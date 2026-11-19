use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use tokio::net::{TcpListener, TcpStream};
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::{ApiConfig, Hub};

/// How long a restart waits for the old listener to close before binding again.
const CLOSE_WAIT: Duration = Duration::from_secs(5);

pub struct Listen {
    pub addr: SocketAddr,
    pub config: ApiConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostStatus {
    Off,
    Listening(SocketAddr),
    Failed(String),
}

/// The API served beside another user of the hub's engine (the app): listeners come and go on [`Host::apply`], one
/// at a time and in order; each outcome goes to `report`. Dropping the host closes the listener.
pub struct Host {
    tx: mpsc::UnboundedSender<Option<Listen>>,
}

impl Host {
    pub fn start(rt: &Handle, hub: Hub, report: impl Fn(HostStatus) + Send + Sync + 'static) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        rt.spawn(control(hub, rx, Arc::new(report)));
        Self { tx }
    }

    /// `None` stops listening; a new `Listen` replaces the listener (open `/events` streams end).
    pub fn apply(&self, want: Option<Listen>) {
        let _ = self.tx.send(want);
    }
}

struct Running {
    stop: CancellationToken,
    closed: oneshot::Receiver<()>,
}

impl Running {
    async fn stop(self) {
        self.stop.cancel();
        let _ = tokio::time::timeout(CLOSE_WAIT, self.closed).await;
    }
}

async fn control(
    hub: Hub,
    mut rx: mpsc::UnboundedReceiver<Option<Listen>>,
    report: Arc<dyn Fn(HostStatus) + Send + Sync>,
) {
    let mut current: Option<Running> = None;
    while let Some(mut want) = rx.recv().await {
        while let Ok(newer) = rx.try_recv() {
            want = newer;
        }
        if let Some(run) = current.take() {
            run.stop().await;
        }
        let Some(listen) = want else {
            report(HostStatus::Off);
            continue;
        };
        match serve(&hub, listen).await {
            Ok((run, addr)) => {
                current = Some(run);
                report(HostStatus::Listening(addr));
            }
            Err(e) => report(HostStatus::Failed(e)),
        }
    }
    if let Some(run) = current.take() {
        run.stop().await;
    }
}

async fn serve(hub: &Hub, listen: Listen) -> Result<(Running, SocketAddr), String> {
    crate::check_bind(listen.addr, listen.config.token.as_deref()).map_err(|e| e.to_string())?;
    let listener = TcpListener::bind(listen.addr).await.map_err(|e| bind_error(listen.addr, &e))?;
    let addr = listener.local_addr().map_err(|e| bind_error(listen.addr, &e))?;
    let stop = CancellationToken::new();
    let (closed_tx, closed) = oneshot::channel();
    let listener = Closing { inner: listener, _closed: closed_tx };
    let app = crate::router(hub.clone(), listen.config, stop.clone());
    tokio::spawn(axum::serve(listener, app).with_graceful_shutdown(stop.clone().cancelled_owned()).into_future());
    Ok((Running { stop, closed }, addr))
}

pub fn bind_error(addr: SocketAddr, e: &io::Error) -> String {
    let port = addr.port();
    match e.kind() {
        io::ErrorKind::AddrInUse => format!("Port {port} is already in use by another program."),
        io::ErrorKind::PermissionDenied => format!("No permission to use port {port}."),
        io::ErrorKind::AddrNotAvailable => format!("The address {} is not available on this computer.", addr.ip()),
        _ => format!("Cannot listen on {addr}: {e}."),
    }
}

/// axum drops the listener as soon as a graceful shutdown starts; its sender going with it says the port is free.
struct Closing {
    inner: TcpListener,
    _closed: oneshot::Sender<()>,
}

impl axum::serve::Listener for Closing {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        axum::serve::Listener::accept(&mut self.inner).await
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}
