//! The API hosted beside another user of the engine, as the app hosts it, against
//! crates/desktop/examples/fake_engine.py.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use mimvox_api::host::{Host, HostStatus, Listen};
use mimvox_api::{ApiConfig, Hub};
use mimvox_engine_client::{EngineClient, EngineConfig, EngineMsg};
use reqwest::StatusCode;
use serde_json::{Value, json};
use tokio::sync::mpsc::{UnboundedReceiver, unbounded_channel};

const WAIT: Duration = Duration::from_secs(20);

fn python() -> PathBuf {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .flat_map(|d| ["python3", "python", "python.exe"].map(|n| d.join(n)))
        // The first one that starts, as the tests write it: on Windows PATH can list a Store alias that does not.
        .filter(|p| p.is_file())
        .find(|p| {
            let p = p.display().to_string().replace('\\', "/");
            std::process::Command::new(p).arg("--version").output().is_ok_and(|o| o.status.success())
        })
        .expect("python3 on PATH")
}

fn config(dir: &Path) -> EngineConfig {
    std::fs::create_dir_all(dir.join("data")).unwrap();
    let python = python().display().to_string().replace('\\', "/");
    std::fs::write(dir.join("runtime.toml"), format!("[python]\nengine = \"{python}\"\n")).unwrap();
    EngineConfig {
        runtime_toml: dir.join("runtime.toml"),
        engine_src: Path::new(env!("CARGO_MANIFEST_DIR")).join("../desktop/examples/fake_engine_src"),
        data_dir: dir.join("data"),
    }
}

/// What the app's side sees: every engine message, as its UI thread gets them.
struct Ui {
    rx: UnboundedReceiver<(u64, EngineMsg)>,
}

impl Ui {
    async fn until(&mut self, mut pred: impl FnMut(&EngineMsg) -> bool) -> EngineMsg {
        tokio::time::timeout(WAIT, async {
            loop {
                let (_, msg) = self.rx.recv().await.expect("the sink is gone");
                if pred(&msg) {
                    return msg;
                }
            }
        })
        .await
        .expect("the UI side got nothing in time")
    }
}

struct Hosted {
    data: PathBuf,
    hub: Hub,
    engine: Arc<EngineClient>,
    ui: Ui,
    host: Host,
    status: UnboundedReceiver<HostStatus>,
    // Last: the engine ends before its folder goes.
    _dir: tempfile::TempDir,
}

async fn hosted() -> Hosted {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(dir.path());
    let (hub, _exits) = Hub::new();
    let (ui_tx, ui_rx) = unbounded_channel();
    let sink = hub.tee_sink(move |client, msg| {
        let _ = ui_tx.send((client, msg));
    });
    let engine = Arc::new(EngineClient::spawn(&cfg, sink).unwrap());
    hub.attach_shared(engine.clone());
    let mut ui = Ui { rx: ui_rx };
    ui.until(|m| matches!(m, EngineMsg::Ready(_))).await;
    let (status_tx, status) = unbounded_channel();
    let host = Host::start(&tokio::runtime::Handle::current(), hub.clone(), move |s| {
        let _ = status_tx.send(s);
    });
    Hosted { data: cfg.data_dir.clone(), hub, engine, ui, host, status, _dir: dir }
}

impl Hosted {
    fn listen(&self, addr: &str, token: Option<&str>) {
        let config =
            ApiConfig { data_dir: self.data.clone(), token: token.map(str::to_owned), max_upload_bytes: 1 << 20 };
        self.host.apply(Some(Listen { addr: addr.parse().unwrap(), config }));
    }

    async fn status(&mut self) -> HostStatus {
        tokio::time::timeout(WAIT, self.status.recv()).await.expect("no status in time").expect("the host is gone")
    }

    async fn listening(&mut self) -> SocketAddr {
        match self.status().await {
            HostStatus::Listening(addr) => addr,
            other => panic!("not listening: {other:?}"),
        }
    }
}

async fn rpc(addr: SocketAddr, token: Option<&str>, method: &str) -> reqwest::Result<(StatusCode, Value)> {
    let mut req = reqwest::Client::new().post(format!("http://{addr}/rpc")).json(&json!({ "method": method }));
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let res = req.send().await?;
    Ok((res.status(), res.json().await?))
}

/// Reads `GET /events` as JSON events.
struct Events {
    res: reqwest::Response,
    buf: String,
}

impl Events {
    async fn open(addr: SocketAddr) -> Self {
        let res = reqwest::get(format!("http://{addr}/events")).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        Self { res, buf: String::new() }
    }

    async fn until(&mut self, pred: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut seen = Vec::new();
        loop {
            while let Some(end) = self.buf.find("\n\n") {
                let block: String = self.buf.drain(..end + 2).collect();
                let data: Vec<&str> =
                    block.lines().filter_map(|l| l.strip_prefix("data:")).map(str::trim_start).collect();
                if data.is_empty() {
                    continue;
                }
                let ev: Value = serde_json::from_str(&data.join("\n")).unwrap();
                let hit = pred(&ev);
                seen.push(ev);
                if hit {
                    return seen;
                }
            }
            let chunk = tokio::time::timeout(WAIT, self.res.chunk()).await.expect("no event in time").unwrap();
            self.buf.push_str(&String::from_utf8_lossy(&chunk.expect("stream ended")));
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn one_engine_serves_the_api_and_the_ui() {
    let mut h = hosted().await;
    h.listen("127.0.0.1:0", None);
    let addr = h.listening().await;

    let health: Value = reqwest::get(format!("http://{addr}/health")).await.unwrap().json().await.unwrap();
    assert_eq!(health["engine"]["version"], "0.1.0-fake");

    // The UI's own request is answered to the UI, not taken by the hub.
    let id = h.engine.request("voices.list", json!({}));
    let reply = h.ui.until(|m| matches!(m, EngineMsg::Response { id: got, .. } if *got == id)).await;
    let EngineMsg::Response { result, .. } = reply else { unreachable!() };
    assert_eq!(result.unwrap()["voices"][0]["id"], "fake-voice");

    // A job started through the API reaches /events and the UI alike.
    let mut events = Events::open(addr).await;
    let started = reqwest::Client::new()
        .post(format!("http://{addr}/rpc"))
        .json(&json!({ "method": "tts.generate", "params": { "voice_id": "fake-voice", "text": "One. Two." } }))
        .send()
        .await
        .unwrap();
    assert_eq!(started.status(), StatusCode::OK);
    let job = started.json::<Value>().await.unwrap()["job"].as_str().unwrap().to_owned();
    let seen = events.until(|e| e["event"] == "tts.done" && e["job"] == job.as_str()).await;
    assert_eq!(seen.iter().filter(|e| e["event"] == "tts.chunk" && e["job"] == job.as_str()).count(), 2);
    h.ui.until(
        |m| matches!(m, EngineMsg::Event(ev) if ev.event == "tts.done" && ev.job.as_deref() == Some(job.as_str())),
    )
    .await;

    // Both sides at once: the API's answer and the UI's own.
    let id = h.engine.request("hello", json!({}));
    let (status, body) = rpc(addr, None, "voices.list").await.unwrap();
    assert_eq!((status, body["voices"][0]["id"].as_str()), (StatusCode::OK, Some("fake-voice")));
    h.ui.until(|m| matches!(m, EngineMsg::Response { id: got, .. } if *got == id)).await;

    drop(events);
    let engine = h.hub.take_engine().unwrap();
    assert!(Arc::ptr_eq(&engine, &h.engine));
}

#[tokio::test(flavor = "multi_thread")]
async fn the_listener_follows_the_settings() {
    let mut h = hosted().await;
    h.listen("127.0.0.1:0", None);
    let first = h.listening().await;
    assert_eq!(rpc(first, None, "voices.list").await.unwrap().0, StatusCode::OK);

    h.host.apply(None);
    assert_eq!(h.status().await, HostStatus::Off);
    assert!(rpc(first, None, "voices.list").await.is_err(), "still listening after it was turned off");

    // The same port again, now asking for the token: the old listener is gone before the new one binds.
    h.listen(&first.to_string(), Some("secret"));
    assert_eq!(h.listening().await, first);
    assert_eq!(rpc(first, None, "voices.list").await.unwrap().0, StatusCode::UNAUTHORIZED);
    assert_eq!(rpc(first, Some("secret"), "voices.list").await.unwrap().0, StatusCode::OK);

    // Another port: the first one closes.
    h.listen("127.0.0.1:0", Some("secret"));
    let second = h.listening().await;
    assert_ne!(second, first);
    assert!(rpc(first, Some("secret"), "voices.list").await.is_err());
    assert_eq!(rpc(second, Some("secret"), "voices.list").await.unwrap().0, StatusCode::OK);

    // A port another program holds, and the local network without a token, fail and say why.
    let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    h.listen(&taken.local_addr().unwrap().to_string(), None);
    match h.status().await {
        HostStatus::Failed(e) => assert!(e.contains("already in use"), "{e}"),
        other => panic!("{other:?}"),
    }
    h.listen("0.0.0.0:0", None);
    assert!(matches!(h.status().await, HostStatus::Failed(e) if e.contains("token")));
    assert!(rpc(second, Some("secret"), "voices.list").await.is_err());

    // Closing the app (dropping the host) ends the listener.
    h.listen("127.0.0.1:0", None);
    let last = h.listening().await;
    drop(h.host);
    tokio::time::timeout(WAIT, async {
        while rpc(last, None, "voices.list").await.is_ok() {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the listener outlived its host");
}
