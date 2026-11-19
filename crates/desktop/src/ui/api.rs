use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use mimvox_api::host::{Host, HostStatus, Listen};
use mimvox_api::token::{self, TokenSource};
use mimvox_api::{ApiConfig, Hub};
use rune::TextEdit;
use serde_json::{Value, json};

use super::components::Tone;
use super::shell::{self, EngineStatus};
use crate::engine::{EngineClient, EngineSink};
use crate::state::AppState;

pub const DEFAULT_PORT: u16 = 8765;
const SETTINGS_KEY: &str = "api";
/// How long a copied token stays on the clipboard.
const COPY_CLEAR: Duration = Duration::from_secs(120);

pub const OTHER_ENGINE: &str = "Another Mimvox engine (mimvox-server) uses this data folder, so the API cannot start \
     here. Stop it, then press Retry.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prefs {
    pub on: bool,
    pub lan: bool,
    pub port: u16,
    pub require_token: bool,
    pub notice_seen: bool,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs { on: false, lan: false, port: DEFAULT_PORT, require_token: false, notice_seen: false }
    }
}

impl Prefs {
    fn from_json(v: Option<&Value>) -> Prefs {
        let d = Prefs::default();
        let Some(v) = v else { return d };
        let flag = |k: &str, or: bool| v.get(k).and_then(Value::as_bool).unwrap_or(or);
        Prefs {
            on: flag("on", d.on),
            lan: flag("lan", d.lan),
            port: v
                .get("port")
                .and_then(Value::as_u64)
                .and_then(|p| u16::try_from(p).ok())
                .filter(|p| *p > 0)
                .unwrap_or(d.port),
            require_token: flag("require_token", d.require_token),
            notice_seen: flag("notice_seen", d.notice_seen),
        }
    }

    fn to_json(self) -> Value {
        json!({
            "on": self.on,
            "lan": self.lan,
            "port": self.port,
            "require_token": self.require_token,
            "notice_seen": self.notice_seen,
        })
    }

    /// The local network always needs the token.
    pub fn needs_token(self) -> bool {
        self.lan || self.require_token
    }

    pub fn addr(self) -> SocketAddr {
        let ip = if self.lan { Ipv4Addr::UNSPECIFIED } else { Ipv4Addr::LOCALHOST };
        SocketAddr::new(IpAddr::V4(ip), self.port)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Want {
    addr: SocketAddr,
    token: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Off,
    Starting,
    Listening(SocketAddr),
    Failed(String),
}

pub struct ApiState {
    /// Every engine the app starts is attached here, so the API shares it.
    pub hub: Hub,
    pub prefs: Prefs,
    pub prefs_read: bool,
    pub status: Status,
    applied: Option<Want>,
    pub port: TextEdit,
    pub port_error: Option<String>,
    pub token: Option<(String, TokenSource)>,
    pub token_error: Option<String>,
    pub token_shown: bool,
    pub notice_open: bool,
    pub confirm_new_token: bool,
    /// This computer's address on the local network, for the URL shown.
    pub lan_ip: Option<IpAddr>,
}

impl Default for ApiState {
    fn default() -> Self {
        ApiState {
            hub: Hub::new().0,
            prefs: Prefs::default(),
            prefs_read: false,
            status: Status::Off,
            applied: None,
            port: TextEdit::new(DEFAULT_PORT.to_string()),
            port_error: None,
            token: None,
            token_error: None,
            token_shown: false,
            notice_open: false,
            confirm_new_token: false,
            lan_ip: None,
        }
    }
}

/// The sink the app spawns its engine with: the API's hub sees every message, then the UI gets it.
pub fn engine_sink(s: &AppState) -> EngineSink {
    s.api.hub.tee_sink(|client, msg| super::events::post(move |s| shell::on_engine_msg(s, client, msg)))
}

pub fn attach(s: &mut AppState, client: EngineClient) {
    let client = Arc::new(client);
    s.api.hub.attach_shared(client.clone());
    s.engine = Some(client);
}

/// Takes the engine from the UI and the API alike, so dropping it ends it.
pub fn detach(s: &mut AppState) -> Option<Arc<EngineClient>> {
    drop(s.api.hub.take_engine());
    s.engine.take()
}

/// Why the API cannot run now.
pub fn blocked(s: &AppState) -> Option<&'static str> {
    match &s.shell.engine {
        EngineStatus::Stopped { detail, .. } if detail.contains("another Mimvox engine") => Some(OTHER_ENGINE),
        _ => None,
    }
}

fn desired(s: &AppState) -> Option<Want> {
    let p = s.api.prefs;
    // After the first engine start: one that finds mimvox-server holding the data folder blocks the API.
    if !p.on || matches!(s.shell.engine, EngineStatus::Idle) || blocked(s).is_some() {
        return None;
    }
    let token = if p.needs_token() { Some(s.api.token.as_ref()?.0.clone()) } else { None };
    Some(Want { addr: p.addr(), token })
}

/// Reads the saved choices once and keeps the listener where they say. True when what the window shows changed.
pub fn settle(s: &mut AppState) -> bool {
    if s.shell.scripted() {
        return false;
    }
    let mut changed = false;
    if !s.api.prefs_read {
        s.api.prefs_read = true;
        s.api.prefs = Prefs::from_json(super::onboarding::setting(SETTINGS_KEY).as_ref());
        s.api.port = TextEdit::new(s.api.prefs.port.to_string());
        if s.api.prefs.lan {
            s.api.lan_ip = lan_ip();
        }
        changed = true;
    }
    let wanted = s.settings.open || (s.api.prefs.on && s.api.prefs.needs_token());
    if wanted && s.api.token.is_none() && s.api.token_error.is_none() {
        load_token(s);
        changed = true;
    }
    let want = desired(s);
    if want != s.api.applied {
        s.api.applied = want.clone();
        s.api.status = if want.is_some() { Status::Starting } else { Status::Off };
        let data_dir = s.shell.data_dir.clone();
        let listen = want.map(|w| Listen {
            addr: w.addr,
            config: ApiConfig {
                data_dir: data_dir.clone(),
                token: w.token,
                max_upload_bytes: mimvox_api::DEFAULT_MAX_UPLOAD_BYTES,
            },
        });
        host_apply(listen, &s.api.hub, &data_dir);
        changed = true;
    }
    changed
}

fn load_token(s: &mut AppState) {
    match token::load_or_create(&s.shell.data_dir) {
        Ok(t) => {
            s.api.token = Some(t);
            s.api.token_error = None;
        }
        Err(e) => s.api.token_error = Some(format!("Cannot read or create the token: {e}")),
    }
}

fn on_status(s: &mut AppState, status: HostStatus) {
    s.api.status = match status {
        HostStatus::Off => Status::Off,
        HostStatus::Listening(addr) => {
            if addr.ip().is_unspecified() && s.api.lan_ip.is_none() {
                s.api.lan_ip = lan_ip();
            }
            Status::Listening(addr)
        }
        HostStatus::Failed(e) => Status::Failed(e),
    };
}

/// The address the system would send from: no packet leaves, a UDP connect only picks the route.
fn lan_ip() -> Option<IpAddr> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.0.2.1:9").ok()?;
    Some(socket.local_addr().ok()?.ip()).filter(|ip| !ip.is_unspecified() && !ip.is_loopback())
}

/// What a client uses: loopback, or this computer's address on the network.
pub fn url(s: &AppState, addr: SocketAddr) -> String {
    match s.api.lan_ip {
        _ if !addr.ip().is_unspecified() => format!("http://{addr}"),
        Some(ip) => format!("http://{}", SocketAddr::new(ip, addr.port())),
        None => format!("http://<this computer>:{}", addr.port()),
    }
}

struct Hosting {
    rt: tokio::runtime::Runtime,
    host: Host,
}

static HOSTING: Mutex<Option<Hosting>> = Mutex::new(None);

fn host_apply(want: Option<Listen>, hub: &Hub, data_dir: &Path) {
    let mut hosting = HOSTING.lock().unwrap_or_else(|e| e.into_inner());
    if hosting.is_none() {
        if want.is_none() {
            return;
        }
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("mimvox-api")
            .enable_all()
            .build();
        let rt = match rt {
            Ok(rt) => rt,
            Err(e) => {
                let message = format!("Cannot start the API: {e}");
                super::events::post(move |s| s.api.status = Status::Failed(message));
                return;
            }
        };
        let host = Host::start(rt.handle(), hub.clone(), |status| super::events::post(move |s| on_status(s, status)));
        let data_dir: PathBuf = data_dir.to_path_buf();
        rt.spawn_blocking(move || mimvox_api::prune_uploads(&data_dir));
        *hosting = Some(Hosting { rt, host });
    }
    if let Some(h) = hosting.as_ref() {
        h.host.apply(want);
    }
}

/// The API ends with the app.
pub fn stop_on_exit() {
    let hosting = HOSTING.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(Hosting { rt, host }) = hosting {
        drop(host);
        rt.shutdown_timeout(Duration::from_secs(1));
    }
}

fn save(s: &AppState) {
    if !s.shell.scripted() {
        let prefs = s.api.prefs.to_json();
        super::onboarding::update_settings(|v| v[SETTINGS_KEY] = prefs);
    }
}

pub fn set_on(s: &mut AppState, on: bool) {
    if on && !s.api.prefs.notice_seen {
        s.api.notice_open = true;
        return;
    }
    s.api.prefs.on = on;
    save(s);
}

pub fn accept_notice(s: &mut AppState) {
    s.api.notice_open = false;
    s.api.prefs.notice_seen = true;
    s.api.prefs.on = true;
    save(s);
}

pub fn decline_notice(s: &mut AppState) {
    s.api.notice_open = false;
}

pub fn set_lan(s: &mut AppState, lan: bool) {
    s.api.prefs.lan = lan;
    if lan && s.api.lan_ip.is_none() && !s.shell.scripted() {
        s.api.lan_ip = lan_ip();
    }
    save(s);
}

pub fn set_require_token(s: &mut AppState, on: bool) {
    if !s.api.prefs.lan {
        s.api.prefs.require_token = on;
        save(s);
    }
}

pub fn parse_port(text: &str) -> Result<u16, String> {
    match text.trim().parse::<u16>() {
        Ok(p) if p > 0 => Ok(p),
        _ => Err("Enter a port from 1 to 65535.".to_owned()),
    }
}

pub fn port_dirty(s: &AppState) -> bool {
    s.api.port.text.trim() != s.api.prefs.port.to_string()
}

pub fn apply_port(s: &mut AppState) {
    match parse_port(&s.api.port.text) {
        Ok(port) => {
            s.api.port_error = None;
            s.api.port = TextEdit::new(port.to_string());
            if port != s.api.prefs.port {
                s.api.prefs.port = port;
                save(s);
            }
        }
        Err(e) => s.api.port_error = Some(e),
    }
}

pub fn copy_token(s: &mut AppState, tx: &mut rune::Tx) {
    if let Some((t, _)) = &s.api.token {
        let _ = tx.copy_secret(t.clone(), COPY_CLEAR);
        shell::toast(s, Tone::Success, "Token copied");
    }
}

pub fn new_token(s: &mut AppState) {
    s.api.confirm_new_token = false;
    if s.shell.scripted() {
        return;
    }
    match token::rotate(&s.shell.data_dir) {
        Ok(t) => {
            s.api.token = Some((t, TokenSource::File(token::token_path(&s.shell.data_dir))));
            s.api.token_error = None;
            shell::toast(s, Tone::Success, "New token made. Clients need it from now on.");
        }
        Err(e) => shell::toast(s, Tone::Danger, format!("Cannot write the token: {e}")),
    }
}

pub fn token_from_env(s: &AppState) -> bool {
    matches!(s.api.token, Some((_, TokenSource::Env)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefs_round_trip_and_fall_back() {
        let p = Prefs { on: true, lan: true, port: 9000, require_token: false, notice_seen: true };
        assert_eq!(Prefs::from_json(Some(&p.to_json())), p);
        assert_eq!(Prefs::from_json(None), Prefs::default());
        let bad = json!({ "on": "yes", "port": 70000 });
        assert_eq!(Prefs::from_json(Some(&bad)), Prefs::default());
        assert_eq!(Prefs::from_json(Some(&json!({ "port": 0 }))).port, DEFAULT_PORT);
    }

    #[test]
    fn the_local_network_needs_the_token() {
        let local = Prefs::default();
        assert!(!local.needs_token());
        assert!(local.addr().ip().is_loopback());
        let lan = Prefs { lan: true, ..local };
        assert!(lan.needs_token());
        assert!(lan.addr().ip().is_unspecified());
        assert!(Prefs { require_token: true, ..local }.needs_token());
    }

    #[test]
    fn ports() {
        assert_eq!(parse_port(" 8765 "), Ok(8765));
        for bad in ["", "0", "65536", "80a", "-1"] {
            assert!(parse_port(bad).is_err(), "{bad}");
        }
    }
}
