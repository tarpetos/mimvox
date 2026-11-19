//! The local HTTP API: contract/engine-protocol.md, "HTTP API".

mod ffmpeg;
mod files;
pub mod host;
mod hub;
mod openai;
pub mod token;

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::Stream;
use mimvox_engine_client::protocol::EngineError;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast::error::RecvError;
use tokio_util::sync::CancellationToken;

pub use files::{UPLOAD_MAX_AGE, prune_uploads, uploads_dir};
pub use hub::{EngineEvent, EngineExit, Hub, Reply};

pub const DEFAULT_BIND: &str = "127.0.0.1:8765";
pub const DEFAULT_MAX_UPLOAD_BYTES: u64 = 2 << 30;
const RPC_BODY_LIMIT: usize = 16 << 20;
const SPEECH_BODY_LIMIT: usize = 1 << 20;
const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);

pub struct ApiConfig {
    pub data_dir: PathBuf,
    /// Asked of every request when set; without it only loopback clients that are not web pages get in.
    pub token: Option<String>,
    pub max_upload_bytes: u64,
}

/// A token is required on any address but loopback.
pub fn check_bind(addr: SocketAddr, token: Option<&str>) -> anyhow::Result<()> {
    if !addr.ip().is_loopback() && token.is_none_or(str::is_empty) {
        anyhow::bail!(
            "refusing to listen on {addr} without a token: create one with `mimvox-server token new` or set {}",
            token::TOKEN_ENV
        );
    }
    Ok(())
}

#[derive(Clone)]
struct AppState {
    hub: Hub,
    cfg: Arc<ApiConfig>,
    speech: Arc<tokio::sync::Mutex<()>>,
    stop: CancellationToken,
}

/// `stop` ends the open `/events` streams, so a graceful shutdown does not wait for them.
pub fn router(hub: Hub, mut cfg: ApiConfig, stop: CancellationToken) -> Router {
    cfg.data_dir = std::path::absolute(&cfg.data_dir).unwrap_or(cfg.data_dir);
    let state = AppState { hub, cfg: Arc::new(cfg), speech: Arc::default(), stop };
    Router::new()
        .route("/health", get(health))
        .route("/rpc", post(rpc).layer(DefaultBodyLimit::max(RPC_BODY_LIMIT)))
        .route("/events", get(events))
        .route("/files", post(files::upload).get(files::download))
        .route("/v1/models", get(openai::models))
        .route("/v1/audio/voices", get(openai::voices))
        .route("/v1/audio/speech", post(openai::speech).layer(DefaultBodyLimit::max(SPEECH_BODY_LIMIT)))
        .layer(middleware::from_fn_with_state(state.clone(), guard))
        .with_state(state)
}

#[derive(Debug)]
pub(crate) struct ApiError {
    status: StatusCode,
    code: String,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, code: &str, message: impl Into<String>) -> Self {
        Self { status, code: code.to_string(), message: message.into() }
    }

    fn bad_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }

    fn internal(e: impl std::fmt::Display) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string())
    }
}

pub fn status_for(code: &str) -> StatusCode {
    match code {
        "bad_request" => StatusCode::BAD_REQUEST,
        "not_found" => StatusCode::NOT_FOUND,
        "busy" => StatusCode::CONFLICT,
        "not_ready" => StatusCode::SERVICE_UNAVAILABLE,
        "too_large" => StatusCode::PAYLOAD_TOO_LARGE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        Self { status: status_for(&e.code), code: e.code, message: e.message }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(e: JsonRejection) -> Self {
        let code = if e.status() == StatusCode::PAYLOAD_TOO_LARGE { "too_large" } else { "bad_request" };
        Self::new(e.status(), code, e.body_text())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({ "code": self.code, "message": self.message }))).into_response()
    }
}

fn is_local_host(host: &str) -> bool {
    let name = match host.strip_prefix('[') {
        Some(rest) => rest.split(']').next().unwrap_or(""),
        None => host.split(':').next().unwrap_or(""),
    };
    name.eq_ignore_ascii_case("localhost") || name.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

/// Without a token, a web page could reach a loopback server through the browser (a cross-site request, or DNS
/// rebinding): only requests addressed to a loopback name and not sent by a remote page get in. Browsers send no
/// `Origin` on a plain GET (`<img>`, `<audio>`, a link) but do send `Sec-Fetch-Site`.
fn local_request(headers: &HeaderMap) -> bool {
    let text = |h: Option<&HeaderValue>| h.map(|v| v.to_str().unwrap_or("\u{0}").to_string());
    let host_ok = text(headers.get(header::HOST)).is_none_or(|h| is_local_host(&h));
    let origin_ok = text(headers.get(header::ORIGIN)).is_none_or(|o| {
        o.split_once("://").is_some_and(|(_, rest)| is_local_host(rest.split('/').next().unwrap_or("")))
    });
    let fetch_ok = text(headers.get("sec-fetch-site")).is_none_or(|s| s == "same-origin" || s == "none");
    host_ok && origin_ok && fetch_ok
}

async fn guard(State(st): State<AppState>, req: Request, next: Next) -> Response {
    match &st.cfg.token {
        Some(expected) => {
            let given = req
                .headers()
                .get(header::AUTHORIZATION)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
                .map(str::trim);
            if !given.is_some_and(|g| token::matches(g, expected)) {
                let mut res = ApiError::new(StatusCode::UNAUTHORIZED, "unauthorized", "missing or wrong bearer token")
                    .into_response();
                res.headers_mut().insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
                return res;
            }
        }
        None if !local_request(req.headers()) => {
            return ApiError::new(StatusCode::FORBIDDEN, "forbidden", "without a token only local clients are served")
                .into_response();
        }
        None => {}
    }
    next.run(req).await
}

async fn health(State(st): State<AppState>) -> Result<Json<Value>, ApiError> {
    let hello = tokio::time::timeout(HEALTH_TIMEOUT, st.hub.call("hello", json!({}))).await.map_err(|_| {
        ApiError::new(StatusCode::SERVICE_UNAVAILABLE, "not_ready", "the engine did not answer in time")
    })??;
    Ok(Json(json!({ "status": "ok", "api_version": env!("CARGO_PKG_VERSION"), "engine": hello })))
}

#[derive(Deserialize)]
struct RpcRequest {
    method: String,
    #[serde(default)]
    params: Value,
}

async fn rpc(
    State(st): State<AppState>,
    req: Result<Json<RpcRequest>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let Json(req) = req?;
    if req.method.trim().is_empty() {
        return Err(ApiError::bad_request("method is empty"));
    }
    let params = if req.params.is_null() { json!({}) } else { req.params };
    Ok(Json(st.hub.call(&req.method, params).await?))
}

async fn events(State(st): State<AppState>) -> Sse<impl Stream<Item = Result<SseEvent, Infallible>>> {
    let rx = st.hub.subscribe();
    let stream = futures_util::stream::unfold((rx, st.stop), |(mut rx, stop)| async move {
        let received = tokio::select! {
            _ = stop.cancelled() => return None,
            r = rx.recv() => r,
        };
        let line = match received {
            Ok(ev) => ev.json.clone(),
            Err(RecvError::Lagged(missed)) => {
                let data = json!({ "missed": missed });
                json!({ "event": "api.lagged", "job": null, "data": data }).to_string()
            }
            Err(RecvError::Closed) => return None,
        };
        Some((Ok(SseEvent::default().data(line)), (rx, stop)))
    });
    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bind_needs_a_token_off_loopback() {
        assert!(check_bind("127.0.0.1:8765".parse().unwrap(), None).is_ok());
        assert!(check_bind("[::1]:8765".parse().unwrap(), None).is_ok());
        assert!(check_bind("0.0.0.0:8765".parse().unwrap(), None).is_err());
        assert!(check_bind("192.168.1.5:8765".parse().unwrap(), Some("")).is_err());
        assert!(check_bind("0.0.0.0:8765".parse().unwrap(), Some("t")).is_ok());
    }

    #[test]
    fn local_hosts() {
        for h in ["127.0.0.1:8765", "localhost", "LOCALHOST:1", "[::1]:8765", "127.0.0.2"] {
            assert!(is_local_host(h), "{h}");
        }
        for h in ["evil.example:8765", "192.168.1.5", "[::2]:1", "localhost.evil.example"] {
            assert!(!is_local_host(h), "{h}");
        }
    }

    #[test]
    fn web_pages_are_refused_without_a_token() {
        let mut h = HeaderMap::new();
        h.insert(header::HOST, HeaderValue::from_static("127.0.0.1:8765"));
        assert!(local_request(&h));
        h.insert(header::ORIGIN, HeaderValue::from_static("http://localhost:3000"));
        assert!(local_request(&h));
        h.insert(header::ORIGIN, HeaderValue::from_static("https://evil.example"));
        assert!(!local_request(&h));
        h.insert(header::ORIGIN, HeaderValue::from_static("null"));
        assert!(!local_request(&h));
        h.remove(header::ORIGIN);
        for (site, ok) in [("cross-site", false), ("same-site", false), ("same-origin", true), ("none", true)] {
            h.insert("sec-fetch-site", HeaderValue::from_static(site));
            assert_eq!(local_request(&h), ok, "{site}");
        }
        h.remove("sec-fetch-site");
        h.remove(header::ORIGIN);
        h.insert(header::HOST, HeaderValue::from_static("rebound.evil.example:8765"));
        assert!(!local_request(&h));
    }

    #[test]
    fn error_codes_map_to_statuses() {
        assert_eq!(status_for("bad_request"), StatusCode::BAD_REQUEST);
        assert_eq!(status_for("not_found"), StatusCode::NOT_FOUND);
        assert_eq!(status_for("busy"), StatusCode::CONFLICT);
        assert_eq!(status_for("not_ready"), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status_for("cancelled"), StatusCode::INTERNAL_SERVER_ERROR);
    }
}
