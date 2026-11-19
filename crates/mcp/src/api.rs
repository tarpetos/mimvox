//! The Mimvox HTTP API as this server uses it (contract/engine-protocol.md, "HTTP API").

use std::fmt;
use std::path::Path;
use std::time::Duration;

use reqwest::{Client, Method, RequestBuilder, Response, StatusCode, Url};
use serde_json::{Value, json};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const RPC_TIMEOUT: Duration = Duration::from_secs(600);
/// `/events` sends a keep-alive every 15 s, so a longer silence means the connection is gone.
const EVENTS_READ_TIMEOUT: Duration = Duration::from_secs(60);

/// An error of the API, of the engine behind it (`code` as the engine sent it) or of reaching it.
#[derive(Clone, Debug)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub status: Option<u16>,
}

impl ApiError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_string(), message: message.into(), status: None }
    }

    pub fn to_json(&self) -> Value {
        json!({ "code": self.code, "message": self.message, "http_status": self.status })
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

pub struct Api {
    base: Url,
    token: Option<String>,
    http: Client,
    events: Client,
}

impl Api {
    pub fn new(base: &str, token: Option<String>) -> anyhow::Result<Self> {
        let base = Url::parse(base).map_err(|e| anyhow::anyhow!("bad API URL {base:?}: {e}"))?;
        if base.scheme() != "http" {
            anyhow::bail!("the API URL must start with http:// (the Mimvox API speaks plain HTTP): {base}");
        }
        let http = Client::builder().connect_timeout(CONNECT_TIMEOUT).build()?;
        let events = Client::builder().connect_timeout(CONNECT_TIMEOUT).read_timeout(EVENTS_READ_TIMEOUT).build()?;
        Ok(Self { base, token, http, events })
    }

    pub fn base(&self) -> &Url {
        &self.base
    }

    fn url(&self, path: &str) -> Url {
        let mut url = self.base.clone();
        url.set_path(&format!("{}/{path}", self.base.path().trim_end_matches('/')));
        url.set_query(None);
        url
    }

    fn request(&self, client: &Client, method: Method, path: &str) -> RequestBuilder {
        let req = client.request(method, self.url(path));
        match &self.token {
            Some(token) => req.bearer_auth(token),
            None => req,
        }
    }

    async fn send(&self, req: RequestBuilder) -> Result<Response, ApiError> {
        let res = req.send().await.map_err(|e| self.transport_error(e))?;
        if res.status().is_success() {
            return Ok(res);
        }
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        Err(self.status_error(status, &body))
    }

    fn transport_error(&self, e: reqwest::Error) -> ApiError {
        if e.is_timeout() {
            return ApiError::new("timeout", format!("the Mimvox API at {} did not answer in time", self.base));
        }
        if e.is_connect() {
            let mut reason = String::new();
            let mut source = std::error::Error::source(&e);
            while let Some(s) = source {
                reason = s.to_string();
                source = s.source();
            }
            return ApiError::new(
                "unreachable",
                format!(
                    "cannot reach the Mimvox API at {} ({reason}). Turn on the API in Mimvox Settings, or run \
                     `mimvox serve` (mimvox-server).",
                    self.base
                ),
            );
        }
        ApiError::new("http_error", format!("request to the Mimvox API at {} failed: {e}", self.base))
    }

    fn status_error(&self, status: StatusCode, body: &str) -> ApiError {
        let parsed: Value = serde_json::from_str(body).unwrap_or(Value::Null);
        let inner = if parsed["error"].is_object() { &parsed["error"] } else { &parsed };
        let code = inner["code"].as_str().map(str::to_string).unwrap_or_else(|| match status {
            StatusCode::UNAUTHORIZED => "unauthorized".into(),
            StatusCode::FORBIDDEN => "forbidden".into(),
            StatusCode::NOT_FOUND => "not_found".into(),
            _ => format!("http_{}", status.as_u16()),
        });
        let mut message =
            inner["message"].as_str().map(str::to_string).unwrap_or_else(|| format!("HTTP {status}: {}", body.trim()));
        match status {
            StatusCode::UNAUTHORIZED => message.push_str(&format!(
                ". The API at {} asks for its token: set {} or pass --token-file (`mimvox-server token show` prints \
                 it; in the app: Settings, API).",
                self.base,
                mimvox_api::token::TOKEN_ENV
            )),
            StatusCode::FORBIDDEN if code == "forbidden" => {
                message.push_str(". Without a token the API serves only loopback clients: give mimvox-mcp the token.")
            }
            _ => {}
        }
        ApiError { code, message, status: Some(status.as_u16()) }
    }

    /// `POST /rpc`: the engine's result as is.
    pub async fn rpc(&self, method: &str, params: Value) -> Result<Value, ApiError> {
        let req = self
            .request(&self.http, Method::POST, "rpc")
            .timeout(RPC_TIMEOUT)
            .json(&json!({ "method": method, "params": params }));
        let res = self.send(req).await?;
        res.json().await.map_err(|e| ApiError::new("bad_response", format!("the API answered {method} with: {e}")))
    }

    /// `GET /events`, the open Server-Sent Events response.
    pub async fn events(&self) -> Result<Response, ApiError> {
        self.send(self.request(&self.events, Method::GET, "events")).await
    }

    /// `POST /files`: streams a local file to the API's uploads folder.
    pub async fn upload(&self, local: &Path) -> Result<Value, ApiError> {
        let file = tokio::fs::File::open(local)
            .await
            .map_err(|e| ApiError::new("bad_request", format!("cannot open {}: {e}", local.display())))?;
        let len = file.metadata().await.map(|m| m.len()).unwrap_or(0);
        let name = local.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "upload".into());
        let body = reqwest::Body::wrap_stream(tokio_util::io::ReaderStream::new(file));
        let req = self
            .request(&self.http, Method::POST, "files")
            .query(&[("name", name.as_str())])
            .header(reqwest::header::CONTENT_LENGTH, len)
            .body(body);
        let res = self.send(req).await?;
        res.json().await.map_err(|e| ApiError::new("bad_response", format!("the API answered the upload with: {e}")))
    }

    /// `GET /files`: the response, to be streamed by the caller.
    pub async fn download(&self, path: &str) -> Result<Response, ApiError> {
        self.send(self.request(&self.http, Method::GET, "files").query(&[("path", path)])).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_keep_a_path_prefix() {
        let api = Api::new("http://127.0.0.1:8765", None).unwrap();
        assert_eq!(api.url("rpc").as_str(), "http://127.0.0.1:8765/rpc");
        let api = Api::new("http://box.lan:9000/mimvox/", None).unwrap();
        assert_eq!(api.url("files").as_str(), "http://box.lan:9000/mimvox/files");
        assert!(Api::new("https://box.lan", None).is_err());
    }

    #[test]
    fn engine_and_openai_error_bodies() {
        let api = Api::new("http://127.0.0.1:8765", None).unwrap();
        let e = api.status_error(StatusCode::NOT_FOUND, r#"{"code":"not_found","message":"unknown method x"}"#);
        assert_eq!((e.code.as_str(), e.message.as_str(), e.status), ("not_found", "unknown method x", Some(404)));
        let e = api.status_error(StatusCode::BAD_REQUEST, r#"{"error":{"message":"bad","code":"bad_request"}}"#);
        assert_eq!(e.code, "bad_request");
        let e = api.status_error(StatusCode::UNAUTHORIZED, "");
        assert_eq!(e.code, "unauthorized");
        assert!(e.message.contains("MIMVOX_API_TOKEN"), "{}", e.message);
    }
}
