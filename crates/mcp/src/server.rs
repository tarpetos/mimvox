//! The MCP tools and resources.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use futures_util::StreamExt;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ListResourceTemplatesResult, ListResourcesResult,
    PaginatedRequestParams, ProgressNotificationParam, ProgressToken, ReadResourceRequestParams, ReadResourceResponse,
    ReadResourceResult, RequestMetaObject, Resource, ResourceContents, ResourceTemplate, ServerCapabilities,
    ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{ErrorData, Peer, RoleServer, ServerHandler, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::io::AsyncWriteExt;
use tokio::sync::broadcast::error::RecvError;
use tokio_util::sync::CancellationToken;

use crate::api::{Api, ApiError};
use crate::events::{Event, JobState, State, Tracker, end_state, progress_of};

/// Audio larger than this is returned as a link only.
pub const AUDIO_INLINE_MAX: u64 = 4 << 20;
/// `resources/read` refuses larger files (download_file streams them instead).
pub const RESOURCE_MAX: u64 = 32 << 20;
const EVENTS_CONNECT: Duration = Duration::from_secs(10);
const SPEAK_TIMEOUT_S: u64 = 600;
const WAIT_TIMEOUT_S: u64 = 300;
const MAX_TIMEOUT_S: u64 = 3600;
const RESOURCE_LIST_LIMIT: u64 = 100;
pub const FILE_URI: &str = "mimvox://file/";

const INSTRUCTIONS: &str = "Mimvox clones a voice and speaks text with it, on the user's machine. This server is a \
client of the Mimvox local HTTP API (on in Mimvox Settings, API, or `mimvox serve`). Paths in requests and answers are \
on the API's machine: with a remote API, upload_file a local file first and download_file results. Only clone the \
user's own voice or one whose owner gave explicit permission; the speech model weights are CC BY-NC 4.0, so \
generated speech is for non-commercial use only.";

const ENGINE_CALL: &str = "Call any Mimvox engine method through the API's /rpc and return its result as is. Covers \
everything the app can do. Method families: hello, components.set; voices.* (list, create, rename, delete, import, \
warm, reference_candidates, set_reference, add_reference, remove_reference, reference_from_media, media_cancel, \
clear_media); media.combine; presets.list; text.analyze, text.import, text.stress; recording.check; dataset.add, \
dataset.remove, dataset.summary; train.start, train.pause, train.resume, train.cancel; tts.generate, tts.cancel, \
tts.languages; outputs.list, outputs.play, outputs.delete; formats.list, audio.export; book.start, book.resume, \
book.pause, book.cancel, book.list, book.get, book.delete, book.reread, book.play; redub.probe, redub.prepare, \
redub.transcribe, redub.dub, redub.render, redub.cancel; shutdown. Parameters and results are documented in \
contract/engine-protocol.md of the Mimvox repository. Methods that answer {job} run in the background: follow them \
with wait_job or job_status. Errors carry the engine's code (bad_request, not_found, busy, not_ready, paused, \
cancelled, internal).";

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Ogg,
    Opus,
    M4a,
    Aiff,
}

impl AudioFormat {
    fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
            Self::Opus => "opus",
            Self::M4a => "m4a",
            Self::Aiff => "aiff",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Quality {
    Best,
    Fast,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct EngineCallArgs {
    /// Engine method, e.g. "voices.list" or "audio.export".
    pub method: String,
    /// The method's parameters (an object); omit for none.
    #[serde(default)]
    pub params: Option<Map<String, Value>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SpeakArgs {
    /// The text to speak.
    pub text: String,
    /// Voice id or name (case-insensitive); default: the first voice of list_voices.
    #[serde(default)]
    pub voice: Option<String>,
    /// ISO 639-1 code of the text (en, uk, de, ...); default: the voice's language.
    #[serde(default)]
    pub language: Option<String>,
    /// Format of the saved file; default wav.
    #[serde(default)]
    pub format: Option<AudioFormat>,
    /// Speaking speed, 0.8 to 1.3; default 1.0.
    #[serde(default)]
    pub speed: Option<f64>,
    /// Return the audio itself when it is at most 4 MB (default true); the file path is always returned.
    #[serde(default)]
    pub include_audio: Option<bool>,
    /// Seconds to wait for the result (default 600); after that the job goes on and wait_job follows it.
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ImportTextArgs {
    /// Absolute path of a document on the API's machine (txt, md, html, epub, docx, odt, rtf, fb2, pdf, srt, vtt).
    pub path: String,
    /// Language code of the text, or "auto" (default).
    #[serde(default)]
    pub language: Option<String>,
    /// Include each chapter's full text (default false: titles and word counts only).
    #[serde(default)]
    pub include_text: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct Chapter {
    pub title: String,
    pub text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StartBookArgs {
    /// A document on the API's machine to read (imported as import_text does). Give this or chapters.
    #[serde(default)]
    pub path: Option<String>,
    /// The chapters to read, instead of a document.
    #[serde(default)]
    pub chapters: Option<Vec<Chapter>>,
    /// Only these chapters of the document (0-based indexes, as import_text lists them).
    #[serde(default)]
    pub chapter_indexes: Option<Vec<usize>>,
    /// Book title; default: the document's title.
    #[serde(default)]
    pub title: Option<String>,
    /// Voice id or name; default: the first voice.
    #[serde(default)]
    pub voice: Option<String>,
    /// Language code of the text; default: the document's detected language, else the voice's.
    #[serde(default)]
    pub language: Option<String>,
    /// Format of each chapter file; default wav.
    #[serde(default)]
    pub format: Option<AudioFormat>,
    /// Also make one .m4b audiobook with chapter marks (default false).
    #[serde(default)]
    pub m4b: Option<bool>,
    /// Read each chapter's title first (default true).
    #[serde(default)]
    pub read_titles: Option<bool>,
    /// Speaking speed, 0.8 to 1.3 (default 1.0).
    #[serde(default)]
    pub speed: Option<f64>,
    /// 0.1 to 1.5 (default 0.6 for books): lower is steadier.
    #[serde(default)]
    pub expressiveness: Option<f64>,
    /// "best" (default) or "fast".
    #[serde(default)]
    pub quality: Option<Quality>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BookStatusArgs {
    /// Book id; omit to list every book.
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BookIdArgs {
    /// Book id (from start_book or book_status).
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct JobArgs {
    /// Job id, e.g. "job-3".
    pub job: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WaitJobArgs {
    /// Job id, e.g. "job-3".
    pub job: String,
    /// Seconds to wait (default 300, at most 3600).
    #[serde(default)]
    pub timeout_s: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UploadArgs {
    /// Path of a file on this machine (where mimvox-mcp runs).
    pub local_path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DownloadArgs {
    /// Absolute path on the API's machine, under <data>/outputs, <data>/tmp or <data>/api/uploads.
    pub path: String,
    /// Where to save it on this machine (a file or an existing folder); default: the Downloads folder.
    #[serde(default)]
    pub dest: Option<String>,
    /// Replace an existing file (default false).
    #[serde(default)]
    pub overwrite: Option<bool>,
}

#[derive(Clone)]
pub struct Mimvox {
    api: Arc<Api>,
    events: Tracker,
    tool_router: ToolRouter<Self>,
}

fn ok(value: Value) -> CallToolResult {
    CallToolResult::structured(value)
}

fn fail(e: &ApiError) -> CallToolResult {
    let mut res = CallToolResult::error(vec![ContentBlock::text(e.to_string())]);
    res.structured_content = Some(json!({ "error": e.to_json() }));
    res
}

fn reply(r: Result<Value, ApiError>) -> CallToolResult {
    match r {
        Ok(v) => ok(v),
        Err(e) => fail(&e),
    }
}

fn timeout_of(given: Option<u64>, default: u64) -> Duration {
    Duration::from_secs(given.unwrap_or(default).clamp(1, MAX_TIMEOUT_S))
}

pub fn mime_for(path: &str) -> &'static str {
    let ext = path.rsplit('.').next().unwrap_or_default().to_ascii_lowercase();
    match ext.as_str() {
        "wav" | "wave" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" | "oga" | "opus" => "audio/ogg",
        "m4a" | "aac" => "audio/mp4",
        "m4b" => "audio/mp4",
        "aif" | "aiff" => "audio/aiff",
        "mp4" | "m4v" => "video/mp4",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "webm" => "video/webm",
        "avi" => "video/x-msvideo",
        "json" => "application/json",
        "txt" | "srt" | "vtt" => "text/plain",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "application/octet-stream",
    }
}

fn file_name(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
}

/// `mimvox://file/<the absolute path, percent-encoded>`: a file the API serves.
pub fn file_uri(path: &str) -> String {
    let mut uri = String::from(FILE_URI);
    for b in path.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            uri.push(b as char);
        } else {
            uri.push_str(&format!("%{b:02X}"));
        }
    }
    uri
}

pub fn path_of_uri(uri: &str) -> Option<String> {
    let enc = uri.strip_prefix(FILE_URI)?.as_bytes();
    let mut out = Vec::with_capacity(enc.len());
    let mut i = 0;
    while i < enc.len() {
        if enc[i] == b'%' {
            let hex = std::str::from_utf8(enc.get(i + 1..i + 3)?).ok()?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(enc[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok().filter(|p| !p.is_empty())
}

fn file_link(path: &str, size: Option<u64>) -> ContentBlock {
    let mut res = Resource::new(file_uri(path), file_name(path)).with_mime_type(mime_for(path));
    if let Some(size) = size {
        res = res.with_size(size);
    }
    ContentBlock::resource_link(res)
}

/// The text of every chapter left out (a book's text can be megabytes).
fn without_texts(mut import: Value) -> Value {
    let strip = |chapters: &mut Value| {
        if let Some(list) = chapters.as_array_mut() {
            for (i, ch) in list.iter_mut().enumerate() {
                if let Some(obj) = ch.as_object_mut() {
                    obj.remove("text");
                    obj.insert("index".into(), json!(i));
                }
            }
        }
    };
    strip(&mut import["chapters"]);
    if import["parts"].is_object() {
        strip(&mut import["parts"]["chapters"]);
    }
    import
}

struct Progress {
    peer: Peer<RoleServer>,
    token: ProgressToken,
    last: f64,
}

impl Progress {
    fn new(meta: &RequestMetaObject, peer: Peer<RoleServer>) -> Option<Self> {
        meta.get_progress_token().map(|token| Self { peer, token, last: f64::NEG_INFINITY })
    }

    async fn send(&mut self, ev: &Event, total_hint: Option<f64>) {
        let Some((value, total, message)) = progress_of(ev, total_hint) else { return };
        // Progress must increase with each notification.
        if value <= self.last {
            return;
        }
        self.last = value;
        let mut p = ProgressNotificationParam::new(self.token.clone(), value).with_message(message);
        if let Some(total) = total {
            p = p.with_total(total);
        }
        let _ = self.peer.notify_progress(p).await;
    }
}

enum Waited {
    Ended(JobState),
    TimedOut(Option<JobState>),
    Cancelled,
}

impl Mimvox {
    pub fn new(api: Arc<Api>, events: Tracker) -> Self {
        Self { api, events, tool_router: Self::tool_router() }
    }

    async fn events_ready(&self) -> Result<(), ApiError> {
        self.events.ready(EVENTS_CONNECT).await.map_err(|reason| {
            ApiError::new(
                "unreachable",
                format!(
                    "cannot follow the Mimvox API's events at {} ({reason}). Turn on the API in Mimvox Settings, or \
                     run `mimvox serve` (mimvox-server).",
                    self.api.base()
                ),
            )
        })
    }

    async fn wait(
        &self,
        job: &str,
        timeout: Duration,
        mut progress: Option<&mut Progress>,
        total_hint: Option<f64>,
        cancel: &CancellationToken,
    ) -> Waited {
        let (mut rx, known) = self.events.watch(job);
        if let Some(st) = known.filter(JobState::ended) {
            return Waited::Ended(st);
        }
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let ev = tokio::select! {
                _ = tokio::time::sleep_until(deadline) => return Waited::TimedOut(self.events.job(job)),
                _ = cancel.cancelled() => return Waited::Cancelled,
                ev = rx.recv() => ev,
            };
            match ev {
                Ok(ev) if ev.job.as_deref() == Some(job) => {
                    if end_state(&ev.event).is_some() {
                        if let Some(st) = self.events.job(job) {
                            return Waited::Ended(st);
                        }
                    } else if let Some(p) = progress.as_deref_mut() {
                        p.send(&ev, total_hint).await;
                    }
                }
                Ok(ev) if ev.event == "engine.exited" => {
                    let lost = self.events.job(job).filter(JobState::ended);
                    return Waited::Ended(lost.unwrap_or_else(|| JobState::lost(job, &ev.event)));
                }
                Ok(_) => {}
                Err(RecvError::Lagged(_)) => {
                    if let Some(st) = self.events.job(job).filter(JobState::ended) {
                        return Waited::Ended(st);
                    }
                }
                Err(RecvError::Closed) => return Waited::TimedOut(self.events.job(job)),
            }
        }
    }

    async fn voices(&self) -> Result<Vec<Value>, ApiError> {
        let list = self.api.rpc("voices.list", json!({})).await?;
        Ok(list["voices"].as_array().cloned().unwrap_or_default())
    }

    /// A voice by id or name (case-insensitive), else the first one.
    async fn voice(&self, wanted: Option<&str>) -> Result<Value, ApiError> {
        let voices = self.voices().await?;
        let names = || voices.iter().filter_map(|v| v["name"].as_str()).collect::<Vec<_>>().join(", ");
        match wanted.map(str::trim).filter(|w| !w.is_empty()) {
            Some(w) => voices
                .iter()
                .find(|v| v["id"].as_str() == Some(w))
                .or_else(|| voices.iter().find(|v| v["name"].as_str().is_some_and(|n| n.eq_ignore_ascii_case(w))))
                .cloned()
                .ok_or_else(|| ApiError::new("not_found", format!("no voice {w:?}; the voices are: {}", names()))),
            None => voices
                .first()
                .cloned()
                .ok_or_else(|| ApiError::new("not_found", "there are no voices yet: make one in the Mimvox app first")),
        }
    }

    async fn speak_inner(
        &self,
        args: SpeakArgs,
        progress: Option<&mut Progress>,
        cancel: &CancellationToken,
    ) -> Result<CallToolResult, ApiError> {
        let voice = self.voice(args.voice.as_deref()).await?;
        let mut params = json!({ "voice_id": voice["id"], "text": args.text });
        if let Some(lang) = args.language.or_else(|| voice["language"].as_str().map(str::to_string)) {
            params["language"] = json!(lang);
        }
        if let Some(format) = args.format {
            params["output_format"] = json!(format.as_str());
        }
        if let Some(speed) = args.speed {
            params["speed"] = json!(speed);
        }
        self.events_ready().await?;
        let started = self.api.rpc("tts.generate", params).await?;
        let job = started["job"].as_str().unwrap_or_default().to_string();
        let segments = started["segments"].as_f64();
        let timeout = timeout_of(args.timeout_s, SPEAK_TIMEOUT_S);
        let st = match self.wait(&job, timeout, progress, segments, cancel).await {
            Waited::Ended(st) => st,
            Waited::TimedOut(_) => {
                return Ok(ok(json!({
                    "job": job,
                    "state": "running",
                    "voice": voice["name"],
                    "message": "still speaking: follow it with wait_job",
                })));
            }
            Waited::Cancelled => {
                let _ = self.api.rpc("tts.cancel", json!({ "job": job })).await;
                return Err(ApiError::new("cancelled", "the request was cancelled, and so was the job"));
            }
        };
        let result = st.result.clone().unwrap_or(Value::Null);
        if st.state != State::Done {
            let code = result["code"].as_str().unwrap_or("internal");
            let message = result["message"].as_str().unwrap_or("the job failed");
            return Err(ApiError::new(code, message));
        }
        let path = result["path"].as_str().unwrap_or_default().to_string();
        let summary = json!({
            "job": job,
            "state": "done",
            "voice": voice["name"],
            "voice_id": voice["id"],
            "path": path,
            "uri": file_uri(&path),
            "format": result["format"],
            "duration_s": result["duration_s"],
            "elapsed_s": result["elapsed_s"],
            "result": result,
        });
        let mut content = vec![ContentBlock::text(summary.to_string())];
        let mut size = None;
        if args.include_audio.unwrap_or(true) {
            match self.fetch(&path, AUDIO_INLINE_MAX).await {
                Ok(Fetched::Bytes(bytes)) => {
                    size = Some(bytes.len() as u64);
                    content.push(ContentBlock::audio(BASE64.encode(&bytes), mime_for(&path)));
                }
                Ok(Fetched::TooLarge(n)) => size = n,
                Err(e) => eprintln!("mimvox-mcp: cannot fetch {path}: {e}"),
            }
        }
        content.push(file_link(&path, size));
        let mut res = CallToolResult::success(content);
        res.structured_content = Some(summary);
        Ok(res)
    }

    /// A file of the API, unless it is larger than `max`.
    async fn fetch(&self, path: &str, max: u64) -> Result<Fetched, ApiError> {
        let res = self.api.download(path).await?;
        if let Some(len) = res.content_length().filter(|n| *n > max) {
            return Ok(Fetched::TooLarge(Some(len)));
        }
        let mut bytes = Vec::new();
        let mut stream = res.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ApiError::new("http_error", format!("reading {path}: {e}")))?;
            if (bytes.len() + chunk.len()) as u64 > max {
                return Ok(Fetched::TooLarge(None));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(Fetched::Bytes(bytes))
    }

    async fn start_book_inner(&self, args: StartBookArgs) -> Result<Value, ApiError> {
        let mut params = Map::new();
        let mut detected = None;
        let mut title = args.title;
        let chapters: Vec<Value> = match (args.path, args.chapters) {
            (Some(path), None) => {
                let doc = self.api.rpc("text.import", json!({ "path": path, "language": "auto" })).await?;
                detected = doc["language"].as_str().map(str::to_string);
                title = title.or_else(|| doc["title"].as_str().filter(|t| !t.is_empty()).map(str::to_string));
                title = title.or_else(|| doc["name"].as_str().map(str::to_string));
                params.insert("source".into(), doc["path"].clone());
                params.insert("format".into(), doc["format"].clone());
                let all = doc["chapters"].as_array().cloned().unwrap_or_default();
                let picked = match &args.chapter_indexes {
                    Some(idx) => idx.iter().filter_map(|i| all.get(*i).cloned()).collect(),
                    None => all,
                };
                picked.into_iter().map(|c| json!({ "title": c["title"], "text": c["text"] })).collect()
            }
            (None, Some(chapters)) => {
                chapters.into_iter().map(|c| json!({ "title": c.title, "text": c.text })).collect()
            }
            _ => return Err(ApiError::new("bad_request", "give either path or chapters")),
        };
        if chapters.is_empty() {
            return Err(ApiError::new("bad_request", "no chapters to read"));
        }
        let voice = self.voice(args.voice.as_deref()).await?;
        params.insert("title".into(), json!(title.unwrap_or_else(|| "Untitled".into())));
        params.insert("chapters".into(), json!(chapters));
        params.insert("voice_id".into(), voice["id"].clone());
        if let Some(lang) = args.language.or(detected).or_else(|| voice["language"].as_str().map(str::to_string)) {
            params.insert("language".into(), json!(lang));
        }
        let optional = [
            ("output_format", args.format.map(|f| json!(f.as_str()))),
            ("m4b", args.m4b.map(Value::from)),
            ("read_titles", args.read_titles.map(Value::from)),
            ("speed", args.speed.map(Value::from)),
            ("expressiveness", args.expressiveness.map(Value::from)),
            ("quality", args.quality.map(|q| json!(if matches!(q, Quality::Fast) { "fast" } else { "best" }))),
        ];
        for (key, value) in optional {
            if let Some(v) = value {
                params.insert(key.into(), v);
            }
        }
        let mut started = self.api.rpc("book.start", Value::Object(params)).await?;
        started["voice"] = voice["name"].clone();
        Ok(started)
    }

    async fn download_inner(&self, args: DownloadArgs) -> Result<Value, ApiError> {
        let name = file_name(&args.path);
        if name.is_empty() {
            return Err(ApiError::new("bad_request", "path names no file"));
        }
        let dest = match args.dest.map(PathBuf::from) {
            Some(d) if d.is_dir() => d.join(name),
            Some(d) => d,
            None => dirs::download_dir().unwrap_or_else(std::env::temp_dir).join(name),
        };
        if dest.exists() && !args.overwrite.unwrap_or(false) {
            return Err(ApiError::new(
                "exists",
                format!("{} already exists: pass overwrite or another dest", dest.display()),
            ));
        }
        let res = self.api.download(&args.path).await?;
        let size = write_stream(res, &dest).await?;
        Ok(json!({ "path": dest.display().to_string(), "size_bytes": size }))
    }

    async fn upload_inner(&self, args: UploadArgs) -> Result<Value, ApiError> {
        let local = Path::new(&args.local_path);
        if !local.is_file() {
            return Err(ApiError::new("not_found", format!("no file {}", local.display())));
        }
        self.api.upload(local).await
    }

    async fn resources(&self) -> Result<Vec<Resource>, ApiError> {
        let list = self.api.rpc("outputs.list", json!({ "limit": RESOURCE_LIST_LIMIT })).await?;
        let mut out = Vec::new();
        for entry in list["results"].as_array().into_iter().flatten() {
            let what = match entry["kind"].as_str() {
                Some("redub") => "Redub result",
                _ => "Speech",
            };
            let mut about = vec![what.to_string()];
            if let Some(v) = entry["voice_name"].as_str() {
                about.push(format!("voice {v}"));
            }
            if let Some(d) = entry["duration_s"].as_f64() {
                about.push(format!("{d:.1} s"));
            }
            if let Some(t) = entry["text"].as_str() {
                let short: String = t.chars().take(120).collect();
                about.push(format!("\"{short}{}\"", if short.len() < t.len() { "..." } else { "" }));
            }
            for key in ["path", "video_path"] {
                if let Some(path) = entry[key].as_str() {
                    out.push(
                        Resource::new(file_uri(path), file_name(path))
                            .with_mime_type(mime_for(path))
                            .with_description(about.join(", ")),
                    );
                }
            }
        }
        Ok(out)
    }
}

enum Fetched {
    Bytes(Vec<u8>),
    TooLarge(Option<u64>),
}

async fn write_stream(res: reqwest::Response, dest: &Path) -> Result<u64, ApiError> {
    let io = |e: std::io::Error| ApiError::new("io_error", format!("{}: {e}", dest.display()));
    let part = dest.with_file_name(format!("{}.part", dest.file_name().unwrap_or_default().to_string_lossy()));
    let mut file = tokio::fs::File::create(&part).await.map_err(io)?;
    let mut size = 0u64;
    let mut stream = res.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = match chunk {
            Ok(c) => c,
            Err(e) => {
                drop(file);
                let _ = tokio::fs::remove_file(&part).await;
                return Err(ApiError::new("http_error", format!("download interrupted: {e}")));
            }
        };
        file.write_all(&chunk).await.map_err(io)?;
        size += chunk.len() as u64;
    }
    file.sync_all().await.map_err(io)?;
    drop(file);
    tokio::fs::rename(&part, dest).await.map_err(io)?;
    Ok(size)
}

#[tool_router]
impl Mimvox {
    #[tool(description = ENGINE_CALL)]
    async fn engine_call(&self, Parameters(args): Parameters<EngineCallArgs>) -> Result<CallToolResult, ErrorData> {
        let params = args.params.map(Value::Object).unwrap_or_else(|| json!({}));
        Ok(reply(self.api.rpc(&args.method, params).await))
    }

    #[tool(
        description = "List the cloned voices: id, name, language (the one it was recorded in), dataset minutes, \
                       whether it is trained, per-language reference clips.",
        annotations(read_only_hint = true)
    )]
    async fn list_voices(&self) -> Result<CallToolResult, ErrorData> {
        Ok(reply(self.api.rpc("voices.list", json!({})).await))
    }

    #[tool(description = "Speak text in a cloned voice and wait for the result. Saves the audio in Mimvox's outputs \
                          (it shows in the app's Results) and returns its path and a mimvox://file/ link, plus the \
                          audio itself when it is at most 4 MB. Long texts can take minutes; progress is reported \
                          per segment. Use only the user's own voice or one they have permission to use.")]
    async fn speak(
        &self,
        Parameters(args): Parameters<SpeakArgs>,
        meta: RequestMetaObject,
        peer: Peer<RoleServer>,
        cancel: CancellationToken,
    ) -> Result<CallToolResult, ErrorData> {
        let mut progress = Progress::new(&meta, peer);
        Ok(self.speak_inner(args, progress.as_mut(), &cancel).await.unwrap_or_else(|e| fail(&e)))
    }

    #[tool(
        description = "Read a document (txt, md, html, epub, docx, odt, rtf, fb2, pdf, srt, vtt) on the API's \
                       machine into chapters: title, detected language, words, estimated reading time per chapter, \
                       metadata and a suggested split of long chapters. Use it before start_book to pick chapters.",
        annotations(read_only_hint = true)
    )]
    async fn import_text(&self, Parameters(args): Parameters<ImportTextArgs>) -> Result<CallToolResult, ErrorData> {
        let mut params = json!({ "path": args.path });
        if let Some(lang) = args.language {
            params["language"] = json!(lang);
        }
        let res = self.api.rpc("text.import", params).await;
        Ok(reply(if args.include_text.unwrap_or(false) { res } else { res.map(without_texts) }))
    }

    #[tool(description = "Start reading an audiobook: a document (path) or given chapters, one audio file per \
                          chapter in <data>/outputs/books/<title>/. Returns the book id and its job; follow it with \
                          wait_job (progress in percent) or book_status. Only one generation runs at a time: while \
                          a book is read, speak answers busy until it is paused.")]
    async fn start_book(&self, Parameters(args): Parameters<StartBookArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(reply(self.start_book_inner(args).await))
    }

    #[tool(
        description = "One book's status (percent, chapters done, time left, chapter files) or, without id, every \
                       book.",
        annotations(read_only_hint = true)
    )]
    async fn book_status(&self, Parameters(args): Parameters<BookStatusArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(reply(match args.id {
            Some(id) => self.api.rpc("book.get", json!({ "id": id })).await,
            None => self.api.rpc("book.list", json!({})).await,
        }))
    }

    #[tool(description = "Pause a book that is being read (it can be resumed later, also after a restart).")]
    async fn pause_book(&self, Parameters(args): Parameters<BookIdArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(reply(self.api.rpc("book.pause", json!({ "id": args.id })).await))
    }

    #[tool(description = "Resume a paused, failed or cancelled book from where it stopped. Returns its new job.")]
    async fn resume_book(&self, Parameters(args): Parameters<BookIdArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(reply(self.api.rpc("book.resume", json!({ "id": args.id })).await))
    }

    #[tool(
        description = "The latest known state of a background job (speech, book, training, Redub, reference from \
                       media) as seen by this server since it started: running, done, paused or error, with its \
                       latest progress and its result.",
        annotations(read_only_hint = true)
    )]
    async fn job_status(&self, Parameters(args): Parameters<JobArgs>) -> Result<CallToolResult, ErrorData> {
        let _ = self.events.ready(Duration::from_secs(2)).await;
        Ok(match self.events.job(&args.job) {
            Some(st) => ok(json!(st)),
            None => ok(json!({
                "job": args.job,
                "state": "unknown",
                "connected": self.events.is_connected(),
                "message": "no event of this job since this server started; it may not have reported yet, or it \
                            ended earlier (for a book use book_status)",
            })),
        })
    }

    #[tool(
        description = "Wait until a background job ends (done, paused or error) and return its result, sending \
                       progress notifications meanwhile. On timeout the job keeps running and the answer says so.",
        annotations(read_only_hint = true)
    )]
    async fn wait_job(
        &self,
        Parameters(args): Parameters<WaitJobArgs>,
        meta: RequestMetaObject,
        peer: Peer<RoleServer>,
        cancel: CancellationToken,
    ) -> Result<CallToolResult, ErrorData> {
        if let Err(e) = self.events_ready().await {
            return Ok(fail(&e));
        }
        let mut progress = Progress::new(&meta, peer);
        let timeout = timeout_of(args.timeout_s, WAIT_TIMEOUT_S);
        Ok(match self.wait(&args.job, timeout, progress.as_mut(), None, &cancel).await {
            Waited::Ended(st) if st.state == State::Error => {
                let r = st.result.clone().unwrap_or(Value::Null);
                let e = ApiError::new(
                    r["code"].as_str().unwrap_or("internal"),
                    r["message"].as_str().unwrap_or("the job failed"),
                );
                let mut res = fail(&e);
                res.structured_content = Some(json!({ "error": e.to_json(), "job": st }));
                res
            }
            Waited::Ended(st) => ok(json!(st)),
            Waited::TimedOut(st) => ok(json!({
                "job": args.job,
                "state": "running",
                "timed_out": true,
                "progress": st.and_then(|s| s.progress),
            })),
            Waited::Cancelled => fail(&ApiError::new("cancelled", "the wait was cancelled (the job goes on)")),
        })
    }

    #[tool(description = "Upload a local file to the API's machine (needed when the API runs on another computer) \
                          and return its path there, to pass to methods that take files. Uploads are deleted after \
                          24 hours.")]
    async fn upload_file(&self, Parameters(args): Parameters<UploadArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(reply(self.upload_inner(args).await))
    }

    #[tool(description = "Download a file the API serves (a result in <data>/outputs, a file in <data>/tmp or an \
                          upload) to this machine.")]
    async fn download_file(&self, Parameters(args): Parameters<DownloadArgs>) -> Result<CallToolResult, ErrorData> {
        Ok(reply(self.download_inner(args).await))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Mimvox {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().enable_resources().build())
            .with_server_info(Implementation::new("mimvox-mcp", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        self.resources()
            .await
            .map(ListResourcesResult::with_all_items)
            .map_err(|e| ErrorData::internal_error(e.to_string(), Some(e.to_json())))
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, ErrorData> {
        let file = ResourceTemplate::new(format!("{FILE_URI}{{path}}"), "file").with_description(
            "A file the Mimvox API serves (under <data>/outputs, <data>/tmp or <data>/api/uploads); {path} is its \
             absolute path, percent-encoded.",
        );
        Ok(ListResourceTemplatesResult::with_all_items(vec![file]))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let uri = request.uri;
        let Some(path) = path_of_uri(&uri) else {
            return Err(ErrorData::invalid_params(format!("not a {FILE_URI} URI: {uri}"), None));
        };
        let fetched = self.fetch(&path, RESOURCE_MAX).await.map_err(|e| match e.code.as_str() {
            "not_found" => ErrorData::resource_not_found(e.to_string(), Some(json!({ "uri": uri }))),
            _ => ErrorData::internal_error(e.to_string(), Some(e.to_json())),
        })?;
        let bytes = match fetched {
            Fetched::Bytes(b) => b,
            Fetched::TooLarge(_) => {
                return Err(ErrorData::invalid_params(
                    format!("{path} is larger than {} MB: use the download_file tool", RESOURCE_MAX >> 20),
                    None,
                ));
            }
        };
        let mime = mime_for(&path);
        let contents = match String::from_utf8(bytes) {
            Ok(text) if mime.starts_with("text/") || mime == "application/json" => ResourceContents::text(text, &uri),
            Ok(text) => ResourceContents::blob(BASE64.encode(text.as_bytes()), &uri),
            Err(e) => ResourceContents::blob(BASE64.encode(e.as_bytes()), &uri),
        }
        .with_mime_type(mime);
        Ok(ReadResourceResult::new(vec![contents]).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uris_round_trip() {
        for p in ["/home/u/.local/share/mimvox/outputs/20261111_101112.wav", r"C:\Users\Ана\outputs\a b%.mp3"] {
            let uri = file_uri(p);
            assert!(uri.starts_with(FILE_URI) && !uri[FILE_URI.len()..].contains('/'), "{uri}");
            assert_eq!(path_of_uri(&uri).as_deref(), Some(p));
        }
        assert_eq!(path_of_uri("mimvox://file/%zz"), None);
        assert_eq!(path_of_uri("file:///x"), None);
    }

    #[test]
    fn import_without_texts() {
        let doc = json!({ "chapters": [{ "title": "One", "text": "...", "words": 1 }], "parts": null });
        assert_eq!(without_texts(doc)["chapters"][0], json!({ "title": "One", "words": 1, "index": 0 }));
    }
}
