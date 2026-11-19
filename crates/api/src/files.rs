use std::path::{Component, Path, PathBuf, Prefix};
use std::time::{Duration, SystemTime};

use axum::Json;
use axum::body::Body;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio_util::io::ReaderStream;

use crate::{ApiError, AppState};

pub const UPLOAD_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);
const NAME_MAX_BYTES: usize = 200;

pub fn uploads_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("api").join("uploads")
}

fn served_roots(data_dir: &Path) -> [PathBuf; 3] {
    [data_dir.join("outputs"), data_dir.join("tmp"), uploads_dir(data_dir)]
}

pub fn prune_uploads(data_dir: &Path) {
    let Ok(entries) = std::fs::read_dir(uploads_dir(data_dir)) else {
        return;
    };
    let Some(cutoff) = SystemTime::now().checked_sub(UPLOAD_MAX_AGE) else {
        return;
    };
    for entry in entries.flatten() {
        let old =
            entry.metadata().ok().filter(|m| m.is_file()).and_then(|m| m.modified().ok()).is_some_and(|t| t < cutoff);
        if old {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The last path component, letters, digits, `.`, `-` and `_` only, at most [`NAME_MAX_BYTES`] bytes.
pub fn safe_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or("");
    let cleaned: String =
        base.chars().map(|c| if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' }).collect();
    let cleaned = cleaned.trim_start_matches('.');
    let cleaned = match cleaned.rsplit_once('.').filter(|(_, ext)| (1..=16).contains(&ext.len())) {
        _ if cleaned.len() <= NAME_MAX_BYTES => cleaned.to_string(),
        Some((stem, ext)) => format!("{}.{ext}", cut(stem, NAME_MAX_BYTES - ext.len() - 1)),
        None => cut(cleaned, NAME_MAX_BYTES).to_string(),
    };
    if cleaned.is_empty() { "upload".to_string() } else { cleaned }
}

fn cut(s: &str, max: usize) -> &str {
    let mut end = max.min(s.len());
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// `path` with `.` and `..` resolved without touching the filesystem; `None` unless it is absolute, and for Windows
/// verbatim (`\\?\`) and device (`\\.\`) paths.
fn lexical(path: &Path) -> Option<PathBuf> {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(p) if matches!(p.kind(), Prefix::Disk(_) | Prefix::UNC(..)) => out.push(component),
            Component::Prefix(_) => return None,
            Component::RootDir => out.push(component),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(name) => out.push(name),
        }
    }
    out.is_absolute().then_some(out)
}

pub fn content_type(path: &Path) -> &'static str {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "wav" | "wave" => "audio/wav",
        "mp3" => "audio/mpeg",
        "flac" => "audio/flac",
        "ogg" | "oga" | "opus" => "audio/ogg",
        "m4a" | "m4b" => "audio/mp4",
        "aac" => "audio/aac",
        "aif" | "aiff" => "audio/aiff",
        "mp4" => "video/mp4",
        "m4v" => "video/x-m4v",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "webm" => "video/webm",
        "avi" => "video/x-msvideo",
        "json" => "application/json",
        "txt" => "text/plain; charset=utf-8",
        "srt" => "application/x-subrip",
        "vtt" => "text/vtt",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        _ => "application/octet-stream",
    }
}

#[derive(Deserialize)]
pub struct UploadQuery {
    name: Option<String>,
}

pub async fn upload(
    State(st): State<AppState>,
    Query(q): Query<UploadQuery>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<Value>, ApiError> {
    let max = st.cfg.max_upload_bytes;
    let too_large =
        || ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "too_large", format!("uploads are limited to {max} bytes"));
    let declared =
        headers.get(header::CONTENT_LENGTH).and_then(|v| v.to_str().ok()).and_then(|v| v.parse::<u64>().ok());
    if declared.is_some_and(|n| n > max) {
        return Err(too_large());
    }
    let dir = uploads_dir(&st.cfg.data_dir);
    tokio::fs::create_dir_all(&dir).await.map_err(ApiError::internal)?;
    let path = dir.join(format!("{}-{}", uuid::Uuid::new_v4(), safe_name(q.name.as_deref().unwrap_or(""))));
    let mut file = tokio::fs::File::create(&path).await.map_err(ApiError::internal)?;

    let mut size = 0u64;
    let mut stream = body.into_data_stream();
    let written: Result<(), ApiError> = async {
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| ApiError::bad_request(format!("the upload was cut off: {e}")))?;
            size += chunk.len() as u64;
            if size > max {
                return Err(too_large());
            }
            file.write_all(&chunk).await.map_err(ApiError::internal)?;
        }
        file.flush().await.map_err(ApiError::internal)
    }
    .await;
    drop(file);
    if let Err(e) = written {
        let _ = tokio::fs::remove_file(&path).await;
        return Err(e);
    }

    let data_dir = st.cfg.data_dir.clone();
    tokio::task::spawn_blocking(move || prune_uploads(&data_dir));
    Ok(Json(json!({ "path": path, "size_bytes": size })))
}

#[derive(Deserialize)]
pub struct DownloadQuery {
    path: String,
}

pub async fn download(State(st): State<AppState>, Query(q): Query<DownloadQuery>) -> Result<Response, ApiError> {
    let requested = Path::new(&q.path);
    if !requested.is_absolute() {
        return Err(ApiError::bad_request("path must be absolute"));
    }
    let not_found = || ApiError::new(StatusCode::NOT_FOUND, "not_found", "no such file");
    let forbidden = || {
        ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "only files under the data folder's outputs, tmp and api/uploads are served",
        )
    };
    let roots = served_roots(&st.cfg.data_dir);
    // Checked before any filesystem access: on Windows even resolving a UNC path connects to (and authenticates with)
    // that host.
    let path = lexical(requested).ok_or_else(forbidden)?;
    let mut real_roots = Vec::new();
    for root in &roots {
        if let Ok(real) = tokio::fs::canonicalize(root).await {
            real_roots.push(real);
        }
    }
    let lexical_roots: Vec<PathBuf> = roots.iter().filter_map(|r| lexical(r)).collect();
    if !lexical_roots.iter().chain(&real_roots).any(|r| path.starts_with(r)) {
        return Err(forbidden());
    }
    let real = tokio::fs::canonicalize(&path).await.map_err(|_| not_found())?;
    if !real_roots.iter().any(|r| real.starts_with(r)) {
        return Err(forbidden());
    }
    let file = tokio::fs::File::open(&real).await.map_err(|_| not_found())?;
    let meta = file.metadata().await.map_err(ApiError::internal)?;
    if !meta.is_file() {
        return Err(ApiError::bad_request("not a file"));
    }
    let headers =
        [(header::CONTENT_TYPE, content_type(&real).to_string()), (header::CONTENT_LENGTH, meta.len().to_string())];
    Ok((headers, Body::from_stream(ReaderStream::new(file))).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_made_safe() {
        assert_eq!(safe_name("voice.wav"), "voice.wav");
        assert_eq!(safe_name("../../etc/passwd"), "passwd");
        assert_eq!(safe_name("C:\\Users\\me\\my take (1).mp3"), "my_take__1_.mp3");
        assert_eq!(safe_name("..."), "upload");
        assert_eq!(safe_name(""), "upload");
        assert_eq!(safe_name(".hidden"), "hidden");
        assert_eq!(safe_name("книга.epub"), "книга.epub");
        let long = format!("{}.flac", "a".repeat(300));
        let short = safe_name(&long);
        assert_eq!(short.len(), NAME_MAX_BYTES);
        assert!(short.ends_with(".flac"));
        let cjk = safe_name(&format!("{}.mp3", "書".repeat(300)));
        assert!(cjk.len() <= NAME_MAX_BYTES && cjk.ends_with(".mp3"), "{cjk}");
        let no_ext = safe_name(&"ж".repeat(300));
        assert!(no_ext.len() <= NAME_MAX_BYTES && no_ext.chars().all(|c| c == 'ж'));
    }

    #[test]
    fn lexical_paths() {
        let p = |s: &str| lexical(Path::new(s));
        assert_eq!(p("relative/x"), None);
        #[cfg(unix)]
        {
            assert_eq!(p("/data/outputs/../secret"), Some(PathBuf::from("/data/secret")));
            assert_eq!(p("/data/./outputs/a.wav"), Some(PathBuf::from("/data/outputs/a.wav")));
            assert_eq!(p("/../../etc/passwd"), Some(PathBuf::from("/etc/passwd")));
        }
        #[cfg(windows)]
        {
            assert_eq!(p(r"C:\data\outputs\..\secret"), Some(PathBuf::from(r"C:\data\secret")));
            assert_eq!(p(r"\\?\C:\data\outputs\a.wav"), None);
            assert_eq!(p(r"\\?\UNC\host\share\a.wav"), None);
            assert_eq!(p(r"\\.\pipe\x"), None);
            assert!(!p(r"\\attacker\s\x").unwrap().starts_with(r"C:\data\outputs"));
        }
    }
}
