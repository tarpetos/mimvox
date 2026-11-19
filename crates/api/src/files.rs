use std::path::{Path, PathBuf};
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
const NAME_MAX_CHARS: usize = 100;

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

/// The last path component, letters, digits, `.`, `-` and `_` only, at most [`NAME_MAX_CHARS`] characters.
pub fn safe_name(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or("");
    let cleaned: String =
        base.chars().map(|c| if c.is_alphanumeric() || matches!(c, '.' | '-' | '_') { c } else { '_' }).collect();
    let cleaned = cleaned.trim_start_matches('.');
    let cleaned = if cleaned.chars().count() > NAME_MAX_CHARS {
        match cleaned.rsplit_once('.').filter(|(_, ext)| (1..=10).contains(&ext.chars().count())) {
            Some((stem, ext)) => {
                let keep = NAME_MAX_CHARS - ext.chars().count() - 1;
                format!("{}.{ext}", stem.chars().take(keep).collect::<String>())
            }
            None => cleaned.chars().take(NAME_MAX_CHARS).collect(),
        }
    } else {
        cleaned.to_string()
    };
    if cleaned.is_empty() { "upload".to_string() } else { cleaned }
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
    let real = tokio::fs::canonicalize(requested).await.map_err(|_| not_found())?;
    let mut inside = false;
    for root in served_roots(&st.cfg.data_dir) {
        if let Ok(root) = tokio::fs::canonicalize(&root).await {
            inside |= real.starts_with(&root);
        }
    }
    if !inside {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "forbidden",
            "only files under the data folder's outputs, tmp and api/uploads are served",
        ));
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
        let cut = safe_name(&long);
        assert_eq!(cut.chars().count(), NAME_MAX_CHARS);
        assert!(cut.ends_with(".flac"));
    }
}
