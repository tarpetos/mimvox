use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use mimvox_engine_client::protocol::EngineError;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast::error::RecvError;

use crate::files::content_type;
use crate::hub::{EngineEvent, Hub};
use crate::{ApiError, AppState};

/// OpenAI's error shape: `{"error":{"message","type","code"}}`.
pub struct OpenAiError(ApiError);

impl<E: Into<ApiError>> From<E> for OpenAiError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for OpenAiError {
    fn into_response(self) -> Response {
        let kind = if self.0.status.is_client_error() { "invalid_request_error" } else { "server_error" };
        let body = json!({ "error": { "message": self.0.message, "type": kind, "code": self.0.code } });
        (self.0.status, Json(body)).into_response()
    }
}

#[derive(Deserialize)]
pub struct SpeechRequest {
    input: String,
    voice: Value,
    response_format: Option<String>,
    speed: Option<f64>,
    /// Not in OpenAI's API: the language of `input`; the voice's own language by default.
    language: Option<String>,
}

pub async fn models() -> Json<Value> {
    Json(
        json!({ "object": "list", "data": [{ "id": "mimvox", "object": "model", "created": 0, "owned_by": "mimvox" }] }),
    )
}

pub async fn voices(State(st): State<AppState>) -> Result<Json<Value>, OpenAiError> {
    Ok(Json(st.hub.call("voices.list", json!({})).await?))
}

/// The engine's `output_format` for an OpenAI `response_format`; `pcm` is cut out of a WAV.
fn engine_format(response_format: &str) -> Option<&'static str> {
    Some(match response_format {
        "mp3" => "mp3",
        "wav" | "pcm" => "wav",
        "flac" => "flac",
        "opus" => "opus",
        "aac" => "m4a",
        _ => return None,
    })
}

/// The samples of a PCM WAV file (its `data` chunk).
pub fn wav_samples(wav: &[u8]) -> Option<&[u8]> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12usize;
    while at + 8 <= wav.len() {
        let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().ok()?) as usize;
        let start = at + 8;
        if &wav[at..at + 4] == b"data" {
            return Some(&wav[start..start.saturating_add(size).min(wav.len())]);
        }
        at = start.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

fn find_voice<'a>(voices: &'a [Value], wanted: &str) -> Option<&'a Value> {
    let wanted = wanted.trim();
    voices
        .iter()
        .find(|v| v["id"].as_str() == Some(wanted))
        .or_else(|| voices.iter().find(|v| v["name"].as_str().is_some_and(|n| n.trim().eq_ignore_ascii_case(wanted))))
}

struct CancelOnDrop {
    hub: Hub,
    job: Option<String>,
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        let (Some(job), Ok(rt)) = (self.job.take(), tokio::runtime::Handle::try_current()) else {
            return;
        };
        let hub = self.hub.clone();
        rt.spawn(async move {
            let _ = hub.call("tts.cancel", json!({ "job": job })).await;
        });
    }
}

fn job_error(ev: &EngineEvent) -> ApiError {
    let code = ev.data["code"].as_str().unwrap_or("internal");
    let message = ev.data["message"].as_str().unwrap_or("speech failed");
    EngineError::new(code, message).into()
}

pub async fn speech(
    State(st): State<AppState>,
    req: Result<Json<SpeechRequest>, JsonRejection>,
) -> Result<Response, OpenAiError> {
    let Json(req) = req.map_err(ApiError::from)?;
    let response_format = req.response_format.as_deref().unwrap_or("mp3");
    let format = engine_format(response_format).ok_or_else(|| {
        ApiError::bad_request(format!(
            "response_format must be one of mp3, opus, aac, flac, wav, pcm (got {response_format})"
        ))
    })?;
    if req.input.trim().is_empty() {
        return Err(ApiError::bad_request("input is empty").into());
    }
    let wanted = match &req.voice {
        Value::String(s) => s.clone(),
        Value::Object(o) => o.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
        _ => String::new(),
    };

    let listed = st.hub.call("voices.list", json!({})).await?;
    let voices = listed["voices"].as_array().cloned().unwrap_or_default();
    let voice = find_voice(&voices, &wanted).ok_or_else(|| {
        ApiError::new(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("no voice with the id or name {wanted:?} (GET /v1/audio/voices lists them)"),
        )
    })?;
    let language =
        req.language.clone().or_else(|| voice["language"].as_str().map(str::to_string)).unwrap_or_else(|| "en".into());
    let mut params = json!({
        "voice_id": voice["id"],
        "text": req.input,
        "language": language,
        "output_format": format,
    });
    if let Some(speed) = req.speed {
        params["speed"] = json!(speed);
    }

    // One generation at a time in the engine: queue here instead of answering busy.
    let _turn = st.speech.lock().await;
    let mut events = st.hub.subscribe();
    let started = st.hub.call("tts.generate", params).await?;
    let job =
        started["job"].as_str().ok_or_else(|| ApiError::internal("tts.generate answered without a job"))?.to_string();
    let mut cancel = CancelOnDrop { hub: st.hub.clone(), job: Some(job.clone()) };
    let done: Arc<EngineEvent> = loop {
        match events.recv().await {
            Ok(ev) if ev.event == "engine.exited" => {
                return Err(ApiError::internal("the engine stopped while speaking").into());
            }
            Ok(ev) if ev.job.as_deref() == Some(job.as_str()) => match ev.event.as_str() {
                "tts.done" => break ev,
                "tts.error" | "job.error" => return Err(job_error(&ev).into()),
                _ => {}
            },
            Ok(_) | Err(RecvError::Lagged(_)) => {}
            Err(RecvError::Closed) => return Err(ApiError::internal("the engine went away").into()),
        }
    };
    cancel.job = None;

    let path = done.data["path"].as_str().unwrap_or_default().to_string();
    let audio = tokio::fs::read(&path).await.map_err(ApiError::internal)?;
    let mut delete = json!({ "path": path });
    if let Some(play) = done.data["path_play"].as_str().filter(|p| *p != path) {
        delete["path_play"] = json!(play);
    }
    // The result was only made for this response; a failed delete leaves it in the app's Results.
    let _ = st.hub.call("outputs.delete", delete).await;

    let saved = done.data["format"].as_str().unwrap_or(format);
    if response_format == "pcm" && saved == "wav" {
        let samples = wav_samples(&audio).ok_or_else(|| ApiError::internal("the engine's WAV has no data chunk"))?;
        return Ok(([(header::CONTENT_TYPE, "audio/pcm")], samples.to_vec()).into_response());
    }
    let ctype = content_type(std::path::Path::new(&format!("x.{saved}")));
    Ok(([(header::CONTENT_TYPE, ctype)], audio).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_data_chunk() {
        let mut wav = b"RIFF\0\0\0\0WAVEfmt \x02\0\0\0abLIST\x01\0\0\0x\0data\x04\0\0\0wxyz".to_vec();
        assert_eq!(wav_samples(&wav), Some(&b"wxyz"[..]));
        wav.truncate(10);
        assert_eq!(wav_samples(&wav), None);
    }

    #[test]
    fn voice_by_id_or_name() {
        let voices = vec![json!({"id": "v1", "name": "Anna"}), json!({"id": "v2", "name": "v1"})];
        assert_eq!(find_voice(&voices, "v1").unwrap()["id"], "v1");
        assert_eq!(find_voice(&voices, " anna ").unwrap()["id"], "v1");
        assert!(find_voice(&voices, "nobody").is_none());
    }
}
