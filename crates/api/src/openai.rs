use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use mimvox_engine_client::protocol::EngineError;
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{OwnedMutexGuard, oneshot};

use crate::files::content_type;
use crate::hub::{EngineEvent, Hub};
use crate::{ApiError, AppState};

pub const MAX_INPUT_CHARS: usize = 4096;
const SPEECH_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const CANCEL_GRACE: Duration = Duration::from_secs(30);
const PCM_RATE: u32 = 24_000;

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

const ENGINE_MODELS: [&str; 3] = ["omnivoice", "voxcpm", "chatterbox"];

#[derive(Deserialize)]
pub struct SpeechRequest {
    input: String,
    voice: Value,
    model: Option<String>,
    response_format: Option<String>,
    speed: Option<f64>,
    /// Not in OpenAI's API: the language of `input`; the voice's own language by default.
    language: Option<String>,
}

pub async fn models() -> Json<Value> {
    Json(json!({ "object": "list", "data": std::iter::once("mimvox").chain(ENGINE_MODELS).map(|id| {
            json!({ "id": id, "object": "model", "created": 0, "owned_by": "mimvox" })
        }).collect::<Vec<_>>() }))
}

pub async fn voices(State(st): State<AppState>) -> Result<Json<Value>, OpenAiError> {
    Ok(Json(st.hub.call("voices.list", json!({})).await?))
}

/// The engine's `output_format` for an OpenAI `response_format`; `pcm` is made from a WAV.
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

/// A chunk of a WAV file by its id.
fn wav_chunk<'a>(wav: &'a [u8], id: &[u8; 4]) -> Option<&'a [u8]> {
    if wav.len() < 12 || &wav[0..4] != b"RIFF" || &wav[8..12] != b"WAVE" {
        return None;
    }
    let mut at = 12usize;
    while at + 8 <= wav.len() {
        let size = u32::from_le_bytes(wav[at + 4..at + 8].try_into().ok()?) as usize;
        let start = at + 8;
        if &wav[at..at + 4] == id {
            return Some(&wav[start..start.saturating_add(size).min(wav.len())]);
        }
        at = start.checked_add(size)?.checked_add(size & 1)?;
    }
    None
}

/// OpenAI's `pcm` from a 16-bit PCM WAV: mono, 24 kHz, 16-bit little-endian.
fn openai_pcm(wav: &[u8]) -> Option<Vec<u8>> {
    let fmt = wav_chunk(wav, b"fmt ")?;
    let field = |at: usize| Some(u16::from_le_bytes(fmt.get(at..at + 2)?.try_into().ok()?));
    let (tag, channels, bits) = (field(0)?, usize::from(field(2)?), field(14)?);
    let rate = u32::from_le_bytes(fmt.get(4..8)?.try_into().ok()?);
    // WAVE_FORMAT_EXTENSIBLE keeps the real format tag at the start of its sub-format GUID.
    let tag = if tag == 0xFFFE { field(24)? } else { tag };
    if tag != 1 || bits != 16 || channels == 0 || rate == 0 {
        return None;
    }
    let samples: Vec<f32> = wav_chunk(wav, b"data")?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|s| f32::from(i16::from_le_bytes(*s)) / 32768.0)
        .collect();
    let mono: Vec<f32> = samples.chunks_exact(channels).map(|f| f.iter().sum::<f32>() / channels as f32).collect();
    let samples = if rate == PCM_RATE || mono.is_empty() {
        mono
    } else {
        let mut resampler = Fft::<f32>::new(rate as usize, PCM_RATE as usize, 1024, 1, FixedSync::Input).ok()?;
        let input = InterleavedSlice::new(&mono, 1, mono.len()).ok()?;
        let mut out = resampler.process_all(&input, mono.len(), None).ok()?.take_data();
        // rubato rounds its float ratio up, which can add a frame.
        out.truncate(((mono.len() as u64 * u64::from(PCM_RATE) + u64::from(rate) / 2) / u64::from(rate)) as usize);
        out
    };
    Some(samples.iter().flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()).collect())
}

fn find_voice<'a>(voices: &'a [Value], wanted: &str) -> Option<&'a Value> {
    let wanted = wanted.trim();
    voices
        .iter()
        .find(|v| v["id"].as_str() == Some(wanted))
        .or_else(|| voices.iter().find(|v| v["name"].as_str().is_some_and(|n| n.trim().eq_ignore_ascii_case(wanted))))
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
    if req.input.chars().count() > MAX_INPUT_CHARS {
        return Err(ApiError::bad_request(format!("input is limited to {MAX_INPUT_CHARS} characters")).into());
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
    if let Some(model) = req.model.as_deref().filter(|m| ENGINE_MODELS.contains(m)) {
        params["model"] = json!(model);
    }

    // OpenAI's aac is raw ADTS; the engine only writes AAC in MP4.
    let adts = match response_format {
        "aac" => Some(crate::ffmpeg::find(&st.cfg.data_dir).ok_or_else(|| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "not_ready",
                "response_format aac needs ffmpeg, which was not found (re-run setup, or install ffmpeg)",
            )
        })?),
        _ => None,
    };

    // One generation at a time in the engine: queue here instead of answering busy.
    let turn = st.speech.clone().lock_owned().await;
    let (tx, rx) = oneshot::channel();
    tokio::spawn(run_job(st.hub.clone(), params, format, adts, turn, tx));
    let (audio, saved) = rx.await.map_err(|_| ApiError::internal("the speech task ended early"))??;

    if response_format == "pcm" && saved == "wav" {
        let pcm = openai_pcm(&audio).ok_or_else(|| ApiError::internal("the engine's WAV is not 16-bit PCM"))?;
        return Ok(([(header::CONTENT_TYPE, "audio/pcm")], pcm).into_response());
    }
    let ctype = content_type(Path::new(&format!("x.{saved}")));
    Ok(([(header::CONTENT_TYPE, ctype)], audio).into_response())
}

type JobResult = Result<(Vec<u8>, String), ApiError>;

/// Holds the speech turn until the job has ended, also when the client left, so the next request never meets a
/// running job; a job given up on is cancelled and its output deleted.
async fn run_job(
    hub: Hub,
    params: Value,
    format: &str,
    adts: Option<PathBuf>,
    _turn: OwnedMutexGuard<()>,
    mut tx: oneshot::Sender<JobResult>,
) {
    let mut events = hub.subscribe();
    let job = match hub.call("tts.generate", params).await {
        Ok(started) => match started["job"].as_str() {
            Some(job) => job.to_string(),
            None => {
                let _ = tx.send(Err(ApiError::internal("tts.generate answered without a job")));
                return;
            }
        },
        Err(e) => {
            let _ = tx.send(Err(e.into()));
            return;
        }
    };

    let timer = tokio::time::sleep(SPEECH_TIMEOUT);
    tokio::pin!(timer);
    let mut gave_up: Option<ApiError> = None;
    let outcome: Result<Value, ApiError> = loop {
        let give_up = tokio::select! {
            _ = tx.closed(), if gave_up.is_none() => Some(ApiError::internal("the client went away")),
            _ = &mut timer => match gave_up.take() {
                Some(e) => break Err(e),
                None => Some(ApiError::new(StatusCode::GATEWAY_TIMEOUT, "timeout", "speech took too long")),
            },
            received = events.recv() => match received {
                Ok(ev) if ev.event == "engine.exited" => break Err(ApiError::internal("the engine stopped while speaking")),
                Ok(ev) if ev.job.as_deref() == Some(job.as_str()) => match ev.event.as_str() {
                    "tts.done" => break Ok(ev.data.clone()),
                    "tts.error" | "job.error" => break Err(gave_up.take().unwrap_or_else(|| job_error(&ev))),
                    _ => None,
                },
                Ok(_) => None,
                // Job ids restart with the engine: after a gap this job's end may be gone, so stop waiting for it.
                Err(RecvError::Lagged(_)) if gave_up.is_none() => Some(ApiError::internal("missed engine events")),
                Err(RecvError::Lagged(_)) => None,
                Err(RecvError::Closed) => break Err(ApiError::internal("the engine went away")),
            },
        };
        if let Some(e) = give_up {
            gave_up = Some(e);
            let _ = hub.call("tts.cancel", json!({ "job": job })).await;
            timer.as_mut().reset(tokio::time::Instant::now() + CANCEL_GRACE);
        }
    };

    let result = match outcome {
        Ok(done) => {
            let path = done["path"].as_str().unwrap_or_default().to_string();
            let mut saved = done["format"].as_str().unwrap_or(format).to_string();
            let audio = match (gave_up, adts) {
                (Some(e), _) => Err(e),
                (None, Some(ffmpeg)) if saved == "m4a" => {
                    saved = "aac".into();
                    crate::ffmpeg::mp4_to_adts(&ffmpeg, Path::new(&path)).await
                }
                (None, _) => tokio::fs::read(&path).await.map_err(ApiError::internal),
            };
            let mut delete = json!({ "path": path });
            if let Some(play) = done["path_play"].as_str().filter(|p| *p != path) {
                delete["path_play"] = json!(play);
            }
            // The result was only made for this response; a failed delete leaves it in the app's Results.
            let _ = hub.call("outputs.delete", delete).await;
            audio.map(|a| (a, saved))
        }
        Err(e) => Err(e),
    };
    let _ = tx.send(result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_data_chunk() {
        let mut wav = b"RIFF\0\0\0\0WAVEfmt \x02\0\0\0abLIST\x01\0\0\0x\0data\x04\0\0\0wxyz".to_vec();
        assert_eq!(wav_chunk(&wav, b"data"), Some(&b"wxyz"[..]));
        wav.truncate(10);
        assert_eq!(wav_chunk(&wav, b"data"), None);
    }

    fn wav_16bit(rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut wav = b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0".to_vec();
        wav.extend(channels.to_le_bytes());
        wav.extend(rate.to_le_bytes());
        wav.extend((rate * 2 * u32::from(channels)).to_le_bytes());
        wav.extend((2 * channels).to_le_bytes());
        wav.extend(16u16.to_le_bytes());
        wav.extend(b"data");
        wav.extend((data.len() as u32).to_le_bytes());
        wav.extend(data);
        wav
    }

    #[test]
    fn pcm_is_24khz_mono() {
        let tone: Vec<i16> = (0..44_100)
            .map(|i| (10_000.0 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / 44_100.0).sin()) as i16)
            .collect();
        let pcm = openai_pcm(&wav_16bit(44_100, 1, &tone)).unwrap();
        assert_eq!(pcm.len(), 2 * 24_000);
        let out: Vec<i16> = pcm.as_chunks::<2>().0.iter().map(|s| i16::from_le_bytes(*s)).collect();
        let crossings = out.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
        assert!((1990..=2010).contains(&crossings), "{crossings}");
        let peak = out[1000..23_000].iter().map(|s| s.unsigned_abs()).max().unwrap();
        assert!((9_500..=10_500).contains(&peak), "{peak}");

        let stereo = wav_16bit(24_000, 2, &[100, 300, -50, -150]);
        assert_eq!(openai_pcm(&stereo).unwrap(), [200i16.to_le_bytes(), (-100i16).to_le_bytes()].concat());
        let mut float = stereo.clone();
        float[20] = 3;
        assert_eq!(openai_pcm(&float), None);
    }

    #[test]
    fn voice_by_id_or_name() {
        let voices = vec![json!({"id": "v1", "name": "Anna"}), json!({"id": "v2", "name": "v1"})];
        assert_eq!(find_voice(&voices, "v1").unwrap()["id"], "v1");
        assert_eq!(find_voice(&voices, " anna ").unwrap()["id"], "v1");
        assert!(find_voice(&voices, "nobody").is_none());
    }
}
