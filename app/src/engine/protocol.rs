//! Typed mirror of contract/engine-protocol.md.

#![allow(dead_code)]

use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

mod de {
    use super::*;

    pub fn f32_or_zero<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
        Ok(Option::<f32>::deserialize(d)?.unwrap_or(0.0))
    }

    pub fn u32_or_zero<'de, D: Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
        Ok(Option::<u32>::deserialize(d)?.unwrap_or(0))
    }

    /// Timestamps arrive as a string, a number or null.
    pub fn string_or_number<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
        Ok(match Value::deserialize(d)? {
            Value::String(s) => s,
            Value::Null => String::new(),
            other => other.to_string(),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineError {
    pub code: String,
    pub message: String,
}

impl EngineError {
    pub fn new(code: &str, message: impl Into<String>) -> Self {
        Self { code: code.to_string(), message: message.into() }
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

impl std::error::Error for EngineError {}

/// Raw event line: `{"event":"...","job":"..."|null,"data":{...}}`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Event {
    pub event: String,
    #[serde(default)]
    pub job: Option<String>,
    #[serde(default)]
    pub data: Value,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Ready,
    Loading,
    /// Unloaded while a voice trains; back to `Ready` after it.
    Paused,
    Missing,
    #[default]
    #[serde(other)]
    Unknown,
}

impl ModelState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Loading => "loading",
            Self::Paused => "paused",
            Self::Missing => "missing",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Ok,
    Warn,
    Bad,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    TooShort,
    ZeroShot,
    Trainable,
    Great,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainStage {
    Preprocess,
    Extract,
    Train,
    Index,
    #[default]
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Device {
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub name: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub vram_gb: f32,
    /// `nvidia` or `amd` for the `cuda` backend, else empty.
    #[serde(default)]
    pub vendor: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Models {
    #[serde(default)]
    pub tts: ModelState,
    #[serde(default)]
    pub rvc: ModelState,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechLanguage {
    /// ISO 639-1 (`yue` for Cantonese).
    pub code: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub native: String,
    #[serde(default)]
    pub common: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Hello {
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub device: Device,
    #[serde(default)]
    pub models: Models,
    #[serde(default)]
    pub languages: Vec<SpeechLanguage>,
    /// Best first (`"nllb-1.3b"`, `"nllb-600m"`).
    #[serde(default)]
    pub translators: Vec<String>,
    #[serde(default)]
    pub zero_shot_vc: bool,
    #[serde(default)]
    pub components: Vec<String>,
    /// Installed but turned off.
    #[serde(default)]
    pub components_off: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Voice {
    pub id: String,
    #[serde(default)]
    pub name: String,
    #[serde(default, deserialize_with = "de::string_or_number")]
    pub created: String,
    #[serde(default)]
    pub preset: Option<String>,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub dataset_minutes: f32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub parts: u32,
    #[serde(default)]
    pub trained: bool,
    #[serde(default)]
    pub trained_epochs: Option<u32>,
    #[serde(default)]
    pub training_paused: Option<PausedTraining>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PausedTraining {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub epoch: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub total_epochs: u32,
    /// Local time without zone, `YYYY-MM-DDTHH:MM:SS`.
    #[serde(default)]
    pub paused_at: Option<String>,
    #[serde(default)]
    pub interrupted: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Part {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub text: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub est_seconds: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Preset {
    /// `quick`, `standard` or `pro`.
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub target_minutes: f32,
    #[serde(default)]
    pub trains_rvc: bool,
    #[serde(default)]
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextAnalysis {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub words: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub est_minutes: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub phoneme_coverage: f32,
    #[serde(default)]
    pub missing_phonemes: Vec<String>,
    #[serde(default)]
    pub level: Level,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RecordingCheck {
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub peak: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub clipping_pct: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub rms_db: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub noise_floor_db: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub snr_db: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub silence_pct: f32,
    #[serde(default)]
    pub verdict: Verdict,
    #[serde(default)]
    pub issues: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PartInfo {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default)]
    pub verdict: Verdict,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetSummary {
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub total_minutes: f32,
    #[serde(default)]
    pub parts: Vec<PartInfo>,
    #[serde(default)]
    pub level: Level,
    #[serde(default)]
    pub message: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub recommended_epochs: u32,
}

/// `voices.list`
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VoicesList {
    #[serde(default)]
    pub voices: Vec<Voice>,
}

/// `voices.create` / `voices.rename`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VoiceResult {
    pub voice: Voice,
}

/// `presets.list`
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PresetsList {
    #[serde(default)]
    pub presets: Vec<Preset>,
}

/// `dataset.add`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetAdded {
    pub part: PartInfo,
    #[serde(default)]
    pub summary: DatasetSummary,
}

/// `dataset.remove`
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DatasetRemoved {
    #[serde(default)]
    pub summary: DatasetSummary,
}

/// `train.start`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JobStarted {
    pub job: String,
}

/// `tts.generate`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsStarted {
    pub job: String,
    /// Equals the number of `tts.chunk` events to expect.
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segments: u32,
    #[serde(default)]
    pub segment_chars: Vec<u32>,
    #[serde(default)]
    pub accent_note: Option<String>,
    #[serde(default)]
    pub reference_note: Option<String>,
}

/// Which reference clip spoke a job (`tts.done`, `redub.done`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReferenceUse {
    /// `own` | `native-us`.
    #[serde(default)]
    pub accent: Option<String>,
    #[serde(default)]
    pub reference_language: Option<String>,
    #[serde(default)]
    pub reference_foreign: bool,
    #[serde(default)]
    pub reference_note: Option<String>,
}

/// `voices.warm`: `warming = false` means no `voice.warm` event follows.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VoiceWarming {
    #[serde(default)]
    pub warming: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

pub fn parse_result<T: DeserializeOwned>(result: Value) -> Result<T, EngineError> {
    serde_json::from_value(result)
        .map_err(|e| EngineError::new("bad_response", format!("unexpected engine response: {e}")))
}

/// `redub.probe`: what the engine found in an audio or video file.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubProbe {
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub sample_rate: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub channels: u32,
    #[serde(default)]
    pub has_audio: bool,
    /// A picture track; cover art does not count.
    #[serde(default)]
    pub has_video: Option<bool>,
    #[serde(default)]
    pub format: String,
    /// A playable WAV copy, set when the file itself is not a WAV.
    #[serde(default)]
    pub preview_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubSegment {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub start: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub end: f32,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub source_text: Option<String>,
    #[serde(default)]
    pub translated: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubTranscript {
    #[serde(default)]
    pub language: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default)]
    pub segments: Vec<RedubSegment>,
    #[serde(default)]
    pub translated: Option<bool>,
    /// The speech language when `language` is that of the translated text.
    #[serde(default)]
    pub source_language: Option<String>,
    #[serde(default)]
    pub target_language: Option<String>,
    /// This transcript is the review of a `redub.dub` with `review: true`.
    #[serde(default)]
    pub review: Option<bool>,
}

/// `text` is the translation, `source_text` the original.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubReviewSegment {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub start: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub end: f32,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub source_text: String,
}

/// `redub.review`: a `redub.dub` with `review: true` stops here, awaiting a `redub.render` respeak.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubReview {
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub source_language: String,
    #[serde(default)]
    pub segments: Vec<RedubReviewSegment>,
}

/// `pct` is 0-100; `segment` / `segments` (1-based k of n) appear only while rendering a re-spoken transcript.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubProgress {
    #[serde(default)]
    pub stage: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub pct: f32,
    #[serde(default)]
    pub segment: Option<u32>,
    #[serde(default)]
    pub segments: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedubDone {
    pub path: PathBuf,
    #[serde(default)]
    pub path_play: Option<PathBuf>,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    /// `"dub"`, `"convert"` or `"respeak"`.
    #[serde(default)]
    pub mode: Option<String>,
    /// Seconds the spoken result ran past the original length (`redub.dub` only).
    #[serde(default)]
    pub drift_s: Option<f32>,
    /// The spoken language for `redub.dub`; respeak reports it in `language`.
    #[serde(default)]
    pub target_language: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub video_path: Option<PathBuf>,
    #[serde(default)]
    pub background_note: Option<String>,
    #[serde(flatten)]
    pub reference: ReferenceUse,
}

impl RedubDone {
    pub fn play_path(&self) -> PathBuf {
        self.path_play.clone().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| self.path.clone())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedubMode {
    /// Only for `redub.dub`; not a `redub.render` mode itself.
    #[default]
    Dub,
    /// Swaps the voice, keeping timing and delivery; needs a trained voice.
    Convert,
    Respeak,
}

impl RedubMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Dub => "dub",
            Self::Convert => "convert",
            Self::Respeak => "respeak",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RedubTiming {
    #[default]
    Keep,
    Natural,
}

impl RedubTiming {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Keep => "keep",
            Self::Natural => "natural",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RedubOutput {
    pub video: bool,
    pub keep_background: bool,
}

impl RedubOutput {
    fn apply(self, params: &mut Value) {
        params["output"] = json!(if self.video { "video" } else { "audio" });
        params["keep_background"] = json!(self.keep_background);
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubSegmentIn {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainProgress {
    #[serde(default)]
    pub stage: TrainStage,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub epoch: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub total_epochs: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub pct: f32,
    #[serde(default)]
    pub eta_s: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainDone {
    #[serde(default)]
    pub voice_id: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub epochs: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
}

/// `train.paused`: the job is over and the GPU is back for speech.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainPaused {
    #[serde(default)]
    pub voice_id: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub epoch: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub total_epochs: u32,
    #[serde(default)]
    pub paused_at: Option<String>,
    /// How far the training got before it stopped (at or after `epoch`).
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub reached_epoch: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsChunk {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    pub path: PathBuf,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segments: u32,
    /// Seconds since the job started.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub ready_s: f32,
    /// Seconds of generation per second of audio.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub rtf: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VoiceWarm {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub ok: bool,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsDone {
    pub path: PathBuf,
    #[serde(default)]
    pub path_play: Option<PathBuf>,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    #[serde(flatten)]
    pub reference: ReferenceUse,
}

impl TtsDone {
    pub fn play_path(&self) -> PathBuf {
        self.path_play.clone().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| self.path.clone())
    }
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    TtsChunk { job: Option<String>, chunk: TtsChunk },
    TtsDone { job: Option<String>, done: TtsDone },
    TtsError { job: Option<String>, error: EngineError },
    TrainProgress { job: Option<String>, progress: TrainProgress },
    TrainDone { job: Option<String>, done: TrainDone },
    TrainPaused { job: Option<String>, paused: TrainPaused },
    TrainError { job: Option<String>, error: EngineError },
    VoiceWarm(VoiceWarm),
    RedubProgress { job: Option<String>, progress: RedubProgress },
    RedubTranscript { job: Option<String>, transcript: RedubTranscript },
    RedubReview { job: Option<String>, review: RedubReview },
    RedubDone { job: Option<String>, done: RedubDone },
    RedubError { job: Option<String>, error: EngineError },
    /// Unknown event or a payload that did not match the contract (kept raw).
    Other(Event),
}

impl EngineEvent {
    pub fn job(&self) -> Option<&str> {
        match self {
            Self::TtsChunk { job, .. }
            | Self::TtsDone { job, .. }
            | Self::TtsError { job, .. }
            | Self::TrainProgress { job, .. }
            | Self::TrainDone { job, .. }
            | Self::TrainPaused { job, .. }
            | Self::TrainError { job, .. }
            | Self::RedubProgress { job, .. }
            | Self::RedubTranscript { job, .. }
            | Self::RedubReview { job, .. }
            | Self::RedubDone { job, .. }
            | Self::RedubError { job, .. } => job.as_deref(),
            Self::VoiceWarm(_) => None,
            Self::Other(e) => e.job.as_deref(),
        }
    }
}

impl Event {
    pub fn parse(&self) -> EngineEvent {
        fn data<T: DeserializeOwned>(e: &Event) -> Option<T> {
            serde_json::from_value(e.data.clone()).ok()
        }
        let job = self.job.clone();
        let typed = match self.event.as_str() {
            "tts.chunk" => data(self).map(|chunk| EngineEvent::TtsChunk { job, chunk }),
            "tts.done" => data(self).map(|done| EngineEvent::TtsDone { job, done }),
            "tts.error" => data(self).map(|error| EngineEvent::TtsError { job, error }),
            "train.progress" => data(self).map(|progress| EngineEvent::TrainProgress { job, progress }),
            "train.done" => data(self).map(|done| EngineEvent::TrainDone { job, done }),
            "train.paused" => data(self).map(|paused| EngineEvent::TrainPaused { job, paused }),
            "train.error" => data(self).map(|error| EngineEvent::TrainError { job, error }),
            "voice.warm" => data(self).map(EngineEvent::VoiceWarm),
            "redub.progress" => data(self).map(|progress| EngineEvent::RedubProgress { job, progress }),
            "redub.transcript" => data(self).map(|transcript| EngineEvent::RedubTranscript { job, transcript }),
            "redub.review" => data(self).map(|review| EngineEvent::RedubReview { job, review }),
            "redub.done" => data(self).map(|done| EngineEvent::RedubDone { job, done }),
            "redub.error" => data(self).map(|error| EngineEvent::RedubError { job, error }),
            _ => None,
        };
        typed.unwrap_or_else(|| EngineEvent::Other(self.clone()))
    }
}

pub type Request = (&'static str, Value);

pub fn hello() -> Request {
    ("hello", json!({}))
}

pub fn voices_list() -> Request {
    ("voices.list", json!({}))
}

pub fn voices_create(name: &str) -> Request {
    ("voices.create", json!({ "name": name }))
}

pub fn voices_rename(id: &str, name: &str) -> Request {
    ("voices.rename", json!({ "id": id, "name": name }))
}

pub fn voices_delete(id: &str) -> Request {
    ("voices.delete", json!({ "id": id }))
}

pub fn presets_list() -> Request {
    ("presets.list", json!({}))
}

pub fn text_analyze(text: &str) -> Request {
    ("text.analyze", json!({ "text": text }))
}

pub fn recording_check(path: &std::path::Path) -> Request {
    ("recording.check", json!({ "path": path }))
}

pub fn dataset_add(voice_id: &str, part_index: u32, path: &std::path::Path) -> Request {
    ("dataset.add", json!({ "voice_id": voice_id, "part_index": part_index, "path": path }))
}

pub fn dataset_remove(voice_id: &str, part_index: u32) -> Request {
    ("dataset.remove", json!({ "voice_id": voice_id, "part_index": part_index }))
}

pub fn dataset_summary(voice_id: &str) -> Request {
    ("dataset.summary", json!({ "voice_id": voice_id }))
}

/// `epochs = None` lets the engine pick from the dataset length.
pub fn train_start(voice_id: &str, epochs: Option<u32>) -> Request {
    ("train.start", json!({ "voice_id": voice_id, "epochs": epochs }))
}

pub fn train_cancel(job: &str) -> Request {
    ("train.cancel", json!({ "job": job }))
}

/// Pauses after the next checkpoint; answered by the `train.paused` event.
pub fn train_pause(job: &str) -> Request {
    ("train.pause", json!({ "job": job }))
}

/// Answers `{"job"}` like `train.start`.
pub fn train_resume(voice_id: &str) -> Request {
    ("train.resume", json!({ "voice_id": voice_id }))
}

/// Throws away a paused training's checkpoints.
pub fn train_discard(voice_id: &str) -> Request {
    ("train.cancel", json!({ "voice_id": voice_id }))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    /// 32 decoding steps.
    #[default]
    Best,
    /// 16 steps: about twice as fast, slightly less polished.
    Fast,
}

impl Quality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Best => "best",
            Self::Fast => "fast",
        }
    }
}

/// Speaker for `accent: "native-us"` (needs a trained voice); `Auto` lets the engine pick by the voice's pitch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeVoice {
    #[default]
    Auto,
    Male,
    Female,
}

impl NativeVoice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Male => "male",
            Self::Female => "female",
        }
    }

    pub fn from_str(v: &str) -> NativeVoice {
        match v {
            "male" => Self::Male,
            "female" => Self::Female,
            _ => Self::Auto,
        }
    }
}

fn with_accent(mut params: Value, accent: Option<NativeVoice>) -> Value {
    if let Some(v) = accent {
        params["accent"] = json!("native-us");
        params["native_voice"] = json!(v.as_str());
    }
    params
}

/// Defaults: `expressiveness` 0.8, `voice_match` 1.0, `speed` 1.0 (0.8..=1.3).
/// `output_format`: `wav`, `mp3`, `flac`, `ogg`, `opus`, `m4a` or `aiff`.
#[allow(clippy::too_many_arguments)]
pub fn tts_generate(
    voice_id: &str,
    text: &str,
    expressiveness: f32,
    voice_match: f32,
    quality: Quality,
    speed: f32,
    language: &str,
    output_format: &str,
    accent: Option<NativeVoice>,
    exact_stress: bool,
) -> Request {
    (
        "tts.generate",
        with_accent(json!({
            "exact_stress": exact_stress,
            "voice_id": voice_id,
            "text": text,
            "expressiveness": expressiveness,
            "voice_match": voice_match,
            "quality": quality.as_str(),
            "speed": speed,
            "language": language,
            "output_format": output_format,
        }), accent),
    )
}

/// Answered by `VoiceWarming`, then the `voice.warm` event.
pub fn voices_warm(id: &str) -> Request {
    ("voices.warm", json!({ "id": id }))
}

pub fn tts_cancel(job: &str) -> Request {
    ("tts.cancel", json!({ "job": job }))
}

pub fn shutdown() -> Request {
    ("shutdown", json!({}))
}

pub fn redub_probe(path: &std::path::Path) -> Request {
    ("redub.probe", json!({ "path": path }))
}

/// `language` is `"auto"` or an ISO code; `translator: None` picks the best one installed.
pub fn redub_transcribe(path: &std::path::Path, language: &str, translate_to: Option<&str>, translator: Option<&str>) -> Request {
    ("redub.transcribe", json!({ "path": path, "language": language, "translate_to": translate_to, "translator": translator }))
}

/// Preloads models for `path`; `separate` also splits voice and background. Failures are harmless.
pub fn redub_prepare(path: &std::path::Path, separate: bool) -> Request {
    ("redub.prepare", json!({ "path": path, "separate": separate }))
}

#[derive(Clone, Debug)]
pub struct RedubDubParams<'a> {
    pub path: &'a std::path::Path,
    pub voice_id: &'a str,
    /// `"auto"` or a language code.
    pub source_language: &'a str,
    pub target_language: &'a str,
    /// `None` picks the best one installed.
    pub translator: Option<&'a str>,
    /// Stop after the translation and send `redub.review`.
    pub review: bool,
    pub voice_match: f32,
    pub expressiveness: f32,
    pub quality: Quality,
    pub speed: f32,
    pub output_format: &'a str,
    pub accent: Option<NativeVoice>,
    /// A video output always keeps the timing.
    pub timing: RedubTiming,
    pub output: RedubOutput,
}

/// Answers `{job}`; then `redub.progress`, and `redub.review` (with `review`), `redub.done` or `redub.error`.
pub fn redub_dub(p: &RedubDubParams<'_>) -> Request {
    let mut params = json!({
        "path": p.path,
        "voice_id": p.voice_id,
        "source_language": p.source_language,
        "target_language": p.target_language,
        "translator": p.translator,
        "review": p.review,
        "voice_match": p.voice_match,
        "expressiveness": p.expressiveness,
        "quality": p.quality.as_str(),
        "speed": p.speed,
        "output_format": p.output_format,
        "timing": p.timing.as_str(),
    });
    p.output.apply(&mut params);
    ("redub.dub", with_accent(params, p.accent))
}

#[derive(Clone, Debug, PartialEq)]
pub enum TextLanguage {
    /// The engine guesses from the script.
    Unknown,
    Spoken(String),
    Translated(String),
}

/// `segments` and `text_language` are only sent in `Respeak` mode.
#[allow(clippy::too_many_arguments)]
pub fn redub_render(
    voice_id: &str,
    path: &std::path::Path,
    mode: RedubMode,
    segments: Option<&[RedubSegmentIn]>,
    text_language: &TextLanguage,
    voice_match: f32,
    expressiveness: f32,
    quality: Quality,
    speed: f32,
    output_format: &str,
    accent: Option<NativeVoice>,
    timing: RedubTiming,
    output: RedubOutput,
) -> Request {
    let mut params = json!({
        "voice_id": voice_id,
        "path": path,
        "mode": mode.as_str(),
        "voice_match": voice_match,
        "expressiveness": expressiveness,
        "quality": quality.as_str(),
        "speed": speed,
        "output_format": output_format,
    });
    output.apply(&mut params);
    if let Some(segs) = segments {
        params["segments"] = json!(segs);
        params["timing"] = json!(timing.as_str());
        match text_language {
            TextLanguage::Unknown => {}
            TextLanguage::Spoken(l) => params["language"] = json!(l),
            TextLanguage::Translated(l) => {
                params["language"] = json!(l);
                params["translated"] = json!(true);
                params["target_language"] = json!(l);
            }
        }
    }
    ("redub.render", with_accent(params, accent))
}

pub fn redub_cancel(job: &str) -> Request {
    ("redub.cancel", json!({ "job": job }))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextChapter {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub text: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub words: u32,
    /// At the language's reading pace, speed 1.0.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub est_s: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextImport {
    #[serde(default)]
    pub path: PathBuf,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub format_label: String,
    #[serde(default)]
    pub title: String,
    /// Plain-text formats only.
    #[serde(default)]
    pub encoding: Option<String>,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub language_detected: bool,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub words: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub est_s: f32,
    #[serde(default)]
    pub chapters: Vec<TextChapter>,
    /// A suggested split when chapters are missing or very long.
    #[serde(default)]
    pub parts: Option<TextParts>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextParts {
    /// `headings` | `breaks` | `length` | `mixed`.
    #[serde(default)]
    pub how: String,
    #[serde(default)]
    pub chapters: Vec<TextChapter>,
}

pub fn text_import(path: &std::path::Path) -> Request {
    ("text.import", json!({ "path": path }))
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookTotals {
    /// 0..100, by characters read.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub percent: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub audio_s: f32,
    /// Across all sessions.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    #[serde(default)]
    pub rtf: Option<f32>,
    #[serde(default)]
    pub eta_s: Option<f32>,
    /// Extrapolated from what is made.
    #[serde(default)]
    pub est_audio_s: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookChapterRow {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    #[serde(default)]
    pub title: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub words: u32,
    /// `waiting` | `partial` | `done`.
    #[serde(default)]
    pub status: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segment: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segments: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub audio_s: f32,
    #[serde(default)]
    pub rtf: Option<f32>,
    #[serde(default)]
    pub path: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookSummary {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub voice_id: Option<String>,
    /// `new` | `running` | `paused` | `done` | `error` | `cancelled`.
    #[serde(default)]
    pub status: String,
    /// Left running by an engine that stopped: resumable.
    #[serde(default)]
    pub interrupted: bool,
    #[serde(default)]
    pub job: Option<String>,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub chapters: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub chapters_done: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub current: u32,
    #[serde(default)]
    pub folder: Option<PathBuf>,
    #[serde(default)]
    pub m4b: Option<PathBuf>,
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub words: u32,
    #[serde(default)]
    pub error: Option<EngineError>,
    #[serde(flatten)]
    pub totals: BookTotals,
    #[serde(default)]
    pub chapter_list: Vec<BookChapterRow>,
}

/// `book.start` / `book.resume`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookStarted {
    pub id: String,
    pub job: String,
    #[serde(default)]
    pub folder: Option<PathBuf>,
    #[serde(default)]
    pub accent_note: Option<String>,
    #[serde(default)]
    pub reference_note: Option<String>,
}

/// `book.progress`: after every segment and when a chapter starts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookProgress {
    pub id: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub chapter: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub chapters: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segment: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segments: u32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub chapter_audio_s: f32,
    /// `m4b` while the whole-book file is made, `memory` while waiting for free memory.
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(default)]
    pub retry_in_s: Option<f32>,
    /// Progress of the M4B: percent and seconds left.
    #[serde(default)]
    pub stage_pct: Option<f32>,
    #[serde(default)]
    pub stage_eta_s: Option<f32>,
    #[serde(flatten)]
    pub totals: BookTotals,
}

/// `book.chapter`
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookChapterDone {
    pub id: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    #[serde(default)]
    pub path: PathBuf,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default)]
    pub rtf: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookDone {
    pub id: String,
    #[serde(default)]
    pub folder: Option<PathBuf>,
    #[serde(default)]
    pub m4b: Option<PathBuf>,
    #[serde(default)]
    pub m4b_requested: bool,
    #[serde(flatten)]
    pub totals: BookTotals,
}

#[derive(Clone, Debug, Serialize)]
pub struct BookChapterIn<'a> {
    pub title: &'a str,
    pub text: &'a str,
}

pub struct BookParams<'a> {
    pub title: &'a str,
    pub source: Option<&'a str>,
    pub format: Option<&'a str>,
    pub chapters: Vec<BookChapterIn<'a>>,
    pub voice_id: &'a str,
    pub language: &'a str,
    pub expressiveness: f32,
    pub voice_match: f32,
    pub quality: Quality,
    pub speed: f32,
    pub accent: Option<NativeVoice>,
    pub output_format: &'a str,
    pub m4b: bool,
    pub read_titles: bool,
    pub exact_stress: bool,
}

pub fn book_start(p: &BookParams<'_>) -> Request {
    let params = json!({
        "exact_stress": p.exact_stress,
        "title": p.title,
        "source": p.source,
        "format": p.format,
        "chapters": p.chapters,
        "voice_id": p.voice_id,
        "language": p.language,
        "expressiveness": p.expressiveness,
        "voice_match": p.voice_match,
        "quality": p.quality.as_str(),
        "speed": p.speed,
        "output_format": p.output_format,
        "m4b": p.m4b,
        "read_titles": p.read_titles,
    });
    ("book.start", with_accent(params, p.accent))
}

pub fn book_id_request(method: &'static str, id: &str) -> Request {
    (method, json!({ "id": id }))
}

pub fn book_reread(id: &str, voice_id: Option<&str>, chapters: Option<&[u32]>) -> Request {
    let mut params = json!({ "id": id });
    if let Some(v) = voice_id {
        params["voice_id"] = json!(v);
    }
    if let Some(c) = chapters {
        params["chapters"] = json!(c);
    }
    ("book.reread", params)
}

pub fn book_play(id: &str, index: u32) -> Request {
    ("book.play", json!({ "id": id, "index": index }))
}

/// A `legacy` entry (no sidecar) knows only `name`, `created`, `duration_s`, `format` and the paths.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputEntry {
    /// The file stem, unique in the outputs folder.
    pub id: String,
    /// `speak` | `redub`.
    pub kind: String,
    #[serde(default)]
    pub legacy: bool,
    #[serde(default)]
    pub name: String,
    /// Unix seconds; the list is sorted by it, newest first.
    #[serde(default)]
    pub created: f64,
    pub path: PathBuf,
    /// `None` until `outputs.play` makes one.
    #[serde(default)]
    pub path_play: Option<PathBuf>,
    #[serde(default)]
    pub video_path: Option<PathBuf>,
    #[serde(default)]
    pub duration_s: Option<f32>,
    #[serde(default)]
    pub elapsed_s: Option<f32>,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub voice_id: Option<String>,
    #[serde(default)]
    pub voice_name: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub accent: Option<String>,
    /// Redub: `dub` | `convert` | `respeak`.
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub source_name: Option<String>,
    /// Redub: the language a dub was spoken in.
    #[serde(default)]
    pub to: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputsPage {
    #[serde(default)]
    pub results: Vec<OutputEntry>,
    /// Older results exist: ask again with `before` = the last entry's `created`.
    #[serde(default)]
    pub more: bool,
}

pub fn outputs_list(kind: &str, limit: u32, before: Option<f64>) -> Request {
    let mut params = json!({ "kind": kind, "limit": limit });
    if let Some(b) = before {
        params["before"] = json!(b);
    }
    ("outputs.list", params)
}

/// Answers `{path, path_play}`.
pub fn outputs_play(path: &std::path::Path) -> Request {
    ("outputs.play", json!({ "path": path }))
}
