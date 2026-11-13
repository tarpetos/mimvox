//! Typed mirror of contract/engine-protocol.md.
//!
//! Three layers:
//! - wire envelope types (`EngineError`, `Event`),
//! - typed payloads (`Hello`, `Voice`, `Preset`, ...) with lenient decoding (missing/null numbers become 0),
//! - request builders returning `(method, params)` plus `parse_result` for typed responses and
//!   `Event::parse` for typed events.
//!
//! Typical use:
//! ```ignore
//! let id = engine.send(protocol::tts_generate(&voice, "Hello", 0.8, 1.0, protocol::Quality::Best, 1.0, "en", "wav"));
//! // later, on EngineMsg::Response { id, result: Ok(v) }:
//! let started: protocol::TtsStarted = protocol::parse_result(v)?;
//! ```

#![allow(dead_code)]

use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};

// ---------------------------------------------------------------------------------------------------------------
// Lenient deserializers
// ---------------------------------------------------------------------------------------------------------------

mod de {
    use super::*;

    /// `null` or missing -> 0.0 (use with `#[serde(default)]`).
    pub fn f32_or_zero<'de, D: Deserializer<'de>>(d: D) -> Result<f32, D::Error> {
        Ok(Option::<f32>::deserialize(d)?.unwrap_or(0.0))
    }

    /// `null` or missing -> 0.
    pub fn u32_or_zero<'de, D: Deserializer<'de>>(d: D) -> Result<u32, D::Error> {
        Ok(Option::<u32>::deserialize(d)?.unwrap_or(0))
    }

    /// Accepts a string, a number or null (timestamps may be either) and returns a string.
    pub fn string_or_number<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
        Ok(match Value::deserialize(d)? {
            Value::String(s) => s,
            Value::Null => String::new(),
            other => other.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Envelope
// ---------------------------------------------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------------------------------------------

/// Readiness of a model in `hello`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    Ready,
    Loading,
    /// Unloaded while a voice trains (the GPU is handed to training); back to `Ready` after it.
    Paused,
    Missing,
    #[default]
    #[serde(other)]
    Unknown,
}

impl ModelState {
    /// The wire spelling (`ready`, `loading`, `missing`), `unknown` for anything else.
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

/// How good a recording or a whole dataset is.
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

/// Amount of material: decides zero-shot vs training.
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

// ---------------------------------------------------------------------------------------------------------------
// hello
// ---------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Device {
    #[serde(default)]
    pub backend: String,
    #[serde(default)]
    pub name: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub vram_gb: f32,
    /// `nvidia` or `amd` for the `cuda` backend; empty for older engines and other backends.
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

/// A language the speech model speaks (`hello` -> `languages`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SpeechLanguage {
    /// ISO 639-1 (`yue` for Cantonese).
    pub code: String,
    /// English name.
    #[serde(default)]
    pub name: String,
    /// Name in the language itself.
    #[serde(default)]
    pub native: String,
    /// One of the languages listed first.
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
    /// Languages the speech model speaks, the common ones first (empty for an engine that does not say).
    #[serde(default)]
    pub languages: Vec<SpeechLanguage>,
    /// Redub translators installed, best first (`"nllb-1.3b"`, `"nllb-600m"`; empty for an engine that does not say).
    #[serde(default)]
    pub translators: Vec<String>,
    /// The zero-shot voice converter is installed: an untrained voice can use the native accent too (`false` for an
    /// engine that does not say).
    #[serde(default)]
    pub zero_shot_vc: bool,
}

// ---------------------------------------------------------------------------------------------------------------
// voices / presets / text
// ---------------------------------------------------------------------------------------------------------------

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
    /// A training of this voice is paused (or was interrupted by a closed app): `train.resume` continues it.
    #[serde(default)]
    pub training_paused: Option<PausedTraining>,
}

/// `voices.list` -> `training_paused`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PausedTraining {
    /// Epoch of the saved checkpoint: where a resume continues.
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub epoch: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub total_epochs: u32,
    /// Local time without zone, `YYYY-MM-DDTHH:MM:SS`.
    #[serde(default)]
    pub paused_at: Option<String>,
    /// The app was closed or crashed during the training (instead of a pause the user asked for).
    #[serde(default)]
    pub interrupted: bool,
}

/// One reading part of a preset or of analysed user text.
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

// ---------------------------------------------------------------------------------------------------------------
// recording / dataset
// ---------------------------------------------------------------------------------------------------------------

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

/// One recorded take inside a dataset.
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

// ---------------------------------------------------------------------------------------------------------------
// Typed responses (`parse_result::<T>`)
// ---------------------------------------------------------------------------------------------------------------

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

/// `tts.generate`: the job plus the exact segmentation, known before the first chunk arrives.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsStarted {
    pub job: String,
    /// Number of text segments (= number of `tts.chunk` events to expect).
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segments: u32,
    /// Characters per segment, in order (weights for progress and for the buffering estimate).
    #[serde(default)]
    pub segment_chars: Vec<u32>,
    /// Why `accent: "native-us"` was not possible and the voice's own reference speaks instead (see [`NativeVoice`]).
    #[serde(default)]
    pub accent_note: Option<String>,
    /// Set when the clip that speaks is in another language than the text (its accent carries over): a sentence for
    /// the user. See [`ReferenceUse`].
    #[serde(default)]
    pub reference_note: Option<String>,
}

/// Which reference clip spoke a job, from its end event (`tts.done`, `redub.done` of a speech job).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ReferenceUse {
    /// `own` | `native-us`: whether the voice's own clip spoke, or the native-accent clip.
    #[serde(default)]
    pub accent: Option<String>,
    /// The language of the clip that spoke (`reference_<lang>.wav` by its name, `reference.wav` as detected, `en` for
    /// the native clip); `None` while unknown.
    #[serde(default)]
    pub reference_language: Option<String>,
    /// The clip is in another language than the speech: its accent carries over.
    #[serde(default)]
    pub reference_foreign: bool,
    /// A sentence for the user when `reference_foreign`.
    #[serde(default)]
    pub reference_note: Option<String>,
}

/// `voices.warm`: `warming = false` means nothing will be reported (`reason`: for example `already_warm`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VoiceWarming {
    #[serde(default)]
    pub warming: bool,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Decodes the `result` of a successful response into a typed payload.
pub fn parse_result<T: DeserializeOwned>(result: Value) -> Result<T, EngineError> {
    serde_json::from_value(result)
        .map_err(|e| EngineError::new("bad_response", format!("unexpected engine response: {e}")))
}

// ---------------------------------------------------------------------------------------------------------------
// Redub (`redub.*`)
// ---------------------------------------------------------------------------------------------------------------

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
    /// The file has a picture track (a video; cover art does not count). Absent from older engines.
    #[serde(default)]
    pub has_video: Option<bool>,
    #[serde(default)]
    pub format: String,
    /// Optional: a WAV copy the app can play as "the original" (set when the file itself is not a WAV).
    #[serde(default)]
    pub preview_path: Option<PathBuf>,
}

/// One transcribed segment (`redub.transcript`).
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
    /// The original-language text when `text` is a translation.
    #[serde(default)]
    pub source_text: Option<String>,
    /// `text` is an English translation of the speech.
    #[serde(default)]
    pub translated: bool,
}

/// `redub.transcript` event.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubTranscript {
    /// The language of the speech (the detected one), or of `text` for an engine that reports the text language.
    #[serde(default)]
    pub language: String,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default)]
    pub segments: Vec<RedubSegment>,
    /// `true` when the segment texts were translated to English (`None` for an engine that does not say).
    #[serde(default)]
    pub translated: Option<bool>,
    /// The language of the speech when `language` describes the (translated) text.
    #[serde(default)]
    pub source_language: Option<String>,
    /// The language the segments were translated into (with `translated: true`).
    #[serde(default)]
    pub target_language: Option<String>,
    /// `true` when a `redub.dub` with `review: true` stopped here: this transcript is the review.
    #[serde(default)]
    pub review: Option<bool>,
}

/// One segment of the `redub.review` event: `text` is the English translation, `source_text` the original.
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

/// `redub.review` event: a `redub.dub` job with `review: true` ends here, after the translation, and waits for the
/// user's edits (then `redub.render` respeak).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubReview {
    /// Language of `text` (`"en"`).
    #[serde(default)]
    pub language: String,
    /// Language of the speech (the detected one when the request said `"auto"`).
    #[serde(default)]
    pub source_language: String,
    #[serde(default)]
    pub segments: Vec<RedubReviewSegment>,
}

/// `redub.progress` event (transcribe stages `decode` / `load_model` / `transcribe`; render stages are free-form).
/// `pct` is 0-100; `segment` / `segments` (1-based k of n) appear while rendering a re-spoken transcript.
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

/// `redub.done` event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RedubDone {
    /// The saved result in `<data>/outputs`, in the requested `output_format`.
    pub path: PathBuf,
    /// A WAV of the same audio to play (the saved file itself when it is a WAV; else one in `<data>/tmp/play`).
    #[serde(default)]
    pub path_play: Option<PathBuf>,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    /// `"dub"`, `"convert"` or `"respeak"`.
    #[serde(default)]
    pub mode: Option<String>,
    /// How far the spoken result ran past the original length, in seconds (`redub.dub` only, when it did).
    #[serde(default)]
    pub drift_s: Option<f32>,
    /// The language the result is spoken in (`redub.dub`: `target_language`; respeak: `language`).
    #[serde(default)]
    pub target_language: Option<String>,
    /// `respeak`: the language spoken.
    #[serde(default)]
    pub language: Option<String>,
    /// The video with the new voice (`output: "video"`), next to the audio result with the same name.
    #[serde(default)]
    pub video_path: Option<PathBuf>,
    /// Why the background could not be kept (no music-removal model, the separation failed), when it was asked for.
    #[serde(default)]
    pub background_note: Option<String>,
    /// The clip that spoke and its language (speech jobs; empty for `convert`).
    #[serde(flatten)]
    pub reference: ReferenceUse,
}

impl RedubDone {
    /// The WAV to play: `path_play`, else the saved file.
    pub fn play_path(&self) -> PathBuf {
        self.path_play.clone().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| self.path.clone())
    }
}

/// What `redub.render` does with the file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RedubMode {
    /// Transcribe, translate to English and speak it in the voice (`redub.dub`; a reviewed one is finished with
    /// `redub.render` in `respeak` mode). Not a `redub.render` mode itself.
    #[default]
    Dub,
    /// Keep the original timing and delivery, swap the voice (needs a trained voice).
    Convert,
    /// Speak the (edited) transcript again in the voice's own delivery.
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

/// How a re-spoken or dubbed text is timed (`timing` of `redub.dub` and `redub.render` in `respeak` mode).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RedubTiming {
    /// Fit each phrase into the original timing (may speed the speech up, never below the engine's floor).
    #[default]
    Keep,
    /// Speak at the natural pace, one phrase after the other; the result can be longer than the source.
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

/// What a Redub job writes for a video source (`output` and `keep_background` of `redub.dub` / `redub.render`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RedubOutput {
    /// A copy of the video with the new voice (`output: "video"`), else audio only.
    pub video: bool,
    /// Keep the original music and sound effects under the new voice.
    pub keep_background: bool,
}

impl RedubOutput {
    fn apply(self, params: &mut Value) {
        params["output"] = json!(if self.video { "video" } else { "audio" });
        params["keep_background"] = json!(self.keep_background);
    }
}

/// A segment sent to `redub.render` in `respeak` mode.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RedubSegmentIn {
    pub start: f32,
    pub end: f32,
    pub text: String,
}

// ---------------------------------------------------------------------------------------------------------------
// Typed events
// ---------------------------------------------------------------------------------------------------------------

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

/// `train.paused`: the training stopped after a checkpoint; the job is over and the GPU is back for speech.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainPaused {
    #[serde(default)]
    pub voice_id: String,
    /// Epoch of the saved checkpoint (where a resume continues).
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub epoch: u32,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub total_epochs: u32,
    #[serde(default)]
    pub paused_at: Option<String>,
    /// How far the training had got when it stopped (at or after `epoch`).
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub reached_epoch: u32,
}

/// One finished text segment, final audio: play at once.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsChunk {
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub index: u32,
    pub path: PathBuf,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    /// Total segments of the job (same as `TtsStarted::segments`).
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub segments: u32,
    /// Seconds since the job started when this chunk became ready.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub ready_s: f32,
    /// Real-time factor of this chunk: seconds of generation per second of audio (below 1 = faster than playback).
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub rtf: f32,
}

/// `voice.warm` event: a voice finished preparing (`ok`), or failed (`message`).
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

/// `tts.done` event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsDone {
    /// The saved result in `<data>/outputs`, in the requested `output_format`.
    pub path: PathBuf,
    /// A WAV of the same audio to play (the saved file itself when it is a WAV; else one in `<data>/tmp/play`).
    #[serde(default)]
    pub path_play: Option<PathBuf>,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub duration_s: f32,
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    /// The clip that spoke and its language.
    #[serde(flatten)]
    pub reference: ReferenceUse,
}

impl TtsDone {
    /// The WAV to play: `path_play`, else the saved file.
    pub fn play_path(&self) -> PathBuf {
        self.path_play.clone().filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| self.path.clone())
    }
}

/// Typed view of `Event`. `job` is the job id the event belongs to.
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
    /// Unknown event name or a payload that did not match the contract (kept raw, never dropped).
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
    /// Decodes into a typed event; falls back to `EngineEvent::Other` for unknown or malformed events.
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

// ---------------------------------------------------------------------------------------------------------------
// Request builders: `(method, params)` for `EngineClient::request` / `EngineClient::send`
// ---------------------------------------------------------------------------------------------------------------

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

/// Pause a running training after its next checkpoint; answered by the `train.paused` event.
pub fn train_pause(job: &str) -> Request {
    ("train.pause", json!({ "job": job }))
}

/// Continue the paused (or interrupted) training of a voice: answers `{"job"}` like `train.start`.
pub fn train_resume(voice_id: &str) -> Request {
    ("train.resume", json!({ "voice_id": voice_id }))
}

/// Throw away a paused training (its checkpoints and the flag).
pub fn train_discard(voice_id: &str) -> Request {
    ("train.cancel", json!({ "voice_id": voice_id }))
}

/// Speed/quality trade-off of `tts.generate`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Quality {
    /// 32 OmniVoice decoding steps (the default).
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

/// The native-accent speaker of `accent: "native-us"` (English spoken by a native American reference voice, then
/// converted to the user's timbre by RVC; needs a trained voice). `Auto` lets the engine pick by the voice's pitch.
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

/// Adds `accent` / `native_voice` to a speech request: `Some` asks for the native accent with that speaker, `None`
/// leaves the default (`own`: the voice's own reference).
fn with_accent(mut params: Value, accent: Option<NativeVoice>) -> Value {
    if let Some(v) = accent {
        params["accent"] = json!("native-us");
        params["native_voice"] = json!(v.as_str());
    }
    params
}

/// `expressiveness` (default 0.8) maps to OmniVoice's sampling temperatures, `voice_match` (default 1.0) to the RVC
/// index rate (capped by the engine for a language the voice was not recorded in). `speed` (0.8..=1.3, default 1.0) is
/// OmniVoice's speaking speed. `language` is the ISO code of `text`. `output_format` is the format the result is saved
/// in (`wav`, `mp3`, `flac`, `ogg`, `opus`, `m4a` or `aiff`). `accent`: see [`NativeVoice`].
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
) -> Request {
    (
        "tts.generate",
        with_accent(json!({
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

/// Loads a voice's models in the background; answered by `VoiceWarming`, then the `voice.warm` event.
pub fn voices_warm(id: &str) -> Request {
    ("voices.warm", json!({ "id": id }))
}

pub fn tts_cancel(job: &str) -> Request {
    ("tts.cancel", json!({ "job": job }))
}

pub fn shutdown() -> Request {
    ("shutdown", json!({}))
}

// ---- redub ----

pub fn redub_probe(path: &std::path::Path) -> Request {
    ("redub.probe", json!({ "path": path }))
}

/// `language`: `"auto"` or an ISO code (`"en"`, `"uk"`, ...). `translate_to`: a language code to translate every segment
/// into (any language the speech model speaks), or `None` to keep the speech's own language. `translator`:
/// `"nllb-1.3b"` | `"nllb-600m"`, or `None` for the best one installed.
pub fn redub_transcribe(path: &std::path::Path, language: &str, translate_to: Option<&str>, translator: Option<&str>) -> Request {
    ("redub.transcribe", json!({ "path": path, "language": language, "translate_to": translate_to, "translator": translator }))
}

/// `redub.prepare`: preloads the speech models for `path` so the first transcribe/dub starts at once, and with
/// `separate` splits the file into voice and background in the background (for a video whose music is kept).
/// Failures (busy, not ready) are harmless and ignored.
pub fn redub_prepare(path: &std::path::Path, separate: bool) -> Request {
    ("redub.prepare", json!({ "path": path, "separate": separate }))
}

/// Parameters of `redub.dub` (transcribe, translate to `target_language`, speak in the voice).
#[derive(Clone, Debug)]
pub struct RedubDubParams<'a> {
    pub path: &'a std::path::Path,
    pub voice_id: &'a str,
    /// `"auto"` or a language code.
    pub source_language: &'a str,
    /// The language to translate to and speak (any language the speech model speaks).
    pub target_language: &'a str,
    /// `"nllb-1.3b"` (better) | `"nllb-600m"` (faster), or `None` for the best one installed.
    pub translator: Option<&'a str>,
    /// Stop after the translation and send `redub.review` instead of speaking.
    pub review: bool,
    pub voice_match: f32,
    pub expressiveness: f32,
    pub quality: Quality,
    pub speed: f32,
    /// The format the result is saved in (see [`tts_generate`]).
    pub output_format: &'a str,
    /// The native accent for an English target (see [`NativeVoice`]).
    pub accent: Option<NativeVoice>,
    /// Keep the original timing or speak at the natural pace (a video output always keeps it).
    pub timing: RedubTiming,
    /// Video or audio, with or without the original background.
    pub output: RedubOutput,
}

/// `redub.dub {path, voice_id, source_language, target_language, translator, review, voice_match, expressiveness, quality, speed,
/// output_format, timing, output, keep_background}`
/// -> `{job}`; events `redub.progress` (stages `decode`, `load_model`, `transcribe`, `translate`, `speak`, `save`),
/// then `redub.review` (when `review`) or `redub.done`, or `redub.error`.
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

/// The language of the segment texts of a `respeak` render: the transcript's own language, or a translation.
#[derive(Clone, Debug, PartialEq)]
pub enum TextLanguage {
    /// Not known: the engine guesses from the script.
    Unknown,
    /// The speech's own language (`language`).
    Spoken(String),
    /// Translated into this language (`translated: true`, `target_language`).
    Translated(String),
}

/// `segments` and `text_language` are only sent in `Respeak` mode (the possibly edited transcript). `output_format` is
/// the format the result is saved in (see [`tts_generate`]). `accent` applies to an English `respeak` (see
/// [`NativeVoice`]).
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

// ---------------------------------------------------------------------------------------------------------------
// Text import and audiobooks (`text.import`, `book.*`)
// ---------------------------------------------------------------------------------------------------------------

/// One chapter of an imported text.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextChapter {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub text: String,
    #[serde(default, deserialize_with = "de::u32_or_zero")]
    pub words: u32,
    /// Estimated speech seconds at the language's reading pace (speed 1.0).
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub est_s: f32,
}

/// `text.import`: the clean text of a file, cut into chapters.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextImport {
    #[serde(default)]
    pub path: PathBuf,
    /// The file name.
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub format_label: String,
    #[serde(default)]
    pub title: String,
    /// The text encoding found (plain-text formats only).
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
    #[serde(default)]
    pub warnings: Vec<String>,
}

pub fn text_import(path: &std::path::Path) -> Request {
    ("text.import", json!({ "path": path }))
}

/// Overall progress of a book (in `book.*` events and summaries).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BookTotals {
    /// 0..100, by characters read.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub percent: f32,
    /// Seconds of audio made so far.
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub audio_s: f32,
    /// Seconds spent making it (all sessions).
    #[serde(default, deserialize_with = "de::f32_or_zero")]
    pub elapsed_s: f32,
    /// Generation seconds per audio second so far.
    #[serde(default)]
    pub rtf: Option<f32>,
    /// Seconds of generation left at the measured pace.
    #[serde(default)]
    pub eta_s: Option<f32>,
    /// The whole book's audio length, extrapolated from what is made.
    #[serde(default)]
    pub est_audio_s: Option<f32>,
}

/// A chapter of a book project with its progress (`book.get` -> `chapter_list`).
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

/// A book project (`book.list` entries, `book.get`, the `book.paused` event).
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
    /// Recorded as running by an engine that stopped (the app closed mid-book): resumable.
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

/// `book.progress`: after every segment (and when a chapter starts).
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
    /// `m4b` while the whole-book file is made.
    #[serde(default)]
    pub stage: Option<String>,
    #[serde(flatten)]
    pub totals: BookTotals,
}

/// `book.chapter`: a chapter file is finished.
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

/// `book.done`.
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

/// One chapter sent with `book.start`.
#[derive(Clone, Debug, Serialize)]
pub struct BookChapterIn<'a> {
    pub title: &'a str,
    pub text: &'a str,
}

/// The voice settings of a book (as `tts.generate`'s) plus its output options.
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
}

pub fn book_start(p: &BookParams<'_>) -> Request {
    let params = json!({
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

pub fn book_play(id: &str, index: u32) -> Request {
    ("book.play", json!({ "id": id, "index": index }))
}

// ---- saved results (outputs.list / outputs.play) ----

/// One saved Speak or Redub result (`outputs.list`), from its sidecar (`<stem>.result.json`) or, for a result saved
/// before sidecars (`legacy`), from the file alone: then only `name`, `created`, `duration_s`, `format` and the paths
/// are known.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputEntry {
    /// The file stem, unique in the outputs folder.
    pub id: String,
    /// `speak` | `redub`.
    pub kind: String,
    #[serde(default)]
    pub legacy: bool,
    /// The saved file's name (what a legacy entry shows as its title).
    #[serde(default)]
    pub name: String,
    /// Unix time in seconds; the list is sorted by it, newest first (`before` pages by it).
    #[serde(default)]
    pub created: f64,
    pub path: PathBuf,
    /// A WAV that plays it now (the file itself, or this session's copy in `<data>/tmp/play`); `None` until
    /// `outputs.play` makes one.
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
    /// Speak: the whole text.
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

/// The answer of `outputs.list`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputsPage {
    #[serde(default)]
    pub results: Vec<OutputEntry>,
    /// Older results exist (ask again with `before` = the last entry's `created`).
    #[serde(default)]
    pub more: bool,
}

/// `outputs.list`: the saved results of `kind` (`speak` | `redub`), newest first, `limit` of them, older than
/// `before` (a `created`) when given.
pub fn outputs_list(kind: &str, limit: u32, before: Option<f64>) -> Request {
    let mut params = json!({ "kind": kind, "limit": limit });
    if let Some(b) = before {
        params["before"] = json!(b);
    }
    ("outputs.list", params)
}

/// `outputs.play`: a WAV to play a saved result with (`{path, path_play}`).
pub fn outputs_play(path: &std::path::Path) -> Request {
    ("outputs.play", json!({ "path": path }))
}
