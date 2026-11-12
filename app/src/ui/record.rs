//! Record screen: pick a preset (or paste your own text), read it part by part, auto-save every take.
//!
//! Shell integration (the shell calls these, see `ui::shell`):
//! - `view` draws the screen on a transparent background (the shell paints chrome and aurora).
//! - `on_engine` receives every engine message, `settle` runs after every event (outgoing requests, following the
//!   header's voice); the live meter is fed by the recorder's level events (`on_capture`).
//! - `on_key` handles Space (record / stop) and Left / Right (previous / next part); the screen root also
//!   handles them itself when it has focus, so `AppSpec::on_key` may forward to it for the unfocused case.
//!
//! The recording goes into the voice that is active in the header. Without one, the screen creates a voice named
//! "My voice" and selects it.
//!
//! **Languages.** The reading presets exist per language (`presets.list {language}` lists the languages; the choice is
//! remembered as `record_language` in settings.json); "Use my own text" detects the language or takes the chosen one
//! (`text.analyze {text, language}`). A new voice is created in the language of what is read (`voices.create
//! {language}`); an empty active voice in another language takes the new language (`voices.rename {id, language}`), and
//! one that already has recordings in another language is left alone: a new voice "<name> (<Language>)" is made. Requests are queued on `RecordState` and sent from `settle` through
//! `shell::request`, so the screen state never needs the engine handle.

mod fixtures;
mod mic;
mod pick;
mod session;
mod widgets;

use super::events::{self, Latest};
use super::shell;
use super::theme;
use crate::audio::{CaptureEvent, MeterSink, Recorder};
use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use rune::{Cx, El, Key, KeyEvent, TextEdit, Tx, col, el};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use widgets::Tone;

pub use fixtures::{
    scene_language_uk, scene_own_text_uk, scene_session_uk, scene_mic_open, scene_mic_picker, scene_mic_warning, scene_notice_saved, scene_own_text, scene_picker, scene_quick_ready, scene_recording, scene_recording_loud, scene_rejected,
    scene_saved_with_issues, scene_summary_ready, scene_import_drop, scene_import_checking, scene_import_saved,
};

#[allow(dead_code)]
#[derive(Clone, Debug, Default, Deserialize)]
pub struct PartText {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub est_seconds: f32,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Preset {
    pub id: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub target_minutes: f32,
    #[serde(default)]
    pub trains_rvc: bool,
    #[serde(default)]
    pub parts: Vec<PartText>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum Level {
    #[default]
    TooShort,
    ZeroShot,
    Trainable,
    Great,
}

impl Level {
    fn parse(s: &str) -> Level {
        match s {
            "zero_shot" => Level::ZeroShot,
            "trainable" => Level::Trainable,
            "great" => Level::Great,
            _ => Level::TooShort,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Level::TooShort => "Too short",
            Level::ZeroShot => "Quick clone",
            Level::Trainable => "Trainable",
            Level::Great => "Great",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Ok,
    Warn,
    Bad,
}

impl Verdict {
    fn parse(s: &str) -> Verdict {
        match s {
            "warn" => Verdict::Warn,
            "bad" => Verdict::Bad,
            _ => Verdict::Ok,
        }
    }
}

/// A language with reading presets (`presets.list` -> `languages`).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct RecLang {
    pub code: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub native: String,
}

#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// The language the text was analysed in, and whether the engine detected it.
    pub language: String,
    pub detected: bool,
    pub words: u32,
    pub est_minutes: f32,
    pub coverage: f32,
    pub missing: Vec<String>,
    pub level: Level,
    pub message: String,
    pub parts: Vec<PartText>,
}

#[derive(Clone, Debug)]
pub struct SummaryPart {
    pub index: u32,
    pub duration_s: f32,
    pub verdict: Verdict,
    pub issues: Vec<String>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct Summary {
    pub total_minutes: f32,
    /// Minutes the engine counts for training (parts with a "bad" verdict are left out).
    pub usable_minutes: f32,
    pub has_reference: bool,
    pub parts: Vec<SummaryPart>,
    pub level: Level,
    pub message: String,
    pub recommended_epochs: Option<u32>,
}

/// What is being recorded.
#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub id: String,
    /// The language of the texts (a new voice is created in it).
    pub language: String,
    pub title: String,
    pub target_minutes: f32,
    pub trains_rvc: bool,
    pub parts: Vec<PartText>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TakeStatus {
    #[default]
    Empty,
    Checking,
    Saving,
    Saved,
    /// The check said "bad": not added to the dataset.
    Rejected,
    Failed,
}

#[derive(Clone, Debug, Default)]
pub struct Take {
    pub status: TakeStatus,
    pub duration_s: f32,
    pub verdict: Option<Verdict>,
    pub issues: Vec<String>,
    pub path: Option<PathBuf>,
    pub error: Option<String>,
    /// File name of an existing recording that was imported instead of recorded live.
    pub imported: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub enum Load<T> {
    #[default]
    NotAsked,
    Loading,
    Ready(T),
    Failed(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Phase {
    #[default]
    Pick,
    Own,
    Session,
    Summary,
}

enum Pending {
    Presets,
    Analyze,
    CreateVoice,
    Check { part: usize, path: PathBuf },
    Add { part: usize, dest: PathBuf },
    /// `media.combine`: several files joined into one take for `part`.
    Combine { part: usize },
    Summary,
}

/// A running capture and its live readings.
pub struct Active {
    recorder: Option<Recorder>,
    /// Tells this capture's level events from those of an earlier one.
    seq: u64,
    pub part: usize,
    pub level: f32,
    pub peak: f32,
    pub elapsed_s: f32,
    pub clipped: bool,
}

#[derive(Default)]
pub struct RecordState {
    /// The voice being recorded into (follows the header's active voice).
    pub voice_id: Option<String>,
    /// The chosen microphone (`None` = system default); persisted as `microphone` in the settings file.
    pub input_device: Option<String>,
    /// Microphone chooser: device list, dropdown and level test.
    pub mic: mic::MicState,
    /// The app data folder (voices, temporary takes); copied from `ShellState::data_dir` by `AppState::default`.
    pub data_dir: PathBuf,

    pub phase: Phase,
    /// The language of the presets (`""` until the saved choice is read: English).
    pub language: String,
    /// Languages that have presets (from `presets.list`).
    pub languages: Vec<RecLang>,
    /// "Use my own text": the chosen language, `None` = detect it.
    pub own_language: Option<String>,
    prefs_loaded: bool,
    pub presets: Load<Vec<Preset>>,
    pub plan: Option<Plan>,
    pub current: usize,
    pub takes: Vec<Take>,
    pub summary: Option<Summary>,

    pub own: TextEdit,
    pub analysis: Option<Analysis>,
    pub analyzing: bool,
    pub analyze_error: Option<String>,

    pub active: Option<Active>,
    /// One-line message above the recorder (auto-advance confirmation, errors).
    pub notice: Option<(String, Tone)>,
    pub playing: bool,
    pub creating_voice: bool,
    /// The name for the voice `ensure_voice` creates next (default "My voice").
    new_voice_name: Option<String>,
    /// A file drag is over the window (highlights the recorder card as a drop target).
    pub import_over: bool,
    /// The "Import file..." dialog is open; its answer arrives as an event.
    importing: bool,
    /// Numbers captures (live recordings and mic tests) so late level events of a stopped one are ignored.
    capture_seq: u64,
    pending: HashMap<u64, Pending>,
    /// Requests waiting for the event to settle (sent through `shell::request`).
    outbox: Vec<(String, Value, Pending)>,
    /// The dataset changed: refresh the shell's voice list (minutes, parts) when the event settles.
    voices_dirty: bool,
}

fn guard<T>(f: impl FnOnce() -> anyhow::Result<T>) -> Result<T, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("Audio is not available yet".to_string()),
    }
}

fn f(v: &Value, k: &str) -> f32 {
    v.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32
}

fn st<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

fn str_list(v: &Value, k: &str) -> Vec<String> {
    v.get(k).and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

fn parse_summary(v: &Value) -> Summary {
    let total = f(v, "total_minutes");
    Summary {
        total_minutes: total,
        usable_minutes: v.get("usable_minutes").and_then(Value::as_f64).map_or(total, |m| m as f32),
        has_reference: v.get("has_reference").and_then(Value::as_bool).unwrap_or(false),
        parts: v
            .get("parts")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .map(|p| SummaryPart { index: f(p, "index") as u32, duration_s: f(p, "duration_s"), verdict: Verdict::parse(st(p, "verdict")), issues: str_list(p, "issues") })
                    .collect()
            })
            .unwrap_or_default(),
        level: Level::parse(st(v, "level")),
        message: st(v, "message").to_string(),
        recommended_epochs: v.get("recommended_epochs").and_then(Value::as_u64).map(|e| e as u32),
    }
}

impl RecordState {
    /// The part being recorded or shown.
    pub fn part(&self) -> Option<&PartText> {
        self.plan.as_ref().and_then(|p| p.parts.get(self.current))
    }

    pub fn recording(&self) -> bool {
        self.active.is_some()
    }

    /// Queues a request; `settle` sends it. Always accepted (a dead engine fails it there).
    fn request(&mut self, method: &str, params: Value, what: Pending) -> bool {
        self.outbox.push((method.to_string(), params, what));
        true
    }

    /// The language of the presets ("en" until chosen).
    pub fn lang(&self) -> &str {
        if self.language.is_empty() { "en" } else { &self.language }
    }

    fn ensure_voice(&mut self) {
        if self.voice_id.is_none() && !self.creating_voice {
            let language = self.plan.as_ref().map_or_else(|| self.lang().to_owned(), |p| p.language.clone());
            let name = self.new_voice_name.take().unwrap_or_else(|| "My voice".to_owned());
            self.creating_voice = self.request("voices.create", json!({"name": name, "language": language}), Pending::CreateVoice);
        }
    }

    /// The header's voice changed: show that voice's recorded parts for the current plan.
    fn set_voice(&mut self, id: Option<String>) {
        self.voice_id = id;
        if let Some(plan) = &self.plan {
            self.takes = vec![Take::default(); plan.parts.len()];
            self.summary = None;
            self.notice = None;
            self.refresh_summary();
        }
    }

    /// A queued request could not be sent: the engine is not running.
    fn send_failed(&mut self, what: Pending) {
        match what {
            Pending::Presets => self.presets = Load::Failed("The engine is not running".into()),
            Pending::Analyze => {
                self.analyzing = false;
                self.analyze_error = Some("The engine is not running".into());
            }
            Pending::CreateVoice => {
                self.creating_voice = false;
                self.notice = Some(("The engine is not running".into(), Tone::Danger));
            }
            Pending::Check { part, .. } | Pending::Add { part, .. } => self.set_take_failed(part, "The engine is not running"),
            Pending::Combine { .. } => {
                self.importing = false;
                self.notice = Some(("The engine is not running".into(), Tone::Danger));
            }
            Pending::Summary => {}
        }
    }

    pub fn refresh_summary(&mut self) {
        if let Some(v) = self.voice_id.clone() {
            self.request("dataset.summary", json!({ "voice_id": v }), Pending::Summary);
        }
    }

    pub fn choose_preset(&mut self, id: &str) {
        let Load::Ready(list) = &self.presets else { return };
        let Some(p) = list.iter().find(|p| p.id == id) else { return };
        let plan = Plan { id: p.id.clone(), language: self.lang().to_owned(), title: p.title.clone(), target_minutes: p.target_minutes, trains_rvc: p.trains_rvc, parts: p.parts.clone() };
        self.begin(plan);
    }

    pub fn begin(&mut self, plan: Plan) {
        self.takes = vec![Take::default(); plan.parts.len()];
        self.plan = Some(plan);
        self.current = 0;
        self.summary = None;
        self.notice = None;
        self.phase = Phase::Session;
        self.ensure_voice();
        self.refresh_summary();
    }

    pub fn analyze_own(&mut self) {
        let text = self.own.text.trim().to_string();
        if text.is_empty() || self.analyzing {
            return;
        }
        self.analyze_error = None;
        let language = self.own_language.clone();
        self.analyzing = self.request("text.analyze", json!({ "text": text, "language": language }), Pending::Analyze);
        if !self.analyzing {
            self.analyze_error = Some("The engine is still starting".into());
        }
    }

    pub fn back_to_pick(&mut self) {
        self.cancel_recording();
        self.phase = Phase::Pick;
    }

    pub fn goto(&mut self, part: usize) {
        if self.active.is_some() {
            return;
        }
        let n = self.plan.as_ref().map_or(0, |p| p.parts.len());
        if part < n {
            self.current = part;
            self.notice = None;
            self.phase = Phase::Session;
        }
    }

    pub fn cancel_recording(&mut self) {
        if let Some(mut a) = self.active.take() {
            if let Some(r) = a.recorder.take() {
                let path = self.tmp_path(a.part);
                let _ = guard(|| r.stop(&path));
            }
        }
    }

    fn tmp_path(&self, part: usize) -> PathBuf {
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
        let dir = self.data_dir.join("tmp");
        let _ = std::fs::create_dir_all(&dir);
        dir.join(format!("record-part{}-{stamp}.wav", part + 1))
    }

    /// Space / the big button.
    pub fn toggle_recording(&mut self) {
        if self.active.is_some() {
            self.stop_recording();
        } else {
            self.start_recording();
        }
    }

    fn start_recording(&mut self) {
        // the level test must not hold the microphone while recording
        self.mic_set_open(false);
        if self.plan.is_none() || self.voice_id.is_none() {
            self.notice = Some(("Preparing your voice, one moment".into(), widgets::Tone::Neutral));
            return;
        }
        self.capture_seq += 1;
        let seq = self.capture_seq;
        match guard(|| Recorder::start(self.input_device.as_deref(), Some(meter_sink(seq, on_capture)))) {
            Ok(rec) => {
                self.notice = None;
                self.active = Some(Active { recorder: Some(rec), seq, part: self.current, level: 0.0, peak: 0.0, elapsed_s: 0.0, clipped: false });
            }
            Err(e) => self.notice = Some((format!("Cannot open the microphone: {e}"), widgets::Tone::Danger)),
        }
    }

    fn stop_recording(&mut self) {
        let Some(mut a) = self.active.take() else { return };
        let part = a.part;
        let Some(rec) = a.recorder.take() else { return };
        let path = self.tmp_path(part);
        match guard(|| rec.stop(&path)) {
            Ok(info) => {
                if let Some(t) = self.takes.get_mut(part) {
                    *t = Take { status: TakeStatus::Checking, duration_s: info.duration_s, path: Some(info.path.clone()), ..Take::default() };
                }
                let p = info.path.clone();
                let expected = self.plan.as_ref().and_then(|pl| pl.parts.get(part)).map(|x| x.est_seconds).filter(|e| *e > 0.0);
                if !self.request("recording.check", json!({ "path": p, "expected_s": expected }), Pending::Check { part, path: info.path }) {
                    self.set_take_failed(part, "The engine is not running");
                }
            }
            Err(e) => self.set_take_failed(part, &format!("Could not save the recording: {e}")),
        }
    }

    /// Several files for the current part: the engine joins them into one WAV (`media.combine`), which is then
    /// imported like a single file.
    fn import_many(&mut self, paths: Vec<PathBuf>) {
        if self.active.is_some() || self.importing {
            return;
        }
        if self.plan.is_none() || self.voice_id.is_none() {
            self.notice = Some(("Preparing your voice, one moment".into(), Tone::Neutral));
            return;
        }
        let n = paths.len();
        let part = self.current;
        if self.request("media.combine", json!({ "paths": paths }), Pending::Combine { part }) {
            self.importing = true;
            self.notice = Some((format!("Combining {n} files into one recording…"), Tone::Neutral));
        }
    }

    fn on_combined(&mut self, part: usize, v: &Value) {
        let path = PathBuf::from(st(v, "path"));
        let used = v.get("files").and_then(Value::as_array).map_or(0, Vec::len);
        let skipped: Vec<String> = v.get("skipped").and_then(Value::as_array).map(|a| {
            a.iter().map(|x| format!("{} ({})", st(x, "name"), st(x, "reason"))).collect()
        }).unwrap_or_default();
        if self.current != part || self.active.is_some() {
            return; // the user moved on meanwhile
        }
        self.import_take(path);
        let mins = f(v, "duration_s") / 60.0;
        let mut msg = format!("{used} files combined into one recording ({mins:.1} min).");
        if !skipped.is_empty() {
            msg.push_str(&format!(" Skipped: {}.", skipped.join(", ")));
        }
        self.notice = Some((msg, if skipped.is_empty() { Tone::Neutral } else { Tone::Warning }));
    }

    /// Uses an existing recording (any format the engine reads) for the current part instead of recording live:
    /// it goes through the same `recording.check` -> `dataset.add` path as a live take.
    fn import_take(&mut self, path: PathBuf) {
        if self.active.is_some() {
            return;
        }
        if self.plan.is_none() || self.voice_id.is_none() {
            self.notice = Some(("Preparing your voice, one moment".into(), Tone::Neutral));
            return;
        }
        if self.phase == Phase::Summary {
            self.phase = Phase::Session;
        }
        let part = self.current;
        let had_saved = self.takes.get(part).is_some_and(|t| t.status == TakeStatus::Saved);
        if had_saved {
            if let Some(v) = self.voice_id.clone() {
                self.request("dataset.remove", json!({"voice_id": v, "part_index": part + 1}), Pending::Summary);
            }
        }
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        if let Some(t) = self.takes.get_mut(part) {
            *t = Take { status: TakeStatus::Checking, path: Some(path.clone()), imported: name, ..Take::default() };
        }
        self.notice = None;
        let expected = self.plan.as_ref().and_then(|pl| pl.parts.get(part)).map(|x| x.est_seconds).filter(|e| *e > 0.0);
        self.request("recording.check", json!({ "path": path, "expected_s": expected }), Pending::Check { part, path });
    }

    fn set_take_failed(&mut self, part: usize, msg: &str) {
        if let Some(t) = self.takes.get_mut(part) {
            t.status = TakeStatus::Failed;
            t.error = Some(msg.to_string());
        }
    }

    pub fn rerecord(&mut self) {
        if self.active.is_some() {
            return;
        }
        let part = self.current;
        let had_saved = self.takes.get(part).is_some_and(|t| t.status == TakeStatus::Saved);
        if let Some(t) = self.takes.get_mut(part) {
            *t = Take::default();
        }
        if had_saved {
            if let Some(v) = self.voice_id.clone() {
                self.request("dataset.remove", json!({"voice_id": v, "part_index": part + 1}), Pending::Summary);
            }
        }
        self.start_recording();
    }

    /// First part that has no saved take, after `from`.
    fn next_open(&self, from: usize) -> Option<usize> {
        (from + 1..self.takes.len()).find(|&i| self.takes[i].status != TakeStatus::Saved)
    }

    /// Minutes recorded so far (saved takes), from the summary when the engine reported one.
    pub fn total_minutes(&self) -> f32 {
        match &self.summary {
            Some(s) => s.usable_minutes,
            None => self.takes.iter().filter(|t| t.status == TakeStatus::Saved && t.verdict != Some(Verdict::Bad)).map(|t| t.duration_s).sum::<f32>() / 60.0,
        }
    }

    pub fn saved_count(&self) -> usize {
        self.takes.iter().filter(|t| t.status == TakeStatus::Saved).count()
    }

    fn apply_summary(&mut self, sum: Summary) {
        let stored = |this: &Self, n: u32| this.voice_id.as_ref().map(|v| this.data_dir.join("voices").join(v).join("dataset").join(format!("part{n}.wav")));
        for p in &sum.parts {
            let path = stored(self, p.index);
            if let Some(t) = (p.index as usize).checked_sub(1).and_then(|i| self.takes.get_mut(i)) {
                if t.status == TakeStatus::Empty || t.status == TakeStatus::Failed {
                    // a part restored from an earlier session: show it as saved, playable from the dataset
                    t.status = if p.verdict == Verdict::Bad { TakeStatus::Rejected } else { TakeStatus::Saved };
                    t.duration_s = p.duration_s;
                    t.verdict = Some(p.verdict);
                    t.issues = p.issues.clone();
                    t.path = path;
                }
            }
        }
        // parts removed from the dataset become empty again
        let have: Vec<usize> = sum.parts.iter().map(|p| p.index as usize).collect();
        for (i, t) in self.takes.iter_mut().enumerate() {
            if t.status == TakeStatus::Saved && !have.contains(&(i + 1)) {
                *t = Take::default();
            }
        }
        self.summary = Some(sum);
    }

    /// Handles the answer to one of our requests. Returns the id of a voice this screen just created.
    fn on_response(&mut self, id: u64, result: &Result<Value, crate::engine::protocol::EngineError>) -> Option<String> {
        let what = self.pending.remove(&id)?;
        let mut created = None;
        match (what, result) {
            (Pending::Presets, Ok(v)) => {
                let list = v.get("presets").cloned().and_then(|p| serde_json::from_value::<Vec<Preset>>(p).ok()).unwrap_or_default();
                if let Some(langs) = v.get("languages").cloned().and_then(|l| serde_json::from_value::<Vec<RecLang>>(l).ok()).filter(|l| !l.is_empty()) {
                    self.languages = langs;
                }
                self.presets = Load::Ready(list);
            }
            (Pending::Presets, Err(e)) => self.presets = Load::Failed(e.message.clone()),
            (Pending::Analyze, Ok(v)) => {
                self.analyzing = false;
                self.analysis = Some(Analysis {
                    language: st(v, "language").to_string(),
                    detected: v.get("language_detected").and_then(Value::as_bool).unwrap_or(false),
                    words: f(v, "words") as u32,
                    est_minutes: f(v, "est_minutes"),
                    coverage: f(v, "phoneme_coverage"),
                    missing: str_list(v, "missing_phonemes"),
                    level: Level::parse(st(v, "level")),
                    message: st(v, "message").to_string(),
                    parts: v.get("parts").cloned().and_then(|p| serde_json::from_value(p).ok()).unwrap_or_default(),
                });
            }
            (Pending::Analyze, Err(e)) => {
                self.analyzing = false;
                self.analyze_error = Some(e.message.clone());
            }
            (Pending::CreateVoice, r) => {
                self.creating_voice = false;
                match r {
                    Ok(v) => {
                        created = v.pointer("/voice/id").and_then(Value::as_str).map(String::from);
                        self.voice_id = created.clone();
                        self.voices_dirty = true;
                        self.refresh_summary();
                    }
                    Err(e) => self.notice = Some((format!("Cannot create a voice: {}", e.message), widgets::Tone::Danger)),
                }
            }
            (Pending::Check { part, path }, Ok(v)) => self.on_check(part, path, v),
            (Pending::Check { part, .. }, Err(e)) => self.set_take_failed(part, &e.message),
            (Pending::Add { part, dest }, Ok(v)) => self.on_added(part, dest, v),
            (Pending::Add { part, .. }, Err(e)) => self.set_take_failed(part, &e.message),
            (Pending::Combine { part }, r) => {
                self.importing = false;
                match r {
                    Ok(v) => self.on_combined(part, v),
                    Err(e) => self.notice = Some((format!("Could not combine the files: {}", e.message), Tone::Danger)),
                }
            }
            (Pending::Summary, Ok(v)) => {
                let s = parse_summary(v.get("summary").unwrap_or(v));
                self.apply_summary(s);
                self.voices_dirty = true;
            }
            (Pending::Summary, Err(_)) => {}
        }
        created
    }

    fn on_check(&mut self, part: usize, path: PathBuf, v: &Value) {
        let verdict = Verdict::parse(st(v, "verdict"));
        let issues = str_list(v, "issues");
        let duration = if f(v, "duration_s") > 0.0 { f(v, "duration_s") } else { self.takes.get(part).map_or(0.0, |t| t.duration_s) };
        let Some(t) = self.takes.get_mut(part) else { return };
        t.verdict = Some(verdict);
        t.issues = issues;
        t.duration_s = duration;
        if verdict == Verdict::Bad {
            t.status = TakeStatus::Rejected;
            return;
        }
        t.status = TakeStatus::Saving;
        let Some(voice) = self.voice_id.clone() else {
            self.set_take_failed(part, "No voice to save into");
            return;
        };
        let dest = self.data_dir.join("voices").join(&voice).join("dataset").join(format!("part{}.wav", part + 1));
        let expected = self.plan.as_ref().and_then(|pl| pl.parts.get(part)).map(|x| x.est_seconds).filter(|e| *e > 0.0);
        let params = json!({"voice_id": voice, "part_index": part + 1, "path": path, "expected_s": expected});
        if !self.request("dataset.add", params, Pending::Add { part, dest }) {
            self.set_take_failed(part, "The engine is not running");
        }
    }

    fn on_added(&mut self, part: usize, dest: PathBuf, v: &Value) {
        let verdict = v.pointer("/part/verdict").and_then(Value::as_str).map(Verdict::parse);
        let dur = v.pointer("/part/duration_s").and_then(Value::as_f64).map(|d| d as f32);
        let issues = v.pointer("/part").map(|p| str_list(p, "issues"));
        let mut rejected = false;
        if let Some(t) = self.takes.get_mut(part) {
            t.status = TakeStatus::Saved;
            t.path = Some(dest);
            if let Some(d) = dur {
                t.duration_s = d;
            }
            // the engine judges the trimmed take against the expected length: its verdict and tips win
            if let Some(vd) = verdict {
                t.verdict = Some(vd);
                if let Some(i) = issues {
                    t.issues = i;
                }
                if vd == Verdict::Bad {
                    t.status = TakeStatus::Rejected;
                    rejected = true;
                }
            }
        }
        if rejected {
            // a bad part is not counted anyway, take it out so the dataset only holds usable audio
            if let Some(v) = self.voice_id.clone() {
                self.request("dataset.remove", json!({"voice_id": v, "part_index": part + 1}), Pending::Summary);
            }
            return;
        }
        if let Some(s) = v.get("summary") {
            let s = parse_summary(s);
            self.summary = Some(s);
            self.voices_dirty = true;
        }
        let (verdict, dur) = self.takes.get(part).map_or((None, 0.0), |t| (t.verdict, t.duration_s));
        // auto-advance only on a clean take, and only while the reader is still on that part
        if verdict == Some(Verdict::Ok) && part == self.current {
            self.notice = Some((format!("Part {} saved ({})", part + 1, widgets::fmt_dur(dur)), widgets::Tone::Success));
            match self.next_open(part) {
                Some(n) => self.current = n,
                None => self.phase = Phase::Summary,
            }
        }
    }

}

/// A meter sink for capture `seq`: the audio thread's level events reach `apply` on the UI thread, newest only (at
/// most one waits at a time), at the recorder's capped rate.
pub(crate) fn meter_sink(seq: u64, apply: fn(&mut AppState, u64, CaptureEvent) -> bool) -> MeterSink {
    let feed = Latest::new(move |s: &mut AppState, ev: CaptureEvent| apply(s, seq, ev));
    std::sync::Arc::new(move |ev| feed.send(ev))
}

/// A level event of the live recording (meter, clip warning, elapsed time).
fn on_capture(s: &mut AppState, seq: u64, ev: CaptureEvent) -> bool {
    let Some(a) = s.record.active.as_mut().filter(|a| a.seq == seq) else { return false };
    match ev {
        CaptureEvent::Level { level, peak, elapsed_s } => {
            a.level = if level > a.level { level } else { a.level * 0.82 + level * 0.18 };
            a.peak = a.peak.max(peak);
            a.elapsed_s = elapsed_s;
            if peak >= 0.99 {
                a.clipped = true;
            }
            true
        }
        // Stop reports it ("nothing was recorded" or a short take); keep the capture until then.
        CaptureEvent::Error(_) => false,
    }
}

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    let body = match s.record.phase {
        Phase::Pick => pick::presets(&s.record, p),
        Phase::Own => pick::own_text(&s.record, p),
        Phase::Session | Phase::Summary => session::view(&s.record, p),
    };
    let mut root = col()
        .key("record")
        .w_full()
        .h_full()
        .min_h(0.0)
        .focusable(true)
        .no_focus_ring()
        .on_key(|s: &mut AppState, ev: &KeyEvent, tx: &mut Tx| on_key(s, ev, tx))
        .child(body)
        .child(el().h(0.0));
    if let Some(menu) = mic::overlay(&s.record, p) {
        root = root.child(menu);
    }
    root
}

/// "Import file...": pick an existing recording for the current part (native dialog on its own thread).
pub fn import_file(s: &mut AppState) {
    if s.record.importing || s.record.active.is_some() || s.shell.scripted() {
        return;
    }
    s.record.importing = true;
    let exts = super::formats::import_extensions(s);
    std::thread::spawn(move || {
        let dialog = rfd::FileDialog::new().set_title("Choose recordings for this part (several are joined into one)").add_filter("Audio and video", &exts);
        let choice = dialog.pick_files().unwrap_or_default();
        events::post(move |s| {
            s.record.importing = false;
            import_paths(s, choice);
        });
    });
}

/// Files dropped on the window while recording: they become the current part's take (several are joined into one).
pub fn import_dropped(s: &mut AppState, files: Vec<PathBuf>) {
    s.record.import_over = false;
    if s.record.plan.is_none() {
        shell::toast(s, Tone::Warning, "Pick a reading first, then drop the recording on a part.");
        return;
    }
    shell::stop_audio(s);
    s.record.playing = false;
    import_paths(s, files);
}

fn import_paths(s: &mut AppState, mut paths: Vec<PathBuf>) {
    match paths.len() {
        0 => {}
        1 => s.record.import_take(paths.remove(0)),
        _ => s.record.import_many(paths),
    }
}

/// Space / the big button: start or stop the capture (stops playback first).
pub fn toggle_recording(s: &mut AppState) {
    if s.record.active.is_none() {
        shell::stop_audio(s);
        s.record.playing = false;
    }
    s.record.toggle_recording();
}

/// "Record again" on a rejected or saved part.
pub fn rerecord(s: &mut AppState) {
    shell::stop_audio(s);
    s.record.playing = false;
    s.record.rerecord();
}

/// Plays or stops the current part's take through the shell's player.
pub fn toggle_play(s: &mut AppState) {
    if s.record.playing {
        shell::stop_audio(s);
        s.record.playing = false;
        return;
    }
    let Some(path) = s.record.takes.get(s.record.current).and_then(|t| t.path.clone()) else { return };
    match shell::play(s, &path) {
        Ok(()) => s.record.playing = true,
        Err(e) => s.record.notice = Some((e, Tone::Danger)),
    }
}

/// Space toggles record/stop, Left/Right move between parts. Returns true when it took the key.
pub fn on_key(s: &mut AppState, ev: &KeyEvent, _tx: &mut Tx) -> bool {
    if s.screen != Screen::Record || !matches!(s.record.phase, Phase::Session | Phase::Summary) {
        return false;
    }
    if ev.mods.cmd || ev.mods.ctrl || ev.mods.alt {
        return false;
    }
    match &ev.key {
        Key::Text(t) if t == " " => {
            if s.record.phase == Phase::Summary {
                s.record.phase = Phase::Session;
            }
            toggle_recording(s);
            true
        }
        Key::Right => {
            let n = s.record.current + 1;
            s.record.goto(n);
            true
        }
        Key::Left => {
            if s.record.current > 0 {
                let n = s.record.current - 1;
                s.record.goto(n);
            }
            true
        }
        _ => false,
    }
}

/// The language of the reading presets was picked: load that language's presets (remembered for next time).
pub fn set_language(s: &mut AppState, code: &str) {
    if s.record.lang() == code || s.record.recording() {
        return;
    }
    s.record.language = code.to_owned();
    s.record.presets = Load::NotAsked;
    if !s.shell.scripted() {
        super::onboarding::save_setting("record_language", json!(code));
    }
}

/// Starts reading `plan` into the right voice: the active one when it is in the plan's language (or still empty: it
/// takes the language), else a new voice for that language.
fn begin_in_language(s: &mut AppState, plan: Plan) {
    let lang = plan.language.clone();
    let active = shell::active_voice(s).cloned();
    match active {
        Some(v) if v.lang() != lang && v.parts == 0 => {
            shell::request(s, "voices.rename", json!({ "id": v.id, "language": lang }));
            if let Some(info) = s.shell.voices.iter_mut().find(|x| x.id == v.id) {
                info.language = lang.clone();
            }
            s.record.begin(plan);
        }
        Some(v) if v.lang() != lang => {
            s.record.voice_id = None;
            s.record.new_voice_name = Some(format!("{} ({})", v.name, super::lang::name(&lang)));
            s.record.begin(plan);
            s.record.notice = Some((
                format!("Recording into a new voice for {}: “{}” keeps its {} recordings.", super::lang::name(&lang), v.name, super::lang::name(v.lang())),
                Tone::Neutral,
            ));
        }
        _ => s.record.begin(plan),
    }
}

/// A preset card was pressed.
pub fn choose_preset(s: &mut AppState, id: &str) {
    let Load::Ready(list) = &s.record.presets else { return };
    let Some(p) = list.iter().find(|p| p.id == id) else { return };
    let plan = Plan { id: p.id.clone(), language: s.record.lang().to_owned(), title: p.title.clone(), target_minutes: p.target_minutes, trains_rvc: p.trains_rvc, parts: p.parts.clone() };
    begin_in_language(s, plan);
}

/// "Record these parts" after analysing the user's own text.
pub fn record_own(s: &mut AppState) {
    let Some(a) = s.record.analysis.clone() else { return };
    let language = if a.language.is_empty() { "en".to_owned() } else { a.language.clone() };
    let plan = Plan { id: "custom".into(), language, title: "Your own text".into(), target_minutes: a.est_minutes, trains_rvc: a.level >= Level::Trainable, parts: a.parts };
    begin_in_language(s, plan);
}

/// Called for every engine message (responses, events, exit).
pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Response { id, result } => {
            if let Some(voice) = s.record.on_response(*id, result) {
                // select the new voice in the header; the refreshed list keeps it
                s.shell.active_voice = Some(voice);
            }
        }
        EngineMsg::Exited { .. } => {
            // requests in flight will never be answered
            let lost: Vec<Pending> = s.record.pending.drain().map(|(_, w)| w).collect();
            for w in lost {
                s.record.send_failed(w);
            }
        }
        _ => {}
    }
}

/// After every event: drop highlight, microphone chooser, voice sync and queued requests.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;
    if s.record.import_over != (s.formats.over && s.screen == Screen::Record) {
        s.record.import_over = s.formats.over && s.screen == Screen::Record;
        changed = true;
    }
    changed |= mic::settle(s);

    // follow the header's voice
    if s.shell.active_voice != s.record.voice_id && !s.record.creating_voice && s.record.active.is_none() {
        let active = s.shell.active_voice.clone();
        s.record.set_voice(active);
        changed = true;
    }
    if s.record.plan.is_some() && s.record.voice_id.is_none() && shell::engine_ready(s) {
        s.record.ensure_voice();
    }
    if !s.record.prefs_loaded && !s.shell.scripted() {
        s.record.prefs_loaded = true;
        if let Some(l) = super::onboarding::setting("record_language").and_then(|v| v.as_str().map(str::to_owned)).filter(|l| !l.is_empty()) {
            s.record.language = l;
        }
    }
    if matches!(s.record.presets, Load::NotAsked) && shell::engine_ready(s) {
        s.record.presets = Load::Loading;
        let language = s.record.lang().to_owned();
        s.record.request("presets.list", json!({ "language": language }), Pending::Presets);
        changed = true;
    }

    // send what the screen queued
    if !s.record.outbox.is_empty() {
        let queued = std::mem::take(&mut s.record.outbox);
        for (method, params, what) in queued {
            match shell::engine_ready(s).then(|| shell::request(s, &method, params)).flatten() {
                Some(id) => {
                    s.record.pending.insert(id, what);
                }
                None => s.record.send_failed(what),
            }
        }
        changed = true;
    }
    if s.record.voices_dirty && shell::engine_ready(s) {
        s.record.voices_dirty = false;
        shell::refresh_voices(s);
    }
    changed
}

/// The player ran dry: the take that was playing has ended.
pub fn on_playback_end(s: &mut AppState) -> bool {
    std::mem::replace(&mut s.record.playing, false)
}
