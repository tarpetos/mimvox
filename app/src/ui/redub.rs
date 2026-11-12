//! Redub screen: take an existing recording (audio or video) and say it again in your voice.
//!
//! Three modes:
//! * **Translate & dub** (the default) takes speech in any language and says it in another one (the "To" picker, English
//!   by default, any language the engine speaks) in your voice, in one job (`redub.dub`: transcribe, translate, speak,
//!   save). With "Review text before dubbing" the job stops after the translation: a two-column table (original |
//!   translation, editable) and "Dub it", which finishes with `redub.render` in `respeak` mode and the edited segments.
//! * **Convert voice** keeps the original timing and delivery and swaps in a trained voice (`redub.render`, mode
//!   `convert`).
//! * **Re-speak** transcribes the speech (`redub.transcribe`), lets you edit or translate the text, then speaks it
//!   again in the voice's own delivery (`redub.render`, mode `respeak`, with the edited segments).
//!
//! `redub.prepare` is sent as soon as a file is chosen, so the first click starts at once. The last mode, spoken language,
//! review choice and target language are remembered in `settings.json` (`redub_mode`, `redub_language`, `redub_review`,
//! `redub_target`), and so is "Translation quality" (`redub_translator`: `nllb-1.3b` | `nllb-600m`), offered when the
//! engine has both translators (`hello` -> `translators`). The language lists come from the engine (`ui::lang`).
//! Translate & dub and Re-speak have "Keep original timing" (`timing`: `keep` | `natural`), on by default for a video
//! and off for an audio file, remembered per kind once changed (`redub_timing`); Redub saves audio only.
//! Dubbing into English, or re-speaking English, with a voice that has no English reference clip shows Speak's
//! "Accent" choice (`speak::accent_block`, the same setting), sent as `accent` with `redub.dub`, with the
//! `redub.render` that finishes a reviewed dub and with a Re-speak's `redub.render`. A job spoken from a clip in another
//! language than its text (`reference_note`) shows a warning toast.
//!
//! The file comes from a drop anywhere on the window (see `formats::window_drops`) or "Choose file..." (rfd on its
//! own thread). The voice is the header's active voice; Voice match, Expressiveness and Quality are the Speak
//! screen's own controls and state. Results land in a session history with A/B playback and the shared
//! "Export as..." control. Requests and events follow `contract/engine-protocol.md` (`redub.*`).

use super::components::{
    Button, Tone, accent_glass, badge, empty_state, glass_card, icon_button, icon_el, icons, progress_bar, section_panel, segmented_help, status_dot, well,
};
use super::formats;
use super::lang;
use super::shell::{self, EngineStatus, VoiceInfo};
use super::speak;
use super::theme::{self, Palette, motion, radius, ty};
use super::trash;
use super::playbar;
use super::voices::kit;
use crate::engine::EngineMsg;
use crate::engine::protocol::{self, EngineEvent, JobStarted, RedubDone, RedubDubParams, RedubMode, RedubProbe, RedubReview, RedubSegmentIn, RedubTiming, RedubTranscript, TextLanguage};
use crate::state::{AppState, Screen};
use rune::{Cx, El, ScrollSpec, TextEdit, col, edit_access, el, row, text, text_input};
use serde_json::json;
use std::path::{Path, PathBuf};

/// A language code ("uk", "zh-CN", "pt") as a name for the screen; an unknown code is shown upper-case.
fn lang_name(code: &str) -> String {
    lang::name(code)
}

fn mode_key(m: RedubMode) -> &'static str {
    m.as_str()
}

fn mode_from_key(k: &str) -> Option<RedubMode> {
    [RedubMode::Dub, RedubMode::Convert, RedubMode::Respeak].into_iter().find(|m| m.as_str() == k)
}

// -------------------------------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Probe {
    /// Not asked yet (the engine may still be starting).
    Pending,
    Asking(u64),
    Ready(RedubProbe),
    Failed(String),
}

#[derive(Clone, Debug)]
struct Source {
    path: PathBuf,
    name: String,
    probe: Probe,
    /// `redub.prepare` was sent for this file (reset when the engine restarts).
    prepared: bool,
}

impl Source {
    /// A file the player can play as "the original": the engine's WAV preview, or the file itself when it is a WAV.
    fn preview(&self) -> Option<PathBuf> {
        if let Probe::Ready(p) = &self.probe {
            if let Some(pv) = &p.preview_path {
                return Some(pv.clone());
            }
        }
        is_wav(&self.path).then(|| self.path.clone())
    }
}

fn is_wav(p: &Path) -> bool {
    p.extension().is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("wav"))
}

/// One editable line of the transcript.
struct Seg {
    key: u64,
    start: f32,
    end: f32,
    /// The original-language text shown next to the editable translation while reviewing a dub.
    source: String,
    edit: TextEdit,
}

/// A running engine job.
#[derive(Clone, Debug)]
struct Run {
    req: u64,
    job: Option<String>,
    started: f64,
    stage: String,
    pct: f32,
    /// (k, n) while a re-spoken transcript renders segment by segment.
    segment: Option<(u32, u32)>,
    /// Cancel pressed before the job id arrived.
    cancel: bool,
    /// The voice speaking (to record its detected reference language from `redub.done`).
    voice_id: String,
    /// The foreign-reference warning was shown with the answer already.
    warned: bool,
}

impl Run {
    fn new(req: u64, now: f64) -> Run {
        Run { req, job: None, started: now, stage: String::new(), pct: 0.0, segment: None, cancel: false, voice_id: String::new(), warned: false }
    }

    fn for_voice(req: u64, now: f64, voice_id: &str) -> Run {
        Run { voice_id: voice_id.to_owned(), ..Run::new(req, now) }
    }
}

/// The language a Re-speak speaks: the transcript's (the target after a translation), else what the options say
/// ("Translate the transcript" to, or the spoken language when it is not Auto-detect).
fn respeak_lang(r: &RedubState) -> Option<String> {
    if let Some(l) = r.transcript_lang.as_deref().filter(|l| !l.trim().is_empty()) {
        return Some(l.to_owned());
    }
    if r.translate {
        return Some(r.to.clone());
    }
    (!r.from.eq_ignore_ascii_case("auto")).then(|| r.from.clone())
}

#[derive(Clone, Debug, Default)]
enum Job {
    #[default]
    Idle,
    Transcribing(Run),
    Rendering { run: Run, mode: RedubMode, source: String, voice: String },
}

/// One finished redub in the session history (newest first).
#[derive(Clone, Debug)]
pub struct Redub {
    pub id: u64,
    pub name: String,
    pub voice: String,
    pub mode: RedubMode,
    /// The saved file (`<data>/outputs`, in the chosen format): Show in folder, Open file, Export.
    pub path: PathBuf,
    /// A WAV of it to play ("Result"; the saved file itself when that is a WAV).
    pub play: PathBuf,
    pub preview: Option<PathBuf>,
    pub duration_s: f32,
    pub elapsed_s: f32,
    /// Seconds the spoken result ran past the original (a dub only; `None` when it did not).
    pub drift_s: Option<f32>,
    /// The language a dub was spoken in.
    pub to: Option<String>,
}

pub struct RedubState {
    file: Option<Source>,
    pub mode: RedubMode,
    /// The language of the speech (`auto` or a code).
    pub from: String,
    /// The language to translate to and speak (Translate & dub, and Re-speak with "Translate").
    pub to: String,
    translate: bool,
    /// "Review text before dubbing" (Translate & dub).
    review_first: bool,
    /// "Translation quality: Faster" (NLLB-200 600M) instead of Better (1.3B); only asked when both are installed.
    faster_translation: bool,
    /// "Keep original timing" as the user set it for audio files (`[0]`) and video files (`[1]`); `None` until they
    /// change it (then it is on for video and off for audio).
    keep_timing: [Option<bool>; 2],
    segments: Vec<Seg>,
    /// The segments are a dub's translation waiting for the user's edits (shown as the two-column review).
    reviewing: bool,
    /// Language of the text in `segments` (the target after a translation, never the source language).
    transcript_lang: Option<String>,
    /// The speech language when the text was translated ("English · from Ukrainian").
    transcript_from: Option<String>,
    transcript_translated: bool,
    prefs_loaded: bool,
    prefs_dirty: bool,
    job: Job,
    pub history: Vec<Redub>,
    /// The latest result for the current file (shown in the result card).
    result: Option<u64>,
    error: Option<String>,
    /// The file dialog is open (its answer arrives as an event).
    picking: bool,
    next_id: u64,
    next_seg: u64,
}

impl Default for RedubState {
    fn default() -> Self {
        RedubState {
            file: None,
            mode: RedubMode::Dub,
            from: "auto".into(),
            to: "en".into(),
            translate: false,
            review_first: false,
            faster_translation: false,
            keep_timing: [None, None],
            segments: Vec::new(),
            reviewing: false,
            transcript_lang: None,
            transcript_from: None,
            transcript_translated: false,
            prefs_loaded: false,
            prefs_dirty: false,
            job: Job::Idle,
            history: Vec::new(),
            result: None,
            error: None,
            picking: false,
            next_id: 0,
            next_seg: 0,
        }
    }
}

impl RedubState {
    /// A transcription or render is running.
    pub fn busy(&self) -> bool {
        !matches!(self.job, Job::Idle)
    }
    fn kept_segments(&self) -> impl Iterator<Item = &Seg> {
        self.segments.iter().filter(|g| !g.edit.text.trim().is_empty())
    }
    /// Forgets the transcript or review (a new file, a new run, another mode).
    fn clear_text(&mut self) {
        self.segments.clear();
        self.reviewing = false;
        self.transcript_lang = None;
        self.transcript_from = None;
        self.transcript_translated = false;
    }
    /// "Ukrainian · from English" for the transcript or review header (never the source language alone for a translation).
    fn lang_label(&self) -> Option<String> {
        let l = self.transcript_lang.as_deref().filter(|l| !l.is_empty())?;
        let mut label = lang_name(l);
        if let Some(f) = self.transcript_from.as_deref() {
            let n = lang_name(f);
            if !n.is_empty() && !f.eq_ignore_ascii_case("auto") && !f.eq_ignore_ascii_case(l) {
                label = format!("{label} · from {n}");
            }
        }
        Some(label)
    }
    /// The language of the segment texts, for `redub.render`.
    fn text_language(&self) -> TextLanguage {
        match self.transcript_lang.clone().filter(|l| !l.trim().is_empty()) {
            Some(l) if self.transcript_translated => TextLanguage::Translated(l),
            Some(l) => TextLanguage::Spoken(l),
            None => TextLanguage::Unknown,
        }
    }
    fn persist(&mut self) {
        self.prefs_dirty = true;
    }
}

/// Extensions taken for video until `redub.probe` says whether the file has a picture track.
const VIDEO_EXTS: &[&str] = &["mp4", "mkv", "mov", "webm", "avi", "m4v"];

/// The chosen file is a video (the probe's `has_video`, else its extension). Without a file: true (fitting the timing
/// is what a video needs, and it was the only behaviour before the switch).
fn source_is_video(r: &RedubState) -> bool {
    let Some(f) = r.file.as_ref() else { return true };
    if let Probe::Ready(RedubProbe { has_video: Some(v), .. }) = &f.probe {
        return *v;
    }
    f.path.extension().and_then(|e| e.to_str()).is_some_and(|e| VIDEO_EXTS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// "Keep original timing": the user's choice for this kind of file, else on for a video and off for audio.
fn keep_timing(r: &RedubState) -> bool {
    let video = source_is_video(r);
    r.keep_timing[usize::from(video)].unwrap_or(video)
}

fn timing(r: &RedubState) -> RedubTiming {
    if keep_timing(r) { RedubTiming::Keep } else { RedubTiming::Natural }
}

fn toggle_keep_timing(s: &mut AppState) {
    let r = &mut s.redub;
    if r.busy() {
        return;
    }
    let (video, on) = (source_is_video(r), keep_timing(r));
    r.keep_timing[usize::from(video)] = Some(!on);
    r.persist();
}

/// The saved `redub_timing` setting: `{"audio": "keep"|"natural", "video": ...}`, each only once the user chose.
fn parse_timing_pref(v: Option<serde_json::Value>) -> [Option<bool>; 2] {
    let get = |k: &str| v.as_ref().and_then(|v| v.get(k)).and_then(|x| x.as_str()).and_then(|x| match x {
        "keep" => Some(true),
        "natural" => Some(false),
        _ => None,
    });
    [get("audio"), get("video")]
}

fn timing_pref(t: [Option<bool>; 2]) -> serde_json::Value {
    let mut v = json!({});
    for (k, on) in ["audio", "video"].into_iter().zip(t) {
        if let Some(on) = on {
            v[k] = json!(if on { "keep" } else { "natural" });
        }
    }
    v
}

/// "Keep original timing" with the hint for its current state (Translate & dub and Re-speak).
fn timing_block(s: &AppState, p: Palette, key: &str) -> El {
    let r = &s.redub;
    let on = keep_timing(r);
    // Redub saves audio only (nothing is muxed back into a video), so for a video the hint says what the choice means
    // for laying the result back over the picture.
    let hint = match (on, source_is_video(r)) {
        (true, true) => "Fits each phrase into the original timing; may speed up speech. Stays in sync with the video (the result is audio only).",
        (true, false) => "Fits each phrase into the original timing; may speed up speech.",
        (false, true) => "Natural pace; the result can be longer, so it drifts out of sync with the video.",
        (false, false) => "Natural pace; the result can be longer.",
    };
    col().key(key.to_owned()).w_full().gap(3.0)
        .child(kit::checkbox(&p, on, "Keep original timing", toggle_keep_timing))
        .child(text(hint).font(ty::CAPTION).color(p.text_faint).wrap().pl(32.0))
}

const TRANSLATOR_BETTER: &str = "nllb-1.3b";
const TRANSLATOR_FASTER: &str = "nllb-600m";

/// The engine has both translators, so "Translation quality" is a choice.
fn translator_choice(s: &AppState) -> bool {
    let has = |t: &str| s.shell.translators.iter().any(|x| x == t);
    has(TRANSLATOR_BETTER) && has(TRANSLATOR_FASTER)
}

/// The `translator` to send: the user's choice when there is one, else the engine picks the best one installed.
fn translator(s: &AppState) -> Option<&'static str> {
    translator_choice(s).then_some(if s.redub.faster_translation { TRANSLATOR_FASTER } else { TRANSLATOR_BETTER })
}

// -------------------------------------------------------------------------------------------------
// Actions
// -------------------------------------------------------------------------------------------------

/// Why the engine cannot take a Redub job right now.
fn engine_blocker(s: &AppState) -> Option<&'static str> {
    match &s.shell.engine {
        EngineStatus::Idle | EngineStatus::Starting { .. } => return Some("Waiting for the engine to start…"),
        EngineStatus::Stopped { .. } | EngineStatus::Unavailable { .. } => return Some("The engine is not running. Press Retry above."),
        EngineStatus::Ready { .. } => {}
    }
    if super::train::is_training(s) {
        return Some("A voice is training and is using the GPU. Redub is back when it finishes.");
    }
    if s.speak.is_generating() {
        return Some("Speech is being generated on the Speak tab. Try again in a moment.");
    }
    None
}

/// Why Transcribe cannot run right now.
fn transcribe_blocker(s: &AppState) -> Option<String> {
    if let Some(b) = engine_blocker(s) {
        return Some(b.to_owned());
    }
    match s.redub.file.as_ref().map(|f| &f.probe) {
        None => Some("Choose a file first.".into()),
        Some(Probe::Failed(m)) => Some(m.clone()),
        Some(Probe::Pending | Probe::Asking(_)) => Some("Reading the file…".into()),
        Some(Probe::Ready(_)) => None,
    }
}

/// Why Render cannot run right now, and whether it deserves the warning colour.
fn render_blocker(s: &AppState) -> Option<(String, bool)> {
    if let Some(b) = transcribe_blocker(s) {
        let calm = b == "Choose a file first." || b == "Reading the file…";
        return Some((b, !calm));
    }
    let Some(v) = shell::active_voice(s) else {
        return Some(("Record your voice first, or pick one at the top right.".into(), true));
    };
    match s.redub.mode {
        RedubMode::Dub if s.redub.reviewing && s.redub.kept_segments().next().is_none() => Some(("Every line is empty. Add some text.".into(), false)),
        RedubMode::Convert if !v.trained => Some((format!("“{}” is not trained yet. Convert voice needs a trained voice.", v.name), true)),
        RedubMode::Respeak if s.redub.kept_segments().next().is_none() => {
            let hint = if s.redub.segments.is_empty() { "Transcribe the file first, then edit the text." } else { "Every segment is empty. Add some text." };
            Some((hint.into(), false))
        }
        RedubMode::Respeak => unspeakable_lang(s).map(|l| (format!("The voice cannot speak {l}. Use Translate & dub, or tick Translate."), true)),
        _ => None,
    }
}

/// Re-speak cannot say this text: it is not translated and not in a language the voice speaks.
fn unspeakable_lang(s: &AppState) -> Option<String> {
    let r = &s.redub;
    if r.segments.is_empty() || r.transcript_translated {
        return None;
    }
    let l = r.transcript_lang.as_deref()?;
    (!l.trim().is_empty() && !lang::is_spoken(s, l)).then(|| lang_name(l))
}

/// The picker chose the language of the speech.
pub fn set_from(s: &mut AppState, code: &str) {
    if s.redub.busy() || s.redub.reviewing {
        return;
    }
    s.redub.from = code.to_owned();
    s.redub.persist();
}

/// The picker chose the language to translate to and speak.
pub fn set_to(s: &mut AppState, code: &str) {
    if s.redub.busy() || s.redub.reviewing {
        return;
    }
    s.redub.to = code.to_owned();
    s.redub.persist();
}

/// Opens `path` as the file to redub (from a drop or the picker). The engine probes it as soon as it is ready.
pub fn set_file(s: &mut AppState, path: PathBuf) {
    if s.redub.busy() {
        shell::toast(s, Tone::Warning, "Wait for the current job to finish, or cancel it, before choosing another file.");
        return;
    }
    if !path.is_file() && !s.shell.scripted() {
        shell::toast(s, Tone::Warning, "That file could not be found.");
        return;
    }
    stop_play(s);
    let name = path.file_name().map_or_else(|| path.display().to_string(), |n| n.to_string_lossy().into_owned());
    let r = &mut s.redub;
    r.file = Some(Source { path, name, probe: Probe::Pending, prepared: false });
    r.clear_text();
    r.result = None;
    r.error = None;
}

fn clear_file(s: &mut AppState) {
    if s.redub.busy() {
        return;
    }
    stop_play(s);
    let r = &mut s.redub;
    r.file = None;
    r.clear_text();
    r.result = None;
    r.error = None;
}

/// "Choose file...": the native picker on its own thread; its answer arrives as an event.
pub fn choose_file(s: &mut AppState) {
    if s.redub.picking || s.redub.busy() || s.shell.scripted() {
        return;
    }
    s.redub.picking = true;
    let exts = formats::import_extensions(s);
    let start = s.redub.file.as_ref().and_then(|f| f.path.parent().map(Path::to_path_buf)).filter(|d| d.is_dir());
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_title("Choose an audio or video file").add_filter("Audio and video", &exts);
        if let Some(d) = start {
            dialog = dialog.set_directory(d);
        }
        let choice = dialog.pick_file();
        crate::ui::events::post(move |s| {
            s.redub.picking = false;
            if let Some(path) = choice {
                set_file(s, path);
            }
        });
    });
}

fn set_mode(s: &mut AppState, mode: RedubMode) {
    if s.redub.busy() {
        return;
    }
    if s.redub.mode != mode && (s.redub.reviewing || mode == RedubMode::Dub) {
        // A dub review and a re-speak transcript are different texts: do not carry one into the other.
        s.redub.clear_text();
    }
    s.redub.mode = mode;
    s.redub.error = None;
    s.redub.persist();
}

fn start_transcribe(s: &mut AppState) {
    if s.redub.busy() {
        return;
    }
    if let Some(why) = transcribe_blocker(s) {
        shell::toast(s, Tone::Warning, why);
        return;
    }
    let Some(path) = s.redub.file.as_ref().map(|f| f.path.clone()) else { return };
    let to = s.redub.to.clone();
    let (method, params) = protocol::redub_transcribe(&path, &s.redub.from, s.redub.translate.then_some(to.as_str()), translator(s));
    let Some(req) = shell::request(s, method, params) else {
        shell::toast(s, Tone::Danger, "Could not reach the engine.");
        return;
    };
    stop_play(s);
    let now = shell::clock_now(s);
    let r = &mut s.redub;
    r.clear_text();
    r.result = None;
    r.error = None;
    r.job = Job::Transcribing(Run::new(req, now));
}

/// The big button: starts the dub (Translate & dub), or, with the review open, speaks the edited translation.
fn start_primary(s: &mut AppState) {
    if s.redub.mode == RedubMode::Dub && !s.redub.reviewing { start_dub(s) } else { start_render(s) }
}

fn start_dub(s: &mut AppState) {
    if s.redub.busy() {
        return;
    }
    if let Some((why, _)) = render_blocker(s) {
        shell::toast(s, Tone::Warning, why);
        return;
    }
    let (Some(src), Some(voice)) = (s.redub.file.clone(), shell::active_voice(s).cloned()) else { return };
    let out = formats::output_format(s);
    let (method, params) = protocol::redub_dub(&RedubDubParams {
        path: &src.path,
        voice_id: &voice.id,
        source_language: &s.redub.from,
        target_language: &s.redub.to,
        translator: translator(s),
        review: s.redub.review_first,
        voice_match: s.speak.voice_match,
        expressiveness: s.speak.expressiveness,
        quality: s.speak.quality,
        speed: s.speak.speed,
        output_format: &out,
        accent: speak::accent_param(s, &s.redub.to),
        timing: timing(&s.redub),
    });
    let Some(req) = shell::request(s, method, params) else {
        shell::toast(s, Tone::Danger, "Could not reach the engine.");
        return;
    };
    stop_play(s);
    let now = shell::clock_now(s);
    let r = &mut s.redub;
    r.clear_text();
    r.result = None;
    r.error = None;
    r.job = Job::Rendering { run: Run::for_voice(req, now, &voice.id), mode: RedubMode::Dub, source: src.name, voice: voice.name };
}

fn start_render(s: &mut AppState) {
    if s.redub.busy() {
        return;
    }
    if let Some((why, _)) = render_blocker(s) {
        shell::toast(s, Tone::Warning, why);
        return;
    }
    let (Some(src), Some(voice)) = (s.redub.file.clone(), shell::active_voice(s).cloned()) else { return };
    let mode = s.redub.mode;
    // A reviewed dub is finished as a re-speak of the edited translation; the history still calls it a dub.
    let wire = if mode == RedubMode::Dub { RedubMode::Respeak } else { mode };
    let segs: Vec<RedubSegmentIn> =
        s.redub.kept_segments().map(|g| RedubSegmentIn { start: g.start, end: g.end, text: g.edit.text.trim().to_owned() }).collect();
    let out = formats::output_format(s);
    // A reviewed dub and a Re-speak in English offer the accent choice (Convert keeps the original speech).
    let accent = match mode {
        RedubMode::Dub => speak::accent_param(s, s.redub.transcript_lang.as_deref().unwrap_or(&s.redub.to)),
        RedubMode::Respeak => respeak_lang(&s.redub).and_then(|l| speak::accent_param(s, &l)),
        RedubMode::Convert => None,
    };
    let (method, params) = protocol::redub_render(
        &voice.id,
        &src.path,
        wire,
        (wire == RedubMode::Respeak).then_some(segs.as_slice()),
        &s.redub.text_language(),
        s.speak.voice_match,
        s.speak.expressiveness,
        s.speak.quality,
        s.speak.speed,
        &out,
        accent,
        timing(&s.redub),
    );
    let Some(req) = shell::request(s, method, params) else {
        shell::toast(s, Tone::Danger, "Could not reach the engine.");
        return;
    };
    stop_play(s);
    s.redub.result = None;
    s.redub.error = None;
    s.redub.job = Job::Rendering { run: Run::for_voice(req, shell::clock_now(s), &voice.id), mode, source: src.name, voice: voice.name };
}

fn cancel(s: &mut AppState) {
    let run = match &mut s.redub.job {
        Job::Idle => return,
        Job::Transcribing(r) | Job::Rendering { run: r, .. } => r,
    };
    match run.job.clone() {
        Some(job) => {
            shell::request(s, "redub.cancel", json!({ "job": job }));
            s.redub.job = Job::Idle;
        }
        None => run.cancel = true,
    }
}

fn remove_segment(s: &mut AppState, key: u64) {
    s.redub.segments.retain(|g| g.key != key);
}

/// A play button (the original, a result): play it, or pause / resume it in the player bar.
fn toggle_play(s: &mut AppState, path: &Path, title: String, what: &str) {
    if let Err(e) = playbar::toggle(s, path, title, what, playbar::Source::Redub) {
        shell::toast(s, Tone::Danger, e);
    }
}

/// Stops playback started on this screen.
fn stop_play(s: &mut AppState) {
    playbar::stop_from(s, playbar::Source::Redub);
}

/// The engine deleted a result's files (`ui::trash`): drop it from the history (and the result card).
pub fn remove_result(s: &mut AppState, id: u64) {
    s.redub.history.retain(|r| r.id != id);
    if s.redub.result == Some(id) {
        s.redub.result = None;
    }
}

fn friendly_error(code: &str, message: &str) -> String {
    match code {
        "not_ready" => "The engine is still loading its models. Try again in a moment.".into(),
        "busy" => "The engine is busy with another job. Try again when it finishes.".into(),
        "not_found" => "That file or voice could not be found.".into(),
        "not_trained" => "This voice is not trained yet. Train it, or use Translate & dub, which works with any voice.".into(),
        "no_audio" => "That file has no audio track.".into(),
        _ if message.is_empty() => "Something went wrong.".into(),
        _ => message.to_owned(),
    }
}

fn fail(s: &mut AppState, text: String) {
    s.redub.job = Job::Idle;
    s.redub.error = Some(text.clone());
    shell::toast(s, Tone::Danger, text);
}

// -------------------------------------------------------------------------------------------------
// Engine messages and events
// -------------------------------------------------------------------------------------------------

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    if matches!(msg, EngineMsg::Exited { .. }) {
        if let Some(f) = s.redub.file.as_mut() {
            f.prepared = false;
        }
    }
    on_probe(s, msg);
    let run = match &s.redub.job {
        Job::Idle => return,
        Job::Transcribing(r) | Job::Rendering { run: r, .. } => r.clone(),
    };
    match msg {
        EngineMsg::Response { id, result } if *id == run.req => match result {
            Ok(v) => {
                let mut run = run;
                if let Ok(JobStarted { job }) = protocol::parse_result(v.clone()) {
                    run.job = Some(job);
                }
                let stop = run.cancel;
                let note = |k: &str| v.get(k).and_then(|n| n.as_str()).map(str::to_owned);
                run.warned = speak::job_notes(s, note("accent_note").as_deref(), note("reference_note").as_deref());
                set_run(s, run);
                if stop {
                    cancel(s);
                }
            }
            Err(e) => {
                let text = friendly_error(&e.code, &e.message);
                fail(s, text);
            }
        },
        EngineMsg::Event(ev) if run.job.is_some() && ev.job == run.job => match ev.parse() {
            EngineEvent::RedubProgress { progress, .. } => {
                let mut run = run;
                if !progress.stage.is_empty() {
                    run.stage = progress.stage;
                }
                run.pct = progress.pct.clamp(0.0, 100.0);
                if let (Some(k), Some(n)) = (progress.segment, progress.segments) {
                    run.segment = Some((k, n));
                }
                set_run(s, run);
            }
            EngineEvent::RedubTranscript { transcript, .. } => on_transcript(s, transcript),
            EngineEvent::RedubReview { review, .. } => on_review(s, review),
            EngineEvent::RedubDone { done, .. } => on_done(s, done),
            EngineEvent::RedubError { error, .. } => {
                if error.code == "cancelled" {
                    s.redub.job = Job::Idle;
                } else {
                    let text = friendly_error(&error.code, &error.message);
                    fail(s, text);
                }
            }
            _ => {}
        },
        EngineMsg::Exited { .. } => fail(s, "The engine stopped while working. Press Retry above and try again.".into()),
        _ => {}
    }
}

fn set_run(s: &mut AppState, run: Run) {
    match &mut s.redub.job {
        Job::Transcribing(r) | Job::Rendering { run: r, .. } => *r = run,
        Job::Idle => {}
    }
}

fn on_probe(s: &mut AppState, msg: &EngineMsg) {
    let Some(Source { probe, .. }) = s.redub.file.as_mut() else { return };
    let Probe::Asking(req) = probe else { return };
    match msg {
        EngineMsg::Response { id, result } if id == req => {
            *probe = match result {
                Ok(v) => match protocol::parse_result::<RedubProbe>(v.clone()) {
                    Ok(p) if !p.has_audio => Probe::Failed("This file has no audio track.".into()),
                    Ok(p) => Probe::Ready(p),
                    Err(e) => Probe::Failed(e.message),
                },
                Err(e) => Probe::Failed(friendly_error(&e.code, &e.message)),
            };
        }
        // The engine went away before it answered: ask again once it is back.
        EngineMsg::Exited { .. } => *probe = Probe::Pending,
        _ => {}
    }
}

fn on_transcript(s: &mut AppState, t: RedubTranscript) {
    // A dub with "Review text before dubbing" ends with its transcript (the engine sends no separate review event).
    let review = t.review.unwrap_or(false) || matches!(s.redub.job, Job::Rendering { mode: RedubMode::Dub, .. });
    let r = &mut s.redub;
    let target = t.target_language.clone().filter(|l| !l.is_empty()).unwrap_or_else(|| r.to.clone());
    let same = |l: &str| l.trim().eq_ignore_ascii_case(&target);
    // The text is in the target language when the engine says it was translated; an engine that does not say is taken
    // at the request.
    let translated = t.segments.iter().any(|g| g.translated) || t.translated.unwrap_or(r.translate && !same(&t.language));
    r.segments = t
        .segments
        .into_iter()
        .map(|g| {
            r.next_seg += 1;
            Seg { key: r.next_seg, start: g.start, end: g.end.max(g.start), source: g.source_text.unwrap_or_default().trim().to_owned(), edit: TextEdit::new(g.text.trim()) }
        })
        .collect();
    r.reviewing = review && !r.segments.is_empty();
    r.transcript_translated = translated;
    if translated {
        r.transcript_lang = Some(target.clone());
        let from = t.source_language.filter(|l| !l.is_empty()).unwrap_or(t.language);
        r.transcript_from = Some(from).filter(|l| !l.is_empty() && !same(l));
    } else {
        r.transcript_lang = Some(t.language).filter(|l| !l.is_empty());
        r.transcript_from = None;
    }
    r.job = Job::Idle;
    if r.segments.is_empty() {
        let msg = "No speech was found in this file.".to_owned();
        fail(s, msg);
    }
}

/// A dub with "Review text before dubbing" stopped after the translation: show the two columns.
fn on_review(s: &mut AppState, rv: RedubReview) {
    if !matches!(s.redub.job, Job::Rendering { mode: RedubMode::Dub, .. }) {
        return;
    }
    s.redub.job = Job::Idle;
    let r = &mut s.redub;
    r.segments = rv
        .segments
        .into_iter()
        .map(|g| {
            r.next_seg += 1;
            Seg { key: r.next_seg, start: g.start, end: g.end.max(g.start), source: g.source_text.trim().to_owned(), edit: TextEdit::new(g.text.trim()) }
        })
        .collect();
    let lang = Some(rv.language).filter(|l| !l.trim().is_empty()).unwrap_or_else(|| r.to.clone());
    r.transcript_from = Some(rv.source_language).filter(|l| !l.trim().is_empty() && !l.eq_ignore_ascii_case("auto") && !l.eq_ignore_ascii_case(&lang));
    r.transcript_lang = Some(lang);
    r.transcript_translated = true;
    r.reviewing = true;
    if r.segments.is_empty() {
        r.reviewing = false;
        fail(s, "No speech was found in this file.".to_owned());
    }
}

fn on_done(s: &mut AppState, done: RedubDone) {
    let Job::Rendering { mode, source, voice, run } = std::mem::take(&mut s.redub.job) else { return };
    let to = (mode == RedubMode::Dub).then(|| {
        done.target_language.clone().filter(|l| !l.is_empty()).or_else(|| s.redub.transcript_lang.clone()).unwrap_or_else(|| s.redub.to.clone())
    });
    // The language spoken and the clip that spoke it (a clip transcribed for the first time is known only now).
    if let Some(spoken) = to.clone().or_else(|| done.language.clone()).filter(|l| !l.is_empty()) {
        speak::note_reference(s, &run.voice_id, &spoken, &done.reference);
    }
    if !run.warned {
        if let Some(n) = done.reference.reference_note.clone().filter(|n| !n.is_empty()) {
            shell::toast(s, Tone::Warning, n);
        }
    }
    if mode == RedubMode::Dub {
        // The dub (or the reviewed one) is finished: the review text is no longer needed.
        s.redub.clear_text();
    }
    let (path, play) = (done.path.clone(), done.play_path());
    let preview = s.redub.file.as_ref().filter(|f| f.name == source).and_then(Source::preview);
    s.redub.next_id += 1;
    let id = s.redub.next_id;
    let elapsed = if done.elapsed_s > 0.0 { done.elapsed_s } else { (s.shell.now - run.started) as f32 };
    s.redub.history.insert(0, Redub { id, name: source, voice, mode, path, play, preview, duration_s: done.duration_s, elapsed_s: elapsed, drift_s: done.drift_s.filter(|d| d.abs() >= 0.5), to });
    s.redub.history.truncate(50);
    s.redub.result = Some(id);
}

/// After every event: saved choices, and the probe / model preload a chosen file waits for until the engine is ready.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;

    // Last time's choices, then every change of them.
    if !s.shell.scripted() {
        if !s.redub.prefs_loaded {
            s.redub.prefs_loaded = true;
            let (mode, from, review, to) = super::onboarding::saved_redub_prefs();
            if let Some(m) = mode.as_deref().and_then(mode_from_key) {
                s.redub.mode = m;
            }
            if let Some(l) = from {
                s.redub.from = l.to_ascii_lowercase();
            }
            if let Some(r) = review {
                s.redub.review_first = r;
            }
            if let Some(l) = to.filter(|l| lang::is_spoken(s, l)) {
                s.redub.to = l.to_ascii_lowercase();
            }
            if let Some(t) = super::onboarding::setting("redub_translator") {
                s.redub.faster_translation = t.as_str() == Some(TRANSLATOR_FASTER);
            }
            s.redub.keep_timing = parse_timing_pref(super::onboarding::setting("redub_timing"));
            changed = true;
        } else if s.redub.prefs_dirty {
            s.redub.prefs_dirty = false;
            super::onboarding::save_redub_prefs(mode_key(s.redub.mode), &s.redub.from, s.redub.review_first, &s.redub.to);
            let t = if s.redub.faster_translation { TRANSLATOR_FASTER } else { TRANSLATOR_BETTER };
            super::onboarding::save_setting("redub_translator", json!(t));
            if s.redub.keep_timing.iter().any(Option::is_some) {
                super::onboarding::save_setting("redub_timing", timing_pref(s.redub.keep_timing));
            }
        }
    }

    // Probe a newly chosen file once the engine can answer.
    if shell::engine_ready(s) && !s.shell.scripted() && matches!(s.redub.file.as_ref().map(|f| &f.probe), Some(Probe::Pending)) {
        let path = s.redub.file.as_ref().map(|f| f.path.clone()).unwrap_or_default();
        let (method, params) = protocol::redub_probe(&path);
        if let Some(req) = shell::request(s, method, params) {
            if let Some(f) = s.redub.file.as_mut() {
                f.probe = Probe::Asking(req);
            }
            changed = true;
        }
    }

    // Preload the speech models for a newly chosen file, so the first click does not wait for them. Failures (busy, not
    // ready) do not matter: the job loads what it needs anyway.
    let want_prepare = s.redub.file.as_ref().is_some_and(|f| !f.prepared && !matches!(f.probe, Probe::Failed(_)));
    if want_prepare && shell::engine_ready(s) && !s.shell.scripted() && !s.redub.busy() && !super::train::is_training(s) && !s.speak.is_generating() {
        let path = s.redub.file.as_ref().map(|f| f.path.clone()).unwrap_or_default();
        let (method, params) = protocol::redub_prepare(&path);
        if shell::request(s, method, params).is_some() {
            if let Some(f) = s.redub.file.as_mut() {
                f.prepared = true;
            }
        }
    }

    changed
}

/// The start of the running job (its elapsed time ticks on the Redub screen).
pub fn clock_origin(s: &AppState) -> Option<f64> {
    match &s.redub.job {
        Job::Idle => None,
        Job::Transcribing(run) | Job::Rendering { run, .. } => Some(run.started),
    }
}

// -------------------------------------------------------------------------------------------------
// View
// -------------------------------------------------------------------------------------------------

fn fmt_time(secs: f32) -> String {
    let t = secs.max(0.0).round() as u32;
    if t >= 3600 { format!("{}:{:02}:{:02}", t / 3600, (t / 60) % 60, t % 60) } else { format!("{}:{:02}", t / 60, t % 60) }
}

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    row().key("redub").w_full().h_full().px(16.0).pb(16.0).gap(16.0)
        .child(main_panel(s, p))
        .child(speak::side_column(s, p, cx, |own_scroll| results_panel(s, p, own_scroll)))
}

fn main_panel(s: &AppState, p: Palette) -> El {
    let mut body = col().key("redub-body").w_full().grow(1.0).min_h(0.0).gap(16.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0)
        .child(source_card(s, p))
        .child(mode_block(s, p));
    if let Some(e) = &s.redub.error {
        body = body.child(kit::notice(&p, Tone::Danger, e));
    }
    body = match s.redub.mode {
        RedubMode::Dub => body.child(dub_block(s, p)),
        RedubMode::Convert => body.child(convert_block(s, p)),
        RedubMode::Respeak => body.child(respeak_block(s, p)),
    };
    let mut bottom = col().key("redub-bottom").w_full().gap(12.0);
    if let Some(r) = s.redub.result.and_then(|id| s.redub.history.iter().find(|h| h.id == id)) {
        if !s.redub.busy() {
            bottom = bottom.child(result_card(s, p, r));
        }
    }
    bottom = bottom.child(action_bar(s, p));
    section_panel(p, "Redub").key("redub-panel").grow(1.0).min_w(0.0).h_full().gap(14.0).child(body).child(bottom)
}

// ---- source file ----

fn source_card(s: &AppState, p: Palette) -> El {
    let over = s.formats.over;
    match &s.redub.file {
        None => drop_zone(s, p, over),
        Some(f) => file_card(s, p, f, over),
    }
}

fn badge_circle(p: Palette, glyph: &'static rune::Icon, ink: rune::Color) -> El {
    let b = el().size(44.0, 44.0).radius(22.0).center().fill(p.glass).gradient(p.glass_top, p.glass_bottom).border_gradient(1.0, p.rim_top, p.rim_bottom)
        .child(icon_el(glyph, 19.0, ink));
    super::components::glass_edge(p, b)
}

fn drop_zone(s: &AppState, p: Palette, over: bool) -> El {
    let (title, sub) = if over {
        ("Release to open this file".to_owned(), "It stays where it is: nothing is copied or changed.".to_owned())
    } else {
        ("Drop an audio or video file here".to_owned(), formats::import_hint(s))
    };
    col().key("drop-zone").role("button").label("Choose a file").w_full().h(172.0).center().gap(9.0).radius(radius::CARD).continuous_corners()
        .fill(if over { p.accent.alpha(0.13) } else { p.well })
        .border(if over { 2.0 } else { 1.0 }, if over { p.accent } else { p.well_rim.alpha(p.well_rim.a * 1.8) })
        .motion(motion::UI)
        .on_press(choose_file)
        .child(badge_circle(p, &icons::FILE, if over { p.accent } else { p.text_dim }))
        .child(text(title).font(ty::TITLE).color(p.text))
        .child(text(sub).font(ty::CAPTION).color(p.text_faint))
        .child(Button::new("Choose file…").small().key("choose-file").on_press(choose_file).build(p))
}

fn file_card(s: &AppState, p: Palette, f: &Source, over: bool) -> El {
    let (meta, tone) = match &f.probe {
        Probe::Pending | Probe::Asking(_) => ("Reading the file…".to_owned(), Tone::Neutral),
        Probe::Failed(m) => (m.clone(), Tone::Danger),
        Probe::Ready(pr) => {
            let khz = pr.sample_rate as f32 / 1000.0;
            let ch = match pr.channels {
                1 => "mono".to_owned(),
                2 => "stereo".to_owned(),
                0 => String::new(),
                n => format!("{n} channels"),
            };
            let mut parts = vec![fmt_time(pr.duration_s)];
            if !pr.format.is_empty() {
                parts.push(pr.format.to_uppercase());
            }
            if pr.sample_rate > 0 {
                parts.push(format!("{khz:.1} kHz").replace(".0 kHz", " kHz"));
            }
            if !ch.is_empty() {
                parts.push(ch);
            }
            (parts.join(" · "), Tone::Neutral)
        }
    };
    let busy = s.redub.busy();
    let preview = f.preview();
    let play: El = match preview {
        Some(path) => {
            let on = playbar::playing(s, &path);
            let name = f.name.clone();
            let b = el().key("play-original").role("button").label(if on { "Pause original" } else { "Play original" }).help(if on { "Pause" } else { "Play the original" })
                .size(40.0, 40.0).radius(20.0).center().hover(rune::Color::WHITE.alpha(0.1)).focusable(true).motion(motion::UI)
                .on_press(move |s: &mut AppState| toggle_play(s, &path, name.clone(), "Redub · original"))
                .child(icon_el(if on { &icons::PAUSE } else { &icons::PLAY }, 16.0, p.on_accent));
            accent_glass(p, b)
        }
        None => icon_button(p, "play-original", &icons::PLAY, "Preview needs a WAV file, but the original is still used for the redub", |_| {}).opacity(0.4),
    };
    let meta_color = if tone == Tone::Danger { p.bad } else { p.text_faint };
    let change = Button::new("Change…").ghost().small().key("change-file").disabled(busy).on_press(choose_file).build(p);
    let remove = icon_button(p, "remove-file", &icons::CLOSE, "Remove the file", clear_file).opacity(if busy { 0.4 } else { 1.0 });
    glass_card(p).key("file-card").w_full().padding(14.0)
        .border_gradient(if over { 2.0 } else { 1.0 }, if over { p.accent } else { p.rim_top.alpha(p.rim_top.a * 0.7) }, if over { p.accent.alpha(0.5) } else { p.rim_bottom })
        .child(row().w_full().gap(14.0).center_y()
            .child(badge_circle(p, &icons::FILE, p.accent))
            .child(col().grow(1.0).min_w(0.0).gap(2.0)
                .child(text(f.name.clone()).font(ty::TITLE).color(p.text).truncate())
                .child(row().gap(8.0).center_y()
                    .children(matches!(f.probe, Probe::Pending | Probe::Asking(_)).then(|| status_dot(p, Tone::Accent, true)))
                    .child(text(if over { "Drop to replace this file".to_owned() } else { meta }).font(ty::CAPTION).color(meta_color))))
            .child(play)
            .child(change)
            .child(remove))
}

// ---- mode ----

fn mode_card(p: Palette, key: &str, title: &str, sub: &str, on: bool, mode: RedubMode) -> El {
    col().key(key.to_owned()).role("button").label(title.to_owned()).checked(on).focusable(true).grow(1.0).min_w(0.0).gap(3.0).padding(12.0)
        .radius(radius::CARD).continuous_corners()
        .fill(if on { p.accent.alpha(0.16) } else { p.well })
        .border(if on { 1.5 } else { 1.0 }, if on { p.accent.alpha(0.7) } else { p.well_rim.alpha(p.well_rim.a * 1.5) })
        .hover(if on { p.accent.alpha(0.22) } else { p.glass_hover })
        .motion(motion::UI)
        .on_press(move |s: &mut AppState| set_mode(s, mode))
        .child(text(title.to_owned()).font(ty::TITLE).color(if on { p.text } else { p.text_dim }).a11y_hidden())
        .child(text(sub.to_owned()).font(ty::CAPTION).color(p.text_faint).wrap().a11y_hidden())
}

fn mode_block(s: &AppState, p: Palette) -> El {
    let m = s.redub.mode;
    col().key("mode-block").w_full().gap(8.0)
        .child(text("Mode").font(ty::TITLE).color(p.text))
        .child(row().w_full().gap(10.0)
            .child(mode_card(p, "mode-dub", "Translate & dub", "Any language to another, in your voice", m == RedubMode::Dub, RedubMode::Dub))
            .child(mode_card(p, "mode-convert", "Convert voice", "Same recording, your voice", m == RedubMode::Convert, RedubMode::Convert))
            .child(mode_card(p, "mode-respeak", "Re-speak", "Edit the words, then speak", m == RedubMode::Respeak, RedubMode::Respeak)))
}

fn voice_chip(p: Palette, v: &VoiceInfo) -> El {
    row().gap(8.0).center_y()
        .child(icon_el(&icons::USER, 14.0, p.accent))
        .child(text(v.name.clone()).font(ty::LABEL).color(p.text))
        .child(badge(p, if v.trained { "Trained" } else { "Zero-shot" }, if v.trained { Tone::Success } else { Tone::Accent }))
}

fn convert_block(s: &AppState, p: Palette) -> El {
    let mut c = col().key("convert-block").w_full().gap(10.0)
        .child(text("Keeps the original timing and delivery, and swaps in your trained voice.").font(ty::CAPTION).color(p.text_dim).wrap());
    match shell::active_voice(s) {
        None => {
            c = c.child(kit::notice(&p, Tone::Warning, "No voice selected. Record your voice first, or pick one at the top right."))
                .child(row().child(Button::new("Record").small().key("convert-record").on_press(|s| shell::set_screen(s, Screen::Record)).build(p)));
        }
        Some(v) if !v.trained => {
            c = c.child(kit::notice(&p, Tone::Warning, &format!("“{}” is not trained yet. Convert voice needs a trained voice: train it first, or use Translate & dub, which works with any voice.", v.name)))
                .child(row().gap(8.0)
                    .child(Button::new("Go to Train").primary().small().icon(&icons::CHIP).key("convert-train").on_press(|s| shell::set_screen(s, Screen::Train)).build(p))
                    .child(Button::new("Use Translate & dub").small().key("convert-dub").on_press(|s| set_mode(s, RedubMode::Dub)).build(p)));
        }
        Some(v) => {
            c = c.child(glass_card(p).w_full().padding(12.0).child(row().w_full().gap(12.0).center_y()
                .child(voice_chip(p, v))
                .child(el().grow(1.0))
                .child(text(format!("{} epochs", v.trained_epochs)).font(ty::CAPTION).color(p.text_faint))));
        }
    }
    c
}

// ---- translate & dub ----

/// The label of the "From" chip: "Auto-detect" or the language.
fn from_label(code: &str) -> String {
    if code.eq_ignore_ascii_case("auto") { "Auto-detect".to_owned() } else { lang::label(code) }
}

/// A caption under "To" when the voice was recorded in another language (its RVC index is capped there).
fn foreign_note(s: &AppState, to: &str) -> Option<String> {
    let v = shell::active_voice(s)?;
    if v.lang() == lang::base(to) {
        return None;
    }
    let (rec, target) = (lang::name(v.lang()), lang::name(to));
    if speak::accent_param(s, to).is_some() {
        return None; // the Accent choice below says what happens
    }
    let mut t = format!("Voice recorded in {rec}: timbre kept");
    if v.trained {
        t.push_str(", Voice match capped at 0.3 so the words stay clear");
    }
    if !v.references.iter().any(|r| *r == lang::base(to)) {
        t.push_str(&format!(". Adding a {target} reference clip in Voices improves the accent."));
    } else {
        t.push_str(&format!(", your {target} reference clip sets the accent."));
    }
    Some(t)
}

fn dub_block(s: &AppState, p: Palette) -> El {
    let r = &s.redub;
    let busy = r.busy();
    let locked = busy || r.reviewing;
    let from = lang::chip(&p, "dub-from", &from_label(&r.from), lang::Target::RedubFrom, !locked);
    let to = lang::chip(&p, "dub-to", &lang::label(&r.to), lang::Target::RedubTo, !locked);
    let review = kit::checkbox(&p, r.review_first, "Review text before dubbing", |s| {
        if !s.redub.busy() && !s.redub.reviewing {
            s.redub.review_first = !s.redub.review_first;
            s.redub.persist();
        }
    });
    let to_name = lang_name(&r.to);
    let pair = row().w_full().gap(12.0).center_y().flex_wrap()
        .child(row().gap(8.0).center_y().child(text("From").font(ty::LABEL).color(p.text)).child(from))
        .child(icon_el(&ARROW, 14.0, p.text_faint))
        .child(row().gap(8.0).center_y().child(text("To").font(ty::LABEL).color(p.text)).child(to));
    let mut options = glass_card(p).key("dub-options").w_full().padding(14.0).gap(12.0).opacity(if locked { 0.6 } else { 1.0 }).child(pair);
    if let Some(note) = foreign_note(s, &r.to) {
        options = options.child(text(note).font(ty::CAPTION).color(p.text_faint).wrap());
    }
    if speak::accent_applies(s, &r.to) {
        options = options.child(el().w_full().max_w(420.0).child(speak::accent_block(s, p, "dub-accent")));
    }
    if translator_choice(s) {
        options = options.child(translation_quality(r, p));
    }
    options = options.child(timing_block(s, p, "dub-timing"));
    options = options.child(col().w_full().gap(3.0).child(review)
        .child(text(format!("Stops after the translation so you can fix the {to_name} before it is spoken.")).font(ty::CAPTION).color(p.text_faint).wrap().pl(32.0)));
    let mut c = col().key("dub-block").w_full().gap(12.0);
    if !r.reviewing {
        // While the review is open the table gets the room; the choices are locked in until it is done or discarded.
        let from_name = if r.from.eq_ignore_ascii_case("auto") { "Any language".to_owned() } else { lang_name(&r.from) };
        c = c.child(text(format!("{from_name} → {to_name}, in your voice.")).font(ty::CAPTION).color(p.text_dim).wrap()).child(options);
    }
    if shell::active_voice(s).is_none() {
        c = c.child(kit::notice(&p, Tone::Warning, "No voice selected. Record your voice first, or pick one at the top right."))
            .child(row().child(Button::new("Record").small().key("dub-record").on_press(|s| shell::set_screen(s, Screen::Record)).build(p)));
    }
    if r.reviewing {
        c = c.child(review_table(s, p));
    }
    c
}

/// "Translation quality: Better (1.3B) / Faster (600M)" (shown when both translators are installed).
fn translation_quality(r: &RedubState, p: Palette) -> El {
    let caption = if r.faster_translation { "NLLB-200 600M · the smaller model, about 1.5× faster" } else { "NLLB-200 1.3B · the larger, generally more accurate model" };
    row().key("translation-quality").w_full().center_y().justify_between().gap(10.0)
        .child(col().grow(1.0).min_w(0.0).gap(1.0)
            .child(text("Translation quality").font(ty::LABEL).color(p.text))
            .child(text(caption).font(ty::CAPTION).color(p.text_faint).wrap()))
        .child(segmented_help(
            p,
            "translator-pick",
            &["Better (1.3B)", "Faster (600M)"],
            &["The larger translation model, generally more accurate (about 3 GB of GPU memory)", "The smaller translation model: quicker, about 1.5 GB of GPU memory"],
            usize::from(r.faster_translation),
            |s, i| {
                if !s.redub.busy() && !s.redub.reviewing {
                    s.redub.faster_translation = i == 1;
                    s.redub.persist();
                }
            },
        ))
}

fn review_table(s: &AppState, p: Palette) -> El {
    let r = &s.redub;
    let total: f32 = r.kept_segments().map(|g| (g.end - g.start).max(0.0)).sum();
    let n = r.segments.len();
    let mut head = row().w_full().gap(10.0).center_y().child(text("Review").font(ty::TITLE).color(p.text));
    if let Some(l) = r.lang_label() {
        head = head.child(badge(p, &l, Tone::Neutral));
    }
    head = head.child(el().grow(1.0)).child(text(format!("{n} line{} · {} of speech", if n == 1 { "" } else { "s" }, fmt_time(total))).font(ty::CAPTION).color(p.text_faint));
    let cols = row().w_full().gap(10.0)
        .child(el().w(84.0))
        .child(text("Original").font(ty::OVERLINE).color(p.text_faint).grow(1.0))
        .child(text(format!("{} (edit if needed)", lang_name(r.transcript_lang.as_deref().unwrap_or(&r.to)))).font(ty::OVERLINE).color(p.text_faint).grow(1.0))
        .child(el().w(32.0));
    let rows = r.segments.iter().enumerate().map(|(i, g)| segment_row(p, i, g, r.busy(), true));
    col().key("review").w_full().gap(10.0)
        .child(head)
        .child(text("Fix any line, then press Dub it. Empty lines are skipped.").font(ty::CAPTION).color(p.text_dim).wrap())
        .child(cols)
        .child(col().w_full().gap(8.0).children(rows))
        .child(row().child(Button::new("Discard review").ghost().small().key("review-discard").on_press(|s| {
            if !s.redub.busy() {
                s.redub.clear_text();
            }
        }).build(p)))
}

// ---- re-speak ----

fn respeak_block(s: &AppState, p: Palette) -> El {
    let r = &s.redub;
    let busy = r.busy();
    let from = lang::chip(&p, "respeak-from", &from_label(&r.from), lang::Target::RedubFrom, !busy);
    let to_name = lang_name(&r.to);
    let translate = kit::checkbox(&p, r.translate, "Translate the transcript", |s| {
        if !s.redub.busy() {
            s.redub.translate = !s.redub.translate;
        }
    });
    let mut tr = row().w_full().gap(12.0).center_y().child(el().grow(1.0).min_w(0.0).child(translate));
    if r.translate {
        tr = tr.child(row().gap(8.0).center_y().child(text("To").font(ty::LABEL).color(p.text)).child(lang::chip(&p, "respeak-to", &lang::label(&r.to), lang::Target::RedubTo, !busy)));
    }
    let tr_note = if r.translate {
        format!("The transcript is written in {to_name}, so your cloned voice speaks {to_name}.")
    } else {
        "Your voice speaks the transcript in its own language.".to_owned()
    };
    let mut options = glass_card(p).key("respeak-options").w_full().padding(14.0).gap(12.0)
        .opacity(if busy { 0.6 } else { 1.0 })
        .child(row().w_full().gap(10.0).center_y().child(text("Spoken language").font(ty::LABEL).color(p.text)).child(from))
        .child(col().w_full().gap(3.0).child(tr)
            .child(text(tr_note).font(ty::CAPTION).color(p.text_faint).wrap().pl(32.0)));
    // English without an English reference clip: the same Accent choice as Speak and Translate & dub.
    if respeak_lang(r).is_some_and(|l| speak::accent_applies(s, &l)) {
        options = options.child(el().w_full().max_w(420.0).child(speak::accent_block(s, p, "respeak-accent")));
    }
    let options = options.child(timing_block(s, p, "respeak-timing")).child(transcribe_row(s, p));
    let mut c = col().key("respeak-block").w_full().gap(12.0)
        .child(text("Transcribes the speech so you can fix or translate the words, then says them again in your voice's own delivery.").font(ty::CAPTION).color(p.text_dim).wrap())
        .child(options);
    if let Some(l) = unspeakable_lang(s) {
        c = c.child(kit::notice(&p, Tone::Warning, &format!("This text is {l}, which your voice cannot speak yet. Use Translate & dub, or tick “Translate the transcript” and transcribe again.")))
            .child(row().child(Button::new("Use Translate & dub").small().key("respeak-use-dub").on_press(|s| set_mode(s, RedubMode::Dub)).build(p)));
    }
    if !r.segments.is_empty() {
        c = c.child(transcript_list(s, p));
    }
    c
}

fn transcribe_row(s: &AppState, p: Palette) -> El {
    if let Job::Transcribing(run) = &s.redub.job {
        let (label, frac) = match run.stage.as_str() {
            "decode" => ("Reading the audio…".to_owned(), None),
            "load_model" => ("Loading the speech model… (the first time takes longer)".to_owned(), None),
            "transcribe" => (format!("Transcribing… {:.0}%", run.pct), Some((run.pct / 100.0).max(0.03))),
            _ => ("Starting…".to_owned(), None),
        };
        let elapsed = (s.shell.now - run.started).max(0.0) as f32;
        return row().key("transcribing").w_full().gap(16.0).center_y()
            .child(col().grow(1.0).min_w(0.0).gap(7.0)
                .child(row().w_full().center_y().justify_between()
                    .child(text(label).font(ty::LABEL).color(p.text))
                    .child(text(fmt_time(elapsed)).font(ty::NUM).color(p.text_dim)))
                .child(progress_bar(p, frac)))
            .child(Button::new("Cancel").small().icon(&icons::STOP).key("cancel-transcribe").on_press(cancel).build(p));
    }
    let block = transcribe_blocker(s);
    let again = !s.redub.segments.is_empty();
    let go = Button::new(if again { "Transcribe again" } else { "Transcribe" }).small().icon(&icons::WAVE).key("transcribe").disabled(block.is_some() || s.redub.busy());
    let go = if again { go } else { go.primary() }.on_press(start_transcribe).build(p);
    let hint = match &block {
        Some(b) => text(b.clone()).font(ty::CAPTION).color(if b == "Choose a file first." || b == "Reading the file…" { p.text_faint } else { p.warn }),
        None => text(if again { "Transcribing again replaces your edits." } else { "Turns the speech into editable text." }).font(ty::CAPTION).color(p.text_faint),
    };
    row().key("transcribe-row").w_full().gap(14.0).center_y().child(go).child(el().grow(1.0).min_w(0.0).child(hint.wrap()))
}

fn transcript_list(s: &AppState, p: Palette) -> El {
    let r = &s.redub;
    let total: f32 = r.kept_segments().map(|g| (g.end - g.start).max(0.0)).sum();
    let n = r.segments.len();
    let mut head = row().w_full().gap(10.0).center_y().child(text("Transcript").font(ty::TITLE).color(p.text));
    if let Some(l) = r.lang_label() {
        head = head.child(badge(p, &l, Tone::Neutral));
    }
    head = head.child(el().grow(1.0)).child(text(format!("{n} segment{} · {} of speech", if n == 1 { "" } else { "s" }, fmt_time(total))).font(ty::CAPTION).color(p.text_faint));
    let busy = r.busy();
    let rows = r.segments.iter().enumerate().map(|(i, g)| segment_row(p, i, g, busy, false));
    col().key("transcript").w_full().gap(10.0).child(head).child(col().w_full().gap(8.0).children(rows))
}

fn segment_row(p: Palette, i: usize, g: &Seg, busy: bool, with_source: bool) -> El {
    let key = g.key;
    let focused = g.edit.focused;
    let field = text_input(&g.edit, ty::BODY)
        .key(format!("seg-text-{key}"))
        .w_full()
        .color(p.text)
        .placeholder("Empty segments are skipped", p.text_faint)
        .caret(p.accent, p.accent.alpha(0.3))
        .on_edit(edit_access(move |s: &mut AppState| {
            let last = s.redub.segments.len().saturating_sub(1);
            &mut s.redub.segments[i.min(last)].edit
        }));
    let well_ = well(p).key(format!("seg-well-{key}")).grow(1.0).min_w(0.0).px(12.0).py(8.0)
        .border(1.0, if focused { p.accent.alpha(0.8) } else { p.well_rim })
        .child(field);
    let stamp = col().w(84.0).pt(8.0).gap(1.0)
        .child(text(fmt_time(g.start)).font(ty::NUM).color(p.text_dim))
        .child(text(format!("to {}", fmt_time(g.end))).font(ty::CAPTION).color(p.text_faint));
    let del = icon_button(p, &format!("seg-del-{key}"), &icons::TRASH, "Remove this segment", move |s| remove_segment(s, key)).opacity(if busy { 0.4 } else { 1.0 });
    let mut line = row().key(format!("seg-{key}")).w_full().gap(10.0).items_start().child(stamp);
    if with_source {
        let src = if g.source.is_empty() { "(no original text)".to_owned() } else { g.source.clone() };
        line = line.child(
            col().key(format!("seg-src-{key}")).grow(1.0).min_w(0.0).px(12.0).py(8.0).radius(radius::CONTROL + 2.0).continuous_corners().fill(p.glass)
                .border(1.0, p.rim_top.alpha(p.rim_top.a * 0.5))
                .child(text(src).font(ty::BODY).color(p.text_dim).wrap()),
        );
    }
    line.child(well_).child(del.mt(4.0))
}

// ---- result and actions ----

fn open_reveal(p: Palette, id: u64, tag: &str, path: &Path) -> (El, El) {
    let (a, b) = (path.to_path_buf(), path.to_path_buf());
    let reveal = icon_button(p, &format!("reveal-{tag}{id}"), &icons::FOLDER, "Show in folder", move |s| {
        if let Err(e) = speak::reveal_file(&a) {
            shell::toast(s, Tone::Danger, e);
        }
    });
    let open = icon_button(p, &format!("open-{tag}{id}"), &icons::OPEN, "Open file", move |s| {
        if let Err(e) = speak::open_file(&b) {
            shell::toast(s, Tone::Danger, e);
        }
    });
    (reveal, open)
}

/// "Original" and "Result" play buttons: press one to hear it, press it again to pause (A/B).
fn ab_buttons(s: &AppState, p: Palette, r: &Redub, tag: &str) -> El {
    let id = r.id;
    let mk = |label: &str, key: String, path: Option<PathBuf>, lead: bool| -> El {
        let on = path.as_ref().is_some_and(|x| playbar::playing(s, x));
        let mut b = Button::new(label).small().grow().key(key).icon(if on { &icons::PAUSE } else { &icons::PLAY });
        if lead && !on {
            b = b.primary();
        }
        let (name, what) = (r.name.clone(), if lead { "Redub · result" } else { "Redub · original" });
        match path {
            Some(path) => b.on_press(move |s| toggle_play(s, &path, name.clone(), what)).build(p),
            None => b.disabled(true).build(p),
        }
    };
    row().w_full().gap(8.0)
        .child(mk("Original", format!("ab-orig-{tag}{id}"), r.preview.clone(), false))
        .child(mk("Result", format!("ab-res-{tag}{id}"), Some(r.play.clone()), true))
}

/// "dubbed to Ukrainian", "converted" or "re-spoken".
fn mode_past(r: &Redub) -> String {
    match r.mode {
        RedubMode::Dub => format!("dubbed to {}", lang_name(r.to.as_deref().unwrap_or("en"))),
        RedubMode::Convert => "converted".to_owned(),
        RedubMode::Respeak => "re-spoken".to_owned(),
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or_else(String::new, |f| f.to_uppercase().collect::<String>() + c.as_str())
}

fn result_card(s: &AppState, p: Palette, r: &Redub) -> El {
    let (reveal, open) = open_reveal(p, r.id, "main", &r.path);
    let mut meta = format!("{} · made in {:.1} s · {}", fmt_time(r.duration_s), r.elapsed_s, mode_past(r));
    if let Some(d) = r.drift_s {
        meta = format!("{meta} · {d:.1} s longer than the original");
    }
    glass_card(p).key("result-card").w_full().padding(12.0).gap(10.0)
        .child(row().w_full().gap(10.0).center_y()
            .child(badge(p, "Done", Tone::Success))
            .child(col().grow(1.0).min_w(0.0).gap(1.0)
                .child(text(format!("Your redub of {} is ready", r.name)).font(ty::LABEL).color(p.text).truncate())
                .child(text(meta).font(ty::CAPTION).color(p.text_faint)))
            .child(formats::export_button(p, format!("export-main{}", r.id), &r.path, &r.play, false))
            .child(reveal)
            .child(open))
        .child(ab_buttons(s, p, r, "main"))
}

/// The progress line and bar of a running dub or render, by engine stage.
fn stage_label(run: &Run, mode: RedubMode) -> (String, Option<f32>) {
    let frac = (run.pct > 0.0).then(|| (run.pct / 100.0).max(0.03));
    match run.stage.as_str() {
        "decode" => ("Reading the audio…".to_owned(), None),
        "load_model" => ("Loading the speech models… (the first time takes longer)".to_owned(), None),
        "transcribe" => (format!("Transcribing… {:.0}%", run.pct), frac),
        "translate" => ("Translating…".to_owned(), frac),
        "save" => ("Saving…".to_owned(), frac),
        "convert" => (format!("Converting the voice… {:.0}%", run.pct), frac),
        _ => match run.segment {
            Some((k, n)) => {
                let k = k.clamp(1, n.max(1));
                let by_part = (k.saturating_sub(1)) as f32 / n.max(1) as f32;
                (format!("Speaking part {k} of {n}…"), Some(if run.pct > 0.0 { run.pct / 100.0 } else { by_part }.max(0.03)))
            }
            None if run.pct > 0.0 => (if mode == RedubMode::Convert { format!("Converting the voice… {:.0}%", run.pct) } else { format!("Working… {:.0}%", run.pct) }, frac),
            None => ("Starting…".to_owned(), None),
        },
    }
}

fn action_bar(s: &AppState, p: Palette) -> El {
    if let Job::Rendering { run, mode, voice, .. } = &s.redub.job {
        let elapsed = (s.shell.now - run.started).max(0.0) as f32;
        let (label, frac) = stage_label(run, *mode);
        let sub = format!("{} as {voice}. Long files take a few minutes; you can use other tabs meanwhile.", if *mode == RedubMode::Dub { "Dubbing" } else { "Speaking" });
        return row().key("bar-rendering").w_full().gap(20.0).center_y()
            .child(col().grow(1.0).min_w(0.0).gap(8.0)
                .child(row().w_full().center_y().justify_between()
                    .child(text(label).font(ty::LABEL).color(p.text))
                    .child(text(fmt_time(elapsed)).font(ty::NUM).color(p.text_dim)))
                .child(progress_bar(p, frac))
                .child(text(sub).font(ty::CAPTION).color(p.text_faint).truncate()))
            .child(Button::new("Cancel").large().icon(&icons::STOP).key("cancel-render").on_press(cancel).build(p));
    }
    let block = render_blocker(s);
    let working = s.redub.busy();
    let voice_line = match shell::active_voice(s) {
        Some(v) => col().gap(2.0)
            .child(row().gap(8.0).center_y().child(text(format!("{} as {}", if s.redub.mode == RedubMode::Dub { "Dubbing" } else { "Redubbing" }, v.name)).font(ty::LABEL).color(p.text))
                .child(badge(p, if v.trained { "Trained" } else { "Zero-shot" }, if v.trained { Tone::Success } else { Tone::Accent })))
            .child(match &block {
                Some((b, serious)) => text(b.clone()).font(ty::CAPTION).color(if *serious { p.warn } else { p.text_faint }),
                None if s.shell.preparing_voice() => text("Preparing voice…").font(ty::CAPTION).color(p.text_dim),
                None if s.redub.mode == RedubMode::Dub && !s.redub.reviewing => text(if s.redub.review_first { format!("Ready. You will review the {} before it is spoken.", lang_name(&s.redub.to)) } else { format!("Ready to translate to {} and dub.", lang_name(&s.redub.to)) }).font(ty::CAPTION).color(p.text_faint),
                None => text("Ready to render.").font(ty::CAPTION).color(p.text_faint),
            }.wrap()),
        None => col().gap(2.0)
            .child(text("No voice selected").font(ty::LABEL).color(p.text))
            .child(text("Record your voice first, or pick one at the top right.").font(ty::CAPTION).color(p.warn)),
    };
    let label = match (s.redub.mode, s.redub.reviewing) {
        (RedubMode::Dub, false) => "Translate & dub",
        (RedubMode::Dub, true) => "Dub it",
        _ => "Render",
    };
    let go = Button::new(label).primary().large().icon(&icons::SPARKLE).key("render").disabled(block.is_some() || working).on_press(start_primary).build(p);
    row().key("bar-idle").w_full().gap(20.0).center_y().child(el().grow(1.0).min_w(0.0).child(voice_line)).child(go)
}

// ---- history ----

fn results_panel(s: &AppState, p: Palette, own_scroll: bool) -> El {
    let h = &s.redub.history;
    let head = speak::results_head(s, p, h.len(), "redub-outfmt");
    let body = if h.is_empty() {
        col().key("redubs-empty").w_full().grow(1.0).justify_center().child(empty_state(p, &ICON, "No redubs yet", "Finished redubs from this session appear here."))
    } else {
        let list = col().w_full().gap(8.0).children(h.iter().map(|r| history_row(s, p, r)));
        if own_scroll {
            col().key("redubs").w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0).child(list)
        } else {
            col().key("redubs").w_full().child(list)
        }
    };
    section_panel(p, "Results").key("results-panel").w_full().grow(1.0).min_h(0.0).gap(10.0).child(head).child(body)
}

fn history_row(s: &AppState, p: Palette, r: &Redub) -> El {
    let (reveal, open) = open_reveal(p, r.id, "h", &r.path);
    let meta = format!("{} · {} · {:.0} s", capitalize(&mode_past(r)), fmt_time(r.duration_s), r.elapsed_s);
    glass_card(p).key(format!("redub{}", r.id)).w_full().padding(10.0).gap(8.0)
        .child(col().w_full().gap(1.0)
            .child(text(r.name.clone()).font(ty::LABEL).color(p.text).truncate())
            .child(text(format!("{meta} · {}", r.voice)).font(ty::CAPTION).color(p.text_faint).truncate()))
        .child(ab_buttons(s, p, r, "h"))
        .child(if trash::confirming(s, trash::Kind::Redub, r.id) {
            trash::confirm_row(s, p, trash::Kind::Redub, r.id, &r.path, &r.play)
        } else {
            row().w_full().gap(8.0).center_y()
                .child(formats::export_button(p, format!("export-h{}", r.id), &r.path, &r.play, true))
                .child(reveal)
                .child(open)
                .child(trash::button(p, trash::Kind::Redub, r.id))
        })
}

/// A right arrow between "From" and "To".
static ARROW: rune::Icon = rune::Icon {
    name: "arrow-right",
    view: 16.0,
    body: r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8H13M9.5 4.5L13 8L9.5 11.5"/></g>"#,
};

/// The tab icon: two wavy lines trading places (swap), outline like the other icons.
pub static ICON: rune::Icon = rune::Icon {
    name: "redub",
    view: 16.0,
    body: concat!(
        r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">"#,
        r#"<path d="M2.2 5.4Q3.8 2.8 5.4 5.4T8.6 5.4T11.8 5.4"/><path d="M10 3.2L12.4 5.4L10 7.6"/>"#,
        r#"<path d="M13.8 10.6Q12.2 8 10.6 10.6T7.4 10.6T4.2 10.6"/><path d="M6 8.4L3.6 10.6L6 12.8"/>"#,
        "</g>"
    ),
};

// -------------------------------------------------------------------------------------------------
// Headless scenes
// -------------------------------------------------------------------------------------------------

const SAMPLE_PATH: &str = "/tmp/interview-take-2.mp3";

fn demo(s: &mut AppState, trained: bool) {
    s.screen = Screen::Redub;
    s.formats.scene_ready();
    s.shell.voices = vec![
        VoiceInfo { id: "me".into(), name: "My voice".into(), preset: Some("standard".into()), dataset_minutes: 6.4, parts: 5, trained, trained_epochs: if trained { 300 } else { 0 }, ..Default::default() },
        VoiceInfo { id: "pro".into(), name: "Studio take".into(), preset: Some("pro".into()), dataset_minutes: 15.0, parts: 12, trained: true, trained_epochs: 300, ..Default::default() },
    ];
    s.shell.active_voice = Some("me".into());
    s.shell.set_engine(EngineStatus::Ready { device: Some("NVIDIA GeForce RTX 4070".into()), tts: Some("ready".into()), rvc: Some("ready".into()) });
}

fn demo_file(s: &mut AppState) {
    s.redub.file = Some(Source {
        path: SAMPLE_PATH.into(),
        name: "interview-take-2.mp3".into(),
        probe: Probe::Ready(RedubProbe { duration_s: 161.0, sample_rate: 44100, channels: 2, has_audio: true, has_video: Some(false), format: "mp3".into(), preview_path: Some("/tmp/interview-take-2-preview.wav".into()) }),
        prepared: true,
    });
}

fn demo_segments(s: &mut AppState) {
    let lines = [
        (0.0, 4.2, "Hello everyone, and welcome back to the show."),
        (4.6, 9.8, "Today we are going to talk about how voices are cloned from just a few minutes of speech."),
        (10.3, 15.1, "It sounds like magic, but it is mostly careful recording and a bit of patience."),
        (15.8, 19.0, "Let's get started."),
    ];
    s.redub.segments = lines
        .iter()
        .enumerate()
        .map(|(i, (a, b, t))| Seg { key: i as u64 + 1, start: *a, end: *b, source: String::new(), edit: TextEdit::new(*t) })
        .collect();
    s.redub.next_seg = lines.len() as u64;
    s.redub.transcript_lang = Some("en".into());
}

fn demo_history(s: &mut AppState) {
    let mk = |id, name: &str, mode, d, e| Redub { id, name: name.into(), voice: "My voice".into(), mode, path: format!("/tmp/redub{id}.wav").into(), play: format!("/tmp/redub{id}.wav").into(), preview: Some(format!("/tmp/orig{id}.wav").into()), duration_s: d, elapsed_s: e, drift_s: None, to: None };
    s.redub.history = vec![
        mk(2, "podcast-intro.wav", RedubMode::Respeak, 48.0, 31.0),
        mk(1, "interview-take-1.mp3", RedubMode::Convert, 161.0, 74.0),
    ];
    s.redub.next_id = 2;
}

fn scene_empty(s: &mut AppState) {
    demo(s, true);
}

fn scene_file(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
}

fn scene_transcribing(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Respeak;
    let mut run = Run::new(1, -14.0);
    run.job = Some("job1".into());
    run.stage = "transcribe".into();
    run.pct = 42.0;
    s.redub.job = Job::Transcribing(run);
}

fn scene_transcript(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Respeak;
    demo_segments(s);
    s.redub.segments[1].edit = TextEdit::new("Today we are going to talk about how a voice is cloned from only a few minutes of speech.");
    demo_history(s);
}

fn scene_rendering(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Respeak;
    demo_segments(s);
    let mut run = Run::new(2, -52.0);
    run.job = Some("job2".into());
    run.stage = "speak".into();
    run.segment = Some((3, 4));
    run.pct = 58.0;
    s.redub.job = Job::Rendering { run, mode: RedubMode::Respeak, source: "interview-take-2.mp3".into(), voice: "My voice".into() };
    demo_history(s);
}

fn scene_done(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Convert;
    demo_history(s);
    s.redub.history.insert(0, Redub { id: 3, name: "interview-take-2.mp3".into(), voice: "My voice".into(), mode: RedubMode::Convert, path: "/tmp/redub3.wav".into(), play: "/tmp/redub3.wav".into(), preview: Some("/tmp/interview-take-2-preview.wav".into()), duration_s: 161.0, elapsed_s: 68.4, drift_s: None, to: None });
    s.redub.next_id = 3;
    s.redub.result = Some(3);
}

fn scene_convert_untrained(s: &mut AppState) {
    demo(s, false);
    demo_file(s);
    s.redub.mode = RedubMode::Convert;
}

fn scene_training_busy(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Respeak;
    s.train.phase = super::train::Phase::Running;
    s.train.pct = 37.0;
}

fn scene_drop_hover(s: &mut AppState) {
    demo(s, true);
    s.formats.over = true;
}

fn scene_export_menu(s: &mut AppState) {
    scene_done(s);
    s.formats.menu = Some(formats::Menu { anchor: "export-main3".into(), src: "/tmp/redub3.wav".into(), audio: "/tmp/redub3.wav".into() });
    s.formats.format = "mp3".into();
}

/// Results saved as FLAC, with the "Save results as" menu open.
fn scene_output_menu(s: &mut AppState) {
    scene_done(s);
    s.formats.scene_ready();
    for r in &mut s.redub.history {
        r.path = r.path.with_extension("flac");
    }
    s.formats.output = "flac".into();
    s.formats.output_menu = Some("redub-outfmt".into());
}

const DUB_LINES: [(f32, f32, &str, &str); 4] = [
    (0.0, 4.2, "Привіт усім, і ласкаво просимо назад на шоу.", "Hello everyone, and welcome back to the show."),
    (4.6, 9.8, "Сьогодні ми поговоримо про те, як клонують голос лише з кількох хвилин мовлення.", "Today we will talk about how a voice is cloned from just a few minutes of speech."),
    (10.3, 15.1, "Це звучить як магія, але насправді це уважний запис і трохи терпіння.", "It sounds like magic, but it is really careful recording and a bit of patience."),
    (15.8, 19.0, "Почнімо.", "Let's begin."),
];

fn demo_dub_file(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Dub;
    s.redub.from = "uk".into();
}

fn scene_dub_ready(s: &mut AppState) {
    demo_dub_file(s);
    demo_history(s);
}

/// A video file instead of the demo's MP3.
fn demo_video(s: &mut AppState) {
    if let Some(f) = s.redub.file.as_mut() {
        f.path = "/tmp/game-tour.mp4".into();
        f.name = "game-tour.mp4".into();
        if let Probe::Ready(p) = &mut f.probe {
            p.has_video = Some(true);
            p.format = "mp4".into();
        }
    }
}

/// Translate & dub of a video: "Keep original timing" is on by default, with the video hint.
fn scene_dub_timing_video(s: &mut AppState) {
    scene_dub_ready(s);
    demo_video(s);
}

/// The same with the switch turned off for videos: natural pace, the hint warns about the sync.
fn scene_dub_timing_video_off(s: &mut AppState) {
    scene_dub_timing_video(s);
    s.redub.keep_timing = [None, Some(false)];
}

/// Re-speak of an audio file with the switch turned on (it is off by default for audio).
fn scene_respeak_timing_keep(s: &mut AppState) {
    scene_transcript(s);
    s.redub.keep_timing = [Some(true), None];
}

/// Translate & dub with both translators installed: "Translation quality" is offered (Better chosen).
fn scene_dub_translator(s: &mut AppState) {
    scene_dub_ready(s);
    s.shell.translators = vec![TRANSLATOR_BETTER.into(), TRANSLATOR_FASTER.into()];
}

/// The same with Faster chosen.
fn scene_dub_translator_faster(s: &mut AppState) {
    scene_dub_translator(s);
    s.redub.faster_translation = true;
}

fn scene_dub_running(s: &mut AppState) {
    demo_dub_file(s);
    let mut run = Run::new(3, -21.0);
    run.job = Some("job3".into());
    run.stage = "translate".into();
    run.pct = 34.0;
    s.redub.job = Job::Rendering { run, mode: RedubMode::Dub, source: "interview-take-2.mp3".into(), voice: "My voice".into() };
    demo_history(s);
}

fn scene_dub_review(s: &mut AppState) {
    demo_dub_file(s);
    s.redub.review_first = true;
    s.redub.segments = DUB_LINES
        .iter()
        .enumerate()
        .map(|(i, (a, b, src, en))| Seg { key: i as u64 + 1, start: *a, end: *b, source: (*src).into(), edit: TextEdit::new(*en) })
        .collect();
    s.redub.next_seg = DUB_LINES.len() as u64;
    s.redub.reviewing = true;
    s.redub.transcript_lang = Some("en".into());
    s.redub.transcript_from = Some("uk".into());
    s.redub.transcript_translated = true;
    demo_history(s);
}

fn scene_dub_done(s: &mut AppState) {
    demo_dub_file(s);
    demo_history(s);
    s.redub.history.insert(0, Redub { id: 3, name: "interview-take-2.mp3".into(), voice: "My voice".into(), mode: RedubMode::Dub, path: "/tmp/redub3.wav".into(), play: "/tmp/redub3.wav".into(), preview: Some("/tmp/interview-take-2-preview.wav".into()), duration_s: 163.2, elapsed_s: 92.0, drift_s: Some(2.2), to: Some("en".into()) });
    s.redub.next_id = 3;
    s.redub.result = Some(3);
}

/// Translate & dub from English to Ukrainian, with the picker open on "To" and a finished Ukrainian dub in the history.
fn scene_dub_to_uk(s: &mut AppState) {
    demo(s, true);
    demo_file(s);
    s.redub.mode = RedubMode::Dub;
    s.redub.from = "en".into();
    s.redub.to = "uk".into();
    demo_history(s);
    s.redub.history.insert(0, Redub { id: 3, name: "podcast-intro.wav".into(), voice: "My voice".into(), mode: RedubMode::Dub, path: "/tmp/redub3.wav".into(), play: "/tmp/redub3.wav".into(), preview: Some("/tmp/orig3.wav".into()), duration_s: 49.5, elapsed_s: 41.0, drift_s: None, to: Some("uk".into()) });
    s.redub.next_id = 3;
}

fn scene_dub_to_uk_picker(s: &mut AppState) {
    scene_dub_to_uk(s);
    s.langpick.open = Some((lang::Target::RedubTo, "dub-to".into()));
}

/// The review of an English -> Ukrainian dub.
fn scene_dub_review_uk(s: &mut AppState) {
    demo_dub_file(s);
    s.redub.from = "en".into();
    s.redub.to = "uk".into();
    s.redub.review_first = true;
    s.redub.segments = DUB_LINES
        .iter()
        .enumerate()
        .map(|(i, (a, b, uk, en))| Seg { key: i as u64 + 1, start: *a, end: *b, source: (*en).into(), edit: TextEdit::new(*uk) })
        .collect();
    s.redub.next_seg = DUB_LINES.len() as u64;
    s.redub.reviewing = true;
    s.redub.transcript_lang = Some("uk".into());
    s.redub.transcript_from = Some("en".into());
    s.redub.transcript_translated = true;
}

/// Ukrainian -> English with a voice recorded in Ukrainian: the Accent choice (native by default).
fn scene_dub_accent(s: &mut AppState) {
    demo_dub_file(s);
    for v in &mut s.shell.voices {
        v.language = "uk".into();
    }
    demo_history(s);
}

/// The same with an untrained voice and the zero-shot converter installed: Native, from the reference clip.
fn scene_dub_accent_zero_shot(s: &mut AppState) {
    scene_dub_accent_untrained(s);
    s.shell.zero_shot_vc = true;
}

/// The same with an untrained voice and no zero-shot converter: the choice is disabled with the hint.
fn scene_dub_accent_untrained(s: &mut AppState) {
    demo(s, false);
    demo_file(s);
    s.redub.mode = RedubMode::Dub;
    s.redub.from = "uk".into();
    for v in &mut s.shell.voices {
        v.language = "uk".into();
    }
}

/// Re-speak with a transcript in a language the voice does not speak (Hebrew): Render is disabled with the hint to
/// translate.
fn scene_respeak_foreign(s: &mut AppState) {
    scene_transcript(s);
    s.redub.transcript_lang = Some("he".into());
    for (g, (_, _, src, _)) in s.redub.segments.iter_mut().zip(DUB_LINES.iter()) {
        g.edit = TextEdit::new(*src);
    }
}

/// The owner's case in Translate & dub: Ukrainian -> English with a voice set to English whose reference clip is
/// Ukrainian speech, not trained: the Accent choice, disabled, with "Add an English clip".
fn scene_dub_accent_foreign_ref(s: &mut AppState) {
    demo(s, false);
    demo_file(s);
    s.redub.mode = RedubMode::Dub;
    s.redub.from = "uk".into();
    s.shell.voices[0].reference_language = Some("uk".into());
}

/// Re-speak of an English transcript with that voice, trained: the Accent choice (Native) under the options.
fn scene_respeak_accent(s: &mut AppState) {
    scene_transcript(s);
    s.shell.voices[0].reference_language = Some("uk".into());
}

/// The same untrained without the zero-shot converter: disabled, with the hint to install it, train or add an
/// English clip.
fn scene_respeak_accent_untrained(s: &mut AppState) {
    scene_respeak_accent(s);
    s.shell.voices[0].trained = false;
    s.shell.voices[0].trained_epochs = 0;
}

/// The same untrained with the zero-shot converter installed: Native, from the reference clip.
fn scene_respeak_accent_zero_shot(s: &mut AppState) {
    scene_respeak_accent_untrained(s);
    s.shell.zero_shot_vc = true;
}

/// "Delete this result?" on a Results row, with the other result paused in the player bar (the action bar above
/// it stays usable).
fn scene_trash_confirm(s: &mut AppState) {
    scene_file(s);
    demo_history(s);
    trash::scene_confirm(s, trash::Kind::Redub, 2);
    playbar::scene_now(s, "/tmp/redub1.wav", "interview-take-1.mp3", "Redub · result", playbar::Source::Redub, playbar::Status::Paused, 97.0, 161.0);
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("redub_trash_confirm", scene_trash_confirm),
        ("redub_dub_accent_foreign_ref", scene_dub_accent_foreign_ref),
        ("redub_respeak_accent", scene_respeak_accent),
        ("redub_respeak_accent_untrained", scene_respeak_accent_untrained),
        ("redub_respeak_accent_zero_shot", scene_respeak_accent_zero_shot),
        ("redub_dub_ready", scene_dub_ready),
        ("redub_dub_timing_video", scene_dub_timing_video),
        ("redub_dub_timing_video_off", scene_dub_timing_video_off),
        ("redub_respeak_timing_keep", scene_respeak_timing_keep),
        ("redub_dub_accent", scene_dub_accent),
        ("redub_dub_accent_untrained", scene_dub_accent_untrained),
        ("redub_dub_accent_zero_shot", scene_dub_accent_zero_shot),
        ("redub_dub_translator", scene_dub_translator),
        ("redub_dub_translator_faster", scene_dub_translator_faster),
        ("redub_dub_to_uk", scene_dub_to_uk),
        ("redub_dub_to_uk_picker", scene_dub_to_uk_picker),
        ("redub_dub_review_uk", scene_dub_review_uk),
        ("redub_dub_running", scene_dub_running),
        ("redub_dub_review", scene_dub_review),
        ("redub_dub_done", scene_dub_done),
        ("redub_respeak_foreign", scene_respeak_foreign),
        ("redub_empty", scene_empty),
        ("redub_drop_hover", scene_drop_hover),
        ("redub_file", scene_file),
        ("redub_transcribing", scene_transcribing),
        ("redub_transcript", scene_transcript),
        ("redub_rendering", scene_rendering),
        ("redub_done", scene_done),
        ("redub_convert_untrained", scene_convert_untrained),
        ("redub_training_busy", scene_training_busy),
        ("redub_export_menu", scene_export_menu),
        ("redub_output_menu", scene_output_menu),
    ]
}

