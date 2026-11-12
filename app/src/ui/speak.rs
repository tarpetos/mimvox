//! Speak screen: type text, press Generate, hear it in your voice while it is still being made.
//!
//! Flow: `tts.generate` -> `tts.chunk` per text segment -> `tts.done` (the result lands in the history list with
//! play / open / reveal actions). Each result is saved as one file in the format of the "Save as" chip in the Results
//! header (`output_format`, shared with Redub; see `formats::output_chip`); a non-WAV result plays from the engine's
//! playback WAV.
//!
//! The output language (a chip in the Style panel, any language the engine speaks) defaults to the language the active
//! voice was recorded in and is remembered per voice (`speak_languages` in `settings.json`); `tts.generate` sends it as
//! `language`. Another language than the voice's own shows a hint about per-language reference clips; Ukrainian
//! shows how to fix a wrong word stress ("+" before the vowel, see `backends/omnivoice_stress.py`).
//!
//! English output without an English reference clip (no `reference_en.wav`, and `reference.wav` is detected as another
//! language, or, while its language is not known yet, the voice was recorded in another one) shows an "Accent" choice
//! (see [`accent_block`], shared with Redub's Translate & dub and Re-speak): "Native (American)" sends
//! `accent: "native-us"` (a native reference voice speaks; a trained voice's RVC model brings the user's timbre back,
//! an untrained voice gets it from its reference clip through the engine's zero-shot converter), "Like my recordings"
//! the voice's own reference. Native is the default. An untrained voice needs the zero-shot converter (`hello` ->
//! `zero_shot_vc`, installed by setup): without it the choice is shown disabled with a way to add an English clip
//! instead. The choice and the native speaker override are remembered in `settings.json` (`accent`, `native_voice`)
//! once the user changes them. A job spoken from a clip in another language than its text (`reference_note`) shows a
//! warning toast.
//!
//! Playback is pre-buffered: a single-segment job plays at once; for longer text the finished segments are held
//! back until, by the engine's real-time factor, the segments still to come will arrive before the buffered audio
//! runs out (see [`can_start`]). So it starts as early as it can without ever stuttering, and never later than
//! the end of the generation.

use super::components::{
    Button, Slider, TextArea, Tone, accent_glass, badge, empty_state, glass_card, icon_button, icon_el, icons, progress_bar, section_panel,
    segmented_fill, segmented_help, status_dot,
};
use super::lang;
use super::playbar;
use super::shell::{self, EngineStatus, VoiceInfo};
use super::trash;
use super::theme::{self, Palette, motion, ty};
use crate::engine::protocol::{self, EngineEvent, NativeVoice, Quality, TtsChunk, TtsStarted};
use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use rune::{Cx, El, Key, KeyEvent, ScrollSpec, TextEdit, Tx, col, el, row, text};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// One finished generation in the history list (newest first).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Take {
    pub id: u64,
    pub text: String,
    pub voice: String,
    /// The saved file (`<data>/outputs`, in the chosen format): Show in folder, Open file, Export.
    pub path: PathBuf,
    /// A WAV of it to play (the saved file itself when that is a WAV).
    pub play: PathBuf,
    pub duration_s: f32,
    pub elapsed_s: f32,
}

/// A generation in flight.
#[derive(Clone, Debug)]
struct Active {
    /// The `tts.generate` request; its answer carries the job id.
    req: u64,
    job: Option<String>,
    started: f64,
    /// Segments received so far.
    chunks: usize,
    /// Segments expected: the engine's number when it tells us (`exact`), else an estimate from the text.
    total: usize,
    exact: bool,
    /// Characters per segment as reported by `tts.generate` (weights for progress and the buffering estimate).
    seg_chars: Vec<u32>,
    /// Seconds of audio received so far.
    audio_s: f32,
    /// Generation seconds per audio second, from the latest chunk (0 = unknown).
    rtf: f32,
    /// Segments waiting for playback to start (see [`can_start`]).
    held: Vec<PathBuf>,
    /// Playback has been released: new segments are queued on the player at once.
    audible: bool,
    /// `play()` has been called for this job (later segments use `enqueue`).
    began: bool,
    text: String,
    voice: String,
    /// The voice id and the language spoken (to record the detected reference language from `tts.done`).
    voice_id: String,
    lang: String,
    /// The foreign-reference warning was shown already (from the answer; the end event then stays quiet).
    warned: bool,
    /// Stop pressed before the job id arrived: cancel as soon as it does.
    cancel: bool,
}

impl Active {
    fn new(req: u64, now: f64, text: String, voice: String) -> Active {
        Active {
            req,
            job: None,
            started: now,
            chunks: 0,
            total: estimate_segments(&text),
            exact: false,
            seg_chars: Vec::new(),
            audio_s: 0.0,
            rtf: 0.0,
            held: Vec::new(),
            audible: false,
            began: false,
            text,
            voice,
            voice_id: String::new(),
            lang: String::new(),
            warned: false,
            cancel: false,
        }
    }

    /// Whether finished segments are being held back for a smooth start.
    fn buffering(&self) -> bool {
        !self.audible && !self.held.is_empty()
    }
}

/// Whether playback can start with `buffered_s` seconds of audio ready while the segments in `remaining_s`
/// (estimated audio seconds each) are still being generated at `rtf` (generation seconds per audio second).
/// True when every remaining segment is done before the audio before it has played out.
fn can_start(buffered_s: f32, rtf: f32, remaining_s: &[f32]) -> bool {
    let (mut need, mut audio) = (0.0f32, buffered_s);
    for d in remaining_s {
        need += rtf * 1.1 * d;
        if need > audio {
            return false;
        }
        audio += d;
    }
    true
}

/// Playback start decision for a job that has `a.chunks` of `a.total` segments ready.
fn should_start(a: &Active) -> bool {
    if a.chunks >= a.total || a.rtf <= 0.0 || a.audio_s <= 0.0 {
        return true;
    }
    let done_chars: u32 = a.seg_chars.iter().take(a.chunks).sum();
    let remaining: Vec<f32> = if a.seg_chars.len() == a.total && done_chars > 0 {
        let per_char = a.audio_s / done_chars as f32;
        a.seg_chars[a.chunks..].iter().map(|c| *c as f32 * per_char).collect()
    } else {
        vec![a.audio_s / a.chunks.max(1) as f32; a.total - a.chunks]
    };
    can_start(a.audio_s, a.rtf, &remaining)
}

#[derive(Clone, Debug, Default)]
enum Gen {
    #[default]
    Idle,
    Active(Active),
}

pub struct SpeakState {
    pub text: TextEdit,
    /// 0.4..=1.0, maps to OmniVoice's sampling temperatures.
    pub expressiveness: f32,
    /// 0.3..=1.0, maps to the RVC index rate.
    pub voice_match: f32,
    /// Best (32 decoding steps) or Fast (16).
    pub quality: Quality,
    /// 0.8..=1.3, speaking speed (engine `duration_factor = 1 / speed`); persisted as `speed` in settings.json.
    pub speed: f32,
    speed_loaded: bool,
    speed_dirty: bool,
    /// Output language per voice id, when it is not the voice's own language (`speak_languages` in settings.json).
    pub languages: HashMap<String, String>,
    languages_dirty: bool,
    /// "Native (American)" accent for English from a voice recorded in another language (default on; `accent` in
    /// settings.json: `native-us` | `own`). Shared with Redub.
    pub native_accent: bool,
    /// Which native speaker (`native_voice` in settings.json; `Auto` picks by the voice's pitch).
    pub native_voice: NativeVoice,
    accent_dirty: bool,
    gen_: Gen,
    pub takes: Vec<Take>,
    next_take: u64,
}

impl Default for SpeakState {
    fn default() -> Self {
        let mut text = TextEdit::new("");
        text.focused = true;
        SpeakState {
            text,
            expressiveness: 0.8,
            voice_match: 1.0,
            quality: Quality::Best,
            speed: 1.0,
            speed_loaded: false,
            speed_dirty: false,
            languages: HashMap::new(),
            languages_dirty: false,
            native_accent: true,
            native_voice: NativeVoice::Auto,
            accent_dirty: false,
            gen_: Gen::Idle,
            takes: Vec::new(),
            next_take: 0,
        }
    }
}

impl SpeakState {
    pub fn is_generating(&self) -> bool {
        matches!(self.gen_, Gen::Active(_))
    }
}

// -------------------------------------------------------------------------------------------------
// Actions
// -------------------------------------------------------------------------------------------------

/// The output language for the active voice: the one picked for it, else the language it was recorded in.
pub fn language(s: &AppState) -> String {
    match shell::active_voice(s) {
        Some(v) => s.speak.languages.get(&v.id).cloned().unwrap_or_else(|| v.lang().to_owned()),
        None => "en".to_owned(),
    }
}

/// The picker chose `code` for the active voice (the voice's own language is stored as "no choice").
pub fn set_language(s: &mut AppState, code: &str) {
    let Some(v) = shell::active_voice(s).cloned() else { return };
    if code == v.lang() {
        s.speak.languages.remove(&v.id);
    } else {
        s.speak.languages.insert(v.id, code.to_owned());
    }
    s.speak.languages_dirty = true;
}

/// Whether the Accent choice applies to speech in `lang` with the active voice: English output with no English
/// reference clip, so the voice's clip in another language would lend English its accent (see
/// [`VoiceInfo::has_reference_in`]: a `reference_en.wav`, or a `reference.wav` detected as English).
pub fn accent_applies(s: &AppState, lang: &str) -> bool {
    shell::active_voice(s).is_some_and(|v| lang::base(lang) == "en" && !v.has_reference_in("en"))
}

/// Whether the active voice can use the native accent: trained (its RVC model gives the timbre), or the engine has
/// the zero-shot converter (it takes the timbre from the reference clip, no training).
pub fn native_possible(s: &AppState) -> bool {
    shell::active_voice(s).is_some_and(|v| v.trained || s.shell.zero_shot_vc)
}

/// The `accent` of a speech request in `lang`: the native speaker when the native accent applies, is chosen and is
/// possible (an untrained voice without the zero-shot converter would lose its timbre), else `None` (the voice's own
/// reference).
pub fn accent_param(s: &AppState, lang: &str) -> Option<NativeVoice> {
    (accent_applies(s, lang) && native_possible(s) && s.speak.native_accent).then_some(s.speak.native_voice)
}

/// The `accent` to warm the active voice with (`voices.warm`): native when chosen and possible and the voice has no
/// English reference clip (its English output would use it).
pub fn warm_accent(s: &AppState) -> Option<NativeVoice> {
    let v = shell::active_voice(s)?;
    (native_possible(s) && !v.has_reference_in("en") && s.speak.native_accent).then_some(s.speak.native_voice)
}

/// Opens Voices with "Add a language" set to `code`: record or import a reference clip in that language.
pub fn add_reference(s: &mut AppState, code: &str) {
    super::refclip::start_language(s, code);
    shell::set_screen(s, Screen::Voices);
}

/// The warning toast of a job's start answer: why the native accent was not used and/or that the clip speaking is in
/// another language (`accent_note`, `reference_note`). True when the reference warning was part of it.
pub(crate) fn job_notes(s: &mut AppState, accent_note: Option<&str>, reference_note: Option<&str>) -> bool {
    let notes: Vec<&str> = [accent_note, reference_note].into_iter().flatten().filter(|n| !n.is_empty()).collect();
    if !notes.is_empty() {
        shell::toast(s, Tone::Warning, notes.join(" "));
    }
    reference_note.is_some_and(|n| !n.is_empty())
}

/// After a speech job: the detected language of `reference.wav` when that clip spoke (not the native clip, not a
/// `reference_<lang>.wav`), so the Accent choice and the Voices screen follow a clip transcribed for the first time.
pub(crate) fn note_reference(s: &mut AppState, voice_id: &str, lang: &str, used: &protocol::ReferenceUse) {
    if used.accent.as_deref() == Some("native-us") || used.reference_language.is_none() {
        return;
    }
    let own_clip = s.shell.voices.iter().find(|v| v.id == voice_id).is_some_and(|v| !v.references.iter().any(|r| lang::base(r) == lang::base(lang)));
    if own_clip {
        let value = json!(used.reference_language);
        shell::set_reference_language(s, voice_id, &value);
    }
}

/// Why Generate cannot run right now (shown next to the button), or `None` when it can.
fn blocker(s: &AppState) -> Option<&'static str> {
    match &s.shell.engine {
        EngineStatus::Idle | EngineStatus::Starting { .. } => return Some("Waiting for the engine to start…"),
        EngineStatus::Stopped { .. } | EngineStatus::Unavailable { .. } => return Some("The engine is not running. Press Retry above."),
        EngineStatus::Ready { .. } => {}
    }
    if shell::active_voice(s).is_none() {
        return Some("Record your voice first, or pick one at the top right.");
    }
    None
}

/// Starts generating the current text with the selected voice.
pub fn generate(s: &mut AppState) {
    if s.speak.is_generating() {
        return;
    }
    let text = s.speak.text.text.trim().to_owned();
    if text.is_empty() {
        return;
    }
    if let Some(why) = blocker(s) {
        shell::toast(s, Tone::Warning, why);
        return;
    }
    let Some(voice) = shell::active_voice(s).cloned() else { return };
    let lang = language(s);
    let out = super::formats::output_format(s);
    let accent = accent_param(s, &lang);
    let (method, params) =
        protocol::tts_generate(&voice.id, &text, s.speak.expressiveness, s.speak.voice_match, s.speak.quality, s.speak.speed, &lang, &out, accent);
    shell::stop_audio(s);
    let Some(req) = shell::request(s, method, params) else {
        shell::toast(s, Tone::Danger, "Could not reach the engine.");
        return;
    };
    let mut a = Active::new(req, shell::clock_now(s), text, voice.name);
    (a.voice_id, a.lang) = (voice.id, lang);
    s.speak.gen_ = Gen::Active(a);
}

/// Stops the running generation (and the audio that is streaming from it).
pub fn cancel(s: &mut AppState) {
    let job = match &mut s.speak.gen_ {
        Gen::Active(a) => match a.job.clone() {
            Some(j) => Some(j),
            None => {
                a.cancel = true;
                return;
            }
        },
        Gen::Idle => return,
    };
    if let Some(job) = job {
        shell::request(s, "tts.cancel", json!({ "job": job }));
    }
    shell::stop_audio(s);
    s.speak.gen_ = Gen::Idle;
}

/// A take's play button: play it, or pause / resume it in the player bar.
fn toggle_take(s: &mut AppState, id: u64) {
    let Some(t) = s.speak.takes.iter().find(|t| t.id == id) else { return };
    let (path, title) = (t.play.clone(), one_line(&t.text));
    if let Err(e) = playbar::toggle(s, &path, title, "Speak · result", playbar::Source::Speak) {
        shell::toast(s, Tone::Danger, e);
    }
}

/// The engine deleted a take's files (`ui::trash`): drop it from the list.
pub fn remove_take(s: &mut AppState, id: u64) {
    s.speak.takes.retain(|t| t.id != id);
}

/// A rough segment count: sentences packed up to ~120 tokens (about 4 characters each).
fn estimate_segments(text: &str) -> usize {
    let (mut segments, mut tokens) = (1usize, 0usize);
    for sentence in text.split_inclusive(['.', '!', '?', '\n']) {
        let t = sentence.trim().chars().count().div_ceil(4);
        if tokens > 0 && tokens + t > 120 {
            segments += 1;
            tokens = 0;
        }
        tokens += t;
    }
    segments
}

fn friendly_error(code: &str, message: &str) -> String {
    match code {
        "not_ready" => "The engine is still loading its models. Try again in a moment.".into(),
        "busy" => "The engine is busy with another job. Try again when it finishes.".into(),
        "not_found" => "That voice could not be found. Pick another one.".into(),
        _ if message.is_empty() => "Generation failed.".into(),
        _ => message.to_owned(),
    }
}

// -------------------------------------------------------------------------------------------------
// Engine messages and events
// -------------------------------------------------------------------------------------------------

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    let Gen::Active(mut a) = s.speak.gen_.clone() else { return };
    match msg {
        EngineMsg::Response { id, result } if *id == a.req => match result {
            Ok(v) => {
                let mut notes = (None, None);
                if let Ok(TtsStarted { job, segments, segment_chars, accent_note, reference_note }) = protocol::parse_result(v.clone()) {
                    a.job = Some(job);
                    if segments > 0 {
                        a.total = segments as usize;
                        a.exact = true;
                    }
                    a.seg_chars = segment_chars;
                    notes = (accent_note, reference_note);
                }
                let stop = a.cancel;
                a.warned = job_notes(s, notes.0.as_deref(), notes.1.as_deref());
                s.speak.gen_ = Gen::Active(a);
                if stop {
                    cancel(s);
                }
            }
            Err(e) => {
                let text = friendly_error(&e.code, &e.message);
                s.speak.gen_ = Gen::Idle;
                shell::toast(s, Tone::Danger, text);
            }
        },
        EngineMsg::Event(ev) if a.job.is_some() && ev.job == a.job => match ev.parse() {
            EngineEvent::TtsChunk { chunk, .. } => on_chunk(s, a, chunk),
            EngineEvent::TtsDone { done, .. } => {
                flush_held(s, &mut a);
                let play = done.play_path();
                s.speak.next_take += 1;
                let take = Take {
                    id: s.speak.next_take,
                    text: a.text.clone(),
                    voice: a.voice.clone(),
                    path: done.path.clone(),
                    play,
                    duration_s: done.duration_s,
                    elapsed_s: if ev.data.get("elapsed_s").is_some() { done.elapsed_s } else { (s.shell.now - a.started) as f32 },
                };
                // What streamed is this take now: the player bar names it and its card shows it playing.
                playbar::finish_live(s, Some((&take.play, one_line(&take.text))));
                s.speak.takes.insert(0, take);
                s.speak.takes.truncate(50);
                s.speak.gen_ = Gen::Idle;
                // A clip transcribed for the first time by this job: its language is known only now.
                note_reference(s, &a.voice_id, &a.lang, &done.reference);
                if !a.warned {
                    if let Some(n) = done.reference.reference_note.clone().filter(|n| !n.is_empty()) {
                        shell::toast(s, Tone::Warning, n);
                    }
                }
            }
            EngineEvent::TtsError { error, .. } => {
                if error.code != "cancelled" {
                    let text = friendly_error(&error.code, &error.message);
                    shell::toast(s, Tone::Danger, text);
                }
                s.speak.gen_ = Gen::Idle;
                // What arrived can still be played to its end.
                playbar::finish_live(s, None);
            }
            _ => {}
        },
        EngineMsg::Exited { .. } => {
            s.speak.gen_ = Gen::Idle;
            playbar::finish_live(s, None);
        }
        _ => {}
    }
}

/// A segment finished: hold it for a smooth start, or queue it on the player when playback is already released.
fn on_chunk(s: &mut AppState, mut a: Active, chunk: TtsChunk) {
    a.chunks += 1;
    if chunk.segments > 0 {
        a.total = chunk.segments as usize;
        a.exact = true;
    }
    a.total = a.total.max(a.chunks);
    a.audio_s += chunk.duration_s;
    a.rtf = if chunk.rtf > 0.0 {
        chunk.rtf
    } else if chunk.ready_s > 0.0 && a.audio_s > 0.0 {
        chunk.ready_s / a.audio_s
    } else {
        0.0
    };
    a.held.push(chunk.path);
    if !a.audible && should_start(&a) {
        a.audible = true;
    }
    if a.audible {
        release(s, &mut a);
    }
    s.speak.gen_ = Gen::Active(a);
}

/// Queues what was held back (the first segment of the job starts playback, the rest follow gaplessly).
fn release(s: &mut AppState, a: &mut Active) {
    for path in std::mem::take(&mut a.held) {
        let first = !a.began;
        let r = if first { playbar::play_live(s, &path, one_line(&a.text)) } else { playbar::enqueue(s, &path) };
        match r {
            Ok(()) => a.began = true,
            Err(e) if first => {
                a.began = true;
                shell::toast(s, Tone::Warning, e);
            }
            Err(_) => {}
        }
    }
}

/// The job ended with segments still held (an engine without exact counts): play them now.
fn flush_held(s: &mut AppState, a: &mut Active) {
    if !a.held.is_empty() {
        a.audible = true;
        release(s, a);
    }
}

/// After every event: the saved speed and output languages once, then every change of them.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;
    if !s.shell.scripted() {
        if !s.speak.speed_loaded {
            s.speak.speed_loaded = true;
            s.speak.speed = super::onboarding::saved_speed();
            if let Some(Value::Object(m)) = super::onboarding::setting("speak_languages") {
                s.speak.languages = m.into_iter().filter_map(|(k, v)| v.as_str().map(|c| (k, c.to_owned()))).collect();
            }
            if let Some(a) = super::onboarding::setting("accent") {
                s.speak.native_accent = a.as_str() != Some("own");
            }
            if let Some(v) = super::onboarding::setting("native_voice") {
                s.speak.native_voice = NativeVoice::from_str(v.as_str().unwrap_or(""));
            }
            changed = true;
        } else {
            if s.speak.speed_dirty {
                s.speak.speed_dirty = false;
                super::onboarding::save_speed(s.speak.speed);
            }
            if s.speak.languages_dirty {
                s.speak.languages_dirty = false;
                super::onboarding::save_setting("speak_languages", json!(s.speak.languages));
            }
            if s.speak.accent_dirty {
                s.speak.accent_dirty = false;
                super::onboarding::save_setting("accent", json!(if s.speak.native_accent { "native-us" } else { "own" }));
                super::onboarding::save_setting("native_voice", json!(s.speak.native_voice.as_str()));
            }
        }
    }
    changed
}

/// The start of the running generation (its elapsed time ticks on the Speak screen).
pub fn clock_origin(s: &AppState) -> Option<f64> {
    match &s.speak.gen_ {
        Gen::Active(a) => Some(a.started),
        Gen::Idle => None,
    }
}

/// Window-level keys: Ctrl/Cmd+Enter generates from anywhere on the Speak screen.
pub fn on_key(s: &mut AppState, ev: &KeyEvent, _tx: &mut Tx) -> bool {
    if s.screen == Screen::Speak && ev.key == Key::Enter && ev.mods.cmd {
        generate(s);
        return true;
    }
    false
}

// -------------------------------------------------------------------------------------------------
// OS integration
// -------------------------------------------------------------------------------------------------

fn spawn(mut cmd: std::process::Command) -> Result<(), String> {
    cmd.spawn().map(|_| ()).map_err(|e| format!("Could not open it: {e}"))
}

/// Opens a file with the system's default application.
pub fn open_file(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    let mut c = std::process::Command::new("explorer");
    #[cfg(target_os = "macos")]
    let mut c = std::process::Command::new("open");
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let mut c = std::process::Command::new("xdg-open");
    c.arg(path);
    spawn(c)
}

/// Shows a file in the system file manager (selected where the platform supports it).
pub fn reveal_file(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        let mut c = std::process::Command::new("explorer");
        c.arg(format!("/select,{}", path.display()));
        spawn(c)
    }
    #[cfg(target_os = "macos")]
    {
        let mut c = std::process::Command::new("open");
        c.arg("-R").arg(path);
        spawn(c)
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        let mut c = std::process::Command::new("xdg-open");
        c.arg(path.parent().unwrap_or(path));
        spawn(c)
    }
}

// -------------------------------------------------------------------------------------------------
// View
// -------------------------------------------------------------------------------------------------

fn fmt_time(secs: f32) -> String {
    let t = secs.max(0.0).round() as u32;
    format!("{}:{:02}", t / 60, t % 60)
}

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    row().key("speak").w_full().h_full().px(16.0).pb(16.0).gap(16.0)
        .child(text_panel(s, p))
        .child(side_column(s, p, cx, |own_scroll| results_panel(s, p, own_scroll)))
}

/// The Results panel never gets shorter than this: below it, the side column scrolls instead.
const RESULTS_MIN_H: f32 = 240.0;

/// The right-hand column of Speak and Redub: Style on top, Results filling the rest. In a short window, where
/// Results would be squeezed below `RESULTS_MIN_H`, the column scrolls and Results keeps that height.
/// `results(own_scroll)` builds the Results panel: with `own_scroll` its list scrolls inside the panel; without it
/// the list grows and the whole column scrolls (one scroller, so the wheel works anywhere over the column).
pub(crate) fn side_column(s: &AppState, p: Palette, cx: &mut Cx<AppState>, results: impl Fn(bool) -> El) -> El {
    let side_h = cx.layout_with("side", |r| r[3]);
    let style_h = cx.layout_with("side-style", |r| r[3]).unwrap_or(0.0);
    let cramped = side_h.is_some_and(|h| h < style_h + 16.0 + RESULTS_MIN_H);
    let style = col().key("side-style").w_full().child(style_panel(s, p));
    let side = col().key("side").w(320.0).h_full();
    if cramped {
        side.scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0)
            .child(col().w_full().gap(16.0).child(style).child(col().key("side-results").w_full().min_h(RESULTS_MIN_H).child(results(false))))
    } else {
        side.gap(16.0).child(style).child(results(true))
    }
}

fn text_panel(s: &AppState, p: Palette) -> El {
    let words = s.speak.text.text.split_whitespace().count();
    let est = (words as f32 / 2.6).round();
    let meta = if words == 0 { String::new() } else { format!("{words} words · about {} of speech", fmt_time(est)) };
    let head = text(meta).font(ty::CAPTION).color(p.text_faint).pin(None, Some(24.0), Some(22.0), None);
    let editor = TextArea::new("speak-text", &s.speak.text, |s| &mut s.speak.text)
        .placeholder(format!("Type or paste anything in {}, then press Generate.", lang::name(&language(s))))
        .on_submit(generate)
        .build(p);
    section_panel(p, "What should I say?").key("text-panel").grow(1.0).min_w(0.0).h_full().gap(12.0).child(head).child(editor).child(action_bar(s, p))
}

fn action_bar(s: &AppState, p: Palette) -> El {
    match &s.speak.gen_ {
        Gen::Active(a) => {
            let elapsed = (s.shell.now - a.started).max(0.0) as f32;
            let k = (a.chunks + 1).min(a.total);
            let of = if a.exact { format!("{}", a.total) } else { format!("~{}", a.total) };
            let (label, sub) = if a.buffering() {
                (format!("Getting ahead · part {k} of {of}"), "Playback starts once the rest can keep up, so it never stutters.".to_owned())
            } else if a.chunks == 0 {
                (if a.total > 1 { format!("Generating part 1 of {of}") } else { "Generating…".to_owned() }, "The first words play as soon as they are ready.".to_owned())
            } else if a.chunks >= a.total {
                ("Finishing up…".to_owned(), "Almost done.".to_owned())
            } else {
                (format!("Generating part {k} of {of}"), "Playing as it arrives.".to_owned())
            };
            let total_chars: u32 = a.seg_chars.iter().sum();
            let frac = if a.seg_chars.len() == a.total && total_chars > 0 {
                Some((a.seg_chars.iter().take(a.chunks).sum::<u32>() as f32 / total_chars as f32).max(0.03))
            } else {
                (a.total > 1).then(|| (a.chunks as f32 / a.total as f32).max(0.03))
            };
            let info = col().grow(1.0).min_w(0.0).gap(8.0)
                .child(row().w_full().center_y().justify_between()
                    .child(text(label).font(ty::LABEL).color(p.text))
                    .child(text(fmt_time(elapsed)).font(ty::NUM).color(p.text_dim)))
                .child(progress_bar(p, frac))
                .child(text(sub).font(ty::CAPTION).color(p.text_faint));
            row().key("bar-active").w_full().gap(20.0).center_y()
                .child(info)
                .child(Button::new("Stop").large().icon(&icons::STOP).key("stop").on_press(cancel).build(p))
        }
        Gen::Idle => {
            let block = blocker(s);
            let empty = s.speak.text.text.trim().is_empty();
            let voice_line = match shell::active_voice(s) {
                Some(v) => col().gap(2.0)
                    .child(row().gap(8.0).center_y().child(text(format!("Speaking as {}", v.name)).font(ty::LABEL).color(p.text))
                        .child(badge(p, if v.trained { "Trained" } else { "Zero-shot" }, if v.trained { Tone::Success } else { Tone::Accent }))
                        .child(Button::new("Reference from media…").ghost().small().icon(&icons::PLUS).key("speak-ref-media").on_press(|s| {
                            shell::set_screen(s, Screen::Voices);
                            super::refmedia::pick_files(s);
                        }).build(p)))
                    .child(hint_line(s, p, block)),
                None => col().gap(2.0)
                    .child(text("No voice selected").font(ty::LABEL).color(p.text))
                    .child(text(block.unwrap_or("")).font(ty::CAPTION).color(p.warn)),
            };
            let go = Button::new("Generate").primary().large().icon(&icons::SPARKLE).key("generate").disabled(empty || block.is_some()).on_press(generate).build(p);
            let mut r = row().key("bar-idle").w_full().gap(20.0).center_y().child(el().grow(1.0).min_w(0.0).child(voice_line));
            if block.is_some() && shell::active_voice(s).is_none() && matches!(s.shell.engine, EngineStatus::Ready { .. }) {
                r = r.child(Button::new("Record").key("go-record").on_press(|s| shell::set_screen(s, Screen::Record)).build(p));
            }
            r.child(go)
        }
    }
}

/// The small line under "Speaking as": why Generate is off, "Preparing voice…" while the engine loads the voice,
/// or the keyboard hint.
fn hint_line(s: &AppState, p: Palette, block: Option<&'static str>) -> El {
    if let Some(b) = block {
        return text(b).font(ty::CAPTION).color(p.warn);
    }
    if s.shell.preparing_voice() {
        return row().key("preparing").gap(6.0).center_y()
            .child(status_dot(p, Tone::Accent, true))
            .child(text("Preparing voice…").font(ty::CAPTION).color(p.text_dim));
    }
    text("Ctrl+Enter to generate").font(ty::CAPTION).color(p.text_faint)
}

pub(crate) fn style_panel(s: &AppState, p: Palette) -> El {
    let trained = shell::active_voice(s).is_some_and(|v: &VoiceInfo| v.trained);
    let speed = Slider::new("speed", "Speed", s.speak.speed, 0.8, 1.3).step(0.05).decimals(2).unit("×").ends("Slower", "Faster")
        .on_change(|s, v| {
            s.speak.speed = (v * 100.0).round() / 100.0;
            s.speak.speed_dirty = true;
        }).build(p);
    let expr = Slider::new("expr", "Expressiveness", s.speak.expressiveness, 0.4, 1.0).step(0.05).ends("Steady", "Lively")
        .on_change(|s, v| s.speak.expressiveness = v).build(p);
    let mut m = Slider::new("match", "Voice match", s.speak.voice_match, 0.3, 1.0).step(0.05).ends("Natural", "Closest to me")
        .on_change(|s, v| s.speak.voice_match = v).build(p);
    if !trained {
        m = m.opacity(0.5);
    }
    let quality = row().key("quality").w_full().center_y().justify_between().gap(10.0)
        .child(col().gap(1.0)
            .child(text("Quality").font(ty::LABEL).color(p.text))
            .child(text(if s.speak.quality == Quality::Best { "Richest detail · 32 steps" } else { "About twice as fast · 16 steps" }).font(ty::CAPTION).color(p.text_faint)))
        .child(segmented_help(
            p,
            "quality-pick",
            &["Best", "Fast"],
            &["32 decoding steps per sentence (the first one uses 16, so sound starts sooner)", "16 decoding steps: about twice as fast, slightly less clean"],
            usize::from(s.speak.quality == Quality::Fast),
            |s, i| s.speak.quality = if i == 1 { Quality::Fast } else { Quality::Best },
        ));
    let mut c = section_panel(p, "Style").key("style-panel").w_full().gap(14.0);
    // Redub shares this panel but chooses its languages itself.
    if s.screen == Screen::Speak {
        c = c.child(language_row(s, p));
        if accent_applies(s, &language(s)) {
            c = c.child(accent_block(s, p, "speak-accent"));
        }
    }
    c = c.child(speed).child(expr).child(m).child(quality);
    if !trained && shell::active_voice(s).is_some() {
        c = c.child(text("Voice match applies once this voice is trained.").font(ty::CAPTION).color(p.text_faint).wrap());
    }
    c
}

/// "Language" with the picker chip, and the hint when it is not the language the voice was recorded in.
fn language_row(s: &AppState, p: Palette) -> El {
    let code = language(s);
    let voice = shell::active_voice(s);
    let head = row().key("speak-lang").w_full().center_y().justify_between().gap(10.0)
        .child(col().gap(1.0).min_w(0.0)
            .child(text("Language").font(ty::LABEL).color(p.text))
            .child(text("What the text is written in").font(ty::CAPTION).color(p.text_faint)))
        .child(lang::chip(&p, "speak-lang-chip", &lang::label(&code), lang::Target::Speak, voice.is_some()));
    let mut c = col().w_full().gap(8.0).child(head);
    // The engine turns "+" before a vowel into an upper-case vowel, the stress hint the speech model follows.
    if code == "uk" {
        c = c.child(text("Wrong stress? Type + before the stressed vowel: співак+ом.").font(ty::CAPTION).color(p.text_faint).wrap());
    }
    if let Some(v) = voice.filter(|v| v.lang() != code) {
        let (rec, target) = (lang::name(v.lang()), lang::name(&code));
        let has_ref = v.references.iter().any(|r| *r == code);
        if accent_applies(s, &code) {
            // The Accent choice below says what happens and offers the English clip.
            return c;
        }
        let hint = if has_ref {
            format!("Voice recorded in {rec}. Your {target} reference clip sets the accent.")
        } else {
            format!("Voice recorded in {rec}: timbre kept, a {target} reference clip improves the accent.")
        };
        let mut box_ = col().key("speak-lang-hint").w_full().gap(8.0).padding(10.0).radius(super::theme::radius::CONTROL)
            .fill(p.accent.alpha(0.07)).border(1.0, p.accent.alpha(0.2))
            .child(text(hint).font(ty::CAPTION).color(p.text_dim).wrap());
        if !has_ref {
            let c2 = code.clone();
            box_ = box_.child(row().child(Button::new(format!("Record a {target} reference")).small().icon(&icons::MIC).key("speak-lang-ref")
                .on_press(move |s| {
                    super::refclip::start_language(s, &c2);
                    shell::set_screen(s, Screen::Voices);
                }).build(p)));
        }
        c = c.child(box_);
    }
    c
}

/// The "Accent" choice (Speak's Style panel, Redub's Translate & dub to English and Re-speak in English): "Native
/// (American)" or "Like my recordings"; with Native, the native speaker (Auto / Male / Female). A trained voice gets
/// Native through its RVC model, an untrained one through the engine's zero-shot converter; an untrained voice on an
/// engine without that converter shows it disabled, with the other ways to English without an accent: install it,
/// train the voice, or add an English clip ("Add an English clip" opens Voices on it).
pub(crate) fn accent_block(s: &AppState, p: Palette, key: &str) -> El {
    let voice = shell::active_voice(s);
    let trained = voice.is_some_and(|v| v.trained);
    let possible = native_possible(s);
    // The language the voice's clip is in: detected, else the one it was recorded in.
    let rec = voice.map(|v| lang::name(v.reference_lang())).unwrap_or_default();
    let detected = voice.is_some_and(|v| v.reference_language.is_some());
    let native = possible && s.speak.native_accent;
    let pick = segmented_fill(
        p,
        &format!("{key}-pick"),
        &["Native (American)", "Like my recordings"],
        if trained {
            &["A native American speaker says it, then your trained voice model gives it your timbre", "Copies the accent of your own recordings"]
        } else if possible {
            &["A native American speaker says it, then it gets the timbre of your reference clip (no training needed)", "Copies the accent of your own recordings"]
        } else {
            &["Needs the zero-shot voice converter (run setup) or a trained voice", "Copies the accent of your own recordings"]
        },
        usize::from(!native),
        possible,
        |s, i| {
            s.speak.native_accent = i == 0;
            s.speak.accent_dirty = true;
        },
    );
    // Whose accent English gets from the voice's own clip: "your Ukrainian reference clip" once its language is
    // detected, else the recording language.
    let source = if detected { format!("your {rec} reference clip") } else { format!("your {rec} recordings") };
    let caption = if native && trained {
        "A native American speaker says it; your trained voice model keeps your timbre.".to_owned()
    } else if native {
        "Native accent from your reference clip (no training needed); training makes it closer to you.".to_owned()
    } else {
        format!("English gets the accent of {source}.")
    };
    let mut c = col().key(key.to_owned()).w_full().gap(8.0)
        .child(text("Accent").font(ty::LABEL).color(p.text))
        .child(pick)
        .child(text(caption).font(ty::CAPTION).color(p.text_faint).wrap());
    if !native {
        let hint = if possible {
            "Pick Native, or add an English reference clip in Voices."
        } else {
            "Run setup to install the zero-shot voice converter, or train this voice, to speak English without an accent (Native accent); or add an English reference clip in Voices."
        };
        c = c.child(col().key(format!("{key}-hint")).w_full().gap(8.0).padding(10.0).radius(theme::radius::CONTROL)
            .fill(p.warn.alpha(0.08)).border(1.0, p.warn.alpha(0.25))
            .child(text(hint).font(ty::CAPTION).color(if possible { p.text_dim } else { p.warn }).wrap())
            .child(row().gap(8.0)
                .child(Button::new("Add an English clip").small().icon(&icons::MIC).key(format!("{key}-add-en"))
                    .on_press(|s| add_reference(s, "en")).build(p))));
    }
    if native {
        let v = s.speak.native_voice;
        let sel = match v {
            NativeVoice::Auto => 0,
            NativeVoice::Male => 1,
            NativeVoice::Female => 2,
        };
        let pick = segmented_fill(
            p,
            &format!("{key}-speaker-pick"),
            &["Auto", "Male", "Female"],
            &["The native speaker whose pitch is closest to yours", "Always the male native speaker", "Always the female native speaker"],
            sel,
            true,
            |s, i| {
                s.speak.native_voice = [NativeVoice::Auto, NativeVoice::Male, NativeVoice::Female][i.min(2)];
                s.speak.accent_dirty = true;
            },
        );
        c = c.child(row().key(format!("{key}-speaker")).w_full().center_y().gap(10.0)
            .child(text("Speaker").font(ty::LABEL).color(p.text))
            .child(el().grow(1.0).min_w(0.0).child(pick)));
    }
    c
}

/// The right end of a Results header (Speak and Redub): the count badge, when there are results, and the "Save as"
/// chip. Pinned beside the title, so the header is as tall as Style's.
pub(crate) fn results_head(s: &AppState, p: Palette, count: usize, chip_key: &str) -> El {
    row().key(format!("{chip_key}-head")).pin(None, Some(16.5), Some(22.0), None).gap(8.0).center_y()
        .children((count > 0).then(|| badge(p, &format!("{count}"), Tone::Neutral)))
        .child(super::formats::output_chip(s, p, chip_key))
}

fn results_panel(s: &AppState, p: Palette, own_scroll: bool) -> El {
    let head = results_head(s, p, s.speak.takes.len(), "speak-outfmt");
    let body = if s.speak.takes.is_empty() {
        col().key("takes-empty").w_full().grow(1.0).justify_center().child(empty_state(p, &icons::WAVE, "Nothing yet", "Generated speech appears here."))
    } else {
        let list = col().w_full().gap(8.0).children(s.speak.takes.iter().map(|t| take_row(s, p, t)));
        if own_scroll {
            col().key("takes").w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0).child(list)
        } else {
            col().key("takes").w_full().child(list)
        }
    };
    section_panel(p, "Results").key("results-panel").w_full().grow(1.0).min_h(0.0).gap(10.0).child(head).child(body)
}

/// A take's text as one short line for its card: line breaks would make the truncated label as tall as the paragraph.
fn one_line(text: &str) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(140) {
        Some((i, _)) => format!("{}…", &flat[..i]),
        None => flat,
    }
}

fn take_row(s: &AppState, p: Palette, t: &Take) -> El {
    let id = t.id;
    let on = playbar::playing(s, &t.play);
    let play = el().key(format!("play{id}")).role("button").label(if on { "Pause" } else { "Play" }).size(36.0, 36.0).radius(18.0).center()
        .hover(rune::Color::WHITE.alpha(0.1)).focusable(true).motion(motion::UI)
        .on_press(move |s: &mut AppState| toggle_take(s, id))
        .child(icon_el(if on { &icons::PAUSE } else { &icons::PLAY }, 15.0, p.on_accent));
    let play = accent_glass(p, play);
    let path = t.path.clone();
    let path2 = t.path.clone();
    let reveal = icon_button(p, &format!("reveal{id}"), &icons::FOLDER, "Show in folder", move |s| {
        if let Err(e) = reveal_file(&path) {
            shell::toast(s, Tone::Danger, e);
        }
    });
    let open = icon_button(p, &format!("open{id}"), &icons::OPEN, "Open file", move |s| {
        if let Err(e) = open_file(&path2) {
            shell::toast(s, Tone::Danger, e);
        }
    });
    let meta = format!("{} · made in {:.1} s", fmt_time(t.duration_s), t.elapsed_s);
    let export = super::formats::export_button(p, format!("export-take{id}"), &t.path, &t.play, true);
    glass_card(p).key(format!("take{id}")).w_full().padding(10.0).gap(10.0)
        .child(row().w_full().gap(10.0).center_y()
            .child(play)
            .child(col().grow(1.0).min_w(0.0).gap(1.0)
                .child(text(one_line(&t.text)).font(ty::LABEL).color(p.text).truncate())
                .child(text(meta).font(ty::CAPTION).color(p.text_faint))))
        .child(if trash::confirming(s, trash::Kind::Speak, id) {
            trash::confirm_row(s, p, trash::Kind::Speak, id, &t.path, &t.play)
        } else {
            row().w_full().gap(8.0).center_y().child(export).child(reveal).child(open).child(trash::button(p, trash::Kind::Speak, id))
        })
}

// -------------------------------------------------------------------------------------------------
// Headless scenes
// -------------------------------------------------------------------------------------------------

const SAMPLE: &str = "Welcome back. In a few seconds this paragraph will be read in my own voice, one sentence after another, \
starting to play before the last sentence has even been generated.";

fn demo_voice(s: &mut AppState, trained: bool) {
    s.screen = Screen::Speak;
    s.shell.voices = vec![
        VoiceInfo { id: "me".into(), name: "My voice".into(), preset: Some("standard".into()), dataset_minutes: 6.4, parts: 5, trained, trained_epochs: if trained { 300 } else { 0 }, ..Default::default() },
        VoiceInfo { id: "pro".into(), name: "Studio take".into(), preset: Some("pro".into()), dataset_minutes: 15.0, parts: 12, trained: true, trained_epochs: 300, ..Default::default() },
    ];
    s.shell.active_voice = Some("me".into());
    s.shell.set_engine(EngineStatus::Ready { device: Some("NVIDIA GeForce RTX 4070".into()), tts: Some("ready".into()), rvc: Some("ready".into()) });
}

fn demo_takes(s: &mut AppState) {
    let mk = |id, text: &str, d, e| Take { id, text: text.into(), voice: "My voice".into(), path: format!("/tmp/out{id}.wav").into(), play: format!("/tmp/out{id}.wav").into(), duration_s: d, elapsed_s: e };
    s.speak.takes = vec![
        mk(3, "Welcome back. In a few seconds this paragraph will be read in my own voice.\n\nA second paragraph follows after an empty line, and a third one after it.\n\nThat is all.", 14.2, 9.1),
        mk(2, "The quick brown fox jumps over the lazy dog.", 3.8, 2.6),
        mk(1, "Hello, this is a test of my cloned voice.", 3.1, 2.2),
    ];
}

pub fn scene_idle(s: &mut AppState) {
    demo_voice(s, true);
}

pub fn scene_typed(s: &mut AppState) {
    demo_voice(s, true);
    s.speak.text = TextEdit::new(SAMPLE);
}

const SAMPLE_UK: &str = "Привіт! За хвилину цей абзац прозвучить моїм власним голосом, речення за реченням, і почне грати ще до того, як буде готове останнє.";

/// Speaking Ukrainian with a voice recorded in English: the language chip and the reference hint.
pub fn scene_language_uk(s: &mut AppState) {
    demo_voice(s, true);
    s.speak.text = TextEdit::new(SAMPLE_UK);
    s.speak.languages.insert("me".into(), "uk".into());
}

/// A voice recorded in Ukrainian, speaking English: the Accent choice, native by default.
fn demo_voice_uk(s: &mut AppState, trained: bool) {
    demo_voice(s, trained);
    for v in &mut s.shell.voices {
        v.language = "uk".into();
    }
    s.speak.languages.insert("me".into(), "en".into());
    s.speak.text = TextEdit::new(SAMPLE);
}

pub fn scene_accent_native(s: &mut AppState) {
    demo_voice_uk(s, true);
    demo_takes(s);
}

/// "Like my recordings" chosen: the language hint about an English reference clip is back.
pub fn scene_accent_own(s: &mut AppState) {
    demo_voice_uk(s, true);
    s.speak.native_accent = false;
}

/// An untrained voice on an engine without the zero-shot converter: the choice is disabled with the hint to install
/// it or train the voice.
pub fn scene_accent_untrained(s: &mut AppState) {
    demo_voice_uk(s, false);
}

/// An untrained voice with the zero-shot converter installed: Native is on by default, from the reference clip.
pub fn scene_accent_zero_shot(s: &mut AppState) {
    demo_voice_uk(s, false);
    s.shell.zero_shot_vc = true;
    demo_takes(s);
}

/// The owner's case: a voice set to English whose reference clip was detected as Ukrainian speech, not trained. The
/// Accent choice shows (English would carry the clip's accent), disabled, with "Add an English clip".
pub fn scene_accent_foreign_ref(s: &mut AppState) {
    demo_voice(s, false);
    s.shell.voices[0].reference_language = Some("uk".into());
    s.shell.voices[0].references = vec!["uk".into()];
    s.speak.text = TextEdit::new(SAMPLE);
}

/// The owner's case with the zero-shot converter installed: Native is on without training.
pub fn scene_accent_foreign_ref_zero_shot(s: &mut AppState) {
    scene_accent_foreign_ref(s);
    s.shell.zero_shot_vc = true;
}

/// The same voice trained: Native is on.
pub fn scene_accent_foreign_ref_trained(s: &mut AppState) {
    scene_accent_foreign_ref(s);
    s.shell.voices[0].trained = true;
    s.shell.voices[0].trained_epochs = 300;
}

/// The output-language picker open.
pub fn scene_language_picker(s: &mut AppState) {
    scene_language_uk(s);
    s.langpick.open = Some((lang::Target::Speak, "speak-lang-chip".into()));
}

pub fn scene_generating(s: &mut AppState) {
    demo_voice(s, true);
    s.speak.text = TextEdit::new(SAMPLE);
    demo_takes(s);
    s.speak.takes.truncate(2);
    let mut a = Active::new(1, -6.6, SAMPLE.into(), "My voice".into());
    a.job = Some("job1".into());
    (a.chunks, a.total, a.exact, a.audible, a.began, a.audio_s) = (1, 3, true, true, true, 9.0);
    a.seg_chars = vec![80, 70, 60];
    s.speak.gen_ = Gen::Active(a);
    super::playbar::scene_live(s, &one_line(SAMPLE), 3.4, 9.0);
}

/// A long job whose first segments are held back for a smooth start.
pub fn scene_buffering(s: &mut AppState) {
    demo_voice(s, true);
    s.speak.text = TextEdit::new(SAMPLE);
    demo_takes(s);
    s.speak.takes.truncate(2);
    let mut a = Active::new(1, -4.2, SAMPLE.into(), "My voice".into());
    a.job = Some("job1".into());
    (a.chunks, a.total, a.exact, a.audio_s, a.rtf) = (1, 4, true, 8.0, 1.2);
    a.seg_chars = vec![90, 80, 85, 70];
    a.held = vec!["/tmp/c0.wav".into()];
    s.speak.gen_ = Gen::Active(a);
}

/// The active voice is still being prepared by the engine (`voices.warm`).
pub fn scene_preparing(s: &mut AppState) {
    demo_voice(s, true);
    s.speak.text = TextEdit::new(SAMPLE);
    s.shell.preparing = Some("me".into());
}

pub fn scene_done(s: &mut AppState) {
    demo_voice(s, true);
    s.speak.text = TextEdit::new(SAMPLE);
    demo_takes(s);
}

/// "Delete this result?" on the second take, while the first one plays in the player bar.
pub fn scene_trash_confirm(s: &mut AppState) {
    scene_done(s);
    trash::scene_confirm(s, trash::Kind::Speak, 2);
    let t = &s.speak.takes[0];
    let (path, title) = (t.play.to_string_lossy().into_owned(), one_line(&t.text));
    playbar::scene_now(s, &path, &title, "Speak · result", playbar::Source::Speak, playbar::Status::Playing, 4.0, 14.2);
}

/// A long warning toast wraps instead of being cut off.
pub fn scene_toast_long(s: &mut AppState) {
    scene_done(s);
    shell::toast(s, Tone::Warning, "The reference clip is Ukrainian speech, so the English carries a Ukrainian accent. Add an English clip under Other languages in Voices, or choose Native (American) in Accent.");
}

/// The shared "Export as..." menu open on a take.
pub fn scene_export_menu(s: &mut AppState) {
    demo_voice(s, true);
    s.formats.scene_ready();
    s.speak.text = TextEdit::new(SAMPLE);
    demo_takes(s);
    s.formats.menu = Some(super::formats::Menu { anchor: "export-take3".into(), src: "/tmp/out3.wav".into(), audio: "/tmp/out3.wav".into() });
    s.formats.format = "opus".into();
}

/// Results saved as MP3, with the "Save results as" menu open.
pub fn scene_output_menu(s: &mut AppState) {
    demo_voice(s, true);
    s.formats.scene_ready();
    s.speak.text = TextEdit::new(SAMPLE);
    demo_takes(s);
    s.formats.output = "mp3".into();
    s.formats.output_menu = Some("speak-outfmt".into());
}

pub fn scene_no_voice(s: &mut AppState) {
    s.screen = Screen::Speak;
    s.shell.set_engine(EngineStatus::Ready { device: None, tts: Some("ready".into()), rvc: Some("ready".into()) });
}

pub fn scene_engine_starting(s: &mut AppState) {
    s.screen = Screen::Speak;
    s.speak.text = TextEdit::new("Hello there.");
    s.shell.set_engine(EngineStatus::Starting { since: 0.0 });
}

pub fn scene_engine_stopped(s: &mut AppState) {
    s.screen = Screen::Speak;
    s.speak.text = TextEdit::new("Hello there.");
    s.shell.set_engine(EngineStatus::Stopped { code: Some(1), detail: "CUDA out of memory while loading the speech model.".into() });
}
