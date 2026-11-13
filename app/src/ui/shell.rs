//! App chrome: glass background, header (title, step tabs, voice selector, engine status), screen switching,
//! toasts, and the event routing.
//!
//! ## Event convention (for screen authors)
//!
//! The UI is event-driven (see `ui::events`): nothing polls the engine, the audio devices or the file system, and
//! an idle window draws nothing. Every screen module exposes these hooks:
//!
//! * `pub fn on_engine(s: &mut AppState, msg: &EngineMsg)` is called for **every** message the engine sends
//!   (responses, events, exit) as it arrives, on every screen, whether or not that screen is visible. Match your own
//!   request ids (store the `u64` returned by [`request`]) and event names (`tts.*`, `train.*`, ...).
//! * `pub fn settle(s: &mut AppState) -> bool` runs after every event (input, an engine message, an audio or dialog
//!   event, a timer): bring derived state up to date (send a request that waited for the engine, follow the
//!   header's voice) and return true when that changed what the screen shows. It must only look at `AppState`:
//!   no I/O to find out whether something happened (that is an event), and no timing.
//! * Background work (a dialog, a disk probe, a subprocess) runs on its own thread and reports with
//!   `events::post(|s| ...)`; playback state lives in `ui::playbar` (screens read [`super::playbar::playing`]), the
//!   recorder reports through its meter events.
//! * A clock that must tick while a job runs (elapsed time) is a deadline: return the job's start time from the
//!   screen's `clock_origin`, and [`settle`] arms a timer for the next whole second while the screen shows it.
//!
//! The shell owns the engine and the player. Never call `EngineClient`/`Player` methods directly: use the guarded
//! helpers [`request`], `playbar::play` / `toggle` / `play_live` / `enqueue` and [`stop_audio`], which never panic
//! and return `None`/`Err` with a friendly message when a backend is unavailable.
//!
//! Screens return a *transparent* element that fills the body (`.w_full().h_full()`); the shell paints the
//! background and header.

use super::components::{Button, Tone, accent_glass, badge, glass_card, glass_edge, icon_el, icons, status_dot, toast_stack};
use super::theme::{self, Palette, motion, radius, ty};
use crate::audio::Player;
use crate::engine::protocol::PausedTraining;
use crate::engine::{EngineClient, EngineConfig, EngineMsg};
use crate::state::{AppState, Screen};
use rune::{Align, ChromeInsets, Color, Cx, El, Side, Tx, col, el, row, text};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::time::{Duration, Instant};

pub use super::components::Toast;

// -------------------------------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------------------------------

/// What the engine is doing, as the header shows it.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum EngineStatus {
    /// Not started yet (started a moment after launch, see `settle`).
    #[default]
    Idle,
    /// Process spawned, waiting for its `ready` line.
    Starting { since: f64 },
    /// Answering requests. `models` are the `hello` model states when the engine reports them.
    Ready { device: Option<String>, tts: Option<String>, rvc: Option<String> },
    /// The process ended or could not be started.
    Stopped { code: Option<i32>, detail: String },
    /// No engine can be started here (not installed, or this build has no engine client).
    Unavailable { reason: String },
}

/// A voice as listed by `voices.list`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub preset: Option<String>,
    pub dataset_minutes: f32,
    pub parts: u32,
    pub trained: bool,
    pub trained_epochs: u32,
    /// A paused (or interrupted) training of this voice that can be resumed.
    pub paused: Option<PausedTraining>,
    /// The language the voice was recorded in (`en` for voices from before languages existed).
    pub language: String,
    /// Languages with their own reference clip (`reference_<lang>.wav`).
    pub references: Vec<String>,
    /// The language actually spoken in `reference.wav`, detected by the engine (`reference_language`); `None` until the
    /// speech worker has transcribed the clip (its next warm-up or generation).
    pub reference_language: Option<String>,
}

impl VoiceInfo {
    /// The recording language (`en` when unknown).
    pub fn lang(&self) -> &str {
        if self.language.is_empty() { "en" } else { &self.language }
    }

    /// The language of `reference.wav` as far as it is known: the detected one, else the recording language.
    pub fn reference_lang(&self) -> &str {
        self.reference_language.as_deref().filter(|l| !l.is_empty()).unwrap_or(self.lang())
    }

    /// `reference.wav` is known to be speech in another language than the voice's own (its accent carries over).
    pub fn reference_mismatch(&self) -> bool {
        self.reference_language.as_deref().is_some_and(|l| !l.is_empty() && super::lang::base(l) != super::lang::base(self.lang()))
    }

    /// Speech in `code` is spoken from a clip in that language: its own `reference_<code>.wav`, or `reference.wav`
    /// when that clip is in `code` (detected, or the recording language while the detection is not known yet).
    pub fn has_reference_in(&self, code: &str) -> bool {
        let code = super::lang::base(code);
        self.references.iter().any(|r| super::lang::base(r) == code) || super::lang::base(self.reference_lang()) == code
    }

    pub fn from_json(v: &Value) -> Option<VoiceInfo> {
        Some(VoiceInfo {
            id: v.get("id")?.as_str()?.to_owned(),
            name: v.get("name").and_then(Value::as_str).unwrap_or("Untitled").to_owned(),
            preset: v.get("preset").and_then(Value::as_str).map(str::to_owned),
            dataset_minutes: v.get("dataset_minutes").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            parts: v.get("parts").and_then(Value::as_u64).unwrap_or(0) as u32,
            trained: v.get("trained").and_then(Value::as_bool).unwrap_or(false),
            trained_epochs: v.get("trained_epochs").and_then(Value::as_u64).unwrap_or(0) as u32,
            paused: v.get("training_paused").filter(|p| p.is_object()).and_then(|p| serde_json::from_value(p.clone()).ok()),
            language: v.get("language").and_then(Value::as_str).filter(|l| !l.is_empty()).unwrap_or("en").to_owned(),
            references: v.get("references").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
            reference_language: v.get("reference_language").and_then(Value::as_str).filter(|l| !l.is_empty()).map(str::to_owned),
        })
    }
}

pub struct ShellState {
    /// Seconds on the app clock (Rune's `Tx::clock`) at the last event. Views read it; code that stamps a start time
    /// outside an event (a press handler) uses [`clock_now`].
    pub now: f64,
    /// The wall-clock instant of app clock 0 (windowed runs; headless runs keep a virtual clock).
    epoch: Option<Instant>,
    /// When the elapsed-time clock of a running job shows its next second (see [`settle`]).
    clock_due: Option<f64>,
    pub engine: EngineStatus,
    /// The engine runs on a CUDA GPU (`hello` -> `device.backend`); sets defaults such as "Remove background music".
    pub gpu: bool,
    /// Voices known to the engine (refreshed by [`refresh_voices`]).
    pub voices: Vec<VoiceInfo>,
    /// Languages the speech model speaks (`hello` -> `languages`; empty until the engine says, see `ui::lang`).
    pub languages: Vec<crate::engine::protocol::SpeechLanguage>,
    /// Redub translators installed (`hello` -> `translators`: `"nllb-1.3b"`, `"nllb-600m"`; empty until the engine says).
    pub translators: Vec<String>,
    /// The engine has the zero-shot voice converter (`hello` -> `zero_shot_vc`): the native accent works for an
    /// untrained voice too.
    pub zero_shot_vc: bool,
    /// The voice used by Speak and shown in the header pill.
    pub active_voice: Option<String>,
    pub voice_menu_open: bool,
    /// Headless previews only: the caption buttons' state to draw as if a Windows window reported it
    /// (`caption::scenes`). `None` in the real app.
    pub caption_preview: Option<rune::CaptionView>,
    /// The app icon, decoded once in `setup` (see [`setup_icon`]); the brand mark draws it.
    pub icon: Option<rune::Image>,
    icon_cache: Option<rune::ImageCache>,
    pub toasts: Vec<Toast>,
    /// Why audio is unavailable (shown on screens that play sound).
    pub audio_error: Option<String>,
    /// The per-user data dir (voices, temporary takes, settings): the engine's `--data-dir`.
    pub data_dir: PathBuf,
    /// A voice that `voices.warm` is still preparing (until its `voice.warm` event arrives).
    pub preparing: Option<String>,
    /// The voice `voices.warm` was last requested for (so a voice is warmed once per engine start).
    warm_for: Option<String>,
    warm_req: Option<u64>,
    next_toast: u64,
    /// Headless snapshot / script run: never start the engine or audio.
    scripted: bool,
    voices_req: Option<u64>,
    first_voice_list: bool,
}

impl ShellState {
    pub fn dismiss_toast(&mut self, id: u64) {
        self.toasts.retain(|t| t.id != id);
    }
    pub fn scripted(&self) -> bool {
        self.scripted
    }
    /// Test/scene helper: pretend the engine is in this state.
    pub fn set_engine(&mut self, status: EngineStatus) {
        self.engine = status;
    }
    /// Whether the active voice is still being prepared by the engine (`voices.warm`).
    pub fn preparing_voice(&self) -> bool {
        self.preparing.is_some() && self.preparing == self.active_voice
    }
}

/// `AppSpec::setup`: decodes the app icon (a 512 px PNG; Rune keeps filtered mip levels, so it stays crisp at any
/// size it is drawn at). The decode takes milliseconds and finishes before the first frame, so the mark never pops in.
pub fn setup_icon(s: &mut AppState, waker: rune::Waker) {
    let cache = rune::ImageCache::new(waker);
    let icon = cache.load(crate::ICON_PNG);
    let _ = icon.wait();
    s.shell.icon = Some(icon);
    s.shell.icon_cache = Some(cache);
}

/// The brand mark: the app icon drawn `size` pt square (the header's, and the welcome card's large one).
pub fn brand_mark(s: &AppState, size: f32) -> El {
    let mark = el().key("brand-mark").size(size, size).a11y_hidden();
    match &s.shell.icon {
        Some(icon) => mark.pixels(icon.pixels()),
        None => mark,
    }
}

/// Headless runs (`--snap`, `--tree`, `--script`, ...) must not start processes or open audio devices. A
/// `--realtime` run is the exception: it keeps pace with the wall clock so it can record a real session (the engine
/// starts and work really runs), for demo videos made with `--script` and `--video`.
pub fn headless_run() -> bool {
    let mut args = std::env::args();
    let headless = args.any(|a| matches!(a.as_str(), "--snap" | "--tree" | "--script" | "--list-scenes" | "--trace" | "--video"));
    headless && !std::env::args().any(|a| a == "--realtime")
}

impl Default for ShellState {
    fn default() -> Self {
        ShellState {
            now: 0.0,
            epoch: None,
            clock_due: None,
            engine: EngineStatus::Idle,
            voices: Vec::new(),
            languages: Vec::new(),
            translators: Vec::new(),
            zero_shot_vc: false,
            active_voice: None,
            voice_menu_open: false,
            caption_preview: None,
            icon: None,
            icon_cache: None,
            toasts: Vec::new(),
            audio_error: None,
            data_dir: crate::engine::default_data_dir().unwrap_or_else(std::env::temp_dir),
            preparing: None,
            warm_for: None,
            warm_req: None,
            next_toast: 0,
            scripted: headless_run(),
            voices_req: None,
            first_voice_list: true,
            gpu: false,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Public helpers for screens
// -------------------------------------------------------------------------------------------------

/// The app clock now: the wall clock in windowed runs (Rune's clock is wall time there), the last event's clock in
/// headless runs. For start times stamped in press handlers, which get no `Tx`.
pub fn clock_now(s: &AppState) -> f64 {
    match s.shell.epoch {
        Some(e) => e.elapsed().as_secs_f64().max(s.shell.now),
        None => s.shell.now,
    }
}

/// Shows a toast for about 4 seconds (errors 7 seconds).
pub fn toast(s: &mut AppState, tone: Tone, text: impl Into<String>) {
    let now = clock_now(s);
    let sh = &mut s.shell;
    sh.next_toast += 1;
    let text = text.into();
    // Time to read it: longer messages stay longer (about 15 characters a second on top of the base time).
    let base = if tone == Tone::Danger { 7.0 } else { 4.0 };
    let life = (base + text.chars().count() as f64 / 15.0).min(14.0);
    sh.toasts.push(Toast { id: sh.next_toast, tone, text, expires: now + life });
    if sh.toasts.len() > 3 {
        sh.toasts.remove(0);
    }
}

/// Whether requests can be sent right now.
pub fn engine_ready(s: &AppState) -> bool {
    s.engine.is_some() && matches!(s.shell.engine, EngineStatus::Ready { .. })
}

/// Runs `f`, turning a panic (an unimplemented backend) into an error string.
pub fn guard<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).map_err(|p| {
        p.downcast_ref::<&str>().map(|m| (*m).to_owned()).or_else(|| p.downcast_ref::<String>().cloned()).unwrap_or_else(|| "internal error".into())
    })
}

/// Sends an engine request. Returns its id (the answer comes to `on_engine` as `EngineMsg::Response`),
/// or `None` when the engine is not running.
pub fn request(s: &mut AppState, method: &str, params: Value) -> Option<u64> {
    let client = s.engine.as_ref()?;
    match guard(|| client.request(method, params)) {
        Ok(id) => Some(id),
        Err(_) => {
            s.engine = None;
            s.shell.engine = EngineStatus::Stopped { code: None, detail: "The engine connection failed.".into() };
            None
        }
    }
}

/// Re-reads the voice list from the engine; the shell updates `shell.voices` and `shell.active_voice`.
pub fn refresh_voices(s: &mut AppState) {
    s.shell.voices_req = request(s, "voices.list", json!({}));
}

/// The voice currently selected in the header.
pub fn active_voice(s: &AppState) -> Option<&VoiceInfo> {
    let id = s.shell.active_voice.as_deref()?;
    s.shell.voices.iter().find(|v| v.id == id)
}

/// Switches screen. First-run onboarding (consent, runtime setup) cannot be skipped: until it is done, every
/// request to leave it keeps the user there.
pub fn set_screen(s: &mut AppState, screen: Screen) {
    s.screen = if screen != Screen::Onboarding && !s.shell.scripted && super::onboarding::needed(s) { Screen::Onboarding } else { screen };
    s.shell.voice_menu_open = false;
}

/// Stops whatever plays and hides the player bar.
pub fn stop_audio(s: &mut AppState) {
    super::playbar::stop(s);
}

pub fn audio_unavailable(s: &AppState) -> String {
    s.shell.audio_error.clone().unwrap_or_else(|| "No audio output is available.".into())
}

/// Starts (or restarts) the engine. Called once a moment after launch (see [`settle`]) and by the Retry buttons.
/// Its messages arrive as events ([`on_engine_msg`]); a replaced engine's late messages are dropped there.
pub fn restart_engine(s: &mut AppState) {
    if let Some(mut old) = s.engine.take() {
        let _ = guard(|| old.shutdown());
    }
    s.shell.engine = EngineStatus::Starting { since: clock_now(s) };
    let cfg = match guard(EngineConfig::discover) {
        Ok(Ok(cfg)) => cfg,
        Ok(Err(e)) => {
            s.shell.engine = EngineStatus::Unavailable { reason: format!("{e:#}") };
            return;
        }
        Err(_) => {
            s.shell.engine = EngineStatus::Unavailable { reason: "This build has no engine client yet.".into() };
            return;
        }
    };
    s.shell.data_dir = cfg.data_dir.clone();
    s.record.data_dir = cfg.data_dir.clone();
    let sink: crate::engine::EngineSink = std::sync::Arc::new(|client, msg| super::events::post(move |s| on_engine_msg(s, client, msg)));
    match guard(|| EngineClient::spawn(&cfg, sink)) {
        Ok(Ok(c)) => s.engine = Some(c),
        Ok(Err(e)) => s.shell.engine = EngineStatus::Stopped { code: None, detail: format!("{e:#}") },
        Err(_) => s.shell.engine = EngineStatus::Unavailable { reason: "This build has no engine client yet.".into() },
    }
}

/// Drops the audio output and opens the default device again: after the PC slept, or the output device changed,
/// the old stream stays open but never asks for sound (nothing plays, the position stays at 0).
pub fn restart_audio(s: &mut AppState) {
    s.player = None;
    s.shell.audio_error = None;
    start_audio(s);
}

fn start_audio(s: &mut AppState) {
    if s.player.is_some() {
        return;
    }
    match guard(|| Player::new(Some(super::playbar::sink()))) {
        Ok(Ok(p)) => s.player = Some(p),
        Ok(Err(e)) => s.shell.audio_error = Some(format!("No audio output: {e}")),
        Err(_) => s.shell.audio_error = Some("Audio is not available in this build.".into()),
    }
}

// -------------------------------------------------------------------------------------------------
// Events, settling and timers
// -------------------------------------------------------------------------------------------------

/// Start the engine this long after launch, so the first frames stay still.
const ENGINE_START_DELAY: f64 = 0.25;

/// The frame hook (`AppSpec::poll`). Rune runs it at the start of a frame, which only happens for input, a running
/// animation, an event (`AppHandle::update`) or a timer that came due (`Tx::wake_at`, armed by [`settle`]); an idle
/// window never runs it. It only settles the state: the engine, the audio and the dialogs report through events.
pub fn on_frame(s: &mut AppState, clock: f64, tx: &mut Tx) -> bool {
    super::stats::count(&super::stats::HOOKS);
    if let Some(t) = super::stats::quit_at() {
        if clock >= t {
            super::stats::write();
            tx.quit();
            return false;
        }
        tx.wake_at(t);
    }
    if s.shell.epoch.is_none() && !s.shell.scripted {
        s.shell.epoch = Instant::now().checked_sub(Duration::from_secs_f64(clock.max(0.0)));
    }
    settle(s, tx)
}

/// Runs after every event: screens bring derived state up to date, a quit or toast expiry is carried out, and the
/// timers the state needs are armed. True when what the window shows changed.
pub fn settle(s: &mut AppState, tx: &mut Tx) -> bool {
    let now = tx.clock().max(s.shell.now);
    s.shell.now = now;
    let mut changed = false;

    if !s.shell.scripted && matches!(s.shell.engine, EngineStatus::Idle) {
        if now < ENGINE_START_DELAY {
            tx.wake_at(ENGINE_START_DELAY);
        } else {
            start_audio(s);
            restart_engine(s);
            changed = true;
        }
    }

    changed |= super::onboarding::settle(s);
    changed |= super::voices::settle(s);
    changed |= super::record::settle(s);
    changed |= super::train::settle(s);
    changed |= super::speak::settle(s);
    changed |= super::book::settle(s);
    changed |= super::redub::settle(s);
    changed |= super::history::settle(s);
    changed |= super::formats::settle(s);
    if super::train::take_quit(s, tx) {
        return true;
    }
    changed |= ensure_warm(s);
    changed |= super::playbar::settle(s, tx);

    // Toasts: drop the expired ones, wake for the next expiry.
    let before = s.shell.toasts.len();
    s.shell.toasts.retain(|t| t.expires > now);
    changed |= s.shell.toasts.len() != before;
    if let Some(next) = s.shell.toasts.iter().map(|t| t.expires).reduce(f64::min) {
        tx.wake_at(next);
    }

    // The elapsed-time clock of a job on the visible screen: one redraw per shown second while it runs.
    if s.shell.clock_due.is_some_and(|due| due <= now + 1e-3) {
        super::stats::count(&super::stats::TIMERS);
        s.shell.clock_due = None;
        changed = true;
    }
    match clock_origin(s) {
        Some(origin) => {
            // `fmt_time` rounds, so the shown second changes at k + 0.5 s.
            let k = (now - origin - 0.5).floor() + 1.0;
            let due = origin + k.max(0.0) + 0.5;
            s.shell.clock_due = Some(due);
            tx.wake_at(due);
        }
        None => s.shell.clock_due = None,
    }

    // Other deadlines screens keep (a debounce).
    if let Some(due) = super::onboarding::next_due(s) {
        tx.wake_at(due);
    }
    changed
}

/// The start (app clock) of the job whose elapsed time the visible screen shows, while one runs.
fn clock_origin(s: &AppState) -> Option<f64> {
    match s.screen {
        Screen::Speak => super::speak::clock_origin(s),
        Screen::Redub => super::redub::clock_origin(s),
        Screen::Train => super::train::clock_origin(s),
        _ => None,
    }
}

/// An engine message (event from the client's reader thread). Messages of an engine that was stopped or replaced
/// meanwhile are dropped.
fn on_engine_msg(s: &mut AppState, client: u64, m: EngineMsg) {
    if s.engine.as_ref().map(EngineClient::id) != Some(client) {
        return;
    }
    on_engine(s, &m);
    super::onboarding::on_engine(s, &m);
    super::voices::on_engine(s, &m);
    super::record::on_engine(s, &m);
    super::train::on_engine(s, &m);
    super::speak::on_engine(s, &m);
    super::book::on_engine(s, &m);
    super::redub::on_engine(s, &m);
    super::formats::on_engine(s, &m);
    super::trash::on_engine(s, &m);
    super::history::on_engine(s, &m);
}

/// The shell's own share of engine messages: engine status, the voice list and voice warm-up.
pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Ready(hello) => {
            let device = (!hello.device.name.is_empty()).then(|| hello.device.name.clone());
            s.shell.gpu = hello.device.backend.eq_ignore_ascii_case("cuda");
            s.shell.engine = EngineStatus::Ready { device, tts: Some(hello.models.tts.as_str().to_owned()), rvc: Some(hello.models.rvc.as_str().to_owned()) };
            if !hello.languages.is_empty() {
                s.shell.languages = hello.languages.clone();
            }
            s.shell.translators = hello.translators.clone();
            s.shell.zero_shot_vc = hello.zero_shot_vc;
            reset_warm(s);
            refresh_voices(s);
        }
        EngineMsg::Response { id, result } if s.shell.voices_req == Some(*id) => {
            s.shell.voices_req = None;
            if let Ok(v) = result {
                let list: Vec<VoiceInfo> =
                    v.get("voices").and_then(Value::as_array).map(|a| a.iter().filter_map(VoiceInfo::from_json).collect()).unwrap_or_default();
                apply_voice_list(s, list);
            }
        }
        // Model status changes after `ready` (load finished, GPU handed over to training and back).
        EngineMsg::Event(ev) if ev.event == "models" => {
            let get = |k: &str| ev.data.get(k).and_then(Value::as_str).map(str::to_owned);
            let (new_tts, new_rvc) = (get("tts"), get("rvc"));
            if let EngineStatus::Ready { tts, rvc, .. } = &mut s.shell.engine {
                let was_ready = tts.as_deref() == Some("ready") && rvc.as_deref() == Some("ready");
                if new_tts.is_some() {
                    *tts = new_tts;
                }
                if new_rvc.is_some() {
                    *rvc = new_rvc;
                }
                let now_ready = tts.as_deref() == Some("ready") && rvc.as_deref() == Some("ready");
                if now_ready && !was_ready {
                    // Fresh models lost any warm state: warm the active voice again.
                    reset_warm(s);
                }
            }
        }
        // `voices.warm` answered: `{warming: false, reason}` means there is nothing to wait for.
        EngineMsg::Response { id, result } if s.shell.warm_req == Some(*id) => {
            s.shell.warm_req = None;
            let warming = result.as_ref().ok().and_then(|v| v.get("warming")).and_then(Value::as_bool).unwrap_or(false);
            if !warming {
                s.shell.preparing = None;
            }
        }
        EngineMsg::Event(ev) if ev.event == "voice.warm" => {
            let id = ev.data.get("id").and_then(Value::as_str);
            if id.is_some() && id == s.shell.preparing.as_deref() {
                s.shell.preparing = None;
            }
            // The warm-up transcribed the reference clip: its detected language is known now.
            if let (Some(id), Some(l)) = (id, ev.data.get("reference_language")) {
                set_reference_language(s, id, l);
            }
        }
        EngineMsg::Event(ev) if ev.event == "train.done" => reset_warm(s),
        EngineMsg::Exited { code, stderr_tail } => {
            s.engine = None;
            reset_warm(s);
            let tail: Vec<&str> = stderr_tail.lines().rev().take(3).collect();
            let detail = tail.into_iter().rev().collect::<Vec<_>>().join("\n");
            s.shell.engine = EngineStatus::Stopped { code: *code, detail };
            toast(s, Tone::Danger, "The voice engine stopped");
        }
        _ => {}
    }
}

/// Records the detected language of a voice's `reference.wav` from an engine message (`voice.warm`, `tts.done`,
/// `redub.done`): a string, or `null` while unknown.
pub fn set_reference_language(s: &mut AppState, voice_id: &str, value: &Value) {
    let lang = value.as_str().filter(|l| !l.is_empty()).map(str::to_owned);
    if let Some(v) = s.shell.voices.iter_mut().find(|v| v.id == voice_id) {
        v.reference_language = lang;
    }
}

/// Forgets what was warmed (the engine restarted, or a voice was just trained).
fn reset_warm(s: &mut AppState) {
    s.shell.warm_for = None;
    s.shell.warm_req = None;
    s.shell.preparing = None;
}

/// Asks the engine to warm up the active voice once per engine start, whenever it becomes active (on launch with a
/// saved voice, on selection). True when the "Preparing voice" state appeared.
fn ensure_warm(s: &mut AppState) -> bool {
    if !engine_ready(s) {
        return false;
    }
    let Some(id) = s.shell.active_voice.clone() else { return false };
    if s.shell.warm_for.as_deref() == Some(id.as_str()) {
        return false;
    }
    s.shell.warm_for = Some(id.clone());
    // A voice whose clip is in another language than English also gets the native-accent clip ready (and, untrained,
    // the zero-shot converter).
    let mut params = json!({ "id": id });
    if let Some(v) = super::speak::warm_accent(s) {
        params["accent"] = json!("native-us");
        params["native_voice"] = json!(v.as_str());
    }
    s.shell.warm_req = request(s, "voices.warm", params);
    if s.shell.warm_req.is_some() {
        s.shell.preparing = Some(id);
        return true;
    }
    false
}

fn apply_voice_list(s: &mut AppState, list: Vec<VoiceInfo>) {
    let keep = s.shell.active_voice.as_ref().filter(|id| list.iter().any(|v| &v.id == *id)).cloned();
    s.shell.active_voice = keep.or_else(|| list.iter().find(|v| v.trained).or(list.first()).map(|v| v.id.clone()));
    let first = std::mem::replace(&mut s.shell.first_voice_list, false);
    s.shell.voices = list;
    // On launch, go straight to Speak when the user already has a voice.
    // Never past first-run consent / setup: that flow is not optional.
    if first && s.screen == Screen::Onboarding && !s.shell.voices.is_empty() && !super::onboarding::needed(s) {
        s.screen = Screen::Speak;
    }
}

// -------------------------------------------------------------------------------------------------
// View
// -------------------------------------------------------------------------------------------------

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    // The window's own controls over the header: macOS traffic lights (leading), Windows caption buttons
    // (trailing, drawn by `caption`); zero where the native frame stays (Linux).
    let mut insets = cx.chrome_insets();
    if let Some(preview) = s.shell.caption_preview {
        insets.trailing = 3.0 * super::caption::CELL;
        insets.caption = preview;
    }
    // Header width from the last layout: below these widths the header compacts so nothing slides under the
    // window's caption buttons (icon-only tabs, then a dot-only engine status).
    let header_w = cx.layout_with("header", |r| r[2]).unwrap_or(1300.0);
    let compact = header_w < 1260.0;
    // Centre the tabs on the window, not between the brand and the right-hand pills: pad the lighter side by
    // the difference (widths from the last layout), but never by more than the free room around the tabs, so a
    // narrow window keeps the tabs as central as it can without squeezing anything.
    let side = |key: &str| cx.layout_with(key, |r| r[2]).unwrap_or(0.0);
    let left = 16.0 + insets.leading + side("brand");
    let right = (if insets.trailing > 0.0 { insets.trailing + 14.0 } else { 12.0 }) + side("right");
    let free = side("header-gap-l") + side("header-gap-r") + side("header-pad-l") + side("header-pad-r");
    let balance = (right - left).clamp(-free, free);
    let fit = HeaderFit { compact, tight: header_w < 1120.0, balance };
    let page = body_page(s, cx);
    let mut body = col().key("body").w_full().grow(1.0).min_h(0.0).pt(14.0).gap(12.0);
    if let Some(b) = engine_banner(s, p) {
        body = body.child(b);
    }
    body = body.child(col().key(format!("screen-{:?}", s.screen)).w_full().grow(1.0).min_h(0.0).child(page));
    // The player bar docks under the screen (the screen gets shorter), so it never covers a screen's controls.
    let dock = super::playbar::view(s, p);
    let toasts_at = if dock.is_some() { super::playbar::DOCK_H + 14.0 } else { 26.0 };
    let mut root = el().key("root").w_full().h_full().fill(p.bg_top).child(background(p)).child(
        col().key("chrome").w_full().h_full().child(header(s, p, insets, fit)).child(body).children(dock),
    );
    if s.shell.voice_menu_open {
        root = root.child(voice_menu(s, p));
    }
    if let Some(dialog) = super::relocate::overlay(s, &p) {
        root = root.child(dialog);
    }
    if let Some(dialog) = super::train::close_overlay(s, &p) {
        root = root.child(dialog);
    }
    if let Some(menu) = super::formats::overlay(s, p) {
        root = root.child(menu);
    }
    if let Some(menu) = super::formats::output_overlay(s, p) {
        root = root.child(menu);
    }
    if let Some(menu) = super::lang::overlay(s, &p) {
        root = root.child(menu);
    }
    if let Some(menu) = super::playbar::overlay(s, p) {
        root = root.child(menu);
    }
    super::formats::window_drops(root, s).child(toast_stack(p, &s.shell.toasts, toasts_at))
}

/// The active screen's element (used when the banner pushes it down).
fn body_page(s: &AppState, cx: &mut Cx<AppState>) -> El {
    match s.screen {
        Screen::Onboarding => super::onboarding::view(s, cx),
        Screen::Voices => super::voices::view(s, cx),
        Screen::Record => super::record::view(s, cx),
        Screen::Train => super::train::view(s, cx),
        Screen::Speak => super::speak::view(s, cx),
        Screen::Redub => super::redub::view(s, cx),
    }
}

/// The window ground: deep navy with a few large, soft, cool light fields (ice blue, teal, steel, deep blue)
/// behind the glass, so the panels' backdrop blur has something to refract. Static on purpose: animating them
/// redrew the whole window (blur included) several times a second and kept the CPU busy while idle.
fn background(p: Palette) -> El {
    // Each field keeps a different fixed strength (what the old breathing animation showed at rest).
    const fn field(_period: f32, phase: f32) -> f32 {
        0.55 + 0.35 * phase
    }
    let a = 0.3;
    let blob = |key: &'static str, c: Color, w: f32, h: f32, strength: f32| {
        let a = a * strength;
        el().key(key).size(w, h).inert().radial_gradient(
            [0.5, 0.5],
            [w / 2.0, h / 2.0],
            [(0.0, c.alpha(a)), (0.4, c.alpha(a * 0.5)), (0.75, c.alpha(a * 0.12)), (1.0, c.alpha(0.0))],
        )
    };
    el().key("bg").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).inert().gradient(p.bg_top, p.bg_bottom)
        .child(blob("field-ice", p.aurora[0], 880.0, 620.0, field(29.0, 0.0)).pin(Some(-260.0), Some(-300.0), None, None))
        .child(blob("field-teal", p.aurora[1], 720.0, 560.0, field(37.0, 0.3)).pin(None, Some(-120.0), Some(-240.0), None))
        .child(blob("field-steel", p.aurora[2], 820.0, 600.0, field(33.0, 0.6)).pin(Some(-80.0), None, None, Some(-340.0)))
        .child(blob("field-blue", p.aurora[3], 700.0, 520.0, field(41.0, 0.15)).pin(None, None, Some(-60.0), Some(-300.0)))
        .child(blob("field-mid", p.aurora[0], 560.0, 420.0, field(47.0, 0.8)).pin(Some(300.0), Some(160.0), None, None))
}

// ---- header ----------------------------------------------------------------------------------------

/// Height of the header, which is also the window's title bar (`main.rs` passes it to `Titlebar::new`).
pub const HEADER_HEIGHT: f32 = 56.0;

/// The glass header is the window's title bar: a full-width smoked-glass strip across the top. Empty parts of it
/// move the window and a double-click maximizes it (`window_drag`); the buttons, tabs and pills in it keep their
/// presses. `insets` keeps the content clear of the window controls: the macOS traffic lights on the left and
/// the Windows minimize/maximize/close buttons on the right (Rune draws those over the header's right end).
/// How much the header must compact to fit beside the window controls.
#[derive(Clone, Copy)]
struct HeaderFit {
    /// Tabs show icons only (names as tooltips).
    compact: bool,
    /// The engine status shows only its dot (the label as a tooltip).
    tight: bool,
    /// Right side minus left side width: positive pads the left gap, negative the right one, so the tabs sit on
    /// the window's centre line.
    balance: f32,
}

fn header(s: &AppState, p: Palette, insets: ChromeInsets, fit: HeaderFit) -> El {
    let mark = brand_mark(s, 32.0);
    let brand = row().key("brand").gap(10.0).center_y().window_drag()
        .child(mark)
        .child(text("Mimvox").font(ty::TITLE).color(p.text))
        .child(text("early version").font(ty::CAPTION).color(p.warn).help("Early version: expect bugs. Report problems on the project's GitHub page."));
    let right = row().key("right").gap(8.0).center_y().window_drag().child(engine_pill(s, p, fit.tight)).child(voice_pill(s, p));
    let bar = col().key("header").w_full().h(HEADER_HEIGHT).fill(p.glass).gradient(p.glass_top, p.glass_bottom)
        .backdrop_blur(24.0).shadow(6.0, 24.0, p.shadow_soft).window_drag()
        .child(row().key("header-row").w_full().h_full().center_y().gap(14.0)
            .pl(16.0 + insets.leading)
            .pr(if insets.trailing > 0.0 { insets.trailing + 14.0 } else { 12.0 })
            .child(brand)
            .child(el().key("header-pad-l").w(fit.balance.max(0.0)).h_full().window_drag())
            .child(el().key("header-gap-l").grow(1.0).h_full().window_drag())
            .child(if s.screen == Screen::Onboarding { el() } else { tabs(s, p, fit.compact) })
            .child(el().key("header-gap-r").grow(1.0).h_full().window_drag())
            .child(el().key("header-pad-r").w((-fit.balance).max(0.0)).h_full().window_drag())
            .child(right))
        .children((insets.trailing > 0.0).then(|| super::caption::buttons(p, insets.caption, HEADER_HEIGHT)))
        .child(el().key("header-rim").pin(Some(0.0), None, Some(0.0), Some(0.0)).h(1.0).inert().fill(p.rim_bottom.alpha(0.16)))
        .child(el().key("header-sheen").pin(Some(0.0), Some(0.0), Some(0.0), None).h(1.0).inert().fill(p.sheen.alpha(0.12)));
    bar
}

const STEPS: [(Screen, &str, &'static rune::Icon); 4] = [
    (Screen::Record, "Record", &icons::MIC),
    (Screen::Train, "Train", &icons::CHIP),
    (Screen::Speak, "Speak", &icons::VOLUME),
    (Screen::Redub, "Redub", &super::redub::ICON),
];

/// The three steps as a glass pill with an accent-tinted glass lens that springs under the current step.
fn tabs(s: &AppState, p: Palette, compact: bool) -> El {
    // The Train tab's "Paused 118/300" badge needs more room than a percentage.
    let paused_badge = super::train::status_line(s).is_some_and(|l| l.starts_with("Paused"));
    let (w, h, pad) = (if compact { 52.0 } else if paused_badge { 132.0 } else { 100.0 }, 38.0, 4.0);
    let idx = STEPS.iter().position(|(sc, ..)| *sc == s.screen);
    let on_lens = p.on_accent;
    let mut strip = row().key("tab-strip");
    for (i, (screen, name, glyph)) in STEPS.iter().enumerate() {
        let on = idx == Some(i);
        let screen = *screen;
        let ink = if on { on_lens } else { p.text_dim };
        let mut tab = row().key(*name).role("tab").label(*name).checked(on).focusable(true).size(w, h).gap(7.0).center().radius(h / 2.0)
            .hover(if on { Color::TRANSPARENT } else { p.glass_hover })
            .on_press(move |s: &mut AppState| set_screen(s, screen))
            .child(icon_el(glyph, 15.0, ink).a11y_hidden())
            .children((!compact).then(|| text(*name).font(ty::LABEL).color(ink).a11y_hidden()));
        if compact {
            tab = tab.help(*name);
        }
        if screen == Screen::Train {
            if let Some(line) = super::train::status_line(s) {
                tab = tab.gap(5.0).child(train_badge(p, &line, on, compact));
            }
        }
        strip = strip.child(tab);
    }
    let lens = el().key("tab-indicator").size(w, h).absolute(pad + idx.unwrap_or(0) as f32 * w, pad).radius(h / 2.0)
        .opacity(if idx.is_some() { 1.0 } else { 0.0 })
        .motion(motion::BOUNCY)
        .lead(motion::SNAP, motion::UI)
        .a11y_hidden();
    let lens = accent_glass(p, lens);
    let bar = el().key("tabs").size(w * STEPS.len() as f32 + pad * 2.0, h + pad * 2.0).padding(pad).radius((h + pad * 2.0) / 2.0)
        .fill(p.well)
        .border_gradient(1.0, Color::BLACK.alpha(0.35), p.well_rim)
        .box_shadow(rune::Shadow { x: 0.0, y: 2.0, blur: 8.0, spread: 0.0, color: Color::BLACK.alpha(0.35), inset: true });
    bar.child(lens).child(strip)
}

/// A small pill on the Train tab while a model trains: the percentage, or an ellipsis while it starts; a pause
/// icon with `k/N` while a training is paused.
fn train_badge(p: Palette, line: &str, on_lens: bool, compact: bool) -> El {
    let paused = line.strip_prefix("Paused ");
    let label = match paused {
        Some(k) => k.to_owned(),
        None => line.rsplit(' ').next().filter(|w| w.ends_with('%')).unwrap_or("…").to_owned(),
    };
    let mut pill = row().key("train-badge").role("status").label(line.to_owned()).h(18.0).px(6.0).gap(3.0).center().radius(9.0)
        .fill(if on_lens { p.on_accent.alpha(0.85) } else { p.accent.alpha(0.2) });
    if paused.is_some() {
        pill = pill.child(icon_el(&icons::PAUSE, 11.0, p.accent).a11y_hidden());
        if compact {
            return pill;
        }
    }
    pill.child(text(label).font(ty::CAPTION).color(p.accent).a11y_hidden())
}

fn engine_pill(s: &AppState, p: Palette, tight: bool) -> El {
    let (tone, label, pulse, retry) = match &s.shell.engine {
        EngineStatus::Idle | EngineStatus::Starting { .. } => (Tone::Warning, "Starting engine…".to_owned(), true, false),
        EngineStatus::Ready { tts, rvc, .. } => {
            let loading = [tts, rvc].iter().any(|m| m.as_deref() == Some("loading"));
            let paused = [tts, rvc].iter().any(|m| m.as_deref() == Some("paused"));
            let missing = [tts, rvc].iter().any(|m| m.as_deref() == Some("missing"));
            if missing {
                (Tone::Danger, "Models missing".to_owned(), false, false)
            } else if paused {
                (Tone::Neutral, "Training · speech paused".to_owned(), false, false)
            } else if loading {
                (Tone::Warning, "Loading models…".to_owned(), true, false)
            } else {
                (Tone::Success, "Ready".to_owned(), false, false)
            }
        }
        EngineStatus::Stopped { .. } | EngineStatus::Unavailable { .. } => (Tone::Danger, "Engine stopped".to_owned(), false, true),
    };
    let mut pill = glass_edge(p, row().key("engine-pill").role("status").label(label.clone()).h(32.0).px(13.0).gap(8.0).center().radius(16.0)
        .fill(p.glass).gradient(p.glass_top, p.glass_bottom).border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(3.0, 10.0, p.shadow_soft))
        .motion(motion::UI)
        .child(status_dot(p, tone, pulse))
        .children((!tight).then(|| text(label.clone()).font(ty::LABEL).color(p.text).a11y_hidden()));
    pill = pill.help(engine_help(&s.shell.engine));
    if retry {
        pill = pill.hover(p.glass_hover).focusable(true)
            .on_press(|s: &mut AppState| restart_engine(s))
            .child(icon_el(&icons::REFRESH, 14.0, p.text_dim));
    }
    pill
}

/// The status pill's tooltip: what the state means, the device, and each model's state.
fn engine_help(engine: &EngineStatus) -> String {
    let model = |m: &Option<String>| match m.as_deref() {
        Some("ready") => "ready".to_owned(),
        Some("loading") => "loading…".to_owned(),
        Some("paused") => "paused while a voice trains".to_owned(),
        Some("missing") => "not installed".to_owned(),
        Some(other) => other.to_owned(),
        None => "not loaded yet".to_owned(),
    };
    match engine {
        EngineStatus::Idle | EngineStatus::Starting { .. } => {
            "Starting the speech engine. Speak and Redub are available in a few seconds.".to_owned()
        }
        EngineStatus::Ready { device, tts, rvc } => {
            let any = |want: &str| [tts, rvc].iter().any(|m| m.as_deref() == Some(want));
            let summary = if any("missing") {
                "Some models are not installed: run setup again."
            } else if any("paused") {
                "A voice is training: the speech models are paused so training has the GPU. They come back when training ends or pauses."
            } else if any("loading") {
                "Loading the speech models into memory. The first start takes up to about half a minute."
            } else {
                "Ready: the speech models are loaded."
            };
            let mut t = summary.to_owned();
            if let Some(d) = device {
                t.push_str(&format!("\nDevice: {d}"));
            }
            t.push_str(&format!("\nSpeech model: {}\nVoice conversion: {}", model(tts), model(rvc)));
            t
        }
        EngineStatus::Stopped { code, detail } => {
            let mut t = match code {
                Some(c) => format!("The engine stopped (exit code {c})."),
                None => "The engine stopped.".to_owned(),
            };
            let last = detail.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
            if !last.is_empty() {
                t.push_str(&format!("\n{last}"));
            }
            t.push_str("\nClick to restart it. Details are in logs/engine.log in the data folder.");
            t
        }
        EngineStatus::Unavailable { reason } => format!("The engine cannot run: {reason}"),
    }
}

fn voice_pill(s: &AppState, p: Palette) -> El {
    let name = active_voice(s).map(|v| v.name.clone()).unwrap_or_else(|| "No voice yet".to_owned());
    let has = active_voice(s).is_some();
    let open = s.shell.voice_menu_open;
    let chip = row().key("voice-pill").role("button").label("Choose voice").h(32.0).pl(12.0).pr(12.0).gap(8.0).center().radius(16.0)
        .fill(if open { p.lens } else { p.glass }).gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, if open { p.accent.alpha(0.4) } else { p.rim_bottom })
        .shadow(3.0, 10.0, p.shadow_soft);
    glass_edge(p, chip)
        .hover(p.glass_hover).focusable(true).motion(motion::UI)
        .on_press(|s: &mut AppState| s.shell.voice_menu_open = !s.shell.voice_menu_open)
        .child(icon_el(&icons::USER, 14.0, if has { p.accent } else { p.text_dim }))
        .child(text(name).font(ty::LABEL).color(if has { p.text } else { p.text_dim }).max_w(110.0).truncate().a11y_hidden())
        .child(icon_el(&icons::CHEVRON_DOWN, 12.0, p.text_dim))
}

fn voice_menu(s: &AppState, p: Palette) -> El {
    let scrim = el().key("voice-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(|s: &mut AppState| s.shell.voice_menu_open = false);
    let mut list = col().w_full().gap(2.0);
    for v in &s.shell.voices {
        let id = v.id.clone();
        let on = s.shell.active_voice.as_deref() == Some(v.id.as_str());
        list = list.child(
            row().key(format!("voice-{}", v.id)).w_full().h(38.0).px(12.0).gap(10.0).center_y().radius(19.0)
                .fill(if on { p.accent.alpha(0.14) } else { Color::TRANSPARENT }).hover(p.glass_hover).focusable(true)
                .on_press(move |s: &mut AppState| {
                    s.shell.active_voice = Some(id.clone());
                    s.shell.voice_menu_open = false;
                })
                .child(text(v.name.clone()).font(ty::LABEL).color(p.text).grow(1.0).truncate())
                .child(badge(p, if v.trained { "Trained" } else { "Zero-shot" }, if v.trained { Tone::Success } else { Tone::Accent }))
                .child(el().w(16.0).children(on.then(|| icon_el(&icons::CHECK, 14.0, p.accent)))),
        );
    }
    if s.shell.voices.is_empty() {
        list = list.child(text("No voices yet").font(ty::CAPTION).color(p.text_faint).px(12.0).py(10.0));
    }
    let manage = Button::new("Manage voices").ghost().small().key("menu-manage").on_press(|s| set_screen(s, Screen::Voices)).build(p);
    let create = Button::new("New voice").primary().small().icon(&icons::PLUS).key("menu-new").on_press(|s| set_screen(s, Screen::Record)).build(p);
    let panel = col().key("voice-menu").w(290.0).padding(8.0).gap(6.0).radius(radius::PANEL + 4.0)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.88))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .backdrop_blur(28.0)
        .anchored("voice-pill", Side::Bottom, Align::End, 8.0)
        .transition(rune::Transition::scale(0.98))
        .child(list)
        .child(row().w_full().justify_between().child(manage).child(create));
    el().key("voice-menu-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).child(scrim).child(panel)
}

/// A glass banner under the header when the engine is not running, with a Retry button.
fn engine_banner(s: &AppState, p: Palette) -> Option<El> {
    let (title, detail) = match &s.shell.engine {
        EngineStatus::Stopped { code, detail } => (
            "The voice engine stopped".to_owned(),
            if detail.is_empty() { code.map(|c| format!("Exit code {c}.")).unwrap_or_default() } else { detail.clone() },
        ),
        EngineStatus::Unavailable { reason } => ("The voice engine is not set up yet".to_owned(), format!("{reason} Run setup.sh (setup.cmd on Windows), then press Retry.")),
        _ => return None,
    };
    let row_ = row().key("engine-banner").w_full().px(14.0).py(9.0).gap(12.0).center_y()
        .child(icon_el(&icons::ALERT, 16.0, p.warn))
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(text(title).font(ty::LABEL).color(p.text))
            .child(text(detail).font(ty::CAPTION).color(p.text_dim).wrap().max_lines(2)))
        .child(Button::new("Retry").small().icon(&icons::REFRESH).key("banner-retry").on_press(|s| restart_engine(s)).build(p));
    Some(el().w_full().px(16.0).child(glass_card(p).w_full().child(row_)))
}
