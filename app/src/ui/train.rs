//! Train screen: dataset summary, epochs, start, live progress, done and error states.
//!
//! Training runs inside the engine, so the other screens stay usable while it goes on; this
//! module keeps listening to `train.*` events whichever screen is shown (see [`is_training`]).

use std::collections::HashMap;
use std::time::Instant;

use rune::{Cx, El, Tx, col, el, row, text};
use serde_json::{Value, json};

use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use crate::ui::theme::{self, Palette, radius, ty};
use crate::engine::protocol::PausedTraining;
use crate::ui::components::{Tone, icon_el, icons};
use crate::ui::shell;
use crate::ui::voices::{
    self,
    kit::{self, BtnKind},
};

const EPOCH_MIN: u32 = 25;
const EPOCH_MAX: u32 = 1000;
const EPOCH_STEP: u32 = 25;
/// Measured on an RTX 4070: 300 epochs on 6.4 minutes of audio took about 55 minutes.
const SECONDS_PER_EPOCH_CUDA: f32 = 11.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Stage {
    #[default]
    Preprocess,
    Extract,
    Train,
    Index,
}

impl Stage {
    const ALL: [Stage; 4] = [Stage::Preprocess, Stage::Extract, Stage::Train, Stage::Index];
    fn parse(s: &str) -> Option<Stage> {
        Some(match s {
            "preprocess" => Stage::Preprocess,
            "extract" => Stage::Extract,
            "train" => Stage::Train,
            "index" => Stage::Index,
            _ => return None,
        })
    }
    fn label(self) -> &'static str {
        match self {
            Stage::Preprocess => "Preprocess",
            Stage::Extract => "Extract",
            Stage::Train => "Train",
            Stage::Index => "Index",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Phase {
    #[default]
    Idle,
    /// `train.start` sent, waiting for the job id.
    Starting,
    Running,
    Done { epochs: u32, elapsed_s: f32 },
    Failed(String),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    pub total_minutes: f32,
    pub parts: u32,
    pub level: String,
    pub message: String,
    pub recommended_epochs: u32,
}

#[derive(Clone, Debug)]
enum Tag {
    Summary(String),
    Start,
    Resume,
    Pause,
    Discard,
    Cancel,
}

/// The "app is being closed while training" dialog (see [`on_close`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum CloseDialog {
    #[default]
    Hidden,
    /// Asking what to do with the running training.
    Ask,
    /// "Pause and close" was chosen: waiting for `train.paused`, then the app exits by itself.
    Pausing,
}

pub struct TrainState {
    /// Summary of the active voice's dataset, with the voice it was fetched for.
    pub summary: Option<(String, Summary)>,
    pub summary_error: Option<String>,
    /// Epochs control: automatic (engine decides) or this many.
    pub manual: bool,
    pub epochs: u32,
    pub phase: Phase,
    /// The voice being trained (or just trained).
    pub voice_id: Option<String>,
    pub job: Option<String>,
    pub stage: Stage,
    pub epoch: u32,
    pub total_epochs: u32,
    pub pct: f32,
    pub eta_s: Option<f32>,
    pub elapsed_s: f32,
    started: Option<Instant>,
    pub confirm_cancel: bool,
    /// `train.pause` was sent and the checkpoint is not on disk yet.
    pub pausing: bool,
    /// The run being started is a resume of a paused one (`train.resume`).
    pub resuming: bool,
    /// Asked "Discard the paused training?" on the paused screen.
    pub confirm_discard: bool,
    /// The window close button was used while training.
    pub close: CloseDialog,
    /// Why the close dialog could not pause (shown inside it).
    pub close_note: Option<String>,
    /// Exit the app when the current event settles (the shell owns the `Tx` that can quit).
    quit_requested: bool,
    /// A short note after a cancel.
    pub note: Option<String>,
    pending: HashMap<u64, Tag>,
    /// What the screen showed when the last event settled, to spot screen and voice changes.
    last_screen: Option<Screen>,
    last_voice: Option<String>,
}

impl Default for TrainState {
    fn default() -> Self {
        TrainState {
            summary: None,
            summary_error: None,
            manual: false,
            epochs: 300,
            phase: Phase::Idle,
            voice_id: None,
            job: None,
            stage: Stage::Preprocess,
            epoch: 0,
            total_epochs: 0,
            pct: 0.0,
            eta_s: None,
            elapsed_s: 0.0,
            started: None,
            confirm_cancel: false,
            pausing: false,
            resuming: false,
            confirm_discard: false,
            close: CloseDialog::Hidden,
            close_note: None,
            quit_requested: false,
            note: None,
            pending: HashMap::new(),
            last_screen: None,
            last_voice: None,
        }
    }
}

/// Training (or starting to train) right now; the shell can show a badge on the Train tab.
pub fn is_training(s: &AppState) -> bool {
    matches!(s.train.phase, Phase::Starting | Phase::Running)
}

/// The paused (or interrupted) training of the active voice, else of any voice.
pub fn paused_run(s: &AppState) -> Option<&PausedTraining> {
    voices::active_voice_info(s).and_then(|v| v.paused.as_ref()).or_else(|| s.shell.voices.iter().find_map(|v| v.paused.as_ref()))
}

/// Short status for a badge: `"Training 42%"`, `"Starting"`, `"Resuming"`, `"Pausing"` or `"Paused 118/300"`.
pub fn status_line(s: &AppState) -> Option<String> {
    match s.train.phase {
        Phase::Starting if s.train.resuming => Some("Resuming".into()),
        Phase::Starting => Some("Starting".into()),
        Phase::Running if s.train.pausing => Some("Pausing".into()),
        Phase::Running => Some(format!("Training {:.0}%", s.train.pct)),
        Phase::Idle => paused_run(s).map(|p| format!("Paused {}/{}", p.epoch, p.total_epochs)),
        _ => None,
    }
}

/// Both speech models are loaded (resuming hands the GPU over again, which needs them idle and ready).
fn models_ready(s: &AppState) -> bool {
    matches!(&s.shell.engine, shell::EngineStatus::Ready { tts, rvc, .. } if tts.as_deref() == Some("ready") && rvc.as_deref() == Some("ready"))
}

/// Overall percentage (`train.progress.pct`) a run has reached at `epoch` of `total`: learning spans 15..95.
fn overall_pct(epoch: u32, total: u32) -> f32 {
    if total == 0 { 15.0 } else { 15.0 + 80.0 * (epoch.min(total) as f32 / total as f32) }
}

/// Fetches the dataset summary for the active voice.
pub fn refresh_summary(s: &mut AppState) {
    if let Some(id) = voices::active_voice(s).map(str::to_string) {
        send(s, Tag::Summary(id.clone()), "dataset.summary", json!({ "voice_id": id }));
    }
}

/// Sends a request and remembers what it was for. `false` when the engine is not running.
fn send(s: &mut AppState, tag: Tag, method: &str, params: Value) -> bool {
    match shell::request(s, method, params) {
        Some(id) => {
            s.train.pending.insert(id, tag);
            true
        }
        None => false,
    }
}

/// After every event: keeps the summary fresh (screen entry, voice change) and drops a close dialog that has
/// nothing left to ask.
pub fn settle(s: &mut AppState) -> bool {
    let screen_changed = s.train.last_screen != Some(s.screen);
    s.train.last_screen = Some(s.screen);
    let voice = voices::active_voice(s).map(str::to_string);
    let voice_changed = s.train.last_voice != voice;
    if voice_changed {
        s.train.last_voice = voice;
        s.train.summary = None;
        s.train.summary_error = None;
    }
    let mut changed = false;
    if (screen_changed && s.screen == Screen::Train || voice_changed) && shell::engine_ready(s) && !is_training(s) {
        refresh_summary(s);
        changed = true;
    }
    if s.train.close == CloseDialog::Ask && !is_training(s) {
        s.train.close = CloseDialog::Hidden; // the training ended meanwhile: nothing left to ask
        changed = true;
    }
    changed
}

/// The start of the running training on the app clock (the Train screen's elapsed time ticks from it).
pub fn clock_origin(s: &AppState) -> Option<f64> {
    let st = s.train.started.filter(|_| is_training(s))?;
    Some(s.shell.now - st.elapsed().as_secs_f64())
}

// ---------------------------------------------------------------------------------------------
// Closing the window while training
// ---------------------------------------------------------------------------------------------


/// `App::on_close`. Rune cannot veto a close: with `keep_running` the window only hides, runs this hook, and the app
/// keeps going. While a training runs the window is shown again with the "Training is running" dialog; otherwise the
/// engine is stopped and the app quits. A quit that is not the user's close (system shutdown) goes straight through.
pub fn on_close(s: &mut AppState, tx: &mut Tx) {
    if is_training(s) && !rune::quitting() {
        if s.train.close == CloseDialog::Hidden {
            s.train.close = CloseDialog::Ask;
            s.train.close_note = None;
        }
        if let Some(app) = crate::ui::events::app() {
            app.open();
        }
        return;
    }
    quit_now(s, tx);
}

/// Stops the engine (which stops Applio) and quits. The UI thread only asks the engine to exit: waiting for it here,
/// with the window still on screen, froze the window and Windows painted it white. `main` waits for the engine
/// after the window is gone (`take_closing_engine`).
fn quit_now(s: &mut AppState, tx: &mut Tx) {
    super::stats::write();
    if let Some(c) = s.engine.take() {
        c.request_shutdown();
        *CLOSING_ENGINE.lock().unwrap_or_else(|e| e.into_inner()) = Some(c);
    }
    tx.quit();
}

static CLOSING_ENGINE: std::sync::Mutex<Option<crate::engine::EngineClient>> = std::sync::Mutex::new(None);

/// The engine a quit asked to exit, for `main` to wait for once the app loop has ended.
pub fn take_closing_engine() -> Option<crate::engine::EngineClient> {
    CLOSING_ENGINE.lock().unwrap_or_else(|e| e.into_inner()).take()
}

/// The shell calls this after every event: true once the app should quit (after "Close anyway" or "Pause and close").
pub fn take_quit(s: &mut AppState, tx: &mut Tx) -> bool {
    if s.train.quit_requested {
        s.train.quit_requested = false;
        quit_now(s, tx);
        return true;
    }
    false
}

fn level_ok(level: &str) -> bool {
    matches!(level, "trainable" | "great")
}

fn start(s: &mut AppState) {
    let Some(id) = voices::active_voice(s).map(str::to_string) else { return };
    let epochs = if s.train.manual { json!(s.train.epochs) } else { Value::Null };
    let t = &mut s.train;
    t.phase = Phase::Starting;
    t.voice_id = Some(id.clone());
    t.job = None;
    t.stage = Stage::Preprocess;
    t.epoch = 0;
    t.total_epochs = if t.manual { t.epochs } else { t.summary.as_ref().map_or(0, |(_, x)| x.recommended_epochs) };
    t.pct = 0.0;
    t.eta_s = None;
    t.elapsed_s = 0.0;
    t.started = Some(Instant::now());
    t.confirm_cancel = false;
    t.pausing = false;
    t.resuming = false;
    t.note = None;
    if !send(s, Tag::Start, "train.start", json!({ "voice_id": id, "epochs": epochs })) {
        finish_run(&mut s.train, Phase::Failed("The voice engine is not running. Check the status at the top and press Retry.".into()));
    }
}

/// Continues the paused (or interrupted) training of the active voice from its saved checkpoint.
fn resume(s: &mut AppState) {
    let Some(id) = voices::active_voice(s).map(str::to_string) else { return };
    let Some(pz) = paused_run(s).cloned() else { return };
    let t = &mut s.train;
    t.phase = Phase::Starting;
    t.resuming = true;
    t.pausing = false;
    t.voice_id = Some(id.clone());
    t.job = None;
    t.stage = Stage::Train;
    t.epoch = pz.epoch;
    t.total_epochs = pz.total_epochs;
    t.pct = overall_pct(pz.epoch, pz.total_epochs);
    t.eta_s = None;
    t.elapsed_s = 0.0;
    t.started = Some(Instant::now());
    t.confirm_cancel = false;
    t.confirm_discard = false;
    t.note = None;
    if !send(s, Tag::Resume, "train.resume", json!({ "voice_id": id })) {
        finish_run(&mut s.train, Phase::Failed("The voice engine is not running. Check the status at the top and press Retry.".into()));
    }
}

/// Asks the engine to pause after the next checkpoint (`train.paused` answers).
fn pause(s: &mut AppState) {
    let Some(job) = s.train.job.clone() else { return };
    if s.train.pausing || s.train.stage != Stage::Train {
        return;
    }
    if send(s, Tag::Pause, "train.pause", json!({ "job": job })) {
        s.train.pausing = true;
        s.train.note = None;
    }
}

/// Throws away the paused training of the active voice.
fn discard(s: &mut AppState) {
    s.train.confirm_discard = false;
    if let Some(id) = voices::active_voice(s).map(str::to_string) {
        send(s, Tag::Discard, "train.cancel", json!({ "voice_id": id }));
    }
}

fn cancel(s: &mut AppState) {
    s.train.confirm_cancel = false;
    if let Some(job) = s.train.job.clone() {
        send(s, Tag::Cancel, "train.cancel", json!({ "job": job }));
        s.train.note = Some("Cancelling...".into());
    } else {
        // Not started on the engine yet: just drop the local state.
        s.train.phase = Phase::Idle;
        s.train.started = None;
    }
}

fn finish_run(t: &mut TrainState, phase: Phase) {
    if let Some(st) = t.started.take() {
        t.elapsed_s = st.elapsed().as_secs_f32();
    }
    t.phase = phase;
    t.job = None;
    t.eta_s = None;
    t.confirm_cancel = false;
    t.pausing = false;
    t.resuming = false;
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Response { id, result } => {
            let Some(tag) = s.train.pending.remove(id) else { return };
            match (tag, result) {
                (Tag::Summary(voice), Ok(v)) => {
                    // Ignore answers for a voice that is no longer active.
                    if voices::active_voice(s) == Some(voice.as_str()) {
                        s.train.summary_error = None;
                        let sum = Summary {
                            total_minutes: v.get("total_minutes").and_then(Value::as_f64).unwrap_or(0.0) as f32,
                            parts: v.get("parts").and_then(Value::as_array).map_or(0, |a| a.len() as u32),
                            level: v.get("level").and_then(Value::as_str).unwrap_or("too_short").to_string(),
                            message: v.get("message").and_then(Value::as_str).unwrap_or("").to_string(),
                            recommended_epochs: v.get("recommended_epochs").and_then(Value::as_u64).unwrap_or(0) as u32,
                        };
                        if !s.train.manual && sum.recommended_epochs > 0 {
                            s.train.epochs = sum.recommended_epochs.clamp(EPOCH_MIN, EPOCH_MAX);
                        }
                        s.train.summary = Some((voice, sum));
                    }
                }
                (Tag::Summary(_), Err(e)) => s.train.summary_error = Some(e.message.clone()),
                (Tag::Start | Tag::Resume, Ok(v)) => {
                    s.train.job = v.get("job").and_then(Value::as_str).map(str::to_string);
                    if s.train.phase == Phase::Starting {
                        s.train.phase = Phase::Running;
                    }
                }
                (Tag::Start | Tag::Resume, Err(e)) => {
                    if e.code == "paused" {
                        // This voice has a paused run: show it (with Resume) instead of an error.
                        finish_run(&mut s.train, Phase::Idle);
                        shell::refresh_voices(s);
                        return;
                    }
                    let text = match e.code.as_str() {
                        "busy" => "The engine is busy with another job. Try again in a moment.".to_string(),
                        "not_ready" => "The speech models are still loading. Try again in a moment.".to_string(),
                        _ => e.message.clone(),
                    };
                    finish_run(&mut s.train, Phase::Failed(text));
                    shell::refresh_voices(s);
                }
                (Tag::Pause, Ok(_)) => {}
                (Tag::Pause, Err(e)) => {
                    s.train.pausing = false;
                    if s.train.close == CloseDialog::Pausing {
                        s.train.close = CloseDialog::Ask;
                        s.train.close_note = Some(e.message.clone());
                    } else {
                        s.train.note = Some(e.message.clone());
                    }
                }
                (Tag::Discard, Ok(_)) => {
                    s.train.note = Some("The paused training was discarded.".into());
                    shell::refresh_voices(s);
                    refresh_summary(s);
                }
                (Tag::Discard, Err(e)) => s.train.note = Some(format!("Could not discard it: {}", e.message)),
                (Tag::Cancel, Ok(_)) => {}
                (Tag::Cancel, Err(e)) => s.train.note = Some(format!("Could not cancel: {}", e.message)),
            }
        }
        EngineMsg::Event(ev) if ev.event.starts_with("train.") => {
            let t = &mut s.train;
            let mine = match (&ev.job, &t.job) {
                (Some(a), Some(b)) => a == b,
                _ => matches!(t.phase, Phase::Starting | Phase::Running),
            };
            if !mine {
                return;
            }
            match ev.event.as_str() {
                "train.progress" => {
                    let d = &ev.data;
                    if t.phase == Phase::Starting {
                        t.phase = Phase::Running;
                    }
                    if let Some(st) = d.get("stage").and_then(Value::as_str).and_then(Stage::parse) {
                        t.stage = st;
                    }
                    if let Some(e) = d.get("epoch").and_then(Value::as_u64) {
                        t.epoch = e as u32;
                    }
                    if let Some(e) = d.get("total_epochs").and_then(Value::as_u64) {
                        t.total_epochs = e as u32;
                    }
                    if let Some(p) = d.get("pct").and_then(Value::as_f64) {
                        t.pct = (p as f32).clamp(0.0, 100.0);
                    }
                    t.eta_s = d.get("eta_s").and_then(Value::as_f64).map(|x| x as f32);
                    if t.job.is_none() {
                        t.job = ev.job.clone();
                    }
                }
                "train.done" => {
                    let epochs = ev.data.get("epochs").and_then(Value::as_u64).unwrap_or(t.total_epochs as u64) as u32;
                    let elapsed = ev.data.get("elapsed_s").and_then(Value::as_f64).map(|x| x as f32);
                    if let Some(st) = t.started.take() {
                        t.elapsed_s = st.elapsed().as_secs_f32();
                    }
                    t.pct = 100.0;
                    t.phase = Phase::Done { epochs, elapsed_s: elapsed.unwrap_or(t.elapsed_s) };
                    t.job = None;
                    t.note = None;
                    t.pausing = false;
                    t.resuming = false;
                    if t.close != CloseDialog::Hidden {
                        t.close = CloseDialog::Hidden;
                        t.quit_requested = true; // the user wanted to leave and the voice is trained: nothing left to wait for
                    }
                    refresh_summary(s);
                    shell::refresh_voices(s);
                }
                "train.paused" => {
                    let epoch = ev.data.get("epoch").and_then(Value::as_u64).unwrap_or(0);
                    let total = ev.data.get("total_epochs").and_then(Value::as_u64).unwrap_or(t.total_epochs as u64);
                    finish_run(t, Phase::Idle);
                    t.note = Some(format!(
                        "Training paused at epoch {epoch} of {total}. Speech is available again as soon as the models have loaded."
                    ));
                    if t.close == CloseDialog::Pausing {
                        t.close = CloseDialog::Hidden;
                        t.quit_requested = true;
                    }
                    shell::refresh_voices(s);
                }
                "train.error" => {
                    let code = ev.data.get("code").and_then(Value::as_str).unwrap_or("");
                    let message = ev.data.get("message").and_then(Value::as_str).unwrap_or("Training failed.").to_string();
                    t.close = CloseDialog::Hidden;
                    if code == "cancelled" {
                        finish_run(t, Phase::Idle);
                        t.note = Some("Training was cancelled. Nothing was changed.".into());
                    } else {
                        finish_run(t, Phase::Failed(message));
                    }
                    shell::refresh_voices(s); // a failed run can leave a resumable checkpoint behind
                }
                _ => {}
            }
        }
        EngineMsg::Exited { .. } => {
            s.train.pending.clear();
            if is_training(s) {
                finish_run(&mut s.train, Phase::Failed("The engine stopped while training. Restart the app and try again.".into()));
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------------------------

fn fmt_dur(secs: f32) -> String {
    let t = secs.max(0.0).round() as u64;
    let (h, m, s) = (t / 3600, (t % 3600) / 60, t % 60);
    if h > 0 { format!("{h}h {m:02}m") } else if m > 0 { format!("{m}m {s:02}s") } else { format!("{s}s") }
}

fn level_label(level: &str) -> (&'static str, Tone) {
    match level {
        "great" => ("Great", Tone::Success),
        "trainable" => ("Trainable", Tone::Success),
        "zero_shot" => ("Quick voice only", Tone::Warning),
        _ => ("Too short", Tone::Danger),
    }
}

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    let t = &s.train;
    let mut body = col().w_full().gap(18.0);

    let running_for = t.voice_id.as_deref().and_then(|id| voices::voice_name(s, id)).map(str::to_string);
    let active_name = voices::active_voice_info(s).map(|v| v.name.clone());
    let name = match &t.phase {
        Phase::Idle => active_name.clone(),
        _ => running_for.or(active_name.clone()),
    };
    let paused_here = t.phase == Phase::Idle && voices::active_voice_info(s).is_some_and(|v| v.paused.is_some());
    let sub = match &name {
        Some(n) if paused_here => format!("The training of \"{n}\" is waiting for you."),
        Some(n) => format!("Training the voice \"{n}\" on your recordings."),
        None => "Teach the app your voice from your recordings.".to_string(),
    };
    body = body.child(kit::header(&p, "Train", &sub));

    match &t.phase {
        Phase::Idle => {
            if name.is_none() {
                body = body.child(no_voice(&p));
            } else if let Some(pz) = voices::active_voice_info(s).and_then(|v| v.paused.clone()) {
                body = body.child(paused_view(s, &p, &pz, name.as_deref().unwrap_or("voice")));
            } else {
                body = body.child(setup_view(s, &p));
            }
        }
        Phase::Starting | Phase::Running => body = body.child(progress_view(t, &p, name.as_deref().unwrap_or("voice"))),
        Phase::Done { epochs, elapsed_s } => body = body.child(done_view(&p, *epochs, *elapsed_s, name.as_deref().unwrap_or("voice"))),
        Phase::Failed(msg) => body = body.child(failed_view(&p, msg)),
    }
    kit::page(body)
}

fn no_voice(p: &Palette) -> El {
    kit::panel(p)
        .items_center()
        .gap(12.0)
        .child(text("Pick a voice first").font(ty::TITLE).color(p.text))
        .child(text("Training needs a voice with recordings. Create or select one on the Voices screen.").font(ty::BODY).color(p.text_dim).wrap())
        .child(kit::button(p, "Go to Voices", BtnKind::Primary, true, |s: &mut AppState| s.screen = Screen::Voices))
}

fn stat(p: &Palette, label: &str, value: String) -> El {
    col()
        .grow(1.0)
        .min_w(0.0)
        .gap(4.0)
        .child(text(label.to_string()).font(ty::OVERLINE).color(p.text_faint))
        .child(text(value).font(rune::TextStyle::new(22.0, 650.0).features("'tnum' 1")).color(p.text))
}

fn setup_view(s: &AppState, p: &Palette) -> El {
    let t = &s.train;
    let sum = t.summary.as_ref().map(|(_, x)| x);
    let mut out = col().w_full().gap(18.0);

    // Dataset summary
    let mut card = kit::card(p).gap(14.0).child(text("Your recordings").font(ty::TITLE).color(p.text));
    match sum {
        Some(x) => {
            let (label, tone) = level_label(&x.level);
            card = card
                .child(
                    row()
                        .w_full()
                        .gap(14.0)
                        .child(stat(p, "RECORDED", format!("{:.1} min", x.total_minutes)))
                        .child(stat(p, "PARTS", x.parts.to_string()))
                        .child(stat(p, "RECOMMENDED", format!("{} epochs", x.recommended_epochs)))
                        .child(col().gap(6.0).child(text("LEVEL").font(ty::OVERLINE).color(p.text_faint)).child(kit::badge(p, label, tone))),
                )
                .child(text(x.message.clone()).font(ty::BODY).color(p.text_dim).wrap());
        }
        None => {
            let msg = t.summary_error.clone().unwrap_or_else(|| "Reading your recordings...".into());
            card = card.child(text(msg).font(ty::BODY).color(p.text_dim));
        }
    }
    out = out.child(card);
    if let Some(n) = &t.note {
        out = out.child(kit::notice(p, Tone::Accent, n));
    }

    let can_train = sum.is_some_and(|x| level_ok(&x.level));
    if !can_train && sum.is_some() {
        let zero = sum.is_some_and(|x| x.level == "zero_shot");
        let msg = if zero {
            "This is enough for a quick voice you can use right away, but not for training. Record more for the full quality."
        } else {
            "There is not enough audio yet. Record at least a few minutes first."
        };
        out = out.child(
            kit::card(p)
                .gap(12.0)
                .child(text(msg).font(ty::BODY).color(p.text).wrap())
                .child(
                    row()
                        .gap(10.0)
                        .child(kit::button(p, "Go to Record", BtnKind::Primary, true, |s: &mut AppState| s.screen = Screen::Record))
                        .child(kit::button(p, "Go to Speak", BtnKind::Glass, zero, |s: &mut AppState| s.screen = Screen::Speak)),
                ),
        );
        return out;
    }

    // Epochs
    out = out.child(epochs_card(s, p));

    // Estimate
    let epochs = if t.manual { t.epochs } else { sum.map_or(t.epochs, |x| x.recommended_epochs.max(1)) };
    let est = match crate::ui::onboarding::backend(s) {
        Some("cuda") => format!("About {} on an RTX 4070-class GPU.", fmt_dur(epochs as f32 * SECONDS_PER_EPOCH_CUDA)),
        Some("mps") => "Expect several hours on an Apple GPU.".to_string(),
        Some("cpu") => "Expect many hours without a GPU.".to_string(),
        _ => format!("About {} on an RTX 4070-class GPU.", fmt_dur(epochs as f32 * SECONDS_PER_EPOCH_CUDA)),
    };
    out.child(
        col()
            .w_full()
            .items_center()
            .gap(10.0)
            .pt(6.0)
            .child(kit::big_button(p, "Start training", can_train, start))
            .child(text(est).font(ty::CAPTION).color(p.text_dim))
            .child(text("You can keep using the other screens while it trains.").font(ty::CAPTION).color(p.text_faint)),
    )
}

fn epochs_card(s: &AppState, p: &Palette) -> El {
    let t = &s.train;
    let rec = t.summary.as_ref().map_or(0, |(_, x)| x.recommended_epochs);
    let shown = if t.manual { t.epochs } else if rec > 0 { rec } else { t.epochs };

    let seg = |label: &'static str, on: bool, manual: bool| {
        row()
            .key(format!("seg-{label}"))
            .role("button")
            .label(label)
            .h(30.0)
            .px(16.0)
            .center()
            .radius(radius::CONTROL)
            .focusable(true)
            .fill(if on { p.accent } else { rune::Color::TRANSPARENT })
            .hover(p.glass_hover)
            .on_press(move |s: &mut AppState| {
                s.train.manual = manual;
                if !manual {
                    if let Some((_, x)) = &s.train.summary {
                        if x.recommended_epochs > 0 {
                            s.train.epochs = x.recommended_epochs.clamp(EPOCH_MIN, EPOCH_MAX);
                        }
                    }
                }
            })
            .child(text(label).font(ty::LABEL).color(if on { p.on_accent } else { p.text_dim }))
    };
    let toggle = row().gap(2.0).padding(3.0).radius(radius::CONTROL).fill(p.well).border(1.0, p.well_rim)
        .child(seg("Auto", !t.manual, false))
        .child(seg("Custom", t.manual, true));

    let mut card = kit::card(p).gap(14.0).child(
        row()
            .w_full()
            .center_y()
            .child(
                col()
                    .grow(1.0)
                    .gap(2.0)
                    .child(text("Epochs").font(ty::TITLE).color(p.text))
                    .child(text("One epoch is one pass over your recordings. More epochs can sound closer but take longer.").font(ty::CAPTION).color(p.text_dim).wrap()),
            )
            .child(toggle),
    );

    let stepper = |label: &'static str, delta: i32| {
        kit::button(p, label, BtnKind::Glass, true, move |s: &mut AppState| {
            let v = (s.train.epochs as i32 + delta).clamp(EPOCH_MIN as i32, EPOCH_MAX as i32);
            s.train.epochs = v as u32;
        })
    };
    let mut line = row().w_full().gap(10.0).center_y();
    if t.manual {
        line = line
            .child(stepper("-", -(EPOCH_STEP as i32)))
            .child(
                el().w(110.0).h(40.0).center().radius(radius::CONTROL).fill(p.well).border(1.0, p.well_rim).child(
                    text(shown.to_string()).font(rune::TextStyle::new(20.0, 650.0).features("'tnum' 1")).color(p.text),
                ),
            )
            .child(stepper("+", EPOCH_STEP as i32));
        for preset in [100u32, 200, 300, 500] {
            let on = t.epochs == preset;
            line = line.child(
                row()
                    .key(format!("preset-{preset}"))
                    .role("button")
                    .label(format!("{preset} epochs"))
                    .h(28.0)
                    .px(12.0)
                    .center()
                    .radius(radius::CONTROL)
                    .focusable(true)
                    .border(1.0, if on { p.accent } else { p.well_rim })
                    .hover(p.glass_hover)
                    .on_press(move |s: &mut AppState| s.train.epochs = preset)
                    .child(text(preset.to_string()).font(ty::CAPTION).color(if on { p.text } else { p.text_dim })),
            );
        }
    } else {
        line = line
            .child(text(if rec > 0 { rec.to_string() } else { "Auto".into() }).font(rune::TextStyle::new(20.0, 650.0).features("'tnum' 1")).color(p.text))
            .child(text("chosen from the length of your recordings").font(ty::CAPTION).color(p.text_dim));
    }
    card = card.child(line);
    card
}

fn stage_chips(p: &Palette, current: Stage, done: bool) -> El {
    let mut r = row().w_full().gap(8.0).center_y().justify_center().flex_wrap();
    for (i, st) in Stage::ALL.iter().enumerate() {
        let idx = Stage::ALL.iter().position(|x| *x == current).unwrap_or(0);
        let (fill, border, ink) = if done || i < idx {
            (p.ok.alpha(0.16), p.ok.alpha(0.4), p.ok)
        } else if i == idx {
            (p.accent.alpha(0.3), p.accent, p.text)
        } else {
            (rune::Color::TRANSPARENT, p.well_rim, p.text_faint)
        };
        if i > 0 {
            r = r.child(text("\u{2192}").font(ty::CAPTION).color(p.text_faint));
        }
        let mut chip = row().key(format!("stage-{i}")).h(28.0).px(14.0).center().radius(radius::CONTROL).fill(fill).border(1.0, border).gap(6.0);
        if done || i < idx {
            chip = chip.child(text("\u{2713}").font(ty::CAPTION).color(ink));
        }
        r = r.child(chip.child(text(st.label()).font(ty::LABEL).color(ink)));
    }
    r
}

fn progress_view(t: &TrainState, p: &Palette, name: &str) -> El {
    let elapsed = t.started.map_or(t.elapsed_s, |st| st.elapsed().as_secs_f32());
    let starting = t.phase == Phase::Starting;
    let headline = if t.pausing {
        "Pausing...".to_string()
    } else if starting && t.resuming {
        "Resuming...".to_string()
    } else if starting {
        "Starting...".to_string()
    } else {
        match t.stage {
            Stage::Preprocess => "Preparing your recordings".to_string(),
            Stage::Extract => "Analysing your voice".to_string(),
            Stage::Train => "Training your voice".to_string(),
            Stage::Index => "Building the voice index".to_string(),
        }
    };
    let epoch_text = if t.stage == Stage::Train && t.total_epochs > 0 {
        format!("Epoch {} / {}", t.epoch, t.total_epochs)
    } else if t.total_epochs > 0 {
        format!("{} epochs planned", t.total_epochs)
    } else {
        String::new()
    };
    let eta = t.eta_s.map_or("calculating...".to_string(), |e| fmt_dur(e));

    let metrics = row()
        .w_full()
        .gap(14.0)
        .child(stat(p, "PROGRESS", format!("{:.0}%", t.pct)))
        .child(stat(p, "ELAPSED", fmt_dur(elapsed)))
        .child(stat(p, "REMAINING", eta));

    let mut actions = row().w_full().gap(10.0).center_y().justify_end();
    if t.confirm_cancel {
        actions = actions
            .child(text("Stop training? Progress of this run is lost.").font(ty::CAPTION).color(p.bad))
            .child(kit::button(p, "Stop", BtnKind::Danger, true, cancel))
            .child(kit::button(p, "Keep training", BtnKind::Glass, true, |s: &mut AppState| s.train.confirm_cancel = false));
    } else {
        let can_pause = t.phase == Phase::Running && t.stage == Stage::Train && !t.pausing && t.job.is_some();
        let label = if t.pausing { "Pausing..." } else { "Pause" };
        actions = actions
            .child(kit::button(p, label, BtnKind::Glass, can_pause, pause))
            .child(kit::button(p, "Cancel", BtnKind::Glass, true, |s: &mut AppState| s.train.confirm_cancel = true));
    }

    let mut card = kit::panel(p)
        .gap(18.0)
        .child(
            row()
                .w_full()
                .center_y()
                .child(
                    col()
                        .grow(1.0)
                        .gap(4.0)
                        .child(text(headline).font(ty::TITLE).color(p.text))
                        .child(text(format!("Voice: {name}")).font(ty::CAPTION).color(p.text_dim)),
                )
                .child(text(epoch_text).font(rune::TextStyle::new(15.0, 620.0).features("'tnum' 1")).color(p.text)),
        )
        .child(stage_chips(p, t.stage, false))
        .child(kit::progress(p, t.pct / 100.0))
        .child(metrics);
    if let Some(n) = &t.note {
        card = card.child(text(n.clone()).font(ty::CAPTION).color(p.text_dim));
    }
    if t.pausing {
        card = card.child(
            text("Pausing at the next saved checkpoint (a few epochs, up to a minute). Cancel still stops everything.")
                .font(ty::CAPTION)
                .color(p.text_dim)
                .wrap(),
        );
    }
    card = card.child(actions);
    let hint = if t.resuming && t.phase == Phase::Starting {
        "Continuing from the saved checkpoint. Speech is unavailable again while the voice trains."
    } else if t.stage == Stage::Train {
        "Pauses after the current epoch; speech becomes available meanwhile. You can also use the other screens while it trains."
    } else {
        "This can take a while. You can use the other screens in the meantime; training continues in the background. Pause becomes available once learning starts."
    };
    col().w_full().gap(14.0).child(card).child(text(hint).font(ty::CAPTION).color(p.text_faint).wrap())
}

fn fmt_when(iso: &str) -> String {
    // `2026-11-11T17:42:10` -> `2026-11-11 17:42`
    let s = iso.replace('T', " ");
    s.get(..16).unwrap_or(&s).to_string()
}

/// A paused or interrupted training of the active voice: where it stopped and what to do next.
fn paused_view(s: &AppState, p: &Palette, pz: &PausedTraining, name: &str) -> El {
    let t = &s.train;
    let ready = models_ready(s);
    let title = if pz.interrupted { "Training was interrupted" } else { "Training paused" };
    let when = pz.paused_at.as_deref().map(fmt_when);
    let body = if pz.interrupted {
        format!(
            "\"{name}\" was training when the app closed. Progress up to epoch {} of {} was saved; at most a few epochs are repeated.",
            pz.epoch, pz.total_epochs
        )
    } else {
        format!("\"{name}\" is paused at epoch {} of {}.", pz.epoch, pz.total_epochs)
    };
    let pct = overall_pct(pz.epoch, pz.total_epochs);
    let mut card = kit::panel(p)
        .items_center()
        .gap(14.0)
        .child(
            el().size(56.0, 56.0).radius(radius::CARD).center().fill(p.warn.alpha(0.18)).border(1.0, p.warn.alpha(0.5)).child(
                icon_el(&icons::PAUSE, 26.0, p.warn),
            ),
        )
        .child(text(title).font(ty::DISPLAY).color(p.text))
        .child(text(body).font(ty::BODY).color(p.text_dim).wrap())
        .child(kit::progress(p, pct / 100.0))
        .child(
            row()
                .w_full()
                .gap(14.0)
                .child(stat(p, "SAVED AT", format!("Epoch {} / {}", pz.epoch, pz.total_epochs)))
                .child(stat(p, "DONE", format!("{pct:.0}%")))
                .child(stat(p, "WHEN", when.unwrap_or_else(|| "-".into()))),
        );
    if let Some(n) = &t.note {
        card = card.child(kit::notice(p, Tone::Accent, n));
    }
    let mut actions = row().gap(10.0).center_y().justify_center().flex_wrap();
    if t.confirm_discard {
        actions = actions
            .child(text("Discard the paused training? Its progress is lost.").font(ty::CAPTION).color(p.bad))
            .child(kit::button(p, "Discard", BtnKind::Danger, true, discard))
            .child(kit::button(p, "Keep it", BtnKind::Glass, true, |s: &mut AppState| s.train.confirm_discard = false));
    } else {
        actions = actions
            .child(kit::big_button(p, "Resume", ready, resume))
            .child(kit::button(p, "Discard", BtnKind::Ghost, true, |s: &mut AppState| s.train.confirm_discard = true));
    }
    card = card.child(actions);
    let hint = if ready {
        "Speak and Redub work while the training is paused. Resuming takes the GPU back for training."
    } else {
        "The speech models are loading again. Resume is available as soon as they are ready."
    };
    col().w_full().gap(14.0).child(card).child(text(hint).font(ty::CAPTION).color(p.text_faint).wrap())
}

/// The dialog shown when the window is closed while a training runs (`None` otherwise).
pub fn close_overlay(s: &AppState, p: &Palette) -> Option<El> {
    let t = &s.train;
    if t.close == CloseDialog::Hidden {
        return None;
    }
    let can_pause = t.stage == Stage::Train && t.phase == Phase::Running && t.job.is_some();
    let has_ckpt = t.stage == Stage::Train && t.epoch >= 5;
    let mut card = col()
        .key("close-dialog")
        .w(500.0)
        .padding(22.0)
        .gap(12.0)
        .radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.96))
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow);
    if t.close == CloseDialog::Pausing {
        card = card
            .child(text("Pausing before closing...").font(ty::TITLE).color(p.text))
            .child(
                text("Waiting for the next saved checkpoint (a few epochs, up to a minute). The app closes by itself as soon as the training is safely paused.")
                    .font(ty::BODY)
                    .color(p.text_dim)
                    .wrap(),
            )
            .child(
                row()
                    .w_full()
                    .justify_end()
                    .gap(10.0)
                    .child(kit::button(p, "Stay open", BtnKind::Glass, true, |s: &mut AppState| s.train.close = CloseDialog::Hidden))
                    .child(kit::button(p, "Close anyway", BtnKind::Ghost, true, |s: &mut AppState| s.train.quit_requested = true)),
            );
    } else {
        card = card
            .child(text("Training is running").font(ty::TITLE).color(p.text))
            .child(
                text("Closing the app now stops the training. You can pause it first and resume it later from where it stopped.")
                    .font(ty::BODY)
                    .color(p.text_dim)
                    .wrap(),
            );
        if let Some(n) = &t.close_note {
            card = card.child(kit::notice(p, Tone::Warning, n));
        }
        if !can_pause {
            card = card.child(text("Pausing is possible once learning starts; before that nothing is saved yet.").font(ty::CAPTION).color(p.text_faint).wrap());
        }
        let anyway = if has_ckpt { "Close anyway (resume later from the last checkpoint)" } else { "Close anyway (this run starts over)" };
        card = card
            .child(
                col()
                    .w_full()
                    .gap(8.0)
                    .child(kit::big_button(p, "Pause and close", can_pause, |s: &mut AppState| {
                        pause(s);
                        if s.train.pausing {
                            s.train.close = CloseDialog::Pausing;
                            s.train.close_note = None;
                        }
                    }))
                    .child(kit::button(p, "Keep training", BtnKind::Glass, true, |s: &mut AppState| s.train.close = CloseDialog::Hidden))
                    .child(kit::button(p, anyway, BtnKind::Ghost, true, |s: &mut AppState| s.train.quit_requested = true)),
            );
    }
    Some(
        el().key("close-layer")
            .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
            .center()
            .fill(rune::Color::BLACK.alpha(0.55))
            .on_press(|_: &mut AppState| {})
            .child(card),
    )
}

fn done_view(p: &Palette, epochs: u32, elapsed_s: f32, name: &str) -> El {
    kit::panel(p)
        .items_center()
        .gap(14.0)
        .child(
            el().size(56.0, 56.0).radius(radius::CARD).center().fill(p.ok.alpha(0.2)).border(1.0, p.ok.alpha(0.5)).child(
                text("\u{2713}").font(rune::TextStyle::new(26.0, 700.0)).color(p.ok),
            ),
        )
        .child(text("Your voice is ready").font(ty::DISPLAY).color(p.text))
        .child(text(format!("\"{name}\" was trained for {epochs} epochs in {}.", fmt_dur(elapsed_s))).font(ty::BODY).color(p.text_dim).wrap())
        .child(stage_chips(p, Stage::Index, true))
        .child(kit::big_button(p, "Go to Speak", true, |s: &mut AppState| {
            s.train.phase = Phase::Idle;
            s.screen = Screen::Speak;
        }))
        .child(kit::button(p, "Train again", BtnKind::Ghost, true, |s: &mut AppState| {
            s.train.phase = Phase::Idle;
            refresh_summary(s);
        }))
}

fn failed_view(p: &Palette, msg: &str) -> El {
    col()
        .w_full()
        .gap(14.0)
        .child(kit::notice(p, Tone::Danger, msg))
        .child(
            row()
                .gap(10.0)
                .child(kit::button(p, "Try again", BtnKind::Primary, true, |s: &mut AppState| {
                    s.train.phase = Phase::Idle;
                    refresh_summary(s);
                }))
                .child(kit::button(p, "Go to Voices", BtnKind::Glass, true, |s: &mut AppState| s.screen = Screen::Voices)),
        )
}

// ---------------------------------------------------------------------------------------------
// Scenes for headless snapshots
// ---------------------------------------------------------------------------------------------

fn base(s: &mut AppState) {
    s.screen = Screen::Train;
    s.shell.set_engine(shell::EngineStatus::Ready { device: None, tts: Some("ready".into()), rvc: Some("ready".into()) });
    s.shell.voices = vec![shell::VoiceInfo {
        preset: None,
        id: "my-voice".into(),
        name: "My voice".into(),
        dataset_minutes: 6.4,
        parts: 5,
        trained: false,
        trained_epochs: 0,
        paused: None,
        ..Default::default()
    }];
    s.shell.active_voice = Some("my-voice".into());
    s.train = TrainState::default();
    s.train.last_voice = Some("my-voice".into());
    s.train.last_screen = Some(Screen::Train);
    s.train.summary = Some((
        "my-voice".into(),
        Summary {
            total_minutes: 6.4,
            parts: 5,
            level: "trainable".into(),
            message: "Good amount of audio. Training will give a clearly better match than a quick voice.".into(),
            recommended_epochs: 300,
        },
    ));
    s.train.epochs = 300;
    s.train.voice_id = Some("my-voice".into());
    s.onboarding.device = Some(crate::ui::onboarding::DeviceInfo {
        backend: Some("cuda".into()),
        name: Some("NVIDIA GeForce RTX 4070".into()),
        vram_gb: Some(12.0),
        version: None,
        vendor: None,
    });
}

fn scene_idle(s: &mut AppState) {
    base(s);
}
fn scene_custom(s: &mut AppState) {
    base(s);
    s.train.manual = true;
    s.train.epochs = 200;
}
fn scene_short(s: &mut AppState) {
    base(s);
    if let Some((_, x)) = &mut s.train.summary {
        x.total_minutes = 0.9;
        x.parts = 1;
        x.level = "zero_shot".into();
        x.message = "About a minute is enough for a quick voice, not for training.".into();
        x.recommended_epochs = 0;
    }
}
fn scene_no_voice(s: &mut AppState) {
    base(s);
    s.shell.voices.clear();
    s.shell.active_voice = None;
    s.train.last_voice = None;
    s.train.summary = None;
}
fn running(s: &mut AppState, stage: Stage, epoch: u32, pct: f32, eta: Option<f32>, elapsed: f32) {
    base(s);
    let t = &mut s.train;
    t.phase = Phase::Running;
    t.stage = stage;
    t.epoch = epoch;
    t.total_epochs = 300;
    t.pct = pct;
    t.eta_s = eta;
    t.elapsed_s = elapsed;
}
fn scene_preprocess(s: &mut AppState) {
    running(s, Stage::Preprocess, 0, 4.0, None, 35.0);
}
fn scene_running(s: &mut AppState) {
    running(s, Stage::Train, 118, 41.0, Some(2010.0), 1380.0);
}
fn scene_cancel_confirm(s: &mut AppState) {
    scene_running(s);
    s.train.confirm_cancel = true;
}
fn scene_done(s: &mut AppState) {
    base(s);
    s.train.phase = Phase::Done { epochs: 300, elapsed_s: 3310.0 };
    s.train.pct = 100.0;
}
fn scene_error(s: &mut AppState) {
    base(s);
    s.train.phase = Phase::Failed("CUDA out of memory while training. Close other programs that use the GPU and try again.".into());
}

fn paused_at(s: &mut AppState, epoch: u32, interrupted: bool) {
    s.shell.voices[0].paused = Some(PausedTraining {
        epoch,
        total_epochs: 300,
        paused_at: Some("2026-11-11T17:42:10".into()),
        interrupted,
    });
}
fn scene_pausing(s: &mut AppState) {
    scene_running(s);
    s.train.pausing = true;
    s.train.job = Some("job-4".into());
}
fn scene_paused(s: &mut AppState) {
    base(s);
    paused_at(s, 115, false);
    s.train.note = Some("Training paused at epoch 115 of 300. Speech is available again as soon as the models have loaded.".into());
}
fn scene_paused_loading(s: &mut AppState) {
    scene_paused(s);
    s.shell.set_engine(shell::EngineStatus::Ready { device: None, tts: Some("loading".into()), rvc: Some("loading".into()) });
}
fn scene_discard_confirm(s: &mut AppState) {
    scene_paused(s);
    s.train.note = None;
    s.train.confirm_discard = true;
}
fn scene_interrupted(s: &mut AppState) {
    base(s);
    paused_at(s, 105, true);
}
fn scene_resuming(s: &mut AppState) {
    base(s);
    paused_at(s, 115, false);
    let t = &mut s.train;
    t.phase = Phase::Starting;
    t.resuming = true;
    t.stage = Stage::Train;
    t.epoch = 115;
    t.total_epochs = 300;
    t.pct = overall_pct(115, 300);
    t.elapsed_s = 6.0;
}
fn scene_close(s: &mut AppState) {
    scene_running(s);
    s.train.job = Some("job-4".into());
    s.train.close = CloseDialog::Ask;
}
fn scene_close_early(s: &mut AppState) {
    scene_preprocess(s);
    s.train.close = CloseDialog::Ask;
}
fn scene_close_pausing(s: &mut AppState) {
    scene_pausing(s);
    s.train.close = CloseDialog::Pausing;
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("train-pausing", scene_pausing),
        ("train-paused", scene_paused),
        ("train-paused-loading", scene_paused_loading),
        ("train-discard", scene_discard_confirm),
        ("train-interrupted", scene_interrupted),
        ("train-resuming", scene_resuming),
        ("train-close", scene_close),
        ("train-close-early", scene_close_early),
        ("train-close-pausing", scene_close_pausing),
        ("train-idle", scene_idle),
        ("train-custom", scene_custom),
        ("train-short", scene_short),
        ("train-novoice", scene_no_voice),
        ("train-preprocess", scene_preprocess),
        ("train-running", scene_running),
        ("train-cancel", scene_cancel_confirm),
        ("train-done", scene_done),
        ("train-error", scene_error),
    ]
}
