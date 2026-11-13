//! The shared mini player: one track plays at a time, from any screen, and a compact glass bar docked at the bottom
//! of the window shows it (title, time, play/pause, stop, seek, loop, speed, close).
//!
//! Screens start sound through [`play`] / [`toggle`] (a file) or [`play_live`] / [`enqueue`] / [`finish_live`]
//! (Speak's streaming playback) and draw their own play buttons from [`playing`], so a card's button and the bar
//! always agree: pressing a card's button again pauses, and once more resumes. Only the shell's audio player makes
//! sound (see `audio::Player`); this module keeps what the UI shows.
//!
//! Event-driven: the audio thread reports the position about [`crate::audio::PROGRESS_HZ`] times a second while
//! sound plays (through `events::Latest`, so a busy UI thread never queues reports up) and once when a track ends;
//! nothing is reported while paused or stopped, and nothing here polls.
//!
//! * Speed (0.5x..2x) keeps the pitch: WSOLA time-stretch in the audio callback (`audio::stretch`); a change applies
//!   within one 16 ms frame, without a reset.
//! * Live playback (Speak, while the text is still being generated): the bar shows what has arrived so far and
//!   seeking is off until the job ends (the end of the arrived audio moves, and a jump past it would only wait).
//!   Loop applies once the job is done. When it ends the track becomes the finished take (its card shows it playing).
//! * After a track plays out the bar stays (play starts it again); Stop and Close end playback and hide the bar.
//! * Keyboard: Space toggles play/pause while focus is in the bar (on Record's session it stays the record key);
//!   Left/Right on the focused seek bar jump 5 s.

use super::components::{Tone, accent_glass, glass_edge, icon_el, icons};
use super::shell;
use super::theme::{Palette, motion, radius, ty};
use crate::state::{AppState, Screen};
use rune::{Align, Color, Drag, El, Key, Phase, Side, col, el, row, text};
use std::path::{Path, PathBuf};

/// The speeds offered.
pub const SPEEDS: [f32; 6] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

/// Seconds the arrow keys move on the seek bar.
const STEP_S: f32 = 5.0;

/// Where a track was started (stopping "my" playback, the bar's caption).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Speak,
    Redub,
    Record,
    Voices,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Playing,
    Paused,
    Ended,
}

/// The track the bar shows.
#[derive(Clone, Debug)]
pub struct Now {
    /// The file; for a live stream its first chunk until [`finish_live`] names the finished take.
    pub path: PathBuf,
    pub title: String,
    /// The caption under the title ("Speak · result").
    pub what: String,
    pub source: Source,
    pub status: Status,
    pub pos_s: f32,
    pub len_s: f32,
    /// Still being generated (Speak): the length grows, seeking is off.
    pub live: bool,
    /// The audio player's id of this track (reports about another one are ignored).
    track: u64,
}

pub struct PlaybarState {
    pub now: Option<Now>,
    /// A track just started: (its player id, app clock deadline). If it has not moved by then, the output device is
    /// stale (the PC slept, the device changed) and the player is reopened once (`settle`).
    watch: Option<(u64, f64)>,
    /// The output was reopened for this track already (no second retry).
    retried: Option<u64>,
    pub looped: bool,
    pub speed: f32,
    /// The speed menu is open.
    pub speed_menu: bool,
}

impl Default for PlaybarState {
    fn default() -> Self {
        PlaybarState { now: None, watch: None, retried: None, looped: false, speed: 1.0, speed_menu: false }
    }
}

// -------------------------------------------------------------------------------------------------
// Actions (the guarded way to the audio player)
// -------------------------------------------------------------------------------------------------

fn player_call<T>(s: &AppState, f: impl FnOnce(&crate::audio::Player) -> anyhow::Result<T>) -> Result<T, String> {
    let p = s.player.as_ref().ok_or_else(|| shell::audio_unavailable(s))?;
    shell::guard(|| f(p)).map_err(|_| "Audio playback is not available.".to_owned())?.map_err(|e| format!("Could not play audio: {e}"))
}

fn player_do(s: &AppState, f: impl FnOnce(&crate::audio::Player)) {
    if let Some(p) = &s.player {
        let _ = shell::guard(|| f(p));
    }
}

fn start(s: &mut AppState, path: &Path, title: String, what: String, source: Source, live: bool) -> Result<(), String> {
    // A stream that reported an error (the device went away) cannot play: open the output again first.
    if s.player.as_ref().is_some_and(|p| shell::guard(|| p.error()).ok().flatten().is_some()) {
        shell::restart_audio(s);
    }
    let (looped, speed) = (s.playbar.looped, s.playbar.speed);
    let (track, len_s) = player_call(s, |p| {
        p.set_loop(looped);
        p.set_speed(speed);
        p.play(path, live)
    })?;
    s.playbar.now = Some(Now { path: path.to_path_buf(), title, what, source, status: Status::Playing, pos_s: 0.0, len_s, live, track });
    s.playbar.watch = Some((track, shell::clock_now(s) + STALL_S));
    Ok(())
}

/// How long a started track may stay at 0:00 before the output is taken for stale.
const STALL_S: f64 = 1.5;

/// The stall check of a just-started track (a one-shot deadline handed to `Tx::wake_at`, cleared by its first
/// progress report): a stale output device is reopened and the track started again, once.
pub fn settle(s: &mut AppState, tx: &mut rune::Tx) -> bool {
    let Some((track, due)) = s.playbar.watch else { return false };
    let now = shell::clock_now(s);
    if now < due {
        tx.wake_at(due);
        return false;
    }
    s.playbar.watch = None;
    let Some(n) = s.playbar.now.clone().filter(|n| n.track == track && n.status == Status::Playing && n.pos_s <= 0.0) else {
        return false;
    };
    if s.playbar.retried == Some(track) {
        shell::toast(s, Tone::Danger, "The audio output does not respond. Check the sound device in the system settings.");
        return true;
    }
    shell::restart_audio(s);
    let source = n.source;
    match start(s, &n.path, n.title.clone(), n.what.clone(), source, n.live) {
        Ok(()) => {
            if let Some(m) = &s.playbar.now {
                s.playbar.retried = Some(m.track);
            }
        }
        Err(e) => shell::toast(s, Tone::Danger, e),
    }
    true
}

/// Stops what plays and plays `path`. `title` names it in the bar, `what` says what it is ("Redub · original").
pub fn play(s: &mut AppState, path: &Path, title: impl Into<String>, what: impl Into<String>, source: Source) -> Result<(), String> {
    start(s, path, title.into(), what.into(), source, false)
}

/// A screen's play button: plays `path`, or pauses / resumes it when it is the track loaded already.
pub fn toggle(s: &mut AppState, path: &Path, title: impl Into<String>, what: impl Into<String>, source: Source) -> Result<(), String> {
    if s.playbar.now.as_ref().is_some_and(|n| n.path == path && !n.live) {
        toggle_pause(s);
        return Ok(());
    }
    play(s, path, title, what, source)
}

/// Whether `path` is the track loaded and audible (not paused, not ended): its play button shows "pause".
pub fn playing(s: &AppState, path: &Path) -> bool {
    s.playbar.now.as_ref().is_some_and(|n| n.status == Status::Playing && n.path == path)
}

/// Starts Speak's streaming playback with its first chunk.
pub fn play_live(s: &mut AppState, first: &Path, title: impl Into<String>) -> Result<(), String> {
    start(s, first, title.into(), "Speak · playing as it is generated".into(), Source::Speak, true)
}

/// Appends a chunk to the live track.
pub fn enqueue(s: &mut AppState, path: &Path) -> Result<(), String> {
    if !s.playbar.now.as_ref().is_some_and(|n| n.live) {
        return Err("Nothing is streaming.".into());
    }
    let len_s = player_call(s, |p| p.enqueue(path))?;
    if let Some(n) = s.playbar.now.as_mut() {
        n.len_s = len_s;
        if n.status == Status::Ended {
            n.status = Status::Playing;
        }
    }
    Ok(())
}

/// The live job ended: the track is complete. With `take` (the finished take's playback file and title) the bar and
/// the take's card show it as that take.
pub fn finish_live(s: &mut AppState, take: Option<(&Path, String)>) {
    if !s.playbar.now.as_ref().is_some_and(|n| n.live) {
        return;
    }
    player_do(s, |p| p.set_live(false));
    let len = s.player.as_ref().and_then(|p| shell::guard(|| p.duration_s()).ok());
    if let Some(n) = s.playbar.now.as_mut() {
        n.live = false;
        n.what = "Speak · result".into();
        if let Some(l) = len {
            n.len_s = l;
        }
        if let Some((path, title)) = take {
            n.path = path.to_path_buf();
            n.title = title;
        }
    }
}

pub fn pause(s: &mut AppState) {
    if !s.playbar.now.as_ref().is_some_and(|n| n.status == Status::Playing) {
        return;
    }
    player_do(s, |p| p.pause());
    let pos = s.player.as_ref().and_then(|p| shell::guard(|| p.position_s()).ok());
    if let Some(n) = s.playbar.now.as_mut() {
        n.status = Status::Paused;
        if let Some(pos) = pos {
            n.pos_s = pos;
        }
    }
}

/// Continues a paused track; an ended one starts again from the beginning.
pub fn resume(s: &mut AppState) {
    let Some(status) = s.playbar.now.as_ref().map(|n| n.status) else { return };
    if status == Status::Playing {
        return;
    }
    let loaded = s.player.as_ref().is_some_and(|p| shell::guard(|| p.resume()).unwrap_or(false));
    if !loaded && !s.shell.scripted() {
        // The audio player lost the track (no device any more): nothing to resume.
        s.playbar.now = None;
        return;
    }
    if let Some(n) = s.playbar.now.as_mut() {
        if n.status == Status::Ended {
            n.pos_s = 0.0;
        }
        n.status = Status::Playing;
    }
}

pub fn toggle_pause(s: &mut AppState) {
    match s.playbar.now.as_ref().map(|n| n.status) {
        Some(Status::Playing) => pause(s),
        Some(_) => resume(s),
        None => {}
    }
}

/// Stops playback and hides the bar.
pub fn stop(s: &mut AppState) {
    player_do(s, |p| p.stop());
    s.playbar.now = None;
    s.playbar.speed_menu = false;
}

/// The bar's Stop: halts the track and rewinds it to the start, the bar stays (play starts it again from 0:00).
/// Live Speak playback cannot be rewound while it is still arriving: there Stop ends it, as Close does.
fn stop_rewind(s: &mut AppState) {
    if s.playbar.now.as_ref().is_none_or(|n| n.live) {
        stop(s);
        return;
    }
    if s.playbar.now.as_ref().is_some_and(|n| n.status == Status::Playing) {
        pause(s);
    }
    seek(s, 0.0);
    if let Some(n) = s.playbar.now.as_mut() {
        n.status = Status::Paused;
        n.pos_s = 0.0;
    }
}

/// Stops playback started from `source` (leaves anything else playing).
pub fn stop_from(s: &mut AppState, source: Source) {
    if s.playbar.now.as_ref().is_some_and(|n| n.source == source) {
        stop(s);
    }
}

/// Stops playback when one of `paths` is loaded (it is about to be deleted).
pub fn stop_path(s: &mut AppState, paths: &[&Path]) {
    if s.playbar.now.as_ref().is_some_and(|n| paths.contains(&n.path.as_path())) {
        stop(s);
    }
}

/// Jumps to `pos_s` seconds (not while live).
pub fn seek(s: &mut AppState, pos_s: f32) {
    let Some(n) = s.playbar.now.as_ref().filter(|n| !n.live) else { return };
    let pos = pos_s.clamp(0.0, n.len_s);
    player_do(s, |p| p.seek(pos));
    if let Some(n) = s.playbar.now.as_mut() {
        n.pos_s = pos;
        if n.status == Status::Ended {
            n.status = Status::Paused;
        }
    }
}

pub fn set_loop(s: &mut AppState, on: bool) {
    s.playbar.looped = on;
    player_do(s, |p| p.set_loop(on));
}

pub fn set_speed(s: &mut AppState, speed: f32) {
    s.playbar.speed = speed;
    s.playbar.speed_menu = false;
    player_do(s, |p| p.set_speed(speed));
}

// -------------------------------------------------------------------------------------------------
// Events from the audio thread
// -------------------------------------------------------------------------------------------------

/// The player sink: positions go through a coalescing feed, the end of a track as its own event.
pub fn sink() -> crate::audio::PlaySink {
    let progress = super::events::Latest::new(|s: &mut AppState, (track, pos, len): (u64, f32, f32)| on_progress(s, track, pos, len));
    std::sync::Arc::new(move |ev| match ev {
        crate::audio::PlayEvent::Progress { track, pos_s, len_s } => progress.send((track, pos_s, len_s)),
        crate::audio::PlayEvent::Ended { track } => super::events::post_if(move |s| on_ended(s, track)),
    })
}

fn on_progress(s: &mut AppState, track: u64, pos_s: f32, len_s: f32) -> bool {
    let Some(n) = s.playbar.now.as_mut().filter(|n| n.track == track && n.status == Status::Playing) else { return false };
    let changed = (n.pos_s - pos_s).abs() > 1e-3 || (n.len_s - len_s).abs() > 1e-3;
    n.pos_s = pos_s;
    n.len_s = len_s;
    if pos_s > 0.0 && s.playbar.watch.is_some_and(|(t, _)| t == track) {
        s.playbar.watch = None; // it plays: the output is alive
    }
    changed
}

fn on_ended(s: &mut AppState, track: u64) -> bool {
    let Some(n) = s.playbar.now.as_mut().filter(|n| n.track == track) else { return false };
    n.status = Status::Ended;
    n.pos_s = n.len_s;
    true
}

/// Space while focus is in the bar: play/pause, except on Record's session, where Space records.
fn on_key(s: &mut AppState, ev: &rune::KeyEvent, tx: &mut rune::Tx) -> bool {
    if !matches!(&ev.key, Key::Text(t) if t == " ") || ev.mods.cmd || ev.mods.ctrl || ev.mods.alt {
        return false;
    }
    if s.screen == Screen::Record && super::record::on_key(s, ev, tx) {
        return true;
    }
    toggle_pause(s);
    true
}

// -------------------------------------------------------------------------------------------------
// View
// -------------------------------------------------------------------------------------------------

/// Height of the docked bar (without the margin under it).
pub const BAR_H: f32 = 56.0;
/// Room the dock takes at the bottom of the window (the toasts move up by this).
pub const DOCK_H: f32 = BAR_H + 12.0;

fn clock(secs: f32) -> String {
    let t = secs.max(0.0).floor() as u32;
    format!("{}:{:02}", t / 60, t % 60)
}

fn speed_label(v: f32) -> String {
    let t = format!("{v:.2}");
    let t = t.trim_end_matches('0').trim_end_matches('.');
    format!("{t}×")
}

/// A round 32 pt control of the bar.
fn round_button(p: Palette, key: &str, glyph: &'static rune::Icon, label: &str, on: bool, f: impl Fn(&mut AppState) + 'static) -> El {
    let b = el().key(key.to_owned()).role("button").label(label.to_owned()).help(label.to_owned()).size(32.0, 32.0).radius(16.0).center()
        .focusable(true).motion(motion::UI).on_press(f);
    let b = if on {
        b.fill(p.accent.alpha(0.2)).border(1.0, p.accent.alpha(0.45)).hover(p.accent.alpha(0.28))
    } else {
        b.hover(p.glass_hover).press(p.glass_press)
    };
    b.child(icon_el(glyph, 15.0, if on { p.accent } else { p.text_dim }))
}

/// The docked bar, while a track is loaded: a full-width strip under the screen (the screen gets shorter, so it
/// never covers the Speak / Redub action bars or anything else).
pub fn view(s: &AppState, p: Palette) -> Option<El> {
    let n = s.playbar.now.as_ref()?;
    let on = n.status == Status::Playing;
    let play = el().key("pb-play").role("button").label(if on { "Pause" } else { "Play" }).help(if on { "Pause (Space)" } else { "Play (Space)" })
        .size(38.0, 38.0).radius(19.0).center().focusable(true).motion(motion::UI)
        .hover(Color::WHITE.alpha(0.1))
        .on_press(toggle_pause)
        .child(icon_el(if on { &icons::PAUSE } else { &icons::PLAY }, 16.0, p.on_accent));
    let play = accent_glass(p, play);
    let stop_b = round_button(p, "pb-stop", &icons::STOP, "Stop (back to the start)", false, stop_rewind);
    let title = col().key("pb-title").w(230.0).min_w(0.0).gap(1.0)
        .child(text(n.title.clone()).font(ty::LABEL).color(p.text).truncate())
        .child(text(n.what.clone()).font(ty::CAPTION).color(p.text_faint).truncate());
    // The total rounds like the result cards' lengths (a 3.8 s take is "0:04" on its card and here).
    let total = if n.live { format!("{}…", clock(n.len_s.round())) } else { clock(n.len_s.round()) };
    let looped = s.playbar.looped;
    let loop_b = round_button(p, "pb-loop", &LOOP, if looped { "Loop: on" } else { "Loop: off" }, looped, move |s| set_loop(s, !looped)).checked(looped);
    let speed = s.playbar.speed;
    let open = s.playbar.speed_menu;
    let chip = row().key("pb-speed").role("button").label(format!("Speed {}", speed_label(speed))).help("Playback speed (the pitch stays)")
        .h(30.0).px(10.0).gap(5.0).center().radius(15.0).focusable(true).motion(motion::UI)
        .fill(if open || speed != 1.0 { p.accent.alpha(0.16) } else { p.glass })
        .border(1.0, if speed != 1.0 { p.accent.alpha(0.4) } else { p.rim_top.alpha(0.25) })
        .hover(p.glass_hover)
        .on_press(|s: &mut AppState| s.playbar.speed_menu = !s.playbar.speed_menu)
        .child(text(speed_label(speed)).font(ty::NUM).color(if speed != 1.0 { p.accent } else { p.text }))
        .child(icon_el(&icons::CHEVRON_DOWN, 11.0, p.text_dim));
    let close = round_button(p, "pb-close", &icons::CLOSE, "Close the player", false, stop);
    // instant_changes: the position moves 20 times a second; animating each move with the layout springs kept the
    // window redrawing (and its blurs) at the display's full rate for as long as anything played.
    let bar = row().key("playbar").role("group").label("Player").instant_changes().w_full().h(BAR_H).pl(9.0).pr(10.0).gap(10.0).center_y()
        .radius(radius::PANEL + 4.0).continuous_corners()
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.9))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(10.0, 30.0, p.shadow)
        .on_key(on_key)
        .child(play)
        .child(stop_b)
        .child(title)
        .child(text(clock(n.pos_s)).font(ty::NUM).color(p.text_dim))
        .child(seek_bar(p, n))
        .child(text(total).font(ty::NUM).color(p.text_dim))
        .child(loop_b)
        .child(chip)
        .child(close);
    Some(row().key("playbar-dock").w_full().px(16.0).pb(12.0).child(glass_edge(p, bar)))
}

/// The seek bar: click or drag to jump; Left/Right when focused. Off (dimmed) while the track is live.
fn seek_bar(p: Palette, n: &Now) -> El {
    const TRACK: f32 = 5.0;
    const KNOB: f32 = 14.0;
    let frac = if n.len_s > 0.0 { (n.pos_s / n.len_s).clamp(0.0, 1.0) } else { 0.0 };
    let len = n.len_s;
    let knob = el().key("pb-knob").size(KNOB, KNOB).pin(None, Some((TRACK - KNOB) / 2.0), Some(-KNOB / 2.0), None).radius(KNOB / 2.0)
        .fill(Color::WHITE.alpha(0.85)).border(1.0, p.accent.alpha(0.6)).shadow(2.0, 6.0, Color::BLACK.alpha(0.5));
    let fill = el().key("pb-fill").h(TRACK).w_frac(frac).radius(TRACK / 2.0).gradient(p.accent_top, p.accent)
        .children((!n.live).then_some(knob));
    let track = el().key("pb-track").w_full().h(TRACK).radius(TRACK / 2.0).fill(p.well).border(1.0, p.well_rim).child(fill);
    let hit = col().key("pb-seek").role("slider").label("Position").w_full().h(28.0).px(KNOB / 2.0).justify_center().grow(1.0).min_w(60.0)
        .range_value(f64::from(n.pos_s), 0.0, f64::from(len.max(0.01)))
        .a11y_value(format!("{} of {}", clock(n.pos_s), clock(len)))
        .child(track);
    if n.live {
        return hit.opacity(0.55).help("Seeking works once the whole text is generated");
    }
    let pos = n.pos_s;
    hit.focusable(true).cursor(rune::Cursor::Pointer)
        .on_step(move |s: &mut AppState, steps, _| seek(s, pos + steps as f32 * STEP_S))
        .on_drag(move |s: &mut AppState, d: &mut Drag, tx| {
            if d.cancelled() || d.phase == Phase::Up {
                return;
            }
            if d.phase == Phase::Move {
                tx.instant();
            }
            let inner = (d.w - KNOB).max(1.0);
            seek(s, ((d.x - KNOB / 2.0) / inner).clamp(0.0, 1.0) * len);
        })
}

/// The speed menu above its chip (drawn at the window root, over everything).
pub fn overlay(s: &AppState, p: Palette) -> Option<El> {
    if !s.playbar.speed_menu || s.playbar.now.is_none() {
        return None;
    }
    let scrim = el().key("pb-speed-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(|s: &mut AppState| s.playbar.speed_menu = false);
    let mut list = col().key("pb-speed-list").role("menu").w_full().gap(2.0).focus_group();
    for v in SPEEDS {
        let on = (s.playbar.speed - v).abs() < 1e-3;
        list = list.child(
            row().key(format!("pb-speed-{v}")).role("menuitem").label(speed_label(v)).checked(on).w_full().h(30.0).px(10.0).gap(8.0).center_y().radius(15.0)
                .fill(if on { p.accent.alpha(0.14) } else { Color::TRANSPARENT }).hover(p.glass_hover).focusable(true)
                .on_press(move |s: &mut AppState| set_speed(s, v))
                .child(text(speed_label(v)).font(ty::NUM).color(p.text).grow(1.0))
                .child(el().w(14.0).children(on.then(|| icon_el(&icons::CHECK, 13.0, p.accent)))),
        );
    }
    let panel = col().key("pb-speed-menu").w(112.0).padding(6.0).gap(4.0).radius(radius::CARD + 2.0)
        .fill(rune::oklcha(0.21, 0.03, 255.0, 0.98))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(14.0, 40.0, p.shadow)
        .anchored("pb-speed", Side::Top, Align::Center, 8.0)
        .transition(rune::Transition::scale(0.98))
        .child(text("Speed").font(ty::CAPTION).color(p.text_faint).px(10.0).pt(2.0))
        .child(list);
    Some(el().key("pb-speed-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).child(scrim).child(panel))
}

/// Loop: two arrows chasing each other round a rounded rectangle.
static LOOP: rune::Icon = rune::Icon {
    name: "loop",
    view: 16.0,
    body: r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.8 7.4V6.6A2.4 2.4 0 0 1 5.2 4.2H12.4M10.6 2.4L12.4 4.2L10.6 6"/><path d="M13.2 8.6V9.4A2.4 2.4 0 0 1 10.8 11.8H3.6M5.4 10L3.6 11.8L5.4 13.6"/></g>"#,
};

// -------------------------------------------------------------------------------------------------
// Headless scenes
// -------------------------------------------------------------------------------------------------

/// Scene helper: show `path` in the bar as if it were loaded.
pub fn scene_now(s: &mut AppState, path: &str, title: &str, what: &str, source: Source, status: Status, pos_s: f32, len_s: f32) {
    s.playbar.now = Some(Now { path: path.into(), title: title.into(), what: what.into(), source, status, pos_s, len_s, live: false, track: 0 });
}

/// Scene helper: Speak's live stream in the bar.
pub fn scene_live(s: &mut AppState, title: &str, pos_s: f32, len_s: f32) {
    scene_now(s, "/tmp/job1/chunk0.wav", title, "Speak · playing as it is generated", Source::Speak, Status::Playing, pos_s, len_s);
    if let Some(n) = s.playbar.now.as_mut() {
        n.live = true;
    }
}

const TAKE3: &str = "Welcome back. In a few seconds this paragraph will be read in my own voice.";

fn scene_playing(s: &mut AppState) {
    super::speak::scene_done(s);
    scene_now(s, "/tmp/out3.wav", TAKE3, "Speak · result", Source::Speak, Status::Playing, 5.2, 14.2);
}

fn scene_paused(s: &mut AppState) {
    super::speak::scene_done(s);
    scene_now(s, "/tmp/out2.wav", "The quick brown fox jumps over the lazy dog.", "Speak · result", Source::Speak, Status::Paused, 1.6, 3.8);
    shell::toast(s, Tone::Success, "Exported as MP3");
}

fn scene_looped(s: &mut AppState) {
    scene_playing(s);
    s.playbar.looped = true;
    if let Some(n) = s.playbar.now.as_mut() {
        n.pos_s = 11.6;
    }
}

fn scene_speed(s: &mut AppState) {
    scene_playing(s);
    s.playbar.speed = 1.5;
}

fn scene_speed_menu(s: &mut AppState) {
    scene_speed(s);
    s.playbar.speed_menu = true;
}

fn scene_live_bar(s: &mut AppState) {
    super::speak::scene_generating(s);
    scene_live(s, "Welcome back. In a few seconds this paragraph will be read in my own voice, one sentence…", 3.4, 9.0);
}

fn scene_record(s: &mut AppState) {
    super::record::scene_notice_saved(s);
    scene_now(s, "/tmp/part1.wav", "Part 1 take", "Record", Source::Record, Status::Playing, 2.0, 21.0);
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("player_playing", scene_playing),
        ("player_paused", scene_paused),
        ("player_looped", scene_looped),
        ("player_speed", scene_speed),
        ("player_speed_menu", scene_speed_menu),
        ("player_live", scene_live_bar),
        ("player_record", scene_record),
    ]
}
