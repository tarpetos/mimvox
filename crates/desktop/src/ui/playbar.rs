//! The shared mini player: one track plays at a time, from any screen.

use super::components::{Tone, accent_glass, glass_edge, icon_el, icons};
use super::shell;
use super::theme::{Palette, motion, radius, ty};
use crate::state::{AppState, Screen};
use rune::{Align, Color, Drag, El, Key, Phase, Side, col, el, row, text};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Acquire, Ordering::Relaxed, Ordering::Release, fence};

pub const SPEEDS: [f32; 6] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0];

const STEP_S: f32 = 5.0;

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

#[derive(Clone, Debug)]
pub struct Now {
    /// The file; for a live stream its first chunk until [`finish_live`] names the finished take.
    pub path: PathBuf,
    pub title: String,
    pub what: String,
    pub source: Source,
    pub status: Status,
    pub pos_s: f32,
    pub len_s: f32,
    pub live: bool,
    /// While `live`: the part of the text generated so far (0..1), `None` when not known.
    pub loaded: Option<f32>,
    /// The audio player's id of this track (reports about another one are ignored).
    track: u64,
}

pub struct PlaybarState {
    pub now: Option<Now>,
    /// (player id, deadline): if the track has not moved by then, the output device is stale (the PC slept, the
    /// device changed) and the player is reopened once.
    watch: Option<(u64, f64)>,
    retried: Option<u64>,
    /// The seek knob is being dragged: whether the track was playing when it was grabbed.
    scrub: Option<bool>,
    pub looped: bool,
    pub speed: f32,
    pub speed_menu: bool,
}

impl Default for PlaybarState {
    fn default() -> Self {
        PlaybarState {
            now: None,
            watch: None,
            retried: None,
            scrub: None,
            looped: false,
            speed: 1.0,
            speed_menu: false,
        }
    }
}

fn player_call<T>(s: &AppState, f: impl FnOnce(&crate::audio::Player) -> anyhow::Result<T>) -> Result<T, String> {
    let p = s.player.as_ref().ok_or_else(|| shell::audio_unavailable(s))?;
    shell::guard(|| f(p))
        .map_err(|_| "Audio playback is not available.".to_owned())?
        .map_err(|e| format!("Could not play audio: {e}"))
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
    s.playbar.now = Some(Now {
        path: path.to_path_buf(),
        title,
        what,
        source,
        status: Status::Playing,
        pos_s: 0.0,
        len_s,
        live,
        loaded: None,
        track,
    });
    s.playbar.watch = Some((track, shell::clock_now(s) + STALL_S));
    jump(track, 0.0, len_s);
    Ok(())
}

/// How long a started track may stay at 0:00 before the output is taken for stale.
const STALL_S: f64 = 1.5;

pub fn settle(s: &mut AppState, tx: &mut rune::Tx) -> bool {
    glide(s, tx);
    let Some((track, due)) = s.playbar.watch else { return false };
    let now = shell::clock_now(s);
    if now < due {
        tx.wake_at(due);
        return false;
    }
    s.playbar.watch = None;
    let Some(n) = s.playbar.now.clone().filter(|n| n.track == track && n.status == Status::Playing && n.pos_s <= 0.0)
    else {
        return false;
    };
    if s.playbar.retried == Some(track) {
        shell::toast(
            s,
            Tone::Danger,
            "The audio output does not respond. Check the sound device in the system settings.",
        );
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

pub fn play(
    s: &mut AppState,
    path: &Path,
    title: impl Into<String>,
    what: impl Into<String>,
    source: Source,
) -> Result<(), String> {
    start(s, path, title.into(), what.into(), source, false)
}

pub fn toggle(
    s: &mut AppState,
    path: &Path,
    title: impl Into<String>,
    what: impl Into<String>,
    source: Source,
) -> Result<(), String> {
    if s.playbar.now.as_ref().is_some_and(|n| n.path == path && !n.live) {
        toggle_pause(s);
        return Ok(());
    }
    play(s, path, title, what, source)
}

/// Loaded and audible (not paused, not ended).
pub fn playing(s: &AppState, path: &Path) -> bool {
    s.playbar.now.as_ref().is_some_and(|n| n.status == Status::Playing && n.path == path)
}

pub fn play_live(s: &mut AppState, first: &Path, title: impl Into<String>) -> Result<(), String> {
    start(s, first, title.into(), "Speak · playing as it is generated".into(), Source::Speak, true)
}

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

/// While streaming: how much of the text has been generated (0..1), `None` when not known. It only grows.
pub fn loading(s: &mut AppState, part: Option<f32>) {
    if let Some(n) = s.playbar.now.as_mut().filter(|n| n.live) {
        n.loaded = match (n.loaded, part) {
            (Some(was), Some(now)) => Some(was.max(now.clamp(0.0, 1.0))),
            (was, now) => now.map(|v| v.clamp(0.0, 1.0)).or(was),
        };
    }
}

/// With `take` (the finished take's playback file and title) the bar shows the track as that take. The length is
/// final now, so the knob appears where the sound is.
pub fn finish_live(s: &mut AppState, take: Option<(&Path, String)>) {
    if !s.playbar.now.as_ref().is_some_and(|n| n.live) {
        return;
    }
    player_do(s, |p| p.set_live(false));
    let len = s.player.as_ref().and_then(|p| shell::guard(|| p.duration_s()).ok());
    let pos = s.player.as_ref().and_then(|p| shell::guard(|| p.position_s()).ok());
    if let Some(n) = s.playbar.now.as_mut() {
        n.live = false;
        n.loaded = None;
        n.what = "Speak · result".into();
        if let Some(l) = len {
            n.len_s = l;
        }
        if let Some(p) = pos {
            n.pos_s = p.min(n.len_s);
        }
        if let Some((path, title)) = take {
            n.path = path.to_path_buf();
            n.title = title;
        }
        if n.len_s > 0.0 {
            place(n.pos_s / n.len_s, true);
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

/// An ended track starts again from the beginning.
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
        jump(n.track, n.pos_s, n.len_s);
    }
}

pub fn toggle_pause(s: &mut AppState) {
    match s.playbar.now.as_ref().map(|n| n.status) {
        Some(Status::Playing) => pause(s),
        Some(_) => resume(s),
        None => {}
    }
}

pub fn stop(s: &mut AppState) {
    player_do(s, |p| p.stop());
    s.playbar.now = None;
    s.playbar.speed_menu = false;
}

pub fn stop_from(s: &mut AppState, source: Source) {
    if s.playbar.now.as_ref().is_some_and(|n| n.source == source) {
        stop(s);
    }
}

pub fn stop_path(s: &mut AppState, paths: &[&Path]) {
    if s.playbar.now.as_ref().is_some_and(|n| paths.contains(&n.path.as_path())) {
        stop(s);
    }
}

fn begin_scrub(s: &mut AppState) {
    if s.playbar.scrub.is_some() {
        return;
    }
    let was_playing = s.playbar.now.as_ref().is_some_and(|n| n.status == Status::Playing);
    if was_playing {
        pause(s);
    }
    s.playbar.scrub = Some(was_playing);
}

/// `at` is `None` when the drag was cancelled.
fn end_scrub(s: &mut AppState, at: Option<f32>) {
    let Some(was_playing) = s.playbar.scrub.take() else {
        if let Some(at) = at {
            seek(s, at); // a click without a drag
        }
        return;
    };
    if let Some(at) = at {
        seek(s, at);
    }
    if was_playing {
        resume(s);
    }
}

pub fn seek(s: &mut AppState, pos_s: f32) {
    let Some(n) = s.playbar.now.as_ref().filter(|n| !n.live) else { return };
    let pos = pos_s.clamp(0.0, n.len_s);
    player_do(s, |p| p.seek(pos));
    if let Some(n) = s.playbar.now.as_mut() {
        n.pos_s = pos;
        if n.status == Status::Ended {
            n.status = Status::Paused;
        }
        jump(n.track, pos, n.len_s);
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

pub fn sink() -> crate::audio::PlaySink {
    let progress = super::events::Latest::new(|s: &mut AppState, (track, pos, len): (u64, f32, f32)| {
        on_progress(s, track, pos, len)
    });
    epoch();
    // Every report feeds the knob's clock; only one that changes the time shown (`shows`) goes to the UI thread.
    let sent = (AtomicU64::new(0), AtomicU64::new(0));
    std::sync::Arc::new(move |ev| match ev {
        crate::audio::PlayEvent::Progress { track, pos_s, len_s, jumped } => {
            report(track, pos_s, len_s, jumped, std::time::Instant::now());
            #[cfg(windows)]
            super::seekstrip::tick();
            let (last_pos, last_len) = unpack(sent.1.load(Relaxed));
            if sent.0.load(Relaxed) != track || (last_len - len_s).abs() > 1e-3 || shows(last_pos, pos_s, len_s) {
                sent.0.store(track, Relaxed);
                sent.1.store(pack(pos_s, len_s), Relaxed);
                progress.send((track, pos_s, len_s));
            }
        }
        crate::audio::PlayEvent::Ended { track } => super::events::post_if(move |s| on_ended(s, track)),
    })
}

// The knob and the fill are placed by live values (`rune::Live`): a set draws one frame of the bar without
// running the view (a view run per position cost three whole-window frames).

/// Points as f32 bits; 0 until the bar has been laid out once.
static RAIL_W: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// The fraction of the track shown now, as f32 bits.
static SHOWN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// A fraction of a pixel, so the knob glides also on a long track.
const KNOB_STEP_PT: f32 = 0.05;
const GLIDE_HZ: f64 = 30.0;
/// Windows: the seek strip draws the fill and the knob, so the bar draws neither and does not glide.
static STRIP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Where the track was at a moment: the audio thread's report, or where a seek or a start put it.
#[derive(Clone, Copy, Debug)]
struct Heard {
    track: u64,
    pos_s: f32,
    len_s: f32,
    at: std::time::Instant,
    /// Changes when the position jumped (a seek, a start, a loop): the knob follows at once, backwards too.
    jump: u64,
}

/// A seqlock with one writer: the writer never waits (it is the audio callback), a reader retries while a write is
/// under way.
struct Slot {
    seq: AtomicU64,
    v: [AtomicU64; 6],
}

impl Slot {
    const fn new() -> Self {
        Slot { seq: AtomicU64::new(0), v: [const { AtomicU64::new(0) }; 6] }
    }

    fn write(&self, v: [u64; 6]) {
        let seq = self.seq.load(Relaxed);
        self.seq.store(seq + 1, Relaxed);
        fence(Release);
        for (slot, x) in self.v.iter().zip(v) {
            slot.store(x, Relaxed);
        }
        self.seq.store(seq + 2, Release);
    }

    /// `None` before the first write.
    fn read(&self) -> Option<[u64; 6]> {
        loop {
            let seq = self.seq.load(Acquire);
            if seq == 0 {
                return None;
            }
            if seq.is_multiple_of(2) {
                let v = self.v.each_ref().map(|slot| slot.load(Relaxed));
                fence(Acquire);
                if self.seq.load(Relaxed) == seq {
                    return Some(v);
                }
            }
            std::hint::spin_loop();
        }
    }
}

fn epoch() -> std::time::Instant {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *EPOCH.get_or_init(std::time::Instant::now)
}

fn nanos(at: std::time::Instant) -> u64 {
    at.saturating_duration_since(epoch()).as_nanos() as u64
}

fn pack(pos_s: f32, len_s: f32) -> u64 {
    (u64::from(pos_s.to_bits()) << 32) | u64::from(len_s.to_bits())
}

fn unpack(v: u64) -> (f32, f32) {
    (f32::from_bits((v >> 32) as u32), f32::from_bits(v as u32))
}

/// After a seek, how long a report from before it may still be on its way.
const QUIET: std::time::Duration = std::time::Duration::from_millis(150);

/// What the knob follows. The audio thread and the UI thread each write a slot of their own, so neither waits for
/// the other and a report from before a seek cannot overwrite it; `heard` picks the one to follow.
struct Feed {
    /// Track, position and length, time, the audio thread's jump count.
    audio: Slot,
    /// Track, position and length, time, the seek count, the audio jump count at the seek, quiet until.
    seek: Slot,
}

impl Feed {
    const fn new() -> Self {
        Feed { audio: Slot::new(), seek: Slot::new() }
    }

    /// Audio thread only.
    fn report(&self, track: u64, pos_s: f32, len_s: f32, jumped: bool, at: std::time::Instant) {
        let jumps = self.audio.read().map_or(0, |v| v[3]) + u64::from(jumped);
        self.audio.write([track, pack(pos_s, len_s), nanos(at), jumps, 0, 0]);
    }

    /// UI thread only.
    fn jump(&self, track: u64, pos_s: f32, len_s: f32, at: std::time::Instant) {
        let seen = self.audio.read().map_or(0, |v| v[3]);
        let seeks = self.seek.read().map_or(0, |v| v[3]) + 1;
        self.seek.write([track, pack(pos_s, len_s), nanos(at), seeks, seen, nanos(at + QUIET)]);
    }

    /// A report counts once the audio thread has jumped since the last seek (the seek itself, a loop) or once it is
    /// newer than `QUIET` after it.
    fn heard(&self) -> Option<Heard> {
        let seek = self.seek.read();
        let audio = self.audio.read();
        let (seeks, seen, quiet) = seek.map_or((0, 0, 0), |v| (v[3], v[4], v[5]));
        let from = match (audio, seek) {
            (Some(a), _) if a[3] > seen || a[2] >= quiet => a,
            (_, Some(s)) => s,
            (Some(a), None) => a,
            (None, None) => return None,
        };
        let (pos_s, len_s) = unpack(from[1]);
        let audio_jumps = audio.map_or(0, |a| a[3]);
        Some(Heard {
            track: from[0],
            pos_s,
            len_s,
            at: epoch() + std::time::Duration::from_nanos(from[2]),
            jump: (seeks << 32) | (audio_jumps & 0xffff_ffff),
        })
    }
}

static FEED: Feed = Feed::new();

/// Audio thread: no locks.
pub(super) fn report(track: u64, pos_s: f32, len_s: f32, jumped: bool, at: std::time::Instant) {
    FEED.report(track, pos_s, len_s, jumped, at);
}

/// UI thread. The knob goes to the new position without easing.
pub(super) fn jump(track: u64, pos_s: f32, len_s: f32) {
    FEED.jump(track, pos_s, len_s, std::time::Instant::now());
}

fn heard() -> Option<Heard> {
    FEED.heard()
}

/// The longest the knob runs on by itself after the audio thread's last report: past it the sound has stalled.
const AHEAD_S: f32 = 0.25;
/// How far the knob may run ahead of the audio clock while it takes up a difference.
const LEAD_S: f32 = 0.05;
/// The knob is this far behind the audio clock only after it stood still: it catches up at once.
const BEHIND_S: f32 = 0.5;
/// How fast the knob takes up the difference to the audio clock (time constant).
const CATCH_S: f32 = 0.3;

/// The knob's clock: it moves on by a monotonic clock, eases towards the audio clock and never goes back unless
/// the position jumped, so late frames and uneven reports neither stop it nor make it hop.
pub(super) struct Glide {
    shown_s: f32,
    at: Option<std::time::Instant>,
    track: u64,
    jump: u64,
}

impl Glide {
    pub(super) const fn new() -> Self {
        Glide { shown_s: 0.0, at: None, track: 0, jump: 0 }
    }

    /// The position (seconds) to show at `now`.
    fn at(&mut self, h: &Heard, speed: f32, now: std::time::Instant) -> f32 {
        let speed = speed.max(0.0);
        let target = h.pos_s + now.saturating_duration_since(h.at).as_secs_f32().min(AHEAD_S) * speed;
        let shown = match self.at {
            Some(last) if self.track == h.track && self.jump == h.jump => {
                let dt = now.saturating_duration_since(last).as_secs_f32();
                let moved = self.shown_s + dt * speed;
                let behind = target - moved;
                let next = if behind > BEHIND_S { target } else { moved + behind * (1.0 - (-dt / CATCH_S).exp()) };
                next.min(target + LEAD_S).max(self.shown_s)
            }
            _ => target,
        };
        self.shown_s = shown.clamp(0.0, h.len_s.max(0.0));
        (self.track, self.jump, self.at) = (h.track, h.jump, Some(now));
        self.shown_s
    }

    /// The fraction of the track to show now; `None` before any report.
    pub(super) fn step(&mut self, speed: f32) -> Option<f32> {
        let h = heard().filter(|h| h.len_s > 0.0)?;
        Some(self.at(&h, speed, std::time::Instant::now()) / h.len_s)
    }

    /// Nothing plays: the next step starts from the report, not from where the glide was.
    pub(super) fn rest(&mut self) {
        self.at = None;
    }
}

fn glide(s: &mut AppState, tx: &mut rune::Tx) {
    static GLIDE: std::sync::Mutex<Glide> = std::sync::Mutex::new(Glide::new());
    let mut glide = GLIDE.lock().unwrap_or_else(|e| e.into_inner());
    // The seek strip moves the knob (Windows); a scripted run keeps its knob where the scene put it.
    let strip = STRIP.load(std::sync::atomic::Ordering::Relaxed);
    let Some(n) =
        s.playbar.now.as_ref().filter(|n| n.status == Status::Playing && !n.live && !strip && !s.shell.scripted())
    else {
        glide.rest();
        return;
    };
    let track = n.track;
    if heard().is_some_and(|h| h.track == track)
        && let Some(frac) = glide.step(s.playbar.speed)
    {
        place(frac, false);
    }
    let clock = tx.clock();
    tx.wake_at(((clock * GLIDE_HZ).floor() + 1.0) / GLIDE_HZ);
}

fn knob_at() -> &'static rune::Live<[f32; 2]> {
    static L: std::sync::OnceLock<rune::Live<[f32; 2]>> = std::sync::OnceLock::new();
    L.get_or_init(|| rune::Live::new([0.0, 0.0]))
}

fn fill_at() -> &'static rune::Live<[f32; 2]> {
    static L: std::sync::OnceLock<rune::Live<[f32; 2]>> = std::sync::OnceLock::new();
    L.get_or_init(|| rune::Live::new([0.0, 0.0]))
}

/// Any thread. Unless `force`d, a move of less than `KNOB_STEP_PT` is left out.
fn place(frac: f32, force: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let w = f32::from_bits(RAIL_W.load(Relaxed));
    let frac = if frac.is_finite() { frac.clamp(0.0, 1.0) } else { 0.0 };
    let shown = f32::from_bits(SHOWN.load(Relaxed));
    if !force && ((frac - shown) * w).abs() < KNOB_STEP_PT {
        return;
    }
    SHOWN.store(frac.to_bits(), Relaxed);
    knob_at().set([frac * w, 0.0]);
    fill_at().set([(frac - 1.0) * w, 0.0]);
}

/// Keeps the live values right after a resize and wherever the position changes without the audio thread.
fn sync_rail(w: Option<f32>, frac: Option<f32>) {
    use std::sync::atomic::Ordering::Relaxed;
    let mut force = false;
    if let Some(w) = w.filter(|w| *w > 0.0) {
        force = RAIL_W.swap(w.to_bits(), Relaxed) != w.to_bits();
    }
    match frac {
        Some(frac) => place(frac, true),
        None if force => place(f32::from_bits(SHOWN.load(Relaxed)), true),
        None => {}
    }
}

fn shows(shown_s: f32, pos_s: f32, len_s: f32) -> bool {
    (shown_s <= 0.0 && pos_s > 0.0) || pos_s.floor() != shown_s.floor() || pos_s >= len_s - 1e-3
}

fn on_progress(s: &mut AppState, track: u64, pos_s: f32, len_s: f32) -> bool {
    let Some(n) = s.playbar.now.as_mut().filter(|n| n.track == track && n.status == Status::Playing) else {
        return false;
    };
    let moved = shows(n.pos_s, pos_s, len_s);
    let changed = moved || (n.len_s - len_s).abs() > 1e-3;
    if moved {
        n.pos_s = pos_s;
    }
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
    super::book::on_track_ended(s);
    true
}

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

pub const BAR_H: f32 = 56.0;
pub const DOCK_H: f32 = BAR_H + 12.0;

fn clock(secs: f32) -> String {
    let t = secs.max(0.0).floor() as u32;
    format!("{}:{:02}", t / 60, t % 60)
}

/// Rounds like the result cards' lengths (a 3.8 s take is "0:04" on its card and here); unknown while generated.
fn total(n: &Now) -> String {
    if n.live { "…".to_owned() } else { clock(n.len_s.round()) }
}

fn speed_label(v: f32) -> String {
    let t = format!("{v:.2}");
    let t = t.trim_end_matches('0').trim_end_matches('.');
    format!("{t}×")
}

fn round_button(
    p: Palette,
    key: &str,
    glyph: &'static rune::Icon,
    label: &str,
    on: bool,
    f: impl Fn(&mut AppState) + 'static,
) -> El {
    let b = el()
        .key(key.to_owned())
        .role("button")
        .label(label.to_owned())
        .help(label.to_owned())
        .size(32.0, 32.0)
        .radius(16.0)
        .center()
        .focusable(true)
        .motion(motion::UI)
        .on_press(f);
    let b = if on {
        b.fill(p.accent.alpha(0.2)).border(1.0, p.accent.alpha(0.45)).hover(p.accent.alpha(0.28))
    } else {
        b.hover(p.glass_hover).press(p.glass_press)
    };
    b.child(icon_el(glyph, 15.0, if on { p.accent } else { p.text_dim }))
}

/// `rail`: the seek rail's place (x, y, width, height in window points) from the last layout, `None` before
/// the first; `covered`: a menu or a dialog is open over the window.
pub fn view(s: &AppState, p: Palette, rail: Option<[f32; 4]>, scale: f32, covered: bool) -> Option<El> {
    let Some(n) = s.playbar.now.as_ref() else {
        strip(None);
        return None;
    };
    let on = n.status == Status::Playing;
    let frac = if n.len_s > 0.0 { (n.pos_s / n.len_s).clamp(0.0, 1.0) } else { 0.0 };
    // While the speech is still generated the bar is a loading bar of its own, without the strip.
    let native = strip(
        rail.filter(|_| !covered && !n.live && !s.shell.scripted())
            .map(|rail| (rail, scale, p, n, s.playbar.speed, frac)),
    );
    sync_rail(rail.map(|r| r[2]), (!on || s.shell.scripted()).then_some(frac)); // (a scripted run has no audio thread)
    let play = el()
        .key("pb-play")
        .role("button")
        .label(if on { "Pause" } else { "Play" })
        .help(if on { "Pause (Space)" } else { "Play (Space)" })
        .size(38.0, 38.0)
        .radius(19.0)
        .center()
        .focusable(true)
        .motion(motion::UI)
        .hover(Color::WHITE.alpha(0.1))
        .on_press(toggle_pause)
        .child(icon_el(if on { &icons::PAUSE } else { &icons::PLAY }, 16.0, p.on_accent));
    let play = accent_glass(p, play);
    let title = col()
        .key("pb-title")
        .w(230.0)
        .min_w(0.0)
        .gap(1.0)
        .child(text(n.title.clone()).font(ty::LABEL).color(p.text).truncate())
        .child(text(n.what.clone()).font(ty::CAPTION).color(p.text_faint).truncate());
    let total = total(n);
    let looped = s.playbar.looped;
    let loop_b = round_button(p, "pb-loop", &LOOP, if looped { "Loop: on" } else { "Loop: off" }, looped, move |s| {
        set_loop(s, !looped)
    })
    .checked(looped);
    let speed = s.playbar.speed;
    let open = s.playbar.speed_menu;
    let chip = row()
        .key("pb-speed")
        .role("button")
        .label(format!("Speed {}", speed_label(speed)))
        .help("Playback speed (the pitch stays)")
        .h(30.0)
        .px(10.0)
        .gap(5.0)
        .center()
        .radius(15.0)
        .focusable(true)
        .motion(motion::UI)
        .fill(if open || speed != 1.0 { p.accent.alpha(0.16) } else { p.glass })
        .border(1.0, if speed != 1.0 { p.accent.alpha(0.4) } else { p.rim_top.alpha(0.25) })
        .hover(p.glass_hover)
        .on_press(|s: &mut AppState| s.playbar.speed_menu = !s.playbar.speed_menu)
        .child(text(speed_label(speed)).font(ty::NUM).color(if speed != 1.0 { p.accent } else { p.text }))
        .child(icon_el(&icons::CHEVRON_DOWN, 11.0, p.text_dim));
    let close = round_button(p, "pb-close", &icons::CLOSE, "Close the player", false, stop);
    // instant_changes: animating each position move with the layout springs kept the window (and its blurs)
    // redrawing at the display's full rate while anything played.
    let bar = row()
        .key("playbar")
        .role("group")
        .label("Player")
        .instant_changes()
        .w_full()
        .h(BAR_H)
        .pl(9.0)
        .pr(10.0)
        .gap(10.0)
        .center_y()
        .radius(radius::PANEL + 4.0)
        .continuous_corners()
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.9))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(10.0, 30.0, p.shadow)
        .on_key(on_key)
        .child(play)
        .child(title)
        .child(text(clock(n.pos_s)).font(ty::NUM).color(p.text_dim))
        .child(seek_bar(p, n, native))
        .child(text(total).font(ty::NUM).color(p.text_dim).min_w(24.0))
        .child(loop_b)
        .child(chip)
        .child(close);
    Some(row().key("playbar-dock").w_full().px(16.0).pb(12.0).child(glass_edge(p, bar)))
}

type StripView<'a> = ([f32; 4], f32, Palette, &'a Now, f32, f32);

/// Windows only. True when the strip shows the fill and the knob, so the bar must not.
#[allow(unused_variables)]
fn strip(over: Option<StripView>) -> bool {
    #[cfg(windows)]
    let shown = match over {
        Some((rail, scale, p, n, speed, frac)) => {
            let rgb = |c: Color| {
                let [r, g, b, _] = c.premul_srgb(1.0);
                [(r * 255.0).round() as u8, (g * 255.0).round() as u8, (b * 255.0).round() as u8]
            };
            let look = super::seekstrip::Look {
                track_pt: 5.0,
                knob_pt: 14.0,
                fill_top: rgb(p.accent_top),
                fill_bottom: rgb(p.accent),
                ring: rgb(p.accent),
            };
            super::seekstrip::show(rail, scale, look, n.status == Status::Playing, speed, frac)
        }
        None => {
            super::seekstrip::hide();
            false
        }
    };
    #[cfg(not(windows))]
    let shown = false;
    STRIP.store(shown, std::sync::atomic::Ordering::Relaxed);
    shown
}

pub const RAIL: &str = "pb-rail";

fn seek_bar(p: Palette, n: &Now, native: bool) -> El {
    const TRACK: f32 = 5.0;
    const KNOB: f32 = 14.0;
    if n.live {
        return loading_bar(p, n.loaded, TRACK, KNOB);
    }
    let len = n.len_s;
    let knob = el()
        .key("pb-knob")
        .size(KNOB, KNOB)
        .pin(Some(-KNOB / 2.0), Some((TRACK - KNOB) / 2.0), None, None)
        .radius(KNOB / 2.0)
        .fill(Color::WHITE.alpha(0.85))
        .border(1.0, p.accent.alpha(0.6))
        .shadow(2.0, 6.0, Color::BLACK.alpha(0.5))
        .live_offset(knob_at());
    let fill = el()
        .key("pb-fill")
        .h(TRACK)
        .w_full()
        .radius(TRACK / 2.0)
        .gradient(p.accent_top, p.accent)
        .live_offset(fill_at());
    let groove = el()
        .key("pb-track")
        .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        .radius(TRACK / 2.0)
        .fill(p.well)
        .border(1.0, p.well_rim)
        .clip_rounded(TRACK / 2.0)
        .child(fill);
    let groove = if native {
        el().key("pb-track")
            .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
            .radius(TRACK / 2.0)
            .fill(p.well)
            .border(1.0, p.well_rim)
    } else {
        groove
    };
    let track = el().key(RAIL).w_full().h(TRACK).child(groove).children((!native).then_some(knob));
    let hit = col()
        .key("pb-seek")
        .role("slider")
        .label("Position")
        .w_full()
        .h(28.0)
        .px(KNOB / 2.0)
        .justify_center()
        .grow(1.0)
        .min_w(60.0)
        .range_value(f64::from(n.pos_s), 0.0, f64::from(len.max(0.01)))
        .a11y_value(format!("{} of {}", clock(n.pos_s), clock(len)))
        .child(track);
    let pos = n.pos_s;
    hit.focusable(true)
        .cursor(rune::Cursor::Pointer)
        .on_step(move |s: &mut AppState, steps, _| seek(s, pos + steps as f32 * STEP_S))
        .on_drag(move |s: &mut AppState, d: &mut Drag, tx| {
            // Dragging only moves the knob; the sound jumps once, to where it is let go (no "scrubbing" noise).
            let inner = (d.w - KNOB).max(1.0);
            let at = ((d.x - KNOB / 2.0) / inner).clamp(0.0, 1.0) * len;
            if d.cancelled() {
                end_scrub(s, None);
                return;
            }
            match d.phase {
                Phase::Up => end_scrub(s, Some(at)),
                _ => {
                    if d.phase == Phase::Move {
                        tx.instant();
                    }
                    begin_scrub(s);
                    if let Some(n) = s.playbar.now.as_mut() {
                        n.pos_s = at;
                    }
                }
            }
        })
}

/// The length is not known until the whole text is generated: the rail fills as it is, with no knob.
fn loading_bar(p: Palette, loaded: Option<f32>, track: f32, knob: f32) -> El {
    let part = el().key("pb-loaded").h(track).radius(track / 2.0).gradient(p.accent_top, p.accent).opacity(0.6);
    let part = match loaded {
        Some(v) => part.w_frac(v),
        None => part.w_frac(0.32).looping(rune::LoopSpec::slide(1.5, [-1.0, 0.0], [3.2, 0.0])),
    };
    let groove = el()
        .key("pb-track")
        .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        .radius(track / 2.0)
        .fill(p.well)
        .border(1.0, p.well_rim)
        .clip_rounded(track / 2.0)
        .child(part);
    let bar = col()
        .key("pb-seek")
        .role("progressbar")
        .label("Generating")
        .help("Seeking works once the whole text is generated")
        .w_full()
        .h(28.0)
        .px(knob / 2.0)
        .justify_center()
        .grow(1.0)
        .min_w(60.0)
        .child(el().key(RAIL).w_full().h(track).child(groove));
    match loaded {
        Some(v) => {
            bar.range_value(f64::from((v * 100.0).round()), 0.0, 100.0).a11y_value(format!("{:.0} %", v * 100.0))
        }
        None => bar,
    }
}

pub fn overlay(s: &AppState, p: Palette) -> Option<El> {
    if !s.playbar.speed_menu || s.playbar.now.is_none() {
        return None;
    }
    let scrim = el()
        .key("pb-speed-scrim")
        .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        .on_press(|s: &mut AppState| s.playbar.speed_menu = false);
    let mut list = col().key("pb-speed-list").role("menu").w_full().gap(2.0).focus_group();
    for v in SPEEDS {
        let on = (s.playbar.speed - v).abs() < 1e-3;
        list = list.child(
            row()
                .key(format!("pb-speed-{v}"))
                .role("menuitem")
                .label(speed_label(v))
                .checked(on)
                .w_full()
                .h(30.0)
                .px(10.0)
                .gap(8.0)
                .center_y()
                .radius(15.0)
                .fill(if on { p.accent.alpha(0.14) } else { Color::TRANSPARENT })
                .hover(p.glass_hover)
                .focusable(true)
                .on_press(move |s: &mut AppState| set_speed(s, v))
                .child(text(speed_label(v)).font(ty::NUM).color(p.text).grow(1.0))
                .child(el().w(14.0).children(on.then(|| icon_el(&icons::CHECK, 13.0, p.accent)))),
        );
    }
    let panel = col()
        .key("pb-speed-menu")
        .w(112.0)
        .padding(6.0)
        .gap(4.0)
        .radius(radius::CARD + 2.0)
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

#[allow(clippy::too_many_arguments)]
pub fn scene_now(
    s: &mut AppState,
    path: &str,
    title: &str,
    what: &str,
    source: Source,
    status: Status,
    pos_s: f32,
    len_s: f32,
) {
    s.playbar.now = Some(Now {
        path: path.into(),
        title: title.into(),
        what: what.into(),
        source,
        status,
        pos_s,
        len_s,
        live: false,
        loaded: None,
        track: 0,
    });
}

pub fn scene_live(s: &mut AppState, title: &str, pos_s: f32, len_s: f32, loaded: Option<f32>) {
    scene_now(
        s,
        "/tmp/job1/chunk0.wav",
        title,
        "Speak · playing as it is generated",
        Source::Speak,
        Status::Playing,
        pos_s,
        len_s,
    );
    if let Some(n) = s.playbar.now.as_mut() {
        n.live = true;
        n.loaded = loaded;
    }
}

const TAKE3: &str = "Welcome back. In a few seconds this paragraph will be read in my own voice.";

fn scene_playing(s: &mut AppState) {
    super::speak::scene_done(s);
    scene_now(s, "/tmp/out3.wav", TAKE3, "Speak · result", Source::Speak, Status::Playing, 5.2, 14.2);
}

fn scene_paused(s: &mut AppState) {
    super::speak::scene_done(s);
    scene_now(
        s,
        "/tmp/out2.wav",
        "The quick brown fox jumps over the lazy dog.",
        "Speak · result",
        Source::Speak,
        Status::Paused,
        1.6,
        3.8,
    );
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

const LIVE_TITLE: &str = "Welcome back. In a few seconds this paragraph will be read in my own voice, one sentence…";

fn scene_generating(s: &mut AppState) {
    super::speak::scene_generating(s);
    scene_live(s, LIVE_TITLE, 3.4, 9.0, Some(0.38));
}

fn scene_generating_unknown(s: &mut AppState) {
    super::speak::scene_generating(s);
    scene_live(s, LIVE_TITLE, 1.2, 4.0, None);
}

fn scene_finished(s: &mut AppState) {
    super::speak::scene_done(s);
    scene_live(s, LIVE_TITLE, 6.1, 14.2, Some(1.0));
    finish_live(s, Some((Path::new("/tmp/out3.wav"), TAKE3.to_owned())));
}

fn scene_record(s: &mut AppState) {
    super::record::scene_notice_saved(s);
    scene_now(s, "/tmp/part1.wav", "Part 1 take", "Record", Source::Record, Status::Playing, 2.0, 21.0);
}

pub fn scenes() -> &'static [super::Scene] {
    &[
        ("player_playing", scene_playing),
        ("player_paused", scene_paused),
        ("player_looped", scene_looped),
        ("player_speed", scene_speed),
        ("player_speed_menu", scene_speed_menu),
        ("player_generating", scene_generating),
        ("player_generating_unknown", scene_generating_unknown),
        ("player_finished", scene_finished),
        ("player_record", scene_record),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn secs(s: f32) -> Duration {
        Duration::from_secs_f32(s)
    }

    /// Reports every 33 ms, each stamped up to 8 ms late or early (callbacks are not on time), of a track playing
    /// from `t0` at normal speed: the latest one at `now`.
    fn report_at(t0: Instant, now: Instant, stall: Option<(f32, f32)>) -> Heard {
        let t = now.duration_since(t0).as_secs_f32();
        let mut k = (t / 0.033).floor() as i32;
        loop {
            let nominal = k as f32 * 0.033;
            let jitter = [0.0, 0.008, -0.006, 0.003, -0.008, 0.005][k.rem_euclid(6) as usize];
            let pos = match stall {
                Some((from, to)) if nominal >= from && nominal < to => None,
                Some((from, to)) if nominal >= to => Some(from + (nominal - to)),
                _ => Some(nominal),
            };
            let at = (nominal + jitter).max(0.0);
            if let Some(pos_s) = pos.filter(|_| at <= t) {
                return Heard { track: 1, pos_s, len_s: 600.0, at: t0 + secs(at), jump: 1 };
            }
            k -= 1;
        }
    }

    fn truth(t: f32, stall: Option<(f32, f32)>) -> f32 {
        match stall {
            Some((from, to)) if t >= from && t < to => from,
            Some((from, to)) if t >= to => from + (t - to),
            _ => t,
        }
    }

    #[test]
    fn the_knob_follows_the_sound_and_never_steps_back() {
        let t0 = Instant::now();
        let mut glide = Glide::new();
        let mut last = 0.0;
        for frame in 1..600 {
            let t = frame as f32 / 60.0;
            let now = t0 + secs(t);
            let shown = glide.at(&report_at(t0, now, None), 1.0, now);
            assert!(shown >= last, "back from {last} to {shown} at {t}");
            assert!((shown - t).abs() < 0.02, "{shown} shown at {t}");
            last = shown;
        }
    }

    #[test]
    fn late_frames_land_where_the_sound_is() {
        let t0 = Instant::now();
        let mut glide = Glide::new();
        let mut last = 0.0;
        // A busy UI thread: frames 16 ms apart, then gaps of 0.4, 0.9 and 1.6 s.
        for t in [0.5, 0.516, 0.533, 0.933, 0.95, 1.85, 1.866, 3.466, 3.483, 3.5] {
            let now = t0 + secs(t);
            let shown = glide.at(&report_at(t0, now, None), 1.0, now);
            assert!(shown >= last);
            assert!((shown - t).abs() < 0.02, "{shown} shown at {t}");
            last = shown;
        }
    }

    #[test]
    fn the_same_moment_shows_the_same_place_at_any_frame_rate() {
        let t0 = Instant::now();
        let (mut fast, mut slow) = (Glide::new(), Glide::new());
        for frame in 1..=300 {
            let t = frame as f32 / 60.0;
            let now = t0 + secs(t);
            let a = fast.at(&report_at(t0, now, None), 1.0, now);
            if frame % 6 == 0 {
                let b = slow.at(&report_at(t0, now, None), 1.0, now);
                assert!((a - b).abs() < 0.01, "{a} at 60 fps, {b} at 10 fps, at {t}");
            }
        }
    }

    #[test]
    fn a_stalled_sound_holds_the_knob_without_going_back() {
        let t0 = Instant::now();
        let stall = Some((1.0, 2.0));
        let mut glide = Glide::new();
        let mut last = 0.0;
        for frame in 1..300 {
            let t = frame as f32 / 60.0;
            let now = t0 + secs(t);
            let shown = glide.at(&report_at(t0, now, stall), 1.0, now);
            assert!(shown >= last, "back from {last} to {shown} at {t}");
            if t < 2.0 {
                assert!(shown <= 1.0 + AHEAD_S + LEAD_S, "{shown} ran on at {t}");
            }
            if t > 3.0 {
                assert!((shown - truth(t, stall)).abs() < 0.02, "{shown} shown at {t}");
            }
            last = shown;
        }
    }

    #[test]
    fn a_seek_or_a_loop_moves_the_knob_at_once_backwards_too() {
        let t0 = Instant::now();
        let mut glide = Glide::new();
        let now = t0 + secs(3.0);
        let h = report_at(t0, now, None);
        assert!(glide.at(&h, 1.0, now) > 2.9);
        let seek = Heard { pos_s: 0.5, at: now, jump: h.jump + 1, ..h };
        assert_eq!(glide.at(&seek, 1.0, now), 0.5);
        let later = now + secs(0.1);
        assert!((glide.at(&seek, 1.0, later) - 0.6).abs() < 1e-3);
        let other = Heard { track: 2, pos_s: 0.0, at: later, ..seek };
        assert_eq!(glide.at(&other, 1.0, later), 0.0);
    }

    #[test]
    fn a_faster_speed_moves_the_knob_faster() {
        let t0 = Instant::now();
        let mut glide = Glide::new();
        let h = Heard { track: 1, pos_s: 1.0, len_s: 60.0, at: t0, jump: 1 };
        assert_eq!(glide.at(&h, 2.0, t0), 1.0);
        assert!((glide.at(&h, 2.0, t0 + secs(0.1)) - 1.2).abs() < 1e-3);
    }

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    #[test]
    fn a_report_from_before_a_seek_does_not_pull_the_knob_back() {
        let feed = Feed::new();
        let t0 = epoch() + secs(1.0);
        feed.report(1, 5.0, 10.0, false, t0);
        feed.jump(1, 2.0, 10.0, t0 + ms(10));
        // Reports made before the seek, published after it: one stamped before the seek, one after.
        feed.report(1, 5.03, 10.0, false, t0 + ms(5));
        assert_eq!(feed.heard().unwrap().pos_s, 2.0);
        feed.report(1, 5.06, 10.0, false, t0 + ms(40));
        assert_eq!(feed.heard().unwrap().pos_s, 2.0);
        // The seek's own report, and the ones after it.
        feed.report(1, 2.0, 10.0, true, t0 + ms(60));
        assert_eq!(feed.heard().unwrap().pos_s, 2.0);
        feed.report(1, 2.03, 10.0, false, t0 + ms(90));
        assert_eq!(feed.heard().unwrap().pos_s, 2.03);
    }

    #[test]
    fn without_a_jump_from_the_audio_thread_reports_count_again_after_the_quiet_time() {
        let feed = Feed::new();
        let t0 = epoch() + secs(1.0);
        feed.jump(1, 2.0, 10.0, t0);
        feed.report(1, 2.03, 10.0, false, t0 + ms(30));
        assert_eq!(feed.heard().unwrap().pos_s, 2.0);
        feed.report(1, 2.2, 10.0, false, t0 + QUIET + ms(10));
        assert_eq!(feed.heard().unwrap().pos_s, 2.2);
    }

    #[test]
    fn a_loop_right_after_a_seek_brings_the_knob_back_to_the_start() {
        let feed = Feed::new();
        let mut glide = Glide::new();
        let t0 = epoch() + secs(1.0);
        feed.report(1, 4.0, 10.0, false, t0);
        feed.jump(1, 9.97, 10.0, t0 + ms(1));
        assert!((glide.at(&feed.heard().unwrap(), 1.0, t0 + ms(1)) - 9.97).abs() < 1e-3);
        feed.report(1, 9.98, 10.0, true, t0 + ms(20));
        feed.report(1, 0.02, 10.0, true, t0 + ms(50));
        let h = feed.heard().unwrap();
        assert_eq!(h.pos_s, 0.02);
        assert!(glide.at(&h, 1.0, t0 + ms(50)) < 0.05);
    }

    #[test]
    fn a_reader_never_sees_half_a_report() {
        let feed = std::sync::Arc::new(Feed::new());
        let writer = {
            let feed = feed.clone();
            std::thread::spawn(move || {
                let t0 = epoch();
                for k in 1..200_000u64 {
                    feed.report(k, k as f32, 2.0 * k as f32, k % 7 == 0, t0);
                }
            })
        };
        while !writer.is_finished() {
            if let Some(h) = feed.heard() {
                assert_eq!(h.pos_s, h.track as f32);
                assert_eq!(h.len_s, 2.0 * h.track as f32);
            }
        }
        writer.join().unwrap();
    }

    #[test]
    fn generating_shows_how_much_is_loaded_then_the_track_with_its_length() {
        let mut s = AppState::default();
        scene_live(&mut s, "Hello", 1.5, 4.0, None);
        let n = s.playbar.now.as_ref().unwrap();
        assert!(n.live && n.loaded.is_none());
        assert_eq!(total(n), "…");
        loading(&mut s, Some(0.4));
        loading(&mut s, Some(0.3));
        assert_eq!(s.playbar.now.as_ref().unwrap().loaded, Some(0.4));
        loading(&mut s, None);
        assert_eq!(s.playbar.now.as_ref().unwrap().loaded, Some(0.4));
        loading(&mut s, Some(1.0));
        if let Some(n) = s.playbar.now.as_mut() {
            n.len_s = 13.6;
        }
        finish_live(&mut s, Some((Path::new("/tmp/take.wav"), "Hello there".into())));
        let n = s.playbar.now.as_ref().unwrap();
        assert!(!n.live && n.loaded.is_none());
        assert_eq!(
            (n.path.as_path(), n.title.as_str(), n.what.as_str()),
            (Path::new("/tmp/take.wav"), "Hello there", "Speak · result")
        );
        assert_eq!(total(n), "0:14");
        loading(&mut s, Some(0.5));
        assert!(s.playbar.now.as_ref().unwrap().loaded.is_none());
    }
}
