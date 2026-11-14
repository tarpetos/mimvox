//! `MIMVOX_FRAME_STATS=<file>` counts what the UI thread does and writes the counts there on quit.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static VIEWS: AtomicU64 = AtomicU64::new(0);
pub static HOOKS: AtomicU64 = AtomicU64::new(0);
pub static FRAMES: AtomicU64 = AtomicU64::new(0);
pub static EVENTS: AtomicU64 = AtomicU64::new(0);
pub static TIMERS: AtomicU64 = AtomicU64::new(0);

fn target() -> Option<&'static (std::path::PathBuf, Instant)> {
    static T: OnceLock<Option<(std::path::PathBuf, Instant)>> = OnceLock::new();
    T.get_or_init(|| std::env::var_os("MIMVOX_FRAME_STATS").filter(|v| !v.is_empty()).map(|p| (p.into(), Instant::now()))).as_ref()
}

#[inline]
pub fn count(what: &AtomicU64) {
    if target().is_some() {
        what.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn write() {
    let Some((path, start)) = target() else { return };
    let line = format!(
        "wall_s={:.1} views={} hooks={} frames={} events={} timers={}\n",
        start.elapsed().as_secs_f64(),
        VIEWS.load(Ordering::Relaxed),
        HOOKS.load(Ordering::Relaxed),
        FRAMES.load(Ordering::Relaxed),
        EVENTS.load(Ordering::Relaxed),
        TIMERS.load(Ordering::Relaxed),
    );
    let _ = std::fs::write(path, line);
}

/// `MIMVOX_FRAME_STATS_SECS=<n>`: quit after `n` seconds on the app clock, writing the counts.
pub fn quit_at() -> Option<f64> {
    static T: OnceLock<Option<f64>> = OnceLock::new();
    *T.get_or_init(|| target().and(std::env::var("MIMVOX_FRAME_STATS_SECS").ok()).and_then(|v| v.trim().parse().ok()))
}
