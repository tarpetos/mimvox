//! How work outside the UI thread reaches the UI. Everything is an event; nothing polls.
//!
//! * Background threads (the engine's stdout reader, audio callbacks, native file dialogs, the setup / move
//!   subprocess readers, disk probes) call [`post`] / [`post_if`]. Rune's `AppHandle::update_if` queues the closure
//!   and wakes the event loop; the closure runs on the UI thread, then `shell::settle` brings derived state up to
//!   date (requests that were waiting for the engine, the timers below) and Rune draws one frame if anything
//!   changed. While no event arrives the UI thread sleeps: no frames, no wake-ups.
//! * Timers (a toast's expiry, the elapsed-time clock of a running job, a debounce) are deadlines on the app clock
//!   handed to Rune with `Tx::wake_at`. When one comes due Rune runs the frame hook (`shell::on_frame`, the app's
//!   `AppSpec::poll`), which settles the state like any other event. Nothing asks for a wake-up unless such a
//!   deadline exists, so an idle window never wakes.
//! * Feeds that can outpace the screen (a microphone level meter) go through [`Latest`]: the newest value replaces
//!   one that is still queued, so at most one closure per feed waits for the UI thread.

use crate::state::AppState;
use rune::AppHandle;
use std::sync::{Arc, Mutex, OnceLock};

static APP: OnceLock<AppHandle<AppState>> = OnceLock::new();

/// Called once from `main`, before the app runs.
pub fn init(handle: AppHandle<AppState>) {
    let _ = APP.set(handle);
}

/// The running app (window requests: open, quit, ...). `None` before `init`.
pub fn app() -> Option<&'static AppHandle<AppState>> {
    APP.get()
}

/// Runs `f` on the UI thread (from any thread) and redraws.
pub fn post(f: impl FnOnce(&mut AppState) + Send + 'static) {
    post_if(move |s| {
        f(s);
        true
    });
}

/// Runs `f` on the UI thread (from any thread). `f` returns whether it changed what the screen shows; the frame is
/// drawn only then (or when the settle step after it changed something).
pub fn post_if(f: impl FnOnce(&mut AppState) -> bool + Send + 'static) {
    let Some(app) = APP.get() else { return };
    app.update_if(move |s, tx| {
        super::stats::count(&super::stats::EVENTS);
        let changed = f(s);
        super::shell::settle(s, tx) | changed
    });
}

/// A coalescing feed: [`Latest::send`] keeps only the newest value; a value is posted to the UI thread only when the
/// previous one has been taken, so a fast producer (an audio callback) never queues up work.
pub struct Latest<T> {
    slot: Arc<Mutex<Option<T>>>,
    apply: Arc<dyn Fn(&mut AppState, T) -> bool + Send + Sync>,
}

impl<T> Clone for Latest<T> {
    fn clone(&self) -> Self {
        Latest { slot: self.slot.clone(), apply: self.apply.clone() }
    }
}

impl<T: Send + 'static> Latest<T> {
    /// `apply` runs on the UI thread with the newest value and returns whether the screen changed.
    pub fn new(apply: impl Fn(&mut AppState, T) -> bool + Send + Sync + 'static) -> Self {
        Latest { slot: Arc::new(Mutex::new(None)), apply: Arc::new(apply) }
    }

    pub fn send(&self, value: T) {
        let was_empty = {
            let mut g = self.slot.lock().unwrap_or_else(|e| e.into_inner());
            let empty = g.is_none();
            *g = Some(value);
            empty
        };
        if was_empty {
            let (slot, apply) = (self.slot.clone(), self.apply.clone());
            post_if(move |s| {
                let value = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
                value.is_some_and(|v| apply(s, v))
            });
        }
    }
}
