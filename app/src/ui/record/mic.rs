//! Microphone selection for the Record screen: a compact glass "mic chip", a dropdown with the input devices
//! and a live level test.
//!
//! - Devices are listed on a worker thread (cpal enumeration can take 100+ ms on Windows): once when the screen is
//!   first entered and again every time the dropdown opens. The list comes back as an event.
//! - The choice is `RecordState::input_device`, saved under `microphone` in `<data dir>/settings.json` and restored
//!   on launch. A saved device that is gone falls back to the system default with a small warning (and is picked
//!   up again when it reappears).
//! - While the dropdown is open a short-lived `Recorder` on the selected device feeds a level meter through its
//!   level events (opened on a worker thread too, dropped when the menu closes). It never writes a file, and switching is locked while a
//!   recording runs.

use super::widgets::*;
use super::{RecordState, guard, meter_sink};
use crate::audio::{self, CaptureEvent, Recorder};
use crate::ui::events;
use crate::state::{AppState, Screen};
use crate::ui::components::{glass_edge, icon_el, icons};
use crate::ui::onboarding;
use crate::ui::theme::{motion, radius, ty};
use rune::{Align, Color, El, ScrollSpec, Side, col, el, row, text};

/// A running level test.
pub struct Live {
    /// `None` only in headless scene fixtures.
    pub rec: Option<Recorder>,
    /// The capture number its level events carry.
    pub seq: u64,
    pub level: f32,
    pub peak: f32,
}

#[derive(Default)]
pub enum Test {
    #[default]
    Off,
    /// Capture `seq` is being opened on a worker thread (its result arrives as an event).
    Starting(u64),
    Running(Live),
    Failed(String),
}

#[derive(Default)]
pub struct MicState {
    /// Input device names; the first is the system default.
    pub devices: Vec<String>,
    pub loaded: bool,
    /// The device list is being read on a worker thread (it arrives as an event).
    listing: bool,
    pub open: bool,
    pub warning: Option<String>,
    /// The saved device that was not found (restored when it shows up again).
    missing: Option<String>,
    pub settings_read: bool,
    pub test: Test,
}

/// Stops a recorder off the UI thread (joining the capture thread can take a moment).
fn drop_later(rec: Recorder) {
    let _ = std::thread::Builder::new().name("mic-test-stop".into()).spawn(move || drop(rec));
}

impl RecordState {
    /// The chip shows on the preset picker and the recording session, not on the own-text editor.
    pub fn mic_visible(&self) -> bool {
        matches!(self.phase, super::Phase::Pick | super::Phase::Session | super::Phase::Summary)
    }

    fn mic_refresh(&mut self) {
        if self.mic.listing {
            return;
        }
        let spawned = std::thread::Builder::new().name("mic-list".into()).spawn(move || {
            let list = guard(|| Ok(audio::input_devices())).unwrap_or_default();
            events::post(move |s| {
                s.record.mic.listing = false;
                s.record.mic_apply_devices(list);
            });
        });
        if spawned.is_ok() {
            self.mic.listing = true;
        } else {
            self.mic.loaded = true;
        }
    }

    fn mic_apply_devices(&mut self, list: Vec<String>) {
        self.mic.devices = list;
        self.mic.loaded = true;
        // An empty list means the enumeration itself failed (or found nothing): never declare the saved device
        // missing on that basis, keep it and let `Recorder::start` report a real problem.
        if self.mic.devices.is_empty() {
            return;
        }
        let known = |devices: &[String], name: &str| devices.iter().any(|d| d.trim().eq_ignore_ascii_case(name.trim()));
        if let Some(cur) = self.input_device.clone() {
            if !known(&self.mic.devices, &cur) {
                self.mic.warning = Some(format!("\"{cur}\" is not available, using the system default"));
                self.mic.missing = Some(cur);
                self.input_device = None;
                if self.mic.open {
                    self.mic_start_test();
                }
            }
        } else if let Some(m) = self.mic.missing.clone() {
            if known(&self.mic.devices, &m) {
                self.input_device = Some(m);
                self.mic.missing = None;
                self.mic.warning = None;
                if self.mic.open {
                    self.mic_start_test();
                }
            }
        }
    }

    pub fn mic_set_open(&mut self, open: bool) {
        if open == self.mic.open {
            return;
        }
        if open {
            if self.recording() {
                return;
            }
            self.mic.open = true;
            self.mic_refresh();
            self.mic_start_test();
        } else {
            self.mic.open = false;
            self.mic_stop_test();
        }
    }

    /// Picks a device (`None` = system default), remembers it and re-points the level test.
    pub fn mic_select(&mut self, name: Option<String>) {
        if self.recording() {
            return;
        }
        self.mic.missing = None;
        self.mic.warning = None;
        onboarding::save_microphone(name.as_deref());
        self.input_device = name;
        if self.mic.open {
            self.mic_start_test();
        }
    }

    fn mic_start_test(&mut self) {
        self.mic_stop_test();
        let device = self.input_device.clone();
        self.capture_seq += 1;
        let seq = self.capture_seq;
        let spawned = std::thread::Builder::new().name("mic-test".into()).spawn(move || {
            let opened = guard(|| Recorder::start(device.as_deref(), Some(meter_sink(seq, on_test_level))));
            events::post(move |s| on_test_opened(s, seq, opened));
        });
        self.mic.test = match spawned {
            Ok(_) => Test::Starting(seq),
            Err(e) => Test::Failed(e.to_string()),
        };
    }

    pub fn mic_stop_test(&mut self) {
        if let Test::Running(Live { rec: Some(rec), .. }) = std::mem::take(&mut self.mic.test) {
            drop_later(rec);
        }
    }
}

/// The level test's microphone opened (or failed to): event from the opening thread.
fn on_test_opened(s: &mut AppState, seq: u64, opened: Result<Recorder, String>) {
    let r = &mut s.record;
    let current = matches!(r.mic.test, Test::Starting(n) if n == seq);
    if !current || !r.mic.open {
        // Closed or re-pointed meanwhile.
        if let Ok(rec) = opened {
            drop_later(rec);
        }
        if current {
            r.mic.test = Test::Off;
        }
        return;
    }
    r.mic.test = match opened {
        Ok(rec) => Test::Running(Live { rec: Some(rec), seq, level: 0.0, peak: 0.0 }),
        Err(e) => Test::Failed(e),
    };
}

/// A level event of the running test (from the audio thread, newest only).
fn on_test_level(s: &mut AppState, seq: u64, ev: CaptureEvent) -> bool {
    let Test::Running(live) = &mut s.record.mic.test else { return false };
    if live.seq != seq {
        return false;
    }
    match ev {
        CaptureEvent::Level { level, .. } => {
            live.level = if level > live.level { level } else { live.level * 0.82 + level * 0.18 };
            live.peak = live.level.max(live.peak * 0.97);
        }
        CaptureEvent::Error(e) => {
            if let Test::Running(Live { rec: Some(rec), .. }) = std::mem::take(&mut s.record.mic.test) {
                drop_later(rec);
            }
            s.record.mic.test = Test::Failed(e);
        }
    }
    true
}

impl RecordState {

    /// Reads the saved choice once. Headless scenes mark it as read so they never touch the settings file.
    fn mic_read_settings(&mut self) {
        if self.mic.settings_read {
            return;
        }
        self.mic.settings_read = true;
        if self.input_device.is_none() {
            self.input_device = onboarding::saved_microphone();
        }
    }
}

/// After every event: the saved choice once, the first device listing, and closing the dropdown (and its test)
/// whenever it should not be there. True when that changed what shows.
pub fn settle(s: &mut AppState) -> bool {
    let on_screen = s.screen == Screen::Record;
    let r = &mut s.record;
    let mut changed = false;
    r.mic_read_settings();
    if r.mic.open && (!on_screen || r.recording() || !r.mic_visible()) {
        r.mic_set_open(false);
        changed = true;
    }
    if on_screen && !r.mic.loaded && !r.mic.listing {
        r.mic_refresh();
    }
    changed
}

fn current_label(r: &RecordState) -> String {
    r.input_device.clone().unwrap_or_else(|| "System default".to_string())
}

/// The compact glass chip (plus the fallback warning when there is one).
pub fn chip(r: &RecordState, p: Pal) -> El {
    let busy = r.recording();
    let open = r.mic.open;
    let warn = r.mic.warning.is_some();
    let name = current_label(r);
    let pill = row()
        .key("mic-chip")
        .role("button")
        .label(format!("Microphone: {name}"))
        .h(32.0)
        .pl(12.0)
        .pr(12.0)
        .gap(8.0)
        .center()
        .radius(16.0)
        .fill(if open { p.lens } else { p.glass })
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, if warn { p.warn.alpha(0.6) } else if open { p.accent.alpha(0.4) } else { p.rim_bottom })
        .shadow(3.0, 10.0, p.shadow_soft);
    let mut pill = glass_edge(p, pill);
    pill = if busy {
        pill.opacity(0.5).help("Stop recording to change the microphone")
    } else {
        pill.hover(p.glass_hover).focusable(true).motion(motion::UI).on_press(|s: &mut AppState| {
            let o = s.record.mic.open;
            s.record.mic_set_open(!o);
        })
    };
    let pill = pill
        .child(icon_el(if warn { &icons::ALERT } else { &icons::MIC }, 14.0, if warn { p.warn } else { p.accent }))
        .child(text(name).font(ty::LABEL).color(p.text).max_w(190.0).truncate().a11y_hidden())
        .child(icon_el(&icons::CHEVRON_DOWN, 12.0, p.text_dim));
    // The warning sits under the chip, right-aligned, so it never runs into the heading next to it.
    let mut out = col().items_end().gap(6.0).child(pill);
    if let Some(w) = &r.mic.warning {
        out = out.child(text(w.clone()).font(ty::CAPTION).color(p.warn).max_w(260.0).max_lines(2));
    }
    out
}

fn device_row(p: Pal, key: String, title: String, caption: Option<String>, on: bool, pick: Option<String>) -> El {
    let mut label = col().grow(1.0).min_w(0.0).child(text(title).font(ty::LABEL).color(p.text).truncate());
    if let Some(c) = caption {
        label = label.child(text(c).font(ty::CAPTION).color(p.text_faint).truncate());
    }
    row()
        .key(key)
        .w_full()
        .min_h(38.0)
        .px(12.0)
        .py(3.0)
        .gap(10.0)
        .center_y()
        .radius(19.0)
        .fill(if on { p.accent.alpha(0.14) } else { Color::TRANSPARENT })
        .hover(p.glass_hover)
        .focusable(true)
        .on_press(move |s: &mut AppState| s.record.mic_select(pick.clone()))
        .child(label)
        .child(el().w(16.0).children(on.then(|| icon_el(&icons::CHECK, 14.0, p.accent))))
}

fn test_block(r: &RecordState, p: Pal) -> El {
    let (line, tone) = match &r.mic.test {
        Test::Off | Test::Starting(_) => ("Opening the microphone...".to_string(), p.text_faint),
        Test::Running(_) => ("Say a few words: the bar should move. Nothing is recorded.".to_string(), p.text_dim),
        Test::Failed(e) => (format!("Cannot use this microphone: {e}"), p.bad),
    };
    let meter_el = match &r.mic.test {
        Test::Running(l) => meter(p, l.level, l.peak),
        _ => meter(p, 0.0, 0.0),
    };
    let ok = matches!(r.mic.test, Test::Running(_));
    col()
        .w_full()
        .px(12.0)
        .py(6.0)
        .gap(8.0)
        .child(row().w_full().center_y().justify_between().child(text("Input level").font(ty::LABEL).color(p.text_dim)).child(
            text(if ok { "Live" } else { "" }).font(ty::CAPTION).color(p.ok),
        ))
        .child(meter_el)
        .child(text(line).font(ty::CAPTION).color(tone).wrap().max_lines(3))
}

/// The dropdown (with its scrim), anchored under the chip. Attach it at the screen root after the chip.
pub fn overlay(r: &RecordState, p: Pal) -> Option<El> {
    if !r.mic.open || !r.mic_visible() {
        return None;
    }
    let scrim = el().key("mic-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(|s: &mut AppState| s.record.mic_set_open(false));

    let default_name = r.mic.devices.first().cloned();
    let mut list = col().w_full().gap(2.0).child(device_row(
        p,
        "mic-default".into(),
        "System default".into(),
        default_name.map(|d| format!("Currently {d}")).or(Some("Follows your system sound settings".into())),
        r.input_device.is_none(),
        None,
    ));
    for (i, d) in r.mic.devices.iter().enumerate() {
        list = list.child(device_row(
            p,
            format!("mic-dev-{i}"),
            d.clone(),
            (i == 0).then(|| "System default device".to_string()),
            r.input_device.as_deref() == Some(d.as_str()),
            Some(d.clone()),
        ));
    }
    if r.mic.devices.is_empty() {
        let msg = if r.mic.loaded { "No other microphones found" } else { "Looking for microphones..." };
        list = list.child(text(msg).font(ty::CAPTION).color(p.text_faint).px(12.0).py(8.0));
    }
    let scroll = col().w_full().max_h(230.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).child(list);

    let mut head = row().w_full().px(12.0).pt(4.0).center_y().justify_between().child(text("Microphone").font(ty::LABEL).color(p.text));
    if r.mic.listing {
        head = head.child(text("Refreshing...").font(ty::CAPTION).color(p.text_faint));
    }
    let done = button(p, "mic-done", "Done", BtnKind::Ghost, true, |s: &mut AppState| s.record.mic_set_open(false));
    let panel = col()
        .key("mic-menu")
        .w(330.0)
        .padding(8.0)
        .gap(6.0)
        .radius(radius::PANEL + 4.0)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 1.0))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .backdrop_blur(28.0)
        .anchored("mic-chip", Side::Bottom, Align::End, 8.0)
        .transition(rune::Transition::scale(0.98))
        .child(head)
        .child(scroll)
        .child(test_block(r, p))
        .child(row().w_full().justify_end().child(done));
    Some(el().key("mic-menu-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).child(scrim).child(panel))
}
