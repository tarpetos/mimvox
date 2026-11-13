//! The update banner: "Mimvox X is available" under the header, the download's progress, then "Restart now".
//!
//! Once per launch (when the engine first settles, so the check never competes with the first frames) a background
//! thread asks for the newest release (`crate::update::check`). A newer one shows the banner, unless the user skipped
//! that version. "Update" downloads and verifies the installer in the background; it installs when the app closes, or
//! at once with "Restart now" (a running training or book pauses and continues after the restart, as on any close).
//! Where the app cannot replace itself (a .deb, a loose binary) the banner offers the download page instead.

use super::components::{Button, glass_card, icon_el, icons, progress_bar};
use super::events;
use super::theme::{Palette, ty};
use crate::state::AppState;
use crate::update::{self, Found, Kind};
use rune::{El, Tx, col, el, row, text};
use serde_json::json;

#[derive(Default)]
pub struct UpdateState {
    /// The check was started this session.
    asked: bool,
    found: Option<Found>,
    phase: Phase,
    /// "Later": the banner is hidden until the next launch.
    hidden: bool,
    /// A link to open once the handler's frame is done (`Tx::open_url`).
    open: Option<&'static str>,
}

#[derive(Default)]
enum Phase {
    #[default]
    Offered,
    Downloading { got: u64, total: Option<u64> },
    Ready,
    Failed(String),
}

/// `"check_updates": false` in settings.json turns the check off; `"skip_update": "<version>"` hides that version.
fn setting(key: &str) -> Option<serde_json::Value> {
    super::onboarding::read_settings().get(key).cloned()
}

/// After every event: start the check once (never in a scripted run or a development build), open a pending link.
pub fn settle(s: &mut AppState, tx: &mut Tx) -> bool {
    if let Some(url) = s.update.open.take() {
        let _ = tx.open_url(url);
    }
    if s.update.asked || s.shell.scripted() || !super::shell::engine_ready(s) {
        return false;
    }
    s.update.asked = true;
    if update::kind() == Kind::Never || setting("check_updates").and_then(|v| v.as_bool()) == Some(false) {
        return false;
    }
    std::thread::spawn(|| {
        if let Ok(Some(found)) = update::check() {
            let skipped = setting("skip_update").is_some_and(|v| v.as_str() == Some(found.version.as_str()));
            if !skipped {
                events::post(move |s| s.update.found = Some(found));
            }
        }
    });
    false
}

fn start_download(s: &mut AppState) {
    let Some(found) = s.update.found.clone() else { return };
    s.update.phase = Phase::Downloading { got: 0, total: None };
    let progress = events::Latest::new(|s: &mut AppState, (got, total): (u64, Option<u64>)| {
        if let Phase::Downloading { .. } = s.update.phase {
            s.update.phase = Phase::Downloading { got, total };
            true
        } else {
            false
        }
    });
    std::thread::spawn(move || {
        let result = update::download(&found, |got, total| progress.send((got, total)));
        events::post(move |s| {
            s.update.phase = match result {
                Ok(()) => Phase::Ready,
                Err(e) => Phase::Failed(e),
            }
        });
    });
}

fn restart_now(s: &mut AppState) {
    update::relaunch_after_install();
    s.train.quit_requested = true;
}

fn skip(s: &mut AppState) {
    if let Some(f) = &s.update.found {
        let version = f.version.clone();
        super::onboarding::update_settings(|v| v["skip_update"] = json!(version));
    }
    s.update.found = None;
}

/// The first line of the release notes that says something (not a heading or an empty line).
fn summary(notes: &str) -> Option<String> {
    let line = notes.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#'))?;
    let line = line.trim_start_matches(['*', '-', ' ']).replace("**", "");
    Some(line)
}

pub fn banner(s: &AppState, p: Palette) -> Option<El> {
    let found = s.update.found.as_ref().filter(|_| !s.update.hidden)?;
    let v = &found.version;
    let auto = update::kind() == Kind::Auto;
    let (title, detail, progress): (String, String, Option<f32>) = match &s.update.phase {
        Phase::Offered => (
            format!("Mimvox {v} is available"),
            found.notes.as_deref().and_then(summary).unwrap_or_else(|| format!("You have {}.", env!("CARGO_PKG_VERSION"))),
            None,
        ),
        Phase::Downloading { got, total } => {
            let mb = |b: u64| b as f64 / 1_048_576.0;
            let size = total.map(|t| format!(" of {:.0} MB", mb(t))).unwrap_or_default();
            (format!("Downloading Mimvox {v}…"), format!("{:.1} MB{size}", mb(*got)), Some(total.map_or(0.0, |t| *got as f32 / t.max(1) as f32)))
        }
        Phase::Ready => (
            format!("Mimvox {v} is ready to install"),
            "It installs when you close Mimvox, or restart now. A running training or book pauses and continues after the restart."
                .to_owned(),
            None,
        ),
        Phase::Failed(e) => (format!("Mimvox {v} could not be downloaded"), e.clone(), None),
    };
    let mut buttons = row().gap(8.0).center_y();
    let whats_new = Button::new("What's new").ghost().small().key("update-notes").on_press(|s| s.update.open = Some(update::RELEASE_PAGE));
    match &s.update.phase {
        Phase::Offered => {
            buttons = buttons
                .child(Button::new("Skip this version").ghost().small().key("update-skip").on_press(skip).build(p))
                .child(Button::new("Later").ghost().small().key("update-later").on_press(|s| s.update.hidden = true).build(p))
                .child(whats_new.build(p));
            buttons = if auto {
                buttons.child(Button::new("Update").primary().small().icon(&icons::SPARKLE).key("update-go").on_press(start_download).build(p))
            } else {
                buttons.child(Button::new("Download").primary().small().icon(&icons::OPEN).key("update-page")
                    .on_press(|s| s.update.open = Some(update::RELEASE_PAGE)).build(p))
            };
        }
        Phase::Downloading { .. } => {}
        Phase::Ready => {
            buttons = buttons
                .child(Button::new("Later").ghost().small().key("update-later").on_press(|s| s.update.hidden = true).build(p))
                .child(Button::new("Restart now").primary().small().icon(&icons::REFRESH).key("update-restart").on_press(restart_now).build(p));
        }
        Phase::Failed(_) => {
            buttons = buttons
                .child(Button::new("Download page").ghost().small().key("update-page").on_press(|s| s.update.open = Some(update::RELEASE_PAGE)).build(p))
                .child(Button::new("Try again").primary().small().icon(&icons::REFRESH).key("update-retry").on_press(start_download).build(p));
        }
    }
    let mut info = col().grow(1.0).min_w(0.0).gap(2.0)
        .child(text(title).font(ty::LABEL).color(p.text))
        .child(text(detail).font(ty::CAPTION).color(p.text_dim).wrap().max_lines(2));
    if let Some(v) = progress {
        info = info.child(el().w_full().pt(4.0).child(progress_bar(p, Some(v))));
    }
    let row_ = row().key("update-banner").w_full().px(14.0).py(9.0).gap(12.0).center_y()
        .child(icon_el(&icons::SPARKLE, 16.0, p.accent))
        .child(info)
        .child(buttons);
    Some(el().w_full().px(16.0).child(glass_card(p).w_full().child(row_)))
}
