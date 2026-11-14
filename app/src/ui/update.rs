//! Updates: the banner under the header ("Mimvox X is available", the download's progress, then "Restart now") and
//! the Updates dialog (the version in the header): every release from `update::OLDEST` on with its notes, "Check for
//! updates", and installing any listed version, newer or older.
//!
//! Once per launch (when the engine has settled: ready, stopped or not installed yet, so the check never competes
//! with the first frames and still runs on the welcome screen) a background thread asks for the newest release
//! (`crate::update::check`). A newer one shows the banner, unless the user skipped that version. "Update" (or a
//! version's button in the dialog) fetches that release's manifest, downloads the installer and verifies it in the
//! background; it installs when the app closes, or at once with "Restart now" (a running training or book pauses and
//! continues after the restart, as on any close). Where the app cannot replace itself (a .deb, a loose binary) the
//! banner and the dialog offer the release's download page instead.

use super::components::{Button, Tone, badge, icon_button, icon_el, icons, progress_bar};
use super::events;
use super::shell::EngineStatus;
use super::theme::{Palette, radius, ty};
use super::voices::kit;
use crate::state::AppState;
use crate::update::{self, Found, Kind, Release};
use rune::{Color, El, ScrollSpec, Tx, col, el, row, text};
use serde_json::json;
use std::cmp::Ordering;

#[derive(Default)]
pub struct UpdateState {
    /// The check was started this session.
    asked: bool,
    /// The version the banner offers or is installing.
    found: Option<Found>,
    phase: Phase,
    /// "Later": the banner is hidden until the next launch.
    hidden: bool,
    /// A link to open once the handler's frame is done (`Tx::open_url`).
    open: Option<String>,
    /// The Updates dialog is open.
    dialog: bool,
    /// The version shown in the dialog; `None`: the one being installed, else the installed one, else the newest.
    selected: Option<String>,
    list: List,
    /// "Check for updates" is running.
    checking: bool,
    /// What the last "Check for updates" said.
    checked: Option<(Tone, String)>,
    /// The scenes show an installed copy's dialog from a development build.
    kind: Option<Kind>,
}

fn kind(s: &AppState) -> Kind {
    s.update.kind.unwrap_or_else(update::kind)
}

#[derive(Default)]
enum Phase {
    #[default]
    Offered,
    Downloading { got: u64, total: Option<u64> },
    Ready,
    Failed(String),
}

/// The releases in the dialog.
#[derive(Default)]
enum List {
    #[default]
    NotAsked,
    Loading,
    Loaded(Vec<Release>),
    Failed(String),
}

/// `"check_updates": false` in settings.json turns the check off; `"skip_update": "<version>"` hides that version.
fn setting(key: &str) -> Option<serde_json::Value> {
    super::onboarding::read_settings().get(key).cloned()
}

/// The engine has done what it does at launch: it answers, it stopped, or there is none to start (the welcome
/// screen before setup).
fn engine_settled(s: &AppState) -> bool {
    matches!(s.shell.engine, EngineStatus::Ready { .. } | EngineStatus::Stopped { .. } | EngineStatus::Unavailable { .. })
}

/// After every event: start the check once (never in a scripted run or a development build), open a pending link.
pub fn settle(s: &mut AppState, tx: &mut Tx) -> bool {
    if let Some(url) = s.update.open.take() {
        let _ = tx.open_url(&url);
    }
    if s.update.asked || s.shell.scripted() || !engine_settled(s) {
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
                events::post(move |s| offer(s, found));
            }
        }
    });
    false
}

/// Shows `found` in the banner, unless a download is already running or waiting to install.
fn offer(s: &mut AppState, found: Found) {
    if matches!(s.update.phase, Phase::Downloading { .. } | Phase::Ready) {
        return;
    }
    s.update.found = Some(found);
    s.update.phase = Phase::Offered;
    s.update.hidden = false;
}

fn downloading(s: &AppState) -> bool {
    matches!(s.update.phase, Phase::Downloading { .. })
}

/// Fetches `version`'s manifest, downloads its installer and verifies it, all in the background. The banner and the
/// dialog show the progress; a finished download installs when the app closes.
fn start_install(s: &mut AppState, version: String) {
    if downloading(s) {
        return;
    }
    update::discard_pending();
    let notes = release(s, &version).map(|r| r.notes.clone()).or_else(|| s.update.found.as_ref().and_then(|f| f.notes.clone()));
    s.update.found = Some(Found::preview(&version, notes.as_deref()));
    s.update.phase = Phase::Downloading { got: 0, total: None };
    s.update.hidden = false;
    let progress = events::Latest::new(|s: &mut AppState, (got, total): (u64, Option<u64>)| {
        if let Phase::Downloading { .. } = s.update.phase {
            s.update.phase = Phase::Downloading { got, total };
            true
        } else {
            false
        }
    });
    std::thread::spawn(move || {
        let result = update::fetch(&version).and_then(|found| {
            update::download(&found, |got, total| progress.send((got, total)))?;
            Ok(found)
        });
        events::post(move |s| match result {
            Ok(found) => {
                s.update.found = Some(found);
                s.update.phase = Phase::Ready;
            }
            Err(e) => s.update.phase = Phase::Failed(e),
        });
    });
}

fn retry(s: &mut AppState) {
    if let Some(v) = s.update.found.as_ref().map(|f| f.version.clone()) {
        start_install(s, v);
    }
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

// ---- the dialog's data -------------------------------------------------------------------------------------------

/// Opens the Updates dialog on `version` (or the default choice) and asks for the releases the first time.
pub fn open_dialog(s: &mut AppState, version: Option<String>) {
    s.update.dialog = true;
    s.update.selected = version;
    if matches!(s.update.list, List::NotAsked | List::Failed(_)) {
        load_list(s);
    }
}

fn close_dialog(s: &mut AppState) {
    s.update.dialog = false;
}

fn load_list(s: &mut AppState) {
    if s.shell.scripted() || matches!(s.update.list, List::Loading) {
        return;
    }
    s.update.list = List::Loading;
    std::thread::spawn(|| {
        let result = update::releases();
        events::post(move |s| {
            s.update.list = match result {
                Ok(list) => List::Loaded(list),
                Err(e) => List::Failed(e),
            }
        });
    });
}

/// "Check for updates": asks for the newest release whatever was skipped, and refreshes the list.
fn check_now(s: &mut AppState) {
    if s.update.checking || s.shell.scripted() {
        return;
    }
    s.update.checking = true;
    s.update.checked = None;
    if !matches!(s.update.list, List::Loading) {
        load_list(s);
    }
    std::thread::spawn(|| {
        let result = update::check();
        events::post(move |s| {
            s.update.checking = false;
            match result {
                Ok(Some(found)) => {
                    let v = found.version.clone();
                    s.update.checked = Some((Tone::Accent, format!("Mimvox {v} is available.")));
                    s.update.selected = Some(v);
                    offer(s, found);
                }
                Ok(None) => {
                    s.update.checked = Some((Tone::Success, format!("You have the newest version, {}.", env!("CARGO_PKG_VERSION"))));
                }
                Err(e) => s.update.checked = Some((Tone::Danger, format!("Could not check for updates: {e}"))),
            }
        });
    });
}

fn releases(s: &AppState) -> &[Release] {
    match &s.update.list {
        List::Loaded(list) => list,
        _ => &[],
    }
}

fn release<'a>(s: &'a AppState, version: &str) -> Option<&'a Release> {
    releases(s).iter().find(|r| r.version == version)
}

/// The version the dialog shows.
fn selected(s: &AppState) -> Option<String> {
    let list = releases(s);
    let installing = s.update.found.as_ref().filter(|_| !matches!(s.update.phase, Phase::Offered)).map(|f| f.version.clone());
    s.update.selected.clone()
        .or(installing)
        .or_else(|| list.iter().find(|r| update::compare(&r.version) == Ordering::Equal).map(|r| r.version.clone()))
        .or_else(|| list.first().map(|r| r.version.clone()))
        .or_else(|| s.update.found.as_ref().map(|f| f.version.clone()))
}

// ---- release notes -----------------------------------------------------------------------------------------------

#[derive(Debug, PartialEq)]
enum Block {
    Heading(String),
    /// A bullet: its bold lead ("Romanian") when it has one, and the rest.
    Item(Option<String>, String),
    Para(String),
}

/// The release notes' Markdown as headings, bullets and paragraphs. A bullet's wrapped lines join it; links keep
/// their text; bold and code marks go.
fn blocks(md: &str) -> Vec<Block> {
    let mut raw: Vec<(u8, String)> = Vec::new(); // kind: 0 heading, 1 item, 2 paragraph
    let mut open = false;
    for line in md.lines() {
        let line = line.trim();
        if line.is_empty() {
            open = false;
        } else if line.starts_with('#') {
            raw.push((0, line.trim_start_matches('#').trim().to_owned()));
            open = false;
        } else if let Some(rest) = line.strip_prefix("* ").or_else(|| line.strip_prefix("- ")) {
            raw.push((1, rest.trim().to_owned()));
            open = true;
        } else if open && let Some(last) = raw.last_mut() {
            last.1.push(' ');
            last.1.push_str(line);
        } else {
            raw.push((2, line.to_owned()));
            open = true;
        }
    }
    raw.into_iter()
        .map(|(kind, body)| match kind {
            0 => Block::Heading(inline(&body)),
            1 => {
                let lead = body.strip_prefix("**").and_then(|b| b.split_once("**")).map(|(lead, rest)| (lead.to_owned(), rest.to_owned()));
                match lead {
                    Some((lead, rest)) => Block::Item(
                        Some(inline(lead.trim_end_matches(':'))),
                        capitalized(inline(rest.trim_start_matches(':').trim())),
                    ),
                    None => Block::Item(None, inline(&body)),
                }
            }
            _ => Block::Para(inline(&body)),
        })
        .collect()
}

/// The text under a bold lead starts a sentence of its own.
fn capitalized(s: String) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_lowercase() => c.to_uppercase().chain(chars).collect(),
        _ => s,
    }
}

/// `[text](url)` keeps its text; `**` and backticks go.
fn inline(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('[') {
        let after = &rest[i + 1..];
        match after.find("](").and_then(|j| after[j + 2..].find(')').map(|k| (j, j + 2 + k))) {
            Some((j, end)) => {
                out.push_str(&rest[..i]);
                out.push_str(&after[..j]);
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[..=i]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out.replace("**", "").replace('`', "")
}

/// The first line of the release notes that says something (not a heading or an empty line).
fn summary(notes: &str) -> Option<String> {
    blocks(notes).into_iter().find_map(|b| match b {
        Block::Item(Some(lead), rest) => Some(format!("{lead}: {rest}")),
        Block::Item(None, t) | Block::Para(t) => Some(t),
        Block::Heading(_) => None,
    })
}

fn notes_view(p: Palette, notes: &str) -> El {
    let mut c = col().w_full().gap(12.0);
    let list = blocks(notes);
    if list.is_empty() {
        return c.child(text("No notes for this version.").font(ty::BODY).color(p.text_faint));
    }
    for b in list {
        c = c.child(match b {
            Block::Heading(h) => text(h).font(ty::TITLE).color(p.text).pt(4.0),
            Block::Para(t) => text(t).font(ty::BODY).color(p.text_dim).wrap(),
            Block::Item(lead, body) => {
                let mut words = col().grow(1.0).min_w(0.0).gap(2.0);
                if let Some(lead) = lead {
                    words = words.child(text(lead).font(ty::LABEL).color(p.text).wrap());
                }
                if !body.is_empty() {
                    words = words.child(text(body).font(ty::BODY).color(p.text_dim).wrap());
                }
                row().w_full().gap(10.0)
                    .child(el().pt(7.0).child(el().size(5.0, 5.0).radius(2.5).fill(p.accent.alpha(0.8))))
                    .child(words)
            }
        });
    }
    c
}

// ---- views -------------------------------------------------------------------------------------------------------

/// The version in the header, a glass pill like the header's others: opens the Updates dialog. A newer version waiting
/// adds a sparkle (the update banner's icon) and turns the version accent.
pub fn version_button(s: &AppState, p: Palette) -> El {
    let waiting = s.update.found.as_ref().is_some_and(|f| update::compare(&f.version) == Ordering::Greater);
    let help = match &s.update.found {
        Some(f) if waiting => format!("Mimvox {} is available: updates and what's new", f.version),
        _ => "Updates and what's new".to_owned(),
    };
    let pill = row().key("version-button").role("button").label("Updates").h(28.0).px(11.0).gap(6.0).center().radius(14.0)
        .fill(if s.update.dialog { p.glass_hover } else { p.glass }).gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, if waiting { p.accent.alpha(0.45) } else { p.rim_bottom })
        .shadow(3.0, 10.0, p.shadow_soft);
    super::components::glass_edge(p, pill)
        .hover(p.glass_hover).focusable(true).cursor(rune::Cursor::Pointer).motion(super::theme::motion::UI).help(help)
        .on_press(|s: &mut AppState| open_dialog(s, None))
        .children(waiting.then(|| icon_el(&icons::SPARKLE, 12.0, p.accent)))
        .child(text(format!("v{}", env!("CARGO_PKG_VERSION"))).font(rune::TextStyle::new(11.5, 600.0).lh(15.0))
            .color(if waiting { p.accent } else { p.text_dim }).a11y_hidden())
}

pub fn banner(s: &AppState, p: Palette) -> Option<El> {
    let found = s.update.found.as_ref().filter(|_| !s.update.hidden)?;
    let v = &found.version;
    let auto = kind(s) == Kind::Auto;
    let (title, detail, progress): (String, String, Option<f32>) = match &s.update.phase {
        Phase::Offered => (
            format!("Mimvox {v} is available"),
            found.notes.as_deref().and_then(summary).unwrap_or_else(|| format!("You have {}.", env!("CARGO_PKG_VERSION"))),
            None,
        ),
        Phase::Downloading { got, total } => {
            (format!("Downloading Mimvox {v}…"), size_line(*got, *total), Some(fraction(*got, *total)))
        }
        Phase::Ready => (
            format!("Mimvox {v} is ready to install"),
            "It installs when you close Mimvox, or restart now. A running training or book pauses and continues after the restart."
                .to_owned(),
            None,
        ),
        Phase::Failed(e) => (format!("Mimvox {v} could not be downloaded"), e.clone(), None),
    };
    let version = v.clone();
    let whats_new = Button::new("What's new").ghost().small().key("update-notes")
        .on_press(move |s| open_dialog(s, Some(version.clone())));
    let mut buttons = row().gap(8.0).center_y();
    match &s.update.phase {
        Phase::Offered => {
            buttons = buttons
                .child(Button::new("Skip this version").ghost().small().key("update-skip").on_press(skip).build(p))
                .child(Button::new("Later").ghost().small().key("update-later").on_press(|s| s.update.hidden = true).build(p))
                .child(whats_new.build(p));
            let version = v.clone();
            buttons = if auto {
                buttons.child(Button::new("Update").primary().small().icon(&icons::SPARKLE).key("update-go")
                    .on_press(move |s| start_install(s, version.clone())).build(p))
            } else {
                buttons.child(Button::new("Download").primary().small().icon(&icons::OPEN).key("update-page")
                    .on_press(move |s| s.update.open = Some(update::release_page(&version))).build(p))
            };
        }
        Phase::Downloading { .. } => {}
        Phase::Ready => {
            buttons = buttons
                .child(Button::new("Later").ghost().small().key("update-later").on_press(|s| s.update.hidden = true).build(p))
                .child(Button::new("Restart now").primary().small().icon(&icons::REFRESH).key("update-restart").on_press(restart_now).build(p));
        }
        Phase::Failed(_) => {
            let version = v.clone();
            buttons = buttons
                .child(Button::new("Download page").ghost().small().key("update-page")
                    .on_press(move |s| s.update.open = Some(update::release_page(&version))).build(p))
                .child(Button::new("Try again").primary().small().icon(&icons::REFRESH).key("update-retry").on_press(retry).build(p));
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
    Some(el().w_full().px(16.0).child(super::components::glass_card(p).w_full().child(row_)))
}

fn mb(b: u64) -> f64 {
    b as f64 / 1_048_576.0
}

fn size_line(got: u64, total: Option<u64>) -> String {
    let size = total.map(|t| format!(" of {:.0} MB", mb(t))).unwrap_or_default();
    format!("{:.1} MB{size}", mb(got))
}

fn fraction(got: u64, total: Option<u64>) -> f32 {
    total.map_or(0.0, |t| got as f32 / t.max(1) as f32)
}

/// The Updates dialog over the window, or `None` when it is closed.
pub fn overlay(s: &AppState, p: &Palette) -> Option<El> {
    if !s.update.dialog {
        return None;
    }
    let p = *p;
    let current = env!("CARGO_PKG_VERSION");
    let kind = kind(s);
    let subtitle = match kind {
        Kind::Auto => format!("You have Mimvox {current}. Any version from {} on can be installed here, older ones too.", update::OLDEST),
        Kind::Manual => format!("You have Mimvox {current}. This copy cannot replace itself: install a version from its download page."),
        Kind::Never => format!("You have Mimvox {current}, a development build: it never installs releases."),
    };
    let check = Button::new(if s.update.checking { "Checking…" } else { "Check for updates" }).ghost().small().icon(&icons::REFRESH)
        .key("updates-check").disabled(s.update.checking).on_press(check_now).build(p);
    let head = row().w_full().gap(12.0).center_y()
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(text("Updates").font(ty::TITLE).color(p.text))
            .child(text(subtitle).font(ty::CAPTION).color(p.text_dim).wrap()))
        .child(check)
        .child(icon_button(p, "updates-close", &icons::CLOSE, "Close", close_dialog));
    let mut card = col().key("updates-dialog").w(860.0).h(560.0).padding(22.0).gap(14.0).radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 1.0)).border_gradient(1.0, p.rim_top, p.rim_bottom).shadow(18.0, 50.0, p.shadow)
        .on_press(|_: &mut AppState| {})
        .child(head);
    if let Some((tone, msg)) = &s.update.checked {
        card = card.child(kit::notice(&p, *tone, msg));
    }
    let chosen = selected(s);
    card = card.child(row().key("updates-body").w_full().grow(1.0).min_h(0.0).gap(14.0)
        .child(version_list(s, p, chosen.as_deref()))
        .child(details(s, p, chosen.as_deref(), kind)));
    let scrim = el().key("updates-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(close_dialog);
    Some(el().key("updates-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).center().fill(Color::BLACK.alpha(0.55))
        .child(scrim).child(card))
}

fn version_list(s: &AppState, p: Palette, chosen: Option<&str>) -> El {
    let mut list = col().w_full().gap(4.0);
    match &s.update.list {
        List::NotAsked | List::Loading => list = list.child(text("Loading versions…").font(ty::CAPTION).color(p.text_faint).px(10.0).py(8.0)),
        List::Failed(e) => {
            list = list
                .child(text(format!("The versions could not be loaded: {e}")).font(ty::CAPTION).color(p.text_dim).wrap().px(10.0).py(6.0))
                .child(el().px(10.0).child(Button::new("Try again").ghost().small().key("updates-reload").on_press(load_list).build(p)));
        }
        List::Loaded(all) if all.is_empty() => list = list.child(text("No versions found.").font(ty::CAPTION).color(p.text_faint).px(10.0).py(8.0)),
        List::Loaded(all) => {
            for r in all {
                let on = chosen == Some(r.version.as_str());
                let version = r.version.clone();
                let tag = match update::compare(&r.version) {
                    Ordering::Equal => Some(badge(p, "Installed", Tone::Success)),
                    Ordering::Greater => Some(badge(p, "New", Tone::Accent)),
                    Ordering::Less => None,
                };
                list = list.child(
                    row().key(format!("updates-v{}", r.version)).role("tab").label(format!("Mimvox {}", r.version)).checked(on)
                        .w_full().h(46.0).px(12.0).gap(8.0).center_y().radius(radius::CARD)
                        .fill(if on { p.accent.alpha(0.14) } else { Color::TRANSPARENT }).hover(p.glass_hover).focusable(true)
                        .on_press(move |s: &mut AppState| s.update.selected = Some(version.clone()))
                        .child(col().grow(1.0).min_w(0.0).gap(1.0)
                            .child(text(r.version.clone()).font(ty::LABEL).color(p.text))
                            .children((!r.date.is_empty()).then(|| text(r.date.clone()).font(ty::CAPTION).color(p.text_faint))))
                        .children(tag),
                );
            }
        }
    }
    col().key("updates-list").w(200.0).h_full().scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(12.0).child(list)
}

fn details(s: &AppState, p: Palette, chosen: Option<&str>, kind: Kind) -> El {
    let pane = super::components::well(p).key("updates-details").grow(1.0).min_w(0.0).h_full().padding(18.0).gap(12.0);
    let Some(v) = chosen else {
        return pane.child(text("Choose a version on the left.").font(ty::BODY).color(p.text_faint));
    };
    let r = release(s, v);
    let notes = r.map(|r| r.notes.clone()).or_else(|| s.update.found.as_ref().filter(|f| f.version == v).and_then(|f| f.notes.clone()));
    let date = r.map(|r| r.date.clone()).filter(|d| !d.is_empty());
    let head = row().w_full().gap(10.0).center_y()
        .child(text(format!("Mimvox {v}")).font(ty::TITLE).color(p.text))
        .children(date.map(|d| text(d).font(ty::CAPTION).color(p.text_faint)))
        .child(el().grow(1.0))
        .child(Button::new("Release page").ghost().small().icon(&icons::OPEN).key("updates-page")
            .on_press({
                let v = v.to_owned();
                move |s| s.update.open = Some(update::release_page(&v))
            }).build(p));
    // Keyed by version: another version's notes start at the top, not where the last ones were scrolled to.
    let body = col().key(format!("updates-notes-{v}")).w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0)
        .child(match notes {
            Some(n) => notes_view(p, &n),
            None => text("Loading the notes…").font(ty::BODY).color(p.text_faint),
        });
    pane.child(head).child(body).child(action(s, p, v, kind))
}

/// What can be done with version `v`: update, install the older version, its progress, or nothing (installed).
fn action(s: &AppState, p: Palette, v: &str, kind: Kind) -> El {
    let foot = col().key("updates-action").w_full().gap(10.0);
    let order = update::compare(v);
    if order == Ordering::Equal {
        return foot.child(row().gap(8.0).center_y()
            .child(icon_el(&icons::CHECK, 14.0, p.ok))
            .child(text("This is the version you have.").font(ty::BODY).color(p.text_dim)));
    }
    let this = s.update.found.as_ref().is_some_and(|f| f.version == v);
    match (&s.update.phase, this) {
        (Phase::Downloading { got, total }, true) => {
            return foot
                .child(text(format!("Downloading Mimvox {v}…  {}", size_line(*got, *total))).font(ty::BODY).color(p.text_dim))
                .child(progress_bar(p, Some(fraction(*got, *total))));
        }
        (Phase::Ready, true) => {
            return foot.child(row().w_full().gap(10.0).center_y()
                .child(text(format!("Mimvox {v} is downloaded and verified. It installs when you close Mimvox.")).font(ty::BODY).color(p.text_dim).wrap().grow(1.0).min_w(0.0))
                .child(Button::new("Restart now").primary().small().icon(&icons::REFRESH).key("updates-restart").on_press(restart_now).build(p)));
        }
        (Phase::Failed(e), true) => {
            return foot
                .child(kit::notice(&p, Tone::Danger, &format!("Mimvox {v} could not be downloaded: {e}")))
                .child(row().w_full().justify_end()
                    .child(Button::new("Try again").primary().small().icon(&icons::REFRESH).key("updates-retry").on_press(retry).build(p)));
        }
        _ => {}
    }
    let older = order == Ordering::Less;
    let mut foot = foot;
    if older {
        foot = foot.child(kit::notice(&p, Tone::Warning,
            "An older version: what came after it is gone until you update again. Your voices, results and settings stay, \
             and it offers the newest version again when it starts."));
    }
    let version = v.to_owned();
    let button = match kind {
        Kind::Auto => {
            let label = if older { format!("Install {v}") } else { format!("Update to {v}") };
            let b = Button::new(label).small().icon(&icons::SPARKLE).key("updates-install").disabled(downloading(s))
                .on_press(move |s| start_install(s, version.clone()));
            if older { b.secondary() } else { b.primary() }
        }
        Kind::Manual | Kind::Never => Button::new("Download page").primary().small().icon(&icons::OPEN).key("updates-install")
            .on_press(move |s| s.update.open = Some(update::release_page(&version))),
    };
    let note = if downloading(s) {
        "Another version is downloading."
    } else if kind == Kind::Auto {
        "It downloads in the background and installs when you close Mimvox (or at once with Restart now)."
    } else {
        ""
    };
    foot.child(row().w_full().gap(10.0).center_y()
        .child(text(note).font(ty::CAPTION).color(p.text_faint).wrap().grow(1.0).min_w(0.0))
        .child(button.build(p)))
}

// -------------------------------------------------------------------------------------------------
// Headless scenes
// -------------------------------------------------------------------------------------------------

const NOTES_030: &str = "\n* **Romanian:** Speak, Book and Redub now offer Romanian. Record has Romanian reading texts too (Quick,\n  Standard and Pro), so a voice can be recorded and trained in Romanian from start to finish.\n* **A voice from recordings you already have:** Record's start page now begins with a card for audio or video\n  files of one person: no reading and no preset.\n* **Player:** no Stop button any more (pause, the knob and Close cover it).\n";
const NOTES_020: &str = "\n* **Installers for every platform:** Windows x64 (setup .exe), Linux x64 (AppImage and .deb) and macOS on Apple\n  Silicon (.dmg), built by GitHub Actions for every release.\n* **Updates from inside the app:** a new version shows a banner under the header.\n* **Install only what you need:** the welcome screen lists the parts of the runtime with their sizes.\n";

fn demo(s: &mut AppState) {
    super::speak::scene_idle(s);
    let current = env!("CARGO_PKG_VERSION").to_owned();
    s.update.list = List::Loaded(vec![
        Release { version: "0.4.0".into(), date: "2026-11-20".into(), notes: "\n* **Updates window:** every version with its notes; install any of them, older ones too.\n* **Check for updates** whenever you like, from the version in the header.\n".into() },
        Release { version: current, date: "2026-11-14".into(), notes: NOTES_030.into() },
        Release { version: "0.2.0".into(), date: "2026-11-13".into(), notes: NOTES_020.into() },
    ]);
    s.update.dialog = true;
    s.update.kind = Some(Kind::Auto);
}

pub fn scene_newer(s: &mut AppState) {
    demo(s);
    s.update.found = Some(Found::preview("0.4.0", Some("\n* **Updates window:** every version with its notes.\n")));
    s.update.selected = Some("0.4.0".into());
    s.update.checked = Some((Tone::Accent, "Mimvox 0.4.0 is available.".into()));
}

pub fn scene_older(s: &mut AppState) {
    demo(s);
    s.update.selected = Some("0.2.0".into());
}

pub fn scene_downloading(s: &mut AppState) {
    demo(s);
    s.update.found = Some(Found::preview("0.4.0", None));
    s.update.phase = Phase::Downloading { got: 96_468_992, total: Some(214_958_080) };
}

/// The header and the banner with a newer version waiting (the dialog closed).
pub fn scene_waiting(s: &mut AppState) {
    super::speak::scene_idle(s);
    s.update.kind = Some(Kind::Auto);
    s.update.found = Some(Found::preview("0.4.0", Some("\n* **Updates window:** every version with its notes.\n")));
}

pub fn scene_loading(s: &mut AppState) {
    demo(s);
    s.update.list = List::Loading;
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("updates_newer", scene_newer),
        ("updates_older", scene_older),
        ("updates_downloading", scene_downloading),
        ("updates_loading", scene_loading),
        ("updates_waiting", scene_waiting),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_become_headings_bullets_and_paragraphs() {
        let md = "## Added\n\n* **Romanian:** Speak and Book now offer it.\n  More [here](https://x.y) and `code`.\n* plain item\n\nA closing line\nthat wraps.\n";
        assert_eq!(blocks(md), vec![
            Block::Heading("Added".into()),
            Block::Item(Some("Romanian".into()), "Speak and Book now offer it. More here and code.".into()),
            Block::Item(None, "plain item".into()),
            Block::Para("A closing line that wraps.".into()),
        ]);
    }

    #[test]
    fn summary_is_the_first_bullet() {
        assert_eq!(summary(NOTES_030).as_deref().map(|s| s.starts_with("Romanian: Speak, Book and Redub")), Some(true));
        assert_eq!(inline("a [b](c) [d e"), "a b [d e");
    }
}
