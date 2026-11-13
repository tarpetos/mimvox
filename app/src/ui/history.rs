//! Speak's and Redub's Results across restarts.
//!
//! Every result the engine saves has a sidecar (`<stem>.result.json` in `<data>/outputs`), so once the engine is
//! ready each list asks `outputs.list` for its kind (newest first, [`PAGE`] at a time) and fills its Results panel;
//! results made this session go on top as they finish. "Show older" at the bottom asks for the next page (`before` =
//! the oldest `created` shown). After an engine restart the list is asked again and merged by file name, so the
//! results already shown keep their cards.
//!
//! The engine's playback WAVs (`<data>/tmp/play`) are emptied at every engine start, so a result from an earlier
//! session saved in another format than WAV has nothing to play yet: its play button asks `outputs.play`, which
//! decodes one (once per session), and plays it when the answer comes ([`fetch_play`]). Trash, Export, Show in folder,
//! Open and "Open video" work on the saved files as for any result.

use super::components::Button;
use super::components::Tone;
use super::redub::Redub;
use super::shell;
use super::speak::Take;
use super::theme::Palette;
use super::trash::Kind;
use crate::engine::EngineMsg;
use crate::engine::protocol::{self, OutputEntry, OutputsPage};
use crate::state::AppState;
use rune::El;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// Results asked for at a time.
pub const PAGE: u32 = 50;

#[derive(Default)]
struct Listing {
    /// The `outputs.list` request waiting for its answer, and whether it asks for older results.
    req: Option<(u64, bool)>,
    /// Asked since the engine became ready.
    listed: bool,
    /// The engine has older results than the ones loaded.
    more: bool,
}

#[derive(Default)]
pub struct HistoryState {
    speak: Listing,
    redub: Listing,
    /// `outputs.play` requests waiting for their answer: (request id, list, result id).
    play: Vec<(u64, Kind, u64)>,
}

impl HistoryState {
    fn listing(&mut self, kind: Kind) -> &mut Listing {
        match kind {
            Kind::Speak => &mut self.speak,
            Kind::Redub => &mut self.redub,
        }
    }

    fn get(&self, kind: Kind) -> &Listing {
        match kind {
            Kind::Speak => &self.speak,
            Kind::Redub => &self.redub,
        }
    }
}

fn kind_str(kind: Kind) -> &'static str {
    match kind {
        Kind::Speak => "speak",
        Kind::Redub => "redub",
    }
}

/// Now as Unix seconds (the `created` of a result made this session).
pub fn now_unix() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64())
}

/// The engine has older results of `kind` than the list shows.
pub fn has_older(s: &AppState, kind: Kind) -> bool {
    s.history.get(kind).more
}

fn request(s: &mut AppState, kind: Kind, before: Option<f64>) {
    let (m, p) = protocol::outputs_list(kind_str(kind), PAGE, before);
    let req = shell::request(s, m, p);
    let l = s.history.listing(kind);
    l.req = req.map(|r| (r, before.is_some()));
    l.listed = true;
}

/// "Show older": the next page, older than the oldest result shown.
pub fn show_older(s: &mut AppState, kind: Kind) {
    if s.history.get(kind).req.is_some() {
        return;
    }
    let oldest = match kind {
        Kind::Speak => s.speak.takes.iter().map(|t| t.created).filter(|c| *c > 0.0).fold(f64::INFINITY, f64::min),
        Kind::Redub => s.redub.history.iter().map(|r| r.created).filter(|c| *c > 0.0).fold(f64::INFINITY, f64::min),
    };
    request(s, kind, oldest.is_finite().then_some(oldest));
}

/// The "Show older" button at the bottom of a Results list ("Loading…" while the page is on its way).
pub fn older_button(s: &AppState, p: Palette, kind: Kind) -> El {
    let loading = s.history.get(kind).req.is_some_and(|(_, older)| older);
    Button::new(if loading { "Loading…" } else { "Show older" })
        .ghost()
        .small()
        .key(format!("{}-older", kind_str(kind)))
        .disabled(loading)
        .on_press(move |s| show_older(s, kind))
        .build(p)
}

/// After every event: ask for both lists once the engine is ready.
pub fn settle(s: &mut AppState) -> bool {
    if s.shell.scripted() || !shell::engine_ready(s) {
        return false;
    }
    for kind in [Kind::Speak, Kind::Redub] {
        if !s.history.get(kind).listed {
            request(s, kind, None);
        }
    }
    false
}

/// Whether `play` is a WAV on disk the player can open now.
fn playable_now(play: &Path) -> bool {
    play.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")) && play.is_file()
}

/// Before playing a result: when it has no WAV to play (`play` is gone or is not a WAV), ask the engine for one
/// (`outputs.play` with the saved `file`) and return true; the answer updates the result and plays it. False when it
/// can play at once.
pub fn fetch_play(s: &mut AppState, kind: Kind, id: u64, file: &Path, play: &Path) -> bool {
    if playable_now(play) {
        return false;
    }
    if s.history.play.iter().any(|(_, k, i)| *k == kind && *i == id) {
        return true; // already asked
    }
    let (m, p) = protocol::outputs_play(file);
    match shell::request(s, m, p) {
        Some(req) => s.history.play.push((req, kind, id)),
        None => shell::toast(s, Tone::Danger, "Could not play the result: the engine is not running."),
    }
    true
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Ready(_) => {
            // A (re)started engine: ask again (merged with what is shown).
            for kind in [Kind::Speak, Kind::Redub] {
                let l = s.history.listing(kind);
                l.listed = false;
                l.req = None;
            }
            s.history.play.clear();
        }
        EngineMsg::Exited { .. } => {
            s.history.speak.req = None;
            s.history.redub.req = None;
            s.history.play.clear();
        }
        EngineMsg::Response { id, result } => {
            for kind in [Kind::Speak, Kind::Redub] {
                if let Some((_, older)) = s.history.get(kind).req.filter(|(r, _)| r == id) {
                    s.history.listing(kind).req = None;
                    match result.clone().and_then(protocol::parse_result::<OutputsPage>) {
                        Ok(page) => apply(s, kind, page, older),
                        Err(e) if older => shell::toast(s, Tone::Danger, format!("Could not load older results: {}", e.message)),
                        Err(_) => {} // the list stays as this session made it
                    }
                    return;
                }
            }
            if let Some(i) = s.history.play.iter().position(|(r, ..)| r == id) {
                let (_, kind, item) = s.history.play.remove(i);
                match result {
                    Ok(v) => {
                        let Some(path) = v.get("path_play").and_then(Value::as_str).filter(|p| !p.is_empty()).map(PathBuf::from) else { return };
                        if !playable_now(&path) {
                            shell::toast(s, Tone::Danger, "Could not play the result: the engine gave no WAV to play.");
                            return;
                        }
                        set_play(s, kind, item, path);
                        match kind {
                            Kind::Speak => super::speak::toggle_take(s, item),
                            Kind::Redub => super::redub::play_result(s, item),
                        }
                    }
                    Err(e) => shell::toast(s, Tone::Danger, format!("Could not play the result: {}", e.message)),
                }
            }
        }
        EngineMsg::Event(_) => {}
    }
}

fn set_play(s: &mut AppState, kind: Kind, id: u64, path: PathBuf) {
    match kind {
        Kind::Speak => {
            if let Some(t) = s.speak.takes.iter_mut().find(|t| t.id == id) {
                t.play = path;
            }
        }
        Kind::Redub => {
            if let Some(r) = s.redub.history.iter_mut().find(|r| r.id == id) {
                r.play = path;
            }
        }
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    a == b || (a.file_name().is_some() && a.file_name() == b.file_name())
}

/// Merges a page of `outputs.list` into a list: a result already shown keeps its card (and what this session knows of
/// it, like the original's preview); new ones are added; the whole list is sorted newest first. A fresh list (not
/// "older") also drops results whose file is gone.
pub fn apply(s: &mut AppState, kind: Kind, page: OutputsPage, older: bool) {
    s.history.listing(kind).more = page.more;
    match kind {
        Kind::Speak => {
            let mut list = std::mem::take(&mut s.speak.takes);
            if !older {
                list.retain(|t| t.path.is_file() || s.shell.scripted());
            }
            for e in page.results {
                if !list.iter().any(|t| same_file(&t.path, &e.path)) {
                    let id = s.speak.new_take_id();
                    list.push(Take::from_entry(id, e));
                }
            }
            list.sort_by(|a, b| b.created.total_cmp(&a.created));
            s.speak.takes = list;
        }
        Kind::Redub => {
            let mut list = std::mem::take(&mut s.redub.history);
            if !older {
                list.retain(|r| r.path.is_file() || s.shell.scripted());
            }
            for e in page.results {
                if !list.iter().any(|r| same_file(&r.path, &e.path)) {
                    let id = s.redub.new_id();
                    list.push(Redub::from_entry(id, e));
                }
            }
            list.sort_by(|a, b| b.created.total_cmp(&a.created));
            s.redub.history = list;
        }
    }
}

/// Scene helper: a page as the engine answers `outputs.list`.
pub fn scene_page(s: &mut AppState, kind: Kind, entries: Vec<OutputEntry>, more: bool) {
    apply(s, kind, OutputsPage { results: entries, more }, false);
}
