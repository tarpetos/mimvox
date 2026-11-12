//! Deleting saved results: the trash button on Speak's take cards and Redub's Results rows, its inline
//! confirmation ("Delete this result?" Delete / Cancel, like deleting a voice) and the engine request.
//!
//! Delete stops the result if it is playing (or loaded in the player bar), then asks the engine (`outputs.delete`,
//! which only deletes files in `<data>/outputs` and `<data>/tmp/play`) to remove the saved file and its playback
//! WAV. The entry leaves the list when the engine confirms; an error leaves it and shows a toast.

use super::components::{Button, Tone, icon_button, icons};
use super::playbar;
use super::shell;
use super::theme::{Palette, ty};
use crate::engine::EngineMsg;
use crate::state::AppState;
use rune::{El, row, text};
use serde_json::json;
use std::path::{Path, PathBuf};

/// Which list a result is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Speak,
    Redub,
}

#[derive(Default)]
pub struct TrashState {
    /// The result whose "Delete this result?" shows.
    confirm: Option<(Kind, u64)>,
    /// `outputs.delete` requests waiting for their answer: (request id, kind, result id).
    pending: Vec<(u64, Kind, u64)>,
}

fn key(kind: Kind, id: u64) -> String {
    format!("{}{id}", if kind == Kind::Speak { "take" } else { "redub" })
}

/// The trash button of a result (asks first).
pub fn button(p: Palette, kind: Kind, id: u64) -> El {
    icon_button(p, &format!("trash-{}", key(kind, id)), &icons::TRASH, "Delete this result", move |s| s.trash.confirm = Some((kind, id)))
}

/// The confirmation row replaces the result's buttons.
pub fn confirming(s: &AppState, kind: Kind, id: u64) -> bool {
    s.trash.confirm == Some((kind, id)) || deleting(s, kind, id)
}

fn deleting(s: &AppState, kind: Kind, id: u64) -> bool {
    s.trash.pending.iter().any(|(_, k, i)| *k == kind && *i == id)
}

/// "Delete this result?" Delete / Cancel (or "Deleting…" while the engine works). `path` is the saved file, `play`
/// its playback WAV (the same file for a WAV result).
pub fn confirm_row(s: &AppState, p: Palette, kind: Kind, id: u64, path: &Path, play: &Path) -> El {
    let k = key(kind, id);
    if deleting(s, kind, id) {
        return row().key(format!("trash-confirm-{k}")).w_full().h(28.0).center_y().child(text("Deleting…").font(ty::CAPTION).color(p.text_dim));
    }
    let (path, play) = (path.to_path_buf(), play.to_path_buf());
    row().key(format!("trash-confirm-{k}")).w_full().gap(8.0).center_y()
        .child(text("Delete this result?").font(ty::CAPTION).color(p.bad).grow(1.0).min_w(0.0).truncate())
        .child(Button::new("Delete").danger().small().key(format!("trash-yes-{k}")).on_press(move |s| delete(s, kind, id, &path, &play)).build(p))
        .child(Button::new("Cancel").ghost().small().key(format!("trash-no-{k}")).on_press(|s| s.trash.confirm = None).build(p))
}

fn delete(s: &mut AppState, kind: Kind, id: u64, path: &PathBuf, play: &PathBuf) {
    s.trash.confirm = None;
    playbar::stop_path(s, &[path.as_path(), play.as_path()]);
    let mut params = json!({ "path": path });
    if play != path {
        params["path_play"] = json!(play);
    }
    match shell::request(s, "outputs.delete", params) {
        Some(req) => s.trash.pending.push((req, kind, id)),
        None => shell::toast(s, Tone::Danger, "Could not delete the result: the engine is not running."),
    }
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Response { id, result } => {
            let Some(i) = s.trash.pending.iter().position(|(req, ..)| req == id) else { return };
            let (_, kind, item) = s.trash.pending.remove(i);
            match result {
                Ok(_) => match kind {
                    Kind::Speak => super::speak::remove_take(s, item),
                    Kind::Redub => super::redub::remove_result(s, item),
                },
                Err(e) => shell::toast(s, Tone::Danger, format!("Could not delete the result: {}", e.message)),
            }
        }
        EngineMsg::Exited { .. } if !s.trash.pending.is_empty() => {
            s.trash.pending.clear();
            shell::toast(s, Tone::Danger, "Could not delete the result: the engine stopped.");
        }
        _ => {}
    }
}

/// Scene helper: the confirmation open on a result.
pub fn scene_confirm(s: &mut AppState, kind: Kind, id: u64) {
    s.trash.confirm = Some((kind, id));
}
