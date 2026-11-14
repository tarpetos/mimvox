//! Voices screen: list, create, rename, delete and select the active voice.
//!
//! Also hosts the small glass widget kit (`kit`) and the engine request plumbing (`Link`, `pump`)
//! shared by the train and onboarding screens, so the three screens look and behave alike.

use std::collections::HashMap;

use rune::{Cx, El, ScrollSpec, TextEdit, col, el, row, text, text_input};
use serde_json::{Value, json};

use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use crate::ui::components::{self, Button, Tone};
use crate::ui::shell::{self, VoiceInfo};
use crate::ui::theme::{self, Palette, radius, ty};

use self::kit::BtnKind;

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Tag {
    Create,
    Rename,
    Delete(String),
}

pub struct Renaming {
    pub id: String,
    pub edit: TextEdit,
}

pub struct VoicesState {
    pub error: Option<String>,
    pub new_name: TextEdit,
    pub renaming: Option<Renaming>,
    pub confirm_delete: Option<String>,
    pending: HashMap<u64, Tag>,
    last_screen: Option<Screen>,
    /// The voice saved in settings is re-selected once, when the first list arrives.
    restore: Option<String>,
    /// The active voice's reference clip (see `refclip`).
    pub refclip: crate::ui::refclip::RefState,
}

impl Default for VoicesState {
    fn default() -> Self {
        VoicesState {
            error: None,
            new_name: TextEdit::new(""),
            renaming: None,
            confirm_delete: None,
            pending: HashMap::new(),
            last_screen: None,
            restore: crate::ui::onboarding::saved_active_voice(),
            refclip: Default::default(),
        }
    }
}

/// Sends a request and remembers what it was for. Without an engine the user sees an error.
fn send(s: &mut AppState, tag: Tag, method: &str, params: Value) {
    match shell::request(s, method, params) {
        Some(id) => {
            s.voices.pending.insert(id, tag);
        }
        None => s.voices.error = Some("The voice engine is not running. Check the status at the top and press Retry.".into()),
    }
}

/// The voice every screen works on (the one picked here or in the header), if any.
pub fn active_voice(s: &AppState) -> Option<&str> {
    s.shell.active_voice.as_deref()
}

pub fn active_voice_info(s: &AppState) -> Option<&VoiceInfo> {
    shell::active_voice(s)
}

pub fn voice_name<'a>(s: &'a AppState, id: &str) -> Option<&'a str> {
    s.shell.voices.iter().find(|v| v.id == id).map(|v| v.name.as_str())
}

/// Selects `id` as the active voice and remembers it.
pub fn select(s: &mut AppState, id: &str) {
    if s.shell.active_voice.as_deref() == Some(id) {
        return;
    }
    s.shell.active_voice = Some(id.to_string());
    crate::ui::onboarding::save_active_voice(Some(id));
}

/// After every event: refresh on screen entry, restore the saved selection.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;
    if s.voices.last_screen != Some(s.screen) {
        s.voices.last_screen = Some(s.screen);
        s.voices.refclip.mark_stale();
        if matches!(s.screen, Screen::Voices | Screen::Train) && shell::engine_ready(s) {
            shell::refresh_voices(s);
        }
    }
    if let Some(id) = s.voices.restore.clone() {
        if !s.shell.voices.is_empty() {
            s.voices.restore = None;
            if s.shell.voices.iter().any(|v| v.id == id) {
                s.shell.active_voice = Some(id);
                changed = true;
            }
        }
    }
    crate::ui::refclip::settle(s) | changed
}

// ---------------------------------------------------------------------------------------------
// Engine messages
// ---------------------------------------------------------------------------------------------

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    crate::ui::refclip::on_engine(s, msg);
    match msg {
        EngineMsg::Event(ev) if ev.event == "train.done" => shell::refresh_voices(s),
        EngineMsg::Exited { .. } => s.voices.pending.clear(),
        EngineMsg::Response { id, result } => {
            let Some(tag) = s.voices.pending.remove(id) else { return };
            match (tag, result) {
                (Tag::Create, Ok(v)) => {
                    s.voices.error = None;
                    if let Some(info) = v.get("voice").and_then(VoiceInfo::from_json) {
                        let id = info.id.clone();
                        s.shell.voices.push(info);
                        select(s, &id);
                        // An empty voice is no use yet: straight on to giving it recordings (files or reading).
                        s.record.back_to_pick();
                        s.record.for_new = false;
                        shell::set_screen(s, Screen::Record);
                    }
                    shell::refresh_voices(s);
                }
                (Tag::Rename, Ok(_)) => shell::refresh_voices(s),
                (Tag::Delete(id), Ok(_)) => {
                    s.shell.voices.retain(|v| v.id != id);
                    if s.shell.active_voice.as_deref() == Some(id.as_str()) {
                        s.shell.active_voice = s.shell.voices.first().map(|v| v.id.clone());
                        crate::ui::onboarding::save_active_voice(s.shell.active_voice.as_deref());
                    }
                    shell::refresh_voices(s);
                }
                (_, Err(e)) => {
                    s.voices.error = Some(e.message.clone());
                    shell::refresh_voices(s);
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------------------------

fn create(s: &mut AppState) {
    let name = s.voices.new_name.text.trim().to_string();
    if name.is_empty() {
        return;
    }
    send(s, Tag::Create, "voices.create", json!({ "name": name }));
    s.voices.new_name = TextEdit::new("");
}

fn commit_rename(s: &mut AppState) {
    if let Some(r) = s.voices.renaming.take() {
        let name = r.edit.text.trim().to_string();
        if !name.is_empty() {
            if let Some(v) = s.shell.voices.iter_mut().find(|v| v.id == r.id) {
                v.name = name.clone();
            }
            send(s, Tag::Rename, "voices.rename", json!({ "id": r.id, "name": name }));
        }
    }
}

fn start_rename(s: &mut AppState, id: &str) {
    if let Some(v) = s.shell.voices.iter().find(|v| v.id == id) {
        let mut edit = TextEdit::new(v.name.clone());
        edit.focused = true;
        s.voices.renaming = Some(Renaming { id: id.to_string(), edit });
        s.voices.confirm_delete = None;
    }
}

fn delete(s: &mut AppState, id: &str) {
    s.voices.confirm_delete = None;
    send(s, Tag::Delete(id.to_string()), "voices.delete", json!({ "id": id }));
}

// ---------------------------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------------------------

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    let v = &s.voices;

    let create_card = kit::card(&p)
        .gap(12.0)
        .child(text("New voice").font(ty::TITLE).color(p.text))
        .child(
            row()
                .w_full()
                .gap(10.0)
                .center_y()
                .child(
                    kit::field(&p, &v.new_name, "Name your voice, for example \"My voice\"", |s: &mut AppState| {
                        &mut s.voices.new_name
                    })
                    .on_submit(|s: &mut AppState, _| create(s)),
                )
                .child(kit::button(&p, "Create voice", BtnKind::Primary, !v.new_name.text.trim().is_empty(), create)),
        );

    let mut list = col().w_full().gap(14.0);
    if s.shell.voices.is_empty() {
        list = list.child(empty_state(&p, shell::engine_ready(s)));
    }
    for info in &s.shell.voices {
        list = list.child(voice_card(s, &p, info));
    }

    let mut body = col().w_full().gap(18.0).child(kit::header(
        &p,
        "Your voices",
        "Each voice keeps its own recordings and trained model. Pick one to work on.",
    ));
    if let Some(err) = &v.error {
        body = body.child(kit::notice(&p, Tone::Danger, err));
    }
    body = body.child(create_card).child(list).child(crate::ui::relocate::footer(s, &p));
    kit::page(body)
}

fn empty_state(p: &Palette, loaded: bool) -> El {
    let (title, sub) = if loaded {
        ("No voices yet", "Create your first voice above, then record about six minutes of reading.")
    } else {
        ("Starting the engine...", "Your voices will appear here in a moment.")
    };
    kit::card(p)
        .center()
        .gap(6.0)
        .py(28.0)
        .child(text(title).font(ty::TITLE).color(p.text))
        .child(text(sub).font(ty::BODY).color(p.text_dim))
}

fn voice_card(s: &AppState, p: &Palette, info: &VoiceInfo) -> El {
    let v = &s.voices;
    let active = s.shell.active_voice.as_deref() == Some(info.id.as_str());
    let id = info.id.clone();
    let renaming = v.renaming.as_ref().filter(|r| r.id == info.id);
    let confirming = v.confirm_delete.as_deref() == Some(info.id.as_str());

    let initial = info.name.chars().next().map(|c| c.to_uppercase().to_string()).unwrap_or_else(|| "?".into());
    let avatar = el()
        .size(48.0, 48.0)
        .radius(radius::CONTROL)
        .center()
        .fill(p.accent)
        .child(text(initial).font(ty::TITLE).color(p.on_accent));

    let title: El = match renaming {
        Some(r) => {
            let rid = id.clone();
            let _ = rid;
            kit::field(p, &r.edit, "Voice name", |s: &mut AppState| {
                &mut s.voices.renaming.as_mut().expect("renaming").edit
            })
            .on_submit(|s: &mut AppState, _| commit_rename(s))
        }
        None => text(info.name.clone()).font(ty::TITLE).color(p.text).truncate(),
    };

    let mut meta = row().gap(8.0).center_y().flex_wrap();
    meta = meta.child(kit::badge(
        p,
        if info.trained { "Trained" } else { "Untrained" },
        if info.trained { Tone::Success } else { Tone::Warning },
    ));
    if let Some(pz) = &info.paused {
        let label = if pz.interrupted { format!("Interrupted {}/{}", pz.epoch, pz.total_epochs) } else { format!("Paused {}/{}", pz.epoch, pz.total_epochs) };
        meta = meta.child(kit::badge(p, &label, Tone::Warning));
    }
    if active {
        meta = meta.child(kit::badge(p, "Active", Tone::Accent));
    }
    let mins = if info.dataset_minutes > 0.0 {
        format!("{:.1} min recorded", info.dataset_minutes)
    } else {
        "No recordings yet".to_string()
    };
    meta = meta.child(text(mins).font(ty::CAPTION).color(p.text_dim));
    if info.trained && info.trained_epochs > 0 {
        meta = meta.child(text(format!("{} epochs", info.trained_epochs)).font(ty::CAPTION).color(p.text_dim));
    }

    let info_col = col().grow(1.0).min_w(0.0).gap(6.0).child(title).child(meta);

    let mut actions = row().gap(8.0).center_y();
    if let Some(_) = renaming {
        actions = actions
            .child(kit::button(p, "Save", BtnKind::Primary, true, commit_rename))
            .child(kit::button(p, "Cancel", BtnKind::Glass, true, |s: &mut AppState| s.voices.renaming = None));
    } else if confirming {
        let did = id.clone();
        actions = actions
            .child(text("Delete this voice and its recordings?").font(ty::CAPTION).color(p.bad))
            .child(kit::button(p, "Delete", BtnKind::Danger, true, move |s: &mut AppState| delete(s, &did)))
            .child(kit::button(p, "Keep", BtnKind::Glass, true, |s: &mut AppState| s.voices.confirm_delete = None));
    } else {
        let (sel, ren, del) = (id.clone(), id.clone(), id.clone());
        if !active {
            actions = actions.child(kit::button(p, "Use this voice", BtnKind::Primary, true, move |s: &mut AppState| {
                select(s, &sel)
            }));
        }
        actions = actions
            .child(kit::button(p, "Rename", BtnKind::Ghost, true, move |s: &mut AppState| start_rename(s, &ren)))
            .child(kit::button(p, "Delete", BtnKind::Ghost, true, move |s: &mut AppState| {
                s.voices.renaming = None;
                s.voices.confirm_delete = Some(del.clone())
            }));
    }

    let mut card = kit::card(p).key(format!("voice-{}", info.id)).gap(14.0);
    if active {
        card = card.border(1.5, p.accent);
    }
    let compact = renaming.is_none() && !confirming;
    let mut top = row().w_full().gap(14.0).center_y().child(avatar).child(info_col);
    if compact {
        top = top.child(actions);
        card = card.child(top);
    } else {
        card = card.child(top).child(row().w_full().gap(8.0).center_y().justify_end().flex_wrap().child(actions));
    }
    if active && renaming.is_none() && !confirming {
        card = card.child(crate::ui::refclip::section(s, p, info));
        let go = row().w_full().gap(8.0).center_y().child(text("Next:").font(ty::CAPTION).color(p.text_faint))
            .child(kit::button(p, "Record", BtnKind::Glass, true, |s: &mut AppState| s.screen = Screen::Record))
            .child(kit::button(p, "Train", BtnKind::Glass, true, |s: &mut AppState| s.screen = Screen::Train))
            .child(kit::button(p, "Speak", BtnKind::Glass, true, |s: &mut AppState| s.screen = Screen::Speak));
        card = card.child(go);
    }
    card
}

// ---------------------------------------------------------------------------------------------
// Scenes for headless snapshots
// ---------------------------------------------------------------------------------------------

fn demo_voices() -> Vec<VoiceInfo> {
    vec![
        VoiceInfo { preset: None, id: "my-voice".into(), name: "My voice".into(), dataset_minutes: 6.4, parts: 5, trained: true, trained_epochs: 300, ..Default::default() },
        VoiceInfo { preset: None, id: "narrator".into(), name: "Narrator".into(), dataset_minutes: 2.1, parts: 2, trained: false, trained_epochs: 0, ..Default::default() },
        VoiceInfo { preset: None, id: "quick".into(), name: "Quick test".into(), dataset_minutes: 0.0, parts: 0, trained: false, trained_epochs: 0, ..Default::default() },
    ]
}

pub(crate) fn scene_list(s: &mut AppState) {
    s.screen = Screen::Voices;
    ready(s);
    s.shell.voices = demo_voices();
    s.shell.active_voice = Some("my-voice".into());
}
fn scene_paused(s: &mut AppState) {
    scene_list(s);
    let paused = |epoch, interrupted| Some(crate::engine::protocol::PausedTraining { epoch, total_epochs: 300, paused_at: Some("2026-11-11T17:42:10".into()), interrupted });
    s.shell.voices[1].paused = paused(118, false);
    s.shell.voices[2].paused = paused(45, true);
}
fn scene_empty(s: &mut AppState) {
    s.screen = Screen::Voices;
    ready(s);
    s.shell.active_voice = None;
}
fn scene_loading(s: &mut AppState) {
    s.screen = Screen::Voices;
    s.shell.set_engine(shell::EngineStatus::Starting { since: 0.0 });
}
fn ready(s: &mut AppState) {
    s.screen = Screen::Voices;
    s.shell.set_engine(shell::EngineStatus::Ready { device: None, tts: Some("ready".into()), rvc: Some("ready".into()) });
}
fn scene_rename(s: &mut AppState) {
    scene_list(s);
    start_rename(s, "narrator");
}
fn scene_delete(s: &mut AppState) {
    scene_list(s);
    s.voices.confirm_delete = Some("narrator".into());
}
fn scene_error(s: &mut AppState) {
    scene_list(s);
    s.voices.error = Some("The engine stopped unexpectedly. Restart the app.".into());
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("voices-list", scene_list),
        ("voices-paused", scene_paused),
        ("voices-empty", scene_empty),
        ("voices-loading", scene_loading),
        ("voices-rename", scene_rename),
        ("voices-delete", scene_delete),
        ("voices-error", scene_error),
    ]
}

// ---------------------------------------------------------------------------------------------
// Glass widget kit (local; mirrors the shell's palette and type scale)
// ---------------------------------------------------------------------------------------------

pub(crate) mod kit {
    use super::*;
    use rune::Shape;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum BtnKind {
        Primary,
        Glass,
        Ghost,
        Danger,
    }

    /// Scrollable page body with the shared gutters; content is limited to a readable width.
    pub fn page(body: El) -> El {
        col()
            .w_full()
            .h_full()
            .min_h(0.0)
            .scroll(ScrollSpec { stick_bottom: false })
            .child(col().w_full().items_center().px(28.0).py(24.0).child(body.w_full().max_w(760.0)))
    }

    pub fn header(p: &Palette, title: &str, sub: &str) -> El {
        col()
            .w_full()
            .gap(6.0)
            .child(text(title.to_string()).font(ty::DISPLAY).color(p.text))
            .child(text(sub.to_string()).font(ty::BODY).color(p.text_dim).wrap())
    }

    /// A glass card with the shared inner padding.
    pub fn card(p: &Palette) -> El {
        components::glass_card(*p).w_full().padding(18.0)
    }

    /// The main content panel of a screen (one of the few blurred surfaces).
    pub fn panel(p: &Palette) -> El {
        components::glass_panel(*p).w_full().padding(24.0)
    }

    pub fn button(p: &Palette, label: &str, kind: BtnKind, enabled: bool, on_press: impl Fn(&mut AppState) + 'static) -> El {
        let b = Button::new(label).small().disabled(!enabled).on_press(on_press);
        match kind {
            BtnKind::Primary => b.primary(),
            BtnKind::Glass => b.secondary(),
            BtnKind::Ghost => b.ghost(),
            BtnKind::Danger => b.danger(),
        }
        .build(*p)
    }

    /// The big call-to-action button.
    pub fn big_button(p: &Palette, label: &str, enabled: bool, on_press: impl Fn(&mut AppState) + 'static) -> El {
        Button::new(label).primary().large().key(format!("big-{label}")).disabled(!enabled).on_press(on_press).build(*p)
    }

    pub fn badge(p: &Palette, label: &str, tone: Tone) -> El {
        components::badge(*p, label, tone)
    }

    /// A banner with a tone-coloured accent for errors and warnings.
    pub fn notice(p: &Palette, tone: Tone, msg: &str) -> El {
        let c = tone.color(*p);
        row()
            .w_full()
            .gap(10.0)
            .padding(12.0)
            .radius(radius::CONTROL)
            .fill(c.alpha(0.12))
            .border(1.0, c.alpha(0.35))
            .child(components::status_dot(*p, tone, false).mt(6.0))
            .child(text(msg.to_string()).font(ty::BODY).color(p.text).wrap().grow(1.0).min_w(0.0))
    }

    /// A single-line text field in an inset well. `access` returns the edit inside the app state.
    pub fn field(
        p: &Palette,
        edit: &TextEdit,
        placeholder: &str,
        access: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
    ) -> El {
        row()
            .key("field-well")
            .radius(radius::CONTROL)
            .fill(p.well)
            .border(1.0, if edit.focused { p.accent.alpha(0.8) } else { p.well_rim })
            .grow(1.0)
            .min_w(0.0)
            .h(42.0)
            .px(14.0)
            .center_y()
            .child(
                text_input(edit, ty::BODY)
                    .key("field")
                    .single_line()
                    .w_full()
                    .color(p.text)
                    .placeholder(placeholder.to_string(), p.text_faint)
                    .caret(p.accent, p.accent.alpha(0.35))
                    .on_edit(access),
            )
    }

    pub fn progress(p: &Palette, frac: f32) -> El {
        components::progress_bar(*p, Some(frac))
    }

    pub fn checkbox(p: &Palette, checked: bool, label: &str, on_toggle: impl Fn(&mut AppState) + 'static) -> El {
        let mark = if checked {
            el().size(20.0, 20.0).radius(radius::CONTROL * 0.4).fill(p.accent).child(
                rune::canvas(
                    Shape::new().stroke(vec![[5.0, 10.5], [8.5, 14.0], [15.0, 6.5]], 2.2, p.on_accent).view_box(20.0, 20.0),
                )
                .size(20.0, 20.0),
            )
        } else {
            el().size(20.0, 20.0).radius(radius::CONTROL * 0.4).fill(p.well).border(1.0, p.rim_top)
        };
        row()
            .key("checkbox")
            .role("checkbox")
            .label(label.to_string())
            .checked(checked)
            .focusable(true)
            .gap(12.0)
            .w_full()
            .items_start()
            .on_press(on_toggle)
            .child(mark.mt(1.0))
            .child(text(label.to_string()).font(ty::BODY).color(p.text).wrap().grow(1.0).min_w(0.0))
    }
}
