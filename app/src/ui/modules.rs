use super::components::{Button, Tone, badge, icon_button, icons};
use super::setuprun::{self, COMPONENTS, Component};
use super::shell::{self, EngineStatus};
use super::theme::{Palette, radius, ty};
use crate::engine::EngineMsg;
use crate::state::AppState;
use rune::{Color, El, ScrollSpec, col, el, row, text};
use serde_json::json;

#[derive(Default)]
pub struct ModulesState {
    pub open: bool,
    toggle_req: Option<(u64, &'static str, bool)>,
    confirm_remove: Option<&'static str>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    On,
    Off,
    Missing,
}

pub fn status(s: &AppState, id: &str) -> Status {
    if s.shell.components_off.iter().any(|c| c == id) {
        Status::Off
    } else if s.shell.components.as_ref().is_some_and(|c| c.iter().any(|x| x == id)) {
        Status::On
    } else {
        Status::Missing
    }
}

pub fn open(s: &mut AppState) {
    s.modules.open = true;
    s.modules.confirm_remove = None;
}

fn close(s: &mut AppState) {
    s.modules.open = false;
    s.modules.confirm_remove = None;
}

/// Changing a part restarts the engine, which would end what it is doing.
fn busy_reason(s: &AppState) -> Option<&'static str> {
    if setuprun::running(s) {
        Some("Setup is running.")
    } else if s.book.running() {
        Some("A book is being read: pause it first.")
    } else if s.speak.is_generating() {
        Some("Speech is being made.")
    } else if super::train::is_training(s) {
        Some("A voice is training.")
    } else if s.redub.busy() {
        Some("Redub is working.")
    } else if !matches!(s.shell.engine, EngineStatus::Ready { .. }) {
        Some("Wait for the engine to start.")
    } else {
        None
    }
}

fn set_on(s: &mut AppState, id: &'static str, on: bool) {
    if busy_reason(s).is_some() || s.modules.toggle_req.is_some() {
        return;
    }
    if let Some(r) = shell::request(s, "components.set", json!({ "id": id, "on": on })) {
        s.modules.toggle_req = Some((r, id, on));
    }
}

fn remove(s: &mut AppState, id: &'static str) {
    s.modules.confirm_remove = None;
    if busy_reason(s).is_some() {
        return;
    }
    setuprun::remove(s, id);
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    let EngineMsg::Response { id, result } = msg else { return };
    let Some((_, name, on)) = s.modules.toggle_req.filter(|(r, _, _)| r == id) else { return };
    s.modules.toggle_req = None;
    let label = setuprun::component(name).map_or(name, |c| c.label);
    match result {
        Ok(_) => {
            shell::toast(s, Tone::Success, if on { format!("{label} is on.") } else { format!("{label} is off; its files stay.") });
            shell::restart_engine(s);
        }
        Err(e) => shell::toast(s, Tone::Danger, e.message.clone()),
    }
}

pub fn overlay(s: &AppState, p: &Palette) -> Option<El> {
    if !s.modules.open {
        return None;
    }
    let p = *p;
    let head = row().w_full().gap(12.0).center_y()
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(text("Modules").font(ty::TITLE).color(p.text))
            .child(text("Optional parts of the speech engine. Turning one off keeps its files; removing it frees the disk. Speech itself is always installed.")
                .font(ty::CAPTION).color(p.text_dim).wrap()))
        .child(icon_button(p, "modules-close", &icons::CLOSE, "Close", close));
    let mut card = col().key("modules-dialog").w(720.0).padding(22.0).gap(12.0).radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 1.0)).border_gradient(1.0, p.rim_top, p.rim_bottom).shadow(18.0, 50.0, p.shadow)
        .on_press(|_: &mut AppState| {})
        .child(head);
    let mut body = col().w_full().gap(12.0);
    if let Some(why) = busy_reason(s).filter(|_| !setuprun::running(s)) {
        body = body.child(super::voices::kit::notice(&p, Tone::Neutral, why));
    }
    let setup_shown = setuprun::running(s) || matches!(s.setup.phase, setuprun::Phase::Error(_) | setuprun::Phase::Finished(false));
    if setup_shown {
        body = body.child(setuprun::panel(s, p));
    }
    for c in COMPONENTS.iter().filter(|c| c.id != "speech") {
        body = body.child(module_row(s, p, c));
    }
    // A scroll area has no height of its own: the dialog takes a fixed one only when setup's steps make it long.
    if setup_shown {
        card = card.h(640.0).child(col().key("modules-list").w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false })
            .scroll_edge_fade(12.0).child(body));
    } else {
        card = card.child(body.key("modules-list"));
    }
    let scrim = el().key("modules-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(close);
    Some(el().key("modules-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).center().fill(Color::BLACK.alpha(0.55))
        .child(scrim).child(card))
}

fn module_row(s: &AppState, p: Palette, c: &'static Component) -> El {
    let id = c.id;
    let state = status(s, id);
    let blocked = busy_reason(s).is_some() || s.modules.toggle_req.is_some();
    let (word, tone) = match state {
        Status::On => ("On", Tone::Success),
        Status::Off => ("Off", Tone::Warning),
        Status::Missing => ("Not installed", Tone::Neutral),
    };
    let info = col().grow(1.0).min_w(0.0).gap(2.0)
        .child(row().gap(8.0).center_y()
            .child(text(c.label).font(ty::LABEL).color(p.text))
            .child(badge(p, word, tone))
            .child(text(format!("{:.1} GB", c.gb)).font(ty::CAPTION).color(p.text_faint)))
        .child(text(c.what).font(ty::CAPTION).color(p.text_dim).wrap());
    let mut actions = row().gap(8.0).center_y();
    if s.modules.confirm_remove == Some(id) {
        actions = actions
            .child(text(format!("Delete {:.1} GB?", c.gb)).font(ty::CAPTION).color(p.text_dim))
            .child(Button::new("Keep").ghost().small().key(format!("module-keep-{id}")).on_press(|s| s.modules.confirm_remove = None).build(p))
            .child(Button::new("Remove").danger().small().key(format!("module-remove-yes-{id}")).on_press(move |s| remove(s, id)).build(p));
    } else {
        match state {
            Status::Missing => {
                actions = actions.child(Button::new("Install").primary().small().icon(&super::formats::DOWNLOAD).key(format!("module-install-{id}"))
                    .disabled(blocked).on_press(move |s| setuprun::install(s, id)).build(p));
            }
            Status::On | Status::Off => {
                let on = state == Status::Off;
                actions = actions
                    .child(Button::new(if on { "Turn on" } else { "Turn off" }).ghost().small().key(format!("module-toggle-{id}"))
                        .disabled(blocked).on_press(move |s| set_on(s, id, on)).build(p))
                    .child(Button::new("Remove").ghost().small().key(format!("module-remove-{id}"))
                        .disabled(blocked).on_press(move |s| s.modules.confirm_remove = Some(id)).build(p));
            }
        }
    }
    row().key(format!("module-{id}")).w_full().gap(14.0).center_y().padding(12.0).radius(radius::CONTROL)
        .fill(p.well).border(1.0, p.well_rim)
        .child(info)
        .child(actions)
}

fn demo(s: &mut AppState) {
    super::speak::scene_idle(s);
    s.shell.components = Some(vec!["speech".into(), "training".into(), "translation".into(), "music_removal".into(), "native_accent".into()]);
    s.shell.components_off = vec!["ukrainian".into()];
    s.modules.open = true;
}

fn scene_open(s: &mut AppState) {
    demo(s);
}

fn scene_confirm(s: &mut AppState) {
    demo(s);
    s.modules.confirm_remove = Some("ukrainian");
}

fn scene_installing(s: &mut AppState) {
    demo(s);
    setuprun::scene_running(s);
    s.setup.adding = Some("ukrainian".into());
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[("modules", scene_open), ("modules_confirm_remove", scene_confirm), ("modules_installing", scene_installing)]
}
