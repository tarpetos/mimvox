use super::components::{Button, Tone, badge, icons};
use super::setuprun::{self, COMPONENTS, Component, SPEECH_MODELS};
use super::shell::{self, EngineStatus};
use super::theme::{Palette, radius, ty};
use crate::engine::EngineMsg;
use crate::state::AppState;
use rune::{El, col, row, text};
use serde_json::json;

#[derive(Default)]
pub struct ModulesState {
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

pub fn reset(s: &mut AppState) {
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
            shell::toast(
                s,
                Tone::Success,
                if on { format!("{label} is on.") } else { format!("{label} is off; its files stay.") },
            );
            shell::restart_engine(s);
        }
        Err(e) => shell::toast(s, Tone::Danger, e.message.clone()),
    }
}

pub fn summary(s: &AppState) -> String {
    if setuprun::running(s) {
        return "Setup is running…".to_owned();
    }
    if s.shell.components.is_none() {
        return "Waiting for the engine".to_owned();
    }
    let count = |want: Status| COMPONENTS.iter().filter(|c| c.id != "speech" && status(s, c.id) == want).count();
    [(count(Status::On), "on"), (count(Status::Off), "off"), (count(Status::Missing), "not installed")]
        .iter()
        .filter(|(n, _)| *n > 0)
        .map(|(n, word)| format!("{n} {word}"))
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn section(s: &AppState, p: Palette) -> El {
    let mut body = col().key("modules-list").w_full().gap(12.0).px(6.0).pb(6.0).child(
        text("Optional parts of the speech engine. Turning one off keeps its files; removing it frees the disk. The speech engine and at least one voice-cloning model are always installed.")
            .font(ty::CAPTION)
            .color(p.text_dim)
            .wrap(),
    );
    if let Some(why) = busy_reason(s).filter(|_| !setuprun::running(s)) {
        body = body.child(super::voices::kit::notice(&p, Tone::Neutral, why));
    }
    if setuprun::running(s) || matches!(s.setup.phase, setuprun::Phase::Error(_) | setuprun::Phase::Finished(false)) {
        body = body.child(setuprun::panel(s, p));
    }
    for c in COMPONENTS.iter().filter(|c| c.id != "speech") {
        body = body.child(module_row(s, p, c));
    }
    body
}

fn module_row(s: &AppState, p: Palette, c: &'static Component) -> El {
    let id = c.id;
    let state = status(s, id);
    let blocked = busy_reason(s).is_some() || s.modules.toggle_req.is_some();
    let last_model = SPEECH_MODELS.contains(&id)
        && state == Status::On
        && SPEECH_MODELS.iter().filter(|m| status(s, m) == Status::On).count() == 1;
    let (word, tone) = match state {
        Status::On => ("On", Tone::Success),
        Status::Off => ("Off", Tone::Warning),
        Status::Missing => ("Not installed", Tone::Neutral),
    };
    let info = col()
        .grow(1.0)
        .min_w(0.0)
        .gap(2.0)
        .child(
            row()
                .gap(8.0)
                .center_y()
                .child(text(c.label).font(ty::LABEL).color(p.text))
                .child(badge(p, word, tone))
                .child(text(format!("{:.1} GB", c.gb)).font(ty::CAPTION).color(p.text_faint)),
        )
        .child(text(c.what).font(ty::CAPTION).color(p.text_dim).wrap())
        .children(last_model.then(|| {
            text("The only voice-cloning model: install another one to turn this one off or remove it.")
                .font(ty::CAPTION)
                .color(p.text_faint)
                .wrap()
        }));
    // Turn off / Remove and the delete confirmation swap at once: cross-fading, both sets overlapped and overflowed.
    let mut actions = row().key(format!("module-actions-{id}")).gap(8.0).center_y().instant_changes();
    if s.modules.confirm_remove == Some(id) {
        actions = actions
            .child(text(format!("Delete {:.1} GB?", c.gb)).font(ty::CAPTION).color(p.text_dim))
            .child(
                Button::new("Keep")
                    .key(format!("module-keep-{id}"))
                    .on_press(|s| s.modules.confirm_remove = None)
                    .build(p),
            )
            .child(
                Button::new("Remove")
                    .danger()
                    .icon(&icons::TRASH)
                    .key(format!("module-remove-yes-{id}"))
                    .on_press(move |s| remove(s, id))
                    .build(p),
            );
    } else {
        match state {
            Status::Missing => {
                actions = actions.child(
                    Button::new("Install")
                        .primary()
                        .icon(&super::formats::DOWNLOAD)
                        .key(format!("module-install-{id}"))
                        .disabled(blocked)
                        .on_press(move |s| setuprun::install(s, id))
                        .build(p),
                );
            }
            Status::On | Status::Off => {
                let on = state == Status::Off;
                actions = actions
                    .child(
                        Button::new(if on { "Turn on" } else { "Turn off" })
                            .icon(&icons::POWER)
                            .key(format!("module-toggle-{id}"))
                            .disabled(blocked || last_model)
                            .on_press(move |s| set_on(s, id, on))
                            .build(p),
                    )
                    .child(
                        Button::new("Remove")
                            .icon(&icons::TRASH)
                            .key(format!("module-remove-{id}"))
                            .disabled(blocked || last_model)
                            .on_press(move |s| s.modules.confirm_remove = Some(id))
                            .build(p),
                    );
            }
        }
    }
    row()
        .key(format!("module-{id}"))
        .w_full()
        .gap(14.0)
        .center_y()
        .padding(12.0)
        .radius(radius::CONTROL)
        .fill(p.well)
        .border(1.0, p.well_rim)
        .child(info)
        .child(actions)
}

pub fn demo(s: &mut AppState) {
    s.shell.components = Some(vec![
        "speech".into(),
        "omnivoice".into(),
        "training".into(),
        "translation".into(),
        "music_removal".into(),
        "native_accent".into(),
    ]);
    s.shell.components_off = vec!["ukrainian".into()];
}

pub fn demo_confirm_remove(s: &mut AppState, id: &'static str) {
    s.modules.confirm_remove = Some(id);
}
