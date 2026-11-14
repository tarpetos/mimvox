//! "Use recordings you already have": the drop card on the Record start page and the "whose voice is it" dialog.

use super::widgets::*;
use super::{FilesStage, Level, RecordState};
use crate::state::AppState;
use crate::ui::components::{icon_el, icons};
use crate::ui::shell;
use crate::ui::theme::{radius, ty};
use crate::ui::voices::kit;
use rune::{Color, El, col, el, row, text};

/// The card: what it is for, where to drop, and what happened to the last files.
pub fn card(r: &RecordState, p: Pal) -> El {
    let over = r.import_over;
    let busy = matches!(r.files.stage, FilesStage::Working { .. });
    let (title, sub) = if over {
        ("Release to use these files", "You choose next whose voice they are.")
    } else {
        (
            "Use recordings you already have",
            "Drop audio or video of one person here: a voice memo, a podcast, a video. No reading needed. One clean minute is enough to speak; three or more can be trained for a closer copy.",
        )
    };
    let well = el().key("files-icon").size(46.0, 46.0).radius(14.0).center().fill(p.accent.alpha(if over { 0.28 } else { 0.14 }))
        .child(icon_el(&icons::FOLDER, 22.0, p.accent));
    let head = row().w_full().gap(16.0).center_y()
        .child(well)
        .child(col().grow(1.0).min_w(0.0).gap(4.0)
            .child(txt(title, 17.0, 650.0, p.text))
            .child(txt(sub, 13.0, 400.0, p.text_dim).wrap()))
        .child(button(p, "files-choose", "Choose files…", BtnKind::Primary, !busy, super::choose_files));
    let mut c = glass(p).key("use-files").w_full().px(22.0).py(18.0).gap(14.0)
        .border(1.5, if over { p.accent } else { p.accent.alpha(0.35) })
        .fill(if over { p.accent.alpha(0.10) } else { Color::TRANSPARENT })
        .child(head);
    match &r.files.stage {
        FilesStage::Idle | FilesStage::Asking { .. } => {}
        FilesStage::Working { voice, note } => {
            c = c.child(row().w_full().gap(10.0).center_y()
                .child(chip(p, "Working", Tone::Accent))
                .child(txt(format!("{voice}: {note}"), 13.5, 500.0, p.text).wrap().grow(1.0)));
        }
        FilesStage::Done { voice, minutes, level, notes } => {
            let can_train = *level >= Level::Trainable;
            let what = if can_train {
                format!("“{voice}” is ready: {} of clean speech. Speak with it now, or train it for a closer copy.", fmt_min(*minutes))
            } else if *level == Level::TooShort {
                format!("“{voice}” has {} of clean speech: it works, more of it sounds closer to the voice.", fmt_min(*minutes))
            } else {
                format!("“{voice}” is ready: {} of clean speech. Three minutes or more can also be trained.", fmt_min(*minutes))
            };
            let mut done = col().w_full().gap(8.0).child(row().w_full().gap(10.0).center_y()
                .child(chip(p, "Ready", Tone::Success))
                .child(txt(what, 13.5, 500.0, p.text).wrap().grow(1.0)));
            for n in notes {
                done = done.child(txt(n.clone(), 12.5, 400.0, p.text_dim).wrap());
            }
            done = done.child(row().w_full().gap(10.0).center_y().flex_wrap()
                .child(button(p, "files-speak", "Speak with it", BtnKind::Primary, true, |s: &mut AppState| {
                    s.record.files_cancel();
                    shell::set_screen(s, crate::state::Screen::Speak);
                }))
                .child(button(p, "files-train", "Train a closer copy", BtnKind::Secondary, can_train, |s: &mut AppState| {
                    s.record.files_cancel();
                    shell::set_screen(s, crate::state::Screen::Train);
                }))
                .child(button(p, "files-ok", "Done", BtnKind::Ghost, true, |s: &mut AppState| s.record.files_cancel())));
            c = c.child(done);
        }
        FilesStage::Failed { message } => {
            c = c.child(row().w_full().gap(10.0).center_y()
                .child(chip(p, "Not added", Tone::Danger))
                .child(txt(message.clone(), 13.5, 500.0, p.text).wrap().grow(1.0))
                .child(button(p, "files-dismiss", "OK", BtnKind::Ghost, true, |s: &mut AppState| s.record.files_cancel())));
        }
    }
    c
}

/// One choice of the dialog: a ring, a title and a line under it.
fn choice(p: Pal, key: &str, on: bool, title: String, sub: &str, pick: fn(&mut AppState)) -> El {
    let ring = el().size(18.0, 18.0).radius(9.0).center().border(1.5, if on { p.accent } else { p.text_faint })
        .children(on.then(|| el().size(9.0, 9.0).radius(4.5).fill(p.accent)));
    row().key(key.to_owned()).role("radio").label(title.clone()).checked(on).w_full().gap(12.0).px(14.0).py(11.0).radius(radius::CONTROL)
        .fill(if on { p.accent.alpha(0.10) } else { Color::TRANSPARENT }).border(1.0, if on { p.accent.alpha(0.45) } else { p.well_rim })
        .hover(p.glass_hover).focusable(true).on_press(pick)
        .child(ring)
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(txt(title, 14.5, 600.0, p.text).a11y_hidden())
            .child(txt(sub, 12.5, 400.0, p.text_dim).wrap().a11y_hidden()))
}

/// The dialog over the window while files wait for their answer.
pub fn overlay(s: &AppState, p: &Pal) -> Option<El> {
    let FilesStage::Asking { paths } = &s.record.files.stage else { return None };
    let p = *p;
    let active = shell::active_voice(s);
    let into = s.record.files.into_current && active.is_some();
    let names: Vec<String> = paths.iter().take(3).filter_map(|f| f.file_name().map(|n| n.to_string_lossy().into_owned())).collect();
    let mut listed = names.join(", ");
    if paths.len() > names.len() {
        listed.push_str(&format!(" and {} more", paths.len() - names.len()));
    }
    let mut card = col().key("files-dialog").w(540.0).padding(24.0).gap(14.0).radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.97)).border_gradient(1.0, p.rim_top, p.rim_bottom).shadow(18.0, 50.0, p.shadow)
        .child(text("Whose voice is in these files?").font(ty::TITLE).color(p.text))
        .child(txt(listed, 12.5, 400.0, p.text_dim).wrap())
        .child(choice(p, "files-new", !into, "A new voice".into(), "The files become a voice of their own.", |s: &mut AppState| s.record.files.into_current = false));
    if !into {
        card = card.child(row().w_full().gap(10.0).center_y().pl(44.0)
            .child(kit::field(&p, &s.record.files.name, "Name of the voice", |s: &mut AppState| &mut s.record.files.name)
                .on_submit(|s: &mut AppState, _| super::confirm_files(s))));
    }
    if let Some(v) = active {
        card = card.child(choice(p, "files-current", into, format!("More recordings of “{}”", v.name),
            "Only when it is the same person: the files join this voice's recordings.", |s: &mut AppState| s.record.files.into_current = true));
    }
    let go = if into { "Add to this voice" } else { "Create the voice" };
    card = card.child(row().w_full().justify_end().gap(10.0)
        .child(button(p, "files-cancel", "Cancel", BtnKind::Ghost, true, |s: &mut AppState| s.record.files_cancel()))
        .child(button(p, "files-go", go, BtnKind::Primary, true, super::confirm_files)));
    Some(el().key("files-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).center().fill(Color::BLACK.alpha(0.55))
        .on_press(|_s: &mut AppState| {}).child(card))
}
