//! Part-by-part recording view: big text, live meter, controls, parts sidebar and the sufficiency footer.

use super::pick::level_tone;
use super::widgets::*;
use super::{Level, Phase, RecordState, Take, TakeStatus, Verdict};
use crate::state::{AppState, Screen};
use crate::ui::theme;
use rune::{Color, El, ScrollSpec, col, el, row};

/// Level to act on: the engine's summary, else an estimate from the minutes saved.
fn level_of(r: &RecordState) -> Level {
    if let Some(s) = &r.summary {
        return s.level;
    }
    let m = r.total_minutes();
    if m < 0.5 {
        Level::TooShort
    } else if m < 5.0 {
        Level::ZeroShot
    } else if m < 15.0 {
        Level::Trainable
    } else {
        Level::Great
    }
}

enum Next {
    Train,
    Speak,
    More,
}

fn next_step(r: &RecordState) -> Next {
    let level = level_of(r);
    let trains = r.plan.as_ref().is_some_and(|p| p.trains_rvc);
    match (trains, level) {
        (_, Level::TooShort) => Next::More,
        (false, _) => Next::Speak,
        (true, Level::ZeroShot) => Next::More,
        (true, _) => Next::Train,
    }
}

pub fn view(r: &RecordState, p: Pal) -> El {
    let Some(plan) = &r.plan else { return el() };
    let top = row()
        .w_full()
        .center_y()
        .gap(12.0)
        .child(button(p, "back", "All presets", BtnKind::Ghost, !r.recording(), |s: &mut AppState| s.record.back_to_pick()))
        .child(txt(plan.title.clone(), 17.0, 650.0, p.text))
        .child(chip(p, format!("goal {}", fmt_min(plan.target_minutes)), Tone::Neutral))
        .child(el().grow(1.0))
        .child(super::mic::chip(r, p))
        .child(txt("Space  record / stop", 12.5, 500.0, p.text_faint));

    let main = if r.phase == Phase::Summary { summary_card(r, p) } else { recorder_card(r, p) };
    let mid = row().w_full().grow(1.0).min_h(0.0).gap(16.0).child(main).child(sidebar(r, p));

    col().w_full().h_full().min_h(0.0).padding(20.0).gap(14.0).child(top).child(mid).child(footer(r, p))
}

fn recorder_card(r: &RecordState, p: Pal) -> El {
    let (Some(part), Some(plan)) = (r.part(), &r.plan) else { return el() };
    let take = r.takes.get(r.current).cloned().unwrap_or_default();
    let active = r.active.as_ref();
    let n = plan.parts.len();

    let header = row()
        .w_full()
        .center_y()
        .gap(10.0)
        .child(txt(format!("Part {} of {}", r.current + 1, n), 14.0, 700.0, p.accent))
        .child(txt(part.title.clone(), 14.0, 500.0, p.text_dim))
        .child(el().grow(1.0))
        .child(chip(p, format!("about {}", fmt_dur(part.est_seconds)), Tone::Neutral))
        .child(button(p, "import", "Import file…", BtnKind::Secondary, !r.recording() && !r.importing, |s: &mut AppState| crate::ui::record::import_file(s)));

    let body = col()
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .scroll(ScrollSpec { stick_bottom: false })
        .scroll_edge_fade(18.0)
        .child(txt_lh(part.text.clone(), 23.0, 500.0, 35.0, p.text).wrap().w_full());

    // live meter
    let (level, clip, elapsed) = active.map_or((0.0, false, 0.0), |a| (a.level, a.clipped || a.peak >= 0.99, a.elapsed_s));
    let too_loud = clip || level_to_meter(level) > 0.92;
    let meter_row = row()
        .w_full()
        .center_y()
        .gap(14.0)
        .child(el().grow(1.0).child(meter(p, level, active.map_or(0.0, |a| a.peak))))
        .child(
            txt(
                if active.is_some() { format!("{} / ~{}", fmt_dur(elapsed), fmt_dur(part.est_seconds)) } else { "0:00".to_string() },
                15.0,
                650.0,
                if active.is_some() { p.text } else { p.text_faint },
            )
            .w(110.0),
        );
    let warn_line: El = if too_loud && active.is_some() {
        txt("Too loud. Move back from the microphone or lower the input volume.", 13.5, 650.0, p.bad)
    } else if active.is_some() {
        txt("Recording. Read the text naturally, press Space when you finish.", 13.5, 500.0, p.text_dim)
    } else {
        el().h(18.0)
    };

    let rec = r.recording();
    let big = record_button(p, rec, r.voice_id.is_some());
    let can_nav = !rec;
    let controls = row()
        .w_full()
        .center_y()
        .gap(14.0)
        .child(button(p, "prev", "Previous", BtnKind::Secondary, can_nav && r.current > 0, |s: &mut AppState| {
            let c = s.record.current;
            s.record.goto(c.wrapping_sub(1));
        }))
        .child(el().grow(1.0))
        .child(
            col().items_center().gap(6.0).child(big).child(txt(
                if rec { "Stop" } else if take.status == TakeStatus::Saved { "Record again" } else { "Record" },
                13.0,
                600.0,
                p.text_dim,
            )),
        )
        .child(el().grow(1.0))
        .child(button(p, "play", if r.playing { "Pause take" } else { "Play take" }, BtnKind::Secondary, can_nav && take.path.is_some(), |s: &mut AppState| {
            crate::ui::record::toggle_play(s)
        }))
        .child(button(p, "next", "Next", BtnKind::Secondary, can_nav && r.current + 1 < n, |s: &mut AppState| {
            let c = s.record.current;
            s.record.goto(c + 1);
        }));

    let over = r.import_over && !rec_now(r);
    let card = panel(p)
        .key("recorder-card")
        .grow(1.0)
        .min_w(0.0)
        .min_h(0.0)
        .padding(24.0)
        .gap(14.0)
        .child(header)
        .child(body)
        .child(take_banner(r, p, &take))
        .child(warn_line)
        .child(meter_row)
        .child(controls);
    if over {
        // A file is being dragged over the window: the card says where it will go.
        card.child(
            col().key("drop-hint").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).center().gap(8.0).radius(theme::radius::PANEL).inert()
                .fill(p.bg_top.alpha(0.9)).border(2.0, p.accent)
                .child(txt(format!("Release to use this file for part {}", r.current + 1), 18.0, 650.0, p.text))
                .child(txt("It is checked and added like a live recording.", 13.5, 450.0, p.text_dim)),
        )
    } else {
        card
    }
}

fn rec_now(r: &RecordState) -> bool {
    r.recording()
}

fn record_button(p: Pal, recording: bool, ready: bool) -> El {
    let (inner_w, inner_r) = if recording { (30.0, 8.0) } else { (54.0, 27.0) };
    let mut b = row()
        .key("rec-toggle")
        .role("button")
        .label(if recording { "Stop recording" } else { "Start recording" })
        .size(84.0, 84.0)
        .center()
        .radius(42.0)
        .border(3.0, if recording { p.bad } else { p.text_faint })
        .child(el().size(inner_w, inner_w).radius(inner_r).fill(p.bad).motion(theme::motion::SNAP));
    if ready || recording {
        b = b.hover(p.glass_hover).press(p.glass_press).focusable(true).on_press(|s: &mut AppState| crate::ui::record::toggle_recording(s));
    } else {
        b = b.opacity(0.45);
    }
    b
}

fn verdict_tone(v: Option<Verdict>) -> Tone {
    match v {
        Some(Verdict::Ok) => Tone::Success,
        Some(Verdict::Warn) => Tone::Warning,
        Some(Verdict::Bad) => Tone::Danger,
        None => Tone::Neutral,
    }
}

fn verdict_word(v: Option<Verdict>) -> &'static str {
    match v {
        Some(Verdict::Ok) => "Sounds good",
        Some(Verdict::Warn) => "Usable, with notes",
        Some(Verdict::Bad) => "Please record again",
        None => "",
    }
}

fn take_banner(r: &RecordState, p: Pal, t: &Take) -> El {
    if let Some((msg, tone)) = &r.notice {
        if r.recording() || t.status == TakeStatus::Empty || *tone == Tone::Danger {
            return txt(msg.clone(), 14.0, 600.0, tone.color(p)).wrap();
        }
    }
    let mut c = col().w_full().gap(6.0);
    match t.status {
        TakeStatus::Empty => {
            return if r.recording() {
                el().h(0.0)
            } else {
                txt("Press Space or the red button, then read the text above out loud.", 14.0, 500.0, p.text_dim)
            };
        }
        TakeStatus::Checking => {
            c = c.child(txt(
                match &t.imported {
                    Some(n) => format!("Checking {n}..."),
                    None => "Checking your recording...".to_string(),
                },
                14.0,
                600.0,
                p.text_dim,
            ))
        }
        TakeStatus::Saving => c = c.child(txt("Saving...", 14.0, 600.0, p.text_dim)),
        TakeStatus::Failed => c = c.child(txt(t.error.clone().unwrap_or_else(|| "Something went wrong".into()), 14.0, 600.0, p.bad).wrap()),
        TakeStatus::Saved | TakeStatus::Rejected => {
            let tone = verdict_tone(t.verdict);
            let head = row()
                .w_full()
                .center_y()
                .gap(10.0)
                .child(chip(p, if t.status == TakeStatus::Saved { "Saved" } else { "Not saved" }, tone))
                .child(txt(format!("{}  {}", fmt_dur(t.duration_s), verdict_word(t.verdict)), 14.0, 650.0, tone.color(p)))
                .child(el().grow(1.0));
            let head = if t.status == TakeStatus::Saved && t.verdict == Some(Verdict::Warn) && r.next_open(r.current).is_some() {
                head.child(button(p, "continue", "Continue", BtnKind::Primary, !r.recording(), |s: &mut AppState| {
                    if let Some(n) = s.record.next_open(s.record.current) {
                        s.record.goto(n);
                    }
                }))
            } else if t.status == TakeStatus::Saved && t.verdict == Some(Verdict::Warn) {
                head.child(button(p, "finish", "Finish", BtnKind::Primary, true, |s: &mut AppState| s.record.phase = Phase::Summary))
            } else if t.status == TakeStatus::Rejected {
                head.child(button(p, "retry", "Record again", BtnKind::Danger, !r.recording(), |s: &mut AppState| crate::ui::record::rerecord(s)))
            } else {
                head
            };
            c = c.child(head);
            if let Some(n) = &t.imported {
                c = c.child(txt(format!("Imported from {n}"), 13.0, 500.0, p.text_faint));
            }
            for i in &t.issues {
                c = c.child(txt(format!("- {i}"), 13.5, 450.0, p.text_dim).wrap());
            }
            if let Some((msg, _)) = &r.notice {
                c = c.child(txt(msg.clone(), 13.5, 600.0, p.ok));
            }
        }
    }
    c
}

fn sidebar(r: &RecordState, p: Pal) -> El {
    let Some(plan) = &r.plan else { return el() };
    let mut list = col().w_full().gap(6.0).scroll(ScrollSpec { stick_bottom: false }).grow(1.0).min_h(0.0);
    for (i, part) in plan.parts.iter().enumerate() {
        let t = r.takes.get(i).cloned().unwrap_or_default();
        let sel = i == r.current && r.phase == Phase::Session;
        let (dot, sub): (Color, String) = match t.status {
            TakeStatus::Saved => (verdict_tone(t.verdict).color(p), fmt_dur(t.duration_s)),
            TakeStatus::Checking | TakeStatus::Saving => (p.text_faint, "checking".into()),
            TakeStatus::Rejected => (p.bad, "re-record".into()),
            TakeStatus::Failed => (p.bad, "error".into()),
            TakeStatus::Empty => (Color::TRANSPARENT, format!("~{}", fmt_dur(part.est_seconds))),
        };
        let recording_here = r.active.as_ref().is_some_and(|a| a.part == i);
        let row_el = row()
            .key(format!("part-{}", i + 1))
            .role("button")
            .label(format!("Part {}", i + 1))
            .w_full()
            .h(46.0)
            .px(12.0)
            .gap(10.0)
            .center_y()
            .radius(theme::radius::CONTROL)
            .fill(if sel { p.glass_press } else { Color::TRANSPARENT })
            .border(1.0, if sel { p.accent } else { Color::TRANSPARENT })
            .hover(p.glass_hover)
            .focusable(true)
            .on_press(move |s: &mut AppState| s.record.goto(i))
            .child(if t.status == TakeStatus::Empty && !recording_here {
                el().size(12.0, 12.0).radius(6.0).border(1.5, p.text_faint)
            } else {
                el().size(12.0, 12.0).radius(6.0).fill(if recording_here { p.bad } else { dot })
            })
            .child(
                col()
                    .grow(1.0)
                    .min_w(0.0)
                    .child(txt(format!("Part {}", i + 1), 13.5, 650.0, p.text).a11y_hidden())
                    .child(txt(part.title.clone(), 11.5, 450.0, p.text_faint).truncate().a11y_hidden()),
            )
            .child(txt(sub, 12.5, 600.0, p.text_dim).a11y_hidden());
        list = list.child(row_el);
    }
    panel(p)
        .w(264.0)
        .min_h(0.0)
        .padding(14.0)
        .gap(10.0)
        .child(txt("Parts", 13.0, 700.0, p.text_dim).pl(6.0))
        .child(list)
}

fn footer(r: &RecordState, p: Pal) -> El {
    let Some(plan) = &r.plan else { return el() };
    let total = r.total_minutes();
    let frac = if plan.target_minutes > 0.0 { total / plan.target_minutes } else { 0.0 };
    let level = level_of(r);
    let (cta, msg): (El, String) = match next_step(r) {
        Next::Train => (
            button(p, "cta", "Ready to train", BtnKind::Primary, !r.recording(), |s: &mut AppState| s.screen = Screen::Train),
            r.summary.as_ref().map(|s| s.message.clone()).filter(|m| !m.is_empty()).unwrap_or_else(|| "You have enough speech to train your voice.".into()),
        ),
        Next::Speak => (
            button(p, "cta", "Ready to speak", BtnKind::Primary, !r.recording(), |s: &mut AppState| s.screen = Screen::Speak),
            "Your voice is ready. You can speak any text with it now.".into(),
        ),
        Next::More => {
            let need = (plan.target_minutes - total).max(0.0);
            (
                button(p, "cta", "Keep recording", BtnKind::Secondary, false, |_| {}),
                if total <= 0.0 { "Record your first part to get started.".to_string() } else { format!("About {} more to go.", fmt_min(need)) },
            )
        }
    };
    panel(p).w_full().px(20.0).py(14.0).child(
        row()
            .w_full()
            .center_y()
            .gap(18.0)
            .child(
                col()
                    .w(240.0)
                    .gap(6.0)
                    .child(
                        row()
                            .w_full()
                            .justify_between()
                            .child(txt(format!("{} / {}", fmt_min(total), fmt_min(plan.target_minutes)), 14.0, 700.0, p.text))
                            .child(txt(format!("{}/{} parts", r.saved_count(), plan.parts.len()), 12.5, 500.0, p.text_faint)),
                    )
                    .child(progress(p, frac, if frac >= 1.0 { p.ok } else { p.accent }, 8.0)),
            )
            .child(chip(p, level.label(), level_tone(level)))
            .child(txt(msg, 13.5, 450.0, p.text_dim).grow(1.0).min_w(0.0).wrap())
            .child(cta),
    )
}

fn summary_card(r: &RecordState, p: Pal) -> El {
    let level = level_of(r);
    let tone = level_tone(level);
    let (title, body) = match next_step(r) {
        Next::Train => ("All parts recorded", "Your recordings are enough to train a high quality copy of your voice."),
        Next::Speak => ("Your voice is ready", "Quick cloning needs no training. Type any text and hear it in your voice."),
        Next::More => ("A little more to record", "Record the parts that are still empty to reach the goal."),
    };
    let msg = r.summary.as_ref().map(|s| s.message.clone()).filter(|m| !m.is_empty()).unwrap_or_default();
    let mut c = panel(p).grow(1.0).min_w(0.0).min_h(0.0).padding(32.0).gap(14.0).center().child(chip(p, level.label(), tone))
        .child(txt(title, 30.0, 700.0, p.text))
        .child(txt(body, 16.0, 450.0, p.text_dim).wrap().max_w(520.0))
        .child(txt(format!("{} recorded in {} parts", fmt_min(r.total_minutes()), r.saved_count()), 15.0, 650.0, p.text));
    if !msg.is_empty() {
        c = c.child(txt(msg, 13.5, 450.0, p.text_faint).wrap().max_w(520.0));
    }
    let mut actions = row().gap(12.0).child(button(p, "review", "Review parts", BtnKind::Secondary, true, |s: &mut AppState| s.record.phase = Phase::Session));
    match next_step(r) {
        Next::Train => actions = actions.child(button(p, "go-train", "Ready to train", BtnKind::Primary, true, |s: &mut AppState| s.screen = Screen::Train)),
        Next::Speak => actions = actions.child(button(p, "go-speak", "Ready to speak", BtnKind::Primary, true, |s: &mut AppState| s.screen = Screen::Speak)),
        Next::More => {}
    }
    c.child(actions)
}
