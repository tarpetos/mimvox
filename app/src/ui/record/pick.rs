//! Preset picker and the "use my own text" analysis screen.

use super::widgets::*;
use super::{Analysis, Level, Load, Phase, Preset, RecLang, RecordState};
use crate::state::AppState;
use crate::ui::components::{TextArea, segmented};
use crate::ui::lang;
use crate::ui::theme::ty;
use rune::{El, ScrollSpec, col, el, row};

fn heading(p: Pal, title: &str, sub: &str) -> El {
    col()
        .gap(6.0)
        .child(txt(title, 30.0, 700.0, p.text))
        .child(txt(sub, 15.0, 400.0, p.text_dim).wrap())
}

fn page(p: Pal, content: El) -> El {
    let _ = p;
    col().w_full().h_full().min_h(0.0).items_center().justify_center().py(28.0).px(28.0).child(content)
}

fn page_scroll(content: El) -> El {
    col().w_full().h_full().min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).items_center().py(24.0).px(28.0).child(content)
}

/// The languages with reading presets (the engine's list, else the six it ships with).
fn rec_languages(r: &RecordState) -> Vec<RecLang> {
    if !r.languages.is_empty() {
        return r.languages.clone();
    }
    ["en", "uk", "de", "fr", "es", "pl"].iter().map(|c| RecLang { code: (*c).into(), name: lang::name(c), native: lang::label(c) }).collect()
}

/// The reading language as a row of native names.
fn language_strip(r: &RecordState, p: Pal) -> El {
    let langs = rec_languages(r);
    let labels: Vec<String> = langs.iter().map(|l| if l.native.is_empty() { l.name.clone() } else { l.native.clone() }).collect();
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let sel = langs.iter().position(|l| l.code == r.lang()).unwrap_or(0);
    let codes: Vec<String> = langs.iter().map(|l| l.code.clone()).collect();
    row().key("record-lang").w_full().gap(12.0).center_y().flex_wrap()
        .child(txt("Reading language", 13.0, 600.0, p.text_dim))
        .child(segmented(p, "record-lang-pick", &refs, sel, move |s: &mut AppState, i| {
            if let Some(c) = codes.get(i) {
                super::set_language(s, c);
            }
        }))
}

/// "Use my own text": detect the language, or one of the preset languages.
fn own_language_strip(r: &RecordState, p: Pal) -> El {
    let langs = rec_languages(r);
    let mut labels = vec!["Detect".to_owned()];
    labels.extend(langs.iter().map(|l| if l.native.is_empty() { l.name.clone() } else { l.native.clone() }));
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let sel = r.own_language.as_ref().and_then(|c| langs.iter().position(|l| &l.code == c)).map_or(0, |i| i + 1);
    let codes: Vec<String> = langs.iter().map(|l| l.code.clone()).collect();
    row().key("own-lang").w_full().gap(12.0).center_y().flex_wrap()
        .child(txt("Language", 13.0, 600.0, p.text_dim))
        .child(segmented(p, "own-lang-pick", &refs, sel, move |s: &mut AppState, i| {
            let pick = if i == 0 { None } else { codes.get(i - 1).cloned() };
            if s.record.own_language != pick {
                s.record.own_language = pick;
                s.record.analysis = None;
            }
        }))
}

pub fn presets(r: &RecordState, p: Pal) -> El {
    let mut cards = row().w_full().gap(16.0).items_stretch();
    match &r.presets {
        Load::Ready(list) => {
            for pr in list {
                cards = cards.child(preset_card(p, pr));
            }
        }
        Load::Failed(e) => {
            cards = cards.child(info_card(p, "Could not load the presets", e, Tone::Danger));
        }
        _ => {
            cards = cards.child(info_card(p, "Starting the engine", "The reading presets will appear in a moment.", Tone::Neutral));
        }
    }
    let head = row().w_full().gap(16.0).items_start().justify_between().child(
        col().grow(1.0).min_w(0.0).child(heading(
            p,
            "What would you like to read?",
            "Pick a preset. You read short parts aloud, we save each one and tell you when it is enough.",
        )),
    ).child(super::mic::chip(r, p));
    let mut c = col().w_full().max_w(900.0).gap(22.0).child(head).child(language_strip(r, p));
    if let Some(plan) = &r.plan {
        let done = r.saved_count();
        let label = format!("Continue: {} ({}/{} parts)", plan.title, done, plan.parts.len());
        c = c.child(
            glass(p).w_full().px(20.0).py(14.0).child(
                row().w_full().center_y().justify_between().child(txt(label, 15.0, 600.0, p.text)).child(button(
                    p,
                    "continue",
                    "Continue recording",
                    BtnKind::Primary,
                    true,
                    |s: &mut AppState| s.record.phase = Phase::Session,
                )),
            ),
        );
    }
    c = c.child(cards).child(
        glass(p).w_full().px(22.0).py(18.0).child(
            row()
                .w_full()
                .center_y()
                .justify_between()
                .child(
                    col()
                        .gap(4.0)
                        .child(txt("Use my own text", 17.0, 650.0, p.text))
                        .child(txt("Paste anything you like to read. We check it covers enough speech sounds.", 13.0, 400.0, p.text_dim)),
                )
                .child(button(p, "own", "Paste text", BtnKind::Secondary, true, |s: &mut AppState| s.record.phase = Phase::Own)),
        ),
    );
    page(p, c)
}

fn info_card(p: Pal, title: &str, body: &str, tone: Tone) -> El {
    glass(p)
        .grow(1.0)
        .p20()
        .gap(8.0)
        .child(txt(title, 17.0, 650.0, if tone == Tone::Neutral { p.text } else { tone.color(p) }))
        .child(txt(body, 13.0, 400.0, p.text_dim).wrap())
}

trait Pad {
    fn p20(self) -> Self;
}
impl Pad for El {
    fn p20(self) -> Self {
        self.padding(20.0)
    }
}

fn preset_card(p: Pal, pr: &Preset) -> El {
    let id = pr.id.clone();
    let mins = if pr.target_minutes < 1.5 { "about 1 min".to_string() } else { format!("about {} min", pr.target_minutes.round() as u32) };
    let (kind, tone) = if pr.trains_rvc { ("Trains your voice", Tone::Accent) } else { ("Ready at once", Tone::Success) };
    glass(p)
        .key(format!("preset-{}", pr.id))
        .role("button")
        .label(pr.title.clone())
        .grow(1.0)
        .w_frac(0.3)
        .padding(22.0)
        .gap(10.0)
        .hover(p.glass_hover)
        .press(p.glass_press)
        .focusable(true)
        .on_press(move |s: &mut AppState| super::choose_preset(s, &id))
        .child(txt(pr.title.clone(), 22.0, 700.0, p.text).a11y_hidden())
        .child(txt(pr.description.clone(), 13.5, 400.0, p.text_dim).wrap().a11y_hidden())
        .child(el().grow(1.0).min_h(8.0))
        .child(
            row()
                .gap(8.0)
                .flex_wrap()
                .child(chip(p, mins, Tone::Neutral))
                .child(chip(p, if pr.parts.len() == 1 { "1 part".to_string() } else { format!("{} parts", pr.parts.len()) }, Tone::Neutral))
                .child(chip(p, kind, tone)),
        )
}


pub fn own_text(r: &RecordState, p: Pal) -> El {
    let editor = TextArea::new("own-text", &r.own, |s: &mut AppState| &mut s.record.own)
        .placeholder("Paste or type the text you want to read aloud (a few minutes of text works best)")
        .font(ty::BODY)
        .min_height(150.0)
        .build(p);
    let editor = col().w_full().h(170.0).child(editor);
    let can = !r.own.text.trim().is_empty() && !r.analyzing;
    let actions = row()
        .w_full()
        .gap(10.0)
        .center_y()
        .child(button(p, "back", "Back", BtnKind::Ghost, true, |s: &mut AppState| s.record.phase = Phase::Pick))
        .child(el().grow(1.0))
        .child(
            button(p, "analyze", if r.analyzing { "Analyzing..." } else { "Analyze text" }, BtnKind::Primary, can, |s: &mut AppState| {
                s.record.analyze_own()
            }),
        );
    let mut c = col().w_full().max_w(900.0).gap(18.0).child(heading(
        p,
        "Your own text",
        "We count the speech sounds in it and tell you how good a voice it can make.",
    ));
    c = c.child(own_language_strip(r, p)).child(editor).child(actions);
    if let Some(e) = &r.analyze_error {
        c = c.child(txt(e.clone(), 13.0, 500.0, p.bad));
    }
    if let Some(a) = &r.analysis {
        c = c.child(analysis_card(p, a));
    }
    page_scroll(c)
}

pub fn level_tone(l: Level) -> Tone {
    match l {
        Level::TooShort => Tone::Danger,
        Level::ZeroShot => Tone::Warning,
        Level::Trainable => Tone::Accent,
        Level::Great => Tone::Success,
    }
}

fn analysis_card(p: Pal, a: &Analysis) -> El {
    let tone = level_tone(a.level);
    let cov_tone = if a.coverage >= 0.9 { p.ok } else if a.coverage >= 0.7 { p.warn } else { p.bad };
    let stat = |label: &str, value: String| col().gap(2.0).child(txt(value, 24.0, 700.0, p.text)).child(txt(label.to_string(), 12.0, 500.0, p.text_faint));
    let mut missing = row().gap(6.0).flex_wrap();
    for m in &a.missing {
        missing = missing.child(chip(p, m.clone(), Tone::Warning));
    }
    let mut card = glass(p)
        .w_full()
        .padding(22.0)
        .gap(16.0)
        .child(
            row()
                .w_full()
                .center_y()
                .gap(28.0)
                .child(stat("words", a.words.to_string()))
                .child(stat("estimated reading", fmt_min(a.est_minutes)))
                .child(stat("parts", a.parts.len().to_string()))
                .child(stat(if a.detected { "language (detected)" } else { "language" }, lang::label(if a.language.is_empty() { "en" } else { &a.language })))
                .child(el().grow(1.0))
                .child(chip(p, a.level.label(), tone)),
        )
        .child(
            col()
                .gap(6.0)
                .child(
                    row()
                        .w_full()
                        .justify_between()
                        .child(txt("Speech sound coverage", 13.0, 600.0, p.text_dim))
                        .child(txt(format!("{}%", (a.coverage * 100.0).round() as u32), 13.0, 700.0, p.text)),
                )
                .child(progress(p, a.coverage, cov_tone, 10.0)),
        );
    if !a.missing.is_empty() {
        card = card.child(
            col().gap(6.0).child(txt("Missing sounds", 13.0, 600.0, p.text_dim)).child(missing),
        );
    }
    card = card.child(txt(a.message.clone(), 14.0, 450.0, p.text).wrap());
    let ok = a.level != Level::TooShort && !a.parts.is_empty();
    card.child(
        row().w_full().justify_end().child(button(
            p,
            "record-own",
            &format!("Record these {} parts", a.parts.len()),
            BtnKind::Primary,
            ok,
            super::record_own,
        )),
    )
}
