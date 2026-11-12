//! Small helpers for the Record screen: thin adapters over the shared theme and components
//! (`ui::theme`, `ui::components`) plus the few record-specific pieces (segmented dB meter, formatters).

use crate::state::AppState;
use crate::ui::components::{self, Button, ButtonKind};
use crate::ui::theme;
use rune::{Color, El, TextStyle, el, text};

pub use crate::ui::components::Tone;
pub use crate::ui::theme::Palette as Pal;

pub fn txt(s: impl Into<String>, size: f32, weight: f32, color: Color) -> El {
    text(s).font(TextStyle::new(size, weight)).color(color)
}

pub fn txt_lh(s: impl Into<String>, size: f32, weight: f32, line: f32, color: Color) -> El {
    text(s).font(TextStyle::new(size, weight).lh(line)).color(color)
}

/// One of the few big blurred glass surfaces of a screen (the shell allows 4 backdrop blurs per frame).
pub fn panel(p: Pal) -> El {
    components::glass_panel(p)
}

/// A lighter glass card without backdrop blur.
pub fn glass(p: Pal) -> El {
    components::glass_card(p)
}

/// A small tinted label.
pub fn chip(p: Pal, label: impl Into<String>, tone: Tone) -> El {
    components::badge(p, &label.into(), tone)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BtnKind {
    Primary,
    Secondary,
    Ghost,
    Danger,
}

/// A pill button from the shared kit.
pub fn button(p: Pal, key: &str, label: &str, kind: BtnKind, enabled: bool, on_press: fn(&mut AppState)) -> El {
    let b = Button::new(label).key(key).disabled(!enabled).on_press(on_press);
    match kind {
        BtnKind::Primary => b.primary(),
        BtnKind::Secondary => b.secondary(),
        BtnKind::Ghost => b.ghost(),
        BtnKind::Danger => b.danger(),
    }
    .build(p)
}

#[allow(dead_code)]
const _: ButtonKind = ButtonKind::Primary;

/// Horizontal progress bar (0..1) in a chosen colour.
pub fn progress(p: Pal, frac: f32, color: Color, h: f32) -> El {
    let f = frac.clamp(0.0, 1.0);
    let track = el().w_full().h(h).radius(h / 2.0).fill(p.well).border(1.0, p.well_rim).clip_rounded(h / 2.0);
    if f <= 0.003 {
        return track;
    }
    track.child(el().h(h).w_frac(f).radius(h / 2.0).fill(color).motion(theme::motion::SOFT))
}

/// Maps a linear 0..1 peak to a 0..1 meter position on a -60..0 dB scale.
pub fn level_to_meter(level: f32) -> f32 {
    if level <= 0.0 {
        return 0.0;
    }
    ((20.0 * level.log10() + 60.0) / 60.0).clamp(0.0, 1.0)
}

/// Live level meter on a dB scale with a peak-hold marker (shared kit meter).
pub fn meter(p: Pal, level: f32, peak: f32) -> El {
    components::level_meter(p, level_to_meter(level), if peak > 0.0 { Some(level_to_meter(peak)) } else { None })
}

pub fn fmt_dur(secs: f32) -> String {
    let s = secs.max(0.0).round() as u32;
    format!("{}:{:02}", s / 60, s % 60)
}

pub fn fmt_min(minutes: f32) -> String {
    if minutes < 1.0 {
        format!("{} s", (minutes * 60.0).round() as u32)
    } else {
        format!("{minutes:.1} min")
    }
}

