//! Reusable "liquid glass" components shared by every screen.
//!
//! All functions take a [`Palette`] (get it with `theme::palette(s, cx)`) and return a plain Rune
//! [`El`], so they compose with the usual `.w_full()`, `.grow(1.0)`, `.child(..)` builders.
//!
//! | Component | Entry point |
//! | --- | --- |
//! | Frosted panel (blur, hairline, top highlight) | [`glass_panel`] |
//! | Panel with a title tag on its border (the main app's section look) | [`section_panel`] |
//! | Lighter card inside a panel (no blur) | [`glass_card`] |
//! | Primary / secondary / ghost / danger button | [`Button`] |
//! | Square ghost icon button | [`icon_button`] |
//! | Dropdown chip (icon, label, chevron) | [`dropdown_chip`] |
//! | Slider with label and value | [`Slider`] |
//! | Multi-line text field | [`TextArea`] |
//! | Progress bar (determinate or shimmering) | [`progress_bar`] |
//! | Level meter (green / yellow / red) | [`level_meter`] |
//! | Badge / status pill | [`badge`], [`status_dot`] |
//! | Toast stack (state lives in `AppState.shell`) | [`toast_stack`], `shell::toast` |
//! | Empty state, section label, divider | [`empty_state`], [`overline`], [`divider`] |
//!
//! Handlers are `Fn(&mut AppState)` closures: the whole app shares one root state.

use super::theme::{Palette, motion, radius, ty};
use crate::state::AppState;
use rune::{
    Align, Color, Cycle, Drag, El, Icon, Key, LoopSpec, Phase, Shadow, TextEdit, TextStyle, Transition, col, edit_access,
    el, icon, row, text, text_input,
};
use std::f32::consts::PI;
use std::rc::Rc;

// -------------------------------------------------------------------------------------------------
// Icons
// -------------------------------------------------------------------------------------------------

/// 16-point stroke icons in one style. Draw with `components::icon_el(&icons::PLAY, 16.0)`
/// (or `rune::icon`) and colour with `.color(..)`.
#[allow(dead_code)]
pub mod icons {
    use rune::Icon;
    macro_rules! stroke_icon {
        ($name:ident, $id:literal, $body:literal) => {
            pub static $name: Icon = Icon {
                name: $id,
                view: 16.0,
                body: concat!(
                    r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">"#,
                    $body,
                    "</g>"
                ),
            };
        };
    }
    stroke_icon!(PLAY, "play", r#"<path d="M5.2 3.4L12.4 8L5.2 12.6Z"/>"#);
    stroke_icon!(PAUSE, "pause", r#"<path d="M5.6 3.8V12.2M10.4 3.8V12.2"/>"#);
    stroke_icon!(STOP, "stop", r#"<rect x="4" y="4" width="8" height="8" rx="1.6"/>"#);
    stroke_icon!(CHIP, "chip", r#"<rect x="4" y="4" width="8" height="8" rx="1.6"/><path d="M6.6 6.6H9.4V9.4H6.6ZM6 1.8V4M10 1.8V4M6 12V14.2M10 12V14.2M1.8 6H4M1.8 10H4M12 6H14.2M12 10H14.2"/>"#);
    stroke_icon!(FOLDER, "folder", r#"<path d="M2 4.6A1.6 1.6 0 0 1 3.6 3H6.2L7.7 4.8H12.4A1.6 1.6 0 0 1 14 6.4V11.4A1.6 1.6 0 0 1 12.4 13H3.6A1.6 1.6 0 0 1 2 11.4Z"/>"#);
    stroke_icon!(OPEN, "open-external", r#"<path d="M9.2 2.8H13.2V6.8M13.2 2.8L7.6 8.4M11 9.6V11.6A1.4 1.4 0 0 1 9.6 13H4.4A1.4 1.4 0 0 1 3 11.6V6.4A1.4 1.4 0 0 1 4.4 5H6.4"/>"#);
    stroke_icon!(FILE, "file", r#"<path d="M4.2 2H9L12.2 5.2V13.2A.8.8 0 0 1 11.4 14H4.6A.8.8 0 0 1 3.8 13.2V2.8A.8.8 0 0 1 4.2 2Z"/><path d="M9 2V5.4H12.2"/>"#);
    stroke_icon!(MIC, "mic", r#"<rect x="6" y="1.8" width="4" height="7.4" rx="2"/><path d="M3.7 7.6A4.3 4.3 0 0 0 12.3 7.6M8 11.9V14.2"/>"#);
    stroke_icon!(WAVE, "waveform", r#"<path d="M2.2 8V8M4.7 5.4V10.6M7.2 2.8V13.2M9.8 5.8V10.2M12.3 4V12M14 7.4V8.6"/>"#);
    stroke_icon!(CHECK, "check", r#"<path d="M3.4 8.4L6.6 11.6L12.6 4.6"/>"#);
    stroke_icon!(CLOSE, "close", r#"<path d="M4 4L12 12M12 4L4 12"/>"#);
    stroke_icon!(REFRESH, "refresh", r#"<path d="M13.2 8A5.2 5.2 0 1 1 11.6 4.2M13.2 2.4V5.6H10"/>"#);
    stroke_icon!(CHEVRON_DOWN, "chevron-down", r#"<path d="M4.2 6.3L8 10L11.8 6.3"/>"#);
    stroke_icon!(ARROW_RIGHT, "arrow-right", r#"<path d="M3 8H13M9 4L13 8L9 12"/>"#);
    stroke_icon!(PLUS, "plus", r#"<path d="M8 3V13M3 8H13"/>"#);
    stroke_icon!(USER, "user", r#"<circle cx="8" cy="5.6" r="2.6"/><path d="M3 13.4C3.4 10.9 5.4 9.6 8 9.6S12.6 10.9 13 13.4"/>"#);
    stroke_icon!(ALERT, "alert", r#"<path d="M8 2.4L14 12.8H2Z"/><path d="M8 6.6V9.4M8 11.2V11.3"/>"#);
    stroke_icon!(SPARKLE, "sparkle", r#"<path d="M7.2 2.6L8.5 6L11.9 7.3L8.5 8.6L7.2 12L5.9 8.6L2.5 7.3L5.9 6Z"/><path d="M12.4 1.8V4.2M11.2 3H13.6"/>"#);
    stroke_icon!(TRASH, "trash", r#"<path d="M3 4.4H13M6.4 4.4V3A.8.8 0 0 1 7.2 2.2H8.8A.8.8 0 0 1 9.6 3V4.4M4.2 4.4L4.8 12.8A1 1 0 0 0 5.8 13.8H10.2A1 1 0 0 0 11.2 12.8L11.8 4.4"/>"#);
    stroke_icon!(VOLUME, "volume", r#"<path d="M2.6 6.2H5L8.2 3.4V12.6L5 9.8H2.6Z" stroke-width="1.3"/><path d="M10.4 5.8A3.2 3.2 0 0 1 10.4 10.2M12.2 4A5.6 5.6 0 0 1 12.2 12"/>"#);
}

/// A themed icon element (`rune::icon` with the colour applied).
pub fn icon_el(i: &'static Icon, size: f32, color: Color) -> El {
    icon(i, size).color(color)
}

// -------------------------------------------------------------------------------------------------
// Tone
// -------------------------------------------------------------------------------------------------

/// Semantic colour of a badge, toast or status dot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger,
}

impl Tone {
    pub fn color(self, p: Palette) -> Color {
        match self {
            Tone::Neutral => p.text_dim,
            Tone::Accent => p.accent,
            Tone::Success => p.ok,
            Tone::Warning => p.warn,
            Tone::Danger => p.bad,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Panels
// -------------------------------------------------------------------------------------------------

/// Adds the liquid-glass edge to a translucent element: a bright 1 pt specular line along the top inside edge
/// and a soft inner glow that brightens the rim, as if the glass were thick and caught the light.
pub fn glass_edge(p: Palette, e: El) -> El {
    e.box_shadow(Shadow { x: 0.0, y: 1.0, blur: 1.5, spread: 0.0, color: p.highlight, inset: true })
        .box_shadow(Shadow { x: 1.0, y: 2.0, blur: 18.0, spread: 0.0, color: p.highlight.alpha(p.highlight.a * 0.3), inset: true })
        .box_shadow(Shadow { x: -1.0, y: -1.0, blur: 10.0, spread: 0.0, color: p.highlight.alpha(p.highlight.a * 0.14), inset: true })
}

/// A panel that opens over the page next to the element keyed `anchor` (below it, or above when there is no room),
/// `width` points wide; a press outside it runs `close`. Add it at the root, above the screens. The page under it
/// does not move: sections that unfolded in place pushed everything below them down and jumped under the pointer.
/// A panel taller than the window scrolls.
pub fn popover(p: Palette, key: &str, anchor: &str, width: f32, align: Align, close: impl Fn(&mut AppState) + 'static, content: El) -> El {
    // The page dims a little, so the panel reads as the thing in front; the panel is opaque (text under a
    // translucent one showed through).
    let scrim = el().key(format!("{key}-scrim")).pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).fill(Color::BLACK.alpha(0.22)).on_press(close);
    let panel = col().key(format!("{key}-panel")).w(width).min_h(0.0).padding(14.0).radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 1.0))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .anchored_fit(anchor.to_owned(), rune::Side::Bottom, align, 6.0)
        .transition(Transition::scale(0.98))
        .child(col().key(format!("{key}-body")).w_full().shrink().min_h(0.0).scroll(rune::ScrollSpec { stick_bottom: false }).scroll_edge_fade(12.0).child(content));
    el().key(format!("{key}-layer")).pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).child(scrim).child(glass_edge(p, panel))
}

/// Accent-tinted luminous glass (the primary button, the active tab lens, the play button): a translucent
/// accent fill with a bright top, a white-to-accent rim and a soft accent glow. Apply to a sized element.
pub fn accent_glass(p: Palette, e: El) -> El {
    // Saturated and translucent rather than milky: a bright ice-blue top deepening to a richer blue, a thin rim.
    let top = rune::oklcha(0.8, 0.1, 245.0, 0.9);
    let bottom = rune::oklcha(0.66, 0.14, 250.0, 0.62);
    let e = e
        .fill(p.lens.alpha(0.12))
        .gradient(top, bottom)
        .border_gradient(1.0, Color::WHITE.alpha(0.55), p.accent.alpha(0.3))
        .shadow(6.0, 26.0, p.accent_glow.alpha(0.4))
        .shadow(1.0, 3.0, p.shadow_soft);
    glass_edge(p, e)
}

/// A liquid-glass panel: a translucent gradient fill over the light fields behind it (they are soft already: a
/// backdrop blur changed almost nothing there, cost a blur pyramid per panel on every frame, and cannot be drawn
/// into a cached layer, see `shell::view`), a bright specular rim (lit top edge fading down), an inner edge glow, a top sheen and layered
/// soft shadows. Returns an empty column: add `.padding(..)`, size and children.
///
/// A frame blurs at most `rune::MAX_BACKDROP_BLURS` (4) elements, so keep real panels to the
/// big surfaces of a screen (the header takes one) and use [`glass_card`] for everything inside them.
pub fn glass_panel(p: Palette) -> El {
    let r = radius::PANEL;
    let e = col()
        .radius(r)
        .continuous_corners()
        .fill(p.glass)
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(20.0, 56.0, p.shadow)
        .shadow(3.0, 10.0, p.shadow_soft)
        .cache();
    glass_edge(p, e).child(sheen(p, r))
}

/// A glass panel with a simple inline title at its top-left. Add the body with `.child(..)`; padding is set.
/// Anything pinned into the title row (a count, a hint) should use `.pin(None, Some(22.0), Some(22.0), None)`.
pub fn section_panel(p: Palette, title: &str) -> El {
    glass_panel(p).padding(22.0).gap(14.0).child(text(title.to_owned()).font(ty::TITLE).color(p.text))
}

/// A lighter card for use inside a [`glass_panel`]: cheap frost (no backdrop blur), rim and edge glow.
pub fn glass_card(p: Palette) -> El {
    let e = col()
        .radius(radius::CARD)
        .continuous_corners()
        .fill(p.glass)
        .gradient(p.glass_top.alpha(p.glass_top.a * 0.7), p.glass_bottom)
        .border_gradient(1.0, p.rim_top.alpha(p.rim_top.a * 0.7), p.rim_bottom)
        .shadow(6.0, 18.0, p.shadow_soft);
    glass_edge(p, e)
}

/// A recessed well: the background of text fields. Darker than the glass around it, with an inner top shadow.
pub fn well(p: Palette) -> El {
    col()
        .radius(radius::CONTROL + 2.0)
        .continuous_corners()
        .fill(p.well)
        .border_gradient(1.0, Color::BLACK.alpha(0.35), p.well_rim.alpha(p.well_rim.a * 1.2))
        .box_shadow(Shadow { x: 0.0, y: 2.0, blur: 8.0, spread: 0.0, color: Color::BLACK.alpha(0.4), inset: true })
}

/// Top sheen of a glass surface: a faint light wash fading down, and a bright hairline that fades out sideways.
fn sheen(p: Palette, r: f32) -> El {
    let wash = el()
        .key("sheen-wash")
        .pin(Some(0.0), Some(0.0), Some(0.0), None)
        .h(r * 4.0)
        .radius(r)
        .linear_gradient(PI, [(0.0, Color::WHITE.alpha(0.035)), (1.0, Color::WHITE.alpha(0.0))])
        .inert();
    let line = el()
        .key("sheen-line")
        .pin(Some(r), Some(0.5), Some(r), None)
        .h(1.0)
        .linear_gradient(PI / 2.0, [(0.0, p.sheen.alpha(0.0)), (0.3, p.sheen.alpha(p.sheen.a * 0.8)), (1.0, p.sheen.alpha(0.0))])
        .inert();
    el().key("sheen").pin(Some(0.0), Some(0.0), Some(0.0), None).h(r * 4.0).inert().child(wash).child(line)
}

/// A centred placeholder for an empty list: icon, title, optional subtitle.
pub fn empty_state(p: Palette, glyph: &'static Icon, title: &str, subtitle: &str) -> El {
    let badge = el()
        .size(48.0, 48.0)
        .radius(24.0)
        .center()
        .fill(p.glass)
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .child(icon_el(glyph, 20.0, p.text_dim));
    let badge = glass_edge(p, badge);
    let mut c = col().w_full().center().gap(10.0).py(18.0).child(badge)
        .child(text(title).font(ty::LABEL).color(p.text_dim).align(Align::Center));
    if !subtitle.is_empty() {
        c = c.child(text(subtitle).font(ty::CAPTION).color(p.text_faint).align(Align::Center).wrap().max_w(260.0));
    }
    c
}

// -------------------------------------------------------------------------------------------------
// Buttons
// -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonKind {
    /// Accent gradient with glow: the one main action of a screen.
    Primary,
    /// Glass: everything else.
    Secondary,
    /// No background until hovered.
    Ghost,
    /// Red gradient for destructive actions.
    Danger,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonSize {
    Small,
    Medium,
    Large,
}

type Handler = Rc<dyn Fn(&mut AppState)>;

/// A pill button.
///
/// ```ignore
/// Button::new("Generate").primary().large().icon(&icons::SPARKLE)
///     .on_press(|s| speak::generate(s)).build(p)
/// ```
pub struct Button {
    label: String,
    glyph: Option<&'static Icon>,
    kind: ButtonKind,
    size: ButtonSize,
    disabled: bool,
    key: Option<String>,
    help: Option<String>,
    grow: bool,
    on_press: Option<Handler>,
}

impl Button {
    pub fn new(label: impl Into<String>) -> Self {
        Button {
            label: label.into(),
            glyph: None,
            kind: ButtonKind::Secondary,
            size: ButtonSize::Medium,
            disabled: false,
            key: None,
            help: None,
            grow: false,
            on_press: None,
        }
    }
    pub fn primary(mut self) -> Self {
        self.kind = ButtonKind::Primary;
        self
    }
    pub fn secondary(mut self) -> Self {
        self.kind = ButtonKind::Secondary;
        self
    }
    pub fn ghost(mut self) -> Self {
        self.kind = ButtonKind::Ghost;
        self
    }
    pub fn danger(mut self) -> Self {
        self.kind = ButtonKind::Danger;
        self
    }
    pub fn small(mut self) -> Self {
        self.size = ButtonSize::Small;
        self
    }
    pub fn large(mut self) -> Self {
        self.size = ButtonSize::Large;
        self
    }
    /// A leading icon.
    pub fn icon(mut self, glyph: &'static Icon) -> Self {
        self.glyph = Some(glyph);
        self
    }
    /// A disabled button is dimmed and ignores presses.
    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }
    /// Stable element key (defaults to the label); give one when two buttons share a label.
    pub fn key(mut self, key: impl Into<String>) -> Self {
        self.key = Some(key.into());
        self
    }
    /// A tooltip.
    #[allow(dead_code)]
    pub fn help(mut self, text: impl Into<String>) -> Self {
        self.help = Some(text.into());
        self
    }
    /// Fill the available width of a row (`grow(1.0)`).
    #[allow(dead_code)]
    pub fn grow(mut self) -> Self {
        self.grow = true;
        self
    }
    pub fn on_press(mut self, f: impl Fn(&mut AppState) + 'static) -> Self {
        self.on_press = Some(Rc::new(f));
        self
    }

    pub fn build(self, p: Palette) -> El {
        let (h, fs, px) = match self.size {
            ButtonSize::Small => (28.0, TextStyle::new(12.0, 500.0).lh(16.0), 11.0),
            ButtonSize::Medium => (34.0, ty::LABEL, 16.0),
            ButtonSize::Large => (46.0, TextStyle::new(13.5, 600.0).lh(18.0), 22.0),
        };
        let ink = match self.kind {
            ButtonKind::Primary => p.on_accent,
            ButtonKind::Danger => p.text,
            ButtonKind::Secondary => p.text,
            ButtonKind::Ghost => p.text_dim,
        };
        let key = self.key.clone().unwrap_or_else(|| self.label.clone());
        let mut b = row().key(key).role("button").label(self.label.clone()).h(h).px(px).gap(8.0).center().radius(h / 2.0)
            .motion(motion::UI);
        if self.grow {
            b = b.grow(1.0);
        }
        b = match self.kind {
            ButtonKind::Primary => accent_glass(p, b).hover(Color::WHITE.alpha(0.1)).press(Color::BLACK.alpha(0.1)),
            ButtonKind::Danger => glass_edge(
                p,
                b.fill(p.bad.alpha(0.3)).gradient(p.bad.alpha(0.5), p.bad.alpha(0.2))
                    .border_gradient(1.0, Color::WHITE.alpha(0.6), p.bad.alpha(0.35))
                    .shadow(6.0, 20.0, p.bad.alpha(0.3)),
            )
            .hover(Color::WHITE.alpha(0.1))
            .press(Color::BLACK.alpha(0.1)),
            ButtonKind::Secondary => glass_edge(
                p,
                b.fill(p.glass).gradient(p.glass_top, p.glass_bottom).border_gradient(1.0, p.rim_top, p.rim_bottom).shadow(4.0, 14.0, p.shadow_soft),
            )
            .hover(p.glass_hover)
            .press(p.glass_press),
            ButtonKind::Ghost => b.hover(p.glass_hover).press(p.glass_press),
        };
        if let Some(g) = self.glyph {
            b = b.child(icon_el(g, h * 0.46, ink));
        }
        if !self.label.is_empty() {
            b = b.child(text(self.label).font(fs).color(ink).a11y_hidden());
        }
        if let Some(t) = self.help {
            b = b.help(t);
        }
        match (self.disabled, self.on_press) {
            (false, Some(f)) => b.focusable(true).on_press(move |s: &mut AppState| f(s)),
            _ => b.disabled(self.disabled).opacity(if self.disabled { 0.42 } else { 1.0 }),
        }
    }
}

/// A round glass icon button (32 pt). `help` is the tooltip and accessible label.
pub fn icon_button(
    p: Palette,
    key: &str,
    glyph: &'static Icon,
    help: &str,
    on_press: impl Fn(&mut AppState) + 'static,
) -> El {
    let b = el().key(key.to_owned())
        .role("button")
        .label(help.to_owned())
        .size(32.0, 32.0)
        .radius(16.0)
        .center()
        .fill(p.glass)
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(3.0, 10.0, p.shadow_soft);
    glass_edge(p, b)
        .hover(p.glass_hover)
        .press(p.glass_press)
        .focusable(true)
        .motion(motion::UI)
        .help(help.to_owned())
        .on_press(on_press)
        .child(icon_el(glyph, 16.0, p.text_dim))
}

/// A compact glass chip that opens a dropdown (the language pickers, the result format): an accent icon, `label` and a
/// chevron, 30 pt tall. `key` anchors the dropdown; `a11y` is the accessible label. A disabled chip is dimmed.
pub fn dropdown_chip(p: Palette, key: &str, glyph: &'static Icon, label: &str, a11y: String, enabled: bool, on_press: impl Fn(&mut AppState) + 'static) -> El {
    let b = row().key(key.to_owned()).role("button").label(a11y).h(30.0).pl(12.0).pr(9.0).gap(7.0).center_y().radius(15.0)
        .fill(p.glass)
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(3.0, 10.0, p.shadow_soft);
    let b = glass_edge(p, b).motion(motion::UI)
        .child(icon_el(glyph, 13.0, p.accent))
        .child(text(label.to_owned()).font(ty::LABEL).color(p.text).max_w(150.0).truncate().a11y_hidden())
        .child(icon_el(&icons::CHEVRON_DOWN, 11.0, p.text_dim));
    if enabled { b.hover(p.glass_hover).focusable(true).on_press(on_press) } else { b.opacity(0.5) }
}

// -------------------------------------------------------------------------------------------------
// Slider
// -------------------------------------------------------------------------------------------------

/// A labelled slider with a live value read-out and optional end captions.
///
/// ```ignore
/// Slider::new("expr", "Expressiveness", s.speak.expressiveness, 0.4, 1.0)
///     .step(0.05).ends("Steady", "Lively")
///     .on_change(|s, v| s.speak.expressiveness = v).build(p)
/// ```
pub struct Slider {
    key: String,
    label: String,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
    decimals: usize,
    unit: &'static str,
    ends: Option<(String, String)>,
    on_change: Option<Rc<dyn Fn(&mut AppState, f32)>>,
}

impl Slider {
    pub fn new(key: impl Into<String>, label: impl Into<String>, value: f32, min: f32, max: f32) -> Self {
        Slider {
            key: key.into(),
            label: label.into(),
            value: value.clamp(min, max),
            min,
            max,
            step: 0.0,
            decimals: 2,
            unit: "",
            ends: None,
            on_change: None,
        }
    }
    /// Snap to multiples of `step` (0 = continuous).
    pub fn step(mut self, step: f32) -> Self {
        self.step = step;
        self
    }
    /// Digits shown after the decimal point.
    #[allow(dead_code)]
    pub fn decimals(mut self, d: usize) -> Self {
        self.decimals = d;
        self
    }
    /// A unit appended to the live value read-out (for example `"×"`).
    pub fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }
    /// Captions under the left and right end of the track.
    pub fn ends(mut self, left: impl Into<String>, right: impl Into<String>) -> Self {
        self.ends = Some((left.into(), right.into()));
        self
    }
    pub fn on_change(mut self, f: impl Fn(&mut AppState, f32) + 'static) -> Self {
        self.on_change = Some(Rc::new(f));
        self
    }

    pub fn build(self, p: Palette) -> El {
        const KNOB: f32 = 22.0;
        const TRACK: f32 = 6.0;
        let frac = ((self.value - self.min) / (self.max - self.min).max(1e-6)).clamp(0.0, 1.0);
        let (min, max, step) = (self.min, self.max, self.step);
        let on_change = self.on_change.clone();
        let to_value = move |x: f32, w: f32| -> f32 {
            let t = ((x - KNOB / 2.0) / (w - KNOB).max(1.0)).clamp(0.0, 1.0);
            let mut v = min + t * (max - min);
            if step > 0.0 {
                v = (min + ((v - min) / step).round() * step).clamp(min, max);
            }
            v
        };
        let on_step = self.on_change.clone();
        let step_size = if step > 0.0 { step } else { (max - min) / 20.0 };
        let (cur, key) = (self.value, self.key.clone());

        let knob = el().key("knob")
            .size(KNOB, KNOB)
            .pin(None, Some((TRACK - KNOB) / 2.0), Some(-KNOB / 2.0), None)
            .radius(KNOB / 2.0)
            .fill(Color::WHITE.alpha(0.2))
            .gradient(Color::WHITE.alpha(0.6), Color::WHITE.alpha(0.12))
            .border_gradient(1.0, Color::WHITE.alpha(0.95), p.accent.alpha(0.5))
            .shadow(3.0, 10.0, Color::BLACK.alpha(0.5))
            .shadow(0.0, 14.0, p.accent_glow)
            .press_scale(1.1);
        let knob = glass_edge(p, knob);
        let fill = el()
            .key("fill")
            .h(TRACK)
            .w_frac(frac)
            .radius(TRACK / 2.0)
            .gradient(p.accent_top, p.accent)
            .shadow(0.0, 12.0, p.accent_glow)
            .motion(motion::SNAP)
            .child(knob);
        let track = el()
            .key("track")
            .w_full()
            .h(TRACK)
            .radius(TRACK / 2.0)
            .fill(p.well)
            .border_gradient(1.0, Color::BLACK.alpha(0.4), p.well_rim)
            .box_shadow(Shadow { x: 0.0, y: 1.0, blur: 3.0, spread: 0.0, color: Color::BLACK.alpha(0.4), inset: true })
            .child(fill);
        let hit = col()
            .key(format!("{key}-hit"))
            .role("slider")
            .label(self.label.clone())
            .range_value(f64::from(cur), f64::from(min), f64::from(max))
            .a11y_value(format!("{:.*}{}", self.decimals, cur, self.unit))
            .w_full()
            .h(26.0)
            .px(KNOB / 2.0)
            .justify_center()
            .focusable(true)
            .cursor(rune::Cursor::Pointer)
            .on_step(move |s: &mut AppState, steps, _| {
                if let Some(f) = &on_step {
                    f(s, (cur + steps as f32 * step_size).clamp(min, max));
                }
            })
            .on_drag(move |s: &mut AppState, d: &mut Drag, tx| {
                if matches!(d.phase, Phase::Move) {
                    tx.instant();
                }
                if !d.cancelled() {
                    if let Some(f) = &on_change {
                        f(s, to_value(d.x, d.w));
                    }
                }
            })
            .child(track);

        let head = row()
            .w_full()
            .center_y()
            .justify_between()
            .child(text(self.label).font(ty::LABEL).color(p.text))
            .child(text(format!("{:.*}{}", self.decimals, self.value, self.unit)).font(ty::NUM).color(p.text_dim));
        let mut c = col().key(self.key).w_full().gap(2.0).child(head).child(hit);
        if let Some((l, r)) = self.ends {
            c = c.child(
                row().w_full().justify_between().child(text(l).font(ty::CAPTION).color(p.text_faint))
                    .child(text(r).font(ty::CAPTION).color(p.text_faint)),
            );
        }
        c
    }
}


// -------------------------------------------------------------------------------------------------
// Text area
// -------------------------------------------------------------------------------------------------

/// A multi-line text field inside an inset well that scrolls and grows with its parent.
///
/// Return inserts a newline; Ctrl/Cmd+Enter runs `on_submit`.
///
/// ```ignore
/// TextArea::new("speak-text", &s.speak.text, |s| &mut s.speak.text)
///     .placeholder("Type anything to hear it in your voice...")
///     .on_submit(|s| speak::generate(s)).build(p)
/// ```
pub struct TextArea<'a> {
    key: String,
    edit: &'a TextEdit,
    access: fn(&mut AppState) -> &mut TextEdit,
    placeholder: String,
    font: TextStyle,
    min_h: f32,
    on_submit: Option<Handler>,
}

impl<'a> TextArea<'a> {
    pub fn new(key: impl Into<String>, edit: &'a TextEdit, access: fn(&mut AppState) -> &mut TextEdit) -> Self {
        TextArea { key: key.into(), edit, access, placeholder: String::new(), font: ty::EDITOR, min_h: 120.0, on_submit: None }
    }
    pub fn placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = text.into();
        self
    }
    pub fn font(mut self, font: TextStyle) -> Self {
        self.font = font;
        self
    }
    /// Minimum height of the editable area in points (default 120).
    pub fn min_height(mut self, h: f32) -> Self {
        self.min_h = h;
        self
    }
    /// Runs on Ctrl/Cmd+Enter.
    pub fn on_submit(mut self, f: impl Fn(&mut AppState) + 'static) -> Self {
        self.on_submit = Some(Rc::new(f));
        self
    }

    pub fn build(self, p: Palette) -> El {
        let access = self.access;
        let mut field = text_input(self.edit, self.font)
            .key(self.key.clone())
            .w_full()
            .min_h(self.min_h)
            .color(p.text)
            .placeholder(self.placeholder, p.text_faint)
            .caret(p.accent, p.accent.alpha(0.3))
            .on_edit(edit_access(move |s: &mut AppState| access(s)))
            .on_submit(move |s: &mut AppState, _| access(s).insert("\n"));
        if let Some(f) = self.on_submit {
            field = field.on_input_key(move |s: &mut AppState, ev, _| {
                if ev.key == Key::Enter && ev.mods.cmd {
                    f(s);
                    true
                } else {
                    false
                }
            });
        }
        let viewport = col()
            .key(format!("{}-scroll", self.key))
            .w_full()
            .grow(1.0)
            .min_h(0.0)
            .scroll(rune::ScrollSpec { stick_bottom: false })
            .scroll_edge_fade(14.0)
            .child(field);
        well(p).w_full().grow(1.0).min_h(0.0).padding(14.0).child(viewport)
    }
}

// -------------------------------------------------------------------------------------------------
// Progress and meters
// -------------------------------------------------------------------------------------------------

/// A thin progress bar. `Some(0.0..=1.0)` is a determinate fill that springs to its value;
/// `None` is an indeterminate shimmer for work of unknown length.
pub fn progress_bar(p: Palette, value: Option<f32>) -> El {
    let h = 8.0;
    let track = el()
        .w_full()
        .h(h)
        .radius(h / 2.0)
        .fill(p.well)
        .border_gradient(1.0, Color::BLACK.alpha(0.35), p.well_rim)
        .clip_rounded(h / 2.0);
    let bar = |e: El| e.radius(h / 2.0).gradient(p.accent_top, p.accent).shadow(0.0, 10.0, p.accent_glow);
    match value {
        Some(v) => track.child(bar(el().key("bar").h(h).w_frac(v.clamp(0.0, 1.0)).motion(motion::SOFT))),
        None => track.child(bar(el().key("shimmer").h(h).w_frac(0.32).looping(LoopSpec::slide(1.5, [-1.0, 0.0], [3.2, 0.0])))),
    }
}

/// An audio level meter: `level` 0..1 lights segments green, then yellow above 0.65 and red above
/// 0.88. `peak` (optional) draws a hold marker. Segments fade smoothly as the level moves.
/// A small segmented toggle (a recessed glass pill; the chosen option is lit with the accent). `on_pick` gets the
/// index of the pressed option.
pub fn segmented(p: Palette, key: &str, options: &[&str], selected: usize, on_pick: impl Fn(&mut AppState, usize) + Clone + 'static) -> El {
    segmented_help(p, key, options, &[], selected, on_pick)
}

/// [`segmented`] with a tooltip per option (`helps[i]`; missing or empty ones get none).
pub fn segmented_help(p: Palette, key: &str, options: &[&str], helps: &[&str], selected: usize, on_pick: impl Fn(&mut AppState, usize) + Clone + 'static) -> El {
    let mut strip = row().key(key.to_owned()).padding(3.0).gap(2.0).radius(17.0)
        .fill(p.well)
        .border_gradient(1.0, Color::BLACK.alpha(0.35), p.well_rim);
    for (i, name) in options.iter().enumerate() {
        let on = i == selected;
        let pick = on_pick.clone();
        let mut opt = row().key(format!("{key}-{i}")).role("button").label(*name).checked(on).focusable(true).h(28.0).px(14.0).center().radius(14.0)
            .fill(if on { p.accent.alpha(0.22) } else { Color::TRANSPARENT })
            .border(1.0, if on { p.accent.alpha(0.5) } else { Color::TRANSPARENT })
            .hover(if on { p.accent.alpha(0.28) } else { p.glass_hover })
            .motion(motion::UI)
            .on_press(move |s: &mut AppState| pick(s, i))
            .child(text(*name).font(ty::LABEL).color(if on { p.text } else { p.text_dim }).a11y_hidden());
        if let Some(h) = helps.get(i).filter(|h| !h.is_empty()) {
            opt = opt.help((*h).to_owned());
        }
        strip = strip.child(opt);
    }
    strip
}

/// A [`segmented_help`] strip that fills its width, the options sharing it equally: long labels in a narrow panel.
/// `enabled: false` dims it and ignores presses (the help texts still show, to say why).
#[allow(clippy::too_many_arguments)]
pub fn segmented_fill(
    p: Palette,
    key: &str,
    options: &[&str],
    helps: &[&str],
    selected: usize,
    enabled: bool,
    on_pick: impl Fn(&mut AppState, usize) + Clone + 'static,
) -> El {
    let mut strip = row().key(key.to_owned()).w_full().padding(3.0).gap(2.0).radius(17.0)
        .fill(p.well)
        .border_gradient(1.0, Color::BLACK.alpha(0.35), p.well_rim);
    for (i, name) in options.iter().enumerate() {
        let on = i == selected;
        let mut opt = row().key(format!("{key}-{i}")).role("button").label(*name).checked(on).focusable(enabled).grow(1.0).min_w(0.0).h(28.0).px(8.0)
            .center().radius(14.0)
            .fill(if on { p.accent.alpha(0.22) } else { Color::TRANSPARENT })
            .border(1.0, if on { p.accent.alpha(0.5) } else { Color::TRANSPARENT })
            .motion(motion::UI)
            .child(text(*name).font(ty::LABEL).color(if on { p.text } else { p.text_dim }).truncate().a11y_hidden());
        if enabled {
            let pick = on_pick.clone();
            opt = opt.hover(if on { p.accent.alpha(0.28) } else { p.glass_hover }).on_press(move |s: &mut AppState| pick(s, i));
        }
        if let Some(h) = helps.get(i).filter(|h| !h.is_empty()) {
            opt = opt.help((*h).to_owned());
        }
        strip = strip.child(opt);
    }
    if enabled { strip } else { strip.opacity(0.5) }
}

pub fn level_meter(p: Palette, level: f32, peak: Option<f32>) -> El {
    const N: usize = 28;
    let level = level.clamp(0.0, 1.0);
    let lit = (level * N as f32).round() as usize;
    let peak_i = peak.map(|v| ((v.clamp(0.0, 1.0) * N as f32).round() as usize).clamp(1, N) - 1);
    let seg = (0..N).map(|i| {
        let t = (i as f32 + 0.5) / N as f32;
        let tone = if t > 0.88 { p.bad } else if t > 0.65 { p.warn } else { p.ok };
        let on = i < lit || peak_i == Some(i);
        el().key(format!("seg{i}"))
            .grow(1.0)
            .h_full()
            .radius(2.0)
            .fill(if on { tone } else { p.well_rim })
            .opacity(if i < lit { 1.0 } else if on { 0.75 } else { 1.0 })
            .motion(motion::SNAP)
    });
    row().w_full().h(14.0).gap(3.0).children(seg)
}

// -------------------------------------------------------------------------------------------------
// Badges and pills
// -------------------------------------------------------------------------------------------------

/// A small tinted label ("Trained", "Zero-shot", "5 parts").
pub fn badge(p: Palette, label: &str, tone: Tone) -> El {
    let c = tone.color(p);
    row()
        .h(20.0)
        .px(8.0)
        .center()
        .radius(10.0)
        .fill(c.alpha(0.14))
        .border(1.0, c.alpha(0.35))
        .child(text(label).font(TextStyle::new(11.0, 600.0).lh(14.0)).color(if tone == Tone::Neutral { p.text_dim } else { c }))
}

/// A coloured dot, optionally pulsing (for "working" states).
pub fn status_dot(p: Palette, tone: Tone, pulsing: bool) -> El {
    const PULSE: Cycle = Cycle { period: 1.4, phase: 0.0, stops: &[(0.0, 1.0), (0.5, 0.35), (1.0, 1.0)], still: 1.0, fps: 12.0 };
    let c = tone.color(p);
    let d = el().size(8.0, 8.0).radius(4.0).fill(c);
    if pulsing { d.cycle(PULSE) } else { d }
}

// -------------------------------------------------------------------------------------------------
// Toasts
// -------------------------------------------------------------------------------------------------

/// One transient message. Push with `shell::toast(s, Tone::Success, "Saved")`.
#[derive(Clone, Debug)]
pub struct Toast {
    pub id: u64,
    pub tone: Tone,
    pub text: String,
    /// Shell clock time at which it disappears.
    pub expires: f64,
}

/// The toast column, pinned bottom-centre, `bottom` points above the window's bottom edge (above the player bar
/// while it shows). The shell draws it once; screens only push toasts.
pub fn toast_stack(p: Palette, toasts: &[Toast], bottom: f32) -> El {
    let items = toasts.iter().map(|t| {
        let id = t.id;
        // Long messages wrap onto more lines inside a bounded card instead of being cut off. The width follows the
        // text (a centred wrapping text would otherwise shrink to its longest word).
        let width = (t.text.chars().count() as f32 * 7.8 + 72.0).clamp(160.0, 640.0);
        row().key(format!("toast{id}"))
            .w(width)
            .px(16.0)
            .py(10.0)
            .min_h(38.0)
            .gap(10.0)
            .center_y()
            .radius(16.0)
            .fill(rune::oklcha(0.22, 0.03, 255.0, 0.97))
            .backdrop_blur(18.0)
            .border_gradient(1.0, p.rim_top, p.rim_bottom)
            .shadow(8.0, 28.0, p.shadow)
            .hover(p.glass_hover)
            .on_press(move |s: &mut AppState| s.shell.dismiss_toast(id))
            .transition(Transition::offset(0.0, 12.0).combined(Transition::opacity()))
            .child(status_dot(p, t.tone, false))
            .child(text(t.text.clone()).font(ty::LABEL).color(p.text).wrap().grow(1.0).min_w(0.0))
    });
    col().key("toasts").pin(Some(0.0), None, Some(0.0), Some(bottom)).items_center().gap(10.0).inert_if_empty(toasts.is_empty()).children(items)
}

trait InertIfEmpty {
    fn inert_if_empty(self, empty: bool) -> Self;
}
impl InertIfEmpty for El {
    /// An empty toast column must not swallow clicks meant for the screen below.
    fn inert_if_empty(self, empty: bool) -> Self {
        if empty { self.inert() } else { self }
    }
}
