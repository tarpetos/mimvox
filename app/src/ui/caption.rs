//! The caption buttons, drawn by the app. On Windows the system takes the pointer over them (non-client hits), so
//! hover and press are drawn from `ChromeInsets::caption`, not from `.hover()` / `.press()`.

use super::components::{glass_edge, icon_el};
use super::theme::{Palette, motion};
use rune::{CaptionButton, CaptionView, Color, El, Icon, Tx, col, el, row};

/// The 32 pt circle plus 12 pt to the next button (or the window's edge).
pub const CELL: f32 = 44.0;
const DIAMETER: f32 = 32.0;

macro_rules! glyph {
    ($name:ident, $id:literal, $body:literal) => {
        static $name: Icon = Icon {
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
glyph!(MINIMIZE, "window-minimize", r#"<path d="M4.4 8H11.6"/>"#);
glyph!(MAXIMIZE, "window-maximize", r#"<rect x="3.8" y="3.8" width="8.4" height="8.4" rx="1.9"/>"#);
glyph!(
    RESTORE,
    "window-restore",
    r#"<rect x="3.4" y="5.8" width="7" height="7" rx="1.7"/><path d="M5.8 3.6A1.4 1.4 0 0 1 7.2 2.8H10.8A2 2 0 0 1 12.8 4.8V8.4A1.4 1.4 0 0 1 12 9.7"/>"#
);
glyph!(CLOSE, "window-close", r#"<path d="M4.4 4.4L11.6 11.6M11.6 4.4L4.4 11.6"/>"#);

pub fn buttons(p: Palette, view: CaptionView, height: f32) -> El {
    let cell = |button: CaptionButton| {
        let (key, label, glyph) = match button {
            CaptionButton::Minimize => ("win-min", "Minimize", &MINIMIZE),
            CaptionButton::Maximize if view.maximized => ("win-max", "Restore", &RESTORE),
            CaptionButton::Maximize => ("win-max", "Maximize", &MAXIMIZE),
            CaptionButton::Close => ("win-close", "Close", &CLOSE),
        };
        let hovered = view.hovered == Some(button);
        let pressed = hovered && view.pressed;
        let close = button == CaptionButton::Close;
        let (fill, ink) = match (close, hovered, pressed) {
            (true, true, true) => (p.bad.alpha(0.55), Color::WHITE),
            (true, true, false) => (p.bad.alpha(0.82), Color::WHITE),
            (false, true, true) => (p.glass_press, p.text),
            (false, true, false) => (p.glass_hover, p.text),
            _ => (Color::TRANSPARENT, p.text_dim),
        };
        let disc = glass_edge(
            p,
            el().key(format!("{key}-disc")).size(DIAMETER, DIAMETER).radius(DIAMETER / 2.0).center()
                .fill(p.glass).gradient(p.glass_top, p.glass_bottom)
                .border_gradient(1.0, p.rim_top, p.rim_bottom)
                .shadow(3.0, 10.0, p.shadow_soft),
        )
        .child(el().key(format!("{key}-wash")).pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).radius(DIAMETER / 2.0).fill(fill).inert().motion(motion::UI))
        .child(icon_el(glyph, 16.0, ink).motion(motion::UI).a11y_hidden());
        // The system presses these (non-client hits); assistive technology invokes them.
        col().key(key).role("button").label(label).focusable(false).w(CELL).h(height).justify_center()
            .on_press_tx(move |_: &mut crate::state::AppState, tx: &mut Tx| match button {
                CaptionButton::Minimize => tx.minimize(),
                CaptionButton::Maximize => tx.toggle_maximized(),
                CaptionButton::Close => tx.close_window(),
            })
            .child(disc)
    };
    row().key("caption").pin(None, Some(0.0), Some(0.0), None).h(height)
        .child(cell(CaptionButton::Minimize))
        .child(cell(CaptionButton::Maximize))
        .child(cell(CaptionButton::Close))
}

pub fn scenes() -> &'static [(&'static str, fn(&mut crate::state::AppState))] {
    &[
        ("caption_idle", |s| preview(s, CaptionView::default())),
        ("caption_crowded", crowded),
        ("caption_maximized", |s| preview(s, CaptionView { maximized: true, ..CaptionView::default() })),
        ("caption_hover_min", |s| preview(s, CaptionView { hovered: Some(CaptionButton::Minimize), ..CaptionView::default() })),
        ("caption_hover_max", |s| preview(s, CaptionView { hovered: Some(CaptionButton::Maximize), ..CaptionView::default() })),
        ("caption_hover_close", |s| preview(s, CaptionView { hovered: Some(CaptionButton::Close), ..CaptionView::default() })),
        ("caption_press_close", |s| preview(s, CaptionView { hovered: Some(CaptionButton::Close), pressed: true, ..CaptionView::default() })),
    ]
}

/// The header at its fullest: models loading (the widest engine status) and a long voice name, under the caption buttons.
fn crowded(s: &mut crate::state::AppState) {
    super::speak::scene_idle(s);
    if let Some(v) = s.shell.voices.first_mut() {
        v.name = "Gigachad Sage (UA)".into();
    }
    s.shell.set_engine(super::shell::EngineStatus::Ready { device: Some("cuda".into()), tts: Some("loading".into()), rvc: Some("loading".into()) });
    preview(s, CaptionView::default());
}

fn preview(s: &mut crate::state::AppState, view: CaptionView) {
    s.screen = crate::state::Screen::Speak;
    s.shell.caption_preview = Some(view);
}
