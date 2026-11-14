//! Design tokens for the "liquid glass" look: the dark smoked-glass palette, type scale, radii and motion.
//!
//! Every screen reads colours from a [`Palette`] and never hard-codes a colour:
//!
//! ```ignore
//! let p = theme::palette(s, cx);          // in a screen's view(s, cx)
//! text("Hello").font(ty::TITLE).color(p.text)
//! ```

use crate::state::AppState;
use rune::{Color, Cx, Spring, TextStyle};

/// The palette (the app has one appearance: dark smoked glass on the Ice navy). Also sets the default text
/// colour on `cx`, so plain text and text fields are readable without an explicit colour.
pub fn palette(_s: &AppState, cx: &mut Cx<AppState>) -> Palette {
    let p = Palette::new();
    cx.foreground(p.text);
    p
}

/// All semantic colours for one appearance.
#[derive(Clone, Copy, Debug)]
pub struct Palette {
    /// Window background gradient (top to bottom).
    pub bg_top: Color,
    pub bg_bottom: Color,
    /// Soft colour blobs drifting behind the glass (violet, blue, pink, teal).
    pub aurora: [Color; 4],
    /// Translucent panel fill and its brighter variants.
    pub glass: Color,
    pub glass_top: Color,
    pub glass_bottom: Color,
    /// Hover / pressed tint over glass controls.
    pub glass_hover: Color,
    pub glass_press: Color,
    /// Rim light: border gradient from the lit top edge to the shaded bottom edge.
    pub rim_top: Color,
    pub rim_bottom: Color,
    /// Specular highlight line along a panel's top edge.
    pub sheen: Color,
    /// Inset input wells (text area, slider tracks).
    pub well: Color,
    pub well_rim: Color,
    pub text: Color,
    pub text_dim: Color,
    pub text_faint: Color,
    /// Text drawn on an accent-coloured surface.
    pub on_accent: Color,
    pub accent: Color,
    pub accent_top: Color,
    #[allow(dead_code)]
    pub accent_bottom: Color,
    pub accent_glow: Color,
    pub ok: Color,
    pub warn: Color,
    pub bad: Color,
    pub shadow: Color,
    pub shadow_soft: Color,
    /// Inner specular highlight along a glass edge (inset shadow colour).
    pub highlight: Color,
    /// Base tint of accent-coloured glass (primary button, active tab lens).
    pub lens: Color,
}

impl Palette {
    /// The app's "Ice" colours: navy `#0B1119`, accent
    /// `#9CD2FF`, text `#F2F6FB` / `#AFBBCB` / `#8A97A9`, state colours. Surfaces are dark smoked glass:
    /// navy-tinted fills over the blur, white only as a thin specular rim and a faint sheen.
    pub fn new() -> Palette {
        let dec = |v: u8| rune::color::srgb_decode(f32::from(v) / 255.0);
        let rgb = |r: u8, g: u8, b: u8, a: f32| Color { r: dec(r), g: dec(g), b: dec(b), a };
        let hex = |h: u32| rgb((h >> 16) as u8, (h >> 8) as u8, h as u8, 1.0);
        let accent = hex(0x9CD2FF);
        let line = |a: f32| rgb(205, 225, 255, a);
        Palette {
            bg_top: hex(0x0B131F),
            bg_bottom: hex(0x070C13),
            // Cool light fields drifting behind the glass: ice blue, teal, steel, deep blue.
            aurora: [hex(0x9CD2FF), hex(0x3FC0CF), hex(0x6283B8), hex(0x2F66C2)],
            // Smoked glass: navy fill, a cool lift at the top, deeper at the bottom.
            glass: rgb(11, 17, 25, 0.58),
            glass_top: rgb(70, 100, 140, 0.12),
            glass_bottom: rgb(3, 6, 12, 0.3),
            glass_hover: accent.alpha(0.12),
            glass_press: accent.alpha(0.2),
            rim_top: rgb(215, 232, 255, 0.4),
            rim_bottom: line(0.07),
            sheen: rgb(235, 245, 255, 0.45),
            well: rgb(14, 24, 42, 0.7),
            well_rim: line(0.1),
            text: hex(0xF2F6FB),
            text_dim: hex(0xAFBBCB),
            text_faint: hex(0x8A97A9),
            on_accent: hex(0x08111B),
            accent,
            accent_top: hex(0xAEDAFF),
            accent_bottom: hex(0x5FAAF0),
            accent_glow: hex(0x4FA3F5).alpha(0.4),
            ok: hex(0x6FD08C),
            warn: hex(0xFFD27A),
            bad: hex(0xFF7A85),
            shadow: Color::BLACK.alpha(0.55),
            shadow_soft: Color::BLACK.alpha(0.35),
            highlight: rgb(200, 224, 255, 0.2),
            lens: accent.alpha(0.3),
        }
    }
}

/// Type scale (Noto Sans, like the main app).
pub mod ty {
    use super::TextStyle;
    /// Big screen headings.
    pub const DISPLAY: TextStyle = TextStyle::new(22.0, 650.0).lh(28.0);
    /// Panel and section titles.
    pub const TITLE: TextStyle = TextStyle::new(14.0, 700.0).lh(19.0);
    /// Default reading text.
    pub const BODY: TextStyle = TextStyle::new(13.0, 400.0).lh(19.0);
    /// The speak text area.
    pub const EDITOR: TextStyle = TextStyle::new(16.0, 400.0).lh(25.0);
    /// Buttons and controls.
    pub const LABEL: TextStyle = TextStyle::new(12.5, 500.0).lh(17.0);
    /// Small secondary text.
    pub const CAPTION: TextStyle = TextStyle::new(11.0, 400.0).lh(15.0);
    /// Tiny section labels.
    pub const OVERLINE: TextStyle = TextStyle::new(10.5, 600.0).lh(14.0).tracking(0.4);
    /// Numbers that change (timers): tabular figures so they do not jitter.
    pub const NUM: TextStyle = TextStyle::new(12.0, 500.0).lh(16.0).features("'tnum' 1");
}

/// Corner radii: restrained glass (panels 16, cards 12, controls 10; buttons, tabs and chips are full pills).
pub mod radius {
    /// Big panels.
    pub const PANEL: f32 = 16.0;
    /// Cards inside panels.
    pub const CARD: f32 = 12.0;
    /// Controls (buttons, fields, tabs).
    pub const CONTROL: f32 = 10.0;
    /// Chips, badges, slider knob.
    #[allow(dead_code)]
    pub const SMALL: f32 = 6.0;
}

/// Springs: calm, no overshoot except `BOUNCY`.
pub mod motion {
    use super::Spring;
    // Hover and press feedback: quick and without overshoot. Every frame of it is a redraw at the display's rate, so
    // a long, bouncy spring (0.34 s, bounce 0.12 before) kept the GPU drawing for half a second per control crossed.
    pub const UI: Spring = Spring::new(0.16, 0.0);
    pub const SNAP: Spring = Spring::new(0.22, 0.0);
    pub const SOFT: Spring = Spring::new(0.45, 0.0);
    pub const BOUNCY: Spring = Spring::new(0.4, 0.2);
}
