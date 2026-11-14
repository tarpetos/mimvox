//! The window opens where and as large as it was closed (`settings.json` `window`), and the first time in the middle
//! of the main display at its default size, not maximized.
//!
//! Saved when the window closes: the outer position in desktop pixels, the content size in points, and whether it
//! was maximized (then the size from before stays, so un-maximizing returns to it; a maximized window is not moved
//! when it opens, because moving it would un-maximize it). A place that is no
//! longer on any display (a monitor unplugged since) is not used: the window opens centred. Wayland hides window
//! positions, so there only the size is kept.

use std::cell::RefCell;

use serde_json::{Value, json};

/// The default content size in points.
pub const DEFAULT_SIZE: (f32, f32) = (1040.0, 700.0);
/// A saved size below this is not trusted.
const MIN_SIZE: (f32, f32) = (640.0, 420.0);

thread_local! {
    /// The app's window, kept from `App::on_window` for [`save`].
    static WINDOW: RefCell<Option<rune::WindowHandle>> = const { RefCell::new(None) };
}

/// How the window opens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opening {
    /// Content size in points.
    pub size: (f32, f32),
    /// Outer position in desktop pixels; None: centre it on the main display.
    pub position: Option<(i32, i32)>,
    pub maximized: bool,
}

impl Default for Opening {
    fn default() -> Self {
        Opening { size: DEFAULT_SIZE, position: None, maximized: false }
    }
}

/// The saved placement, or the default.
pub fn opening() -> Opening {
    parse(super::onboarding::setting("window").as_ref(), on_a_display)
}

fn parse(saved: Option<&Value>, visible: impl Fn(i32, i32, f32, f32) -> bool) -> Opening {
    let Some(v) = saved else { return Opening::default() };
    let num = |k: &str| v.get(k).and_then(Value::as_f64);
    let (Some(w), Some(h)) = (num("w"), num("h")) else { return Opening::default() };
    let (w, h) = (w as f32, h as f32);
    if !(w.is_finite() && h.is_finite()) || w < MIN_SIZE.0 || h < MIN_SIZE.1 || w > 20000.0 || h > 20000.0 {
        return Opening::default();
    }
    let scale = num("scale").filter(|s| *s > 0.0).unwrap_or(1.0) as f32;
    let position = match (num("x"), num("y")) {
        (Some(x), Some(y)) if visible(x as i32, y as i32, w * scale, h * scale) => Some((x as i32, y as i32)),
        _ => None,
    };
    Opening { size: (w, h), position, maximized: v.get("maximized").and_then(Value::as_bool).unwrap_or(false) }
}

/// Keeps the window for [`save`] (`App::on_window`).
pub fn keep(window: &rune::WindowHandle) {
    WINDOW.with_borrow_mut(|w| *w = Some(window.clone()));
}

/// Saves where the window is now (when it closes). UI thread.
pub fn save() {
    let Some(place) = WINDOW.with_borrow(|w| w.as_ref().and_then(rune::WindowHandle::placement)) else { return };
    if place.minimized || place.full_screen {
        return; // keep what was saved before: neither is a place to reopen in
    }
    super::onboarding::update_settings(|v| {
        if place.maximized {
            // The maximized frame is not the size to return to: keep the last normal one.
            if !v["window"].is_object() {
                v["window"] = json!({ "w": DEFAULT_SIZE.0, "h": DEFAULT_SIZE.1 });
            }
            v["window"]["maximized"] = json!(true);
            return;
        }
        let scale = if place.scale > 0.0 { place.scale } else { 1.0 };
        v["window"] = json!({
            "x": place.position.0, "y": place.position.1,
            "w": (f64::from(place.content_size.0) / scale).round(), "h": (f64::from(place.content_size.1) / scale).round(),
            "scale": scale, "maximized": false,
        });
    });
}

/// Whether a window of `w` x `h` pixels at (`x`, `y`) would have its title area on a display.
#[cfg(windows)]
fn on_a_display(x: i32, y: i32, w: f32, _h: f32) -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN};
    // SAFETY: plain metric reads.
    let (vx, vy, vw, vh) = unsafe { (GetSystemMetrics(SM_XVIRTUALSCREEN), GetSystemMetrics(SM_YVIRTUALSCREEN), GetSystemMetrics(SM_CXVIRTUALSCREEN), GetSystemMetrics(SM_CYVIRTUALSCREEN)) };
    if vw <= 0 || vh <= 0 {
        return true;
    }
    // At least 120 px of the title bar's width and its top 40 px are inside the desktop.
    let right = x.saturating_add(w as i32);
    right > vx + 120 && x < vx + vw - 120 && y > vy - 20 && y < vy + vh - 40
}

#[cfg(not(windows))]
fn on_a_display(x: i32, y: i32, _w: f32, _h: f32) -> bool {
    // No cheap way to ask here: trust a plausible place (the window manager keeps a window reachable).
    (-20000..20000).contains(&x) && (-20000..20000).contains(&y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_saved_opens_centred_at_the_default_size() {
        assert_eq!(parse(None, |_, _, _, _| true), Opening::default());
        assert_eq!(parse(Some(&json!({ "w": 100, "h": 50 })), |_, _, _, _| true), Opening::default());
    }

    #[test]
    fn a_saved_place_is_used_when_it_is_on_a_display() {
        let saved = json!({ "x": 200, "y": 120, "w": 1300, "h": 800, "scale": 2.0, "maximized": false });
        let o = parse(Some(&saved), |x, y, w, h| (x, y, w, h) == (200, 120, 2600.0, 1600.0));
        assert_eq!(o, Opening { size: (1300.0, 800.0), position: Some((200, 120)), maximized: false });
    }

    #[test]
    fn a_place_off_every_display_keeps_the_size_and_centres() {
        let saved = json!({ "x": 9000, "y": 120, "w": 1300, "h": 800, "maximized": true });
        let o = parse(Some(&saved), |_, _, _, _| false);
        assert_eq!(o, Opening { size: (1300.0, 800.0), position: None, maximized: true });
    }
}
