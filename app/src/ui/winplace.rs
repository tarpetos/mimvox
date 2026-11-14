//! The window opens where and as large as it was closed (`settings.json` `window`), and the first time in the middle
//! of the main display at its default size, not maximized.
//!
//! Saved when the window closes: the outer position in desktop pixels, the content size in points, and whether it
//! was maximized. A maximized window saves the place it returns to when un-maximized (on Windows, its restore
//! rectangle), which is on the display it was maximized on: the next start creates the window there and maximizes
//! it, so it comes back on the same display and not on the one the app was started from. It is placed when it is
//! created, never afterwards: moving a maximized window un-maximizes it. A place that is no
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
    let scale = if place.scale > 0.0 { place.scale } else { 1.0 };
    // The place a maximized window returns to; it also says which display it is maximized on.
    let restore = if place.maximized { WINDOW.with_borrow(|w| w.as_ref().and_then(restore_rect)) } else { None };
    super::onboarding::update_settings(|v| {
        if place.maximized {
            // The maximized frame is not the size to return to: keep the last normal one.
            if !v["window"].is_object() {
                v["window"] = json!({ "w": DEFAULT_SIZE.0, "h": DEFAULT_SIZE.1 });
            }
            v["window"]["maximized"] = json!(true);
            if let Some((x, y, w, h)) = restore {
                let saved = (v["window"]["w"].as_f64().unwrap_or(0.0), v["window"]["h"].as_f64().unwrap_or(0.0));
                merge_restore(&mut v["window"], (x, y), (f64::from(w) / scale, f64::from(h) / scale), saved, scale);
            }
            return;
        }
        v["window"] = json!({
            "x": place.position.0, "y": place.position.1,
            "w": (f64::from(place.content_size.0) / scale).round(), "h": (f64::from(place.content_size.1) / scale).round(),
            "scale": scale, "maximized": false,
        });
    });
}

/// Puts a maximized window's restore place into the saved `window`: always where, and the size only when it differs
/// clearly from the saved one (the restore rectangle is the outer frame, the saved size the content: taking a
/// near-equal one each time would let the window creep).
fn merge_restore(window: &mut Value, at: (i32, i32), size: (f64, f64), saved: (f64, f64), scale: f64) {
    window["x"] = json!(at.0);
    window["y"] = json!(at.1);
    window["scale"] = json!(scale);
    let sane = size.0 >= f64::from(MIN_SIZE.0) && size.1 >= f64::from(MIN_SIZE.1);
    if sane && ((size.0 - saved.0).abs() > 16.0 || (size.1 - saved.1).abs() > 16.0) {
        window["w"] = json!(size.0.round());
        window["h"] = json!(size.1.round());
    }
}

/// Where a maximized window goes back to when it is un-maximized: (x, y, width, height) in desktop pixels.
#[cfg(windows)]
fn restore_rect(window: &rune::WindowHandle) -> Option<(i32, i32, i32, i32)> {
    use rune::raw_window_handle::RawWindowHandle;
    use windows_sys::Win32::Foundation::RECT;
    use windows_sys::Win32::UI::WindowsAndMessaging::{GetWindowPlacement, SPI_GETWORKAREA, SystemParametersInfoW, WINDOWPLACEMENT};
    let hwnd = window.with_raw_handle(|h| match h.as_raw() {
        RawWindowHandle::Win32(w) => Some(w.hwnd.get() as windows_sys::Win32::Foundation::HWND),
        _ => None,
    }).ok()??;
    // SAFETY: plain reads into structures of the size given; `hwnd` is the live window `with_raw_handle` lent.
    unsafe {
        let mut wp: WINDOWPLACEMENT = std::mem::zeroed();
        wp.length = std::mem::size_of::<WINDOWPLACEMENT>() as u32;
        if GetWindowPlacement(hwnd, &mut wp) == 0 {
            return None;
        }
        // The restore rectangle is in workspace coordinates: desktop pixels counted from the main display's work
        // area (they differ only with a taskbar at the top or the left).
        let mut work: RECT = std::mem::zeroed();
        if SystemParametersInfoW(SPI_GETWORKAREA, 0, (&mut work as *mut RECT).cast(), 0) == 0 {
            work = std::mem::zeroed();
        }
        let r = wp.rcNormalPosition;
        Some((r.left + work.left, r.top + work.top, r.right - r.left, r.bottom - r.top))
    }
}

#[cfg(not(windows))]
fn restore_rect(_window: &rune::WindowHandle) -> Option<(i32, i32, i32, i32)> {
    None
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
    fn a_maximized_window_saves_where_it_returns_to() {
        // On the second display, with a frame a few pixels off the saved content size: the place moves, the size stays.
        let mut w = json!({ "x": 200, "y": 120, "w": 1300, "h": 800, "scale": 2.0, "maximized": true });
        merge_restore(&mut w, (4040, 300), (1301.0, 800.5), (1300.0, 800.0), 1.0);
        assert_eq!(w, json!({ "x": 4040, "y": 300, "w": 1300, "h": 800, "scale": 1.0, "maximized": true }));
        // Resized before it was maximized: the new size is taken too.
        merge_restore(&mut w, (4040, 300), (1500.0, 900.0), (1300.0, 800.0), 1.0);
        assert_eq!((w["w"].as_f64(), w["h"].as_f64()), (Some(1500.0), Some(900.0)));
        // A nonsense rectangle keeps the saved size.
        merge_restore(&mut w, (4040, 300), (10.0, 10.0), (1500.0, 900.0), 1.0);
        assert_eq!((w["w"].as_f64(), w["h"].as_f64()), (Some(1500.0), Some(900.0)));
    }

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
