//! Windows: the moving part of the player's seek bar (the fill and the knob) in a small window of its own.
//!
//! Why: every frame Rune draws is presented as the whole window, and Windows then composites the whole window again.
//! Measured on an RTX 4070 (idle 11.4 W): a knob that moves 60 times a second through the window's own frames keeps
//! the card at about 36 W, however cheap each frame is to draw (0.08 ms), while a small window that redraws just the
//! strip costs about 14 W. So the knob glides in a layered child window over the seek rail: a few thousand pixels
//! drawn on the CPU on a 16 ms window timer, and nothing of the main window is redrawn while a track plays (its
//! clock still changes once a second).
//!
//! The strip takes no input (`HTTRANSPARENT`): clicks and drags reach the seek bar under it. When it cannot be made
//! (a system without layered child windows) or is not wanted (another element covers the bar, a headless run), the
//! bar draws its own knob, as on the other systems (`playbar::seek_bar`).

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, SIZE, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    CreateCompatibleDC, CreateDIBSection, DeleteDC, DeleteObject, SelectObject, AC_SRC_ALPHA, AC_SRC_OVER, BITMAPINFO, BITMAPINFOHEADER,
    BI_RGB, BLENDFUNCTION, DIB_RGB_COLORS, HBITMAP, HDC,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Threading::GetCurrentProcessId;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, EnumWindows, GetWindow, GetWindowThreadProcessId, IsWindowVisible, KillTimer,
    RegisterClassW, SetTimer, SetWindowPos, ShowWindow, UpdateLayeredWindow, GW_OWNER, HTTRANSPARENT, SWP_NOACTIVATE, SWP_NOZORDER,
    SW_HIDE, SW_SHOWNA, ULW_ALPHA, WM_NCHITTEST, WM_TIMER, WNDCLASSW, WS_CHILD, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TRANSPARENT,
};

/// The knob's frames a second while a track plays (the window timer's period, rounded down to whole milliseconds).
const TICK_MS: u32 = 16;
/// Points the strip reaches past the rail on each side (the knob's radius and its shadow).
const REACH_PT: f32 = 14.0;
/// The strip's height in points (the knob with its shadow).
const HEIGHT_PT: f32 = 30.0;

/// How the strip is drawn: sizes in points, colours as straight sRGB.
#[derive(Clone, Copy, PartialEq)]
pub struct Look {
    pub track_pt: f32,
    pub knob_pt: f32,
    pub fill_top: [u8; 3],
    pub fill_bottom: [u8; 3],
    pub ring: [u8; 3],
    /// Without the knob (a track still being generated cannot be sought).
    pub knob: bool,
}

/// The audio thread's last report: (track, seconds played, the track's length, when).
static HEARD: Mutex<Option<(u64, f32, f32, Instant)>> = Mutex::new(None);
/// What the strip follows: the playing track, its speed, and whether it plays (else `PLACED` is shown).
static TRACK: AtomicU64 = AtomicU64::new(0);
static SPEED: AtomicU32 = AtomicU32::new(0);
static PLAYING: AtomicBool = AtomicBool::new(false);
static PLACED: AtomicU32 = AtomicU32::new(0);

struct Strip {
    hwnd: HWND,
    dc: HDC,
    bitmap: HBITMAP,
    bits: *mut u32,
    /// The strip's size in pixels and where the rail is in it: its left edge, its width, the centre line.
    size: (i32, i32),
    rail: (f32, f32, f32),
    scale: f32,
    look: Look,
    ticking: bool,
    shown: bool,
    drawn: f32,
}

thread_local! {
    static STRIP: RefCell<Option<Strip>> = const { RefCell::new(None) };
    static FAILED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The audio thread heard the track at `pos_s` of `len_s` (any thread).
pub fn heard(track: u64, pos_s: f32, len_s: f32) {
    *HEARD.lock().unwrap_or_else(|e| e.into_inner()) = Some((track, pos_s, len_s, Instant::now()));
}

/// Puts the strip over the seek rail (`rail_pt`: x, y, width, height in window points) and says what it shows: the
/// playing `track` gliding on at `speed`, or the still fraction `frac`. UI thread, from the view. False when there is
/// no strip (the bar then draws its own knob).
pub fn show(rail_pt: [f32; 4], scale: f32, look: Look, track: u64, playing: bool, speed: f32, frac: f32) -> bool {
    if FAILED.get() || rail_pt[2] < 4.0 || scale <= 0.0 {
        return false;
    }
    TRACK.store(track, Relaxed);
    SPEED.store(speed.to_bits(), Relaxed);
    PLACED.store(frac.to_bits(), Relaxed);
    PLAYING.store(playing, Relaxed);
    let px = |v: f32| (v * scale).round() as i32;
    let reach = px(REACH_PT);
    let (w, h) = (px(rail_pt[2]) + 2 * reach, px(HEIGHT_PT));
    let x = px(rail_pt[0]) - reach;
    let y = ((rail_pt[1] + rail_pt[3] / 2.0) * scale - h as f32 / 2.0).round() as i32;
    STRIP.with_borrow_mut(|slot| {
        if slot.as_ref().is_some_and(|s| s.size != (w, h)) {
            destroy(slot.take());
        }
        if slot.is_none() {
            match create(x, y, w, h) {
                Some(strip) => *slot = Some(strip),
                None => {
                    FAILED.set(true);
                    return false;
                }
            }
        }
        let s = slot.as_mut().expect("just made");
        s.rail = (reach as f32, rail_pt[2] * scale, h as f32 / 2.0);
        (s.scale, s.look) = (scale, look);
        // SAFETY: the strip's own window, on its thread.
        unsafe {
            SetWindowPos(s.hwnd, std::ptr::null_mut(), x, y, w, h, SWP_NOACTIVATE | SWP_NOZORDER);
            if !s.shown {
                ShowWindow(s.hwnd, SW_SHOWNA);
                s.shown = true;
            }
            if playing != s.ticking {
                if playing { SetTimer(s.hwnd, 1, TICK_MS, None); } else { KillTimer(s.hwnd, 1); }
                s.ticking = playing;
            }
        }
        s.drawn = f32::NAN; // the look or the size may have changed: draw
        paint(s);
        true
    })
}

/// Hides the strip (the bar is gone, or something covers it). UI thread.
pub fn hide() {
    STRIP.with_borrow_mut(|slot| {
        if let Some(s) = slot.as_mut().filter(|s| s.shown) {
            // SAFETY: the strip's own window, on its thread.
            unsafe {
                KillTimer(s.hwnd, 1);
                ShowWindow(s.hwnd, SW_HIDE);
            }
            (s.shown, s.ticking) = (false, false);
        }
    });
}

/// The fraction of the track to show now.
fn fraction() -> f32 {
    if PLAYING.load(Relaxed) {
        let heard = *HEARD.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, pos_s, len_s, at)) = heard.filter(|h| h.0 == TRACK.load(Relaxed) && h.2 > 0.0) {
            let now_s = pos_s + at.elapsed().as_secs_f32() * f32::from_bits(SPEED.load(Relaxed));
            return (now_s / len_s).clamp(0.0, 1.0);
        }
    }
    f32::from_bits(PLACED.load(Relaxed)).clamp(0.0, 1.0)
}

/// Draws the strip at the current fraction and hands it to the compositor, unless it already shows it.
fn paint(s: &mut Strip) {
    let frac = fraction();
    let (x0, rail_w, cy) = s.rail;
    let at = x0 + frac * rail_w;
    if (at - s.drawn).abs() < 0.05 {
        return;
    }
    s.drawn = at;
    let (w, h) = (s.size.0 as usize, s.size.1 as usize);
    // SAFETY: the DIB section's pixels, `w * h` of them, owned by this strip and touched only on its thread.
    let pixels = unsafe { std::slice::from_raw_parts_mut(s.bits, w * h) };
    pixels.fill(0);
    let look = s.look;
    let half = look.track_pt * s.scale / 2.0;
    let radius = look.knob_pt * s.scale / 2.0;
    let shadow_reach = 5.0 * s.scale;
    let shadow_drop = 2.0 * s.scale;
    // Only the columns that can have paint: the fill's span and the knob's.
    let x_end = ((at + radius + shadow_reach + 2.0).ceil() as usize).min(w);
    for y in 0..h {
        let py = y as f32 + 0.5;
        let dy = py - cy;
        let row = &mut pixels[y * w..(y + 1) * w];
        // The fill: a capsule from the rail's start to the knob, lit from the top.
        let in_fill_rows = dy.abs() <= half + 1.0;
        let t = ((dy + half) / (2.0 * half)).clamp(0.0, 1.0);
        let fill_rgb = [mix(look.fill_top[0], look.fill_bottom[0], t), mix(look.fill_top[1], look.fill_bottom[1], t), mix(look.fill_top[2], look.fill_bottom[2], t)];
        let (a, b) = (x0 + half, (at).max(x0 + half));
        let x_start = ((x0 - 1.0).floor().max(0.0)) as usize;
        for (x, out) in row.iter_mut().enumerate().take(x_end).skip(x_start) {
            let pxc = x as f32 + 0.5;
            let mut c = [0.0f32; 4]; // premultiplied r, g, b, a
            if in_fill_rows && frac > 0.0 {
                let nearest = pxc.clamp(a, b);
                let d = ((pxc - nearest).powi(2) + dy * dy).sqrt() - half;
                let cover = (0.5 - d).clamp(0.0, 1.0);
                if cover > 0.0 {
                    c = [fill_rgb[0] * cover, fill_rgb[1] * cover, fill_rgb[2] * cover, cover];
                }
            }
            if look.knob {
                let dx = pxc - at;
                // A soft shadow under the knob, the knob's face, and a thin ring in the accent colour.
                let ds = (dx * dx + (dy - shadow_drop).powi(2)).sqrt() - radius;
                let shade = 0.45 * (1.0 - (ds / shadow_reach).clamp(0.0, 1.0)).powi(2);
                if shade > 0.0 {
                    c = over([0.0, 0.0, 0.0, shade], c);
                }
                let d = (dx * dx + dy * dy).sqrt() - radius;
                let cover = (0.5 - d).clamp(0.0, 1.0);
                if cover > 0.0 {
                    let ring = (1.0 - ((d + s.scale).abs() / s.scale.max(1.0))).clamp(0.0, 1.0) * 0.6;
                    let face = [
                        mixf(0.93, f32::from(look.ring[0]) / 255.0, ring),
                        mixf(0.95, f32::from(look.ring[1]) / 255.0, ring),
                        mixf(0.98, f32::from(look.ring[2]) / 255.0, ring),
                    ];
                    let alpha = 0.92 * cover;
                    c = over([face[0] * alpha, face[1] * alpha, face[2] * alpha, alpha], c);
                }
            }
            if c[3] > 0.0 {
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                *out = (q(c[3]) << 24) | (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2]);
            }
        }
    }
    let size = SIZE { cx: s.size.0, cy: s.size.1 };
    let origin = POINT { x: 0, y: 0 };
    let blend = BLENDFUNCTION { BlendOp: AC_SRC_OVER as u8, BlendFlags: 0, SourceConstantAlpha: 255, AlphaFormat: AC_SRC_ALPHA as u8 };
    // SAFETY: the strip's window and its memory DC with the DIB section selected; the structs live through the call.
    unsafe {
        UpdateLayeredWindow(s.hwnd, std::ptr::null_mut(), std::ptr::null(), &size, s.dc, &origin, 0, &blend, ULW_ALPHA);
    }
}

fn mix(a: u8, b: u8, t: f32) -> f32 {
    mixf(f32::from(a) / 255.0, f32::from(b) / 255.0, t)
}

fn mixf(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Premultiplied `top` over premultiplied `bottom`.
fn over(top: [f32; 4], bottom: [f32; 4]) -> [f32; 4] {
    let k = 1.0 - top[3];
    [top[0] + bottom[0] * k, top[1] + bottom[1] * k, top[2] + bottom[2] * k, top[3] + bottom[3] * k]
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    match msg {
        // Input goes to the seek bar under the strip.
        WM_NCHITTEST => HTTRANSPARENT as LRESULT,
        WM_TIMER => {
            // `try_borrow_mut`: a timer message dispatched while `show` holds the strip waits for the next one.
            STRIP.with(|slot| {
                if let Ok(mut slot) = slot.try_borrow_mut() {
                    if let Some(s) = slot.as_mut().filter(|s| s.hwnd == hwnd && s.shown) {
                        paint(s);
                    }
                }
            });
            0
        }
        // SAFETY: the default handling of this window's own message.
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

/// The app's window: the visible, unowned top-level window of this process.
fn main_window() -> Option<HWND> {
    unsafe extern "system" fn each(hwnd: HWND, found: LPARAM) -> i32 {
        let mut pid = 0;
        // SAFETY: a window handle the system enumerates; `found` is the `HWND` slot `main_window` passed.
        unsafe {
            GetWindowThreadProcessId(hwnd, &mut pid);
            if pid == GetCurrentProcessId() && IsWindowVisible(hwnd) != 0 && GetWindow(hwnd, GW_OWNER).is_null() {
                *(found as *mut HWND) = hwnd;
                return 0;
            }
        }
        1
    }
    let mut found: HWND = std::ptr::null_mut();
    // SAFETY: `each` writes only through the pointer to `found`, which outlives the call.
    unsafe { EnumWindows(Some(each), &mut found as *mut HWND as LPARAM) };
    (!found.is_null()).then_some(found)
}

fn create(x: i32, y: i32, w: i32, h: i32) -> Option<Strip> {
    let parent = main_window()?;
    let class: Vec<u16> = "MimvoxSeekStrip\0".encode_utf16().collect();
    // SAFETY: plain Win32 window and GDI object creation; every handle made here is owned by the returned strip (or
    // released before returning None).
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        static REGISTERED: AtomicBool = AtomicBool::new(false);
        if !REGISTERED.swap(true, Relaxed) {
            let wc = WNDCLASSW {
                style: 0, lpfnWndProc: Some(wndproc), cbClsExtra: 0, cbWndExtra: 0, hInstance: instance, hIcon: std::ptr::null_mut(),
                hCursor: std::ptr::null_mut(), hbrBackground: std::ptr::null_mut(), lpszMenuName: std::ptr::null(), lpszClassName: class.as_ptr(),
            };
            RegisterClassW(&wc);
        }
        let hwnd = CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE, class.as_ptr(), std::ptr::null(), WS_CHILD, x, y, w, h,
            parent, std::ptr::null_mut(), instance, std::ptr::null());
        if hwnd.is_null() {
            return None; // no layered child windows here: the bar draws its own knob
        }
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        let mut info: BITMAPINFO = std::mem::zeroed();
        info.bmiHeader = BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32, biWidth: w, biHeight: -h, biPlanes: 1, biBitCount: 32,
            biCompression: BI_RGB, biSizeImage: 0, biXPelsPerMeter: 0, biYPelsPerMeter: 0, biClrUsed: 0, biClrImportant: 0,
        };
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if dc.is_null() || bitmap.is_null() || bits.is_null() {
            if !bitmap.is_null() { DeleteObject(bitmap); }
            if !dc.is_null() { DeleteDC(dc); }
            DestroyWindow(hwnd);
            return None;
        }
        SelectObject(dc, bitmap);
        Some(Strip {
            hwnd, dc, bitmap, bits: bits.cast(), size: (w, h), rail: (0.0, 1.0, h as f32 / 2.0), scale: 1.0,
            look: Look { track_pt: 5.0, knob_pt: 14.0, fill_top: [0; 3], fill_bottom: [0; 3], ring: [0; 3], knob: true },
            ticking: false, shown: false, drawn: f32::NAN,
        })
    }
}

fn destroy(strip: Option<Strip>) {
    if let Some(s) = strip {
        // SAFETY: the strip's own handles, released once.
        unsafe {
            KillTimer(s.hwnd, 1);
            DestroyWindow(s.hwnd);
            DeleteDC(s.dc);
            DeleteObject(s.bitmap);
        }
    }
}
