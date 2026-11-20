//! Windows: the seek knob glides in a layered child window, so the main window is not redrawn while a track plays.
//! Measured on an RTX 4070: about 36 W redrawing the main window, about 14 W with the strip. The strip runs on a
//! thread of its own, ticked by the audio thread, so a UI thread waiting for a busy GPU cannot stop the knob.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU32, Ordering::Relaxed, Ordering::SeqCst};

use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, POINT, SIZE, WPARAM};
use windows_sys::Win32::Graphics::Gdi::{
    AC_SRC_ALPHA, AC_SRC_OVER, BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION, CreateCompatibleDC,
    CreateDIBSection, DIB_RGB_COLORS, DeleteDC, DeleteObject, HBITMAP, HDC, SelectObject,
};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, EnumWindows, GW_OWNER, GetMessageW, GetWindow,
    GetWindowThreadProcessId, HTTRANSPARENT, IsWindowVisible, MSG, PM_NOREMOVE, PeekMessageW, PostThreadMessageW,
    RegisterClassW, SW_HIDE, SW_SHOWNA, SWP_NOACTIVATE, SWP_NOZORDER, SetWindowPos, ShowWindow, ULW_ALPHA,
    UpdateLayeredWindow, WM_APP, WM_DESTROY, WM_NCHITTEST, WNDCLASSW, WS_CHILD, WS_EX_LAYERED, WS_EX_NOACTIVATE,
    WS_EX_NOPARENTNOTIFY, WS_EX_TRANSPARENT,
};

/// The knob's radius and its shadow.
const REACH_PT: f32 = 14.0;
/// The knob with its shadow.
const HEIGHT_PT: f32 = 30.0;

const WM_SYNC: u32 = WM_APP + 1;
const WM_TICK: u32 = WM_APP + 2;

/// Sizes in points, colours as straight sRGB.
#[derive(Clone, Copy, PartialEq)]
pub struct Look {
    pub track_pt: f32,
    pub knob_pt: f32,
    pub fill_top: [u8; 3],
    pub fill_bottom: [u8; 3],
    pub ring: [u8; 3],
}

/// What the UI thread asks the strip to show. `rect`: x, y, width, height in the main window's pixels; `rail`:
/// the rail's left edge, width and centre line in the strip's pixels.
#[derive(Clone, Copy, PartialEq)]
struct Want {
    shown: bool,
    rect: [i32; 4],
    rail: (f32, f32, f32),
    scale: f32,
    look: Look,
    playing: bool,
    speed: f32,
    frac: f32,
}

static WANT: Mutex<Option<Want>> = Mutex::new(None);
/// The strip thread's id once its message queue exists.
static THREAD: AtomicU32 = AtomicU32::new(0);
static STARTED: AtomicBool = AtomicBool::new(false);
static FAILED: AtomicBool = AtomicBool::new(false);
static TICKING: AtomicBool = AtomicBool::new(false);
static TICK_QUEUED: AtomicBool = AtomicBool::new(false);
static SYNC_QUEUED: AtomicBool = AtomicBool::new(false);
/// The strip's window while it exists (it goes with the main window).
static WINDOW: AtomicIsize = AtomicIsize::new(0);

struct Strip {
    hwnd: HWND,
    dc: HDC,
    bitmap: HBITMAP,
    bits: *mut u32,
    /// Pixels.
    size: (i32, i32),
    want: Want,
    shown: bool,
    drawn: f32,
    glide: super::playbar::Glide,
}

/// `rail_pt`: x, y, width, height in window points. UI thread. False when there is no strip (the bar then draws
/// its own knob).
pub fn show(rail_pt: [f32; 4], scale: f32, look: Look, playing: bool, speed: f32, frac: f32) -> bool {
    if FAILED.load(Relaxed) || rail_pt[2] < 4.0 || scale <= 0.0 {
        hide();
        return false;
    }
    let px = |v: f32| (v * scale).round() as i32;
    let reach = px(REACH_PT);
    let (w, h) = (px(rail_pt[2]) + 2 * reach, px(HEIGHT_PT));
    let x = px(rail_pt[0]) - reach;
    let y = ((rail_pt[1] + rail_pt[3] / 2.0) * scale - h as f32 / 2.0).round() as i32;
    let want = Want {
        shown: true,
        rect: [x, y, w, h],
        rail: (reach as f32, rail_pt[2] * scale, h as f32 / 2.0),
        scale,
        look,
        playing,
        speed,
        frac,
    };
    TICKING.store(playing, Relaxed);
    {
        let mut asked = WANT.lock().unwrap_or_else(|e| e.into_inner());
        if *asked == Some(want) {
            return true;
        }
        *asked = Some(want);
    }
    if !STARTED.swap(true, SeqCst) && std::thread::Builder::new().name("seek-strip".into()).spawn(run).is_err() {
        FAILED.store(true, Relaxed);
        return false;
    }
    post(WM_SYNC, &SYNC_QUEUED);
    true
}

/// UI thread.
pub fn hide() {
    TICKING.store(false, Relaxed);
    let mut asked = WANT.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(w) = asked.as_mut().filter(|w| w.shown) {
        w.shown = false;
        drop(asked);
        post(WM_SYNC, &SYNC_QUEUED);
    }
}

/// Audio thread: one more frame of the knob while the track plays.
pub fn tick() {
    if TICKING.load(Relaxed) {
        post(WM_TICK, &TICK_QUEUED);
    }
}

/// At most one of each message waits in the strip's queue.
fn post(msg: u32, queued: &AtomicBool) {
    let thread = THREAD.load(SeqCst);
    if thread == 0 || queued.swap(true, SeqCst) {
        return;
    }
    // SAFETY: posting a message with no pointers in it to a thread of this process.
    if unsafe { PostThreadMessageW(thread, msg, 0, 0) } == 0 {
        queued.store(false, SeqCst);
    }
}

/// The strip thread: its window, its message loop.
fn run() {
    // SAFETY: `msg` is written by the message calls before it is read; the queue belongs to this thread.
    unsafe {
        let mut msg: MSG = std::mem::zeroed();
        PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_NOREMOVE); // makes this thread's queue
        THREAD.store(GetCurrentThreadId(), SeqCst);
        let mut strip = None;
        sync(&mut strip);
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            match msg.message {
                WM_SYNC if msg.hwnd.is_null() => {
                    SYNC_QUEUED.store(false, SeqCst);
                    sync(&mut strip);
                }
                WM_TICK if msg.hwnd.is_null() => {
                    TICK_QUEUED.store(false, SeqCst);
                    if let Some(s) = strip.as_mut().filter(|s| s.shown && s.want.playing && alive(s)) {
                        paint(s);
                    }
                }
                _ => {
                    DispatchMessageW(&msg);
                }
            }
        }
    }
}

fn fail() {
    FAILED.store(true, Relaxed);
    TICKING.store(false, Relaxed);
    super::events::post(|_| {}); // the view runs again and the bar draws its own knob
}

fn alive(s: &Strip) -> bool {
    WINDOW.load(SeqCst) == s.hwnd as isize
}

fn sync(slot: &mut Option<Strip>) {
    let Some(want) = *WANT.lock().unwrap_or_else(|e| e.into_inner()) else { return };
    if slot.as_ref().is_some_and(|s| !alive(s)) {
        release(slot.take());
    }
    if !want.shown {
        if let Some(s) = slot.as_mut().filter(|s| s.shown) {
            // SAFETY: the strip's own window, on its thread.
            unsafe { ShowWindow(s.hwnd, SW_HIDE) };
            s.shown = false;
            s.glide.rest();
        }
        return;
    }
    if slot.is_none() {
        *slot = create(&want);
        if slot.is_none() {
            fail();
            return;
        }
    }
    let Some(s) = slot.as_mut() else { return };
    let [x, y, w, h] = want.rect;
    if s.size != (w, h) {
        // SAFETY: the strip's memory DC; the old bitmap is released once the new one is selected.
        match unsafe { surface(s.dc, w, h) } {
            Some((bitmap, bits)) => {
                // SAFETY: as above.
                unsafe { DeleteObject(std::mem::replace(&mut s.bitmap, bitmap)) };
                (s.bits, s.size) = (bits, (w, h));
            }
            None => {
                release(slot.take());
                fail();
                return;
            }
        }
    }
    if s.want.rect != want.rect || !s.shown {
        // SAFETY: the strip's own window, on its thread.
        unsafe { SetWindowPos(s.hwnd, std::ptr::null_mut(), x, y, w, h, SWP_NOACTIVATE | SWP_NOZORDER) };
    }
    if !want.playing {
        s.glide.rest();
    }
    s.want = want;
    s.drawn = f32::NAN;
    paint(s);
    if !s.shown {
        // SAFETY: as above.
        unsafe { ShowWindow(s.hwnd, SW_SHOWNA) };
        s.shown = true;
    }
}

fn fraction(s: &mut Strip) -> f32 {
    if s.want.playing
        && let Some(frac) = s.glide.step(s.want.speed)
    {
        return frac.clamp(0.0, 1.0);
    }
    s.want.frac.clamp(0.0, 1.0)
}

#[cfg(test)]
static PAINTS: AtomicU32 = AtomicU32::new(0);

fn paint(s: &mut Strip) {
    let frac = fraction(s);
    let (x0, rail_w, cy) = s.want.rail;
    let at = x0 + frac * rail_w;
    if (at - s.drawn).abs() < 0.05 {
        return;
    }
    s.drawn = at;
    let (w, h) = (s.size.0 as usize, s.size.1 as usize);
    // SAFETY: the DIB section's pixels, `w * h` of them, owned by this strip and touched only on its thread.
    let pixels = unsafe { std::slice::from_raw_parts_mut(s.bits, w * h) };
    pixels.fill(0);
    let look = s.want.look;
    let half = look.track_pt * s.want.scale / 2.0;
    let radius = look.knob_pt * s.want.scale / 2.0;
    let shadow_reach = 5.0 * s.want.scale;
    let shadow_drop = 2.0 * s.want.scale;
    let x_end = ((at + radius + shadow_reach + 2.0).ceil() as usize).min(w);
    for y in 0..h {
        let py = y as f32 + 0.5;
        let dy = py - cy;
        let row = &mut pixels[y * w..(y + 1) * w];
        let in_fill_rows = dy.abs() <= half + 1.0;
        let t = ((dy + half) / (2.0 * half)).clamp(0.0, 1.0);
        let fill_rgb = [
            mix(look.fill_top[0], look.fill_bottom[0], t),
            mix(look.fill_top[1], look.fill_bottom[1], t),
            mix(look.fill_top[2], look.fill_bottom[2], t),
        ];
        let (a, b) = (x0 + half, (at).max(x0 + half));
        let knob_left = at - radius - shadow_reach - 2.0;
        let x_start = (x0 - 1.0).min(knob_left).floor().max(0.0) as usize;
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
            let dx = pxc - at;
            let ds = (dx * dx + (dy - shadow_drop).powi(2)).sqrt() - radius;
            let shade = 0.45 * (1.0 - (ds / shadow_reach).clamp(0.0, 1.0)).powi(2);
            if shade > 0.0 {
                c = over([0.0, 0.0, 0.0, shade], c);
            }
            let d = (dx * dx + dy * dy).sqrt() - radius;
            let cover = (0.5 - d).clamp(0.0, 1.0);
            if cover > 0.0 {
                let ring = (1.0 - ((d + s.want.scale).abs() / s.want.scale.max(1.0))).clamp(0.0, 1.0) * 0.6;
                let face = [
                    mixf(0.93, f32::from(look.ring[0]) / 255.0, ring),
                    mixf(0.95, f32::from(look.ring[1]) / 255.0, ring),
                    mixf(0.98, f32::from(look.ring[2]) / 255.0, ring),
                ];
                let alpha = 0.92 * cover;
                c = over([face[0] * alpha, face[1] * alpha, face[2] * alpha, alpha], c);
            }
            if c[3] > 0.0 {
                let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32;
                *out = (q(c[3]) << 24) | (q(c[0]) << 16) | (q(c[1]) << 8) | q(c[2]);
            }
        }
    }
    let size = SIZE { cx: s.size.0, cy: s.size.1 };
    let origin = POINT { x: 0, y: 0 };
    let blend = BLENDFUNCTION {
        BlendOp: AC_SRC_OVER as u8,
        BlendFlags: 0,
        SourceConstantAlpha: 255,
        AlphaFormat: AC_SRC_ALPHA as u8,
    };
    // SAFETY: the strip's window and its memory DC with the DIB section selected; the structs live through the call.
    let _done = unsafe {
        UpdateLayeredWindow(s.hwnd, std::ptr::null_mut(), std::ptr::null(), &size, s.dc, &origin, 0, &blend, ULW_ALPHA)
    };
    #[cfg(test)]
    if _done != 0 {
        PAINTS.fetch_add(1, Relaxed);
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
        WM_NCHITTEST => HTTRANSPARENT as LRESULT,
        WM_DESTROY => {
            let _ = WINDOW.compare_exchange(hwnd as isize, 0, SeqCst, SeqCst);
            0
        }
        // SAFETY: the default handling of this window's own message.
        _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
    }
}

#[cfg(test)]
static TEST_PARENT: AtomicIsize = AtomicIsize::new(0);

fn parent() -> Option<HWND> {
    #[cfg(test)]
    if TEST_PARENT.load(SeqCst) != 0 {
        return Some(TEST_PARENT.load(SeqCst) as HWND);
    }
    main_window()
}

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

/// A 32-bit top-down DIB section selected into `dc`, and its pixels.
///
/// # Safety
/// `dc` is a valid memory DC.
unsafe fn surface(dc: HDC, w: i32, h: i32) -> Option<(HBITMAP, *mut u32)> {
    // SAFETY: plain GDI calls on the caller's DC; the info struct lives through the call.
    unsafe {
        let mut info: BITMAPINFO = std::mem::zeroed();
        info.bmiHeader = BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w,
            biHeight: -h,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB,
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        };
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        let bitmap = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &mut bits, std::ptr::null_mut(), 0);
        if bitmap.is_null() || bits.is_null() {
            if !bitmap.is_null() {
                DeleteObject(bitmap);
            }
            return None;
        }
        SelectObject(dc, bitmap);
        Some((bitmap, bits.cast()))
    }
}

fn create(want: &Want) -> Option<Strip> {
    let parent = parent()?;
    let class: Vec<u16> = "MimvoxSeekStrip\0".encode_utf16().collect();
    let [x, y, w, h] = want.rect;
    // SAFETY: plain Win32 window and GDI object creation; every handle made here is owned by the returned strip (or
    // released before returning None).
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        static REGISTERED: AtomicBool = AtomicBool::new(false);
        if !REGISTERED.swap(true, Relaxed) {
            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: instance,
                hIcon: std::ptr::null_mut(),
                hCursor: std::ptr::null_mut(),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            RegisterClassW(&wc);
        }
        // No parent notification: it would be sent to the UI thread and wait for it.
        let hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_NOPARENTNOTIFY,
            class.as_ptr(),
            std::ptr::null(),
            WS_CHILD,
            x,
            y,
            w,
            h,
            parent,
            std::ptr::null_mut(),
            instance,
            std::ptr::null(),
        );
        if hwnd.is_null() {
            return None; // no layered child windows here: the bar draws its own knob
        }
        let dc = CreateCompatibleDC(std::ptr::null_mut());
        let made = if dc.is_null() { None } else { surface(dc, w, h) };
        let Some((bitmap, bits)) = made else {
            if !dc.is_null() {
                DeleteDC(dc);
            }
            DestroyWindow(hwnd);
            return None;
        };
        WINDOW.store(hwnd as isize, SeqCst);
        Some(Strip {
            hwnd,
            dc,
            bitmap,
            bits,
            size: (w, h),
            want: Want { shown: false, ..*want },
            shown: false,
            drawn: f32::NAN,
            glide: super::playbar::Glide::new(),
        })
    }
}

fn release(strip: Option<Strip>) {
    if let Some(s) = strip {
        // SAFETY: the strip's own window (unless the main window took it along) and GDI handles, released once.
        unsafe {
            if alive(&s) {
                WINDOW.store(0, SeqCst);
                DestroyWindow(s.hwnd);
            }
            DeleteDC(s.dc);
            DeleteObject(s.bitmap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};
    use windows_sys::Win32::UI::WindowsAndMessaging::{PM_REMOVE, WS_POPUP};

    fn pump(how_long: Duration) {
        let end = Instant::now() + how_long;
        // SAFETY: `msg` is written by PeekMessageW before it is dispatched.
        unsafe {
            let mut msg: MSG = std::mem::zeroed();
            while Instant::now() < end {
                while PeekMessageW(&mut msg, std::ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                    DispatchMessageW(&msg);
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    #[test]
    fn the_knob_moves_on_while_the_ui_thread_is_busy() {
        let class: Vec<u16> = "MimvoxSeekStripTest\0".encode_utf16().collect();
        // SAFETY: a hidden window of this test, made and destroyed on this thread.
        let parent = unsafe {
            let instance = GetModuleHandleW(std::ptr::null());
            let wc = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(DefWindowProcW),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: instance,
                hIcon: std::ptr::null_mut(),
                hCursor: std::ptr::null_mut(),
                hbrBackground: std::ptr::null_mut(),
                lpszMenuName: std::ptr::null(),
                lpszClassName: class.as_ptr(),
            };
            RegisterClassW(&wc);
            CreateWindowExW(
                0,
                class.as_ptr(),
                std::ptr::null(),
                WS_POPUP,
                0,
                0,
                800,
                100,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                instance,
                std::ptr::null(),
            )
        };
        assert!(!parent.is_null());
        TEST_PARENT.store(parent as isize, SeqCst);
        super::super::playbar::jump(1, 0.0, 30.0);
        let look = Look { track_pt: 5.0, knob_pt: 14.0, fill_top: [120; 3], fill_bottom: [90; 3], ring: [90; 3] };
        assert!(show([20.0, 40.0, 600.0, 5.0], 1.0, look, true, 1.0, 0.0));
        let deadline = Instant::now() + Duration::from_secs(5);
        while PAINTS.load(SeqCst) == 0 && !FAILED.load(SeqCst) && Instant::now() < deadline {
            pump(Duration::from_millis(10));
        }
        assert!(!FAILED.load(SeqCst), "no strip");
        assert!(PAINTS.load(SeqCst) > 0, "the strip never drew");
        let before = PAINTS.load(SeqCst);
        // This thread stands for a busy UI thread: it handles no messages while the audio thread ticks.
        let audio = std::thread::spawn(|| {
            let start = Instant::now();
            for _ in 0..15 {
                super::super::playbar::report(1, start.elapsed().as_secs_f32(), 30.0, false, Instant::now());
                tick();
                std::thread::sleep(Duration::from_millis(33));
            }
        });
        std::thread::sleep(Duration::from_millis(500));
        audio.join().unwrap();
        let drawn = PAINTS.load(SeqCst) - before;
        assert!(drawn >= 8, "{drawn} frames in half a second");
        hide();
        pump(Duration::from_millis(50));
        // SAFETY: the window made above, on its thread.
        unsafe { DestroyWindow(parent) };
    }
}
