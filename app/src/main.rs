#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod audio;
mod engine;
mod state;
mod update;
mod ui;

use rune::{App, AppSpec, Titlebar};
use state::AppState;

/// Noto Sans (SIL OFL, see assets/fonts/OFL.txt).
const NOTO_REGULAR: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");
const NOTO_BOLD: &[u8] = include_bytes!("../assets/fonts/NotoSans-Bold.ttf");
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

/// AppRun points `PYTHONHOME` and the search paths into the image; child processes must not inherit them.
fn leave_appimage_env() {
    let Some(appdir) = std::env::var_os("APPDIR").filter(|_| std::env::var_os("APPIMAGE").is_some()) else { return };
    let appdir = std::path::PathBuf::from(appdir);
    let inside = |p: &std::path::Path| p.starts_with(&appdir);
    for var in ["PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "XDG_DATA_DIRS", "PERLLIB", "QT_PLUGIN_PATH",
        "GST_PLUGIN_SYSTEM_PATH", "GST_PLUGIN_SYSTEM_PATH_1_0", "GSETTINGS_SCHEMA_DIR", "PYTHONHOME"] {
        let Some(value) = std::env::var_os(var) else { continue };
        let kept: Vec<_> = std::env::split_paths(&value).filter(|p| !p.as_os_str().is_empty() && !inside(p)).collect();
        // SAFETY: called first thing in `main`, before any other thread exists.
        unsafe {
            match std::env::join_paths(&kept) {
                Ok(joined) if !kept.is_empty() => std::env::set_var(var, joined),
                _ => std::env::remove_var(var),
            }
        }
    }
    // SAFETY: as above.
    unsafe { std::env::remove_var("PYTHONDONTWRITEBYTECODE") };
}

/// The AppImage runtime's inherited fds (mount pipe, mount dir) keep an old image mounted while any child lives:
/// they stay open here but close on exec.
#[cfg(target_os = "linux")]
fn keep_appimage_fds_to_self() {
    if std::env::var_os("APPIMAGE").is_none() {
        return;
    }
    let Ok(dir) = std::fs::read_dir("/proc/self/fd") else { return };
    let fds: Vec<i32> = dir.flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).filter(|fd| *fd > 2).collect();
    for fd in fds {
        // SAFETY: fcntl on a descriptor number only reads and sets its flags; a stale number (the directory listing's own
        // descriptor, closed by now) fails with EBADF and is skipped.
        unsafe {
            let flags = libc::fcntl(fd, libc::F_GETFD);
            if flags >= 0 {
                libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC);
            }
        }
    }
}

const APP_ID: &str = "io.github.tarpetos.mimvox";
/// The installer and uninstaller wait while it is held (scripts/build/inno.py, APP_MUTEX).
#[cfg(windows)]
const APP_MUTEX: &str = "Local\\Mimvox-running";

#[cfg(windows)]
fn hold_app_mutex() {
    use windows_sys::Win32::System::Threading::CreateMutexW;
    let name: Vec<u16> = APP_MUTEX.encode_utf16().chain(std::iter::once(0)).collect();
    // SAFETY: a NUL-terminated name; the handle is never closed, so the mutex lives as long as the process.
    let _ = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
}

fn main() {
    leave_appimage_env();
    #[cfg(target_os = "linux")]
    keep_appimage_fds_to_self();
    // Headless runs (snapshots, scripted recordings) may run beside the app.
    if !rune::is_headless() {
        if !rune::single_instance(APP_ID) {
            return;
        }
        #[cfg(windows)]
        hold_app_mutex();
    }
    // Before `run`, so every window opened afterwards shows it.
    let _ = rune::set_app_icon(ICON_PNG);
    let opening = if rune::is_headless() { ui::winplace::Opening::default() } else { ui::winplace::opening() };
    let spec = AppSpec::<AppState> {
        min_size: Some((1040.0, 620.0)),
        scenes: ui::all_scenes(),
        fonts: &[NOTO_REGULAR, NOTO_BOLD],
        families: ("Noto Sans", "Noto Sans"),
        setup: Some(ui::shell::setup_icon),
        poll: Some(ui::shell::on_frame),
        on_key: Some(ui::on_key),
        ..AppSpec::new("Mimvox", opening.size, AppState::default, ui::view)
    };
    // Windows draws its caption buttons itself (`ui::caption`, cells Rune hit-tests as the system's); macOS insets
    // the traffic lights; Linux keeps the native frame.
    let titlebar = Titlebar::new(ui::shell::HEADER_HEIGHT)
        .traffic_lights(18.0, (ui::shell::HEADER_HEIGHT - 14.0) / 2.0)
        .custom_caption(ui::caption::CELL);
    // Rune cannot veto a close: the window only hides and `train::on_close` decides.
    // A maximized window is created at its saved place and never moved: moving it un-maximizes it.
    let mut options = rune::WindowOptions::default();
    if opening.maximized {
        options.position = opening.position;
    }
    let app = App::new(spec)
        .window_options(options)
        .titlebar(titlebar)
        .keep_running()
        .maximized(opening.maximized)
        .on_window(ui::winplace::keep)
        .on_close(ui::train::on_close)
        .on_focus(ui::on_focus)
        .on_frame_drawn(|_| ui::stats::count(&ui::stats::FRAMES));
    ui::events::init(app.handle());
    match opening.position {
        _ if opening.maximized => {}
        Some(position) => app.handle().set_position(position),
        None if !rune::is_headless() => ui::shell::center_on_open(),
        None => {}
    }
    app.run();
    ui::setuprun::stop_on_exit();
    if let Some(mut engine) = ui::train::take_closing_engine() {
        engine.shutdown();
    }
    // Only after the app and the engine are gone.
    update::install_pending();
    ui::stats::write();
}
