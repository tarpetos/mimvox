//! Mimvox: record your voice, train it, speak any text with it.

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
/// The app icon (same artwork as `tools/release/icons`): the window and taskbar icon on Windows, the Dock icon on
/// macOS. On Windows the executable also carries it as a resource for Explorer (see `build.rs`).
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

/// An AppImage's launcher (AppRun) points `PYTHONHOME` into the image and puts the image's folders first in `PATH`,
/// `LD_LIBRARY_PATH`, `PYTHONPATH`, `XDG_DATA_DIRS` and a few more. Every process the app starts inherits them: the
/// setup's and the engine's Python then cannot find their standard library, and a terminal or `xdg-open` may load the
/// wrong libraries. The image bundles no libraries the app needs at run time, so the entries inside it are dropped
/// before anything starts (still one thread here, so changing the environment is sound).
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

/// The app's id: the single-instance pipe (`rune::single_instance`) and the Windows mutex below are named after it.
const APP_ID: &str = "io.github.tarpetos.mimvox";
/// Held while the app runs (Windows): the installer and the uninstaller check it (scripts/build/inno.py, APP_MUTEX) and
/// install or remove nothing while it is held.
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
    // One Mimvox at a time: a second launch brings the running one's window forward and ends here. Headless runs
    // (snapshots, scripted recordings) open no window and may run beside the app.
    if !rune::is_headless() {
        if !rune::single_instance(APP_ID) {
            return;
        }
        #[cfg(windows)]
        hold_app_mutex();
    }
    // Before `run`: every window opened afterwards shows it. A failure only costs the icon.
    let _ = rune::set_app_icon(ICON_PNG);
    // Where and how large the window was closed; the first time, the default size in the middle of the main display.
    let opening = if rune::is_headless() { ui::winplace::Opening::default() } else { ui::winplace::opening() };
    let spec = AppSpec::<AppState> {
        min_size: Some((1040.0, 620.0)),
        scenes: ui::all_scenes(),
        fonts: &[NOTO_REGULAR, NOTO_BOLD],
        families: ("Noto Sans", "Noto Sans"),
        setup: Some(ui::shell::setup_icon),
        // The frame hook: settles state and fires due timers; it never polls the engine or audio (see `ui::events`).
        poll: Some(ui::shell::on_frame),
        on_key: Some(ui::on_key),
        ..AppSpec::new("Mimvox", opening.size, AppState::default, ui::view)
    };
    // The glass header is the title bar: Windows loses its native title bar (resize edges, Aero Snap and the snap
    // layouts keep working) and the app draws the caption buttons itself (`ui::caption`, over cells Rune
    // hit-tests as the system's caption buttons), macOS keeps its traffic lights inset into the header. Linux keeps
    // the native frame.
    let titlebar = Titlebar::new(ui::shell::HEADER_HEIGHT)
        .traffic_lights(18.0, (ui::shell::HEADER_HEIGHT - 14.0) / 2.0)
        .custom_caption(ui::caption::CELL);
    // Rune cannot veto a close, so the window only hides (`keep_running`) and `train::on_close` quits, or, while a
    // training runs, brings the window back with the "Training is running" dialog.
    // A window that opens maximized is created at its saved place, on the display it was maximized on, and maximizes
    // there. It is not moved afterwards: moving or centring a maximized window puts it back to its normal size (on
    // Windows, setting the position un-maximizes it).
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
    // Background threads (engine, audio, dialogs) reach the UI through this handle (`ui::events`).
    ui::events::init(app.handle());
    match opening.position {
        _ if opening.maximized => {}
        Some(position) => app.handle().set_position(position),
        None if !rune::is_headless() => ui::shell::center_on_open(),
        None => {}
    }
    app.run();
    // A setup still running stops with the app (it resumes on the next run).
    ui::setuprun::stop_on_exit();
    // The window is gone: now wait for the engine a quit asked to exit (it is killed after a grace period).
    if let Some(mut engine) = ui::train::take_closing_engine() {
        engine.shutdown();
    }
    // An update downloaded this session installs now that the app and the engine are gone.
    update::install_pending();
    ui::stats::write();
}
