#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod audio;
#[cfg(target_os = "linux")]
mod desktop;
mod state;
mod ui;
mod update;

use mimvox_engine_client as engine;
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
    for var in [
        "PATH",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "XDG_DATA_DIRS",
        "PERLLIB",
        "QT_PLUGIN_PATH",
        "GST_PLUGIN_SYSTEM_PATH",
        "GST_PLUGIN_SYSTEM_PATH_1_0",
        "GSETTINGS_SCHEMA_DIR",
        "PYTHONHOME",
    ] {
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
    let fds: Vec<i32> =
        dir.flatten().filter_map(|e| e.file_name().to_str()?.parse().ok()).filter(|fd| *fd > 2).collect();
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

/// `mimvox serve ...` / `mimvox mcp ...`: the mimvox-server / mimvox-mcp next to this binary, with the remaining
/// arguments. An AppImage starts only its main binary, so this is how the bundled ones are reached.
fn run_sibling(name: &str, args: impl Iterator<Item = std::ffi::OsString>) -> ! {
    let exe = std::env::current_exe().ok();
    // macOS reports a symlink's own path; Windows' canonical form would be a \\?\ path.
    #[cfg(unix)]
    let exe = exe.map(|p| p.canonicalize().unwrap_or(p));
    let dir = exe.and_then(|p| p.parent().map(std::path::Path::to_path_buf)).unwrap_or_default();
    let bin = dir.join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    if !bin.is_file() {
        eprintln!("mimvox: {} not found", bin.display());
        std::process::exit(2);
    }
    let mut cmd = std::process::Command::new(&bin);
    cmd.args(args);
    #[cfg(unix)]
    let err = std::os::unix::process::CommandExt::exec(&mut cmd);
    #[cfg(not(unix))]
    let err = match cmd.status() {
        Ok(status) => std::process::exit(status.code().unwrap_or(1)),
        Err(e) => e,
    };
    eprintln!("mimvox: cannot start {}: {err}", bin.display());
    std::process::exit(126);
}

const APP_ID: &str = "io.github.tarpetos.mimvox";

fn tiling_wm() -> bool {
    if !cfg!(target_os = "linux") {
        return false;
    }
    let socket = ["I3SOCK", "SWAYSOCK", "HYPRLAND_INSTANCE_SIGNATURE", "NIRI_SOCKET"]
        .iter()
        .any(|v| std::env::var_os(v).is_some());
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default().to_ascii_lowercase();
    socket
        || desktop.split(':').any(|d| {
            matches!(
                d,
                "i3" | "sway" | "hyprland" | "niri" | "bspwm" | "dwm" | "awesome" | "qtile" | "river" | "xmonad"
            )
        })
}

fn main() {
    leave_appimage_env();
    let mut args = std::env::args_os().skip(1).peekable();
    // Before the AppImage fds turn close-on-exec: the program run in this process's place keeps the image mounted.
    if args.next_if(|a| a == "serve").is_some() {
        run_sibling("mimvox-server", args);
    }
    if args.next_if(|a| a == "mcp").is_some() {
        run_sibling("mimvox-mcp", args);
    }
    #[cfg(target_os = "linux")]
    keep_appimage_fds_to_self();
    // Headless runs (snapshots, scripted recordings) may run beside the app.
    if !rune::is_headless() {
        if !rune::single_instance(APP_ID) {
            return;
        }
        #[cfg(windows)]
        engine::hold_app_mutex();
    }
    // Before `run`, so every window opened afterwards shows it.
    let _ = rune::set_app_icon(ICON_PNG);
    #[cfg(target_os = "linux")]
    {
        rune::set_app_id(desktop::app_id());
        if !rune::is_headless() {
            desktop::integrate();
        }
    }
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
    let titlebar =
        Titlebar::new(ui::shell::HEADER_HEIGHT).traffic_lights(18.0, (ui::shell::HEADER_HEIGHT - 14.0) / 2.0);
    let titlebar = if tiling_wm() { titlebar } else { titlebar.custom_caption(ui::caption::CELL) };
    // Rune cannot veto a close: the window only hides and `train::on_close` decides.
    // A maximized window is created at its saved place and never moved: moving it un-maximizes it.
    let mut options =
        rune::WindowOptions { transparent: cfg!(target_os = "linux") && !tiling_wm(), ..Default::default() };
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
    ui::api::stop_on_exit();
    if let Some(engine) = ui::train::take_closing_engine() {
        engine.shutdown();
    }
    // Only after the app and the engine are gone.
    update::install_pending();
    ui::stats::write();
}
