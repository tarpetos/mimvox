//! Mimvox: record your voice, train it, speak any text with it.

#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod audio;
mod engine;
mod state;
mod ui;

use rune::{App, AppSpec, Titlebar};
use state::AppState;

/// Noto Sans (SIL OFL, see assets/fonts/OFL.txt).
const NOTO_REGULAR: &[u8] = include_bytes!("../assets/fonts/NotoSans-Regular.ttf");
const NOTO_BOLD: &[u8] = include_bytes!("../assets/fonts/NotoSans-Bold.ttf");
/// The app icon (same artwork as `tools/release/icons`): the window and taskbar icon on Windows, the Dock icon on
/// macOS. On Windows the executable also carries it as a resource for Explorer (see `build.rs`).
const ICON_PNG: &[u8] = include_bytes!("../assets/icon.png");

fn main() {
    // Before `run`: every window opened afterwards shows it. A failure only costs the icon.
    let _ = rune::set_app_icon(ICON_PNG);
    let spec = AppSpec::<AppState> {
        min_size: Some((1040.0, 620.0)),
        scenes: ui::all_scenes(),
        fonts: &[NOTO_REGULAR, NOTO_BOLD],
        families: ("Noto Sans", "Noto Sans"),
        setup: Some(ui::shell::setup_icon),
        // The frame hook: settles state and fires due timers; it never polls the engine or audio (see `ui::events`).
        poll: Some(ui::shell::on_frame),
        on_key: Some(ui::on_key),
        ..AppSpec::new("Mimvox", (1040.0, 700.0), AppState::default, ui::view)
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
    let app = App::new(spec)
        .titlebar(titlebar)
        .keep_running()
        .on_close(ui::train::on_close)
        .on_focus(ui::on_focus)
        .on_frame_drawn(|_| ui::stats::count(&ui::stats::FRAMES));
    // Background threads (engine, audio, dialogs) reach the UI through this handle (`ui::events`).
    ui::events::init(app.handle());
    app.run();
    // The window is gone: now wait for the engine a quit asked to exit (it is killed after a grace period).
    if let Some(mut engine) = ui::train::take_closing_engine() {
        engine.shutdown();
    }
    ui::stats::write();
}
