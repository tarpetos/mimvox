//! UI: shell (navigation, glass chrome), theme, shared components and one module per screen.
//!
//! * `theme`: the dark smoked-glass palette, type scale, radii, springs.
//! * `components`: GlassPanel, Button, Slider, TextArea, ProgressBar, LevelMeter, Badge, Toast, icons.
//! * `shell`: header, tabs, engine status, toasts and the event routing (see its module docs for the
//!   `on_engine` / `settle` convention every screen implements).
//! * `events`: how background threads (engine, audio, dialogs) wake the UI; there is no polling.

pub mod book;
pub mod caption;
pub mod components;
pub mod events;
pub mod formats;
pub mod lang;
pub mod onboarding;
pub mod playbar;
pub mod record;
pub mod refclip;
pub mod refmedia;
pub mod redub;
pub mod relocate;
pub mod shell;
pub mod speak;
pub mod stats;
pub mod theme;
pub mod train;
pub mod trash;
pub mod voices;

use crate::state::AppState;
use rune::{Cx, El, KeyEvent, Tx};
use std::sync::OnceLock;

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    stats::count(&stats::VIEWS);
    shell::view(s, cx)
}

/// Named states for headless snapshots (`--snap out.png --scene <name>`).
pub const SCENES: &[(&str, fn(&mut AppState))] = &[
    ("record_picker", record::scene_picker),
    ("record_own_text", record::scene_own_text),
    ("record_language_uk", record::scene_language_uk),
    ("record_session_uk", record::scene_session_uk),
    ("record_own_text_uk", record::scene_own_text_uk),
    ("record_recording", record::scene_recording),
    ("record_recording_loud", record::scene_recording_loud),
    ("record_saved", record::scene_notice_saved),
    ("record_saved_issues", record::scene_saved_with_issues),
    ("record_rejected", record::scene_rejected),
    ("record_summary_ready", record::scene_summary_ready),
    ("record_quick_ready", record::scene_quick_ready),
    ("record_mic_picker", record::scene_mic_picker),
    ("record_mic_open", record::scene_mic_open),
    ("record_mic_warning", record::scene_mic_warning),
    ("record_import_drop", record::scene_import_drop),
    ("record_import_checking", record::scene_import_checking),
    ("record_import_saved", record::scene_import_saved),
    ("speak_idle", speak::scene_idle),
    ("speak_typed", speak::scene_typed),
    ("speak_language_uk", speak::scene_language_uk),
    ("speak_language_picker", speak::scene_language_picker),
    ("speak_accent_native", speak::scene_accent_native),
    ("speak_accent_own", speak::scene_accent_own),
    ("speak_accent_untrained", speak::scene_accent_untrained),
    ("speak_accent_zero_shot", speak::scene_accent_zero_shot),
    ("speak_accent_foreign_ref", speak::scene_accent_foreign_ref),
    ("speak_accent_foreign_ref_zero_shot", speak::scene_accent_foreign_ref_zero_shot),
    ("speak_accent_foreign_ref_trained", speak::scene_accent_foreign_ref_trained),
    ("speak_generating", speak::scene_generating),
    ("speak_buffering", speak::scene_buffering),
    ("speak_preparing", speak::scene_preparing),
    ("speak_done", speak::scene_done),
    ("speak_trash_confirm", speak::scene_trash_confirm),
    ("speak_toast_long", speak::scene_toast_long),
    ("speak_export_menu", speak::scene_export_menu),
    ("speak_output_menu", speak::scene_output_menu),
    ("move_confirm", relocate::scene_confirm),
    ("move_running", relocate::scene_running),
    ("move_failed", relocate::scene_failed),
    ("speak_no_voice", speak::scene_no_voice),
    ("engine_starting", speak::scene_engine_starting),
    ("engine_stopped", speak::scene_engine_stopped),
];

type Scene = (&'static str, fn(&mut AppState));

/// Every scene of every screen (the screens each list their own).
pub fn all_scenes() -> &'static [Scene] {
    static ALL: OnceLock<Vec<Scene>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut all: Vec<Scene> = SCENES.to_vec();
        all.extend_from_slice(caption::scenes());
        all.extend_from_slice(onboarding::scenes());
        all.extend_from_slice(voices::scenes());
        all.extend_from_slice(refclip::scenes());
        all.extend_from_slice(refmedia::scenes());
        all.extend_from_slice(train::scenes());
        all.extend_from_slice(redub::scenes());
        all.extend_from_slice(playbar::scenes());
        all.extend_from_slice(book::scenes());
        all
    })
}

/// `App::on_focus`: coming back to the window is when an outside change (setup finished in its terminal) shows.
pub fn on_focus(s: &mut AppState, focused: bool, _tx: &mut Tx) {
    if focused {
        onboarding::on_focus(s);
    }
}

/// Window-level keys: each screen gets a chance, the first to take the key wins (Space records on Record,
/// Ctrl+Enter generates on Speak, ...), so they work without clicking into anything first.
pub fn on_key(s: &mut AppState, ev: &KeyEvent, tx: &mut Tx) -> bool {
    record::on_key(s, ev, tx) || speak::on_key(s, ev, tx)
}
