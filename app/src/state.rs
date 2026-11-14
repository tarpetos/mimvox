//! Root application state. Each screen owns its sub-state struct in its own `ui/<screen>.rs` file.
//!
//! The root also owns the two long-lived backends: the engine process client and the audio player.
//! Both are `Option`s: `None` until the shell has started them (and for good if they cannot start,
//! e.g. no engine installed or no audio device), so every screen must cope with `None` and show a
//! friendly state instead of failing. Use the guarded helpers (`ui::shell::request`, `ui::playbar::play` and
//! friends, `ui::shell::stop_audio`) rather than touching them directly.

use crate::audio::Player;
use crate::engine::EngineClient;
use crate::ui::{
    book::BookState, formats::FormatsState, onboarding::OnboardingState, record::RecordState, redub::RedubState, relocate::RelocateState, shell::ShellState, speak::SpeakState, train::TrainState,
    voices::VoicesState,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    #[default]
    Onboarding,
    Voices,
    Record,
    Train,
    Speak,
    Redub,
}

impl Screen {
    /// The tab's name in `settings.json` (`screen`: the tab open when the app was closed). The welcome screen is not
    /// a tab to come back to.
    pub fn key(self) -> Option<&'static str> {
        Some(match self {
            Screen::Onboarding => return None,
            Screen::Voices => "voices",
            Screen::Record => "record",
            Screen::Train => "train",
            Screen::Speak => "speak",
            Screen::Redub => "redub",
        })
    }

    pub fn from_key(key: &str) -> Option<Screen> {
        [Screen::Voices, Screen::Record, Screen::Train, Screen::Speak, Screen::Redub].into_iter().find(|s| s.key() == Some(key))
    }
}

pub struct AppState {
    pub screen: Screen,
    /// Chrome state: theme, engine status, voice list, toasts (see `ui::shell`).
    pub shell: ShellState,
    /// The engine process; its messages arrive as events (see `ui::events`).
    pub engine: Option<EngineClient>,
    /// Audio playback; `None` when no output device is available.
    pub player: Option<Player>,
    pub onboarding: OnboardingState,
    pub voices: VoicesState,
    pub record: RecordState,
    pub train: TrainState,
    pub speak: SpeakState,
    /// Speak's Book mode: imported books and the book being read (see `ui::book`).
    pub book: BookState,
    pub redub: RedubState,
    /// Audio formats from the engine and the shared "Export as..." menu (see `ui::formats`).
    pub formats: FormatsState,
    /// Moving the data folder to another location (see `ui::relocate`).
    pub relocate: RelocateState,
    /// The shared language picker (see `ui::lang`).
    pub langpick: crate::ui::lang::PickerState,
    /// What the shared player bar shows (see `ui::playbar`).
    pub playbar: crate::ui::playbar::PlaybarState,
    /// Deleting saved results (see `ui::trash`).
    pub trash: crate::ui::trash::TrashState,
    /// Speak's and Redub's Results from earlier sessions (see `ui::history`).
    pub history: crate::ui::history::HistoryState,
    /// A newer release and its download (see `ui::update`).
    pub update: crate::ui::update::UpdateState,
    /// Setup run from inside the app (see `ui::setuprun`).
    pub setup: crate::ui::setuprun::SetupRunState,
}

impl Default for AppState {
    fn default() -> Self {
        let shell = ShellState::default();
        // The recorder writes into the same data dir as the engine: one source of truth on the shell.
        let mut record = RecordState::default();
        record.data_dir = shell.data_dir.clone();
        // Set up already: no welcome screen. With a voice, the tab that was open when the app was closed (Speak the
        // first time); without one, Voices (where "Get started" leads).
        let onboarding = OnboardingState::default();
        let screen = if onboarding.runtime_present && !crate::ui::shell::headless_run() {
            if has_voices(&shell.data_dir) { last_screen(crate::ui::onboarding::setting("screen").as_ref()) } else { Screen::Voices }
        } else {
            Screen::default()
        };
        AppState {
            screen,
            shell,
            engine: None,
            player: None,
            onboarding,
            voices: VoicesState::default(),
            record,
            train: TrainState::default(),
            speak: SpeakState::default(),
            book: BookState::default(),
            redub: RedubState::default(),
            formats: FormatsState::default(),
            relocate: RelocateState::default(),
            langpick: Default::default(),
            playbar: Default::default(),
            trash: Default::default(),
            history: Default::default(),
            update: Default::default(),
            setup: Default::default(),
        }
    }
}

/// Whether the data folder holds at least one voice (`voices/<id>/voice.json`), read once at start.
/// The tab saved at the last close; Speak when nothing (or nothing known) is saved.
fn last_screen(saved: Option<&serde_json::Value>) -> Screen {
    saved.and_then(serde_json::Value::as_str).and_then(Screen::from_key).unwrap_or(Screen::Speak)
}

fn has_voices(data_dir: &std::path::Path) -> bool {
    std::fs::read_dir(data_dir.join("voices"))
        .map(|dir| dir.flatten().any(|e| e.path().join("voice.json").is_file()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn the_last_tab_comes_back_and_anything_else_is_speak() {
        for tab in [Screen::Voices, Screen::Record, Screen::Train, Screen::Speak, Screen::Redub] {
            assert_eq!(last_screen(Some(&json!(tab.key().unwrap()))), tab);
        }
        assert_eq!(Screen::Onboarding.key(), None);
        assert_eq!(last_screen(None), Screen::Speak);
        assert_eq!(last_screen(Some(&json!("onboarding"))), Screen::Speak);
        assert_eq!(last_screen(Some(&json!(3))), Screen::Speak);
    }
}
