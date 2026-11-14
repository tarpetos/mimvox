//! Root application state. `engine` and `player` stay `None` when they cannot start: use the guarded helpers
//! (`ui::shell::request`, `ui::playbar::play`, `ui::shell::stop_audio`) rather than touching them directly.

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
    /// The tab's name in `settings.json`; `None` for the welcome screen.
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
    pub shell: ShellState,
    pub engine: Option<EngineClient>,
    /// `None` when no output device is available.
    pub player: Option<Player>,
    pub onboarding: OnboardingState,
    pub voices: VoicesState,
    pub record: RecordState,
    pub train: TrainState,
    pub speak: SpeakState,
    pub book: BookState,
    pub redub: RedubState,
    pub formats: FormatsState,
    pub relocate: RelocateState,
    pub langpick: crate::ui::lang::PickerState,
    pub playbar: crate::ui::playbar::PlaybarState,
    pub trash: crate::ui::trash::TrashState,
    pub history: crate::ui::history::HistoryState,
    pub update: crate::ui::update::UpdateState,
    pub setup: crate::ui::setuprun::SetupRunState,
}

impl Default for AppState {
    fn default() -> Self {
        let shell = ShellState::default();
        let mut record = RecordState::default();
        record.data_dir = shell.data_dir.clone();
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
