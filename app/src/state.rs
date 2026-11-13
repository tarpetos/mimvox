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
}

impl Default for AppState {
    fn default() -> Self {
        let shell = ShellState::default();
        // The recorder writes into the same data dir as the engine: one source of truth on the shell.
        let mut record = RecordState::default();
        record.data_dir = shell.data_dir.clone();
        AppState {
            screen: Screen::default(),
            shell,
            engine: None,
            player: None,
            onboarding: OnboardingState::default(),
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
        }
    }
}
