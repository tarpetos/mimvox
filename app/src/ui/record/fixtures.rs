//! Headless scenes for snapshots: `scenes()` returns state fixtures the shell can register in `AppSpec.scenes`.

use super::*;
use widgets::Tone;

fn part(i: u32, title: &str, text: &str, secs: f32) -> PartText {
    PartText { index: i, title: title.into(), text: text.into(), est_seconds: secs }
}

const TEXT_1: &str = "The old lighthouse keeper climbed the spiral stairs every evening, long before the first star appeared. He carried a lantern, a thermos of tea, and a notebook where he wrote down the weather, the ships, and the small things nobody else would ever notice. On clear nights he could see the whole coast stretching away like a silver thread.";
const TEXT_2: &str = "Yesterday I measured the garden twice, and the zebra still would not fit through the quiet wooden gate. Judge for yourself whether such a vision is pleasure or measure; the treasure was a huge jar of jam.";

fn parts5() -> Vec<PartText> {
    vec![
        part(1, "A quiet evening", TEXT_1, 70.0),
        part(2, "Everyday sounds", TEXT_2, 62.0),
        part(3, "Numbers and dates", "On the third of March, twenty-four people met at nine thirty to count seventeen boxes of blue ceramic cups.", 75.0),
        part(4, "Questions", "Why does the river bend so sharply here? Who built the bridge, and when did it first open to travellers?", 80.0),
        part(5, "Closing", "That is all for today. Thank you for listening, and take care until we speak again.", 68.0),
    ]
}

fn presets_fixture() -> Vec<Preset> {
    vec![
        Preset {
            id: "quick".into(),
            title: "Quick".into(),
            description: "Read one short passage. Your voice works at once, no training needed.".into(),
            target_minutes: 1.0,
            trains_rvc: false,
            parts: parts5().into_iter().take(1).collect(),
        },
        Preset {
            id: "standard".into(),
            title: "Standard".into(),
            description: "About six minutes of reading, then a short training run for a close copy of your voice.".into(),
            target_minutes: 6.5,
            trains_rvc: true,
            parts: parts5(),
        },
        Preset {
            id: "pro".into(),
            title: "Pro".into(),
            description: "Fifteen to twenty minutes for the best similarity and the most natural expression.".into(),
            target_minutes: 17.0,
            trains_rvc: true,
            parts: parts5().into_iter().cycle().take(12).collect(),
        },
    ]
}

fn base(s: &mut AppState) {
    s.screen = Screen::Record;
    s.record.presets = Load::Ready(presets_fixture());
    s.record.voice_id = Some("demo".into());
    s.shell.active_voice = Some("demo".into());
    s.shell.voices = vec![crate::ui::shell::VoiceInfo { id: "demo".into(), name: "My voice".into(), ..Default::default() }];
    s.shell.set_engine(crate::ui::shell::EngineStatus::Ready { device: Some("NVIDIA GeForce RTX 4070".into()), tts: Some("ready".into()), rvc: Some("ready".into()) });
}

fn standard(s: &mut AppState) {
    base(s);
    s.record.choose_preset("standard");
    s.record.phase = Phase::Session;
}

fn saved(d: f32, v: Verdict, issues: &[&str]) -> Take {
    Take { status: TakeStatus::Saved, duration_s: d, verdict: Some(v), issues: issues.iter().map(|s| s.to_string()).collect(), path: Some(PathBuf::from("part.wav")), error: None, imported: None }
}

fn summary(total: f32, parts: &[(u32, f32, Verdict)], level: Level, message: &str) -> Summary {
    let parts = parts.iter().map(|&(index, duration_s, verdict)| SummaryPart { index, duration_s, verdict, issues: vec![] }).collect();
    Summary { total_minutes: total, usable_minutes: total, has_reference: true, parts, level, message: message.into(), recommended_epochs: Some(300) }
}

pub fn scene_picker(s: &mut AppState) {
    base(s);
}

pub fn scene_own_text(s: &mut AppState) {
    base(s);
    s.record.phase = Phase::Own;
    s.record.own.replace_text(format!("{TEXT_1}\n\n{TEXT_2}"));
    s.record.analysis = Some(Analysis {
        language: "en".into(),
        detected: true,
        words: 842,
        est_minutes: 5.6,
        coverage: 0.86,
        missing: vec!["ZH".into(), "OY".into(), "UH".into()],
        level: Level::Trainable,
        message: "Good length. A few sounds are rare in your text, so add a sentence with \"vision\", \"toy\" and \"book\" for a better result.".into(),
        parts: parts5().into_iter().take(4).collect(),
    });
}

const UK_1: &str = "Привіт! Це мій голос, і зараз я записую його, щоб комп'ютер навчився звучати точнісінько як я. Сьогодні звичайний спокійний день. Я сиджу в тихій кімнаті з горнятком м'ятного чаю й читаю ці рядки неквапом, ніби розмовляю з добрим другом.";

fn uk_presets() -> Vec<Preset> {
    let mut list = presets_fixture();
    for p in &mut list {
        for (i, part) in p.parts.iter_mut().enumerate() {
            part.title = ["Одна хвилина, один запис", "Звуки щодня", "Числа й дати", "Запитання", "На прощання"][i % 5].into();
            part.text = UK_1.into();
        }
    }
    list
}

fn uk_languages(s: &mut AppState) {
    s.record.languages = [("en", "English", "English"), ("uk", "Ukrainian", "Українська"), ("de", "German", "Deutsch"), ("fr", "French", "Français"), ("es", "Spanish", "Español"), ("pl", "Polish", "Polski")]
        .iter()
        .map(|(c, n, nat)| RecLang { code: (*c).into(), name: (*n).into(), native: (*nat).into() })
        .collect();
}

/// The preset picker in Ukrainian.
pub fn scene_language_uk(s: &mut AppState) {
    base(s);
    uk_languages(s);
    s.record.language = "uk".into();
    s.record.presets = Load::Ready(uk_presets());
}

/// Reading a Ukrainian preset into a new voice (the active one was recorded in English).
pub fn scene_session_uk(s: &mut AppState) {
    scene_language_uk(s);
    s.shell.voices[0].parts = 5;
    choose_preset(s, "standard");
    s.record.phase = Phase::Session;
}

/// "Use my own text" with a detected Ukrainian text.
pub fn scene_own_text_uk(s: &mut AppState) {
    base(s);
    uk_languages(s);
    s.record.phase = Phase::Own;
    s.record.own.replace_text(UK_1.to_owned());
    s.record.analysis = Some(Analysis {
        language: "uk".into(),
        detected: true,
        words: 46,
        est_minutes: 0.4,
        coverage: 0.84,
        missing: vec!["dzʲ".into(), "ɡ".into()],
        level: Level::ZeroShot,
        message: "Enough for a quick clone. For training, read a few minutes more.".into(),
        parts: uk_presets().remove(0).parts,
    });
}

pub fn scene_recording(s: &mut AppState) {
    standard(s);
    s.record.current = 1;
    s.record.takes[0] = saved(68.0, Verdict::Ok, &[]);
    s.record.summary = Some(summary(1.1, &[(1, 68.0, Verdict::Ok)], Level::ZeroShot, "Enough for a quick clone. Keep going to train."));
    s.record.active = Some(Active { recorder: None, seq: 0, part: 1, level: 0.42, peak: 0.55, elapsed_s: 37.0, clipped: false });
}

pub fn scene_recording_loud(s: &mut AppState) {
    scene_recording(s);
    s.record.active = Some(Active { recorder: None, seq: 0, part: 1, level: 0.97, peak: 1.0, elapsed_s: 12.0, clipped: true });
}

pub fn scene_saved_with_issues(s: &mut AppState) {
    standard(s);
    s.record.current = 2;
    s.record.takes[0] = saved(68.0, Verdict::Ok, &[]);
    s.record.takes[1] = saved(64.0, Verdict::Ok, &[]);
    s.record.takes[2] = saved(72.0, Verdict::Warn, &["Background noise is a bit high: close the window or move away from the fan", "A few words were slightly clipped: lower the input volume"]);
    s.record.summary = Some(summary(3.4, &[(1, 68.0, Verdict::Ok), (2, 64.0, Verdict::Ok), (3, 72.0, Verdict::Warn)], Level::ZeroShot, "Almost there. Two more parts to reach a trainable voice."));
}

pub fn scene_rejected(s: &mut AppState) {
    standard(s);
    s.record.current = 1;
    s.record.takes[0] = saved(68.0, Verdict::Ok, &[]);
    s.record.takes[1] = Take {
        status: TakeStatus::Rejected,
        duration_s: 9.0,
        verdict: Some(Verdict::Bad),
        issues: vec!["Too quiet: speak closer to the mic".into(), "Too short: read the whole part".into()],
        path: Some(PathBuf::from("take.wav")),
        error: None,
        imported: None,
    };
    s.record.notice = None;
}

pub fn scene_summary_ready(s: &mut AppState) {
    standard(s);
    let durs = [(1, 68.0), (2, 63.0), (3, 80.0), (4, 85.0), (5, 70.0)];
    for (i, d) in durs {
        s.record.takes[i as usize - 1] = saved(d, if i == 3 { Verdict::Warn } else { Verdict::Ok }, &[]);
    }
    let parts: Vec<_> = durs.iter().map(|(i, d)| (*i, *d, Verdict::Ok)).collect();
    s.record.summary = Some(summary(6.1, &parts, Level::Trainable, "Great: 6.1 minutes is enough to train a close copy of your voice (about 300 epochs, around 55 minutes)."));
    s.record.phase = Phase::Summary;
}

pub fn scene_quick_ready(s: &mut AppState) {
    base(s);
    s.record.choose_preset("quick");
    s.record.takes[0] = saved(61.0, Verdict::Ok, &[]);
    s.record.summary = Some(summary(1.0, &[(1, 61.0, Verdict::Ok)], Level::ZeroShot, "Enough for a quick clone."));
    s.record.phase = Phase::Summary;
}

pub fn scene_notice_saved(s: &mut AppState) {
    scene_recording(s);
    s.record.active = None;
    s.record.current = 2;
    s.record.takes[1] = saved(64.0, Verdict::Ok, &[]);
    s.record.notice = Some(("Part 2 saved (1:04)".into(), Tone::Success));
}

fn mic_devices(s: &mut AppState) {
    s.record.mic.settings_read = true;
    s.record.mic.loaded = true;
    s.record.mic.devices = vec!["Microphone (Realtek(R) Audio)".into(), "Blue Yeti Stereo Microphone".into(), "Headset Microphone (Logi USB Headset)".into()];
}

/// The chip on the picker, showing a chosen device.
pub fn scene_mic_picker(s: &mut AppState) {
    base(s);
    mic_devices(s);
    s.record.input_device = Some("Blue Yeti Stereo Microphone".into());
}

/// The chip on the session with the dropdown open and a live level.
pub fn scene_mic_open(s: &mut AppState) {
    standard(s);
    mic_devices(s);
    s.record.input_device = Some("Blue Yeti Stereo Microphone".into());
    s.record.mic.open = true;
    s.record.mic.test = mic::Test::Running(mic::Live { rec: None, seq: 0, level: 0.38, peak: 0.52 });
}

/// The saved device is gone: fallback to the system default with a warning.
pub fn scene_mic_warning(s: &mut AppState) {
    standard(s);
    mic_devices(s);
    s.record.mic.warning = Some("\"USB Podcast Mic\" is not available, using the system default".into());
}

/// A file is being dragged over the recorder card.
pub fn scene_import_drop(s: &mut AppState) {
    standard(s);
    s.record.current = 1;
    s.record.takes[0] = saved(68.0, Verdict::Ok, &[]);
    s.record.import_over = true;
    s.formats.over = true;
    s.formats.scene_ready();
}

/// An existing recording (an MP3) was imported for part 2 and is being checked.
pub fn scene_import_checking(s: &mut AppState) {
    standard(s);
    s.record.current = 1;
    s.record.takes[0] = saved(68.0, Verdict::Ok, &[]);
    s.record.takes[1] = Take { status: TakeStatus::Checking, path: Some(PathBuf::from("/home/me/Voice Memos/evening-reading.mp3")), imported: Some("evening-reading.mp3".into()), ..Take::default() };
}

/// The imported recording passed the check with notes and was added to the dataset.
pub fn scene_import_saved(s: &mut AppState) {
    standard(s);
    s.record.current = 1;
    s.record.takes[0] = saved(68.0, Verdict::Ok, &[]);
    let mut t = saved(61.0, Verdict::Warn, &["A little background noise: a quieter room helps"]);
    t.imported = Some("evening-reading.mp3".into());
    s.record.takes[1] = t;
    s.record.summary = Some(summary(2.1, &[(1, 68.0, Verdict::Ok), (2, 61.0, Verdict::Warn)], Level::ZeroShot, "Enough for a quick clone. Keep going to train."));
}
