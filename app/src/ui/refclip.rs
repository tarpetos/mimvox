//! "Reference clip" section of the active voice card (Voices screen).
//!
//! The speech model (OmniVoice) copies the tempo and manner of the voice's `reference.wav`, so the clip decides how fast
//! and natural the cloned voice sounds. The engine picks one automatically; here the user sees which clip is in use and
//! can open a glass list of the best candidates (`voices.reference_candidates`), play each one and switch with "Use this
//! one" (`voices.set_reference`, which re-conditions the speaker; the engine warms the voice again by itself).
//!
//! **Per-language references.** For speech in another language than the one the voice was recorded in, a short clip in
//! that language (`reference_<lang>.wav`, about 10 s) carries its accent over. "Add a language..." picks the language,
//! shows a native sample text to read (from that language's presets, `presets.list`), records it with the same recorder
//! and quality check as the Record screen (`recording.check`), or imports a file, and stores it with
//! `voices.add_reference`; `voices.remove_reference` drops one. The voice list names the languages that have one.
//!
//! **Spoken language.** Under the clip in use, the language the engine detected in it (`voices.list`
//! `reference_language`, e.g. "Ukrainian speech"); when it is not the language the voice was recorded in, a note that
//! speech in that language carries the clip's accent, and how to avoid it.
//!
//! **Reference from media** (`refmedia`): many audio / video files of the same speaker are combined into a pool and its
//! best window becomes the reference; the pool's other good windows join the candidate list (`source: "media"`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use rune::{El, col, el, row, text};
use serde_json::{Value, json};

use crate::audio::{CaptureEvent, Recorder};
use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use crate::ui::components::{Button, Tone, accent_glass, glass_card, icon_el, icons, level_meter};
use crate::ui::lang;
use crate::ui::playbar;
use crate::ui::shell::{self, VoiceInfo};
use crate::ui::theme::{Palette, motion, radius, ty};
use crate::ui::voices::kit::{self, BtnKind};

/// One clip of the recordings, as described by the engine.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct Clip {
    /// `part` (a recorded part) or `media` (a window of the combined media files).
    pub source: String,
    /// The media file the window comes from (1-based, in the order the files were given).
    pub file: Option<u32>,
    pub part: Option<u32>,
    pub title: Option<String>,
    pub start_s: f32,
    pub end_s: f32,
    pub duration_s: f32,
    /// Syllables per second.
    pub rate: f32,
    /// Fraction of the clip spent in pauses of 400 ms or more.
    pub pauses: f32,
    pub snr_db: f32,
    /// `natural`, `slow` or `fast` (against the voice's median rate).
    pub hint: String,
    pub recommended: bool,
    pub is_current: bool,
    pub manual: bool,
    pub preview_path: String,
}

impl Clip {
    fn from_json(v: &Value) -> Option<Clip> {
        let f = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        let b = |k: &str| v.get(k).and_then(Value::as_bool).unwrap_or(false);
        let st = |k: &str| v.get(k).and_then(Value::as_str).map(str::to_owned);
        Some(Clip {
            source: st("source").unwrap_or_else(|| "part".into()),
            file: v.get("file").and_then(Value::as_u64).map(|p| p as u32),
            part: v.get("part").and_then(Value::as_u64).map(|p| p as u32),
            title: st("title").filter(|t| !t.is_empty()),
            start_s: f("start_s"),
            end_s: f("end_s"),
            duration_s: f("duration_s"),
            rate: f("rate"),
            pauses: f("pauses"),
            snr_db: f("snr_db"),
            hint: st("hint").unwrap_or_else(|| "natural".into()),
            recommended: b("recommended"),
            is_current: b("is_current"),
            manual: b("manual"),
            preview_path: st("preview_path")?,
        })
    }

    pub fn is_media(&self) -> bool {
        self.source == "media"
    }
}

/// The combined media files of the voice (`voices.reference_candidates` -> `media`).
#[derive(Clone, Debug, Default)]
pub struct Pool {
    pub files: usize,
    pub duration_s: f32,
    /// `None`: its clips go to `reference.wav`; else to `reference_<language>.wav`.
    pub language: Option<String>,
}

impl Pool {
    fn from_json(v: &Value) -> Option<Pool> {
        v.is_object().then(|| Pool {
            files: v.get("files").and_then(Value::as_u64).unwrap_or(0) as usize,
            duration_s: v.get("duration_s").and_then(Value::as_f64).unwrap_or(0.0) as f32,
            language: v.get("language").and_then(Value::as_str).map(str::to_owned),
        })
    }
}

enum Kind {
    List(String),
    /// `voices.set_reference`; the language it wrote when it was a media clip for another language.
    Set(Option<String>),
    /// `presets.list {language}` for the sample text of a language reference.
    Sample(String),
    /// `recording.check` of a language reference take.
    Check(PathBuf),
    /// `voices.add_reference`.
    Add(String, String),
    /// `voices.remove_reference`.
    Remove(String, String),
}

/// Where adding a language reference stands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AddPhase {
    /// Showing the text to read: Record or Import.
    #[default]
    Ready,
    Recording,
    Checking,
    /// The quality check answered: Save, or record again.
    Checked,
    Saving,
}

/// Adding a reference clip for one language.
#[derive(Default)]
pub struct LangRef {
    /// The voice it is for.
    pub voice: String,
    pub lang: String,
    /// Text in that language to read aloud (about 10 s), when one is known.
    pub sample: Option<String>,
    pub phase: AddPhase,
    recorder: Option<Recorder>,
    seq: u64,
    pub level: f32,
    pub peak: f32,
    pub elapsed_s: f32,
    /// The recorded or imported take.
    pub take: Option<PathBuf>,
    pub verdict: Option<String>,
    pub issues: Vec<String>,
    pub duration_s: f32,
    pub error: Option<String>,
    importing: bool,
}

/// The extra sections under the reference clip, folded until asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum More {
    /// "Use other recordings": the reference from media files.
    Media,
    /// Reference clips in other languages.
    Languages,
}

#[derive(Default)]
pub struct RefState {
    pending: HashMap<u64, Kind>,
    /// The voice the data below belongs to.
    voice: Option<String>,
    /// The voice the last request was for (no request loop while the answer is missing).
    asked: Option<String>,
    /// Ask again when the next event settles (entering the screen, a clip was changed).
    stale: bool,
    pub current: Option<Clip>,
    pub candidates: Vec<Clip>,
    pub loading: bool,
    pub applying: bool,
    pub open: bool,
    /// Which of the card's extra sections is unfolded (one at a time, so the card stays short).
    pub more: Option<More>,
    pub error: Option<String>,
    /// A language reference being added (record / import, check, save).
    pub adding: Option<LangRef>,
    capture_seq: u64,
    /// The combined media files, when the voice has them.
    pub pool: Option<Pool>,
    /// "Reference from media" (see `refmedia`).
    pub media: super::refmedia::MediaState,
}

impl RefState {
    pub fn mark_stale(&mut self) {
        self.stale = true;
    }
}

// ---------------------------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------------------------

fn request_list(s: &mut AppState, id: &str) {
    s.voices.refclip.asked = Some(id.to_owned());
    s.voices.refclip.stale = false;
    match shell::request(s, "voices.reference_candidates", json!({ "id": id })) {
        Some(req) => {
            s.voices.refclip.loading = true;
            s.voices.refclip.pending.insert(req, Kind::List(id.to_owned()));
        }
        None => s.voices.refclip.loading = false,
    }
}

fn toggle_open(s: &mut AppState) {
    stop_preview(s);
    s.voices.refclip.open = !s.voices.refclip.open;
    s.voices.refclip.error = None;
}

/// Stops a clip preview started on this screen (the player bar plays something else on).
pub(super) fn stop_preview(s: &mut AppState) {
    playbar::stop_from(s, playbar::Source::Voices);
}

/// Whether the preview `path` is playing (its button shows "pause").
pub(super) fn previewing(s: &AppState, path: &str) -> bool {
    playbar::playing(s, Path::new(path))
}

/// What the player bar calls a preview: the candidate's place, a language's clip, a fresh take.
fn preview_title(s: &AppState, path: &str) -> String {
    if let Some(c) = s.voices.refclip.candidates.iter().find(|c| c.preview_path == path) {
        return where_line(c);
    }
    if let Some(code) = Path::new(path).file_stem().and_then(|n| n.to_str()).and_then(|n| n.strip_prefix("reference_")) {
        return format!("{} reference", lang::label(code));
    }
    "Reference clip".to_owned()
}

fn toggle_play(s: &mut AppState, path: &str) {
    let title = preview_title(s, path);
    let what = shell::active_voice(s).map(|v| format!("Voices · {}", v.name)).unwrap_or_else(|| "Voices".to_owned());
    if let Err(e) = playbar::toggle(s, Path::new(path), title, what, playbar::Source::Voices) {
        shell::toast(s, Tone::Danger, e);
    }
}

fn use_clip(s: &mut AppState, i: usize) {
    let (Some(id), Some(c)) = (s.shell.active_voice.clone(), s.voices.refclip.candidates.get(i).cloned()) else { return };
    if s.voices.refclip.applying {
        return;
    }
    let (params, lang_code) = if c.is_media() {
        let code = s.voices.refclip.pool.as_ref().and_then(|p| p.language.clone());
        (json!({ "id": id, "source": "media", "start_s": c.start_s, "end_s": c.end_s }), code)
    } else {
        let Some(part) = c.part else { return };
        (json!({ "id": id, "part": part, "start_s": c.start_s, "end_s": c.end_s }), None)
    };
    stop_preview(s);
    match shell::request(s, "voices.set_reference", params) {
        Some(req) => {
            s.voices.refclip.applying = true;
            s.voices.refclip.error = None;
            s.voices.refclip.pending.insert(req, Kind::Set(lang_code));
        }
        None => s.voices.refclip.error = Some("The voice engine is not running.".into()),
    }
}

// ---- per-language references ----

/// About 10 s of text in languages without reading presets (Latin and Cyrillic scripts; the app font draws them).
const SAMPLES: &[(&str, &str)] = &[
    ("it", "Stamattina sono uscito presto per una passeggiata lungo il fiume. L'aria era fresca e il cielo limpido, e ho pensato che fosse il momento giusto per registrare la mia voce."),
    ("pt", "Esta manhã saí cedo para caminhar junto ao rio. O ar estava fresco e o céu limpo, e pensei que era o momento certo para gravar a minha voz."),
    ("cs", "Dnes ráno jsem vyšel brzy na procházku podél řeky. Vzduch byl svěží a nebe jasné, a napadlo mě, že je to ta pravá chvíle nahrát svůj hlas."),
    ("ru", "Сегодня утром я рано вышел прогуляться вдоль реки. Воздух был свежий, небо ясное, и я подумал, что это подходящий момент записать свой голос."),
    ("nl", "Vanochtend ging ik vroeg wandelen langs de rivier. De lucht was fris en helder, en ik dacht dat het het juiste moment was om mijn stem op te nemen."),
    ("sk", "Dnes ráno som vyšiel skoro na prechádzku popri rieke. Vzduch bol svieži a obloha jasná, a napadlo mi, že je to vhodná chvíľa nahrať svoj hlas."),
    ("sv", "I morse gick jag ut tidigt på en promenad längs ån. Luften var frisk och himlen klar, och jag tänkte att det var rätt tillfälle att spela in min röst."),
    ("be", "Сёння раніцай я рана выйшаў пагуляць уздоўж ракі. Паветра было свежае, неба яснае, і я падумаў, што гэта добры момант запісаць свой голас."),
    ("bg", "Тази сутрин излязох рано на разходка покрай реката. Въздухът беше свеж, небето ясно, и си помислих, че е подходящ момент да запиша гласа си."),
    ("hr", "Jutros sam rano izašao u šetnju uz rijeku. Zrak je bio svjež, a nebo vedro, i pomislio sam da je pravi trenutak da snimim svoj glas."),
    ("tr", "Bu sabah erkenden nehir kıyısında yürüyüşe çıktım. Hava serin ve gökyüzü açıktı, sesimi kaydetmek için doğru an olduğunu düşündüm."),
    ("da", "I morges gik jeg tidligt en tur langs åen. Luften var frisk og himlen klar, og jeg tænkte, at det var det rette tidspunkt at optage min stemme."),
    ("fi", "Tänä aamuna lähdin aikaisin kävelylle joen rantaan. Ilma oli raikas ja taivas kirkas, ja ajattelin, että nyt on oikea hetki äänittää ääneni."),
    ("ro", "Azi-dimineață am ieșit devreme la o plimbare pe malul râului. Aerul era proaspăt și cerul senin, și m-am gândit că este momentul potrivit să îmi înregistrez vocea."),
    ("hu", "Ma reggel korán elindultam sétálni a folyó mentén. A levegő friss volt, az ég tiszta, és arra gondoltam, hogy most van itt a jó pillanat felvenni a hangomat."),
];

/// Languages with reading presets (`presets.list`).
const PRESET_LANGS: [&str; 7] = ["en", "uk", "de", "fr", "es", "pl", "ro"];

/// The first sentences of a reading text, about `secs` seconds of it (`est_s` is the whole text's reading time).
fn first_sentences(text: &str, est_s: f32, secs: f32) -> String {
    let words = text.split_whitespace().count().max(1) as f32;
    let per_word = if est_s > 0.0 { est_s / words } else { 0.4 };
    let mut out = String::new();
    for sentence in text.split_inclusive(['.', '!', '?']) {
        out.push_str(sentence);
        if out.split_whitespace().count() as f32 * per_word >= secs {
            break;
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Starts adding a reference for `code` (from the picker, or Speak's "Record a ... reference"): fetches the sample text.
pub fn start_language(s: &mut AppState, code: &str) {
    stop_preview(s);
    cancel_adding(s);
    s.voices.refclip.more = Some(More::Languages); // the panel the new clip is recorded in
    let code = lang::base(code);
    let sample = SAMPLES.iter().find(|(c, _)| *c == code).map(|(_, t)| (*t).to_owned());
    let voice = s.shell.active_voice.clone().unwrap_or_default();
    s.voices.refclip.adding = Some(LangRef { voice, lang: code.clone(), sample, ..Default::default() });
    s.voices.refclip.error = None;
    if PRESET_LANGS.contains(&code.as_str()) {
        if let Some(req) = shell::request(s, "presets.list", json!({ "language": code })) {
            s.voices.refclip.pending.insert(req, Kind::Sample(code));
        }
    }
}

/// Stops a running capture (the take is thrown away) and closes the panel.
fn cancel_adding(s: &mut AppState) {
    if let Some(mut a) = s.voices.refclip.adding.take() {
        if let Some(r) = a.recorder.take() {
            let _ = shell::guard(|| drop(r));
        }
    }
}

fn adding_tmp(s: &AppState, code: &str) -> PathBuf {
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis());
    let dir = s.shell.data_dir.join("tmp");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("reference-{code}-{stamp}.wav"))
}

fn toggle_capture(s: &mut AppState) {
    let recording = s.voices.refclip.adding.as_ref().is_some_and(|a| a.recorder.is_some());
    if recording { stop_capture(s) } else { start_capture(s) }
}

fn start_capture(s: &mut AppState) {
    stop_preview(s);
    s.voices.refclip.capture_seq += 1;
    let seq = s.voices.refclip.capture_seq;
    let device = s.record.input_device.clone();
    let started = shell::guard(|| Recorder::start(device.as_deref(), Some(super::record::meter_sink(seq, on_capture)))).map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string()));
    let Some(a) = s.voices.refclip.adding.as_mut() else { return };
    match started {
        Ok(rec) => {
            *a = LangRef { voice: a.voice.clone(), lang: a.lang.clone(), sample: a.sample.take(), phase: AddPhase::Recording, recorder: Some(rec), seq, ..Default::default() };
        }
        Err(e) => a.error = Some(format!("Cannot open the microphone: {e}")),
    }
}

fn stop_capture(s: &mut AppState) {
    let path = match s.voices.refclip.adding.as_ref() {
        Some(a) => adding_tmp(s, &a.lang),
        None => return,
    };
    let Some(a) = s.voices.refclip.adding.as_mut() else { return };
    let Some(rec) = a.recorder.take() else { return };
    match shell::guard(|| rec.stop(&path)) {
        Ok(Ok(info)) => {
            a.duration_s = info.duration_s;
            check_take(s, info.path);
        }
        Ok(Err(e)) => fail_adding(s, format!("Could not save the recording: {e}")),
        Err(e) => fail_adding(s, e),
    }
}

fn fail_adding(s: &mut AppState, msg: String) {
    if let Some(a) = s.voices.refclip.adding.as_mut() {
        a.phase = AddPhase::Ready;
        a.error = Some(msg);
    }
}

/// The take goes through the same quality check as a recorded part.
fn check_take(s: &mut AppState, path: PathBuf) {
    match shell::request(s, "recording.check", json!({ "path": path, "expected_s": 10.0 })) {
        Some(req) => {
            s.voices.refclip.pending.insert(req, Kind::Check(path.clone()));
            if let Some(a) = s.voices.refclip.adding.as_mut() {
                a.phase = AddPhase::Checking;
                a.take = Some(path);
                a.error = None;
            }
        }
        None => fail_adding(s, "The voice engine is not running.".into()),
    }
}

/// "Import file...": an existing recording in that language.
fn import_take(s: &mut AppState) {
    if s.shell.scripted() || s.voices.refclip.adding.as_ref().is_none_or(|a| a.importing || a.recorder.is_some()) {
        return;
    }
    if let Some(a) = s.voices.refclip.adding.as_mut() {
        a.importing = true;
    }
    let exts = super::formats::import_extensions(s);
    std::thread::spawn(move || {
        let choice = rfd::FileDialog::new().set_title("Choose a short recording in this language").add_filter("Audio and video", &exts).pick_file();
        super::events::post(move |s| {
            if let Some(a) = s.voices.refclip.adding.as_mut() {
                a.importing = false;
            }
            if let Some(path) = choice {
                check_take(s, path);
            }
        });
    });
}

fn save_take(s: &mut AppState) {
    let (Some(id), Some(a)) = (s.shell.active_voice.clone(), s.voices.refclip.adding.as_ref()) else { return };
    let (Some(path), code) = (a.take.clone(), a.lang.clone()) else { return };
    stop_preview(s);
    match shell::request(s, "voices.add_reference", json!({ "id": id, "language": code, "path": path })) {
        Some(req) => {
            s.voices.refclip.pending.insert(req, Kind::Add(id, code));
            if let Some(a) = s.voices.refclip.adding.as_mut() {
                a.phase = AddPhase::Saving;
            }
        }
        None => fail_adding(s, "The voice engine is not running.".into()),
    }
}

fn remove_lang(s: &mut AppState, code: &str) {
    let Some(id) = s.shell.active_voice.clone() else { return };
    stop_preview(s);
    if let Some(req) = shell::request(s, "voices.remove_reference", json!({ "id": id, "language": code })) {
        s.voices.refclip.pending.insert(req, Kind::Remove(id, code.to_owned()));
    }
}

/// The voice's languages with a reference, as the engine just reported them.
fn set_refs(s: &mut AppState, id: &str, v: &Value) {
    if let Some(list) = v.get("references").and_then(Value::as_array) {
        let refs: Vec<String> = list.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect();
        if let Some(info) = s.shell.voices.iter_mut().find(|x| x.id == id) {
            info.references = refs;
        }
    }
}

/// A level event of the reference capture.
fn on_capture(s: &mut AppState, seq: u64, ev: CaptureEvent) -> bool {
    let Some(a) = s.voices.refclip.adding.as_mut().filter(|a| a.seq == seq && a.recorder.is_some()) else { return false };
    match ev {
        CaptureEvent::Level { level, peak, elapsed_s } => {
            a.level = if level > a.level { level } else { a.level * 0.82 + level * 0.18 };
            a.peak = a.peak.max(peak);
            a.elapsed_s = elapsed_s;
            true
        }
        CaptureEvent::Error(_) => false,
    }
}

fn ref_path(s: &AppState, id: &str, code: &str) -> PathBuf {
    s.shell.data_dir.join("voices").join(id).join(format!("reference_{code}.wav"))
}

fn friendly(code: &str, message: &str) -> String {
    match code {
        "busy" => "A voice is being generated right now. Change the clip when it has finished.".into(),
        "not_found" => "That recording could not be found.".into(),
        _ if message.is_empty() => "The reference clip could not be changed.".into(),
        _ => message.to_owned(),
    }
}

// ---------------------------------------------------------------------------------------------
// Engine messages and events
// ---------------------------------------------------------------------------------------------

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    super::refmedia::on_engine(s, msg);
    match msg {
        EngineMsg::Response { id, result } => {
            let Some(kind) = s.voices.refclip.pending.remove(id) else { return };
            match (kind, result) {
                (Kind::List(voice), Ok(v)) => {
                    let r = &mut s.voices.refclip;
                    r.loading = false;
                    r.voice = Some(voice);
                    r.current = v.get("current").and_then(Clip::from_json);
                    r.candidates = v.get("candidates").and_then(Value::as_array).map(|a| a.iter().filter_map(Clip::from_json).collect()).unwrap_or_default();
                    r.pool = v.get("media").and_then(Pool::from_json);
                }
                (Kind::List(_), Err(_)) => s.voices.refclip.loading = false,
                (Kind::Set(code), Ok(_)) => {
                    let r = &mut s.voices.refclip;
                    r.applying = false;
                    r.open = false;
                    r.stale = true;
                    match code {
                        Some(c) => shell::toast(s, Tone::Success, format!("{} reference changed. Speech in {} now uses it.", lang::name(&c), lang::name(&c))),
                        None => shell::toast(s, Tone::Success, "Reference clip changed. The next speech uses its pace."),
                    }
                }
                (Kind::Set(..), Err(e)) => {
                    s.voices.refclip.applying = false;
                    s.voices.refclip.error = Some(friendly(&e.code, &e.message));
                }
                (Kind::Sample(code), Ok(v)) => {
                    let quick = v.get("presets").and_then(Value::as_array).and_then(|a| a.iter().find(|p| p.get("id").and_then(Value::as_str) == Some("quick")).or(a.first()).cloned());
                    let part = quick.as_ref().and_then(|p| p.get("parts")).and_then(Value::as_array).and_then(|a| a.first()).cloned();
                    if let (Some(a), Some(part)) = (s.voices.refclip.adding.as_mut().filter(|a| a.lang == code), part) {
                        let t = part.get("text").and_then(Value::as_str).unwrap_or("");
                        let est = part.get("est_seconds").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                        if !t.trim().is_empty() {
                            a.sample = Some(first_sentences(t, est, 10.0));
                        }
                    }
                }
                (Kind::Sample(_), Err(_)) => {}
                (Kind::Check(path), Ok(v)) => {
                    if let Some(a) = s.voices.refclip.adding.as_mut().filter(|a| a.take.as_ref() == Some(&path)) {
                        a.phase = AddPhase::Checked;
                        a.verdict = v.get("verdict").and_then(Value::as_str).map(str::to_owned);
                        a.issues = v.get("issues").and_then(Value::as_array).map(|x| x.iter().filter_map(|i| i.as_str().map(str::to_owned)).collect()).unwrap_or_default();
                        a.duration_s = v.get("duration_s").and_then(Value::as_f64).map_or(a.duration_s, |d| d as f32);
                    }
                }
                (Kind::Check(_), Err(e)) => fail_adding(s, friendly(&e.code, &e.message)),
                (Kind::Add(id, code), Ok(v)) => {
                    set_refs(s, &id, v);
                    s.voices.refclip.adding = None;
                    shell::toast(s, Tone::Success, format!("{} reference saved. Speech in {} now uses it.", lang::name(&code), lang::name(&code)));
                    shell::refresh_voices(s);
                }
                (Kind::Add(..), Err(e)) => {
                    let msg = friendly(&e.code, &e.message);
                    if let Some(a) = s.voices.refclip.adding.as_mut() {
                        a.phase = AddPhase::Checked;
                        a.error = Some(msg);
                    }
                }
                (Kind::Remove(id, code), Ok(v)) => {
                    set_refs(s, &id, v);
                    shell::toast(s, Tone::Neutral, format!("{} reference removed.", lang::name(&code)));
                }
                (Kind::Remove(..), Err(e)) => s.voices.refclip.error = Some(friendly(&e.code, &e.message)),
            }
        }
        EngineMsg::Exited { .. } => {
            let r = &mut s.voices.refclip;
            r.pending.clear();
            r.loading = false;
            r.applying = false;
            r.asked = None;
            if let Some(a) = r.adding.as_mut().filter(|a| matches!(a.phase, AddPhase::Checking | AddPhase::Saving)) {
                a.phase = AddPhase::Ready;
                a.error = Some("The voice engine stopped. Try again when it is back.".into());
            }
        }
        _ => {}
    }
}

/// After every event: (re)load the clips of the voice shown, close the picker when leaving the screen.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;
    if s.shell.scripted() {
        return changed;
    }
    changed |= super::refmedia::settle(s);
    if s.screen != Screen::Voices {
        // A preview keeps playing in the player bar.
        if s.voices.refclip.open {
            s.voices.refclip.open = false;
            changed = true;
        }
        // Leaving the screen while recording a language reference: the take is thrown away.
        if s.voices.refclip.adding.as_ref().is_some_and(|a| a.recorder.is_some()) {
            cancel_adding(s);
            changed = true;
        }
        return changed;
    }
    if let Some(id) = s.shell.active_voice.clone() {
        let other = s.voices.refclip.voice.as_deref() != Some(id.as_str());
        if s.voices.refclip.adding.as_ref().is_some_and(|a| a.voice != id) {
            cancel_adding(s);
            changed = true;
        }
        if other && s.voices.refclip.voice.is_some() {
            changed = true;
            stop_preview(s);
            let r = &mut s.voices.refclip;
            r.voice = None;
            r.current = None;
            r.candidates.clear();
            r.pool = None;
            r.open = false;
        }
        let want = s.voices.refclip.stale || s.voices.refclip.asked.as_deref() != Some(id.as_str());
        if want && !s.voices.refclip.loading && shell::engine_ready(s) {
            request_list(s, &id);
            changed = true;
        }
    }
    changed
}

// ---------------------------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------------------------

fn clock(secs: f32) -> String {
    let t = secs.max(0.0).round() as u32;
    format!("{}:{:02}", t / 60, t % 60)
}

fn pace_badge(p: &Palette, c: &Clip) -> El {
    let (label, tone) = match c.hint.as_str() {
        "slow" => ("Slow", Tone::Warning),
        "fast" => ("Fast", Tone::Accent),
        _ => ("Natural pace", Tone::Success),
    };
    kit::badge(p, label, tone)
}

fn pause_note(c: &Clip) -> &'static str {
    if c.pauses >= 0.12 {
        "many pauses"
    } else if c.pauses >= 0.05 {
        "some pauses"
    } else {
        "few pauses"
    }
}

fn where_line(c: &Clip) -> String {
    let span = format!("{}-{}", clock(c.start_s), clock(c.end_s));
    if c.is_media() {
        return match (c.file, c.title.as_deref()) {
            (Some(n), Some(t)) => format!("Media file {n} · {t}"),
            (Some(n), None) => format!("Media file {n}"),
            (None, Some(t)) => format!("Media · {t}"),
            (None, None) => "From your media".to_owned(),
        };
    }
    match (c.part, c.title.as_deref()) {
        (Some(n), Some(t)) => format!("Part {n} · {t}"),
        (Some(n), None) => format!("Part {n} · {span}"),
        (None, _) => "Imported clip".to_owned(),
    }
}

pub(super) fn play_button(p: &Palette, key: String, path: String, on: bool) -> El {
    let b = el().key(key).role("button").label(if on { "Pause" } else { "Play" }).size(36.0, 36.0).radius(18.0).center()
        .hover(rune::Color::WHITE.alpha(0.1)).focusable(true).motion(motion::UI)
        .on_press(move |s: &mut AppState| toggle_play(s, &path))
        .child(icon_el(if on { &icons::PAUSE } else { &icons::PLAY }, 15.0, p.on_accent));
    accent_glass(*p, b)
}

fn clip_row(s: &AppState, p: &Palette, key: &str, c: &Clip, action: Option<El>) -> El {
    let on = previewing(s, &c.preview_path);
    let mut badges = row().gap(8.0).center_y().flex_wrap().child(pace_badge(p, c));
    if c.recommended {
        badges = badges.child(kit::badge(p, "Recommended", Tone::Accent));
    }
    if c.manual {
        badges = badges.child(kit::badge(p, "Chosen by you", Tone::Neutral));
    }
    badges = badges.child(text(pause_note(c)).font(ty::CAPTION).color(p.text_dim));
    let span = if c.is_media() {
        format!("{:.0} s · SNR {:.0} dB", c.duration_s, c.snr_db)
    } else {
        format!("{}-{} · {:.0} s", clock(c.start_s), clock(c.end_s), c.duration_s)
    };
    let mut r = row().w_full().gap(10.0).center_y()
        .child(play_button(p, format!("{key}-play"), c.preview_path.clone(), on))
        .child(col().grow(1.0).min_w(0.0).gap(3.0)
            .child(text(where_line(c)).font(ty::LABEL).color(p.text).truncate())
            .child(text(span).font(ty::CAPTION).color(p.text_faint))
            .child(badges));
    if let Some(a) = action {
        r = r.child(a);
    }
    glass_card(*p).key(key.to_owned()).w_full().padding(10.0).child(r)
}

/// "Other languages": the languages with their own clip (play, remove), "Add a language..." and the add panel.
fn languages_block(s: &AppState, p: &Palette, info: &VoiceInfo) -> El {
    let r = &s.voices.refclip;
    let own = lang::name(info.lang());
    let busy = r.adding.as_ref().is_some_and(|a| a.phase != AddPhase::Ready && a.phase != AddPhase::Checked);
    let head = row().w_full().gap(10.0).center_y().justify_between()
        .child(col().gap(1.0).min_w(0.0).grow(1.0)
            .child(text(format!("Recorded in {own}. A short clip in another language makes speech in it sound native.")).font(ty::CAPTION).color(p.text_faint).wrap()))
        .child(lang::chip(p, "refclip-add-lang", "Add a language", lang::Target::Reference, !busy));
    let mut c = col().key("refclip-langs").w_full().gap(8.0).child(head);
    if !info.references.is_empty() {
        let mut pills = row().w_full().gap(8.0).flex_wrap();
        for code in &info.references {
            let path = ref_path(s, &info.id, code).to_string_lossy().into_owned();
            let on = previewing(s, &path);
            let (pp, rc) = (path.clone(), code.clone());
            pills = pills.child(
                row().key(format!("refclip-lang-{code}")).h(32.0).pl(4.0).pr(4.0).gap(6.0).center_y().radius(16.0).fill(p.glass).border(1.0, p.rim_top)
                    .child(el().key(format!("refclip-lang-play-{code}")).role("button").label(if on { "Pause" } else { "Play" }).size(24.0, 24.0).radius(12.0).center()
                        .fill(p.accent.alpha(0.25)).hover(p.accent.alpha(0.4)).focusable(true)
                        .on_press(move |s: &mut AppState| toggle_play(s, &pp))
                        .child(icon_el(if on { &icons::PAUSE } else { &icons::PLAY }, 11.0, p.text)))
                    .child(text(lang::label(code)).font(ty::LABEL).color(p.text))
                    .child(el().key(format!("refclip-lang-del-{code}")).role("button").label("Remove").help("Remove this reference").size(24.0, 24.0).radius(12.0).center()
                        .hover(p.glass_hover).focusable(true)
                        .on_press(move |s: &mut AppState| remove_lang(s, &rc))
                        .child(icon_el(&icons::CLOSE, 10.0, p.text_dim))),
            );
        }
        c = c.child(pills);
    }
    if let Some(a) = r.adding.as_ref().filter(|a| a.voice == info.id) {
        c = c.child(adding_panel(s, p, a));
    }
    c
}

fn verdict_badge(p: &Palette, v: Option<&str>) -> El {
    match v {
        Some("ok") => kit::badge(p, "Sounds good", Tone::Success),
        Some("warn") => kit::badge(p, "Usable", Tone::Warning),
        Some("bad") => kit::badge(p, "Not usable", Tone::Danger),
        _ => kit::badge(p, "Checked", Tone::Neutral),
    }
}

/// Recording (or importing) the clip for one language.
fn adding_panel(s: &AppState, p: &Palette, a: &LangRef) -> El {
    let name = lang::name(&a.lang);
    let head = row().w_full().gap(10.0).center_y()
        .child(text(format!("Reference for {}", lang::label(&a.lang))).font(ty::TITLE).color(p.text).grow(1.0).min_w(0.0).truncate())
        .child(Button::new("Cancel").ghost().small().key("refadd-cancel").on_press(cancel_adding).build(*p));
    let intro = match &a.sample {
        Some(_) => format!("Read this aloud in {name} at your natural pace (about 10 seconds):"),
        None => format!("Read two or three sentences aloud in {name} at your natural pace (about 10 seconds)."),
    };
    let mut c = col().key("refadd").w_full().gap(10.0).padding(12.0).radius(radius::CONTROL)
        .fill(p.accent.alpha(0.06)).border(1.0, p.accent.alpha(0.25))
        .child(head)
        .child(text(intro).font(ty::CAPTION).color(p.text_dim).wrap());
    if let Some(t) = &a.sample {
        c = c.child(col().key("refadd-sample").w_full().padding(12.0).radius(radius::CONTROL).fill(p.glass).border(1.0, p.rim_top.alpha(p.rim_top.a * 0.6))
            .child(text(t.clone()).font(ty::BODY).color(p.text).wrap()));
    }
    match a.phase {
        AddPhase::Recording => {
            c = c.child(row().w_full().gap(12.0).center_y()
                .child(Button::new("Stop").primary().small().icon(&icons::STOP).key("refadd-stop").on_press(toggle_capture).build(*p))
                .child(el().grow(1.0).min_w(0.0).child(level_meter(*p, a.level, Some(a.peak))))
                .child(text(clock(a.elapsed_s)).font(ty::NUM).color(if a.elapsed_s >= 8.0 { p.ok } else { p.text_dim })));
        }
        AddPhase::Checking | AddPhase::Saving => {
            let what = if a.phase == AddPhase::Checking { "Checking the recording…" } else { "Saving…" };
            c = c.child(row().gap(8.0).center_y().child(crate::ui::components::status_dot(*p, Tone::Accent, true)).child(text(what).font(ty::CAPTION).color(p.text_dim)));
        }
        AddPhase::Checked => {
            let take = a.take.as_ref().map(|t| t.to_string_lossy().into_owned()).unwrap_or_default();
            let on = previewing(s, &take);
            let mut line = row().w_full().gap(10.0).center_y()
                .child(play_button(p, "refadd-play".into(), take, on))
                .child(verdict_badge(p, a.verdict.as_deref()))
                .child(text(format!("{:.0} s", a.duration_s)).font(ty::CAPTION).color(p.text_faint))
                .child(el().grow(1.0));
            line = line.child(Button::new("Record again").small().icon(&icons::MIC).key("refadd-again").on_press(toggle_capture).build(*p));
            let ok = a.verdict.as_deref() != Some("bad");
            line = line.child(Button::new(format!("Use for {name}")).primary().small().key("refadd-save").disabled(!ok).on_press(save_take).build(*p));
            c = c.child(line);
            for i in &a.issues {
                c = c.child(text(format!("• {i}")).font(ty::CAPTION).color(if ok { p.text_dim } else { p.bad }).wrap());
            }
        }
        AddPhase::Ready => {
            c = c.child(row().w_full().gap(10.0).center_y()
                .child(Button::new("Record").primary().small().icon(&icons::MIC).key("refadd-record").on_press(toggle_capture).build(*p))
                .child(Button::new("Import file…").small().key("refadd-import").disabled(a.importing).on_press(import_take).build(*p))
                .child(text("Quiet room, normal distance to the mic.").font(ty::CAPTION).color(p.text_faint)));
        }
    }
    if let Some(e) = &a.error {
        c = c.child(kit::notice(p, Tone::Danger, e));
    }
    c
}

/// The heading above the media clips in the "Choose..." list.
fn media_heading(p: &Palette, pool: Option<&Pool>, after_parts: bool) -> El {
    let mut line = "From your media".to_owned();
    if let Some(pool) = pool {
        line.push_str(&format!(" · {} file{} · {}", pool.files, if pool.files == 1 { "" } else { "s" }, clock(pool.duration_s)));
        if let Some(code) = &pool.language {
            line.push_str(&format!(" · for {}", lang::name(code)));
        }
    }
    let h = row().key("refclip-media-head").w_full().gap(8.0).center_y()
        .child(icon_el(&icons::FOLDER, 12.0, p.text_faint))
        .child(text(line.to_uppercase()).font(ty::OVERLINE).color(p.text_faint).truncate());
    if after_parts { h.pt(6.0) } else { h }
}

/// The detected language of `reference.wav` ("Ukrainian speech") and, when it is not the voice's own language and
/// that language has no clip of its own, what that means for speech in it.
fn language_line(p: &Palette, info: &VoiceInfo, zero_shot_vc: bool) -> El {
    let Some(code) = info.reference_language.as_deref() else {
        return row().key("refclip-spoken").w_full().child(
            text("Spoken language: not checked yet (the next time this voice is prepared or speaks).").font(ty::CAPTION).color(p.text_faint).wrap(),
        );
    };
    let mismatch = info.reference_mismatch() && !info.references.iter().any(|r| lang::base(r) == lang::base(info.lang()));
    let head = row().key("refclip-spoken").w_full().gap(8.0).center_y()
        .child(kit::badge(p, &format!("{} speech", lang::name(code)), if mismatch { Tone::Warning } else { Tone::Neutral }))
        .child(text("detected in this clip").font(ty::CAPTION).color(p.text_faint));
    if !mismatch {
        return head;
    }
    let own = lang::name(info.lang());
    let note = if lang::base(info.lang()) == "en" {
        let fix = if info.trained || zero_shot_vc { "or choose Native accent when it speaks" } else { "or train the voice for a native accent" };
        format!("English will carry this clip's accent. Add an English clip under Other languages, {fix}.")
    } else {
        format!("{own} will carry this clip's accent. Add {} {own} clip under Other languages.", if own.starts_with(['A', 'E', 'I', 'O', 'U']) { "an" } else { "a" })
    };
    let mut c = col().w_full().gap(6.0).child(head).child(text(note).key("refclip-spoken-note").font(ty::CAPTION).color(p.warn).wrap());
    // The voice's language says what it was recorded in: when the clip is clearly another language, offer to fix it
    // (it decides which output counts as foreign: the accent choice, and how strongly the voice model applies).
    if lang::base(code) != lang::base(info.lang()) {
        let (id, to) = (info.id.clone(), lang::base(code));
        let label = format!("Set the voice to {}", lang::name(&to));
        c = c.child(row().key("refclip-set-lang").gap(10.0).center_y()
            .child(kit::button(p, &label, BtnKind::Glass, true, move |s: &mut AppState| set_voice_language(s, &id, &to)))
            .child(text("Changes only how languages are treated; recordings and training stay as they are.").font(ty::CAPTION).color(p.text_faint).wrap().grow(1.0).min_w(0.0)));
    }
    c
}

/// "Set the voice to <language>": the recording language follows the detected language of its clip.
fn set_voice_language(s: &mut AppState, id: &str, code: &str) {
    if let Some(v) = s.shell.voices.iter_mut().find(|v| v.id == id) {
        v.language = code.to_owned();
    }
    if shell::request(s, "voices.rename", json!({ "id": id, "language": code })).is_some() {
        shell::refresh_voices(s);
        shell::toast(s, Tone::Success, &format!("The voice is now set to {}.", lang::name(code)));
    } else {
        shell::toast(s, Tone::Danger, "The engine is not running: try again once it is ready.");
    }
}

/// The section shown inside the active voice's card.
pub fn section(s: &AppState, p: &Palette, info: &VoiceInfo) -> El {
    let r = &s.voices.refclip;
    let head_action = if r.candidates.is_empty() && !r.open {
        None
    } else if r.open {
        Some(kit::button(p, "Close", BtnKind::Ghost, true, toggle_open))
    } else {
        Some(kit::button(p, "Choose…", BtnKind::Glass, !r.loading, toggle_open))
    };
    let mut head = row().w_full().gap(10.0).center_y().justify_between()
        .child(col().gap(1.0)
            .child(text("Reference clip").font(ty::LABEL).color(p.text))
            .child(text("The pace and manner of your cloned voice are copied from this clip.").font(ty::CAPTION).color(p.text_faint)));
    if let Some(a) = head_action {
        head = head.child(a);
    }

    let mut c = col().key("refclip").w_full().gap(10.0).child(head);
    match &r.current {
        Some(cur) => c = c.child(clip_row(s, p, "refclip-current", cur, None)).child(language_line(p, info, s.shell.zero_shot_vc)),
        None if r.loading => c = c.child(text("Checking the clip…").font(ty::CAPTION).color(p.text_dim)),
        None => c = c.child(text("No reference clip yet. Record this voice, or add media files of it below.").font(ty::CAPTION).color(p.text_dim)),
    }
    if let Some(e) = &r.error {
        c = c.child(kit::notice(p, Tone::Danger, e));
    }
    if r.open {
        if r.candidates.is_empty() {
            c = c.child(text("No other clips to choose from yet.").font(ty::CAPTION).color(p.text_dim));
        } else {
            let tips = "Pick one that sounds like your natural, relaxed reading. A slow clip makes the voice slow.";
            c = c.child(text(tips).font(ty::CAPTION).color(p.text_dim).wrap());
            let mut list = col().key("refclip-list").w_full().gap(8.0);
            let parts = r.candidates.iter().any(|c| !c.is_media());
            let mut media_head = false;
            for (i, cand) in r.candidates.iter().enumerate() {
                if cand.is_media() && !media_head {
                    media_head = true;
                    list = list.child(media_heading(p, r.pool.as_ref(), parts));
                }
                let action = if cand.is_current {
                    kit::badge(p, "In use", Tone::Success)
                } else {
                    kit::button(p, if r.applying { "Switching…" } else { "Use this one" }, BtnKind::Primary, !r.applying,
                        move |s: &mut AppState| use_clip(s, i))
                };
                list = list.child(clip_row(s, p, &format!("refclip-c{i}"), cand, Some(action)));
            }
            c = c.child(list);
        }
    }
    // The two extra sections are one line each: all of it open at once was more than anyone could take in. A press
    // opens the section in a panel over the page ([`more_overlay`]), so the card and the voices under it do not
    // move. The line says what is going on in a section that is closed.
    let media = &r.media;
    let mine = media.voice.as_deref() == Some(info.id.as_str());
    let media_note = if mine && media.busy() {
        "Working on your files…"
    } else if mine && media.error.is_some() {
        "The files could not be used: open to see why"
    } else if mine && media.outcome.is_some() {
        "A new reference is ready: open to listen"
    } else if r.pool.is_some() {
        "Combined files are in use"
    } else {
        "Combine files of the same speaker into a cleaner reference"
    };
    c = c.child(el().w_full().h(1.0).fill(p.well_rim)).child(more_row(p, "more-media", "Use other recordings", media_note, media_open(s), More::Media));
    if r.current.is_some() || !info.references.is_empty() || r.adding.is_some() {
        let have: Vec<String> = info.references.iter().map(|c| super::lang::label(c)).collect();
        let note = match &r.adding {
            Some(a) => format!("Adding {}: open to go on", super::lang::label(&a.lang)),
            None if have.is_empty() => "A short clip in another language makes speech in it sound native".to_owned(),
            None => have.join(", "),
        };
        c = c.child(el().w_full().h(1.0).fill(p.well_rim)).child(more_row(p, "more-langs", "Other languages", &note, r.more == Some(More::Languages), More::Languages));
    }
    el().w_full().padding(12.0).radius(radius::CONTROL).fill(p.well).border(1.0, p.well_rim).child(c)
}

/// Whether "Use other recordings" is open: asked for, or files are being dragged over the Voices screen.
fn media_open(s: &AppState) -> bool {
    s.voices.refclip.more == Some(More::Media) || (s.formats.over && s.screen == Screen::Voices)
}

/// The line of an extra section: its name, what it holds, and "Open". A press opens it in a panel over the page.
fn more_row(p: &Palette, key: &str, title: &str, note: &str, open: bool, which: More) -> El {
    row().key(key.to_owned()).role("button").label(title.to_owned()).w_full().gap(10.0).center_y().px(6.0).py(7.0).radius(radius::CONTROL)
        .fill(if open { p.glass_hover } else { rune::Color::TRANSPARENT })
        .hover(p.glass_hover).focusable(true)
        .on_press(move |s: &mut AppState| {
            let r = &mut s.voices.refclip;
            r.more = if r.more == Some(which) { None } else { Some(which) };
        })
        .child(text(title.to_owned()).font(ty::LABEL).color(p.text).a11y_hidden())
        .child(text(note.to_owned()).font(ty::CAPTION).color(p.text_faint).grow(1.0).min_w(0.0).truncate().a11y_hidden())
        .child(text("Open").font(ty::CAPTION).color(p.accent).a11y_hidden())
}

/// The open extra section of the active voice, in a panel over the page under its line (added at the root by the
/// shell). `None` when neither is open.
pub fn more_overlay(s: &AppState, p: &Palette) -> Option<El> {
    if s.screen != Screen::Voices {
        return None;
    }
    let r = &s.voices.refclip;
    let info = shell::active_voice(s)?;
    let which = if media_open(s) { More::Media } else { r.more? };
    let (anchor, title, body) = match which {
        More::Media => ("more-media", "Use other recordings", super::refmedia::block(s, p, info, r.pool.is_some())),
        More::Languages => ("more-langs", "Other languages", languages_block(s, p, info)),
    };
    let content = col().w_full().gap(12.0)
        .child(row().w_full().center_y().justify_between()
            .child(text(title).font(ty::TITLE).color(p.text))
            .child(kit::button(p, "Done", BtnKind::Glass, true, close_more)))
        .child(body);
    Some(crate::ui::components::popover(*p, "voice-more", anchor, 640.0, rune::Align::Start, close_more, content))
}

/// Closes the open section's panel. A clip that is being recorded keeps it open: the recording would go on unseen.
fn close_more(s: &mut AppState) {
    let r = &mut s.voices.refclip;
    if r.more == Some(More::Languages) && r.adding.as_ref().is_some_and(|a| a.recorder.is_some()) {
        return;
    }
    r.more = None;
}

// ---------------------------------------------------------------------------------------------
// Scenes for headless snapshots
// ---------------------------------------------------------------------------------------------

fn clip(part: u32, title: &str, start: f32, end: f32, rate: f32, pauses: f32, hint: &str) -> Clip {
    Clip {
        part: Some(part), title: Some(title.into()), start_s: start, end_s: end, duration_s: end - start, rate, pauses, snr_db: 38.0,
        hint: hint.into(), preview_path: format!("/tmp/demo/part{part}.wav"), ..Default::default()
    }
}

fn demo(s: &mut AppState) {
    super::voices::scene_list(s);
    let own = s.shell.voices[0].lang().to_owned();
    s.shell.voices[0].reference_language = Some(own);
    let r = &mut s.voices.refclip;
    r.voice = Some("my-voice".into());
    let mut current = clip(5, "Questions, questions", 31.6, 50.9, 3.4, 0.17, "slow");
    current.preview_path = "/tmp/demo/reference.wav".into();
    r.current = Some(current);
    let mut best = clip(4, "Two friends, one argument", 8.2, 26.4, 4.7, 0.03, "natural");
    best.recommended = true;
    let mut cur = clip(5, "Questions, questions", 31.6, 50.9, 3.4, 0.17, "slow");
    cur.is_current = true;
    r.candidates = vec![
        best,
        clip(3, "The lighthouse keeper", 14.0, 31.5, 4.5, 0.05, "natural"),
        clip(1, "A slow Sunday morning", 40.1, 58.0, 4.2, 0.08, "natural"),
        clip(8, "Cooking dinner", 3.0, 21.9, 5.6, 0.02, "fast"),
        cur,
    ];
}

fn scene_current(s: &mut AppState) {
    demo(s);
}

/// The reference section of the demo voice (for `refmedia`'s scenes).
pub(super) fn scene_demo(s: &mut AppState) {
    demo(s);
}

/// The demo voice after combining media: the current clip comes from file 3, the list shows media clips too.
pub(super) fn scene_demo_media(s: &mut AppState) {
    demo(s);
    let r = &mut s.voices.refclip;
    let media = |file: u32, title: &str, start: f32, end: f32, rate: f32, snr: f32, hint: &str| Clip {
        source: "media".into(), file: Some(file), title: Some(title.into()), start_s: start, end_s: end, duration_s: end - start, rate, pauses: 0.0,
        snr_db: snr, hint: hint.into(), preview_path: format!("/tmp/demo/cand_media{file}.wav"), ..Default::default()
    };
    let mut cur = media(3, "podcast-ep-14.m4a", 131.2, 142.6, 4.6, 41.0, "natural");
    cur.manual = true;
    cur.preview_path = "/tmp/demo/reference.wav".into();
    r.current = Some(cur);
    let mut best = media(3, "podcast-ep-14.m4a", 131.2, 142.6, 4.6, 41.0, "natural");
    best.recommended = true;
    best.is_current = true;
    r.candidates.truncate(2);
    r.candidates.extend([
        best,
        media(7, "conference-talk-2024.mp4", 288.0, 298.4, 4.8, 37.5, "natural"),
        media(1, "interview-radio.mp3", 12.5, 23.9, 5.4, 33.0, "fast"),
    ]);
    r.pool = Some(Pool { files: 12, duration_s: 452.0, language: None });
    r.open = true;
}

fn scene_choose(s: &mut AppState) {
    demo(s);
    s.voices.refclip.open = true;
    playbar::scene_now(s, "/tmp/demo/part3.wav", "Part 3 · The lighthouse keeper", "Voices · My voice", playbar::Source::Voices, playbar::Status::Playing, 6.0, 19.0);
}

fn scene_choose_busy(s: &mut AppState) {
    demo(s);
    s.voices.refclip.open = true;
    s.voices.refclip.error = Some(friendly("busy", ""));
}

fn scene_loading(s: &mut AppState) {
    super::voices::scene_list(s);
    s.voices.refclip.loading = true;
}

fn scene_none(s: &mut AppState) {
    super::voices::scene_list(s);
}

/// References in Ukrainian and German, and a Polish one checked and ready to save.
fn scene_langs(s: &mut AppState) {
    demo(s);
    s.voices.refclip.more = Some(More::Languages);
    s.shell.voices[0].references = vec!["uk".into(), "de".into()];
    s.voices.refclip.adding = Some(LangRef {
        voice: "my-voice".into(),
        lang: "pl".into(),
        sample: Some("Wczoraj wieczorem usiadłem przy oknie z kubkiem herbaty i patrzyłem, jak deszcz spływa po szybie. Na ulicy było cicho, tylko czasem przejeżdżał tramwaj.".into()),
        phase: AddPhase::Checked,
        take: Some("/tmp/demo/reference-pl.wav".into()),
        verdict: Some("ok".into()),
        duration_s: 10.4,
        ..Default::default()
    });
}

/// Recording a Ukrainian reference (live meter).
fn scene_langs_recording(s: &mut AppState) {
    demo(s);
    s.voices.refclip.more = Some(More::Languages);
    s.shell.voices[0].references = vec!["de".into()];
    s.voices.refclip.adding = Some(LangRef {
        voice: "my-voice".into(),
        lang: "uk".into(),
        sample: Some("Сьогодні вранці я вийшов на прогулянку вздовж річки. Повітря було свіже, небо ясне, і я подумав, що це слушна мить записати свій голос.".into()),
        phase: AddPhase::Recording,
        level: 0.55,
        peak: 0.7,
        elapsed_s: 6.4,
        ..Default::default()
    });
}

/// The language picker open on "Add a language".
fn scene_langs_picker(s: &mut AppState) {
    demo(s);
    s.voices.refclip.more = Some(More::Languages);
    s.shell.voices[0].references = vec!["uk".into()];
    s.langpick.open = Some((lang::Target::Reference, "refclip-add-lang".into()));
}

/// The owner's case: a voice set to English whose clip is Ukrainian speech (untrained): the warning note.
fn scene_spoken_mismatch(s: &mut AppState) {
    demo(s);
    s.shell.voices[0].language = "en".into();
    s.shell.voices[0].trained = false;
    s.shell.voices[0].reference_language = Some("uk".into());
    s.shell.voices[0].references = vec!["uk".into()];
}

/// The clip's language matches the voice's: just the badge.
fn scene_spoken_match(s: &mut AppState) {
    demo(s);
    s.shell.voices[0].reference_language = Some("en".into());
}

/// A new clip the speech worker has not transcribed yet.
fn scene_spoken_unknown(s: &mut AppState) {
    demo(s);
    s.shell.voices[0].reference_language = None;
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("voices_reference_spoken_mismatch", scene_spoken_mismatch),
        ("voices_reference_spoken_match", scene_spoken_match),
        ("voices_reference_spoken_unknown", scene_spoken_unknown),
        ("voices-reference", scene_current),
        ("voices-reference-choose", scene_choose),
        ("voices-reference-error", scene_choose_busy),
        ("voices-reference-loading", scene_loading),
        ("voices-reference-none", scene_none),
        ("voices_reference_langs", scene_langs),
        ("voices_reference_langs_recording", scene_langs_recording),
        ("voices_reference_langs_picker", scene_langs_picker),
    ]
}
