//! "Reference from media" block of the active voice's reference section (Voices screen).
//!
//! The user picks (or drops onto the Voices screen) many audio and video files of the same speaker. The engine
//! combines them into one cleaned pool (`<voice>/reference_media.wav`, at most 20 minutes) and makes its best 8-12 s
//! window the voice's reference (`voices.reference_from_media`, a job: `voices.media.progress` / `.done` / `.error`).
//! The language chip says which reference it becomes: the voice's own language writes `reference.wav`, another one
//! `reference_<lang>.wav`. The pool's other good windows then show up in the "Choose..." list of the reference section
//! (`source: "media"` candidates), so the user can listen and pick another one.
//!
//! "Remove background music" sends `remove_music`: the engine then keeps only the vocals of each file (Hybrid Demucs)
//! before cleaning it. The choice is remembered as `remove_music` in settings.json; until the user makes one it is on
//! when the engine runs on a CUDA GPU. Record's multi-file import sends the remembered choice too (`media.combine`).

use std::path::PathBuf;

use rune::{El, col, el, row, text};
use serde_json::{Value, json};

use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use crate::ui::components::{Button, Tone, icon_el, icons, progress_bar, status_dot};
use crate::ui::lang;
use crate::ui::shell::{self, VoiceInfo};
use crate::ui::theme::{Palette, motion, radius, ty};
use crate::ui::voices::kit;

/// A file the engine could not use, and why.
#[derive(Clone, Debug, Default)]
pub struct Skipped {
    pub name: String,
    pub reason: String,
}

/// What a finished job reported.
#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub files: usize,
    pub duration_s: f32,
    /// `None`: the voice's own language (`reference.wav`).
    pub language: Option<String>,
    pub applied: bool,
    pub ref_duration_s: f32,
    pub ref_file: Option<u32>,
    pub ref_title: Option<String>,
    pub preview_path: Option<String>,
    pub message: Option<String>,
    pub skipped: Vec<Skipped>,
    pub truncated: Vec<String>,
    /// Files whose background music was removed (`music_files`), and why not when none was (`music_reason`).
    pub music_files: usize,
    pub music_reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum Phase {
    #[default]
    Idle,
    /// `voices.reference_from_media` sent, waiting for the job id.
    Starting,
    /// The job runs.
    Running,
}

#[derive(Default)]
pub struct MediaState {
    /// The voice the job / result below belongs to.
    pub voice: Option<String>,
    /// The chosen language (`None`: the voice's own).
    pub lang: Option<String>,
    pub phase: Phase,
    req: Option<u64>,
    clear_req: Option<u64>,
    pub job: Option<String>,
    pub picking: bool,
    pub cancelling: bool,
    /// Files sent, file being processed (1-based), its name, stage and overall percent.
    pub files: usize,
    pub file: usize,
    pub name: String,
    pub stage: String,
    pub pct: f32,
    pub outcome: Option<Outcome>,
    pub error: Option<String>,
    pub skipped: Vec<Skipped>,
    /// "Remove background music" as the user last set it (`None`: never set, see [`remove_music`]).
    pub remove_music: Option<bool>,
    prefs_loaded: bool,
}

impl MediaState {
    pub fn busy(&self) -> bool {
        self.phase != Phase::Idle
    }
}

fn m(s: &mut AppState) -> &mut MediaState {
    &mut s.voices.refclip.media
}

// ---------------------------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------------------------

/// The language the media reference is for (the chip's check mark): the chosen one, else the voice's own.
pub fn language(s: &AppState) -> String {
    s.voices.refclip.media.lang.clone().unwrap_or_else(|| shell::active_voice(s).map(|v| v.lang().to_owned()).unwrap_or_else(|| "en".into()))
}

/// The language picker answered.
pub fn set_language(s: &mut AppState, code: &str) {
    let own = shell::active_voice(s).map(|v| lang::base(v.lang()));
    let code = lang::base(code);
    m(s).lang = if own.as_deref() == Some(code.as_str()) { None } else { Some(code) };
}

/// "Remove background music": the remembered choice, else on when the engine has a CUDA GPU.
pub fn remove_music(s: &AppState) -> bool {
    super::setuprun::has(s, "music_removal") && s.voices.refclip.media.remove_music.unwrap_or(s.shell.gpu)
}

/// The remembered choice only (`None` when the user never set it: the engine then decides, `"auto"`).
pub fn saved_remove_music() -> Option<bool> {
    super::onboarding::setting("remove_music").and_then(|v| v.as_bool())
}

fn toggle_music(s: &mut AppState) {
    let on = !remove_music(s);
    m(s).remove_music = Some(on);
    if !s.shell.scripted() {
        super::onboarding::save_setting("remove_music", json!(on));
    }
}

/// "Add media...": the native multi-file picker on its own thread; the choice comes back as an event.
pub fn pick_files(s: &mut AppState) {
    if s.shell.scripted() || m(s).picking || m(s).busy() {
        return;
    }
    m(s).picking = true;
    let exts = super::formats::import_extensions(s);
    std::thread::spawn(move || {
        let choice = rfd::FileDialog::new().set_title("Choose recordings or videos of the same speaker").add_filter("Audio and video", &exts).pick_files();
        super::events::post(move |s| {
            m(s).picking = false;
            if let Some(paths) = choice.filter(|p| !p.is_empty()) {
                start(s, paths);
            }
        });
    });
}

/// Files dropped on the Voices screen.
pub fn drop_files(s: &mut AppState, files: &[PathBuf]) {
    if files.is_empty() {
        return;
    }
    if shell::active_voice(s).is_none() {
        shell::toast(s, Tone::Warning, "Create or pick a voice first, then drop its media files.");
        return;
    }
    start(s, files.to_vec());
}

fn start(s: &mut AppState, paths: Vec<PathBuf>) {
    let Some(id) = s.shell.active_voice.clone() else { return };
    if m(s).busy() {
        shell::toast(s, Tone::Warning, "Media files are already being combined. Wait for them or press Cancel.");
        return;
    }
    super::refclip::stop_preview(s);
    let mut params = json!({ "id": id, "paths": paths, "remove_music": remove_music(s) });
    if let Some(code) = m(s).lang.clone() {
        params["language"] = json!(code);
    }
    let n = paths.len();
    match shell::request(s, "voices.reference_from_media", params) {
        Some(req) => {
            let st = m(s);
            *st = MediaState {
                voice: Some(id), lang: st.lang.clone(), phase: Phase::Starting, req: Some(req), files: n, stage: "start".into(),
                remove_music: st.remove_music, prefs_loaded: st.prefs_loaded, ..Default::default()
            };
        }
        None => m(s).error = Some("The voice engine is not running. Check the status at the top and press Retry.".into()),
    }
}

fn cancel(s: &mut AppState) {
    let Some(job) = m(s).job.clone() else { return };
    if shell::request(s, "voices.media_cancel", json!({ "job": job })).is_some() {
        m(s).cancelling = true;
    }
}

fn dismiss(s: &mut AppState) {
    super::refclip::stop_preview(s);
    let st = m(s);
    st.outcome = None;
    st.error = None;
    st.skipped.clear();
}

/// "Remove combined media": deletes the pool and its candidates (references cut from it stay).
fn clear_pool(s: &mut AppState) {
    let Some(id) = s.shell.active_voice.clone() else { return };
    super::refclip::stop_preview(s);
    m(s).clear_req = shell::request(s, "voices.clear_media", json!({ "id": id }));
}

fn skipped_list(v: &Value) -> Vec<Skipped> {
    v.get("skipped").and_then(Value::as_array).map(|a| a.iter().map(|x| Skipped {
        name: x.get("name").and_then(Value::as_str).unwrap_or("").to_owned(),
        reason: x.get("reason").and_then(Value::as_str).unwrap_or("").to_owned(),
    }).collect()).unwrap_or_default()
}

fn outcome(v: &Value) -> Outcome {
    let f = |k: &str| v.get(k).and_then(Value::as_f64).unwrap_or(0.0) as f32;
    let r = v.get("reference").filter(|r| r.is_object());
    Outcome {
        files: v.get("files").and_then(Value::as_u64).unwrap_or(0) as usize,
        duration_s: f("duration_s"),
        language: v.get("language").and_then(Value::as_str).map(str::to_owned),
        applied: v.get("applied").and_then(Value::as_bool).unwrap_or(false),
        ref_duration_s: r.and_then(|r| r.get("duration_s")).and_then(Value::as_f64).unwrap_or(0.0) as f32,
        ref_file: r.and_then(|r| r.get("file")).and_then(Value::as_u64).map(|n| n as u32),
        ref_title: r.and_then(|r| r.get("title")).and_then(Value::as_str).map(str::to_owned),
        preview_path: v.get("preview_path").and_then(Value::as_str).map(str::to_owned),
        message: v.get("message").and_then(Value::as_str).map(str::to_owned),
        skipped: skipped_list(v),
        truncated: v.get("truncated").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
        music_files: v.get("music_files").and_then(Value::as_u64).unwrap_or(0) as usize,
        music_reason: v.get("music_reason").and_then(Value::as_str).filter(|r| !r.is_empty()).map(str::to_owned),
    }
}

fn friendly(code: &str, message: &str) -> String {
    match code {
        "busy" if message.contains("generation") => "A voice is being generated right now. Add media when it has finished.".into(),
        "not_found" => "That voice could not be found.".into(),
        _ if message.is_empty() => "The media files could not be combined.".into(),
        _ => message.to_owned(),
    }
}

// ---------------------------------------------------------------------------------------------
// Engine messages
// ---------------------------------------------------------------------------------------------

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Response { id, result } if m(s).req == Some(*id) => {
            let st = m(s);
            st.req = None;
            match result {
                Ok(v) => {
                    st.job = v.get("job").and_then(Value::as_str).map(str::to_owned);
                    st.phase = if st.job.is_some() { Phase::Running } else { Phase::Idle };
                }
                Err(e) => {
                    st.phase = Phase::Idle;
                    st.error = Some(friendly(&e.code, &e.message));
                }
            }
        }
        EngineMsg::Response { id, result } if m(s).clear_req == Some(*id) => {
            m(s).clear_req = None;
            match result {
                Ok(_) => {
                    dismiss(s);
                    s.voices.refclip.mark_stale();
                    shell::toast(s, Tone::Neutral, "Combined media removed. The reference clip in use stays.");
                }
                Err(e) => shell::toast(s, Tone::Danger, friendly(&e.code, &e.message)),
            }
        }
        EngineMsg::Event(ev) if ev.job.is_some() && ev.job == m(s).job => match ev.event.as_str() {
            "voices.media.progress" => {
                let d = &ev.data;
                let st = m(s);
                st.pct = d.get("pct").and_then(Value::as_f64).unwrap_or(0.0) as f32;
                st.stage = d.get("stage").and_then(Value::as_str).unwrap_or("").to_owned();
                if let Some(k) = d.get("file").and_then(Value::as_u64) {
                    st.file = k as usize;
                    st.name = d.get("name").and_then(Value::as_str).unwrap_or("").to_owned();
                }
                if let Some(n) = d.get("files").and_then(Value::as_u64) {
                    st.files = n as usize;
                }
            }
            "voices.media.done" => {
                let o = outcome(&ev.data);
                let st = m(s);
                st.phase = Phase::Idle;
                st.job = None;
                st.cancelling = false;
                let (applied, lang_code) = (o.applied, o.language.clone());
                st.outcome = Some(o);
                s.voices.refclip.mark_stale();
                if lang_code.is_some() {
                    shell::refresh_voices(s);
                }
                if applied {
                    let what = lang_code.map_or_else(|| "Reference clip".to_owned(), |c| format!("{} reference", lang::name(&c)));
                    shell::toast(s, Tone::Success, format!("{what} set from your media. The next speech uses it."));
                }
            }
            "voices.media.error" => {
                let code = ev.data.get("code").and_then(Value::as_str).unwrap_or("internal").to_owned();
                let message = ev.data.get("message").and_then(Value::as_str).unwrap_or("").to_owned();
                let skipped = skipped_list(&ev.data);
                let st = m(s);
                st.phase = Phase::Idle;
                st.job = None;
                st.cancelling = false;
                st.skipped = skipped;
                if code == "cancelled" {
                    shell::toast(s, Tone::Neutral, "Cancelled. Nothing was changed.");
                } else {
                    m(s).error = Some(friendly(&code, &message));
                }
            }
            _ => {}
        },
        EngineMsg::Exited { .. } => {
            let st = m(s);
            st.req = None;
            st.clear_req = None;
            if st.busy() {
                st.phase = Phase::Idle;
                st.job = None;
                st.cancelling = false;
                st.error = Some("The voice engine stopped. Try again when it is back.".into());
            }
        }
        _ => {}
    }
}

/// After every event: the saved "Remove background music" once; a result of another voice is not shown for this one.
pub fn settle(s: &mut AppState) -> bool {
    if !m(s).prefs_loaded && !s.shell.scripted() {
        let saved = saved_remove_music();
        let st = m(s);
        st.prefs_loaded = true;
        st.remove_music = saved;
    }
    let active = s.shell.active_voice.clone();
    let st = m(s);
    if !st.busy() && st.voice.is_some() && st.voice != active && (st.outcome.is_some() || st.error.is_some()) {
        st.outcome = None;
        st.error = None;
        st.skipped.clear();
        st.lang = None;
        st.voice = active;
        return true;
    }
    false
}

// ---------------------------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------------------------

fn clock(secs: f32) -> String {
    let t = secs.max(0.0).round() as u32;
    format!("{}:{:02}", t / 60, t % 60)
}

fn stage_label(stage: &str) -> &'static str {
    match stage {
        "decode" => "Cleaning",
        "separate" => "Removing background music",
        "combine" => "Combining",
        "analyse" => "Finding the best clip",
        "apply" => "Saving the reference",
        _ => "Starting",
    }
}

fn skipped_lines(p: &Palette, list: &[Skipped]) -> El {
    let mut c = col().key("refmedia-skipped").w_full().gap(2.0);
    let n = list.len();
    c = c.child(text(if n == 1 { "1 file skipped:".to_owned() } else { format!("{n} files skipped:") }).font(ty::CAPTION).color(p.warn));
    for x in list.iter().take(6) {
        c = c.child(text(format!("• {} · {}", x.name, x.reason)).font(ty::CAPTION).color(p.text_dim).truncate());
    }
    if n > 6 {
        c = c.child(text(format!("… and {} more", n - 6)).font(ty::CAPTION).color(p.text_faint));
    }
    c
}

/// "Background music removed from 3 of 12 files." / "Background music not removed: <why>." / nothing (turned off).
fn music_line(o: &Outcome) -> Option<String> {
    match (o.music_files, o.music_reason.as_deref()) {
        (0, None | Some("turned off")) => None,
        (0, Some(why)) => Some(format!("Background music not removed: {why}.")),
        (k, why) => {
            let files = if o.files <= 1 { String::new() } else if k >= o.files { format!(" from all {} files", o.files) } else { format!(" from {k} of {} files", o.files) };
            Some(match why {
                Some(w) => format!("Background music removed{files}; then {w}."),
                None => format!("Background music removed{files}."),
            })
        }
    }
}

/// "12 files combined · 7:32 · reference set: 11.4 s from file 3".
fn summary_line(o: &Outcome) -> String {
    let files = if o.files == 1 { "1 file combined".to_owned() } else { format!("{} files combined", o.files) };
    let mut t = format!("{files} · {}", clock(o.duration_s));
    if o.applied {
        let from = match (o.ref_file, o.ref_title.as_deref()) {
            (Some(n), _) => format!(" from file {n}"),
            (None, Some(name)) => format!(" from {name}"),
            _ => String::new(),
        };
        let what = o.language.as_deref().map_or_else(|| "reference set".to_owned(), |c| format!("{} reference set", lang::name(c)));
        t.push_str(&format!(" · {what}: {:.1} s{from}", o.ref_duration_s));
    }
    t
}

fn drop_well(s: &AppState, p: &Palette, over: bool) -> El {
    let (title, sub) = if over {
        ("Release to combine these files".to_owned(), "They stay where they are: nothing is copied or changed.".to_owned())
    } else {
        ("Drop recordings or videos of this voice here".to_owned(), format!("Many files are fine (up to 20 minutes are used). {}.", super::formats::import_hint(s)))
    };
    let picking = s.voices.refclip.media.picking;
    row().key("refmedia-drop").w_full().gap(12.0).padding(12.0).center_y().radius(radius::CONTROL).continuous_corners()
        .fill(if over { p.accent.alpha(0.13) } else { p.glass.alpha(p.glass.a * 0.6) })
        .border(if over { 2.0 } else { 1.0 }, if over { p.accent } else { p.rim_top.alpha(p.rim_top.a * 0.7) })
        .motion(motion::UI)
        .child(el().size(36.0, 36.0).radius(18.0).center().fill(if over { p.accent.alpha(0.2) } else { p.well }).child(icon_el(&icons::FOLDER, 16.0, if over { p.accent } else { p.text_dim })))
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(text(title).font(ty::LABEL).color(p.text).truncate())
            .child(text(sub).font(ty::CAPTION).color(p.text_faint).wrap()))
        .child(Button::new(if picking { "Choosing…" } else { "Add media…" }).small().icon(&icons::PLUS).key("refmedia-add").disabled(picking).on_press(pick_files).build(*p))
}

/// The block inside the reference section.
pub fn block(s: &AppState, p: &Palette, info: &VoiceInfo, has_pool: bool) -> El {
    let st = &s.voices.refclip.media;
    let mine = st.voice.as_deref() == Some(info.id.as_str());
    let busy = st.busy() && mine;
    let code = st.lang.clone().unwrap_or_else(|| info.lang().to_owned());
    let chip_label = if st.lang.is_none() { format!("{} (voice)", lang::label(&code)) } else { lang::label(&code) };
    let head = row().w_full().gap(10.0).center_y()
        .child(col().gap(1.0).min_w(0.0).grow(1.0)
            .child(text("Reference from media").font(ty::LABEL).color(p.text))
            .child(text("Combine many files of the same speaker; the cleanest 8-12 s become the reference.").font(ty::CAPTION).color(p.text_faint).wrap()))
        .child(lang::chip(p, "refmedia-lang", &chip_label, lang::Target::MediaRef, !busy));
    let mut c = col().key("refmedia").w_full().gap(10.0).child(head);

    if busy {
        let line = if (st.stage == "decode" || st.stage == "separate") && st.file > 0 {
            format!("File {} of {} · {}", st.file, st.files, st.name)
        } else {
            format!("{} files", st.files)
        };
        let label = if st.cancelling { "Cancelling…" } else { stage_label(&st.stage) };
        c = c.child(col().key("refmedia-run").w_full().gap(8.0).padding(12.0).radius(radius::CONTROL).fill(p.accent.alpha(0.06)).border(1.0, p.accent.alpha(0.25))
            .child(row().w_full().gap(10.0).center_y()
                .child(status_dot(*p, Tone::Accent, true))
                .child(col().grow(1.0).min_w(0.0).gap(1.0)
                    .child(text(label).font(ty::LABEL).color(p.text))
                    .child(text(line).font(ty::CAPTION).color(p.text_dim).truncate()))
                .child(text(format!("{:.0}%", st.pct)).font(ty::NUM).color(p.text_dim))
                .child(Button::new("Cancel").ghost().small().key("refmedia-cancel").disabled(st.cancelling || st.job.is_none()).on_press(cancel).build(*p)))
            .child(progress_bar(*p, Some(st.pct / 100.0))));
        return c;
    }

    let over = s.formats.over && s.screen == Screen::Voices;
    c = c.child(drop_well(s, p, over));
    c = c.child(if super::setuprun::has(s, "music_removal") {
        col().key("refmedia-music").w_full().gap(2.0)
            .child(kit::checkbox(p, remove_music(s), "Remove background music", toggle_music))
            .child(text("Cleaner reference from videos with music; a little slower.").font(ty::CAPTION).color(p.text_faint).wrap().pl(32.0))
    } else {
        super::setuprun::install_line(s, *p, "music_removal", "Removing background music")
    });
    if let Some(o) = st.outcome.as_ref().filter(|_| mine) {
        let mut res = row().key("refmedia-done").w_full().gap(10.0).center_y();
        if let Some(path) = o.preview_path.clone().filter(|_| o.applied) {
            let on = super::refclip::previewing(s, &path);
            res = res.child(super::refclip::play_button(p, "refmedia-play".into(), path, on));
        }
        res = res
            .child(col().grow(1.0).min_w(0.0).gap(2.0)
                .child(text(summary_line(o)).font(ty::LABEL).color(p.text).wrap())
                .child(text(if o.applied { "Listen to it, or open Choose… to try other clips from these files." } else { "Open Choose… to pick a clip from these files." }).font(ty::CAPTION).color(p.text_faint).wrap()))
            .child(el().key("refmedia-dismiss").role("button").label("Dismiss").size(28.0, 28.0).radius(14.0).center().hover(p.glass_hover).focusable(true)
                .on_press(dismiss).child(icon_el(&icons::CLOSE, 11.0, p.text_dim)));
        let mut box_ = col().w_full().gap(8.0).padding(12.0).radius(radius::CONTROL).fill(p.ok.alpha(if o.applied { 0.07 } else { 0.0 }))
            .border(1.0, if o.applied { p.ok.alpha(0.3) } else { p.well_rim }).child(res);
        if let Some(line) = music_line(o) {
            let tone = if o.music_files > 0 { p.text_dim } else { p.warn };
            box_ = box_.child(text(line).key("refmedia-music-done").font(ty::CAPTION).color(tone).wrap());
        }
        if let Some(msg) = &o.message {
            box_ = box_.child(kit::notice(p, Tone::Warning, msg));
        }
        if !o.truncated.is_empty() {
            box_ = box_.child(text(format!("Only the first part of {} was used (20-minute limit).", o.truncated.join(", "))).font(ty::CAPTION).color(p.text_dim).wrap());
        }
        if !o.skipped.is_empty() {
            box_ = box_.child(skipped_lines(p, &o.skipped));
        }
        c = c.child(box_);
    }
    if let Some(e) = st.error.as_ref().filter(|_| mine) {
        c = c.child(kit::notice(p, Tone::Danger, e));
        if !st.skipped.is_empty() {
            c = c.child(skipped_lines(p, &st.skipped));
        }
    }
    if has_pool {
        c = c.child(row().w_full().justify_end().child(Button::new("Remove combined media").ghost().small().icon(&icons::TRASH).key("refmedia-clear").on_press(clear_pool).build(*p)));
    }
    c
}

// ---------------------------------------------------------------------------------------------
// Scenes for headless snapshots
// ---------------------------------------------------------------------------------------------

/// The default on a CUDA machine: "Remove background music" ticked (no saved choice).
fn scene_idle(s: &mut AppState) {
    super::refclip::scene_demo(s);
    s.shell.gpu = true;
}

fn scene_drop(s: &mut AppState) {
    super::refclip::scene_demo(s);
    s.formats.over = true;
}

fn scene_running(s: &mut AppState) {
    super::refclip::scene_demo(s);
    s.voices.refclip.media = MediaState {
        voice: Some("my-voice".into()), phase: Phase::Running, job: Some("job-4".into()), files: 12, file: 5,
        name: "Interview with the author (full).mp4".into(), stage: "decode".into(), pct: 35.4, ..Default::default()
    };
}

/// Removing the music of file 2 of 6.
fn scene_separating(s: &mut AppState) {
    super::refclip::scene_demo(s);
    s.voices.refclip.media = MediaState {
        voice: Some("my-voice".into()), phase: Phase::Running, job: Some("job-4".into()), files: 6, file: 2,
        name: "Channel trailer with music.mp4".into(), stage: "separate".into(), pct: 21.8, remove_music: Some(true), ..Default::default()
    };
}

/// The checkbox off (the user's saved choice).
fn scene_music_off(s: &mut AppState) {
    super::refclip::scene_demo(s);
    s.voices.refclip.media.remove_music = Some(false);
}

fn scene_done(s: &mut AppState) {
    super::refclip::scene_demo_media(s);
    s.voices.refclip.media = MediaState {
        voice: Some("my-voice".into()),
        outcome: Some(Outcome {
            files: 12, duration_s: 452.0, applied: true, ref_duration_s: 11.4, ref_file: Some(3), ref_title: Some("podcast-ep-14.m4a".into()),
            preview_path: Some("/tmp/demo/reference.wav".into()),
            skipped: vec![
                Skipped { name: "trailer.webm".into(), reason: "no clear speech found".into() },
                Skipped { name: "notes.mp3".into(), reason: "Cannot read this audio file (Invalid data found when processing input)".into() },
            ],
            ..Default::default()
        }),
        ..Default::default()
    };
    s.voices.refclip.open = false;
}

/// After combining: the "Choose..." list with the media clips under their own heading.
fn scene_choose(s: &mut AppState) {
    scene_done(s);
    s.voices.refclip.open = true;
}

fn scene_done_lang(s: &mut AppState) {
    scene_done(s);
    if let Some(c) = s.voices.refclip.candidates.first().cloned() {
        // reference.wav is still the part clip; the media clips went to reference_uk.wav
        let mut cur = c;
        cur.preview_path = "/tmp/demo/reference.wav".into();
        cur.recommended = false;
        cur.manual = true;
        s.voices.refclip.current = Some(cur);
    }
    if let Some(pool) = s.voices.refclip.pool.as_mut() {
        pool.language = Some("uk".into());
    }
    s.shell.voices[0].references = vec!["uk".into()];
    s.voices.refclip.media.lang = Some("uk".into());
    if let Some(o) = s.voices.refclip.media.outcome.as_mut() {
        o.language = Some("uk".into());
        o.skipped.clear();
        o.files = 4;
        o.duration_s = 1200.0;
        o.truncated = vec!["lecture-2h.mkv".into()];
        o.message = Some("The 20-minute limit was reached; 3 file(s) were not used.".into());
    }
    s.voices.refclip.open = false;
}

/// Done with the music removed from most files.
fn scene_done_music(s: &mut AppState) {
    scene_done(s);
    s.voices.refclip.media.remove_music = Some(true);
    if let Some(o) = s.voices.refclip.media.outcome.as_mut() {
        o.music_files = 9;
        o.skipped.clear();
    }
}

/// Done, but the separation model is missing (an older install): the files were used as they are.
fn scene_done_nomodel(s: &mut AppState) {
    scene_done(s);
    s.voices.refclip.media.remove_music = Some(true);
    if let Some(o) = s.voices.refclip.media.outcome.as_mut() {
        o.music_reason = Some("the music-removal model is not installed (run setup again)".into());
        o.skipped.clear();
    }
}

fn scene_error(s: &mut AppState) {
    super::refclip::scene_demo(s);
    let st = &mut s.voices.refclip.media;
    st.voice = Some("my-voice".into());
    st.error = Some("None of the files has usable speech.".into());
    st.skipped = vec![
        Skipped { name: "intro-music.mp3".into(), reason: "no clear speech found".into() },
        Skipped { name: "screen-capture.mp4".into(), reason: "This file has no audio track".into() },
    ];
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("voices-media", scene_idle),
        ("voices-media-drop", scene_drop),
        ("voices-media-running", scene_running),
        ("voices-media-separating", scene_separating),
        ("voices-media-music-off", scene_music_off),
        ("voices-media-done", scene_done),
        ("voices-media-choose", scene_choose),
        ("voices-media-done-lang", scene_done_lang),
        ("voices-media-done-music", scene_done_music),
        ("voices-media-done-nomodel", scene_done_nomodel),
        ("voices-media-error", scene_error),
    ]
}
