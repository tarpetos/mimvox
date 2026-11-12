//! Audio formats, app-wide: what the engine can read (`formats.list`), the reusable "Export as..." control, and
//! the window-level file drop.
//!
//! * Import: the engine lists the extensions it can decode (audio and video). Redub's drop zone and file picker,
//!   Record's "Import file..." and the window drop all filter with [`import_extensions`].
//! * Export: [`export_button`] is a small button; pressing it opens a menu (format grid, quality, Export...) drawn
//!   by [`overlay`] (the shell adds it above the screen). The chosen format and quality are remembered in
//!   `settings.json` (`export_format`, `export_quality`). Export opens the native save dialog (rfd, on its own
//!   thread) and then asks the engine to `audio.export`; a WAV copy of a WAV needs no engine at all.
//! * Window drops: a file dropped anywhere lands on the screen that can use it (see [`window_drops`]); on Voices
//!   every dropped file goes into the active voice's "Reference from media".

use super::components::{Button, Tone, glass_edge, segmented};
use super::shell;
use super::theme::{Palette, motion, radius, ty};
use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use rune::{Align, El, Icon, Side, col, el, row, text};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Every export format the app knows: (wire name = file extension, label).
pub const EXPORT_ALL: [(&str, &str); 7] =
    [("wav", "WAV"), ("mp3", "MP3"), ("flac", "FLAC"), ("ogg", "OGG"), ("opus", "Opus"), ("m4a", "M4A"), ("aiff", "AIFF")];

/// Used until (or unless) the engine answers `formats.list`.
const DEFAULT_IMPORT: &[&str] = &[
    "wav", "mp3", "flac", "ogg", "oga", "opus", "m4a", "aac", "wma", "aiff", "aif", "mp4", "mkv", "mov", "avi", "webm", "m4v", "wmv", "flv", "mpg", "mpeg", "3gp",
];

pub static DOWNLOAD: Icon = Icon {
    name: "download",
    view: 16.0,
    body: concat!(
        r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">"#,
        r#"<path d="M8 2.6V10M4.8 7.2L8 10.4L11.2 7.2M3 13H13"/>"#,
        "</g>"
    ),
};

// -------------------------------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------------------------------

/// How much to compress a lossy export.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportQuality {
    High,
    #[default]
    Standard,
    Small,
}

impl ExportQuality {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::High => "high",
            Self::Standard => "standard",
            Self::Small => "small",
        }
    }
    fn parse(s: &str) -> Option<Self> {
        match s {
            "high" => Some(Self::High),
            "standard" => Some(Self::Standard),
            "small" => Some(Self::Small),
            _ => None,
        }
    }
    fn index(self) -> usize {
        match self {
            Self::High => 0,
            Self::Standard => 1,
            Self::Small => 2,
        }
    }
    fn from_index(i: usize) -> Self {
        [Self::High, Self::Standard, Self::Small][i.min(2)]
    }
}

/// The open "Export as..." menu: which button it hangs from and what it exports.
#[derive(Clone, Debug)]
pub struct Menu {
    pub anchor: String,
    pub src: PathBuf,
}

#[derive(Clone, Debug)]
struct Job {
    src: PathBuf,
    format: String,
    quality: ExportQuality,
}

#[derive(Debug, Default)]
enum Phase {
    #[default]
    Idle,
    /// The save dialog is open.
    Picking(Job),
    /// `audio.export` is running.
    Exporting { req: u64, dest: PathBuf },
}

pub struct FormatsState {
    /// Extensions the engine can read (lower case, no dot).
    pub import: Vec<String>,
    /// Formats the engine can write.
    pub export: Vec<String>,
    pub format: String,
    pub quality: ExportQuality,
    pub menu: Option<Menu>,
    /// A file drag is over the window (drives the drop highlight).
    pub over: bool,
    phase: Phase,
    list_req: Option<u64>,
    requested: bool,
    pub settings_read: bool,
}

impl Default for FormatsState {
    fn default() -> Self {
        FormatsState {
            import: DEFAULT_IMPORT.iter().map(|e| (*e).to_owned()).collect(),
            export: EXPORT_ALL.iter().map(|(e, _)| (*e).to_owned()).collect(),
            format: "wav".into(),
            quality: ExportQuality::Standard,
            menu: None,
            over: false,
            phase: Phase::Idle,
            list_req: None,
            requested: false,
            settings_read: false,
        }
    }
}

impl FormatsState {
    /// An export is in progress (dialog or engine).
    pub fn busy(&self) -> bool {
        !matches!(self.phase, Phase::Idle)
    }
    /// Headless scenes: never touch the settings file.
    pub fn scene_ready(&mut self) {
        self.settings_read = true;
    }
}

fn lossless(ext: &str) -> bool {
    matches!(ext, "wav" | "flac" | "aiff")
}

fn label_of(ext: &str) -> &'static str {
    EXPORT_ALL.iter().find(|(e, _)| *e == ext).map_or("", |(_, l)| l)
}

fn blurb(ext: &str) -> &'static str {
    match ext {
        "wav" => "Uncompressed. Best for editing.",
        "mp3" => "Small files that play everywhere.",
        "flac" => "Lossless, about half the size of WAV.",
        "ogg" => "Open lossy format, small and clean.",
        "opus" => "The best sound for its small size.",
        "m4a" => "AAC, for Apple devices and video tools.",
        "aiff" => "Uncompressed, for Apple audio tools.",
        _ => "",
    }
}

fn has_ext(path: &Path, exts: &[String]) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| exts.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Extensions the engine can read (audio and video).
pub fn import_extensions(s: &AppState) -> Vec<String> {
    s.formats.import.clone()
}

/// A one-line summary for hints: "WAV, MP3, FLAC, M4A, MP4, MKV...".
pub fn import_hint(s: &AppState) -> String {
    let mut shown: Vec<String> = ["wav", "mp3", "flac", "m4a", "ogg", "mp4", "mkv", "mov"]
        .iter()
        .filter(|e| s.formats.import.iter().any(|x| x == *e))
        .map(|e| e.to_uppercase())
        .collect();
    if shown.is_empty() {
        shown = s.formats.import.iter().take(6).map(|e| e.to_uppercase()).collect();
    }
    format!("{} and more", shown.join(", "))
}

// -------------------------------------------------------------------------------------------------
// Settings (`export_format`, `export_quality` in <data dir>/settings.json)
// -------------------------------------------------------------------------------------------------

fn settings_path() -> PathBuf {
    super::onboarding::data_dir().join("settings.json")
}

fn read_settings(s: &mut AppState) {
    s.formats.settings_read = true;
    let v: Value = std::fs::read_to_string(settings_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}));
    if let Some(f) = v.get("export_format").and_then(Value::as_str) {
        if EXPORT_ALL.iter().any(|(e, _)| *e == f) {
            s.formats.format = f.to_owned();
        }
    }
    if let Some(q) = v.get("export_quality").and_then(Value::as_str).and_then(ExportQuality::parse) {
        s.formats.quality = q;
    }
}

fn write_settings(format: &str, quality: ExportQuality) {
    let path = settings_path();
    let mut v: Value = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}));
    if !v.is_object() {
        v = json!({});
    }
    v["export_format"] = json!(format);
    v["export_quality"] = json!(quality.as_str());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(t) = serde_json::to_string_pretty(&v) {
        let _ = std::fs::write(path, t);
    }
}

// -------------------------------------------------------------------------------------------------
// Engine and events
// -------------------------------------------------------------------------------------------------

fn ext_list(v: Option<&Value>) -> Vec<String> {
    let Some(a) = v.and_then(Value::as_array) else { return Vec::new() };
    a.iter()
        .filter_map(|x| match x {
            Value::String(t) => Some(t.clone()),
            Value::Object(o) => ["ext", "id", "name"].iter().find_map(|k| o.get(*k).and_then(Value::as_str)).map(str::to_owned),
            _ => None,
        })
        .map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase())
        .filter(|e| !e.is_empty())
        .collect()
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Response { id, result } if s.formats.list_req == Some(*id) => {
            s.formats.list_req = None;
            if let Ok(v) = result {
                let (imp, exp) = (ext_list(v.get("import")), ext_list(v.get("export")));
                if !imp.is_empty() {
                    s.formats.import = imp;
                }
                if !exp.is_empty() {
                    s.formats.export = exp;
                    // A remembered format the engine cannot write falls back to WAV.
                    if !s.formats.export.iter().any(|e| *e == s.formats.format) {
                        s.formats.format = s.formats.export.first().cloned().unwrap_or_else(|| "wav".into());
                    }
                }
            }
        }
        EngineMsg::Response { id, result } => {
            let Phase::Exporting { req, dest } = &s.formats.phase else { return };
            if req != id {
                return;
            }
            let dest = dest.clone();
            s.formats.phase = Phase::Idle;
            match result {
                Ok(v) => {
                    let size = v.get("size_bytes").and_then(Value::as_u64).map(size_text).map(|t| format!(" ({t})")).unwrap_or_default();
                    shell::toast(s, Tone::Success, format!("Exported {}{size}", file_name(&dest)));
                }
                Err(e) => shell::toast(s, Tone::Danger, format!("Export failed: {}", e.message)),
            }
        }
        EngineMsg::Exited { .. } => {
            s.formats.list_req = None;
            s.formats.requested = false;
            if matches!(s.formats.phase, Phase::Exporting { .. }) {
                s.formats.phase = Phase::Idle;
                shell::toast(s, Tone::Danger, "Export failed: the engine stopped");
            }
        }
        _ => {}
    }
}

/// After every event: the saved export choice once, and the format list once the engine is ready.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;
    if !s.formats.settings_read && !s.shell.scripted() {
        read_settings(s);
        changed = true;
    }
    if !s.formats.requested && shell::engine_ready(s) && !s.shell.scripted() {
        s.formats.requested = true;
        s.formats.list_req = shell::request(s, "formats.list", json!({}));
    }
    changed
}

/// The save dialog answered (event from its thread).
fn on_picked(s: &mut AppState, choice: Option<PathBuf>) {
    if let Phase::Picking(job) = std::mem::take(&mut s.formats.phase) {
        if let Some(dest) = choice {
            finish_pick(s, job, dest);
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

pub fn size_text(bytes: u64) -> String {
    let b = bytes as f64;
    if b >= 1e6 { format!("{:.1} MB", b / 1e6) } else { format!("{:.0} KB", (b / 1e3).max(1.0)) }
}

// -------------------------------------------------------------------------------------------------
// Export
// -------------------------------------------------------------------------------------------------

/// A small "Export as..." button for `src`. `key` must be unique on screen (the menu hangs from it).
pub fn export_button(p: Palette, key: impl Into<String>, src: &Path, grow: bool) -> El {
    let (key, src) = (key.into(), src.to_path_buf());
    let k = key.clone();
    let mut b = Button::new("Export as…").small().icon(&DOWNLOAD).key(key).on_press(move |s: &mut AppState| toggle_menu(s, &k, &src));
    if grow {
        b = b.grow();
    }
    b.build(p)
}

fn toggle_menu(s: &mut AppState, key: &str, src: &Path) {
    if s.formats.menu.as_ref().is_some_and(|m| m.anchor == key) {
        s.formats.menu = None;
    } else {
        s.formats.menu = Some(Menu { anchor: key.to_owned(), src: src.to_path_buf() });
    }
}

fn pick_format(s: &mut AppState, ext: &str) {
    s.formats.format = ext.to_owned();
    write_settings(&s.formats.format, s.formats.quality);
}

fn pick_quality(s: &mut AppState, q: ExportQuality) {
    s.formats.quality = q;
    write_settings(&s.formats.format, s.formats.quality);
}

/// "Export..." in the menu: close it and open the save dialog.
fn start_export(s: &mut AppState) {
    let Some(menu) = s.formats.menu.take() else { return };
    if s.formats.busy() {
        shell::toast(s, Tone::Warning, "An export is already running.");
        return;
    }
    if s.shell.scripted() {
        return;
    }
    let job = Job { src: menu.src, format: s.formats.format.clone(), quality: s.formats.quality };
    let name = format!("{}.{}", job.src.file_stem().map_or_else(|| "audio".into(), |n| n.to_string_lossy().into_owned()), job.format);
    let (ext, label) = (job.format.clone(), label_of(&job.format).to_owned());
    let dir = job.src.parent().map(Path::to_path_buf).filter(|d| d.is_dir());
    s.formats.phase = Phase::Picking(job);
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_title("Export audio").set_file_name(name).add_filter(label, &[ext]);
        if let Some(d) = dir {
            dialog = dialog.set_directory(d);
        }
        let choice = dialog.save_file();
        crate::ui::events::post(move |s| on_picked(s, choice));
    });
}

fn finish_pick(s: &mut AppState, job: Job, mut dest: PathBuf) {
    if dest.extension().is_none_or(|e| !e.to_string_lossy().eq_ignore_ascii_case(&job.format)) {
        let mut name = dest.as_os_str().to_owned();
        name.push(".");
        name.push(&job.format);
        dest = PathBuf::from(name);
    }
    let src_is_wav = job.src.extension().is_some_and(|e| e.to_string_lossy().eq_ignore_ascii_case("wav"));
    if job.format == "wav" && src_is_wav {
        match std::fs::copy(&job.src, &dest) {
            Ok(n) => shell::toast(s, Tone::Success, format!("Exported {} ({})", file_name(&dest), size_text(n))),
            Err(e) => shell::toast(s, Tone::Danger, format!("Could not save the file: {e}")),
        }
        return;
    }
    let params = json!({ "path": job.src, "format": job.format, "dest": dest, "quality": job.quality.as_str() });
    match shell::request(s, "audio.export", params) {
        Some(req) => s.formats.phase = Phase::Exporting { req, dest },
        None => shell::toast(s, Tone::Danger, "The engine is not running, so this format cannot be made yet."),
    }
}

/// The export menu, to be added at the root (after the screens) so it draws above them. `None` when closed.
pub fn overlay(s: &AppState, p: Palette) -> Option<El> {
    let m = s.formats.menu.as_ref()?;
    let scrim = el().key("export-scrim").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).on_press(|s: &mut AppState| s.formats.menu = None);

    let avail: Vec<&(&str, &str)> = EXPORT_ALL.iter().filter(|(e, _)| s.formats.export.iter().any(|x| x == e)).collect();
    let avail = if avail.is_empty() { EXPORT_ALL.iter().collect() } else { avail };
    let mut grid = col().w_full().gap(6.0);
    for chunk in avail.chunks(4) {
        let mut r = row().w_full().gap(6.0);
        for (ext, label) in chunk {
            let on = s.formats.format == *ext;
            let ext_owned = (*ext).to_owned();
            r = r.child(
                row().key(format!("export-fmt-{ext}")).role("button").label(*label).checked(on).focusable(true).grow(1.0).h(32.0).center().radius(16.0)
                    .fill(if on { p.accent.alpha(0.22) } else { p.well })
                    .border(1.0, if on { p.accent.alpha(0.55) } else { p.well_rim })
                    .hover(if on { p.accent.alpha(0.28) } else { p.glass_hover })
                    .motion(motion::UI)
                    .on_press(move |s: &mut AppState| pick_format(s, &ext_owned))
                    .child(text(*label).font(ty::LABEL).color(if on { p.text } else { p.text_dim }).a11y_hidden()),
            );
        }
        // Keep the last row's chips the same width as the rows above.
        for _ in chunk.len()..4 {
            r = r.child(el().grow(1.0).h(32.0));
        }
        grid = grid.child(r);
    }

    let lossless_now = lossless(&s.formats.format);
    let quality = segmented(p, "export-quality", &["High", "Standard", "Small"], s.formats.quality.index(), |s, i| pick_quality(s, ExportQuality::from_index(i)));
    let quality_block = col().w_full().gap(6.0)
        .child(row().w_full().center_y().justify_between()
            .child(text("Quality").font(ty::LABEL).color(p.text))
            .child(quality))
        .opacity(if lossless_now { 0.4 } else { 1.0 });
    let blurb_text = if lossless_now { "Lossless: every detail is kept, so quality does not apply.".to_owned() } else { blurb(&s.formats.format).to_owned() };

    let name = file_name(&m.src);
    let go = Button::new("Export…").primary().small().icon(&DOWNLOAD).key("export-go").on_press(start_export).build(p);
    let cancel = Button::new("Cancel").ghost().small().key("export-cancel").on_press(|s| s.formats.menu = None).build(p);
    let panel = col().key("export-menu").w(318.0).padding(14.0).gap(12.0).radius(radius::PANEL + 4.0)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.97))
        .gradient(p.glass_top, p.glass_bottom)
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .backdrop_blur(28.0)
        .anchored(m.anchor.clone(), Side::Bottom, Align::End, 8.0)
        .transition(rune::Transition::scale(0.98))
        .child(col().w_full().gap(1.0)
            .child(text("Export as").font(ty::TITLE).color(p.text))
            .child(text(name).font(ty::CAPTION).color(p.text_faint).truncate()))
        .child(grid)
        .child(text(blurb_text).font(ty::CAPTION).color(p.text_dim).wrap())
        .child(quality_block)
        .child(row().w_full().justify_between().center_y().child(cancel).child(go));
    let panel = glass_edge(p, panel);
    Some(el().key("export-menu-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).child(scrim).child(panel))
}

// -------------------------------------------------------------------------------------------------
// Window-level drops
// -------------------------------------------------------------------------------------------------

/// Makes `root` take files dropped anywhere in the window: Record imports them into the current part, Voices combines
/// them into the active voice's reference from media, every other main screen opens them in Redub. Files the engine
/// cannot read are refused with a message.
pub fn window_drops(root: El, s: &AppState) -> El {
    let exts = s.formats.import.clone();
    root.on_drop_files(|s: &mut AppState, files, _| on_window_drop(s, files))
        .accepts_files(move |p| has_ext(p, &exts))
        .on_drop_hover(|s: &mut AppState, over, _| s.formats.over = over)
        .on_drop_rejected(|s: &mut AppState, _, _| shell::toast(s, Tone::Warning, "That kind of file is not supported. Try WAV, MP3, FLAC, M4A, MP4 or MKV."))
}

fn on_window_drop(s: &mut AppState, files: &[PathBuf]) {
    s.formats.over = false;
    let Some(path) = files.first().cloned() else { return };
    match s.screen {
        Screen::Onboarding => {}
        Screen::Voices => super::refmedia::drop_files(s, files),
        // Speak has no input file, and Redub takes one: files dropped there (or several anywhere) build the voice's
        // reference from media, shown on the Voices screen.
        Screen::Speak | Screen::Redub | Screen::Train if files.len() > 1 || s.screen == Screen::Speak => {
            shell::set_screen(s, Screen::Voices);
            super::refmedia::drop_files(s, files);
        }
        Screen::Record => super::record::import_dropped(s, files.to_vec()),
        _ => {
            shell::set_screen(s, Screen::Redub);
            super::redub::set_file(s, path);
        }
    }
}
