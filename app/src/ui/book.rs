//! Book mode of Speak: read a whole book (EPUB, DOCX, ODT, FB2, PDF with text, TXT, Markdown, HTML, RTF, subtitles)
//! in the active voice, one file per chapter.
//!
//! A mode of Speak rather than a tab: a book uses exactly Speak's voice settings (the Style panel stays on the right:
//! voice, language, accent, speed, expressiveness, voice match, quality), and the format chip of Speak's results.
//! The left panel switches between "Text" and "Book"; in Book mode the right column shows the book's output options and
//! the recent books instead of the take history.
//!
//! Flow: "Open book…" (or a text file dropped on the window, or "Open text file…" in Text mode) sends `text.import`;
//! a short text (up to [`LONG_WORDS`] words) from Text mode goes into the editor, anything longer opens here as a
//! draft: the chapters found in the file, each with an include box, a title that can be renamed and merge / split
//! buttons. "Read the book" sends `book.start` with the included chapters; the engine reads them in one job and
//! reports `book.progress` (after every segment), `book.chapter` (a chapter file is done), `book.done`,
//! `book.paused` or `book.error`. Pause / Resume / Cancel map to `book.pause` / `book.resume` / `book.cancel`.
//!
//! The book's state lives in the engine (`<data>/books/<id>/`), so the app can close mid-book: after the engine
//! starts, `book.list` names the books, and the most recent unfinished one is shown again, ready to resume. A book
//! holds the engine's one generation slot, so Text mode's Generate is blocked while one is read.
//!
//! Finished chapters play in the shared player bar (`book.play` returns a WAV to play; the chapter file itself when it
//! is a WAV).

use super::components::{Button, Tone, accent_glass, badge, empty_state, glass_card, icon_button, icon_el, icons, progress_bar, section_panel, segmented, status_dot};
use super::playbar;
use super::shell::{self, EngineStatus};
use super::theme::{Palette, motion, radius, ty};
use super::voices::kit;
use crate::engine::EngineMsg;
use crate::engine::protocol::{self, BookChapterDone, BookChapterIn, BookChapterRow, BookDone, BookParams, BookProgress, BookStarted, BookSummary, TextImport};
use crate::state::{AppState, Screen};
use rune::{El, ScrollSpec, TextEdit, col, el, row, text};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// Texts longer than this many words open in Book mode instead of the Speak editor.
pub const LONG_WORDS: u32 = 5000;

/// File types `text.import` reads.
pub const TEXT_EXTS: &[&str] = &["txt", "md", "markdown", "html", "htm", "xhtml", "epub", "docx", "odt", "rtf", "fb2", "pdf", "srt", "vtt"];

/// Generation seconds per audio second assumed before a book has measured its own (an RTX 4070 measured 0.2-0.4 with
/// Best for an untrained voice; voice conversion adds to it, so the estimate stays on the safe side).
fn guess_rtf(quality: protocol::Quality) -> f32 {
    if quality == protocol::Quality::Fast { 0.3 } else { 0.5 }
}

/// Whether `text.import` can read this file (by its extension).
pub fn is_text_path(p: &Path) -> bool {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
    name.ends_with(".fb2.zip") || p.extension().and_then(|e| e.to_str()).is_some_and(|e| TEXT_EXTS.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Purpose {
    /// From Text mode or a drop: a short text goes into the editor, a long one opens as a book.
    Auto,
    /// From Book mode: always a book.
    Book,
}

/// A chapter of the draft (an imported file not started yet).
#[derive(Clone, Debug)]
pub struct DraftChapter {
    pub title: String,
    pub text: String,
    pub words: u32,
    pub est_s: f32,
    pub include: bool,
}

/// An imported file, before "Read the book".
#[derive(Clone, Debug)]
pub struct Draft {
    pub file: TextImport,
    pub chapters: Vec<DraftChapter>,
}

pub struct BookState {
    /// Speak shows Book mode (else Text mode).
    pub shown: bool,
    picking: bool,
    importing: Option<(u64, Purpose)>,
    pub draft: Option<Draft>,
    /// The book project shown (from `book.start`, `book.get` or the list), with its chapters.
    pub project: Option<BookSummary>,
    /// "Pausing…" / "Cancelling…" until the job's end event arrives.
    stopping: Option<&'static str>,
    start_req: Option<u64>,
    resume_req: Option<u64>,
    get_req: Option<u64>,
    list_req: Option<u64>,
    play_req: Option<(u64, u32)>,
    /// `book.list` was asked since the engine became ready.
    listed: bool,
    /// The first list after the engine started: it may bring back an unfinished book.
    first_list: bool,
    pub recent: Vec<BookSummary>,
    /// Also make one M4B with chapter marks (`m4b`; settings.json `book_m4b`).
    pub m4b: bool,
    /// Read each chapter's title before it (`read_titles`; settings.json `book_read_titles`).
    pub read_titles: bool,
    /// A book's own expressiveness, calmer than Speak's by default (settings.json `book_expressiveness`); the Style
    /// panel's slider sets it in Book mode.
    pub expressiveness: f32,
    settings_loaded: bool,
    pub(crate) settings_dirty: bool,
    renaming: Option<(usize, TextEdit)>,
    /// The engine is making the whole-book M4B (the last step).
    m4b_making: bool,
}

/// The engine's default for books (`narration.EXPRESSIVENESS`): a steady narrator.
pub const EXPRESSIVENESS: f32 = 0.6;

impl Default for BookState {
    fn default() -> Self {
        BookState {
            shown: false,
            picking: false,
            importing: None,
            draft: None,
            project: None,
            stopping: None,
            start_req: None,
            resume_req: None,
            get_req: None,
            list_req: None,
            play_req: None,
            listed: false,
            first_list: false,
            recent: Vec::new(),
            m4b: false,
            read_titles: true,
            expressiveness: EXPRESSIVENESS,
            settings_loaded: false,
            settings_dirty: false,
            renaming: None,
            m4b_making: false,
        }
    }
}

impl BookState {
    /// A book is being read by the engine (it holds the generation slot).
    pub fn running(&self) -> bool {
        self.project.as_ref().is_some_and(|p| p.status == "running") || self.start_req.is_some() || self.resume_req.is_some()
    }
}

// -------------------------------------------------------------------------------------------------
// Opening files
// -------------------------------------------------------------------------------------------------

/// "Open book…" / "Open text file…": the native picker on its own thread; the answer arrives as an event.
pub fn choose_file(s: &mut AppState, book: bool) {
    if s.book.picking || s.shell.scripted() {
        return;
    }
    s.book.picking = true;
    let purpose = if book { Purpose::Book } else { Purpose::Auto };
    std::thread::spawn(move || {
        let choice = rfd::FileDialog::new()
            .set_title(if book { "Choose a book" } else { "Choose a text file" })
            .add_filter("Books and text", TEXT_EXTS)
            .pick_file();
        crate::ui::events::post(move |s| {
            s.book.picking = false;
            if let Some(path) = choice {
                import(s, path, purpose);
            }
        });
    });
}

/// A text file dropped on the window: it opens on Speak (a short text in the editor, a long one as a book; in Book
/// mode always as a book).
pub fn drop_file(s: &mut AppState, path: PathBuf) {
    let purpose = if s.screen == Screen::Speak && s.book.shown { Purpose::Book } else { Purpose::Auto };
    shell::set_screen(s, Screen::Speak);
    import(s, path, purpose);
}

fn import(s: &mut AppState, path: PathBuf, purpose: Purpose) {
    if !shell::engine_ready(s) {
        shell::toast(s, Tone::Warning, "Wait for the engine to start, then open the file again.");
        return;
    }
    let (method, params) = protocol::text_import(&path);
    match shell::request(s, method, params) {
        Some(id) => s.book.importing = Some((id, purpose)),
        None => shell::toast(s, Tone::Danger, "Could not reach the engine."),
    }
}

pub fn set_mode(s: &mut AppState, book: bool) {
    s.book.shown = book;
    s.book.renaming = None;
}

fn opened(s: &mut AppState, file: TextImport, purpose: Purpose) {
    // The text's language becomes the output language of the active voice (the Style panel shows it).
    if !file.language.is_empty() && s.shell.languages.iter().any(|l| l.code == file.language) && file.language != super::speak::language(s) {
        super::speak::set_language(s, &file.language);
        if file.language_detected {
            shell::toast(s, Tone::Neutral, format!("Language set to {} (detected in the text).", super::lang::name(&file.language)));
        }
    }
    for w in file.warnings.iter().take(2) {
        shell::toast(s, Tone::Warning, w.clone());
    }
    if purpose == Purpose::Auto && file.words <= LONG_WORDS {
        let joined = file.chapters.iter().map(|c| if file.chapters.len() > 1 && !c.title.is_empty() { format!("{}\n\n{}", c.title, c.text) } else { c.text.clone() })
            .collect::<Vec<_>>().join("\n\n");
        let mut edit = TextEdit::new(joined);
        edit.focused = true;
        s.speak.text = edit;
        s.book.shown = false;
        return;
    }
    if purpose == Purpose::Auto {
        shell::toast(s, Tone::Neutral, format!("{} is long ({} words): it opened in Book mode, where each chapter becomes its own file.", file.name, group(file.words)));
    }
    let chapters = file.chapters.iter().map(|c| DraftChapter { title: c.title.clone(), text: c.text.clone(), words: c.words, est_s: c.est_s, include: true }).collect();
    let mut file = file;
    file.chapters.clear(); // the texts live in the draft chapters (edited there)
    s.book.draft = Some(Draft { file, chapters });
    s.book.renaming = None;
    s.book.shown = true;
}

// -------------------------------------------------------------------------------------------------
// Chapter editing (draft)
// -------------------------------------------------------------------------------------------------

fn toggle_include(s: &mut AppState, i: usize) {
    if let Some(c) = s.book.draft.as_mut().and_then(|d| d.chapters.get_mut(i)) {
        c.include = !c.include;
    }
}

fn start_rename(s: &mut AppState, i: usize) {
    commit_rename(s);
    if let Some(c) = s.book.draft.as_ref().and_then(|d| d.chapters.get(i)) {
        let mut edit = TextEdit::new(c.title.clone());
        edit.focused = true;
        s.book.renaming = Some((i, edit));
    }
}

fn commit_rename(s: &mut AppState) {
    let Some((i, edit)) = s.book.renaming.take() else { return };
    let title = edit.text.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(c) = s.book.draft.as_mut().and_then(|d| d.chapters.get_mut(i)) {
        if !title.is_empty() {
            c.title = title;
        }
    }
}

/// Joins chapter `i + 1` into chapter `i` (its title becomes a paragraph of the text).
fn merge_next(s: &mut AppState, i: usize) {
    commit_rename(s);
    let Some(d) = s.book.draft.as_mut() else { return };
    if i + 1 >= d.chapters.len() {
        return;
    }
    let b = d.chapters.remove(i + 1);
    let a = &mut d.chapters[i];
    a.text = format!("{}\n\n{}\n\n{}", a.text, b.title, b.text);
    a.words += b.words;
    a.est_s += b.est_s;
    a.include = a.include || b.include;
}

/// Cuts chapter `i` in two at the paragraph break nearest its middle.
fn split(s: &mut AppState, i: usize) {
    commit_rename(s);
    let Some(d) = s.book.draft.as_mut() else { return };
    let Some(c) = d.chapters.get(i).cloned() else { return };
    let half = c.text.len() / 2;
    let cut = c.text.match_indices("\n\n").map(|(k, _)| k).min_by_key(|k| k.abs_diff(half));
    let Some(cut) = cut.filter(|k| *k > 0 && *k + 2 < c.text.len()) else {
        shell::toast(s, Tone::Warning, "This chapter is a single paragraph: there is nowhere to split it.");
        return;
    };
    let (head, tail) = (c.text[..cut].trim().to_owned(), c.text[cut..].trim().to_owned());
    let wa = head.split_whitespace().count() as u32;
    let wb = tail.split_whitespace().count() as u32;
    let total = (wa + wb).max(1) as f32;
    let first = DraftChapter { title: c.title.clone(), text: head, words: wa, est_s: c.est_s * wa as f32 / total, include: c.include };
    let second = DraftChapter { title: format!("{} (2)", c.title), text: tail, words: wb, est_s: c.est_s * wb as f32 / total, include: c.include };
    d.chapters.splice(i..=i, [first, second]);
}

// -------------------------------------------------------------------------------------------------
// Book actions
// -------------------------------------------------------------------------------------------------

/// Why "Read the book" cannot run now, or `None`.
fn start_blocker(s: &AppState) -> Option<&'static str> {
    match &s.shell.engine {
        EngineStatus::Ready { .. } => {}
        EngineStatus::Idle | EngineStatus::Starting { .. } => return Some("Waiting for the engine to start…"),
        _ => return Some("The engine is not running. Press Retry above."),
    }
    if shell::active_voice(s).is_none() {
        return Some("Record your voice first, or pick one at the top right.");
    }
    if s.speak.is_generating() {
        return Some("Wait for the text being spoken to finish.");
    }
    if s.book.running() {
        return Some("Another book is being read: pause it first.");
    }
    if s.book.draft.as_ref().is_some_and(|d| !d.chapters.iter().any(|c| c.include)) {
        return Some("Tick at least one chapter.");
    }
    None
}

fn start(s: &mut AppState) {
    commit_rename(s);
    if let Some(why) = start_blocker(s) {
        shell::toast(s, Tone::Warning, why);
        return;
    }
    let (Some(d), Some(voice)) = (s.book.draft.as_ref(), shell::active_voice(s)) else { return };
    let lang = super::speak::language(s);
    let out = super::formats::output_format(s);
    let title = if d.file.title.trim().is_empty() { d.file.name.clone() } else { d.file.title.clone() };
    let req = protocol::book_start(&BookParams {
        title: &title,
        source: d.file.path.to_str(),
        format: Some(&d.file.format),
        chapters: d.chapters.iter().filter(|c| c.include).map(|c| BookChapterIn { title: &c.title, text: &c.text }).collect(),
        voice_id: &voice.id,
        language: &lang,
        expressiveness: s.book.expressiveness,
        voice_match: s.speak.voice_match,
        quality: s.speak.quality,
        speed: s.speak.speed,
        accent: super::speak::accent_param(s, &lang),
        output_format: &out,
        m4b: s.book.m4b,
        read_titles: s.book.read_titles,
    });
    shell::stop_audio(s);
    match shell::request(s, req.0, req.1) {
        Some(id) => s.book.start_req = Some(id),
        None => shell::toast(s, Tone::Danger, "Could not reach the engine."),
    }
}

fn resume(s: &mut AppState) {
    let Some(id) = s.book.project.as_ref().map(|p| p.id.clone()) else { return };
    if s.speak.is_generating() {
        shell::toast(s, Tone::Warning, "Wait for the text being spoken to finish.");
        return;
    }
    let (m, p) = protocol::book_id_request("book.resume", &id);
    match shell::request(s, m, p) {
        Some(r) => s.book.resume_req = Some(r),
        None => shell::toast(s, Tone::Danger, "Could not reach the engine."),
    }
}

fn stop(s: &mut AppState, how: &'static str) {
    let Some(id) = s.book.project.as_ref().map(|p| p.id.clone()) else { return };
    let method = if how == "pause" { "book.pause" } else { "book.cancel" };
    let (m, p) = protocol::book_id_request(method, &id);
    if shell::request(s, m, p).is_some() {
        let running = s.book.project.as_ref().is_some_and(|p| p.status == "running");
        if running {
            s.book.stopping = Some(if how == "pause" { "Pausing…" } else { "Cancelling…" });
        } else if how == "cancel" {
            // A paused book: the engine just drops the unfinished chapter.
            if let Some(p) = s.book.project.as_mut() {
                p.status = "cancelled".into();
                reset_partial(p);
            }
        }
    }
}

fn reset_partial(p: &mut BookSummary) {
    for c in &mut p.chapter_list {
        if c.status != "done" {
            c.status = "waiting".into();
            c.segment = 0;
            c.audio_s = 0.0;
        }
    }
}

fn open_book(s: &mut AppState, id: &str) {
    let (m, p) = protocol::book_id_request("book.get", id);
    s.book.get_req = shell::request(s, m, p);
    s.book.draft = None;
    s.book.shown = true;
}

fn forget_book(s: &mut AppState, id: &str) {
    let (m, p) = protocol::book_id_request("book.delete", id);
    if shell::request(s, m, p).is_some() {
        s.book.recent.retain(|b| b.id != id);
        if s.book.project.as_ref().is_some_and(|p| p.id == id) {
            s.book.project = None;
        }
    }
}

fn close(s: &mut AppState) {
    s.book.renaming = None;
    if s.book.draft.take().is_none() && !s.book.running() {
        s.book.project = None;
    }
}

fn open_folder(s: &mut AppState) {
    let Some(folder) = s.book.project.as_ref().and_then(|p| p.folder.clone()) else { return };
    let target = if folder.is_dir() { folder } else { folder.parent().map(Path::to_path_buf).unwrap_or(folder) };
    if let Err(e) = super::speak::open_file(&target) {
        shell::toast(s, Tone::Danger, e);
    }
}

/// A done chapter's play button: ask the engine for a WAV to play (or pause / resume it when it is the one playing).
fn play_chapter(s: &mut AppState, index: u32) {
    let Some(p) = s.book.project.as_ref() else { return };
    let Some(c) = p.chapter_list.iter().find(|c| c.index == index) else { return };
    if let Some(path) = c.path.as_ref().filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav"))) {
        let (path, title) = (path.clone(), c.title.clone());
        toggle_play(s, &path, title, index);
        return;
    }
    let id = p.id.clone();
    if let Some(now) = s.playbar.now.as_ref().filter(|n| n.what == chapter_caption(index)) {
        let path = now.path.clone();
        let title = now.title.clone();
        toggle_play(s, &path, title, index);
        return;
    }
    let (m, params) = protocol::book_play(&id, index);
    if let Some(r) = shell::request(s, m, params) {
        s.book.play_req = Some((r, index));
    }
}

fn chapter_caption(index: u32) -> String {
    format!("Book · chapter {}", index + 1)
}

fn toggle_play(s: &mut AppState, path: &Path, title: String, index: u32) {
    if let Err(e) = playbar::toggle(s, path, title, chapter_caption(index), playbar::Source::Speak) {
        shell::toast(s, Tone::Danger, e);
    }
}

fn chapter_playing(s: &AppState, index: u32) -> bool {
    s.playbar.now.as_ref().is_some_and(|n| n.what == chapter_caption(index) && n.status == playbar::Status::Playing)
}

// -------------------------------------------------------------------------------------------------
// Engine messages
// -------------------------------------------------------------------------------------------------

fn friendly(e: &protocol::EngineError) -> String {
    match e.code.as_str() {
        "not_ready" => format!("The engine is not ready: {}.", e.message),
        "busy" => format!("The engine is busy: {}.", e.message),
        _ if e.message.is_empty() => "Something went wrong.".into(),
        _ => e.message.clone(),
    }
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    match msg {
        EngineMsg::Ready(_) => {
            s.book.listed = false;
            s.book.first_list = true;
        }
        EngineMsg::Exited { .. } => {
            s.book.listed = false;
            s.book.stopping = None;
            s.book.start_req = None;
            s.book.resume_req = None;
            if let Some(p) = s.book.project.as_mut().filter(|p| p.status == "running") {
                p.status = "paused".into();
                p.interrupted = true;
                p.job = None;
            }
        }
        EngineMsg::Response { id, result } => on_response(s, *id, result),
        EngineMsg::Event(ev) if ev.event.starts_with("book.") => on_event(s, &ev.event, &ev.data),
        _ => {}
    }
}

fn on_response(s: &mut AppState, id: u64, result: &Result<Value, protocol::EngineError>) {
    let b = &mut s.book;
    if let Some((req, purpose)) = b.importing.filter(|(r, _)| *r == id) {
        let _ = req;
        b.importing = None;
        match result.clone().and_then(protocol::parse_result::<TextImport>) {
            Ok(file) => opened(s, file, purpose),
            Err(e) => shell::toast(s, Tone::Danger, friendly(&e)),
        }
    } else if b.start_req == Some(id) {
        b.start_req = None;
        match result.clone().and_then(protocol::parse_result::<BookStarted>) {
            Ok(st) => {
                let Some(d) = b.draft.take() else { return };
                let rows = d.chapters.iter().filter(|c| c.include).enumerate()
                    .map(|(i, c)| BookChapterRow { index: i as u32, title: c.title.clone(), words: c.words, status: "waiting".into(), ..Default::default() })
                    .collect::<Vec<_>>();
                b.project = Some(BookSummary {
                    id: st.id.clone(),
                    title: if d.file.title.is_empty() { d.file.name.clone() } else { d.file.title.clone() },
                    format: Some(d.file.format_label.clone()),
                    language: Some(d.file.language.clone()),
                    status: "running".into(),
                    job: Some(st.job.clone()),
                    chapters: rows.len() as u32,
                    words: rows.iter().map(|r| r.words).sum(),
                    folder: st.folder.clone(),
                    chapter_list: rows,
                    ..Default::default()
                });
                super::speak::job_notes(s, st.accent_note.as_deref(), st.reference_note.as_deref());
                request_list(s);
            }
            Err(e) => shell::toast(s, Tone::Danger, friendly(&e)),
        }
    } else if b.resume_req == Some(id) {
        b.resume_req = None;
        match result.clone().and_then(protocol::parse_result::<BookStarted>) {
            Ok(st) => {
                if let Some(p) = b.project.as_mut().filter(|p| p.id == st.id) {
                    p.status = "running".into();
                    p.job = Some(st.job.clone());
                    p.error = None;
                    p.interrupted = false;
                }
                super::speak::job_notes(s, st.accent_note.as_deref(), st.reference_note.as_deref());
            }
            Err(e) => shell::toast(s, Tone::Danger, friendly(&e)),
        }
    } else if b.get_req == Some(id) {
        b.get_req = None;
        match result.clone().and_then(protocol::parse_result::<BookSummary>) {
            Ok(p) => b.project = Some(p),
            Err(e) => shell::toast(s, Tone::Danger, friendly(&e)),
        }
    } else if b.list_req == Some(id) {
        b.list_req = None;
        let list: Vec<BookSummary> = result.as_ref().ok().and_then(|v| v.get("books")).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
        b.recent = list;
        if std::mem::replace(&mut b.first_list, false) {
            // After an engine start: refresh the book shown, or bring back the latest unfinished one.
            let want = match &b.project {
                Some(p) => Some(p.id.clone()),
                None if b.draft.is_none() => b.recent.iter().find(|r| unfinished(r)).map(|r| r.id.clone()),
                None => None,
            };
            if let Some(id) = want {
                let (m, p) = protocol::book_id_request("book.get", &id);
                s.book.get_req = shell::request(s, m, p);
            }
        }
    } else if let Some((_, index)) = b.play_req.filter(|(r, _)| *r == id) {
        b.play_req = None;
        match result {
            Ok(v) => {
                let path = v.get("path_play").and_then(Value::as_str).map(PathBuf::from);
                let title = s.book.project.as_ref().and_then(|p| p.chapter_list.iter().find(|c| c.index == index)).map(|c| c.title.clone()).unwrap_or_default();
                if let Some(path) = path {
                    toggle_play(s, &path, title, index);
                }
            }
            Err(e) => shell::toast(s, Tone::Danger, friendly(e)),
        }
    }
}

fn unfinished(b: &BookSummary) -> bool {
    matches!(b.status.as_str(), "paused" | "error" | "running")
}

fn on_event(s: &mut AppState, name: &str, data: &Value) {
    let id = data.get("id").and_then(Value::as_str).unwrap_or("");
    let shown = s.book.project.as_ref().is_some_and(|p| p.id == id);
    match name {
        "book.progress" if shown => {
            let Ok(pr) = serde_json::from_value::<BookProgress>(data.clone()) else { return };
            let p = s.book.project.as_mut().expect("shown");
            p.totals = pr.totals;
            p.current = pr.chapter;
            p.status = "running".into();
            if let Some(c) = p.chapter_list.iter_mut().find(|c| c.index == pr.chapter) {
                if c.status != "done" {
                    c.status = "partial".into();
                    c.segment = pr.segment;
                    c.segments = pr.segments;
                    c.audio_s = pr.chapter_audio_s;
                }
            }
            s.book.m4b_stage(pr.stage.as_deref() == Some("m4b"));
        }
        "book.chapter" if shown => {
            let Ok(ch) = serde_json::from_value::<BookChapterDone>(data.clone()) else { return };
            let totals = serde_json::from_value(data.clone()).unwrap_or_default();
            let p = s.book.project.as_mut().expect("shown");
            p.totals = totals;
            p.chapters_done += 1;
            if let Some(c) = p.chapter_list.iter_mut().find(|c| c.index == ch.index) {
                c.status = "done".into();
                c.path = Some(ch.path);
                c.audio_s = ch.duration_s;
                c.rtf = ch.rtf;
                c.segment = c.segments;
            }
        }
        "book.done" => {
            if shown {
                let Ok(done) = serde_json::from_value::<BookDone>(data.clone()) else { return };
                let p = s.book.project.as_mut().expect("shown");
                p.status = "done".into();
                p.job = None;
                p.totals = done.totals;
                p.m4b = done.m4b.clone();
                p.chapters_done = p.chapters;
                let hours = fmt_len(p.totals.audio_s);
                let note = if done.m4b_requested && done.m4b.is_none() { " The M4B could not be made (it needs ffmpeg with AAC)." } else { "" };
                shell::toast(s, Tone::Success, format!("The book is ready: {hours} of audio.{note}"));
            }
            s.book.stopping = None;
            request_list(s);
        }
        "book.paused" => {
            if shown {
                if let Ok(sum) = serde_json::from_value::<BookSummary>(data.clone()) {
                    let p = s.book.project.as_mut().expect("shown");
                    p.status = "paused".into();
                    p.job = None;
                    p.totals = sum.totals;
                }
            }
            s.book.stopping = None;
            request_list(s);
        }
        "book.error" => {
            if shown {
                let code = data.get("code").and_then(Value::as_str).unwrap_or("internal").to_owned();
                let message = data.get("message").and_then(Value::as_str).unwrap_or("").to_owned();
                let p = s.book.project.as_mut().expect("shown");
                p.job = None;
                if code == "cancelled" {
                    p.status = "cancelled".into();
                    reset_partial(p);
                } else {
                    p.status = "error".into();
                    p.error = Some(protocol::EngineError::new(&code, message.clone()));
                    shell::toast(s, Tone::Danger, format!("The book stopped: {message}"));
                }
            }
            s.book.stopping = None;
            request_list(s);
        }
        _ => {}
    }
}

impl BookState {
    fn m4b_stage(&mut self, on: bool) {
        if on {
            self.stopping = None;
        }
        self.m4b_making = on;
    }
}

fn request_list(s: &mut AppState) {
    let (m, p) = ("book.list", json!({}));
    s.book.list_req = shell::request(s, m, p);
    s.book.listed = true;
}

/// After every event: the saved options once, their changes, and the book list once the engine is ready.
pub fn settle(s: &mut AppState) -> bool {
    let mut changed = false;
    if !s.shell.scripted() {
        if !s.book.settings_loaded {
            s.book.settings_loaded = true;
            s.book.m4b = super::onboarding::setting("book_m4b").and_then(|v| v.as_bool()).unwrap_or(false);
            s.book.read_titles = super::onboarding::setting("book_read_titles").and_then(|v| v.as_bool()).unwrap_or(true);
            s.book.expressiveness = super::onboarding::setting("book_expressiveness").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(EXPRESSIVENESS);
            changed = true;
        } else if s.book.settings_dirty {
            s.book.settings_dirty = false;
            super::onboarding::save_setting("book_m4b", json!(s.book.m4b));
            super::onboarding::save_setting("book_read_titles", json!(s.book.read_titles));
            super::onboarding::save_setting("book_expressiveness", json!((s.book.expressiveness * 100.0).round() / 100.0));
        }
        if !s.book.listed && shell::engine_ready(s) {
            request_list(s);
        }
    }
    changed
}

// -------------------------------------------------------------------------------------------------
// View
// -------------------------------------------------------------------------------------------------

/// "3 h 05 min", "12 min", "40 s".
pub fn fmt_len(secs: f32) -> String {
    let t = secs.max(0.0).round() as u64;
    if t >= 3600 {
        format!("{} h {:02} min", t / 3600, (t % 3600) / 60)
    } else if t >= 60 {
        format!("{} min", (t + 30) / 60)
    } else {
        format!("{t} s")
    }
}

/// "86,400".
fn group(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

/// The Text / Book switch at the top right of Speak's left panel.
pub fn mode_switch(s: &AppState, p: Palette) -> El {
    segmented(p, "speak-mode", &["Text", "Book"], usize::from(s.book.shown), |s, i| set_mode(s, i == 1))
}

/// The left panel in Book mode.
pub fn panel(s: &AppState, p: Palette) -> El {
    let head = row().key("book-head").pin(None, Some(14.5), Some(22.0), None).gap(10.0).center_y().child(mode_switch(s, p));
    let mut c = section_panel(p, "Book").key("book-panel").grow(1.0).min_w(0.0).h_full().gap(12.0).child(head);
    c = if let Some(d) = &s.book.draft {
        c.child(draft_view(s, p, d))
    } else if let Some(b) = &s.book.project {
        c.child(project_view(s, p, b))
    } else {
        c.child(empty_view(s, p))
    };
    c
}

fn empty_view(s: &AppState, p: Palette) -> El {
    let busy = s.book.importing.is_some() || s.book.get_req.is_some();
    let open = Button::new(if busy { "Opening…" } else { "Open book…" }).primary().large().icon(&icons::FILE).key("book-open")
        .disabled(busy || !matches!(s.shell.engine, EngineStatus::Ready { .. })).on_press(|s| choose_file(s, true)).build(p);
    col().key("book-empty").w_full().grow(1.0).min_h(0.0).center().gap(14.0)
        .child(empty_state(p, &icons::FILE, "Read a whole book in your voice",
            "EPUB, DOCX, ODT, FB2, PDF with a text layer, TXT, Markdown, HTML, RTF or subtitles. Every chapter becomes its own file, and you can pause and continue later, even after closing the app."))
        .child(open)
        .child(text("or drop the file on this window").font(ty::CAPTION).color(p.text_faint))
}

/// File name, format and language, with Open another / Close.
fn file_card(p: Palette, title: &str, sub: String, right: El) -> El {
    glass_card(p).key("book-file").w_full().padding(12.0)
        .child(row().w_full().gap(12.0).center_y()
            .child(el().size(36.0, 36.0).radius(radius::CONTROL).center().fill(p.accent.alpha(0.14)).child(icon_el(&icons::FILE, 17.0, p.accent)))
            .child(col().grow(1.0).min_w(0.0).gap(2.0)
                .child(text(title.to_owned()).font(ty::TITLE).color(p.text).truncate())
                .child(text(sub).font(ty::CAPTION).color(p.text_faint).truncate()))
            .child(right))
}

fn draft_view(s: &AppState, p: Palette, d: &Draft) -> El {
    let f = &d.file;
    let mut sub = vec![if f.format_label.is_empty() { f.format.to_uppercase() } else { f.format_label.clone() }];
    if !f.language.is_empty() {
        sub.push(format!("{}{}", super::lang::name(&f.language), if f.language_detected { " (detected)" } else { "" }));
    }
    if let Some(e) = f.encoding.as_ref().filter(|e| !e.is_empty()) {
        sub.push(e.to_uppercase());
    }
    let title = if f.title.trim().is_empty() { f.name.clone() } else { f.title.clone() };
    let name_line = if f.title.trim().is_empty() || f.title == f.name { sub.join(" · ") } else { format!("{} · {}", f.name, sub.join(" · ")) };
    let right = row().gap(8.0).center_y()
        .child(Button::new("Open another…").ghost().small().key("book-open-other").on_press(|s| choose_file(s, true)).build(p))
        .child(icon_button(p, "book-close", &icons::CLOSE, "Close this file", close));
    let inc: Vec<&DraftChapter> = d.chapters.iter().filter(|c| c.include).collect();
    let words: u32 = inc.iter().map(|c| c.words).sum();
    let speed = s.speak.speed.max(0.5);
    let audio = inc.iter().map(|c| c.est_s).sum::<f32>() / speed;
    let make = audio * guess_rtf(s.speak.quality);
    let totals = row().key("book-totals").w_full().gap(8.0).center_y().flex_wrap()
        .child(badge(p, &format!("{} of {} chapters", inc.len(), d.chapters.len()), Tone::Accent))
        .child(badge(p, &format!("{} words", group(words)), Tone::Neutral))
        .child(badge(p, &format!("about {} of audio", fmt_len(audio)), Tone::Neutral))
        .child(badge(p, &format!("about {} to make", fmt_len(make)), Tone::Neutral));
    let list = col().w_full().gap(2.0).children(d.chapters.iter().enumerate().map(|(i, c)| draft_row(s, p, i, c, i + 1 == d.chapters.len())));
    let block = start_blocker(s);
    let voice = shell::active_voice(s).map(|v| v.name.clone()).unwrap_or_else(|| "no voice".into());
    let hint = block.map(str::to_owned).unwrap_or_else(|| {
        format!("Read by {voice} in {} · saved as {} files, one per chapter", super::lang::name(&super::speak::language(s)), super::formats::output_format(s).to_uppercase())
    });
    let starting = s.book.start_req.is_some();
    let bar = row().key("book-bar").w_full().gap(16.0).center_y()
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(text("Ready to read").font(ty::LABEL).color(p.text))
            .child(text(hint).font(ty::CAPTION).color(if block.is_some() { p.warn } else { p.text_faint }).wrap()))
        .child(Button::new(if starting { "Starting…" } else { "Read the book" }).primary().large().icon(&icons::SPARKLE).key("book-start")
            .disabled(block.is_some() || starting).on_press(start).build(p));
    let mut c = col().key("book-draft").w_full().grow(1.0).min_h(0.0).gap(12.0)
        .child(file_card(p, &title, name_line, right))
        .child(totals);
    if let Some(w) = f.warnings.first() {
        c = c.child(text(w.clone()).font(ty::CAPTION).color(p.warn).wrap());
    }
    c.child(chapter_scroller(p, list)).child(bar)
}

fn chapter_scroller(p: Palette, list: El) -> El {
    super::components::well(p).key("book-chapters").w_full().grow(1.0).min_h(120.0).padding(6.0)
        .child(col().key("book-chapters-scroll").w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0).child(list))
}

fn check_box(p: Palette, key: String, on: bool, on_press: impl Fn(&mut AppState) + 'static) -> El {
    let b = el().key(key).role("checkbox").label("Include").checked(on).focusable(true).size(20.0, 20.0).radius(radius::SMALL).center()
        .motion(motion::UI).on_press(on_press);
    if on {
        b.fill(p.accent).child(icon_el(&icons::CHECK, 13.0, p.on_accent))
    } else {
        b.fill(p.well).border(1.0, p.rim_top).hover(p.glass_hover)
    }
}

fn draft_row(s: &AppState, p: Palette, i: usize, c: &DraftChapter, last: bool) -> El {
    let renaming = s.book.renaming.as_ref().filter(|(k, _)| *k == i);
    let title: El = match renaming {
        Some((_, edit)) => kit::field(&p, edit, "Chapter title", |s: &mut AppState| &mut s.book.renaming.as_mut().expect("renaming").1)
            .on_submit(|s: &mut AppState, _| commit_rename(s)),
        None => text(c.title.clone()).font(ty::LABEL).color(if c.include { p.text } else { p.text_faint }).truncate().grow(1.0).min_w(0.0)
            .role("button").label(format!("Rename {}", c.title)).help("Click to rename").on_press(move |s: &mut AppState| start_rename(s, i)),
    };
    let meta = format!("{} words · {}", group(c.words), fmt_len(c.est_s / s.speak.speed.max(0.5)));
    let mut r = row().key(format!("book-ch{i}")).w_full().min_h(40.0).px(8.0).gap(10.0).center_y().radius(radius::CONTROL).hover(p.glass_hover)
        .child(check_box(p, format!("book-inc{i}"), c.include, move |s| toggle_include(s, i)))
        .child(text(format!("{}", i + 1)).font(ty::NUM).color(p.text_faint).w(26.0))
        .child(el().grow(1.0).min_w(0.0).child(title))
        .child(text(meta).font(ty::CAPTION).color(p.text_faint));
    r = r.child(small_icon(p, format!("book-split{i}"), &SPLIT, "Split this chapter in two", move |s| split(s, i)));
    r = r.child(if last {
        el().size(28.0, 28.0)
    } else {
        small_icon(p, format!("book-merge{i}"), &MERGE, "Join the next chapter into this one", move |s| merge_next(s, i))
    });
    if !c.include {
        r = r.opacity(0.7);
    }
    r
}

fn small_icon(p: Palette, key: String, glyph: &'static rune::Icon, help: &str, on_press: impl Fn(&mut AppState) + 'static) -> El {
    el().key(key).role("button").label(help.to_owned()).help(help.to_owned()).size(28.0, 28.0).radius(14.0).center()
        .hover(p.glass_hover).focusable(true).motion(motion::UI).on_press(on_press)
        .child(icon_el(glyph, 14.0, p.text_dim))
}

fn status_word(b: &BookSummary) -> (&'static str, Tone) {
    match b.status.as_str() {
        "running" => ("Reading", Tone::Accent),
        "paused" if b.interrupted => ("Interrupted", Tone::Warning),
        "paused" => ("Paused", Tone::Warning),
        "done" => ("Finished", Tone::Success),
        "error" => ("Stopped by an error", Tone::Danger),
        "cancelled" => ("Cancelled", Tone::Neutral),
        _ => ("Not started", Tone::Neutral),
    }
}

fn project_view(s: &AppState, p: Palette, b: &BookSummary) -> El {
    let (word, tone) = status_word(b);
    let mut sub = Vec::new();
    if let Some(f) = b.format.as_ref().filter(|f| !f.is_empty()) {
        sub.push(f.to_uppercase());
    }
    sub.push(format!("{} chapters", b.chapters));
    if let Some(l) = b.language.as_ref().filter(|l| !l.is_empty()) {
        sub.push(super::lang::name(l).to_string());
    }
    if b.words > 0 {
        sub.push(format!("{} words", group(b.words)));
    }
    let running = b.status == "running";
    let right = row().gap(8.0).center_y()
        .child(badge(p, word, tone))
        .children(b.folder.is_some().then(|| icon_button(p, "book-folder", &icons::FOLDER, "Open the book's folder", open_folder)))
        .children((!running).then(|| icon_button(p, "book-close", &icons::CLOSE, "Close this book (it stays in the list on the right)", close)));
    let t = &b.totals;
    let k = (b.current + 1).min(b.chapters.max(1));
    let label = match b.status.as_str() {
        "running" if s.book.m4b_making => "Making the M4B…".to_owned(),
        "running" => format!("Chapter {k} of {} · {:.0}%", b.chapters, t.percent),
        "done" => format!("All {} chapters read", b.chapters),
        _ => format!("{} of {} chapters done · {:.0}%", b.chapters_done, b.chapters, t.percent),
    };
    let mut facts = vec![format!("{} of audio", fmt_len(t.audio_s))];
    if let Some(eta) = t.eta_s.filter(|_| b.status != "done") {
        facts.push(format!("about {} left", fmt_len(eta)));
    }
    if let Some(r) = t.rtf {
        facts.push(format!("{r:.2}× real time"));
    }
    if b.status == "done" && t.elapsed_s > 0.0 {
        facts.push(format!("made in {}", fmt_len(t.elapsed_s)));
    }
    let progress = col().key("book-progress").w_full().gap(8.0)
        .child(row().w_full().center_y().justify_between().gap(10.0)
            .child(text(label).font(ty::LABEL).color(p.text))
            .child(text(facts.join(" · ")).font(ty::NUM).color(p.text_dim)))
        .child(progress_bar(p, Some((t.percent / 100.0).max(if running { 0.01 } else { 0.0 }))));
    let list = col().w_full().gap(2.0).children(b.chapter_list.iter().map(|c| project_row(s, p, b, c)));
    let mut c = col().key("book-project").w_full().grow(1.0).min_h(0.0).gap(12.0)
        .child(file_card(p, &b.title, sub.join(" · "), right))
        .child(progress);
    if let Some(e) = b.error.as_ref().filter(|_| b.status == "error") {
        c = c.child(text(format!("Stopped: {}. Resume tries again from the same place.", e.message)).font(ty::CAPTION).color(p.bad).wrap());
    }
    if b.interrupted && b.status == "paused" {
        c = c.child(text("The app closed while this book was being read. Resume continues where it stopped.").font(ty::CAPTION).color(p.text_dim).wrap());
    }
    c.child(chapter_scroller(p, list)).child(project_bar(s, p, b))
}

fn project_row(s: &AppState, p: Palette, b: &BookSummary, c: &BookChapterRow) -> El {
    let idx = c.index;
    let running = b.status == "running";
    let (mark, meta): (El, String) = match c.status.as_str() {
        "done" => (icon_el(&icons::CHECK, 14.0, p.ok), {
            let mut m = fmt_len(c.audio_s);
            if let Some(r) = c.rtf {
                m.push_str(&format!(" · {r:.2}×"));
            }
            m
        }),
        "partial" if running => (status_dot(p, Tone::Accent, true), if c.segments > 0 { format!("part {} of {}", c.segment.min(c.segments), c.segments) } else { "starting…".into() }),
        "partial" => (icon_el(&icons::PAUSE, 13.0, p.warn), format!("stopped at part {} of {}", c.segment, c.segments.max(c.segment))),
        _ => (el().size(10.0, 10.0).radius(5.0).border(1.0, p.text_faint), if c.words > 0 { format!("{} words", group(c.words)) } else { String::new() }),
    };
    let on = c.status == "partial" && running;
    let mut r = row().key(format!("book-row{idx}")).w_full().min_h(40.0).px(8.0).gap(10.0).center_y().radius(radius::CONTROL)
        .fill(if on { p.accent.alpha(0.08) } else { rune::Color::TRANSPARENT })
        .child(el().w(20.0).center().child(mark))
        .child(text(format!("{}", idx + 1)).font(ty::NUM).color(p.text_faint).w(26.0))
        .child(text(c.title.clone()).font(ty::LABEL).color(if c.status == "waiting" { p.text_dim } else { p.text }).truncate().grow(1.0).min_w(0.0))
        .child(text(meta).font(ty::CAPTION).color(p.text_faint));
    if c.status == "done" {
        let playing = chapter_playing(s, idx);
        let btn = el().key(format!("book-play{idx}")).role("button").label(if playing { "Pause" } else { "Play" }).size(28.0, 28.0).radius(14.0).center()
            .hover(rune::Color::WHITE.alpha(0.1)).focusable(true).motion(motion::UI)
            .on_press(move |s: &mut AppState| play_chapter(s, idx))
            .child(icon_el(if playing { &icons::PAUSE } else { &icons::PLAY }, 12.0, p.on_accent));
        r = r.child(accent_glass(p, btn));
    } else {
        r = r.child(el().size(28.0, 28.0));
    }
    r
}

fn project_bar(s: &AppState, p: Palette, b: &BookSummary) -> El {
    let voice = b.voice_id.as_ref().and_then(|id| s.shell.voices.iter().find(|v| &v.id == id)).map(|v| v.name.clone());
    let (title, hint) = match b.status.as_str() {
        "running" => (s.book.stopping.unwrap_or("Reading…").to_owned(), "Pause any time: it continues from the same sentence. Speak, Redub and training wait until the book is paused or done.".to_owned()),
        "done" => ("Finished".to_owned(), match &b.m4b {
            Some(m) => format!("Chapter files and {} are in the book's folder.", m.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()),
            None => "One file per chapter, in the book's folder.".to_owned(),
        }),
        "cancelled" => ("Cancelled".to_owned(), "Finished chapters are kept. Resume starts again at the first unfinished chapter.".to_owned()),
        _ => (if b.status == "new" { "Not started".to_owned() } else { "Paused".to_owned() }, match voice {
            Some(v) => format!("Resume reads on with {v} and the settings the book started with."),
            None => "Resume reads on with the settings the book started with.".to_owned(),
        }),
    };
    let mut r = row().key("book-bar").w_full().gap(12.0).center_y()
        .child(col().grow(1.0).min_w(0.0).gap(2.0)
            .child(text(title).font(ty::LABEL).color(p.text))
            .child(text(hint).font(ty::CAPTION).color(p.text_faint).wrap()));
    let waiting = s.book.stopping.is_some() || s.book.resume_req.is_some();
    match b.status.as_str() {
        "running" => {
            r = r.child(Button::new("Cancel").ghost().key("book-cancel").disabled(waiting).on_press(|s| stop(s, "cancel")).build(p))
                .child(Button::new("Pause").large().icon(&icons::PAUSE).key("book-pause").disabled(waiting).on_press(|s| stop(s, "pause")).build(p));
        }
        "done" => {
            r = r.child(Button::new("New book").ghost().key("book-new").on_press(|s| choose_file(s, true)).build(p))
                .child(Button::new("Open folder").primary().large().icon(&icons::FOLDER).key("book-open-folder").on_press(open_folder).build(p));
        }
        _ => {
            if b.status == "paused" || b.status == "error" {
                r = r.child(Button::new("Cancel").ghost().key("book-cancel").help("Drop the unfinished chapter; finished chapters are kept").on_press(|s| stop(s, "cancel")).build(p));
            }
            let blocked = s.speak.is_generating() || s.book.recent.iter().any(|x| x.status == "running" && x.id != b.id);
            r = r.child(Button::new(if s.book.resume_req.is_some() { "Resuming…" } else { "Resume" }).primary().large().icon(&icons::PLAY).key("book-resume")
                .disabled(waiting || blocked || !matches!(s.shell.engine, EngineStatus::Ready { .. })).on_press(resume).build(p));
        }
    }
    r
}

/// The right column's lower panel in Book mode: the output options and the recent books.
pub fn side_panel(s: &AppState, p: Palette, own_scroll: bool) -> El {
    let head = row().key("book-side-head").pin(None, Some(16.5), Some(22.0), None).gap(8.0).center_y()
        .child(super::formats::output_chip(s, p, "book-outfmt"));
    let m4b = kit::checkbox(&p, s.book.m4b, "Also make one M4B audiobook with chapter marks", |s| {
        s.book.m4b = !s.book.m4b;
        s.book.settings_dirty = true;
    });
    let titles = kit::checkbox(&p, s.book.read_titles, "Read each chapter's title first", |s| {
        s.book.read_titles = !s.book.read_titles;
        s.book.settings_dirty = true;
    });
    let opts = col().key("book-opts").w_full().gap(10.0).child(m4b).child(titles);
    let mut list = col().w_full().gap(6.0);
    let shown = s.book.project.as_ref().map(|b| b.id.as_str());
    for b in s.book.recent.iter().take(12) {
        let id = b.id.clone();
        let id2 = b.id.clone();
        let (word, tone) = status_word(b);
        let on = shown == Some(b.id.as_str());
        let detail = format!("{} of {} chapters · {}", b.chapters_done, b.chapters, fmt_len(b.totals.audio_s));
        let mut card = row().key(format!("book-recent-{}", b.id)).w_full().px(10.0).py(8.0).gap(8.0).center_y().radius(radius::CONTROL)
            .fill(if on { p.accent.alpha(0.12) } else { p.glass_hover.alpha(0.0) }).hover(p.glass_hover).focusable(true)
            .on_press(move |s: &mut AppState| open_book(s, &id))
            .child(col().grow(1.0).min_w(0.0).gap(2.0)
                .child(text(b.title.clone()).font(ty::LABEL).color(p.text).truncate())
                .child(row().gap(6.0).center_y().child(status_dot(p, tone, b.status == "running")).child(text(format!("{word} · {detail}")).font(ty::CAPTION).color(p.text_faint).truncate())));
        if b.status != "running" {
            card = card.child(small_icon(p, format!("book-forget-{}", b.id), &icons::TRASH, "Remove from the list (the audio files stay)", move |s| forget_book(s, &id2)));
        }
        list = list.child(card);
    }
    let books = if s.book.recent.is_empty() {
        col().key("book-recent-empty").w_full().child(text("Books you read appear here, so you can come back to them.").font(ty::CAPTION).color(p.text_faint).wrap())
    } else {
        col().key("book-recent").w_full().gap(8.0).child(text("Your books").font(ty::LABEL).color(p.text_dim)).child(list)
    };
    let body = col().w_full().gap(16.0).child(opts).child(books);
    let body = if own_scroll {
        col().key("book-side-scroll").w_full().grow(1.0).min_h(0.0).scroll(ScrollSpec { stick_bottom: false }).scroll_edge_fade(14.0).child(body)
    } else {
        body
    };
    section_panel(p, "Output").key("book-side").w_full().grow(1.0).min_h(0.0).gap(12.0).child(head).child(body)
}

/// Join: two lines meeting in one.
static MERGE: rune::Icon = rune::Icon {
    name: "merge",
    view: 16.0,
    body: r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 2.5V5.5A2.5 2.5 0 0 0 6.5 8H8M12 2.5V5.5A2.5 2.5 0 0 1 9.5 8H8V13.5M5.8 11.4L8 13.6L10.2 11.4"/></g>"#,
};

/// Split: one line forking in two.
static SPLIT: rune::Icon = rune::Icon {
    name: "split",
    view: 16.0,
    body: r#"<g fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2.5V7M8 7L4 11V13.5M8 7L12 11V13.5"/></g>"#,
};

// -------------------------------------------------------------------------------------------------
// Headless scenes
// -------------------------------------------------------------------------------------------------

// A public-domain book (Herman Melville, 1851) for the snapshots.
const CHAPTERS: [(&str, u32); 9] = [
    ("Etymology", 640),
    ("Chapter 1. Loomings", 2210),
    ("Chapter 2. The Carpet-Bag", 1900),
    ("Chapter 3. The Spouter-Inn", 5150),
    ("Chapter 4. The Counterpane", 1550),
    ("Chapter 5. Breakfast", 900),
    ("Chapter 6. The Street", 1100),
    ("Chapter 7. The Chapel", 1100),
    ("Chapter 8. The Pulpit", 1050),
];

fn scene_base(s: &mut AppState) {
    super::speak::scene_idle(s);
    s.book.shown = true;
    s.book.m4b = true;
    s.book.read_titles = true;
    s.shell.languages = Vec::new();
}

fn scene_file() -> TextImport {
    TextImport {
        path: "/home/me/Books/Moby-Dick.epub".into(),
        name: "Moby-Dick.epub".into(),
        format: "epub".into(),
        format_label: "EPUB".into(),
        title: "Moby-Dick".into(),
        encoding: None,
        language: "en".into(),
        language_detected: true,
        words: CHAPTERS.iter().map(|c| c.1).sum(),
        est_s: CHAPTERS.iter().map(|c| c.1 as f32 / 145.0 * 60.0).sum(),
        chapters: Vec::new(),
        warnings: Vec::new(),
    }
}

fn scene_project(s: &mut AppState, status: &str, done: usize, seg: (u32, u32)) {
    scene_base(s);
    let rows = CHAPTERS.iter().enumerate().map(|(i, (t, w))| {
        let d = *w as f32 / 145.0 * 60.0;
        let st = if i < done { "done" } else if i == done && seg.0 > 0 { "partial" } else { "waiting" };
        BookChapterRow {
            index: i as u32,
            title: (*t).into(),
            words: *w,
            status: st.into(),
            segment: if st == "partial" { seg.0 } else { 0 },
            segments: if st == "partial" { seg.1 } else { 0 },
            audio_s: if st == "done" { d } else if st == "partial" { d * seg.0 as f32 / seg.1 as f32 } else { 0.0 },
            rtf: (st == "done").then_some(0.30 + 0.01 * i as f32),
            path: (st == "done").then(|| PathBuf::from(format!("/tmp/books/{:02} - {t}.mp3", i + 1))),
            ..Default::default()
        }
    }).collect::<Vec<_>>();
    let audio: f32 = rows.iter().map(|r| r.audio_s).sum();
    let total: f32 = CHAPTERS.iter().map(|c| c.1 as f32 / 145.0 * 60.0).sum();
    let finished = status == "done";
    s.book.project = Some(BookSummary {
        id: "under-the-old-lime-tree-20261113-101500".into(),
        title: "Moby-Dick".into(),
        format: Some("EPUB".into()),
        language: Some("en".into()),
        voice_id: Some("me".into()),
        status: status.into(),
        job: (status == "running").then(|| "job-4".into()),
        chapters: rows.len() as u32,
        chapters_done: done as u32,
        current: done.min(rows.len() - 1) as u32,
        folder: Some("/tmp/books/Moby-Dick".into()),
        m4b: finished.then(|| PathBuf::from("/tmp/books/Moby-Dick/Moby-Dick.m4b")),
        words: CHAPTERS.iter().map(|c| c.1).sum(),
        totals: protocol::BookTotals {
            percent: 100.0 * audio / total,
            audio_s: audio,
            elapsed_s: audio * 0.32,
            rtf: Some(0.32),
            eta_s: (!finished).then_some((total - audio) * 0.32),
            est_audio_s: Some(total),
        },
        chapter_list: rows,
        ..Default::default()
    });
    s.book.recent = vec![
        s.book.project.clone().unwrap_or_default(),
        BookSummary { id: "kobzar-20261101-090000".into(), title: "Кобзар".into(), status: "done".into(), chapters: 24, chapters_done: 24, totals: protocol::BookTotals { audio_s: 31_200.0, ..Default::default() }, ..Default::default() },
    ];
}

/// A book just imported: the chapters found, with include boxes and the totals.
pub fn scene_imported(s: &mut AppState) {
    scene_base(s);
    let chapters = CHAPTERS.iter().enumerate().map(|(i, (t, w))| DraftChapter { title: (*t).into(), text: String::new(), words: *w, est_s: *w as f32 / 145.0 * 60.0, include: i > 0 }).collect();
    s.book.draft = Some(Draft { file: scene_file(), chapters });
    s.book.recent = vec![BookSummary { id: "kobzar-20261101-090000".into(), title: "Кобзар".into(), status: "done".into(), chapters: 24, chapters_done: 24, totals: protocol::BookTotals { audio_s: 31_200.0, ..Default::default() }, ..Default::default() }];
}

/// Chapter 4 of 9 being read.
pub fn scene_generating(s: &mut AppState) {
    scene_project(s, "running", 3, (37, 112));
}

/// Paused in the middle of chapter 4 (resumable).
pub fn scene_paused(s: &mut AppState) {
    scene_project(s, "paused", 3, (37, 112));
}

/// The app closed mid-book: brought back after the next start, ready to resume.
pub fn scene_interrupted(s: &mut AppState) {
    scene_project(s, "paused", 3, (37, 112));
    if let Some(p) = s.book.project.as_mut() {
        p.interrupted = true;
    }
}

/// Every chapter read, a chapter playing in the player bar.
pub fn scene_finished(s: &mut AppState) {
    scene_project(s, "done", CHAPTERS.len(), (0, 0));
    playbar::scene_now(s, "/tmp/books/02.wav", CHAPTERS[1].0, &chapter_caption(1), playbar::Source::Speak, playbar::Status::Playing, 312.0, 2031.0);
}

/// Book mode before a file is opened.
pub fn scene_empty(s: &mut AppState) {
    scene_base(s);
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("book_empty", scene_empty),
        ("book_imported", scene_imported),
        ("book_generating", scene_generating),
        ("book_paused", scene_paused),
        ("book_interrupted", scene_interrupted),
        ("book_finished", scene_finished),
    ]
}
