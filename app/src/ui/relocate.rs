//! "Move data…": relocate the whole data folder (runtime, models, voices) to another drive or folder.
//!
//! Everything that happens off the UI thread (the dialog, the measurements, the subprocess's output and exit) comes
//! back as events (`ui::events::post`).
//!
//! Flow: native folder picker (rfd, on its own thread) -> confirm card (size to move, free space on the target)
//! -> the engine is stopped -> `setup.sh|setup.cmd --move-to <path> --yes` runs as a hidden subprocess; its
//! `[move] NN% a/b GB` lines drive a progress bar -> the engine is restarted from wherever the data dir now
//! resolves (`EngineConfig::discover`). When the move fails, the old location keeps working: the engine is simply
//! restarted from it and the error is shown.

use super::components::Tone;
use super::events;
use super::shell::{self, EngineStatus};
use super::theme::{Palette, radius, ty};
use super::voices::kit::{self, BtnKind};
use crate::state::AppState;
use rune::{El, col, el, row, text};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// What the move subprocess reported so far (its output lines arrive as events).
#[derive(Default)]
struct Shared {
    /// `(percent 0..=100, "a/b GB")` from the latest `[move] NN% ...` line.
    progress: Option<(f32, String)>,
    /// The last lines of output (stdout and stderr) for error messages.
    tail: Vec<String>,
}

/// Result of the background measurements shown on the confirm card.
#[derive(Clone, Debug, Default)]
struct Facts {
    /// Bytes in the current data folder.
    size: Option<u64>,
    /// Free bytes on the target drive.
    free: Option<u64>,
    done: bool,
}

#[derive(Default)]
enum Phase {
    #[default]
    Idle,
    /// The folder dialog is open.
    Picking,
    Confirm { target: PathBuf, facts: Facts },
    Running { shared: Shared },
    Failed(String),
}

#[derive(Default)]
pub struct RelocateState {
    phase: Phase,
}

impl RelocateState {
    /// A move (or its dialog) is in progress: the rest of the app must wait.
    pub fn busy(&self) -> bool {
        !matches!(self.phase, Phase::Idle)
    }
}

// -------------------------------------------------------------------------------------------------
// Actions
// -------------------------------------------------------------------------------------------------

/// Opens the folder picker for the new location.
pub fn start(s: &mut AppState) {
    if s.relocate.busy() || s.shell.scripted() {
        return;
    }
    s.relocate.phase = Phase::Picking;
    let start = s.shell.data_dir.parent().map(Path::to_path_buf).filter(|d| d.is_dir());
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_title("Choose the new folder for your voice data");
        if let Some(dir) = start {
            dialog = dialog.set_directory(dir);
        }
        let choice = dialog.pick_folder();
        events::post(move |s| {
            if !matches!(s.relocate.phase, Phase::Picking) {
                return;
            }
            match choice {
                Some(dir) => begin_confirm(s, dir),
                None => s.relocate.phase = Phase::Idle,
            }
        });
    });
}

fn dir_size(dir: &Path) -> u64 {
    let Ok(rd) = std::fs::read_dir(dir) else { return 0 };
    rd.flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_dir() => dir_size(&e.path()),
            Ok(t) if t.is_file() => e.metadata().map_or(0, |m| m.len()),
            _ => 0,
        })
        .sum()
}

/// The folder the data lands in: `<chosen>/mimvox` unless the chosen folder already has that name.
fn final_target(chosen: &Path) -> PathBuf {
    if chosen.file_name().is_some_and(|n| n == "mimvox") { chosen.to_path_buf() } else { chosen.join("mimvox") }
}

fn begin_confirm(s: &mut AppState, chosen: PathBuf) {
    let target = final_target(&chosen);
    let (from, probe, for_target) = (s.shell.data_dir.clone(), chosen.display().to_string(), target.clone());
    std::thread::spawn(move || {
        let size = dir_size(&from);
        let free = super::onboarding::free_bytes(&probe);
        events::post(move |s| {
            if let Phase::Confirm { target, facts } = &mut s.relocate.phase {
                if *target == for_target {
                    *facts = Facts { size: Some(size), free, done: true };
                }
            }
        });
    });
    s.relocate.phase = Phase::Confirm { target, facts: Facts::default() };
}

fn cancel(s: &mut AppState) {
    s.relocate.phase = Phase::Idle;
}

/// Why a move cannot start right now.
fn blocked(s: &AppState, target: &Path) -> Option<String> {
    let cur = &s.shell.data_dir;
    if target == cur.as_path() {
        return Some("That is the folder the data is already in.".into());
    }
    if target.starts_with(cur) {
        return Some("Choose a folder outside the current data folder.".into());
    }
    if std::env::var_os("MIMVOX_DATA").is_some_and(|v| !v.is_empty()) {
        return Some("The data folder is fixed by the MIMVOX_DATA setting, so it cannot be moved from here.".into());
    }
    if super::train::is_training(s) {
        return Some("A voice is training. Wait for it to finish, then move the data.".into());
    }
    if s.speak.is_generating() {
        return Some("Speech is being generated. Try again in a moment.".into());
    }
    None
}

fn run(s: &mut AppState) {
    let Phase::Confirm { target, .. } = &s.relocate.phase else { return };
    let target = target.clone();
    if blocked(s, &target).is_some() {
        return;
    }
    let Some(root) = super::onboarding::find_setup_root() else {
        s.relocate.phase = Phase::Failed("Could not find setup.sh / setup.cmd next to the app.".into());
        return;
    };
    // Stop the engine first: it holds files in the data folder open.
    if let Some(mut old) = s.engine.take() {
        let _ = shell::guard(|| old.shutdown());
    }
    s.shell.engine = EngineStatus::Starting { since: shell::clock_now(s) };
    s.shell.preparing = None;

    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/c", "setup.cmd"]);
        c
    } else {
        let mut c = Command::new("sh");
        c.arg("setup.sh");
        c
    };
    cmd.arg("--move-to").arg(&target).arg("--yes").current_dir(&root).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            fail(s, format!("Could not start the move: {e}"));
            return;
        }
    };
    let (out, err) = (child.stdout.take(), child.stderr.take());
    let mut readers = Vec::new();
    for stream in [out.map(|o| Box::new(o) as Box<dyn std::io::Read + Send>), err.map(|e| Box::new(e) as Box<dyn std::io::Read + Send>)].into_iter().flatten() {
        readers.push(std::thread::spawn(move || {
            // Blocks on each line and hands it to the UI thread as it comes.
            for line in BufReader::new(stream).split(b'\n').flatten() {
                // Progress bars rewrite one line with \r: look at the last piece.
                let line = String::from_utf8_lossy(&line).into_owned();
                let line = line.rsplit('\r').next().unwrap_or("").trim().to_owned();
                if !line.is_empty() {
                    events::post(move |s| on_line(s, line));
                }
            }
        }));
    }
    std::thread::spawn(move || {
        let code = child.wait().ok().and_then(|st| st.code());
        // Every line is posted before the exit, so the error message sees the last of them.
        for r in readers {
            let _ = r.join();
        }
        events::post(move |s| on_exit(s, code));
    });
    s.relocate.phase = Phase::Running { shared: Shared::default() };
}

/// `[move] 42% 8.1/19.3 GB` -> `(42.0, "8.1/19.3 GB")`.
fn parse_progress(line: &str) -> Option<(f32, String)> {
    let rest = line.trim().strip_prefix("[move]")?.trim();
    let (pct, detail) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
    let pct: f32 = pct.strip_suffix('%')?.parse().ok()?;
    Some((pct.clamp(0.0, 100.0), detail.trim().to_owned()))
}

/// The move ended without success: bring the engine back from the old location and show why.
fn fail(s: &mut AppState, why: String) {
    shell::restart_engine(s);
    s.relocate.phase = Phase::Failed(why);
}

/// A line of the move's output (event from a reader thread).
fn on_line(s: &mut AppState, line: String) {
    let Phase::Running { shared } = &mut s.relocate.phase else { return };
    if let Some(p) = parse_progress(&line) {
        shared.progress = Some(p);
    }
    shared.tail.push(line);
    if shared.tail.len() > 12 {
        shared.tail.remove(0);
    }
}

/// The move ended (event from its waiter thread; `None` code = killed by a signal).
fn on_exit(s: &mut AppState, code: Option<i32>) {
    let Phase::Running { shared } = &s.relocate.phase else { return };
    if code == Some(0) {
        // Success: the pointer file now names the new folder, so discover() finds it.
        s.relocate.phase = Phase::Idle;
        super::onboarding::recheck(s);
        shell::restart_engine(s);
        shell::toast(s, Tone::Success, "Your data was moved");
        return;
    }
    let tail = shared.tail.last().cloned().unwrap_or_default();
    let code = code.map_or("no exit code".to_owned(), |c| format!("exit code {c}"));
    let why = if tail.is_empty() { format!("The move failed ({code}).") } else { format!("The move failed ({code}): {tail}") };
    fail(s, why);
}

// -------------------------------------------------------------------------------------------------
// View
// -------------------------------------------------------------------------------------------------

/// The footer block of the Voices screen.
pub fn footer(s: &AppState, p: &Palette) -> El {
    kit::card(p).gap(10.0)
        .child(text("Data location").font(ty::TITLE).color(p.text))
        .child(text(format!("Your voices, recordings and models live in {}", s.shell.data_dir.display())).font(ty::CAPTION).color(p.text_dim).wrap())
        .child(row().child(kit::button(p, "Move data…", BtnKind::Glass, !s.relocate.busy() && shell::engine_ready(s), start)))
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

/// The dialog layer above everything (confirm, progress, error), or `None` when nothing is happening.
pub fn overlay(s: &AppState, p: &Palette) -> Option<El> {
    let card = match &s.relocate.phase {
        Phase::Idle | Phase::Picking => return None,
        Phase::Confirm { target, facts } => {
            let f = facts.clone();
            let block = blocked(s, target);
            let short = matches!((f.size, f.free), (Some(sz), Some(fr)) if fr < sz + sz / 20);
            let measured = if f.done {
                format!("{} to move. {} free on the new drive.", f.size.map_or("Unknown size".into(), gb), f.free.map_or("Unknown space".into(), gb))
            } else {
                "Measuring…".to_owned()
            };
            let mut c = dialog(p, "Move your data?")
                .child(text("Everything moves to:").font(ty::BODY).color(p.text_dim))
                .child(text(target.display().to_string()).font(ty::LABEL).color(p.text).wrap())
                .child(text(measured).font(ty::BODY).color(p.text_dim).wrap())
                .child(text("The voice engine stops while the files are copied and restarts afterwards. This can take several minutes.").font(ty::CAPTION).color(p.text_faint).wrap());
            if short {
                c = c.child(kit::notice(p, Tone::Danger, "The new drive does not have enough free space."));
            }
            if let Some(why) = &block {
                c = c.child(kit::notice(p, Tone::Warning, why));
            }
            c.child(row().w_full().justify_end().gap(10.0)
                .child(kit::button(p, "Cancel", BtnKind::Glass, true, cancel))
                .child(kit::button(p, "Move", BtnKind::Primary, f.done && !short && block.is_none(), run)))
        }
        Phase::Running { shared } => {
            let (pct, detail) = shared.progress.clone().unwrap_or((0.0, String::new()));
            dialog(p, "Moving your data…")
                .child(kit::progress(p, (pct / 100.0).max(0.02)))
                .child(text(if detail.is_empty() { format!("{pct:.0}%") } else { format!("{pct:.0}%  ·  {detail}") }).font(ty::NUM).color(p.text_dim))
                .child(text("Please keep the app open. Nothing is deleted from the old folder until everything has been copied.").font(ty::CAPTION).color(p.text_faint).wrap())
        }
        Phase::Failed(msg) => dialog(p, "The move did not finish")
            .child(kit::notice(p, Tone::Danger, msg))
            .child(text("Your data is still in its old location and keeps working.").font(ty::BODY).color(p.text_dim).wrap())
            .child(row().w_full().justify_end().child(kit::button(p, "Close", BtnKind::Primary, true, cancel))),
    };
    Some(
        el().key("relocate-layer").pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0)).center()
            .fill(rune::Color::BLACK.alpha(0.55)).on_press(|_: &mut AppState| {}).child(card),
    )
}

fn dialog(p: &Palette, title: &str) -> El {
    col().key("relocate-dialog").w(480.0).padding(22.0).gap(12.0).radius(radius::PANEL)
        .fill(rune::oklcha(0.2, 0.03, 255.0, 0.96))
        .border_gradient(1.0, p.rim_top, p.rim_bottom)
        .shadow(18.0, 50.0, p.shadow)
        .child(text(title.to_owned()).font(ty::TITLE).color(p.text))
}

// -------------------------------------------------------------------------------------------------
// Headless scenes
// -------------------------------------------------------------------------------------------------

fn demo(s: &mut AppState) {
    super::voices::scenes()[0].1(s);
    s.shell.data_dir = PathBuf::from("/home/me/.local/share/mimvox");
}

pub fn scene_confirm(s: &mut AppState) {
    demo(s);
    let facts = Facts { size: Some(31_400_000_000), free: Some(212_000_000_000), done: true };
    s.relocate.phase = Phase::Confirm { target: PathBuf::from("/mnt/d/Voices/mimvox"), facts };
}

pub fn scene_running(s: &mut AppState) {
    demo(s);
    let shared = Shared { progress: Some((42.0, "13.2/31.4 GB".into())), ..Shared::default() };
    s.relocate.phase = Phase::Running { shared };
}

pub fn scene_failed(s: &mut AppState) {
    demo(s);
    s.relocate.phase = Phase::Failed("The move failed (exit code 1): not enough free space on the target drive".into());
}
