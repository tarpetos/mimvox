//! Runs `tools/setup/setup.py --gui` with `uv`; its `@@mimvox {json}` lines are the wire format of `event()` in setup.py.

use super::components::{Tone, icon_el, icons, progress_bar, status_dot};
use super::events;
use super::theme::{Palette, ty};
use crate::state::AppState;
use rune::{El, col, el, row, text};
use serde_json::Value;
use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;

const SETUP_PYTHON: &str = "3.12";
const LOG_KEEP: usize = 400;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StepState {
    Waiting,
    Running,
    Ok,
    Failed,
    Skipped,
}

pub struct Step {
    pub id: String,
    pub label: String,
    pub state: StepState,
    pub message: Option<String>,
    /// Its share of the progress bar.
    pub gb: f32,
    pub done: f32,
    /// False when it found everything in place: left out of the progress.
    pub work: bool,
    /// GB it wrote so far, as setup measures it.
    pub written: Option<f32>,
}

#[derive(Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Idle,
    Running,
    /// True when every step worked.
    Finished(bool),
    Error(String),
    /// Cancel was pressed.
    Stopped,
}

pub const STOPPED: &str = "Setup was stopped. Run it again to continue where it stopped.";

#[derive(Default)]
pub struct SetupRunState {
    pub phase: Phase,
    pub steps: Vec<Step>,
    pub log: VecDeque<String>,
    pub show_log: bool,
    /// `None`: a full setup or a runtime update.
    pub adding: Option<String>,
    pub removing: Option<String>,
    pub left_out: Vec<String>,
}

/// Must match setup.py's COMPONENTS, sizes included.
pub struct Component {
    pub id: &'static str,
    pub label: &'static str,
    pub gb: f32,
    pub what: &'static str,
}

pub const COMPONENTS: [Component; 10] = [
    Component {
        id: "speech",
        label: "Speech engine",
        gb: 9.3,
        what: "The engine and Whisper, which every voice model and Redub use. Always installed.",
    },
    Component {
        id: "omnivoice",
        label: "OmniVoice",
        gb: 3.3,
        what: "Voice cloning in about 60 languages, the fastest. Speak, Book and Redub.",
    },
    Component {
        id: "voxcpm",
        label: "VoxCPM2",
        gb: 6.0,
        what: "Voice cloning in 30 languages: closest to your voice in other languages, slower.",
    },
    Component {
        id: "chatterbox",
        label: "Chatterbox",
        gb: 4.3,
        what: "Voice cloning in 23 languages: very close to your voice in English, slower.",
    },
    Component {
        id: "training",
        label: "Voice training",
        gb: 12.0,
        what: "Train your voice: voice match for trained voices and Redub's Convert voice.",
    },
    Component { id: "translation", label: "Translation", gb: 2.5, what: "Redub's Translate & dub (NLLB-200 600M)." },
    Component {
        id: "better_translation",
        label: "Better translation",
        gb: 2.7,
        what: "Fewer mistranslations in Redub (NLLB-200 1.3B). Installed only with an NVIDIA or AMD GPU.",
    },
    Component {
        id: "native_accent",
        label: "Native accent",
        gb: 2.0,
        what: "English with a native accent for voices that are not trained (Seed-VC, GPL-3.0).",
    },
    Component {
        id: "music_removal",
        label: "Music removal",
        gb: 0.3,
        what: "Keeps only the voice: Redub keeps a video's background, references come out clean.",
    },
    Component {
        id: "ukrainian",
        label: "Ukrainian speech",
        gb: 1.5,
        what: "Ukrainian with exact word stress in Speak, Book and Redub (HolosTTS): every word as the dictionary and the sentence say.",
    },
];

pub const SPEECH_MODELS: [&str; 3] = ["omnivoice", "voxcpm", "chatterbox"];

pub fn component(id: &str) -> Option<&'static Component> {
    COMPONENTS.iter().find(|c| c.id == id)
}

/// True until the engine has said, so nothing flashes at start.
pub fn has(s: &AppState, id: &str) -> bool {
    s.shell.components.as_ref().is_none_or(|c| c.iter().any(|x| x == id))
}

static RUNNING: Mutex<Option<u32>> = Mutex::new(None);

fn uv_path() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(p) = std::env::var_os("MIMVOX_UV").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let res = crate::engine::find_resources();
    if let Some(p) = res.as_ref().map(|r| r.join("bin").join(exe)).filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(p) =
        res.as_ref().map(|r| r.join("bin").join("uv.gz")).filter(|p| p.is_file()).and_then(|gz| unpack_uv(&gz))
    {
        return Some(p);
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    if let Some(p) = std::env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.is_file()) {
        return Some(p);
    }
    let home = dirs::home_dir()?;
    [home.join(".local").join("bin").join(exe), home.join(".cargo").join("bin").join(exe)]
        .into_iter()
        .find(|p| p.is_file())
}

/// The Linux installers carry uv gzipped (the AppImage tool would rewrite a bare one and break it): unpacked once into
/// the user's cache folder, under a name that changes with the packed file.
fn unpack_uv(gz: &std::path::Path) -> Option<PathBuf> {
    let size = std::fs::metadata(gz).ok()?.len();
    let dir = dirs::cache_dir()?.join("mimvox").join(format!("uv-{size}"));
    let target = dir.join("uv");
    if target.is_file() {
        return Some(target);
    }
    std::fs::create_dir_all(&dir).ok()?;
    let tmp = dir.join("uv.part");
    let mut reader = flate2::read::GzDecoder::new(std::fs::File::open(gz).ok()?);
    let mut out = std::fs::File::create(&tmp).ok()?;
    std::io::copy(&mut reader, &mut out).ok()?;
    drop(out);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755)).ok()?;
    }
    std::fs::rename(&tmp, &target).ok()?;
    Some(target)
}

pub fn command(args: &[OsString]) -> Result<Command, String> {
    let res = crate::engine::find_resources()
        .ok_or("The app's setup files are missing (engine, tools). Reinstall Mimvox.")?;
    let script = res.join("tools").join("setup").join("setup.py");
    if !script.is_file() {
        return Err(format!("The setup program is missing: {}. Reinstall Mimvox.", script.display()));
    }
    let uv = uv_path().ok_or("uv (the Python installer setup runs with) was not found. Reinstall Mimvox.")?;
    let mut cmd = Command::new(&uv);
    cmd.args(["run", "--no-project", "--python", SETUP_PYTHON, "python", "-u"])
        .arg(&script)
        .args(args)
        .current_dir(&res)
        .env("MIMVOX_UV", &uv)
        .env("PYTHONUNBUFFERED", "1")
        // No __pycache__ next to the installed sources.
        .env("PYTHONPYCACHEPREFIX", super::onboarding::data_dir().join("cache").join("pycache"))
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0); // its own group, so Cancel stops everything it started
    }
    Ok(cmd)
}

pub fn running(s: &AppState) -> bool {
    s.setup.phase == Phase::Running
}

pub fn start(s: &mut AppState, data_dir: &str) {
    let chosen: Vec<&str> =
        COMPONENTS.iter().map(|c| c.id).filter(|id| !s.setup.left_out.iter().any(|x| x == id)).collect();
    let mut args = base_args(data_dir);
    args.push("--components".into());
    args.push(chosen.join(",").into());
    start_with(s, args, None, data_dir);
}

pub fn update_runtime(s: &mut AppState) {
    let dir = super::onboarding::data_dir().display().to_string();
    start_with(s, base_args(&dir), None, &dir);
}

pub fn install(s: &mut AppState, id: &str) {
    let dir = super::onboarding::data_dir().display().to_string();
    let mut args = base_args(&dir);
    args.push("--add".into());
    args.push(id.into());
    start_with(s, args, Some(id.to_owned()), &dir);
}

pub fn remove(s: &mut AppState, id: &str) {
    if running(s) || s.shell.scripted() {
        return;
    }
    let dir = super::onboarding::data_dir().display().to_string();
    let mut args = base_args(&dir);
    args.push("--remove".into());
    args.push(id.into());
    start_with(s, args, None, &dir);
    s.setup.removing = Some(id.to_owned());
}

fn base_args(data_dir: &str) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["--gui".into()];
    if !data_dir.trim().is_empty() {
        args.push("--data-dir".into());
        args.push(data_dir.trim().into());
    }
    args
}

/// Does nothing while one runs.
fn start_with(s: &mut AppState, args: Vec<OsString>, adding: Option<String>, data_dir: &str) {
    if running(s) || s.shell.scripted() {
        return;
    }
    *CANCELLED.lock().unwrap_or_else(|e| e.into_inner()) = false;
    // Setup may update the engine's environment, and Windows refuses to replace files a running engine holds open.
    if let Some(old) = super::api::detach(s) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| old.shutdown()));
        s.shell.engine = super::shell::EngineStatus::Stopped { code: None, detail: "Stopped while setup runs.".into() };
    }
    let left_out = std::mem::take(&mut s.setup.left_out);
    s.setup =
        SetupRunState { phase: Phase::Running, show_log: s.setup.show_log, adding, left_out, ..Default::default() };
    let cache = PathBuf::from(data_dir.trim()).join("cache").join("uv");
    let spawned = command(&args).and_then(|mut c| {
        if !data_dir.trim().is_empty() {
            if std::env::var_os("UV_CACHE_DIR").is_none() {
                c.env("UV_CACHE_DIR", &cache);
            }
            // The Python this runs on stays in the data folder too, next to the ones setup.py installs there.
            if std::env::var_os("UV_PYTHON_INSTALL_DIR").is_none() {
                c.env("UV_PYTHON_INSTALL_DIR", PathBuf::from(data_dir.trim()).join("runtime").join("python"));
            }
            c.env("PYTHONPYCACHEPREFIX", PathBuf::from(data_dir.trim()).join("cache").join("pycache"));
        }
        c.spawn().map_err(|e| format!("Setup could not start: {e}"))
    });
    let mut child = match spawned {
        Ok(c) => c,
        Err(e) => {
            s.setup.phase = Phase::Error(e);
            return;
        }
    };
    *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = Some(child.id());
    let streams: Vec<Box<dyn std::io::Read + Send>> = [
        child.stdout.take().map(|o| Box::new(o) as Box<dyn std::io::Read + Send>),
        child.stderr.take().map(|e| Box::new(e) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .flatten()
    .collect();
    let readers: Vec<_> = streams
        .into_iter()
        .map(|stream| {
            std::thread::spawn(move || {
                // Progress bars rewrite a line with \r: only the last piece is kept.
                for line in BufReader::new(stream).split(b'\n').map_while(Result::ok) {
                    let line = String::from_utf8_lossy(&line).into_owned();
                    let line = line.rsplit('\r').find(|p| !p.trim().is_empty()).unwrap_or("").trim_end().to_owned();
                    if !line.is_empty() {
                        events::post(move |s| on_line(s, line));
                    }
                }
            })
        })
        .collect();
    std::thread::spawn(move || {
        let status = child.wait();
        for r in readers {
            let _ = r.join();
        }
        *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let code = status.as_ref().ok().and_then(|s| s.code());
        // A process killed by a signal has no exit code: say which one (Cancel sends SIGTERM).
        #[cfg(unix)]
        let signal = {
            use std::os::unix::process::ExitStatusExt;
            status.as_ref().ok().and_then(|s| s.signal())
        };
        #[cfg(not(unix))]
        let signal: Option<i32> = None;
        events::post(move |s| on_exit(s, code, signal));
    });
}

fn on_line(s: &mut AppState, line: String) {
    if let Some(json) = line.strip_prefix("@@mimvox ") {
        if let Ok(v) = serde_json::from_str::<Value>(json) {
            on_event(s, &v);
        }
        return;
    }
    let line = strip_ansi(&line);
    let line = line.strip_prefix("[setup] ").map(str::to_owned).unwrap_or(line);
    // Advice meant for a developer's terminal, not for this screen.
    let line = match line.find("Warning: You are sending unauthenticated requests") {
        Some(at) => line[..at].trim_end().to_owned(),
        None => line,
    };
    if line.contains("hf skills add") || line.trim().is_empty() {
        return;
    }
    let line = tidy_progress(&line).unwrap_or(line);
    s.setup.log.push_back(line);
    while s.setup.log.len() > LOG_KEEP {
        s.setup.log.pop_front();
    }
}

/// A tqdm bar ("model.safetensors:  41%|████      | 1.35G/3.30G [00:42<01:01, 31.8MB/s]") as
/// "model.safetensors: 41% · 1.35G/3.30G · 01:01 left".
fn tidy_progress(line: &str) -> Option<String> {
    let (head, rest) = line.split_once("%|")?;
    let (desc, pct) = head.rsplit_once(':')?;
    let pct = pct.trim();
    if pct.is_empty() || !pct.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let (_, tail) = rest.split_once('|')?;
    let (counts, timing) = tail.split_once('[').unwrap_or((tail, ""));
    let mut out = format!("{}: {pct}%", desc.trim());
    let counts = counts.trim();
    if !counts.is_empty() {
        out.push_str(&format!(" · {counts}"));
    }
    let left = timing.split_once('<').and_then(|(_, r)| r.split([',', ']']).next()).map(str::trim);
    if let Some(left) = left.filter(|l| !l.is_empty() && !l.contains('?')) {
        out.push_str(&format!(" · {left} left"));
    }
    Some(out)
}

/// The text without terminal colour codes (`ESC [ ... m` and the like).
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                // Parameters, then one final letter.
                for d in chars.by_ref() {
                    if d.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

fn on_event(s: &mut AppState, v: &Value) {
    let str_of = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
    match v.get("event").and_then(Value::as_str) {
        Some("plan") => {
            s.setup.steps = v
                .get("steps")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|st| Step {
                            id: st.get("id").and_then(Value::as_str).unwrap_or("").to_owned(),
                            label: st.get("label").and_then(Value::as_str).unwrap_or("").to_owned(),
                            state: StepState::Waiting,
                            message: None,
                            gb: st.get("gb").and_then(Value::as_f64).unwrap_or(0.2) as f32,
                            done: 0.0,
                            work: true,
                            written: None,
                        })
                        .collect()
                })
                .unwrap_or_default();
        }
        Some("step") => {
            let id = str_of("id");
            let state = match str_of("state").as_str() {
                "run" => StepState::Running,
                "ok" => StepState::Ok,
                "fail" => StepState::Failed,
                _ => StepState::Skipped,
            };
            if let Some(step) = s.setup.steps.iter_mut().find(|st| st.id == id) {
                step.state = state;
                step.message = v.get("message").and_then(Value::as_str).map(str::to_owned);
                step.work = v.get("work").and_then(Value::as_bool).unwrap_or(true);
                if let Some(gb) = v.get("gb").and_then(Value::as_f64) {
                    step.written = Some(gb as f32);
                }
            }
        }
        Some("progress") => {
            let id = str_of("id");
            if let Some(step) = s.setup.steps.iter_mut().find(|st| st.id == id) {
                step.done = v.get("done").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 1.0) as f32;
                step.written = v.get("gb").and_then(Value::as_f64).map(|g| g as f32);
            }
        }
        Some("done") => s.setup.phase = Phase::Finished(v.get("ok").and_then(Value::as_bool).unwrap_or(false)),
        _ => {}
    }
}

fn on_exit(s: &mut AppState, code: Option<i32>, signal: Option<i32>) {
    if s.setup.phase == Phase::Running {
        // Ended without its final report: stopped, or failed before the steps (the log says why).
        let last = s.setup.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
        let cancelled = std::mem::take(&mut *CANCELLED.lock().unwrap_or_else(|e| e.into_inner()));
        s.setup.phase = match code {
            Some(0) => Phase::Finished(true),
            _ if cancelled => Phase::Stopped,
            None => Phase::Error(match signal {
                Some(sig) => format!(
                    "Setup crashed (signal {sig}). Run it again; if it keeps crashing, report it on the project's GitHub page."
                ),
                None => "Setup ended unexpectedly. Run it again to continue where it stopped.".into(),
            }),
            Some(c) if last.is_empty() => Phase::Error(format!("Setup stopped (exit code {c}).")),
            Some(_) => Phase::Error(last),
        };
    }
    let stopped = s.setup.phase == Phase::Stopped;
    for step in s.setup.steps.iter_mut().filter(|st| st.state == StepState::Running) {
        step.state = if stopped { StepState::Waiting } else { StepState::Failed };
    }
    // The engine reads what is installed only when it starts, so it is restarted.
    let ok = s.setup.phase == Phase::Finished(true);
    super::onboarding::recheck(s);
    if s.onboarding.runtime_present && (ok || !super::shell::engine_ready(s)) {
        super::shell::restart_engine(s);
    }
    if ok {
        if let Some(c) = s.setup.adding.as_deref().and_then(component) {
            super::shell::toast(s, Tone::Success, format!("{} is installed.", c.label));
        }
        if let Some(c) = s.setup.removing.as_deref().and_then(component) {
            super::shell::toast(s, Tone::Success, format!("{} is removed.", c.label));
        }
    }
}

pub fn cancel(_s: &mut AppState) {
    stop_on_exit();
}

/// Cancel was pressed, so the end of the process is not a crash.
static CANCELLED: Mutex<bool> = Mutex::new(false);

/// Called on Cancel and by `main` when the app closes.
pub fn stop_on_exit() {
    *CANCELLED.lock().unwrap_or_else(|e| e.into_inner()) = true;
    let Some(pid) = *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) else { return };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ =
            Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).creation_flags(0x0800_0000).status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-TERM", "--", &format!("-{pid}")]).status();
    }
}

pub fn progress(s: &AppState) -> f32 {
    let counted = s.setup.steps.iter().filter(|st| st.work || !matches!(st.state, StepState::Ok | StepState::Skipped));
    let (mut total, mut done) = (0.0, 0.0);
    // A finished step weighs what it wrote and a running one at least that, so a size guessed wrong never jumps.
    for st in counted {
        let (weight, part) = match (st.state, st.written) {
            (StepState::Waiting, _) => (st.gb, 0.0),
            (StepState::Running, Some(w)) => {
                let weight = st.gb.max(w);
                (weight, w.min(weight * 0.97))
            }
            (StepState::Running, None) => (st.gb, st.gb * st.done),
            (_, Some(w)) => (w.max(0.01), w.max(0.01)),
            (_, None) => (st.gb, st.gb),
        };
        total += weight;
        done += part;
    }
    if total > 0.0 { (done / total).clamp(0.0, 1.0) } else { 0.0 }
}

pub fn current(s: &AppState) -> Option<&Step> {
    s.setup.steps.iter().find(|st| st.state == StepState::Running)
}

fn size(gb: f32) -> String {
    if gb < 1.0 { format!("{:.0} MB", gb * 1000.0) } else { format!("{gb:.1} GB") }
}

/// The running step's bytes against its size, then the last log line unless it is a download counter.
pub fn status_line(s: &AppState) -> String {
    let line = last_line(s).unwrap_or("");
    let counter = line.contains("% ·") || line.trim_end().ends_with('%');
    let written = current(s).and_then(|st| st.written.filter(|w| *w >= 0.001).map(|w| (w, st.gb.max(w))));
    match written {
        Some((w, of)) if counter || line.is_empty() => format!("Downloaded {} of about {}", size(w), size(of)),
        Some((w, of)) => format!("{} of about {} · {line}", size(w), size(of)),
        None => line.to_owned(),
    }
}

pub fn last_line(s: &AppState) -> Option<&str> {
    s.setup.log.iter().rev().map(String::as_str).find(|l| !l.trim().is_empty())
}

pub fn panel(s: &AppState, p: Palette) -> El {
    let mut list = col().key("setup-steps").w_full().gap(6.0);
    for st in &s.setup.steps {
        let mark = match st.state {
            StepState::Waiting => el().size(10.0, 10.0).radius(5.0).border(1.0, p.text_faint),
            StepState::Running => status_dot(p, Tone::Accent, true),
            StepState::Ok => icon_el(&icons::CHECK, 13.0, p.ok),
            StepState::Failed => icon_el(&icons::ALERT, 13.0, p.bad),
            StepState::Skipped => el().size(10.0, 2.0).fill(p.text_faint),
        };
        let color = match st.state {
            StepState::Waiting | StepState::Skipped => p.text_faint,
            StepState::Failed => p.bad,
            _ => p.text,
        };
        let mut r = col().w_full().gap(2.0).child(
            row()
                .w_full()
                .gap(10.0)
                .center_y()
                .child(el().w(16.0).center().child(mark))
                .child(text(st.label.clone()).font(ty::LABEL).color(color)),
        );
        if let Some(m) = st.message.as_ref().filter(|_| st.state == StepState::Failed) {
            r = r.child(text(m.clone()).font(ty::CAPTION).color(p.text_dim).wrap().pl(26.0));
        }
        list = list.child(r);
    }
    let mut c = col().key("setup-panel").w_full().gap(12.0);
    if running(s) {
        c = c.child(progress_bar(p, (!s.setup.steps.is_empty()).then(|| progress(s))));
    }
    c = c.child(list);
    let line = status_line(s);
    if running(s) && !line.is_empty() {
        c = c.child(text(line).font(ty::CAPTION).color(p.text_dim).truncate());
    }
    if s.setup.show_log && !s.setup.log.is_empty() {
        // One line each, cut with an ellipsis: a long path or a progress bar never widens the card.
        let start = s.setup.log.len().saturating_sub(14);
        let mut body = col().key("setup-log").w_full().padding(10.0).radius(8.0).fill(p.well).gap(2.0);
        for line in s.setup.log.iter().skip(start) {
            body = body.child(text(line.clone()).font(ty::CAPTION).color(p.text_dim).truncate().w_full().min_w(0.0));
        }
        c = c.child(body);
    }
    c
}

pub fn toggle_log(s: &mut AppState) {
    s.setup.show_log = !s.setup.show_log;
}

pub fn scene_running(s: &mut AppState) {
    let states = [
        StepState::Ok,
        StepState::Ok,
        StepState::Ok,
        StepState::Running,
        StepState::Waiting,
        StepState::Waiting,
        StepState::Waiting,
        StepState::Waiting,
    ];
    let plan = [
        ("toolchain", "System tools"),
        ("applio_source", "Voice training tools: source"),
        ("engine_env", "Speech engine"),
        ("omnivoice", "OmniVoice: model"),
        ("speech_models", "Transcription and translation models"),
        ("zero_shot_vc", "Native accent converter"),
        ("applio_env", "Voice training tools: environment"),
        ("runtime_toml", "Finishing"),
    ];
    s.setup.phase = Phase::Running;
    s.setup.steps = plan
        .iter()
        .zip(states)
        .map(|((id, label), state)| Step {
            id: (*id).into(),
            label: (*label).into(),
            done: if state == StepState::Running { 0.41 } else { 0.0 },
            state,
            message: None,
            gb: 1.0,
            work: true,
            written: None,
        })
        .collect();
    s.setup.log = [
        "== Speech model ==",
        "OmniVoice model: downloading k2-fsa/OmniVoice (about 3.3 GB, resumable)...",
        "model.safetensors:  41%|████      | 1.35G/3.30G [00:42<01:01, 31.8MB/s]",
    ]
    .iter()
    .map(|l| (*l).to_owned())
    .collect();
    s.setup.show_log = true;
}

pub fn chosen_gb(s: &AppState) -> f32 {
    COMPONENTS.iter().filter(|c| c.id == "speech" || !s.setup.left_out.iter().any(|x| x == c.id)).map(|c| c.gb).sum()
}

pub fn chooser(s: &AppState, p: Palette) -> El {
    use super::voices::kit;
    let left_out = |id: &str| s.setup.left_out.iter().any(|x| x == id);
    let models_on = SPEECH_MODELS.iter().filter(|m| !left_out(m)).count();
    let mut total = 0.0;
    let mut row_for = |c: &'static Component| {
        let id = c.id;
        let required = id == "speech";
        let on = required || !left_out(id);
        if on {
            total += c.gb;
        }
        let label = format!("{} · {:.1} GB{}", c.label, c.gb, if required { " · required" } else { "" });
        let last_model = on && models_on == 1 && SPEECH_MODELS.contains(&id);
        let locked = required || last_model;
        let mut check = kit::checkbox_if(&p, on, !locked, &label, move |s: &mut AppState| {
            if let Some(i) = s.setup.left_out.iter().position(|x| x == id) {
                s.setup.left_out.remove(i);
            } else {
                s.setup.left_out.push(id.to_owned());
            }
        });
        if locked {
            check = check.help(if required {
                "Always installed: everything else builds on it."
            } else {
                "At least one voice-cloning model is installed."
            });
        }
        col()
            .key(format!("component-{id}"))
            .w_full()
            .gap(2.0)
            .child(check)
            .child(text(c.what).font(ty::CAPTION).color(p.text_faint).wrap().pl(32.0))
    };
    let mut list = col().key("setup-components").w_full().gap(10.0);
    let speech = COMPONENTS.iter().filter(|c| c.id == "speech");
    for c in speech {
        list = list.child(row_for(c));
    }
    list = list.child(text("Voice-cloning models · at least one").font(ty::OVERLINE).color(p.text_faint).pt(4.0));
    for c in COMPONENTS.iter().filter(|c| SPEECH_MODELS.contains(&c.id)) {
        list = list.child(row_for(c));
    }
    list = list.child(text("More").font(ty::OVERLINE).color(p.text_faint).pt(4.0));
    for c in COMPONENTS.iter().filter(|c| c.id != "speech" && !SPEECH_MODELS.contains(&c.id)) {
        list = list.child(row_for(c));
    }
    let all = s.setup.left_out.is_empty();
    let keep = SPEECH_MODELS.iter().copied().find(|m| !left_out(m)).unwrap_or("omnivoice");
    let toggle_all = move |s: &mut AppState| {
        s.setup.left_out = if all {
            COMPONENTS.iter().filter(|c| c.id != "speech" && c.id != keep).map(|c| c.id.to_owned()).collect()
        } else {
            Vec::new()
        };
    };
    col()
        .w_full()
        .gap(10.0)
        .child(
            row()
                .w_full()
                .center_y()
                .justify_between()
                .child(text(format!("What to install · about {total:.0} GB")).font(ty::LABEL).color(p.text))
                .child(
                    super::components::Button::new(if all { "Only what is required" } else { "Everything" })
                        .ghost()
                        .small()
                        .key(if all { "components-required" } else { "components-all" })
                        .on_press(toggle_all)
                        .build(p),
                ),
        )
        .child(list)
        .child(
            text("Anything left out can be installed later in one click, on the screen that needs it.")
                .font(ty::CAPTION)
                .color(p.text_dim)
                .wrap(),
        )
}

fn turned_off(s: &AppState, id: &str) -> bool {
    s.shell.components_off.iter().any(|c| c == id)
}

pub fn install_card(s: &AppState, p: Palette, id: &str, needed_for: &str) -> El {
    use super::components::{Button, glass_card};
    let Some(c) = component(id) else { return el() };
    if turned_off(s, id) {
        let row_ = row()
            .key(format!("install-card-{id}"))
            .w_full()
            .px(14.0)
            .py(12.0)
            .gap(12.0)
            .center_y()
            .child(icon_el(&icons::SPARKLE, 16.0, p.accent))
            .child(
                col()
                    .grow(1.0)
                    .min_w(0.0)
                    .gap(4.0)
                    .child(text(format!("{} is turned off", c.label)).font(ty::LABEL).color(p.text))
                    .child(
                        text(
                            "Its files are still installed. Turn it on in Settings (the status button), under Modules.",
                        )
                        .font(ty::CAPTION)
                        .color(p.text_dim)
                        .wrap(),
                    ),
            )
            .child(
                Button::new("Modules")
                    .ghost()
                    .small()
                    .key(format!("install-{id}-modules"))
                    .on_press_tx(super::settings::open_modules)
                    .build(p),
            );
        return glass_card(p).w_full().child(row_);
    }
    let this = s.setup.adding.as_deref() == Some(id);
    let mut info = col().grow(1.0).min_w(0.0).gap(4.0);
    let button: Option<El> = if running(s) && this {
        let step = current(s).map(|st| st.label.clone()).unwrap_or_else(|| "Starting…".into());
        info = info
            .child(text(format!("Installing {}: {step}", c.label)).font(ty::LABEL).color(p.text))
            .child(text(status_line(s)).font(ty::CAPTION).color(p.text_dim).truncate())
            .child(el().w_full().pt(4.0).child(progress_bar(p, Some(progress(s)))));
        Some(Button::new("Cancel").ghost().small().key(format!("install-{id}-cancel")).on_press(cancel).build(p))
    } else {
        let stopped = this && s.setup.phase == Phase::Stopped;
        let failed = this && matches!(s.setup.phase, Phase::Error(_) | Phase::Finished(false));
        info = info.child(text(format!("{} is not installed", c.label)).font(ty::LABEL).color(p.text)).child(
            text(format!(
                "{}{} About {:.1} GB; it installs in the background, and the engine restarts when it is done.",
                if needed_for.is_empty() { String::new() } else { format!("{needed_for} ") },
                c.what,
                c.gb
            ))
            .font(ty::CAPTION)
            .color(p.text_dim)
            .wrap(),
        );
        if failed {
            let why = match &s.setup.phase {
                Phase::Error(e) => e.clone(),
                _ => s.setup.steps.iter().find_map(|st| st.message.clone()).unwrap_or_else(|| "A step failed.".into()),
            };
            info = info.child(text(format!("The last attempt stopped: {why}")).font(ty::CAPTION).color(p.bad).wrap());
        } else if stopped {
            info = info.child(text(STOPPED).font(ty::CAPTION).color(p.text_dim).wrap());
        }
        let id_owned = id.to_owned();
        Some(
            Button::new(if failed { "Try again".to_owned() } else { format!("Install · {:.1} GB", c.gb) })
                .primary()
                .small()
                .icon(&icons::SPARKLE)
                .key(format!("install-{id}"))
                .disabled(running(s))
                .help(if running(s) { "Another setup is running; this can start when it is done." } else { "" })
                .on_press(move |s| install(s, &id_owned))
                .build(p),
        )
    };
    let row_ = row()
        .key(format!("install-card-{id}"))
        .w_full()
        .px(14.0)
        .py(12.0)
        .gap(12.0)
        .center_y()
        .child(icon_el(&icons::SPARKLE, 16.0, p.accent))
        .child(info)
        .children(button);
    glass_card(p).w_full().child(row_)
}

fn mid_sentence(label: &str) -> String {
    if label.starts_with("Ukrainian") {
        return label.to_owned();
    }
    let mut chars = label.chars();
    chars.next().map(|c| c.to_lowercase().chain(chars).collect()).unwrap_or_default()
}

pub fn install_line(s: &AppState, p: Palette, id: &str, what: &str) -> El {
    use super::components::Button;
    let Some(c) = component(id) else { return el() };
    let this = s.setup.adding.as_deref() == Some(id) && running(s);
    let id_owned = id.to_owned();
    if turned_off(s, id) {
        return row()
            .key(format!("install-line-{id}"))
            .w_full()
            .gap(8.0)
            .center_y()
            .child(
                text(format!("{what}: {} is turned off.", mid_sentence(c.label)))
                    .font(ty::CAPTION)
                    .color(p.text_dim)
                    .grow(1.0)
                    .min_w(0.0)
                    .wrap(),
            )
            .child(
                Button::new("Modules")
                    .ghost()
                    .small()
                    .key(format!("install-line-{id}-modules"))
                    .on_press_tx(super::settings::open_modules)
                    .build(p),
            );
    }
    let line = if this {
        format!("Installing {}… {:.0}%", c.label, progress(s) * 100.0)
    } else {
        format!("{what} needs {} ({:.1} GB).", mid_sentence(c.label), c.gb)
    };
    row()
        .key(format!("install-line-{id}"))
        .w_full()
        .gap(8.0)
        .center_y()
        .child(text(line).font(ty::CAPTION).color(p.text_dim).grow(1.0).min_w(0.0).wrap())
        .children((!this).then(|| {
            Button::new("Install")
                .ghost()
                .small()
                .key(format!("install-line-{id}-go"))
                .disabled(running(s))
                .on_press(move |s| install(s, &id_owned))
                .build(p)
        }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(s: &mut AppState, v: serde_json::Value) {
        on_line(s, format!("@@mimvox {v}"));
    }

    #[test]
    fn progress_follows_the_running_step_and_skips_steps_already_done() {
        let mut s = AppState::default();
        s.setup.phase = Phase::Running;
        let plan = serde_json::json!({"event": "plan", "steps": [
            {"id": "engine_env", "label": "Speech engine", "gb": 7.7},
            {"id": "chatterbox_env", "label": "Chatterbox: environment", "gb": 1.3},
            {"id": "chatterbox_models", "label": "Chatterbox: model", "gb": 3.0},
        ]});
        feed(&mut s, plan);
        assert_eq!(progress(&s), 0.0);
        feed(&mut s, serde_json::json!({"event": "step", "id": "engine_env", "state": "ok", "work": false}));
        assert_eq!(progress(&s), 0.0);
        feed(&mut s, serde_json::json!({"event": "step", "id": "chatterbox_env", "state": "run"}));
        feed(&mut s, serde_json::json!({"event": "progress", "id": "chatterbox_env", "done": 0.5}));
        assert!((progress(&s) - 0.65 / 4.3).abs() < 1e-4);
    }

    #[test]
    fn a_step_that_wrote_less_than_its_size_does_not_jump_the_bar() {
        let mut s = AppState::default();
        s.setup.phase = Phase::Running;
        let plan = serde_json::json!({"event": "plan", "steps": [
            {"id": "holos_env", "label": "Ukrainian speech: environment", "gb": 0.6},
            {"id": "holos_models", "label": "Ukrainian speech: models", "gb": 0.9},
            {"id": "chatterbox_env", "label": "Chatterbox: environment", "gb": 0.05},
        ]});
        feed(&mut s, plan);
        feed(&mut s, serde_json::json!({"event": "step", "id": "holos_env", "state": "run"}));
        feed(&mut s, serde_json::json!({"event": "progress", "id": "holos_env", "done": 0.17, "gb": 0.1}));
        let running = progress(&s);
        feed(&mut s, serde_json::json!({"event": "step", "id": "holos_env", "state": "ok", "work": true, "gb": 0.1}));
        let after = progress(&s);
        assert!(after >= running && after - running < 0.05, "{running} -> {after}");
    }

    #[test]
    fn the_status_line_counts_what_the_step_downloaded() {
        let mut s = AppState::default();
        s.setup.phase = Phase::Running;
        feed(
            &mut s,
            serde_json::json!({"event": "plan", "steps": [{"id": "holos_models", "label": "Ukrainian speech: models", "gb": 0.9}]}),
        );
        feed(&mut s, serde_json::json!({"event": "step", "id": "holos_models", "state": "run"}));
        on_line(&mut s, "Fetching 3 files:   0%|          | 0/3 [00:00<?, ?it/s]".into());
        assert_eq!(status_line(&s), "Fetching 3 files: 0% · 0/3");
        feed(&mut s, serde_json::json!({"event": "progress", "id": "holos_models", "done": 0.46, "gb": 0.412}));
        assert_eq!(status_line(&s), "Downloaded 412 MB of about 900 MB");
        on_line(&mut s, "Ukrainian speech: downloading Stanza's Ukrainian models (about 0.2 GB)...".into());
        assert_eq!(
            status_line(&s),
            "412 MB of about 900 MB · Ukrainian speech: downloading Stanza's Ukrainian models (about 0.2 GB)..."
        );
    }

    #[test]
    fn a_tqdm_bar_reads_as_text() {
        let bar = "model.safetensors:  41%|████      | 1.35G/3.30G [00:42<01:01, 31.8MB/s]";
        assert_eq!(tidy_progress(bar).as_deref(), Some("model.safetensors: 41% · 1.35G/3.30G · 01:01 left"));
        let files = "Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]";
        assert_eq!(tidy_progress(files).as_deref(), Some("Fetching 13 files: 0% · 0/13"));
        assert_eq!(tidy_progress("Downloading torch (782.3MiB)"), None);
    }

    #[test]
    fn the_hub_token_advice_is_cut_from_a_line() {
        let mut s = AppState::default();
        on_line(
            &mut s,
            "Fetching 13 files:   0%|  | 0/13Warning: You are sending unauthenticated requests to the HF Hub.".into(),
        );
        assert_eq!(s.setup.log.back().map(String::as_str), Some("Fetching 13 files: 0% · 0/13"));
    }
}
