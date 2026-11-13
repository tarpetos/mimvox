//! Setup run by the app itself: no terminal window, no script to find or start by hand.
//!
//! The app starts `tools/setup/setup.py --gui` (from the bundled resources) with `uv` (bundled in the installers under
//! `<resources>/bin`, else the one on PATH) and reads its output: readable lines become the log, and `@@mimvox {json}`
//! lines (`plan`, `step`, `done`; see `event()` in setup.py) drive the step list and the progress bar. `--gui` asks no
//! questions and installs no system packages, so nothing needs a terminal or administrator rights.
//!
//! Cancel stops the whole process tree (uv, Python and whatever they started); setup resumes where it stopped on the
//! next run. Closing the app stops a running setup the same way ([`stop_on_exit`]).
//! `setup.sh` / `setup.cmd` stay in the repository for developers only.

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

/// Python version setup runs on (uv downloads it when the system has none).
const SETUP_PYTHON: &str = "3.12";
/// Log lines kept for the details view.
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
}

#[derive(Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Idle,
    Running,
    /// Setup ended: true when every step worked.
    Finished(bool),
    /// Setup could not start, or ended without its final report.
    Error(String),
}

#[derive(Default)]
pub struct SetupRunState {
    pub phase: Phase,
    pub steps: Vec<Step>,
    pub log: VecDeque<String>,
    pub show_log: bool,
}

/// The running setup's process id (for Cancel and for closing the app).
static RUNNING: Mutex<Option<u32>> = Mutex::new(None);

/// The `uv` setup runs with: `MIMVOX_UV`, the installer's copy, then PATH and uv's usual install folders.
fn uv_path() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(p) = std::env::var_os("MIMVOX_UV").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(p) = crate::engine::find_resources().map(|r| r.join("bin").join(exe)).filter(|p| p.is_file()) {
        return Some(p);
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    if let Some(p) = std::env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.is_file()) {
        return Some(p);
    }
    let home = dirs::home_dir()?;
    [home.join(".local").join("bin").join(exe), home.join(".cargo").join("bin").join(exe)].into_iter().find(|p| p.is_file())
}

/// `uv run ... python -u tools/setup/setup.py <args>` in the resources folder, with no console window and the output
/// piped. Also used by the data-folder move (`ui::relocate`).
pub fn command(args: &[OsString]) -> Result<Command, String> {
    let res = crate::engine::find_resources().ok_or("The app's setup files are missing (engine, tools). Reinstall Mimvox.")?;
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

/// Starts setup into `data_dir`. Does nothing while one runs.
pub fn start(s: &mut AppState, data_dir: &str) {
    if running(s) || s.shell.scripted() {
        return;
    }
    s.setup = SetupRunState { phase: Phase::Running, show_log: s.setup.show_log, ..Default::default() };
    let mut args: Vec<OsString> = vec!["--gui".into()];
    if !data_dir.trim().is_empty() {
        args.push("--data-dir".into());
        args.push(data_dir.trim().into());
    }
    let mut child = match command(&args).and_then(|mut c| c.spawn().map_err(|e| format!("Setup could not start: {e}"))) {
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
                // Blocks on each line; progress bars rewrite a line with \r: only the last piece is kept.
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
        let code = status.ok().and_then(|s| s.code());
        events::post(move |s| on_exit(s, code));
    });
}

fn on_line(s: &mut AppState, line: String) {
    if let Some(json) = line.strip_prefix("@@mimvox ") {
        if let Ok(v) = serde_json::from_str::<Value>(json) {
            on_event(s, &v);
        }
        return;
    }
    let line = line.strip_prefix("[setup] ").map(str::to_owned).unwrap_or(line);
    s.setup.log.push_back(line);
    while s.setup.log.len() > LOG_KEEP {
        s.setup.log.pop_front();
    }
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
            }
        }
        Some("done") => s.setup.phase = Phase::Finished(v.get("ok").and_then(Value::as_bool).unwrap_or(false)),
        _ => {}
    }
}

fn on_exit(s: &mut AppState, code: Option<i32>) {
    if s.setup.phase == Phase::Running {
        // Ended without its final report: stopped, or failed before the steps (the log says why).
        let last = s.setup.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
        s.setup.phase = match code {
            Some(0) => Phase::Finished(true),
            None => Phase::Error("Setup was stopped. Run it again to continue where it stopped.".into()),
            Some(c) if last.is_empty() => Phase::Error(format!("Setup stopped (exit code {c}).")),
            Some(_) => Phase::Error(last),
        };
    }
    for step in s.setup.steps.iter_mut().filter(|st| st.state == StepState::Running) {
        step.state = StepState::Failed;
    }
    // A runtime that is now there (or up to date): the onboarding screen and the engine follow.
    super::onboarding::recheck(s);
    if s.onboarding.runtime_present && !super::shell::engine_ready(s) {
        super::shell::restart_engine(s);
    }
}

/// Stops the running setup and everything it started (it resumes on the next run).
pub fn cancel(_s: &mut AppState) {
    stop_on_exit();
}

/// Called on Cancel and by `main` when the app closes.
pub fn stop_on_exit() {
    let Some(pid) = *RUNNING.lock().unwrap_or_else(|e| e.into_inner()) else { return };
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"]).creation_flags(0x0800_0000).status();
    }
    #[cfg(unix)]
    {
        let _ = Command::new("kill").args(["-TERM", "--", &format!("-{pid}")]).status();
    }
}

/// Rough share of the work per step (download and install size), for the progress bar.
fn weight(id: &str) -> f32 {
    match id {
        "engine_env" => 6.0,
        "omnivoice" => 3.3,
        "speech_models" => 4.5,
        "better_translation" => 5.5,
        "zero_shot_vc" => 2.0,
        "applio_env" => 8.0,
        "applio_models" => 4.0,
        "separation" => 0.3,
        _ => 0.2,
    }
}

/// Share of the plan that is done (0..1).
pub fn progress(s: &AppState) -> f32 {
    let total: f32 = s.setup.steps.iter().map(|st| weight(&st.id)).sum();
    let done: f32 = s.setup.steps.iter().filter(|st| !matches!(st.state, StepState::Waiting | StepState::Running)).map(|st| weight(&st.id)).sum();
    if total > 0.0 { done / total } else { 0.0 }
}

/// The step that runs now.
pub fn current(s: &AppState) -> Option<&Step> {
    s.setup.steps.iter().find(|st| st.state == StepState::Running)
}

/// The latest log line (what setup is doing right now).
pub fn last_line(s: &AppState) -> Option<&str> {
    s.setup.log.iter().rev().map(String::as_str).find(|l| !l.trim().is_empty())
}

/// The step list, the progress bar, the latest line and (on request) the log.
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
            row().w_full().gap(10.0).center_y().child(el().w(16.0).center().child(mark)).child(text(st.label.clone()).font(ty::LABEL).color(color)),
        );
        if let Some(m) = st.message.as_ref().filter(|_| st.state == StepState::Failed) {
            r = r.child(text(m.clone()).font(ty::CAPTION).color(p.text_dim).wrap().pl(26.0));
        }
        list = list.child(r);
    }
    let mut c = col().key("setup-panel").w_full().gap(12.0);
    if !s.setup.steps.is_empty() {
        c = c.child(progress_bar(p, Some(progress(s))));
    } else if running(s) {
        c = c.child(progress_bar(p, None));
    }
    c = c.child(list);
    if running(s) {
        if let Some(line) = last_line(s) {
            c = c.child(text(line.to_owned()).font(ty::CAPTION).color(p.text_dim).truncate());
        }
    }
    if s.setup.show_log && !s.setup.log.is_empty() {
        let lines: Vec<&str> = s.setup.log.iter().rev().take(14).map(String::as_str).collect();
        let body = lines.into_iter().rev().collect::<Vec<_>>().join("\n");
        c = c.child(
            el().key("setup-log").w_full().padding(10.0).radius(8.0).fill(p.well)
                .child(text(body).font(ty::CAPTION).color(p.text_dim).wrap()),
        );
    }
    c
}

pub fn toggle_log(s: &mut AppState) {
    s.setup.show_log = !s.setup.show_log;
}

/// Scene helper: setup halfway through, with the log open.
pub fn scene_running(s: &mut AppState) {
    let states = [StepState::Ok, StepState::Ok, StepState::Ok, StepState::Running, StepState::Waiting, StepState::Waiting, StepState::Waiting, StepState::Waiting];
    let plan = [
        ("toolchain", "System tools"),
        ("applio_source", "Voice training tools: source"),
        ("engine_env", "Speech engine"),
        ("omnivoice", "Speech model"),
        ("speech_models", "Transcription and translation models"),
        ("zero_shot_vc", "Native accent converter"),
        ("applio_env", "Voice training tools: environment"),
        ("runtime_toml", "Finishing"),
    ];
    s.setup.phase = Phase::Running;
    s.setup.steps = plan
        .iter()
        .zip(states)
        .map(|((id, label), state)| Step { id: (*id).into(), label: (*label).into(), state, message: None })
        .collect();
    s.setup.log = ["== Speech model ==", "OmniVoice model: downloading k2-fsa/OmniVoice (about 3.3 GB, resumable)...", "model.safetensors:  41%|████      | 1.35G/3.30G [00:42<01:01, 31.8MB/s]"]
        .iter()
        .map(|l| (*l).to_owned())
        .collect();
    s.setup.show_log = true;
}
