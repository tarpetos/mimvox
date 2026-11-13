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
    /// The component an "Install" button is adding (None: a full setup or a runtime update).
    pub adding: Option<String>,
    /// Optional components unticked on the start screen (all are ticked at first).
    pub left_out: Vec<String>,
}

/// One part of the runtime that can be installed on its own (setup's COMPONENTS; the sizes match).
pub struct Component {
    pub id: &'static str,
    pub label: &'static str,
    pub gb: f32,
    pub what: &'static str,
}

pub const COMPONENTS: [Component; 6] = [
    Component { id: "speech", label: "Speech", gb: 12.6, what: "Speak and Book in about 60 languages, and Redub's Re-speak. Always installed." },
    Component { id: "training", label: "Voice training", gb: 12.0, what: "Train your voice: voice match for trained voices and Redub's Convert voice." },
    Component { id: "translation", label: "Translation", gb: 2.5, what: "Redub's Translate & dub (NLLB-200 600M)." },
    Component {
        id: "better_translation",
        label: "Better translation",
        gb: 2.7,
        what: "Fewer mistranslations in Redub (NLLB-200 1.3B). Installed only with an NVIDIA or AMD GPU.",
    },
    Component { id: "native_accent", label: "Native accent", gb: 2.0, what: "English with a native accent for voices that are not trained (Seed-VC, GPL-3.0)." },
    Component { id: "music_removal", label: "Music removal", gb: 0.3, what: "Keeps only the voice: Redub keeps a video's background, references come out clean." },
];

pub fn component(id: &str) -> Option<&'static Component> {
    COMPONENTS.iter().find(|c| c.id == id)
}

/// The engine has this component installed (true until the engine has said, so nothing flashes at start).
pub fn has(s: &AppState, id: &str) -> bool {
    s.shell.components.as_ref().is_none_or(|c| c.iter().any(|x| x == id))
}

/// The running setup's process id (for Cancel and for closing the app).
static RUNNING: Mutex<Option<u32>> = Mutex::new(None);

/// The `uv` setup runs with: `MIMVOX_UV`, the installer's copy, then PATH and uv's usual install folders.
fn uv_path() -> Option<PathBuf> {
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    if let Some(p) = std::env::var_os("MIMVOX_UV").map(PathBuf::from).filter(|p| p.is_file()) {
        return Some(p);
    }
    let res = crate::engine::find_resources();
    if let Some(p) = res.as_ref().map(|r| r.join("bin").join(exe)).filter(|p| p.is_file()) {
        return Some(p);
    }
    if let Some(p) = res.as_ref().map(|r| r.join("bin").join("uv.gz")).filter(|p| p.is_file()).and_then(|gz| unpack_uv(&gz)) {
        return Some(p);
    }
    let path = std::env::var_os("PATH").unwrap_or_default();
    if let Some(p) = std::env::split_paths(&path).map(|d| d.join(exe)).find(|p| p.is_file()) {
        return Some(p);
    }
    let home = dirs::home_dir()?;
    [home.join(".local").join("bin").join(exe), home.join(".cargo").join("bin").join(exe)].into_iter().find(|p| p.is_file())
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

/// The first setup from the start screen: into `data_dir`, with the components left ticked.
pub fn start(s: &mut AppState, data_dir: &str) {
    let chosen: Vec<&str> = COMPONENTS.iter().map(|c| c.id).filter(|id| !s.setup.left_out.iter().any(|x| x == id)).collect();
    let mut args = base_args(data_dir);
    args.push("--components".into());
    args.push(chosen.join(",").into());
    start_with(s, args, None);
}

/// Setup again with the components installed before (a runtime update).
pub fn update_runtime(s: &mut AppState) {
    let args = base_args(&super::onboarding::data_dir().display().to_string());
    start_with(s, args, None);
}

/// Adds one component to the installed runtime (the "Install" buttons on the screens that need it).
pub fn install(s: &mut AppState, id: &str) {
    let mut args = base_args(&super::onboarding::data_dir().display().to_string());
    args.push("--add".into());
    args.push(id.into());
    start_with(s, args, Some(id.to_owned()));
}

fn base_args(data_dir: &str) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec!["--gui".into()];
    if !data_dir.trim().is_empty() {
        args.push("--data-dir".into());
        args.push(data_dir.trim().into());
    }
    args
}

/// Starts setup with `args`. Does nothing while one runs.
fn start_with(s: &mut AppState, args: Vec<OsString>, adding: Option<String>) {
    if running(s) || s.shell.scripted() {
        return;
    }
    *CANCELLED.lock().unwrap_or_else(|e| e.into_inner()) = false;
    let left_out = std::mem::take(&mut s.setup.left_out);
    s.setup = SetupRunState { phase: Phase::Running, show_log: s.setup.show_log, adding, left_out, ..Default::default() };
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

fn on_exit(s: &mut AppState, code: Option<i32>, signal: Option<i32>) {
    if s.setup.phase == Phase::Running {
        // Ended without its final report: stopped, or failed before the steps (the log says why).
        let last = s.setup.log.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
        let cancelled = std::mem::take(&mut *CANCELLED.lock().unwrap_or_else(|e| e.into_inner()));
        s.setup.phase = match code {
            Some(0) => Phase::Finished(true),
            None if cancelled => Phase::Error("Setup was stopped. Run it again to continue where it stopped.".into()),
            None => Phase::Error(match signal {
                Some(sig) => format!("Setup crashed (signal {sig}). Run it again; if it keeps crashing, report it on the project's GitHub page."),
                None => "Setup ended unexpectedly. Run it again to continue where it stopped.".into(),
            }),
            Some(c) if last.is_empty() => Phase::Error(format!("Setup stopped (exit code {c}).")),
            Some(_) => Phase::Error(last),
        };
    }
    for step in s.setup.steps.iter_mut().filter(|st| st.state == StepState::Running) {
        step.state = StepState::Failed;
    }
    // A runtime that is now there, up to date or with a new component: the onboarding screen and the engine follow
    // (the engine reads what is installed when it starts).
    let ok = s.setup.phase == Phase::Finished(true);
    super::onboarding::recheck(s);
    if s.onboarding.runtime_present && (ok || !super::shell::engine_ready(s)) {
        super::shell::restart_engine(s);
    }
    if ok {
        if let Some(c) = s.setup.adding.as_deref().and_then(component) {
            super::shell::toast(s, Tone::Success, format!("{} is installed.", c.label));
        }
    }
}

/// Stops the running setup and everything it started (it resumes on the next run).
pub fn cancel(_s: &mut AppState) {
    stop_on_exit();
}

/// Cancel was pressed (the end of the process is then not a crash).
static CANCELLED: Mutex<bool> = Mutex::new(false);

/// Called on Cancel and by `main` when the app closes.
pub fn stop_on_exit() {
    *CANCELLED.lock().unwrap_or_else(|e| e.into_inner()) = true;
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

/// Disk use of the components ticked on the start screen.
pub fn chosen_gb(s: &AppState) -> f32 {
    COMPONENTS.iter().filter(|c| c.id == "speech" || !s.setup.left_out.iter().any(|x| x == c.id)).map(|c| c.gb).sum()
}

/// The start screen's choice of components: Speech (always), the others ticked at first, and the total size.
pub fn chooser(s: &AppState, p: Palette) -> El {
    use super::voices::kit;
    let mut list = col().key("setup-components").w_full().gap(10.0);
    let mut total = 0.0;
    for c in &COMPONENTS {
        let required = c.id == "speech";
        let on = required || !s.setup.left_out.iter().any(|x| x == c.id);
        if on {
            total += c.gb;
        }
        let label = format!("{} · {:.1} GB{}", c.label, c.gb, if required { " · required" } else { "" });
        let id = c.id;
        let toggle = move |s: &mut AppState| {
            if id == "speech" {
                return;
            }
            if let Some(i) = s.setup.left_out.iter().position(|x| x == id) {
                s.setup.left_out.remove(i);
            } else {
                s.setup.left_out.push(id.to_owned());
            }
        };
        // Speech cannot be left out: a dimmed, fixed tick instead of a checkbox.
        let check = if required {
            row().w_full().gap(12.0).items_start().opacity(0.5).help("Always installed: everything else builds on it.")
                .child(el().size(20.0, 20.0).radius(super::theme::radius::CONTROL * 0.4).fill(p.text_faint).center()
                    .child(icon_el(&icons::CHECK, 13.0, p.bg_top)))
                .child(text(label).font(ty::BODY).color(p.text))
        } else {
            kit::checkbox(&p, on, &label, toggle)
        };
        list = list.child(
            col().key(format!("component-{id}")).w_full().gap(2.0)
                .child(check)
                .child(text(c.what).font(ty::CAPTION).color(p.text_faint).wrap().pl(32.0)),
        );
    }
    let all = s.setup.left_out.is_empty();
    let toggle_all = move |s: &mut AppState| {
        s.setup.left_out = if all { COMPONENTS.iter().filter(|c| c.id != "speech").map(|c| c.id.to_owned()).collect() } else { Vec::new() };
    };
    col().w_full().gap(10.0)
        .child(row().w_full().center_y().justify_between()
            .child(text(format!("What to install · about {total:.0} GB")).font(ty::LABEL).color(p.text))
            .child(super::components::Button::new(if all { "Only what is required" } else { "Everything" }).ghost().small()
                .key("components-all").on_press(toggle_all).build(p)))
        .child(list)
        .child(text("Anything left out can be installed later in one click, on the screen that needs it.").font(ty::CAPTION).color(p.text_dim).wrap())
}

/// In place of a feature whose component is not installed: what it is, its size and an Install button; while it
/// installs, the progress.
pub fn install_card(s: &AppState, p: Palette, id: &str, needed_for: &str) -> El {
    use super::components::{Button, glass_card};
    let Some(c) = component(id) else { return el() };
    let this = s.setup.adding.as_deref() == Some(id);
    let mut info = col().grow(1.0).min_w(0.0).gap(4.0);
    let button: Option<El> = if running(s) && this {
        let step = current(s).map(|st| st.label.clone()).unwrap_or_else(|| "Starting…".into());
        info = info
            .child(text(format!("Installing {}: {step}", c.label)).font(ty::LABEL).color(p.text))
            .child(text(last_line(s).unwrap_or("").to_owned()).font(ty::CAPTION).color(p.text_dim).truncate())
            .child(el().w_full().pt(4.0).child(progress_bar(p, Some(progress(s)))));
        Some(Button::new("Cancel").ghost().small().key(format!("install-{id}-cancel")).on_press(cancel).build(p))
    } else {
        let failed = this && matches!(s.setup.phase, Phase::Error(_) | Phase::Finished(false));
        info = info
            .child(text(format!("{} is not installed", c.label)).font(ty::LABEL).color(p.text))
            .child(text(format!("{}{} About {:.1} GB; it installs in the background, and the engine restarts when it is done.",
                if needed_for.is_empty() { String::new() } else { format!("{needed_for} ") }, c.what, c.gb))
                .font(ty::CAPTION).color(p.text_dim).wrap());
        if failed {
            let why = match &s.setup.phase {
                Phase::Error(e) => e.clone(),
                _ => s.setup.steps.iter().find_map(|st| st.message.clone()).unwrap_or_else(|| "A step failed.".into()),
            };
            info = info.child(text(format!("The last attempt stopped: {why}")).font(ty::CAPTION).color(p.bad).wrap());
        }
        let id_owned = id.to_owned();
        Some(
            Button::new(if failed { "Try again".to_owned() } else { format!("Install · {:.1} GB", c.gb) })
                .primary().small().icon(&icons::SPARKLE).key(format!("install-{id}"))
                .disabled(running(s))
                .help(if running(s) { "Another setup is running; this can start when it is done." } else { "" })
                .on_press(move |s| install(s, &id_owned))
                .build(p),
        )
    };
    let row_ = row().key(format!("install-card-{id}")).w_full().px(14.0).py(12.0).gap(12.0).center_y()
        .child(icon_el(&icons::SPARKLE, 16.0, p.accent))
        .child(info)
        .children(button);
    glass_card(p).w_full().child(row_)
}

/// A small inline "Install" for an option that needs a component (a checkbox, a mode): one line with the button.
pub fn install_line(s: &AppState, p: Palette, id: &str, what: &str) -> El {
    use super::components::Button;
    let Some(c) = component(id) else { return el() };
    let this = s.setup.adding.as_deref() == Some(id) && running(s);
    let id_owned = id.to_owned();
    let line = if this {
        format!("Installing {}… {:.0}%", c.label, progress(s) * 100.0)
    } else {
        format!("{what} needs {} ({:.1} GB).", c.label.to_lowercase(), c.gb)
    };
    row().key(format!("install-line-{id}")).w_full().gap(8.0).center_y()
        .child(text(line).font(ty::CAPTION).color(p.text_dim).grow(1.0).min_w(0.0).wrap())
        .children((!this).then(|| {
            Button::new("Install").ghost().small().key(format!("install-line-{id}-go")).disabled(running(s))
                .on_press(move |s| install(s, &id_owned)).build(p)
        }))
}
