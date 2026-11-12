//! Onboarding (first run): welcome, how it works, engine/runtime status and setup.

use std::path::PathBuf;
use std::process::Command;

use rune::{Cx, El, TextEdit, col, el, row, text};
use serde_json::{Value, json};

use crate::engine::EngineMsg;
use crate::state::{AppState, Screen};
use crate::ui::theme::{self, Palette, radius, ty};
use crate::ui::components::Tone;
use crate::ui::events;
use crate::ui::shell::EngineStatus as ShellEngine;
use crate::ui::voices::kit::{self, BtnKind};

// ---------------------------------------------------------------------------------------------
// Paths and the settings file
// ---------------------------------------------------------------------------------------------

/// The per-user data dir (same resolution as the engine client: env, `location.toml` pointer, platform default).
pub fn data_dir() -> PathBuf {
    crate::engine::default_data_dir().unwrap_or_else(|| PathBuf::from("."))
}

/// `MIMVOX_RUNTIME`, else `<data dir>/runtime.toml`.
pub fn runtime_toml_path() -> PathBuf {
    std::env::var_os("MIMVOX_RUNTIME").map(PathBuf::from).unwrap_or_else(|| data_dir().join("runtime.toml"))
}

/// Shown on the welcome page: the app is an early version.
const EARLY_VERSION: &str = "Early version: expect bugs. Some things may not work, work badly or change between \
    versions. Keep your own copies of recordings that matter, and report problems on the project's GitHub page.";

fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

fn read_settings() -> Value {
    std::fs::read_to_string(settings_path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_else(|| json!({}))
}

fn write_settings(v: &Value) {
    let path = settings_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(t) = serde_json::to_string_pretty(v) {
        let _ = std::fs::write(path, t);
    }
}

fn update_settings(f: impl FnOnce(&mut Value)) {
    let mut v = read_settings();
    if !v.is_object() {
        v = json!({});
    }
    f(&mut v);
    write_settings(&v);
}

/// One value of the settings file (`None` when absent).
pub fn setting(key: &str) -> Option<Value> {
    read_settings().get(key).cloned()
}

/// Sets one value of the settings file, keeping the others.
pub fn save_setting(key: &str, value: Value) {
    update_settings(|v| v[key] = value);
}

/// The voice selected last time, if any.
pub fn saved_active_voice() -> Option<String> {
    read_settings().get("active_voice").and_then(Value::as_str).map(str::to_string)
}

pub fn save_active_voice(id: Option<&str>) {
    update_settings(|v| {
        v["active_voice"] = match id {
            Some(id) => json!(id),
            None => Value::Null,
        }
    });
}

/// The speaking speed chosen on the Speak screen (0.8..=1.3, default 1.0).
pub fn saved_speed() -> f32 {
    read_settings().get("speed").and_then(Value::as_f64).map_or(1.0, |v| (v as f32).clamp(0.8, 1.3))
}

pub fn save_speed(speed: f32) {
    update_settings(|v| v["speed"] = json!((f64::from(speed) * 100.0).round() / 100.0));
}

/// What the Redub screen remembers: the mode (`dub` | `convert` | `respeak`), the spoken-language code (`auto`, `uk`, ...),
/// "Review text before dubbing" and the language to translate to (`redub_target`). Each part is `None` when never saved.
pub fn saved_redub_prefs() -> (Option<String>, Option<String>, Option<bool>, Option<String>) {
    let v = read_settings();
    let text = |k: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    (text("redub_mode"), text("redub_language"), v.get("redub_review").and_then(Value::as_bool), text("redub_target"))
}

pub fn save_redub_prefs(mode: &str, language: &str, review: bool, target: &str) {
    update_settings(|v| {
        v["redub_mode"] = json!(mode);
        v["redub_language"] = json!(language);
        v["redub_review"] = json!(review);
        v["redub_target"] = json!(target);
    });
}

/// The microphone chosen on the Record screen (`None` = system default).
pub fn saved_microphone() -> Option<String> {
    read_settings().get("microphone").and_then(Value::as_str).filter(|m| !m.is_empty()).map(str::to_string)
}

/// Remembers the microphone (`None` = system default) without touching the other settings.
pub fn save_microphone(name: Option<&str>) {
    update_settings(|v| {
        v["microphone"] = match name {
            Some(n) => json!(n),
            None => Value::Null,
        }
    });
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

/// What `hello` told us about the machine.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceInfo {
    pub backend: Option<String>,
    pub name: Option<String>,
    pub vram_gb: Option<f32>,
    pub version: Option<String>,
}

impl DeviceInfo {
    fn from_hello(h: &crate::engine::protocol::Hello) -> DeviceInfo {
        let some = |v: &str| (!v.is_empty()).then(|| v.to_owned());
        DeviceInfo {
            backend: some(&h.device.backend),
            name: some(&h.device.name),
            vram_gb: (h.device.vram_gb > 0.0).then_some(h.device.vram_gb),
            version: some(&h.version),
        }
    }
}

pub struct OnboardingState {
    pub runtime_present: bool,
    pub device: Option<DeviceInfo>,
    /// Result of the last "Run setup" press (a hint or an error).
    pub setup_note: Option<(Tone, String)>,
    pub setup_started: bool,
    /// Where setup installs the runtime and models (editable; passed to setup as `--data-dir`).
    pub location: TextEdit,
    /// Free space of the drive holding `location`: (path it was measured for, bytes).
    free_space: Option<(String, Option<u64>)>,
    /// The path the last measurement was started for.
    free_asked: String,
    /// When to measure the edited path (app clock): a short debounce while the user types, counted from the last
    /// change (`free_seen`).
    free_due: Option<f64>,
    free_seen: String,
    /// The native folder dialog is open (its answer arrives as an event).
    picking: bool,
}

/// What setup downloads, shown before it runs.
const SETUP_SIZE_WARNING: &str = "Setup downloads about 28 GB (engine and speech model ~11 GB, voice-training tools ~12 GB, \
Redub models ~4.5 GB, music removal ~0.3 GB), plus ~2.7 GB for the better translation model with an NVIDIA GPU. \
Make sure the drive has at least 32 GB free (38 GB with an NVIDIA GPU).";
const NEEDED_FREE_BYTES: u64 = 32 * 1_000_000_000;

/// Free bytes on the drive that holds `path` (or its nearest existing parent). `None` when it cannot be told.
pub(crate) fn free_bytes(path: &str) -> Option<u64> {
    let mut p = PathBuf::from(path);
    while !p.exists() {
        if !p.pop() {
            return None;
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let script = format!(
            "[System.IO.DriveInfo]::new((Get-Item -LiteralPath '{}').PSDrive.Root).AvailableFreeSpace",
            p.display().to_string().replace('\'', "''")
        );
        let out = Command::new("powershell").args(["-NoProfile", "-Command", &script]).creation_flags(0x0800_0000).output().ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(windows))]
    {
        let out = Command::new("df").args(["-Pk"]).arg(&p).output().ok()?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let kib: u64 = text.lines().nth(1)?.split_whitespace().nth(3)?.parse().ok()?;
        Some(kib * 1024)
    }
}

impl Default for OnboardingState {
    fn default() -> Self {
        OnboardingState {
            runtime_present: runtime_toml_path().is_file(),
            device: None,
            setup_note: None,
            setup_started: false,
            location: TextEdit::new(data_dir().display().to_string()),
            free_space: None,
            free_asked: String::new(),
            free_due: None,
            free_seen: String::new(),
            picking: false,
        }
    }
}

/// True when the first-run flow still has something to do (runtime setup).
pub fn needed(s: &AppState) -> bool {
    !s.onboarding.runtime_present
}

/// The compute backend reported by the engine (`cuda`, `mps`, `cpu`), if known.
pub fn backend(s: &AppState) -> Option<&str> {
    s.onboarding.device.as_ref().and_then(|d| d.backend.as_deref())
}

/// Re-checks `runtime.toml` now. Done on "Check again", when the window gets the focus back (setup ran in a
/// terminal window), when a setup terminal that can be waited for ends, and after a data move. True when it changed.
pub fn recheck(s: &mut AppState) -> bool {
    let was = s.onboarding.runtime_present;
    s.onboarding.runtime_present = runtime_toml_path().is_file();
    if s.onboarding.runtime_present && !was {
        s.onboarding.setup_note = Some((Tone::Success, "Setup finished. Restart the app if the engine does not start by itself.".into()));
        // Setup may have moved the data dir (location.toml pointer): follow it.
        let dir = data_dir();
        s.shell.data_dir = dir.clone();
        s.record.data_dir = dir;
    }
    s.onboarding.runtime_present != was
}

/// The window got the focus back.
pub fn on_focus(s: &mut AppState) {
    if !s.onboarding.runtime_present && !s.shell.scripted() {
        recheck(s);
    }
}

/// Free space is measured this long after the install path last changed.
const FREE_DEBOUNCE_S: f64 = 0.4;

/// After every event: measures the free space of an edited install path (after a short debounce).
pub fn settle(s: &mut AppState) -> bool {
    if s.screen != Screen::Onboarding || s.onboarding.runtime_present || s.shell.scripted() {
        s.onboarding.free_due = None;
        return false;
    }
    let now = s.shell.now;
    let o = &mut s.onboarding;
    let path = o.location.text.trim().to_owned();
    if path == o.free_asked {
        o.free_due = None;
        return false;
    }
    if path != o.free_seen || o.free_due.is_none() {
        o.free_seen = path.clone();
        o.free_due = Some(now + FREE_DEBOUNCE_S);
        return false;
    }
    match o.free_due {
        Some(due) if now + 1e-3 >= due => {
            o.free_due = None;
            o.free_asked = path.clone();
            std::thread::spawn(move || {
                let bytes = free_bytes(&path);
                events::post(move |s| s.onboarding.free_space = Some((path, bytes)));
            });
        }
        _ => {}
    }
    false
}

/// The debounce deadline [`settle`] waits for, if any (the shell arms a timer for it).
pub fn next_due(s: &AppState) -> Option<f64> {
    s.onboarding.free_due
}

pub fn on_engine(s: &mut AppState, msg: &EngineMsg) {
    if let EngineMsg::Ready(h) = msg {
        s.onboarding.device = Some(DeviceInfo::from_hello(h));
        s.onboarding.runtime_present = true;
    }
}

// ---------------------------------------------------------------------------------------------
// Running the setup script in a terminal window
// ---------------------------------------------------------------------------------------------

/// Finds the repo or bundle root that holds `setup.sh` / `setup.cmd`.
pub(crate) fn find_setup_root() -> Option<PathBuf> {
    let script = if cfg!(windows) { "setup.cmd" } else { "setup.sh" };
    let mut starts: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("MIMVOX_ROOT") {
        starts.push(PathBuf::from(p));
    }
    if let Ok(exe) = std::env::current_exe() {
        starts.push(exe);
    }
    if let Ok(cwd) = std::env::current_dir() {
        starts.push(cwd);
    }
    starts.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    for start in starts {
        for dir in start.ancestors().take(7) {
            if dir.join(script).is_file() {
                return Some(dir.to_path_buf());
            }
        }
    }
    None
}

/// Opens a terminal window running the setup script. Returns a message for the user.
/// The terminal process is waited for on its own thread; when it ends, `runtime.toml` is checked again (terminals that
/// hand the window to another process end at once: then the check on focus or "Check again" sees the result).
fn launch_setup(location: &str) -> Result<String, String> {
    let root = find_setup_root().ok_or_else(|| {
        "Could not find setup.sh / setup.cmd next to the app. Run it from the project folder by hand.".to_string()
    })?;
    let mut child = spawn_terminal(&root, location)?;
    std::thread::spawn(move || {
        let _ = child.wait();
        events::post_if(recheck);
    });
    Ok(format!("Setup is running in a terminal window ({}). This can take a while.", root.display()))
}

#[cfg(windows)]
fn spawn_terminal(root: &std::path::Path, location: &str) -> Result<std::process::Child, String> {
    Command::new("cmd")
        .args(["/c", "start", "Mimvox setup", "cmd", "/k", "setup.cmd"])
        .args(setup_args(location))
        .current_dir(root)
        .spawn()
        .map_err(|e| format!("Could not open a terminal: {e}. Run setup.cmd by hand."))
}

#[cfg(target_os = "macos")]
fn spawn_terminal(root: &std::path::Path, location: &str) -> Result<std::process::Child, String> {
    let quote = |v: &str| format!("'{}'", v.replace('\'', "'\\''"));
    let args: Vec<String> = setup_args(location).iter().map(|a| quote(a)).collect();
    let cmd = format!("cd {} && ./setup.sh {}", quote(&root.display().to_string()), args.join(" "));
    let script = format!("tell application \"Terminal\" to do script \"{}\"", cmd.replace('"', "\\\""));
    Command::new("osascript")
        .args(["-e", &script, "-e", "tell application \"Terminal\" to activate"])
        .spawn()
        .map_err(|e| format!("Could not open Terminal: {e}. Run ./setup.sh by hand."))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn spawn_terminal(root: &std::path::Path, location: &str) -> Result<std::process::Child, String> {
    let inner = "./setup.sh \"$@\"; echo; echo 'Setup finished. Press Enter to close this window.'; read _";
    let terminals: [(&str, &[&str]); 6] = [
        ("x-terminal-emulator", &["-e"]),
        ("gnome-terminal", &["--"]),
        ("konsole", &["-e"]),
        ("xfce4-terminal", &["-x"]),
        ("alacritty", &["-e"]),
        ("xterm", &["-e"]),
    ];
    for (term, pre) in terminals {
        let mut c = Command::new(term);
        c.args(pre).args(["bash", "-c", inner, "setup"]).args(setup_args(location)).current_dir(root);
        if let Ok(child) = c.spawn() {
            return Ok(child);
        }
    }
    Err("No terminal emulator found. Open a terminal in the project folder and run ./setup.sh.".into())
}

/// `--data-dir <path>` for the setup script (the install location chosen on this screen).
fn setup_args(location: &str) -> Vec<String> {
    let loc = location.trim();
    if loc.is_empty() { Vec::new() } else { vec!["--data-dir".to_owned(), loc.to_owned()] }
}

/// Opens the system folder picker on its own thread (the frame loop never waits for it).
fn pick_location(s: &mut AppState) {
    if s.onboarding.picking || s.shell.scripted() {
        return;
    }
    s.onboarding.picking = true;
    let start = PathBuf::from(s.onboarding.location.text.trim());
    std::thread::spawn(move || {
        let mut dialog = rfd::FileDialog::new().set_title("Choose where to install the speech engine and models");
        if let Some(dir) = Some(start).filter(|d| d.is_dir()).or_else(|| dirs::home_dir()) {
            dialog = dialog.set_directory(dir);
        }
        let choice = dialog.pick_folder();
        events::post(move |s| {
            s.onboarding.picking = false;
            if let Some(dir) = choice {
                s.onboarding.location = TextEdit::new(dir.display().to_string());
            }
        });
    });
}

fn run_setup(s: &mut AppState) {
    let location = s.onboarding.location.text.clone();
    s.onboarding.setup_note = Some(match launch_setup(&location) {
        Ok(m) => {
            s.onboarding.setup_started = true;
            (Tone::Accent, m)
        }
        Err(e) => (Tone::Danger, e),
    });
}

fn finish(s: &mut AppState) {
    s.screen = Screen::Voices;
}

// ---------------------------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------------------------

pub fn view(s: &AppState, cx: &mut Cx<AppState>) -> El {
    let p = theme::palette(s, cx);
    let o = &s.onboarding;

    let hero = row()
        .w_full()
        .gap(16.0)
        .center_y()
        .child(crate::ui::shell::brand_mark(s, 56.0))
        .child(
            col()
                .gap(4.0)
                .child(text("Welcome").font(ty::DISPLAY).color(p.text))
                .child(text("Mimvox").font(ty::TITLE).color(p.text_dim)),
        );

    let intro = text(
        "Clone your own voice on this computer, then type any text and hear it spoken the way you would say it. \
         You read a short guided text for about 6 minutes, the app trains your voice for roughly an hour on an \
         RTX-class GPU, and from then on you can speak anything, instantly: it speaks English, Ukrainian and 60+ other \
         languages in your voice. Everything stays on your machine.",
    )
    .font(ty::BODY)
    .color(p.text_dim)
    .wrap();

    let steps = row()
        .w_full()
        .gap(12.0)
        .child(step(&p, 1, "Record", "About 6 minutes of guided reading"))
        .child(step(&p, 2, "Train", "About 1 hour on an RTX-class GPU"))
        .child(step(&p, 3, "Speak", "Any text in 60+ languages, in your voice"));

    let mut page = col().w_full().gap(18.0).child(
        kit::panel(&p).gap(18.0).child(hero).child(intro).child(steps).child(kit::notice(&p, Tone::Warning, EARLY_VERSION)),
    );
    page = page.child(engine_card(&p, s));

    let ready = o.runtime_present;
    let hint = if !o.runtime_present {
        "Run setup first, then you can continue."
    } else {
        "You are all set."
    };
    page = page.child(
        row()
            .w_full()
            .gap(14.0)
            .center_y()
            .justify_end()
            .child(text(hint).font(ty::CAPTION).color(p.text_dim))
            .child(kit::big_button(&p, "Get started", ready, finish)),
    );
    kit::page(page)
}

fn step(p: &Palette, n: u32, title: &str, sub: &str) -> El {
    col()
        .grow(1.0)
        .min_w(0.0)
        .gap(8.0)
        .padding(14.0)
        .radius(radius::CARD)
        .fill(p.well)
        .border(1.0, p.well_rim)
        .child(
            el().size(24.0, 24.0).radius(radius::CONTROL).center().fill(p.accent.alpha(0.25)).child(
                text(n.to_string()).font(ty::LABEL).color(p.accent),
            ),
        )
        .child(text(title.to_string()).font(ty::TITLE).color(p.text))
        .child(text(sub.to_string()).font(ty::CAPTION).color(p.text_dim).wrap())
}

/// The install folder (editable) and the free space on its drive.
fn location_block(p: &Palette, o: &OnboardingState) -> El {
    let free = o.free_space.clone();
    let (line, tone_ok) = match free {
        Some((path, Some(b))) if path == o.location.text.trim() => {
            (format!("{:.1} GB free on this drive", b as f64 / 1e9), b >= NEEDED_FREE_BYTES)
        }
        Some((path, None)) if path == o.location.text.trim() => ("Free space could not be read for this folder.".to_owned(), true),
        _ => ("Checking free space…".to_owned(), true),
    };
    col().key("location").w_full().gap(8.0)
        .child(text("Install location").font(ty::LABEL).color(p.text))
        .child(row().w_full().gap(10.0).center_y()
            .child(kit::field(p, &o.location, "Folder for the speech models and your voices", |s: &mut AppState| &mut s.onboarding.location))
            .child(kit::button(p, "Change location…", BtnKind::Glass, !o.picking, pick_location)))
        .child(text(format!("{line}. Pick a folder, or edit the path, to change where everything is installed.")).font(ty::CAPTION).color(if tone_ok { p.text_faint } else { p.warn }).wrap())
}

fn engine_card(p: &Palette, s: &AppState) -> El {
    let o = &s.onboarding;
    let mut card = kit::card(p).gap(14.0).child(text("Engine and device").font(ty::TITLE).color(p.text));

    // Runtime
    let (badge, line) = if o.runtime_present {
        (kit::badge(p, "Runtime found", Tone::Success), "The speech engine is installed.")
    } else {
        (kit::badge(p, "Runtime missing", Tone::Warning), "The speech engine is not installed yet.")
    };
    card = card.child(row().w_full().gap(10.0).center_y().child(badge).child(text(line).font(ty::BODY).color(p.text_dim)));

    if !o.runtime_present {
        card = card
            .child(kit::notice(p, Tone::Warning, SETUP_SIZE_WARNING))
            .child(
                text(
                    "Setup downloads the speech models and creates the runtime (runtime.toml). A terminal window opens and \
                     shows the progress; this screen updates when it is done.",
                )
                .font(ty::CAPTION)
                .color(p.text_dim)
                .wrap(),
            )
            .child(location_block(p, o))
            .child(
                row()
                    .gap(10.0)
                    .child(kit::button(p, if o.setup_started { "Run setup again" } else { "Run setup" }, BtnKind::Primary, true, run_setup))
                    .child(kit::button(p, "Check again", BtnKind::Glass, true, |s: &mut AppState| {
                        recheck(s);
                    })),
            );
    }
    if let Some((tone, msg)) = &o.setup_note {
        card = card.child(kit::notice(p, *tone, msg));
    }

    // Engine
    match &s.shell.engine {
        ShellEngine::Idle | ShellEngine::Starting { .. } if o.runtime_present => {
            card = card.child(text("Starting the engine...").font(ty::CAPTION).color(p.text_faint));
        }
        ShellEngine::Stopped { code, detail } => {
            let last = detail.lines().last().unwrap_or("").trim();
            let why = match code {
                Some(c) if !last.is_empty() => format!("exit code {c}: {last}"),
                Some(c) => format!("exit code {c}"),
                None if !last.is_empty() => last.to_string(),
                None => "no details".to_string(),
            };
            card = card.child(kit::notice(p, Tone::Danger, &format!("The engine stopped: {why}")));
        }
        ShellEngine::Unavailable { reason } => {
            card = card.child(kit::notice(p, Tone::Warning, reason));
        }
        _ => {}
    }

    // Device
    if let Some(d) = &o.device {
        card = card.child(el().w_full().h(1.0).fill(p.rim_bottom));
        let backend = d.backend.clone().unwrap_or_else(|| "unknown".into()).to_uppercase();
        let mut rows = col().gap(6.0);
        rows = rows.child(kv(p, "Device", d.name.clone().unwrap_or_else(|| "Unknown".into())));
        if let Some(v) = d.vram_gb {
            rows = rows.child(kv(p, "Video memory", format!("{v:.0} GB")));
        }
        rows = rows.child(kv(p, "Backend", backend));
        card = card.child(rows);
        match d.backend.as_deref() {
            Some("cpu") => {
                card = card.child(kit::notice(
                    p,
                    Tone::Warning,
                    "No GPU found. Training on the CPU would take many hours; speaking with a quick (zero-shot) voice still works.",
                ))
            }
            Some("mps") => {
                card = card.child(kit::notice(
                    p,
                    Tone::Warning,
                    "Apple GPU (MPS) is supported on a best-effort basis. Training will be noticeably slower than on an NVIDIA GPU.",
                ))
            }
            _ => {
                if d.vram_gb.is_some_and(|v| v < 6.0) {
                    card = card.child(kit::notice(p, Tone::Warning, "Less than 6 GB of video memory: training may run out of memory."));
                }
            }
        }
    }
    card
}

fn kv(p: &Palette, k: &str, v: String) -> El {
    row()
        .w_full()
        .gap(12.0)
        .child(text(k.to_string()).font(ty::CAPTION).color(p.text_faint).w(110.0))
        .child(text(v).font(ty::BODY).color(p.text).grow(1.0).min_w(0.0).truncate())
}

// ---------------------------------------------------------------------------------------------
// Scenes for headless snapshots
// ---------------------------------------------------------------------------------------------

fn base(s: &mut AppState) {
    s.screen = Screen::Onboarding;
    s.shell.set_engine(ShellEngine::Idle);
    s.onboarding = OnboardingState {
        runtime_present: false,
        device: None,
        setup_note: None,
        setup_started: false,
        ..OnboardingState::default()
    };
    let loc = "/home/me/.local/share/mimvox".to_string();
    s.onboarding.location = TextEdit::new(loc.clone());
    s.onboarding.free_asked = loc.clone();
    s.onboarding.free_space = Some((loc, Some(412_300_000_000)));
}

fn device(backend: &str, name: &str, vram: f32) -> Option<DeviceInfo> {
    Some(DeviceInfo { backend: Some(backend.into()), name: Some(name.into()), vram_gb: Some(vram), version: Some("0.1.0".into()) })
}

fn scene_welcome(s: &mut AppState) {
    base(s);
}
fn scene_setup_missing(s: &mut AppState) {
    base(s);
    s.onboarding.setup_note = Some((Tone::Accent, "Setup is running in a terminal window. This can take a while.".into()));
    s.onboarding.setup_started = true;
}
fn scene_ready_gpu(s: &mut AppState) {
    base(s);
    s.onboarding.runtime_present = true;
    s.shell.set_engine(ShellEngine::Ready { device: None, tts: Some("ready".into()), rvc: Some("ready".into()) });
    s.onboarding.device = device("cuda", "NVIDIA GeForce RTX 4070", 12.0);
}
fn scene_cpu_warning(s: &mut AppState) {
    scene_ready_gpu(s);
    s.onboarding.device = device("cpu", "CPU", 0.0);
    s.onboarding.device.as_mut().unwrap().vram_gb = None;
}
fn scene_engine_stopped(s: &mut AppState) {
    scene_ready_gpu(s);
    s.onboarding.device = None;
    s.shell.set_engine(ShellEngine::Stopped { code: Some(1), detail: "ModuleNotFoundError: No module named 'torch'".into() });
}

pub fn scenes() -> &'static [(&'static str, fn(&mut AppState))] {
    &[
        ("onboarding-welcome", scene_welcome),
        ("onboarding-setup", scene_setup_missing),
        ("onboarding-ready", scene_ready_gpu),
        ("onboarding-cpu", scene_cpu_warning),
        ("onboarding-stopped", scene_engine_stopped),
    ]
}
