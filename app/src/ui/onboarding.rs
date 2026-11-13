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

fn settings_path() -> PathBuf {
    data_dir().join("settings.json")
}

pub(crate) fn read_settings() -> Value {
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

pub(crate) fn update_settings(f: impl FnOnce(&mut Value)) {
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
    pub vendor: Option<String>,
}

impl DeviceInfo {
    fn from_hello(h: &crate::engine::protocol::Hello) -> DeviceInfo {
        let some = |v: &str| (!v.is_empty()).then(|| v.to_owned());
        DeviceInfo {
            backend: some(&h.device.backend),
            name: some(&h.device.name),
            vram_gb: (h.device.vram_gb > 0.0).then_some(h.device.vram_gb),
            version: some(&h.version),
            vendor: some(&h.device.vendor),
        }
    }
}

pub struct OnboardingState {
    pub runtime_present: bool,
    /// The runtime was set up for an older version of the app than this one needs (see [`runtime_outdated`]).
    pub runtime_outdated: bool,
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
            runtime_outdated: runtime_outdated(),
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
    let was_outdated = s.onboarding.runtime_outdated;
    s.onboarding.runtime_outdated = runtime_outdated();
    if was_outdated && !s.onboarding.runtime_outdated {
        // Setup brought the runtime up to date: the running engine still has the old packages loaded.
        super::shell::restart_engine(s);
        super::shell::toast(s, Tone::Success, "Setup finished: the engine restarted with the updated runtime.");
        return true;
    }
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

/// The runtime version this copy of the app needs: `tools/setup/runtime_version.txt` in the bundled resources (or the
/// repo), the file setup also reads. `None` when it cannot be found (nothing to compare).
fn required_runtime() -> Option<u32> {
    let text = std::fs::read_to_string(crate::engine::find_resources()?.join("tools").join("setup").join("runtime_version.txt")).ok()?;
    text.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with('#'))?.parse().ok()
}

/// The runtime version setup wrote to `runtime.toml` (`[setup] runtime_version`); runtimes from before it count as 1.
fn installed_runtime() -> u32 {
    std::fs::read_to_string(runtime_toml_path())
        .ok()
        .and_then(|t| t.parse::<toml::Table>().ok())
        .and_then(|t| t.get("setup")?.get("runtime_version")?.as_integer())
        .map_or(1, |v| v.max(0) as u32)
}

/// An installed runtime older than this app needs (after an update that brought new packages or models).
fn runtime_outdated() -> bool {
    runtime_toml_path().is_file() && required_runtime().is_some_and(|need| installed_runtime() < need)
}

/// "Run setup" from the banner: setup again into the current data folder (it installs only what changed).
fn run_setup_update(s: &mut AppState) {
    super::setuprun::update_runtime(s);
}

/// Under the header while the runtime is older than this version needs, and while that setup runs.
pub fn runtime_banner(s: &AppState, p: Palette) -> Option<El> {
    use super::components::{Button, glass_card, icon_el, icons, progress_bar};
    use super::setuprun;
    let running = setuprun::running(s);
    if !(s.onboarding.runtime_outdated || running) || s.screen == Screen::Onboarding {
        return None;
    }
    let (title, detail) = if running {
        let step = setuprun::current(s).map(|st| st.label.clone()).unwrap_or_else(|| "Starting…".into());
        (format!("Updating the runtime: {step}"), setuprun::last_line(s).unwrap_or("").to_owned())
    } else if let setuprun::Phase::Error(e) = &s.setup.phase {
        ("The setup update stopped".to_owned(), e.clone())
    } else if let setuprun::Phase::Finished(false) = &s.setup.phase {
        ("The setup update did not finish".to_owned(), "Some steps failed; run it again to retry them.".to_owned())
    } else {
        (
            "This version needs a setup update".to_owned(),
            format!(
                "Mimvox {} uses newer engine packages or models. Setup installs only what changed; your voices and results stay as they are.",
                env!("CARGO_PKG_VERSION")
            ),
        )
    };
    let mut info = col().grow(1.0).min_w(0.0).gap(2.0)
        .child(text(title).font(ty::LABEL).color(p.text))
        .child(text(detail).font(ty::CAPTION).color(p.text_dim).wrap().max_lines(2));
    if running {
        info = info.child(el().w_full().pt(4.0).child(progress_bar(p, Some(setuprun::progress(s)))));
    }
    let button = if running {
        Button::new("Cancel").ghost().small().key("runtime-cancel").on_press(setuprun::cancel).build(p)
    } else {
        Button::new("Run setup").primary().small().icon(&icons::REFRESH).key("runtime-setup").on_press(run_setup_update).build(p)
    };
    let row_ = row().key("runtime-banner").w_full().px(14.0).py(9.0).gap(12.0).center_y()
        .child(icon_el(&icons::ALERT, 16.0, p.warn))
        .child(info)
        .child(button);
    Some(el().w_full().px(16.0).child(glass_card(p).w_full().child(row_)))
}

/// The window got the focus back.
pub fn on_focus(s: &mut AppState) {
    if (!s.onboarding.runtime_present || s.onboarding.runtime_outdated) && !s.shell.scripted() {
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
    s.onboarding.setup_started = true;
    super::setuprun::start(s, &location);
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
        kit::panel(&p).gap(18.0).child(hero).child(intro).child(steps),
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
fn location_block(p: &Palette, o: &OnboardingState, needed_gb: f32) -> El {
    let free = o.free_space.clone();
    let (line, tone_ok) = match free {
        Some((path, Some(b))) if path == o.location.text.trim() => {
            let ok = b as f64 / 1e9 >= f64::from(needed_gb);
            let short = if ok { String::new() } else { format!(" (about {needed_gb:.0} GB needed)") };
            (format!("{:.1} GB free on this drive{short}", b as f64 / 1e9), ok)
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
        .children((!o.location.text.trim().is_empty() && !std::path::Path::new(o.location.text.trim()).is_dir()).then(|| {
            text("This folder does not exist yet: setup creates it.").font(ty::CAPTION).color(p.text_dim).wrap()
        }))
}

fn engine_card(p: &Palette, s: &AppState) -> El {
    let o = &s.onboarding;
    let mut card = kit::card(p).gap(14.0).child(text("Engine and device").font(ty::TITLE).color(p.text));

    // Runtime
    let (badge, line) = if super::setuprun::running(s) {
        (kit::badge(p, "Setting up", Tone::Accent), "Installing the speech engine and its models.")
    } else if o.runtime_present {
        (kit::badge(p, "Runtime found", Tone::Success), "The speech engine is installed.")
    } else {
        (kit::badge(p, "Runtime missing", Tone::Warning), "The speech engine is not installed yet.")
    };
    card = card.child(row().w_full().gap(10.0).center_y().child(badge).child(text(line).font(ty::BODY).color(p.text_dim)));

    let run = &s.setup;
    let running = super::setuprun::running(s);
    if !o.runtime_present || running || run.phase != super::setuprun::Phase::Idle {
        if !o.runtime_present && run.phase == super::setuprun::Phase::Idle {
            card = card
                .child(
                    text("Setup downloads the speech engine and its models into the folder below. It runs here, in this window; \
                          you can leave it running and come back. If it stops, it continues where it stopped.")
                        .font(ty::CAPTION)
                        .color(p.text_dim)
                        .wrap(),
                )
                .child(super::setuprun::chooser(s, *p))
                .child(location_block(p, o, super::setuprun::chosen_gb(s) + 4.0));
        } else {
            card = card.child(super::setuprun::panel(s, *p));
        }
        match &run.phase {
            super::setuprun::Phase::Error(e) => card = card.child(kit::notice(p, Tone::Danger, e)),
            super::setuprun::Phase::Finished(true) => card = card.child(kit::notice(p, Tone::Success, "Setup finished.")),
            super::setuprun::Phase::Finished(false) => {
                card = card.child(kit::notice(p, Tone::Danger, "Some steps failed (see above). Run setup again: it retries them and keeps what is done."))
            }
            _ => {}
        }
        let mut buttons = row().gap(10.0);
        if running {
            buttons = buttons.child(kit::button(p, "Cancel", BtnKind::Glass, true, super::setuprun::cancel));
        } else if !o.runtime_present || !matches!(run.phase, super::setuprun::Phase::Finished(true)) {
            let label = if o.setup_started { "Run setup again" } else { "Run setup" };
            buttons = buttons.child(kit::button(p, label, BtnKind::Primary, true, run_setup));
        }
        if !run.log.is_empty() {
            let label = if run.show_log { "Hide details" } else { "Show details" };
            buttons = buttons.child(kit::button(p, label, BtnKind::Glass, true, super::setuprun::toggle_log));
        }
        card = card.child(buttons);
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
        ShellEngine::Unavailable { reason } if reason != super::shell::NOT_INSTALLED => {
            card = card.child(kit::notice(p, Tone::Warning, reason));
        }
        _ => {}
    }

    // Device
    if let Some(d) = &o.device {
        card = card.child(el().w_full().h(1.0).fill(p.rim_bottom));
        let backend = match (d.backend.as_deref(), d.vendor.as_deref()) {
            (Some("cuda"), Some("amd")) => "ROCm".to_string(),
            (b, _) => b.unwrap_or("unknown").to_uppercase(),
        };
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
                    "Apple GPU (MPS) is supported on a best-effort basis. Training will be noticeably slower than on an NVIDIA or AMD GPU.",
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
    Some(DeviceInfo { backend: Some(backend.into()), name: Some(name.into()), vram_gb: Some(vram), version: Some("0.1.0".into()), vendor: None })
}

fn scene_welcome(s: &mut AppState) {
    base(s);
}
fn scene_setup_missing(s: &mut AppState) {
    base(s);
}

/// Train without the voice training component: the install card.
fn scene_install_training(s: &mut AppState) {
    scene_ready_gpu(s);
    s.shell.components = Some(vec!["speech".into(), "translation".into()]);
    s.screen = Screen::Train;
}

fn scene_setup_running(s: &mut AppState) {
    base(s);
    s.onboarding.setup_started = true;
    super::setuprun::scene_running(s);
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
        ("onboarding-setup-running", scene_setup_running),
        ("install-training", scene_install_training),
        ("onboarding-ready", scene_ready_gpu),
        ("onboarding-cpu", scene_cpu_warning),
        ("onboarding-stopped", scene_engine_stopped),
    ]
}
