//! Engine client: spawns the Python engine and speaks contract/engine-protocol.md over stdin/stdout.
//! Event-driven: a reader thread hands every engine message to an [`EngineSink`] as it arrives; the UI never polls
//! and never waits on the engine.

#![allow(dead_code)]

pub mod protocol;

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use anyhow::{Context, anyhow};
use serde::Deserialize;
use serde_json::Value;

/// Folder name used inside the platform data dir.
const APP_DIR: &str = "mimvox";
/// Number of stderr lines kept for crash reports.
const STDERR_TAIL_LINES: usize = 200;

/// Where to find the engine and its runtime. `discover()` reads `<data dir>/runtime.toml`
/// (or `MIMVOX_RUNTIME`), the engine sources (`<resources>/engine/src`, where `<resources>` is found via
/// `MIMVOX_RESOURCES`, the install layout or a repo checkout; `MIMVOX_ENGINE_SRC` overrides only the sources)
/// and the data dir (see [`default_data_dir`]: `MIMVOX_DATA`, the `location.toml` pointer, the platform default).
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub runtime_toml: PathBuf,
    pub engine_src: PathBuf,
    pub data_dir: PathBuf,
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The fixed per-user folder of the pointer file (it never moves, so the data dir can).
/// Windows: `%LOCALAPPDATA%`; macOS: Application Support; Linux: `~/.config`.
fn pointer_base() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    return dirs::data_local_dir();
    #[cfg(target_os = "macos")]
    return dirs::data_dir();
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    return dirs::config_dir();
}

/// The default data dir when nothing overrides it. Windows: `%LOCALAPPDATA%` (not the roaming profile).
fn platform_data_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    return dirs::data_local_dir();
    #[cfg(not(target_os = "windows"))]
    return dirs::data_dir();
}

#[derive(Deserialize)]
struct LocationToml {
    data_dir: String,
}

/// The per-user data dir, in this order: `MIMVOX_DATA`; `data_dir = "<abs path>"` in the pointer file
/// `<pointer base>/mimvox/location.toml` (written by setup); the platform default + `mimvox`.
pub fn default_data_dir() -> Option<PathBuf> {
    if let Some(p) = env_path("MIMVOX_DATA") {
        return Some(p);
    }
    let pointed = pointer_base()
        .map(|b| b.join(APP_DIR).join("location.toml"))
        .and_then(|f| fs::read_to_string(f).ok())
        .and_then(|text| toml::from_str::<LocationToml>(&text).ok())
        .map(|l| PathBuf::from(l.data_dir.trim()))
        .filter(|p| p.is_absolute());
    pointed.or_else(|| platform_data_dir().map(|d| d.join(APP_DIR)))
}

/// True when `dir` is an engine source root (contains the `mimvox_engine` package).
fn is_engine_src(dir: &Path) -> bool {
    dir.join("mimvox_engine").join("__init__.py").is_file()
}

/// True when `dir` is a resources root (contains `engine/src/mimvox_engine`).
fn is_resources_root(dir: &Path) -> bool {
    is_engine_src(&dir.join("engine").join("src"))
}

/// Finds the resources root `<RES>` (the folder that holds `engine/`, `presets/`, `tools/`, `contract/`).
/// First hit wins (see the header of `app/Packager.toml`):
/// 1. `MIMVOX_RESOURCES`;
/// 2. next to the executable: `<exe_dir>` (Windows), `<exe_dir>/../Resources` (macOS),
///    `<exe_dir>/../lib/mimvox` (Linux deb / AppImage), `<exe_dir>/resources` (portable);
/// 3. a repo checkout: walk up from the executable's folder, then from `CARGO_MANIFEST_DIR/..`.
fn find_resources() -> Option<PathBuf> {
    if let Some(p) = env_path("MIMVOX_RESOURCES") {
        return Some(p);
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.canonicalize().ok().or(Some(p)))
        .and_then(|p| p.parent().map(Path::to_path_buf));

    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = &exe_dir {
        for rel in ["", "../Resources", "../lib/mimvox", "resources"] {
            candidates.push(if rel.is_empty() { dir.clone() } else { dir.join(rel) });
        }
        candidates.extend(dir.ancestors().map(Path::to_path_buf));
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    candidates.extend(manifest.ancestors().map(Path::to_path_buf));
    candidates.into_iter().find(|c| is_resources_root(c))
}

impl EngineConfig {
    pub fn discover() -> anyhow::Result<Self> {
        let data_dir = default_data_dir().ok_or_else(|| anyhow!("cannot determine the data directory; set MIMVOX_DATA"))?;

        let runtime_toml = match env_path("MIMVOX_RUNTIME") {
            Some(p) => p,
            None => {
                let in_data = data_dir.join("runtime.toml");
                if in_data.is_file() {
                    in_data
                } else {
                    // Portable layouts: runtime.toml next to the executable.
                    let beside_exe = std::env::current_exe()
                        .ok()
                        .and_then(|p| p.parent().map(|d| d.join("runtime.toml")))
                        .filter(|p| p.is_file());
                    beside_exe.unwrap_or(in_data)
                }
            }
        };

        let engine_src = match env_path("MIMVOX_ENGINE_SRC") {
            Some(p) => p,
            None => find_resources().map(|r| r.join("engine").join("src")).ok_or_else(|| {
                anyhow!(
                    "engine sources not found (looked next to the executable, in bundle resources and in \
                     engine/src of a repo checkout); set MIMVOX_RESOURCES or MIMVOX_ENGINE_SRC"
                )
            })?,
        };

        Ok(Self { runtime_toml, engine_src, data_dir })
    }

    /// Path of the engine log (stderr of the engine process).
    pub fn log_path(&self) -> PathBuf {
        self.data_dir.join("logs").join("engine.log")
    }
}

// ---------------------------------------------------------------------------------------------------------------
// runtime.toml
// ---------------------------------------------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct RuntimeToml {
    python: RuntimePython,
}

#[derive(Debug, Deserialize)]
struct RuntimePython {
    engine: String,
}

/// Reads `[python] engine` from runtime.toml (relative paths resolve against the file's directory).
fn read_engine_python(runtime_toml: &Path) -> anyhow::Result<PathBuf> {
    let text = fs::read_to_string(runtime_toml).with_context(|| {
        format!(
            "cannot read {} (run the setup script first, or set MIMVOX_RUNTIME)",
            runtime_toml.display()
        )
    })?;
    let parsed: RuntimeToml = toml::from_str(&text)
        .with_context(|| format!("invalid runtime.toml at {}: expected [python] engine = \"...\"", runtime_toml.display()))?;
    let python = PathBuf::from(&parsed.python.engine);
    let python = if python.is_relative() {
        runtime_toml.parent().map(|d| d.join(&python)).unwrap_or(python)
    } else {
        python
    };
    if !python.is_file() {
        return Err(anyhow!(
            "engine interpreter from runtime.toml does not exist: {} (re-run the setup script)",
            python.display()
        ));
    }
    Ok(python)
}

// ---------------------------------------------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------------------------------------------

/// Everything the engine can tell the app, in arrival order.
#[derive(Debug)]
pub enum EngineMsg {
    Ready(protocol::Hello),
    Response { id: u64, result: Result<serde_json::Value, protocol::EngineError> },
    Event(protocol::Event),
    Exited { code: Option<i32>, stderr_tail: String },
}

/// Where the client delivers what the engine says: called on the client's reader thread for every message, in
/// order (the app turns each call into a UI event, see `ui::events`). The `u64` is [`EngineClient::id`], so
/// messages from an engine that was replaced meanwhile can be told apart.
pub type EngineSink = Arc<dyn Fn(u64, EngineMsg) + Send + Sync>;

type StderrTail = Arc<Mutex<VecDeque<String>>>;

fn tail_string(tail: &StderrTail) -> String {
    let t = tail.lock().unwrap_or_else(|e| e.into_inner());
    t.iter().cloned().collect::<Vec<_>>().join("\n")
}

fn push_tail(tail: &StderrTail, line: String) {
    let mut t = tail.lock().unwrap_or_else(|e| e.into_inner());
    if t.len() >= STDERR_TAIL_LINES {
        t.pop_front();
    }
    t.push_back(line);
}

/// How long `shutdown` lets the engine exit by itself before it is killed.
const SHUTDOWN_GRACE: Duration = Duration::from_millis(2500);
/// How long the exit report waits for the last stderr lines after the process is gone.
const STDERR_FLUSH: Duration = Duration::from_millis(300);

/// A running engine process. Event-driven: a reader thread blocks on the engine's stdout and hands every message to
/// the [`EngineSink`] as it arrives; nothing polls. Requests are written to stdin right away and never wait.
pub struct EngineClient {
    id: u64,
    child: Arc<Mutex<Child>>,
    stdin: Mutex<Option<ChildStdin>>,
    sink: EngineSink,
    /// Signalled by the stdout reader once the process has exited (after stdout closed).
    exited: Mutex<Receiver<()>>,
    next_id: AtomicU64,
    stderr_tail: StderrTail,
    log_path: PathBuf,
    shut_down: AtomicBool,
}

impl EngineClient {
    pub fn spawn(cfg: &EngineConfig, sink: EngineSink) -> anyhow::Result<Self> {
        static NEXT_CLIENT: AtomicU64 = AtomicU64::new(1);
        let client_id = NEXT_CLIENT.fetch_add(1, Ordering::Relaxed);
        let python = read_engine_python(&cfg.runtime_toml)?;
        if !is_engine_src(&cfg.engine_src) {
            return Err(anyhow!(
                "engine sources not found at {} (expected mimvox_engine/__init__.py inside)",
                cfg.engine_src.display()
            ));
        }
        fs::create_dir_all(&cfg.data_dir)
            .with_context(|| format!("cannot create data dir {}", cfg.data_dir.display()))?;

        let mut pythonpath = OsString::from(&cfg.engine_src);
        if let Some(existing) = std::env::var_os("PYTHONPATH").filter(|v| !v.is_empty()) {
            let joined = std::env::join_paths(std::env::split_paths(&pythonpath).chain(std::env::split_paths(&existing)))
                .context("cannot build PYTHONPATH")?;
            pythonpath = joined;
        }

        let mut cmd = Command::new(&python);
        // Presets live beside the engine sources in the resources root (`<RES>/engine/src` -> `<RES>/presets`).
        if std::env::var_os("MIMVOX_PRESETS").is_none_or(|v| v.is_empty()) {
            let presets = cfg.engine_src.parent().and_then(Path::parent).map(|r| r.join("presets"));
            if let Some(dir) = presets.filter(|d| d.is_dir()) {
                cmd.env("MIMVOX_PRESETS", dir);
            }
        }
        cmd.arg("-m")
            .arg("mimvox_engine")
            .arg("serve")
            .arg("--runtime")
            .arg(&cfg.runtime_toml)
            .arg("--data-dir")
            .arg(&cfg.data_dir)
            .env("PYTHONPATH", pythonpath)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("cannot start the engine with {}", python.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("engine stdin unavailable"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("engine stdout unavailable"))?;
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("engine stderr unavailable"))?;

        let child = Arc::new(Mutex::new(child));
        let stderr_tail: StderrTail = Arc::new(Mutex::new(VecDeque::new()));
        let log_path = cfg.log_path();
        let (stderr_done_tx, stderr_done_rx) = mpsc::channel::<()>();
        let (exited_tx, exited_rx) = mpsc::channel::<()>();

        // stderr: tail ring buffer + log file. Blocks on each line; ends at EOF.
        {
            let tail = stderr_tail.clone();
            let log_path = log_path.clone();
            thread::Builder::new()
                .name("engine-stderr".into())
                .spawn(move || {
                    let mut log = log_path
                        .parent()
                        .and_then(|d| fs::create_dir_all(d).ok())
                        .and_then(|_| File::create(&log_path).ok());
                    for line in BufReader::new(stderr).split(b'\n') {
                        let Ok(line) = line else { break };
                        let line = String::from_utf8_lossy(&line).trim_end_matches(['\r', '\n']).to_string();
                        if let Some(f) = log.as_mut() {
                            let _ = writeln!(f, "{line}");
                        }
                        push_tail(&tail, line);
                    }
                    let _ = stderr_done_tx.send(());
                })
                .context("cannot start the stderr reader")?;
        }

        // stdout: every protocol line becomes a message for the sink as soon as it is read; at EOF the exit is reported.
        {
            let sink = sink.clone();
            let child = child.clone();
            let tail = stderr_tail.clone();
            thread::Builder::new()
                .name("engine-stdout".into())
                .spawn(move || {
                    for line in BufReader::new(stdout).split(b'\n') {
                        let Ok(line) = line else { break };
                        let line = String::from_utf8_lossy(&line);
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        match parse_line(line) {
                            Ok(Some(msg)) => sink(client_id, msg),
                            Ok(None) => {}
                            // Not protocol traffic (a library printed to stdout): keep it for diagnostics.
                            Err(()) => push_tail(&tail, format!("[stdout] {line}")),
                        }
                    }
                    // EOF: the process is ending. Wait for it (stdout closes at the very end of the interpreter's
                    // shutdown), then give stderr a moment to deliver its last lines.
                    let code = child.lock().unwrap_or_else(|e| e.into_inner()).wait().ok().and_then(|s| s.code());
                    let _ = exited_tx.send(());
                    let _ = stderr_done_rx.recv_timeout(STDERR_FLUSH);
                    sink(client_id, EngineMsg::Exited { code, stderr_tail: tail_string(&tail) });
                })
                .context("cannot start the stdout reader")?;
        }

        Ok(Self {
            id: client_id,
            child,
            stdin: Mutex::new(Some(stdin)),
            sink,
            exited: Mutex::new(exited_rx),
            next_id: AtomicU64::new(1),
            stderr_tail,
            log_path,
            shut_down: AtomicBool::new(false),
        })
    }

    /// Unique per spawned process (the first argument of every [`EngineSink`] call).
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Sends a request and returns its id at once; the answer arrives later as `EngineMsg::Response`.
    /// If the engine cannot be written to, an `internal` error response is delivered to the sink instead.
    pub fn request(&self, method: &str, params: serde_json::Value) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let line = serde_json::json!({ "id": id.to_string(), "method": method, "params": params }).to_string();
        if let Err(e) = self.write_line(&line) {
            (self.sink)(
                self.id,
                EngineMsg::Response { id, result: Err(protocol::EngineError::new("internal", format!("engine is not running: {e}"))) },
            );
        }
        id
    }

    /// Same as `request` for the typed builders in `protocol` (e.g. `engine.send(protocol::voices_list())`).
    pub fn send(&self, req: protocol::Request) -> u64 {
        self.request(req.0, req.1)
    }

    fn write_line(&self, line: &str) -> std::io::Result<()> {
        let mut guard = self.stdin.lock().unwrap_or_else(|e| e.into_inner());
        let stdin = guard
            .as_mut()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::BrokenPipe, "stdin closed"))?;
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()
    }

    /// Last lines the engine wrote to stderr.
    pub fn stderr_tail(&self) -> String {
        tail_string(&self.stderr_tail)
    }

    /// Log file with the engine's stderr (truncated at every start).
    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    /// Asks the engine to exit and waits (blocking, up to [`SHUTDOWN_GRACE`]) for the stdout reader to see it go;
    /// kills it when it does not.
    /// Asks the engine to exit (the shutdown request, then end of stdin) without waiting for it: for the UI
    /// thread while the window is still on screen. [`EngineClient::shutdown`] (or dropping the client) finishes it.
    pub fn request_shutdown(&self) {
        let (method, params) = protocol::shutdown();
        let line = serde_json::json!({ "id": "0", "method": method, "params": params }).to_string();
        let _ = self.write_line(&line);
        self.stdin.lock().unwrap_or_else(|e| e.into_inner()).take();
    }

    pub fn shutdown(&mut self) {
        if self.shut_down.swap(true, Ordering::SeqCst) {
            return;
        }
        // Closing stdin also makes the engine exit (a no-op after `request_shutdown`).
        self.request_shutdown();

        let exited = self.exited.lock().unwrap_or_else(|e| e.into_inner()).recv_timeout(SHUTDOWN_GRACE);
        if exited.is_ok() {
            return;
        }
        // Still running. The reader holds the child only once stdout has closed (the process is then already at
        // its very end), so never block on it here.
        if let Ok(mut child) = self.child.try_lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for EngineClient {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Parses one stdout line. `Err(())` = not protocol traffic; `Ok(None)` = protocol line nobody waits for.
fn parse_line(line: &str) -> Result<Option<EngineMsg>, ()> {
    let v: Value = serde_json::from_str(line).map_err(|_| ())?;
    let obj = v.as_object().ok_or(())?;

    if let Some(name) = obj.get("event").and_then(Value::as_str) {
        let data = obj.get("data").cloned().unwrap_or(Value::Null);
        if name == "ready" {
            let hello = serde_json::from_value(data).unwrap_or_default();
            return Ok(Some(EngineMsg::Ready(hello)));
        }
        return Ok(Some(EngineMsg::Event(protocol::Event {
            event: name.to_string(),
            job: obj.get("job").and_then(Value::as_str).map(str::to_string),
            data,
        })));
    }

    let id = match obj.get("id").ok_or(())? {
        Value::String(s) => s.parse::<u64>().map_err(|_| ())?,
        Value::Number(n) => n.as_u64().ok_or(())?,
        _ => return Err(()),
    };
    if id == 0 {
        // Answer to the internal shutdown request.
        return Ok(None);
    }
    let result = if let Some(err) = obj.get("error").filter(|e| !e.is_null()) {
        Err(serde_json::from_value(err.clone()).unwrap_or_else(|_| {
            protocol::EngineError::new("internal", format!("malformed error from engine: {err}"))
        }))
    } else {
        Ok(obj.get("result").cloned().unwrap_or(Value::Null))
    };
    Ok(Some(EngineMsg::Response { id, result }))
}
