//! Spawns the engine, prints hello, lists voices, runs tts.generate and plays chunks as they arrive.
//!
//! Real engine:  cargo run --example engine_smoke -- --text "Hello there." [--voice ID]
//!               (uses EngineConfig::discover(); override with --runtime, --data-dir, --engine-src)
//! Fake engine:  cargo run --example engine_smoke -- --fake --text "One. Two. Three."
//!               (examples/fake_engine.py through a shim package; needs python3 on PATH)
//! Add --no-play to skip audio output.

#[path = "../src/audio/mod.rs"]
mod audio;
#[path = "../src/engine/mod.rs"]
mod engine;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use engine::protocol::{self, EngineEvent};
use engine::{EngineClient, EngineConfig, EngineMsg};

struct Args {
    runtime: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    engine_src: Option<PathBuf>,
    voice: Option<String>,
    text: String,
    fake: bool,
    no_play: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut a = Args {
        runtime: None,
        data_dir: None,
        engine_src: None,
        voice: None,
        text: "Hello there. This is a quick test of my own voice.".to_string(),
        fake: false,
        no_play: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| anyhow::anyhow!("{name} needs a value"));
        match arg.as_str() {
            "--runtime" => a.runtime = Some(value("--runtime")?.into()),
            "--data-dir" => a.data_dir = Some(value("--data-dir")?.into()),
            "--engine-src" => a.engine_src = Some(value("--engine-src")?.into()),
            "--voice" => a.voice = Some(value("--voice")?),
            "--text" => a.text = value("--text")?,
            "--fake" => a.fake = true,
            "--no-play" => a.no_play = true,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    Ok(a)
}

fn find_python() -> anyhow::Result<PathBuf> {
    let path = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path) {
        for name in ["python3", "python", "python.exe"] {
            let p = dir.join(name);
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    anyhow::bail!("python3 not found on PATH (needed for --fake)")
}

fn build_config(a: &Args) -> anyhow::Result<EngineConfig> {
    if a.fake {
        let data_dir = a.data_dir.clone().unwrap_or_else(|| std::env::temp_dir().join("mimvox-fake-engine-data"));
        std::fs::create_dir_all(&data_dir)?;
        let runtime = a.runtime.clone().unwrap_or_else(|| data_dir.join("runtime.toml"));
        if !runtime.exists() {
            let py = find_python()?;
            std::fs::write(&runtime, format!("[python]\nengine = '{}'\nrvc = '{}'\n", py.display(), py.display()))?;
        }
        let engine_src = a
            .engine_src
            .clone()
            .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/fake_engine_src"));
        return Ok(EngineConfig { runtime_toml: runtime, engine_src, data_dir });
    }
    let mut cfg = EngineConfig::discover()?;
    if let Some(p) = &a.runtime {
        cfg.runtime_toml = p.clone();
    }
    if let Some(p) = &a.data_dir {
        cfg.data_dir = p.clone();
    }
    if let Some(p) = &a.engine_src {
        cfg.engine_src = p.clone();
    }
    Ok(cfg)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("engine_smoke: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let args = parse_args()?;
    let cfg = build_config(&args)?;
    println!("config: {cfg:#?}");

    let mut engine = EngineClient::spawn(&cfg)?;
    println!("engine log: {}", engine.log_path().display());

    let player = if args.no_play {
        None
    } else {
        match audio::Player::new() {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("warning: playback disabled: {e:#}");
                None
            }
        }
    };

    // Wait for `ready`.
    let started = Instant::now();
    let mut ready = false;
    let mut voices_req: Option<u64> = None;
    let mut tts_req: Option<u64> = None;
    let mut job: Option<String> = None;
    let mut voice_id = args.voice.clone();
    let mut finished = false;

    while started.elapsed() < Duration::from_secs(300) {
        for msg in engine.poll() {
            match msg {
                EngineMsg::Ready(hello) => {
                    println!("hello: {hello:#?}");
                    ready = true;
                    voices_req = Some(engine.send(protocol::voices_list()));
                }
                EngineMsg::Response { id, result } if Some(id) == voices_req => {
                    let list: protocol::VoicesList = protocol::parse_result(result.map_err(anyhow::Error::from)?)?;
                    println!("voices: {} found", list.voices.len());
                    for v in &list.voices {
                        println!("  {} \"{}\" trained={} parts={}", v.id, v.name, v.trained, v.parts);
                    }
                    if voice_id.is_none() {
                        voice_id = list.voices.first().map(|v| v.id.clone());
                    }
                    let Some(vid) = &voice_id else {
                        anyhow::bail!("no voices to speak with; create one first");
                    };
                    println!("tts.generate voice={vid} text={:?}", args.text);
                    tts_req = Some(engine.send(protocol::tts_generate(vid, &args.text, 0.8, 1.0, protocol::Quality::Best, 1.0)));
                }
                EngineMsg::Response { id, result } if Some(id) == tts_req => {
                    let started: protocol::JobStarted = protocol::parse_result(result.map_err(anyhow::Error::from)?)?;
                    println!("job {}", started.job);
                    job = Some(started.job);
                }
                EngineMsg::Response { id, result } => println!("response {id}: {result:?}"),
                EngineMsg::Event(ev) => match ev.parse() {
                    EngineEvent::TtsChunk { chunk, .. } => {
                        println!("chunk {} {:.2}s {}", chunk.index, chunk.duration_s, chunk.path.display());
                        if let Some(p) = &player {
                            if let Err(e) = p.enqueue(&chunk.path) {
                                eprintln!("cannot play chunk: {e:#}");
                            }
                        }
                    }
                    EngineEvent::TtsDone { done, .. } => {
                        println!(
                            "done: {} ({:.2}s audio in {:.2}s)",
                            done.path_wav.display(),
                            done.duration_s,
                            done.elapsed_s
                        );
                        finished = true;
                    }
                    EngineEvent::TtsError { error, .. } => anyhow::bail!("tts failed: {error}"),
                    other => println!("event: {other:?}"),
                },
                EngineMsg::Exited { code, stderr_tail } => {
                    anyhow::bail!("engine exited (code {code:?}); stderr tail:\n{stderr_tail}");
                }
            }
        }
        if finished && player.as_ref().map_or(true, |p| !p.is_playing()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(30));
    }
    if !ready {
        anyhow::bail!("engine did not become ready in time; stderr tail:\n{}", engine.stderr_tail());
    }
    if !finished {
        anyhow::bail!("timed out waiting for tts.done (job {job:?})");
    }
    println!("shutting down");
    engine.shutdown();
    println!("ok");
    Ok(())
}
