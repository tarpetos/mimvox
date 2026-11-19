use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, anyhow};
use clap::{Parser, Subcommand};
use mimvox_api::token::{self, TokenSource};
use mimvox_api::{ApiConfig, EngineExit, Hub};
use mimvox_engine_client::{EngineClient, EngineConfig, SpawnOptions};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_util::sync::CancellationToken;

/// Mimvox's local HTTP API without the window: starts the engine and serves it (see contract/engine-protocol.md).
#[derive(Parser)]
#[command(name = "mimvox-server", version)]
struct Cli {
    /// Address to listen on; any but loopback needs a token.
    #[arg(long, default_value = mimvox_api::DEFAULT_BIND)]
    bind: SocketAddr,
    /// Data folder (default: MIMVOX_DATA, then location.toml, then the platform default, as the app).
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// Ask for the token on loopback too.
    #[arg(long)]
    require_token: bool,
    /// Largest upload accepted by POST /files, in MiB.
    #[arg(long, default_value_t = mimvox_api::DEFAULT_MAX_UPLOAD_BYTES >> 20)]
    max_upload_mb: u64,
    /// Print the API token (creating it when there is none) and exit.
    #[arg(long)]
    show_token: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Print or replace the API token.
    Token {
        #[command(subcommand)]
        action: TokenAction,
    },
}

#[derive(Subcommand)]
enum TokenAction {
    /// Print the token, creating it when there is none.
    Show,
    /// Replace the token file with a new token.
    New,
}

/// A restart within this time counts towards giving up.
const QUICK_EXIT: Duration = Duration::from_secs(30);
const MAX_QUICK_EXITS: u32 = 3;
const RESTART_DELAY: Duration = Duration::from_secs(1);
const DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

fn main() -> ExitCode {
    #[cfg(windows)]
    mimvox_engine_client::hold_app_mutex();
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mimvox-server: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<()> {
    let data_dir = match &cli.data_dir {
        Some(dir) => std::path::absolute(dir)?,
        None => mimvox_engine_client::default_data_dir()
            .ok_or_else(|| anyhow!("cannot determine the data directory; pass --data-dir or set MIMVOX_DATA"))?,
    };
    match cli.command {
        Some(Command::Token { action: TokenAction::Show }) => return show_token(&data_dir),
        Some(Command::Token { action: TokenAction::New }) => return new_token(&data_dir),
        None if cli.show_token => return show_token(&data_dir),
        None => {}
    }
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(serve(cli, data_dir))
}

fn show_token(data_dir: &Path) -> anyhow::Result<()> {
    let (token, source) = token::load_or_create(data_dir).context("cannot read or create the token")?;
    println!("{token}");
    match source {
        TokenSource::Env => eprintln!("(from {})", token::TOKEN_ENV),
        TokenSource::File(path) => eprintln!("(from {})", path.display()),
    }
    Ok(())
}

fn new_token(data_dir: &Path) -> anyhow::Result<()> {
    let token = token::rotate(data_dir).context("cannot write the token")?;
    println!("{token}");
    eprintln!("(written to {})", token::token_path(data_dir).display());
    if token::from_env().is_some() {
        eprintln!("note: {} is set and is used instead of this file", token::TOKEN_ENV);
    }
    Ok(())
}

/// Off loopback the token must exist beforehand, so it is never made up on a reachable address unseen.
fn api_token(cli: &Cli, data_dir: &Path) -> anyhow::Result<Option<String>> {
    if cli.bind.ip().is_loopback() {
        if !cli.require_token {
            return Ok(None);
        }
        let (token, source) = token::load_or_create(data_dir).context("cannot read or create the token")?;
        if let TokenSource::File(path) = source {
            eprintln!("mimvox-server: token required, read it from {}", path.display());
        }
        return Ok(Some(token));
    }
    let token = token::existing(data_dir).context("cannot read the token")?.map(|(t, _)| t);
    mimvox_api::check_bind(cli.bind, token.as_deref())?;
    Ok(token)
}

fn spawn_engine(cfg: &EngineConfig, hub: &Hub) -> anyhow::Result<EngineClient> {
    EngineClient::spawn_with(cfg, hub.sink(), SpawnOptions { own_process_group: true })
}

async fn serve(cli: Cli, data_dir: PathBuf) -> anyhow::Result<()> {
    let token = api_token(&cli, &data_dir)?;
    let cfg = EngineConfig::discover_in(Some(data_dir.clone()))?;
    let listener =
        tokio::net::TcpListener::bind(cli.bind).await.with_context(|| format!("cannot listen on {}", cli.bind))?;
    let addr = listener.local_addr()?;

    let (hub, exits) = Hub::new();
    hub.attach(spawn_engine(&cfg, &hub)?);
    tokio::task::spawn_blocking({
        let data_dir = data_dir.clone();
        move || mimvox_api::prune_uploads(&data_dir)
    });
    tokio::spawn(log_engine(hub.clone()));

    let stop = CancellationToken::new();
    let api =
        ApiConfig { data_dir: data_dir.clone(), token, max_upload_bytes: cli.max_upload_mb.saturating_mul(1 << 20) };
    let app = mimvox_api::router(hub.clone(), api, stop.clone());
    let mut server =
        tokio::spawn(axum::serve(listener, app).with_graceful_shutdown(stop.clone().cancelled_owned()).into_future());
    let mut supervisor = tokio::spawn(supervise(hub.clone(), cfg, exits, stop.clone()));
    eprintln!("mimvox-server: listening on http://{addr} (data folder {})", data_dir.display());

    let outcome = tokio::select! {
        _ = shutdown_signal() => Ok(()),
        r = &mut server => r.map_err(anyhow::Error::from).and_then(|r| r.map_err(anyhow::Error::from)).and(Err(anyhow!("the HTTP server stopped"))),
        r = &mut supervisor => r.map_err(anyhow::Error::from).and_then(|r| r),
    };
    eprintln!("mimvox-server: stopping");
    stop.cancel();
    supervisor.abort();
    let _ = tokio::time::timeout(DRAIN_TIMEOUT, server).await;
    if let Some(engine) = hub.take_engine() {
        tokio::task::spawn_blocking(move || stop_engine(engine)).await?;
    }
    outcome
}

fn stop_engine(engine: Arc<EngineClient>) {
    match Arc::try_unwrap(engine) {
        Ok(engine) => engine.shutdown(),
        // A request is being written right now; the last owner's drop finishes the shutdown.
        Err(shared) => shared.request_shutdown(),
    }
}

async fn supervise(
    hub: Hub,
    cfg: EngineConfig,
    mut exits: UnboundedReceiver<EngineExit>,
    stop: CancellationToken,
) -> anyhow::Result<()> {
    let mut started = Instant::now();
    let mut quick_exits = 0u32;
    while let Some(exit) = exits.recv().await {
        // Gives a signal that also reached the engine time to arrive first.
        tokio::select! {
            _ = stop.cancelled() => return Ok(()),
            _ = tokio::time::sleep(RESTART_DELAY) => {}
        }
        let crashed = exit.code != Some(0);
        if crashed {
            let tail: Vec<&str> = exit.stderr_tail.lines().rev().take(20).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            eprintln!("mimvox-server: the engine exited (code {:?}); its last lines:\n{}", exit.code, tail.join("\n"));
        } else {
            eprintln!("mimvox-server: the engine shut down; starting it again");
        }
        quick_exits = if crashed && started.elapsed() < QUICK_EXIT { quick_exits + 1 } else { 0 };
        if quick_exits >= MAX_QUICK_EXITS {
            return Err(anyhow!("the engine keeps exiting right after it starts (see {})", cfg.log_path().display()));
        }
        let engine = tokio::task::spawn_blocking({
            let (cfg, hub) = (cfg.clone(), hub.clone());
            move || spawn_engine(&cfg, &hub)
        })
        .await??;
        hub.attach(engine);
        started = Instant::now();
    }
    Ok(())
}

async fn log_engine(hub: Hub) {
    let mut events = hub.subscribe();
    loop {
        match events.recv().await {
            Ok(ev) if ev.event == "ready" => {
                let version = ev.data["version"].as_str().unwrap_or("?");
                let device = ev.data["device"]["name"].as_str().unwrap_or("?");
                eprintln!("mimvox-server: engine {version} ready on {device}");
            }
            Ok(ev) if ev.event == "models" => eprintln!("mimvox-server: models {}", ev.data),
            Ok(_) | Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {}
            Err(tokio::sync::broadcast::error::RecvError::Closed) => return,
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let (Ok(mut term), Ok(mut hup)) = (signal(SignalKind::terminate()), signal(SignalKind::hangup())) else {
            let _ = tokio::signal::ctrl_c().await;
            return;
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = term.recv() => {}
            _ = hup.recv() => {}
        }
    }
    #[cfg(windows)]
    {
        let Ok(mut close) = tokio::signal::windows::ctrl_close() else {
            let _ = tokio::signal::ctrl_c().await;
            return;
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = close.recv() => {}
        }
    }
}
