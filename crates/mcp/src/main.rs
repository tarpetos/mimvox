mod api;
mod events;
mod server;

use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, anyhow};
use clap::Parser;
use mimvox_api::token;
use rmcp::ServiceExt;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use tokio_util::sync::CancellationToken;

use crate::api::Api;
use crate::events::Tracker;
use crate::server::Mimvox;

/// MCP server for Mimvox: a client of its local HTTP API (see contract/engine-protocol.md, "MCP server").
#[derive(Parser)]
#[command(name = "mimvox-mcp", version)]
struct Cli {
    /// The Mimvox API (the app's API setting, or mimvox-server / mimvox serve).
    #[arg(long, env = "MIMVOX_URL", default_value = "http://127.0.0.1:8765")]
    url: String,
    /// A file holding the API token (MIMVOX_API_TOKEN wins; default on loopback: <data>/api/token when it exists).
    #[arg(long)]
    token_file: Option<PathBuf>,
    /// Serve MCP over streamable HTTP at ADDR (a port alone means 127.0.0.1:PORT) on /mcp, instead of stdio.
    #[arg(long, value_name = "ADDR")]
    http: Option<String>,
    /// Let --http listen on an address other than loopback. The MCP endpoint has no authentication of its own.
    #[arg(long)]
    http_allow_remote: bool,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("mimvox-mcp: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<()> {
    let (token, from) = api_token(&cli)?;
    let api = Arc::new(Api::new(&cli.url, token)?);
    eprintln!("mimvox-mcp: Mimvox API {}{from}", api.base());
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(async move {
        let events = Tracker::start(api.clone());
        let server = Mimvox::new(api, events);
        match &cli.http {
            Some(addr) => serve_http(server, http_addr(addr)?, cli.http_allow_remote).await,
            None => serve_stdio(server).await,
        }
    })
}

/// `MIMVOX_API_TOKEN`, else --token-file, else on loopback the server's own token file when it exists.
fn api_token(cli: &Cli) -> anyhow::Result<(Option<String>, String)> {
    if let Some(t) = token::from_env() {
        return Ok((Some(t), format!(", token from {}", token::TOKEN_ENV)));
    }
    if let Some(path) = &cli.token_file {
        let text = std::fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
        let t = text.trim();
        if t.is_empty() {
            return Err(anyhow!("{} is empty", path.display()));
        }
        return Ok((Some(t.to_string()), format!(", token from {}", path.display())));
    }
    let loopback = reqwest::Url::parse(&cli.url).ok().and_then(|u| u.host_str().map(str::to_string)).is_some_and(|h| {
        let h = h.trim_start_matches('[').trim_end_matches(']');
        h.eq_ignore_ascii_case("localhost") || h.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
    });
    if loopback && let Some(data) = mimvox_engine_client::default_data_dir() {
        let path = token::token_path(&data);
        if let Ok(text) = std::fs::read_to_string(&path)
            && !text.trim().is_empty()
        {
            return Ok((Some(text.trim().to_string()), format!(", token from {}", path.display())));
        }
    }
    Ok((None, String::new()))
}

fn http_addr(addr: &str) -> anyhow::Result<SocketAddr> {
    if let Ok(port) = addr.parse::<u16>() {
        return Ok(SocketAddr::from((Ipv4Addr::LOCALHOST, port)));
    }
    addr.parse().with_context(|| format!("--http wants ADDR:PORT or a port, not {addr:?}"))
}

async fn serve_stdio(server: Mimvox) -> anyhow::Result<()> {
    let service = server.serve(rmcp::transport::stdio()).await.context("MCP over stdio")?;
    service.waiting().await?;
    Ok(())
}

async fn serve_http(server: Mimvox, addr: SocketAddr, allow_remote: bool) -> anyhow::Result<()> {
    if !addr.ip().is_loopback() && !allow_remote {
        return Err(anyhow!(
            "refusing to serve MCP on {addr}: it has no authentication of its own and uses the API's token; pass \
             --http-allow-remote to do it anyway"
        ));
    }
    let stop = CancellationToken::new();
    let mut config = StreamableHttpServerConfig::default().with_cancellation_token(stop.child_token());
    if !addr.ip().is_loopback() {
        config = config.disable_allowed_hosts();
    }
    let service: StreamableHttpService<Mimvox, LocalSessionManager> =
        StreamableHttpService::new(move || Ok(server.clone()), Default::default(), config);
    let app = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("cannot listen on {addr}"))?;
    eprintln!("mimvox-mcp: MCP over HTTP at http://{}/mcp", listener.local_addr()?);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            stop.cancel();
        })
        .await?;
    Ok(())
}
