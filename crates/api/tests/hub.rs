//! Hub against app/examples/fake_engine.py.

use std::path::{Path, PathBuf};
use std::time::Duration;

use mimvox_api::Hub;
use mimvox_engine_client::{EngineClient, EngineConfig};
use serde_json::json;

fn python() -> PathBuf {
    let path = std::env::var_os("PATH").unwrap_or_default();
    std::env::split_paths(&path)
        .flat_map(|d| ["python3", "python", "python.exe"].map(|n| d.join(n)))
        // The first one that starts, as the tests write it: on Windows PATH can list a Store alias that does not.
        .filter(|p| p.is_file())
        .find(|p| {
            let p = p.display().to_string().replace('\\', "/");
            std::process::Command::new(p).arg("--version").output().is_ok_and(|o| o.status.success())
        })
        .expect("python3 on PATH")
}

fn config() -> EngineConfig {
    let dir = std::env::temp_dir().join(format!("mimvox-hub-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(dir.join("data")).unwrap();
    let python = python().display().to_string().replace('\\', "/");
    std::fs::write(dir.join("runtime.toml"), format!("[python]\nengine = \"{python}\"\n")).unwrap();
    EngineConfig {
        runtime_toml: dir.join("runtime.toml"),
        engine_src: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/examples/fake_engine_src"),
        data_dir: dir.join("data"),
    }
}

#[tokio::test]
async fn a_client_that_exited_before_attach_is_dropped() {
    let cfg = config();
    let (hub, mut exits) = Hub::new();
    let client = EngineClient::spawn(&cfg, hub.sink()).unwrap();
    client.request("shutdown", json!({}));
    let exit = tokio::time::timeout(Duration::from_secs(20), exits.recv()).await.unwrap().unwrap();
    assert_eq!(exit.client, client.id());

    hub.attach(client);
    assert!(hub.take_engine().is_none());
    // Its data dir lock went with it.
    let next = EngineClient::spawn(&cfg, hub.sink()).unwrap();
    hub.attach(next);
    let hello = tokio::time::timeout(Duration::from_secs(20), hub.call("hello", json!({}))).await.unwrap().unwrap();
    assert_eq!(hello["version"], "0.1.0-fake");
    let _ = std::fs::remove_dir_all(cfg.data_dir.parent().unwrap());
}
