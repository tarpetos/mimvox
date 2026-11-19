use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use mimvox_engine_client::EngineConfig;
use serde::Deserialize;
use tokio::process::Command;

use crate::ApiError;

const EXE: &str = std::env::consts::EXE_SUFFIX;
const REMUX_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Deserialize, Default)]
struct RuntimeToml {
    #[serde(default)]
    paths: RuntimePaths,
}

#[derive(Deserialize, Default)]
struct RuntimePaths {
    applio: Option<String>,
    omnivoice_model: Option<String>,
}

fn file_in(folder: &Path) -> Option<PathBuf> {
    Some(folder.join(format!("ffmpeg{EXE}"))).filter(|p| p.is_file())
}

/// Where the engine looks for ffmpeg (contract, "Audio formats"): `MIMVOX_FFMPEG`, Applio's copy, the portable one in
/// `<runtime>/ffmpeg`, then `PATH`.
pub fn find(data_dir: &Path) -> Option<PathBuf> {
    if let Some(env) = std::env::var_os("MIMVOX_FFMPEG").filter(|v| !v.is_empty()).map(PathBuf::from) {
        let folder = if env.is_dir() { Some(env.as_path()) } else { env.parent() };
        if let Some(found) = folder.and_then(file_in) {
            return Some(found);
        }
    }
    let runtime_toml = EngineConfig::discover_in(Some(data_dir.to_path_buf())).ok().map(|c| c.runtime_toml);
    if let Some(runtime_toml) = runtime_toml {
        let paths = std::fs::read_to_string(&runtime_toml)
            .ok()
            .and_then(|text| toml::from_str::<RuntimeToml>(&text).ok())
            .unwrap_or_default()
            .paths;
        let base = runtime_toml.parent().unwrap_or(Path::new(""));
        let applio = paths.applio.map(|p| base.join(p));
        let root = match &applio {
            Some(applio) => applio.parent().map(Path::to_path_buf),
            None => paths.omnivoice_model.and_then(|m| base.join(m).parent()?.parent().map(Path::to_path_buf)),
        };
        if let Some(found) = [applio, root.map(|r| r.join("ffmpeg"))].iter().flatten().find_map(|f| file_in(f)) {
            return Some(found);
        }
    }
    std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).find_map(|d| file_in(&d))
}

/// The AAC track of an MP4 file as a raw ADTS stream, copied without re-encoding.
pub async fn mp4_to_adts(ffmpeg: &Path, input: &Path) -> Result<Vec<u8>, ApiError> {
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-nostdin", "-v", "error", "-i"])
        .arg(input)
        .args(["-map", "0:a:0", "-c:a", "copy", "-f", "adts", "pipe:1"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let out = tokio::time::timeout(REMUX_TIMEOUT, cmd.output())
        .await
        .map_err(|_| ApiError::internal("ffmpeg took too long to remux the AAC"))?
        .map_err(|e| ApiError::internal(format!("ffmpeg could not be started: {e}")))?;
    if !out.status.success() || out.stdout.is_empty() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(ApiError::internal(format!("ffmpeg could not remux the AAC: {}", err.trim())));
    }
    Ok(out.stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn m4a_becomes_adts() {
        let Some(ffmpeg) =
            std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).find_map(|d| file_in(&d))
        else {
            eprintln!("skipped: no ffmpeg on PATH");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let m4a = dir.path().join("x.m4a");
        let made = std::process::Command::new(&ffmpeg)
            .args(["-nostdin", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac"])
            .arg(&m4a)
            .status()
            .unwrap();
        if !made.success() {
            eprintln!("skipped: this ffmpeg cannot encode AAC");
            return;
        }
        let adts = mp4_to_adts(&ffmpeg, &m4a).await.unwrap();
        assert!(adts[0] == 0xFF && adts[1] & 0xF6 == 0xF0, "no ADTS sync word: {:02x?}", &adts[..2]);
        assert!(mp4_to_adts(&ffmpeg, &dir.path().join("missing.m4a")).await.is_err());
    }
}
