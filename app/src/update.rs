//! App updates: finding a newer release, downloading it and installing it when the app closes.
//!
//! Every release (built by `.github/workflows/build.yml`) publishes `latest.json`: the version, its notes and, for each
//! platform, the installer to download and its signature (minisign, made with the project's private key, which only
//! the release workflow holds). The app asks for `latest.json` once per launch ([`check`]), downloads the installer
//! for its platform on request ([`download`]: the signature is verified against [`PUBKEY`] before anything is kept),
//! and installs it after the window is gone and the engine has exited ([`install_pending`], called by `main`):
//!
//! * Windows (setup .exe, or .msi when installed from one): the installer runs without questions; it starts the app
//!   again when the user chose "Restart now".
//! * Linux AppImage: the new AppImage replaces the file the app was started from.
//! * macOS: the new `Mimvox.app` replaces the installed one.
//!
//! A `.deb` install, a loose binary or an app bundle outside an `.app` cannot replace itself: the app only says that
//! a new version exists and opens its download page ([`Kind::Manual`]). A build from a repo checkout never checks.
//! The check is the app's only network request of its own; `"check_updates": false` in settings.json turns it off.

use cargo_packager_updater::{Config, Update, UpdaterBuilder, WindowsConfig, WindowsUpdateInstallMode};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// The newest release's manifest (GitHub serves the latest non-pre-release's asset at this address).
const ENDPOINT: &str = "https://github.com/tarpetos/mimvox/releases/latest/download/latest.json";
/// The newest release's page (what's new, the installers for a manual update).
pub const RELEASE_PAGE: &str = "https://github.com/tarpetos/mimvox/releases/latest";
/// The public half of the key the release workflow signs installers with (`cargo packager signer generate`).
const PUBKEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEQ5RTBDMENBNjhDNkQ0MTUKUldRVjFNWm95c0RnMlNRSXdyc0NFd3BFTlYrcGdmb1BjMG5aQy8vZXBza2hpVWcrREsxYzVOQjAK";

/// How this copy of the app can be updated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Download, verify and install by itself.
    Auto,
    /// Only point at the download page (a .deb, a loose binary).
    Manual,
    /// A development build from a repo checkout: never checks.
    Never,
}

pub fn kind() -> Kind {
    if crate::engine::find_resources().is_some_and(|r| r.join("app").join("Cargo.toml").is_file()) {
        return Kind::Never;
    }
    if cfg!(target_os = "linux") {
        return if std::env::var_os("APPIMAGE").is_some() { Kind::Auto } else { Kind::Manual };
    }
    if cfg!(target_os = "macos") {
        return if app_bundle().is_some() { Kind::Auto } else { Kind::Manual };
    }
    Kind::Auto
}

/// The `.app` bundle the running executable is in (macOS).
fn app_bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    exe.ancestors().find(|a| a.extension().is_some_and(|e| e == "app")).map(PathBuf::from)
}

/// The manifest entry for this install: `<os>-<arch>`, except a Windows copy installed from the .msi (under Program
/// Files; the setup .exe installs per user), which updates with the .msi (`windows-<arch>-msi`).
fn target() -> Option<String> {
    if !cfg!(windows) {
        return None;
    }
    let exe = std::env::current_exe().ok()?;
    let program_files = ["ProgramFiles", "ProgramW6432"].iter().filter_map(std::env::var_os).map(PathBuf::from);
    let from_msi = program_files.into_iter().any(|dir| exe.starts_with(dir));
    from_msi.then(|| format!("windows-{}-msi", std::env::consts::ARCH))
}

fn config(relaunch: bool) -> Config {
    // NSIS: "/P" installs with only a progress bar, "/R" starts the app again afterwards.
    let windows = if relaunch {
        WindowsConfig { install_mode: Some(WindowsUpdateInstallMode::Passive), installer_args: None }
    } else {
        WindowsConfig { install_mode: Some(WindowsUpdateInstallMode::BasicUi), installer_args: Some(vec!["/P".into()]) }
    };
    Config { endpoints: vec![ENDPOINT.parse().expect("valid endpoint")], pubkey: PUBKEY.into(), windows: Some(windows) }
}

/// A newer version than this one.
#[derive(Clone, Debug)]
pub struct Found {
    pub version: String,
    /// The release notes (Markdown), when the manifest has them.
    pub notes: Option<String>,
    update: Update,
}

/// Asks for the newest release (blocking: call it on a background thread). `Ok(None)`: this is the newest.
pub fn check() -> Result<Option<Found>, String> {
    let current = env!("CARGO_PKG_VERSION").parse().map_err(|e| format!("{e}"))?;
    let mut builder = UpdaterBuilder::new(current, config(false)).timeout(Duration::from_secs(30));
    if let Some(t) = target() {
        builder = builder.target(t);
    }
    let update = builder.build().map_err(|e| e.to_string())?.check().map_err(|e| e.to_string())?;
    Ok(update.map(|u| Found { version: u.version.clone(), notes: u.body.clone(), update: u }))
}

/// A downloaded, verified installer, waiting for the app to close.
static READY: Mutex<Option<(Update, Vec<u8>)>> = Mutex::new(None);
/// Start the app again after installing ("Restart now"), not only install (a plain close).
static RELAUNCH: AtomicBool = AtomicBool::new(false);

/// Downloads the installer of `found` and verifies its signature (blocking: call it on a background thread).
/// `progress` gets the bytes received so far and the size when known.
pub fn download(found: &Found, progress: impl Fn(u64, Option<u64>)) -> Result<(), String> {
    let got = std::sync::atomic::AtomicU64::new(0);
    let bytes = found
        .update
        .download_extended(|n, total| progress(got.fetch_add(n as u64, Ordering::Relaxed) + n as u64, total), || {})
        .map_err(|e| e.to_string())?;
    *READY.lock().unwrap_or_else(|e| e.into_inner()) = Some((found.update.clone(), bytes));
    Ok(())
}

/// "Restart now": install when the app closes, then start it again.
pub fn relaunch_after_install() {
    RELAUNCH.store(true, Ordering::Relaxed);
}

/// Called by `main` once the window is gone and the engine has exited: installs a downloaded update. On Windows the
/// installer takes over and this process ends here.
pub fn install_pending() {
    let Some((mut update, bytes)) = READY.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let relaunch = RELAUNCH.load(Ordering::Relaxed);
    update.config = config(relaunch);
    if let Err(e) = update.install(bytes) {
        eprintln!("mimvox: the update could not be installed: {e}");
        return;
    }
    if relaunch {
        let started = if cfg!(target_os = "macos") {
            app_bundle().map(|app| std::process::Command::new("open").arg("-n").arg(app).spawn())
        } else {
            std::env::var_os("APPIMAGE").map(|image| std::process::Command::new(image).spawn())
        };
        if let Some(Err(e)) = started {
            eprintln!("mimvox: the updated app could not be started: {e}");
        }
    }
}
