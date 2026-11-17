use cargo_packager_updater::semver::Version;
use cargo_packager_updater::{Config, Update, UpdaterBuilder, WindowsConfig, WindowsUpdateInstallMode, reqwest};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// GitHub serves the latest non-pre-release's asset at this address.
const ENDPOINT: &str = "https://github.com/tarpetos/mimvox/releases/latest/download/latest.json";
const RELEASES_API: &str = "https://api.github.com/repos/tarpetos/mimvox/releases?per_page=100";
/// 0.2.0 is the first release with signed installers and a `latest.json`.
pub const OLDEST: &str = "0.2.0";
/// Public half of the release workflow's signing key (`cargo packager signer generate`).
const PUBKEY: &str = "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IEQ5RTBDMENBNjhDNkQ0MTUKUldRVjFNWm95c0RnMlNRSXdyc0NFd3BFTlYrcGdmb1BjMG5aQy8vZXBza2hpVWcrREsxYzVOQjAK";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Auto,
    /// A .deb or a loose binary: only point at the download page.
    Manual,
    /// A build from a repo checkout.
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

fn app_bundle() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?.canonicalize().ok()?;
    exe.ancestors().find(|a| a.extension().is_some_and(|e| e == "app")).map(PathBuf::from)
}

pub fn release_page(version: &str) -> String {
    format!("https://github.com/tarpetos/mimvox/releases/tag/v{version}")
}

fn manifest(version: &str) -> String {
    format!("https://github.com/tarpetos/mimvox/releases/download/v{version}/latest.json")
}

fn config(relaunch: bool) -> Config {
    config_at(ENDPOINT, relaunch)
}

fn config_at(endpoint: &str, relaunch: bool) -> Config {
    // The updater passes nothing of its own with BasicUi; these are Inno Setup's options for a quiet install.
    let mut args: Vec<String> = ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART"].map(String::from).to_vec();
    if relaunch {
        args.push("/RELAUNCH=1".into());
    }
    let windows = WindowsConfig { install_mode: Some(WindowsUpdateInstallMode::BasicUi), installer_args: Some(args) };
    let endpoint = endpoint.parse().unwrap_or_else(|_| ENDPOINT.parse().expect("valid endpoint"));
    Config { endpoints: vec![endpoint], pubkey: PUBKEY.into(), windows: Some(windows) }
}

#[derive(Clone, Debug)]
pub struct Found {
    pub version: String,
    pub notes: Option<String>,
    /// `None` only in a headless scene ([`Found::preview`]).
    update: Option<Update>,
}

impl Found {
    /// Shows like a real one but cannot be downloaded.
    pub fn preview(version: &str, notes: Option<&str>) -> Self {
        Found { version: version.to_owned(), notes: notes.map(str::to_owned), update: None }
    }
}

fn current() -> Result<Version, String> {
    env!("CARGO_PKG_VERSION").parse().map_err(|e| format!("{e}"))
}

/// `Greater` when `version` is newer than the installed one; an unreadable version sorts as older.
pub fn compare(version: &str) -> std::cmp::Ordering {
    match (Version::parse(version), current()) {
        (Ok(v), Ok(c)) => v.cmp(&c),
        _ => std::cmp::Ordering::Less,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Release {
    pub version: String,
    /// `2026-11-14`, empty when unknown.
    pub date: String,
    pub notes: String,
}

#[derive(serde::Deserialize)]
struct ApiRelease {
    tag_name: String,
    #[serde(default)]
    body: Option<String>,
    #[serde(default)]
    published_at: Option<String>,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
}

/// From [`OLDEST`] on, newest first, without drafts and pre-releases. Blocking.
pub fn releases() -> Result<Vec<Release>, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .user_agent(concat!("mimvox/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client.get(RELEASES_API).header("Accept", "application/vnd.github+json").send().map_err(|e| e.to_string())?;
    let status = response.status();
    if status.as_u16() == 403 || status.as_u16() == 429 {
        return Err("GitHub is limiting requests from this network for now. Try again in an hour.".into());
    }
    if !status.is_success() {
        return Err(format!("GitHub answered {status}."));
    }
    let list: Vec<ApiRelease> = response.json().map_err(|e| e.to_string())?;
    Ok(listed(list))
}

fn listed(list: Vec<ApiRelease>) -> Vec<Release> {
    let oldest = Version::parse(OLDEST).expect("valid OLDEST");
    let mut out: Vec<(Version, Release)> = list
        .into_iter()
        .filter(|r| !r.draft && !r.prerelease)
        .filter_map(|r| {
            let v = Version::parse(r.tag_name.trim_start_matches('v')).ok().filter(|v| *v >= oldest && v.pre.is_empty())?;
            let date = r.published_at.as_deref().and_then(|d| d.get(..10)).unwrap_or_default().to_owned();
            Some((v.clone(), Release { version: v.to_string(), date, notes: r.body.unwrap_or_default() }))
        })
        .collect();
    out.sort_by(|a, b| b.0.cmp(&a.0));
    out.dedup_by(|a, b| a.0 == b.0);
    out.into_iter().map(|(_, r)| r).collect()
}

/// Blocking. The installed version itself is refused.
pub fn fetch(version: &str) -> Result<Found, String> {
    let current = current()?;
    let builder = UpdaterBuilder::new(current, config_at(&manifest(version), false))
        // reqwest counts this against the whole download body: 215 MB needs ~30 min on a 1 Mbit/s link
        .timeout(Duration::from_secs(30 * 60))
        .version_comparator(|current, release| release.version != current);
    let update = builder.build().map_err(|e| e.to_string())?.check().map_err(|e| e.to_string())?;
    let update = update.ok_or_else(|| format!("Mimvox {version} is the version you have."))?;
    Ok(Found { version: update.version.clone(), notes: update.body.clone(), update: Some(update) })
}

/// Blocking. `Ok(None)`: this is the newest.
pub fn check() -> Result<Option<Found>, String> {
    let builder = UpdaterBuilder::new(current()?, config(false)).timeout(Duration::from_secs(30));
    let update = builder.build().map_err(|e| e.to_string())?.check().map_err(|e| e.to_string())?;
    Ok(update.map(|u| Found { version: u.version.clone(), notes: u.body.clone(), update: Some(u) }))
}

static READY: Mutex<Option<(Update, Vec<u8>)>> = Mutex::new(None);
static RELAUNCH: AtomicBool = AtomicBool::new(false);

/// Blocking. `progress` gets the bytes received so far and the size when known.
pub fn download(found: &Found, progress: impl Fn(u64, Option<u64>)) -> Result<(), String> {
    let update = found.update.as_ref().ok_or("This is a preview; there is nothing to download.")?;
    let got = std::sync::atomic::AtomicU64::new(0);
    let bytes = update
        .download_extended(|n, total| progress(got.fetch_add(n as u64, Ordering::Relaxed) + n as u64, total), || {})
        .map_err(|e| e.to_string())?;
    *READY.lock().unwrap_or_else(|e| e.into_inner()) = Some((update.clone(), bytes));
    Ok(())
}

pub fn discard_pending() {
    READY.lock().unwrap_or_else(|e| e.into_inner()).take();
}

pub fn relaunch_after_install() {
    RELAUNCH.store(true, Ordering::Relaxed);
}

/// On Windows the installer takes over and this process ends here.
pub fn install_pending() {
    let Some((mut update, bytes)) = READY.lock().unwrap_or_else(|e| e.into_inner()).take() else { return };
    let relaunch = RELAUNCH.load(Ordering::Relaxed);
    update.config = config(relaunch);
    let new_version = update.version.clone();
    if let Err(e) = update.install(bytes) {
        eprintln!("mimvox: the update could not be installed: {e}");
        return;
    }
    let appimage = std::env::var_os("APPIMAGE").map(PathBuf::from).map(|path| rename_appimage(path, &new_version));
    if relaunch {
        let started = if cfg!(target_os = "macos") {
            app_bundle().map(|app| std::process::Command::new("open").arg("-n").arg(app).spawn())
        } else {
            appimage.map(|image| std::process::Command::new(image).spawn())
        };
        if let Some(Err(e)) = started {
            eprintln!("mimvox: the updated app could not be started: {e}");
        }
    }
}

/// A name carrying the old version (as downloaded) changes to the new one; any other name stays.
fn rename_appimage(path: PathBuf, new_version: &str) -> PathBuf {
    let old = format!("_{}_", env!("CARGO_PKG_VERSION"));
    let Some(name) = path.file_name().and_then(|n| n.to_str()).filter(|n| n.contains(&old)) else { return path };
    let renamed = path.with_file_name(name.replacen(&old, &format!("_{new_version}_"), 1));
    if renamed.exists() {
        return path; // never overwrite another file
    }
    match std::fs::rename(&path, &renamed) {
        Ok(()) => renamed,
        Err(e) => {
            eprintln!("mimvox: the updated AppImage keeps its old name ({e})");
            path
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api(tag: &str, prerelease: bool) -> ApiRelease {
        ApiRelease {
            tag_name: tag.into(),
            body: Some(format!("notes {tag}")),
            published_at: Some("2026-11-14T16:45:45Z".into()),
            draft: false,
            prerelease,
        }
    }

    #[test]
    fn lists_releases_from_the_oldest_installable_one_newest_first() {
        let got = listed(vec![api("v0.2.0", false), api("v0.1.5", true), api("v0.1.4", false), api("v0.3.0", false), api("v0.4.0", true), api("v0.10.0", false)]);
        let versions: Vec<_> = got.iter().map(|r| r.version.as_str()).collect();
        assert_eq!(versions, ["0.10.0", "0.3.0", "0.2.0"]);
        assert_eq!(got[1].date, "2026-11-14");
        assert_eq!(got[1].notes, "notes v0.3.0");
    }
}
