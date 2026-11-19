use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use subtle::ConstantTimeEq;

pub const TOKEN_ENV: &str = "MIMVOX_API_TOKEN";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenSource {
    Env,
    File(PathBuf),
}

pub fn token_path(data_dir: &Path) -> PathBuf {
    data_dir.join("api").join("token")
}

/// 32 random bytes, URL-safe base64 without padding.
pub fn generate() -> String {
    let mut bytes = [0u8; 32];
    getrandom::fill(&mut bytes).expect("the system's random source failed");
    URL_SAFE_NO_PAD.encode(bytes)
}

pub fn from_env() -> Option<String> {
    std::env::var(TOKEN_ENV).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

/// `MIMVOX_API_TOKEN`, else the token file; `None` when neither has one.
pub fn existing(data_dir: &Path) -> io::Result<Option<(String, TokenSource)>> {
    if let Some(token) = from_env() {
        return Ok(Some((token, TokenSource::Env)));
    }
    let path = token_path(data_dir);
    match fs::read_to_string(&path) {
        Ok(text) if !text.trim().is_empty() => Ok(Some((text.trim().to_string(), TokenSource::File(path)))),
        Ok(_) => Ok(None),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

pub fn load_or_create(data_dir: &Path) -> io::Result<(String, TokenSource)> {
    match existing(data_dir)? {
        Some(found) => Ok(found),
        None => Ok((rotate(data_dir)?, TokenSource::File(token_path(data_dir)))),
    }
}

/// Writes a new token file (readable by the user only) and returns the token.
pub fn rotate(data_dir: &Path) -> io::Result<String> {
    let token = generate();
    write_private(&token_path(data_dir), &token)?;
    Ok(token)
}

pub fn matches(given: &str, expected: &str) -> bool {
    given.as_bytes().ct_eq(expected.as_bytes()).into()
}

fn write_private(path: &Path, token: &str) -> io::Result<()> {
    let dir = path.parent().ok_or_else(|| io::Error::other("the token path has no folder"))?;
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(dir)?;

    let tmp = dir.join(".token.tmp");
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    let mut file = opts.open(&tmp)?;
    // The mode above applies only to a new file.
    #[cfg(unix)]
    file.set_permissions(std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
    file.write_all(token.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    drop(file);
    fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tokens_are_url_safe_and_distinct() {
        let (a, b) = (generate(), generate());
        assert_eq!(a.len(), 43);
        assert_ne!(a, b);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'));
    }

    #[test]
    fn compare() {
        assert!(matches("abc", "abc"));
        assert!(!matches("abd", "abc"));
        assert!(!matches("ab", "abc"));
        assert!(!matches("", "abc"));
    }

    #[test]
    fn rotate_writes_a_private_file() {
        let dir = std::env::temp_dir().join(format!("mimvox-token-{}", uuid::Uuid::new_v4()));
        let first = rotate(&dir).unwrap();
        assert_eq!(fs::read_to_string(token_path(&dir)).unwrap().trim(), first);
        let second = rotate(&dir).unwrap();
        assert_ne!(first, second);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(token_path(&dir)).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
