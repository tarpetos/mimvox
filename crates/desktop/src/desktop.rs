use std::path::{Path, PathBuf};

pub fn app_id() -> &'static str {
    if std::env::var_os("APPIMAGE").is_some() { APPIMAGE_ID } else { "mimvox" }
}

const APPIMAGE_ID: &str = "mimvox-appimage";

pub fn integrate() {
    let (Some(image), Some(data)) = (std::env::var_os("APPIMAGE").map(PathBuf::from), dirs::data_dir()) else {
        return;
    };
    if let Err(e) = add_entry(&image, &data) {
        eprintln!("mimvox: the AppImage was not added to the list of apps: {e}");
        return;
    }
    show_icon(&image, &data);
}

fn icon_path(data: &Path) -> PathBuf {
    data.join("icons/hicolor/512x512/apps").join(format!("{APPIMAGE_ID}.png"))
}

fn link_path(data: &Path) -> PathBuf {
    data.join(APPIMAGE_ID).join("mimvox")
}

fn add_entry(image: &Path, data: &Path) -> std::io::Result<()> {
    write_if_changed(&icon_path(data), crate::ICON_PNG)?;
    let link = link_path(data);
    if std::fs::read_link(&link).ok().as_deref() != Some(image) {
        if let Some(folder) = link.parent() {
            std::fs::create_dir_all(folder)?;
        }
        let fresh = link.with_extension("new");
        let _ = std::fs::remove_file(&fresh);
        std::os::unix::fs::symlink(image, &fresh)?;
        std::fs::rename(&fresh, &link)?;
    }
    let entry = data.join("applications").join(format!("{APPIMAGE_ID}.desktop"));
    write_if_changed(&entry, desktop_entry(&link).as_bytes())
}

fn show_icon(image: &Path, data: &Path) {
    let uri = format!("file://{}", icon_path(data).display());
    let shown = std::process::Command::new("gio").args(["info", "-a", "metadata::custom-icon"]).arg(image).output();
    if shown.is_ok_and(|shown| !String::from_utf8_lossy(&shown.stdout).contains(&uri)) {
        let _ = std::process::Command::new("gio")
            .args(["set", "-t", "string"])
            .arg(image)
            .args(["metadata::custom-icon", &uri])
            .status();
    }
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    if let Some(folder) = path.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let fresh = path.with_extension("new");
    std::fs::write(&fresh, bytes)?;
    std::fs::rename(&fresh, path)
}

fn desktop_entry(image: &Path) -> String {
    let mut quoted = String::new();
    for c in image.to_string_lossy().chars() {
        if matches!(c, '"' | '`' | '$' | '\\') {
            quoted.push('\\');
        }
        quoted.push(c);
    }
    format!(
        "[Desktop Entry]\nType=Application\nName=Mimvox\nComment=Clone your own voice and speak any text with it\n\
         Exec=\"{quoted}\"\nIcon={APPIMAGE_ID}\nTerminal=false\nCategories=Audio;AudioVideo;\n\
         StartupWMClass={APPIMAGE_ID}\n"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_appimage_adds_its_icon_and_its_entry() {
        let folder = std::env::temp_dir().join(format!("mimvox-desktop-{}", std::process::id()));
        let image = folder.join("my apps").join("Mimvox_0.6.0_x86_64.AppImage");
        std::fs::create_dir_all(image.parent().unwrap()).unwrap();
        std::fs::write(&image, b"").unwrap();
        let data = folder.join("share");
        add_entry(&image, &data).unwrap();
        let entry = std::fs::read_to_string(data.join("applications/mimvox-appimage.desktop")).unwrap();
        let link = data.join("mimvox-appimage/mimvox");
        assert!(entry.contains(&format!("Exec=\"{}\"\n", link.display())));
        assert!(entry.contains("StartupWMClass=mimvox-appimage"));
        assert_eq!(std::fs::read_link(&link).unwrap(), image);
        assert_eq!(
            std::fs::read(data.join("icons/hicolor/512x512/apps/mimvox-appimage.png")).unwrap(),
            crate::ICON_PNG
        );
        let moved = image.with_file_name("Mimvox_0.6.1_x86_64.AppImage");
        std::fs::rename(&image, &moved).unwrap();
        add_entry(&moved, &data).unwrap();
        assert_eq!(std::fs::read_link(&link).unwrap(), moved);
        assert_eq!(std::fs::read_to_string(data.join("applications/mimvox-appimage.desktop")).unwrap(), entry);
        std::fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn an_exec_path_is_quoted() {
        let entry = desktop_entry(Path::new("/home/a\"b/$x/Mimvox.AppImage"));
        assert!(entry.contains("Exec=\"/home/a\\\"b/\\$x/Mimvox.AppImage\"\n"));
    }
}
