//! Windows only: embeds the app icon and version info into the .exe as resources, so Explorer, the taskbar and the
//! installer show the icon. It needs the MSVC resource compiler (`rc.exe`, part of the Windows SDK in the Build
//! Tools). On Linux and macOS this does nothing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    // Resource id 1: the first icon, which Explorer shows.
    res.set_icon("assets/icon.ico");
    res.set("ProductName", "Mimvox");
    res.set("FileDescription", "Mimvox");
    res.set("CompanyName", "tarpetos");
    if let Err(e) = res.compile() {
        panic!("embedding the Windows icon failed (needs rc.exe from the Windows SDK / Build Tools): {e}");
    }
}
