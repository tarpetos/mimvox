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
    // The supported systems: without this Windows runs the app in its Vista compatibility mode, where a child window
    // cannot be layered (the seek strip, src/ui/seekstrip.rs).
    res.set_manifest(
        r#"<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security><requestedPrivileges><requestedExecutionLevel level="asInvoker" uiAccess="false"/></requestedPrivileges></security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
      <supportedOS Id="{1f676c76-80e1-4239-95bb-83d0f6d0da78}"/>
      <supportedOS Id="{4a2f28e3-53b9-4441-ba9c-d69d4a4a6e38}"/>
      <supportedOS Id="{35138b9a-5d96-4fbd-8e2d-a2440225f93a}"/>
    </application>
  </compatibility>
</assembly>"#,
    );
    if let Err(e) = res.compile() {
        panic!("embedding the Windows icon failed (needs rc.exe from the Windows SDK / Build Tools): {e}");
    }
}
