fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../../tools/release/icons/api.ico");
    #[cfg(windows)]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../../tools/release/icons/api.ico");
        res.set("ProductName", "Mimvox");
        res.set("FileDescription", "Mimvox API server");
        res.set("CompanyName", "tarpetos");
        if let Err(e) = res.compile() {
            panic!("embedding the Windows icon failed (needs rc.exe from the Windows SDK / Build Tools): {e}");
        }
    }
}
