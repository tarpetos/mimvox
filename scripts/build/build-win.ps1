# Build the Mimvox desktop installer on Windows: a setup .exe (Inno Setup, default) or an .msi (WiX).
# Run it from anywhere:  .\scripts\build\build-win.ps1
#
#   -Format setup|msi|both  installer type (default: setup). setup = per-user setup .exe, no admin rights needed;
#                           msi = WiX package through cargo-packager.
#   -SkipCompile            reuse app\target\release (only re-stage resources and re-package)
#
# Output: dist\mimvox_<version>_<x64|arm64>-setup.exe and/or dist\*.msi. With $env:CARGO_PACKAGER_SIGN_PRIVATE_KEY
# (and _PASSWORD) set, each installer also gets the .sig the app's updater checks.
#
# If PowerShell 5.1 says "running scripts is disabled on this system", run this once in the same window first:
#   Set-ExecutionPolicy -Scope Process -Bypass
#
# Env: $env:CARGO_PACKAGER_VERSION, $env:CARGO_BUILD_JOBS (default 6), $env:CARGO_TARGET_DIR.
param(
    [ValidateSet("setup", "msi", "both")]
    [string]$Format = "setup",
    [switch]$SkipCompile
)
$ErrorActionPreference = "Stop"

$packagerVersion = if ($env:CARGO_PACKAGER_VERSION) { $env:CARGO_PACKAGER_VERSION } else { "0.11.8" }
$binName = "mimvox"
# a full Rust + wgpu build is memory hungry
$jobs = if ($env:CARGO_BUILD_JOBS) { $env:CARGO_BUILD_JOBS } else { "6" }
$env:CARGO_BUILD_JOBS = $jobs

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$appDir = Join-Path $repoRoot "app"
$distDir = Join-Path $repoRoot "dist"
$targetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $appDir "target" }
Set-Location -LiteralPath $appDir

function Require-Command($name, $hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "$name is required but is not installed. $hint"
        exit 1
    }
}
function Check-Exit($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$what failed (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
}

Require-Command cargo "Install Rust from https://rustup.rs (choose the default MSVC toolchain)."
Require-Command rustc "Install Rust from https://rustup.rs."
Require-Command uv 'Install it: powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"'

$hostLine = (rustc -vV | Select-String "^host:").Line
if ($hostLine -notmatch "msvc") {
    Write-Host "The active Rust toolchain is not an MSVC one ($hostLine)."
    Write-Host "Run: rustup default stable-x86_64-pc-windows-msvc"
    exit 1
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$hasMsvc = $false
if (Test-Path $vswhere) {
    $vcTools = if ($hostLine -match "aarch64") { "Microsoft.VisualStudio.Component.VC.Tools.ARM64" } else { "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" }
    $vs = & $vswhere -latest -products * -requires $vcTools -property installationPath
    $hasMsvc = [bool]$vs
}
if (-not $hasMsvc -and -not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
    Write-Host "Visual Studio Build Tools with the C++ workload are required."
    Write-Host "Install: winget install Microsoft.VisualStudio.2022.BuildTools --override `"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended`""
    exit 1
}

$cargoVersion = ((Get-Content Cargo.toml) | Select-String '^version\s*=\s*"([^"]*)"' | Select-Object -First 1).Matches[0].Groups[1].Value
$packagerCfgVersion = ((Get-Content Packager.toml) | Select-String '^version\s*=\s*"([^"]*)"' | Select-Object -First 1).Matches[0].Groups[1].Value
if ($cargoVersion -ne $packagerCfgVersion) {
    Write-Host "Version mismatch: app\Cargo.toml has $cargoVersion, app\Packager.toml has $packagerCfgVersion."
    exit 1
}

if (-not (Get-Command cargo-packager -ErrorAction SilentlyContinue)) {
    Write-Host "[build-win] Installing cargo-packager $packagerVersion..."
    cargo install cargo-packager --locked -j $jobs --version $packagerVersion
    Check-Exit "cargo install cargo-packager"
}

Write-Host "[build-win] Staging bundled resources..."
$uvTarget = if ($hostLine -match "aarch64") { "aarch64-pc-windows-msvc" } else { "x86_64-pc-windows-msvc" }
uv run --no-project python (Join-Path $repoRoot "scripts\build\stage_resources.py") --uv $uvTarget
Check-Exit "Staging resources"

if (-not $SkipCompile) {
    Write-Host "[build-win] Compiling the release binary..."
    cargo build --release --locked -j $jobs
    Check-Exit "cargo build"
}
$releaseDir = Join-Path $targetDir "release"
if (-not (Test-Path (Join-Path $releaseDir "$binName.exe"))) {
    Write-Host "Release binary not found: $releaseDir\$binName.exe"
    exit 1
}

# cargo-packager resolves the binary through Packager.toml's binaries-dir ("target/release", relative to app/) even when
# --binaries-dir points elsewhere, so a build into another CARGO_TARGET_DIR is copied there first.
$appRelease = Join-Path $appDir "target\release"
if ((Resolve-Path $releaseDir).Path -ne (Join-Path (Resolve-Path $appDir).Path "target\release")) {
    New-Item -ItemType Directory -Force -Path $appRelease | Out-Null
    Copy-Item -Force (Join-Path $releaseDir "$binName.exe") (Join-Path $appRelease "$binName.exe")
    $releaseDir = $appRelease
}

New-Item -ItemType Directory -Force -Path $distDir | Out-Null
$arch = if ($hostLine -match "aarch64") { "arm64" } else { "x64" }

if ($Format -ne "msi") {
    $stage = Join-Path $appDir "target\setup-stage"
    if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
    New-Item -ItemType Directory -Force -Path $stage | Out-Null
    Copy-Item (Join-Path $releaseDir "$binName.exe") $stage
    Copy-Item -Recurse -Force (Join-Path $appDir "target\bundle-resources\*") $stage
    $innoCache = if ($env:MIMVOX_INNO_CACHE) { $env:MIMVOX_INNO_CACHE } else { Join-Path $appDir "target\inno" }
    Write-Host "[build-win] Building the setup .exe (Inno Setup)..."
    uv run --no-project --with pillow python (Join-Path $repoRoot "scripts\build\inno.py") --version $cargoVersion --arch $arch `
        --stage $stage --out $distDir --cache $innoCache
    Check-Exit "Inno Setup"
    if ($env:CARGO_PACKAGER_SIGN_PRIVATE_KEY) {
        cargo packager signer sign (Join-Path $distDir "mimvox_${cargoVersion}_${arch}-setup.exe")
        Check-Exit "Signing the setup .exe"
    }
}
if ($Format -ne "setup") {
    Write-Host "[build-win] Packaging the .msi..."
    cargo packager --config Packager.toml --formats wix --binaries-dir $releaseDir --out-dir $distDir
    Check-Exit "cargo packager"
}

Write-Host "[build-win] Done. Installers in ${distDir}:"
Get-ChildItem $distDir -File | ForEach-Object { Write-Host ("  {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB)) }
