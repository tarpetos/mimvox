@echo off
rem Set Mimvox up on Windows with one command: setup.cmd [options]
rem It installs uv when missing (Astral's installer; it adds uv to new terminals), then runs tools\setup\setup.py with
rem uv's Python. That installs the build toolchains (Rust, MSVC Build Tools), the voice runtime (the engine with OmniVoice, Applio, their
rem models) and writes runtime.toml, and ends with the command that starts the app. Safe to run again: it resumes.
rem Without --yes or --data-dir it asks where to install (default: %LOCALAPPDATA%\mimvox); a custom place is
rem remembered in %LOCALAPPDATA%\mimvox\location.toml for the app.
rem Options: --data-dir DIR  --skip-toolchain  --skip-models  --cpu  --yes  --quick-check  (see tools\setup\setup.py)
setlocal
set UV_VERSION=0.12.18
cd /d "%~dp0"
where uv >nul 2>&1 && goto run
echo [setup] Installing uv %UV_VERSION%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/%UV_VERSION%/install.ps1 | iex" || exit /b 1
rem Where the installer put it: the first of its candidates, in its own order, that holds uv.exe.
set "UV_BIN="
if defined UV_INSTALL_DIR if exist "%UV_INSTALL_DIR%\uv.exe" set "UV_BIN=%UV_INSTALL_DIR%"
if not defined UV_BIN if defined XDG_BIN_HOME if exist "%XDG_BIN_HOME%\uv.exe" set "UV_BIN=%XDG_BIN_HOME%"
if not defined UV_BIN if defined XDG_DATA_HOME if exist "%XDG_DATA_HOME%\..\bin\uv.exe" set "UV_BIN=%XDG_DATA_HOME%\..\bin"
if not defined UV_BIN if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV_BIN=%USERPROFILE%\.local\bin"
if not defined UV_BIN echo [setup] uv was installed, but none of the folders its installer uses holds it; open a new window and run this again.& exit /b 1
set "PATH=%UV_BIN%;%PATH%"
set "MIMVOX_UV=%UV_BIN%\uv.exe"
:run
uv run --no-project --python 3.12 python tools\setup\setup.py %*
set CODE=%ERRORLEVEL%
rem This cmd session keeps the PATH that finds uv, so the command setup prints works here too.
endlocal & set "PATH=%PATH%" & exit /b %CODE%
