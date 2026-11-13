#!/usr/bin/env python3
"""Build the Windows installer (setup.exe) with Inno Setup, dark and in the app's colours.

Run by build-win.ps1 (``uv run --no-project --with pillow python scripts/build/inno.py ...``):

    --version 0.1.6 --arch x64|arm64 --stage <folder with mimvox.exe and the bundled resources> --out <dist>
    --cache <folder for the portable Inno Setup>

What it does: draws the wizard's pictures from the app icon and the app's dark palette (the Welcome/Finish side banner
and the top-right icon, at every display scale), writes ``mimvox.iss`` and compiles it with a pinned, portable Inno
Setup (downloaded once into ``--cache``, checked against its SHA-256). The wizard is forced dark
(``WizardStyle=modern dark``) in the app's palette (app/src/ui/theme.rs), and its title bar is painted the same colour.

The installer installs per user (no administrator prompt), into ``%LOCALAPPDATA%\\Programs\\Mimvox`` by default. An
update from inside the app runs it with ``/SILENT /SUPPRESSMSGBOXES /NORESTART`` (and ``/RELAUNCH=1`` for "Restart
now"; see app/src/update.rs): it closes the app if it still runs, replaces the app's folders and starts the new version
when asked. The uninstaller removes the app's folders whole; the user's data folder is elsewhere and stays.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ICON_PNG = REPO / "tools" / "release" / "icons" / "icon_512x512.png"
ICON_ICO = REPO / "tools" / "release" / "icons" / "icon.ico"
FONT_BOLD = REPO / "app" / "assets" / "fonts" / "NotoSans-Bold.ttf"
FONT_REGULAR = REPO / "app" / "assets" / "fonts" / "NotoSans-Regular.ttf"

APP = "Mimvox"
BINARY = "mimvox.exe"
PUBLISHER = "tarpetos"
URL = "https://github.com/tarpetos/mimvox"
TAGLINE = "Your voice, any text"
# The app's top-level folders (scripts/build/stage_resources.py): replaced whole on install, removed on uninstall.
APP_DIRS = ("engine", "presets", "tools", "contract", "bin")

# The app's dark palette (app/src/ui/theme.rs, Palette::new): navy background, the four "aurora" light fields drifting
# behind its glass (ice, teal, steel, deep blue), the ice accent and its gradient, the text colours.
BG_TOP = "#0B131F"
BG_BOTTOM = "#070C13"
GLASS = "#101A28"   # smoked glass over the background: the header strip
AURORA_ICE = "#9CD2FF"
AURORA_TEAL = "#3FC0CF"
AURORA_STEEL = "#6283B8"
AURORA_BLUE = "#2F66C2"
ACCENT = "#9CD2FF"
ACCENT_TOP = "#AEDAFF"
ACCENT_BOTTOM = "#5FAAF0"
TEXT = "#F2F6FB"
TEXT_DIM = "#AFBBCB"
WELL = "#0E182A"    # the app's recessed wells: the progress bar's track
BAD = "#FF7A85"

# Pinned, because the download is executed: the digest GitHub publishes for the release asset.
INNO_VERSION = "6.7.3"
INNO_URL = f"https://github.com/jrsoftware/issrc/releases/download/is-{INNO_VERSION.replace('.', '_')}/innosetup-{INNO_VERSION}.exe"
INNO_SHA256 = "9c73c3bae7ed48d44112a0f48e66742c00090bdb5bef71d9d3c056c66e97b732"

# The image areas at 100-250 % display scaling (Inno Setup 6.7 help, WizardImageFile / WizardSmallImageFile); Setup
# picks the closest, so every scale gets a sharp picture.
BANNER_SIZES = ((202, 386), (269, 515), (336, 643), (403, 772), (430, 824), (498, 953), (534, 1022))
SMALL_SIZES = (58, 77, 97, 116, 124, 143, 159)

# DwmSetWindowAttribute's DWMWA_CAPTION_COLOR (Windows 11; older Windows refuses it and keeps its own title bar).
DWMWA_CAPTION_COLOR = 35


def rgb(colour: str) -> tuple[int, int, int]:
    value = colour.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def colorref(colour: str) -> str:
    """``#RRGGBB`` as a Pascal COLORREF literal (blue, green, red)."""
    r, g, b = rgb(colour)
    return f"${b:02X}{g:02X}{r:02X}"


# ------------------------------------------------------------------------------------------------- the pictures

def draw_banner(width: int, height: int, out: Path) -> None:
    """The Welcome/Finish side banner, drawn like the app's window: the navy gradient with its aurora light fields
    (ice, teal, steel, deep blue), the icon on an accent glow, the name, an accent-gradient rule and the tagline."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: PLC0415 - only the Windows build draws

    top, bottom = rgb(BG_TOP), rgb(BG_BOTTOM)
    base = Image.new("RGB", (width, height))
    pixels = base.load()
    for y in range(height):
        t = y / max(height - 1, 1)
        row = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
        for x in range(width):
            pixels[x, y] = row
    image = base.convert("RGBA")

    def field(colour: str, cx: float, cy: float, rx: float, ry: float, alpha: int) -> None:
        """One soft light field, as the app's background draws them (a blurred ellipse)."""
        nonlocal image
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        box = (round((cx - rx) * width), round((cy - ry) * height), round((cx + rx) * width), round((cy + ry) * height))
        ImageDraw.Draw(layer).ellipse(box, fill=(*rgb(colour), alpha))
        image = Image.alpha_composite(image, layer.filter(ImageFilter.GaussianBlur(width * 0.16)))

    field(AURORA_ICE, 0.0, 0.04, 0.7, 0.2, 92)
    field(AURORA_TEAL, 1.0, 0.3, 0.5, 0.16, 84)
    field(AURORA_STEEL, 0.05, 0.9, 0.6, 0.18, 74)
    field(AURORA_BLUE, 1.0, 0.75, 0.55, 0.18, 96)
    field(AURORA_ICE, 0.5, 0.36, 0.4, 0.12, 58)   # the light behind the icon

    icon = Image.open(ICON_PNG).convert("RGBA")
    side = round(width * 0.46)
    icon = icon.resize((side, side), Image.LANCZOS)
    icon_top = round(height * 0.24)
    image.alpha_composite(icon, ((width - side) // 2, icon_top))

    draw = ImageDraw.Draw(image)
    name_font = ImageFont.truetype(str(FONT_BOLD), round(width * 0.12))
    tag_font = ImageFont.truetype(str(FONT_REGULAR), round(width * 0.062))
    name_y = icon_top + side + round(height * 0.045)
    draw.text((width / 2, name_y), APP, font=name_font, fill=rgb(TEXT), anchor="mt")
    # The rule: the accent's own gradient, light to deep, like the app's primary buttons.
    rule_y = name_y + round(width * 0.17)
    x0, x1 = round(width * 0.38), round(width * 0.62)
    thick = max(2, width // 90)
    a, b = rgb(ACCENT_TOP), rgb(ACCENT_BOTTOM)
    for x in range(x0, x1 + 1):
        t = (x - x0) / max(x1 - x0, 1)
        draw.line((x, rule_y, x, rule_y + thick - 1), fill=tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))
    draw.text((width / 2, rule_y + round(width * 0.05)), TAGLINE, font=tag_font, fill=rgb(TEXT_DIM), anchor="mt")
    image.convert("RGB").save(out)


def draw_small(side: int, out: Path) -> None:
    """The top-right icon on the inner pages: the app icon on a transparent margin."""
    from PIL import Image  # noqa: PLC0415

    small = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    inner = round(side * 0.78)
    icon = Image.open(ICON_PNG).convert("RGBA").resize((inner, inner), Image.LANCZOS)
    small.alpha_composite(icon, ((side - inner) // 2, (side - inner) // 2))
    small.save(out)


def draw_images(folder: Path) -> tuple[list[Path], list[Path]]:
    folder.mkdir(parents=True, exist_ok=True)
    banners, smalls = [], []
    for width, height in BANNER_SIZES:
        banners.append(folder / f"banner-{width}x{height}.png")
        draw_banner(width, height, banners[-1])
    for side in SMALL_SIZES:
        smalls.append(folder / f"small-{side}.png")
        draw_small(side, smalls[-1])
    return banners, smalls


# ------------------------------------------------------------------------------------------------- uninstalling

# The mutex the app holds while it runs (app/src/main.rs, APP_MUTEX).
APP_MUTEX = "Local\\Mimvox-running"
# Inno's uninstall entry: <AppId>_is1 under the install mode's root (HKA: per user here).
UNINSTALL_KEY = f"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{APP}_is1"
# What the data folder holds that Mimvox made (scripts never delete the folder blindly: a folder the user chose may
# hold other files). ENGINE: the speech engine, models, caches; YOURS: what the user made, and the settings.
ENGINE_ENTRIES = ("runtime", "cache", "tmp", "runtime.toml")
USER_ENTRIES = ("voices", "outputs", "books", "logs", "settings.json")

# The uninstaller's one window (Pascal). "Apps & features" runs it with /SILENT (RegisterOneWindowUninstall), so Inno's
# own "Remove ...?" box never shows; /VERYSILENT (a tool removing apps unattended) keeps every bit of data. The choice:
# 0 the app only, 1 also the speech engine and models, 2 everything; /REMOVE=<n> answers it without the form. A run
# started by hand (unins000.exe without /SILENT) shows the form too, then Inno's box.
UNINSTALL_CODE = r"""
var
  RemoveChoice: Integer;
  UserStarted: Boolean;
  ChoiceForm: TSetupForm;
  ChoiceAccepted: Boolean;
  OptionApp, OptionEngine, OptionAll: TNewRadioButton;
  AllWarning, RunningNotice: TNewStaticText;
  FormUninstallButton, FormCancelButton: TNewButton;
  FormButtonsTop: Integer;
  RunningTimer: Longword;
  PointerUsed: Boolean;

function SetTimer(hWnd, nIDEvent, uElapse, lpTimerFunc: Longword): Longword;
  external 'SetTimer@user32.dll stdcall';
function KillTimer(hWnd, nIDEvent: Longword): Bool;
  external 'KillTimer@user32.dll stdcall';

function AppRunning(): Boolean;
begin
  Result := CheckForMutexes('@MUTEX@');
end;

{ The app's files are held while it runs: nothing is removed until it is closed. The form watches it live. }
procedure ShowRunningState();
var
  Running: Boolean;
  Extra: Integer;
begin
  Running := AppRunning();
  RunningNotice.Visible := Running;
  FormUninstallButton.Enabled := not Running;
  Extra := 0;
  if Running then Extra := RunningNotice.Height + ScaleY(12);
  FormUninstallButton.Top := FormButtonsTop + Extra;
  FormCancelButton.Top := FormButtonsTop + Extra;
  ChoiceForm.ClientHeight := FormButtonsTop + Extra + FormUninstallButton.Height + ScaleY(20);
end;

procedure RunningTimerProc(Arg1, Arg2, Arg3, Arg4: Longword);
begin
  if ChoiceForm <> nil then ShowRunningState();
end;

{ A silent run (an update's, or a script's) waits up to 30 s for the app to close. }
function WaitForAppToClose(): Boolean;
var
  I: Integer;
begin
  for I := 1 to 60 do
  begin
    if not AppRunning() then
    begin
      Result := True;
      exit;
    end;
    Sleep(500);
  end;
  Result := not AppRunning();
end;

procedure RegisterOneWindowUninstall();
var
  Command: String;
begin
  if RegQueryStringValue(HKA, '@KEY@', 'UninstallString', Command) then
  begin
    RegWriteStringValue(HKA, '@KEY@', 'UninstallString', Command + ' /SILENT');
    RegWriteStringValue(HKA, '@KEY@', 'QuietUninstallString', Command + ' /VERYSILENT');
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then RegisterOneWindowUninstall();
end;

function VerySilentRun(): Boolean;
var
  I: Integer;
begin
  Result := False;
  for I := 1 to ParamCount do
    if CompareText(ParamStr(I), '/VERYSILENT') = 0 then Result := True;
end;

{ The data folder, as the app finds it: MIMVOX_DATA, else location.toml's data_dir, else the default. }
function DataDir(): String;
var
  Pointer: String;
  Text: AnsiString;
  Value: String;
  First, Last: Integer;
begin
  PointerUsed := False;
  Result := GetEnv('MIMVOX_DATA');
  if Result <> '' then exit;
  Result := ExpandConstant('{localappdata}\mimvox');
  Pointer := ExpandConstant('{localappdata}\mimvox\location.toml');
  if not LoadStringFromFile(Pointer, Text) then exit;
  First := Pos('"', Text);
  if First = 0 then exit;
  Text := Copy(Text, First + 1, Length(Text));
  Last := Pos('"', Text);
  if Last <= 1 then exit;
  Value := String(Copy(Text, 1, Last - 1));
  StringChangeEx(Value, '\\', '\', True);  { TOML escapes each backslash }
  Result := Value;
  PointerUsed := True;
end;

procedure ChoiceChanged(Sender: TObject);
begin
  AllWarning.Visible := OptionAll.Checked;
end;

procedure UninstallClick(Sender: TObject);
begin
  if AppRunning() then
  begin
    ShowRunningState();
    exit;
  end;
  ChoiceAccepted := True;
  ChoiceForm.Close();
end;

function NewOption(Top: Integer; Caption, Detail: String; var DetailText: TNewStaticText): TNewRadioButton;
begin
  Result := TNewRadioButton.Create(ChoiceForm);
  Result.Parent := ChoiceForm;
  Result.Left := ScaleX(24);
  Result.Top := Top;
  Result.Width := ChoiceForm.ClientWidth - ScaleX(48);
  Result.Height := ScaleY(22);
  Result.Caption := Caption;
  Result.OnClick := @ChoiceChanged;
  DetailText := TNewStaticText.Create(ChoiceForm);
  DetailText.Parent := ChoiceForm;
  DetailText.WordWrap := True;
  DetailText.Left := ScaleX(46);
  DetailText.Top := Result.Top + Result.Height;
  DetailText.Width := ChoiceForm.ClientWidth - ScaleX(70);
  DetailText.Caption := Detail;
  DetailText.AdjustHeight();
  DetailText.StyleElements := DetailText.StyleElements - [seFont];
  DetailText.Font.Color := @DIM@;
end;

function AskWhatToRemove(): Boolean;
var
  Title: TNewStaticText;
  D1, D2, D3: TNewStaticText;
  UninstallButton, CancelButton: TNewButton;
  ButtonsTop, ButtonWidth: Integer;
begin
  Result := False;
  ChoiceForm := CreateCustomForm(ScaleX(520), ScaleY(300), True, True);
  try
    ChoiceForm.ClientWidth := ScaleX(520);
    ChoiceForm.Caption := 'Uninstall Mimvox';
    Title := TNewStaticText.Create(ChoiceForm);
    Title.Parent := ChoiceForm;
    Title.Left := ScaleX(24);
    Title.Top := ScaleY(20);
    Title.Width := ChoiceForm.ClientWidth - ScaleX(48);
    Title.WordWrap := True;
    Title.Caption := 'What should be removed?';
    Title.Font.Style := [fsBold];
    Title.AdjustHeight();
    OptionApp := NewOption(Title.Top + Title.Height + ScaleY(14), 'Only the app',
      'Your voices, results, settings and the speech engine stay. Installing Mimvox again continues where you left off.', D1);
    OptionEngine := NewOption(D1.Top + D1.Height + ScaleY(12), 'The app, the speech engine and its models',
      'Frees almost all the space Mimvox uses. Your voices, recordings, results and settings stay.', D2);
    OptionAll := NewOption(D2.Top + D2.Height + ScaleY(12), 'Everything',
      'Also your voices, recordings, trained voice models, results, audiobooks and settings.', D3);
    AllWarning := TNewStaticText.Create(ChoiceForm);
    AllWarning.Parent := ChoiceForm;
    AllWarning.WordWrap := True;
    AllWarning.Left := ScaleX(46);
    AllWarning.Top := D3.Top + D3.Height + ScaleY(6);
    AllWarning.Width := ChoiceForm.ClientWidth - ScaleX(70);
    AllWarning.Caption := 'This cannot be undone: copy any recordings and results you want to keep first.';
    AllWarning.AdjustHeight();
    AllWarning.StyleElements := AllWarning.StyleElements - [seFont];
    AllWarning.Font.Color := @BAD@;
    AllWarning.Visible := False;
    OptionApp.Checked := True;
    RunningNotice := TNewStaticText.Create(ChoiceForm);
    RunningNotice.Parent := ChoiceForm;
    RunningNotice.WordWrap := True;
    RunningNotice.Left := ScaleX(24);
    RunningNotice.Top := AllWarning.Top + AllWarning.Height + ScaleY(14);
    RunningNotice.Width := ChoiceForm.ClientWidth - ScaleX(48);
    RunningNotice.Caption := 'Mimvox is running. Close it to uninstall: this window notices by itself.';
    RunningNotice.AdjustHeight();
    RunningNotice.StyleElements := RunningNotice.StyleElements - [seFont];
    RunningNotice.Font.Color := @BAD@;
    ButtonsTop := AllWarning.Top + AllWarning.Height + ScaleY(20);
    ButtonWidth := ChoiceForm.CalculateButtonWidth(['Uninstall', 'Cancel']);
    CancelButton := TNewButton.Create(ChoiceForm);
    CancelButton.Parent := ChoiceForm;
    CancelButton.Caption := 'Cancel';
    CancelButton.Width := ButtonWidth;
    CancelButton.Height := ScaleY(25);
    CancelButton.Top := ButtonsTop;
    CancelButton.Left := ChoiceForm.ClientWidth - ScaleX(24) - ButtonWidth;
    CancelButton.ModalResult := mrCancel;
    CancelButton.Cancel := True;
    UninstallButton := TNewButton.Create(ChoiceForm);
    UninstallButton.Parent := ChoiceForm;
    UninstallButton.Caption := 'Uninstall';
    UninstallButton.Width := ButtonWidth;
    UninstallButton.Height := ScaleY(25);
    UninstallButton.Top := ButtonsTop;
    UninstallButton.Left := CancelButton.Left - ScaleX(10) - ButtonWidth;
    UninstallButton.OnClick := @UninstallClick;
    UninstallButton.Default := True;
    FormUninstallButton := UninstallButton;
    FormCancelButton := CancelButton;
    FormButtonsTop := ButtonsTop;
    ShowRunningState();
    RunningTimer := SetTimer(0, 0, 700, CreateCallback(@RunningTimerProc));
    PaintTitleBar(ChoiceForm.Handle);
    ChoiceAccepted := False;
    ChoiceForm.ShowModal();
    if ChoiceAccepted then
    begin
      if OptionAll.Checked then RemoveChoice := 2
      else if OptionEngine.Checked then RemoveChoice := 1
      else RemoveChoice := 0;
      Result := True;
    end;
  finally
    if RunningTimer <> 0 then KillTimer(0, RunningTimer);
    RunningTimer := 0;
    ChoiceForm.Free();
    ChoiceForm := nil;
  end;
end;

function InitializeUninstall(): Boolean;
var
  ResultCode: Integer;
begin
  Result := True;
  RemoveChoice := 0;
  UserStarted := False;
  { /REMOVE=0|1|2 answers the question up front (scripts, tests). }
  { A script's run (/VERYSILENT) or a handed-over one (/REMOVE) waits for the app; the person's run shows the form,
    which says the app runs and waits for it live. }
  if (VerySilentRun() or (ExpandConstant('{param:REMOVE|}') <> '')) and AppRunning() and not WaitForAppToClose() then
  begin
    Log('Mimvox is still running: nothing is removed');
    Result := False;
    exit;
  end;
  if ExpandConstant('{param:REMOVE|}') <> '' then
  begin
    RemoveChoice := StrToIntDef(ExpandConstant('{param:REMOVE|0}'), 0);
    { /SHOWDONE=1: handed over from a run the user started, which showed the form. }
    UserStarted := ExpandConstant('{param:SHOWDONE|0}') = '1';
    Log('Chosen on the command line: ' + IntToStr(RemoveChoice));
    exit;
  end;
  if VerySilentRun() then
  begin
    Log('A very silent uninstall: the data folder is kept');
    exit;
  end;
  if not AskWhatToRemove() then
  begin
    Log('Cancelled: nothing is removed');
    Result := False;
    exit;
  end;
  UserStarted := True;
  Log('Chosen: ' + IntToStr(RemoveChoice) + ' (0 the app, 1 also the engine and models, 2 everything)');
  if UninstallSilent() then exit;
  { Started without /SILENT (unins000.exe by hand): Inno would ask its own Yes/No and show its own "removed" box.
    A silent run of the same uninstaller takes the choice over; this one ends without removing anything. }
  Result := False;
  if not Exec(ExpandConstant('{uninstallexe}'), '/SILENT /REMOVE=' + IntToStr(RemoveChoice) + ' /SHOWDONE=1', '',
              SW_SHOW, ewNoWait, ResultCode) then
    MsgBox('The uninstaller could not continue. Run it again from Settings > Apps.', mbError, MB_OK);
end;

procedure RemoveEntry(Dir, Name: String);
var
  Path: String;
begin
  Path := AddBackslash(Dir) + Name;
  if DirExists(Path) then
  begin
    if DelTree(Path, True, True, True) then Log('Removed ' + Path) else Log('Could not remove ' + Path);
  end
  else if FileExists(Path) then
  begin
    if DeleteFile(Path) then Log('Removed ' + Path) else Log('Could not remove ' + Path);
  end;
end;

procedure RemoveChosenData();
var
  Data: String;
begin
  if RemoveChoice = 0 then exit;
  Data := DataDir();
  { Only a folder Mimvox set up (it holds runtime.toml, a runtime or voices), and only what Mimvox made in it. }
  if not (FileExists(AddBackslash(Data) + 'runtime.toml') or DirExists(AddBackslash(Data) + 'runtime')
          or DirExists(AddBackslash(Data) + 'voices')) then
  begin
    Log('No Mimvox data folder at ' + Data + ': nothing else is removed');
    exit;
  end;
  UninstallProgressForm.StatusLabel.Caption := 'Removing the speech engine and its models...';
@ENGINE@
  if RemoveChoice = 2 then
  begin
    UninstallProgressForm.StatusLabel.Caption := 'Removing your voices, results and settings...';
@USER@
    { The pointer goes only with the folder it names (MIMVOX_DATA may point elsewhere). }
    if PointerUsed then DeleteFile(ExpandConstant('{localappdata}\mimvox\location.toml'));
  end;
  RemoveDir(Data);
  RemoveDir(ExpandConstant('{localappdata}\mimvox'));
end;

procedure InitializeUninstallProgressForm();
begin
  PaintTitleBar(UninstallProgressForm.Handle);
  { The style's progress bar is green; the steps are short, a status line says what happens. }
  UninstallProgressForm.ProgressBar.Visible := False;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usUninstall then RemoveChosenData();
  { Ours is the one "removed" box: the uninstall always runs silent, and Inno shows its own only when it is not. }
  if (CurUninstallStep = usDone) and UserStarted and UninstallSilent() then
    MsgBox('Mimvox was removed.', mbInformation, MB_OK);
end;
"""


def uninstall_code() -> str:
    def lines(entries: tuple[str, ...]) -> str:
        return "\n".join(f"  RemoveEntry(Data, '{name}');" for name in entries)
    return (UNINSTALL_CODE.replace("@KEY@", UNINSTALL_KEY).replace("@MUTEX@", APP_MUTEX).replace("@DIM@", colorref(TEXT_DIM))
            .replace("@BAD@", colorref(BAD)).replace("@ENGINE@", lines(ENGINE_ENTRIES))
            .replace("@USER@", lines(USER_ENTRIES)))


# ------------------------------------------------------------------------------------------------- the script

def iss(version: str, arch: str, stage: Path, out: Path, banners: list[Path], smalls: list[Path]) -> str:
    arch_allowed = "arm64" if arch == "arm64" else "x64compatible"
    base = f"mimvox_{version}_{arch}-setup"
    lines = [
        "; Generated by scripts/build/inno.py - do not edit by hand.",
        "[Setup]",
        f"AppId={APP}",
        f"AppName={APP}",
        f"AppVersion={version}",
        f"AppPublisher={PUBLISHER}",
        f"AppPublisherURL={URL}",
        f"AppSupportURL={URL}/issues",
        f"UninstallDisplayName={APP}",
        f"UninstallDisplayIcon={{app}}\\{BINARY}",
        f"SetupIconFile={ICON_ICO}",
        f"DefaultDirName={{autopf}}\\{APP}",
        f"DefaultGroupName={APP}",
        "DisableProgramGroupPage=yes",
        # Lets /NOICONS skip the Start menu entry (a test install, a scripted one).
        "AllowNoIcons=yes",
        "AllowRootDirectory=no",
        "PrivilegesRequired=lowest",
        # Off: it is inherited by what Setup starts (the app after installing, and after an update), and it blocks the
        # links uv's Python installs are made of (cpython-3.11-... -> cpython-3.11.13-...): the engine's python.exe then
        # fails with "uv trampoline failed to spawn Python child process: entity not found". It protects elevated
        # installers writing to shared folders; this one installs per user into the user's own folders.
        "RedirectionGuard=no",
        f"ArchitecturesAllowed={arch_allowed}",
        f"ArchitecturesInstallIn64BitMode={arch_allowed}",
        # Closes a running Mimvox (Restart Manager) before its files are replaced.
        "CloseApplications=yes",
        "RestartApplications=no",
        "SetupLogging=yes",
        "Compression=lzma2/max",
        "SolidCompression=yes",
        "LZMANumBlockThreads=4",
        # The look: forced dark, the app's background everywhere, our pictures.
        # Inno's own dark style (neutral, its default button only outlined in light blue, like the app's glass
        # buttons), coloured with the app's palette below; no bevel lines (the app has none).
        "WizardStyle=modern dark hidebevels",
        f"WizardBackColor={BG_TOP}",
        f"WizardImageFile={','.join(str(p) for p in banners)}",
        f"WizardImageBackColor={BG_TOP}",
        f"WizardSmallImageFile={','.join(str(p) for p in smalls)}",
        "WizardSmallImageBackColor=none",
        "DisableWelcomePage=no",
        # One settings page: the folder and the desktop shortcut together (see [Code]); Install is pressed there.
        "DisableReadyPage=yes",
        f"OutputDir={out}",
        f"OutputBaseFilename={base}",
        "",
        "[Messages]",
        "WelcomeLabel2=This installs [name/ver] on your computer.%n%nThe speech engine and its models are not in "
        "this installer: the app downloads them on its first start, and you choose which parts you want.",
        f"FinishedLabel={APP} is installed. On its first start it sets up the speech engine for you.",
        # The folder page is the last before installing (no Ready page): its button installs.
        "SelectDirBrowseLabel=To install, click Install. To choose a different folder, click Browse.",
        "",
        "[InstallDelete]",
        # An update replaces the app's folders whole, so nothing an older version shipped stays behind.
        *(f'Type: filesandordirs; Name: "{{app}}\\{name}"' for name in APP_DIRS),
        "",
        "[Files]",
        f'Source: "{stage}\\*"; DestDir: "{{app}}"; Flags: recursesubdirs createallsubdirs ignoreversion',
        "",
        "[Icons]",
        f'Name: "{{group}}\\{APP}"; Filename: "{{app}}\\{BINARY}"; Check: StartMenuWanted',
        f'Name: "{{autodesktop}}\\{APP}"; Filename: "{{app}}\\{BINARY}"; Check: DesktopWanted',
        "",
        "[UninstallDelete]",
        # Whatever the app wrote inside its own folders goes with them; the data folder is elsewhere.
        *(f'Type: filesandordirs; Name: "{{app}}\\{name}"' for name in APP_DIRS),
        'Type: dirifempty; Name: "{app}"',
        "",
        "[Run]",
        f'Filename: "{{app}}\\{BINARY}"; Description: "Start {APP}"; Flags: nowait postinstall skipifsilent',
        # An update from the app with "Restart now" passes /RELAUNCH=1: the silent install starts the new version.
        f'Filename: "{{app}}\\{BINARY}"; Flags: nowait; Check: WizardSilent and Relaunch',
        "",
        "[Code]",
        "function DwmSetWindowAttribute(Wnd: HWND; Attribute: Cardinal; var Value: Cardinal; Size: Cardinal): Integer;",
        "  external 'DwmSetWindowAttribute@dwmapi.dll stdcall delayload';",
        "",
        "var",
        "  DesktopCheck, StartMenuCheck: TNewCheckBox;",
        "  WizardSilentWaited: Integer;",
        "  BarTrack, BarFill: TBitmapImage;",
        "",
        "function InitializeSetup(): Boolean;",
        "begin",
        "  Result := True;",
        "  { Nothing is installed over a running Mimvox: an update waits for it to close, a person is asked. }",
        f"  while CheckForMutexes('{APP_MUTEX}') do",
        "  begin",
        "    if WizardSilent then",
        "    begin",
        "      Sleep(500);",
        "      if WizardSilentWaited >= 60 then",
        "      begin",
        "        Log('Mimvox is still running: Setup stops');",
        "        Result := False;",
        "        exit;",
        "      end;",
        "      WizardSilentWaited := WizardSilentWaited + 1;",
        "    end",
        f"    else if MsgBox('{APP} is running. Close it, then click OK to continue.', mbError, MB_OKCANCEL) = IDCANCEL then",
        "    begin",
        "      Result := False;",
        "      exit;",
        "    end;",
        "  end;",
        "end;",
        "",
        "function DesktopWanted: Boolean;",
        "begin",
        "  { A silent update keeps what the user has: a shortcut only when there already is one. }",
        "  if WizardSilent then",
        f"    Result := FileExists(ExpandConstant('{{autodesktop}}\\{APP}.lnk'))",
        "  else",
        "    Result := DesktopCheck.Checked;",
        "end;",
        "",
        "function StartMenuWanted: Boolean;",
        "begin",
        "  if WizardSilent then",
        f"    Result := FileExists(ExpandConstant('{{group}}\\{APP}.lnk'))",
        "  else",
        "    Result := StartMenuCheck.Checked;",
        "end;",
        "",
        "function Relaunch: Boolean;",
        "begin",
        "  Result := ExpandConstant('{param:RELAUNCH|0}') = '1';",
        "end;",
        "",
        "procedure PaintTitleBar(Handle: HWND);",
        "var",
        "  Colour: Cardinal;",
        "begin",
        f"  Colour := {colorref(BG_TOP)};",
        "  try",
        f"    DwmSetWindowAttribute(Handle, {DWMWA_CAPTION_COLOR}, Colour, 4);",
        "  except",
        "  end;",
        "end;",
        "",
        "procedure InitializeWizard;",
        "begin",
        "  { The app's palette: the header strip is smoked glass, titles in the text colour, the rest dimmer. }",
        f"  WizardForm.MainPanel.Color := {colorref(GLASS)};",
        f"  WizardForm.PageNameLabel.Font.Color := {colorref(TEXT)};",
        f"  WizardForm.PageDescriptionLabel.Font.Color := {colorref(TEXT_DIM)};",
        f"  WizardForm.WelcomeLabel1.Font.Color := {colorref(TEXT)};",
        f"  WizardForm.WelcomeLabel2.Font.Color := {colorref(TEXT_DIM)};",
        f"  WizardForm.FinishedHeadingLabel.Font.Color := {colorref(TEXT)};",
        f"  WizardForm.FinishedLabel.Font.Color := {colorref(TEXT_DIM)};",
        "",
        "  { The shortcuts on the folder page: one settings page instead of two. }",
        "  DesktopCheck := TNewCheckBox.Create(WizardForm);",
        "  DesktopCheck.Parent := WizardForm.SelectDirPage;",
        "  DesktopCheck.Caption := 'Create a desktop shortcut';",
        "  DesktopCheck.Left := WizardForm.DirEdit.Left;",
        "  DesktopCheck.Top := WizardForm.DirEdit.Top + WizardForm.DirEdit.Height + ScaleY(24);",
        "  DesktopCheck.Width := WizardForm.DirEdit.Width;",
        "  DesktopCheck.Height := ScaleY(22);",
        "  DesktopCheck.Checked := True;",
        "  StartMenuCheck := TNewCheckBox.Create(WizardForm);",
        "  StartMenuCheck.Parent := WizardForm.SelectDirPage;",
        "  StartMenuCheck.Caption := 'Add to the Start menu';",
        "  StartMenuCheck.Left := DesktopCheck.Left;",
        "  StartMenuCheck.Top := DesktopCheck.Top + DesktopCheck.Height + ScaleY(8);",
        "  StartMenuCheck.Width := DesktopCheck.Width;",
        "  StartMenuCheck.Height := DesktopCheck.Height;",
        "  StartMenuCheck.Checked := True;",
        "  { An update keeps the shortcuts the user has (or does not have). }",
        f"  if RegKeyExists(HKA, '{UNINSTALL_KEY}') then",
        "  begin",
        f"    DesktopCheck.Checked := FileExists(ExpandConstant('{{autodesktop}}\\{APP}.lnk'));",
        f"    StartMenuCheck.Checked := FileExists(ExpandConstant('{{group}}\\{APP}.lnk'));",
        "  end;",
        "",
        "  { The progress bar in the app's colours (the style's own is green): a well and an accent fill. }",
        "  BarTrack := TBitmapImage.Create(WizardForm);",
        "  BarTrack.Parent := WizardForm.InstallingPage;",
        "  BarTrack.SetBounds(WizardForm.ProgressGauge.Left, WizardForm.ProgressGauge.Top,",
        "    WizardForm.ProgressGauge.Width, ScaleY(8));",
        f"  BarTrack.BackColor := {colorref(WELL)};",
        "  BarFill := TBitmapImage.Create(WizardForm);",
        "  BarFill.Parent := WizardForm.InstallingPage;",
        "  BarFill.SetBounds(BarTrack.Left, BarTrack.Top, 0, BarTrack.Height);",
        f"  BarFill.BackColor := {colorref(ACCENT)};",
        "  WizardForm.ProgressGauge.Visible := False;",
        "end;",
        "",
        "procedure CurInstallProgressChanged(CurProgress, MaxProgress: Integer);",
        "begin",
        "  if MaxProgress > 0 then",
        "    BarFill.Width := MulDiv(BarTrack.Width, CurProgress, MaxProgress);",
        "end;",
        "",
        "procedure CurPageChanged(CurPageID: Integer);",
        "begin",
        "  PaintTitleBar(WizardForm.Handle);",
        "  if CurPageID = wpSelectDir then",
        "    WizardForm.NextButton.Caption := SetupMessage(msgButtonInstall);",
        "  { Cancel is hidden on the last page: Finish takes its place at the right edge. }",
        "  if CurPageID = wpFinished then",
        "    WizardForm.NextButton.Left := WizardForm.CancelButton.Left;",
        "end;",
        "",
    ]
    return "\n".join(lines) + "\n" + uninstall_code()


# ------------------------------------------------------------------------------------------------- the compiler

def ensure_iscc(cache: Path) -> Path:
    """The pinned Inno Setup's compiler, unpacked once (portable, per user, no administrator prompt) into ``cache``."""
    iscc = cache / f"inno-{INNO_VERSION}" / "ISCC.exe"
    if iscc.is_file():
        return iscc
    cache.mkdir(parents=True, exist_ok=True)
    installer = cache / f"innosetup-{INNO_VERSION}.exe"
    if not installer.is_file():
        print(f"[inno] Downloading Inno Setup {INNO_VERSION}...", flush=True)
        with urllib.request.urlopen(INNO_URL, timeout=300) as response:  # noqa: S310 - pinned https URL
            data = response.read()
        if hashlib.sha256(data).hexdigest() != INNO_SHA256:
            raise SystemExit(f"[inno] {INNO_URL}: checksum mismatch")
        installer.write_bytes(data)
    print(f"[inno] Unpacking Inno Setup into {iscc.parent}...", flush=True)
    subprocess.run([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", "/CURRENTUSER",
                    "/PORTABLE=1", f"/DIR={iscc.parent}"], check=True)
    if not iscc.is_file():
        raise SystemExit(f"[inno] Inno Setup did not unpack {iscc}")
    return iscc


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Windows installer with Inno Setup.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--arch", choices=("x64", "arm64"), required=True)
    parser.add_argument("--stage", type=Path, required=True, help="mimvox.exe and the bundled resources")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="where the portable Inno Setup is kept")
    args = parser.parse_args()
    if not (args.stage / BINARY).is_file():
        raise SystemExit(f"[inno] {args.stage / BINARY} is missing")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        banners, smalls = draw_images(Path(tmp) / "images")
        script = Path(tmp) / "mimvox.iss"
        script.write_text(iss(args.version, args.arch, args.stage.resolve(), args.out.resolve(), banners, smalls),
                          encoding="utf-8-sig")
        iscc = ensure_iscc(args.cache)
        print(f"[inno] Compiling {script.name}...", flush=True)
        subprocess.run([str(iscc), "/Q", str(script)], check=True)
    print(f"[inno] Wrote {args.out / f'mimvox_{args.version}_{args.arch}-setup.exe'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
