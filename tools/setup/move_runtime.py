"""Move an installed data directory (runtime, models, voices) to another place: ``setup.py --move-to <new dir>``.

Standard library only. Output the app can parse: progress lines ``[move] 37% 11.4/31.0 GB`` while copying, then
``[move] done`` (exit code 0) or ``[move] failed: <reason>`` (non-zero). Everything else is ``[move] <text>``.

Safety:
* refuses while the engine (or anything from the old runtime) is running;
* the old directory is deleted last, only after the copy is verified (same files, same sizes) and the moved runtime
  passed the quick check; an interrupted move resumes (rerun the same command, files already copied are skipped);
* on the same volume the directory is renamed instead (atomic, no copy).

After the files are in place, absolute paths of the old location are rewritten inside the virtual environments: the
editable-install ``.pth`` / finder modules (``_editable_impl_indextts.pth``, ``__editable__*.py``), ``direct_url.json``,
``pyvenv.cfg`` and the activation / console scripts. Windows ``.exe`` console launchers keep the path in a PE resource
that cannot be edited in place; they are not used by the engine and are only reported (``pip``-style reinstall refreshes them).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Callable
from urllib.parse import quote

MARKER = ".mimvox-move.json"
GB = 1e9
TEXT_LIMIT = 2_000_000


class MoveError(RuntimeError):
    """The move cannot continue; the message is shown to the user."""


def out(line: str) -> None:
    print(f"[move] {line}", flush=True)


# ------------------------------------------------------------------------------------------------ preconditions

def running_processes(old: Path, windows: bool) -> list[str]:
    """Descriptions of processes that use the runtime: the engine (``mimvox_engine``) or any program started
    from inside ``old`` (the venv interpreters, Applio training)."""
    needle = os.path.normcase(str(old)).rstrip("\\/")
    found: list[str] = []
    me = os.getpid()
    if windows:
        script = ("Get-CimInstance Win32_Process | ForEach-Object { \"$($_.ProcessId)`t$($_.ExecutablePath)`t$($_.CommandLine)\" }")
        try:
            text = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                                  text=True, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
        except (OSError, subprocess.SubprocessError):
            raise MoveError("cannot list running processes to check that the engine is stopped") from None
        rows = [ln.split("\t", 2) for ln in text.splitlines() if ln.count("\t") >= 2]
        for pid, exe, cmd in rows:
            if not pid.strip().isdigit() or int(pid) == me:
                continue
            if (exe and os.path.normcase(exe).startswith(needle)) or "mimvox_engine" in cmd:
                found.append(f"pid {pid.strip()}: {(exe or cmd)[:100]}")
    else:
        for entry in Path("/proc").glob("[0-9]*") if Path("/proc").is_dir() else []:
            if int(entry.name) == me:
                continue
            try:
                exe = os.readlink(entry / "exe")
                cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if exe.startswith(needle) or "mimvox_engine" in cmd:
                found.append(f"pid {entry.name}: {(exe or cmd)[:100]}")
        if not Path("/proc").is_dir():  # macOS
            try:
                text = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=30).stdout
            except (OSError, subprocess.SubprocessError):
                raise MoveError("cannot list running processes to check that the engine is stopped") from None
            for ln in text.splitlines():
                pid, _, cmd = ln.strip().partition(" ")
                if pid.isdigit() and int(pid) != me and (needle in cmd or "mimvox_engine" in cmd):
                    found.append(f"pid {pid}: {cmd[:100]}")
    return found


def existing_parent(path: Path) -> Path:
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


# Top-level folders a move leaves behind (and deletes with the old folder): uv's package cache and the temporary
# files, both made again when needed. Copying the cache to another drive would also undo its hard links.
NOT_MOVED = ("cache", "tmp")


def _prune_top(root: Path, dirpath: str, dirs: list[str]) -> None:
    if Path(dirpath) == root:
        dirs[:] = [d for d in dirs if d not in NOT_MOVED]


def tree_size(root: Path) -> int:
    total = 0
    for dirpath, dirs, files in os.walk(root):
        _prune_top(root, dirpath, dirs)
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def _inside(child: Path, parent: Path) -> bool:
    c, p = os.path.normcase(str(child)), os.path.normcase(str(parent))
    return c == p or c.startswith(p.rstrip("\\/") + os.sep)


# ------------------------------------------------------------------------------------------------------- copying

def _walk_files(root: Path):
    """Yield ``(relative path, is_symlink)`` of every file and symlink, and directories as ``(rel, None)``."""
    for dirpath, dirs, files in os.walk(root):
        _prune_top(root, dirpath, dirs)
        rel_dir = Path(dirpath).relative_to(root)
        for d in list(dirs):
            if os.path.islink(os.path.join(dirpath, d)):  # a symlinked directory is copied as a link
                dirs.remove(d)
                files.append(d)
        yield rel_dir, None
        for name in files:
            yield rel_dir / name, os.path.islink(os.path.join(dirpath, name))


def copy_tree(src: Path, dst: Path, old: Path, total: int, progress: Callable[[int], None]) -> None:
    """Copy ``src`` to ``dst``. Files are written as ``name.mimvox-part`` and renamed, so a half copy never looks complete;
    a file that is already there with the same size is skipped (resume)."""
    dst.mkdir(parents=True, exist_ok=True)
    done = 0
    for rel, link in _walk_files(src):
        s, d = src / rel, dst / rel
        if link is None:
            d.mkdir(parents=True, exist_ok=True)
            continue
        if link:
            target = os.readlink(s)
            if os.path.normcase(target).startswith(os.path.normcase(str(src))):
                target = str(dst) + target[len(str(src)):]
            if os.path.lexists(d):
                d.unlink()
            os.symlink(target, d, target_is_directory=s.is_dir())
            continue
        size = s.stat().st_size
        if d.is_file() and d.stat().st_size == size:
            done += size
            progress(done)
            continue
        tmp = d.with_name(d.name + ".mimvox-part")
        with open(s, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                block = fin.read(8 * 1024 * 1024)
                if not block:
                    break
                fout.write(block)
                done += len(block)
                progress(done)
        shutil.copystat(s, tmp)
        os.replace(tmp, d)
    progress(total)


def verify_tree(src: Path, dst: Path) -> None:
    """Same set of files and links with the same sizes, else :class:`MoveError`."""
    want = {rel.as_posix(): (None if link else (src / rel).stat().st_size) for rel, link in _walk_files(src) if link is not None}
    have = {rel.as_posix(): (None if link else (dst / rel).stat().st_size) for rel, link in _walk_files(dst) if link is not None}
    have.pop(MARKER, None)
    missing = [k for k in want if k not in have]
    wrong = [k for k, v in want.items() if k in have and have[k] != v]
    stray = [k for k in have if k not in want and not k.endswith(".mimvox-part")]
    parts = [k for k in have if k.endswith(".mimvox-part")]
    if missing or wrong or parts:
        raise MoveError(f"the copy is not complete ({len(missing)} missing, {len(wrong)} with a different size, "
                        f"{len(parts)} unfinished); e.g. {(missing + wrong + parts)[0]}. Run the move again to resume.")
    del stray  # files only the target has (for example the marker) are fine


def remove_tree(path: Path) -> None:
    def onexc(func, p, _exc):  # read-only files (git objects) on Windows
        try:
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            func(p)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=onexc)
    else:
        shutil.rmtree(path, onerror=lambda f, p, e: onexc(f, p, e))


# ------------------------------------------------------------------------------------------------------- path fix

def _forms(old: Path, new: Path) -> list[tuple[bytes, bytes]]:
    o, n = str(old), str(new)
    pairs = [
        (o.replace("\\", "\\\\"), n.replace("\\", "\\\\")),
        (o, n),
        (old.as_posix(), new.as_posix()),
        (quote(old.as_posix(), safe=":/"), quote(new.as_posix(), safe=":/")),
    ]
    seen: set[str] = set()
    result = []
    for a, b in pairs:
        if a not in seen:
            seen.add(a)
            result.append((a.encode("utf-8"), b.encode("utf-8")))
    return result


class PathFixer:
    def __init__(self, old: Path, new: Path, windows: bool) -> None:
        self.pairs = _forms(old, new)
        flags = re.IGNORECASE if windows else 0
        self.rx = [(re.compile(re.escape(a), flags), b) for a, b in self.pairs]
        self.old_b = str(old).encode("utf-8")
        self.new_b = str(new).encode("utf-8")
        self.fixed: list[str] = []
        self.unfixed: list[str] = []

    def replace(self, data: bytes) -> bytes:
        for rx, b in self.rx:
            data = rx.sub(lambda _m, b=b: b, data)
        return data

    def has_old(self, data: bytes) -> bool:
        return any(rx.search(data) for rx, _ in self.rx)

    def text_file(self, path: Path) -> None:
        try:
            if path.stat().st_size > TEXT_LIMIT:
                return
            data = path.read_bytes()
        except OSError:
            return
        if b"\0" in data or not self.has_old(data):
            return
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return
        path.write_bytes(self.replace(data))
        self.fixed.append(str(path))

    def launcher(self, path: Path) -> None:
        """A Windows console-script launcher (``.exe``) keeps the python path in a PE resource, whose size cannot
        change in place. The engine never starts them (it runs ``python.exe`` directly), so they are only reported."""
        try:
            data = path.read_bytes()
        except OSError:
            return
        if self.has_old(data) or any(a.decode("utf-8").encode("utf-16le") in data for a, _ in self.pairs):
            self.unfixed.append(str(path))


def fix_venv_paths(root: Path, old: Path, windows: bool) -> PathFixer:
    """Rewrite ``old`` -> ``root`` inside every virtual environment under ``<root>/runtime``."""
    fixer = PathFixer(old, root, windows)
    for venv in sorted(root.glob("runtime/*/.venv")) + sorted(root.glob("runtime/*/venv")):
        fixer.text_file(venv / "pyvenv.cfg")
        for sub in ("Scripts", "bin"):
            for f in sorted((venv / sub).glob("*")) if (venv / sub).is_dir() else []:
                if f.is_file() and not f.is_symlink():
                    (fixer.launcher if f.suffix.lower() == ".exe" else fixer.text_file)(f)
        sites = list(venv.glob("Lib/site-packages")) + list(venv.glob("lib/python*/site-packages"))
        for site in sites:
            for pattern in ("*.pth", "*.egg-link", "__editable__*.py", "_editable_impl*.py"):
                for f in sorted(site.glob(pattern)):
                    fixer.text_file(f)
            for f in sorted(site.glob("*.dist-info/direct_url.json")):
                fixer.text_file(f)
    return fixer


# --------------------------------------------------------------------------------------------------------- driver

def prepare(old: Path | None, new: Path, windows: bool, force: bool) -> tuple[Path, bool, bool]:
    """Validate the request. Returns ``(old, resume, same_volume)``."""
    marker = new / MARKER
    resume = marker.is_file()
    if resume:
        try:
            old = Path(json.loads(marker.read_text(encoding="utf-8"))["from"])
        except (OSError, ValueError, KeyError):
            raise MoveError(f"{marker} is unreadable; remove the unfinished folder {new} and start again") from None
        out(f"Resuming an interrupted move from {old}")
    if old is None:
        raise MoveError("no current data directory found; pass --data-dir <current dir>")
    if _inside(new, old) or _inside(old, new):
        raise MoveError("the new folder must not be inside the old one (or the other way round)")
    if not resume:
        if not (old / "runtime").is_dir() and not (old / "runtime.toml").is_file():
            raise MoveError(f"nothing to move: no runtime in {old}")
        if new.exists() and any(new.iterdir()):
            raise MoveError(f"{new} is not empty")
    busy = running_processes(old, windows)
    if busy:
        raise MoveError("the engine or the runtime is still running; close the app first (" + "; ".join(busy[:3]) + ")")
    same = False
    if old.exists():
        try:
            same = os.stat(old).st_dev == os.stat(existing_parent(new)).st_dev
        except OSError:
            same = False
    if os.environ.get("MIMVOX_MOVE_FORCE_COPY"):  # testing: take the copy path on one volume
        same = False
    return old, resume, same


def move_files(old: Path, new: Path, resume: bool, same_volume: bool, force: bool) -> None:
    """Get the files to ``new`` (rename or verified copy). The old directory is left in place when it was copied."""
    if not old.exists():
        if not (new / MARKER).is_file():
            raise MoveError(f"{old} does not exist")
        return  # renamed earlier; only the fix-up is left
    marker_data = json.dumps({"from": str(old), "to": str(new)})
    if same_volume and not resume:
        (old / MARKER).write_text(marker_data, encoding="utf-8")
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            new.rmdir()
        try:
            os.rename(old, new)
            out(f"Renamed {old} to {new} (same volume, nothing copied)")
            out("100% done")
            return
        except OSError as exc:
            (old / MARKER).unlink(missing_ok=True)
            out(f"Rename not possible ({exc}); copying instead")
    total = tree_size(old)
    free = shutil.disk_usage(existing_parent(new)).free
    already = tree_size(new) if new.exists() else 0
    need = max(0, total - already) + int(1 * GB)
    if free < need and not force:
        raise MoveError(f"not enough free space on the target: {free / GB:.1f} GB free, {need / GB:.1f} GB needed")
    new.mkdir(parents=True, exist_ok=True)
    (new / MARKER).write_text(marker_data, encoding="utf-8")
    last = [-1]

    def progress(done: int) -> None:
        pct = int(done * 100 / total) if total else 100
        if pct != last[0]:
            last[0] = pct
            out(f"{pct}% {done / GB:.1f}/{total / GB:.1f} GB")

    out(f"Copying {total / GB:.1f} GB from {old} to {new}")
    copy_tree(old, new, old, total, progress)
    out("Verifying the copy")
    verify_tree(old, new)
    out("Copy verified")
