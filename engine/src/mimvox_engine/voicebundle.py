from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Any

from . import __version__, config, paths, reflang, voices
from .registry import Context, EngineError

FORMAT = "mimvox-voice"
VERSION = 1
MANIFEST = "manifest.json"
MAX_JSON = 1 << 20
MAX_MEMBER = 2 << 30
MAX_TOTAL = 4 << 30
_FIXED_TIME = (1980, 1, 1, 0, 0, 0)
_VOLATILE = ("training_run", "training_paused")
_MEMBER = re.compile(r"voice\.json|reference(?:_[a-z]{2,3})?\.(?:wav|omnivoice\.json)|rvc/model\.(?:pth|index)")
_CHUNK = 1 << 20


def _voice_files(vdir: Path, voice_json: bytes) -> list[str]:
    clips = ["reference.wav", *(f"reference_{code}.wav" for code in voices._language_references(vdir))]
    names = [n for n in [*clips, "rvc/model.pth", "rvc/model.index"] if vdir.joinpath(*n.split("/")).is_file()]
    names += [
        Path(reflang.sidecar_path(c)).name for c in clips if c in names and reflang.read_sidecar(vdir / c) is not None
    ]
    total = len(voice_json)
    for name in names:
        size = vdir.joinpath(*name.split("/")).stat().st_size
        total += size
        if size > (MAX_JSON if name.endswith(".json") else MAX_MEMBER) or total > MAX_TOTAL:
            raise EngineError("bad_request", f"{name} is too large for a voice bundle ({size} bytes)")
    if len(voice_json) > MAX_JSON:
        raise EngineError("bad_request", "voice.json is too large for a voice bundle")
    return sorted(names)


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=paths.json_default) + "\n").encode(
        "utf-8"
    )


def _zipinfo(name: str, size: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, _FIXED_TIME)
    zi.compress_type = zipfile.ZIP_STORED  # deflate output differs between zlib builds, so the hash would too
    zi.create_system = 3
    zi.external_attr = 0o100644 << 16
    zi.file_size = size
    return zi


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def export_voice(data: config.DataDir, voice_id: str, dest: Any) -> dict[str, Any]:
    if dest not in (None, "") and (not isinstance(dest, str) or not Path(dest).is_absolute()):
        raise EngineError("bad_request", "'dest' must be an absolute file path")
    if dest and Path(dest).is_dir():
        raise EngineError("bad_request", f"'dest' is a folder: {dest}")
    meta = {k: v for k, v in voices.read_meta(data, voice_id).items() if k not in _VOLATILE}
    voice_json = _canonical(meta)
    vdir = data.voice(voice_id)
    names = _voice_files(vdir, voice_json)
    if "reference.wav" not in names:
        raise EngineError("bad_request", f"voice {voice_id!r} has no reference clip yet")
    out_dir = Path(dest).parent if dest else data.tmp / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    part = out_dir / f".{uuid.uuid4().hex}.part"
    files: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(part, "w") as zf:
            for name in sorted([*names, "voice.json"]):
                h = hashlib.sha256()
                if name == "voice.json":
                    zf.writestr(_zipinfo(name, len(voice_json)), voice_json)
                    h.update(voice_json)
                    size = len(voice_json)
                else:
                    src = vdir.joinpath(*name.split("/"))
                    size = 0
                    with src.open("rb") as fh, zf.open(_zipinfo(name, src.stat().st_size), "w") as out:
                        while chunk := fh.read(_CHUNK):
                            h.update(chunk)
                            out.write(chunk)
                            size += len(chunk)
                files.append({"path": name, "bytes": size, "sha256": h.hexdigest()})
            manifest = _canonical(
                {
                    "format": FORMAT,
                    "version": VERSION,
                    "id": voice_id,
                    "name": meta.get("name", voice_id),
                    "files": files,
                }
            )
            zf.writestr(_zipinfo(MANIFEST, len(manifest)), manifest)
        digest = _sha256_file(part)
        path = Path(dest).absolute() if dest else out_dir / f"{voice_id}-{digest[:12]}.mimvoice"
        part.replace(path)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    return {"path": path, "sha256": digest, "bytes": path.stat().st_size, "files": files, "engine": __version__}


def _bad(message: str) -> EngineError:
    return EngineError("bad_request", f"not a usable voice bundle: {message}")


def _safe_name(name: str) -> bool:
    parts = name.split("/")
    return not ("\\" in name or ":" in name or name.startswith("/") or any(p in ("", ".", "..") for p in parts))


def _check_members(zf: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in zf.infolist():
        name = info.filename
        if not _safe_name(name):
            raise _bad(f"unsafe file name {name!r}")
        if name in members:
            raise _bad(f"{name!r} is in it twice")
        if name != MANIFEST and not _MEMBER.fullmatch(name):
            raise _bad(f"unexpected file {name!r}")
        cap = MAX_JSON if name.endswith(".json") else MAX_MEMBER
        total += info.file_size
        if info.file_size > cap or total > MAX_TOTAL:
            raise _bad(f"{name!r} is too large ({info.file_size} bytes)")
        members[name] = info
    return members


def _read_manifest(zf: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo]) -> dict[str, Any]:
    if MANIFEST not in members:
        raise _bad("no manifest.json")
    try:
        manifest = json.loads(zf.read(members[MANIFEST]).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise _bad(f"manifest.json is not valid JSON ({exc})") from None
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise _bad("manifest.json is not a Mimvox voice manifest")
    version = manifest.get("version")
    if not isinstance(version, int) or version > VERSION:
        raise _bad(f"bundle format version {version!r} is newer than this engine reads ({VERSION})")
    files = manifest.get("files")
    if not isinstance(files, list) or not all(
        isinstance(f, dict)
        and isinstance(f.get("path"), str)
        and isinstance(f.get("bytes"), int)
        and isinstance(f.get("sha256"), str)
        for f in files
    ):
        raise _bad("manifest.json has no valid file list")
    listed = [f["path"] for f in files]
    if len(set(listed)) != len(listed) or set(listed) != set(members) - {MANIFEST}:
        raise _bad("the files do not match manifest.json")
    if "voice.json" not in listed or "reference.wav" not in listed:
        raise _bad("voice.json or reference.wav is missing")
    return manifest


def _extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, entry: dict[str, Any], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    with zf.open(info) as src, dst.open("wb") as out:
        while chunk := src.read(_CHUNK):
            size += len(chunk)
            if size > entry["bytes"]:
                break
            h.update(chunk)
            out.write(chunk)
    if size != entry["bytes"] or h.hexdigest() != entry["sha256"].lower():
        raise _bad(f"{info.filename!r} does not match its hash in manifest.json")


def _keep_sidecars(vdir: Path) -> None:
    # A sidecar is valid only for its clip's [mtime_ns, size]; the bytes are verified, so the mtime is restored.
    for wav in vdir.iterdir():
        name = wav.name
        if not name.endswith(".wav"):
            continue
        try:
            with Path(reflang.sidecar_path(wav)).open(encoding="utf-8") as fh:
                sig = json.load(fh).get("sig")
            if isinstance(sig, list) and len(sig) == 2 and sig[1] == wav.stat().st_size:
                os.utime(wav, ns=(int(sig[0]), int(sig[0])))
        except OverflowError:
            raise _bad(f"the sidecar of {name!r} names an impossible time") from None
        except (OSError, ValueError, AttributeError, TypeError, RecursionError):
            continue


def _clean_meta(raw: bytes) -> dict[str, Any]:
    from .dataset import lang

    try:
        meta = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        meta = None
    if not isinstance(meta, dict) or not isinstance(meta.get("name"), str) or not meta["name"].strip():
        raise _bad("voice.json has no voice name")
    try:
        meta["language"] = lang.normalize_code(meta.get("language")) or lang.DEFAULT
    except ValueError as exc:
        heard = reflang.norm(meta.get("language"))  # a voice made from media keeps the language heard in it
        if not heard or not re.fullmatch(r"[a-z]{2,3}", heard):
            raise _bad(str(exc)) from None
        meta["language"] = heard
    return {k: v for k, v in meta.items() if k not in _VOLATILE}


def import_bundle(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    for key in ("reference_path", "dataset_dir", "rvc_model", "rvc_index"):
        if params.get(key) not in (None, ""):
            raise EngineError("bad_request", f"'bundle' cannot be combined with '{key}'")
    path = voices._existing_file(params, "bundle", required=True)
    assert path is not None
    name = voices._clean_name(params) if params.get("name") not in (None, "") else None
    stage = data.tmp / "import" / uuid.uuid4().hex
    try:
        try:
            with zipfile.ZipFile(path) as zf:
                members = _check_members(zf)
                manifest = _read_manifest(zf, members)
                for entry in manifest["files"]:
                    _extract(zf, members[entry["path"]], entry, stage.joinpath(*entry["path"].split("/")))
        except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError, EOFError) as exc:
            raise _bad(str(exc)) from None
        meta = _clean_meta((stage / "voice.json").read_bytes())
        if name is not None:
            meta["name"] = name
        (stage / "dataset").mkdir(parents=True, exist_ok=True)
        _keep_sidecars(stage)
        data.voices.mkdir(parents=True, exist_ok=True)
        with voices._LOCK:
            wanted = manifest.get("id")
            if (
                isinstance(wanted, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", wanted)
                and not data.voice(wanted).exists()
            ):
                vid = wanted
            else:
                base = voices.slugify(meta["name"])
                vid, n = base, 2
                while data.voice(vid).exists():
                    vid, n = f"{base}-{n}", n + 1
            meta["id"] = vid
            voices._write_json(stage / "voice.json", meta)
            stage.rename(data.voice(vid))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"voice": voices.describe(data, vid)}
