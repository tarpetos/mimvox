from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
import zipfile
from typing import Any

from . import __version__, config, reflang, voices
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


def _voice_files(vdir: str, voice_json: bytes) -> list[str]:
    clips = ["reference.wav", *(f"reference_{code}.wav" for code in voices._language_references(vdir))]
    names = [n for n in [*clips, "rvc/model.pth", "rvc/model.index"] if os.path.isfile(os.path.join(vdir, *n.split("/")))]
    names += [os.path.basename(reflang.sidecar_path(c)) for c in clips
              if c in names and reflang.read_sidecar(os.path.join(vdir, c)) is not None]
    total = len(voice_json)
    for name in names:
        size = os.path.getsize(os.path.join(vdir, *name.split("/")))
        total += size
        if size > (MAX_JSON if name.endswith(".json") else MAX_MEMBER) or total > MAX_TOTAL:
            raise EngineError("bad_request", f"{name} is too large for a voice bundle ({size} bytes)")
    if len(voice_json) > MAX_JSON:
        raise EngineError("bad_request", "voice.json is too large for a voice bundle")
    return sorted(names)


def _canonical(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _zipinfo(name: str, size: int) -> zipfile.ZipInfo:
    zi = zipfile.ZipInfo(name, _FIXED_TIME)
    zi.compress_type = zipfile.ZIP_STORED  # deflate output differs between zlib builds, so the hash would too
    zi.create_system = 3
    zi.external_attr = 0o100644 << 16
    zi.file_size = size
    return zi


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def export_voice(data: config.DataDir, voice_id: str, dest: Any) -> dict[str, Any]:
    if dest not in (None, "") and (not isinstance(dest, str) or not os.path.isabs(dest)):
        raise EngineError("bad_request", "'dest' must be an absolute file path")
    if dest and os.path.isdir(dest):
        raise EngineError("bad_request", f"'dest' is a folder: {dest}")
    meta = {k: v for k, v in voices.read_meta(data, voice_id).items() if k not in _VOLATILE}
    voice_json = _canonical(meta)
    vdir = data.voice(voice_id)
    names = _voice_files(vdir, voice_json)
    if "reference.wav" not in names:
        raise EngineError("bad_request", f"voice {voice_id!r} has no reference clip yet")
    out_dir = os.path.dirname(dest) if dest else os.path.join(data.tmp, "export")
    os.makedirs(out_dir, exist_ok=True)
    part = os.path.join(out_dir, f".{uuid.uuid4().hex}.part")
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
                    src = os.path.join(vdir, *name.split("/"))
                    size = 0
                    with open(src, "rb") as fh, zf.open(_zipinfo(name, os.path.getsize(src)), "w") as out:
                        while chunk := fh.read(_CHUNK):
                            h.update(chunk)
                            out.write(chunk)
                            size += len(chunk)
                files.append({"path": name, "bytes": size, "sha256": h.hexdigest()})
            manifest = _canonical({"format": FORMAT, "version": VERSION, "id": voice_id,
                                   "name": meta.get("name", voice_id), "files": files})
            zf.writestr(_zipinfo(MANIFEST, len(manifest)), manifest)
        digest = _sha256_file(part)
        path = os.path.abspath(dest) if dest else os.path.join(out_dir, f"{voice_id}-{digest[:12]}.mimvoice")
        os.replace(part, path)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    return {"path": path, "sha256": digest, "bytes": os.path.getsize(path), "files": files, "engine": __version__}


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
            isinstance(f, dict) and isinstance(f.get("path"), str) and isinstance(f.get("bytes"), int)
            and isinstance(f.get("sha256"), str) for f in files):
        raise _bad("manifest.json has no valid file list")
    listed = [f["path"] for f in files]
    if len(set(listed)) != len(listed) or set(listed) != set(members) - {MANIFEST}:
        raise _bad("the files do not match manifest.json")
    if "voice.json" not in listed or "reference.wav" not in listed:
        raise _bad("voice.json or reference.wav is missing")
    return manifest


def _extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, entry: dict[str, Any], dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    h = hashlib.sha256()
    size = 0
    with zf.open(info) as src, open(dst, "wb") as out:
        while chunk := src.read(_CHUNK):
            size += len(chunk)
            if size > entry["bytes"]:
                break
            h.update(chunk)
            out.write(chunk)
    if size != entry["bytes"] or h.hexdigest() != entry["sha256"].lower():
        raise _bad(f"{info.filename!r} does not match its hash in manifest.json")


def _keep_sidecars(vdir: str) -> None:
    # A sidecar is valid only for its clip's [mtime_ns, size]; the bytes are verified, so the mtime is restored.
    for name in os.listdir(vdir):
        if not name.endswith(".wav"):
            continue
        wav = os.path.join(vdir, name)
        try:
            with open(reflang.sidecar_path(wav), encoding="utf-8") as fh:
                sig = json.load(fh).get("sig")
            if isinstance(sig, list) and len(sig) == 2 and sig[1] == os.path.getsize(wav):
                os.utime(wav, ns=(int(sig[0]), int(sig[0])))
        except OverflowError:
            raise _bad(f"the sidecar of {name!r} names an impossible time") from None
        except (OSError, ValueError, AttributeError, TypeError, RecursionError):
            continue


def _clean_meta(raw: bytes) -> dict[str, Any]:
    from .dataset import lang  # noqa: PLC0415 - lazy, as in voices

    try:
        meta = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, RecursionError):
        meta = None
    if not isinstance(meta, dict) or not isinstance(meta.get("name"), str) or not meta["name"].strip():
        raise _bad("voice.json has no voice name")
    try:
        meta["language"] = lang.normalize_code(meta.get("language")) or lang.DEFAULT
    except ValueError as exc:
        raise _bad(str(exc)) from None
    return {k: v for k, v in meta.items() if k not in _VOLATILE}


def import_bundle(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    for key in ("reference_path", "dataset_dir", "rvc_model", "rvc_index"):
        if params.get(key) not in (None, ""):
            raise EngineError("bad_request", f"'bundle' cannot be combined with '{key}'")
    path = voices._existing_file(params, "bundle", required=True)
    assert path is not None
    name = voices._clean_name(params) if params.get("name") not in (None, "") else None
    stage = os.path.join(data.tmp, "import", uuid.uuid4().hex)
    try:
        try:
            with zipfile.ZipFile(path) as zf:
                members = _check_members(zf)
                manifest = _read_manifest(zf, members)
                for entry in manifest["files"]:
                    _extract(zf, members[entry["path"]], entry, os.path.join(stage, *entry["path"].split("/")))
        except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError, EOFError) as exc:
            raise _bad(str(exc)) from None
        with open(os.path.join(stage, "voice.json"), "rb") as fh:
            meta = _clean_meta(fh.read())
        if name is not None:
            meta["name"] = name
        os.makedirs(os.path.join(stage, "dataset"), exist_ok=True)
        _keep_sidecars(stage)
        os.makedirs(data.voices, exist_ok=True)
        with voices._LOCK:
            wanted = manifest.get("id")
            if isinstance(wanted, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", wanted) \
                    and not os.path.exists(data.voice(wanted)):
                vid = wanted
            else:
                base = voices.slugify(meta["name"])
                vid, n = base, 2
                while os.path.exists(data.voice(vid)):
                    vid, n = f"{base}-{n}", n + 1
            meta["id"] = vid
            voices._write_json(os.path.join(stage, "voice.json"), meta)
            os.rename(stage, data.voice(vid))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return {"voice": voices.describe(data, vid)}
