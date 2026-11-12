"""voices.list / create / rename / delete, plus helpers other modules use to read a voice."""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import threading
import time
import unicodedata
from collections.abc import Callable
from typing import Any

import soundfile as sf

from . import config, media
from .registry import Context, EngineError, method

_LOCK = threading.RLock()
_DURATIONS: dict[tuple[str, int, int], float] = {}  # (path, mtime_ns, size) -> seconds


# Cyrillic -> Latin: the Ukrainian national transliteration (KMU 2010; "Мій голос" -> "mii-holos"), plus the
# letters of Russian, Belarusian and the other Cyrillic alphabets. Word-initial forms are in _CYR_FIRST.
_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh", "з": "z", "и": "y",
    "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ь": "", "ю": "iu",
    "я": "ia", "'": "", "ʼ": "", "’": "",
    "ё": "io", "ъ": "", "ы": "y", "э": "e", "ў": "u", "ђ": "dj", "ј": "j", "љ": "lj", "њ": "nj", "ћ": "c",
    "џ": "dz", "ѓ": "gj", "ќ": "kj", "ѕ": "dz", "қ": "q", "ғ": "gh", "ң": "ng", "ү": "u", "ұ": "u", "һ": "h",
    "ә": "a", "ө": "o", "ҙ": "z", "ҡ": "q", "ҫ": "s",
}
_CYR_FIRST = {"є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya", "ё": "yo"}
_GREEK = {
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i", "κ": "k", "λ": "l",
    "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y", "φ": "f",
    "χ": "ch", "ψ": "ps", "ω": "o",
}
_LATIN = {"ß": "ss", "æ": "ae", "œ": "oe", "ø": "o", "ł": "l", "đ": "d", "ð": "d", "þ": "th", "ı": "i"}


def transliterate(text: str) -> str:
    """Latin letters for a name in another script (Cyrillic, Greek; accents dropped from Latin letters). Characters
    of other scripts are dropped."""
    low = text.lower()
    out: list[str] = []
    prev_letter = False
    for i, ch in enumerate(low):
        if ch == "г" and i > 0 and low[i - 1] == "з":
            out.append("gh")  # KMU 2010: "зг" is "zgh", so it is not read as "zh"
        elif ch in _CYR or ch in _GREEK or ch in _LATIN:
            if not prev_letter and ch in _CYR_FIRST:
                out.append(_CYR_FIRST[ch])
            else:
                out.append(_CYR.get(ch) or _GREEK.get(ch) or _LATIN.get(ch, ""))
        else:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(_GREEK.get(base) or "".join(c for c in base if c.isascii()))
        prev_letter = ch.isalpha() or ch in "'ʼ’"
    return "".join(out)


def slugify(name: str) -> str:
    """The voice id made from a name: ``Anna`` -> ``anna``, ``Мій голос`` -> ``mii-holos``, ``Zoë`` -> ``zoe``."""
    slug = re.sub(r"[^a-z0-9]+", "-", transliterate(name)).strip("-")
    return slug[:48].strip("-") or "voice"


def _write_json(path: str, obj: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def read_meta(data: config.DataDir, voice_id: str) -> dict[str, Any]:
    path = os.path.join(data.voice(voice_id), "voice.json")
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except FileNotFoundError:
        raise EngineError("not_found", f"voice {voice_id!r} does not exist") from None
    except (OSError, ValueError) as exc:
        raise EngineError("internal", f"cannot read {path}: {exc}") from exc
    meta.setdefault("id", voice_id)
    return meta


def update_meta(data: config.DataDir, voice_id: str, mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Read ``voice.json``, let ``mutate`` change it in place and write it back atomically (under the module lock)."""
    with _LOCK:
        meta = read_meta(data, voice_id)
        mutate(meta)
        _write_json(os.path.join(data.voice(voice_id), "voice.json"), meta)
        return meta


def paused_training(meta: dict[str, Any]) -> dict[str, Any] | None:
    """The public part of a paused (or interrupted) training run recorded in ``voice.json``, or ``None``."""
    p = meta.get("training_paused")
    if not isinstance(p, dict) or not isinstance(p.get("epoch"), int) or not isinstance(p.get("total_epochs"), int):
        return None
    return {
        "epoch": p["epoch"],
        "total_epochs": p["total_epochs"],
        "paused_at": p.get("paused_at"),
        "interrupted": bool(p.get("interrupted", False)),
    }


def _wav_seconds(path: str) -> float:
    try:
        st = os.stat(path)
    except OSError:
        return 0.0
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _DURATIONS:
        try:
            _DURATIONS[key] = float(sf.info(path).duration)
        except Exception:  # noqa: BLE001 - corrupt take: count as empty
            _DURATIONS[key] = 0.0
    return _DURATIONS[key]


def _trained_epochs(vdir: str, meta: dict[str, Any]) -> int | None:
    if isinstance(meta.get("trained_epochs"), int):
        return meta["trained_epochs"]
    for name in ("meta.json", "train.json"):  # written by the training module, if it records it
        try:
            with open(os.path.join(vdir, "rvc", name), encoding="utf-8") as fh:
                value = json.load(fh).get("epochs")
            if isinstance(value, int):
                return value
        except (OSError, ValueError, AttributeError):
            continue
    return None


def describe(data: config.DataDir, voice_id: str) -> dict[str, Any]:
    """The voice object of the contract (``voices.list`` item)."""
    meta = read_meta(data, voice_id)
    vdir = data.voice(voice_id)
    parts = sorted(glob.glob(os.path.join(vdir, "dataset", "part*.wav")))
    pth, _ = data.rvc_model(voice_id)
    trained = os.path.isfile(pth)
    return {
        "id": meta["id"],
        "name": meta.get("name", voice_id),
        "created": meta.get("created"),
        "preset": meta.get("preset"),
        "language": meta.get("language") or "en",
        "dataset_minutes": round(sum(_wav_seconds(p) for p in parts) / 60.0, 2),
        "parts": len(parts),
        "trained": trained,
        "trained_epochs": _trained_epochs(vdir, meta) if trained else None,
        "trained_at": meta.get("trained_at") if trained else None,
        "training_paused": paused_training(meta),
        "references": _language_references(vdir),
        **_reference_languages(vdir),
    }


def _language_references(vdir: str) -> list[str]:
    from .dataset.reference import language_references  # noqa: PLC0415 - lazy: the dataset package imports this one

    return language_references(vdir)


def _reference_languages(vdir: str) -> dict[str, Any]:
    """``reference_language`` / ``reference_languages``: the real language of the clips (reflang.py)."""
    from . import reflang  # noqa: PLC0415 - lazy, as above

    try:
        return reflang.describe(vdir)
    except Exception:  # noqa: BLE001 - a broken sidecar must not hide the voice
        return {"reference_language": None, "reference_languages": {}}


def list_voice_ids(data: config.DataDir) -> list[str]:
    try:
        names = os.listdir(data.voices)
    except FileNotFoundError:
        return []
    return sorted(n for n in names if os.path.isfile(os.path.join(data.voices, n, "voice.json")))


def _clean_name(params: dict[str, Any]) -> str:
    name = params.get("name")
    if not isinstance(name, str) or not name.strip():
        raise EngineError("bad_request", "'name' must be a non-empty string")
    return name.strip()[:80]


def _clean_language(params: dict[str, Any]) -> str:
    """The voice's recording language (ISO 639-1, see ``dataset.lang``); ``en`` when absent or ``auto``."""
    from .dataset import lang  # noqa: PLC0415 - lazy: keeps this module importable on its own

    try:
        return lang.normalize_code(params.get("language")) or lang.DEFAULT
    except ValueError as exc:
        raise EngineError("bad_request", str(exc)) from None


@method("voices.list")
def voices_list(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    out = []
    for vid in list_voice_ids(data):
        try:
            out.append(describe(data, vid))
        except EngineError:
            continue
    out.sort(key=lambda v: v.get("created") or "")
    return {"voices": out}


@method("voices.create")
def voices_create(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    name = _clean_name(params)
    preset = params.get("preset")
    language = _clean_language(params)
    with _LOCK:
        base = slugify(name)
        vid, n = base, 2
        while os.path.exists(data.voice(vid)):
            vid, n = f"{base}-{n}", n + 1
        vdir = data.voice(vid)
        os.makedirs(os.path.join(vdir, "dataset"))
        _write_json(os.path.join(vdir, "voice.json"), {
            "id": vid,
            "name": name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preset": preset if isinstance(preset, str) else None,
            "language": language,
            "reference": "reference.wav",
        })
    return {"voice": describe(data, vid)}


@method("voices.rename")
def voices_rename(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """``{"id","name"}`` renames; ``{"id","language"}`` (name optional) changes the recording language, e.g. when the
    first texts read into a still empty voice are in another language."""
    data = config.data_dir(ctx)
    vid = params.get("id")
    name = _clean_name(params) if "name" in params or "language" not in params else None
    language = _clean_language(params) if params.get("language") not in (None, "") else None
    with _LOCK:
        meta = read_meta(data, str(vid))
        if name is not None:
            meta["name"] = name
        if language is not None:
            meta["language"] = language
        _write_json(os.path.join(data.voice(meta["id"]), "voice.json"), meta)
    return {"voice": describe(data, meta["id"])}


@method("voices.delete")
def voices_delete(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    vid = str(params.get("id"))
    with _LOCK:
        read_meta(data, vid)  # not_found if missing
        shutil.rmtree(data.voice(vid))
    try:  # a paused training keeps GBs of features and checkpoints in Applio's log folder
        from .training.resume import discard_logdir  # noqa: PLC0415 - lazy: the training package imports this module
        discard_logdir(ctx.runtime, vid)
    except Exception:  # noqa: BLE001 - never fail a delete over leftovers
        pass
    return {}


# --------------------------------------------------------------------------------------------------- voices.import

_PART_FILE = re.compile(
    r"^part(\d+)\.(?:wav|mp3|m4a|aac|flac|ogg|oga|opus|wma|aif|aiff|mp4|mkv|mov|webm|avi)$", re.IGNORECASE)
_EPOCHS_IN_NAME = re.compile(r"_(\d+)e(?:_|\.)")


def _existing_file(params: dict[str, Any], key: str, required: bool = False) -> str | None:
    value = params.get(key)
    if value is None or value == "":
        if required:
            raise EngineError("bad_request", f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise EngineError("bad_request", f"'{key}' must be a path string")
    path = os.path.abspath(value)
    if not os.path.isfile(path):
        raise EngineError("not_found", f"{key}: file not found: {path}")
    return path


def _atomic_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def import_voice(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Create a voice from material that already exists (recordings and, optionally, a trained RVC model).

    ``reference_path`` becomes ``reference.wav`` (mono 16-bit WAV, as recorded: no trimming). Every ``part<N>.wav`` in
    ``dataset_dir`` goes through ``dataset.add`` (trim, normalise, quality verdict). ``rvc_model`` / ``rvc_index``
    are copied to ``rvc/model.pth`` / ``rvc/model.index`` and the voice counts as trained: no retraining needed.
    On any failure the half-made voice is removed again.
    """
    data = config.data_dir(ctx)
    ref_src = _existing_file(params, "reference_path")
    pth_src = _existing_file(params, "rvc_model")
    idx_src = _existing_file(params, "rvc_index")
    if idx_src and not pth_src:
        raise EngineError("bad_request", "'rvc_index' needs 'rvc_model'")
    ds_dir = params.get("dataset_dir")
    parts: list[tuple[int, str]] = []
    if ds_dir not in (None, ""):
        if not isinstance(ds_dir, str) or not os.path.isdir(ds_dir):
            raise EngineError("not_found", f"dataset_dir: folder not found: {ds_dir!r}")
        for name in os.listdir(ds_dir):
            m = _PART_FILE.match(name)
            if m:
                parts.append((int(m.group(1)), os.path.join(os.path.abspath(ds_dir), name)))
        parts.sort()
        if not parts:
            raise EngineError("bad_request", "dataset_dir has no part<N> audio files (wav, mp3, m4a, flac, ...)")
    if not (ref_src or parts):
        raise EngineError("bad_request", "give a 'reference_path' and/or a 'dataset_dir' with part<N>.wav files")

    created = voices_create(ctx, {"name": params.get("name"), "preset": params.get("preset"),
                                  "language": params.get("language")})["voice"]
    vid = created["id"]
    vdir = data.voice(vid)
    try:
        if parts:
            from .registry import methods  # noqa: PLC0415

            add = methods().get("dataset.add")
            if add is None:
                raise EngineError("not_ready", "the dataset module is not available")
            for n, path in parts:
                add(ctx, {"voice_id": vid, "part_index": n, "path": path})
        if ref_src:
            y, sr = media.read_audio(ref_src)  # any format the shared decoder reads
            sf.write(data.reference(vid), y, sr, subtype="PCM_16", format="WAV")
        if not os.path.isfile(data.reference(vid)):
            raise EngineError("bad_request", "no usable reference could be made from these recordings")
        if pth_src:
            pth, idx = data.rvc_model(vid)
            _atomic_copy(pth_src, pth)
            if idx_src:
                _atomic_copy(idx_src, idx)
            with _LOCK:
                meta = read_meta(data, vid)
                m = _EPOCHS_IN_NAME.search(os.path.basename(pth_src))
                meta.update(trained=True, trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"), imported=True)
                if m:
                    meta["trained_epochs"] = int(m.group(1))
                _write_json(os.path.join(vdir, "voice.json"), meta)
    except BaseException:
        shutil.rmtree(vdir, ignore_errors=True)
        raise
    return {"voice": describe(data, vid)}


@method("voices.import")
def voices_import(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    return import_voice(ctx, params)
