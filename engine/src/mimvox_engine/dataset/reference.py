"""``voices.reference_candidates`` / ``voices.set_reference``: choose which clip of the recordings is ``reference.wav``.

The speech model copies tempo and manner from the reference, so the clip decides how fast and how natural the cloned voice
sounds. ``dataset.add`` picks one automatically (see ``pace.py``); these two methods let the user hear the best few
and pick another. A manual pick is kept until its part is removed or re-recorded. Windows of the media pool
(``voices.reference_from_media``, see ``media_ref.py``) are offered too (``source: "media"``); picking one writes the
pool's target (``reference.wav``, or ``reference_<lang>.wav`` when the pool was made for another language).
"""
from __future__ import annotations

import os
import shutil
from typing import Any

import numpy as np
import soundfile as sf

from ..registry import EngineError, method
from . import pace, store
from ._audio import read_audio, write_wav

N_CANDIDATES = 5
MAX_PER_PART = 2
SET_MIN_S, SET_MAX_S = 3.0, 40.0


def _vdir(ctx: Any, params: dict[str, Any]) -> tuple[str, str]:
    vid = params.get("id", params.get("voice_id"))
    vdir = store._voice_dir(ctx, {"voice_id": vid})
    return str(vid), vdir


def _ensure_meta(vdir: str, meta: dict[str, Any]) -> bool:
    """Meta + pace analysis for every stored part (parts copied in by hand, voices from before the pace analysis)."""
    changed = False
    for n in store._existing_parts(vdir):
        if str(n) not in meta["parts"]:
            meta["parts"][str(n)] = store._analyse_stored(vdir, n)
            changed = True
    return store._ensure_analysed(vdir, meta) or changed


def _public(c: dict[str, Any], title: str | None, median: float) -> dict[str, Any]:
    return {
        "source": c.get("source") or "part", "file": c.get("file"),
        "part": c.get("part"), "title": title,
        "start_s": round(float(c["start_s"]), 2), "end_s": round(float(c["end_s"]), 2),
        "duration_s": round(float(c["duration_s"]), 1),
        "rate": round(float(c["rate"]), 2), "pauses": round(float(c["pauses"]), 3),
        "snr_db": round(float(c["snr_db"]), 1),
        "hint": pace.hint(float(c["rate"]), median),
    }


def _current(vdir: str, meta: dict[str, Any], titles: dict[int, str], median: float) -> dict[str, Any] | None:
    ref_path = os.path.join(vdir, "reference.wav")
    if not os.path.isfile(ref_path):
        return None
    cur = meta.get("reference") if isinstance(meta.get("reference"), dict) else None
    media = cur is not None and cur.get("source") == "media"
    try:
        if media:  # cut from the media pool: describe the clip itself
            y, sr = read_audio(ref_path)
            an = pace.PartAnalysis(y, sr)
            a, b, part = 0, len(an.fdb), None
        elif cur and isinstance(cur.get("part"), int) and os.path.isfile(store._part_path(vdir, int(cur["part"]))):
            y, sr = read_audio(store._part_path(vdir, int(cur["part"])))
            an = pace.PartAnalysis(y, sr)
            a = int(float(cur["start_s"]) / pace.HOP_S)
            b = min(int(float(cur["end_s"]) / pace.HOP_S), len(an.fdb))
            part: int | None = int(cur["part"])
        else:  # reference.wav that no part explains (voices.import with a reference_path)
            y, sr = read_audio(ref_path)
            an = pace.PartAnalysis(y, sr)
            a, b, part = 0, len(an.fdb), None
        m = an.measure(a, b) if an.ok and b > a else {"start_s": 0.0, "end_s": len(y) / sr, "duration_s": len(y) / sr,
                                                      "rate": 0.0, "pauses": 0.0, "snr_db": 0.0}
    except Exception:  # noqa: BLE001 - a broken file must not break the screen
        return None
    if media:
        assert cur is not None
        m = {**m, "start_s": float(cur["start_s"]), "end_s": float(cur["end_s"])}
        out = _public({"source": "media", "file": cur.get("file"), "part": None, **m}, cur.get("title"), median)
    else:
        out = _public({"part": part, **m}, titles.get(part) if part else None, median)
    out["manual"] = bool(cur and cur.get("manual"))
    out["preview_path"] = ref_path
    return out


def _is_current(c: dict[str, Any], cur: dict[str, Any] | None) -> bool:
    return bool(cur and cur.get("source", "part") == c.get("source", "part") and cur.get("part") == c.get("part")
                and pace._overlap(cur, c) > 0.8)


def _media_candidates(vdir: str, meta: dict[str, Any], current: dict[str, Any] | None, pdir: str,
                      start: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """The best windows of the media pool (with preview clips) and a summary of the pool, or ``([], None)``."""
    from . import media_ref  # noqa: PLC0415

    mm = meta.get("media")
    if not isinstance(mm, dict) or not os.path.isfile(os.path.join(vdir, store.MEDIA_POOL)):
        return [], None
    code = mm.get("language") or None
    target = media_ref.target_name(code)
    in_use = None
    if code is None:  # reference.wav: the current clip, when it was cut from this pool
        ref = meta.get("reference") if isinstance(meta.get("reference"), dict) else None
        if current is not None and ref and ref.get("source") == "media" and ref.get("pool") == mm.get("pool"):
            in_use = {"source": "media", "part": None, "start_s": ref["start_s"], "end_s": ref["end_s"]}
    elif isinstance(mm.get("current"), dict) and os.path.isfile(os.path.join(vdir, target)):
        in_use = {"source": "media", "part": None, **mm["current"]}
    median = float(mm.get("rate_median") or 0.0)
    out = []
    for k, c in enumerate(media_ref.show_candidates(mm)):
        f = media_ref.file_at(mm, float(c["start_s"]), float(c["end_s"])) or {}
        clip, sr = store.cut_media(vdir, float(c["start_s"]), float(c["end_s"]))
        path = os.path.join(pdir, f"cand{start + k + 1}_media.wav")
        write_wav(path, clip, sr)
        item = _public({**c, "source": "media", "part": None, "file": f.get("index")}, f.get("name"), median)
        item.update(recommended=(k == 0), is_current=_is_current({**c, "source": "media", "part": None}, in_use),
                    preview_path=path)
        out.append(item)
    summary = {"files": len(mm.get("files") or []), "duration_s": mm.get("duration_s"), "language": code,
               "target": target, "created": mm.get("created"), "skipped": len(mm.get("skipped") or [])}
    return out, summary


@method("voices.reference_candidates")
def reference_candidates(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    vid, vdir = _vdir(ctx, params)
    with store._LOCK:
        meta = store._load_meta(vdir)
        if _ensure_meta(vdir, meta):
            store._save_meta(vdir, meta)
        ranked, median = store.rank_candidates(vdir, meta)
    titles = store._titles(vdir)
    chosen = pace.pick_distinct(ranked, N_CANDIDATES, MAX_PER_PART)
    current = _current(vdir, meta, titles, median)

    pdir = os.path.join(ctx.data_dir, "tmp", "refcand", vid)
    shutil.rmtree(pdir, ignore_errors=True)
    os.makedirs(pdir, exist_ok=True)
    audio: dict[int, tuple[np.ndarray, int]] = {}
    out = []
    for k, c in enumerate(chosen):
        n = int(c["part"])
        if n not in audio:
            audio[n] = read_audio(store._part_path(vdir, n))
        y, sr = audio[n]
        clip = store._fade(y[int(c["start_s"] * sr): int(c["end_s"] * sr)], sr)
        path = os.path.join(pdir, f"cand{k + 1}_part{n}.wav")
        write_wav(path, clip, sr)
        item = _public(c, titles.get(n), median)
        item.update(recommended=(k == 0), is_current=_is_current(c, current), preview_path=path)
        out.append(item)
    try:
        media, pool = _media_candidates(vdir, meta, current, pdir, len(out))
    except EngineError:
        media, pool = [], None
    from .. import reflang  # noqa: PLC0415

    # The detected language of reference.wav (None until the speech worker has transcribed this clip: the next
    # warm-up or generation does; its `voice.warm` event carries it).
    return {"candidates": out + media, "current": current, "rate_median": round(median, 2), "media": pool,
            "reference_language": reflang.detected(os.path.join(vdir, "reference.wav"))[0]}


@method("voices.set_reference")
def set_reference(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    from .. import tts  # noqa: PLC0415 - lazy: tts imports the voices module, which may import this package

    vid, vdir = _vdir(ctx, params)
    if params.get("source") == "media":
        return _set_from_media(ctx, vid, vdir, params)
    part = params.get("part")
    if isinstance(part, bool) or not isinstance(part, int) or part < 1:
        raise EngineError("bad_request", "'part' must be an integer starting at 1")
    nums = []
    for key in ("start_s", "end_s"):
        v = params.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise EngineError("bad_request", f"'{key}' must be a number of seconds")
        nums.append(float(v))
    start_s, end_s = nums
    if not os.path.isfile(store._part_path(vdir, part)):
        raise EngineError("not_found", f"part {part} does not exist")
    if tts.is_busy():
        raise EngineError("busy", tts.busy_message("a generation is running; change the reference clip when it has finished"))
    with store._LOCK:
        meta = store._load_meta(vdir)
        y, sr = read_audio(store._part_path(vdir, part))
        total = len(y) / sr
        start_s, end_s = max(0.0, start_s), min(total, end_s)
        if not SET_MIN_S <= end_s - start_s <= SET_MAX_S:
            raise EngineError("bad_request", f"the clip must be {SET_MIN_S:g}-{SET_MAX_S:g} s long")
        an = pace.PartAnalysis(y, sr)
        a, b = int(start_s / pace.HOP_S), min(int(end_s / pace.HOP_S), len(an.fdb))
        m = an.measure(a, b) if an.ok and b > a else {"rate": 0.0, "pauses": 0.0}
        store.write_reference(vdir, meta, part, start_s, end_s, manual=True,
                              rate=round(m["rate"], 3), pauses=round(m["pauses"], 3))
        store._save_meta(vdir, meta)
        ref = dict(meta["reference"])
    tts.forget_reference()
    return {"reference": ref, "warming": _warm(ctx, vid)}


def _warm(ctx: Any, vid: str) -> bool:
    """The existing warm path: conditioning (and the RVC model) are prepared in the background."""
    try:
        from ..registry import methods  # noqa: PLC0415

        warm = methods().get("voices.warm")
        if warm is not None:
            return bool(warm(ctx, {"id": vid}).get("warming"))
    except EngineError:
        pass  # not ready / busy: the next generation conditions on the new clip itself
    return False


def _times(params: dict[str, Any]) -> tuple[float, float]:
    nums = []
    for key in ("start_s", "end_s"):
        v = params.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise EngineError("bad_request", f"'{key}' must be a number of seconds")
        nums.append(float(v))
    return nums[0], nums[1]


def _set_from_media(ctx: Any, vid: str, vdir: str, params: dict[str, Any]) -> dict[str, Any]:
    """``set_reference`` with ``source: "media"``: a window of the media pool becomes the pool's target."""
    from .. import tts  # noqa: PLC0415
    from . import media_ref  # noqa: PLC0415

    start_s, end_s = _times(params)
    pool = os.path.join(vdir, store.MEDIA_POOL)
    if tts.is_busy():
        raise EngineError("busy", tts.busy_message("a generation is running; change the reference clip when it has finished"))
    with store._LOCK:
        meta = store._load_meta(vdir)
        mm = meta.get("media")
        if not isinstance(mm, dict) or not os.path.isfile(pool):
            raise EngineError("not_found", "this voice has no combined media; add media files first")
        total = float(sf.info(pool).duration)
        start_s, end_s = max(0.0, start_s), min(total, end_s)
        if not SET_MIN_S <= end_s - start_s <= SET_MAX_S:
            raise EngineError("bad_request", f"the clip must be {SET_MIN_S:g}-{SET_MAX_S:g} s long")
        code = mm.get("language") or None
        clip, sr = store.cut_media(vdir, start_s, end_s)
        an = pace.PartAnalysis(clip, sr)
        m = an.measure(0, len(an.fdb)) if an.ok else {"rate": 0.0, "pauses": 0.0, "snr_db": 0.0}
        f = media_ref.file_at(mm, start_s, end_s) or {}
        info = {"file": f.get("index"), "title": f.get("name"), "rate": round(m["rate"], 3),
                "pauses": round(m["pauses"], 3), "snr_db": round(m["snr_db"], 2)}
        ref = media_ref.write_target(vdir, meta, start_s, end_s, code, info)
        store._save_meta(vdir, meta)
    tts.forget_reference()
    return {"reference": ref, "target": media_ref.target_name(code), "language": code,
            "warming": _warm(ctx, vid) if code is None else False}


# ------------------------------------------------------------------------- per-language references (reference_<lang>)

LANG_REF_MIN_S, LANG_REF_MAX_S = 3.0, 40.0


def _speech_language(value: Any) -> str:
    """A language the speech model speaks (``bad_request`` otherwise)."""
    from .. import backends  # noqa: PLC0415
    from ..redub import langs  # noqa: PLC0415

    code = langs.norm(value) if isinstance(value, str) else ""
    if not code or backends.for_language(code) is None:
        raise EngineError("bad_request", f"'language' must be a language the voice can speak "
                                         f"({', '.join(sorted(backends.languages()))})")
    return code


def language_references(vdir: str) -> list[str]:
    """Languages that have a ``reference_<lang>.wav`` in the voice folder, sorted."""
    try:
        names = os.listdir(vdir)
    except OSError:
        return []
    out = []
    for n in names:
        if n.startswith("reference_") and n.endswith(".wav") and not n.endswith(".tmp.wav"):
            code = n[len("reference_"):-len(".wav")]  # (reference_media.wav is the media pool, not a language)
            if code.isascii() and code.isalpha() and code.islower() and 2 <= len(code) <= 3:
                out.append(code)
    return sorted(out)


@method("voices.add_reference")
def add_reference(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Store a recording as ``reference_<language>.wav`` (trimmed at both ends, normalised): the speech model uses it
    instead of ``reference.wav`` when it speaks that language (accent and prosody of the clip carry over)."""
    from .. import tts  # noqa: PLC0415
    from . import quality  # noqa: PLC0415

    vid, vdir = _vdir(ctx, params)
    code = _speech_language(params.get("language"))
    path = params.get("path")
    if not isinstance(path, str) or not path:
        raise EngineError("bad_request", "'path' must be an audio file path")
    if tts.is_busy():
        raise EngineError("busy", tts.busy_message("a generation is running; add the reference when it has finished"))
    y, sr = read_audio(os.path.abspath(path))
    if len(y) == 0:
        raise EngineError("bad_request", "The audio file is empty.")
    if sr > store.MAX_RATE:
        y, sr = store.resample(y, sr, 44100), 44100
    y = y - float(np.mean(y))  # DC offset
    trimmed = store.trim_silence(y, sr, quality.measure(y, sr))
    clip = store.normalise(trimmed, quality.measure(trimmed, sr))
    duration = len(clip) / float(sr)
    if duration < LANG_REF_MIN_S:
        raise EngineError("bad_request", f"The clip has only {duration:.1f} s of speech; record at least "
                                         f"{LANG_REF_MIN_S:g} s (about 10 s is best).")
    if duration > LANG_REF_MAX_S:
        raise EngineError("bad_request", f"The clip is {duration:.0f} s long; use at most {LANG_REF_MAX_S:g} s "
                                         "(about 10 s is best).")
    write_wav(os.path.join(vdir, f"reference_{code}.wav"), store._fade(clip, sr), sr)
    tts.forget_reference()
    return {"language": code, "duration_s": round(duration, 2), "references": language_references(vdir)}


@method("voices.remove_reference")
def remove_reference(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    from .. import tts  # noqa: PLC0415

    _vid, vdir = _vdir(ctx, params)
    code = _speech_language(params.get("language"))
    path = os.path.join(vdir, f"reference_{code}.wav")
    if not os.path.isfile(path):
        raise EngineError("not_found", f"this voice has no reference for {code!r}")
    if tts.is_busy():
        raise EngineError("busy", tts.busy_message("a generation is running; remove the reference when it has finished"))
    os.remove(path)
    sidecar = os.path.join(vdir, f"reference_{code}.omnivoice.json")
    if os.path.isfile(sidecar):
        os.remove(sidecar)
    tts.forget_reference()
    return {"references": language_references(vdir)}
