"""Audiobook jobs. Each segment is appended to the chapter PCM, then recorded in state.json; a resume truncates the PCM
to the recorded length, so a crash between the two writes neither loses nor repeats audio."""

from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import accent, aio, backends, config, media, narration, paths, reflang, rvc, speech, tts, voices, zsvc
from . import audio_util as au
from .backends import SynthParams
from .registry import Context, EngineError, method
from .workerproc import WorkerDied

log = logging.getLogger("mimvox.book")

SR = au.OUT_SR
BYTES_PER_S = SR * 2
QUALITY_CHECKS = True
ENCODE_QUALITY = "standard"
M4B_BITRATE = "96k"
MAX_CHARS = 20_000_000
WAV_LIMIT = 0xFFFFFFFF - 1024  # a RIFF WAV cannot pass 4 GB: RF64 above that
HOLDER = "a book is being read; pause it (Speak, Book) to use speech, Redub or training"
MEMORY_RETRY_WAIT_S = (20.0, 60.0, 180.0, 300.0)

_RUNNING: dict[str, str] = {}  # loop thread only
_STOP: dict[str, str] = {}
_STARTING: set[str] = set()  # loop thread only: refs whose book.start is under way
_REREADING: set[str] = set()  # loop thread only
# Changed on IO threads (the producer cuts the next chapter while the consumer appends to this one) and read on the
# loop: every change and save holds this lock.
_LOCK = threading.RLock()


def books_dir(data: config.DataDir) -> Path:
    return data.root / "books"


def book_dir(data: config.DataDir, book_id: str) -> Path:
    if not isinstance(book_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", book_id):
        raise EngineError("bad_request", f"invalid book id {book_id!r}")
    return books_dir(data) / book_id


def _read_json(path: str | Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None


def _write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=paths.json_default)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(5):  # Windows: a reader (an antivirus) may hold the target for a moment
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _load(data: config.DataDir, book_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    d = book_dir(data, book_id)
    book = _read_json(d / "book.json")
    if book is None:
        raise EngineError("not_found", f"book {book_id!r} does not exist")
    state = _read_json(d / "state.json") or _new_state(book)
    n = len(book.get("chapters") or [])
    chs = state.setdefault("chapters", [])
    while len(chs) < n:
        chs.append(_new_chapter_state())
    return book, state


def _new_chapter_state() -> dict[str, Any]:
    return {
        "status": "waiting",
        "segment": 0,
        "segments": 0,
        "bytes": 0,
        "chars_done": 0,
        "audio_s": 0.0,
        "elapsed_s": 0.0,
        "path": None,
    }


def _new_state(book: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "new",
        "current": 0,
        "audio_s": 0.0,
        "elapsed_s": 0.0,
        "chapters": [_new_chapter_state() for _ in book.get("chapters") or []],
        "m4b": None,
        "error": None,
        "updated": time.time(),
    }


def _save_state(data: config.DataDir, book_id: str, state: dict[str, Any]) -> None:
    with _LOCK:
        state["updated"] = time.time()
        _write_json(book_dir(data, book_id) / "state.json", state)


_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(name: str, limit: int = 80) -> str:
    """No characters Windows forbids, no trailing dots or spaces, not a reserved device name."""
    s = re.sub(r"\s+", " ", _BAD_NAME.sub(" ", name or "")).strip(" .")
    s = s[:limit].rstrip(" .")
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com\d|lpt\d)(\..*)?", s):
        s = f"_{s}"
    return s


def _work(data: config.DataDir, book_id: str) -> Path:
    return book_dir(data, book_id) / "work"


def _pcm_path(data: config.DataDir, book_id: str, k: int) -> Path:
    return _work(data, book_id) / f"{k + 1:03d}.pcm"


def _segs_path(data: config.DataDir, book_id: str, k: int) -> Path:
    return _work(data, book_id) / f"{k + 1:03d}.segments.json"


def _timing_path(data: config.DataDir, book_id: str, k: int) -> Path:
    return book_dir(data, book_id) / "segments" / f"{k + 1:03d}.json"


# Cyrillic and Greek letters typed in place of Latin ones in Roman numerals ("РОЗДІЛ І" for "РОЗДІЛ I").
_ROMAN_LOOKALIKE = str.maketrans("\u0456\u0445\u0441\u043c\u03b9\u03c7\u03bc", "ixcmixm")
_ROMAN = re.compile(r"m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _heading_token(token: str) -> str:
    if token.isdigit():
        return str(int(token))
    r = token.translate(_ROMAN_LOOKALIKE)
    if _ROMAN.fullmatch(r):
        v = [_ROMAN_VALUES[ch] for ch in r]
        return str(sum(-a if a < b else a for a, b in zip(v, [*v[1:], 0])))
    return token


def _heading_key(s: str) -> str:
    """Case, punctuation, compatibility forms (Ⅻ, ｉ) and how a number is written (1, I, Ⅰ) do not count."""
    return " ".join(_heading_token(t) for t in re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", s).casefold()))


def heading_in_text(title: str, text: str) -> bool:
    """The chapter's text opens with its own heading (often after the book's title and author)."""
    key = _heading_key(title)
    if not key:
        return False
    for para in re.split(r"\n\s*\n|\n", text.strip())[:5]:
        if len(para) > 80:
            return False
        if _heading_key(para) == key:
            return True
    return False


def spoken_text(chapter: dict[str, Any], read_titles: bool) -> str:
    title = re.sub(r"\s+", " ", str(chapter.get("title") or "")).strip()
    text = str(chapter.get("text") or "").strip()
    if not read_titles or not title or heading_in_text(title, text):
        return text
    if not re.search(r"[.!?…:;]$", title):
        title += "."
    return f"{title}\n\n{text}" if text else title


def _flat_chars(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


def _status(book_id: str, state: dict[str, Any]) -> tuple[str, bool]:
    """A book recorded as running whose job is not (the engine stopped mid-book) is paused and interrupted."""
    st = str(state.get("status") or "new")
    if st == "running" and book_id not in _RUNNING:
        return "paused", True
    return st, False


def _totals(book: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    chapters = book.get("chapters") or []
    chs = state.get("chapters") or []
    total = sum(int(c.get("chars") or 0) for c in chapters) or 1
    done_chars = 0
    for i, c in enumerate(chapters):
        cs = chs[i] if i < len(chs) else {}
        done_chars += int(c.get("chars") or 0) if cs.get("status") == "done" else int(cs.get("chars_done") or 0)
    audio = float(state.get("audio_s") or 0.0)
    elapsed = float(state.get("elapsed_s") or 0.0)
    eta = None
    if done_chars > 0 and elapsed > 0:
        eta = round((total - done_chars) * elapsed / done_chars, 1)
    est_audio = audio * total / done_chars if done_chars > 0 and audio > 0 else None
    return {
        "percent": round(100.0 * min(done_chars, total) / total, 2),
        "audio_s": round(audio, 2),
        "elapsed_s": round(elapsed, 1),
        "rtf": round(elapsed / audio, 3) if audio > 0 else None,
        "eta_s": eta,
        "est_audio_s": round(est_audio, 1) if est_audio else None,
        "chars": total,
        "chars_done": done_chars,
    }


def summary(data: config.DataDir, book: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    bid = str(book["id"])
    status, interrupted = _status(bid, state)
    chs = state.get("chapters") or []
    out = {
        "id": bid,
        "title": book.get("title"),
        "source": book.get("source"),
        "format": book.get("format"),
        "language": (book.get("settings") or {}).get("language"),
        "voice_id": (book.get("settings") or {}).get("voice_id"),
        "status": status,
        "interrupted": interrupted,
        "job": _RUNNING.get(bid),
        "chapters": len(book.get("chapters") or []),
        "chapters_done": sum(1 for c in chs if c.get("status") == "done"),
        "current": int(state.get("current") or 0),
        "folder": book.get("folder"),
        "m4b": state.get("m4b"),
        "output_format": (book.get("settings") or {}).get("output_format"),
        "words": book.get("words"),
        "ref": book.get("ref"),
        "created": book.get("created"),
        "updated": state.get("updated"),
        "error": state.get("error"),
    }
    out.update(_totals(book, state))
    return out


def _chapter_rows(book: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    chs = state.get("chapters") or []
    for i, c in enumerate(book.get("chapters") or []):
        cs = chs[i] if i < len(chs) else _new_chapter_state()
        audio, el = float(cs.get("audio_s") or 0.0), float(cs.get("elapsed_s") or 0.0)
        rows.append(
            {
                "index": i,
                "title": c.get("title"),
                "words": c.get("words"),
                "chars": c.get("chars"),
                "status": cs.get("status") or "waiting",
                "segment": int(cs.get("segment") or 0),
                "segments": int(cs.get("segments") or 0),
                "audio_s": round(audio, 2),
                "elapsed_s": round(el, 1),
                "rtf": round(el / audio, 3) if audio > 0 else None,
                "path": cs.get("path"),
                "segments_path": cs.get("segments_path"),
            }
        )
    return rows


@dataclass
class _Voice:
    backend: backends.TtsBackend
    lang: str
    ref: str | Path
    pth: str | Path
    idx: str | Path
    rvc_rate: float
    worker: rvc.RvcWorker | None
    choice: accent.Choice
    zs: zsvc.Bound | None
    voice_match: float
    temperature: float
    quality: str
    duration_factor: float
    out_fmt: str
    ref_fields: dict[str, Any]


def _settings(params: dict[str, Any]) -> dict[str, Any]:
    voice_id = params.get("voice_id")
    if not isinstance(voice_id, str) or not voice_id:
        raise EngineError("bad_request", "'voice_id' is required")
    quality = params.get("quality") or "best"
    if quality not in ("best", "fast"):
        raise EngineError("bad_request", '\'quality\' must be "best" or "fast"')
    lang, _ = tts.speech_language(params)
    m4b = params.get("m4b", False)
    titles = params.get("read_titles", True)
    keep_voice = params.get("keep_voice", False)
    if not isinstance(m4b, bool) or not isinstance(titles, bool) or not isinstance(keep_voice, bool):
        raise EngineError("bad_request", "'m4b', 'read_titles' and 'keep_voice' must be true or false")
    return {
        "voice_id": voice_id,
        "language": lang,
        "exact_stress": params.get("exact_stress", True),
        "model": params.get("model") or "auto",
        "expressiveness": tts._num(params, "expressiveness", narration.EXPRESSIVENESS, 0.1, 1.5),
        "voice_match": tts._num(params, "voice_match", 1.0, 0.0, 1.0),
        "speed": tts._num(params, "speed", 1.0, tts.SPEED_MIN, tts.SPEED_MAX),
        "quality": quality,
        "accent": accent.parse(params),
        "native_voice": accent.parse_voice(params),
        "output_format": params.get("output_format") or "wav",
        "m4b": m4b,
        "read_titles": titles,
        "keep_voice": keep_voice,
    }


async def _voice(ctx: Context, data: config.DataDir, st: dict[str, Any]) -> _Voice:
    lang, backend = tts.speech_language(st)
    tts.check_backend_ready(backend)
    voice_id = st["voice_id"]
    meta = await aio.run_io(voices.read_meta, data, voice_id)
    out_fmt = await aio.run_io(media.output_format, st)
    voice_match = float(st["voice_match"])
    rvc_rate = rvc.index_rate_for(voice_match, lang, meta.get("language"))
    ref = data.reference(voice_id)
    pth, idx = data.rvc_model(voice_id)
    has_ref, has_pth = await aio.run_io(lambda: (ref.is_file(), pth.is_file()))
    if not has_ref:
        raise EngineError("not_found", "this voice has no recording yet (reference.wav is missing)")
    use_rvc = has_pth and voice_match > 0
    worker = rvc.get(ctx.runtime) if use_rvc else None
    if use_rvc and (worker is None or tts.models_status()["rvc"] != "ready"):
        raise EngineError("not_ready", "the voice-conversion model is not ready")
    requested = st.get("accent") or "own"
    zs_worker = await aio.run_io(zsvc.get, ctx.runtime, data.root) if requested == "native-us" else None
    choice = accent.choose(
        requested, st.get("native_voice") or "auto", lang, has_pth, voice_match, zs_worker is not None
    )
    zs = tts.zero_shot(ctx, choice, zs_worker, ref, st["quality"])
    if zs is None and not use_rvc and st.get("keep_voice", False):
        zs = await tts.holos_timbre(ctx, backend, data, ref, st["quality"])
    ref_fields = await aio.run_io(reflang.fields, ref, lang, choice.accent == "native-us")
    return _Voice(
        backend,
        lang,
        ref,
        pth,
        idx,
        rvc_rate,
        worker,
        choice,
        zs,
        voice_match,
        float(st["expressiveness"]),
        st["quality"],
        tts.duration_factor_for(st["speed"]),
        out_fmt,
        ref_fields,
    )


def _clean_chapters(raw: Any) -> list[dict[str, Any]]:
    from .dataset.text import count_words

    if not isinstance(raw, list) or not raw:
        raise EngineError("bad_request", '\'chapters\' must be a non-empty list of {"title", "text"}')
    out, total = [], 0
    for i, c in enumerate(raw):
        if not isinstance(c, dict) or not isinstance(c.get("text", ""), str) or not isinstance(c.get("title", ""), str):
            raise EngineError("bad_request", f"chapter {i + 1}: need a string 'title' and 'text'")
        text = str(c.get("text") or "").strip()
        title = re.sub(r"\s+", " ", str(c.get("title") or "")).strip() or f"Chapter {i + 1}"
        if not count_words(text):
            continue
        total += len(text)
        out.append({"title": title, "text": text, "words": count_words(text)})
    if not out:
        raise EngineError("bad_request", "nothing to read: every chapter is empty")
    if total > MAX_CHARS:
        raise EngineError("bad_request", f"the book is too long ({total:,} characters; at most {MAX_CHARS:,})")
    return out


def _create(
    data: config.DataDir,
    title: str,
    source: str | None,
    fmt: str | None,
    chapters: list[dict[str, Any]],
    settings: dict[str, Any],
    ref: str | None = None,
) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{voices.slugify(title)[:32].strip('-') or 'book'}-{stamp}"
    bid, n = base, 1
    while (books_dir(data) / bid).exists():
        n += 1
        bid = f"{base}-{n}"
    name = safe_name(title) or "Book"
    root = data.outputs / "books"
    folder, n = root / name, 1
    while folder.exists() or any(_folder_taken(data, folder)):
        n += 1
        folder = root / f"{name} ({n})"
    width = max(2, len(str(len(chapters))))
    for i, c in enumerate(chapters):
        c["file"] = f"{i + 1:0{width}d} - {safe_name(c['title'], 60) or 'Chapter'}"
        c["chars"] = _flat_chars(spoken_text(c, settings["read_titles"]))
    book = {
        "id": bid,
        "title": title,
        "source": source,
        "format": fmt,
        "created": time.time(),
        "folder": str(folder),
        "settings": settings,
        "words": sum(int(c["words"]) for c in chapters),
        "chapters": chapters,
    }
    if ref is not None:
        book["ref"] = ref
    d = books_dir(data) / bid
    (d / "work").mkdir(parents=True, exist_ok=True)
    _write_json(d / "book.json", book)
    _save_state(data, bid, _new_state(book))
    return book


def _folder_taken(data: config.DataDir, folder: Path) -> list[bool]:
    out = []
    try:
        dirs = list(books_dir(data).iterdir())
    except OSError:
        return out
    for d in dirs:
        b = _read_json(d / "book.json")
        out.append(bool(b) and os.path.normcase(str(b.get("folder"))) == os.path.normcase(folder))
    return out


def find_ref(data: config.DataDir, ref: str) -> str | None:
    """The id of the oldest project whose book.json has this ref."""
    try:
        dirs = list(books_dir(data).iterdir())
    except OSError:
        return None
    found = []
    for d in dirs:
        b = _read_json(d / "book.json")
        if b and b.get("ref") == ref and b.get("id") == d.name:
            found.append((float(b.get("created") or 0.0), d.name))
    return min(found)[1] if found else None


@method("book.start")
async def book_start(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    ref = params.get("ref")
    if ref is None:
        return await _start_new(ctx, data, params, None)
    if not isinstance(ref, str) or not ref.strip():
        raise EngineError("bad_request", "'ref' must be a non-empty string")
    if ref in _STARTING:
        raise EngineError("busy", "a book with this ref is being started")
    _STARTING.add(ref)
    try:
        bid = await aio.run_io(find_ref, data, ref)
        if bid is None:
            return await _start_new(ctx, data, params, ref)
        running = bid in _RUNNING
        book, state = await aio.run_io(_load, data, bid)
        if running and bid not in _RUNNING:  # the job ended during the read, which may hold its state from before
            book, state = await aio.run_io(_load, data, bid)
        out = {
            "id": bid,
            "folder": book["folder"],
            "chapters": len(book["chapters"]),
            "words": book.get("words"),
            "ref": ref,
            "existing": True,
        }
        if bid in _RUNNING:
            return {**out, "job": _RUNNING[bid], "status": "running"}
        if state.get("status") == "done":
            return {**out, "job": None, "status": "done"}
        return {**out, **await _resume(ctx, data, bid, book), "status": "running"}
    finally:
        _STARTING.discard(ref)


async def _start_new(ctx: Context, data: config.DataDir, params: dict[str, Any], ref: str | None) -> dict[str, Any]:
    settings = _settings(params)
    chapters = _clean_chapters(params.get("chapters"))
    title = re.sub(r"\s+", " ", str(params.get("title") or "")).strip() or "Book"
    v = await _voice(ctx, data, settings)
    if tts.is_busy():
        raise EngineError("busy", tts.busy_message("another generation is running"))
    settings["output_format"] = v.out_fmt
    source = params.get("source") if isinstance(params.get("source"), str) else None
    fmt = params.get("format") if isinstance(params.get("format"), str) else None
    book = await aio.run_io(_create, data, title, source, fmt, chapters, settings, ref)
    try:
        job = _start(ctx, data, book["id"], v)
    except EngineError:  # another job took the slot while the project was written
        await aio.run_io(shutil.rmtree, book_dir(data, book["id"]), ignore_errors=True)
        raise
    return {
        "id": book["id"],
        "job": job,
        "folder": book["folder"],
        "chapters": len(chapters),
        "words": book["words"],
        "ref": ref,
        "existing": False,
        "status": "running",
        **v.choice.fields(),
        **v.ref_fields,
    }


@method("book.resume")
async def book_resume(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    book, state = await aio.run_io(_load, data, bid)
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read already")
    if state.get("status") == "done":
        raise EngineError("bad_request", "this book is finished")
    return {
        "id": bid,
        "folder": book["folder"],
        "chapters": len(book["chapters"]),
        **await _resume(ctx, data, bid, book),
    }


async def _resume(ctx: Context, data: config.DataDir, bid: str, book: dict[str, Any]) -> dict[str, Any]:
    v = await _voice(ctx, data, book["settings"])
    return {"job": _start(ctx, data, bid, v), **v.choice.fields(), **v.ref_fields}


def _start(ctx: Context, data: config.DataDir, bid: str, v: _Voice) -> str:
    """Loop thread, with no await since the busy checks."""
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read already")
    if bid in _REREADING:
        raise EngineError("busy", "this book is being reread")
    job = tts.start_exclusive(ctx, lambda j: _run(ctx, j, data, bid, v), "another generation is running", HOLDER)
    _RUNNING[bid] = job
    _STOP.pop(bid, None)
    return job


@method("book.pause")
async def book_pause(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    bid = str(params.get("id") or "")
    job = _RUNNING.get(bid)
    if job is None:
        raise EngineError("bad_request", "this book is not being read")
    _STOP[bid] = "pause"
    ctx.cancel(job)
    return {}


@method("book.cancel")
async def book_cancel(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    job = _RUNNING.get(bid)
    if job is not None:
        _STOP[bid] = "cancel"
        ctx.cancel(job)
        return {}
    _book, state = await aio.run_io(_load, data, bid)
    if state.get("status") != "done":
        await aio.run_io(_drop_partial, data, bid, state, "cancelled")
    return {}


def _drop_partial(data: config.DataDir, bid: str, state: dict[str, Any], status: str) -> None:
    for k, cs in enumerate(state.get("chapters") or []):
        if cs.get("status") != "done":
            cs.clear()
            cs.update(_new_chapter_state())
            with contextlib.suppress(OSError):
                _timing_path(data, bid, k).unlink()
    shutil.rmtree(_work(data, bid), ignore_errors=True)
    _work(data, bid).mkdir(parents=True, exist_ok=True)
    state["status"] = status
    _save_state(data, bid, state)


@method("book.list")
async def book_list(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)

    def scan() -> list[dict[str, Any]]:
        out = []
        try:
            names = sorted(p.name for p in books_dir(data).iterdir())
        except FileNotFoundError:
            return out
        for n in names:
            try:
                book, state = _load(data, n)
            except EngineError:
                continue
            out.append(summary(data, book, state))
        out.sort(key=lambda b: -float(b.get("updated") or 0.0))
        return out

    return {"books": await aio.run_io(scan)}


@method("book.get")
async def book_get(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    book, state = await aio.run_io(_load, data, bid)
    rows = _chapter_rows(book, state)
    if params.get("text") is True:
        for r, c in zip(rows, book["chapters"]):
            r["text"] = c.get("text")
    return {**summary(data, book, state), "settings": book.get("settings"), "chapter_list": rows}


@method("book.delete")
async def book_delete(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    d = book_dir(data, bid)
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read; pause or cancel it first")
    if not await aio.run_io(d.is_dir):
        raise EngineError("not_found", f"book {bid!r} does not exist")
    await aio.run_io(shutil.rmtree, d, ignore_errors=True)
    return {}


@method("book.reread")
async def book_reread(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read; pause it first")
    if bid in _REREADING:
        raise EngineError("busy", "this book is being reread")
    _REREADING.add(bid)
    try:
        return await _reread_checked(data, bid, params)
    finally:
        _REREADING.discard(bid)


async def _reread_checked(data: config.DataDir, bid: str, params: dict[str, Any]) -> dict[str, Any]:
    book, state = await aio.run_io(_load, data, bid)
    n = len(book["chapters"])
    raw = params.get("chapters")
    if raw is None:
        chapters = list(range(n))
    elif isinstance(raw, list) and all(isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n for k in raw):
        chapters = sorted(set(raw))
    else:
        raise EngineError("bad_request", f"'chapters' must be chapter indexes from 0 to {n - 1}")
    voice_id = params.get("voice_id")
    if voice_id is not None:
        if not isinstance(voice_id, str) or not voice_id:
            raise EngineError("bad_request", "'voice_id' must be a voice id")
        await aio.run_io(voices.read_meta, data, voice_id)
        if not await aio.run_io(data.reference(voice_id).is_file):
            raise EngineError("not_found", "this voice has no recording yet (reference.wav is missing)")
    await aio.run_io(_reread, data, bid, book, state, voice_id, chapters)
    return {**summary(data, book, state), "settings": book.get("settings"), "chapter_list": _chapter_rows(book, state)}


def _reread(
    data: config.DataDir,
    bid: str,
    book: dict[str, Any],
    state: dict[str, Any],
    voice_id: str | None,
    chapters: list[int],
) -> None:
    with _LOCK:
        if voice_id is not None and voice_id != book["settings"].get("voice_id"):
            book["settings"]["voice_id"] = voice_id
            _write_json(book_dir(data, bid) / "book.json", book)
        for k in chapters:
            cs = state["chapters"][k]
            state["audio_s"] = max(0.0, float(state.get("audio_s") or 0.0) - float(cs.get("audio_s") or 0.0))
            state["elapsed_s"] = max(0.0, float(state.get("elapsed_s") or 0.0) - float(cs.get("elapsed_s") or 0.0))
            old = [
                cs.get("path"),
                _pcm_path(data, bid, k),
                _segs_path(data, bid, k),
                _timing_path(data, bid, k),
                book_dir(data, bid) / "stats" / f"{k + 1:03d}.json",
            ]
            for path in filter(None, old):
                with contextlib.suppress(OSError):
                    Path(path).unlink()
            cs.clear()
            cs.update(_new_chapter_state())
        if chapters and state.get("m4b"):
            with contextlib.suppress(OSError):
                Path(state["m4b"]).unlink()
            state["m4b"] = None
        chs = state["chapters"]
        if not all(cs.get("status") == "done" for cs in chs):
            started = any(cs.get("status") in ("done", "partial") for cs in chs)
            state["status"] = "paused" if started else "new"
            state["error"] = None
        state["current"] = next((k for k, cs in enumerate(chs) if cs.get("status") != "done"), 0)
        _save_state(data, bid, state)


@method("book.play")
async def book_play(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """The app plays WAV only: another format is decoded to ``<data>/tmp/play``."""
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    k = params.get("index")
    if isinstance(k, bool) or not isinstance(k, int):
        raise EngineError("bad_request", "'index' must be a chapter index")
    _book, state = await aio.run_io(_load, data, bid)
    chs = state.get("chapters") or []
    if not 0 <= k < len(chs) or chs[k].get("status") != "done" or not chs[k].get("path"):
        raise EngineError("not_found", "this chapter is not finished yet")
    path = str(chs[k]["path"])
    return {"path": path, "path_play": await aio.run_io(_playable, data, bid, k, path)}


def _playable(data: config.DataDir, bid: str, k: int, path: str | Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise EngineError("not_found", "the chapter file is gone")
    if path.name.lower().endswith(".wav"):
        return path
    dst = data.tmp / media.PLAY_DIR / f"book-{bid}-{k + 1:03d}.wav"
    if dst.is_file() and dst.stat().st_mtime >= path.stat().st_mtime:
        return dst
    return Path(media.play_wav(path, dst, SR, channels=1))


def _pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def _prepare_chapter(
    data: config.DataDir, bid: str, k: int, book: dict[str, Any], state: dict[str, Any], count: Any
) -> list[tuple[str, int, str]]:
    with _LOCK:
        return _prepare_chapter_locked(data, bid, k, book, state, count)


def _prepare_chapter_locked(
    data: config.DataDir, bid: str, k: int, book: dict[str, Any], state: dict[str, Any], count: Any
) -> list[tuple[str, int, str]]:
    path = _segs_path(data, bid, k)
    cs = state["chapters"][k]
    saved = _read_json(path)
    segs: list[tuple[str, int, str]] | None = None
    if saved and cs.get("status") == "partial":
        segs = [(str(t), int(p), str(kind)) for t, p, kind in saved.get("segments") or []]
    if not segs:
        ch = book["chapters"][k]
        text = str(ch.get("text") or "")
        title = ch.get("title") if book["settings"].get("read_titles", True) else None
        if title and heading_in_text(str(title), text):
            title = None
        segs = narration.segments(title, text, count)
        if not segs:
            segs = [(".", narration.PAUSES["end"], "end")]
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"segments": [list(s) for s in segs]})
        cs.clear()
        cs.update(_new_chapter_state())
        cs["status"] = "partial"
        cs["segments"] = len(segs)
        for p in (_pcm_path(data, bid, k), _timing_path(data, bid, k)):
            with contextlib.suppress(OSError):
                p.unlink()
    pcm = _pcm_path(data, bid, k)
    keep = int(cs.get("bytes") or 0)
    if pcm.exists() or keep:
        with pcm.open("ab"):  # creates it when missing
            pass
        size = pcm.stat().st_size
        if size < keep:
            log.warning("book %s chapter %d: %d bytes on disk, %d recorded; starting it again", bid, k + 1, size, keep)
            cs.update({"segment": 0, "bytes": 0, "chars_done": 0, "audio_s": 0.0, "stats": []})
            keep = 0
        if size != keep:
            with pcm.open("r+b") as fh:
                fh.truncate(keep)
    cs["status"] = "partial"
    cs["segments"] = len(segs)
    _save_state(data, bid, state)
    return segs


def _segment_audio(src: str | Path, chars: int) -> tuple[np.ndarray, dict[str, Any]]:
    x, sr = au.read_mono(src)
    x = narration.trim(au.resample(x, sr, SR), SR)
    return x, narration.measure(x, SR, chars)


def _commit(
    data: config.DataDir,
    bid: str,
    state: dict[str, Any],
    k: int,
    i: int,
    x: np.ndarray,
    pause_ms: int,
    chars: int,
    dt: float,
    stat: dict[str, Any],
) -> float:
    parts = [au.silence(narration.LEAD_MS)] if i == 0 else []
    parts += [x, au.silence(narration.gap(pause_ms))]
    data_bytes = _pcm(np.concatenate(parts))
    pcm = _pcm_path(data, bid, k)
    with _LOCK:
        return _append(data, bid, state, k, i, pcm, data_bytes, chars, dt, stat)


def _append(
    data: config.DataDir,
    bid: str,
    state: dict[str, Any],
    k: int,
    i: int,
    pcm: Path,
    data_bytes: bytes,
    chars: int,
    dt: float,
    stat: dict[str, Any],
) -> float:
    cs = state["chapters"][k]
    stats = cs.setdefault("stats", [])
    del stats[i:]
    lead = narration.LEAD_MS / 1000.0 if i == 0 else 0.0
    stats.append({**stat, "at": round(int(cs.get("bytes") or 0) / BYTES_PER_S + lead, 3)})
    with pcm.open("ab") as fh:
        if fh.tell() != int(cs.get("bytes") or 0):
            fh.truncate(int(cs.get("bytes") or 0))
            fh.seek(int(cs.get("bytes") or 0))
        fh.write(data_bytes)
        fh.flush()
        os.fsync(fh.fileno())
        size = fh.tell()
    added = len(data_bytes) / BYTES_PER_S
    cs.update(
        {
            "segment": i + 1,
            "bytes": size,
            "chars_done": int(cs.get("chars_done") or 0) + chars,
            "audio_s": round(float(cs.get("audio_s") or 0.0) + added, 3),
            "elapsed_s": round(float(cs.get("elapsed_s") or 0.0) + dt, 3),
        }
    )
    state["audio_s"] = round(float(state.get("audio_s") or 0.0) + added, 3)
    state["elapsed_s"] = round(float(state.get("elapsed_s") or 0.0) + dt, 3)
    state["current"] = k
    _save_state(data, bid, state)
    return added


def _finish_chapter(
    data: config.DataDir, bid: str, book: dict[str, Any], state: dict[str, Any], k: int, fmt: str
) -> Path:
    import soundfile as sf

    pcm = _pcm_path(data, bid, k)
    folder = Path(book["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    stem = book["chapters"][k]["file"]
    wav = folder / f"{stem}.wav" if fmt == "wav" else _work(data, bid) / f"{k + 1:03d}.wav"
    part = wav.with_name(wav.name + ".part")
    size = pcm.stat().st_size
    block_bytes = 1 << 22

    def blocks() -> Iterator[bytes]:
        with pcm.open("rb") as src:
            while True:
                b = src.read(block_bytes)
                if not b:
                    return
                yield np.frombuffer(b[: len(b) // 2 * 2], dtype="<i2").astype(np.float32) / 32767.0

    meter = narration.ChapterMeter(SR)
    for x in blocks():
        meter.feed(x)
    gain = meter.gain()
    with sf.SoundFile(part, "w", SR, 1, "PCM_16", format="RF64" if size > WAV_LIMIT else "WAV") as out:
        for x in blocks():
            out.write(np.clip(x * gain, -1.0, 1.0))
    part.replace(wav)
    loud_in = meter.loudness()
    level = {
        "lufs_in": round(loud_in, 2) if loud_in is not None else None,
        "gain_db": round(20 * np.log10(gain), 2),
        "peak_dbfs": round(20 * np.log10(max(meter.peak * gain, 1e-9)), 2),
    }
    path = wav
    if fmt != "wav":
        ext = media._EXPORT[fmt]["ext"]
        dest = folder / f"{stem}.{ext}"
        try:
            media.export(wav, fmt, dest, ENCODE_QUALITY)
            wav.unlink()
            path = dest
        except EngineError as exc:
            log.warning("book %s chapter %d: saving as %s failed (%s); keeping it as WAV", bid, k + 1, fmt, exc.message)
            path = folder / f"{stem}.wav"
            wav.replace(path)
    with _LOCK:
        cs = state["chapters"][k]
        stats = cs.pop("stats", [])
        title = book["chapters"][k]["title"]
        duration_s = round(size / BYTES_PER_S, 3)
        (book_dir(data, bid) / "stats").mkdir(parents=True, exist_ok=True)
        _write_json(
            book_dir(data, bid) / "stats" / f"{k + 1:03d}.json",
            {"title": title, "level": level, "segments": stats},
        )
        saved = _read_json(_segs_path(data, bid, k)) or {}
        timed = timeline(saved.get("segments") or [], stats, duration_s, title)
        timing = None
        if timed is None:
            log.warning("book %s chapter %d: segment records incomplete, no segment timing", bid, k + 1)
        else:
            timing = _timing_path(data, bid, k)
            timing.parent.mkdir(parents=True, exist_ok=True)
            _write_json(
                timing, {"chapter": k, "title": title, "path": path, "duration_s": duration_s, "segments": timed}
            )
        cs.update(
            {
                "status": "done",
                "path": str(path),
                "duration_s": duration_s,
                "level": level,
                "segments_path": str(timing) if timing else None,
            }
        )
        _save_state(data, bid, state)
    for p in (pcm, _segs_path(data, bid, k)):
        with contextlib.suppress(OSError):
            p.unlink()
    return path


def timeline(
    segs: list[Any], stats: list[dict[str, Any]], duration_s: float, title: str | None = None
) -> list[dict[str, Any]] | None:
    """Each segment's speech in the chapter file, from ``_commit``'s layout: LEAD_MS of silence, then per segment its
    trimmed audio (starting at its stat's ``at``, ``dur`` long) and the gap after it. None when a stat is missing."""
    by_i = {m.get("i"): m for m in stats}
    if not segs or any(i not in by_i or by_i[i].get("at") is None for i in range(len(segs))):
        return None
    kinds = [str(s[2]) for s in segs]
    n_title = kinds.index("title") + 1 if "title" in kinds else 0
    out = []
    for i, s in enumerate(segs):
        m = by_i[i]
        start = float(m["at"])
        row: dict[str, Any] = {
            "text": str(s[0]),
            "start": round(start, 3),
            "end": round(min(start + float(m.get("dur") or 0.0), duration_s), 3),
        }
        if i < n_title:
            row["kind"] = "title"
            if n_title == 1 and title:
                row["text"] = title
        out.append(row)
    return out


def _ffmeta(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def _make_m4b(
    data: config.DataDir,
    bid: str,
    book: dict[str, Any],
    state: dict[str, Any],
    on_progress: Any = None,
    stop: threading.Event | None = None,
) -> Path | None:
    exe = media.ffmpeg_path()
    if not exe:
        log.info("book %s: no ffmpeg, no M4B", bid)
        return None
    _, _, encoders, _ = media._ffmpeg_lists()
    if "aac" not in encoders:
        log.info("book %s: this ffmpeg has no AAC encoder, no M4B", bid)
        return None
    work = _work(data, bid)
    work.mkdir(parents=True, exist_ok=True)
    lst, meta = work / "m4b.txt", work / "m4b.meta"
    rows, t = [], 0
    lines = [";FFMETADATA1", f"title={_ffmeta(str(book.get('title') or 'Book'))}", "genre=Audiobook"]
    for c, cs in zip(book["chapters"], state["chapters"]):
        if cs.get("status") != "done" or not cs.get("path"):
            continue
        rows.append("file '" + str(cs["path"]).replace("'", "'\\''") + "'")
        dur = round(float(cs.get("duration_s") or cs.get("audio_s") or 0.0) * 1000)
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t}", f"END={t + dur}", f"title={_ffmeta(str(c['title']))}"]
        t += dur
    if not rows:
        return None
    lst.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    dest = Path(book["folder"]) / f"{safe_name(str(book.get('title') or 'Book')) or 'Book'}.m4b"
    part = dest.with_name(dest.name + ".part.m4b")
    cmd = [
        exe,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        lst,
        "-i",
        meta,
        "-map",
        "0:a",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        M4B_BITRATE,
        "-ac",
        "1",
        "-movflags",
        "+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "mp4",
        part,
    ]
    code, err = _ffmpeg_with_progress(cmd, t / 1000.0, on_progress, stop)
    if code != 0 or not part.is_file():
        with contextlib.suppress(OSError):
            part.unlink()
        if not (stop and stop.is_set()):
            log.warning("book %s: the M4B failed: %s", bid, media._tail(err))
        return None
    part.replace(dest)
    for p in (lst, meta):
        with contextlib.suppress(OSError):
            p.unlink()
    state["m4b"] = str(dest)
    _save_state(data, bid, state)
    return dest


def _ffmpeg_with_progress(
    cmd: list[str | Path], total_s: float, on_progress: Any, stop: threading.Event | None
) -> tuple[int, bytes]:
    """Runs ffmpeg with ``-progress pipe:1``: ``on_progress(pct, eta_s)`` about once a second; ``stop`` kills it."""
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=media._FLAGS
        )
    except OSError as exc:
        return -1, str(exc).encode()
    err: list[bytes] = []
    reader = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)  # type: ignore[union-attr]
    reader.start()
    t0, last = time.time(), 0.0
    for line in proc.stdout:  # type: ignore[union-attr]
        if stop is not None and stop.is_set():
            proc.kill()
            break
        key, _, value = line.decode("ascii", errors="replace").strip().partition("=")
        if key != "out_time_us" or not value.isdigit() or on_progress is None or total_s <= 0:
            continue
        done = int(value) / 1e6
        now = time.time()
        if done > 0 and now - last >= 1.0:
            last = now
            on_progress(min(100.0, 100.0 * done / total_s), (now - t0) * max(0.0, total_s - done) / done)
    proc.wait()
    reader.join(timeout=5)
    return proc.returncode, b"".join(err)


def _progress(book: dict[str, Any], state: dict[str, Any], k: int) -> dict[str, Any]:
    with _LOCK:
        return _progress_locked(book, state, k)


def _progress_locked(book: dict[str, Any], state: dict[str, Any], k: int) -> dict[str, Any]:
    cs = state["chapters"][k]
    return {
        "id": book["id"],
        "chapter": k,
        "chapters": len(book["chapters"]),
        "title": book["chapters"][k]["title"],
        "segment": int(cs.get("segment") or 0),
        "segments": int(cs.get("segments") or 0),
        "chapter_audio_s": round(float(cs.get("audio_s") or 0.0), 2),
        **_totals(book, state),
    }


async def _run(ctx: Context, job: str, data: config.DataDir, bid: str, v: _Voice) -> None:
    t_start = time.time()
    jobdir = data.tmp / job
    book: dict[str, Any] | None = None
    state: dict[str, Any] | None = None

    def end(event: str, payload: dict[str, Any]) -> None:
        _RUNNING.pop(bid, None)
        tts.release_job(job)  # free the slot before the app sees the end (it may start the next job)
        ctx.emit(event, {"id": bid, **payload}, job=job)

    try:
        book, state = await aio.run_io(_load, data, bid)
        state["status"], state["error"] = "running", None
        await aio.run_io(_save_state, data, bid, state)
        await aio.run_io(jobdir.mkdir, parents=True, exist_ok=True)
        choice, zs, index_rate, worker = v.choice, v.zs, v.rvc_rate, v.worker
        native = (
            await aio.run_io(accent.prepare, choice, v.ref, v.voice_match) if choice.accent == "native-us" else None
        )
        if choice.accent == "native-us" and native is None:
            choice = accent.Choice("own", choice.native_voice, "unavailable")
        if native is None and choice.converter == "zero-shot":
            zs = None
        synth_ref, ref_text, pitch = (native.ref.path, native.ref.text, native.pitch) if native else (v.ref, None, 0)
        if native is not None:
            index_rate = native.index_rate
        await speech.yield_gpu()
        todo_ch = [k for k, cs in enumerate(state["chapters"]) if cs.get("status") != "done"]
        log.info(
            "book %s (%s): %d of %d chapter(s) to read, %s, accent %s",
            bid,
            job,
            len(todo_ch),
            len(book["chapters"]),
            v.lang,
            choice.accent,
        )
        turn = zs.turn if zs is not None else None
        todo: asyncio.Queue[tuple[int, int, int, str | None, str, int, str] | None] = asyncio.Queue(maxsize=2)
        last_at = [time.time()]
        session = {"audio_s": 0.0, "elapsed_s": 0.0, "retries": 0}
        judge = narration.Judge()
        for cs in state["chapters"]:
            for m in cs.get("stats") or []:
                if m.get("spc") is not None and m.get("kind") != "title":
                    judge.accept(m)

        async def synth(k: int, i: int, seg: str, tag: str = "tts") -> str:
            async with zsvc.gpu_turn(turn):
                return await v.backend.synthesize(
                    seg,
                    v.lang,
                    synth_ref,
                    SynthParams(
                        out_path=jobdir / f"c{k}_s{i}_{tag}.wav",
                        temperature=v.temperature,
                        quality=v.quality,
                        duration_factor=v.duration_factor,
                        reference_text=ref_text,
                    ),
                )

        async def convert(k: int, i: int, path: str, tag: str = "") -> str | Path:
            if zs is not None:
                dst = jobdir / f"c{k}_s{i}_vc{tag}.wav"
                await zs.convert(path, dst)
                return dst
            if worker is not None:
                dst = jobdir / f"c{k}_s{i}_rvc{tag}.wav"
                await worker.convert(path, dst, v.pth, v.idx, index_rate, pitch=pitch)
                return dst
            return path

        async def produce() -> None:
            for k in todo_ch:
                segs = await aio.run_io(_prepare_chapter, data, bid, k, book, state, v.backend.count_tokens)
                start = int(state["chapters"][k].get("segment") or 0)
                ctx.emit("book.progress", _progress(book, state, k), job=job)
                for i in range(start, len(segs)):
                    seg, pause, kind = segs[i]
                    t0 = time.time()
                    path = await synth(k, i, seg)
                    log.info(
                        "book %s ch %d seg %d/%d (%s): %.2fs", bid, k + 1, i + 1, len(segs), kind, time.time() - t0
                    )
                    await todo.put((k, i, len(segs), path, seg, pause, kind))
                if start >= len(segs):  # every segment was committed before a stop: only the chapter file is missing
                    await todo.put((k, len(segs) - 1, len(segs), None, "", 0, ""))
            await todo.put(None)

        async def add_segment(k: int, i: int, path: str, seg: str, pause: int, kind: str) -> None:
            files = [path]
            src = await convert(k, i, path)
            files.append(src)
            x, m = await aio.run_io(_segment_audio, src, len(seg))
            # a title is read differently on purpose: it neither judges nor is judged
            score, why = judge.deviation(m) if QUALITY_CHECKS and kind != "title" else (0.0, [])
            retried = None
            if score > 1.0:
                session["retries"] += 1
                p2 = await synth(k, i, seg, "tts2")
                s2 = await convert(k, i, p2, "2")
                files += [p2, s2]
                x2, m2 = await aio.run_io(_segment_audio, s2, len(seg))
                score2, _why2 = judge.deviation(m2)
                retried = {
                    "why": why,
                    "score": round(score, 2),
                    "score2": round(score2, 2),
                    "kept": "second" if score2 < score else "first",
                }
                log.info(
                    "book %s ch %d seg %d: outlier (%s), made again: %s (%.2f -> %.2f)",
                    bid,
                    k + 1,
                    i + 1,
                    ", ".join(why),
                    retried["kept"],
                    score,
                    score2,
                )
                if score2 < score:
                    x, m = x2, m2
            fix_db = judge.level_fix_db(m) if QUALITY_CHECKS else 0.0
            if fix_db:
                x = x * np.float32(10 ** (fix_db / 20.0))
            if kind != "title":
                judge.accept(m)
            stat = {
                "i": i,
                "kind": kind,
                "pause_ms": pause,
                "chars": len(seg),
                "dur": m["dur"],
                "spc": round(m["spc"], 5),
                "lufs": None if m["lufs"] is None else round(m["lufs"], 2),
                "f0": None if m["f0"] is None else round(m["f0"], 1),
                "fix_db": round(fix_db, 2),
                "retried": retried,
            }
            now = time.time()
            dt, last_at[0] = now - last_at[0], now
            # The commit finishes even when the job is stopped meanwhile: the PCM and its record stay together.
            fut = asyncio.ensure_future(aio.run_io(_commit, data, bid, state, k, i, x, pause, len(seg), dt, stat))
            try:
                added = await asyncio.shield(fut)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await fut
                raise
            session["audio_s"] += added
            session["elapsed_s"] += dt
            for p in {Path(f) for f in files}:
                with contextlib.suppress(OSError):
                    await aio.run_io(p.unlink)
            ctx.emit(
                "book.progress",
                {**_progress(book, state, k), "segment_rtf": round(dt / added, 3) if added else None},
                job=job,
            )

        async def consume() -> None:
            async for k, i, n, path, seg, pause, kind in aio.queue_items(todo):
                if path is not None:
                    await add_segment(k, i, path, seg, pause, kind)
                if i == n - 1:
                    # Shielded like the commit; a stop meanwhile takes effect once the chapter is reported.
                    fut = asyncio.ensure_future(aio.run_io(_finish_chapter, data, bid, book, state, k, v.out_fmt))
                    stop: asyncio.CancelledError | None = None
                    try:
                        out = await asyncio.shield(fut)
                    except asyncio.CancelledError as exc:
                        stop = exc
                        out = await fut
                    cs = state["chapters"][k]
                    audio, el = float(cs.get("audio_s") or 0.0), float(cs.get("elapsed_s") or 0.0)
                    log.info(
                        "book %s chapter %d done: %.1fs of audio in %.1fs (RTF %.2f), level %s",
                        bid,
                        k + 1,
                        audio,
                        el,
                        el / audio if audio else 0.0,
                        cs.get("level"),
                    )
                    ctx.emit(
                        "book.chapter",
                        {
                            "id": bid,
                            "index": k,
                            "title": book["chapters"][k]["title"],
                            "path": out,
                            "segments_path": cs.get("segments_path"),
                            "duration_s": cs.get("duration_s"),
                            "audio_s": audio,
                            "elapsed_s": round(el, 1),
                            "rtf": round(el / audio, 3) if audio else None,
                            "level": cs.get("level"),
                            **_totals(book, state),
                        },
                        job=job,
                    )
                    last_at[0] = time.time()
                    if stop is not None:
                        raise stop

        attempt, audio_at_failure = 0, 0.0
        while True:
            try:
                await aio.gather_or_cancel(produce(), consume())
                break
            except Exception as exc:
                if not _short_of_memory(exc):
                    raise
                if session["audio_s"] > audio_at_failure:
                    attempt = 0
                if attempt >= len(MEMORY_RETRY_WAIT_S):
                    raise
                wait, audio_at_failure = MEMORY_RETRY_WAIT_S[attempt], session["audio_s"]
                attempt += 1
                log.warning(
                    "book %s: short of memory (%s: %s); trying again in %.0fs (%d of %d)",
                    bid,
                    type(exc).__name__,
                    exc,
                    wait,
                    attempt,
                    len(MEMORY_RETRY_WAIT_S),
                )
                k = next((k for k, cs in enumerate(state["chapters"]) if cs.get("status") != "done"), 0)
                ctx.emit("book.progress", {**_progress(book, state, k), "stage": "memory", "retry_in_s": wait}, job=job)
                gc.collect()
                await asyncio.sleep(wait)
                todo_ch = [k for k, cs in enumerate(state["chapters"]) if cs.get("status") != "done"]
                todo = asyncio.Queue(maxsize=2)
                last_at[0] = time.time()
        m4b = None
        if book["settings"].get("m4b"):
            last_k = len(book["chapters"]) - 1
            ctx.emit("book.progress", {**_progress(book, state, last_k), "stage": "m4b", "stage_pct": 0.0}, job=job)

            def m4b_progress(pct: float, eta_s: float) -> None:
                ctx.emit(
                    "book.progress",
                    {
                        **_progress(book, state, last_k),
                        "stage": "m4b",
                        "stage_pct": round(pct, 1),
                        "stage_eta_s": round(eta_s),
                    },
                    job=job,
                )

            stop_m4b = threading.Event()
            try:
                m4b = await aio.run_io(_make_m4b, data, bid, book, state, m4b_progress, stop_m4b)
            except asyncio.CancelledError:
                stop_m4b.set()
                raise
        state["status"] = "done"
        await aio.run_io(_save_state, data, bid, state)
        await aio.run_io(shutil.rmtree, _work(data, bid), ignore_errors=True)
        log.info("book %s done in %.1fs (this session: %.1fs of audio)", bid, time.time() - t_start, session["audio_s"])
        end(
            "book.done",
            {
                "folder": book["folder"],
                "m4b": m4b,
                "m4b_requested": bool(book["settings"].get("m4b")),
                "chapters": len(book["chapters"]),
                **_totals(book, state),
                "session_elapsed_s": round(time.time() - t_start, 1),
                "session_audio_s": round(session["audio_s"], 2),
                "retries": session["retries"],
                **choice.fields(),
                **v.ref_fields,
            },
        )
    except asyncio.CancelledError:
        why = _STOP.pop(bid, None)
        if state is not None and why == "cancel":
            await aio.run_io(_drop_partial, data, bid, state, "cancelled")
            end("book.error", {"code": "cancelled", "message": "the book was cancelled"})
        elif state is not None and why == "pause":
            state["status"] = "paused"
            await aio.run_io(_save_state, data, bid, state)
            _RUNNING.pop(bid, None)
            end("book.paused", summary(data, book, state) if book is not None else {})
        else:  # the engine is shutting down: state.json still says "running", so the next engine offers a resume
            _RUNNING.pop(bid, None)
            tts.release_job(job)
            raise
    except EngineError as exc:
        await _failed(data, bid, state, exc.code, exc.message)
        end("book.error", {"code": exc.code, "message": exc.message})
    except Exception as exc:
        log.exception("book %s failed", bid)
        await _failed(data, bid, state, "internal", f"{type(exc).__name__}: {exc}")
        end("book.error", {"code": "internal", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        _RUNNING.pop(bid, None)
        _STOP.pop(bid, None)
        tts.release_job(job)
        await aio.run_io(shutil.rmtree, jobdir, ignore_errors=True)


def _short_of_memory(exc: BaseException | None) -> bool:
    """Worth waiting for: not when the worker says retrying cannot help (too little RAM in all)."""
    chain = []
    while exc is not None:
        chain.append(exc)
        exc = exc.__cause__
    if any(getattr(e, "permanent", False) for e in chain):
        return False
    return any(
        isinstance(e, (MemoryError, WorkerDied)) or getattr(e, "code", None) in ("no_memory", "no_vram") for e in chain
    )


async def _failed(data: config.DataDir, bid: str, state: dict[str, Any] | None, code: str, message: str) -> None:
    if state is None:
        return
    state["status"] = "error"
    state["error"] = {"code": code, "message": message}
    with contextlib.suppress(Exception):
        await aio.run_io(_save_state, data, bid, state)
