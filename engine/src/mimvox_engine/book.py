"""Audiobooks: ``book.start`` / ``book.resume`` / ``book.pause`` / ``book.cancel`` / ``book.list`` / ``book.get`` /
``book.play`` / ``book.delete``. One job reads a whole book, chapter by chapter, segment by segment, in the voice and
style of Speak, and writes one file per chapter (plus an optional M4B with chapter marks).

The speech pipeline is ``tts.generate``'s: the same backend, native accent (the zero-shot converter or RVC with the
native clip), RVC and the one generation slot (``tts.start_exclusive``, so Speak, Redub and training answer ``busy``
while a book is read). The text is cut and paced for listening by ``narration.py`` (whole sentences, pauses by
punctuation, trimmed and levelled segments, an outlier made again once; every chapter normalized to -18 LUFS). And the
audio goes to disk, not to the app:

- Nothing is held in memory. Every finished segment (resampled to 44.1 kHz, its pause appended) is appended as 16-bit
  PCM to ``<data>/books/<id>/work/<NN>.pcm``; when a chapter's last segment is in, that file becomes the chapter's
  output file (``<data>/outputs/books/<book>/<NN> - <title>.<ext>``; RF64 when a WAV would pass 4 GB) and goes away.
- After every segment ``state.json`` records how far the chapter got: the next segment and the PCM length that goes
  with it (written after the PCM, atomically). A resume truncates the PCM to that length, so a segment written but
  not recorded (a crash between the two writes) is dropped and spoken again: no gap and no repeat. The segments of a
  chapter are fixed in ``work/<NN>.segments.json`` when it starts, so a resume continues the same cut.
- Pause and cancel stop the job at once (the segment being made is spoken again on resume) and free the slot. An
  engine that stops mid-book (the app closed) leaves ``status: "running"`` on disk; the next engine reports such a book
  as ``paused`` (``interrupted``) and ``book.resume`` continues it.

Everything runs on the engine loop; file work goes to the IO pool (``aio.run_io``). A producer task synthesizes (one
segment ahead, also across chapter ends) and a consumer converts, appends and finishes chapters, as in ``tts.py``.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import accent, aio, backends, config, media, narration, reflang, rvc, speech, tts, voices, zsvc
from . import audio_util as au
from .backends import SynthParams
from .registry import Context, EngineError, method

log = logging.getLogger("mimvox.book")

SR = au.OUT_SR                # chapter audio: 44.1 kHz mono 16-bit
BYTES_PER_S = SR * 2
# Segmentation, pauses, trimming, levels and the outlier checks: narration.py (one table of constants there).
QUALITY_CHECKS = True         # re-make outlier segments once and even out level steps (narration.Judge)
ENCODE_QUALITY = "standard"   # chapter files: MP3 192 kbps, Opus 96 kbps, AAC 192 kbps, Vorbis q5 (speech needs no more)
M4B_BITRATE = "96k"           # the whole-book M4B (AAC, mono)
MAX_CHARS = 20_000_000        # a book's text (about 3.5 million words)
WAV_LIMIT = 0xFFFFFFFF - 1024  # a RIFF WAV cannot pass 4 GB: RF64 above that
HOLDER = "a book is being read; pause it (Speak, Book) to use speech, Redub or training"

_RUNNING: dict[str, str] = {}   # book id -> job (loop thread only)
_STOP: dict[str, str] = {}      # book id -> "pause" | "cancel" while its job is being stopped
# A running book's state dict is changed on IO threads (the producer cuts the next chapter while the consumer appends
# to this one) and read on the loop: every change and save holds this lock.
_LOCK = threading.RLock()


# ------------------------------------------------------------------------------------------------ files

def books_dir(data: config.DataDir) -> str:
    return os.path.join(data.root, "books")


def book_dir(data: config.DataDir, book_id: str) -> str:
    if not isinstance(book_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", book_id):
        raise EngineError("bad_request", f"invalid book id {book_id!r}")
    return os.path.join(books_dir(data), book_id)


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        return obj if isinstance(obj, dict) else None
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None


def _write_json(path: str, obj: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(5):  # Windows: a reader (an antivirus) may hold the target for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _load(data: config.DataDir, book_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(book.json, state.json)`` of a book; ``not_found`` when it does not exist."""
    d = book_dir(data, book_id)
    book = _read_json(os.path.join(d, "book.json"))
    if book is None:
        raise EngineError("not_found", f"book {book_id!r} does not exist")
    state = _read_json(os.path.join(d, "state.json")) or _new_state(book)
    n = len(book.get("chapters") or [])
    chs = state.setdefault("chapters", [])
    while len(chs) < n:
        chs.append(_new_chapter_state())
    return book, state


def _new_chapter_state() -> dict[str, Any]:
    return {"status": "waiting", "segment": 0, "segments": 0, "bytes": 0, "chars_done": 0, "audio_s": 0.0,
            "elapsed_s": 0.0, "path": None}


def _new_state(book: dict[str, Any]) -> dict[str, Any]:
    return {"status": "new", "current": 0, "audio_s": 0.0, "elapsed_s": 0.0,
            "chapters": [_new_chapter_state() for _ in book.get("chapters") or []], "m4b": None, "error": None,
            "updated": time.time()}


def _save_state(data: config.DataDir, book_id: str, state: dict[str, Any]) -> None:
    with _LOCK:
        state["updated"] = time.time()
        _write_json(os.path.join(book_dir(data, book_id), "state.json"), state)


_BAD_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def safe_name(name: str, limit: int = 80) -> str:
    """A file or folder name from a title: no characters Windows forbids, no trailing dots or spaces, not a device
    name, at most ``limit`` characters."""
    s = re.sub(r"\s+", " ", _BAD_NAME.sub(" ", name or "")).strip(" .")
    s = s[:limit].rstrip(" .")
    if re.fullmatch(r"(?i)(con|prn|aux|nul|com\d|lpt\d)(\..*)?", s):
        s = f"_{s}"
    return s


def _work(data: config.DataDir, book_id: str) -> str:
    return os.path.join(book_dir(data, book_id), "work")


def _pcm_path(data: config.DataDir, book_id: str, k: int) -> str:
    return os.path.join(_work(data, book_id), f"{k + 1:03d}.pcm")


def _segs_path(data: config.DataDir, book_id: str, k: int) -> str:
    return os.path.join(_work(data, book_id), f"{k + 1:03d}.segments.json")


def spoken_text(chapter: dict[str, Any], read_titles: bool) -> str:
    """What is read for a chapter: its title as a sentence of its own (with ``read_titles``), then its text."""
    title = re.sub(r"\s+", " ", str(chapter.get("title") or "")).strip()
    text = str(chapter.get("text") or "").strip()
    if not read_titles or not title:
        return text
    if not re.search(r"[.!?…:;]$", title):
        title += "."
    return f"{title}\n\n{text}" if text else title


def _flat_chars(text: str) -> int:
    return len(re.sub(r"\s+", " ", text).strip())


# ------------------------------------------------------------------------------------------- summaries

def _status(book_id: str, state: dict[str, Any]) -> tuple[str, bool]:
    """``(status, interrupted)``: a book recorded as running whose job is not running (the engine stopped mid-book) is
    ``paused`` and ``interrupted``."""
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
    return {"percent": round(100.0 * min(done_chars, total) / total, 2), "audio_s": round(audio, 2),
            "elapsed_s": round(elapsed, 1), "rtf": round(elapsed / audio, 3) if audio > 0 else None, "eta_s": eta,
            "est_audio_s": round(est_audio, 1) if est_audio else None, "chars": total, "chars_done": done_chars}


def summary(data: config.DataDir, book: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    bid = str(book["id"])
    status, interrupted = _status(bid, state)
    chs = state.get("chapters") or []
    out = {"id": bid, "title": book.get("title"), "source": book.get("source"), "format": book.get("format"),
           "language": (book.get("settings") or {}).get("language"), "voice_id": (book.get("settings") or {}).get("voice_id"),
           "status": status, "interrupted": interrupted, "job": _RUNNING.get(bid),
           "chapters": len(book.get("chapters") or []),
           "chapters_done": sum(1 for c in chs if c.get("status") == "done"),
           "current": int(state.get("current") or 0), "folder": book.get("folder"), "m4b": state.get("m4b"),
           "output_format": (book.get("settings") or {}).get("output_format"), "words": book.get("words"),
           "created": book.get("created"), "updated": state.get("updated"), "error": state.get("error")}
    out.update(_totals(book, state))
    return out


def _chapter_rows(book: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    chs = state.get("chapters") or []
    for i, c in enumerate(book.get("chapters") or []):
        cs = chs[i] if i < len(chs) else _new_chapter_state()
        audio, el = float(cs.get("audio_s") or 0.0), float(cs.get("elapsed_s") or 0.0)
        rows.append({"index": i, "title": c.get("title"), "words": c.get("words"), "chars": c.get("chars"),
                     "status": cs.get("status") or "waiting", "segment": int(cs.get("segment") or 0),
                     "segments": int(cs.get("segments") or 0), "audio_s": round(audio, 2), "elapsed_s": round(el, 1),
                     "rtf": round(el / audio, 3) if audio > 0 else None, "path": cs.get("path")})
    return rows


# ------------------------------------------------------------------------------------------- voice setup

@dataclass
class _Voice:
    """Everything a book job speaks with, checked when it starts (as ``tts.generate`` checks it)."""
    backend: backends.TtsBackend
    lang: str
    ref: str
    pth: str
    idx: str
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
    """The voice settings of a ``book.start`` request, checked and normalized (stored in ``book.json``)."""
    voice_id = params.get("voice_id")
    if not isinstance(voice_id, str) or not voice_id:
        raise EngineError("bad_request", "'voice_id' is required")
    quality = params.get("quality") or "best"
    if quality not in ("best", "fast"):
        raise EngineError("bad_request", "'quality' must be \"best\" or \"fast\"")
    lang, _ = tts.speech_language(params)
    m4b = params.get("m4b", False)
    titles = params.get("read_titles", True)
    if not isinstance(m4b, bool) or not isinstance(titles, bool):
        raise EngineError("bad_request", "'m4b' and 'read_titles' must be true or false")
    return {"voice_id": voice_id, "language": lang,
            "expressiveness": tts._num(params, "expressiveness", narration.EXPRESSIVENESS, 0.1, 1.5),
            "voice_match": tts._num(params, "voice_match", 1.0, 0.0, 1.0),
            "speed": tts._num(params, "speed", 1.0, tts.SPEED_MIN, tts.SPEED_MAX), "quality": quality,
            "accent": accent.parse(params), "native_voice": accent.parse_voice(params),
            "output_format": params.get("output_format") or "wav", "m4b": m4b, "read_titles": titles}


async def _voice(ctx: Context, data: config.DataDir, st: dict[str, Any]) -> _Voice:
    """The speech setup of a book's settings: the checks and choices of ``tts.generate`` (models ready, the voice and
    its reference, RVC, the native accent and its converter)."""
    lang, backend = tts.speech_language(st)
    tts.check_backend_ready(backend)
    voice_id = st["voice_id"]
    meta = await aio.run_io(voices.read_meta, data, voice_id)  # not_found
    out_fmt = await aio.run_io(media.output_format, st)
    voice_match = float(st["voice_match"])
    rvc_rate = rvc.index_rate_for(voice_match, lang, meta.get("language"))
    ref = data.reference(voice_id)
    pth, idx = data.rvc_model(voice_id)
    has_ref, has_pth = await aio.run_io(lambda: (os.path.isfile(ref), os.path.isfile(pth)))
    if not has_ref:
        raise EngineError("not_found", "this voice has no recording yet (reference.wav is missing)")
    use_rvc = has_pth and voice_match > 0
    worker = rvc.get(ctx.runtime) if use_rvc else None
    if use_rvc and (worker is None or tts.models_status()["rvc"] != "ready"):
        raise EngineError("not_ready", "the voice-conversion model is not ready")
    requested = st.get("accent") or "own"
    zs_worker = await aio.run_io(zsvc.get, ctx.runtime, data.root) if requested == "native-us" else None
    choice = accent.choose(requested, st.get("native_voice") or "auto", lang, has_pth, voice_match,
                           zs_worker is not None)
    zs = tts.zero_shot(ctx, choice, zs_worker, ref, st["quality"])
    ref_fields = await aio.run_io(reflang.fields, ref, lang, choice.accent == "native-us")
    return _Voice(backend, lang, ref, pth, idx, rvc_rate, worker, choice, zs, voice_match,
                  float(st["expressiveness"]), st["quality"], tts.duration_factor_for(st["speed"]), out_fmt, ref_fields)


# ------------------------------------------------------------------------------------------- the methods

def _clean_chapters(raw: Any) -> list[dict[str, Any]]:
    from .dataset.text import count_words  # noqa: PLC0415

    if not isinstance(raw, list) or not raw:
        raise EngineError("bad_request", "'chapters' must be a non-empty list of {\"title\", \"text\"}")
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


def _create(data: config.DataDir, title: str, source: str | None, fmt: str | None, chapters: list[dict[str, Any]],
            settings: dict[str, Any]) -> dict[str, Any]:
    """IO thread: a new book project (``book.json``, ``state.json``) and its output folder name."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{voices.slugify(title)[:32].strip('-') or 'book'}-{stamp}"
    bid, n = base, 1
    while os.path.exists(os.path.join(books_dir(data), bid)):
        n += 1
        bid = f"{base}-{n}"
    name = safe_name(title) or "Book"
    root = os.path.join(data.outputs, "books")
    folder, n = os.path.join(root, name), 1
    while os.path.exists(folder) or any(_folder_taken(data, folder)):
        n += 1
        folder = os.path.join(root, f"{name} ({n})")
    width = max(2, len(str(len(chapters))))
    for i, c in enumerate(chapters):
        c["file"] = f"{i + 1:0{width}d} - {safe_name(c['title'], 60) or 'Chapter'}"
        c["chars"] = _flat_chars(spoken_text(c, settings["read_titles"]))
    book = {"id": bid, "title": title, "source": source, "format": fmt, "created": time.time(), "folder": folder,
            "settings": settings, "words": sum(int(c["words"]) for c in chapters), "chapters": chapters}
    d = os.path.join(books_dir(data), bid)
    os.makedirs(os.path.join(d, "work"), exist_ok=True)
    _write_json(os.path.join(d, "book.json"), book)
    _save_state(data, bid, _new_state(book))
    return book


def _folder_taken(data: config.DataDir, folder: str) -> list[bool]:
    """Whether another book project already writes into ``folder`` (not created yet on disk)."""
    out = []
    try:
        names = os.listdir(books_dir(data))
    except OSError:
        return out
    for n in names:
        b = _read_json(os.path.join(books_dir(data), n, "book.json"))
        out.append(bool(b) and os.path.normcase(str(b.get("folder"))) == os.path.normcase(folder))
    return out


@method("book.start")
async def book_start(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Create a book project from chapters and start reading it."""
    data = config.data_dir(ctx)
    settings = _settings(params)
    chapters = _clean_chapters(params.get("chapters"))
    title = re.sub(r"\s+", " ", str(params.get("title") or "")).strip() or "Book"
    v = await _voice(ctx, data, settings)  # the checks, before anything is written
    if tts.is_busy():
        raise EngineError("busy", tts.busy_message("another generation is running"))
    settings["output_format"] = v.out_fmt
    source = params.get("source") if isinstance(params.get("source"), str) else None
    fmt = params.get("format") if isinstance(params.get("format"), str) else None
    book = await aio.run_io(_create, data, title, source, fmt, chapters, settings)
    try:
        job = _start(ctx, data, book["id"], v)
    except EngineError:  # another job took the slot while the project was written
        await aio.run_io(shutil.rmtree, book_dir(data, book["id"]), ignore_errors=True)
        raise
    return {"id": book["id"], "job": job, "folder": book["folder"], "chapters": len(chapters),
            "words": book["words"], **v.choice.fields(), **v.ref_fields}


@method("book.resume")
async def book_resume(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Continue a paused, interrupted, failed or cancelled book from its next unfinished segment."""
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    book, state = await aio.run_io(_load, data, bid)
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read already")
    if state.get("status") == "done":
        raise EngineError("bad_request", "this book is finished")
    v = await _voice(ctx, data, book["settings"])
    job = _start(ctx, data, bid, v)
    return {"id": bid, "job": job, "folder": book["folder"], "chapters": len(book["chapters"]),
            **v.choice.fields(), **v.ref_fields}


def _start(ctx: Context, data: config.DataDir, bid: str, v: _Voice) -> str:
    """Take the generation slot for the book (loop thread; no await since the busy checks)."""
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read already")
    job = tts.start_exclusive(ctx, lambda j: _run(ctx, j, data, bid, v), "another generation is running", HOLDER)
    _RUNNING[bid] = job
    _STOP.pop(bid, None)
    return job


@method("book.pause")
async def book_pause(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Stop reading now; ``book.paused`` follows. The segment being made is spoken again on resume."""
    bid = str(params.get("id") or "")
    job = _RUNNING.get(bid)
    if job is None:
        raise EngineError("bad_request", "this book is not being read")
    _STOP[bid] = "pause"
    ctx.cancel(job)
    return {}


@method("book.cancel")
async def book_cancel(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Stop reading and drop the unfinished chapter's progress (finished chapter files stay). A running job ends with
    ``book.error {"code":"cancelled"}``. The book can still be resumed: from the start of that chapter."""
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    job = _RUNNING.get(bid)
    if job is not None:
        _STOP[bid] = "cancel"
        ctx.cancel(job)
        return {}
    book, state = await aio.run_io(_load, data, bid)
    if state.get("status") != "done":
        await aio.run_io(_drop_partial, data, bid, state, "cancelled")
    return {}


def _drop_partial(data: config.DataDir, bid: str, state: dict[str, Any], status: str) -> None:
    """IO thread: forget the unfinished chapter (its PCM and cut) and record ``status``."""
    for cs in state.get("chapters") or []:
        if cs.get("status") != "done":
            cs.clear()
            cs.update(_new_chapter_state())
    shutil.rmtree(_work(data, bid), ignore_errors=True)
    os.makedirs(_work(data, bid), exist_ok=True)
    state["status"] = status
    _save_state(data, bid, state)


@method("book.list")
async def book_list(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Every book project, the most recently changed first (without the chapter texts)."""
    data = config.data_dir(ctx)

    def scan() -> list[dict[str, Any]]:
        out = []
        try:
            names = sorted(os.listdir(books_dir(data)))
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
    """One book: its summary, settings and chapters with their progress (``text: true`` adds the chapter texts)."""
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
    """Forget a book project (its progress and texts). The chapter files in ``outputs/books`` stay."""
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    d = book_dir(data, bid)
    if bid in _RUNNING:
        raise EngineError("busy", "this book is being read; pause or cancel it first")
    if not await aio.run_io(os.path.isdir, d):
        raise EngineError("not_found", f"book {bid!r} does not exist")
    await aio.run_io(shutil.rmtree, d, ignore_errors=True)
    return {}


@method("book.play")
async def book_play(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """A WAV to play a finished chapter with (the app plays WAV only): the chapter file itself when it is a WAV, else a
    decoded copy in ``<data>/tmp/play`` (gone at the next engine start)."""
    data = config.data_dir(ctx)
    bid = str(params.get("id") or "")
    k = params.get("index")
    if isinstance(k, bool) or not isinstance(k, int):
        raise EngineError("bad_request", "'index' must be a chapter index")
    book, state = await aio.run_io(_load, data, bid)
    chs = state.get("chapters") or []
    if not 0 <= k < len(chs) or chs[k].get("status") != "done" or not chs[k].get("path"):
        raise EngineError("not_found", "this chapter is not finished yet")
    path = str(chs[k]["path"])
    return {"path": path, "path_play": await aio.run_io(_playable, data, bid, k, path)}


def _playable(data: config.DataDir, bid: str, k: int, path: str) -> str:
    if not os.path.isfile(path):
        raise EngineError("not_found", "the chapter file is gone")
    if path.lower().endswith(".wav"):
        return path
    dst = os.path.join(data.tmp, media.PLAY_DIR, f"book-{bid}-{k + 1:03d}.wav")
    if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(path):
        return dst
    return media.play_wav(path, dst, SR, channels=1)  # streams: a long chapter never sits in memory


# ------------------------------------------------------------------------------------------- the job

def _pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2").tobytes()


def _prepare_chapter(data: config.DataDir, bid: str, k: int, book: dict[str, Any], state: dict[str, Any],
                     count: Any) -> list[tuple[str, int, str]]:
    """IO thread: the chapter's segments (fixed in ``work/<NN>.segments.json`` the first time) and its PCM cut back to
    the length ``state.json`` recorded (a segment written but never recorded is dropped and spoken again)."""
    with _LOCK:
        return _prepare_chapter_locked(data, bid, k, book, state, count)


def _prepare_chapter_locked(data: config.DataDir, bid: str, k: int, book: dict[str, Any], state: dict[str, Any],
                            count: Any) -> list[tuple[str, int, str]]:
    path = _segs_path(data, bid, k)
    cs = state["chapters"][k]
    saved = _read_json(path)
    segs: list[tuple[str, int, str]] | None = None
    if saved and cs.get("status") == "partial":
        segs = [(str(t), int(p), str(kind)) for t, p, kind in saved.get("segments") or []]
    if not segs:
        ch = book["chapters"][k]
        titles = bool(book["settings"].get("read_titles", True))
        segs = narration.segments(ch.get("title") if titles else None, str(ch.get("text") or ""), count)
        if not segs:
            segs = [(".", narration.PAUSES["end"], "end")]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_json(path, {"segments": [list(s) for s in segs]})
        cs.clear()
        cs.update(_new_chapter_state())
        cs["status"] = "partial"
        cs["segments"] = len(segs)
        with contextlib.suppress(OSError):
            os.remove(_pcm_path(data, bid, k))
    pcm = _pcm_path(data, bid, k)
    keep = int(cs.get("bytes") or 0)
    if os.path.exists(pcm) or keep:
        with open(pcm, "ab") as fh:  # creates it when missing (then it is shorter than recorded: the cut is re-made)
            pass
        size = os.path.getsize(pcm)
        if size < keep:  # lost audio that was recorded (should not happen): start the chapter again
            log.warning("book %s chapter %d: %d bytes on disk, %d recorded; starting it again", bid, k + 1, size, keep)
            cs.update({"segment": 0, "bytes": 0, "chars_done": 0, "audio_s": 0.0, "stats": []})
            keep = 0
        if size != keep:
            with open(pcm, "r+b") as fh:
                fh.truncate(keep)
    cs["status"] = "partial"
    cs["segments"] = len(segs)
    _save_state(data, bid, state)
    return segs


def _segment_audio(src: str, chars: int) -> tuple[np.ndarray, dict[str, Any]]:
    """IO thread: a converted segment at 44.1 kHz, trimmed of the model's silence and faded, with its measures."""
    x, sr = au.read_mono(src)
    x = narration.trim(au.resample(x, sr, SR), SR)
    return x, narration.measure(x, SR, chars)


def _commit(data: config.DataDir, bid: str, state: dict[str, Any], k: int, i: int, x: np.ndarray, pause_ms: int,
            chars: int, dt: float, stat: dict[str, Any]) -> float:
    """IO thread: append segment ``i`` of chapter ``k`` (the chapter's lead-in before segment 0, the pause after it)
    to the chapter PCM, then record it (and its measures, ``stat``) in ``state.json``. Returns the seconds added."""
    parts = [au.silence(narration.LEAD_MS)] if i == 0 else []
    parts += [x, au.silence(narration.gap(pause_ms))]
    data_bytes = _pcm(np.concatenate(parts))
    pcm = _pcm_path(data, bid, k)
    with _LOCK:
        return _append(data, bid, state, k, i, pcm, data_bytes, chars, dt, stat)


def _append(data: config.DataDir, bid: str, state: dict[str, Any], k: int, i: int, pcm: str, data_bytes: bytes,
            chars: int, dt: float, stat: dict[str, Any]) -> float:
    cs = state["chapters"][k]
    stats = cs.setdefault("stats", [])
    del stats[i:]  # (a segment made again after a resume replaces its old record)
    lead = narration.LEAD_MS / 1000.0 if i == 0 else 0.0
    stats.append({**stat, "at": round(int(cs.get("bytes") or 0) / BYTES_PER_S + lead, 3)})
    with open(pcm, "ab") as fh:
        if fh.tell() != int(cs.get("bytes") or 0):  # (never: the chapter was cut back to the record when it started)
            fh.truncate(int(cs.get("bytes") or 0))
            fh.seek(int(cs.get("bytes") or 0))
        fh.write(data_bytes)
        fh.flush()
        os.fsync(fh.fileno())
        size = fh.tell()
    added = len(data_bytes) / BYTES_PER_S
    cs.update({"segment": i + 1, "bytes": size, "chars_done": int(cs.get("chars_done") or 0) + chars,
               "audio_s": round(float(cs.get("audio_s") or 0.0) + added, 3),
               "elapsed_s": round(float(cs.get("elapsed_s") or 0.0) + dt, 3)})
    state["audio_s"] = round(float(state.get("audio_s") or 0.0) + added, 3)
    state["elapsed_s"] = round(float(state.get("elapsed_s") or 0.0) + dt, 3)
    state["current"] = k
    _save_state(data, bid, state)
    return added


def _finish_chapter(data: config.DataDir, bid: str, book: dict[str, Any], state: dict[str, Any], k: int,
                    fmt: str) -> str:
    """IO thread: the chapter PCM becomes its output file (WAV streamed in blocks, RF64 past 4 GB; another format is
    encoded from a WAV in the work folder, kept as WAV if the encoder fails). Records it and drops the work files."""
    import soundfile as sf  # noqa: PLC0415

    pcm = _pcm_path(data, bid, k)
    folder = book["folder"]
    os.makedirs(folder, exist_ok=True)
    stem = book["chapters"][k]["file"]
    wav = os.path.join(folder, f"{stem}.wav") if fmt == "wav" else os.path.join(_work(data, bid), f"{k + 1:03d}.wav")
    part = f"{wav}.part"
    size = os.path.getsize(pcm)
    block_bytes = 1 << 22

    def blocks():  # the chapter PCM as float blocks, never all of it at once
        with open(pcm, "rb") as src:
            while True:
                b = src.read(block_bytes)
                if not b:
                    return
                yield np.frombuffer(b[: len(b) // 2 * 2], dtype="<i2").astype(np.float32) / 32767.0

    meter = narration.ChapterMeter(SR)  # pass 1: loudness and peak; pass 2: the file at TARGET_LUFS
    for x in blocks():
        meter.feed(x)
    gain = meter.gain()
    with sf.SoundFile(part, "w", SR, 1, "PCM_16", format="RF64" if size > WAV_LIMIT else "WAV") as out:
        for x in blocks():
            out.write(np.clip(x * gain, -1.0, 1.0))
    os.replace(part, wav)
    loud_in = meter.loudness()
    level = {"lufs_in": round(loud_in, 2) if loud_in is not None else None, "gain_db": round(20 * np.log10(gain), 2),
             "peak_dbfs": round(20 * np.log10(max(meter.peak * gain, 1e-9)), 2)}
    path = wav
    if fmt != "wav":
        ext = media._EXPORT[fmt]["ext"]
        dest = os.path.join(folder, f"{stem}.{ext}")
        try:
            media.export(wav, fmt, dest, ENCODE_QUALITY)
            os.remove(wav)
            path = dest
        except EngineError as exc:
            log.warning("book %s chapter %d: saving as %s failed (%s); keeping it as WAV", bid, k + 1, fmt, exc.message)
            path = os.path.join(folder, f"{stem}.wav")
            os.replace(wav, path)
    with _LOCK:
        cs = state["chapters"][k]
        os.makedirs(os.path.join(book_dir(data, bid), "stats"), exist_ok=True)
        _write_json(os.path.join(book_dir(data, bid), "stats", f"{k + 1:03d}.json"),
                    {"title": book["chapters"][k]["title"], "level": level, "segments": cs.pop("stats", [])})
        cs.update({"status": "done", "path": path, "duration_s": round(size / BYTES_PER_S, 3), "level": level})
        _save_state(data, bid, state)
    for p in (pcm, _segs_path(data, bid, k)):
        with contextlib.suppress(OSError):
            os.remove(p)
    return path


def _ffmeta(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s)


def _make_m4b(data: config.DataDir, bid: str, book: dict[str, Any], state: dict[str, Any]) -> str | None:
    """IO thread: the whole book as one M4B (AAC) with a chapter mark per chapter, when ffmpeg can make it."""
    exe = media.ffmpeg_path()
    if not exe:
        log.info("book %s: no ffmpeg, no M4B", bid)
        return None
    _, _, encoders, _ = media._ffmpeg_lists()
    if "aac" not in encoders:
        log.info("book %s: this ffmpeg has no AAC encoder, no M4B", bid)
        return None
    work = _work(data, bid)
    os.makedirs(work, exist_ok=True)
    lst, meta = os.path.join(work, "m4b.txt"), os.path.join(work, "m4b.meta")
    rows, t = [], 0
    lines = [";FFMETADATA1", f"title={_ffmeta(str(book.get('title') or 'Book'))}", "genre=Audiobook"]
    for c, cs in zip(book["chapters"], state["chapters"]):
        if cs.get("status") != "done" or not cs.get("path"):
            continue
        rows.append("file '" + str(cs["path"]).replace("'", "'\\''") + "'")
        dur = int(round(float(cs.get("duration_s") or cs.get("audio_s") or 0.0) * 1000))
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t}", f"END={t + dur}", f"title={_ffmeta(str(c['title']))}"]
        t += dur
    if not rows:
        return None
    with open(lst, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(rows) + "\n")
    with open(meta, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    dest = os.path.join(book["folder"], f"{safe_name(str(book.get('title') or 'Book')) or 'Book'}.m4b")
    part = f"{dest}.part.m4b"
    cmd = [exe, "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", meta,
           "-map", "0:a", "-map_metadata", "1", "-map_chapters", "1", "-c:a", "aac", "-b:a", M4B_BITRATE, "-ac", "1",
           "-movflags", "+faststart", "-f", "mp4", part]
    res = media._run(cmd, timeout=6 * 3600.0)
    if res.returncode != 0 or not os.path.isfile(part):
        with contextlib.suppress(OSError):
            os.remove(part)
        log.warning("book %s: the M4B failed: %s", bid, media._tail(res.stderr))
        return None
    os.replace(part, dest)
    for p in (lst, meta):
        with contextlib.suppress(OSError):
            os.remove(p)
    state["m4b"] = dest
    _save_state(data, bid, state)
    return dest


def _progress(book: dict[str, Any], state: dict[str, Any], k: int) -> dict[str, Any]:
    with _LOCK:
        return _progress_locked(book, state, k)


def _progress_locked(book: dict[str, Any], state: dict[str, Any], k: int) -> dict[str, Any]:
    cs = state["chapters"][k]
    return {"id": book["id"], "chapter": k, "chapters": len(book["chapters"]), "title": book["chapters"][k]["title"],
            "segment": int(cs.get("segment") or 0), "segments": int(cs.get("segments") or 0),
            "chapter_audio_s": round(float(cs.get("audio_s") or 0.0), 2), **_totals(book, state)}


async def _run(ctx: Context, job: str, data: config.DataDir, bid: str, v: _Voice) -> None:
    t_start = time.time()
    jobdir = os.path.join(data.tmp, job)
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
        await aio.run_io(os.makedirs, jobdir, exist_ok=True)
        choice, zs, index_rate, worker = v.choice, v.zs, v.rvc_rate, v.worker
        native = await aio.run_io(accent.prepare, choice, v.ref, v.voice_match) if choice.accent == "native-us" else None
        if choice.accent == "native-us" and native is None:
            choice = accent.Choice("own", choice.native_voice, "unavailable")
        if native is None:
            zs = None
        synth_ref, ref_text, pitch = (native.ref.path, native.ref.text, native.pitch) if native else (v.ref, None, 0)
        if native is not None:
            index_rate = native.index_rate
        await speech.yield_gpu()
        todo_ch = [k for k, cs in enumerate(state["chapters"]) if cs.get("status") != "done"]
        log.info("book %s (%s): %d of %d chapter(s) to read, %s, accent %s", bid, job, len(todo_ch),
                 len(book["chapters"]), v.lang, choice.accent)
        turn = zs.turn if zs is not None else None
        todo: asyncio.Queue[tuple[int, int, int, str, str, int, str] | None] = asyncio.Queue(maxsize=2)
        last_at = [time.time()]
        session = {"audio_s": 0.0, "elapsed_s": 0.0, "retries": 0}
        judge = narration.Judge()
        for cs in state["chapters"]:  # a resumed chapter: the medians of what is already read
            for m in cs.get("stats") or []:
                if m.get("spc") is not None and m.get("kind") != "title":
                    judge.accept(m)

        async def synth(k: int, i: int, seg: str, tag: str = "tts") -> str:
            async with zsvc.gpu_turn(turn):
                return await v.backend.synthesize(seg, v.lang, synth_ref, SynthParams(
                    out_path=os.path.join(jobdir, f"c{k}_s{i}_{tag}.wav"), temperature=v.temperature,
                    quality=v.quality, duration_factor=v.duration_factor, reference_text=ref_text))

        async def convert(k: int, i: int, path: str, tag: str = "") -> str:
            if zs is not None:
                dst = os.path.join(jobdir, f"c{k}_s{i}_vc{tag}.wav")
                await zs.convert(path, dst)
                return dst
            if worker is not None:
                dst = os.path.join(jobdir, f"c{k}_s{i}_rvc{tag}.wav")
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
                    log.info("book %s ch %d seg %d/%d (%s): %.2fs", bid, k + 1, i + 1, len(segs), kind,
                             time.time() - t0)
                    await todo.put((k, i, len(segs), path, seg, pause, kind))
            await todo.put(None)

        async def consume() -> None:
            async for k, i, n, path, seg, pause, kind in aio.queue_items(todo):
                files = [path]
                src = await convert(k, i, path)
                files.append(src)
                x, m = await aio.run_io(_segment_audio, src, len(seg))
                # (a title is read differently on purpose: it neither judges nor is judged)
                score, why = judge.deviation(m) if QUALITY_CHECKS and kind != "title" else (0.0, [])
                retried = None
                if score > 1.0:  # an outlier: made again once, the better take kept
                    session["retries"] += 1
                    p2 = await synth(k, i, seg, "tts2")
                    s2 = await convert(k, i, p2, "2")
                    files += [p2, s2]
                    x2, m2 = await aio.run_io(_segment_audio, s2, len(seg))
                    score2, why2 = judge.deviation(m2)
                    retried = {"why": why, "score": round(score, 2), "score2": round(score2, 2),
                               "kept": "second" if score2 < score else "first"}
                    log.info("book %s ch %d seg %d: outlier (%s), made again: %s (%.2f -> %.2f)", bid, k + 1, i + 1,
                             ", ".join(why), retried["kept"], score, score2)
                    if score2 < score:
                        x, m = x2, m2
                fix_db = judge.level_fix_db(m) if QUALITY_CHECKS else 0.0
                if fix_db:
                    x = x * np.float32(10 ** (fix_db / 20.0))
                if kind != "title":
                    judge.accept(m)
                stat = {"i": i, "kind": kind, "pause_ms": pause, "chars": len(seg), "dur": m["dur"],
                        "spc": round(m["spc"], 5), "lufs": None if m["lufs"] is None else round(m["lufs"], 2),
                        "f0": None if m["f0"] is None else round(m["f0"], 1), "fix_db": round(fix_db, 2),
                        "retried": retried}
                now = time.time()
                dt, last_at[0] = now - last_at[0], now
                # The commit finishes even when the job is stopped meanwhile: the PCM and its record stay together.
                fut = aio.run_io(_commit, data, bid, state, k, i, x, pause, len(seg), dt, stat)
                try:
                    added = await asyncio.shield(fut)
                except asyncio.CancelledError:
                    with contextlib.suppress(Exception):
                        await fut
                    raise
                session["audio_s"] += added
                session["elapsed_s"] += dt
                for p in set(files):
                    with contextlib.suppress(OSError):
                        await aio.run_io(os.remove, p)
                ctx.emit("book.progress", {**_progress(book, state, k),
                                          "segment_rtf": round(dt / added, 3) if added else None}, job=job)
                if i == n - 1:
                    out = await aio.run_io(_finish_chapter, data, bid, book, state, k, v.out_fmt)
                    cs = state["chapters"][k]
                    audio, el = float(cs.get("audio_s") or 0.0), float(cs.get("elapsed_s") or 0.0)
                    log.info("book %s chapter %d done: %.1fs of audio in %.1fs (RTF %.2f), level %s", bid, k + 1,
                             audio, el, el / audio if audio else 0.0, cs.get("level"))
                    ctx.emit("book.chapter", {"id": bid, "index": k, "title": book["chapters"][k]["title"],
                                              "path": out, "duration_s": cs.get("duration_s"), "audio_s": audio,
                                              "elapsed_s": round(el, 1), "rtf": round(el / audio, 3) if audio else None,
                                              "level": cs.get("level"), **_totals(book, state)}, job=job)
                    last_at[0] = time.time()

        await aio.gather_or_cancel(produce(), consume())
        m4b = None
        if book["settings"].get("m4b"):
            ctx.emit("book.progress", {**_progress(book, state, len(book["chapters"]) - 1), "stage": "m4b"}, job=job)
            m4b = await aio.run_io(_make_m4b, data, bid, book, state)
        state["status"] = "done"
        await aio.run_io(_save_state, data, bid, state)
        await aio.run_io(shutil.rmtree, _work(data, bid), ignore_errors=True)
        log.info("book %s done in %.1fs (this session: %.1fs of audio)", bid, time.time() - t_start,
                 session["audio_s"])
        end("book.done", {"folder": book["folder"], "m4b": m4b, "m4b_requested": bool(book["settings"].get("m4b")),
                          "chapters": len(book["chapters"]), **_totals(book, state),
                          "session_elapsed_s": round(time.time() - t_start, 1),
                          "session_audio_s": round(session["audio_s"], 2), "retries": session["retries"],
                          **choice.fields(), **v.ref_fields})
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
    except Exception as exc:  # noqa: BLE001
        log.exception("book %s failed", bid)
        await _failed(data, bid, state, "internal", f"{type(exc).__name__}: {exc}")
        end("book.error", {"code": "internal", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        _RUNNING.pop(bid, None)
        _STOP.pop(bid, None)
        tts.release_job(job)
        await aio.run_io(shutil.rmtree, jobdir, ignore_errors=True)


async def _failed(data: config.DataDir, bid: str, state: dict[str, Any] | None, code: str, message: str) -> None:
    if state is None:
        return
    state["status"] = "error"
    state["error"] = {"code": code, "message": message}
    with contextlib.suppress(Exception):
        await aio.run_io(_save_state, data, bid, state)
