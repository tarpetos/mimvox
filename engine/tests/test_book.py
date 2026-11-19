import asyncio
import json
import os
import threading
import time
import types

import numpy as np
import pytest
import soundfile as sf
from mimvox_engine import accent, book, config, narration

TEXT = ("The quick brown fox jumps over the lazy dog. " * 3 + "\n\n") * 3


class Crash(BaseException):
    pass


class Backend:
    def count_tokens(self, s):
        return max(1, len(s) // 4)

    async def synthesize(self, seg, lang, ref, p):
        sr = 24000
        t = np.arange(int(sr * 0.4)) / sr
        x = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.3 * np.sin(2 * np.pi * 3 * t))
        await asyncio.sleep(0)
        sf.write(p.out_path, x.astype(np.float32), sr)
        return p.out_path


class Ctx:
    def __init__(self, data_dir=""):
        self.data_dir = data_dir
        self.events = []

    def emit(self, event, data, job=None):
        self.events.append((event, data))

    def names(self, *wanted):
        return [e for e, _ in self.events if e in wanted]

    def chapters(self):
        return [d["index"] for e, d in self.events if e == "book.chapter"]


@pytest.fixture(autouse=True)
def no_quality_checks(monkeypatch):
    monkeypatch.setattr(book, "QUALITY_CHECKS", False)


@pytest.fixture
def data(tmp_path):
    os.makedirs(tmp_path / "tmp")
    return config.DataDir(str(tmp_path))


def make(data, read_titles=False):
    settings = {"voice_id": "v", "language": "en", "read_titles": read_titles, "output_format": "wav", "m4b": False}
    chapters = [{"title": f"Chapter {i + 1}", "text": TEXT, "words": 30} for i in range(2)]
    return book._create(data, "Test Book", None, None, chapters, settings)["id"]


def voice():
    return book._Voice(Backend(), "en", "", "", "", 0.0, None, accent.Choice("own", "auto", None), None, 0.0, 0.8,
                       "fast", 1.0, "wav", {})


def run(data, bid, on_start=None):
    """Runs the book job to its end; a ``Crash`` raised inside it is caught and recorded."""
    ctx = Ctx()

    async def main():
        job = f"job{time.time_ns()}"
        book._RUNNING[bid] = job
        task = asyncio.ensure_future(book._run(ctx, job, data, bid, voice()))
        if on_start:
            on_start(asyncio.get_running_loop(), task)
        try:
            await task
        except Crash:
            ctx.events.append(("crash", {}))

    asyncio.run(main())
    return ctx


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def chapters_on_disk(data, bid):
    b, st = book._load(data, bid)
    out = []
    for c, cs in zip(b["chapters"], st["chapters"]):
        f = os.path.join(b["folder"], c["file"] + ".wav")
        out.append((cs["status"], cs.get("segment"), cs.get("segments"), os.path.isfile(f)))
    return st["status"], out


def pause(task, bid, done=None):
    book._STOP[bid] = "pause"
    task.cancel()
    if done is not None:
        done.set()


def assert_finished(data, bid):
    status, chs = chapters_on_disk(data, bid)
    assert status == "done"
    assert all(s == "done" and file for s, _, _, file in chs)
    _, st = book._load(data, bid)
    assert all(os.path.isfile(cs["segments_path"]) for cs in st["chapters"])


def test_reads_a_book(data):
    bid = make(data, read_titles=True)
    ctx = run(data, bid)
    assert ctx.names("book.done", "book.paused", "book.error") == ["book.done"]
    assert ctx.chapters() == [0, 1]
    assert_finished(data, bid)
    b, st = book._load(data, bid)
    cs = st["chapters"][0]
    doc = read_json(cs["segments_path"])
    assert doc["path"] == cs["path"] and doc["chapter"] == 0 and doc["title"] == "Chapter 1"
    rows = doc["segments"]
    assert len(rows) == cs["segments"]
    assert rows[0] == {**rows[0], "kind": "title", "text": "Chapter 1", "start": narration.LEAD_MS / 1000}
    assert all("kind" not in r for r in rows[1:])
    assert all(a["end"] <= b["start"] for a, b in zip(rows, rows[1:]))
    assert all(r["start"] < r["end"] <= doc["duration_s"] for r in rows)
    w, sr = sf.read(cs["path"])
    assert abs(len(w) / sr - doc["duration_s"]) < 1e-3
    assert book._chapter_rows(b, st)[1]["segments_path"] == st["chapters"][1]["segments_path"]


def test_crash_after_the_last_commit_then_resume_finishes_the_chapter(data, monkeypatch):
    bid = make(data)
    real = book._finish_chapter

    def crash(*args):
        raise Crash()

    monkeypatch.setattr(book, "_finish_chapter", crash)
    first = run(data, bid)
    assert first.names("crash") == ["crash"]
    status, chs = chapters_on_disk(data, bid)
    assert status == "running" and chs[0][0] == "partial" and chs[0][1] == chs[0][2] and not chs[0][3]
    assert book.summary(data, *book._load(data, bid))["interrupted"] is True

    monkeypatch.setattr(book, "_finish_chapter", real)
    second = run(data, bid)
    assert second.names("book.done", "book.paused", "book.error") == ["book.done"]
    assert second.chapters() == [0, 1]
    assert_finished(data, bid)


def test_pause_after_the_last_commit_then_resume(data, monkeypatch):
    bid = make(data)
    real = book._commit
    holder = {}

    def commit_then_pause(d, b, st, k, i, *rest):
        out = real(d, b, st, k, i, *rest)
        if k == 0 and i == st["chapters"][0]["segments"] - 1:
            holder["loop"].call_soon_threadsafe(pause, holder["task"], b)
        return out

    monkeypatch.setattr(book, "_commit", commit_then_pause)
    first = run(data, bid, lambda loop, task: holder.update(loop=loop, task=task))
    assert first.names("book.done", "book.paused", "book.error") == ["book.paused"]
    assert first.chapters() == []
    status, chs = chapters_on_disk(data, bid)
    assert status == "paused" and chs[0][0] == "partial" and chs[0][1] == chs[0][2] and not chs[0][3]

    monkeypatch.setattr(book, "_commit", real)
    second = run(data, bid)
    assert second.chapters() == [0, 1]
    assert_finished(data, bid)


def test_pause_while_the_chapter_file_is_written(data, monkeypatch):
    bid = make(data)
    real = book._finish_chapter
    holder, order = {}, []
    cancelled = threading.Event()

    def finish_with_pause(d, b, bk, st, k, fmt):
        if k == 0:
            holder["loop"].call_soon_threadsafe(pause, holder["task"], b, cancelled)
            assert cancelled.wait(10)
        out = real(d, b, bk, st, k, fmt)
        order.append("finished")
        return out

    monkeypatch.setattr(book, "_finish_chapter", finish_with_pause)
    first = run(data, bid, lambda loop, task: holder.update(loop=loop, task=task))
    order += first.names("book.chapter", "book.paused")
    assert order == ["finished", "book.chapter", "book.paused"]
    status, chs = chapters_on_disk(data, bid)
    assert status == "paused" and chs[0][0] == "done" and chs[0][3] and chs[1][0] != "done"

    monkeypatch.setattr(book, "_finish_chapter", real)
    second = run(data, bid)
    assert second.chapters() == [1]
    assert_finished(data, bid)


def test_pause_while_a_segment_is_committed(data, monkeypatch):
    bid = make(data)
    real = book._commit
    holder, order = {}, []
    cancelled = threading.Event()

    def commit_with_pause(d, b, st, k, i, *rest):
        if k == 0 and i == 1:
            holder["loop"].call_soon_threadsafe(pause, holder["task"], b, cancelled)
            assert cancelled.wait(10)
        out = real(d, b, st, k, i, *rest)
        if k == 0 and i == 1:
            order.append("committed")
        return out

    monkeypatch.setattr(book, "_commit", commit_with_pause)
    first = run(data, bid, lambda loop, task: holder.update(loop=loop, task=task))
    order += first.names("book.paused")
    assert order == ["committed", "book.paused"]
    _, chs = chapters_on_disk(data, bid)
    assert chs[0][:2] == ("partial", 2)

    monkeypatch.setattr(book, "_commit", real)
    run(data, bid)
    assert_finished(data, bid)


def test_reread_and_cancel_remove_the_timing_file(data):
    bid = make(data)
    run(data, bid)
    _, st = book._load(data, bid)
    first, second = (cs["segments_path"] for cs in st["chapters"])
    ctx = Ctx(data.root)
    out = asyncio.run(book.book_reread(ctx, {"id": bid, "chapters": [0]}))
    assert out["chapter_list"][0]["segments_path"] is None
    assert not os.path.exists(first) and not os.path.exists(st["chapters"][0]["path"])
    assert os.path.isfile(second)

    book._write_json(first, {})
    asyncio.run(book.book_cancel(ctx, {"id": bid}))
    assert not os.path.exists(first) and os.path.isfile(second)
    _, st = book._load(data, bid)
    assert st["status"] == "cancelled" and st["chapters"][1]["status"] == "done"


def commit_chapter(data, bid, state, k, segs, lengths, rng):
    book._write_json(book._segs_path(data, bid, k), {"segments": [list(s) for s in segs]})
    state["chapters"][k].update(status="partial", segments=len(segs))
    for i, (text, pause_ms, kind) in enumerate(segs):
        x = (np.sign(rng.standard_normal(lengths[i])) * 0.3).astype(np.float32)
        stat = {"i": i, "kind": kind, "pause_ms": pause_ms, "chars": len(text), "dur": round(len(x) / book.SR, 3)}
        book._commit(data, bid, state, k, i, x, pause_ms, len(text), 0.1, stat)


def test_segment_timing_matches_the_audio(tmp_path):
    data = types.SimpleNamespace(root=str(tmp_path), outputs=str(tmp_path / "outputs"))
    bid = "b1"
    bk = {"id": bid, "title": "T", "folder": str(tmp_path / "out"), "settings": {"read_titles": True},
          "chapters": [{"title": "Chapter One", "text": "x", "file": "01 - One"},
                       {"title": "Two", "text": "y", "file": "02 - Two"}]}
    os.makedirs(tmp_path / "books" / bid / "work")
    state = book._new_state(bk)
    rng = np.random.default_rng(1)
    segs = [("Chapter One.", 1200, "title"), ("First sentence here.", 440, "sentence"), ("A clause,", 220, "clause"),
            ("Last one.", 700, "end")]
    commit_chapter(data, bid, state, 0, segs, [30001, 51234, 17777, 40000], rng)
    path = book._finish_chapter(data, bid, bk, state, 0, "wav")

    cs = state["chapters"][0]
    assert cs["segments_path"] == book._timing_path(data, bid, 0)
    assert not os.path.exists(book._segs_path(data, bid, 0))
    doc = read_json(cs["segments_path"])
    assert doc["path"] == path and doc["chapter"] == 0
    assert [r.get("kind") for r in doc["segments"]] == ["title", None, None, None]
    assert [r["text"] for r in doc["segments"]] == ["Chapter One", "First sentence here.", "A clause,", "Last one."]
    assert doc["segments"][0]["start"] == narration.LEAD_MS / 1000

    w, sr = sf.read(path)
    assert sr == book.SR and abs(len(w) / sr - doc["duration_s"]) < 1e-3
    ms = sr // 1000 + 1
    edges = [0] + [x for r in doc["segments"] for x in (round(r["start"] * sr), round(r["end"] * sr))] + [len(w)]
    for j in range(0, len(edges), 2):
        a, b = edges[j], edges[j + 1]
        assert not np.any(w[a + (ms if j else 0):max(a, b - ms)]), ("silence expected", j)
    for j in range(1, len(edges) - 1, 2):
        a, b = edges[j], edges[j + 1]
        assert np.all(w[a + ms:b - ms] != 0), ("speech expected", j)
        assert w[a - ms] == 0 and w[a + ms] != 0 and w[b + ms] == 0 and w[b - ms] != 0

    commit_chapter(data, bid, state, 1, [("A very long", 220, "clause"), ("title.", 1200, "title"),
                                         ("Body.", 700, "end")], [9000, 9000, 9000], rng)
    book._finish_chapter(data, bid, bk, state, 1, "wav")
    doc = read_json(state["chapters"][1]["segments_path"])
    assert [r.get("kind") for r in doc["segments"]] == ["title", "title", None]
    assert doc["segments"][0]["text"] == "A very long"


def test_timeline_needs_every_stat():
    assert book.timeline([["a", 440, "end"]], [], 3.0) is None
    assert book.timeline([["a", 440, "end"]], [{"i": 0, "at": 1.6, "dur": 1.0}], 3.0) == [
        {"text": "a", "start": 1.6, "end": 2.6}]
    assert book.timeline([["a", 440, "end"]], [{"i": 0, "at": 1.6, "dur": 9.0}], 3.0)[0]["end"] == 3.0


@pytest.fixture
def starts(monkeypatch):
    calls = []

    async def fake_voice(ctx, data, st):
        await asyncio.sleep(0)
        return voice()

    def fake_start(ctx, data, bid, v):
        job = f"job{len(calls) + 1}"
        calls.append(bid)
        book._RUNNING[bid] = job
        return job

    monkeypatch.setattr(book.tts, "speech_language", lambda params, key="language": ("en", Backend()))
    monkeypatch.setattr(book, "_voice", fake_voice)
    monkeypatch.setattr(book, "_start", fake_start)
    yield calls
    book._RUNNING.clear()


def start(data, ref=None, n=2, **params):
    chapters = [{"title": f"Chapter {i + 1}", "text": TEXT} for i in range(n)]
    p = {"title": "Test Book", "voice_id": "v", "language": "en", "chapters": chapters, **params}
    if ref is not None:
        p["ref"] = ref
    return asyncio.run(book.book_start(Ctx(data.root), p))


def books(data):
    return asyncio.run(book.book_list(Ctx(data.root), {}))["books"]


def set_status(data, bid, status):
    _, st = book._load(data, bid)
    st["status"] = status
    book._save_state(data, bid, st)


def test_start_with_a_ref_stores_it(data, starts):
    out = start(data, "pub-1")
    assert out["ref"] == "pub-1" and out["existing"] is False and out["status"] == "running"
    assert starts == [out["id"]]
    assert read_json(os.path.join(book.book_dir(data, out["id"]), "book.json"))["ref"] == "pub-1"
    assert [b["ref"] for b in books(data)] == ["pub-1"]
    assert asyncio.run(book.book_get(Ctx(data.root), {"id": out["id"]}))["ref"] == "pub-1"


def test_start_without_a_ref_always_creates(data, starts):
    first, second = start(data), start(data)
    assert first["id"] != second["id"] and first["ref"] is None
    assert len(books(data)) == 2 and all(b["ref"] is None for b in books(data))


def test_start_with_the_ref_of_a_running_book_returns_its_job(data, starts):
    first = start(data, "pub-1")
    again = start(data, "pub-1")
    assert again["id"] == first["id"] and again["job"] == first["job"]
    assert again["existing"] is True and again["status"] == "running"
    assert starts == [first["id"]] and len(books(data)) == 1


@pytest.mark.parametrize("status", ["paused", "new", "running", "error", "cancelled"])
def test_start_with_the_ref_of_a_stopped_book_resumes_it(data, starts, status):
    first = start(data, "pub-1")
    book._RUNNING.clear()
    set_status(data, first["id"], status)  # "running" without a job: interrupted
    again = start(data, "pub-1", n=1)
    assert again["id"] == first["id"] and again["existing"] is True and again["status"] == "running"
    assert again["job"] == "job2" and starts == [first["id"], first["id"]]
    assert again["chapters"] == 2 and again["accent"] == "own"
    assert len(books(data)) == 1


def test_start_with_the_ref_of_a_finished_book_starts_nothing(data, starts):
    first = start(data, "pub-1")
    book._RUNNING.clear()
    set_status(data, first["id"], "done")
    again = start(data, "pub-1")
    assert again == {"id": first["id"], "folder": first["folder"], "chapters": 2, "words": first["words"],
                     "ref": "pub-1", "existing": True, "job": None, "status": "done"}
    assert starts == [first["id"]] and len(books(data)) == 1


def test_another_ref_with_other_chapters_is_another_book(data, starts):
    first = start(data, "pc-a", n=3)
    book._RUNNING.clear()
    set_status(data, first["id"], "paused")
    taken = start(data, "pc-b", n=1)
    assert taken["id"] != first["id"] and taken["existing"] is False and taken["chapters"] == 1
    assert sorted((b["ref"], b["chapters"]) for b in books(data)) == [("pc-a", 3), ("pc-b", 1)]
    book._RUNNING.clear()
    again = start(data, "pc-a", n=1)
    assert again["id"] == first["id"] and again["chapters"] == 3


def test_the_same_ref_twice_at_once_starts_one_book(data, starts):
    chapters = [{"title": "One", "text": TEXT}]
    p = {"title": "T", "voice_id": "v", "language": "en", "chapters": chapters, "ref": "pub-1"}

    async def both():
        return await asyncio.gather(*(book.book_start(Ctx(data.root), dict(p)) for _ in range(2)),
                                    return_exceptions=True)

    a, b = asyncio.run(both())
    assert isinstance(a, dict) and isinstance(b, book.EngineError) and b.code == "busy"
    assert len(books(data)) == 1 and start(data, "pub-1")["id"] == a["id"]


@pytest.mark.parametrize("ref", ["", "  ", 5, ["x"]])
def test_a_bad_ref_is_refused(data, starts, ref):
    with pytest.raises(book.EngineError) as e:
        start(data, ref)
    assert e.value.code == "bad_request" and books(data) == []
