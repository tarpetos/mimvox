"""Fake engine that speaks contract/engine-protocol.md, for testing the Rust client without the ML stack.

Run by the client as `python -m mimvox_engine serve --runtime X --data-dir Y` with
PYTHONPATH=examples/fake_engine_src (the shim package forwards here). tts.generate "speaks" by writing
sine-tone WAV chunks (one per sentence) and emitting tts.chunk / tts.done.
"""
import json
import math
import os
import struct
import sys
import threading
import time
import wave

OUT_LOCK = threading.Lock()
CANCELLED = set()
JOB_COUNTER = [0]
BOOKS = {}
VOICES = [
    {"id": "fake-voice", "name": "Fake voice", "created": "2026-01-01T00:00:00Z", "preset": "standard",
     "dataset_minutes": 6.4, "parts": 5, "trained": True, "trained_epochs": 300}
]


def send(obj):
    with OUT_LOCK:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()


def log(msg):
    sys.stderr.write(f"[fake-engine] {msg}\n")
    sys.stderr.flush()


def hello():
    return {"version": "0.1.0-fake", "device": {"backend": "cpu", "name": "Fake CPU", "vram_gb": 0.0},
            "models": {"tts": "ready", "rvc": "ready"}}


def write_tone(path, seconds, freq, rate=24000):
    n = int(seconds * rate)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            fade = min(1.0, i / 400, (n - i) / 400)
            frames += struct.pack("<h", int(9000 * fade * math.sin(2 * math.pi * freq * i / rate)))
        w.writeframes(bytes(frames))


def tts_job(data_dir, job, params):
    text = params["text"].strip()
    segments = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()] or [text]
    tmp = os.path.join(data_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    started = time.time()
    total = 0.0
    for i, seg in enumerate(segments):
        time.sleep(0.4)  # pretend to synthesise
        if job in CANCELLED:
            send({"event": "tts.error", "job": job, "data": {"code": "cancelled", "message": "cancelled"}})
            return
        dur = max(0.6, min(3.0, len(seg) * 0.06))
        path = os.path.join(tmp, f"{job}_{i}.wav")
        write_tone(path, dur, 330 + 60 * i)
        total += dur
        send({"event": "tts.chunk", "job": job, "data": {"index": i, "path": os.path.abspath(path), "duration_s": dur}})
    out = os.path.join(data_dir, "outputs")
    os.makedirs(out, exist_ok=True)
    final = os.path.join(out, f"{job}.wav")
    write_tone(final, total, 330, rate=44100)
    send({"event": "tts.done", "job": job, "data": {"path": os.path.abspath(final), "path_play": os.path.abspath(final),
                                                    "duration_s": total, "elapsed_s": time.time() - started}})


def next_job():
    JOB_COUNTER[0] += 1
    return "job%d" % JOB_COUNTER[0]


def import_text(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    chapters, title = [], None
    for line in lines:
        if line.startswith("Chapter "):
            chapters.append({"title": line.strip(), "text": ""})
        elif chapters:
            chapters[-1]["text"] = (chapters[-1]["text"] + "\n" + line).strip()
        elif line.strip() and title is None:
            title = line.strip()
    for c in chapters:
        c["words"] = len(c["text"].split())
    return {"path": path, "name": os.path.basename(path), "format": "txt", "title": title, "language": "en",
            "words": sum(c["words"] for c in chapters), "chapters": chapters, "parts": None, "warnings": [],
            "meta": {"authors": [], "description": None, "language_tag": None, "date": None, "cover": None}}


def book_summary(book):
    done = sum(1 for c in book["chapters"] if c.get("done"))
    return {"id": book["id"], "title": book["title"], "status": book["status"], "job": book.get("job"),
            "chapters": len(book["chapters"]), "chapters_done": done,
            "percent": 100.0 * done / len(book["chapters"])}


def book_job(book, job):
    n = len(book["chapters"])
    for k, chapter in enumerate(book["chapters"]):
        if chapter.get("done"):
            continue
        time.sleep(0.4)  # pretend to read the chapter
        if book.get("pause"):
            book["status"], book["pause"] = "paused", False
            send({"event": "book.paused", "job": job, "data": book_summary(book)})
            return
        chapter["done"] = True
        summary = book_summary(book)
        send({"event": "book.progress", "job": job,
              "data": {"id": book["id"], "chapter": k, "chapters": n, "percent": summary["percent"]}})
        send({"event": "book.chapter", "job": job, "data": {"id": book["id"], "index": k, "title": chapter["title"]}})
    book["status"] = "done"
    send({"event": "book.done", "job": job, "data": {"id": book["id"], "folder": book["folder"], "chapters": n}})


def start_book(data_dir, book):
    job = next_job()
    book["status"], book["job"] = "running", job
    threading.Thread(target=book_job, args=(book, job), daemon=True).start()
    return {"id": book["id"], "job": job, "folder": book["folder"], "chapters": len(book["chapters"])}


def handle(data_dir, req):
    rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
    try:
        if method == "hello":
            result = hello()
        elif method == "voices.list":
            result = {"voices": VOICES}
        elif method == "voices.create":
            v = {"id": "v%d" % (len(VOICES) + 1), "name": params["name"], "created": "now", "preset": None,
                 "dataset_minutes": 0.0, "parts": 0, "trained": False, "trained_epochs": None}
            VOICES.append(v)
            result = {"voice": v}
        elif method == "presets.list":
            result = {"presets": [{"id": "quick", "title": "Quick", "description": "1 minute", "target_minutes": 1,
                                   "trains_rvc": False,
                                   "parts": [{"index": 1, "title": "Part 1", "text": "Hello there.", "est_seconds": 5}]}]}
        elif method == "text.analyze":
            words = len(params["text"].split())
            result = {"words": words, "est_minutes": words / 150, "phoneme_coverage": 0.9, "missing_phonemes": ["ZH"],
                      "level": "zero_shot", "message": "fake", "parts": []}
        elif method == "tts.generate":
            if not any(v["id"] == params.get("voice_id") for v in VOICES):
                raise KeyError("voice")
            job = next_job()
            threading.Thread(target=tts_job, args=(data_dir, job, params), daemon=True).start()
            result = {"job": job}
        elif method == "tts.cancel":
            CANCELLED.add(params.get("job"))
            result = {}
        elif method == "outputs.list":
            out = os.path.join(data_dir, "outputs")
            names = sorted(os.listdir(out), reverse=True) if os.path.isdir(out) else []
            result = {"results": [{"id": os.path.splitext(n)[0], "kind": "speak", "name": n,
                                   "path": os.path.abspath(os.path.join(out, n)), "duration_s": 1.0,
                                   "voice_name": "Fake voice", "text": "Hello there."}
                                  for n in names if n.endswith(".wav")], "more": False}
        elif method == "text.import":
            result = import_text(params["path"])
        elif method == "book.start":
            if not any(v["id"] == params.get("voice_id") for v in VOICES):
                raise KeyError("voice")
            book_id = "book%d" % (len(BOOKS) + 1)
            book = {"id": book_id, "title": params["title"], "params": params,
                    "chapters": [dict(c) for c in params["chapters"]],
                    "folder": os.path.abspath(os.path.join(data_dir, "outputs", "books", params["title"]))}
            BOOKS[book_id] = book
            result = start_book(data_dir, book)
        elif method == "book.pause":
            BOOKS[params["id"]]["pause"] = True
            result = {}
        elif method == "book.resume":
            result = start_book(data_dir, BOOKS[params["id"]])
        elif method == "book.get":
            book = BOOKS[params["id"]]
            result = {**book_summary(book), "settings": book["params"],
                      "chapter_list": [{"index": i, "title": c["title"], "status": "done" if c.get("done") else "waiting"}
                                       for i, c in enumerate(book["chapters"])]}
        elif method == "book.list":
            result = {"books": [book_summary(b) for b in BOOKS.values()]}
        elif method == "shutdown":
            send({"id": rid, "result": {}})
            log("shutdown requested")
            os._exit(0)
        else:
            send({"id": rid, "error": {"code": "not_found", "message": f"unknown method {method}"}})
            return
        send({"id": rid, "result": result})
    except KeyError as e:
        send({"id": rid, "error": {"code": "not_found", "message": f"not found: {e}"}})
    except Exception as e:  # noqa: BLE001
        send({"id": rid, "error": {"code": "internal", "message": str(e)}})


def main():
    args = sys.argv[1:]
    if not args or args[0] != "serve":
        log("usage: serve --runtime X --data-dir Y")
        return 2
    data_dir = args[args.index("--data-dir") + 1]
    log(f"starting, data dir {data_dir}")
    time.sleep(0.3)
    send({"event": "ready", "data": hello()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            log(f"bad json: {line}")
            continue
        handle(data_dir, req)
    log("stdin closed, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
