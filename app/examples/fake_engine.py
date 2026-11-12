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
    write_tone(final, total, 330)
    send({"event": "tts.done", "job": job, "data": {"path": os.path.abspath(final), "path_play": os.path.abspath(final),
                                                    "duration_s": total, "elapsed_s": time.time() - started}})


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
            JOB_COUNTER[0] += 1
            job = "job%d" % JOB_COUNTER[0]
            threading.Thread(target=tts_job, args=(data_dir, job, params), daemon=True).start()
            result = {"job": job}
        elif method == "tts.cancel":
            CANCELLED.add(params.get("job"))
            result = {}
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
