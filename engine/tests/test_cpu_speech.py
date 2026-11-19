import asyncio
import mmap
import sys
import textwrap

import pytest
from mimvox_engine import book, worker_kit
from mimvox_engine.backends import SynthParams, omnivoice
from mimvox_engine.registry import EngineError
from mimvox_engine.workerproc import WorkerClient, WorkerError

FAKE_WORKER = textwrap.dedent("""
    import json, sys, time
    print(json.dumps({"event": "ready"}), flush=True)
    for line in iter(sys.stdin.readline, ""):
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        if req.get("op") == "cancel":
            continue
        for _ in range(req.get("beats", 0)):
            time.sleep(req["every"])
            print(json.dumps({"id": req["id"], "progress": {"alive": True}}), flush=True)
        time.sleep(req.get("sleep", 0))
        print(json.dumps({"id": req["id"], "ok": True, "result": {"op": req["op"]}}), flush=True)
""")

GB = 2**30


def _run(tmp_path, body):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")

    async def main():
        client = WorkerClient("fake", [sys.executable, "-I", str(script)], {}, start_timeout=30.0)
        try:
            return await body(client)
        finally:
            await client.close()

    return asyncio.run(main())


def test_progress_keeps_a_slow_call_alive_past_the_stall_limit(tmp_path):
    async def body(client):
        return await client.call("synth", {"beats": 8, "every": 0.1}, stall=0.5)

    assert _run(tmp_path, body) == {"op": "synth"}


def test_silence_ends_the_call_and_a_new_worker_takes_the_next(tmp_path):
    async def body(client):
        pid = None
        with pytest.raises(WorkerError, match="made no progress for 0 s in synth"):
            call = asyncio.ensure_future(client.call("synth", {"sleep": 5}, stall=0.3))
            await asyncio.sleep(0.1)
            pid = client.pid
            await call
        return pid, await client.call("synth", {}, stall=0.3), client.pid

    old, res, new = _run(tmp_path, body)
    assert res == {"op": "synth"} and old != new


def test_timeout_caps_a_call_that_reports_progress(tmp_path):
    async def body(client):
        with pytest.raises(WorkerError, match="timed out in synth"):
            await client.call("synth", {"beats": 20, "every": 0.1}, timeout=0.5, stall=5.0)

    _run(tmp_path, body)


def test_a_fixed_timeout_alone_still_works(tmp_path):
    async def body(client):
        return await client.call("synth", {"sleep": 0.2}, timeout=5.0)

    assert _run(tmp_path, body) == {"op": "synth"}


@pytest.mark.parametrize(("total", "available", "out", "short"), [
    (8 * GB, 0, 3 * GB, True),                    # the QA laptop: nothing available, the model compressed away
    (8 * GB, int(0.4 * GB), int(1.5 * GB), True),
    (8 * GB, 2 * GB, 3 * GB, False),              # paged out but RAM free again: the pages come back once
    (8 * GB, 0, int(0.5 * GB), False),            # RAM full, the model itself in it
    (64 * GB, 2 * GB, 2 * GB, True),              # 5 % of 64 GB is above the 0.5 GB floor
    (64 * GB, 4 * GB, 2 * GB, False),
])
def test_thrashing(total, available, out, short):
    assert worker_kit.thrashing(total, available, out) is short


def test_ram_short_counts_the_weights_outside_ram(monkeypatch):
    ranges = [(0x10000, 2 * GB), (0x90000000, 2 * GB)]
    monkeypatch.setattr(worker_kit, "physical_memory", lambda: (8 * GB, 0))
    monkeypatch.setattr(worker_kit, "resident_share", lambda r: 0.25)
    msg = worker_kit.ram_short(ranges)
    assert msg is not None and "3.0 GB of the speech model" in msg and "8 GB of RAM (0.0 GB free)" in msg
    monkeypatch.setattr(worker_kit, "resident_share", lambda r: 0.9)
    assert worker_kit.ram_short(ranges) is None
    monkeypatch.setattr(worker_kit, "resident_share", lambda r: pytest.fail("not asked while RAM is free"))
    monkeypatch.setattr(worker_kit, "physical_memory", lambda: (8 * GB, 3 * GB))
    assert worker_kit.ram_short(ranges) is None


def test_resident_share_of_touched_memory():
    import ctypes

    buf = mmap.mmap(-1, 64 * worker_kit.PAGE)
    buf.write(b"x" * len(buf))
    addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    share = worker_kit.resident_share([(addr, len(buf))])
    if share is None:
        pytest.skip("no residency query on this platform")
    assert share == 1.0
    fresh = mmap.mmap(-1, 64 * worker_kit.PAGE)  # never touched: no page of it is in RAM yet
    assert worker_kit.resident_share([(ctypes.addressof(ctypes.c_char.from_buffer(fresh)), len(fresh))]) == 0.0


def test_physical_memory():
    mem = worker_kit.physical_memory()
    if mem is None:
        pytest.skip("not known on this platform")
    assert 0 < mem[1] <= mem[0]


class StubClient:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def call(self, op, params=None, progress=None, timeout=None, start=True, retry=True, stall=None):
        self.calls.append((op, timeout, stall))
        if self.error is not None:
            raise self.error
        return {"out": params["out"], "duration_s": 1.0}


def _backend(device, loaded=None, error=None):
    b = omnivoice.OmniVoiceBackend()
    b._client, b._device, b._status = StubClient(error), device, "ready"
    if loaded:
        b.last_load = {"device": loaded}
    return b


def _synth(b, tmp_path):
    ref = tmp_path / "reference.wav"
    ref.write_bytes(b"")
    params = SynthParams(out_path=str(tmp_path / "out.wav"), reference_text="hello")
    return asyncio.run(b.synthesize("Hello there.", "en", str(ref), params))


def test_gpu_keeps_the_fixed_limit(tmp_path):
    b = _backend("cuda")
    _synth(b, tmp_path)
    assert b._client.calls == [("synth", omnivoice.SYNTH_TIMEOUT_S, None)]


@pytest.mark.parametrize(("device", "loaded"), [("cpu", None), ("cuda", "cpu")])
def test_cpu_has_no_fixed_limit_only_the_stall_watchdog(tmp_path, device, loaded):
    b = _backend(device, loaded)
    _synth(b, tmp_path)
    assert b._client.calls == [("synth", None, omnivoice.CPU_STALL_S)]


def test_a_thrashing_cpu_worker_is_short_of_memory_for_a_book(tmp_path):
    msg = worker_kit.NO_RAM_CPU.format(out=3.0, total=8, free=0.0)
    b = _backend("cpu", error=WorkerError("no_memory", msg))
    with pytest.raises(EngineError, match="not enough free memory") as info:
        _synth(b, tmp_path)
    assert book._short_of_memory(info.value)
