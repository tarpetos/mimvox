import asyncio
import mmap
import os
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf
from mimvox_engine import aio, book, speech, worker_kit
from mimvox_engine.backends import SynthParams, omnivoice
from mimvox_engine.registry import EngineError
from mimvox_engine.workerproc import WorkerClient, WorkerDied, WorkerError

FAKE_WORKER = textwrap.dedent("""
    import json, os, sys, time
    print(json.dumps({"event": "ready"}), flush=True)
    for line in iter(sys.stdin.readline, ""):
        req = json.loads(line)
        if req.get("op") == "quit":
            break
        if req.get("op") == "cancel":
            continue
        if req.get("exit_once") and not os.path.exists(req["exit_once"]):
            open(req["exit_once"], "w").close()
            sys.exit(0)
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
    assert info.value.code == "no_memory"
    assert book._short_of_memory(info.value)


def test_too_little_ram_in_all_is_not_worth_a_book_retry(tmp_path):
    msg = worker_kit.NO_RAM_CPU_TOTAL.format(need=11, total=7.4)
    b = _backend("cpu", error=WorkerError("no_memory", msg, permanent=True))
    with pytest.raises(EngineError) as info:
        _synth(b, tmp_path)
    assert info.value.code == "no_memory" and info.value.message == f"OmniVoice: {msg}"
    assert not book._short_of_memory(info.value)


def test_no_memory_keeps_its_code_from_warm_and_the_speech_worker(tmp_path):
    b = _backend("cpu", error=WorkerError("no_memory", "not enough memory (RAM)"))
    with pytest.raises(EngineError) as info:
        asyncio.run(b.warm(str(tmp_path / "reference.wav"), reference_text="hello"))
    assert info.value.code == "no_memory"
    b = _backend("cpu", error=WorkerError("internal", "boom"))
    with pytest.raises(EngineError) as info:
        asyncio.run(b.warm(str(tmp_path / "reference.wav"), reference_text="hello"))
    assert info.value.code == "internal"
    assert speech.SpeechError("no_memory", "full").engine_error().code == "no_memory"
    assert speech.SpeechError("weird", "x").engine_error().code == "internal"


def test_ram_too_small(monkeypatch):
    monkeypatch.setattr(worker_kit, "physical_memory", lambda: (int(7.4 * GB), 2 * GB))
    msg = worker_kit.ram_too_small(11)
    assert msg is not None and "at least 11 GB of RAM and this one has 7.4 GB" in msg
    monkeypatch.setattr(worker_kit, "physical_memory", lambda: (int(11.7 * GB), 0))
    assert worker_kit.ram_too_small(11) is None   # a full 12 GB computer is the other check's business
    monkeypatch.setattr(worker_kit, "physical_memory", lambda: None)
    assert worker_kit.ram_too_small(11) is None


def test_a_worker_that_dies_is_retried_once(tmp_path):
    async def body(client):
        res = await client.call("synth", {"exit_once": str(tmp_path / "died")})
        return res, client.alive()

    assert _run(tmp_path, body) == ({"op": "synth"}, True)


def test_no_retry_or_restart_while_the_engine_shuts_down(tmp_path, monkeypatch):
    starts = []

    async def body(client):
        spawn = client._spawn

        async def counted():
            starts.append(1)
            await spawn()

        client._spawn = counted
        await client.call("ping", {})
        monkeypatch.setattr(aio, "_closing", [True])
        with pytest.raises(WorkerDied, match="exited"):
            await client.call("synth", {"exit_once": str(tmp_path / "died")})
        with pytest.raises(WorkerDied, match="shutting down"):
            await client.call("synth", {})
        return client.alive()

    assert _run(tmp_path, body) is False
    assert len(starts) == 1


STUB_TORCH = textwrap.dedent("""
    import os, types
    float16, float32 = "float16", "float32"
    _CUDA = os.environ.get("FAKE_CUDA") == "1"

    class _Cuda:
        def is_available(self):
            return _CUDA

        def is_initialized(self):
            return _CUDA

        def mem_get_info(self):
            return 8 << 30, 12 << 30

        def memory_reserved(self):
            return 0

        def __getattr__(self, name):
            return lambda *a, **k: None

    cuda = _Cuda()
    backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    nn = types.SimpleNamespace(modules=types.SimpleNamespace(module=types.SimpleNamespace(
        register_module_forward_pre_hook=lambda hook: None)))

    def from_numpy(x):
        return x
""")

STUB_OMNIVOICE = textwrap.dedent("""
    import os
    import numpy as np

    class OmniVoice:
        sampling_rate = 24000

        @classmethod
        def from_pretrained(cls, path, device_map=None, dtype=None):
            open(os.environ["FAKE_LOADED"], "w").close()
            return cls()

        def parameters(self):
            return []

        def create_voice_clone_prompt(self, audio, ref_text=None):
            return "prompt"

        def generate(self, **kw):
            return [np.zeros(2400, dtype=np.float32)]
""")

LAUNCH = "import runpy, sys; sys.path.insert(0, sys.argv[1]); sys.argv = sys.argv[2:]; runpy.run_path(sys.argv[0], " \
         "run_name='__main__')"


def _omnivoice_worker(tmp_path, device, need_gb, ops):
    """The real OmniVoice worker script over stub torch and omnivoice modules: the result or the error code per op."""
    stubs = tmp_path / "stubs"
    (stubs / "torch").mkdir(parents=True, exist_ok=True)
    (stubs / "omnivoice").mkdir(exist_ok=True)
    (stubs / "torch" / "__init__.py").write_text(STUB_TORCH, encoding="utf-8")
    (stubs / "omnivoice" / "__init__.py").write_text(STUB_OMNIVOICE, encoding="utf-8")
    ref = tmp_path / "reference.wav"
    sf.write(str(ref), np.zeros(24000, dtype=np.float32), 24000)
    env = dict(os.environ, MIMVOX_OMNIVOICE_FAST="0", MIMVOX_CPU_RAM_NEED_GB=str(need_gb),
               FAKE_CUDA="1" if device == "cuda" else "0", FAKE_LOADED=str(tmp_path / "loaded"))
    argv = [sys.executable, "-c", LAUNCH, str(stubs), omnivoice.WORKER, "--model-dir", str(tmp_path / "model"),
            "--device", device]
    params = {"ref": str(ref), "ref_text": "hello", "text": "Hello there.", "out": str(tmp_path / "out.wav"),
              "normalize": False}

    async def main():
        client = WorkerClient("omnivoice", argv, env, start_timeout=60.0)
        answers = []
        try:
            for op in ops:
                try:
                    answers.append(await client.call(op, params, timeout=60.0, retry=False))
                except WorkerError as exc:
                    answers.append((exc.code, exc.permanent))
        finally:
            await client.close()
        return answers

    return asyncio.run(main())


needs_ram_figure = pytest.mark.skipif(worker_kit.physical_memory() is None, reason="RAM size not known here")


@needs_ram_figure
def test_cpu_refuses_every_pass_up_front_on_a_computer_with_too_little_ram(tmp_path):
    assert _omnivoice_worker(tmp_path, "cpu", 10**6, ["warm", "synth", "inspect"]) == [("no_memory", True)] * 3
    assert not (tmp_path / "loaded").exists()   # refused before the model was loaded


@needs_ram_figure
def test_cpu_speaks_when_the_ram_is_enough(tmp_path):
    warm, synth = _omnivoice_worker(tmp_path, "cpu", 0, ["warm", "synth"])
    assert "elapsed_s" in warm and synth["out"] == str(tmp_path / "out.wav")


@needs_ram_figure
def test_the_gpu_skips_the_ram_check(tmp_path):
    (synth,) = _omnivoice_worker(tmp_path, "cuda", 10**6, ["synth"])
    assert synth["out"] == str(tmp_path / "out.wav")
