"""JSON-lines server: ``python -m mimvox_engine serve --runtime <runtime.toml> --data-dir <dir>``.

Everything runs on one asyncio event loop (see ``aio.py``):

- stdin: the protocol pipe is detached from the process's standard input (``aio.detach_stdin``) and read line by line
  as it arrives (``aio.read_lines``); each request becomes a task.
- stdout: the real stdout is moved to a private descriptor and fd 1 / ``sys.stdout`` point at stderr, so library prints
  (also from C code) cannot corrupt the stream. Results and events go through one queue drained by a single writer
  coroutine (lines keep their order; a slow reader never blocks the loop: the bytes are written on a dedicated thread).
- jobs are tasks; ``*.cancel`` cancels the task. Shutdown cancels jobs, stops the workers and Applio (process exits are
  awaited) and only then leaves.
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import importlib
import inspect
import itertools
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable
from typing import Any, BinaryIO

from . import __version__, aio, config, media
from .registry import EngineError, handlers, method, methods

log = logging.getLogger("mimvox.server")

SHUTDOWN_GRACE_S = 20.0


class Server:
    """Implements ``registry.Context``. Created inside the running loop."""

    def __init__(self, out: BinaryIO, runtime: dict[str, Any], data_dir: str) -> None:
        self.loop = asyncio.get_running_loop()
        aio.bind_loop(self.loop)
        self._out = out
        self._outq: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._out_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="stdout")
        self._writer = aio.spawn(self._write_lines(), name="stdout-writer")
        self.runtime = runtime
        self.data_dir = data_dir
        self._job_ids = itertools.count(1)
        self._jobs: dict[str, asyncio.Task[Any]] = {}
        self._cancelled: set[str] = set()
        self._requests: set[asyncio.Task[Any]] = set()
        self._closing = False
        self.stopped: asyncio.Future[None] = self.loop.create_future()
        self.device: dict[str, Any] = {"backend": config.backend(runtime), "name": config.backend(runtime).upper(),
                                       "vram_gb": 0.0}

    # ---- output ----
    def _put(self, obj: dict[str, Any]) -> None:
        line = (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        aio.call_soon(self._outq.put_nowait, line)

    async def _write_lines(self) -> None:
        async for line in aio.queue_items(self._outq):
            batch = [line]
            for _ in range(self._outq.qsize()):  # whatever queued meanwhile goes out in the same write
                nxt = self._outq.get_nowait()
                if nxt is None:
                    self._outq.put_nowait(None)
                    break
                batch.append(nxt)
            await self.loop.run_in_executor(self._out_exec, self._write_bytes, b"".join(batch))

    def _write_bytes(self, data: bytes) -> None:
        try:
            self._out.write(data)
            self._out.flush()
        except (OSError, ValueError):
            log.warning("the app closed the protocol channel")
            self.loop.call_soon_threadsafe(self.request_shutdown)

    async def flush(self) -> None:
        self._outq.put_nowait(None)
        try:
            await asyncio.wait_for(self._writer, 10.0)
        except asyncio.TimeoutError:
            log.warning("stdout writer did not finish")

    def emit(self, event: str, data: dict[str, Any], job: str | None = None) -> None:
        self._put({"event": event, "job": job, "data": data})

    def _result(self, rid: Any, result: dict[str, Any]) -> None:
        self._put({"id": rid, "result": result})

    def _error(self, rid: Any, code: str, message: str) -> None:
        self._put({"id": rid, "error": {"code": code, "message": message}})

    # ---- jobs ----
    def start_job(self, fn: Callable[[str], Any]) -> str:
        job = f"job-{next(self._job_ids)}"
        aio.call_soon(self._create_job, job, fn)
        return job

    def _create_job(self, job: str, fn: Callable[[str], Any]) -> None:
        if inspect.iscoroutinefunction(fn):
            body = fn(job)
        else:
            body = self.loop.run_in_executor(aio.IO, fn, job)
        self._jobs[job] = self.loop.create_task(self._run_job(job, body), name=job)

    async def _run_job(self, job: str, body: Any) -> None:
        try:
            await body
        except asyncio.CancelledError:
            if not self._closing:
                log.info("job %s cancelled", job)
        except Exception as exc:  # noqa: BLE001 - jobs report their own errors; this is the last resort
            log.exception("job %s crashed", job)
            self.emit("job.error", {"code": "internal", "message": f"{type(exc).__name__}: {exc}"}, job=job)
        finally:
            self._jobs.pop(job, None)
            self._cancelled.discard(job)

    def cancelled(self, job: str) -> bool:
        return job in self._cancelled

    def cancel(self, job: str) -> None:
        aio.call_soon(self._cancel, job)

    def _cancel(self, job: str) -> None:
        task = self._jobs.get(job)  # (start_job schedules the task before it returns the id, so it exists by now)
        if task is not None and not task.done():
            self._cancelled.add(job)
            task.cancel()

    def job_running(self, job: str) -> bool:
        return job in self._jobs

    # ---- requests ----
    def on_line(self, raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or self._closing:
            return
        task = self.loop.create_task(self._handle(line))
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    async def _handle(self, line: str) -> None:
        rid: Any = None
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("request must be an object")
            rid = req.get("id")
            name = req.get("method")
            params = req.get("params") or {}
            if not isinstance(name, str) or not isinstance(params, dict):
                raise EngineError("bad_request", "need a string 'method' and an object 'params'")
            if name == "shutdown":
                self._result(rid, {})
                self.request_shutdown()
                return
            handler = handlers().get(name)
            if handler is None:
                raise EngineError("not_found", f"unknown method {name!r}")
            if inspect.iscoroutinefunction(handler):
                result = await handler(self, params)
            else:
                result = await self.loop.run_in_executor(aio.IO, handler, self, params)
            self._result(rid, result if isinstance(result, dict) else {})
        except EngineError as exc:
            self._error(rid, exc.code, exc.message)
        except ValueError as exc:
            self._error(rid, "bad_request", f"invalid request: {exc}")
        except asyncio.CancelledError:
            self._error(rid, "cancelled", "the engine is shutting down")
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("request failed: %s", traceback.format_exc())
            self._error(rid, "internal", f"{type(exc).__name__}: {exc}")

    # ---- shutdown ----
    def request_shutdown(self) -> None:
        if not self.stopped.done():
            self.stopped.set_result(None)

    async def close(self) -> None:
        """Stop everything this engine started and wait for it: Applio (process tree), jobs, workers."""
        if self._closing:
            return
        self._closing = True
        aio.set_closing()
        try:
            from .training import handlers as training  # noqa: PLC0415
            await training.on_shutdown()  # kills Applio first, leaving the run resumable (interrupted)
        except Exception:  # noqa: BLE001
            log.exception("could not stop the training")
        requests = [t for t in self._requests if not t.done()]
        if requests:  # requests are short: let the ones in flight answer (a job they start is cancelled below)
            await asyncio.wait(requests, timeout=10.0)
        tasks = [t for t in list(self._jobs.values()) + list(self._requests) if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=SHUTDOWN_GRACE_S)
            if pending:
                log.warning("%d task(s) did not end within %.0f s", len(pending), SHUTDOWN_GRACE_S)
        await _stop_workers()


async def _stop_workers() -> None:
    from . import backends, rvc, speech  # noqa: PLC0415
    from .training import proc  # noqa: PLC0415
    results = await asyncio.gather(rvc.shutdown(), speech.shutdown(), proc.kill_all(),
                                   *(b.close() for b in backends.all_backends()), return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            log.error("shutdown step failed", exc_info=r)


def _device_info(runtime: dict[str, Any]) -> dict[str, Any]:
    backend = config.backend(runtime)
    info: dict[str, Any] = {"backend": backend, "name": backend.upper(), "vram_gb": 0.0}
    if backend == "cuda":
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=True, stdin=subprocess.DEVNULL,
                creationflags=aio.NO_WINDOW,
            ).stdout.strip().splitlines()[0]
            name, mem = [p.strip() for p in out.split(",")]
            info.update(name=name, vram_gb=round(float(mem) / 1024.0, 1))
        except Exception:  # noqa: BLE001
            info["name"] = "NVIDIA GPU"
    elif backend == "mps":
        info["name"] = "Apple Silicon GPU"
    return info


def _models() -> dict[str, str]:
    from . import tts
    return tts.models_status()


def speech_languages() -> list[dict[str, Any]]:
    """The languages the speech model speaks, with names (``hello`` / ``ready`` ``languages``)."""
    from . import backends  # noqa: PLC0415
    from .backends import names  # noqa: PLC0415
    return names.describe(backends.languages())


def hello_info(ctx: Server) -> dict[str, Any]:
    return {"version": __version__, "device": ctx.device, "models": _models(), "languages": speech_languages()}


@method("hello")
async def hello(ctx: Server, params: dict[str, Any]) -> dict[str, Any]:
    return hello_info(ctx)


@method("tts.languages")
async def tts_languages(ctx: Server, params: dict[str, Any]) -> dict[str, Any]:
    return {"languages": speech_languages()}


def _load_handlers() -> None:
    from . import media, tts, voices  # noqa: F401 - register their methods
    for pkg in ("dataset", "training", "redub"):  # written by other modules; may not exist in every checkout
        try:
            importlib.import_module(f"{__package__}.{pkg}")
        except ImportError as exc:
            log.warning("optional module %r not loaded: %s", pkg, exc)


def _prepare_data_dir(data: config.DataDir) -> None:
    data.ensure()
    shutil.rmtree(data.tmp, ignore_errors=True)  # scratch only
    os.makedirs(data.tmp, exist_ok=True)


async def serve(runtime_path: str, data_dir: str, out: BinaryIO, inp: BinaryIO) -> int:
    runtime = config.load_runtime(runtime_path)
    data = config.DataDir(os.path.abspath(data_dir))
    server = Server(out, runtime, data.root)
    await aio.run_io(_prepare_data_dir, data)
    server.device = await aio.run_io(_device_info, runtime)
    media.configure(runtime)
    media.add_to_path()
    await aio.run_io(_load_handlers)
    from . import tts
    try:  # a training the previous engine did not finish (app closed or crashed) becomes a resumable paused run
        from .training.handlers import recover_interrupted
        await aio.run_io(recover_interrupted, server)
    except Exception:  # noqa: BLE001
        log.exception("could not check for interrupted trainings")

    server.emit("ready", hello_info(server))
    aio.spawn(tts.load_models(server), name="model-loader")

    async def read_requests() -> None:
        async for raw in aio.read_lines(inp):
            server.on_line(raw)
        log.info("stdin closed: shutting down")
        server.request_shutdown()

    reader = aio.spawn(read_requests(), name="stdin")
    await server.stopped
    reader.cancel()
    await server.close()
    await server.flush()
    return 0


class _CliContext:
    """Just enough of ``registry.Context`` for handlers that need no running engine (no events, no jobs)."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.runtime: dict[str, Any] = {}

    def emit(self, event: str, data: dict[str, Any], job: str | None = None) -> None:
        pass


def _import_cli(args: argparse.Namespace) -> int:
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
    data = config.DataDir(os.path.abspath(args.data_dir))
    data.ensure()
    try:  # the Applio ffmpeg is found through runtime.toml when it sits in the data directory
        media.configure(config.load_runtime(os.path.join(data.root, "runtime.toml")))
    except (OSError, ValueError):
        pass
    _load_handlers()
    params = {"name": args.name, "reference_path": args.reference, "dataset_dir": args.dataset_dir,
              "rvc_model": args.rvc_model, "rvc_index": args.rvc_index, "language": args.language}
    try:
        result = methods()["voices.import"](_CliContext(data.root), params)  # type: ignore[arg-type]
    except EngineError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mimvox_engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("serve", help="run the JSON-lines engine on stdin/stdout")
    sp.add_argument("--runtime", required=True, help="path to runtime.toml")
    sp.add_argument("--data-dir", required=True, help="data directory")
    ip = sub.add_parser("import", help="create a voice from existing recordings / RVC model and print it as JSON")
    ip.add_argument("--data-dir", required=True, help="data directory")
    ip.add_argument("--name", required=True, help="voice name")
    ip.add_argument("--reference", help="reference clip (becomes reference.wav)")
    ip.add_argument("--dataset-dir", help="folder with part<N>.wav files")
    ip.add_argument("--rvc-model", help="trained RVC .pth")
    ip.add_argument("--rvc-index", help="matching RVC .index")
    ip.add_argument("--language", default=None,
                    help="language the recordings are in (ISO 639-1: en, uk, de, fr, es, pl; default en)")
    args = ap.parse_args(argv)
    if args.cmd == "import":
        return _import_cli(args)

    # Move the protocol channel to a private fd, then send everything else to stderr (also native writes to fd 1).
    out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    inp = aio.detach_stdin()
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    code = 1
    try:
        code = asyncio.run(serve(args.runtime, args.data_dir, out, inp))
    except BaseException:  # noqa: BLE001
        log.exception("engine failed")
        from .training import proc  # noqa: PLC0415
        proc.kill_all_sync()  # never leave an Applio training behind
    try:
        out.flush()
    except OSError:
        pass
    sys.stderr.flush()
    os._exit(code)  # do not wait for CUDA teardown or the stdin reader thread
