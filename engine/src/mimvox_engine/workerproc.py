"""Async client of a persistent JSON-lines worker process (``rvc_worker.py``, ``speech_worker.py``; wire format in
``worker_kit.py``).

Design (safe for any number of concurrent callers by construction):

- **One owner task per worker** consumes a request queue. It starts the process when needed (the next request after a
  crash or an idle exit restarts it), writes one request, awaits its reply, then takes the next one. Only the owner
  writes requests, so nothing is interleaved, and no lock is ever held while a reply is awaited.
- **A reader task** per process does ``async for line in proc.stdout`` and resolves the request future with the
  matching ``id``; ``progress`` lines call the request's callback. At EOF it awaits the process exit and fails every
  pending future at once (``WorkerDied``), so a crash never leaves a caller waiting.
- A caller that is cancelled while its request is queued just drops out; if the request is running, a
  ``{"op":"cancel","id"}`` is sent and the owner waits for the worker's answer before sending the next request.
- Timeouts are timers on the awaited reply (``asyncio.wait``), never polls.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from . import aio

log = logging.getLogger("mimvox.worker")

Progress = Callable[[dict[str, Any]], None]


class WorkerDied(Exception):
    """The worker process ended (crash, kill, idle exit) while a request was queued for it or running."""


class WorkerError(Exception):
    """The worker answered ``ok: false`` (``code`` is the worker's error code), or did not start / answer in time."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerCancelled(Exception):
    """The worker reports the op as cancelled."""


class _Request:
    __slots__ = ("id", "msg", "done", "progress", "timeout", "start", "abandoned")

    def __init__(self, rid: int, msg: dict[str, Any], progress: Progress | None, timeout: float | None,
                 start: bool) -> None:
        self.id = rid
        self.msg = msg
        self.done: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.progress = progress
        self.timeout = timeout
        self.start = start
        self.abandoned = False


class WorkerClient:
    def __init__(self, name: str, argv: list[str], env: dict[str, str], start_timeout: float = 240.0) -> None:
        self.name = name
        self.argv = argv
        self.env = env
        self.start_timeout = start_timeout
        self._ids = itertools.count(1)
        self._queue: asyncio.Queue[_Request | None] | None = None
        self._owner: asyncio.Task[None] | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._exited: asyncio.Future[int] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._pending: dict[int, _Request] = {}
        self.current: _Request | None = None   # the request the worker is running now
        self.last_error: str | None = None
        self.pid: int | None = None

    # ---- state ----
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    def busy(self) -> bool:
        return self.current is not None or (self._queue is not None and not self._queue.empty())

    # ---- requests ----
    async def call(self, op: str, params: dict[str, Any] | None = None, progress: Progress | None = None,
                   timeout: float | None = None, start: bool = True, retry: bool = True) -> dict[str, Any]:
        """Run ``op`` and return its result. A worker that dies under the request is restarted and the request retried
        once (``retry``). ``start=False``: answer ``{}`` without starting a process when none is running."""
        try:
            return await self._call_once(op, params, progress, timeout, start)
        except WorkerDied as exc:
            self.last_error = str(exc)
            if not retry:
                raise
            log.warning("%s worker died under %s (%s); retrying once", self.name, op, exc)
        return await self._call_once(op, params, progress, timeout, start)

    async def _call_once(self, op: str, params: dict[str, Any] | None, progress: Progress | None,
                         timeout: float | None, start: bool) -> dict[str, Any]:
        req = _Request(next(self._ids), {"op": op, **(params or {})}, progress, timeout, start)
        self._ensure_owner()
        assert self._queue is not None
        self._queue.put_nowait(req)
        try:
            return await asyncio.shield(req.done)
        except asyncio.CancelledError:
            req.abandoned = True
            if self.current is req and self.alive():
                self._write({"op": "cancel", "id": req.id})
            raise

    def cancel_current(self, op: str | None = None, when: Callable[[dict[str, Any]], bool] | None = None) -> None:
        """Ask the worker to stop the op it is running (only if it is ``op``, when given, and ``when(request)`` holds,
        when given). Its caller gets ``WorkerCancelled`` (or the result, if the op cannot stop any more)."""
        cur = self.current
        if (cur is not None and (op is None or cur.msg.get("op") == op) and (when is None or when(cur.msg))
                and self.alive()):
            self._write({"op": "cancel", "id": cur.id})

    def _ensure_owner(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._owner is None or self._owner.done():
            self._owner = aio.spawn(self._serve(self._queue), name=f"{self.name}-owner")

    async def _serve(self, queue: asyncio.Queue[_Request | None]) -> None:
        async for req in aio.queue_items(queue):
            if req.abandoned or req.done.done():
                continue
            try:
                if not self.alive():
                    if not req.start:
                        req.done.set_result({})
                        continue
                    await self._spawn()
                await self._exchange(req)
            except Exception as exc:  # noqa: BLE001 - the error belongs to this request
                if not req.done.done():
                    req.done.set_exception(exc)
            finally:
                self.current = None

    async def _exchange(self, req: _Request) -> None:
        self.current = req
        self._pending[req.id] = req
        self._write({"id": req.id, **req.msg})
        if req.abandoned:  # cancelled between dequeuing and sending
            self._write({"op": "cancel", "id": req.id})
        done, _ = await asyncio.wait({req.done}, timeout=req.timeout)
        if not done:
            self._pending.pop(req.id, None)
            req.done.set_exception(WorkerError("internal", f"the {self.name} worker timed out in {req.msg['op']}"))
            await self.stop(graceful=False)
        if req.done.done() and not req.done.cancelled():
            req.done.exception()  # mark it retrieved even when the caller has gone away

    # ---- process ----
    async def _spawn(self) -> None:
        loop = asyncio.get_running_loop()
        kwargs: dict[str, Any] = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = aio.NO_WINDOW
        log.info("starting the %s worker", self.name)
        proc = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=None, env=self.env,
            limit=64 * 1024 * 1024, **kwargs)
        self._proc, self.pid = proc, proc.pid
        self._ready = loop.create_future()
        exited: asyncio.Future[int] = loop.create_future()
        self._exited = exited
        aio.spawn(self._read(proc, exited), name=f"{self.name}-reader")
        done, _ = await asyncio.wait({self._ready, exited}, timeout=self.start_timeout,
                                     return_when=asyncio.FIRST_COMPLETED)
        if self._ready in done and not self._ready.cancelled():
            log.info("%s worker ready (pid %s)", self.name, proc.pid)
            return
        if exited.done():
            raise WorkerDied(f"the {self.name} worker exited during start-up (code {exited.result()}; see the log)")
        await self.stop(graceful=False)
        raise WorkerError("internal", f"the {self.name} worker did not start in time")

    async def _read(self, proc: asyncio.subprocess.Process, exited: asyncio.Future[int]) -> None:
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                self._dispatch(raw)
        except (ValueError, OSError) as exc:
            log.warning("%s worker output unreadable: %s", self.name, exc)
        finally:
            code = await proc.wait()
            if self._proc is proc:
                self._proc = None
            err = WorkerDied(f"the {self.name} worker exited (code {code})")
            for req in list(self._pending.values()):
                if not req.done.done():
                    req.done.set_exception(err)
            self._pending.clear()
            if self._ready is not None and not self._ready.done():
                self._ready.cancel()
            if not exited.done():
                exited.set_result(code)
            log.info("%s worker (pid %s) exited with code %s", self.name, proc.pid, code)

    def _dispatch(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw)
            if not isinstance(msg, dict):
                raise ValueError("not an object")
        except ValueError:
            log.warning("unexpected %s worker output: %s", self.name, raw[:300])
            return
        if "event" in msg:
            if msg["event"] == "ready":
                if self._ready is not None and not self._ready.done():
                    self._ready.set_result(None)
            else:
                log.info("%s worker: %s", self.name, msg)
            return
        req = self._pending.get(msg.get("id"))  # type: ignore[arg-type]
        if req is None:
            return
        if "progress" in msg:
            if req.progress is not None and not req.abandoned:
                try:
                    req.progress(msg["progress"])
                except Exception:  # noqa: BLE001 - a progress callback must not break the channel
                    log.exception("%s progress callback failed", self.name)
            return
        del self._pending[req.id]
        if req.done.done():
            return
        if msg.get("ok"):
            req.done.set_result(msg.get("result") or {})
        elif msg.get("code") == "cancelled":
            req.done.set_exception(WorkerCancelled())
        else:
            req.done.set_exception(WorkerError(str(msg.get("code") or "internal"), str(msg.get("error") or "failed")))

    def _write(self, obj: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.returncode is not None:
            return
        try:
            proc.stdin.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        except (OSError, RuntimeError) as exc:
            log.warning("%s worker pipe closed: %s", self.name, exc)

    async def stop(self, graceful: bool = True) -> None:
        """End the process: ``quit`` and await its exit (5 s), else kill. Pending requests fail with ``WorkerDied``."""
        proc, exited = self._proc, self._exited
        if proc is None or proc.returncode is not None:
            return
        if graceful:
            self._write({"op": "quit"})
            try:
                if proc.stdin is not None:
                    await proc.stdin.drain()
            except (OSError, ConnectionError):
                pass
            if exited is not None:
                done, _ = await asyncio.wait({exited}, timeout=5.0)
                if done:
                    return
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        if exited is not None:
            await asyncio.shield(exited)

    async def close(self) -> None:
        """Stop the process and the owner task (engine shutdown)."""
        await self.stop()
        if self._queue is not None:
            self._queue.put_nowait(None)


def isolated() -> list[str]:
    """Interpreter flags for a worker: ``-I`` (isolated: no user site, no PYTHON* variables), plus the engine's bytecode
    cache folder (``PYTHONPYCACHEPREFIX``, which ``-I`` would ignore), so no ``__pycache__`` lands next to the sources."""
    prefix = sys.pycache_prefix
    return ["-I", "-X", f"pycache_prefix={prefix}"] if prefix else ["-I"]


def child_env(**extra: str) -> dict[str, str]:
    """Environment of a worker: UTF-8, unbuffered, and without the engine's PYTHONPATH (it imports nothing from here)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", **extra)
    env.pop("PYTHONPATH", None)
    return env
