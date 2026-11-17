from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import logging
import os
import signal
import subprocess
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, BinaryIO, TypeVar

log = logging.getLogger("mimvox.aio")
T = TypeVar("T")

# One thread: in-process model calls are serialized by design, a second request queues behind the first.
GPU = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu")
IO = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="io")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: int | None = None
_tasks: set[asyncio.Task[Any]] = set()
_closing = [False]


def closing() -> bool:
    return _closing[0]


def set_closing() -> None:
    _closing[0] = True


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Called once by the server from inside the running loop."""
    global _loop, _loop_thread
    _loop, _loop_thread = loop, threading.get_ident()


def loop() -> asyncio.AbstractEventLoop:
    if _loop is None:
        raise RuntimeError("the engine event loop is not running")
    return _loop


def on_loop_thread() -> bool:
    return _loop_thread == threading.get_ident()


def call_soon(fn: Callable[..., Any], *args: Any) -> None:
    """Safe from any thread."""
    if on_loop_thread():
        fn(*args)
    else:
        loop().call_soon_threadsafe(fn, *args)


def spawn(coro: Awaitable[T], name: str | None = None) -> asyncio.Task[T]:
    """Keeps a strong reference until the task ends and logs an unexpected failure."""
    task = asyncio.ensure_future(coro)
    if name:
        task.set_name(name)
    _tasks.add(task)

    def done(t: asyncio.Task[Any]) -> None:
        _tasks.discard(t)
        if not t.cancelled() and t.exception() is not None:
            log.error("background task %s failed", t.get_name(), exc_info=t.exception())

    task.add_done_callback(done)
    return task


def spawn_threadsafe(coro_fn: Callable[[], Awaitable[Any]], name: str | None = None) -> None:
    call_soon(lambda: spawn(coro_fn(), name))


async def run_gpu(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.get_running_loop().run_in_executor(GPU, functools.partial(fn, *args, **kwargs))


async def run_io(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    return await asyncio.get_running_loop().run_in_executor(IO, functools.partial(fn, *args, **kwargs))


def run_sync(coro: Awaitable[T]) -> T:
    if on_loop_thread():
        raise RuntimeError("run_sync() called on the loop thread")
    return asyncio.run_coroutine_threadsafe(coro, loop()).result()  # type: ignore[arg-type]


def detach_stdin() -> BinaryIO:
    """Point fd 0 / ``STD_INPUT_HANDLE`` at NUL and return the protocol pipe as a private file. Windows serializes I/O
    on a synchronous pipe, so a pending ReadFile on stdin blocked DLL loads (``import torch``) until the next line."""
    fd = os.dup(0)  # non-inheritable (PEP 446)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if os.name == "nt":
        import ctypes  # noqa: PLC0415
        import msvcrt  # noqa: PLC0415

        k32 = ctypes.windll.kernel32
        k32.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
        k32.SetStdHandle(ctypes.c_ulong(-10 & 0xFFFFFFFF), ctypes.c_void_p(msvcrt.get_osfhandle(0)))
    return os.fdopen(fd, "rb")


async def read_lines(stream: BinaryIO) -> AsyncIterator[bytes]:
    """Yield lines until EOF. Windows anonymous pipes are synchronous and cannot be read by the proactor, so there (and
    for a non-pipe stream) a dedicated thread blocks on ``readline()``."""
    loop_ = asyncio.get_running_loop()
    if os.name != "nt":
        reader = asyncio.StreamReader(limit=64 * 1024 * 1024)
        try:
            await loop_.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), stream)
        except (OSError, ValueError):
            reader = None  # type: ignore[assignment]  # a regular file: fall back to the thread below
        if reader is not None:
            try:
                async for line in reader:
                    yield line
                return
            except ValueError:  # readline() raises for a single line longer than the 64 MiB reader limit
                log.warning("oversized stdin line: treating the rest of the stream as EOF")
                return
    queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def pump() -> None:
        try:
            for raw in iter(stream.readline, b""):
                loop_.call_soon_threadsafe(queue.put_nowait, raw)
        except (OSError, ValueError):
            pass
        finally:
            try:
                loop_.call_soon_threadsafe(queue.put_nowait, None)
            except RuntimeError:
                pass  # the loop is already closed

    threading.Thread(target=pump, name="stdin-reader", daemon=True).start()
    line = await queue.get()
    while line is not None:
        yield line
        line = await queue.get()


async def gather_or_cancel(*aws: Awaitable[Any]) -> list[Any]:
    """``asyncio.TaskGroup`` on Python 3.10: the first failure cancels the others, waits for them, is raised."""
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in tasks:
            if t in done and not t.cancelled() and t.exception() is not None:
                raise t.exception()  # type: ignore[misc]
        return [t.result() for t in tasks]
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def queue_items(queue: asyncio.Queue[T | None]) -> AsyncIterator[T]:
    """Yield items until a ``None`` arrives."""
    item = await queue.get()
    while item is not None:
        yield item
        item = await queue.get()


def new_group_kwargs() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def kill_tree(pid: int, wait_for: Awaitable[Any] | None = None, grace_s: float = 5.0) -> None:
    """Stop ``pid`` and everything it started (Applio's train.py spawns workers)."""
    waiter = asyncio.ensure_future(wait_for) if wait_for is not None else None
    if sys.platform == "win32":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(pid), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, creationflags=NO_WINDOW)
            await asyncio.wait_for(killer.wait(), 30)
        except (OSError, asyncio.TimeoutError):
            log.warning("taskkill of pid %s failed", pid)
        if waiter is not None:
            done, _ = await asyncio.wait({waiter}, timeout=grace_s)
            if not done:
                log.warning("process %s still running after taskkill", pid)
                waiter.cancel()
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        if waiter is not None:
            waiter.cancel()
        return
    if waiter is None:
        return
    done, _ = await asyncio.wait({waiter}, timeout=grace_s)
    if not done:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        await waiter


def kill_tree_sync(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30,
                           creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
        else:
            os.killpg(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
