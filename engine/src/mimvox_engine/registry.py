"""Method registry shared by every engine module.

Handlers register with ``@method("name")`` and receive ``(ctx, params)``; they return a JSON-serialisable dict or raise
``EngineError``. A handler is either

- ``async def``: runs on the engine's event loop (keep it non-blocking: await ``aio.run_io`` / ``aio.run_gpu`` for
  blocking work). Use this for anything that touches shared state (job slots, workers, model status): the loop thread
  is the only one that changes it, so no locks are needed.
- a plain ``def``: runs on the IO thread pool (file and CPU work that needs no shared state, e.g. ``dataset.*``).

Long work calls ``ctx.start_job(fn)`` and reports with ``ctx.emit(event, data, job=...)``. The server (``server.py``)
owns stdin/stdout; nothing else may print to stdout.
"""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, Union

Handler = Callable[["Context", dict[str, Any]], Union[dict[str, Any], Awaitable[dict[str, Any]]]]
_METHODS: dict[str, Handler] = {}


class EngineError(Exception):
    """An error reported to the app as ``{"code", "message"}`` (codes are listed in contract/engine-protocol.md)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Context(Protocol):
    """What a handler may use. Implemented by ``server.py``."""

    data_dir: str            # absolute <data> directory
    runtime: dict[str, Any]  # parsed runtime.toml

    def emit(self, event: str, data: dict[str, Any], job: str | None = None) -> None:
        """Send an event line to the app (callable from any thread; events keep their order)."""

    def start_job(self, fn: Callable[[str], Any]) -> str:
        """Start ``fn(job_id)`` and return the job id at once. An ``async def fn`` runs as a task on the loop (cancelled
        by :meth:`cancel`); a plain ``def fn`` runs on the IO pool and must check :meth:`cancelled` itself."""

    def cancelled(self, job: str) -> bool:
        """True once the app asked to cancel ``job``."""

    def cancel(self, job: str) -> None:
        """Cancel ``job``: mark it and cancel its task (the job reports its own ``*.error`` with ``cancelled``)."""


def method(name: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        if name in _METHODS:
            raise RuntimeError(f"method {name!r} registered twice")
        _METHODS[name] = fn
        return fn
    return register


def handlers() -> dict[str, Handler]:
    """The registered handlers as they are (the server dispatches these)."""
    return dict(_METHODS)


def methods() -> dict[str, Callable[["Context", dict[str, Any]], dict[str, Any]]]:
    """Every handler as a plain blocking call, for sync code that calls another method (``voices.import`` calls
    ``dataset.add``, ``voices.set_reference`` calls ``voices.warm``). Call it from an IO-pool thread or the CLI, never
    on the loop thread: an async handler is run on the loop and the calling thread waits for its future."""
    return {name: (_blocking(fn) if inspect.iscoroutinefunction(fn) else fn) for name, fn in _METHODS.items()}  # type: ignore[misc]


def _blocking(fn: Handler) -> Callable[["Context", dict[str, Any]], dict[str, Any]]:
    def call(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
        from . import aio  # noqa: PLC0415

        try:
            aio.loop()
        except RuntimeError:  # no engine loop (the `import` command line): run it on a private loop
            import asyncio  # noqa: PLC0415
            return asyncio.run(fn(ctx, params))  # type: ignore[arg-type]
        return aio.run_sync(fn(ctx, params))  # type: ignore[arg-type]
    return call
