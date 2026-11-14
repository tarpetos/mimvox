"""Engine method registry. ``async def`` handlers run on the event loop, the only thread that changes shared state;
plain ``def`` handlers run on the IO pool. Only ``server.py`` may write to stdout."""
from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, Union

Handler = Callable[["Context", dict[str, Any]], Union[dict[str, Any], Awaitable[dict[str, Any]]]]
_METHODS: dict[str, Handler] = {}


class EngineError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Context(Protocol):
    data_dir: str
    runtime: dict[str, Any]

    def emit(self, event: str, data: dict[str, Any], job: str | None = None) -> None:
        """Callable from any thread; events keep their order."""

    def start_job(self, fn: Callable[[str], Any]) -> str:
        """Return the job id at once. An async fn runs on the loop; a plain fn runs on the IO pool and must poll
        :meth:`cancelled` itself."""

    def cancelled(self, job: str) -> bool:
        """True once the app asked to cancel ``job``."""

    def cancel(self, job: str) -> None:
        """The job reports its own ``*.error`` with ``cancelled``."""


def method(name: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        if name in _METHODS:
            raise RuntimeError(f"method {name!r} registered twice")
        _METHODS[name] = fn
        return fn
    return register


def handlers() -> dict[str, Handler]:
    return dict(_METHODS)


def methods() -> dict[str, Callable[["Context", dict[str, Any]], dict[str, Any]]]:
    """Handlers as blocking calls for sync code. Never on the loop thread: an async handler waits on the loop."""
    return {name: (_blocking(fn) if inspect.iscoroutinefunction(fn) else fn) for name, fn in _METHODS.items()}  # type: ignore[misc]


def _blocking(fn: Handler) -> Callable[["Context", dict[str, Any]], dict[str, Any]]:
    def call(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
        from . import aio  # noqa: PLC0415

        try:
            aio.loop()
        except RuntimeError:  # no engine loop: the `import` command line
            import asyncio  # noqa: PLC0415
            return asyncio.run(fn(ctx, params))  # type: ignore[arg-type]
        return aio.run_sync(fn(ctx, params))  # type: ignore[arg-type]
    return call
