"""Run an Applio CLI step as an asyncio subprocess, stream its output line by line, stop its whole process tree.

The output is read with ``await stdout.read()`` as it arrives (tqdm redraws with ``\\r``, so lines are split on ``\\r``
and ``\\n``); the exit is awaited. A stop (cancel, pause, engine shutdown) kills the tree (Windows: ``taskkill /T``,
POSIX: the process group) and awaits the exit. No polling anywhere.
"""
from __future__ import annotations

import asyncio
import codecs
import os
import re
from collections import deque
from collections.abc import Callable

from .. import aio

_SPLIT = re.compile(r"[\r\n]+")
_BAR = re.compile(r"\d+%\||\d+/\d+ \[")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_LIVE: set[asyncio.subprocess.Process] = set()


async def kill_all() -> None:
    """Stop every running Applio step (engine shutdown) and await the exits."""
    procs = list(_LIVE)
    if procs:
        await asyncio.gather(*(aio.kill_tree(p.pid, p.wait()) for p in procs), return_exceptions=True)


def kill_all_sync() -> None:
    """Last resort when the loop is gone."""
    for p in list(_LIVE):
        aio.kill_tree_sync(p.pid)


class Step:
    """One Applio subprocess. ``await run(on_line)`` returns its exit code; ``await kill()`` stops its tree."""

    def __init__(self, cmd: list[str], cwd: str, env: dict[str, str] | None = None) -> None:
        self.cmd = cmd
        self.cwd = cwd
        self.env = env
        self.proc: asyncio.subprocess.Process | None = None
        self.killed = False
        self.tail: deque[str] = deque(maxlen=40)

    async def kill(self) -> None:
        self.killed = True
        proc = self.proc
        if proc is not None and proc.returncode is None:
            await aio.kill_tree(proc.pid, proc.wait())

    async def run(self, on_line: Callable[[str], None]) -> int:
        if self.killed:
            return -1
        env = dict(os.environ)
        env.update(self.env or {})
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        proc = await asyncio.create_subprocess_exec(
            *self.cmd, cwd=self.cwd, env=env, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, **aio.new_group_kwargs())
        self.proc = proc
        _LIVE.add(proc)
        try:
            if self.killed:  # stopped while the process was being created
                await self.kill()
            await self._pump(proc, on_line)
            return await proc.wait()
        except asyncio.CancelledError:
            await asyncio.shield(self.kill())
            raise
        finally:
            _LIVE.discard(proc)

    async def _pump(self, proc: asyncio.subprocess.Process, on_line: Callable[[str], None]) -> None:
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        chunk = await proc.stdout.read(65536)
        while chunk:  # until EOF; every read is awaited
            pending += decoder.decode(chunk)
            parts = _SPLIT.split(pending)
            pending = parts.pop()  # incomplete tail; tqdm redraws with \r so it is flushed by the next update
            for part in parts:
                self._line(part, on_line)
            if len(pending) > 4096:
                self._line(pending, on_line)
                pending = ""
            chunk = await proc.stdout.read(65536)
        pending += decoder.decode(b"", final=True)
        if pending:
            self._line(pending, on_line)

    def _line(self, raw: str, on_line: Callable[[str], None]) -> None:
        line = _ANSI.sub("", raw).strip()
        if not line:
            return
        if not _BAR.search(line):
            self.tail.append(line)
        on_line(line)
