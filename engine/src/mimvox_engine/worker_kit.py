"""Event-driven skeleton shared by the standalone worker processes (``rvc_worker.py``, ``speech_worker.py``).

Standalone on purpose: the workers load this file by path (they run with ``-I`` in other interpreters and must not
import anything else from ``mimvox_engine``). Standard library only.

Wire format (one JSON object per line each way):
  request   {"id":N,"op":"<name>", ...params}
            {"op":"cancel","id":N}   (no reply: op N stops at its next checkpoint, or right away if it has not started;
                                      without "id" it means the op running now)
            {"op":"quit"}            (no reply: the process exits)
  reply     {"id":N,"progress":{...}}                                    (zero or more, only from ops that report it)
            {"id":N,"ok":true,"result":{...}}  or  {"id":N,"ok":false,"code":"...","error":"..."}
  events    {"event":"ready"} once at start, plus worker-specific ones ({"event":"unloaded"}, {"event":"exit"}, ...)

How it runs: one asyncio loop. A dedicated thread blocks on ``readline()`` of the (detached, see ``open_channels``)
stdin pipe and hands each line to the loop with ``call_soon_threadsafe``. Ops run on a single-thread executor (so they
are serialized by design) and resolve futures; idle actions are ``loop.call_later`` timers re-armed after every op.
Nothing polls and nothing waits with a timeout.
"""
import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import traceback


class Cancelled(Exception):
    """Raised inside an op by :meth:`Worker.check_cancel` after the engine cancelled it."""


class OpError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def open_channels():
    """Call first, before importing anything chatty. Returns ``(input, output)`` binary/text files of the protocol.

    - stdout: the protocol moves to a private descriptor; fd 1 and ``sys.stdout`` point at stderr, so library prints
      (also from C code) cannot corrupt it.
    - stdin: the pipe moves to a private descriptor and fd 0 / ``STD_INPUT_HANDLE`` point at NUL. On Windows a thread
      blocked in ``ReadFile`` on a synchronous pipe blocks every other request on that file object, and loading DLLs
      (``import torch``) makes one on the standard input handle: the op then hung until the next line arrived.
    """
    out = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    fd = os.dup(0)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    if os.name == "nt":
        import ctypes
        import msvcrt

        k32 = ctypes.windll.kernel32
        k32.SetStdHandle.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
        k32.SetStdHandle(ctypes.c_ulong(-10 & 0xFFFFFFFF), ctypes.c_void_p(msvcrt.get_osfhandle(0)))
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    return os.fdopen(fd, "rb"), out


class Worker:
    """Serves ``ops`` (name -> ``fn(request) -> result dict``) until quit, EOF on stdin, or an idle exit.

    ``idle`` is a list of ``(seconds, action)``: after that much time without an op, ``action`` runs. An action is
    ``"exit"`` or a callable run on the op thread (serialized with the ops) that returns an event dict to send or None.
    """

    def __init__(self, name, inp, out, ops, idle=()):
        self.name = name
        self.inp, self.out = inp, out
        self.ops = dict(ops)
        self.idle = list(idle)
        self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="op")
        self._send_lock = threading.Lock()
        self._cancelled = set()      # ids cancelled by the engine (an id is checked when its op starts and runs)
        self.current = None          # id of the op running now (read by check_cancel on the op thread)
        self._busy = 0               # ops queued or running
        self._timers = []
        self.loop = None
        self._finished = None

    # ---- output (any thread) ----
    def send(self, obj):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._send_lock:
            try:
                self.out.write(line)
                self.out.flush()
            except (OSError, ValueError):
                pass  # the engine is gone; stdin EOF ends us

    def log(self, msg):
        print(f"[{self.name}] {msg}", file=sys.stderr, flush=True)

    # ---- helpers for ops (op thread) ----
    def check_cancel(self):
        if self.current is not None and self.current in self._cancelled:
            raise Cancelled()

    def progress(self, payload):
        self.send({"id": self.current, "progress": payload})

    # ---- loop side ----
    def run(self):
        """Blocks until the worker finishes; then the caller should ``os._exit``."""
        asyncio.run(self._main())

    async def _main(self):
        self.loop = asyncio.get_running_loop()
        self._finished = self.loop.create_future()
        threading.Thread(target=self._pump, name="stdin-reader", daemon=True).start()
        self.send({"event": "ready"})
        self._arm_idle()
        reason = await self._finished
        self.log(f"exiting ({reason})")

    def _pump(self):
        try:
            for raw in iter(self.inp.readline, b""):
                self.loop.call_soon_threadsafe(self._on_line, raw)
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.loop.call_soon_threadsafe(self._finish, "stdin closed")
            except RuntimeError:
                pass

    def _finish(self, reason):
        if not self._finished.done():
            self._finished.set_result(reason)

    def _on_line(self, raw):
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            return
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError("not an object")
        except ValueError:
            self.log(f"unreadable request: {line[:200]}")
            return
        op = req.get("op")
        if op == "quit":
            self._finish("quit")
            return
        if op == "cancel":
            rid = req.get("id", self.current)
            if rid is not None:
                self._cancelled.add(rid)
            return
        self._busy += 1
        self._disarm_idle()
        fut = self.loop.run_in_executor(self.exec, self._run_op, req)
        fut.add_done_callback(self._op_done)

    def _op_done(self, _fut):
        self._busy -= 1
        if self._busy == 0:
            self._arm_idle()

    def _run_op(self, req):
        """Op thread: run one op and send its reply."""
        rid = req.get("id")
        self.current = rid
        try:
            self.check_cancel()
            fn = self.ops.get(req.get("op"))
            if fn is None:
                raise OpError("bad_request", f"unknown op {req.get('op')!r}")
            result = fn(req)
            self.send({"id": rid, "ok": True, "result": result if isinstance(result, dict) else {}})
        except Cancelled:
            self.send({"id": rid, "ok": False, "code": "cancelled", "error": "cancelled"})
        except OpError as exc:
            self.send({"id": rid, "ok": False, "code": exc.code, "error": str(exc)})
        except MemoryError:
            self.send({"id": rid, "ok": False, "code": "internal", "error": "out of host memory"})
        except BaseException as exc:  # noqa: BLE001 - every request gets an answer
            self.log(f"op {req.get('op')} failed: {traceback.format_exc()}")
            self.send({"id": rid, "ok": False, "code": "internal", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            self.current = None
            self._cancelled.discard(rid)

    # ---- idle timers ----
    def _disarm_idle(self):
        for t in self._timers:
            t.cancel()
        self._timers = []

    def _arm_idle(self):
        self._disarm_idle()
        for seconds, action in self.idle:
            self._timers.append(self.loop.call_later(seconds, self._idle_fire, action))

    def _idle_fire(self, action):
        if self._busy:
            return  # an op started meanwhile (its end re-arms the timers)
        if action == "exit":
            self.send({"event": "exit", "reason": "idle"})
            self._finish("idle")
            return
        fut = self.loop.run_in_executor(self.exec, action)

        def sent(f):
            if not f.cancelled() and f.exception() is None and f.result():
                self.send(f.result())

        fut.add_done_callback(sent)
