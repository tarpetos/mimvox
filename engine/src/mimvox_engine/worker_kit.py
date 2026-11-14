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


# ------------------------------------------------------------------------------------------- GPU memory guard
#
# Mimvox must never push the graphics card past full. On Windows the NVIDIA driver's "CUDA sysmem fallback" (on by
# default since driver 536.40) does not fail an allocation that does not fit: it moves it into system RAM. A process
# that overfills the card then eats RAM and the system commit limit, other programs crash for lack of memory, and the
# desktop loses its own GPU resources (wallpaper, screenshots, a frozen Task Manager). So every worker caps its PyTorch
# allocator at what it holds plus what is free on the card beyond a reserve for the desktop and other apps (measured
# on the whole card, other processes included): past that, PyTorch raises its out-of-memory error instead, the op
# fails with code "no_vram", and the engine frees GPU memory elsewhere and retries (workerproc.py).

VRAM_RESERVE_GB = float(os.environ.get("MIMVOX_VRAM_RESERVE_GB", "1.5"))
# A model worker that nobody has used for this long exits: its VRAM, its RAM and the 1.5 GB a PyTorch process costs by
# itself all go back to the system (an idle Mimvox held 19 GB of commit and 6.6 GB of RAM before). The next request
# starts it again (workerproc.py). 0 keeps workers for the engine's life.
IDLE_EXIT_S = float(os.environ.get("MIMVOX_IDLE_EXIT_S", "600"))
NO_VRAM = "not enough free GPU memory: close programs that use the graphics card (games, video, browsers) and try again"
NO_RAM = "not enough free memory (RAM): close other programs and try again"


def _cuda_torch(start=False):
    """The imported torch module when CUDA is usable and (unless ``start``) already started in this process."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        if not start and not torch.cuda.is_initialized():
            return None
    except Exception:  # noqa: BLE001
        return None
    return torch


def gpu_free_gb():
    """VRAM free on the whole card beyond the reserve, in GB (0 without CUDA). Starts CUDA when needed."""
    torch = _cuda_torch(start=True)
    if torch is None:
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return max(0.0, free / 2**30 - VRAM_RESERVE_GB)


def gpu_fence(start=False):
    """Cap this process's PyTorch CUDA allocator at the memory it holds now plus what the card has free beyond
    ``VRAM_RESERVE_GB``. Runs before every op (a no-op until the process uses CUDA); call it with ``start=True``
    right before a model is first loaded onto the GPU. Returns the cap in GB, or None without CUDA."""
    torch = _cuda_torch(start)
    if torch is None:
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        held = torch.cuda.memory_reserved()
        allow = held + max(0, free - int(VRAM_RESERVE_GB * 2**30))
        torch.cuda.set_per_process_memory_fraction(max(0.01, min(1.0, allow / total)))
        return allow / 2**30
    except Exception:  # noqa: BLE001 - the guard must never break an op by itself
        return None


def is_vram_error(exc):
    """PyTorch's CUDA out-of-memory error (also the one ``gpu_fence``'s cap raises), or a CUDA error that says so."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc)
    return "CUDA out of memory" in text or "CUBLAS_STATUS_ALLOC_FAILED" in text or "cudaErrorMemoryAllocation" in text


def gpu_release_cache():
    """Hand the PyTorch allocator's cached GPU blocks back to the driver. Runs after every op: on Windows (WDDM) every
    GPU block also counts against the system's commit limit, cached ones included, so a worker that keeps its cache
    holds RAM commit for memory it is not using (measured: a 2 GB block freed but cached still commits 2 GB; the RVC
    worker sat at 6.6 GB of commit with 1.1 GB on the card). The next op allocates again in milliseconds."""
    torch = _cuda_torch()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110 - best effort
            pass


def gpu_cleanup():
    """Collect garbage and give cached GPU blocks back after a failed op."""
    import gc

    gc.collect()
    gpu_release_cache()


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
            gpu_fence()
            result = fn(req)
            self.send({"id": rid, "ok": True, "result": result if isinstance(result, dict) else {}})
        except Cancelled:
            self.send({"id": rid, "ok": False, "code": "cancelled", "error": "cancelled"})
        except OpError as exc:
            self.send({"id": rid, "ok": False, "code": exc.code, "error": str(exc)})
        except MemoryError:
            gpu_cleanup()
            self.log(f"op {req.get('op')} ran out of RAM: {traceback.format_exc()}")
            self.send({"id": rid, "ok": False, "code": "no_memory", "error": NO_RAM})
        except BaseException as exc:  # noqa: BLE001 - every request gets an answer
            if is_vram_error(exc):
                gpu_cleanup()
                self.log(f"op {req.get('op')} ran out of GPU memory: {exc}")
                self.send({"id": rid, "ok": False, "code": "no_vram", "error": NO_VRAM})
                return
            self.log(f"op {req.get('op')} failed: {traceback.format_exc()}")
            self.send({"id": rid, "ok": False, "code": "internal", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            gpu_release_cache()
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
