"""Event loop shared by the standalone workers. They load this file by path with ``-I`` in other interpreters, so it
uses the standard library only. Protocol: one JSON object per line each way; ``cancel`` and ``quit`` get no reply.
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


# On Windows the NVIDIA "CUDA sysmem fallback" (default since driver 536.40) moves an allocation that does not fit
# into system RAM instead of failing, starving the desktop; so every worker caps its PyTorch allocator itself.

VRAM_RESERVE_GB = float(os.environ.get("MIMVOX_VRAM_RESERVE_GB", "1.5"))
# Measured: an idle Mimvox that kept its workers held 19 GB of commit and 6.6 GB of RAM. 0 keeps workers forever.
IDLE_EXIT_S = float(os.environ.get("MIMVOX_IDLE_EXIT_S", "600"))
NO_VRAM = "not enough free GPU memory: close programs that use the graphics card (games, video, browsers) and try again"
NO_RAM = "not enough free memory (RAM): close other programs and try again"


def _cuda_torch(start=False):
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
    """Starts CUDA when needed."""
    torch = _cuda_torch(start=True)
    if torch is None:
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return max(0.0, free / 2**30 - VRAM_RESERVE_GB)


def gpu_fence(start=False):
    """Call it with ``start=True`` right before a model is first loaded onto the GPU. Returns the cap in GB."""
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
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc)
    return "CUDA out of memory" in text or "CUBLAS_STATUS_ALLOC_FAILED" in text or "cudaErrorMemoryAllocation" in text


def gpu_release_cache():
    """Under WDDM cached GPU blocks still count against the system commit limit (measured: a freed 2 GB block still
    committed 2 GB)."""
    torch = _cuda_torch()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001, S110 - best effort
            pass


# Loading a model leaves pages in the working set that are not used again (measured: the RVC worker holds 2.0 GB after
# its first conversion and touches 0.4 GB per conversion). Windows keeps them unless the process trims itself.

TRIM_ABOVE_STEADY = 512 * 2**20


def _working_set():
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        *[(f"f{i}", ctypes.c_size_t) for i in range(6)]]

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.WorkingSetSize
    except Exception:  # noqa: BLE001
        return None


def trim_working_set():
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
        psapi.EmptyWorkingSet(k32.GetCurrentProcess())
    except Exception:  # noqa: BLE001, S110 - best effort
        pass


class _Trimmer:
    """The steady size is the working set after the op that follows a trim."""

    def __init__(self):
        self.steady = 0          # 0: nothing measured yet, so the first load is trimmed
        self.just_trimmed = False

    def after_op(self):
        now = _working_set()
        if now is None:
            return
        if self.just_trimmed:
            self.steady, self.just_trimmed = max(self.steady, now), False
            return
        if now <= self.steady + TRIM_ABOVE_STEADY:
            self.steady = max(self.steady, now)   # a small op after a trim must not set the bar low for the next one
            return
        import gc

        gc.collect()
        trim_working_set()
        self.just_trimmed = True


def gpu_cleanup():
    import gc

    gc.collect()
    gpu_release_cache()


def open_channels():
    """Call first, before importing anything chatty: library prints (also from C code) go to stderr.

    stdin points at NUL: on Windows a thread blocked in ``ReadFile`` on a synchronous pipe blocks every other request on
    that handle, and loading DLLs (``import torch``) makes one, so the op hung until the next line arrived.
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
    """``idle``: ``(seconds, action)`` pairs; an action is ``"exit"`` or a callable run on the op thread that returns an
    event dict to send or None."""

    def __init__(self, name, inp, out, ops, idle=()):
        self.name = name
        self.inp, self.out = inp, out
        self.ops = dict(ops)
        self.idle = list(idle)
        self.exec = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="op")
        self._send_lock = threading.Lock()
        self._cancelled = set()
        self.current = None          # read by check_cancel on the op thread
        self._busy = 0
        self._timers = []
        self._trimmer = _Trimmer()
        self.loop = None
        self._finished = None

    # any thread
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

    # op thread
    def check_cancel(self):
        if self.current is not None and self.current in self._cancelled:
            raise Cancelled()

    def progress(self, payload):
        self.send({"id": self.current, "progress": payload})

    def run(self):
        """Then the caller should ``os._exit``."""
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
        """Op thread."""
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
            self._trimmer.after_op()
            self.current = None
            self._cancelled.discard(rid)

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
