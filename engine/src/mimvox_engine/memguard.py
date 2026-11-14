"""Host memory guard: Mimvox must never run the computer out of memory.

Windows: the engine puts itself in a Job Object at start-up; every worker it starts (and their children: training,
ffmpeg) lands in the same job. The job caps the memory all of them may commit together (``JOB_OBJECT_LIMIT_JOB_MEMORY``,
75 % of the installed RAM by default, ``MIMVOX_RAM_LIMIT_GB`` to override). Past the cap an allocation fails inside a
Mimvox process (the op fails or the worker restarts, the app says why) instead of the system commit limit filling up,
where other programs crash and the desktop breaks. The job also ends with the engine (``KILL_ON_JOB_CLOSE``), so a
crashed engine never leaves a worker holding VRAM and RAM.

Linux: the kernel's OOM killer picks the process with the highest score when memory runs out; the engine raises its
own score (inherited by the workers), so Mimvox is stopped first and other programs are left alone.

macOS: nothing to set (memory pressure handling is the system's).
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger("mimvox.memory")

DEFAULT_SHARE = 0.75          # of the installed RAM, for everything Mimvox's engine runs
LINUX_OOM_SCORE_ADJ = 500     # 0..1000 added to the OOM killer's score (raising it needs no privileges)

_job: dict[str, object] = {}  # Windows: {"handle", "limit"} once the job is set up


def setup() -> None:
    """Call once, first thing in the engine process (before any worker starts). Never raises."""
    try:
        if sys.platform == "win32":
            _setup_windows()
        elif sys.platform.startswith("linux"):
            _setup_linux()
    except Exception:  # the guard must not stop the engine
        log.exception("could not set up the memory guard")


def peak_gb() -> float | None:
    """Windows: the most memory the job (the engine and all its workers) has committed at once, in GB."""
    info = _query()
    return None if info is None else info.PeakJobMemoryUsed / 2**30


def report(when: str) -> None:
    """Log the job's peak commit against its cap (Windows; nothing elsewhere)."""
    peak = peak_gb()
    if peak is not None:
        log.info("memory %s: peak %.1f GB of the %.1f GB cap", when, peak, float(_job["limit"]) / 2**30)


# --------------------------------------------------------------------------------------------------- Windows

def _structs():  # type: ignore[no-untyped-def]
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class Io(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                   "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", Io), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64), ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64), ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64), ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    return ctypes, wintypes, Extended, MemStatus


JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


def _setup_windows() -> None:
    ctypes, wintypes, Extended, MemStatus = _structs()
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.GlobalMemoryStatusEx.argtypes = [ctypes.c_void_p]

    ms = MemStatus()
    ms.dwLength = ctypes.sizeof(ms)
    if not k32.GlobalMemoryStatusEx(ctypes.byref(ms)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    limit = _limit_bytes(ms.ullTotalPhys)

    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    info = Extended()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = limit
    if not k32.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                       ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    _job.update(handle=job, limit=limit)  # the handle stays open for the engine's life (closing it ends the job)
    log.info("memory cap for the engine and its workers: %.1f GB (%.1f GB RAM installed)", limit / 2**30,
             ms.ullTotalPhys / 2**30)


def _limit_bytes(total_phys: int) -> int:
    override = os.environ.get("MIMVOX_RAM_LIMIT_GB")
    if override:
        return int(float(override) * 2**30)
    return int(total_phys * DEFAULT_SHARE)


def _query():  # type: ignore[no-untyped-def]
    if "handle" not in _job:
        return None
    try:
        ctypes, wintypes, Extended, _ = _structs()
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                                  ctypes.c_void_p]
        info = Extended()
        if not k32.QueryInformationJobObject(_job["handle"], JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                             ctypes.sizeof(info), None):
            return None
        return info
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------------------------------- Linux

def _setup_linux() -> None:
    path = "/proc/self/oom_score_adj"
    with open(path, encoding="ascii") as fh:
        current = int(fh.read().strip() or 0)
    if current >= LINUX_OOM_SCORE_ADJ:
        return
    with open(path, "w", encoding="ascii") as fh:
        fh.write(str(LINUX_OOM_SCORE_ADJ))
    log.info("OOM score raised to %d: Mimvox is stopped first if memory runs out", LINUX_OOM_SCORE_ADJ)
