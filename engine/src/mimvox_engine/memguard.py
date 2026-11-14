"""Host memory guard: Mimvox must never run the computer out of memory.

Windows: the engine puts itself in a Job Object at start-up; every worker it starts (and their children: training,
ffmpeg) lands in the same job. The job caps the memory all of them may commit together (``JOB_OBJECT_LIMIT_JOB_MEMORY``):
the system's commit limit (RAM + page file, counting how far a page file Windows may still grow can grow on its disk)
minus what every other program has committed, minus a reserve (3 % of the limit, at least 1.5 GB). It is set at start-up and recomputed before every worker op (``refresh``), so it shrinks when
other programs grow and grows when they shrink; ``MIMVOX_RAM_LIMIT_GB`` fixes it instead. Not a share of the RAM: under WDDM every GPU
allocation is charged to the process's commit too (so it can be paged out), so a worker with 6 GB on the card commits
6 GB plus its own RAM (measured: a PyTorch process holding 2 GB of VRAM commits 3.6 GB; the engine with OmniVoice, RVC
and the zero-shot converter peaked at 18.9 GB on a 12 GB card). Past the cap an allocation fails inside a Mimvox
process (the op fails or the worker restarts, the app says why) instead of the system commit limit filling up, where
other programs crash and the desktop breaks. The job also ends with the engine (``KILL_ON_JOB_CLOSE``), so a
crashed engine never leaves a worker holding VRAM and RAM.

Linux: the kernel's OOM killer picks the process with the highest score when memory runs out; the engine raises its
own score (inherited by the workers), so Mimvox is stopped first and other programs are left alone.

macOS: nothing to set (memory pressure handling is the system's).
"""
from __future__ import annotations

import functools
import logging
import os
import sys

log = logging.getLogger("mimvox.memory")

RESERVE_SHARE = 0.03          # of the system commit limit, always left to everything else
RESERVE_MIN_GB = 1.5
LOW_CAP_GB = 12.0             # below this, warn: generation with voice match may not fit
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


def refresh() -> None:
    """Windows: recompute the cap from the system's commit now (cheap; a no-op elsewhere or with a fixed cap)."""
    if "handle" not in _job or os.environ.get("MIMVOX_RAM_LIMIT_GB"):
        return
    try:
        commit_limit, committed = _system_commit()
        ours = _job_commit()
        old, new = int(_job["limit"]), _limit_bytes(commit_limit, committed - ours)
        _set_limit(new)
        if abs(new - old) >= 2**30:
            log.info("memory cap now %.1f GB (Mimvox uses %.1f GB, other programs %.1f GB of the %.1f GB limit)",
                     new / 2**30, ours / 2**30, (committed - ours) / 2**30, commit_limit / 2**30)
    except Exception:  # keep the previous cap
        log.debug("could not refresh the memory cap", exc_info=True)


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

@functools.cache
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

    class PerfInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                    ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                    ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                    ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                    ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                    ("HandleCount", wintypes.DWORD), ("ProcessCount", wintypes.DWORD), ("ThreadCount", wintypes.DWORD)]

    return ctypes, wintypes, Extended, PerfInfo


JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


@functools.cache
def _k32():  # type: ignore[no-untyped-def]
    ctypes, wintypes, _, _ = _structs()
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.c_void_p]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    return k32


def _system_commit() -> tuple[int, int]:
    """(commit limit, committed now) of the whole system, in bytes. The limit includes what the page files may still
    grow by (``page_file_growth``): Windows enlarges them when commit nears the limit."""
    ctypes, wintypes, _, PerfInfo = _structs()
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetPerformanceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    pi = PerfInfo()
    pi.cb = ctypes.sizeof(pi)
    if not psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb):
        raise OSError(ctypes.get_last_error(), "GetPerformanceInfo failed")
    growth = 0
    try:
        growth = page_file_growth(_paging_files(), pi.PhysicalTotal * pi.PageSize, _file_size, _disk_free)
    except Exception:  # count no growth
        log.debug("could not read the page file settings", exc_info=True)
    return pi.CommitLimit * pi.PageSize + growth, pi.CommitTotal * pi.PageSize


def page_file_growth(entries: list[str], phys_total: int, file_size, disk_free) -> int:  # type: ignore[no-untyped-def]
    """Bytes the page files may still grow by. ``entries`` are the ``PagingFiles`` registry lines (``"d:\\pagefile.sys
    16384 16384"``: initial and maximum MB, ``0 0`` or none: managed by the system, ``"?:\\pagefile.sys"``: all of it
    managed by the system, on the system drive). A system-managed file may reach three times the RAM; growth stops at
    the disk's free space."""
    total = 0
    for entry in entries:
        parts = entry.split()
        if not parts:
            continue
        path = parts[0]
        if path.startswith("?"):
            path = os.environ.get("SystemDrive", "C:") + path[2:]
        nums = [int(n) for n in parts[1:3] if n.isdigit()]
        maximum = nums[1] * 2**20 if len(nums) == 2 and nums[1] > 0 else 3 * phys_total
        current = file_size(path)
        total += max(0, min(maximum, current + disk_free(path)) - current)
    return total


def _paging_files() -> list[str]:
    import winreg

    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management") as key:
        value, _ = winreg.QueryValueEx(key, "PagingFiles")
    return [v for v in (value if isinstance(value, list) else [value]) if v.strip()]


def _file_size(path: str) -> int:
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _disk_free(path: str) -> int:
    import shutil

    try:
        return shutil.disk_usage(os.path.splitdrive(path)[0] + "\\").free
    except OSError:
        return 0


def _job_commit() -> int:
    """What the job's processes have committed now (their private bytes), in bytes."""
    ctypes, wintypes, _, _ = _structs()
    k32 = _k32()
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]

    class PidList(ctypes.Structure):
        _fields_ = [("NumberOfAssignedProcesses", wintypes.DWORD), ("NumberOfProcessIdsInList", wintypes.DWORD),  # noqa: RUF012
                    ("ProcessIdList", ctypes.c_size_t * 256)]

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),  # noqa: RUF012 - ctypes layout
                    *[(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                     "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                     "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                                     "PrivateUsage")]]

    ids = PidList()
    if not k32.QueryInformationJobObject(_job["handle"], JOB_OBJECT_BASIC_PROCESS_ID_LIST, ctypes.byref(ids),
                                         ctypes.sizeof(ids), None):
        raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
    total = 0
    for pid in ids.ProcessIdList[:ids.NumberOfProcessIdsInList]:
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            continue
        try:
            c = Counters()
            c.cb = ctypes.sizeof(c)
            if psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                total += c.PrivateUsage
        finally:
            k32.CloseHandle(h)
    return total


def _set_limit(limit: int) -> None:
    ctypes, _, Extended, _ = _structs()
    info = Extended()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_JOB_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.JobMemoryLimit = limit
    if not _k32().SetInformationJobObject(_job["handle"], JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                                          ctypes.sizeof(info)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    _job["limit"] = limit


def _setup_windows() -> None:
    ctypes, _, _, _ = _structs()
    k32 = _k32()
    commit_limit, committed = _system_commit()
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    _job["handle"] = job  # stays open for the engine's life (closing it ends the job and its processes)
    limit = _limit_bytes(commit_limit, committed)
    _set_limit(limit)
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        _job.clear()
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    log.info("memory cap for the engine and its workers: %.1f GB (system commit limit %.1f GB, %.1f GB in use at "
             "start)", limit / 2**30, commit_limit / 2**30, committed / 2**30)
    if limit < LOW_CAP_GB * 2**30:
        log.warning("little memory is free (%.1f GB for Mimvox): close other programs or enlarge the page file, "
                    "or longer jobs may stop with 'not enough free memory'", limit / 2**30)


def _limit_bytes(commit_limit: int, others: int) -> int:
    """The cap: the commit limit minus what the other programs have committed and the reserve; 1 GB at the least."""
    override = os.environ.get("MIMVOX_RAM_LIMIT_GB")
    if override:
        return int(float(override) * 2**30)
    reserve = max(RESERVE_MIN_GB * 2**30, RESERVE_SHARE * commit_limit)
    return int(max(2**30, commit_limit - others - reserve))


def _query():  # type: ignore[no-untyped-def]
    if "handle" not in _job:
        return None
    try:
        ctypes, _, Extended, _ = _structs()
        info = Extended()
        if not _k32().QueryInformationJobObject(_job["handle"], JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                                ctypes.byref(info), ctypes.sizeof(info), None):
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
