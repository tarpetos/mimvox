"""``train.start`` / ``train.pause`` / ``train.resume`` / ``train.cancel``: train an RVC model for one voice through
Applio's CLI (``core.py``), with pause and resume (see ``resume.py`` for how Applio's checkpoints are used).

Event-driven on the engine loop: the job is a task, each Applio step an asyncio subprocess whose output lines drive the
progress events and the checkpoint bookkeeping (``_Watch``). ``train.cancel`` cancels the task (the step kills its
process tree and awaits the exit); ``train.pause`` lets the watcher stop the tree right after a checkpoint is safely
on disk, or when a timer (``PAUSE_PATIENCE_S``) runs out. Blocking file work goes to the IO pool.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

from .. import aio, config, tts, voices
from ..registry import Context, EngineError, method
from .plan import (
    MIN_TRAIN_MINUTES,
    dataset_parts,
    free_vram_gb,
    pick_batch_size,
    recommended_epochs,
    wav_seconds,
)
from .proc import Step
from .progress import Tracker
from .resume import (
    PAUSE_PATIENCE_S,
    checkpoint_every,
    discard_logdir,
    features_ready,
    kill_orphans,
    logdir_for,
    model_name,
    prune_checkpoints,
    repair,
)

SAMPLE_RATE = 40000
EMIT_EVERY_S = 0.5
log = logging.getLogger("mimvox.training")
_VOICE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
_FINAL_PTH = re.compile(r"^(?P<m>.+)_(?P<e>\d+)e_(?P<s>\d+)s\.pth$")

# Loop-thread state (only the event loop changes it, so no locks).
_runs: dict[str, "_Run"] = {}   # job id -> run (pause needs the live process)
_active: str | None = None      # voice id being trained (one training at a time)


class _Cancelled(Exception):
    pass


class _Paused(Exception):
    """The Applio process was stopped on purpose after a checkpoint (``train.pause``)."""


class _Shutdown(Exception):
    """The engine is shutting down: Applio was stopped, the run marker stays (an "interrupted" run at next start)."""


class _StepFailed(Exception):
    def __init__(self, stage: str, code: int, tail: list[str]) -> None:
        super().__init__(f"{stage} exited with code {code}")
        self.stage, self.code, self.tail = stage, code, tail


class _Run:
    """State of one training job (what ``train.pause`` needs to stop it)."""

    def __init__(self) -> None:
        self.step: Step | None = None
        self.tracker: Tracker | None = None
        self.watch: _Watch | None = None
        self.pause_requested = False
        self.pause_t = 0.0
        self.pause_timer: asyncio.TimerHandle | None = None
        self.pause_stop = False     # the process is being killed because of a pause, not a cancel
        self.kept_epoch: int | None = None  # set when a failed run left a resumable checkpoint behind

    def request_pause(self) -> None:
        if not self.pause_requested:
            self.pause_requested = True
            self.pause_t = time.monotonic()
            # A timer, not a poll: if no checkpoint is confirmed within the patience, stop anyway.
            self.pause_timer = asyncio.get_running_loop().call_later(PAUSE_PATIENCE_S + 0.1, self._recheck)
        self._recheck()

    def _recheck(self) -> None:
        watch = self.watch
        if watch is not None:
            watch.check()

    def stop_for_pause(self) -> None:
        if self.pause_stop:
            return
        self.pause_stop = True
        step = self.step
        if step is not None:
            aio.spawn(step.kill(), name="train-pause-kill")

    def close(self) -> None:
        if self.pause_timer is not None:
            self.pause_timer.cancel()
            self.pause_timer = None


class _Watch:
    """Follows the end-of-epoch lines of one Applio training process to know which checkpoint is safely on disk,
    and decides when a requested pause may stop the process.

    Applio prints the end-of-epoch line first and saves the checkpoint right after it, so the save of epoch ``e`` is
    known to be complete only once a line of epoch ``e + 1`` shows up. A kill in between would tear the files; the
    pause therefore waits for that confirmation (older pairs are removed only then, see ``resume.prune_checkpoints``).
    """

    def __init__(self, run: "_Run", tracker: Tracker, logdir: Path, total: int, saved: tuple[int, int] | None,
                 spe: int | None, on_confirm) -> None:
        self.run, self.tracker, self.logdir, self.total = run, tracker, logdir, total
        self.every = checkpoint_every(total)
        self.saved = saved                      # (epoch, step) of the newest confirmed checkpoint
        self.spe = spe                          # steps per epoch, learned from the end-of-epoch lines
        self.inflight: tuple[int, int] | None = None
        self.on_confirm = on_confirm
        self.last_done = tracker.epochs_done
        self._books: asyncio.Future[Any] | None = None   # the last bookkeeping write (they run in order)

    def feed(self) -> None:
        """Called for every output line of the train step (loop thread)."""
        t = self.tracker
        if t.epochs_done != self.last_done:
            self.last_done = t.epochs_done
            if t.epochs_done and t.last_step:
                self.spe = max(1, t.last_step // t.epochs_done)
            if t.epochs_done % self.every == 0 and t.epochs_done < self.total and t.last_step:
                self.inflight = (t.epochs_done, t.last_step)
        if self.inflight is not None and t.epoch > self.inflight[0]:
            self.saved, self.inflight = self.inflight, None
            self._book(self.saved[0], self.saved[1], self.spe)
        self._maybe_stop()

    def _book(self, epoch: int, step_no: int, spe: int | None) -> None:
        """Prune older checkpoints and record the confirmed one, on the IO pool, after the previous bookkeeping."""
        prev = self._books

        async def work() -> None:
            if prev is not None:
                await asyncio.gather(prev, return_exceptions=True)
            try:
                await aio.run_io(prune_checkpoints, self.logdir, step_no)
                await aio.run_io(self.on_confirm, epoch, step_no, spe)
            except Exception:  # noqa: BLE001 - bookkeeping must never break the training
                log.exception("checkpoint bookkeeping failed")

        self._books = aio.spawn(work(), name="train-checkpoint")

    async def settle(self) -> None:
        """Wait for the bookkeeping writes in flight (before the run's end is recorded)."""
        if self._books is not None:
            await asyncio.gather(self._books, return_exceptions=True)

    def check(self) -> None:
        self._maybe_stop()

    def _maybe_stop(self) -> None:
        run, t = self.run, self.tracker
        if not run.pause_requested or run.pause_stop or t.stage != "train":
            return
        if self.inflight is not None or t.epochs_done >= self.total:
            return  # a save is being written, or the training is about to finish by itself
        if self.saved is None:
            return  # nothing to resume from yet: wait for the first checkpoint
        if self.saved[0] >= t.epochs_done or time.monotonic() - run.pause_t >= PAUSE_PATIENCE_S:
            run.stop_for_pause()


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def _dataset_sig(dataset_dir: Path) -> str:
    """Fingerprint of the recorded parts: a paused run only resumes on the data it extracted its features from."""
    h = hashlib.sha1()
    for p in dataset_parts(dataset_dir):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _paused_info(ctx: Context, voice_id: str) -> dict[str, Any] | None:
    meta = voices.read_meta(config.data_dir(ctx), voice_id)
    p = meta.get("training_paused")
    return p if isinstance(p, dict) and isinstance(p.get("epoch"), int) else None


def _set_meta(ctx: Context, voice_id: str, **changes: Any) -> None:
    """Set keys of ``voice.json``; a value of ``None`` removes the key."""
    def mutate(meta: dict[str, Any]) -> None:
        for k, v in changes.items():
            if v is None:
                meta.pop(k, None)
            else:
                meta[k] = v
    voices.update_meta(config.data_dir(ctx), voice_id, mutate)


def _check_runtime(ctx: Context) -> tuple[str, str]:
    paths = ctx.runtime.get("paths", {})
    interp = ctx.runtime.get("python", {}).get("rvc")
    applio = paths.get("applio")
    if not interp or not applio or not Path(interp).exists() or not (Path(applio) / "core.py").is_file():
        raise EngineError("not_ready", "RVC training runtime (Applio) is not installed; run the setup again")
    return interp, applio


def _discard_paused(ctx: Context, voice_id: str) -> None:
    discard_logdir(ctx.runtime, voice_id)
    _set_meta(ctx, voice_id, training_paused=None, training_run=None)


@method("train.start")
async def train_start(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    voice_id = params.get("voice_id")
    if not isinstance(voice_id, str) or not _VOICE_ID.match(voice_id):
        raise EngineError("bad_request", "voice_id is missing or invalid")
    epochs = params.get("epochs")
    if epochs is not None:
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 2000:
            raise EngineError("bad_request", "epochs must be an integer from 1 to 2000, or null for auto")
    restart = params.get("restart", False)
    if not isinstance(restart, bool):
        raise EngineError("bad_request", "restart must be true or false")
    voice_dir = Path(ctx.data_dir) / "voices" / voice_id
    if not await aio.run_io((voice_dir / "voice.json").is_file):
        raise EngineError("not_found", f"voice {voice_id!r} does not exist")
    paused = await aio.run_io(_paused_info, ctx, voice_id)
    if paused is not None and not restart:
        raise EngineError(
            "paused",
            f"A training of this voice is paused at epoch {paused['epoch']} of {paused['total_epochs']}. "
            "Resume it, or discard it to start over.",
        )
    return await _launch(ctx, voice_id, voice_dir, epochs, None, discard_first=paused is not None)


@method("train.resume")
async def train_resume(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    voice_id = params.get("voice_id")
    if not isinstance(voice_id, str) or not _VOICE_ID.match(voice_id):
        raise EngineError("bad_request", "voice_id is missing or invalid")
    voice_dir = Path(ctx.data_dir) / "voices" / voice_id
    if not await aio.run_io((voice_dir / "voice.json").is_file):
        raise EngineError("not_found", f"voice {voice_id!r} does not exist")
    paused = await aio.run_io(_paused_info, ctx, voice_id)
    if paused is None:
        raise EngineError("bad_request", "this voice has no paused training to resume")
    if paused.get("dataset_sig") not in (None, await aio.run_io(_dataset_sig, voice_dir / "dataset")):
        raise EngineError(
            "bad_request",
            "The recordings changed after the training was paused, so it cannot continue. Discard the paused "
            "training and train again.",
        )
    return await _launch(ctx, voice_id, voice_dir, int(paused["total_epochs"]), paused, discard_first=False)


def _dataset_seconds(dataset_dir: Path) -> tuple[int, float]:
    parts = dataset_parts(dataset_dir)
    return len(parts), sum(wav_seconds(p) for p in parts)


async def _launch(ctx: Context, voice_id: str, voice_dir: Path, epochs: int | None, resume: dict[str, Any] | None,
                  discard_first: bool) -> dict[str, Any]:
    global _active
    await aio.run_io(_check_runtime, ctx)
    nparts, seconds = await aio.run_io(_dataset_seconds, voice_dir / "dataset")
    if not nparts:
        raise EngineError("bad_request", "this voice has no recorded dataset yet")
    if seconds < MIN_TRAIN_MINUTES * 60:
        raise EngineError(
            "bad_request",
            f"only {seconds / 60:.1f} min of speech recorded; at least {MIN_TRAIN_MINUTES:.0f} min is needed to train",
        )
    total = int(epochs) if epochs is not None else recommended_epochs(seconds / 60.0)

    if tts.models_status()["tts"] in ("loading", "paused"):
        raise EngineError("not_ready", "the speech models are still loading; try again in a moment")
    if _active is not None:
        raise EngineError("busy", "a training is already running")
    if tts.is_busy():
        raise EngineError("busy", "a generation is running; wait for it or cancel it before training")
    _active = voice_id   # (no await since the checks: the loop makes check-and-take atomic)
    run = _Run()
    gpu = contextlib.AsyncExitStack()
    try:
        # Hand the GPU over before answering (awaited: the models are unloaded and the worker processes have exited).
        # The context is closed by the job when it ends (done, error, cancel or pause): the speech models then reload
        # in the background.
        await gpu.enter_async_context(tts.gpu_released(ctx))
        if discard_first:
            await aio.run_io(_discard_paused, ctx, voice_id)

        async def body(job_id: str) -> None:
            await _job(ctx, job_id, run, voice_id, voice_dir, total, seconds, gpu, resume)

        job = ctx.start_job(body)
    except BaseException:
        await gpu.aclose()
        _active = None
        raise
    return {"job": job}


@method("train.pause")
async def train_pause(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    job = params.get("job")
    if not isinstance(job, str):
        raise EngineError("bad_request", "job is missing")
    run = _runs.get(job)
    if run is None:
        raise EngineError("not_found", "no training is running with this job id")
    tracker = run.tracker
    if tracker is None or tracker.stage != "train":
        raise EngineError(
            "not_ready",
            "A training can be paused once it has started learning (not while the recordings are being prepared). "
            "Try again in a moment.",
        )
    run.request_pause()  # the answer is the `train.paused` event, after the next checkpoint
    return {"pausing": True}


@method("train.cancel")
async def train_cancel(ctx: Context, params: dict[str, Any]) -> dict[str, Any]:
    """Cancel a running job (``{"job"}``: its task is cancelled, which stops the Applio tree), or discard a paused run
    (``{"voice_id"}``)."""
    job = params.get("job")
    voice_id = params.get("voice_id")
    if isinstance(job, str):
        ctx.cancel(job)
        return {}
    if isinstance(voice_id, str) and _VOICE_ID.match(voice_id):
        if _active == voice_id:
            raise EngineError("busy", "this voice is training; cancel the job instead")
        await aio.run_io(voices.read_meta, config.data_dir(ctx), voice_id)  # not_found when the voice is gone
        await aio.run_io(_discard_paused, ctx, voice_id)
        return {}
    raise EngineError("bad_request", "job or voice_id is missing")


# --------------------------------------------------------------------------------------------------------------

async def _job(ctx: Context, job: str, run: _Run, voice_id: str, voice_dir: Path, total: int, seconds: float,
               gpu: contextlib.AsyncExitStack, resume: dict[str, Any] | None) -> None:
    global _active
    started = time.monotonic()
    _runs[job] = run
    kept = ""
    try:
        await _train(ctx, job, run, voice_id, voice_dir, total, seconds, started, resume)
    except _Paused:
        pass  # `train.paused` was emitted by _train
    except _Shutdown:
        pass  # the engine is going away: the run stays recorded as running and becomes "interrupted" at next start
    except (_Cancelled, asyncio.CancelledError, KeyboardInterrupt):
        if not aio.closing():
            ctx.emit("train.error", {"code": "cancelled", "message": "Training was cancelled."}, job=job)
    except _StepFailed as e:
        if run.kept_epoch is not None:
            kept = f" Progress up to epoch {run.kept_epoch} was kept; you can resume from there."
        ctx.emit("train.error", {"code": "internal", "message": _explain(e) + kept}, job=job)
    except EngineError as e:
        ctx.emit("train.error", {"code": e.code, "message": e.message}, job=job)
    except Exception as e:  # noqa: BLE001 - always answer the app
        if run.kept_epoch is not None:
            kept = f" Progress up to epoch {run.kept_epoch} was kept; you can resume from there."
        ctx.emit("train.error", {"code": "internal", "message": f"Training failed: {e}{kept}"}, job=job)
    finally:
        run.close()
        await gpu.aclose()  # reload the speech model and the RVC worker (also after an error, a cancel or a pause)
        _runs.pop(job, None)
        _active = None


def _is_oom(tail: list[str]) -> bool:
    """GPU out-of-memory, or the Triton/ptxas kernel compiler failing to get host memory (same cure: free memory)."""
    low = "\n".join(tail).lower()
    return any(m in low for m in ("out of memory", "ptxas fatal", "memory allocation failure", "bad allocation"))


def _explain(e: _StepFailed) -> str:
    if _is_oom(e.tail):
        return "Training ran out of memory. Close other programs that use the GPU or a lot of memory and try again."
    last = e.tail[-1] if e.tail else "no output"
    return f"Training failed in the {e.stage} step (exit code {e.code}): {last}"


async def _train(ctx: Context, job: str, run: _Run, voice_id: str, voice_dir: Path, total: int, seconds: float,
                 started: float, resume: dict[str, Any] | None) -> None:
    applio = Path(ctx.runtime["paths"]["applio"])
    interp = str(ctx.runtime["python"]["rvc"])
    backend = str(ctx.runtime.get("device", {}).get("backend", "cuda"))
    model = model_name(voice_id)
    logdir = applio / "logs" / model
    core = str(applio / "core.py")
    tracker = Tracker(total)
    run.tracker = tracker
    last_emit = [0.0]
    log_dir = Path(ctx.data_dir) / "tmp"
    await aio.run_io(log_dir.mkdir, parents=True, exist_ok=True)
    # Full Applio output for diagnostics; a resumed run appends to the log of the paused one.
    applog = await aio.run_io((log_dir / f"train_{voice_id}.log").open, "a" if resume else "w", encoding="utf-8")
    paused = False
    kept = False
    shutdown = False

    def emit(force: bool = False) -> None:
        now = time.monotonic()
        if force or now - last_emit[0] >= EMIT_EVERY_S:
            last_emit[0] = now
            ctx.emit("train.progress", tracker.snapshot(), job=job)

    async def step(stage: str, args: list[str]) -> None:
        if aio.closing():
            raise _Shutdown()
        s = Step([interp, core, *args], cwd=str(applio))
        run.step = s
        if run.pause_stop:
            s.killed = True

        def on_line(line: str) -> None:
            applog.write(f"[{stage}] {line}\n")
            applog.flush()
            changed = tracker.feed(line)
            watch = run.watch
            if watch is not None and stage == "train":
                watch.feed()
            if changed:
                emit(force=tracker.stage != stage)

        code = await s.run(on_line)
        run.step = None
        if code == 0 and any("Training failed for model" in ln for ln in s.tail):
            code = 1  # Applio reports a crashed training worker on its output but still exits with 0
        if aio.closing():
            raise _Shutdown()
        if ctx.cancelled(job):
            raise _Cancelled()
        if run.pause_stop:
            raise _Paused()
        if s.killed:
            raise _Cancelled()
        if code != 0:
            raise _StepFailed(stage, code, list(s.tail))

    def marker(dataset_sig: str, **kw: Any) -> dict[str, Any]:
        return {"model_name": model, "total_epochs": total, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset_sig": dataset_sig, **kw}

    try:
        start_epoch = 0
        saved: tuple[int, int] | None = None
        spe: int | None = None
        batch = cache = None
        skip_prepare = False
        if resume is not None:
            await aio.run_io(kill_orphans, logdir)
            if await aio.run_io(lambda: logdir.is_dir() and features_ready(logdir)):
                skip_prepare = True
                step_no = await aio.run_io(repair, logdir)
                spe = resume.get("steps_per_epoch") if isinstance(resume.get("steps_per_epoch"), int) else None
                if step_no is not None:
                    ep = resume["epoch"] if resume.get("checkpoint_step") == step_no else (
                        step_no // spe if spe else resume["epoch"])
                    start_epoch = min(int(ep), total - 1)
                    saved = (start_epoch, step_no)
                else:
                    applog.write("[plan] no valid checkpoint left; training again from epoch 1 on the saved features\n")
                if isinstance(resume.get("batch_size"), int):
                    batch, cache = resume["batch_size"], bool(resume.get("cache_in_gpu"))
            else:
                applog.write("[plan] the saved features are gone; running the full pipeline again\n")
                await aio.run_io(shutil.rmtree, logdir, ignore_errors=True)
            applog.write(f"[plan] resume: skip_prepare={skip_prepare} start_epoch={start_epoch}\n")

        sig = await aio.run_io(_dataset_sig, voice_dir / "dataset")
        await aio.run_io(_set_meta, ctx, voice_id, training_paused=None, training_run=marker(
            sig, **({"checkpoint_epoch": saved[0], "checkpoint_step": saved[1]} if saved else {}),
            **({"steps_per_epoch": spe} if spe else {})))

        if not skip_prepare:
            # a fresh model every time (stale features would be reused)
            await aio.run_io(shutil.rmtree, logdir, ignore_errors=True)
            tracker.set_stage("preprocess")
            emit(force=True)
            await step("preprocess", [
                "preprocess", "--model-name", model, "--dataset-path", str(voice_dir / "dataset"),
                "--sample-rate", str(SAMPLE_RATE), "--cut-preprocess", "Automatic",
                "--chunk-len", "3.0", "--overlap-len", "0.3", "--normalization-mode", "pre",
            ])

            tracker.set_stage("extract")
            emit(force=True)
            await step("extract", [
                "extract", "--model-name", model, "--f0-method", "rmvpe", "--gpu", "0",
                "--sample-rate", str(SAMPLE_RATE), "--embedder-model", "contentvec",
            ])

        free_gb, _total_gb = await aio.run_io(free_vram_gb)
        if batch is None or cache is None:
            batch, cache = pick_batch_size(free_gb, seconds, backend)
        applog.write(f"[plan] epochs={total} batch={batch} cache_in_gpu={cache} free_vram_gb={free_gb} "
                     f"dataset_s={seconds:.0f}\n")

        def on_confirm(epoch: int, step_no: int, steps_per_epoch: int | None) -> None:  # IO pool
            _set_meta(ctx, voice_id, training_run=marker(
                sig, checkpoint_epoch=epoch, checkpoint_step=step_no, batch_size=batch, cache_in_gpu=cache,
                **({"steps_per_epoch": steps_per_epoch} if steps_per_epoch else {})))

        for attempt in range(3):
            tracker.set_stage("train")
            tracker.epoch = tracker.epochs_done = start_epoch
            tracker.frac = start_epoch / total
            run.watch = _Watch(run, tracker, logdir, total, saved, spe, on_confirm)
            await aio.run_io(_set_meta, ctx, voice_id, training_run=marker(
                sig, batch_size=batch, cache_in_gpu=cache,
                **({"checkpoint_epoch": saved[0], "checkpoint_step": saved[1]} if saved else {}),
                **({"steps_per_epoch": spe} if spe else {})))
            emit(force=True)
            try:
                await step("train", _train_args(model, total, batch, cache))
                break
            except _StepFailed as e:
                oom = _is_oom(e.tail)
                if not oom or attempt == 2 or (batch <= 2 and not cache):
                    raise
                if cache:
                    cache = False
                else:
                    batch = max(2, batch // 2)
                if resume is None:
                    await aio.run_io(_drop_checkpoints, logdir)
                    saved = None
                log.warning("out of memory, retrying with batch %s, cache=%s", batch, cache)
        if run.watch is not None:
            await run.watch.settle()
        run.watch = None

        # `core.py train` builds the index itself; build it explicitly only if that silently failed.
        index = logdir / f"{model}.index"
        if not await aio.run_io(index.is_file):
            tracker.set_stage("index")
            emit(force=True)
            await step("index", ["index", "--model-name", model, "--index-algorithm", "Auto"])
        pth = await aio.run_io(_final_pth, logdir, model)
        if pth is None or not await aio.run_io(index.is_file):
            have = sorted(p.name for p in logdir.glob("*.pth")) + sorted(p.name for p in logdir.glob("*.index"))
            applog.write(f"[error] final files missing; logdir has: {have}\n")
            raise EngineError(
                "internal",
                f"Training finished but the model or index file is missing (see {applog.name}).",
            )
        done_epochs = int(_FINAL_PTH.match(pth.name).group("e"))  # type: ignore[union-attr]

        rvc_dir = voice_dir / "rvc"
        await aio.run_io(_atomic_copy, pth, rvc_dir / "model.pth")
        await aio.run_io(_atomic_copy, index, rvc_dir / "model.index")
        await aio.run_io(_set_meta, ctx, voice_id, trained=True, trained_epochs=done_epochs,
                         trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"), training_run=None, training_paused=None)

        tracker.set_stage("index")
        tracker.frac = 1.0
        emit(force=True)
        ctx.emit("train.done", {"voice_id": voice_id, "epochs": done_epochs,
                                "elapsed_s": round(time.monotonic() - started, 1)}, job=job)
    except _Paused:
        paused = True
        if run.watch is not None:
            await run.watch.settle()
        await aio.run_io(_commit_pause, ctx, job, run, voice_id, voice_dir, logdir, total, model)
        raise
    except _Shutdown:
        shutdown = True
        raise
    except (_Cancelled, asyncio.CancelledError, KeyboardInterrupt):
        if aio.closing():  # cancelled by the engine shutdown, not by the user: leave the run resumable
            shutdown = True
            raise
        await aio.run_io(_set_meta, ctx, voice_id, training_run=None, training_paused=None)
        raise
    except BaseException:
        # A failure while learning (out of memory, a crashed worker) keeps what was saved so it can be resumed.
        if run.watch is not None:
            await run.watch.settle()
        step_no = (await aio.run_io(repair, logdir)
                   if tracker.stage == "train" and await aio.run_io(logdir.is_dir) else None)
        watch = run.watch
        spe_now = watch.spe if watch else None
        epoch_now: int | None = None
        if step_no is not None:
            if watch and watch.saved and watch.saved[1] == step_no:
                epoch_now = watch.saved[0]
            elif spe_now:
                epoch_now = step_no // spe_now
            elif resume is not None:
                epoch_now = start_epoch
        if step_no is not None and epoch_now is not None and epoch_now < total:
            kept = True
            run.kept_epoch = epoch_now
            await aio.run_io(_set_meta, ctx, voice_id, training_run=None, training_paused={
                "epoch": epoch_now, "total_epochs": total, "model_name": model, "interrupted": True,
                "paused_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset_sig": _dataset_sig(voice_dir / "dataset"), "checkpoint_step": step_no,
                **({"steps_per_epoch": spe_now} if spe_now else {}),
                **({"batch_size": batch, "cache_in_gpu": bool(cache)} if batch else {}),
            })
        else:
            await aio.run_io(_set_meta, ctx, voice_id, training_run=None, training_paused=None)
        raise
    finally:
        applog.close()
        run.watch = None
        keep_env = bool(os.environ.get("MIMVOX_KEEP_APPLIO_LOGS"))
        if not paused and not kept and not keep_env and not shutdown:
            # Applio keeps GBs of features and checkpoints per model
            await aio.run_io(shutil.rmtree, logdir, ignore_errors=True)


def _commit_pause(ctx: Context, job: str, run: _Run, voice_id: str, voice_dir: Path, logdir: Path, total: int,
                  model: str) -> None:
    """The Applio process is gone: record where a resume continues and tell the app."""
    watch = run.watch
    tracker = run.tracker
    step_no = repair(logdir)  # also drops a torn newer pair, so Applio cannot pick it up on resume
    spe = watch.spe if watch else None
    epoch = 0
    if step_no is not None:
        if watch and watch.saved and watch.saved[1] == step_no:
            epoch = watch.saved[0]
        elif spe:
            epoch = step_no // spe
        elif watch and watch.saved:
            epoch = watch.saved[0]
    train_args = _read_plan(ctx, voice_id)
    info = {
        "epoch": epoch, "total_epochs": total, "model_name": model,
        "paused_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_sig": _dataset_sig(voice_dir / "dataset"),
        **({"steps_per_epoch": spe} if spe else {}),
        **({"checkpoint_step": step_no} if step_no is not None else {}),
        **train_args,
    }
    _set_meta(ctx, voice_id, training_run=None, training_paused=info)
    ctx.emit("train.paused", {
        "voice_id": voice_id, "epoch": epoch, "total_epochs": total, "paused_at": info["paused_at"],
        "reached_epoch": tracker.epochs_done if tracker else epoch,
    }, job=job)


def _read_plan(ctx: Context, voice_id: str) -> dict[str, Any]:
    """Batch size / GPU cache of the run that is ending, taken from its ``training_run`` marker."""
    try:
        run = voices.read_meta(config.data_dir(ctx), voice_id).get("training_run") or {}
    except EngineError:
        return {}
    out: dict[str, Any] = {}
    if isinstance(run.get("batch_size"), int):
        out["batch_size"] = run["batch_size"]
        out["cache_in_gpu"] = bool(run.get("cache_in_gpu"))
    return out


# -------------------------------------------------------------------------------------- start-up recovery

def recover_interrupted(ctx: Context) -> None:
    """Called once when the engine starts: a voice whose ``training_run`` marker is still set lost its engine
    (app closed or crashed) during a training. If its Applio folder holds a valid checkpoint the run becomes a paused
    one with ``interrupted: true`` so it can be resumed; the folder itself is never touched here beyond dropping a
    torn newer checkpoint pair."""
    data = config.data_dir(ctx)
    for vid in voices.list_voice_ids(data):
        try:
            meta = voices.read_meta(data, vid)
            run = meta.get("training_run")
            if not isinstance(run, dict):
                continue
            logdir = logdir_for(ctx.runtime, vid)
            info: dict[str, Any] | None = None
            if logdir is not None and logdir.is_dir():
                kill_orphans(logdir)
                step_no = repair(logdir)
                spe = run.get("steps_per_epoch") if isinstance(run.get("steps_per_epoch"), int) else None
                total = int(run.get("total_epochs") or 0)
                if step_no is not None and total > 0 and features_ready(logdir):
                    if run.get("checkpoint_step") == step_no and isinstance(run.get("checkpoint_epoch"), int):
                        epoch = run["checkpoint_epoch"]
                    elif spe:
                        epoch = step_no // spe
                    else:
                        epoch = int(run.get("checkpoint_epoch") or 0)
                    if epoch < total:
                        try:
                            when = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
                                (logdir / f"G_{step_no}.pth").stat().st_mtime))
                        except OSError:
                            when = time.strftime("%Y-%m-%dT%H:%M:%S")
                        info = {
                            "epoch": epoch, "total_epochs": total, "model_name": model_name(vid),
                            "paused_at": when, "interrupted": True, "checkpoint_step": step_no,
                            **{k: run[k] for k in ("steps_per_epoch", "dataset_sig", "batch_size", "cache_in_gpu")
                               if k in run},
                        }
            _set_meta(ctx, vid, training_run=None, training_paused=info)
            log.info("training of %s was interrupted: %s", vid,
                     f"resumable from epoch {info['epoch']}" if info else "nothing to resume")
        except Exception:  # noqa: BLE001 - one broken voice must not stop the engine from starting
            log.exception("could not check the interrupted training of %s", vid)


def _train_args(model: str, total: int, batch: int, cache: bool) -> list[str]:
    args = [
        "train", "--model-name", model, "--sample-rate", str(SAMPLE_RATE), "--total-epoch", str(total),
        # G/D checkpoints are what a resume continues from (pause, closed app, crash): one pair every few epochs.
        # No --save-only-latest: that overwrites one file in place, so a kill during the save would destroy the only
        # checkpoint. The watcher (_Watch) removes older pairs once the next epoch runs. The final weights are always
        # exported at the last epoch.
        "--save-every-epoch", str(checkpoint_every(total)),
        "--batch-size", str(batch), "--gpu", "0", "--pretrained", "--vocoder", "HiFi-GAN",
        "--index-algorithm", "Auto",
    ]
    if cache:
        args.append("--cache-data-in-gpu")
    return args


def _final_pth(logdir: Path, model: str) -> Path | None:
    best: tuple[int, Path] | None = None
    for p in logdir.glob(f"{model}_*e_*s.pth"):
        m = _FINAL_PTH.match(p.name)
        if m and m.group("m") == model and (best is None or int(m.group("e")) > best[0]):
            best = (int(m.group("e")), p)
    return best[1] if best else None


def _drop_checkpoints(logdir: Path) -> None:
    for pattern in ("G_*.pth", "D_*.pth"):
        for p in logdir.glob(pattern):
            p.unlink(missing_ok=True)


async def on_shutdown() -> None:
    """Engine shutdown: stop the Applio tree of a running training and await its exit. The run's marker stays in
    ``voice.json``, so the next engine start turns it into an interrupted, resumable run (see ``recover_interrupted``)."""
    from . import proc  # noqa: PLC0415

    await proc.kill_all()
