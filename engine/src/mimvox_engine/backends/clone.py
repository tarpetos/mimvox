"""Voice-cloning models beside OmniVoice, each in its own venv and worker process. A model loads on its first use, so
an installed one costs no memory until it speaks."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .. import aio, components, config, media, reflang
from .. import audio_util as au
from ..registry import EngineError
from ..workerproc import WorkerCancelled, WorkerClient, WorkerDied, WorkerError, child_env, isolated, stop_idle
from . import SynthParams, TtsBackend

log = logging.getLogger("mimvox.clone")

HERE = Path(__file__).absolute().parent
STAMP = ".mimvox-setup-ok"
LOAD_TIMEOUT_S = 600.0
SYNTH_TIMEOUT_S = 600.0
CPU_STALL_S = 900.0


class WorkerBackend(TtsBackend):
    """``missing`` (not installed), ``idle`` (installed, loads when it first speaks), ``ready`` or ``paused``."""

    def __init__(
        self, name: str, label: str, languages: frozenset[str], required: tuple[str, ...], transcript: bool
    ) -> None:
        self.name, self.label, self.languages = name, label, languages
        self.required = required
        self.transcript = transcript
        self._client: WorkerClient | None = None
        self._runtime: dict[str, Any] = {}
        self._status = "missing"
        self._loaded = False
        self.last_load: dict[str, Any] = {}

    def python(self, runtime: dict[str, Any]) -> Path | None:
        return config.runtime_path(runtime, "python", self.name)

    def model_dir(self, runtime: dict[str, Any]) -> Path | None:
        d = config.runtime_path(runtime, "paths", f"{self.name}_model")
        if d:
            return d
        root = config.runtime_root(runtime)
        return root / "models" / self.name if root else None

    def installed(self, runtime: dict[str, Any], honour_off: bool = True) -> bool:
        if honour_off and components.off(self.name):
            return False
        py, model = self.python(runtime), self.model_dir(runtime)
        return bool(
            py
            and model
            and py.is_file()
            and (model / STAMP).is_file()
            and all((model / f).is_file() for f in self.required)
        )

    def status(self) -> str:
        return self._status

    def available(self) -> bool:
        return self._status in ("idle", "ready")

    def _device(self) -> str:
        device = config.backend(self._runtime)
        return device if device in ("cuda", "mps", "cpu") else "cuda"

    def _ensure_client(self) -> WorkerClient:
        if self._client is None:
            model_dir = self.model_dir(self._runtime)
            argv = [
                self.python(self._runtime),
                *isolated(),
                HERE / f"{self.name}_worker.py",
                "--model-dir",
                model_dir,
                "--device",
                self._device(),
            ]
            env = child_env(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                HF_HOME=str(model_dir / "hf-home"),
                PKUSEG_HOME=str(model_dir / "pkuseg"),
            )
            self._client = WorkerClient(self.name, argv, env, start_timeout=240.0)
            self._client.on_release.append(self._unloaded)
        return self._client

    def _unloaded(self) -> None:
        self._loaded = False

    def probe(self, runtime: dict[str, Any]) -> None:
        """Before the models load (the ``hello`` at start): installed or not, from the files alone."""
        if self._status == "missing":
            self._runtime = runtime
            self._status = "idle" if self.installed(runtime) else "missing"

    async def load(self, runtime: dict[str, Any]) -> None:
        self._runtime = runtime
        self._status = "idle" if self.installed(runtime) else "missing"

    async def release(self) -> None:
        if self._status in ("idle", "ready"):
            self._status = "paused"
        self._loaded = False
        if self._client is not None:
            await self._client.stop()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()

    def _limits(self) -> dict[str, float | None]:
        return {"timeout": None, "stall": CPU_STALL_S} if self._device() == "cpu" else {"timeout": SYNTH_TIMEOUT_S}

    async def _call(self, op: str, params: dict[str, Any], **limits: Any) -> dict[str, Any]:
        client = self._ensure_client()
        try:
            try:
                return await client.call(op, params, **limits)
            except WorkerError as exc:
                if exc.code != "no_vram":
                    raise
                stopped = await stop_idle(client)  # the released models' processes still held too much
                if not stopped:
                    raise
                log.warning("%s: still no room on the GPU; stopped %s and trying again", self.label, ", ".join(stopped))
                return await client.call(op, params, retry=False, **limits)
        except WorkerCancelled as exc:
            raise EngineError("cancelled", "generation cancelled") from exc
        except WorkerError as exc:
            code = exc.code if exc.code in ("bad_request", "not_found", "no_memory") else "internal"
            raise EngineError(code, f"{self.label}: {exc.message}") from exc
        except WorkerDied as exc:
            raise EngineError("internal", f"the {self.label} worker stopped: {exc}") from exc

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        for other in _ALL:  # one of these models at a time: two of them do not fit beside OmniVoice on 12 GB
            if other is not self and other._client is not None and other._client.alive() and not other._client.busy():
                await other._client.stop()
                other._loaded = False
        t0 = time.time()
        self.last_load = await self._call("load", {}, timeout=LOAD_TIMEOUT_S)
        self._loaded, self._status = True, "ready"
        log.info("%s loaded in %.1fs: %s", self.label, time.time() - t0, self.last_load)

    async def _prompt(self, reference: str, lang: str, params: SynthParams) -> dict[str, Any]:
        """The clip and, when known, the stretch of it Whisper transcribed (start, end, text)."""
        if params.reference_text:
            return {"ref": reference, "ref_text": params.reference_text}
        from .omnivoice import OMNIVOICE, reference_for

        ref = reference_for(reference, lang)
        side = reflang.read_sidecar(ref)
        if self.transcript and side is None:
            await OMNIVOICE.inspect(str(ref), detect=False)
            side = reflang.read_sidecar(ref)
            if side is None:
                log.warning("%s: no transcript of %s; cloning from the audio only", self.label, ref)
        prompt: dict[str, Any] = {"ref": ref}
        if side and side.get("text") and side.get("end"):
            prompt.update(ref_text=str(side["text"]), start=float(side.get("start") or 0.0), end=float(side["end"]))
        return prompt

    async def synthesize(self, text: str, lang: str, reference: str, params: SynthParams) -> str:
        lang = (lang or "").lower()
        if lang not in self.languages:
            raise EngineError("bad_request", f"{self.label} does not speak {lang!r}")
        if self._status in ("missing", "paused"):
            raise EngineError("not_ready", f"{self.label} is not available")
        await self._ensure_loaded()
        prompt = await self._prompt(reference, lang, params)
        t0 = time.time()
        res = await self._call(
            "synth",
            {
                "text": text,
                "lang": lang,
                "out": params.out_path,
                **prompt,
                "quality": params.quality,
                "expressiveness": params.temperature,
            },
            **self._limits(),
        )
        out = str(res.get("out") or params.out_path)
        if abs(float(params.duration_factor or 1.0) - 1.0) > 0.01:
            await aio.run_io(_stretch, out, 1.0 / float(params.duration_factor))
        log.info(
            "%s %s: %.2fs for %.2fs of audio", self.label, lang, time.time() - t0, float(res.get("duration_s", 0.0))
        )
        return out


def _stretch(path: str, rate: float) -> None:
    x, sr = au.read_mono(path)
    done = media.time_stretch(x, sr, rate)
    if done is not None:
        au.write_wav(path, done[0], sr)


VOXCPM_LANGS = frozenset(
    {
        "ar",
        "my",
        "zh",
        "da",
        "nl",
        "en",
        "fi",
        "fr",
        "de",
        "el",
        "he",
        "hi",
        "id",
        "it",
        "ja",
        "km",
        "ko",
        "lo",
        "ms",
        "no",
        "pl",
        "pt",
        "ru",
        "es",
        "sw",
        "sv",
        "tl",
        "th",
        "tr",
        "vi",
    }
)
CHATTERBOX_LANGS = frozenset(
    {
        "ar",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hi",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ru",
        "sv",
        "sw",
        "tr",
        "zh",
    }
)

_ALL: list[WorkerBackend] = []

VOXCPM = WorkerBackend(
    "voxcpm", "VoxCPM2", VOXCPM_LANGS, ("model.safetensors", "audiovae.pth", "config.json"), transcript=True
)
CHATTERBOX = WorkerBackend(
    "chatterbox",
    "Chatterbox",
    CHATTERBOX_LANGS,
    ("t3_mtl23ls_v2.safetensors", "s3gen.pt", "ve.pt", "conds.pt"),
    transcript=False,
)
_ALL += [VOXCPM, CHATTERBOX]
