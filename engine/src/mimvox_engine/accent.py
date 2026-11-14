"""English from a shipped native reference, converted to the user's timbre: OmniVoice copies its reference's accent."""
from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import audio_util as au
from .registry import EngineError

log = logging.getLogger("mimvox.accent")

NATIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "native")
ACCENTS = ("own", "native-us")
NATIVE_VOICES = ("auto", "male", "female")
NATIVE_LANG = "en"

# Low so the retrieval index (the user's own phoneme features) does not pull the native vowels back towards the
# user's accent. Not yet tuned by listening tests.
NATIVE_INDEX_RATE = float(os.environ.get("MIMVOX_RVC_NATIVE_INDEX", "0.2"))
MAX_TRANSPOSE = 12

F0_MIN_HZ, F0_MAX_HZ = 60.0, 500.0
F0_SR = 16000
F0_MAX_S = 30.0
F0_MIN_VOICED = 25

FALLBACK_TEXT = {
    "no_converter": "The native accent of a voice that is not trained needs the zero-shot voice converter, which is "
                    "not installed: it spoke with its own reference instead. Run setup to install it, or train the "
                    "voice.",
    "not_english": "The native accent is for English output only: it spoke with the voice's own reference.",
    "unavailable": "The native reference clips are missing from this install: it spoke with the voice's own reference.",
}


def parse(params: dict[str, Any], key: str = "accent") -> str:
    v = params.get(key)
    if v in (None, ""):
        return "own"
    if v not in ACCENTS:
        raise EngineError("bad_request", f"'{key}' must be \"native-us\" or \"own\"")
    return str(v)


def parse_voice(params: dict[str, Any], key: str = "native_voice") -> str:
    v = params.get(key)
    if v in (None, ""):
        return "auto"
    if v not in NATIVE_VOICES:
        raise EngineError("bad_request", f"'{key}' must be \"auto\", \"male\" or \"female\"")
    return str(v)


@dataclass(frozen=True)
class NativeRef:
    gender: str
    path: str
    text: str
    f0: float
    speaker: str


_NATIVE: dict[str, NativeRef] | None = None
_NATIVE_LOCK = threading.Lock()


def native_refs() -> dict[str, NativeRef]:
    global _NATIVE
    with _NATIVE_LOCK:
        if _NATIVE is None:
            _NATIVE = _load_native()
        return _NATIVE


def _load_native() -> dict[str, NativeRef]:
    out: dict[str, NativeRef] = {}
    try:
        with open(os.path.join(NATIVE_DIR, "voices.json"), encoding="utf-8") as fh:
            manifest = json.load(fh)
        for gender, v in manifest.get("voices", {}).items():
            wav = os.path.join(NATIVE_DIR, v["file"])
            with open(os.path.join(NATIVE_DIR, v["text_file"]), encoding="utf-8") as fh:
                text = " ".join(fh.read().split())
            if os.path.isfile(wav) and text:
                out[gender] = NativeRef(gender, wav, text, float(v["f0_median"]), str(v.get("speaker", "")))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log.warning("native reference clips unavailable (%s): %s", NATIVE_DIR, exc)
    return out


def available() -> bool:
    return bool(native_refs())


def median_f0(x: np.ndarray, sr: int) -> float | None:
    """None without enough voicing. Seconds of CPU: call it on an IO thread."""
    import librosa  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float32)[: int(F0_MAX_S * sr)]
    y = au.resample(x, sr, F0_SR)
    if len(y) < F0_SR // 2:
        return None
    f0, voiced, _ = librosa.pyin(y, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=F0_SR, frame_length=1024, hop_length=320)
    v = f0[voiced & np.isfinite(f0)]
    if len(v) < F0_MIN_VOICED:
        return None
    return float(np.median(v))


_F0_CACHE: dict[tuple, float | None] = {}
_F0_LOCK = threading.Lock()


def clip_f0(path: str) -> float | None:
    """Cached per file state for the session. Blocking: IO thread."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (os.path.abspath(path), st.st_mtime_ns, st.st_size)
    with _F0_LOCK:
        if key in _F0_CACHE:
            return _F0_CACHE[key]
    try:
        x, sr = au.read_mono(path)
        f0 = median_f0(x, sr)
    except Exception:  # noqa: BLE001 - no pitch is a usable answer (transpose 0)
        log.exception("could not measure the pitch of %s", path)
        f0 = None
    with _F0_LOCK:
        _F0_CACHE[key] = f0
    log.info("median F0 of %s: %s Hz", path, f"{f0:.1f}" if f0 else "unknown")
    return f0


def peek_native(user_reference: str, choice: str = "auto") -> tuple[bool, NativeRef | None]:
    """(whether the pitch is measured already, the native clip it picks), without measuring anything."""
    try:
        st = os.stat(user_reference)
    except OSError:
        return False, None
    key = (os.path.abspath(user_reference), st.st_mtime_ns, st.st_size)
    with _F0_LOCK:
        if key not in _F0_CACHE:
            return False, None
        f0 = _F0_CACHE[key]
    return True, pick_native(f0, choice)


def transpose(user_f0: float | None, native_f0: float) -> int:
    if not user_f0 or not native_f0:
        return 0
    return max(-MAX_TRANSPOSE, min(MAX_TRANSPOSE, round(12.0 * math.log2(user_f0 / native_f0))))


def pick_native(user_f0: float | None, choice: str = "auto") -> NativeRef | None:
    refs = native_refs()
    if not refs:
        return None
    if choice in refs:
        return refs[choice]
    if not user_f0:
        return refs.get("male") or next(iter(refs.values()))
    return min(refs.values(), key=lambda r: abs(math.log2(user_f0 / r.f0)))


@dataclass(frozen=True)
class Choice:
    accent: str
    native_voice: str = "auto"
    fallback: str | None = None
    converter: str | None = None

    def fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {"accent": self.accent}
        if self.converter:
            out["accent_converter"] = self.converter
        if self.fallback:
            out["accent_fallback"] = self.fallback
            out["accent_note"] = FALLBACK_TEXT[self.fallback]
        return out


def choose(requested: str, native_voice: str, lang: str, trained: bool, voice_match: float,
           zero_shot: bool = False) -> Choice:
    """The zero-shot converter wins even for trained voices: RVC turned the native clip metallic with an overshooting
    pitch shift (measured: high-band flatness 0.32 vs 0.26 zero-shot, median pitch 148 Hz vs the recording's 135 Hz)."""
    if requested != "native-us":
        return Choice("own")
    converter = "zero-shot" if zero_shot else "rvc" if trained and voice_match > 0 else None
    reason = ("not_english" if (lang or "").lower() != NATIVE_LANG
              else "unavailable" if not available()
              else "no_converter" if converter is None
              else None)
    if reason:
        return Choice("own", native_voice, reason)
    return Choice("native-us", native_voice, converter=converter)


@dataclass(frozen=True)
class Native:
    ref: NativeRef
    pitch: int
    index_rate: float
    user_f0: float | None
    converter: str = "rvc"

    def fields(self) -> dict[str, Any]:
        return {"native_voice": self.ref.gender, "pitch": self.pitch}


def prepare(choice: Choice, user_reference: str, voice_match: float) -> Native | None:
    """None for ``own``. Measures the user's pitch: IO thread."""
    if choice.accent != "native-us":
        return None
    user_f0 = clip_f0(user_reference)
    ref = pick_native(user_f0, choice.native_voice)
    if ref is None:
        return None
    if choice.converter == "zero-shot":
        n = Native(ref, 0, 0.0, user_f0, "zero-shot")
        log.info("native accent: %s (%s, %.0f Hz) for a voice at %s Hz, zero-shot conversion", ref.gender,
                 ref.speaker, ref.f0, f"{user_f0:.0f}" if user_f0 else "?")
        return n
    n = Native(ref, transpose(user_f0, ref.f0), min(voice_match, NATIVE_INDEX_RATE), user_f0)
    log.info("native accent: %s (%s, %.0f Hz) for a voice at %s Hz: RVC pitch %+d, index %.2f", ref.gender,
             ref.speaker, ref.f0, f"{user_f0:.0f}" if user_f0 else "?", n.pitch, n.index_rate)
    return n
