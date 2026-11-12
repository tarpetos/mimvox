"""Native accent: English spoken from a native American reference voice, then turned into the user's timbre by RVC.

OmniVoice copies the accent of its reference clip, so a voice recorded in Ukrainian speaks English with a Ukrainian
accent, and RVC only changes the timbre of what it is given. With ``accent: "native-us"`` (``tts.generate``, Redub
``dub`` / ``respeak`` into English) the pipeline speaks from a shipped native reference instead of the user's clip and
lets the voice's RVC model put the user's timbre on it:

1. OmniVoice synthesizes with ``assets/native/en-us-{male,female}.wav`` (CMU ARCTIC speakers ``bdl`` and ``slt``,
   US English; the transcript ships next to each clip, so no Whisper pass and no sidecar are needed). The voice is
   picked from the user's median F0: the native speaker whose median is closer on a log scale (smallest pitch shift),
   unless ``native_voice`` overrides it.
2. RVC converts the result with the voice's model, transposed by ``round(12 * log2(user F0 / native F0))`` semitones
   so the model gets a pitch in the user's own range, at a low index rate (``NATIVE_INDEX_RATE``): the retrieval index
   holds the user's own phoneme features, and a high rate pulls the native vowels back towards them (the accent this
   is meant to remove); the timbre comes mostly from the model itself.

Without a trained voice (or with ``voice_match`` 0, which skips RVC) the native reference would replace the user's
timbre, so the request falls back to the voice's own reference and says so (``accent: "own"``,
``accent_fallback``). See contract/engine-protocol.md, "Native accent".
"""
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

# RVC index rate for native-accent speech (capped by voice_match), below the 0.3 that rvc.FOREIGN_INDEX_RATE allows
# for foreign speech: here the point is to keep the native vowels, which the index (the user's own phoneme features)
# pulls back towards the user's accent. Not yet tuned by listening tests; MIMVOX_RVC_NATIVE_INDEX overrides it.
NATIVE_INDEX_RATE = float(os.environ.get("MIMVOX_RVC_NATIVE_INDEX", "0.2"))
MAX_TRANSPOSE = 12  # semitones either way

F0_MIN_HZ, F0_MAX_HZ = 60.0, 500.0
F0_SR = 16000
F0_MAX_S = 30.0          # the median is taken over at most this much of the clip
F0_MIN_VOICED = 25       # frames (20 ms each): less voiced speech than 0.5 s gives no F0

FALLBACK_TEXT = {
    "untrained": "This voice is not trained, so a native accent would replace its timbre: it spoke with its own "
                 "reference instead. Train the voice to use a native accent.",
    "voice_match_off": "Voice match is 0 (no voice conversion), so a native accent would replace the timbre: it spoke "
                       "with the voice's own reference instead.",
    "not_english": "The native accent is for English output only: it spoke with the voice's own reference.",
    "unavailable": "The native reference clips are missing from this install: it spoke with the voice's own reference.",
}


# ------------------------------------------------------------------------------------------------ parameters

def parse(params: dict[str, Any], key: str = "accent") -> str:
    """The requested accent (default ``own``); anything else than ``own`` / ``native-us`` is ``bad_request``."""
    v = params.get(key)
    if v in (None, ""):
        return "own"
    if v not in ACCENTS:
        raise EngineError("bad_request", f"'{key}' must be \"native-us\" or \"own\"")
    return str(v)


def parse_voice(params: dict[str, Any], key: str = "native_voice") -> str:
    """Which native speaker to use (default ``auto``: picked from the voice's pitch)."""
    v = params.get(key)
    if v in (None, ""):
        return "auto"
    if v not in NATIVE_VOICES:
        raise EngineError("bad_request", f"'{key}' must be \"auto\", \"male\" or \"female\"")
    return str(v)


# ------------------------------------------------------------------------------------------------ native clips

@dataclass(frozen=True)
class NativeRef:
    gender: str          # "male" | "female"
    path: str            # the WAV OmniVoice clones (mono 24 kHz, under 12 s, so it is used whole)
    text: str            # its transcript
    f0: float            # its median F0 (Hz), measured with median_f0
    speaker: str         # corpus speaker id ("bdl", "slt")


_NATIVE: dict[str, NativeRef] | None = None
_NATIVE_LOCK = threading.Lock()


def native_refs() -> dict[str, NativeRef]:
    """The shipped native references by gender (``assets/native/voices.json``); empty when they are missing."""
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


# ------------------------------------------------------------------------------------------------ pitch

def median_f0(x: np.ndarray, sr: int) -> float | None:
    """Median F0 (Hz) of the voiced frames of a mono clip (pYIN, 60-500 Hz, 20 ms hop); None without enough voicing.
    CPU, a few seconds for 20 s of audio: call it on an IO thread."""
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
    """``median_f0`` of a WAV, cached per file state (path, mtime, size) for the session. IO thread."""
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
    """(whether the voice's pitch is measured already, the native clip it picks): ``voices.warm``'s "already warm"
    check, without measuring anything. IO thread (a stat)."""
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
    """Semitones that move the native speaker's median pitch to the user's (RVC ``pitch``), within +-12."""
    if not user_f0 or not native_f0:
        return 0
    return max(-MAX_TRANSPOSE, min(MAX_TRANSPOSE, round(12.0 * math.log2(user_f0 / native_f0))))


def pick_native(user_f0: float | None, choice: str = "auto") -> NativeRef | None:
    """The native reference to speak from: ``choice`` when it is ``male`` / ``female``, else the one whose median F0
    is closest to the user's on a log scale (the smallest RVC transpose). Unknown user pitch: male."""
    refs = native_refs()
    if not refs:
        return None
    if choice in refs:
        return refs[choice]
    if not user_f0:
        return refs.get("male") or next(iter(refs.values()))
    return min(refs.values(), key=lambda r: abs(math.log2(user_f0 / r.f0)))


# ------------------------------------------------------------------------------------------------ the plan

@dataclass(frozen=True)
class Choice:
    """What a request asked for and what it gets, decided at request time (cheap, no audio is read)."""
    accent: str                    # effective: "own" | "native-us"
    native_voice: str = "auto"     # the request's override
    fallback: str | None = None    # why "native-us" became "own" (a FALLBACK_TEXT key)

    def fields(self) -> dict[str, Any]:
        """``accent`` (+ ``accent_fallback`` / ``accent_note``) for responses and end events."""
        out: dict[str, Any] = {"accent": self.accent}
        if self.fallback:
            out["accent_fallback"] = self.fallback
            out["accent_note"] = FALLBACK_TEXT[self.fallback]
        return out


def choose(requested: str, native_voice: str, lang: str, trained: bool, voice_match: float) -> Choice:
    """Decide the accent of a request speaking ``lang`` with a voice that is ``trained`` (has an RVC model)."""
    if requested != "native-us":
        return Choice("own")
    reason = ("not_english" if (lang or "").lower() != NATIVE_LANG
              else "untrained" if not trained
              else "voice_match_off" if voice_match <= 0
              else "unavailable" if not available()
              else None)
    if reason:
        return Choice("own", native_voice, reason)
    return Choice("native-us", native_voice)


@dataclass(frozen=True)
class Native:
    """How a native-accent job speaks: the clip OmniVoice clones and the RVC settings that bring the user back."""
    ref: NativeRef
    pitch: int            # RVC transpose (semitones)
    index_rate: float
    user_f0: float | None

    def fields(self) -> dict[str, Any]:
        return {"native_voice": self.ref.gender, "pitch": self.pitch}


def prepare(choice: Choice, user_reference: str, voice_match: float) -> Native | None:
    """The native setup of a ``native-us`` choice (None for ``own``): measures the user's pitch (cached) and picks
    the native clip. IO thread."""
    if choice.accent != "native-us":
        return None
    user_f0 = clip_f0(user_reference)
    ref = pick_native(user_f0, choice.native_voice)
    if ref is None:  # vanished since choose(); not worth a failure
        return None
    n = Native(ref, transpose(user_f0, ref.f0), min(voice_match, NATIVE_INDEX_RATE), user_f0)
    log.info("native accent: %s (%s, %.0f Hz) for a voice at %s Hz: RVC pitch %+d, index %.2f", ref.gender,
             ref.speaker, ref.f0, f"{user_f0:.0f}" if user_f0 else "?", n.pitch, n.index_rate)
    return n
