"""Narration: how a book is cut, paced and levelled so it is pleasant to listen to (used by ``book.py``).

Text
    ``clean_text`` normalizes typography without removing punctuation (OmniVoice's prosody follows it, and
    ``backends/omnivoice_text.normalize`` keeps it too): "..." -> "…", a spaced hyphen or "--" -> an em dash, ``x''``
    TeX quotes -> curly ones, runs of "!!!" -> "!", no space before , . ; : ? !.

Segments
    ``segments`` cuts a chapter into whole sentences packed up to ``BUDGET_TOKENS`` (``tts.segment_text`` with a fixed
    budget: no streaming schedule, never across a paragraph). A sentence is cut only when it alone is over the budget,
    then at clause punctuation (; : — ,) with the no-stub rules of ``tts.segment_text``. Questions, exclamations,
    quotations and dialogue lines therefore stay whole. Each segment gets a ``kind`` from how it ends and what follows,
    and the pause after it comes from ``PAUSES`` (the one table of pause lengths).

Audio
    Every segment is trimmed of the model's own leading and trailing silence (``trim``) and faded in and out over
    ``FADE_MS``, so the inserted pauses are exactly ``PAUSES`` and no join clicks. ``measure`` gives its loudness
    (ITU-R BS.1770 K-weighted, gated), median pitch and speaking rate; ``Judge`` keeps the running medians of the
    chapter so far, flags an outlier (``LOUDNESS_OUTLIER_DB``, ``PITCH_OUTLIER_ST``, a speaking rate outside
    ``RATE_RANGE``: skipped or garbled words) for one re-generation, and evens out small level steps. A finished
    chapter is normalized as a whole to ``TARGET_LUFS`` with peaks under ``PEAK_DBFS`` (``ChapterMeter``, streamed:
    a chapter never sits in memory).

Pure numpy / scipy; no model calls here.
"""
from __future__ import annotations

import math
import re
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.signal import resample_poly, sosfilt, sosfilt_zi

# ------------------------------------------------------------------------------------------------ constants

BUDGET_TOKENS = 90          # whole sentences per segment, up to this (OmniVoice tokens; ~270 characters, ~15 s)
EXPRESSIVENESS = 0.6        # a book's default: a steadier narrator than Speak's 0.8 (the user's setting wins)

PAUSES: dict[str, int] = {  # silence heard after a segment (ms), by how it ends (see ``gap``)
    "clause": 220,      # a long sentence cut at , ; : —
    "sentence": 440,    # . (also when ? or ! end it: see below)
    "question": 440,
    "exclaim": 440,
    "ellipsis": 620,    # … mid-paragraph
    "dialogue": 650,    # a paragraph end followed by a dialogue line
    "paragraph": 820,   # any other paragraph end
    "title": 1200,      # after the chapter title
    "end": 700,         # the end of a chapter file (the next file starts with LEAD_MS)
}
LEAD_MS = 1600              # silence at the start of every chapter file
FADE_MS = 8                 # fade in / out at every cut
TRIM_DB = -42.0             # trim: frames quieter than this relative to the loudest frame are silence...
TRIM_PAD_MS = (30, 60)      # ...keeping this much before the first / after the last loud frame (breath, decay)

TARGET_LUFS = -18.0         # a chapter's integrated loudness
PEAK_DBFS = -1.0            # ...with sample peaks under this
MAX_GAIN_DB = 18.0

LOUDNESS_OUTLIER_DB = 4.0   # a segment this far from the running median is made again (once)
PITCH_OUTLIER_ST = 4.0
RATE_RANGE = (0.62, 1.6)    # seconds per character relative to the running median (outside: words skipped / babble)
LEVEL_TOLERANCE_DB = 1.5    # level steps within this stay (natural dynamics); larger ones are pulled back to it
LEVEL_MAX_FIX_DB = 6.0
JUDGE_MIN_CHARS = 25        # shorter segments (a title, "Yes.") neither judge nor are judged
JUDGE_HISTORY = 25
JUDGE_MIN_HISTORY = 4

_DIALOGUE = re.compile(r"^\s*[—–\-«\"“„‘'‹]")
_CLOSERS = "\"'”’»)]›"


# ------------------------------------------------------------------------------------------------ text

def clean_text(text: str) -> str:
    """Typography normalized for speech, punctuation kept (see the module notes). Paragraph breaks stay."""
    pars = []
    for par in re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n")):
        p = re.sub(r"\s+", " ", par).strip()
        if not p:
            continue
        p = re.sub(r"\.\s?\.\s?\.(?:\s?\.)*", "…", p)                 # ... / . . . -> …
        p = re.sub(r"(?<=\S) ?--+ ?(?=\S)", " — ", p)                # word--word / word -- word
        p = re.sub(r"(?<=\w) [-–] (?=\w)", " — ", p)                 # spaced hyphen or en dash between words
        p = re.sub(r"^[-–]\s", "— ", p)                              # a dialogue line opened with a hyphen
        p = p.replace("―", "—").replace("``", "“").replace("''", "”")
        p = re.sub(r"!{2,}", "!", p)
        p = re.sub(r"\?{2,}", "?", p)
        p = re.sub(r"\s+([,.;:?!…])", r"\1", p)
        p = re.sub(r"([,;:])(?=[^\s\d\"'”’»)\]])", r"\1 ", p)          # a missing space after , ; :
        pars.append(re.sub(r" {2,}", " ", p).strip())
    return "\n\n".join(pars)


def is_dialogue(paragraph: str) -> bool:
    return bool(_DIALOGUE.match(paragraph))


def _end_kind(seg: str) -> str:
    tail = seg.rstrip().rstrip(_CLOSERS).rstrip()
    if tail.endswith("…"):
        return "ellipsis"
    if tail.endswith("?") or tail.endswith("?!"):
        return "question"
    if tail.endswith("!"):
        return "exclaim"
    if tail.endswith("."):
        return "sentence"
    return "clause"


def segments(title: str | None, text: str, count: Callable[[str], int]) -> list[tuple[str, int, str]]:
    """``(segment, pause_after_ms, kind)`` for a chapter: its ``title`` (when given) as a segment of its own, then the
    text paragraph by paragraph (see the module notes)."""
    pars = [p for p in clean_text(text).split("\n\n") if p.strip()]
    from . import tts  # noqa: PLC0415 - tts imports the speech stack; narration stays importable without it

    out: list[tuple[str, int, str]] = []
    if title and title.strip():
        t = clean_text(title).replace("\n\n", " ")
        if not re.search(r"[.!?…:;]$", t.rstrip(_CLOSERS)):
            t += "."
        for seg, _ in tts.segment_text(t, count, fixed=BUDGET_TOKENS):
            out.append((seg, PAUSES["clause"], "clause"))
        if out:
            out[-1] = (out[-1][0], PAUSES["title"], "title")
    for j, par in enumerate(pars):
        segs = tts.segment_text(par, count, fixed=BUDGET_TOKENS)
        for i, (seg, _) in enumerate(segs):
            if i < len(segs) - 1:
                kind = _end_kind(seg)
            elif j + 1 < len(pars):
                kind = "dialogue" if is_dialogue(pars[j + 1]) else "paragraph"
            else:
                kind = "end"
            out.append((seg, PAUSES[kind], kind))
    if out and out[-1][2] != "end":
        out[-1] = (out[-1][0], PAUSES["end"], "end")
    return out


# ------------------------------------------------------------------------------------------------ audio

def gap(pause_ms: int) -> int:
    """The silence to insert for a pause of ``pause_ms``: ``trim`` already leaves ``TRIM_PAD_MS`` of near-silence around
    the speech, so the pause heard is what ``PAUSES`` says."""
    return max(0, int(pause_ms) - sum(TRIM_PAD_MS))


def fade(x: np.ndarray, sr: int, ms: float = FADE_MS) -> np.ndarray:
    n = min(int(sr * ms / 1000.0), len(x) // 2)
    if n <= 1:
        return x
    x = x.copy()
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, n))).astype(np.float32)
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def trim(x: np.ndarray, sr: int) -> np.ndarray:
    """The speech without the model's leading / trailing silence (10 ms frames under ``TRIM_DB`` of the loudest
    frame), keeping ``TRIM_PAD_MS``, then faded."""
    hop = max(1, sr // 100)
    n = len(x) // hop
    if n < 3:
        return fade(x.astype(np.float32), sr)
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
    loud = np.flatnonzero(rms > rms.max() * 10 ** (TRIM_DB / 20.0))
    if len(loud) == 0:
        return fade(x.astype(np.float32), sr)
    a = max(0, loud[0] * hop - int(sr * TRIM_PAD_MS[0] / 1000))
    b = min(len(x), (loud[-1] + 1) * hop + int(sr * TRIM_PAD_MS[1] / 1000))
    return fade(np.asarray(x[a:b], dtype=np.float32), sr)


def _k_sos(sr: int) -> np.ndarray:
    """BS.1770 K-weighting (the pre-filter's high shelf, then the RLB high-pass) as second-order sections for ``sr``."""
    g, q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    a_ = 10 ** (g / 40.0)
    w0 = 2 * math.pi * fc / sr
    al = math.sin(w0) / (2 * q)
    c = math.cos(w0)
    s1 = [a_ * ((a_ + 1) + (a_ - 1) * c + 2 * math.sqrt(a_) * al), -2 * a_ * ((a_ - 1) + (a_ + 1) * c),
          a_ * ((a_ + 1) + (a_ - 1) * c - 2 * math.sqrt(a_) * al),
          (a_ + 1) - (a_ - 1) * c + 2 * math.sqrt(a_) * al, 2 * ((a_ - 1) - (a_ + 1) * c),
          (a_ + 1) - (a_ - 1) * c - 2 * math.sqrt(a_) * al]
    q2, fc2 = 0.5003270373238773, 38.13547087602444
    w0 = 2 * math.pi * fc2 / sr
    al = math.sin(w0) / (2 * q2)
    c = math.cos(w0)
    s2 = [(1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al]
    sos = np.array([s1, s2], dtype=np.float64)
    sos[:, :3] /= sos[:, 3:4]
    sos[:, 3:] /= sos[:, 3:4]
    return sos


def _gated(energies: np.ndarray) -> float | None:
    """Integrated loudness (LUFS) from 100 ms K-weighted mean squares: 400 ms blocks with 75 % overlap, the absolute
    (-70 LUFS) and relative (-10 LU) gates. None for silence."""
    if len(energies) == 0:
        return None
    if len(energies) < 4:
        z = np.array([energies.mean()])
    else:
        z = np.convolve(energies, np.ones(4) / 4.0, mode="valid")
    lk = -0.691 + 10 * np.log10(z + 1e-20)
    z = z[lk > -70.0]
    if len(z) == 0:
        return None
    rel = -0.691 + 10 * math.log10(z.mean()) - 10.0
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * math.log10(z.mean())) if len(z) else None


def lufs(x: np.ndarray, sr: int) -> float | None:
    sub = sr // 10
    y = sosfilt(_k_sos(sr), np.asarray(x, dtype=np.float64))
    n = len(y) // sub
    if n == 0:
        return _gated(np.array([np.mean(y ** 2)])) if len(y) else None
    return _gated(np.mean(y[: n * sub].reshape(n, sub) ** 2, axis=1))


def f0_track(x: np.ndarray, sr: int) -> np.ndarray:
    """A rough pitch track (Hz, 0 = unvoiced) every 20 ms: normalized autocorrelation at 8 kHz, 60-400 Hz."""
    y = resample_poly(np.asarray(x, dtype=np.float64), 8000, sr) if sr != 8000 else np.asarray(x, dtype=np.float64)
    win, hop = 320, 160
    n = 1 + (len(y) - win) // hop if len(y) >= win else 0
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx] * np.hanning(win)[None, :]
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    spec = np.fft.rfft(fr, 1024, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, 1024, axis=1)[:, :win]
    ac /= ac[:, :1] + 1e-12
    lo, hi = 8000 // 400, 8000 // 60
    lag = lo + np.argmax(ac[:, lo:hi], axis=1)
    peak = ac[np.arange(n), lag]
    voiced = (peak > 0.5) & (rms > rms.max() * 10 ** (-30 / 20.0))
    return np.where(voiced, 8000.0 / lag, 0.0)


def measure(x: np.ndarray, sr: int, chars: int) -> dict[str, Any]:
    """Loudness (LUFS), median pitch (Hz or None), length and seconds per character of a trimmed segment."""
    f0 = f0_track(x, sr)
    v = f0[f0 > 0]
    dur = len(x) / float(sr)
    return {"lufs": lufs(x, sr), "f0": float(np.median(v)) if len(v) >= 5 else None, "dur": round(dur, 3),
            "spc": dur / max(chars, 1), "chars": chars}


class Judge:
    """Running medians of a chapter's segments; flags outliers and evens out level steps."""

    def __init__(self) -> None:
        self.hist: deque[dict[str, Any]] = deque(maxlen=JUDGE_HISTORY)

    @staticmethod
    def _med(vals: list[float]) -> float | None:
        return float(np.median(vals)) if vals else None

    def medians(self) -> tuple[float | None, float | None, float | None]:
        h = list(self.hist)
        return (self._med([m["lufs"] for m in h if m.get("lufs") is not None]),
                self._med([m["f0"] for m in h if m.get("f0")]),
                self._med([m["spc"] for m in h]))

    def deviation(self, m: dict[str, Any]) -> tuple[float, list[str]]:
        """How far ``m`` is from the running medians (1.0 = at an outlier limit) and the limits it passes."""
        if m["chars"] < JUDGE_MIN_CHARS or len(self.hist) < JUDGE_MIN_HISTORY:
            return 0.0, []
        lm, fm, sm = self.medians()
        score, why = 0.0, []
        if lm is not None and m.get("lufs") is not None:
            d = abs(m["lufs"] - lm) / LOUDNESS_OUTLIER_DB
            score = max(score, d)
            if d > 1:
                why.append(f"loudness {m['lufs'] - lm:+.1f} dB")
        if fm and m.get("f0"):
            d = abs(12 * math.log2(m["f0"] / fm)) / PITCH_OUTLIER_ST
            score = max(score, d)
            if d > 1:
                why.append(f"pitch {12 * math.log2(m['f0'] / fm):+.1f} st")
        if sm:
            r = m["spc"] / sm
            d = math.log(r) / math.log(RATE_RANGE[1] if r > 1 else 1 / RATE_RANGE[0])
            score = max(score, abs(d))
            if not RATE_RANGE[0] <= r <= RATE_RANGE[1]:
                why.append(f"pace x{r:.2f}")
        return score, why

    def accept(self, m: dict[str, Any]) -> None:
        if m["chars"] >= JUDGE_MIN_CHARS and m["dur"] >= 1.0:
            self.hist.append(m)

    def level_fix_db(self, m: dict[str, Any]) -> float:
        """The gain that pulls a level step larger than ``LEVEL_TOLERANCE_DB`` back to it (0 when within)."""
        lm = self.medians()[0]
        if lm is None or m.get("lufs") is None or len(self.hist) < JUDGE_MIN_HISTORY:
            return 0.0
        d = lm - m["lufs"]
        if abs(d) <= LEVEL_TOLERANCE_DB:
            return 0.0
        return float(np.clip(d - math.copysign(LEVEL_TOLERANCE_DB, d), -LEVEL_MAX_FIX_DB, LEVEL_MAX_FIX_DB))


class ChapterMeter:
    """Integrated loudness and sample peak of a chapter fed block by block (any block size, filter state kept)."""

    def __init__(self, sr: int) -> None:
        self.sr = sr
        self.sub = sr // 10
        self.sos = _k_sos(sr)
        self.zi = sosfilt_zi(self.sos) * 0.0
        self.acc = np.zeros(0)
        self.energies: list[float] = []
        self.peak = 0.0

    def feed(self, x: np.ndarray) -> None:
        if len(x) == 0:
            return
        self.peak = max(self.peak, float(np.max(np.abs(x))))
        y, self.zi = sosfilt(self.sos, np.asarray(x, dtype=np.float64), zi=self.zi)
        y = np.concatenate([self.acc, y])
        n = len(y) // self.sub
        if n:
            self.energies.extend(np.mean(y[: n * self.sub].reshape(n, self.sub) ** 2, axis=1).tolist())
        self.acc = y[n * self.sub:]

    def loudness(self) -> float | None:
        return _gated(np.asarray(self.energies))

    def gain(self) -> float:
        """The linear gain to ``TARGET_LUFS`` with peaks kept under ``PEAK_DBFS`` (1.0 for silence)."""
        lk = self.loudness()
        if lk is None or self.peak <= 0:
            return 1.0
        g_db = min(TARGET_LUFS - lk, MAX_GAIN_DB)
        g_db = min(g_db, PEAK_DBFS - 20 * math.log10(self.peak))
        return 10 ** (g_db / 20.0)
