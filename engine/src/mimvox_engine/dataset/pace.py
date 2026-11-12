"""Reference-clip analysis: candidate windows scored by SNR, speaking rate and long pauses.

The speech model copies the tempo and manner of ``reference.wav``, so a clean but slow, halting clip makes the cloned voice
sound slow. Every part is analysed once (on ``dataset.add``) into a handful of 15-20 s candidate windows. The final
pick happens when the whole voice is known: windows whose speaking rate is close to the voice's own median rate, with
few long pauses and a clean signal, win; parts whose preset title suggests unnatural reading (tongue twisters,
questions, numbers) are avoided.

CPU-light on purpose: one dB envelope per part (10 ms frames) plus ``scipy.signal.find_peaks``; windows are scored from
cumulative sums, no per-window audio processing.

Measures (per window):

- ``rate``: syllable nuclei per second of *active* time (window minus gaps of 400 ms or more). Syllable nuclei are
  peaks of the smoothed energy envelope (at least 3 dB above the dips around them, 70 ms apart).
- ``pauses``: fraction of the window spent in silent gaps of 400 ms or more (0..1).
- ``snr_db``: 90th minus 10th percentile of the frame level inside the window.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np

from ._audio import HOP_S, frame_db

WIN_MIN_S, WIN_MAX_S, WIN_FALLBACK_MAX_S, WIN_IDEAL_S = 15.0, 20.0, 30.0, 17.5
CUT_PAUSE_FRAMES = 20          # a pause of 200 ms is a safe place to cut
LONG_GAP_FRAMES = 40           # a gap of 400 ms or more counts as a "pause" (the speaker stopped)
PEAK_PROMINENCE_DB = 3.0
PEAK_DISTANCE_FRAMES = 7       # 70 ms: at most about 14 syllables per second
SMOOTH_FRAMES = 5              # 50 ms moving average of the dB envelope
MIN_ACTIVE_S = 3.0             # less speech than this in a window: no usable rate
MAX_CANDIDATES_PER_PART = 10
MAX_OVERLAP = 0.35             # two candidates of one part may share at most this fraction of the shorter one

# Score weights (score units are roughly "dB of SNR / 2"; a perfect studio clip scores about 20).
SNR_CAP_DB, SNR_WEIGHT = 40.0, 0.5
PAUSE_WEIGHT = 50.0            # 10 % of the window in long gaps costs 5
RATE_WEIGHT = 30.0             # 20 % off the median rate costs about 5.5
DURATION_WEIGHT = 0.2          # per second away from WIN_IDEAL_S
TITLE_PENALTY = 10.0
CLIP_PENALTY = 40.0

RATE_SLOW, RATE_FAST = 0.88, 1.12   # ratio to the voice's median rate for the "slow" / "fast" hints
MANY_PAUSES = 0.12

# English plus the words the other preset languages use for those parts (uk, de, fr, es, pl).
_BAD_TITLE = re.compile(
    r"tongue|twister|question|number|digit|dates?\b|times\b|spell|alphabet|acronym"
    r"|скоромов|питанн|числа|цифр|дати\b|абетк"
    r"|zungenbrecher|frage|zahlen|daten\b|uhrzeit"
    r"|virelangue|chiffre|nombres\b|heures\b"
    r"|trabalengua|pregunta|número|fechas?\b|horas\b|abecedario|alfabet"
    r"|łamań|pytani|liczb|daty\b|godziny\b", re.IGNORECASE)


def title_penalty(title: str | None) -> float:
    """Penalty for parts whose title suggests unnatural reading (tongue twisters, questions, numbers ...)."""
    return TITLE_PENALTY if title and _BAD_TITLE.search(title) else 0.0


class PartAnalysis:
    """Per-frame arrays of one part and O(1) window measures."""

    def __init__(self, y: np.ndarray, sr: int) -> None:
        from scipy.ndimage import uniform_filter1d  # noqa: PLC0415
        from scipy.signal import find_peaks  # noqa: PLC0415

        self.sr = sr
        self.total_s = len(y) / sr if sr else 0.0
        self.fdb = frame_db(y, sr)
        n = len(self.fdb)
        live = self.fdb[self.fdb > -90.0]
        self.ok = live.size >= 10 and self.total_s >= 3.0
        if not self.ok:
            self.thr = 0.0
            self.sil = np.ones(n, dtype=bool)
            self.cum_peaks = np.zeros(n + 1)
            self.cum_gap = np.zeros(n + 1)
            self.cuts = [0, n]
            return
        floor, speech = float(np.percentile(live, 10)), float(np.percentile(live, 90))
        self.thr = max(floor + 6.0, speech - 30.0)
        self.sil = self.fdb < self.thr
        peaks = np.zeros(n, dtype=np.int8)
        smooth = uniform_filter1d(self.fdb, SMOOTH_FRAMES, mode="nearest")
        idx, _ = find_peaks(smooth, height=self.thr + 3.0, prominence=PEAK_PROMINENCE_DB, distance=PEAK_DISTANCE_FRAMES)
        peaks[idx] = 1
        gap = np.zeros(n, dtype=np.int8)
        cuts = [0]
        for a, b in self._silent_runs():
            if b - a >= LONG_GAP_FRAMES:
                gap[a:b] = 1
            if b - a >= CUT_PAUSE_FRAMES:
                cuts.append((a + b) // 2)
        cuts.append(n)
        self.cuts = cuts
        self.cum_peaks = np.concatenate([[0], np.cumsum(peaks)])
        self.cum_gap = np.concatenate([[0], np.cumsum(gap)])

    def _silent_runs(self) -> list[tuple[int, int]]:
        s = np.concatenate([[0], self.sil.astype(np.int8), [0]])
        d = np.diff(s)
        return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))

    def measure(self, a: int, b: int) -> dict[str, float]:
        """Rate / pauses / SNR of frames ``[a, b)``."""
        n = max(b - a, 1)
        gap = float(self.cum_gap[b] - self.cum_gap[a])
        active_s = (n - gap) * HOP_S
        syll = float(self.cum_peaks[b] - self.cum_peaks[a])
        w = self.fdb[a:b]
        lv = w[w > -90.0]
        snr = float(np.percentile(lv, 90) - np.percentile(lv, 10)) if lv.size >= 10 else 0.0
        return {
            "start_s": a * HOP_S, "end_s": min(b * HOP_S, self.total_s), "duration_s": n * HOP_S,
            "rate": syll / active_s if active_s >= MIN_ACTIVE_S else 0.0,
            "pauses": gap / n, "snr_db": snr,
        }

    def windows(self, lo_s: float = WIN_MIN_S, hi_s: float = WIN_MAX_S, fallback_hi_s: float = WIN_FALLBACK_MAX_S,
                ideal_s: float = WIN_IDEAL_S) -> list[tuple[int, int]]:
        """Pause-aligned windows of ``lo_s``-``hi_s`` (15-20 s by default; widened to ``fallback_hi_s``, then fixed
        windows of ``ideal_s``, when the part has none)."""
        n, cuts = len(self.fdb), self.cuts
        for lo, hi in ((lo_s, hi_s), (lo_s, fallback_hi_s)):
            out = []
            for ia in range(len(cuts)):
                for ib in range(ia + 1, len(cuts)):
                    d = (cuts[ib] - cuts[ia]) * HOP_S
                    if d > hi:
                        break
                    if d >= lo:
                        out.append((cuts[ia], cuts[ib]))
            if out:
                return out
        if self.total_s <= fallback_hi_s:
            return [(0, n)] if self.total_s >= lo_s - 5.0 else []
        step, wlen = int(min(5.0, ideal_s / 2.0) / HOP_S), int(ideal_s / HOP_S)
        return [(a, min(a + wlen, n)) for a in range(0, max(n - wlen, 1), step)]


def part_rate(an: PartAnalysis) -> float:
    """Median-friendly speaking rate of a whole part (syllables per active second)."""
    if not an.ok:
        return 0.0
    return an.measure(0, len(an.fdb))["rate"]


def window_score(m: dict[str, float], median_rate: float, title_pen: float = 0.0, clip_pen: float = 0.0,
                 ideal_s: float = WIN_IDEAL_S) -> float:
    rate = m["rate"]
    if rate > 0 and median_rate > 0:
        rate_pen = RATE_WEIGHT * abs(math.log(rate / median_rate))
    else:
        rate_pen = RATE_WEIGHT * 0.5   # unknown rate: a mild penalty
    return round(SNR_WEIGHT * min(m["snr_db"], SNR_CAP_DB) - PAUSE_WEIGHT * m["pauses"] - rate_pen
                 - DURATION_WEIGHT * abs(m["duration_s"] - ideal_s) - title_pen - clip_pen, 2)


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    inter = min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"])
    short = min(a["end_s"] - a["start_s"], b["end_s"] - b["start_s"])
    return max(inter, 0.0) / short if short > 0 else 0.0


def pick_distinct(cands: list[dict[str, Any]], count: int, max_per_part: int | None = None) -> list[dict[str, Any]]:
    """Greedy best-first selection of ``count`` mutually (mostly) non-overlapping candidates; ``cands`` must carry a
    ``score`` and be sorted best first. ``max_per_part`` needs a ``part`` key."""
    out: list[dict[str, Any]] = []
    per_part: dict[Any, int] = {}
    for c in cands:
        if len(out) >= count:
            break
        if any(c.get("part") == o.get("part") and _overlap(c, o) > MAX_OVERLAP for o in out):
            continue
        if max_per_part is not None and per_part.get(c.get("part"), 0) >= max_per_part:
            continue
        out.append(c)
        per_part[c.get("part")] = per_part.get(c.get("part"), 0) + 1
    return out


def analyse_part(y: np.ndarray, sr: int, proxy_median: float = 0.0, clip_pen: float = 0.0) -> dict[str, Any]:
    """The meta entry fragment of a part: its overall ``rate`` and its best candidate windows (``cands``).

    ``proxy_median`` (the median rate of the voice's other parts, 0 when unknown) only decides which windows are worth
    keeping; the final ranking is redone by :func:`rank` once the whole voice is known.
    """
    an = PartAnalysis(y, sr)
    if not an.ok:
        return {"rate": 0.0, "cands": []}
    rate = part_rate(an)
    median = proxy_median or rate
    scored = []
    for a, b in an.windows():
        m = an.measure(a, b)
        scored.append({**{k: round(v, 3) for k, v in m.items()}, "score": window_score(m, median, 0.0, clip_pen)})
    scored.sort(key=lambda c: -c["score"])
    cands = pick_distinct(scored, MAX_CANDIDATES_PER_PART)
    return {"rate": round(rate, 3), "cands": cands}


def voice_median_rate(parts_meta: dict[str, dict[str, Any]], usable: set[int] | None = None) -> float:
    """Median of the per-part rates (parts that are usable and have a rate)."""
    rates = [float(i["rate"]) for k, i in parts_meta.items()
             if isinstance(i.get("rate"), (int, float)) and i["rate"] > 0 and (usable is None or int(k) in usable)]
    return float(np.median(rates)) if rates else 0.0


def rank(parts_meta: dict[str, dict[str, Any]], titles: dict[int, str], usable: set[int]) -> tuple[list[dict[str, Any]], float]:
    """All stored candidates of the usable parts, best first, scored against the voice's median rate.

    Returns ``(candidates, median_rate)``; every candidate has ``part``, ``start_s``, ``end_s``, ``duration_s``,
    ``rate``, ``pauses``, ``snr_db``, ``score`` and ``hint``.
    """
    median = voice_median_rate(parts_meta, usable)
    out: list[dict[str, Any]] = []
    for key, info in parts_meta.items():
        n = int(key)
        if n not in usable:
            continue
        clip_pen = CLIP_PENALTY if info.get("clipping_pct", 0.0) > 0.05 else 0.0
        t_pen = title_penalty(titles.get(n))
        for c in info.get("cands", []):
            m = {k: float(c[k]) for k in ("start_s", "end_s", "duration_s", "rate", "pauses", "snr_db")}
            out.append({"part": n, **c, "score": window_score(m, median, t_pen, clip_pen), "hint": hint(c["rate"], median)})
    out.sort(key=lambda c: -c["score"])
    return out, median


def hint(rate: float, median: float) -> str:
    """``"slow"`` / ``"fast"`` / ``"natural"`` against the voice's median rate."""
    if rate <= 0 or median <= 0:
        return "natural"
    r = rate / median
    return "slow" if r < RATE_SLOW else "fast" if r > RATE_FAST else "natural"
