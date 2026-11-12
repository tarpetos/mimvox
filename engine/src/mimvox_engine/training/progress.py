"""Turn Applio's console output into ``train.progress`` numbers (stage weights, epoch, ETA)."""
from __future__ import annotations

import re
import time
from collections import deque

# Share of the overall percentage per stage; train covers 15..95 and ends with the index build.
WEIGHTS = {"preprocess": (0.0, 5.0), "extract": (5.0, 15.0), "train": (15.0, 95.0), "index": (95.0, 100.0)}

_BAR = re.compile(r"(\d+)/(\d+)\s*\[")
_EPOCH = re.compile(r"\| epoch=(\d+) \| step=(\d+) \|(.*)")
_BATCH = re.compile(r"batch=(\d+)/(\d+)")


class Tracker:
    """Feed it lines of the current stage; read ``snapshot()`` for the numbers to emit."""

    def __init__(self, total_epochs: int) -> None:
        self.total_epochs = total_epochs
        self.stage = "preprocess"
        self.frac = 0.0            # 0..1 inside the current stage
        self.epoch = 0             # epoch being trained (1-based) once training started
        self.epochs_done = 0
        self._phase = 0            # extract: 0 pitch, 1 embeddings
        self._epoch_marks: deque[float] = deque(maxlen=31)  # monotonic time of recent epoch ends
        self.last_step: int | None = None
        self._batches_seen = 0     # batch lines seen in the current epoch (cached data is shuffled, so ids are not ordered)

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.frac = 0.0
        self._phase = 0

    def feed(self, line: str) -> bool:
        """Update from one output line; True when something progress-relevant changed."""
        st = self.stage
        if st == "train":
            return self._feed_train(line)
        if st == "extract":
            low = line.lower()
            if "embedding extraction" in low and "completed" not in low:
                self._phase, self.frac = 1, 0.5
                return True
            if "pitch extraction" in low and "starting" in low:
                self._phase, self.frac = 0, 0.0
                return True
        m = _BAR.search(line)
        if m and int(m.group(2)) > 0:
            f = min(1.0, int(m.group(1)) / int(m.group(2)))
            new = (self._phase + f) / 2.0 if st == "extract" else f
            self.frac = max(self.frac, new)
            return True
        return False

    def _feed_train(self, line: str) -> bool:
        low = line.lower()
        if "generating index" in low or "training has been successfully completed" in low:
            self.set_stage("index")
            self.epochs_done = self.total_epochs
            return True
        m = _EPOCH.search(line)
        if m:
            epoch = int(m.group(1))
            if epoch < self.epoch:  # worker output can arrive slightly out of order; never go backwards
                return False
            if epoch != self.epoch:
                self._batches_seen = 0
            self.epoch = epoch
            self.last_step = int(m.group(2))
            if "time=" in m.group(3):  # end-of-epoch line
                self.epochs_done = self.epoch
                self._epoch_marks.append(time.monotonic())
                self.frac = max(self.frac, min(1.0, self.epochs_done / self.total_epochs))
            else:
                b = _BATCH.search(m.group(3))
                if b and int(b.group(2)) > 0:
                    self._batches_seen = min(int(b.group(2)), self._batches_seen + 1)
                    done = (self.epoch - 1) + self._batches_seen / int(b.group(2))
                    self.frac = max(self.frac, min(1.0, done / self.total_epochs))
            return True
        return False

    def eta_s(self) -> int | None:
        if self.stage != "train" or len(self._epoch_marks) < 3:
            return None
        marks = list(self._epoch_marks)
        # Median of recent epoch durations, ignoring the slow outliers (checkpoint saves, a busy system) so the
        # estimate moves smoothly instead of doubling after one slow epoch.
        gaps = sorted(b - a for a, b in zip(marks, marks[1:]))
        median = gaps[len(gaps) // 2]
        typical = [g for g in gaps if g <= 1.5 * median] or gaps
        per_epoch = sum(typical) / len(typical)
        remaining = max(0, self.total_epochs - self.epochs_done)
        return int(round(per_epoch * remaining + 20))  # +20 s for the final export and index

    def snapshot(self) -> dict:
        lo, hi = WEIGHTS[self.stage]
        pct = lo + (hi - lo) * max(0.0, min(1.0, self.frac))
        return {
            "stage": self.stage,
            "epoch": self.epoch if self.stage in ("train", "index") else 0,
            "total_epochs": self.total_epochs,
            "pct": round(pct, 1),
            "eta_s": self.eta_s(),
        }
