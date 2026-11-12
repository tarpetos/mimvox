"""Dataset levels, human messages and the training-epoch recommendation (shared by text analysis and the store).

Level thresholds (minutes of speech) and why:

* ``< 0.5``  too_short: the speech model needs a few seconds of reference, but under 30 s the best 15 s clip is hard to choose
  and there is no variety (pitch, pace, emotion) to speak of.
* ``< 3``    zero_shot: enough for an instant zero-shot clone (the reference is 15-30 s) but too little for RVC training,
  which overfits and sounds metallic on so little data.
* ``3 - 10`` trainable: RVC training gives a clear gain. The validated setup is 6.4 min.
* ``>= 10``  great: best similarity; more data keeps helping up to about 20 min.

Low phoneme coverage (below the language's ``low_coverage``, 0.80 of the inventory for every language today: the 39
English ARPAbet phonemes, the 38 Ukrainian phonemes, ... see ``dataset.lang``) downgrades the level by one step,
because the model would never have heard those sounds.
"""
from __future__ import annotations

TOO_SHORT_MIN = 0.5
ZERO_SHOT_MIN = 3.0
GREAT_MIN = 10.0
LOW_COVERAGE = 0.80

LEVELS = ["too_short", "zero_shot", "trainable", "great"]
TRAINABLE_LEVELS = ("trainable", "great")


def level_for(minutes: float, coverage: float | None = None, low_coverage: float = LOW_COVERAGE) -> str:
    if minutes < TOO_SHORT_MIN:
        idx = 0
    elif minutes < ZERO_SHOT_MIN:
        idx = 1
    elif minutes < GREAT_MIN:
        idx = 2
    else:
        idx = 3
    if coverage is not None and coverage < low_coverage and idx >= 2:
        idx -= 1
    return LEVELS[idx]


def recommended_epochs(minutes: float) -> int:
    """RVC epochs for ``minutes`` of speech; 0 when the data is too small to train (below ``ZERO_SHOT_MIN``).

    The curve itself lives in ``training.plan.recommended_epochs`` (the same number ``train.start`` uses for
    ``epochs: null``), so ``dataset.summary`` and the actual training can never disagree.
    """
    if minutes < ZERO_SHOT_MIN:
        return 0
    from ..training.plan import recommended_epochs as plan_epochs  # noqa: PLC0415 - training imports the registry only

    return plan_epochs(minutes)


def fmt_minutes(minutes: float) -> str:
    if minutes < 1.0:
        return f"{int(round(minutes * 60))} s"
    return f"{minutes:.1f} min"


def message_for(minutes: float, level: str, extra: str = "", wpm: float = 145.0) -> str:
    t = fmt_minutes(minutes)
    if level == "too_short":
        per_min = int(wpm / 10.0 + 0.5) * 10
        text = f"Only about {t} so far. Record at least a minute (about {per_min} words) for an instant voice."
    elif level == "zero_shot":
        text = (
            f"About {t}: enough to clone your voice instantly, but too little to train a custom model. "
            f"Reach {ZERO_SHOT_MIN:.0f} minutes or more to unlock training."
        )
    elif level == "trainable":
        text = f"About {t}: good for training a custom voice model. {GREAT_MIN:.0f} minutes or more gives the best result."
    else:
        text = f"About {t}: excellent, this is plenty for the best possible similarity."
    return f"{text} {extra}".strip()
