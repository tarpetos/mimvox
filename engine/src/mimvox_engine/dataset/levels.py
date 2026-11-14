from __future__ import annotations

TOO_SHORT_MIN = 0.5
ZERO_SHOT_MIN = 3.0   # below this RVC training overfits and sounds metallic
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
