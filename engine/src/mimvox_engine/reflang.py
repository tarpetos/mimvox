"""The real language of a voice's reference clips, and which clip speaks a given output language.

OmniVoice copies the accent of the reference clip, so English spoken from a clip of Ukrainian speech has a Ukrainian
accent, whatever ``voice.json`` ``language`` says. The language is read, never guessed from the voice:

- ``reference_<lang>.wav`` is in ``<lang>`` by its name (``voices.add_reference``, the media pool for another
  language).
- ``reference.wav``: the OmniVoice worker transcribes it with Whisper turbo once per clip state and records Whisper's
  detected language in the sidecar ``reference.omnivoice.json`` (``"language"``). Sidecars written before that field
  existed have only the transcript: a cheap script heuristic on it is the fallback (``script_language``). Without a
  current sidecar (a new clip the worker has not seen yet) the language is unknown (``None``) until the next warm-up or
  generation builds it; Whisper never runs in the engine process.

For output language L the clip is ``reference_L.wav`` when it exists, else ``reference.wav``; when the latter is known
to be in another language than L the result is marked *foreign* (``reference_foreign``, ``reference_note``): the
output carries that clip's accent. See contract/engine-protocol.md, "Reference language".
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from .backends import names

SIDECAR_SUFFIX = ".omnivoice.json"

# -------------------------------------------------------------------------------------- script heuristic (fallback)

_UK_ONLY = set("іїєґ")
_RU_ONLY = set("ыэъё")
_BE_ONLY = set("ў")
_SR_ONLY = set("ђјљњћџ")
_KK_ONLY = set("әғқңөұүһ")

# A few very frequent function words per Latin-script language (lower case, whole words). Enough to tell the common
# languages apart in a 10-20 s transcript; a tie or no hit is "unknown".
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the and is are was were to of a in that it you i this with for not have be on what".split()),
    "de": frozenset("der die das und ist nicht ich du ein eine zu den mit sie es wir auf für dass".split()),
    "fr": frozenset("le la les et est un une des je tu il elle que pas pour dans ce qui avec nous vous".split()),
    "es": frozenset("el la los las y es un una que no por con para lo como pero yo se del está".split()),
    "it": frozenset("il lo la gli le e è un una che non per con di sono io ma come questo della".split()),
    "pt": frozenset("o a os as e é um uma que não por com para do da eu se mas como está você".split()),
    "pl": frozenset("i w na nie się to jest że z do jak ale co tak ja ty go mi już".split()),
    "cs": frozenset("a je to se na v že s jsem ale jak tak co ne by do mi už".split()),
    "nl": frozenset("de het een en is van ik je dat niet op te zijn met voor maar wat".split()),
}
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def script_language(text: str) -> str | None:
    """A cheap guess of the language of a transcript from its script and letters (no model): Cyrillic is Ukrainian
    when it has і ї є ґ, Russian when it has ы э ъ ё (the larger count wins), Belarusian with ў, Serbian with ђ ј љ њ ћ
    џ, Kazakh with its own letters; Greek, Hebrew, Arabic, Devanagari, Hangul, kana, Han, Thai and Georgian by their
    script; Latin text by a few frequent words of English and eight other languages. ``None`` when it cannot tell
    (Cyrillic without any of those letters, Latin without a clear winner, no letters)."""
    low = (text or "").lower()
    counts = {"cyr": 0, "lat": 0, "grk": 0, "heb": 0, "ara": 0, "dev": 0, "han": 0, "kana": 0, "hang": 0,
              "thai": 0, "geo": 0}
    for ch in low:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            counts["cyr"] += 1
        elif ch.isascii() and ch.isalpha() or 0x00C0 <= o <= 0x024F:
            counts["lat"] += 1
        elif 0x0370 <= o <= 0x03FF:
            counts["grk"] += 1
        elif 0x0590 <= o <= 0x05FF:
            counts["heb"] += 1
        elif 0x0600 <= o <= 0x06FF:
            counts["ara"] += 1
        elif 0x0900 <= o <= 0x097F:
            counts["dev"] += 1
        elif 0x3040 <= o <= 0x30FF:
            counts["kana"] += 1
        elif 0x4E00 <= o <= 0x9FFF:
            counts["han"] += 1
        elif 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF:
            counts["hang"] += 1
        elif 0x0E00 <= o <= 0x0E7F:
            counts["thai"] += 1
        elif 0x10A0 <= o <= 0x10FF:
            counts["geo"] += 1
    script, n = max(counts.items(), key=lambda kv: kv[1])
    if n == 0:
        return None
    if script == "cyr":
        marks = {code: sum(1 for ch in low if ch in letters)
                 for code, letters in (("uk", _UK_ONLY), ("ru", _RU_ONLY), ("be", _BE_ONLY), ("sr", _SR_ONLY),
                                       ("kk", _KK_ONLY))}
        code, hits = max(marks.items(), key=lambda kv: kv[1])
        if hits == 0 or sum(1 for v in marks.values() if v == hits) > 1:
            return None
        return code
    if script == "lat":
        words = _WORD.findall(low)
        scores = {code: sum(1 for w in words if w in stop) for code, stop in _STOPWORDS.items()}
        best = max(scores.values())
        winners = [c for c, v in scores.items() if v == best]
        return winners[0] if best >= 2 and len(winners) == 1 else None
    if script == "han" and counts["kana"]:
        return "ja"
    return {"grk": "el", "heb": "he", "ara": "ar", "dev": "hi", "han": "zh", "kana": "ja", "hang": "ko",
            "thai": "th", "geo": "ka"}.get(script)


# ------------------------------------------------------------------------------------------------- the sidecar

_ALIAS = {"iw": "he", "in": "id", "jv": "jw", "nb": "no", "cmn": "zh"}  # as redub/langs.py


def norm(code: Any) -> str | None:
    """A language code as the engine writes it (``uk``, ``en``; Whisper's aliases mapped), or None."""
    if not isinstance(code, str) or not code.strip():
        return None
    c = code.strip().lower().replace("_", "-").split("-")[0]
    return _ALIAS.get(c, c) or None


def sidecar_path(wav: str) -> str:
    return os.path.splitext(wav)[0] + SIDECAR_SUFFIX


def read_sidecar(wav: str) -> dict[str, Any] | None:
    """The OmniVoice sidecar of ``wav`` when it describes the clip as it is now (same mtime and size), else None.
    IO thread (a stat and a small read)."""
    try:
        st = os.stat(wav)
        with open(sidecar_path(wav), encoding="utf-8") as fh:
            side = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(side, dict) or side.get("sig") != [st.st_mtime_ns, st.st_size]:
        return None
    return side


def detected(wav: str) -> tuple[str | None, str | None]:
    """(language, how) of a clip from its current sidecar: ``how`` is ``"whisper"`` (recorded by the worker),
    ``"transcript"`` (the script heuristic on an older sidecar's transcript) or None (no current sidecar, or the
    transcript says nothing)."""
    side = read_sidecar(wav)
    if side is None:
        return None, None
    code = norm(side.get("language"))
    if code:
        return code, "whisper"
    code = script_language(str(side.get("text") or ""))
    return (code, "transcript") if code else (None, None)


def per_language(vdir: str) -> list[str]:
    from .dataset.reference import language_references  # noqa: PLC0415 - lazy: the dataset package imports voices

    return language_references(vdir)


def describe(vdir: str) -> dict[str, Any]:
    """``voices.list`` fields: ``reference_language`` (the detected language of ``reference.wav``, None while unknown)
    and ``reference_languages`` (``{lang: detected}`` for every ``reference_<lang>.wav``; the detected language, None
    while the worker has not transcribed it). IO thread."""
    out: dict[str, Any] = {"reference_language": detected(os.path.join(vdir, "reference.wav"))[0]}
    out["reference_languages"] = {code: detected(os.path.join(vdir, f"reference_{code}.wav"))[0]
                                  for code in per_language(vdir)}
    return out


# ------------------------------------------------------------------------------------------- output-time choice

def for_output(reference: str, lang: str) -> tuple[str, str | None]:
    """(clip, its language) that speaks output language ``lang`` for the voice whose reference is ``reference``:
    ``reference_<lang>.wav`` (language ``lang``) when it exists, else ``reference.wav`` with its detected language
    (None while unknown). The same choice as ``backends.omnivoice.reference_for``. IO thread."""
    lang = (lang or "").lower()
    alt = os.path.join(os.path.dirname(reference), f"reference_{lang}.wav")
    if lang and os.path.isfile(alt):
        return alt, lang
    return reference, detected(reference)[0]


def note(clip_lang: str, lang: str) -> str:
    """The warning for speech in ``lang`` spoken from a clip in ``clip_lang``."""
    clip, out = names.name(clip_lang), names.name(lang)
    text = (f"The reference clip is {clip} speech, so the {out} carries a {clip} accent. Add an {out} reference "
            if out[:1] in "AEIOU" else
            f"The reference clip is {clip} speech, so the {out} carries a {clip} accent. Add a {out} reference ")
    text += "clip in Voices"
    if lang == "en":
        text += ", or train the voice and use the native accent"
    return text + "."


def fields(reference: str, lang: str, native: bool = False) -> dict[str, Any]:
    """End-event / answer fields of a speech job in ``lang``: ``reference_language`` (the language of the clip that
    speaks; ``en`` for the native accent's clip, None while unknown), ``reference_foreign`` and, when foreign,
    ``reference_note``. IO thread."""
    if native:
        return {"reference_language": "en", "reference_foreign": False}
    _clip, clip_lang = for_output(reference, lang)
    foreign = bool(clip_lang) and clip_lang != (lang or "").lower()
    out: dict[str, Any] = {"reference_language": clip_lang, "reference_foreign": foreign}
    if foreign:
        assert clip_lang is not None
        out["reference_note"] = note(clip_lang, (lang or "").lower())
    return out
