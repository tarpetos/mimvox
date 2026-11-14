"""Languages of the reading texts: codes, reading pace, phoneme inventories, coverage thresholds and detection.

Codes are ISO 639-1 (``en``, ``uk``, ``de``, ``fr``, ``es``, ``pl``, ``ro``). English keeps the ARPAbet analysis of
``dataset.phonemes`` (g2p_en / CMUdict); every other language has a pure-Python rule-based G2P in this package
(``uk``, ``pl``, ``es``, ``ro``: letter/digraph rules, the spelling is close to phonemic; ``de``, ``fr``: documented
grapheme-cluster heuristics). Each module exposes ``PHONEMES`` (the inventory, canonical order), ``EXAMPLES``
(phoneme -> a native example word) and ``phonemes_of(text)``.

Reading pace (``wpm``, words per minute read aloud at the calm pace used for recording) is the English 145 wpm
scaled by how long words are in each language (syllables per word in running text) and by the typical syllable rate
of read speech (faster in Spanish and French, a little slower in German): in the presets a Ukrainian or Polish word has
about 2.0 syllables, a Romanian one 1.8, a German one 1.6, an English one 1.4, which puts every language at about 3.3-4.2 syllables per
second. These are estimates for planning (``est_seconds``), not measurements.

Coverage thresholds: ``low_coverage`` is the share of the inventory below which ``trainable``/``great`` drop by one
level (0.80 everywhere: a single 60-90 s preset part already reaches 0.84-1.0 in every language, so only a monotonous
or very short text falls below it); ``hint_coverage`` is the share below which ``text.analyze`` names the missing
sounds. Every standard and pro preset covers its whole inventory.
"""
from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ._common import normalize

DEFAULT = "en"


@dataclass(frozen=True)
class Language:
    code: str
    name: str        # English name (UI language)
    native: str      # name in the language itself
    wpm: float       # reading pace, words per minute
    low_coverage: float = 0.80
    hint_coverage: float = 0.95


LANGUAGES: dict[str, Language] = {
    "en": Language("en", "English", "English", 145.0),
    "uk": Language("uk", "Ukrainian", "Українська", 110.0),
    "de": Language("de", "German", "Deutsch", 120.0),
    "fr": Language("fr", "French", "Français", 155.0),
    "es": Language("es", "Spanish", "Español", 140.0),
    "pl": Language("pl", "Polish", "Polski", 110.0),
    "ro": Language("ro", "Romanian", "Română", 135.0),
}


def normalize_code(value: Any) -> str | None:
    """``"uk"``, ``"UK"``, ``"uk-UA"``, ``"uk_UA"`` -> ``"uk"``; ``None`` for ``None``/``""``/``"auto"``.

    Raises ``ValueError`` for a non-string or an unsupported language.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("'language' must be a string such as \"en\" or \"uk\"")
    code = value.strip().lower().replace("_", "-").split("-", 1)[0]
    if code in ("", "auto"):
        return None
    if code not in LANGUAGES:
        raise ValueError(f"Unsupported language {value!r}; supported: {', '.join(LANGUAGES)}")
    return code


def get(code: str | None) -> Language:
    return LANGUAGES.get(code or DEFAULT, LANGUAGES[DEFAULT])


def describe(code: str) -> dict[str, str]:
    lang = get(code)
    return {"code": lang.code, "name": lang.name, "native": lang.native}


def module(code: str) -> ModuleType:
    if code == "en":
        from .. import phonemes  # noqa: PLC0415

        return phonemes
    return importlib.import_module(f".{code}", __name__)


def backend(code: str) -> str:
    """Which analysis is active: g2p_en / cmudict / heuristic for English, ``rules`` for the others."""
    return module("en").backend() if code == "en" else "rules"


def inventory(code: str) -> list[str]:
    return list(module(code).PHONEMES)


def example(code: str, phoneme: str) -> str:
    return str(module(code).EXAMPLES.get(phoneme, ""))


def coverage(text: str, code: str) -> tuple[float, list[str], dict[str, int]]:
    """(coverage 0..1, missing phonemes in canonical order, counts per phoneme) of ``text`` in language ``code``."""
    mod = module(code)
    if code == "en":
        return mod.coverage(text)
    inv = mod.PHONEMES
    allowed = set(inv)
    counts: dict[str, int] = {}
    for p in mod.phonemes_of(text):
        if p in allowed:
            counts[p] = counts.get(p, 0) + 1
    missing = [p for p in inv if p not in counts]
    return (len(inv) - len(missing)) / len(inv), missing, counts


# ------------------------------------------------------------------------------------------------- detection

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[a-zà-ɏ]")
_TOKEN = re.compile(r"[^\W\d_]+")
# Letters that only one of the supported Latin-script languages uses (é and ó are shared, so they do not count).
# â and î are French and Romanian alike, so they count for neither.
_MARKS = {"pl": "ąęłśźżćń", "de": "äöüß", "fr": "èêàçœùôûëï", "es": "ñáíú¿¡", "ro": "ășțşţ"}
_STOPWORDS = {
    "en": "the and is are was of to in that it you he she we they this with for not have be on at but what",
    "de": "der die das und ist nicht ich du er sie wir ein eine mit sich auf den dem zu es auch noch wie aber",
    "fr": "le la les et est un une des que qui je tu il elle nous vous pas ce dans pour sur avec au du mais",
    "es": "el la los las y es un una que en por con para no lo se del al como pero muy yo tú está son",
    "pl": "i w nie się na że to jest z do jak ale co tak czy już mi go jego ja ty my wy był być",
    "ro": "și în să nu că este mai din care sunt dar pentru cu pe am ai ce de la o un eu tu îmi sau iar",
}
_STOP = {code: set(words.split()) for code, words in _STOPWORDS.items()}


def detect(text: str) -> str:
    """Cheap guess of the language of ``text``: Cyrillic script -> ``uk`` (the only Cyrillic language supported);
    Latin script -> the language with the most stopwords plus language-specific letters (3 points each); ``en`` when
    nothing points anywhere (also for empty text)."""
    t = normalize(text)
    cyr, lat = len(_CYRILLIC.findall(t)), len(_LATIN.findall(t))
    if cyr == 0 and lat == 0:
        return DEFAULT
    if cyr > lat:
        return "uk"
    scores = {code: 0.0 for code in _STOP}
    for tok in _TOKEN.findall(t):
        for code, stop in _STOP.items():
            if tok in stop:
                scores[code] += 1.0
    for code, marks in _MARKS.items():
        scores[code] += 3.0 * sum(t.count(ch) for ch in marks)
    best = max(scores, key=lambda c: (scores[c], c == DEFAULT))
    return best if scores[best] > 0 else DEFAULT
