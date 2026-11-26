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
    name: str
    native: str
    wpm: float  # English 145 wpm scaled by syllables per word: an estimate, not measured
    low_coverage: float = 0.80  # one 60-90 s preset part reaches 0.84-1.0 in every language
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
    """``"uk-UA"``, ``"UK"`` -> ``"uk"``; ``None`` for ``None``/``""``/``"auto"``; ``ValueError`` when unsupported."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('\'language\' must be a string such as "en" or "uk"')  # noqa: TRY004
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
        from .. import phonemes

        return phonemes
    return importlib.import_module(f".{code}", __name__)


def backend(code: str) -> str:
    """g2p_en / cmudict / heuristic for English, ``rules`` for the others."""
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


_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[a-zà-ɏ]")
_TOKEN = re.compile(r"[^\W\d_]+")
CYRILLIC_LANGUAGES = frozenset({"uk", "ru", "be", "bg"})
# Letters that set a language apart from the others of its script.
_MARKS = {
    "uk": "їєґ",
    "ru": "ыэё",
    "be": "ўыэё",
    "bg": "ъ",
    "pl": "ąęłśźżćń",
    "de": "äöüß",
    "fr": "èêàçœùôûëï",
    "es": "ñáíú¿¡",
    "ro": "ășțşţ",
    "it": "ìò",
    "pt": "ãõ",
    "cs": "ěřůčšž",
    "nl": "",
}
_STOPWORDS = {
    "uk": "і й та що не на в у з це як він вона вони ми ви був була було його її але так ще вже чи до від",
    "ru": "и в не на что я с он как это она они мы вы был была было его её но так ещё уже ли до от",
    "be": "і ў не на што я з ён як гэта яна яны мы вы быў была было яго але так яшчэ ўжо ці да ад",
    "bg": "и в не на че аз с той как това тази този тя те ние вие беше бе да от за се ще са съм ако",
    "en": "the and is are was of to in that it you he she we they this with for not have be on at but what",
    "de": "der die das und ist nicht ich du er sie wir ein eine mit sich auf den dem zu es auch noch wie aber",
    "fr": "le la les et est un une des que qui je tu il elle nous vous pas ce dans pour sur avec au du mais",
    "es": "el la los las y es un una que en por con para no lo se del al como pero muy yo tú está son",
    "pl": "i w nie się na że to jest z do jak ale co tak czy już mi go jego ja ty my wy był być",
    "ro": "și în să nu că este mai din care sunt dar pentru cu pe am ai ce de la o un eu tu îmi sau iar",
    "it": "il lo la gli le e è un una che di non per con del della sono ma come io tu lui lei noi questo",
    "pt": "o os as e é um uma que de não para com do da em mas como eu você ele ela nós isso muito",
    "cs": "a je se na že to v z do jak ale co tak jsem jsi není byl byla být ten ta já ty my",
    "nl": "de het een en van ik je dat niet is zijn met op te maar wat hij zij wij voor er",
}
# Letters a Cyrillic language never writes: ru and be share ы э ё, so и щ ъ against і ў tell them apart.
_FOREIGN = {"uk": "ыэёъў", "ru": "іїєґў", "be": "ищъїєґ", "bg": "ыэёіїєґў"}
_STOP = {code: set(words.split()) for code, words in _STOPWORDS.items()}
# Every language ``detect`` can tell apart; ``LANGUAGES`` (the recording languages) is a subset.
DETECTABLE = frozenset(_STOP)
# Tie-break order, so the result never depends on set iteration (the hash seed).
_PREFERENCE = ("en", "uk", "ru", "be", "bg", "de", "fr", "es", "pl", "ro", "it", "pt", "cs", "nl")


def detect(text: str, among: frozenset[str] | set[str] | None = None) -> str:
    """The language among ``among`` (default: the recording languages) that the text's script, common words and
    language-specific letters (3 points each, minus 3 for a letter it never writes) fit best; ``en`` when nothing
    fits; ties go to the earlier language in ``_PREFERENCE``."""
    pool = set(LANGUAGES) if among is None else set(among)
    t = normalize(text)
    cyr, lat = len(_CYRILLIC.findall(t)), len(_LATIN.findall(t))
    if cyr == 0 and lat == 0:
        return DEFAULT
    script = CYRILLIC_LANGUAGES if cyr > lat else DETECTABLE - CYRILLIC_LANGUAGES
    candidates = [c for c in _PREFERENCE if c in pool and c in script]
    if not candidates:
        return DEFAULT
    scores = dict.fromkeys(candidates, 0.0)
    for tok in _TOKEN.findall(t):
        for code in candidates:
            if tok in _STOP[code]:
                scores[code] += 1.0
    for code in candidates:
        scores[code] += 3.0 * sum(t.count(ch) for ch in _MARKS.get(code, ""))
        scores[code] -= 3.0 * sum(t.count(ch) for ch in _FOREIGN.get(code, ""))
    best = max(candidates, key=lambda c: scores[c])
    return best if scores[best] > 0 else (DEFAULT if DEFAULT in candidates else best)
