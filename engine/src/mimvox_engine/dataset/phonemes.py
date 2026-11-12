"""Text -> ARPAbet phonemes (stress stripped) with graceful degradation.

Order of preference: g2p_en (present in the engine venv; its CMUdict lookup plus the neural fallback for unknown
words, run without the POS tagger so no NLTK tagger download is needed) -> NLTK's CMUdict if only that is available ->
a small letter/digraph heuristic. ``backend()`` tells which one is active.
"""
from __future__ import annotations

import contextlib
import re
import sys
import unicodedata
from functools import lru_cache
from typing import Any

PHONEMES = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
EXAMPLES = {
    "AA": "odd", "AE": "at", "AH": "hut", "AO": "ought", "AW": "cow", "AY": "hide", "B": "be", "CH": "cheese",
    "D": "dee", "DH": "thee", "EH": "Ed", "ER": "hurt", "EY": "ate", "F": "fee", "G": "green", "HH": "he",
    "IH": "it", "IY": "eat", "JH": "gee", "K": "key", "L": "lee", "M": "me", "N": "knee", "NG": "ping", "OW": "oat",
    "OY": "toy", "P": "pee", "R": "read", "S": "sea", "SH": "she", "T": "tea", "TH": "theta", "UH": "hood",
    "UW": "two", "V": "vee", "W": "we", "Y": "yield", "Z": "zee", "ZH": "seizure",
}

_state: dict[str, Any] = {"backend": None, "g2p": None, "cmu": None}
_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)*")


def _load() -> None:
    if _state["backend"] is not None:
        return
    try:
        # g2p_en calls nltk.download() at import when data is missing, and nltk prints to stdout: keep stdout clean.
        with contextlib.redirect_stdout(sys.stderr):
            from g2p_en import G2p
            from g2p_en.expand import normalize_numbers

            g = G2p()
        _state.update(backend="g2p_en", g2p=g, normalize_numbers=normalize_numbers)
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from nltk.corpus import cmudict

            d = cmudict.dict()
        _state.update(backend="cmudict", cmu=d)
        return
    except Exception:  # noqa: BLE001
        pass
    _state["backend"] = "heuristic"


def backend() -> str:
    _load()
    return str(_state["backend"])


def _strip_stress(p: str) -> str:
    return p.rstrip("012")


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    return text.replace("’", "'")


@lru_cache(maxsize=20000)
def _g2p_word(word: str) -> tuple[str, ...]:
    g = _state["g2p"]
    pron = g.cmu[word][0] if word in g.cmu else g.predict(word)
    return tuple(_strip_stress(p) for p in pron if p[:1].isalpha() and not p.startswith("<"))


@lru_cache(maxsize=20000)
def _cmu_word(word: str) -> tuple[str, ...]:
    prons = _state["cmu"].get(word)
    if prons:
        return tuple(_strip_stress(p) for p in prons[0])
    return _heuristic_word(word)


_DIGRAPHS = [
    ("tch", ["CH"]), ("sch", ["S", "K"]), ("dge", ["JH"]), ("igh", ["AY"]), ("ough", ["AH", "F"]),
    ("sion", ["ZH", "AH", "N"]), ("tion", ["SH", "AH", "N"]), ("sure", ["ZH", "ER"]), ("ture", ["CH", "ER"]),
    ("ous", ["AH", "S"]), ("ing", ["IH", "NG"]), ("ang", ["AE", "NG"]), ("ong", ["AO", "NG"]), ("ung", ["AH", "NG"]),
    ("sh", ["SH"]), ("ch", ["CH"]), ("th", ["TH"]), ("ph", ["F"]), ("wh", ["W"]), ("ck", ["K"]), ("qu", ["K", "W"]),
    ("ng", ["NG"]), ("nk", ["NG", "K"]), ("kn", ["N"]), ("wr", ["R"]), ("gh", []), ("ee", ["IY"]), ("ea", ["IY"]),
    ("oo", ["UW"]), ("ai", ["EY"]), ("ay", ["EY"]), ("oi", ["OY"]), ("oy", ["OY"]), ("ou", ["AW"]), ("ow", ["OW"]),
    ("au", ["AO"]), ("aw", ["AO"]), ("oa", ["OW"]), ("ar", ["AA", "R"]), ("er", ["ER"]), ("ir", ["ER"]),
    ("ur", ["ER"]), ("or", ["AO", "R"]), ("ie", ["IY"]), ("ew", ["UW"]), ("ey", ["EY"]),
]
_LETTERS = {
    "a": ["AE"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH"], "f": ["F"], "g": ["G"], "h": ["HH"], "i": ["IH"],
    "j": ["JH"], "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"], "o": ["AA"], "p": ["P"], "q": ["K"], "r": ["R"],
    "s": ["S"], "t": ["T"], "u": ["AH"], "v": ["V"], "w": ["W"], "x": ["K", "S"], "y": ["Y"], "z": ["Z"],
}
_FUNCTION_DH = {"the", "this", "that", "these", "those", "they", "them", "then", "there", "than", "though"}
_LONG = {"a": ["EY"], "e": ["IY"], "i": ["AY"], "o": ["OW"], "u": ["UW"]}


def _heuristic_word(word: str) -> tuple[str, ...]:
    if word in _FUNCTION_DH:
        return ("DH",) + _heuristic_word(word[2:])
    out: list[str] = []
    i = 0
    w = word.replace("'", "")
    magic_e = len(w) > 3 and w.endswith("e") and w[-2] not in "aeiouy" and w[-3] in "aeiou"
    while i < len(w):
        for pat, ph in _DIGRAPHS:
            if w.startswith(pat, i):
                out.extend(ph)
                i += len(pat)
                break
        else:
            ch = w[i]
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if ch == "c" and nxt in "eiy":
                out.append("S")
            elif ch == "g" and nxt in "eiy" and i < len(w) - 1:
                out.append("JH")
            elif ch == "e" and i == len(w) - 1 and i > 1:
                pass  # silent final e
            elif ch == "y" and i > 0:
                out.append("IY" if i == len(w) - 1 else "IH")
            elif ch in _LONG and magic_e and i == len(w) - 3:
                out.extend(_LONG[ch])
            else:
                out.extend(_LETTERS.get(ch, []))
            i += 1
    return tuple(out)


_DIGIT_WORDS = "zero one two three four five six seven eight nine".split()


def _expand_digits(text: str) -> str:
    return re.sub(r"\d", lambda m: f" {_DIGIT_WORDS[int(m.group())]} ", text)


def phonemes_of(text: str) -> list[str]:
    """ARPAbet phoneme sequence (no stress, no word boundaries) for ``text``."""
    _load()
    b = _state["backend"]
    if b == "g2p_en":
        try:
            text = _state["normalize_numbers"](text)
        except Exception:  # noqa: BLE001
            text = _expand_digits(text)
    else:
        text = _expand_digits(text)
    out: list[str] = []
    for word in _WORD_RE.findall(_clean(text)):
        if b == "g2p_en":
            try:
                out.extend(_g2p_word(word))
                continue
            except Exception:  # noqa: BLE001
                pass
        out.extend(_cmu_word(word) if b == "cmudict" else _heuristic_word(word))
    return [p for p in out if p in EXAMPLES]


def coverage(text: str) -> tuple[float, list[str], dict[str, int]]:
    """(coverage 0..1, missing phonemes in canonical order, counts per phoneme)."""
    counts: dict[str, int] = {}
    for p in phonemes_of(text):
        counts[p] = counts.get(p, 0) + 1
    missing = [p for p in PHONEMES if p not in counts]
    return (len(PHONEMES) - len(missing)) / len(PHONEMES), missing, counts
