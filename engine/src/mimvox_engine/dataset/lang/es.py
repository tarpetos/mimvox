"""Spanish: rule-based grapheme-to-phoneme conversion (the orthography is close to phonemic)."""
from __future__ import annotations

from ._common import words

CODE = "es"
NAME = "Spanish"
NATIVE = "Español"

PHONEMES = [
    "a", "e", "i", "o", "u", "j", "w",
    "p", "b", "t", "d", "k", "g", "f", "θ", "s", "x", "tʃ", "ʝ", "m", "n", "ɲ", "l", "ɾ", "r",
]
EXAMPLES = {
    "a": "casa", "e": "mesa", "i": "sí", "o": "todo", "u": "luna", "j": "tierra", "w": "bueno", "p": "pan",
    "b": "vino", "t": "taza", "d": "dedo", "k": "casa", "g": "gato", "f": "fuego", "θ": "cielo", "s": "sol",
    "x": "jamón", "tʃ": "chico", "ʝ": "llave", "m": "madre", "n": "noche", "ɲ": "niño", "l": "luz", "ɾ": "pero",
    "r": "perro",
}
DIGITS = "cero uno dos tres cuatro cinco seis siete ocho nueve".split()

_ACCENT = str.maketrans("áéóàèò", "aeoaeo")
_VOWELS = set("aeiouáéíóúü")
_X_AS_JOTA = ("méxico", "mexicano", "mexicana", "texas", "oaxaca", "xavier")
_SIMPLE = {"b": "b", "d": "d", "f": "f", "k": "k", "l": "l", "m": "m", "n": "n", "ñ": "ɲ", "p": "p", "s": "s",
           "t": "t", "v": "b", "w": "w", "z": "θ", "j": "x"}


def word_phonemes(word: str) -> list[str]:
    w = word.replace("'", "")
    jota_x = w.startswith(_X_AS_JOTA)
    out: list[str] = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else "#"
        nxt2 = w[i + 2] if i + 2 < n else "#"
        prev = w[i - 1] if i else "#"
        if c in "iu":
            near_vowel = (nxt in _VOWELS and nxt not in "íú") or (prev in _VOWELS and prev not in "iuíúü")
            if near_vowel:
                out.append("j" if c == "i" else "w")
            else:
                out.append(c)
        elif c in "íú":
            out.append("i" if c == "í" else "u")
        elif c == "ü":
            out.append("w")
        elif c in "aeoáéó":
            out.append(c.translate(_ACCENT))
        elif c == "c":
            if nxt == "h":
                out.append("tʃ")
                i += 1
            else:
                out.append("θ" if nxt in "eiéí" else "k")
        elif c == "l" and nxt == "l":
            out.append("ʝ")
            i += 1
        elif c == "r":
            if nxt == "r":
                out.append("r")
                i += 1
            else:
                out.append("r" if i == 0 or prev in "lns" else "ɾ")
        elif c == "q":
            out.append("k")
            if nxt == "u":
                i += 1
        elif c == "g":
            if nxt == "u" and nxt2 in "eiéí":
                out.append("g")
                i += 1
            else:
                out.append("x" if nxt in "eiéí" else "g")
        elif c == "h":
            pass
        elif c == "y":
            if nxt in _VOWELS:
                out.append("ʝ")
            else:
                out.append("j" if prev in _VOWELS else "i")
        elif c == "x":
            out += ["x"] if jota_x else ["k", "s"]
        elif c in _SIMPLE:
            out.append(_SIMPLE[c])
        i += 1
    return out


def phonemes_of(text: str) -> list[str]:
    out: list[str] = []
    for w in words(text, DIGITS):
        out.extend(word_phonemes(w))
    return out
