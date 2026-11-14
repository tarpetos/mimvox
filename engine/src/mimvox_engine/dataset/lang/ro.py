"""Romanian rule-based grapheme-to-phoneme conversion (the orthography is close to phonemic)."""
from __future__ import annotations

from ._common import words

CODE = "ro"
NAME = "Romanian"
NATIVE = "Română"

PHONEMES = [
    "a", "e", "i", "o", "u", "ə", "ɨ", "j", "w", "e̯", "o̯", "ʲ",
    "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "h", "ts", "tʃ", "dʒ", "m", "n", "l", "r",
]
EXAMPLES = {
    "a": "apă", "e": "mere", "i": "inimă", "o": "om", "u": "unde", "ə": "mără", "ɨ": "pâine", "j": "iarnă",
    "w": "ou", "e̯": "seară", "o̯": "soare", "ʲ": "lupi", "p": "pom", "b": "bun", "t": "tată", "d": "drum",
    "k": "casă", "g": "gară", "f": "foc", "v": "vară", "s": "sare", "z": "zi", "ʃ": "șapte", "ʒ": "joc",
    "h": "hartă", "ts": "țară", "tʃ": "cer", "dʒ": "ger", "m": "mamă", "n": "nor", "l": "lună", "r": "râu",
}
DIGITS = "zero unu doi trei patru cinci șase șapte opt nouă".split()

_CEDILLA = str.maketrans({"ş": "ș", "ţ": "ț"})
_VOWELS = set("aăâeiîou")
_SIMPLE_V = {"a": "a", "ă": "ə", "â": "ɨ", "î": "ɨ"}
_SIMPLE = {
    "b": "b", "d": "d", "f": "f", "h": "h", "j": "ʒ", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "q": "k",
    "r": "r", "s": "s", "ș": "ʃ", "t": "t", "ț": "ts", "v": "v", "w": "v", "z": "z",
}
# Old spellings that start with the glide j: the pronouns and the forms of "a fi".
_J_INITIAL = frozenset("el ea ei ele eu este ești e era eram erai erau erați".split())


def word_phonemes(word: str) -> list[str]:
    w = word.replace("'", "").translate(_CEDILLA).replace("y", "i")
    n = len(w)
    out: list[str] = ["j"] if w in _J_INITIAL else []
    if w == "ea":
        return ["j", "a"]
    last_vowel = max((k for k, ch in enumerate(w) if ch in _VOWELS), default=-1)
    first_vowel = next((k for k, ch in enumerate(w) if ch in _VOWELS), -1)
    i = 0
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else "#"
        nxt2 = w[i + 2] if i + 2 < n else "#"
        prev = w[i - 1] if i else "#"
        prev2 = w[i - 2] if i > 1 else "#"
        if c in "cg":
            soft, hard = ("tʃ", "k") if c == "c" else ("dʒ", "g")
            if nxt == "h" and nxt2 in "ei":
                out.append(hard)
                i += 1
            elif nxt in "ei":
                out.append(soft)
                # ce/ci/ge/gi before a o u: the e/i only marks the consonant.
                if (nxt == "e" and nxt2 == "a") or (nxt == "i" and nxt2 in "aou"):
                    i += 1
            else:
                out.append(hard)
        elif c == "x":
            out += ["g", "z"] if prev == "e" and i == 1 and nxt in _VOWELS else ["k", "s"]
        elif c == "e":
            out.append("e̯" if nxt == "a" else "e")
        elif c == "o":
            out.append("o̯" if nxt == "a" else "o")
        elif c == "u":
            out.append("w" if prev in _VOWELS else "u")
        elif c == "i":
            final = i == n - 1
            after_cluster = prev in "lr" and prev2 not in _VOWELS and prev2 != "#"
            if prev in _VOWELS:
                out.append("i" if out and out[-1] == "j" else "j")  # copiii: i j i
            elif final and prev != "#":
                out.append("i" if after_cluster or first_vowel == last_vowel else "ʲ")
            elif nxt in _VOWELS and nxt != "i":
                ending = i == n - 2 and nxt in "eau"  # -ie, -ia, -iu
                before_w = nxt == "u" and nxt2 in _VOWELS  # ziua: the u is the glide
                out.append("i" if prev != "#" and (after_cluster or ending or before_w) else "j")
            else:
                out.append("i")
        elif c in _SIMPLE_V:
            out.append(_SIMPLE_V[c])
        elif c in _SIMPLE:
            out.append(_SIMPLE[c])
        i += 1
    return out


def phonemes_of(text: str) -> list[str]:
    out: list[str] = []
    for w in words(text, DIGITS):
        out.extend(word_phonemes(w))
    return out
