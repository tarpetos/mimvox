"""German grapheme-cluster heuristic, good enough to measure phoneme coverage (no G2P model or lexicon is shipped)."""
from __future__ import annotations

from ._common import words

CODE = "de"
NAME = "German"
NATIVE = "Deutsch"

PHONEMES = [
    "iː", "ɪ", "yː", "ʏ", "eː", "ɛ", "øː", "œ", "uː", "ʊ", "oː", "ɔ", "aː", "a", "ə", "ɐ", "aɪ", "aʊ", "ɔʏ",
    "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ç", "x", "h", "j", "m", "n", "ŋ", "l", "ʁ", "pf",
    "ts", "tʃ",
]
EXAMPLES = {
    "iː": "Liebe", "ɪ": "Mitte", "yː": "über", "ʏ": "Glück", "eː": "See", "ɛ": "Bett", "øː": "schön", "œ": "können",
    "uː": "gut", "ʊ": "Mutter", "oː": "Boot", "ɔ": "Sonne", "aː": "Tag", "a": "Stadt", "ə": "Rose", "ɐ": "Vater",
    "aɪ": "Zeit", "aʊ": "Haus", "ɔʏ": "heute", "p": "Post", "b": "Ball", "t": "Tisch", "d": "Dach", "k": "Kalt",
    "g": "gehen", "f": "Fisch", "v": "Wasser", "s": "Haus", "z": "Sonne", "ʃ": "Schule", "ç": "ich", "x": "Bach",
    "h": "Hund", "j": "ja", "m": "Mond", "n": "Nase", "ŋ": "lang", "l": "Licht", "ʁ": "rot", "pf": "Apfel",
    "ts": "Zeit", "tʃ": "Deutsch",
}
DIGITS = "null eins zwei drei vier fünf sechs sieben acht neun".split()

_V = set("aeiouäöüy")
_LONG = {"a": "aː", "e": "eː", "i": "iː", "o": "oː", "u": "uː", "ä": "eː", "ö": "øː", "ü": "yː", "y": "yː"}
_SHORT = {"a": "a", "e": "ɛ", "i": "ɪ", "o": "ɔ", "u": "ʊ", "ä": "ɛ", "ö": "œ", "ü": "ʏ", "y": "ʏ"}
_SHORT_WORDS = {
    "an", "am", "im", "in", "um", "mit", "das", "des", "es", "man", "ob", "ab", "bis", "hat", "was", "von", "vom",
    "zum", "ins", "ums", "wann", "dann", "denn", "wenn", "bin", "hin", "drin", "weg", "ich", "mich", "dich", "sich",
    "nicht", "doch", "noch", "ist", "und", "mal", "ganz", "jetzt", "es", "uns", "bis", "kann", "will",
}
_CONS_CLUSTERS = [("tsch", ["tʃ"]), ("sch", ["ʃ"]), ("chs", ["k", "s"]), ("ck", ["k"]), ("tz", ["ts"]),
                  ("pf", ["pf"]), ("dt", ["t"]), ("ph", ["f"]), ("th", ["t"]), ("qu", ["k", "v"]), ("ng", ["ŋ"]), ("nk", ["ŋ", "k"]),
                  ("ss", ["s"]), ("ß", ["s"])]
_SINGLE = {"b": "b", "c": "k", "d": "d", "f": "f", "g": "g", "h": "h", "j": "j", "k": "k", "l": "l", "m": "m",
           "n": "n", "p": "p", "r": "ʁ", "s": "s", "t": "t", "v": "f", "w": "v", "x": "ks", "z": "ts"}
_DEVOICE = {"b": "p", "d": "t", "g": "k"}


def _consonants_after(w: str, i: int) -> int:
    """Consonant letters from ``i`` to the next vowel; sch/ck/ß count as two."""
    j = i
    while j < len(w) and w[j] not in _V:
        j += 1
    span = w[i:j]
    if span in ("sch", "ck", "tz", "ß", "ss", "ch") and j < len(w):
        return 2 if span != "ch" else 1
    return len(span) + (1 if span.endswith(("ß", "ss", "ck", "tz")) else 0)


def word_phonemes(word: str) -> list[str]:
    w = word.replace("'", "")
    out: list[str] = []
    n = len(w)
    i = 0
    if w.startswith(("sp", "st")):
        out.append("ʃ")
        i = 1
    while i < n:
        c = w[i]
        rest = w[i:]
        nxt = w[i + 1] if i + 1 < n else "#"
        if rest == "ig" and i > 0:
            out += ["ɪ", "ç"]
            break
        if c in _V:
            if c == "e" and i > 0 and rest in ("e", "en", "el", "em", "es", "ern", "end"):
                if rest in ("ern",):
                    out += ["ɐ", "n"]
                    break
                out.append("ə")
                i += 1
                continue
            if c == "e" and rest == "er" and n > 3:
                out.append("ɐ")
                break
            if c == "e" and i == 1 and w[0] in "bg" and n > 3 and nxt not in _V and nxt not in ("h", "r"):
                out.append("ə")
                i += 1
                continue
            if rest.startswith("ie"):
                out.append("iː")
                i += 2
            elif rest.startswith(("ei", "ai", "ey", "ay")):
                out.append("aɪ")
                i += 2
            elif rest.startswith(("eu", "äu")):
                out.append("ɔʏ")
                i += 2
            elif rest.startswith("au"):
                out.append("aʊ")
                i += 2
            elif nxt == c and c in "aeo":
                out.append(_LONG[c])
                i += 2
            elif nxt == "h" and (i + 2 >= n or w[i + 2] not in _V):
                out.append(_LONG[c])
                i += 2
            else:
                k = _consonants_after(w, i + 1)
                end_single = i + 1 + k >= n and k == 1
                if k >= 2 or (end_single and w in _SHORT_WORDS) or (k == 0 and i + 1 < n):
                    out.append(_SHORT[c])
                else:
                    out.append(_LONG[c])
                i += 1
            # vocalic r after a long vowel at the end of a syllable (mir, Uhr, vor, Tier)
            if i < n and w[i] == "r" and out[-1].endswith("ː") and (i + 1 >= n or w[i + 1] not in _V):
                out.append("ɐ")
                i += 1
            continue
        if c == "c" and nxt == "h" and not rest.startswith("chs"):
            prev = out[-1] if out else ""
            out.append("x" if prev in ("a", "aː", "o", "oː", "ɔ", "u", "uː", "ʊ", "aʊ") else "ç")
            i += 2
            continue
        cl = next((cl for cl in _CONS_CLUSTERS if rest.startswith(cl[0])), None)
        if cl:
            out += cl[1]
            i += len(cl[0])
            continue
        if c == nxt:
            i += 1
            continue
        if c == "h" and i > 0:
            i += 1  # h after a consonant or between vowels is silent (gehen)
            continue
        if c not in _SINGLE:
            i += 1
            continue
        ph = _SINGLE[c]
        prev_l = w[i - 1] if i else "#"
        if c == "s" and nxt in _V and (i == 0 or prev_l in _V or prev_l in "lnmr"):
            ph = "z"
        elif c == "v" and prev_l in _V and nxt in _V:
            ph = "v"
        elif c == "c" and nxt in "eiäy":
            ph = "ts"
        elif c in _DEVOICE and i == n - 1:
            ph = _DEVOICE[c]
        if ph == "ks":
            out += ["k", "s"]
        else:
            out.append(ph)
        i += 1
    return out


def phonemes_of(text: str) -> list[str]:
    out: list[str] = []
    for w in words(text, DIGITS):
        out.extend(word_phonemes(w))
    return out
