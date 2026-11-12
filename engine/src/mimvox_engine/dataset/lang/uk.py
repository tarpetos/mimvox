"""Ukrainian: rule-based grapheme-to-phoneme conversion (the orthography is close to phonemic).

Inventory (38, IPA): 6 vowels and 32 consonants, counting the soft (palatalized) dentals as their own phonemes.

Rules:
* Letters map one to one: а a, е e, и ɪ, і i, о o, у u, б b, в ʋ, г ɦ, ґ g, д d, ж ʒ, з z, й j, к k, л l, м m, н n,
  п p, р r, с s, т t, ф f, х x, ц ts, ч tʃ, ш ʃ; щ is ʃ + tʃ.
* дж / дз are single affricates (джерело, дзвін, бджола), except across the prefixes від-, під-, над-, перед-, од-
  (піджарити, підземний), where they are д + ж / д + з.
* Softening: д т з с ц л н р дз become dʲ tʲ zʲ sʲ tsʲ lʲ nʲ rʲ dzʲ before ь and before і я ю є. Other consonants stay
  hard (their slight palatalization before і is an allophone).
* Iotation: я ю є are j + a/u/e at the start of a word, after a vowel, after the apostrophe or after ь; after a
  consonant they only soften it. ї is always j + i.
* -ться / -тся are tsʲ tsʲ a / ts ts a, -шся is sʲ sʲ a (the usual assimilations). Other assimilations are ignored.
* Stress and final voicing do not change phonemes here (Ukrainian keeps voiced consonants at the end of a word).
"""
from __future__ import annotations

from ._common import words

CODE = "uk"
NAME = "Ukrainian"
NATIVE = "Українська"

PHONEMES = [
    "a", "e", "ɪ", "i", "o", "u",
    "b", "ʋ", "ɦ", "g", "d", "dʲ", "dz", "dzʲ", "dʒ", "ʒ", "z", "zʲ", "j", "k", "l", "lʲ", "m", "n", "nʲ", "p",
    "r", "rʲ", "s", "sʲ", "t", "tʲ", "f", "x", "ts", "tsʲ", "tʃ", "ʃ",
]
EXAMPLES = {
    "a": "мама", "e": "село", "ɪ": "ми", "i": "ліс", "o": "дорога", "u": "рука",
    "b": "бук", "ʋ": "вода", "ɦ": "гора", "g": "ґанок", "d": "дім", "dʲ": "день", "dz": "дзвін", "dzʲ": "дзьоб",
    "dʒ": "джерело", "ʒ": "жук", "z": "зима", "zʲ": "князь", "j": "мій", "k": "кіт", "l": "лампа", "lʲ": "люди",
    "m": "море", "n": "ніс", "nʲ": "кінь", "p": "пан", "r": "рука", "rʲ": "рік", "s": "сад", "sʲ": "сьогодні",
    "t": "тато", "tʲ": "тінь", "f": "фарба", "x": "хата", "ts": "цукор", "tsʲ": "кінець", "tʃ": "час", "ʃ": "шапка",
}
DIGITS = "нуль один два три чотири п'ять шість сім вісім дев'ять".split()

_VOWELS = {"а": "a", "е": "e", "и": "ɪ", "і": "i", "о": "o", "у": "u"}
_IOTATED = {"я": "a", "ю": "u", "є": "e"}
_CONS = {
    "б": "b", "в": "ʋ", "г": "ɦ", "ґ": "g", "д": "d", "ж": "ʒ", "з": "z", "й": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "п": "p", "р": "r", "с": "s", "т": "t", "ф": "f", "х": "x", "ц": "ts", "ч": "tʃ", "ш": "ʃ",
}
_SOFTENABLE = {"d", "t", "z", "s", "ts", "l", "n", "r", "dz"}
_SOFTENERS = set("ьіяює")
_D_PREFIXES = ("перед", "від", "під", "над", "од")
_VOWEL_LETTERS = set("аеиіоуяюєї")


def word_phonemes(w: str) -> list[str]:
    out: list[str] = []
    n = len(w)
    i = 0
    while i < n:
        c = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        if c in ("'", "ь"):
            i += 1
            continue
        if c in _VOWELS:
            out.append(_VOWELS[c])
            i += 1
            continue
        if c == "ї":
            out += ["j", "i"]
            i += 1
            continue
        if c in _IOTATED:
            prev = w[i - 1] if i else ""
            if not prev or prev in _VOWEL_LETTERS or prev in ("'", "ь"):
                out.append("j")
            out.append(_IOTATED[c])
            i += 1
            continue
        if c == "щ":
            out += ["ʃ", "tʃ"]
            i += 1
            continue
        # -ться / -тся / -шся at the end of a word (сміється, вчишся)
        if c in "тш" and w.endswith(("ться", "тся", "шся"), i) and w[i:] in ("ться", "тся", "шся"):
            ph = {"ться": "tsʲ", "тся": "ts", "шся": "sʲ"}[w[i:]]
            out += [ph, ph, "a"]
            break
        if c not in _CONS:
            i += 1  # a letter of another script: ignored
            continue
        ph = _CONS[c]
        step = 1
        if c == "д" and nxt in ("ж", "з") and not any(w.startswith(p) and i + 1 == len(p) for p in _D_PREFIXES):
            ph = "dʒ" if nxt == "ж" else "dz"
            step = 2
        after = w[i + step] if i + step < n else ""
        if ph in _SOFTENABLE and after in _SOFTENERS:
            ph += "ʲ"
        out.append(ph)
        i += step
    return out


def phonemes_of(text: str) -> list[str]:
    out: list[str] = []
    for w in words(text, DIGITS):
        out.extend(word_phonemes(w))
    return out
