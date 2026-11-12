"""Polish: rule-based grapheme-to-phoneme conversion (the orthography is close to phonemic).

Inventory (37, IPA): 8 vowels (a ɛ i ɨ ɔ u and the nasal ɛ̃ ɔ̃) and 29 consonants.

Rules:
* Digraphs: ch/h x, cz tʂ, sz ʂ, rz ʐ (ʂ after p t k ch f: przy, trzy, krzak), dż dʐ, dź dʑ, dz dz; single letters
  ć tɕ, ś ɕ, ź ʑ, ż ʐ, ń ɲ, ł w, w v, c ts, y ɨ, ó u, e ɛ, o ɔ.
* Soft spelling with i: ci si zi ni dzi are tɕ ɕ ʑ ɲ dʑ; before another vowel the i only marks softness (ciepło,
  siostra), otherwise it is also the vowel i (cicho, nic). After any other consonant, i before a vowel is j (pies,
  biały); elsewhere i is the vowel.
* Nasal vowels: ą ę are the nasal vowels ɔ̃ ɛ̃ before fricatives and (ą) at the end of a word; before a stop or an
  affricate they are ɔ/ɛ + a nasal consonant (m before p b, ɲ before ć dź ci dzi, n otherwise: ręka, kąt, będzie);
  before l ł and ę at the end of a word they lose nasality (wziął, idę).
* Final devoicing and voicing assimilation are ignored (they do not remove phonemes from the inventory).
"""
from __future__ import annotations

from ._common import words

CODE = "pl"
NAME = "Polish"
NATIVE = "Polski"

PHONEMES = [
    "a", "ɛ", "i", "ɨ", "ɔ", "u", "ɛ̃", "ɔ̃",
    "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ɕ", "ʑ", "ʂ", "ʐ", "x", "ts", "dz", "tɕ", "dʑ", "tʂ", "dʐ",
    "m", "n", "ɲ", "l", "w", "r", "j",
]
EXAMPLES = {
    "a": "mama", "ɛ": "teraz", "i": "lis", "ɨ": "ryba", "ɔ": "kot", "u": "góra", "ɛ̃": "język", "ɔ̃": "są",
    "p": "pan", "b": "brat", "t": "tata", "d": "dom", "k": "kot", "g": "góra", "f": "fala", "v": "woda", "s": "sok",
    "z": "zima", "ɕ": "siostra", "ʑ": "źle", "ʂ": "szafa", "ʐ": "żaba", "x": "chleb", "ts": "córka", "dz": "dzwon",
    "tɕ": "ciepło", "dʑ": "dzień", "tʂ": "czas", "dʐ": "dżem", "m": "matka", "n": "noc", "ɲ": "koń", "l": "las",
    "w": "łódź", "r": "rok", "j": "jajko",
}
DIGITS = "zero jeden dwa trzy cztery pięć sześć siedem osiem dziewięć".split()

_VOWEL_LETTERS = set("aąeęioóuy")
_SIMPLE_V = {"a": "a", "e": "ɛ", "o": "ɔ", "ó": "u", "u": "u", "y": "ɨ"}
_SOFT_I = {"dzi": "dʑ", "ci": "tɕ", "si": "ɕ", "zi": "ʑ", "ni": "ɲ"}
_MULTI = [("dż", "dʐ"), ("dź", "dʑ"), ("dz", "dz"), ("ch", "x"), ("cz", "tʂ"), ("sz", "ʂ"), ("rz", "ʐ")]
_SINGLE = {
    "b": "b", "c": "ts", "ć": "tɕ", "d": "d", "f": "f", "g": "g", "h": "x", "j": "j", "k": "k", "l": "l", "ł": "w",
    "m": "m", "n": "n", "ń": "ɲ", "p": "p", "r": "r", "s": "s", "ś": "ɕ", "t": "t", "w": "v", "z": "z", "ź": "ʑ",
    "ż": "ʐ", "v": "v", "q": "k",
}
_RZ_VOICELESS = {"p", "t", "k", "x", "f"}
_STOPS = {"p", "b", "t", "d", "k", "g", "ts", "dz", "tɕ", "dʑ", "tʂ", "dʐ"}


def _units(w: str) -> list[tuple[str, str]]:
    """(grapheme, phoneme) units; nasal letters are kept as ("ą"/"ę", "") and resolved afterwards."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(w)
    while i < n:
        c = w[i]
        if c == "'":
            i += 1
            continue
        soft = next((g for g in _SOFT_I if w.startswith(g, i)), None)
        if soft:
            out.append((soft, _SOFT_I[soft]))
            i += len(soft)
            if i < n and w[i] in _VOWEL_LETTERS and w[i] != "i":
                continue  # the i only marks softness
            out.append(("i", "i"))
            continue
        multi = next((m for m in _MULTI if w.startswith(m[0], i)), None)
        if multi:
            ph = multi[1]
            if multi[0] == "rz" and out and out[-1][1] in _RZ_VOICELESS:
                ph = "ʂ"
            out.append((multi[0], ph))
            i += 2
            continue
        if c in "ąę":
            out.append((c, ""))
        elif c in _SIMPLE_V:
            out.append((c, _SIMPLE_V[c]))
        elif c == "i":
            prev_cons = bool(out) and out[-1][0] not in _VOWEL_LETTERS and out[-1][0] not in "ąę"
            nxt_vowel = i + 1 < n and w[i + 1] in _VOWEL_LETTERS
            out.append(("i", "j" if prev_cons and nxt_vowel else "i"))
        elif c == "x":
            out += [("x", "k"), ("x", "s")]
        elif c in _SINGLE:
            out.append((c, _SINGLE[c]))
        i += 1
    return out


def word_phonemes(w: str) -> list[str]:
    units = _units(w)
    out: list[str] = []
    for k, (g, ph) in enumerate(units):
        if g not in ("ą", "ę"):
            out.append(ph)
            continue
        oral, nasal = ("ɔ", "ɔ̃") if g == "ą" else ("ɛ", "ɛ̃")
        nxt = units[k + 1] if k + 1 < len(units) else None
        if nxt is None:
            out.append(nasal if g == "ą" else oral)
        elif nxt[1] in _STOPS:
            out += [oral, "m" if nxt[1] in ("p", "b") else "ɲ" if nxt[1] in ("tɕ", "dʑ") else "n"]
        elif nxt[1] in ("l", "w"):
            out.append(oral)
        else:
            out.append(nasal)
    return out


def phonemes_of(text: str) -> list[str]:
    out: list[str] = []
    for w in words(text, DIGITS):
        out.extend(word_phonemes(w))
    return out
