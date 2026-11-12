"""French: grapheme-cluster heuristic (no G2P model or lexicon is shipped; good enough to measure phoneme coverage).

Inventory (34, IPA): 14 vowels (i e ɛ a ɔ o u y ø œ ə and the nasal ɛ̃ ɑ̃ ɔ̃; ɑ is merged with a and œ̃ with ɛ̃ as in
most of today's French), the semivowels j w ɥ and 17 consonants (ŋ of loanwords is left out).

Rules, in order of priority:
* Elision: l' d' j' qu' n' s' c' m' t' are read as their consonant before the next word.
* Common short words come from a small table (le, les, est, et, un, des, ...).
* Silent endings: final e/es (in words of three letters or more), final s x z t d p g (unless the word ended in e),
  final -er/-ez/-et are e/e/ɛ (except mer, fer, hier, cher, fier, hiver: ɛʁ), -ent is ɑ̃ (verb endings cannot be told
  apart without a lexicon).
* Vowel clusters: eau/au o, ail(l)/eil(l)/euil(l)/ouil(l) aj/ɛj/œj/uj, ay/oy/uy + vowel ɛj/waj/ɥij, ai/ei ɛ, oi wa, ou u (w before a vowel), eu/œu ø (œ before a pronounced r l f v n),
  an/am/en/em ɑ̃, on/om ɔ̃, in/im/ain/ein/un/um/yn/ym ɛ̃, ien jɛ̃, oin wɛ̃ (nasal only when not followed by a vowel
  or by another n/m), ill j after a vowel and ij after a consonant (ville, mille, tranquille: il), final -il after a
  vowel j, y between vowels j, ui ɥi, u y (ɥ before a vowel), i j before a vowel, é e, è ê ë ɛ, e ɛ before two
  consonants or a final pronounced consonant and ə otherwise, o o at the end of a word and before s+e, ɔ otherwise.
* Consonants: gn ɲ, ch ʃ, ph f, th t, qu k, gu (before e i) g, g ʒ before e i y, c s before e i y, ç s, s z between
  vowels, ss s, x ks (gz in ex- before a vowel), h silent, r ʁ, j ʒ, -tion sjɔ̃, doubled consonants once.
Liaisons and the many exceptions of French spelling are ignored.
"""
from __future__ import annotations

from ._common import words

CODE = "fr"
NAME = "French"
NATIVE = "Français"

PHONEMES = [
    "i", "e", "ɛ", "a", "ɔ", "o", "u", "y", "ø", "œ", "ə", "ɛ̃", "ɑ̃", "ɔ̃", "j", "w", "ɥ",
    "p", "b", "t", "d", "k", "g", "f", "v", "s", "z", "ʃ", "ʒ", "m", "n", "ɲ", "l", "ʁ",
]
EXAMPLES = {
    "i": "lit", "e": "été", "ɛ": "mère", "a": "chat", "ɔ": "porte", "o": "beau", "u": "loup", "y": "rue",
    "ø": "deux", "œ": "peur", "ə": "petit", "ɛ̃": "vin", "ɑ̃": "temps", "ɔ̃": "bon", "j": "fille", "w": "oui",
    "ɥ": "nuit", "p": "pain", "b": "bon", "t": "tête", "d": "doux", "k": "cou", "g": "gare", "f": "feu",
    "v": "vous", "s": "sel", "z": "rose", "ʃ": "chat", "ʒ": "joue", "m": "main", "n": "nez", "ɲ": "montagne",
    "l": "lune", "ʁ": "rouge",
}
DIGITS = "zéro un deux trois quatre cinq six sept huit neuf".split()

_FUNCTION = {
    "le": ["l", "ə"], "je": ["ʒ", "ə"], "de": ["d", "ə"], "ce": ["s", "ə"], "me": ["m", "ə"], "te": ["t", "ə"],
    "se": ["s", "ə"], "ne": ["n", "ə"], "que": ["k", "ə"], "les": ["l", "e"], "des": ["d", "e"], "mes": ["m", "e"],
    "tes": ["t", "e"], "ses": ["s", "e"], "ces": ["s", "e"], "et": ["e"], "est": ["ɛ"], "es": ["ɛ"], "un": ["ɛ̃"],
    "une": ["y", "n"], "il": ["i", "l"], "ils": ["i", "l"], "elle": ["ɛ", "l"], "elles": ["ɛ", "l"], "a": ["a"],
    "à": ["a"], "y": ["i"], "on": ["ɔ̃"], "en": ["ɑ̃"], "au": ["o"], "aux": ["o"], "eu": ["y"], "plus": ["p", "l", "y"],
    "fils": ["f", "i", "s"], "os": ["ɔ", "s"], "tous": ["t", "u", "s"], "six": ["s", "i", "s"], "dix": ["d", "i", "s"],
    "sept": ["s", "ɛ", "t"], "huit": ["ɥ", "i", "t"], "neuf": ["n", "œ", "f"], "oeuf": ["œ", "f"], "œuf": ["œ", "f"],
    "monsieur": ["m", "ə", "s", "j", "ø"], "femme": ["f", "a", "m"], "second": ["s", "ə", "g", "ɔ̃"],
}
_ER_PRONOUNCED = {"mer", "fer", "hier", "cher", "fier", "hiver", "amer", "enfer", "super", "hamster"}
_ILL_AS_IL = ("ville", "mille", "tranquill")
_VOWEL_L = set("aeiouyéèêëàâîïôûùüœæ")
_FINAL_SILENT = set("sxztdpg")
_SIMPLE_C = {"b": "b", "d": "d", "f": "f", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p", "r": "ʁ", "t": "t",
             "v": "v", "w": "w", "z": "z", "j": "ʒ"}


def _strip_silent(w: str) -> tuple[str, list[str]]:
    """Drop silent final letters; returns (core spelling, phonemes to append at the end)."""
    tail: list[str] = []
    ended_e = False
    if len(w) > 3 and w.endswith("es"):
        w, ended_e = w[:-2], True
    elif len(w) > 2 and w.endswith("e") and w[-2] not in _VOWEL_L - {"u"}:
        w, ended_e = w[:-1], True
    if ended_e:
        if w.endswith(("g", "c")):  # soft g / c before the silent e: page, place
            w = w[:-1] + ("j" if w[-1] == "g" else "ç")
        return w, tail
    if w.endswith("ent") and len(w) > 4:
        return w[:-3], ["ɑ̃"]
    if w.endswith(("er", "ez")) and len(w) > 2 and w not in _ER_PRONOUNCED:
        return w[:-2], ["e"]
    if w.endswith("et") and len(w) > 2:
        return w[:-2], ["ɛ"]
    while len(w) > 1 and w[-1] in _FINAL_SILENT:
        w = w[:-1]
    return w, tail


def _nasal_ok(w: str, j: int) -> bool:
    """True when the vowel+n/m ending before index ``j`` is nasal: ``j`` is the end or a consonant other than n/m/h."""
    return j >= len(w) or (w[j] not in _VOWEL_L and w[j] not in "nmh")


def word_phonemes(word: str) -> list[str]:
    if word in _FUNCTION:
        return list(_FUNCTION[word])
    w, tail = _strip_silent(word)
    out: list[str] = []
    n = len(w)
    i = 0
    while i < n:
        r = w[i:]
        c = w[i]
        prev = w[i - 1] if i else "#"
        nxt = w[i + 1] if i + 1 < n else "#"

        def at(k: int) -> str:
            return w[i + k] if i + k < n else "#"

        # --- vowels
        if r.startswith("eau"):
            out.append("o"); i += 3; continue  # noqa: E702
        if r.startswith(("ien",)) and _nasal_ok(w, i + 3):
            out += ["j", "ɛ̃"]; i += 3; continue  # noqa: E702
        if r.startswith("oin") and _nasal_ok(w, i + 3):
            out += ["w", "ɛ̃"]; i += 3; continue  # noqa: E702
        if r.startswith(("ain", "ein", "aim")) and _nasal_ok(w, i + 3):
            out.append("ɛ̃"); i += 3; continue  # noqa: E702
        if r.startswith("ill") and i > 0:
            if prev in _VOWEL_L:
                out.append("j")
            elif w.startswith(_ILL_AS_IL) or word.startswith(_ILL_AS_IL):
                out += ["i", "l"]
            else:
                out += ["i", "j"]
            i += 3
            continue
        if r == "il" and prev in _VOWEL_L:
            out.append("j"); i += 2; continue  # noqa: E702
        if r.startswith(("aill", "eill", "euill", "ouill")) or r in ("ail", "eil", "euil", "ouil"):
            v = {"a": ["a"], "e": ["ɛ"], "u": ["œ"], "o": ["u"]}[c if c != "e" or nxt != "u" else "u"]
            out += v + ["j"]
            i += next((len(p) for p in ("aill", "eill", "euill", "ouill") if r.startswith(p)), len(r))
            continue
        if r.startswith(("ay", "oy", "uy")) and (at(2) in _VOWEL_L or (at(2) == "#" and tail)):
            out += {"a": ["ɛ", "j"], "o": ["w", "a", "j"], "u": ["ɥ", "i", "j"]}[c]
            i += 2
            continue
        if r.startswith(("au",)):
            out.append("o"); i += 2; continue  # noqa: E702
        if r.startswith(("ai", "ei")):
            out.append("ɛ"); i += 2; continue  # noqa: E702
        if r.startswith(("oi", "oî")):
            out += ["w", "a"]; i += 2; continue  # noqa: E702
        if r.startswith(("ou", "où", "oû")):
            out.append("w" if at(2) in _VOWEL_L else "u"); i += 2; continue  # noqa: E702
        if r.startswith(("œu", "eu")):
            out.append("œ" if at(2) in "rlfvn" and at(2) else "ø"); i += 2; continue  # noqa: E702
        if c in "ae" and nxt in "nm" and _nasal_ok(w, i + 2):
            out.append("ɑ̃"); i += 2; continue  # noqa: E702
        if c == "o" and nxt in "nm" and _nasal_ok(w, i + 2):
            out.append("ɔ̃"); i += 2; continue  # noqa: E702
        if c in "iuy" and nxt in "nm" and _nasal_ok(w, i + 2):
            out.append("ɛ̃"); i += 2; continue  # noqa: E702
        if r.startswith("tion") and i > 0:
            out += ["s", "j", "ɔ̃"]; i += 4; continue  # noqa: E702
        if c == "u" or c in "ûù":
            if nxt == "i" or (nxt in _VOWEL_L and prev not in "qg"):
                out.append("ɥ")
            elif prev in "qg" and c == "u":
                pass  # qu, gu: the u is silent (handled here when the consonant did not consume it)
            else:
                out.append("y")
            i += 1
            continue
        if c in "iïî":
            out.append("j" if nxt in _VOWEL_L and i > 0 else "i"); i += 1; continue  # noqa: E702
        if c == "y":
            if prev in _VOWEL_L and nxt in _VOWEL_L:
                out += ["j"] if prev in "ou" else ["j"]
            else:
                out.append("j" if nxt in _VOWEL_L else "i")
            i += 1
            continue
        if c == "é":
            out.append("e"); i += 1; continue  # noqa: E702
        if c in "èêë":
            out.append("ɛ"); i += 1; continue  # noqa: E702
        if c in "aàâ":
            out.append("a"); i += 1; continue  # noqa: E702
        if c in "oô":
            out.append("o" if (i == n - 1 or c == "ô") and not tail else "ɔ"); i += 1; continue  # noqa: E702
        if c == "œ":
            out.append("œ"); i += 1; continue  # noqa: E702
        if c == "e":
            k = 0
            while i + 1 + k < n and w[i + 1 + k] not in _VOWEL_L:
                k += 1
            if i == n - 1:
                out.append("ə")
            elif nxt == "x":
                out.append("ɛ")
            elif k >= 2 and not r.startswith(("ech", "eph", "eth", "egn")) or (i + 1 + k >= n and k >= 1):
                out.append("ɛ")
            else:
                out.append("ə")
            i += 1
            continue
        # --- consonants
        if r.startswith("gn"):
            out.append("ɲ"); i += 2; continue  # noqa: E702
        if r.startswith("ch"):
            out.append("ʃ"); i += 2; continue  # noqa: E702
        if r.startswith("ph"):
            out.append("f"); i += 2; continue  # noqa: E702
        if r.startswith("th"):
            out.append("t"); i += 2; continue  # noqa: E702
        if r.startswith("qu"):
            out.append("k"); i += 2; continue  # noqa: E702
        if r.startswith("gu") and at(2) in "eiyéèê":
            out.append("g"); i += 2; continue  # noqa: E702
        if c == "g":
            out.append("ʒ" if nxt in "eiyéèê" else "g"); i += 1; continue  # noqa: E702
        if c == "c":
            out.append("s" if nxt in "eiyéèê" else "k"); i += 1; continue  # noqa: E702
        if c == "ç":
            out.append("s"); i += 1; continue  # noqa: E702
        if c == "s":
            if nxt == "s":
                out.append("s"); i += 2; continue  # noqa: E702
            out.append("z" if prev in _VOWEL_L and nxt in _VOWEL_L else "s"); i += 1; continue  # noqa: E702
        if c == "x":
            out += ["g", "z"] if prev == "e" and i == 1 and nxt in _VOWEL_L else ["k", "s"]
            i += 1
            continue
        if c == "h" or c == "'":
            i += 1
            continue
        if c == nxt:
            i += 1
            continue
        if c in _SIMPLE_C:
            out.append(_SIMPLE_C[c])
        i += 1
    return out + tail


def phonemes_of(text: str) -> list[str]:
    out: list[str] = []
    for w in words(text, DIGITS):
        if "'" in w:  # elision: l'eau, qu'il, aujourd'hui
            head, _, rest = w.partition("'")
            if head.endswith("qu"):  # qu', jusqu', lorsqu', puisqu'
                out.extend((word_phonemes(head[:-2]) if len(head) > 2 else []) + ["k"])
                w = rest
            elif len(head) <= 2:  # l' d' j' n' s' c' m' t'
                out.extend(word_phonemes(head + "e")[:-1])
                w = rest
            else:
                w = head + rest
        out.extend(word_phonemes(w))
    return out
