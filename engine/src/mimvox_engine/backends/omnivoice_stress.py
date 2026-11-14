"""Word stress for OmniVoice, given to the model as an upper-case vowel.

Measured: an upper-case vowel put a final stress on it 63 % of the time (52 % unmarked); U+0301 there pulled it a
syllable earlier. ``omnivoice_text.py`` loads this file by path, so it imports only the standard library.
"""
import os
import re

ACUTE = "\u0301"
_GRAVE = "\u0300"
KEEP = {"uk", "ru", "be"}
_VOWELS = "аеєиіїоуюяыэёАЕЄИІЇОУЮЯЫЭЁ"
_CYR = "а-яёіїєґўА-ЯЁІЇЄҐЎ"
_APOS = "'’ʼ"
_PLUS = re.compile(r"(?:(?<=[" + _CYR + _APOS + r"])|(?<![\w+]))\+(?=[" + _VOWELS + r"])")
_SPACING_ACUTE = re.compile(r"(?<=[" + _VOWELS + r"])[´ˊ]")
_MARKED = re.compile(r"[" + _CYR + _APOS + r"\0\u0301]+")
_WORD = re.compile(r"[" + _CYR + r"\u0301]+(?:[" + _APOS + r"][" + _CYR + r"\u0301]+)*")

_dict = None


def _mark(word, i):
    if not any(c.isalpha() for c in word[:i]):
        return word[:i + 1] + ACUTE + word[i + 1:]   # a word-initial vowel: a capital there means nothing
    return word[:i] + word[i].upper() + word[i + 1:]


def _convert(word):
    """Marks in ``word`` are U+0301 after a vowel or "\\0" before one."""
    plain, stressed = [], []
    for i, c in enumerate(word):
        if c == "\0":
            if i + 1 < len(word) and word[i + 1] in _VOWELS:
                stressed.append(len(plain))
        elif c == ACUTE:
            if plain and plain[-1] in _VOWELS:
                stressed.append(len(plain) - 1)
        else:
            plain.append(c)
    text = "".join(plain)
    if not stressed or text.isupper():   # a word in capitals cannot show an upper-case mark
        return text
    for k in sorted(set(stressed), reverse=True):
        text = _mark(text, k)
    return text


def user_marks(text, lang):
    text = text.replace(_GRAVE, "")
    if lang not in KEEP:
        return text.replace(ACUTE, "")
    text = _PLUS.sub("\0", text)
    text = _SPACING_ACUTE.sub(ACUTE, text).replace("´", "'").replace("ˊ", "'")
    if ACUTE in text or "\0" in text:
        text = _MARKED.sub(lambda m: _convert(m.group()), text).replace("\0", "")
    return text


def _load():
    global _dict
    if _dict is None:
        _dict = False
        try:
            import marisa_trie
            import ukrainian_word_stress

            trie = marisa_trie.BytesTrie()
            trie.load(os.path.join(os.path.dirname(ukrainian_word_stress.__file__), "data", "stress.trie"))
            _dict = trie
        except (ImportError, OSError, RuntimeError):
            pass
    return _dict


def _stress_of(trie, word):
    """Letter index of the stressed vowel, or None unless the dictionary has exactly one.

    Trie value: accent positions (index after the stressed letter) as bytes; readings split by 0xFF, tags after 0xFE.
    """
    key = re.sub("[" + _APOS + "]", "'", word)
    for cand in (key.lower(), key, key.capitalize()):
        if cand in trie:
            value = trie[cand][0]
            break
    else:
        return None
    readings = {tuple(item.partition(b"\xfe")[0]) for item in value.split(b"\xff") if item}
    if len(readings) != 1:
        return None   # a heteronym: the model decides
    accents = next(iter(readings))
    if len(accents) != 1:
        return None   # two accepted stresses (по́ми́лка), or a compound
    i = accents[0] - 1
    return i if 0 <= i < len(word) and word[i] in _VOWELS else None


def auto(text, lang):
    # Off by default: measured, marked words improved (57 % -> 61 %) but the rest got worse (65 % -> 50 %).
    scope = os.environ.get("MIMVOX_STRESS_AUTO", "").strip().lower()
    if lang != "uk" or scope not in ("last", "all"):
        return text
    trie = _load()
    if not trie:
        return text

    def one(m):
        w = m.group()
        vowels = [i for i, c in enumerate(w) if c in _VOWELS]
        if len(vowels) < 2 or ACUTE in w or any(c.isupper() for c in w[1:]):
            return w
        i = _stress_of(trie, w)
        if i is None or (scope == "last" and i != vowels[-1]):
            return w
        return _mark(w, i)

    return _WORD.sub(one, text)
