"""Word stress for OmniVoice: the user's stress marks, and automatic stress for Ukrainian, as an upper-case vowel.

OmniVoice was not trained on stress-marked text. Measured on Ukrainian, with the prominent syllable found from vowel
duration, loudness and pitch, it does react to some hints:

- an upper-case stressed vowel (співакОм) moves the stress to that vowel wherever it is in the word. On words stressed
  on the last syllable, the stress landed on the marked syllable 63 % of the time, against 52 % with no mark; on
  співаком, козаком, юнаком, читачем and співакові 88 % against 73 %. On other syllables it was no better than no mark
  (68 % against 72 %), the model gets those right more often by itself;
- a combining acute after the vowel (U+0301) works on other syllables, but on a stressed last syllable it pulls the
  stress one syllable earlier: співако́м came out as співАком (stress on the marked syllable 24 % of the time, 52 %
  with no mark);
- a "+" before the vowel is sometimes read aloud as "плюс" and makes the words less clear.

Russian was measured with the acute only (two-syllable heteronyms: 74 % on the marked syllable against 52 % without);
the upper-case form is used there too, on the Ukrainian evidence.

So a hint reaches the model as an upper-case vowel. The exception is a stressed vowel at the start of a word: there an
upper-case letter would look like an ordinary capital, so it gets the acute, which works there.

- The user's own marks: a "+" right before a vowel, or a vowel followed by U+0301 or by the spacing acute "´". These
  apply in Ukrainian, Russian and Belarusian text ("´" after a consonant is a typed apostrophe and becomes "'"). An
  upper-case vowel inside a lower-case word (співакОм) is already in the model's form and is left as it is. In other
  languages the marks are removed, as before.
- Automatic stress for Ukrainian, off by default (``MIMVOX_STRESS_AUTO`` = ``last``, ``all`` or ``off``): unmarked
  words get the mark from lang-uk's stress dictionary (``ukrainian-word-stress`` in dictionary-only mode,
  ``marisa-trie``). ``last`` marks only words stressed on the last of two or more syllables, ``all`` every word with
  two or more vowels. Words with more than one possible stress (heteronyms such as зАмок / замОк), words the dictionary
  does not know, and words with a mark or a capital inside them are left alone, so the user's mark always wins. It is
  off because it did not help whole sentences: with ``last`` the marked words improved (stress on the right syllable
  57 % -> 61 %), but the other words in the same sentences got worse (65 % -> 50 %); ``all`` was worse overall.

Standalone on purpose: ``omnivoice_text.py`` loads it by its path; it needs only ``re``. ``marisa_trie`` and the
dictionary are needed only for the automatic stress, which is skipped without them.

    user_marks(text, lang) -> text      the user's marks in the model's form (kept languages) or removed (others)
    auto(text, lang) -> text            dictionary stress for unmarked Ukrainian words (off by default)
"""
import os
import re

ACUTE = "\u0301"
_GRAVE = "\u0300"   # secondary stress in dictionaries: no use to the model
# Languages whose stress marks reach the model; the rest have them removed (they are not part of the spelling there).
KEEP = {"uk", "ru", "be"}
_VOWELS = "аеєиіїоуюяыэёАЕЄИІЇОУЮЯЫЭЁ"
_CYR = "а-яёіїєґўА-ЯЁІЇЄҐЎ"
_APOS = "'’ʼ"
# A "+" right before a Cyrillic vowel, at the start of a word or inside it ("з+амок", "+атлас").
_PLUS = re.compile(r"(?:(?<=[" + _CYR + _APOS + r"])|(?<![\w+]))\+(?=[" + _VOWELS + r"])")
_SPACING_ACUTE = re.compile(r"(?<=[" + _VOWELS + r"])[´ˊ]")
_MARKED = re.compile(r"[" + _CYR + _APOS + r"\0\u0301]+")
_WORD = re.compile(r"[" + _CYR + r"\u0301]+(?:[" + _APOS + r"][" + _CYR + r"\u0301]+)*")

_dict = None   # the stress trie; False when it cannot be loaded


def _mark(word, i):
    """``word`` with the vowel at letter index ``i`` marked in the model's form."""
    if not any(c.isalpha() for c in word[:i]):
        return word[:i + 1] + ACUTE + word[i + 1:]   # a word-initial vowel: a capital there means nothing
    return word[:i] + word[i].upper() + word[i + 1:]


def _convert(word):
    """One word with the user's marks (U+0301 after a vowel, "\\0" before one) in the model's form."""
    plain, stressed = [], []   # the word without marks; the letter indexes the user marked
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
    if not stressed or text.isupper():   # no mark, or a word in capitals where a mark cannot be shown
        return text
    for k in sorted(set(stressed), reverse=True):
        text = _mark(text, k)
    return text


def user_marks(text, lang):
    """The user's stress marks in the model's form for KEEP languages, removed for the others."""
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
        except (ImportError, OSError, RuntimeError):   # no package or no dictionary: no automatic stress
            pass
    return _dict


def _stress_of(trie, word):
    """Letter index of the single stressed vowel of ``word``, or None when the dictionary has no single answer.

    A value is the dictionary's own record: accent positions (the index just after the stressed letter) as bytes;
    several readings are split by a record byte and carry tags after a POS byte. Lower case is tried first: a capital
    at the start of a sentence is not a proper name ("Новий" would otherwise be the surname Но́вий).
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
    """Mark the stress of unmarked Ukrainian words from the dictionary, when ``MIMVOX_STRESS_AUTO`` asks for it."""
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
