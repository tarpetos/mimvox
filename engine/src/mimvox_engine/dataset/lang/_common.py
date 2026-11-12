"""Helpers shared by the rule-based grapheme-to-phoneme modules (pure Python, no dependencies)."""
from __future__ import annotations

import re
import unicodedata

# A word: letters of any script, optionally joined by apostrophes (l'eau, п'ять, don't). Digits are expanded first.
WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'", "´": "'"})


def normalize(text: str) -> str:
    """NFC, lower case, every apostrophe variant as ``'``."""
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHES).lower()


def expand_digits(text: str, digit_words: list[str]) -> str:
    """Read stray digits one by one (presets spell numbers out; this only keeps typed digits from being ignored)."""
    return re.sub(r"\d", lambda m: f" {digit_words[int(m.group())]} ", text)


def words(text: str, digit_words: list[str]) -> list[str]:
    return WORD_RE.findall(expand_digits(normalize(text), digit_words))
