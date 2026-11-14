from __future__ import annotations

import re
import unicodedata

WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "`": "'", "´": "'"})


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_APOSTROPHES).lower()


def expand_digits(text: str, digit_words: list[str]) -> str:
    return re.sub(r"\d", lambda m: f" {digit_words[int(m.group())]} ", text)


def words(text: str, digit_words: list[str]) -> list[str]:
    return WORD_RE.findall(expand_digits(normalize(text), digit_words))
