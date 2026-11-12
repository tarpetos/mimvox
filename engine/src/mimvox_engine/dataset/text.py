"""Reading-text analysis: word count, estimated reading time, phoneme coverage, level and ~60-90 s parts.

Every measure is per language (``dataset.lang``): the reading pace for the time estimates, the phoneme inventory and
the coverage thresholds. ``text.analyze`` detects the language when ``language`` is omitted.
"""
from __future__ import annotations

import re
from typing import Any

from ..registry import EngineError, method
from . import lang
from .levels import level_for, message_for

WPM = 145.0              # English reading pace (words per minute); other languages: ``lang.LANGUAGES[code].wpm``
PART_MIN_S = 60.0
PART_TARGET_S = 75.0
PART_MAX_S = 90.0
MERGE_TAIL_S = 35.0      # a final part shorter than this is merged into the previous one when it still fits

# Letters of any script and digits, joined by apostrophes or hyphens (don't, п'ять, l'eau, well-known).
_WORD_RE = re.compile(r"[^\W_]+(?:['’ʼ\-][^\W_]+)*")
# A sentence ends at . ! ? … (plus closing quotes/brackets) followed by space; the next one must start with an
# opening quote/bracket/¿/¡/dash or an upper-case letter or a digit (checked in ``_split_sentences``).
_SENT_END = re.compile(r"(?<=[.!?…])[\"'”’»)\]]*\s+")
_SENT_OPENERS = "\"'“„«([¿¡—–-"


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def est_seconds(words: int, wpm: float = WPM) -> int:
    return int(round(words / wpm * 60.0))


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    pos = 0
    for m in _SENT_END.finditer(text):
        nxt = text[m.end():m.end() + 1]
        if nxt and (nxt in _SENT_OPENERS or nxt.isupper() or nxt.isdigit()):
            out.append(text[pos:m.end()].strip())
            pos = m.end()
    out.append(text[pos:].strip())
    return [s for s in out if s]


def _units(text: str) -> list[tuple[str, int, bool]]:
    """Paragraph units as (text, words, ends_paragraph)."""
    out = []
    for para in re.split(r"\n+", text.replace("\r\n", "\n").replace("\r", "\n")):
        para = para.strip()
        if para:
            out.append((para, count_words(para), True))
    return out


def _sentences(unit: tuple[str, int, bool]) -> list[tuple[str, int, bool]]:
    text, _, _ = unit
    sents = _split_sentences(text)
    res = [(s, count_words(s), False) for s in sents]
    if res:
        res[-1] = (res[-1][0], res[-1][1], True)
    return res


def split_parts(text: str, target_s: float = PART_TARGET_S, min_s: float = PART_MIN_S,
                max_s: float = PART_MAX_S, wpm: float = WPM) -> list[dict[str, Any]]:
    """Split ``text`` at paragraph/sentence boundaries into parts of about ``min_s``..``max_s`` seconds of reading."""
    per_s = wpm / 60.0
    target_w, min_w, max_w = target_s * per_s, min_s * per_s, max_s * per_s
    queue = _units(text)
    groups: list[list[tuple[str, int, bool]]] = []
    cur: list[tuple[str, int, bool]] = []
    curw = 0
    while queue:
        u = queue.pop(0)
        if cur and curw + u[1] > max_w:
            if curw < min_w and u[2] and u[1] > per_s * 15:
                sents = _sentences(u)
                if len(sents) > 1:
                    queue = sents + queue
                    continue
            groups.append(cur)
            cur, curw = [], 0
        if not cur and u[1] > max_w and u[2]:
            sents = _sentences(u)
            if len(sents) > 1:
                queue = sents + queue
                continue
        cur.append(u)
        curw += u[1]
        if curw >= target_w or (curw >= min_w and u[2]):
            groups.append(cur)
            cur, curw = [], 0
    if cur:
        tail_w = curw
        if groups and tail_w < MERGE_TAIL_S * per_s and sum(x[1] for x in groups[-1]) + tail_w <= max_w * 1.25:
            groups[-1].extend(cur)
        else:
            groups.append(cur)
    parts = []
    for i, g in enumerate(groups, 1):
        text_out = ""
        for j, (t, _, para_end) in enumerate(g):
            text_out += t
            if j < len(g) - 1:
                text_out += "\n\n" if para_end else " "
        words = sum(x[1] for x in g)
        parts.append({"index": i, "title": f"Part {i}", "text": text_out, "words": words, "est_seconds": est_seconds(words, wpm)})
    return parts


def _coverage_note(missing: list[str], code: str) -> str:
    if not missing:
        return ""
    shown = ", ".join(f"{p} (as in \"{lang.example(code, p)}\")" for p in missing[:4])
    more = f" and {len(missing) - 4} more" if len(missing) > 4 else ""
    return f"Missing sounds: {shown}{more}. Add words that use them."


def analyze(text: str, language: str | None = None) -> dict[str, Any]:
    """Analysis of ``text`` read in ``language`` (an ISO 639-1 code known to ``dataset.lang``; detected when None)."""
    detected = language is None
    code = lang.detect(text) if detected else language
    info = lang.get(code)
    inv = lang.inventory(code)
    base = {"language": code, "language_detected": detected, "phonemes_total": len(inv)}
    words = count_words(text)
    if not words:
        return {**base, "words": 0, "est_minutes": 0.0, "phoneme_coverage": 0.0, "missing_phonemes": inv,
                "level": "too_short", "message": "Type or paste the text you want to read aloud.", "parts": []}
    minutes = words / info.wpm
    cov, missing, _ = lang.coverage(text, code)
    level = level_for(minutes, cov, info.low_coverage)
    note = _coverage_note(missing, code) if cov < info.hint_coverage else ""
    if note and cov < info.low_coverage:
        note = "Low sound variety. " + note
    parts = [{k: v for k, v in p.items() if k != "words"} for p in split_parts(text, wpm=info.wpm)]
    return {
        **base,
        "words": words,
        "est_minutes": round(minutes, 2),
        "phoneme_coverage": round(cov, 3),
        "missing_phonemes": missing,
        "level": level,
        "message": message_for(minutes, level, note, info.wpm),
        "parts": parts,
    }


@method("text.analyze")
def text_analyze(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    text = params.get("text")
    if not isinstance(text, str):
        raise EngineError("bad_request", "text.analyze needs a string 'text'.")
    try:
        code = lang.normalize_code(params.get("language"))
    except ValueError as exc:
        raise EngineError("bad_request", str(exc)) from None
    return analyze(text, code)
