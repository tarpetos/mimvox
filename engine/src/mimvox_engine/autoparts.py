"""Automatic parts for a book whose chapters are missing or very long (``text.import`` -> ``parts``).

Many files have no chapter marks the importer can see (``textimport.py`` reads headings, the EPUB table of contents,
FB2 sections, PDF bookmarks and "Chapter 1" lines): a whole novella comes in as one chapter of two hours, which is
awkward to listen to, to resume and to find a place in. This module suggests a split; the app applies it with one
button and puts the original back with another, so nothing here changes the import itself.

Language-independent on purpose (Mimvox reads ~60 languages): it looks at the shape of the text, not its words. For
every chapter longer than LONG_WORDS (about half an hour at 145 words a minute), the first of these that applies:

1. **Hidden headings**: short paragraphs that look like chapter titles but were not marked as headings: the importer's
   own heading lines ("Chapter 4", "Розділ IV", "Prologue"), a bare number or Roman numeral ("12", "XII."), or a short
   line in capitals without a full stop ("THE RETURN"). At least two of one kind, with text between them, each
   becoming the start (and title) of a part. Real chapters: kept whatever their length.
2. **Scene breaks**: a paragraph of only separator signs (``* * *``, ``***``, ``⁂``, ``———``, ``#``, ``~``, ``§``).
   Scenes are grouped into parts of about TARGET_WORDS (a scene is never cut), so a short story with many breaks does
   not turn into dozens of tiny parts.
3. **Length**: parts of about TARGET_WORDS, cut between paragraphs where the text changes most: lexical cohesion as in
   TextTiling (word stems of the ~300 words before against the ~300 after; a stem is the first five letters, which
   works for inflected languages without a stemmer), plus the end of a dialogue and the start of a long narrative
   paragraph. Never inside a paragraph, never after one that does not end a sentence.

A part still longer than twice TARGET_WORDS after 2 (one very long scene) is split by length as in 3.

Titles: a hidden heading is the title. Other parts of a file that had a single chapter are "<Part> N" in the book's
language when the word is known (Частина, Part, Teil, Partie, ...), else just "N"; parts of a longer chapter keep its
title, then "(2)", "(3)" for the following ones, as the manual split does. Chapter titles are read aloud before each
chapter, so they must read well in the book's language.

    suggest(chapters, lang) -> {"how": "headings"|"breaks"|"length"|"mixed", "chapters": [{"title", "text"}]} or None
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .dataset.text import count_words

LONG_WORDS = 4000        # a chapter longer than this (~28 min at 145 wpm) is worth splitting
TARGET_WORDS = 1800      # ~12 min: the length a part aims at
MIN_WORDS = 600          # a part is never shorter (a short tail joins the part before it)
WINDOW_WORDS = 300       # text compared on each side of a candidate cut (length split)
HEADING_MAX_WORDS = 10
HEADING_GAP_WORDS = 150  # hidden headings must have at least this much text between them

PART_WORD = {
    "uk": "Частина", "en": "Part", "de": "Teil", "fr": "Partie", "es": "Parte", "it": "Parte", "pt": "Parte",
    "pl": "Część", "ro": "Partea", "cs": "Část", "sk": "Časť", "ru": "Часть", "be": "Частка", "bg": "Част",
    "nl": "Deel", "sv": "Del", "no": "Del", "da": "Del", "fi": "Osa", "et": "Osa", "hu": "Rész", "hr": "Dio",
    "sr": "Део", "sl": "Del", "lt": "Dalis", "lv": "Daļa", "el": "Μέρος", "tr": "Bölüm", "ca": "Part",
}

_SEPARATOR = re.compile(r"^[\s*⁂~#§•·◆◇○●■□—–\-_=+]+$")
_NUMBER = re.compile(r"^(\d{1,3}|[IVXLCDM]{1,8})[.)]?$")
_DIALOGUE = tuple("—–-\"«„“'‹")
_SENTENCE_END = ".!?…»\"”’)]"
_WORD = re.compile(r"[^\W\d_]{4,}")


def _is_roman(s: str) -> bool:
    return bool(re.fullmatch(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", s)) and bool(s)


def _heading_kind(p: str) -> str | None:
    """The kind of chapter title a paragraph looks like, or None."""
    from .textimport import _heading_match   # noqa: PLC0415 - textimport imports this module lazily too

    s = p.strip()
    if not s or count_words(s) > HEADING_MAX_WORDS or len(s) > 80:
        return None
    m = _heading_match(s)
    if m:
        return "kw" if m[0] == "kw" else m[0]
    n = _NUMBER.match(s)
    if n and (n.group(1).isdigit() or _is_roman(n.group(1))):
        return "number"
    letters = [c for c in s if c.isalpha()]
    if len(letters) >= 3 and all(c.isupper() for c in letters) and s[-1] not in ".,;:!?…" and not s.startswith(_DIALOGUE):
        return "caps"
    return None


def _words(paras: list[str]) -> list[int]:
    return [count_words(p) for p in paras]


# ---------------------------------------------------------------------------------------------- 1. hidden headings

def _by_headings(paras: list[str]) -> list[tuple[str | None, list[str]]] | None:
    kinds = [_heading_kind(p) for p in paras]
    w = _words(paras)
    cum = [0]
    for x in w:
        cum.append(cum[-1] + x)
    best: list[int] = []
    for kind in ("kw", "roman", "number", "caps"):
        idx = [i for i, k in enumerate(kinds) if k == kind or (kind == "kw" and k == "single")]
        # Lines of this kind with little text between them are a sign, a list or a poem, not chapter titles
        # ("TRESPASSERS / WILL BE / PROSECUTED"): every member of such a cluster is dropped.
        near = set()
        for a, b in zip(idx, idx[1:]):
            if cum[b] - cum[a + 1] < HEADING_GAP_WORDS:
                near |= {a, b}
        idx = [i for i in idx if i not in near]
        if len(idx) >= 2:
            best = idx
            break   # the first kind that works (keywords are the surest)
    if not best:
        return None
    # Text before the first heading: its own part when it is long enough (a prologue without a title), else it opens
    # the first part (an epigraph, a dedication). None as a title: the chapter keeps its own.
    pre = paras[:best[0]]
    pre_long = sum(_words(pre)) >= MIN_WORDS
    out: list[tuple[str | None, list[str]]] = [(None, pre)] if pre_long else []
    for k, (a, b) in enumerate(zip(best, best[1:] + [len(paras)])):
        body = paras[a + 1:b]
        if k == 0 and pre and not pre_long:
            body = pre + body
        out.append((paras[a].strip().rstrip("."), body))
    return out


# --------------------------------------------------------------------------------------------- 2. scene breaks

def _scenes(paras: list[str]) -> list[list[str]] | None:
    cuts = [i for i, p in enumerate(paras) if _SEPARATOR.match(p) and any(not c.isspace() for c in p)]
    if not cuts:
        return None
    scenes, start = [], 0
    for i in cuts + [len(paras)]:
        body = paras[start:i]
        if body:
            scenes.append(body)
        start = i + 1
    return scenes if len(scenes) >= 2 else None


def _group(scenes: list[list[str]]) -> list[list[str]]:
    """Whole scenes into parts of about TARGET_WORDS (closing a part before a scene that would overshoot it)."""
    parts: list[list[str]] = []
    cur: list[str] = []
    cur_w = 0
    for sc in scenes:
        w = sum(_words(sc))
        if cur and cur_w >= MIN_WORDS and cur_w + w > TARGET_WORDS * 1.3:
            parts.append(cur)
            cur, cur_w = [], 0
        cur = cur + sc
        cur_w += w
    if cur:
        if parts and cur_w < MIN_WORDS:
            parts[-1] = parts[-1] + cur
        else:
            parts.append(cur)
    return parts


# ------------------------------------------------------------------------------------------------- 3. length

def _stems(text: str) -> Counter:
    return Counter(w[:5] for w in _WORD.findall(text.lower()))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _cut_score(paras: list[str], w: list[int], i: int) -> float:
    """How good a cut before paragraph ``i`` is (higher is better; -inf where a cut must not be)."""
    prev, nxt = paras[i - 1].rstrip(), paras[i].lstrip()
    if not prev or prev[-1] not in _SENTENCE_END:
        return -math.inf
    left, n = [], 0
    for j in range(i - 1, -1, -1):
        left.append(paras[j])
        n += w[j]
        if n >= WINDOW_WORDS:
            break
    right, n = [], 0
    for j in range(i, len(paras)):
        right.append(paras[j])
        n += w[j]
        if n >= WINDOW_WORDS:
            break
    score = 1.0 - _cosine(_stems(" ".join(left)), _stems(" ".join(right)))
    if prev.startswith(_DIALOGUE) and not nxt.startswith(_DIALOGUE):
        score += 0.15          # a dialogue ends, the narration takes over
    if nxt.startswith(_DIALOGUE):
        score -= 0.1           # do not open a part in the middle of a conversation
    if w[i] >= 40 and not nxt.startswith(_DIALOGUE):
        score += 0.05          # a long narrative paragraph opens a part well
    return score


def _by_length(paras: list[str]) -> list[list[str]]:
    w = _words(paras)
    total = sum(w)
    if total <= TARGET_WORDS * 1.4 or len(paras) < 2:
        return [paras]
    # Even parts: as many as TARGET_WORDS suggests, each total / n long (4300 words: two of 2150, not 1800 + 1800 + 700).
    target = total / max(2, round(total / TARGET_WORDS))
    cum = [0]
    for x in w:
        cum.append(cum[-1] + x)
    cuts, start = [], 0
    while total - cum[start] > target * 1.4:
        lo, hi = cum[start] + target * 0.7, cum[start] + target * 1.3
        cands = [i for i in range(start + 1, len(paras)) if lo <= cum[i] <= hi and total - cum[i] >= MIN_WORDS]
        scored = [(s, i) for i in cands if (s := _cut_score(paras, w, i)) > -math.inf]
        if not scored:   # paragraphs longer than the window: the sentence-ending boundary nearest to the target
            near = [i for i in range(start + 1, len(paras)) if total - cum[i] >= MIN_WORDS
                    and _cut_score(paras, w, i) > -math.inf]
            if not near:
                break
            i = min(near, key=lambda k: abs(cum[k] - cum[start] - target))
        else:
            # Prefer a cut near the target when scores are close: 0.2 of penalty at the window's edges.
            i = max(scored, key=lambda si: si[0] - 0.2 * abs(cum[si[1]] - cum[start] - target) / (target * 0.3))[1]
        cuts.append(i)
        start = i
    bounds = [0] + cuts + [len(paras)]
    return [paras[a:b] for a, b in zip(bounds, bounds[1:]) if paras[a:b]]


# ------------------------------------------------------------------------------------------------------ public

def _split(paras: list[str]) -> tuple[list[tuple[str | None, list[str]]], str] | None:
    heads = _by_headings(paras)
    if heads and len(heads) >= 2:
        return heads, "headings"
    scenes = _scenes(paras)
    if scenes:
        parts: list[list[str]] = []
        for group in _group(scenes):
            parts += _by_length(group) if sum(_words(group)) > TARGET_WORDS * 2 else [group]
        if len(parts) >= 2:
            return [("", p) for p in parts], "breaks"
    parts = _by_length(paras)
    if len(parts) >= 2:
        return [("", p) for p in parts], "length"
    return None


def suggest(chapters: list[dict], lang: str | None) -> dict | None:
    """Parts for the long chapters of an import, or None when none is long or nothing could be split."""
    if not any(count_words(c.get("text", "")) > LONG_WORDS for c in chapters):
        return None
    word = PART_WORD.get((lang or "").lower())
    # One chapter with real text (a title page or an "about this edition" next to it does not count): its parts are
    # "<Part> N", not "<title> (N)".
    single = sum(count_words(c.get("text", "")) >= MIN_WORDS for c in chapters) <= 1
    out: list[dict] = []
    hows: set[str] = set()
    n = 0
    for ch in chapters:
        text = ch.get("text", "")
        title = ch.get("title", "")
        if count_words(text) <= LONG_WORDS:
            out.append({"title": title, "text": text})
            continue
        paras = [p for p in text.split("\n\n") if p.strip()]
        found = _split(paras)
        if not found:
            out.append({"title": title, "text": text})
            continue
        pieces, how = found
        hows.add(how)
        for k, (head, body) in enumerate(pieces, 1):
            n += 1
            if head:
                t = head
            elif head is None:         # text before the first hidden heading: the chapter's own title
                t = title or (f"{word} {k}" if word else str(k))
            elif single:
                t = f"{word} {k}" if word else str(k)
            else:
                t = title if k == 1 else f"{title} ({k})"
            out.append({"title": t, "text": "\n\n".join(body)})
    if not hows:
        return None
    return {"how": hows.pop() if len(hows) == 1 else "mixed", "chapters": out}
