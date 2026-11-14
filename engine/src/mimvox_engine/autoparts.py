"""Suggest parts for imported chapters that are missing or very long, from the shape of the text, not its words."""
from __future__ import annotations

import math
import re
from collections import Counter

from .dataset.text import count_words

LONG_WORDS = 4000        # ~28 min at 145 wpm
TARGET_WORDS = 1800      # ~12 min
MIN_WORDS = 600
WINDOW_WORDS = 300
HEADING_MAX_WORDS = 10
HEADING_GAP_WORDS = 150

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


def _by_headings(paras: list[str]) -> list[tuple[str | None, list[str]]] | None:
    kinds = [_heading_kind(p) for p in paras]
    w = _words(paras)
    cum = [0]
    for x in w:
        cum.append(cum[-1] + x)
    best: list[int] = []
    for kind in ("kw", "roman", "number", "caps"):
        idx = [i for i, k in enumerate(kinds) if k == kind or (kind == "kw" and k == "single")]
        # Close lines of one kind are a list or a poem ("TRESPASSERS / WILL BE"), not chapter titles.
        near = set()
        for a, b in zip(idx, idx[1:]):
            if cum[b] - cum[a + 1] < HEADING_GAP_WORDS:
                near |= {a, b}
        idx = [i for i in idx if i not in near]
        if len(idx) >= 2:
            best = idx
            break
    if not best:
        return None
    pre = paras[:best[0]]
    pre_long = sum(_words(pre)) >= MIN_WORDS
    out: list[tuple[str | None, list[str]]] = [(None, pre)] if pre_long else []
    for k, (a, b) in enumerate(zip(best, best[1:] + [len(paras)])):
        body = paras[a + 1:b]
        if k == 0 and pre and not pre_long:
            body = pre + body
        out.append((paras[a].strip().rstrip("."), body))
    return out


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
    """Higher is better; -inf where a cut is not allowed."""
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
        score += 0.15
    if nxt.startswith(_DIALOGUE):
        score -= 0.1
    if w[i] >= 40 and not nxt.startswith(_DIALOGUE):
        score += 0.05
    return score


def _by_length(paras: list[str]) -> list[list[str]]:
    w = _words(paras)
    total = sum(w)
    if total <= TARGET_WORDS * 1.4 or len(paras) < 2:
        return [paras]
    # Even parts: 4300 words are two of 2150, not 1800 + 1800 + 700.
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
    """None when no chapter is long or nothing could be split."""
    if not any(count_words(c.get("text", "")) > LONG_WORDS for c in chapters):
        return None
    word = PART_WORD.get((lang or "").lower())
    # A title page or an "about this edition" next to the one real chapter does not count.
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
