"""Book import for audiobooks: clean text and chapter structure from a text file (``text.import``).

Every format is turned into the same thing: a book title and a list of chapters, each a title plus paragraphs. Most
readers first produce a flat list of *blocks* (a heading with its level, or a paragraph) and one shared step cuts it
into chapters: the highest heading level that occurs at least twice starts the chapters, a single heading above it at
the start is the book title, other headings above it are read as the first paragraph of the next chapter (a "Part
Two" between chapters), deeper headings stay in the text as paragraphs. EPUB (its table of contents), FB2 (nested
sections), PDF (bookmarks) and plain text (heading lines such as ``Chapter 1`` / ``Розділ 1``) bring their own chapters.

Shared rules (``_finish``): text before the first chapter is kept as an "Opening" chapter when it has at least 150
words, otherwise dropped as a title page (unless it is all there is); a chapter does not repeat its title as its first
paragraph; chapters without words are dropped (a "Part" chapter with little text of its own hands its title and text to
the next chapter instead); a table of contents (a chapter named so, or made mostly of chapter headings; in plain text
three or more headings in a row with no text between them) and a leading title-page chapter are dropped; paragraphs
are joined with one blank line, whitespace is collapsed, control, zero-width and soft-hyphen characters are removed
and typographic ligatures expanded. Footnotes, note references, page-number markers and illustration captions are not
part of the text.

Readers use the standard library only: ``zipfile`` + ``ElementTree`` for EPUB, DOCX, ODT and FB2, ``html.parser`` for
HTML/XHTML, a small converter of our own for RTF (``\\uN`` and ``\\'hh`` escapes in the document's or font's code page,
heading styles from the stylesheet). PDF needs ``pypdf`` (imported when used; text layer only, no OCR; AES-encrypted
files would need its optional ``cryptography`` backend and are refused with a clear message). Markdown code blocks
are dropped: they are not meant to be read aloud. Subtitles (SRT/VTT) become flowing text, with sound descriptions
such as ``[Music]`` removed and a paragraph break wherever the speech pauses for 2 s or more.
"""
from __future__ import annotations

import codecs
import html
import itertools
import logging
import os
import posixpath
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from html.entities import html5 as _HTML5
from html.parser import HTMLParser
from typing import Any
from collections.abc import Iterable, Iterator
from urllib.parse import unquote
from xml.etree import ElementTree as ET

from .dataset import lang
from .dataset.text import WPM, count_words, est_seconds
from .registry import EngineError, method

log = logging.getLogger("mimvox.textimport")

SUPPORTED: dict[str, str] = {
    ".txt": "Plain text", ".md": "Markdown", ".markdown": "Markdown", ".html": "HTML", ".htm": "HTML",
    ".xhtml": "XHTML", ".epub": "EPUB", ".docx": "Word document", ".odt": "OpenDocument text", ".rtf": "Rich Text",
    ".fb2": "FictionBook", ".fb2.zip": "FictionBook (zipped)", ".pdf": "PDF", ".srt": "SubRip subtitles",
    ".vtt": "WebVTT subtitles",
}
_FORMAT = {".txt": "txt", ".md": "md", ".markdown": "md", ".html": "html", ".htm": "html", ".xhtml": "html",
           ".epub": "epub", ".docx": "docx", ".odt": "odt", ".rtf": "rtf", ".fb2": "fb2", ".fb2.zip": "fb2",
           ".pdf": "pdf", ".srt": "srt", ".vtt": "vtt"}
MAX_BYTES = 300 * 1024 * 1024
MAX_MEMBER = 512 * 1024 * 1024     # uncompressed size of one archive member we read (zip bombs)
OPENING_WORDS = 150                # text before the first chapter is kept as its own chapter from this many words
PART_WORDS = 60                    # a "Part" with less text of its own (an epigraph) opens the next chapter instead
SAMPLE_CHARS = 20_000              # language detection sample, spread over the book


@dataclass
class _Ch:
    title: str
    paras: list[str] = field(default_factory=list)
    part: bool = False             # a "Part"/"Book" heading: with (almost) no text, it opens the next chapter
    lead: list[str] = field(default_factory=list)


@dataclass
class _Book:
    title: str = ""
    chapters: list[_Ch] = field(default_factory=list)
    preamble: list[str] = field(default_factory=list)
    encoding: str | None = None
    warnings: list[str] = field(default_factory=list)


Block = tuple[str, int, str]       # ("h", level, text) | ("p", 0, text)


# ------------------------------------------------------------------------------------------------------ cleaning

_TRANS: dict[int, Any] = {c: " " for c in range(0x20)}
_TRANS.update({0x7F: " ", **{c: " " for c in range(0x80, 0xA0)}})
_TRANS.update({c: None for c in (0x00AD, 0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0xFFFD, 0x200E, 0x200F,
                                 *range(0x202A, 0x202F), *range(0x2066, 0x206A))})
_TRANS.update({c: " " for c in (0x00A0, 0x1680, *range(0x2000, 0x200B), 0x2028, 0x2029, 0x202F, 0x205F, 0x3000)})
_TRANS.update({0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "st", 0xFB06: "st"})
_WS = re.compile(r"\s+")
_WORDISH = re.compile(r"[^\W_]")


def _clean(text: str) -> str:
    """One paragraph: NFC, no control/zero-width/soft-hyphen characters, single spaces, trimmed."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text).translate(_TRANS)).strip()


def _key(text: str) -> str:
    """Comparison key: lower-case words only."""
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})", re.IGNORECASE)


def _is_roman(word: str) -> bool:
    return bool(word) and len(word) <= 9 and _ROMAN.fullmatch(word) is not None


def _roman_value(word: str) -> int:
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    w = word.lower()
    for i, ch in enumerate(w):
        v = vals[ch]
        total += -v if i + 1 < len(w) and vals[w[i + 1]] > v else v
    return total


def _tidy_title(title: str) -> str:
    """Clean a title; an ALL-CAPS one gets its words capitalised (roman numerals stay upper case)."""
    t = _clean(title).strip(" .:-–—")
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):
        t = " ".join(w if _is_roman(w.strip(".:,")) else w[:1] + w[1:].lower() for w in t.split(" "))
    return t[:200]


def _join_title(head: str, sub: str) -> str:
    head = head.strip()
    return f"{head} {sub}" if head[-1:] in ".:!?—–-" else f"{head}. {sub}"


# ------------------------------------------------------------------------------------------------------ decoding

_BOMS = ((codecs.BOM_UTF32_LE, "utf-32-le"), (codecs.BOM_UTF32_BE, "utf-32-be"), (codecs.BOM_UTF8, "utf-8"),
         (codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be"))


def _plausible(text: str) -> float:
    """Share of characters that are letters, digits, whitespace or common punctuation."""
    if not text:
        return 0.0
    good = sum(1 for c in text if c.isalnum() or c.isspace() or c in ".,;:!?'\"-()–—«»„“”‘’…")
    return good / len(text)


def _decode(data: bytes, hint: str | None = None) -> tuple[str, str]:
    """Decode a text file: BOM, BOM-less UTF-16, strict UTF-8, ``hint`` (a declared charset), then a guess between
    windows-1251 (Cyrillic), windows-1250 (Central European) and windows-1252 (Western)."""
    for bom, enc in _BOMS:
        if data.startswith(bom):
            return data[len(bom):].decode(enc, errors="replace"), enc
    head = data[:200_000]
    if head.count(0) > len(head) // 20:          # many NULs: UTF-16 without a BOM
        best: tuple[float, str, str] | None = None
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                text = data[:len(data) // 2 * 2].decode(enc)
            except UnicodeDecodeError:
                continue
            score = _plausible(text[:50_000])
            if best is None or score > best[0]:
                best = (score, text, enc)
        if best and best[0] > 0.9:
            return best[1], best[2]
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    if hint:
        try:
            enc = codecs.lookup(hint).name
            if enc not in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
                return data.decode(enc, errors="replace"), enc
        except LookupError:
            pass
    high = sum(1 for b in head if b >= 0x80)
    ascii_letters = sum(1 for b in head if 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A)
    cyr_text = head.decode("cp1251", errors="replace")
    cyr = sum(1 for c in cyr_text if "\u0400" <= c <= "\u04ff")
    lower = sum(1 for c in cyr_text if "а" <= c <= "я" or c in "іїєґё")
    if cyr and cyr / max(1, cyr + ascii_letters) >= 0.3:
        if lower >= cyr * 0.5:
            return data.decode("cp1251", errors="replace"), "cp1251"
        return data.decode("koi8-u", errors="replace"), "koi8-u"   # KOI8 swaps the cases of windows-1251
    # Central European: bytes that are letters in windows-1250 but superscripts/symbols in windows-1252 (ą ł ś ź ż)
    ce = sum(1 for b in head if b in (0xB9, 0xB3, 0x9C, 0x9F, 0xA5, 0xA3, 0x8C, 0x8F, 0xBF))
    if high and ce / high >= 0.25:
        return data.decode("cp1250", errors="replace"), "cp1250"
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _newlines(text: str) -> str:
    return (text.replace("\r\n", "\n").replace("\r", "\n").replace("\u2028", "\n").replace("\u2029", "\n\n")
            .replace("\f", "\n\n").replace("\x85", "\n"))


_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w.:-]+)|<\?xml[^>]+encoding\s*=\s*["']([\w.:-]+)""",
                         re.IGNORECASE)


def _markup_charset(data: bytes) -> str | None:
    m = _CHARSET_RE.search(data[:4096])
    return (m.group(1) or m.group(2)).decode("ascii", "replace") if m else None


# ------------------------------------------------------------------------------------------------- chaptering

def _chapters_from_blocks(blocks: list[Block], max_level: int = 3) -> tuple[list[_Ch], list[str], str]:
    """(chapters, preamble paragraphs, title candidate) from headings + paragraphs (see the module docstring)."""
    counts: dict[int, int] = {}
    for kind, level, _ in blocks:
        if kind == "h" and level <= max_level:
            counts[level] = counts.get(level, 0) + 1
    split = next((lv for lv in sorted(counts) if counts[lv] >= 2), None)
    chapters: list[_Ch] = []
    preamble: list[str] = []
    title = ""
    carry: list[str] = []
    for kind, level, text in blocks:
        if kind == "h" and split is not None and level == split:
            chapters.append(_Ch(text, lead=carry))
            carry = []
        elif kind == "h" and level <= max_level and (split is None or level < split):
            if not chapters and not title and not any(count_words(p) for p in preamble):
                title = text
            elif chapters:
                carry.append(text)
            else:
                preamble.append(text)
        elif chapters:
            chapters[-1].paras.append(text)
        else:
            preamble.append(text)
    if carry and chapters:
        chapters[-1].paras.extend(carry)
    if split is None and title:                 # one chapter under a single heading
        return [_Ch(title, preamble)], [], title
    return chapters, preamble, title


def _strip_title(title: str, paras: list[str]) -> list[str]:
    """Drop up to two leading short paragraphs that repeat the chapter title (or a part of it)."""
    key = _key(title)
    if not key:
        return paras
    i = dropped = 0
    while i < len(paras) and dropped < 2:
        pk = _key(paras[i])
        if not pk:
            i += 1
            continue
        if len(pk.split()) > 15:
            break
        if pk == key or f" {pk} " in f" {key} " or (f" {key} " in f" {pk} " and len(pk.split()) <= 12):
            i += 1
            dropped += 1
        else:
            break
    return paras[i:]


_TOC_NAMES = {"contents", "table of contents", "list of illustrations", "illustrations", "зміст", "содержание",
              "оглавление", "inhalt", "inhaltsverzeichnis", "table des matières", "sommaire", "índice", "indice",
              "contenido", "contenidos", "spis treści", "spis tresci"}


def _is_toc(ch: _Ch, title_keys: set[str]) -> bool:
    """A table of contents: named so, or text made mostly of chapter headings / titles of other chapters."""
    if _key(ch.title) in _TOC_NAMES:
        return True
    total = toc = 0
    for p in ch.paras:
        n = count_words(p)
        total += n
        k = _key(p)
        if k and (k in title_keys or k in _TOC_NAMES or _heading_match(p)):
            toc += n
    return len(ch.paras) >= 3 and toc >= 0.6 * total


def _finish(book: _Book, fallback_title: str) -> tuple[str, list[dict[str, str]]]:
    title = _tidy_title(book.title) or fallback_title
    title_keys = {_key(c.title) for c in book.chapters} - {""}
    chapters = [c for c in book.chapters if not _is_toc(c, title_keys)]
    if len(chapters) < len(book.chapters):
        book.preamble = [p for p in book.preamble if not (_heading_match(p) or _key(p) in title_keys | _TOC_NAMES)]
    # a title page listed as a chapter ("<book title>", "by <author>")
    while len(chapters) > 1 and _key(chapters[0].title) == _key(title) and \
            sum(count_words(p) for p in chapters[0].paras) < OPENING_WORDS:
        chapters.pop(0)
    pre = [p for p in (_clean(x) for x in book.preamble) if p]
    pre_words = sum(count_words(p) for p in pre)
    has_words = any(count_words(p) for ch in chapters for p in ch.paras + ch.lead)
    if pre_words >= OPENING_WORDS or (pre_words and not has_words):
        chapters.insert(0, _Ch("Opening" if has_words else title, pre))
    out: list[dict[str, str]] = []
    carry: list[str] = []
    carry_text = False
    for ch in chapters:
        ctitle = _tidy_title(ch.title)
        paras = [p for p in (_clean(x) for x in ch.paras) if p and count_words(p)]
        paras = _strip_title(ctitle, paras) if ctitle else paras
        lead = [p for p in (_clean(x) for x in ch.lead) if p and count_words(p)]
        if ch.part and sum(count_words(x) for x in paras) < PART_WORDS:
            carry = carry + lead + ([ctitle] if ctitle else []) + paras
            carry_text = carry_text or bool(paras)
            continue
        if not paras:
            continue
        out.append({"title": ctitle, "text": "\n\n".join(carry + lead + paras)})
        carry, carry_text = [], False
    if carry_text and out:                  # a trailing "Part" with a little text of its own: keep it
        out[-1]["text"] += "\n\n" + "\n\n".join(carry)
    for i, ch in enumerate(out, 1):
        if not ch["title"]:
            ch["title"] = title if len(out) == 1 else f"Chapter {i}"
    return title, out


# ------------------------------------------------------------------------------------- plain text: heading lines

_KW = (r"chapter|part|book|volume|розділ|глава|частина|книга|том|kapitel|teil|buch|chapitre|partie|livre|tome|"
       r"capítulo|capitulo|parte|libro|rozdział|rozdzial|część|czesc|księga|ksiega")
_PART_KW = {"part", "book", "volume", "частина", "книга", "том", "teil", "buch", "partie", "livre", "tome", "parte",
            "libro", "część", "czesc", "księga", "ksiega"}
_SINGLE = (r"prologue|epilogue|preface|foreword|introduction|afterword|interlude|пролог|епілог|эпилог|передмова|"
           r"післямова|вступ|предисловие|послесловие|prolog|epilog|vorwort|nachwort|einleitung|épilogue|préface|"
           r"avant-propos|prólogo|epílogo|prefacio|introducción|wstęp|przedmowa|posłowie")
_KW_RE = re.compile(rf"^({_KW})\b\.?\s*(.*)$", re.IGNORECASE)
_SINGLE_RE = re.compile(rf"^({_SINGLE})\b\s*(.*)$", re.IGNORECASE)
_NUMBER_WORDS = set((
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred first second third fourth fifth "
    "sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
    "eighteenth nineteenth twentieth thirtieth last final "
    "eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf "
    "un une deux trois quatre cinq sept huit neuf dix onze douze premier première dernier "
    "uno dos tres cuatro cinco seis siete ocho nueve diez once doce primero primera segundo segunda tercero tercera "
    "pierwszy pierwsza drugi druga trzeci trzecia czwarty czwarta piąty piąta szósty szósta siódmy siódma ósmy ósma "
    "dziewiąty dziewiąta dziesiąty dziesiąta ostatni ostatnia "
    "один одна два дві три чотири останній остання").split())
_SLAV_ORD = re.compile(
    r"(?:перш|друг|трет|четверт|п['’ʼ]?ят|шост|сьом|восьм|дев['’ʼ]?ят|десят|одинадцят|дванадцят|тринадцят|"
    r"чотирнадцят|шістнадцят|сімнадцят|вісімнадцят|двадцят|тридцят|сороков|перв|втор|шест|седьм|девят|"
    r"одиннадцат|двенадцат|двадцат|тридцат)[а-яіїєґё]*", re.IGNORECASE)
_SUBTITLE_END = ".,;:!?…»\"”"


def _number_token(word: str) -> bool:
    w = word.lower().strip(".:")
    if not w:
        return False
    if w.isdigit() and len(w) <= 4:
        return True
    if _is_roman(w):
        return True
    parts = re.split(r"-", w)
    return all(p in _NUMBER_WORDS or _SLAV_ORD.fullmatch(p) for p in parts if p) and any(parts)


def _heading_match(line: str) -> tuple[str, bool, bool] | None:
    """(kind "kw"/"single"/"roman", is_part, has_title_on_line) when ``line`` looks like a chapter heading."""
    s = line.strip()
    if not s or len(s) > 80:
        return None
    m = _KW_RE.match(s)
    if m:
        rest = m.group(2).strip()
        tokens = rest.split()
        if not tokens:
            return None
        n = 0
        if tokens[0].lower() == "the" and len(tokens) > 1:   # "Chapter the First"
            n = 1
        if not _number_token(re.sub(r"[.:)\-–—]+$", "", tokens[n])):
            return None
        n += 1
        if n < len(tokens) and _number_token(tokens[n].rstrip(".:")) and not _is_roman(tokens[n].rstrip(".:")):
            n += 1                                           # "twenty one", "двадцять перший"
        after = " ".join(tokens[n:]).lstrip(".:)-–— ").strip()
        if after and (after[0].islower() or len(after) > 70 or after[-1] in ",;"):
            return None
        return "kw", m.group(1).lower() in _PART_KW, bool(after)
    m = _SINGLE_RE.match(s)
    if m:
        after = m.group(2).lstrip(".:)-–— ").strip()
        if after and (after[0].islower() or len(after) > 70 or after[-1] in ",;"):
            return None
        return "single", False, bool(after)
    bare = s.rstrip(".")
    if bare.isupper() and _is_roman(bare):
        return "roman", False, False
    return None


def _is_subtitle(s: str, wrapped: bool) -> bool:
    if not s or len(s) > 70 or len(s.split()) > 12 or _heading_match(s):
        return False
    first = s[0]
    if not (first.isupper() or first.isdigit() or first in "\"'«“„‘"):
        return False
    if s[-1] in ",;":
        return False
    return wrapped or s[-1] not in _SUBTITLE_END


def _find_headings(lines: list[str], wrapped: bool) -> list[tuple[int, int, str, bool]]:
    """Heading lines as (first line, line after the heading, title, is_part); [] unless at least two are found."""
    found: dict[str, list[tuple[int, int, str, bool]]] = {"kw": [], "single": [], "roman": []}
    n = len(lines)
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s or len(s) > 80:
            continue
        hm = _heading_match(s)
        if not hm:
            continue
        kind, part, has_title = hm
        if wrapped and i > 0 and lines[i - 1].strip():
            continue
        end, title = i + 1, _tidy_title(s)
        if not has_title:
            j = i + 1
            if j < n and not lines[j].strip() and j + 1 < n:
                j += 1
            cand = lines[j].strip() if j < n else ""
            after_blank = j + 1 >= n or not lines[j + 1].strip()
            if cand and _is_subtitle(cand, wrapped) and (after_blank or not wrapped):
                end, title = j + 1, _join_title(title, _tidy_title(cand))
        if kind == "roman" and wrapped and end < n and lines[end].strip() and end == i + 1:
            continue                                         # a lone numeral must stand alone
        if wrapped and has_title and end < n and lines[end].strip():
            run = end
            while run < n and lines[run].strip():
                run += 1
            if run - end >= 2:
                continue                                     # the first line of a paragraph ("Book I. (Folio) ...")
            title, end = f"{title} {lines[end].strip()}", end + 1     # a title wrapped onto a second line
        found[kind].append((i, end, title, part))
    roman = found["roman"]
    if len(roman) >= 2:
        vals = [_roman_value(lines[i].strip().rstrip(".")) for i, _, _, _ in roman]
        steps = sum(1 for a, b in itertools.pairwise(vals) if b == a + 1)
        if steps < 0.6 * (len(vals) - 1) or vals[0] > 2:
            roman = []
    if len(found["kw"]) >= 2:
        heads = found["kw"] + found["single"]
    elif len(roman) >= 2:
        heads = roman + found["single"]
    elif len(found["kw"]) + len(found["single"]) >= 2:
        heads = found["kw"] + found["single"]
    else:
        return []
    heads.sort()
    out: list[tuple[int, int, str, bool]] = []
    for h in heads:
        if not out or h[0] >= out[-1][1]:
            out.append(h)
    return out if len(out) >= 2 else []


def _drop_toc_runs(lines: list[str], heads: list[tuple[int, int, str, bool]]) -> list[tuple[int, int, str, bool]]:
    """A table of contents: three or more headings in a row with (almost) no text between them. Its lines are blanked
    and its headings dropped; the text after its last entry (front matter) stays."""
    small = [count_words(" ".join(lines[heads[k][1]:heads[k + 1][0]])) < 20 for k in range(len(heads) - 1)]
    keep = [True] * len(heads)
    k = 0
    while k < len(small):
        if not small[k]:
            k += 1
            continue
        m = k
        while m < len(small) and small[m]:
            m += 1
        if m - k + 1 >= 3:                                   # heads k..m
            first = heads[k][0]
            for i in range(first - 1, max(-1, first - 12), -1):    # its "Contents" header and untitled entries
                s = lines[i].strip()
                if len(s) > 80 or len(s.split()) > 12:
                    break
                if _key(s) in _TOC_NAMES:
                    first = i
                    break
            for i in range(first, heads[m][1]):
                lines[i] = ""
            for i in range(k, m + 1):
                keep[i] = False
        k = m + 1
    return [h for h, ok in zip(heads, keep) if ok]


def _join_lines(parts: list[str], dehyphenate: bool) -> str:
    """Join wrapped lines of one paragraph, healing words broken at a line end."""
    out: list[str] = []
    for s in parts:
        if out:
            prev = out[-1]
            if prev.endswith("\u00ad"):
                out[-1] = prev[:-1]
            elif (prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and s[:1].isalpha()):
                if dehyphenate and s[:1].islower():
                    out[-1] = prev[:-1]
            else:
                out.append(" ")
        out.append(s)
    return "".join(out)


def _line_width(lines: list[str]) -> int:
    lens = sorted(len(s.strip()) for s in lines if s.strip())
    return lens[int(len(lens) * 0.9)] if lens else 0


def _is_wrapped(lines: list[str]) -> bool:
    """Hard-wrapped text: lines of a fixed width (~60-100 characters), most of them close to it."""
    nonblank = [s.strip() for s in lines if s.strip()]
    if len(nonblank) < 8:
        return False
    width = _line_width(lines)
    if not 30 <= width <= 100:
        return False
    near = sum(1 for s in nonblank if width * 0.7 <= len(s) <= width + 2)
    return near >= 0.35 * len(nonblank)


def _lines_to_paras(lines: list[str], wrapped: bool, width: int, indent_breaks: bool) -> list[str]:
    if not wrapped:
        return [s.strip() for s in lines if s.strip()]
    paras: list[str] = []
    cur: list[str] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            if cur:
                paras.append(_join_lines(cur, False))
                cur = []
            continue
        if cur and ((indent_breaks and raw[:1] in " \t") or len(cur[-1]) < 0.6 * width):
            paras.append(_join_lines(cur, False))
            cur = []
        cur.append(s)
    if cur:
        paras.append(_join_lines(cur, False))
    return paras


# _word_ / 1_st_ (Gutenberg italics), also across a line break but not across paragraphs
_UNDERSCORE_ITALICS = re.compile(r"_(?=[^\s_])((?:[^_\n]|\n(?![ \t]*\n)){1,300}?)(?<=\S)_")
_GUTENBERG_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE)
_GUTENBERG_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.MULTILINE | re.IGNORECASE)


def _plain_text_book(text: str) -> _Book:
    book = _Book()
    text = _newlines(text)
    start = _GUTENBERG_START.search(text)
    if start:
        m = re.search(r"^Title:\s*(.+)$", text[:start.start()], re.MULTILINE)
        if m:
            book.title = m.group(1).strip()
        text = text[start.end():]
        end = _GUTENBERG_END.search(text)
        if end:
            text = text[:end.start()]
    lines = _UNDERSCORE_ITALICS.sub(r"\1", text).split("\n")
    wrapped = _is_wrapped(lines)
    width = _line_width(lines)
    indented = sum(1 for s in lines if s.strip() and s[:1] in " \t")
    nonblank = sum(1 for s in lines if s.strip())
    indent_breaks = wrapped and 0.03 * nonblank <= indented <= 0.5 * nonblank
    heads = _find_headings(lines, wrapped)
    if heads:
        heads = _drop_toc_runs(lines, heads)
        if len(heads) < 2:
            heads = _find_headings(lines, wrapped) if not heads else heads
            heads = heads if len(heads) >= 2 else []
    if not heads:
        book.chapters = [_Ch("", _lines_to_paras(lines, wrapped, width, indent_breaks))]
        if count_words(text) > 5000:
            book.warnings.append("No chapter headings were found, so the text is one chapter.")
        return book
    book.preamble = _lines_to_paras(lines[:heads[0][0]], wrapped, width, indent_breaks)
    first = next((x.strip() for x in book.preamble if x.strip()), "")
    if not book.title and first and len(first) <= 80 and len(first.split()) <= 12 and \
            sum(count_words(x) for x in book.preamble) < OPENING_WORDS:
        book.title = first                                   # a short title page: its first line names the book
    for k, (_, end, title, part) in enumerate(heads):
        stop = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        book.chapters.append(_Ch(title, _lines_to_paras(lines[end:stop], wrapped, width, indent_breaks), part))
    return book


# -------------------------------------------------------------------------------------------------- markdown

_MD_INLINE = (
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),                       # images
    (re.compile(r"!\[[^\]]*\]\[[^\]]*\]"), ""),
    (re.compile(r"\[\^[^\]]+\]"), ""),                                # footnote references
    (re.compile(r"\[([^\]]*)\]\([^)]*\)"), r"\1"),                    # links -> their text
    (re.compile(r"\[([^\]]+)\]\[[^\]]*\]"), r"\1"),
    (re.compile(r"<(?:https?|ftp|mailto):[^>]+>"), ""),               # autolinks
    (re.compile(r"<!--.*?-->"), ""),
    (re.compile(r"</?[A-Za-z][^>]*>"), ""),                           # inline HTML
    (re.compile(r"``?([^`]+)``?"), r"\1"),                            # inline code
    (re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1"), r"\2"),
    (re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w_])_(?=\S)(.+?)(?<=\S)_(?![\w_])"), r"\1"),
    (re.compile(r"~~(.+?)~~"), r"\1"),
    (re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])"), r"\1"),
)


def _md_inline(s: str) -> str:
    for rx, rep in _MD_INLINE:
        s = rx.sub(rep, s)
    return html.unescape(s)


def _markdown_book(text: str) -> _Book:
    book = _Book()
    lines = _newlines(text).split("\n")
    if lines and lines[0].strip() == "---":                           # YAML front matter
        for k in range(1, min(len(lines), 200)):
            if lines[k].strip() in ("---", "..."):
                for fm in lines[1:k]:
                    m = re.match(r"^title:\s*(.+)$", fm.strip(), re.IGNORECASE)
                    if m:
                        book.title = m.group(1).strip().strip("\"'")
                lines = lines[k + 1:]
                break
    blocks: list[Block] = []
    para: list[str] = []
    fence = ""
    in_comment = False

    def flush() -> None:
        if para:
            blocks.append(("p", 0, _md_inline(" ".join(para))))
            para.clear()

    for line in lines:
        s = line.strip()
        if in_comment:
            if "-->" in s:
                in_comment = False
            continue
        if fence:
            if s.startswith(fence):
                fence = ""
            continue
        fm = re.match(r"^(`{3,}|~{3,})", s)
        if fm:
            flush()
            fence = fm.group(1)
            continue
        if s.startswith("<!--") and "-->" not in s:
            flush()
            in_comment = True
            continue
        if not s:
            flush()
            continue
        hm = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", s)
        if hm:
            flush()
            blocks.append(("h", len(hm.group(1)), _md_inline(hm.group(2))))
            continue
        if para and re.fullmatch(r"=+", s):
            blocks.append(("h", 1, _md_inline(" ".join(para))))
            para.clear()
            continue
        if para and re.fullmatch(r"-+", s):
            blocks.append(("h", 2, _md_inline(" ".join(para))))
            para.clear()
            continue
        if re.fullmatch(r"([-*_])(\s*\1){2,}", s):                    # horizontal rule
            flush()
            continue
        if re.match(r"^\[\^?[^\]]+\]:\s*\S", s):                      # reference / footnote definitions
            flush()
            continue
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", s):   # table separator row
            continue
        if s.startswith("|"):
            flush()
            cells = [c.strip() for c in s.strip("|").split("|")]
            blocks.append(("p", 0, _md_inline(", ".join(c for c in cells if c))))
            continue
        s = re.sub(r"^(>\s?)+", "", s).strip()
        lm = re.match(r"^(?:[-*+]|\d{1,3}[.)])\s+(.*)$", s)
        if lm:
            flush()
            s = lm.group(1)
        if s:
            para.append(s)
    flush()
    chapters, preamble, title = _chapters_from_blocks(blocks)
    book.title = book.title or title
    book.chapters, book.preamble = chapters, preamble
    return book


# ------------------------------------------------------------------------------------------------------ HTML

_BLOCK_TAGS = {"p", "div", "section", "article", "main", "li", "ul", "ol", "dl", "dt", "dd", "blockquote", "pre",
               "table", "tr", "header", "footer", "figure", "figcaption", "hr", "body", "address", "center",
               "caption", "details", "summary", "form", "fieldset", "legend", "tbody", "thead", "tfoot", "aside",
               "hgroup", "html"}
_SKIP_TAGS = {"script", "style", "nav", "noscript", "template", "svg", "math", "iframe", "object", "button", "select",
              "textarea", "rt", "rp", "audio", "video", "canvas", "map", "figcaption"}
_VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "area", "base", "col", "embed", "source", "track", "wbr"}
# Not read aloud: page-break markers, note references and footnote/endnote bodies (EPUB 3 semantics, ARIA roles, and
# the class names Project Gutenberg and common converters use, on inline/paragraph elements only).
_NOISE = re.compile(r"\b(?:doc-)?(?:pagebreak|noteref|footnotes?|endnotes?|rearnotes?)\b", re.IGNORECASE)
_NOISE_CLASS = re.compile(r"^(?:foot|end|rear)notes?$|^page-?(?:num|number|break)$|^noteref$|^fnref$|^footnote-?ref$|"
                          r"^caption$|^pg-boilerplate$", re.IGNORECASE)
_NOISE_IDS = {"pg-header", "pg-footer", "pg-machine-header", "pg-start-separator", "pg-end-separator"}  # Gutenberg
_NOISE_CLASS_TAGS = {"a", "span", "sup", "sub", "aside", "p", "div", "small"}


class _HtmlBlocks(HTMLParser):
    """HTML/XHTML -> blocks (headings h1-h6, paragraphs); element ids map to the index of the block they start."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.anchors: dict[str, int] = {}
        self.title = ""
        self._buf: list[str] = []
        self._heading = 0
        self._skip = ""
        self._skip_depth = 0
        self._in_title = False
        self._title: list[str] = []

    def _flush(self) -> None:
        text = _clean("".join(self._buf))
        self._buf = []
        if text:
            self.blocks.append(("h", self._heading, text) if self._heading else ("p", 0, text))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._skip:
            if tag == self._skip:
                self._skip_depth += 1
            return
        a = {k.lower(): (v or "") for k, v in attrs}
        noisy = (tag not in ("body", "html") and any(_NOISE.search(a.get(k, "")) for k in ("epub:type", "role"))
                 or tag in _NOISE_CLASS_TAGS and any(_NOISE_CLASS.match(c) for c in a.get("class", "").split())
                 or "hidden" in a or a.get("id", "") in _NOISE_IDS)
        if tag not in _VOID_TAGS and (tag in _SKIP_TAGS or noisy):
            self._skip, self._skip_depth = tag, 1
            return
        for anchor in (a.get("id"), a.get("name") if tag == "a" else None):
            if anchor and anchor not in self.anchors:
                self.anchors[anchor] = len(self.blocks)
        if tag == "title":
            self._in_title = True
        elif len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self._flush()
            self._heading = int(tag[1])
        elif tag == "br":
            if self._heading:
                self._buf.append(" ")
            else:
                self._flush()
        elif tag in _BLOCK_TAGS:
            if not self._heading:
                self._flush()
        elif tag in ("td", "th"):
            self._buf.append(" ")
        elif tag == "img":
            alt = a.get("alt", "").strip()
            if self._heading or (len(alt) == 1 and alt.isalpha()):   # a heading or a drop-cap letter drawn as an image
                self._buf.append(alt)

    def handle_endtag(self, tag: str) -> None:
        if self._skip:
            if tag == self._skip:
                self._skip_depth -= 1
                if self._skip_depth <= 0:
                    self._skip = ""
            return
        if tag == "title":
            self._in_title = False
            self.title = _clean("".join(self._title))
        elif len(tag) == 2 and tag[0] == "h" and tag[1] in "123456":
            self._flush()
            self._heading = 0
        elif tag in _BLOCK_TAGS and not self._heading:
            self._flush()
        elif tag in ("td", "th"):
            self._buf.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        if self._in_title:
            self._title.append(data)
        else:
            self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()


def _html_blocks(text: str) -> _HtmlBlocks:
    p = _HtmlBlocks()
    p.feed(text)
    p.close()
    return p


def _html_book(text: str) -> _Book:
    p = _html_blocks(text)
    chapters, preamble, title = _chapters_from_blocks(p.blocks)
    page_title = re.sub(r"\s+\|\s+[^|]{1,40}$", "", p.title)        # "Book | Site name"
    return _Book(title=page_title or title, chapters=chapters, preamble=preamble)


# ------------------------------------------------------------------------------------------------- XML helpers

def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _find(elem: ET.Element, *path: str) -> ET.Element | None:
    """First descendant chain matching local names (``_find(root, "description", "title-info")``)."""
    cur: ET.Element | None = elem
    for name in path:
        if cur is None:
            return None
        cur = next((c for c in cur.iter() if c is not cur and _local(c.tag) == name), None)
    return cur


def _all(elem: ET.Element, name: str) -> Iterator[ET.Element]:
    return (c for c in elem.iter() if _local(c.tag) == name)


def _text_of(elem: ET.Element | None) -> str:
    return _clean("".join(elem.itertext())) if elem is not None else ""


def _attr(elem: ET.Element, name: str) -> str:
    """Attribute by local name (any namespace)."""
    for k, v in elem.attrib.items():
        if _local(k) == name:
            return v
    return ""


def _xml(data: bytes) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError:
        # HTML named entities (&nbsp; ...) are not XML: replace them and try again
        text, _ = _decode(data, _markup_charset(data))
        text = re.sub(r"<\?xml[^>]*\?>", "", text, count=1)

        def entity(m: re.Match[str]) -> str:
            name = m.group(1)
            if name in ("amp", "lt", "gt", "quot", "apos"):
                return m.group(0)
            return html.unescape(m.group(0)) if (name + ";") in _HTML5 else ""
        return ET.fromstring(re.sub(r"&([A-Za-z][A-Za-z0-9]*);", entity, text))


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    info = zf.getinfo(name)
    if info.file_size > MAX_MEMBER:
        raise EngineError("bad_request", f"The file is too large to read ({name} unpacks to "
                                         f"{info.file_size // (1024 * 1024)} MB).")
    return zf.read(info)


def _zip_names(zf: zipfile.ZipFile) -> dict[str, str]:
    """Lower-case (and unquoted) member name -> real name, for tolerant lookups."""
    out: dict[str, str] = {}
    for n in zf.namelist():
        out.setdefault(n.lower(), n)
        out.setdefault(unquote(n).lower(), n)
    return out


# ------------------------------------------------------------------------------------------------------ EPUB

@dataclass
class _TocNode:
    title: str
    path: str
    frag: str
    kids: list[_TocNode] = field(default_factory=list)


class _NavParser(HTMLParser):
    """Every ``<nav>`` of an EPUB 3 navigation document as a tree; :meth:`toc` picks the table of contents."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.navs: list[tuple[str, list[_TocNode]]] = []
        self._root: list[_TocNode] | None = None
        self._root_depth = 0
        self._depth = 0
        self._lists: list[list[_TocNode]] = []
        self._pushed: list[bool] = []
        self._li: list[_TocNode] = []
        self._collect = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "nav":
            self._depth += 1
            if self._root is None:
                self._root, self._root_depth = [], self._depth
                self.navs.append(((a.get("epub:type", "") + " " + a.get("role", "")).lower(), self._root))
                self._lists, self._pushed, self._li = [self._root], [], []
            return
        if self._root is None:
            return
        if tag in ("ol", "ul"):
            pushed = bool(self._li)
            if pushed:
                self._lists.append(self._li[-1].kids)
            self._pushed.append(pushed)
        elif tag == "li":
            node = _TocNode("", "", "")
            self._lists[-1].append(node)
            self._li.append(node)
        elif tag == "a" and self._li:
            self._li[-1].path = a.get("href", "")
            self._collect = True
        elif tag == "span" and self._li and not self._li[-1].path:
            self._collect = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "nav":
            if self._root is not None and self._depth == self._root_depth:
                self._root = None
            self._depth -= 1
            return
        if self._root is None:
            return
        if tag in ("ol", "ul"):
            if self._pushed and self._pushed.pop():
                self._lists.pop()
        elif tag == "li":
            if self._li:
                self._li.pop()
        elif tag in ("a", "span"):
            self._collect = False

    def handle_data(self, data: str) -> None:
        if self._root is not None and self._collect and self._li:
            self._li[-1].title += data

    def toc(self) -> list[_TocNode]:
        for kind, root in self.navs:
            if "toc" in kind and root:
                return root
        return next((root for kind, root in self.navs if root and "landmarks" not in kind and "page-list" not in kind),
                    [])


def _resolve(base_dir: str, href: str) -> tuple[str, str]:
    href, _, frag = href.partition("#")
    if not href:
        return "", unquote(frag)
    return posixpath.normpath(posixpath.join(base_dir, unquote(href))).lstrip("/"), unquote(frag)


def _ncx_nodes(elem: ET.Element, base: str) -> list[_TocNode]:
    out: list[_TocNode] = []
    for np_ in (c for c in elem if _local(c.tag) == "navPoint"):
        label = _find(np_, "navLabel")
        content = next((c for c in np_ if _local(c.tag) == "content"), None)
        path, frag = _resolve(base, content.get("src", "")) if content is not None else ("", "")
        node = _TocNode(_text_of(label), path, frag)
        node.kids = _ncx_nodes(np_, base)
        out.append(node)
    return out


def _fix_nodes(nodes: list[_TocNode], base: str) -> None:
    """Nav hrefs (relative to the nav document) -> archive paths + fragments; titles cleaned."""
    for n in nodes:
        n.title = _clean(n.title)
        if n.path:
            n.path, n.frag = _resolve(base, n.path)
        _fix_nodes(n.kids, base)


def _epub_book(path: str) -> _Book:
    book = _Book()
    with zipfile.ZipFile(path) as zf:
        names = _zip_names(zf)

        def get(name: str) -> bytes | None:
            real = names.get(name.lower()) or names.get(unquote(name).lower())
            return _read_member(zf, real) if real else None

        enc = get("META-INF/encryption.xml")
        if enc:
            refs = [_attr(c, "URI") for c in _all(_xml(enc), "CipherReference")]
            if any(re.search(r"\.(x?html?|xml)$", r, re.IGNORECASE) for r in refs):
                raise EngineError("bad_request", "This EPUB is protected by DRM (its text is encrypted). Mimvox can "
                                                 "only read EPUB files without DRM.")
        container = get("META-INF/container.xml")
        rootfile = next((r.get("full-path", "") for r in _all(_xml(container), "rootfile")), "") if container else ""
        if not rootfile:
            rootfile = next((n for n in zf.namelist() if n.lower().endswith(".opf")), "")
        opf_data = get(rootfile) if rootfile else None
        if not opf_data:
            raise EngineError("bad_request", "This EPUB is damaged: its package file (OPF) is missing.")
        opf = _xml(opf_data)
        base = posixpath.dirname(rootfile)
        book.title = _text_of(_find(opf, "metadata", "title"))
        manifest: dict[str, tuple[str, str, str]] = {}
        for item in _all(opf, "item"):
            manifest[item.get("id", "")] = (_resolve(base, item.get("href", ""))[0], item.get("media-type", ""),
                                            item.get("properties", ""))
        spine_el = _find(opf, "spine")
        spine: list[str] = []
        nav_path = next((p for p, _, props in manifest.values() if "nav" in props.split()), "")
        if spine_el is not None:
            for ref in (c for c in spine_el if _local(c.tag) == "itemref"):
                it = manifest.get(ref.get("idref", ""))
                if it and ("html" in it[1] or it[0].lower().endswith((".html", ".htm", ".xhtml"))) \
                        and it[0] != nav_path:
                    spine.append(it[0])
        # table of contents: EPUB 3 nav, else the EPUB 2 NCX
        toc: list[_TocNode] = []
        if nav_path and (data := get(nav_path)):
            nav = _NavParser()
            nav.feed(_decode(data, _markup_charset(data))[0])
            nav.close()
            toc = nav.toc()
            _fix_nodes(toc, posixpath.dirname(nav_path))
        if not toc:
            ncx_id = spine_el.get("toc", "") if spine_el is not None else ""
            ncx_path = manifest.get(ncx_id, ("",))[0] or next(
                (p for p, mt, _ in manifest.values() if mt == "application/x-dtbncx+xml"), "")
            if ncx_path and (data := get(ncx_path)):
                nav_map = _find(_xml(data), "navMap")
                if nav_map is not None:
                    toc = _ncx_nodes(nav_map, posixpath.dirname(ncx_path))
        # the spine documents, concatenated as blocks
        blocks: list[Block] = []
        start: dict[str, int] = {}
        anchors: dict[str, dict[str, int]] = {}
        for doc in spine:
            data = get(doc)
            if data is None:
                book.warnings.append(f"A part of the book is missing from the EPUB ({posixpath.basename(doc)}).")
                continue
            p = _html_blocks(_decode(data, _markup_charset(data))[0])
            start[doc] = len(blocks)
            anchors[doc] = p.anchors
            blocks.extend(p.blocks)
    if not blocks:
        return book

    def pos(n: _TocNode) -> int | None:
        if n.path not in start:
            return None
        return start[n.path] + anchors[n.path].get(n.frag, 0) if n.frag else start[n.path]

    # Chapters: the top level of the TOC; a node whose children are separate documents (a part split into chapter
    # files, or a lone root entry) is replaced by itself + its children.
    picked: list[tuple[int, str, bool]] = []

    def pick(nodes: list[_TocNode], depth: int) -> None:
        for n in nodes:
            kids = [k for k in n.kids if pos(k) is not None]
            spread = any(k.path != n.path for k in kids) or (len(nodes) == 1 and len({pos(k) for k in kids}) >= 2)
            p = pos(n)
            if kids and depth < 3 and spread:
                if p is not None:
                    picked.append((p, n.title, True))
                pick(n.kids, depth + 1)
            elif p is not None:
                picked.append((p, n.title, False))

    pick(toc, 0)
    seen: dict[int, tuple[int, str, bool]] = {}
    for item in picked:
        seen.setdefault(item[0], item)
    cuts = sorted(seen.values())
    if len(cuts) < 2:
        chapters, preamble, title = _chapters_from_blocks(blocks)
        book.title = book.title or title
        book.chapters, book.preamble = chapters, preamble
        if len(chapters) < 2 and toc:
            book.warnings.append("The table of contents of this EPUB could not be matched to its text, so chapters "
                                 "come from its headings.")
        return book
    book.preamble = [t for _, _, t in blocks[:cuts[0][0]]]
    for k, (p, title, part) in enumerate(cuts):
        stop = cuts[k + 1][0] if k + 1 < len(cuts) else len(blocks)
        book.chapters.append(_Ch(title, [t for _, _, t in blocks[p:stop]], part))
    return book


# ------------------------------------------------------------------------------------------------------ DOCX

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_book(path: str) -> _Book:
    book = _Book()
    with zipfile.ZipFile(path) as zf:
        names = _zip_names(zf)
        if "word/document.xml" not in names:
            raise EngineError("bad_request", "This is not a Word document (word/document.xml is missing).")
        doc = _xml(_read_member(zf, names["word/document.xml"]))
        styles: dict[str, tuple[str, int | None, str]] = {}
        if "word/styles.xml" in names:
            for st in _all(_xml(_read_member(zf, names["word/styles.xml"])), "style"):
                sid = st.get(f"{_W}styleId", "")
                name_el = st.find(f"{_W}name")
                based = st.find(f"{_W}basedOn")
                lvl_el = st.find(f"{_W}pPr/{_W}outlineLvl")
                lvl = int(lvl_el.get(f"{_W}val", "9")) if lvl_el is not None else None
                styles[sid] = ((name_el.get(f"{_W}val", "") if name_el is not None else "").lower(), lvl,
                               based.get(f"{_W}val", "") if based is not None else "")
        if "docprops/core.xml" in names:
            book.title = _text_of(_find(_xml(_read_member(zf, names["docprops/core.xml"])), "title"))

    def style_level(sid: str) -> int | None:
        for _ in range(6):
            name, lvl, based = styles.get(sid, ("", None, ""))
            name = name or sid.lower()
            if name.startswith(("toc", "table of contents")):
                return -1
            if name == "title":
                return 0
            m = re.fullmatch(r"heading\s*(\d)", name)
            if m:
                return int(m.group(1))
            if lvl is not None:
                return lvl + 1 if lvl < 9 else None
            if not based:
                return None
            sid = based
        return None

    def runs(elem: ET.Element, out: list[str]) -> None:
        for c in elem:
            tag = c.tag
            if tag == f"{_W}t":
                out.append(c.text or "")
            elif tag in (f"{_W}tab", f"{_W}ptab"):
                out.append(" ")
            elif tag in (f"{_W}br", f"{_W}cr"):
                out.append(" " if c.get(f"{_W}type") in ("page", "column") else "\n")
            elif tag == f"{_W}noBreakHyphen":
                out.append("-")
            elif tag in (f"{_W}del", f"{_W}drawing", f"{_W}pict", f"{_W}footnoteReference", f"{_W}endnoteReference",
                         f"{_W}commentReference", f"{_W}instrText", f"{_W}delText", f"{_W}object", f"{_W}pPr",
                         f"{_W}rPr") or _local(tag) == "AlternateContent":
                continue
            else:
                runs(c, out)

    blocks: list[Block] = []

    def walk(elem: ET.Element) -> None:
        for c in elem:
            if c.tag == f"{_W}p":
                ppr = c.find(f"{_W}pPr")
                level: int | None = None
                if ppr is not None:
                    ps = ppr.find(f"{_W}pStyle")
                    if ps is not None:
                        level = style_level(ps.get(f"{_W}val", ""))
                    ol = ppr.find(f"{_W}outlineLvl")
                    if ol is not None and level is None:
                        v = int(ol.get(f"{_W}val", "9"))
                        level = v + 1 if v < 9 else None
                if level == -1:
                    continue
                out: list[str] = []
                runs(c, out)
                pieces = [p for p in (_clean(x) for x in "".join(out).split("\n")) if p]
                if level is not None and pieces:
                    blocks.append(("h", level, " ".join(pieces)))
                else:
                    blocks.extend(("p", 0, p) for p in pieces)
            elif c.tag in (f"{_W}sectPr",) or _local(c.tag) == "AlternateContent":
                continue
            else:
                walk(c)

    body = doc.find(f"{_W}body")
    walk(body if body is not None else doc)
    chapters, preamble, title = _chapters_from_blocks(blocks)
    book.title = book.title or title
    book.chapters, book.preamble = chapters, preamble
    return book


# ------------------------------------------------------------------------------------------------------- ODT

_ODT_SKIP = {"tracked-changes", "annotation", "annotation-end", "note", "table-of-content", "alphabetical-index",
             "illustration-index", "table-index", "object-index", "user-index", "bibliography", "sequence-decls",
             "variable-decls", "user-field-decls", "forms"}


def _odt_book(path: str) -> _Book:
    book = _Book()
    with zipfile.ZipFile(path) as zf:
        names = _zip_names(zf)
        if "content.xml" not in names:
            raise EngineError("bad_request", "This is not an OpenDocument text file (content.xml is missing).")
        content = _xml(_read_member(zf, names["content.xml"]))
        if "meta.xml" in names:
            book.title = _text_of(_find(_xml(_read_member(zf, names["meta.xml"])), "meta", "title"))
        styles_root = _xml(_read_member(zf, names["styles.xml"])) if "styles.xml" in names else None
    parents: dict[str, str] = {}
    for root in (content, styles_root):
        if root is not None:
            for st in _all(root, "style"):
                parents[_attr(st, "name")] = _attr(st, "parent-style-name")

    def para_level(style: str) -> int | None:
        for _ in range(6):
            s = style.replace("_20_", " ").lower()
            if s == "title":
                return 0
            m = re.fullmatch(r"heading\s*(\d)", s)
            if m:
                return int(m.group(1))
            style = parents.get(style, "")
            if not style:
                return None
        return None

    def inline(elem: ET.Element, out: list[str]) -> None:
        if elem.text:
            out.append(elem.text)
        for c in elem:
            name = _local(c.tag)
            if name == "s":
                out.append(" " * int(_attr(c, "c") or "1"))
            elif name == "tab":
                out.append(" ")
            elif name == "line-break":
                out.append("\n")
            elif name not in _ODT_SKIP and name not in ("frame", "image", "bookmark-ref"):
                inline(c, out)
            if c.tail:
                out.append(c.tail)

    blocks: list[Block] = []

    def walk(elem: ET.Element) -> None:
        for c in elem:
            name = _local(c.tag)
            if name in _ODT_SKIP:
                continue
            if name in ("h", "p"):
                out: list[str] = []
                inline(c, out)
                level: int | None = None
                if name == "h":
                    level = int(_attr(c, "outline-level") or "1")
                else:
                    level = para_level(_attr(c, "style-name"))
                pieces = [p for p in (_clean(x) for x in "".join(out).split("\n")) if p]
                if level is not None and pieces:
                    blocks.append(("h", level, " ".join(pieces)))
                else:
                    blocks.extend(("p", 0, p) for p in pieces)
            else:
                walk(c)

    body = _find(content, "body", "text")
    walk(body if body is not None else content)
    chapters, preamble, title = _chapters_from_blocks(blocks)
    book.title = book.title or title
    book.chapters, book.preamble = chapters, preamble
    return book


# -------------------------------------------------------------------------------------------------------- FB2

_FB2_INLINE_SKIP = {"image", "binary"}


def _fb2_inline(elem: ET.Element) -> str:
    out: list[str] = []

    def rec(e: ET.Element) -> None:
        if e.text:
            out.append(e.text)
        for c in e:
            name = _local(c.tag)
            if not (name in _FB2_INLINE_SKIP or (name == "a" and _attr(c, "type") == "note")):
                rec(c)
            if c.tail:
                out.append(c.tail)
    rec(elem)
    return _clean("".join(out))


def _fb2_paras(children: Iterable[ET.Element], out: list[str]) -> None:
    """Paragraph-level content of a section / epigraph / cite / poem (nested sections excluded)."""
    for c in children:
        name = _local(c.tag)
        if name in ("p", "subtitle", "v", "text-author", "td", "th"):
            text = _fb2_inline(c)
            if text:
                out.append(text)
        elif name in ("poem", "stanza", "epigraph", "cite", "title", "table", "tr", "annotation"):
            _fb2_paras(list(c), out)


def _fb2_title(sec: ET.Element) -> str:
    t = next((c for c in sec if _local(c.tag) == "title"), None)
    if t is None:
        return ""
    parts = [x for x in (_fb2_inline(p) for p in t if _local(p.tag) == "p") if x]
    title = ""
    for p in parts:
        title = _join_title(title, p) if title else p
    return title


def _fb2_book(path: str) -> _Book:
    book = _Book()
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith(".fb2")), None)
            if not member:
                raise EngineError("bad_request", "This ZIP file does not contain an .fb2 book.")
            data = _read_member(zf, member)
    else:
        with open(path, "rb") as f:
            data = f.read()
    root = _xml(data)
    book.title = _text_of(_find(root, "description", "title-info", "book-title"))
    bodies = [b for b in root if _local(b.tag) == "body"]
    main = [b for b in bodies if not b.get("name")] or bodies[:1]

    def section(sec: ET.Element, depth: int) -> None:
        title = _fb2_title(sec)
        kids = [c for c in sec if _local(c.tag) == "section"]
        if kids and depth < 4:
            ch = _Ch(title, part=True)
            book.chapters.append(ch)
            target = ch
            for c in sec:
                name = _local(c.tag)
                if name == "section":
                    section(c, depth + 1)
                    target = book.chapters[-1]
                elif name != "title":
                    _fb2_paras([c], target.paras)
        else:
            paras: list[str] = []
            for c in sec:
                name = _local(c.tag)
                if name == "section":                     # too deep: inline the sub-section with its title
                    sub = _fb2_title(c)
                    if sub:
                        paras.append(sub)
                    _fb2_paras(list(c), paras)
                elif name != "title":
                    _fb2_paras([c], paras)
            book.chapters.append(_Ch(title, paras))

    for body in main:
        for c in body:
            name = _local(c.tag)
            if name == "section":
                section(c, 0)
            elif name == "epigraph":
                _fb2_paras(list(c), book.preamble if not book.chapters else book.chapters[-1].paras)
    return book


# -------------------------------------------------------------------------------------------------------- RTF

_RTF_TOKEN = re.compile(r"\\([a-zA-Z]{1,32})(-?\d{1,10})? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|[\r\n]+|"
                        r"([^\\{}\r\n]+)")
_RTF_SKIP = {"colortbl", "pict", "object", "header", "headerl", "headerr", "headerf", "footer", "footerl", "footerr",
             "footerf", "footnote", "fldinst", "xe", "tc", "txe", "themedata", "colorschememapping", "latentstyles",
             "datastore", "listtable", "listoverridetable", "rsidtbl", "generator", "xmlnstbl", "mmathPr",
             "nonshppict", "private", "revtbl", "filetbl", "pgdsctbl", "operator", "bkmkstart", "bkmkend", "shp",
             "shpinst", "sp", "annotation", "atnid", "atnauthor", "comment", "objdata", "pn", "pntext", "listtext",
             "fonttbl_skip", "author", "company", "keywords", "subject", "doccomm", "userprops", "template"}
_RTF_CHARS = {"tab": " ", "emdash": "—", "endash": "–", "lquote": "‘", "rquote": "’", "ldblquote": "“",
              "rdblquote": "”", "bullet": "•", "emspace": " ", "enspace": " ", "qmspace": " ", "cell": " "}
_RTF_BREAKS = {"par", "line", "row", "sect", "page", "softline"}
_CHARSET_CP = {0: None, 1: None, 2: "cp1252", 77: "mac_roman", 128: "cp932", 129: "cp949", 130: "johab",
               134: "gbk", 136: "big5", 161: "cp1253", 162: "cp1254", 163: "cp1258", 177: "cp1255", 178: "cp1256",
               186: "cp1257", 204: "cp1251", 222: "cp874", 238: "cp1250", 255: "cp437"}


@dataclass
class _RtfState:
    skip: bool = False
    dest: str = ""                 # "", "fonttbl", "font", "stylesheet", "style", "info", "title"
    uc: int = 1
    cp: str = "cp1252"
    style: int = 0
    outline: int | None = None
    entry: dict[str, Any] | None = None


def _rtf_book(data: bytes) -> _Book:
    if not data.lstrip().startswith(b"{\\rtf"):
        raise EngineError("bad_request", "This is not an RTF file (it does not start with {\\rtf).")
    src = data.decode("latin-1")
    book = _Book()
    default_cp = "cp1252"
    fonts: dict[int, str] = {}
    styles: dict[int, tuple[str, int | None]] = {}
    deff = 0
    st = _RtfState()
    stack: list[_RtfState] = []
    pending = bytearray()
    para: list[str] = []
    blocks: list[Block] = []
    title: list[str] = []
    skip_chars = 0
    high_surrogate = 0
    star = False

    def flush_bytes() -> None:
        if pending:
            try:
                text = pending.decode(st.cp, errors="replace")
            except LookupError:
                text = pending.decode(default_cp, errors="replace")
            pending.clear()
            emit(text)

    def emit(text: str) -> None:
        if st.skip:
            return
        if st.dest == "":
            para.append(text)
        elif st.dest == "title":
            title.append(text)
        elif st.dest in ("font", "style") and st.entry is not None:
            st.entry["name"] = st.entry.get("name", "") + text

    def end_para() -> None:
        text = _clean("".join(para))
        para.clear()
        if not text:
            return
        level = st.outline
        name, slevel = styles.get(st.style, ("", None))
        if level is None:
            if name == "title":
                level = -1
            elif (m := re.fullmatch(r"heading\s*(\d)", name)):
                level = int(m.group(1)) - 1
            else:
                level = slevel
        if level is not None and 0 <= level + 1 <= 6:
            blocks.append(("h", level + 1, text))
        else:
            blocks.append(("p", 0, text))

    pos, n = 0, len(src)
    while pos < n:
        m = _RTF_TOKEN.match(src, pos)
        if not m:
            pos += 1
            continue
        pos = m.end()
        word, param, hexbyte, sym, brace, plain = m.groups()
        if plain is not None:
            if skip_chars:
                cut = min(skip_chars, len(plain))
                skip_chars -= cut
                plain = plain[cut:]
            if plain and not st.skip:
                pending.extend(plain.encode("latin-1", errors="replace"))
            continue
        if hexbyte is not None:
            if skip_chars:
                skip_chars -= 1
            elif not st.skip:
                pending.append(int(hexbyte, 16))
            continue
        if brace == "{":
            flush_bytes()
            stack.append(st)
            st = _RtfState(st.skip, st.dest, st.uc, st.cp, st.style, st.outline, st.entry)
            if st.dest == "fonttbl":
                st.dest, st.entry = "font", {}
            elif st.dest == "stylesheet":
                st.dest, st.entry = "style", {"s": 0}
            skip_chars = 0
            continue
        if brace == "}":
            flush_bytes()
            entry, dest = st.entry, st.dest
            if stack:
                parent = stack.pop()
                if dest == "font" and parent.dest == "fonttbl" and entry and "f" in entry:
                    cp = entry.get("cpg") or _CHARSET_CP.get(entry.get("charset", 0)) or None
                    if cp:
                        fonts[entry["f"]] = cp
                elif dest == "style" and parent.dest == "stylesheet" and entry is not None:
                    styles[entry.get("s", 0)] = (_clean(entry.get("name", "")).rstrip(";").strip().lower(),
                                                 entry.get("outline"))
                st = parent
            skip_chars = 0
            star = False
            continue
        if m.group(0).startswith(("\r", "\n")):
            continue
        if sym is not None:
            if sym == "*":
                star = True
                continue
            if skip_chars:
                skip_chars -= 1
                continue
            if sym in "\\{}":
                pending.extend(sym.encode())
            elif sym == "~":
                flush_bytes()
                emit(" ")
            elif sym == "_":
                pending.extend(b"-")
            elif sym in "\r\n" and not st.skip and st.dest == "":
                flush_bytes()
                end_para()
            continue
        # control word
        num = int(param) if param is not None else None
        if star:
            star = False
            if word not in ("title",):
                st.skip = True
                continue
        if word == "bin" and num:
            pos += num
            continue
        if word in ("fonttbl", "stylesheet", "info"):
            flush_bytes()
            st.dest = word
            continue
        if word == "title" and st.dest == "info":
            st.dest = "title"
            continue
        if word in _RTF_SKIP or (st.dest == "info" and word not in ("title",)):
            st.skip = True
            continue
        if st.dest == "font" and st.entry is not None:
            if word == "f" and num is not None:
                st.entry["f"] = num
            elif word == "fcharset" and num is not None:
                st.entry["charset"] = num
            elif word == "cpg" and num:
                st.entry["cpg"] = f"cp{num}"
            continue
        if st.dest == "style" and st.entry is not None:
            if word == "s" and num is not None:
                st.entry["s"] = num
            elif word == "outlinelevel" and num is not None:
                st.entry["outline"] = num
            continue
        if word == "ansicpg" and num:
            try:
                default_cp = codecs.lookup(f"cp{num}").name
            except LookupError:
                pass
            st.cp = fonts.get(deff, default_cp)
            continue
        if word in ("ansi", "mac", "pc", "pca"):
            default_cp = {"ansi": "cp1252", "mac": "mac_roman", "pc": "cp437", "pca": "cp850"}[word]
            st.cp = default_cp
            continue
        if word == "deff" and num is not None:
            deff = num
            continue
        if word == "f" and num is not None and st.dest == "":
            flush_bytes()
            st.cp = fonts.get(num, default_cp)
            continue
        if word == "uc" and num is not None:
            st.uc = num
            continue
        if word == "u" and num is not None:
            flush_bytes()
            cp = num + 65536 if num < 0 else num
            if 0xD800 <= cp < 0xDC00:
                high_surrogate = cp
            elif 0xDC00 <= cp < 0xE000 and high_surrogate:
                emit(chr(0x10000 + ((high_surrogate - 0xD800) << 10) + (cp - 0xDC00)))
                high_surrogate = 0
            else:
                emit(chr(cp))
            skip_chars = st.uc
            continue
        if skip_chars:
            skip_chars -= 1
            continue
        if word == "pard":
            st.style, st.outline = 0, None
            continue
        if word == "plain":
            st.cp = fonts.get(deff, default_cp)
            continue
        if word == "s" and num is not None and st.dest == "":
            st.style = num
            continue
        if word == "outlinelevel" and num is not None and st.dest == "":
            st.outline = num if num < 9 else None
            continue
        if word in _RTF_BREAKS:
            flush_bytes()
            if not st.skip and st.dest == "":
                end_para()
            continue
        if word in _RTF_CHARS:
            flush_bytes()
            emit(_RTF_CHARS[word])
            continue
    flush_bytes()
    end_para()
    book.title = _clean("".join(title))
    chapters, preamble, htitle = _chapters_from_blocks(blocks)
    book.title = book.title or htitle
    book.chapters, book.preamble = chapters, preamble
    return book


# -------------------------------------------------------------------------------------------------------- PDF

_PAGE_NUM = re.compile(r"^[\W_]*(?:page|стор(?:інка)?\.?|seite|p\.)?\s*(\d{1,4}|[ivxlcdm]{1,7})"
                       r"(?:\s*(?:of|/|з|из|von)\s*\d{1,4})?[\W_]*$", re.IGNORECASE)
_TERMINAL = ".!?…:\"'»”’)"


def _pdf_book(path: str) -> _Book:
    try:
        from pypdf import PdfReader  # noqa: PLC0415
        from pypdf.errors import DependencyError  # noqa: PLC0415
    except ImportError as e:
        raise EngineError("not_ready", "Reading PDF files needs the pypdf package, which is not installed in the "
                                       "engine environment. Run setup again to install it.") from e
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    book = _Book()
    aes = EngineError("bad_request", "This PDF is encrypted with AES, which Mimvox cannot open. Print it to a new PDF "
                                     "or save it in another format, then import that.")
    try:
        reader = PdfReader(path, strict=False)              # opens a PDF with an empty user password by itself
        ok = 1
        if reader.is_encrypted:
            try:
                ok = reader.decrypt("")
            except DependencyError:
                raise
            except Exception:                               # noqa: BLE001  (bad keys)
                ok = 0
    except DependencyError as e:                            # AES needs the optional `cryptography` package
        raise aes from e
    if reader.is_encrypted and not ok:
        raise EngineError("bad_request", "This PDF is password-protected. Remove the password (for example by "
                                         "printing it to a new PDF) and import it again.")
    n_pages = len(reader.pages)
    if n_pages == 0:
        raise EngineError("bad_request", "This PDF has no pages.")
    pages: list[list[str]] = []
    failed = 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except DependencyError as e:
            raise aes from e
        except Exception:                                   # noqa: BLE001  (a broken page should not stop the book)
            text, failed = "", failed + 1
        pages.append([s.strip() for s in _newlines(text).split("\n") if s.strip()])
    chars = [sum(len(s.replace(" ", "")) for s in p) for p in pages]
    textless = sum(1 for c in chars if c < 40)
    if sum(chars) < 15 * n_pages or textless == n_pages:
        raise EngineError("bad_request", "This PDF has no text layer (it is a scanned image). Mimvox cannot read "
                                         "scanned PDFs; run it through OCR first or use another format.")
    if failed:
        book.warnings.append(f"{failed} page(s) of the PDF could not be read and were skipped.")
    if n_pages >= 4 and textless >= max(2, n_pages // 10):
        book.warnings.append(f"{textless} of {n_pages} pages have no text layer (scanned images or pictures) and "
                             "were skipped.")
    # running headers/footers: the first/last two lines of a page that repeat (numbers ignored) on many pages
    if n_pages >= 4:
        seen: dict[str, int] = {}
        for p in pages:
            for key in {re.sub(r"\d+", "#", s.lower()) for s in p[:2] + p[-2:]}:
                seen[key] = seen.get(key, 0) + 1
        repeated = {k for k, c in seen.items() if c >= max(3, 0.3 * n_pages)}
    else:
        repeated = set()
    for k, p in enumerate(pages):
        edge = set(range(min(2, len(p)))) | set(range(max(0, len(p) - 2), len(p)))
        pages[k] = [s for i, s in enumerate(p) if not (i in edge and (
            _PAGE_NUM.fullmatch(s) or re.sub(r"\d+", "#", s.lower()) in repeated))]
    flat = [s for p in pages for s in p]
    lens = sorted(len(s) for s in flat if len(s) >= 15)
    width = lens[int(len(lens) * 0.75)] if lens else 60
    page_start: list[int] = []
    total = 0
    for p in pages:
        page_start.append(total)
        total += len(p)

    def paras(lines: list[str]) -> list[str]:
        out: list[str] = []
        cur: list[str] = []
        for s in lines:
            if cur:
                prev = cur[-1]
                if len(prev) < 0.5 * width or (len(prev) < 0.85 * width and prev[-1] in _TERMINAL):
                    out.append(_join_lines(cur, True))
                    cur = []
            cur.append(s)
        if cur:
            out.append(_join_lines(cur, True))
        return out

    # chapters: the outline (bookmarks) when it has two or more entries at one level, else heading lines
    entries: list[tuple[int, str, int]] = []

    def walk(items: list[Any], depth: int) -> None:
        for it in items:
            if isinstance(it, list):
                walk(it, depth + 1)
                continue
            try:
                pg = reader.get_destination_page_number(it)
                title = _clean(str(getattr(it, "title", "") or ""))
            except Exception:                               # noqa: BLE001
                continue
            if pg is not None and pg >= 0 and title:
                entries.append((depth, title, pg))
    try:
        walk(reader.outline, 0)
    except Exception:                                       # noqa: BLE001  (a broken outline: use heading lines)
        entries = []
    # (first line, title, is_part, heading lines to skip)
    cuts: list[tuple[int, str, bool, int]] = []
    for depth in sorted({d for d, _, _ in entries}):
        level = [(t, pg) for d, t, pg in entries if d == depth]
        if len({pg for _, pg in level}) >= 2:
            for t, pg in level:
                at, skip = (page_start[pg] if pg < len(page_start) else total), 0
                tk = f" {_key(t)} "
                lines = pages[pg] if pg < len(pages) else []
                for i, s in enumerate(lines[:12]):
                    sk = _key(s)
                    if sk and (f" {sk} " in tk or tk.strip().startswith(sk) or sk.startswith(tk.strip())):
                        at = page_start[pg] + i
                        skip = 1                             # the heading line(s) as printed on the page
                        while i + skip < len(lines) and skip < 3 and _key(lines[i + skip]) and \
                                f" {_key(lines[i + skip])} " in tk:
                            skip += 1
                        break
                cuts.append((at, t, False, skip))
            break
    if len({c[0] for c in cuts}) < 2:
        cuts = []
        starts = set(page_start)
        for i, s in enumerate(flat):
            if len(s) > 70 or len(s) >= 0.7 * width and i:
                continue
            hm = _heading_match(s)
            if not hm or hm[0] == "roman":
                continue
            prev = flat[i - 1] if i else ""
            if prev and not (prev[-1] in _TERMINAL or len(prev) < 0.5 * width or i in starts):
                continue
            title, skip = s, 1
            nxt = flat[i + 1] if i + 1 < len(flat) else ""
            if hm[2] and s[-1] not in _TERMINAL and nxt and len(nxt) < 0.5 * width and len(nxt.split()) <= 8:
                title, skip = f"{s} {nxt}", 2                # a title wrapped onto a second line
            cuts.append((i, title, hm[1], skip))
        if len(cuts) < 2:
            cuts = []
    cuts = sorted({c[0]: c for c in reversed(cuts)}.values())
    if not cuts:
        book.chapters = [_Ch("", paras(flat))]
        if count_words(" ".join(flat[:20000])) > 5000:
            book.warnings.append("No bookmarks or chapter headings were found, so the text is one chapter.")
    else:
        book.preamble = paras(flat[:cuts[0][0]])
        for k, (at, title, part, skip) in enumerate(cuts):
            stop = cuts[k + 1][0] if k + 1 < len(cuts) else len(flat)
            book.chapters.append(_Ch(title, paras(flat[min(at + skip, stop):stop]), part))
    meta_title = ""
    try:
        meta_title = _clean(str(reader.metadata.title or "")) if reader.metadata else ""
    except Exception:                                       # noqa: BLE001
        pass
    meta_title = re.sub(r"^Microsoft (Word|PowerPoint) - ", "", meta_title)
    if meta_title and not re.fullmatch(r"(untitled|document\d*|.*\.(docx?|odt|rtf|tex|indd|pdf))", meta_title,
                                       re.IGNORECASE):
        book.title = meta_title
    return book


# -------------------------------------------------------------------------------------------------- subtitles

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_CUE_TAGS = re.compile(r"<[^>]*>|\{\\[^}]*\}|\{[^}]*\}")
_SOUND = re.compile(r"\[[^\]]*\]|[♪♫♬]+")


def _seconds(m: re.Match[str]) -> float:
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _subtitles_book(text: str) -> _Book:
    book = _Book()
    paras: list[str] = []
    cur: list[str] = []
    last_line = ""
    prev_end: float | None = None
    for block in re.split(r"\n[ \t]*\n", _newlines(text).strip()):
        lines = [s for s in block.split("\n")]
        first = lines[0].strip() if lines else ""
        if first.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")) and "-->" not in first:
            continue
        idx = next((i for i, s in enumerate(lines) if "-->" in s), None)
        if idx is None:
            continue
        found = list(_TIME.finditer(lines[idx]))
        start = _seconds(found[0]) if found else None
        end = _seconds(found[1]) if len(found) > 1 else start
        if prev_end is not None and start is not None and start - prev_end >= 2.0 and cur:
            paras.append(" ".join(cur))
            cur = []
        if end is not None:
            prev_end = max(prev_end or 0.0, end)
        for raw in lines[idx + 1:]:
            s = html.unescape(_CUE_TAGS.sub("", raw))
            s = _SOUND.sub(" ", s)
            s = re.sub(r"^\s*[-–—]\s*", "", s)
            s = _clean(s)
            if not s or re.fullmatch(r"\([^)]*\)", s) or not _WORDISH.search(s):
                continue
            if s == last_line:
                continue                                    # rolling captions repeat the previous line
            last_line = s
            cur.append(s)
    if cur:
        paras.append(" ".join(cur))
    book.chapters = [_Ch("", paras)]
    return book


# ---------------------------------------------------------------------------------------------------- extract

def _format_of(path: str) -> str:
    low = path.lower()
    if low.endswith(".fb2.zip"):
        return ".fb2.zip"
    return os.path.splitext(low)[1]


def _bad_file(label: str, what: str) -> EngineError:
    return EngineError("bad_request", f"This {label} file could not be read: {what}.")


def extract(path: str) -> dict[str, Any]:
    """Read ``path`` and return ``{"format", "format_label", "title", "chapters": [{"title", "text"}], "encoding",
    "warnings"}``. Raises :class:`EngineError` (``bad_request`` / ``not_ready``)."""
    ext = _format_of(path)
    if ext not in SUPPORTED:
        kind = f"'{ext}'" if ext else "(a file without an extension)"
        raise EngineError("bad_request", f"Unsupported file type {kind}. Supported: {', '.join(SUPPORTED)}.")
    fmt, label = _FORMAT[ext], SUPPORTED[ext]
    stem = os.path.basename(path)[:-len(ext)] if ext else os.path.basename(path)
    fallback = _clean(stem.replace("_", " ")) or "Untitled"
    try:
        if os.path.getsize(path) > MAX_BYTES:
            raise EngineError("bad_request", f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
        if fmt in ("txt", "md", "html", "srt", "vtt", "rtf"):
            with open(path, "rb") as f:
                data = f.read()
            if fmt == "rtf":
                book = _rtf_book(data)
            else:
                text, enc = _decode(data, _markup_charset(data) if fmt == "html" else None)
                book = {"txt": _plain_text_book, "md": _markdown_book, "html": _html_book,
                        "srt": _subtitles_book, "vtt": _subtitles_book}[fmt](text)
                book.encoding = enc
        elif fmt == "epub":
            book = _epub_book(path)
        elif fmt == "docx":
            book = _docx_book(path)
        elif fmt == "odt":
            book = _odt_book(path)
        elif fmt == "fb2":
            book = _fb2_book(path)
        else:
            book = _pdf_book(path)
    except EngineError:
        raise
    except zipfile.BadZipFile as e:
        raise _bad_file(label, "it is damaged or not really this format (not a valid ZIP archive)") from e
    except ET.ParseError as e:
        raise _bad_file(label, f"its XML is damaged ({e})") from e
    except (MemoryError, RecursionError) as e:
        raise _bad_file(label, "it is too large or too deeply nested") from e
    except OSError as e:
        raise EngineError("bad_request", f"The file could not be opened: {e.strerror or e}.") from e
    except Exception as e:                                  # noqa: BLE001  (a parser library's own error types)
        log.warning("text.import failed for %s", path, exc_info=True)
        raise _bad_file(label, f"{type(e).__name__}: {e}") from e
    title, chapters = _finish(book, fallback)
    if not chapters:
        raise EngineError("bad_request", "No readable text was found in this file.")
    return {"format": fmt, "format_label": label, "title": title, "chapters": chapters, "encoding": book.encoding,
            "warnings": book.warnings}


def _sample(chapters: list[dict[str, str]]) -> str:
    text = "\n\n".join(c["text"] for c in chapters)
    if len(text) <= SAMPLE_CHARS:
        return text
    n, size = 10, SAMPLE_CHARS // 10
    step = (len(text) - size) // (n - 1)
    return " ".join(text[i * step:i * step + size] for i in range(n))


@method("text.import")
def text_import(ctx: Any, params: dict[str, Any]) -> dict[str, Any]:
    path = params.get("path")
    if not isinstance(path, str) or not path.strip():
        raise EngineError("bad_request", "'path' is required")
    if not os.path.isabs(path):
        raise EngineError("bad_request", "'path' must be an absolute path")
    if not os.path.exists(path):
        raise EngineError("bad_request", f"File not found: {path}")
    if os.path.isdir(path):
        raise EngineError("bad_request", "This is a folder; choose a text file.")
    raw = params.get("language")
    detected = raw is None or (isinstance(raw, str) and raw.strip().lower() in ("", "auto"))
    code: str | None = None
    if not detected:
        try:
            code = lang.normalize_code(raw)
        except ValueError as e:
            base = raw.strip().lower().replace("_", "-").split("-", 1)[0] if isinstance(raw, str) else ""
            if not re.fullmatch(r"[a-z]{2,3}", base):
                raise EngineError("bad_request", f"Unknown language {raw!r}: use a language code such as \"en\" or "
                                                 f"\"uk\", or \"auto\".") from e
            code = base
    book = extract(path)
    chapters = book["chapters"]
    if detected:
        code = lang.detect(_sample(chapters))
    language = lang.LANGUAGES.get(code or "")
    wpm = language.wpm if language else WPM
    out_chapters = []
    for ch in chapters:
        words = count_words(ch["text"])
        out_chapters.append({"title": ch["title"], "text": ch["text"], "words": words, "chars": len(ch["text"]),
                             "est_s": est_seconds(words, wpm)})
    words = sum(c["words"] for c in out_chapters)
    return {"path": path, "name": os.path.basename(path), "format": book["format"],
            "format_label": book["format_label"], "title": book["title"], "encoding": book["encoding"],
            "language": code, "language_detected": detected, "words": words,
            "chars": sum(c["chars"] for c in out_chapters), "est_s": est_seconds(words, wpm),
            "chapters": out_chapters, "warnings": book["warnings"]}
