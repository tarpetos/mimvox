import zipfile

import pytest
from factories import body_paras, docx, epub, fb2, odt, text_file, write_zip
from mimvox_engine.registry import EngineError


def titles(result):
    return [c["title"] for c in result["chapters"]]


@pytest.mark.parametrize("toc", ["nav", "ncx", "both"])
def test_epub_chapters_follow_the_table_of_contents(tmp_path, import_book, toc):
    version = "2.0" if toc == "ncx" else "3.0"
    r = import_book(epub(tmp_path / "b.epub", version=version, toc=toc))
    assert r["format"] == "epub"
    assert r["title"] == "Test Book"
    assert titles(r) == ["Chapter 1", "Chapter 2"]
    first = r["chapters"][0]["text"]
    assert first.startswith("Paragraph 0 of chapter 1")
    assert "Chapter 1" not in first
    assert r["chapters"][0]["words"] > 0 and r["chapters"][0]["est_s"] > 0


def test_epub_nav_wins_over_ncx(tmp_path, import_book):
    path = epub(tmp_path / "b.epub", toc="both")
    with zipfile.ZipFile(path) as z:
        members = {n: z.read(n) for n in z.namelist() if n != "mimetype"}
    members["OEBPS/toc.ncx"] = members["OEBPS/toc.ncx"].replace(b"Chapter", b"Ncx entry")
    write_zip(tmp_path / "c.epub", members, "application/epub+zip")
    assert titles(import_book(tmp_path / "c.epub")) == ["Chapter 1", "Chapter 2"]


def test_epub_drops_table_of_contents_and_title_page(tmp_path, import_book):
    chapters = [
        ("Test Book", ["by Jane Doe"]),
        ("Contents", ["Chapter 1", "Chapter 2", "Chapter 3"]),
        *[(f"Chapter {n}", body_paras(f"chapter {n}")) for n in (1, 2, 3)],
    ]
    r = import_book(epub(tmp_path / "b.epub", chapters=chapters))
    assert titles(r) == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_epub_repeated_heading_is_stripped_from_the_text(tmp_path, import_book):
    chapters = [(f"Chapter {n}", [f"CHAPTER {n}", *body_paras(f"chapter {n}")]) for n in (1, 2)]
    r = import_book(epub(tmp_path / "b.epub", chapters=chapters))
    assert r["chapters"][0]["text"].startswith("Paragraph 0 of chapter 1")


def test_short_first_sentence_containing_the_title_is_kept(tmp_path, import_book):
    chapters = [("The Storm", ["It was the storm.", *body_paras("one")]), ("The Calm", body_paras("two"))]
    r = import_book(epub(tmp_path / "b.epub", chapters=chapters))
    assert r["chapters"][0]["text"].startswith("It was the storm.")


def test_numbered_heading_containing_the_title_is_stripped(tmp_path, import_book):
    chapters = [("The Storm", ["Chapter 1. The Storm", *body_paras("one")]), ("The Calm", body_paras("two"))]
    r = import_book(epub(tmp_path / "b.epub", chapters=chapters))
    assert r["chapters"][0]["text"].startswith("Paragraph 0 of one")


def encryption(uri):
    return (
        '<?xml version="1.0"?><encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
        'xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData><enc:CipherData>'
        f'<enc:CipherReference URI="{uri}"/></enc:CipherData></enc:EncryptedData></encryption>'
    )


def test_drm_encrypted_epub_is_rejected(tmp_path, import_book):
    path = epub(tmp_path / "drm.epub", files={"META-INF/encryption.xml": encryption("OEBPS/c1.xhtml")})
    with pytest.raises(EngineError) as e:
        import_book(path)
    assert e.value.code == "bad_request" and "DRM" in e.value.message


def test_obfuscated_fonts_are_not_drm(tmp_path, import_book):
    path = epub(tmp_path / "fonts.epub", files={"META-INF/encryption.xml": encryption("OEBPS/fonts/serif.otf")})
    assert titles(import_book(path)) == ["Chapter 1", "Chapter 2"]


def test_fb2_sections_are_chapters(tmp_path, import_book):
    path = fb2(
        tmp_path / "k.fb2",
        "<book-title>Кобзар</book-title><lang>uk</lang>",
        [("Розділ 1", ["Текст першого розділу для перевірки. " * 10]), ("Розділ 2", ["Ще текст розділу. " * 10])],
    )
    r = import_book(path)
    assert r["format"] == "fb2" and r["title"] == "Кобзар"
    assert titles(r) == ["Розділ 1", "Розділ 2"]
    assert r["chapters"][1]["text"].startswith("Ще текст розділу.")


def test_docx_headings_are_chapters(tmp_path, import_book):
    blocks = [("Title", "My Document")]
    for n in (1, 2):
        blocks += [("Heading1", f"Chapter {n}"), (None, f"Text of chapter {n}. " * 20)]
    r = import_book(docx(tmp_path / "d.docx", blocks))
    assert r["format"] == "docx" and r["title"] == "My Document"
    assert titles(r) == ["Chapter 1", "Chapter 2"]
    assert r["chapters"][0]["text"].startswith("Text of chapter 1.")


def test_odt_headings_are_chapters(tmp_path, import_book):
    blocks = []
    for n in (1, 2):
        blocks += [(1, f"Part {n}"), (None, f"Words of part {n}. " * 80)]
    r = import_book(odt(tmp_path / "o.odt", blocks))
    assert r["format"] == "odt"
    assert titles(r) == ["Part 1", "Part 2"]


UKRAINIAN = (
    "Розділ 1\n\n"
    + "Він сказав, що вона вже пішла додому, але ми ще чекали її біля школи. " * 3
    + "\n\nРозділ 2\n\n"
    + "Це була їхня улюблена пісня, і ми її співали до ранку. " * 3
)


@pytest.mark.parametrize(
    ("encoding", "prefix", "expected"),
    [
        ("utf-8", b"", "utf-8"),
        ("utf-8", b"\xef\xbb\xbf", "utf-8"),
        ("utf-16", b"", "utf-16"),
        ("utf-16-le", b"", "utf-16-le"),
        ("cp1251", b"", "cp1251"),
    ],
)
def test_text_encodings(tmp_path, import_book, encoding, prefix, expected):
    path = tmp_path / "book.txt"
    path.write_bytes(prefix + UKRAINIAN.encode(encoding))
    r = import_book(path)
    assert r["encoding"].replace("_", "-").startswith(expected)
    assert titles(r) == ["Розділ 1", "Розділ 2"]
    assert "улюблена пісня" in r["chapters"][1]["text"]
    assert "\ufeff" not in r["chapters"][0]["text"]
    assert r["language"] == "uk"


def test_markdown_headings(tmp_path, import_book):
    md = "# The Book\n\n"
    for n in (1, 2):
        md += f"## Chapter {n}\n\nSome **bold** text and a [link](http://example.com), part {n}.\n\n"
    md += "Setext heading\n--------------\n\nA paragraph under it.\n"
    r = import_book(text_file(tmp_path / "b.md", md))
    assert r["format"] == "md" and r["title"] == "The Book"
    assert titles(r) == ["Chapter 1", "Chapter 2", "Setext heading"]
    assert r["chapters"][0]["text"] == "Some bold text and a link, part 1."


def plain(headings, words=40):
    return "\n\n".join(f"{h}\n\n" + f"Body text under {h} goes on. " * (words // 5) for h in headings)


@pytest.mark.parametrize(
    ("headings", "expected"),
    [
        (["Chapter 1", "Chapter 2", "Chapter 3"], ["Chapter 1", "Chapter 2", "Chapter 3"]),
        (["CHAPTER I. THE BEGINNING", "CHAPTER II. THE END"], ["Chapter I. The Beginning", "Chapter II. The End"]),
        (["I", "II", "III"], ["I", "II", "III"]),
        (["Розділ перший", "Розділ другий"], ["Розділ перший", "Розділ другий"]),
        (["Розділ 1", "Розділ 2"], ["Розділ 1", "Розділ 2"]),
        (["Prologue", "Chapter 1", "Epilogue"], ["Prologue", "Chapter 1", "Epilogue"]),
    ],
)
def test_plain_text_chapter_headings(tmp_path, import_book, headings, expected):
    r = import_book(text_file(tmp_path / "book.txt", plain(headings)))
    assert titles(r) == expected


def test_plain_text_heading_with_subtitle_line(tmp_path, import_book):
    text = "Chapter 1\n\nThe Storm\n\n" + "Rain fell. " * 30 + "\n\nChapter 2\n\nThe Calm\n\n" + "Sun rose. " * 30
    assert titles(import_book(text_file(tmp_path / "b.txt", text))) == ["Chapter 1. The Storm", "Chapter 2. The Calm"]


TOC = "My Novel\n\nContents\n\nChapter 1\nChapter 2\nChapter 3\n\n"


def test_plain_text_table_of_contents_is_dropped(tmp_path, import_book):
    foreword = "A short foreword that stands between the contents and the first chapter of this book. " * 2
    text = TOC + foreword + "\n\n" + plain(["Chapter 1", "Chapter 2", "Chapter 3"])
    r = import_book(text_file(tmp_path / "b.txt", text))
    assert titles(r) == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert r["title"] == "My Novel"
    assert r["chapters"][0]["text"].startswith("Body text under Chapter 1")


@pytest.mark.parametrize("words", [40, 400])
def test_plain_text_table_of_contents_right_before_chapter_one(tmp_path, import_book, words):
    r = import_book(text_file(tmp_path / "b.txt", TOC + plain(["Chapter 1", "Chapter 2", "Chapter 3"], words)))
    assert titles(r) == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert r["title"] == "My Novel"
    assert r["chapters"][0]["text"].startswith("Body text under Chapter 1")


def test_plain_text_table_of_contents_right_before_a_part_and_its_first_chapter(tmp_path, import_book):
    toc = "My Novel\n\nContents\n\nPart 1\nChapter 1\nChapter 2\nPart 2\nChapter 3\n\n"
    body = "Part 1\n\n" + plain(["Chapter 1", "Chapter 2"]) + "\n\nPart 2\n\n" + plain(["Chapter 3"])
    r = import_book(text_file(tmp_path / "b.txt", toc + body))
    assert titles(r) == ["Chapter 1", "Chapter 2", "Chapter 3"]
    assert "Body text under Chapter 1" in r["chapters"][0]["text"]
    assert "Contents" not in " ".join(c["text"] for c in r["chapters"])


def test_plain_text_without_headings_is_one_chapter(tmp_path, import_book):
    r = import_book(text_file(tmp_path / "notes.txt", "Just a few lines.\n\nOf plain text."))
    assert len(r["chapters"]) == 1
    assert r["title"] == "notes"


def test_bad_requests(tmp_path, import_book):
    with pytest.raises(EngineError, match="absolute"):
        import_book("relative.txt")
    with pytest.raises(EngineError, match="not found"):
        import_book(tmp_path / "missing.txt")
    (tmp_path / "x.xyz").write_text("text")
    with pytest.raises(EngineError, match="Unsupported"):
        import_book(tmp_path / "x.xyz")
    (tmp_path / "broken.epub").write_bytes(b"not a zip")
    with pytest.raises(EngineError, match="not a valid ZIP"):
        import_book(tmp_path / "broken.epub")
    (tmp_path / "empty.txt").write_text("   \n\n  ")
    with pytest.raises(EngineError, match="No readable text"):
        import_book(tmp_path / "empty.txt")


def long_book(n_words=9000):
    words = ["river", "garden", "window", "morning", "letter", "harbour", "candle", "winter", "station", "orchard"]
    paras, total, i = [], 0, 0
    while total < n_words:
        s = " ".join(words[(i + k) % len(words)] for k in range(12))
        paras.append(f"The {s} went on. Then {s} stopped.")
        total += 28
        i += 1
    return "\n\n".join(paras)


def test_split_auto_applies_the_suggested_parts(tmp_path, import_book):
    path = text_file(tmp_path / "long.txt", long_book())
    plain = import_book(path)
    assert len(plain["chapters"]) == 1 and plain["parts"] is not None
    split = import_book(path, split="auto")
    assert split["parts"] is None
    assert split["chapters"] == plain["parts"]["chapters"] and len(split["chapters"]) > 1
    assert split["words"] == sum(c["words"] for c in split["chapters"])


def test_split_auto_leaves_a_short_book_alone(tmp_path, import_book):
    path = text_file(tmp_path / "b.txt", plain(["Chapter 1", "Chapter 2"]))
    plain_r, split = import_book(path), import_book(path, split="auto")
    assert plain_r["parts"] is None and split["parts"] is None
    assert split["chapters"] == plain_r["chapters"]


def test_split_must_be_auto(tmp_path, import_book):
    with pytest.raises(EngineError) as e:
        import_book(text_file(tmp_path / "b.txt", "Some text."), split="yes")
    assert e.value.code == "bad_request"
