import hashlib
import os

import pytest
from factories import JPEG, PNG, docx, epub, fb2, fb2_binary, odt, text_file

EPUB2_META = """<dc:title>Old Book</dc:title>
<dc:creator opf:role="ill" opf:file-as="Painter, Pat">Pat Painter</dc:creator>
<dc:creator opf:role="aut" opf:file-as="Doe, Jane">Jane Doe</dc:creator>
<dc:creator opf:role="aut">John  Roe</dc:creator>
<dc:creator opf:role="aut">Jane Doe</dc:creator>
<dc:description>&lt;div&gt;&lt;p&gt;A &lt;b&gt;bold&lt;/b&gt; story&amp;nbsp;of
   things.&lt;/p&gt;&lt;p&gt;Second &amp;amp; last.&lt;/p&gt;&lt;/div&gt;</dc:description>
<dc:language>en-US</dc:language>
<dc:date opf:event="modification">2020-01-01</dc:date>
<dc:date opf:event="publication">1999-05</dc:date>
<dc:date opf:event="creation">1998</dc:date>
<meta name="cover" content="cimg"/>"""

EPUB3_META = """<dc:title>New Book</dc:title>
<dc:creator id="a1">Ann Author</dc:creator><meta refines="#a1" property="role" scheme="marc:relators">aut</meta>
<dc:creator id="t1">Tom Translator</dc:creator><meta refines="#t1" property="role" scheme="marc:relators">trl</meta>
<dc:creator id="a2">Bob Writer</dc:creator><meta refines="#a2" property="role" scheme="marc:relators">aut</meta>
<dc:description>Plain   description
 over lines.</dc:description>
<dc:language>uk-UA</dc:language><dc:date>2021-03-04T00:00:00Z</dc:date>
<meta property="dcterms:modified">2022-01-01T00:00:00Z</meta>"""


def image_item(ident, href, media_type="image/jpeg", props=""):
    p = f' properties="{props}"' if props else ""
    return f'<item id="{ident}" href="{href}" media-type="{media_type}"{p}/>'


def cover_bytes(meta):
    with open(meta["cover"]["path"], "rb") as fh:
        return fh.read()


def test_epub2_meta(tmp_path, import_book):
    path = epub(tmp_path / "e2.epub", version="2.0", toc="ncx", metadata=EPUB2_META,
                extra_items=image_item("cimg", "images/front.jpeg"), files={"OEBPS/images/front.jpeg": JPEG})
    meta = import_book(path)["meta"]
    assert meta["authors"] == ["Jane Doe", "John Roe"]
    assert meta["description"] == "A bold story of things. Second & last."
    assert meta["language_tag"] == "en-US"
    assert meta["date"] == "1999-05"
    assert meta["cover"]["media_type"] == "image/jpeg"
    assert cover_bytes(meta) == JPEG


def test_epub3_meta(tmp_path, import_book):
    items = image_item("x1", "img/c.png", "image/png", "cover-image") + image_item("cover", "img/other.jpg")
    path = epub(tmp_path / "e3.epub", metadata=EPUB3_META, extra_items=items,
                files={"OEBPS/img/c.png": PNG, "OEBPS/img/other.jpg": JPEG})
    meta = import_book(path)["meta"]
    assert meta["authors"] == ["Ann Author", "Bob Writer"]
    assert meta["description"] == "Plain description over lines."
    assert meta["language_tag"] == "uk-UA"
    assert meta["date"] == "2021-03-04T00:00:00Z"
    assert meta["cover"]["media_type"] == "image/png" and cover_bytes(meta) == PNG


def test_epub_without_metadata(tmp_path, import_book):
    meta = import_book(epub(tmp_path / "bare.epub", metadata="<dc:title>Bare</dc:title>"))["meta"]
    assert meta == {"authors": [], "description": None, "language_tag": None, "date": None, "cover": None}


def test_epub_without_author_roles_keeps_every_creator(tmp_path, import_book):
    md = ('<dc:creator opf:role="edt">Ed Editor</dc:creator><dc:creator opf:role="ill">Ill Ustrator</dc:creator>'
          '<dc:description><p xmlns="http://www.w3.org/1999/xhtml">Nested</p>'
          '<p xmlns="http://www.w3.org/1999/xhtml">XHTML desc</p></dc:description>'
          '<dc:date opf:event="modification">2020</dc:date>')
    meta = import_book(epub(tmp_path / "e.epub", version="2.0", toc="ncx", metadata=md))["meta"]
    assert meta["authors"] == ["Ed Editor", "Ill Ustrator"]
    assert meta["description"] == "Nested XHTML desc"
    assert meta["date"] is None


def test_epub_cover_found_by_name(tmp_path, import_book):
    path = epub(tmp_path / "e.epub", extra_items=image_item("img9", "pics/Cover_Art.jpg"),
                files={"OEBPS/pics/Cover_Art.jpg": JPEG})
    assert cover_bytes(import_book(path)["meta"]) == JPEG


@pytest.mark.parametrize(
    ("items", "files"),
    [
        (image_item("x1", "img/c.svg", "image/svg+xml", "cover-image"),
         {"OEBPS/img/c.svg": "<svg xmlns='http://www.w3.org/2000/svg'/>"}),
        (image_item("cover", "missing.jpg"), {}),
        (image_item("cover", "fake.png", "image/png"), {"OEBPS/fake.png": b"not an image"}),
        (image_item("x1", "big.jpg", props="cover-image"), {"OEBPS/big.jpg": JPEG + b"\0" * (21 * 1024 * 1024)}),
    ],
    ids=["svg", "missing", "not-an-image", "oversized"],
)
def test_unusable_epub_cover_is_null(tmp_path, import_book, items, files):
    r = import_book(epub(tmp_path / "e.epub", extra_items=items, files=files))
    assert r["meta"]["cover"] is None
    assert [c["title"] for c in r["chapters"]] == ["Chapter 1", "Chapter 2"]


def test_cover_file_is_named_by_content_hash(tmp_path, ctx, import_book):
    a = epub(tmp_path / "a.epub", extra_items=image_item("x", "c.jpg", props="cover-image"),
             files={"OEBPS/c.jpg": JPEG})
    b = epub(tmp_path / "b.epub", extra_items=image_item("y", "cover.jpg"), files={"OEBPS/cover.jpg": JPEG})
    first, second = import_book(a)["meta"]["cover"], import_book(b)["meta"]["cover"]
    assert first == second
    assert os.path.isabs(first["path"])
    assert os.path.basename(first["path"]) == hashlib.sha256(JPEG).hexdigest()[:32] + ".jpg"
    folder = os.path.join(ctx.data_dir, "tmp", "covers")
    assert os.path.dirname(first["path"]) == os.path.abspath(folder)
    assert os.listdir(folder) == [os.path.basename(first["path"])]


FB2_INFO = """<genre>prose</genre>
<author><first-name>Тарас</first-name><middle-name>Григорович</middle-name><last-name>Шевченко</last-name></author>
<author><nickname>anon</nickname></author>
<book-title>Кобзар</book-title>
<annotation><p>Перший рядок.</p><p>Другий <emphasis>рядок</emphasis>.</p></annotation>
<date value="1840-01-01">1840</date><coverpage><image l:href="#cover.jpg"/></coverpage><lang>uk</lang>"""

SECTIONS = [("Розділ 1", ["Текст розділу для перевірки. " * 20]), ("Розділ 2", ["Ще текст розділу. " * 20])]


def test_fb2_meta(tmp_path, import_book):
    r = import_book(fb2(tmp_path / "k.fb2", FB2_INFO, SECTIONS, fb2_binary("cover.jpg", JPEG)))
    meta = r["meta"]
    assert r["title"] == "Кобзар"
    assert meta["authors"] == ["Тарас Григорович Шевченко", "anon"]
    assert meta["description"] == "Перший рядок. Другий рядок."
    assert meta["language_tag"] == "uk"
    assert meta["date"] == "1840"
    assert cover_bytes(meta) == JPEG


def test_fb2_date_value_and_broken_cover(tmp_path, import_book):
    info = ('<author><last-name>Only</last-name></author><date value="2001-02-03"/>'
            '<coverpage><image l:href="#c"/></coverpage>')
    broken = '<binary id="c" content-type="image/jpeg">!!!===A</binary>'
    r = import_book(fb2(tmp_path / "x.fb2", info, SECTIONS, broken))
    assert r["meta"]["authors"] == ["Only"]
    assert r["meta"]["date"] == "2001-02-03"
    assert r["meta"]["cover"] is None
    assert len(r["chapters"]) == 2


def test_docx_meta(tmp_path, import_book):
    core = ("<dc:title>Doc T</dc:title><dc:creator>Alice; Bob</dc:creator><dc:description>Docx  desc</dc:description>"
            "<dc:language>de-DE</dc:language>")
    r = import_book(docx(tmp_path / "d.docx", [(None, "Hello world text. " * 30)], core))
    assert r["title"] == "Doc T"
    assert r["meta"] == {"authors": ["Alice", "Bob"], "description": "Docx desc", "language_tag": "de-DE",
                         "date": None, "cover": None}


def test_odt_meta_prefers_initial_creator(tmp_path, import_book):
    meta = ("<dc:title>Odt T</dc:title><meta:initial-creator>Orig Author</meta:initial-creator>"
            "<dc:creator>Last Editor</dc:creator><dc:description>Odt desc</dc:description>"
            "<dc:language>fr</dc:language>")
    r = import_book(odt(tmp_path / "o.odt", [(None, "Odt text here. " * 30)], meta))
    assert r["title"] == "Odt T"
    assert r["meta"]["authors"] == ["Orig Author"]
    assert r["meta"]["description"] == "Odt desc" and r["meta"]["language_tag"] == "fr"


def test_plain_text_has_empty_meta(tmp_path, import_book):
    r = import_book(text_file(tmp_path / "t.txt", "Some text. " * 20))
    assert r["meta"] == {"authors": [], "description": None, "language_tag": None, "date": None, "cover": None}
