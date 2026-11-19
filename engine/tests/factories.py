import base64
import os
import zipfile

JPEG = b"\xff\xd8\xff\xe0" + b"JFIF-test-jpeg" * 10
PNG = b"\x89PNG\r\n\x1a\n" + b"test-png" * 10

CONTAINER = (
    '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>'
    "</container>"
)


def body_paras(label, n=5):
    return [f"Paragraph {i} of {label} has some words to read aloud for the test." for i in range(n)]


def xhtml(heading, paras):
    ps = "".join(f"<p>{p}</p>" for p in paras)
    h = f"<h1>{heading}</h1>" if heading else ""
    return (
        '<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml">'
        f"<head><title>{heading}</title></head><body>{h}{ps}</body></html>"
    )


def nav(entries):
    lis = "".join(f'<li><a href="{href}">{title}</a></li>' for title, href in entries)
    return (
        '<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        f'<body><nav epub:type="toc"><ol>{lis}</ol></nav></body></html>'
    )


def ncx(entries):
    points = "".join(
        f'<navPoint id="n{i}"><navLabel><text>{title}</text></navLabel><content src="{href}"/></navPoint>'
        for i, (title, href) in enumerate(entries)
    )
    return (
        '<?xml version="1.0"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">'
        f"<navMap>{points}</navMap></ncx>"
    )


def opf(version, metadata, docs, extra_items="", nav_doc=False, ncx_doc=False):
    """``docs``: manifest ids and hrefs of the spine documents, in reading order."""
    items = "".join(f'<item id="{i}" href="{h}" media-type="application/xhtml+xml"/>' for i, h in docs)
    if nav_doc:
        items += '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
    if ncx_doc:
        items += '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>'
    spine = "".join(f'<itemref idref="{i}"/>' for i, _ in docs)
    toc = ' toc="ncx"' if ncx_doc else ""
    return (
        f'<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="{version}" '
        'unique-identifier="id"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
        f'xmlns:opf="http://www.idpf.org/2007/opf">{metadata}</metadata><manifest>{items}{extra_items}</manifest>'
        f"<spine{toc}>{spine}</spine></package>"
    )


def write_zip(path, members, mimetype=None):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        if mimetype:
            z.writestr("mimetype", mimetype, compress_type=zipfile.ZIP_STORED)
        for name, data in members.items():
            z.writestr(name, data)
    return str(path)


def epub(path, version="3.0", metadata="<dc:title>Test Book</dc:title>", chapters=None, toc="nav", extra_items="",
         files=None):
    """An EPUB of ``chapters`` (title, paragraphs); ``toc`` is "nav", "ncx" or "both"."""
    chapters = chapters if chapters is not None else [(f"Chapter {n}", body_paras(f"chapter {n}")) for n in (1, 2)]
    docs = [(f"c{n}", f"c{n}.xhtml") for n in range(1, len(chapters) + 1)]
    entries = [(t, h) for (t, _), (_, h) in zip(chapters, docs)]
    members = {"META-INF/container.xml": CONTAINER}
    members["OEBPS/content.opf"] = opf(version, metadata, docs, extra_items, nav_doc=toc in ("nav", "both"),
                                       ncx_doc=toc in ("ncx", "both"))
    for (title, paras), (_, href) in zip(chapters, docs):
        members[f"OEBPS/{href}"] = xhtml(title, paras)
    if toc in ("nav", "both"):
        members["OEBPS/nav.xhtml"] = nav(entries)
    if toc in ("ncx", "both"):
        members["OEBPS/toc.ncx"] = ncx(entries)
    members.update(files or {})
    return write_zip(path, members, "application/epub+zip")


def fb2(path, title_info, sections, binaries=""):
    secs = "".join(
        f"<section><title><p>{t}</p></title>" + "".join(f"<p>{p}</p>" for p in paras) + "</section>"
        for t, paras in sections
    )
    doc = (
        '<?xml version="1.0" encoding="utf-8"?><FictionBook xmlns="http://www.gribuser.ru/xml/fictionbook/2.0" '
        f'xmlns:l="http://www.w3.org/1999/xlink"><description><title-info>{title_info}</title-info></description>'
        f"<body>{secs}</body>{binaries}</FictionBook>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
    return str(path)


def fb2_binary(ident, data, media_type="image/jpeg"):
    return f'<binary id="{ident}" content-type="{media_type}">{base64.b64encode(data).decode()}</binary>'


def docx(path, blocks, core=None):
    """``blocks``: (style id or None, text)."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    ps = "".join(
        "<w:p>" + (f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else "")
        + f"<w:r><w:t>{text}</w:t></w:r></w:p>"
        for style, text in blocks
    )
    document = f'<?xml version="1.0"?><w:document xmlns:w="{w}"><w:body>{ps}</w:body></w:document>'
    members = {"word/document.xml": document}
    if core:
        members["docProps/core.xml"] = (
            '<?xml version="1.0"?><cp:coreProperties '
            'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/">'
            f"{core}</cp:coreProperties>"
        )
    return write_zip(path, members)


def odt(path, blocks, meta=None):
    """``blocks``: (outline level or None, text)."""
    office = "urn:oasis:names:tc:opendocument:xmlns:office:1.0"
    text_ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    body = "".join(
        f'<text:h text:outline-level="{lvl}">{t}</text:h>' if lvl else f"<text:p>{t}</text:p>" for lvl, t in blocks
    )
    members = {"content.xml": f'<?xml version="1.0"?><office:document-content xmlns:office="{office}" '
                              f'xmlns:text="{text_ns}"><office:body><office:text>{body}</office:text></office:body>'
                              "</office:document-content>"}
    if meta:
        members["meta.xml"] = (
            f'<?xml version="1.0"?><office:document-meta xmlns:office="{office}" '
            'xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0" xmlns:dc="http://purl.org/dc/elements/1.1/">'
            f"<office:meta>{meta}</office:meta></office:document-meta>"
        )
    return write_zip(path, members, "application/vnd.oasis.opendocument.text")


def text_file(path, text, encoding="utf-8"):
    with open(path, "wb") as fh:
        fh.write(text.encode(encoding))
    return str(os.fspath(path))
