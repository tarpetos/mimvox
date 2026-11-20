#!/usr/bin/env python3
"""Generate the app icon set, and its API and MCP variants for mimvox-server and mimvox-mcp, into
tools/release/icons/ (committed):
``uv run --with pillow python tools/release/make_icons.py``."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent / "icons"
PNG_SIZES = (16, 32, 48, 64, 128, 256, 512, 1024)
LABELS = {"api": "API", "mcp": "MCP"}
LABEL_SIZES = (16, 24, 32, 48, 64, 256)
# The label: round-capped strokes on a 64-unit tile, as crates/api/assets/docs-icon.svg draws "API". Up to 32 px,
# where strokes blur, a 7-pixel bitmap font instead; up to 24 px the label alone fills the tile.
COMPACT_MAX = 24
BITMAP_MAX = 32
STROKE = 4.6
CAP = 15.0
LABEL_TOP = 37.5
LABEL_CX = 32.4
KERNING = {"AP": 4.2, "PI": 4.94, "MC": 6.2, "CP": 6.0}
BITMAP_GLYPHS = {
    "A": (".#.", "#.#", "#.#", "###", "#.#", "#.#", "#.#"),
    "C": (".##", "#..", "#..", "#..", "#..", "#..", ".##"),
    "I": ("###", ".#.", ".#.", ".#.", ".#.", ".#.", "###"),
    "M": ("#...#", "##.##", "#.#.#", "#.#.#", "#...#", "#...#", "#...#"),
    "P": ("##.", "#.#", "#.#", "##.", "#..", "#..", "#.."),
}
BOLD_BITMAP_GLYPHS = {
    "A": (".###.", "##.##", "##.##", "#####", "##.##", "##.##", "##.##"),
    "C": (".####", "##...", "##...", "##...", "##...", "##...", ".####"),
    "I": ("####", ".##.", ".##.", ".##.", ".##.", ".##.", "####"),
    "M": ("##...##", "###.###", "##.#.##", "##.#.##", "##...##", "##...##", "##...##"),
    "P": ("####.", "##.##", "##.##", "####.", "##...", "##...", "##..."),
}

TOP = (34, 52, 78)
BOTTOM = (7, 12, 21)
GLOW = (156, 210, 255)
BAR = (196, 228, 255)


def lerp(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, ...]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def diagonal_mask(size: int) -> Image.Image:
    v = Image.linear_gradient("L").resize((size, size), Image.BILINEAR)
    h = v.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
    return Image.blend(v, h, 0.5)


def arc(cx: float, cy: float, rx: float, ry: float, start: float, end: float) -> list[tuple[float, float]]:
    steps = 24
    return [(cx + rx * math.cos(math.radians(start + (end - start) * i / steps)),
             cy + ry * math.sin(math.radians(start + (end - start) * i / steps))) for i in range(steps + 1)]


# Each letter: its width and its strokes, x from 0, y from 0 (top) to CAP.
GLYPHS: dict[str, tuple[float, list[list[tuple[float, float]]]]] = {
    "A": (12.0, [[(0, CAP), (6, 0), (12, CAP)], [(3, 10.4), (9, 10.4)]]),
    "C": (9.7, [arc(5.5, CAP / 2, 5.5, CAP / 2, -40, -320)]),
    "I": (0.0, [[(0, 0), (0, CAP)]]),
    "M": (11.0, [[(0, CAP), (0, 0), (5.5, 8.5), (11, 0), (11, CAP)]]),
    "P": (6.86, [[(0, CAP), (0, 0), (3.16, 0), *arc(3.16, 3.7, 3.7, 3.7, -90, 90)[1:], (0, 7.4)]]),
}


def label_strokes(label: str) -> list[list[tuple[float, float]]]:
    """The label's strokes, centred on x = 0 with its top at y = 0."""
    gaps = [KERNING[a + b] for a, b in zip(label, label[1:])] + [0]
    x = -(sum(GLYPHS[ch][0] for ch in label) + sum(gaps)) / 2
    strokes = []
    for ch, gap in zip(label, gaps):
        w, lines = GLYPHS[ch]
        strokes += [[(x + px, py) for px, py in line] for line in lines]
        x += w + gap
    return strokes


def draw_strokes(draw: ImageDraw.ImageDraw, strokes: list[list[tuple[float, float]]], unit: float, x0: float,
                 y0: float, width: float) -> None:
    """Round caps and joins: a disc at every step along each stroke."""
    r = width * unit / 2
    for line in strokes:
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            steps = max(1, math.ceil(math.hypot(bx - ax, by - ay) * unit / (r * 0.15)))
            for i in range(steps + 1):
                x = x0 + (ax + (bx - ax) * i / steps) * unit
                y = y0 + (ay + (by - ay) * i / steps) * unit
                draw.ellipse((x - r, y - r, x + r, y + r), fill=255)


def draw_bitmap_label(img: Image.Image, label: str, cy: float) -> None:
    glyphs = BITMAP_GLYPHS if img.width <= 16 else BOLD_BITMAP_GLYPHS
    rows = [".".join(glyphs[ch][r] for ch in label) for r in range(7)]
    x0, y0 = (img.width - len(rows[0]) + 1) // 2, round(cy - len(rows) / 2)
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell == "#":
                img.putpixel((x0 + x, y0 + y), BAR + (255,))


def render(size: int, bars: int, label: str | None = None) -> Image.Image:
    s = size * 4
    margin = round(s * 0.07)
    box = (margin, margin, s - margin, s - margin)
    radius = round((s - 2 * margin) * 0.235)

    shape = Image.new("L", (s, s), 0)
    ImageDraw.Draw(shape).rounded_rectangle(box, radius, fill=255)

    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    shadow = Image.new("L", (s, s), 0)
    off = round(s * 0.018)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (box[0], box[1] + off, box[2], box[3] + off), radius, fill=120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(s * 0.022))
    img.paste(Image.new("RGBA", (s, s), (2, 4, 8, 255)), (0, 0), shadow)

    body = Image.composite(
        Image.new("RGBA", (s, s), TOP + (255,)),
        Image.new("RGBA", (s, s), BOTTOM + (255,)),
        diagonal_mask(s).point(lambda p: 255 - p),
    )
    img.paste(body, (0, 0), shape)

    fade = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((s, s), Image.BILINEAR)
    fade = fade.point(lambda p: int(p * 0.12 if p > 128 else p * 0.12 * (p / 128) ** 2))
    fade = ImageChops.multiply(fade, shape)
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), fade)))

    bloom = Image.new("L", (s, s), 0)
    ImageDraw.Draw(bloom).ellipse(
        (s * 0.45, s * 0.55, s * 1.05, s * 1.15), fill=38)
    bloom = ImageChops.multiply(bloom.filter(ImageFilter.GaussianBlur(s * 0.09)), shape)
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), GLOW[0]), Image.new("L", (s, s), GLOW[1]), Image.new("L", (s, s), GLOW[2]), bloom)))

    rim_w = max(2, round(s * 0.014))
    inner = Image.new("L", (s, s), 0)
    ImageDraw.Draw(inner).rounded_rectangle(
        (box[0] + rim_w, box[1] + rim_w, box[2] - rim_w, box[3] - rim_w), radius - rim_w, fill=255)
    ring = ImageChops.subtract(shape, inner)
    vgrad = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((s, s), Image.BILINEAR)
    ring = ImageChops.multiply(ring, vgrad.point(lambda p: 25 + int(p * 0.45)))
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), ring)))

    inner_side = s - 2 * margin
    cx, cy = s / 2, s / 2
    scale = 1.0
    compact = label is not None and size <= COMPACT_MAX
    span = inner_side * 0.58
    if label and not compact:
        cy, scale, span = margin + inner_side * 0.318, 0.8, inner_side * 0.534
    pitch = span / (bars - 1)
    bar_w = pitch * 0.52
    heights = []
    for i in range(bars):
        t = (i / (bars - 1)) * 2 - 1
        env = math.cos(t * math.pi / 2) ** 1.2   # 0 at the edges, 1 at the centre
        ripple = 0.82 + 0.18 * math.cos(i * 2.1)
        heights.append(max(0.16, env * ripple) * inner_side * 0.52 * scale)

    wave = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(wave)
    if not compact:
        for i, h in enumerate(heights):
            x = cx - span / 2 + i * pitch
            d.rounded_rectangle((x - bar_w / 2, cy - h / 2, x + bar_w / 2, cy + h / 2), bar_w / 2, fill=255)
    letters = Image.new("L", (s, s), 0)
    if label and size > BITMAP_MAX:
        unit = s / 64
        draw_strokes(ImageDraw.Draw(letters), label_strokes(label), unit, LABEL_CX * unit, LABEL_TOP * unit, STROKE)

    glow = wave.filter(ImageFilter.GaussianBlur(s * 0.03)).point(lambda p: min(255, int(p * 1.1)))
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), GLOW[0]), Image.new("L", (s, s), GLOW[1]), Image.new("L", (s, s), GLOW[2]), glow)))

    shade = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((s, s), Image.BILINEAR)
    bar_alpha = ImageChops.multiply(wave, shade.point(lambda p: 215 + int(p * 40 / 255)))
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), BAR[0]), Image.new("L", (s, s), BAR[1]), Image.new("L", (s, s), BAR[2]),
        ImageChops.lighter(bar_alpha, letters))))

    img = img.resize((size, size), Image.LANCZOS)
    if label and size <= BITMAP_MAX:
        draw_bitmap_label(img, label, size / 2 if compact else (LABEL_TOP + CAP / 2) * size / 64)
    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    images: dict[int, Image.Image] = {}
    for n in PNG_SIZES:
        # Fewer, fatter bars stay legible when tiny.
        images[n] = render(n, bars=5 if n <= 48 else 7)
        images[n].save(OUT / f"icon_{n}x{n}.png")
    images[1024].save(OUT / "icon.png")

    images[256].save(OUT / "icon.ico", sizes=[(n, n) for n in (16, 32, 48, 64, 128, 256)])
    images[1024].save(OUT / "icon.icns", sizes=[(n, n) for n in (16, 32, 64, 128, 256, 512, 1024)])

    for name, label in LABELS.items():
        labelled = {n: render(n, bars=5, label=label) for n in LABEL_SIZES}
        for n, image in labelled.items():
            image.save(OUT / f"{name}_{n}x{n}.png")
        labelled[256].save(OUT / f"{name}.ico", sizes=[(n, n) for n in LABEL_SIZES],
                           append_images=[labelled[n] for n in LABEL_SIZES if n != 256])
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
