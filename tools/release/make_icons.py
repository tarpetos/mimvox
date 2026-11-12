#!/usr/bin/env python3
"""Generate the app icon set into tools/release/icons/.

Design: a sound wave inside a rounded "glass" square (indigo -> cyan gradient, frosted top highlight,
soft inner rim, glowing white wave bars).

Run from anywhere (Pillow is fetched on the fly, nothing is installed globally):

    uv run --with pillow python tools/release/make_icons.py

Outputs (all committed, so a normal build does not need to run this):
    icons/icon_<N>x<N>.png   N = 16 32 48 64 128 256 512 1024
    icons/icon.png       1024 px master
    icons/icon.ico       Windows (16..256 embedded)
    icons/icon.icns      macOS (16..1024 embedded)
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent / "icons"
PNG_SIZES = (16, 32, 48, 64, 128, 256, 512, 1024)

TOP = (34, 52, 78)        # lit steel-navy (top-left)
BOTTOM = (7, 12, 21)      # deep navy #0B1119-ish (bottom-right)
GLOW = (156, 210, 255)    # Ice accent #9CD2FF
BAR = (196, 228, 255)     # pale ice for the wave bars


def lerp(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, ...]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def diagonal_mask(size: int) -> Image.Image:
    """Top-left (black) to bottom-right (white) 'L' gradient."""
    v = Image.linear_gradient("L").resize((size, size), Image.BILINEAR)
    h = v.rotate(90).transpose(Image.FLIP_LEFT_RIGHT)
    return Image.blend(v, h, 0.5)


def render(size: int, bars: int) -> Image.Image:
    """Render one RGBA icon of `size` px at 4x supersampling."""
    s = size * 4
    margin = round(s * 0.07)
    box = (margin, margin, s - margin, s - margin)
    radius = round((s - 2 * margin) * 0.235)

    shape = Image.new("L", (s, s), 0)
    ImageDraw.Draw(shape).rounded_rectangle(box, radius, fill=255)

    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))

    # Soft drop shadow for depth.
    shadow = Image.new("L", (s, s), 0)
    off = round(s * 0.018)
    ImageDraw.Draw(shadow).rounded_rectangle(
        (box[0], box[1] + off, box[2], box[3] + off), radius, fill=120)
    shadow = shadow.filter(ImageFilter.GaussianBlur(s * 0.022))
    img.paste(Image.new("RGBA", (s, s), (2, 4, 8, 255)), (0, 0), shadow)

    # Glass body: diagonal gradient.
    body = Image.composite(
        Image.new("RGBA", (s, s), TOP + (255,)),
        Image.new("RGBA", (s, s), BOTTOM + (255,)),
        diagonal_mask(s).point(lambda p: 255 - p),
    )
    img.paste(body, (0, 0), shape)

    # Frosted highlight over the upper half (white fading to nothing).
    fade = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((s, s), Image.BILINEAR)
    fade = fade.point(lambda p: int(p * 0.12 if p > 128 else p * 0.12 * (p / 128) ** 2))
    fade = ImageChops.multiply(fade, shape)
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), fade)))

    # Light lower-right bloom for a liquid feel.
    bloom = Image.new("L", (s, s), 0)
    ImageDraw.Draw(bloom).ellipse(
        (s * 0.45, s * 0.55, s * 1.05, s * 1.15), fill=38)
    bloom = ImageChops.multiply(bloom.filter(ImageFilter.GaussianBlur(s * 0.09)), shape)
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), GLOW[0]), Image.new("L", (s, s), GLOW[1]), Image.new("L", (s, s), GLOW[2]), bloom)))

    # Rim: bright at the top, faint at the bottom (a 1.5% stroke inside the shape).
    rim_w = max(2, round(s * 0.014))
    inner = Image.new("L", (s, s), 0)
    ImageDraw.Draw(inner).rounded_rectangle(
        (box[0] + rim_w, box[1] + rim_w, box[2] - rim_w, box[3] - rim_w), radius - rim_w, fill=255)
    ring = ImageChops.subtract(shape, inner)
    vgrad = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((s, s), Image.BILINEAR)
    ring = ImageChops.multiply(ring, vgrad.point(lambda p: 25 + int(p * 0.45)))
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), Image.new("L", (s, s), 255), ring)))

    # Sound wave: capsules with a smooth envelope, tallest at the centre.
    cx, cy = s / 2, s / 2
    span = (s - 2 * margin) * 0.58
    pitch = span / (bars - 1)
    bar_w = pitch * 0.52
    heights = []
    for i in range(bars):
        t = (i / (bars - 1)) * 2 - 1            # -1 .. 1
        env = math.cos(t * math.pi / 2) ** 1.2   # 0 at the edges, 1 at the centre
        ripple = 0.82 + 0.18 * math.cos(i * 2.1)
        heights.append(max(0.16, env * ripple) * (s - 2 * margin) * 0.52)

    wave = Image.new("L", (s, s), 0)
    d = ImageDraw.Draw(wave)
    for i, h in enumerate(heights):
        x = cx - span / 2 + i * pitch
        d.rounded_rectangle((x - bar_w / 2, cy - h / 2, x + bar_w / 2, cy + h / 2), bar_w / 2, fill=255)

    glow = wave.filter(ImageFilter.GaussianBlur(s * 0.03)).point(lambda p: min(255, int(p * 1.1)))
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), GLOW[0]), Image.new("L", (s, s), GLOW[1]), Image.new("L", (s, s), GLOW[2]), glow)))

    # White bars, slightly translucent toward the bottom to read as glass.
    shade = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((s, s), Image.BILINEAR)
    bar_alpha = ImageChops.multiply(wave, shade.point(lambda p: 215 + int(p * 40 / 255)))
    img = Image.alpha_composite(img, Image.merge("RGBA", (
        Image.new("L", (s, s), BAR[0]), Image.new("L", (s, s), BAR[1]), Image.new("L", (s, s), BAR[2]), bar_alpha)))

    return img.resize((size, size), Image.LANCZOS)


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
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
