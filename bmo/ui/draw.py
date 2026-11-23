"""Shared PIL drawing helpers for the kiosk windows."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

CARD_BG = "#FFFFFF"
CARD_BORDER = "#FFFFFF"
CARD_SHADOW = "#7BAADF"
TEXT_MAIN = "#0F172A"


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def blend(c1, c2, t):
    return tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))


_font_cache = {}
_FONTS_BOLD = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONTS_REG = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def get_font(size=14, bold=False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    for p in (_FONTS_BOLD if bold else _FONTS_REG):
        if os.path.exists(p):
            try:
                f = ImageFont.truetype(p, size)
                _font_cache[key] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def draw_text(draw, x, y, text, size=14, fill=TEXT_MAIN, bold=False, anchor="lm"):
    if not text:
        return
    f = get_font(size, bold)
    try:
        draw.text((x, y), text, font=f, fill=hex2rgb(fill), anchor=anchor)
    except Exception:
        draw.text((x, y), text, font=f, fill=hex2rgb(fill))


def draw_glass_card(img, x, y, w, h, r=26, bg_color=CARD_BG, shadow_color=CARD_SHADOW,
                    border_color=CARD_BORDER, shadow_offset=(0, 9),
                    shadow_blur=20, shadow_alpha=115):
    pad = shadow_blur * 2
    shadow_img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    sc = hex2rgb(shadow_color)
    sy0 = pad + shadow_offset[1]
    ImageDraw.Draw(shadow_img).rounded_rectangle(
        [pad, sy0, pad + w, sy0 + h], radius=r, fill=sc + (shadow_alpha,))
    shadow_blurred = shadow_img.filter(ImageFilter.GaussianBlur(shadow_blur))
    img.paste(shadow_blurred, (x - pad, y - pad), shadow_blurred)

    card_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(card_img)
    top, bot = (255, 255, 255), (244, 250, 255)
    for row in range(h):
        c = blend(top, bot, row / max(h - 1, 1))
        cdraw.line([(0, row), (w, row)], fill=c + (252,))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    cdraw.rounded_rectangle([0, 0, w - 1, h - 1], radius=r,
                            outline=hex2rgb(border_color) + (255,), width=2)
    img.paste(card_img, (x, y), mask)


_gradient_pill_cache = {}


def draw_gradient_pill(img, x, y, w, h, c1_hex, c2_hex, outline_col=None):
    if w <= 0 or h <= 0:
        return
    key = (w, h, c1_hex, c2_hex, outline_col)
    cached = _gradient_pill_cache.get(key)
    if cached is not None:
        img.paste(cached[0], (x, y), cached[1])
        return
    r = h // 2
    c1, c2 = hex2rgb(c1_hex), hex2rgb(c2_hex)
    t = np.linspace(0, 1, max(w, 1), dtype=np.float32)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for i in range(3):
        arr[:, :, i] = (c1[i] * (1 - t) + c2[i] * t).astype(np.uint8)
    arr[:, :, 3] = 255
    grad = Image.fromarray(arr, "RGBA")
    if outline_col:
        ImageDraw.Draw(grad).rounded_rectangle(
            [0, 0, w - 1, h - 1], radius=r, outline=hex2rgb(outline_col) + (200,), width=1)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    _gradient_pill_cache[key] = (grad.copy(), mask.copy())
    img.paste(grad, (x, y), mask)


def create_ambient_background(W, H):
    bg_top, bg_mid, bg_bot = (210, 232, 254), (228, 243, 254), (192, 222, 253)
    bg = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / max(H - 1, 1)
        c = blend(bg_top, bg_mid, t / 0.35) if t < 0.35 else blend(bg_mid, bg_bot, (t - 0.35) / 0.65)
        draw.line([(0, y), (W, y)], fill=c + (255,))

    glows = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    g = ImageDraw.Draw(glows)
    g.ellipse([W - 680, -260, W + 360, 580], fill=(255, 255, 255, 140))
    g.ellipse([-220, -220, 680, 520], fill=(165, 220, 255, 110))
    g.ellipse([-150, 400, 450, 900], fill=(180, 225, 255, 80))
    g.ellipse([-320, H - 560, 520, H + 320], fill=(145, 202, 253, 95))
    g.ellipse([W - 720, H - 460, W + 260, H + 260], fill=(155, 212, 254, 95))
    bg.alpha_composite(glows.filter(ImageFilter.GaussianBlur(85)))

    curves = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    c = ImageDraw.Draw(curves)
    c.ellipse([W - 760, -190, W + 320, 560], fill=(238, 248, 255, 90))
    c.ellipse([-260, H - 410, 490, H + 230], fill=(232, 245, 255, 80))
    bg.alpha_composite(curves.filter(ImageFilter.GaussianBlur(36)))
    return bg.convert("RGB")
