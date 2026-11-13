"""
SHIELD FSL — Shared UI Kit (ui_kit.py)
Extracted neumorphic glassmorphic color palette, PIL drawing helpers,
font cache, and ambient background generator.
Consolidated from main.py and non_fsl_window.py.
"""

import os
import math
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont

# ─────────────────────────────────────────────
#  Modern Colour Palette (Project B.M.O)
# ─────────────────────────────────────────────
# Tkinter hex colours & Neumorphic Palette (Modernized Sky Blue)
BG_LIGHT      = "#D2E8FE"   # Luminous fresh sky blue canvas
BG_CENTER     = "#E4F3FE"   # Soft center ambient highlight
BG_DEEP       = "#C0DEFD"   # Rich pastel blue

CARD_BG       = "#FFFFFF"   # Luminous frosted glass card body
CARD_BORDER   = "#FFFFFF"   # Specular white rim highlight
CARD_RIM_SUB  = "#D2E4F8"   # Subtle bottom rim line
CARD_SHADOW   = "#7BAADF"   # Rich sky-blue diffused drop shadow (NOT grey!)

BLUE_PRIMARY  = "#1D68FE"   # Vibrant electric cobalt blue
BLUE_LIGHT    = "#4F9BFE"   # Bright electric azure
BLUE_CYAN     = "#38BDF8"   # Sky cyan highlight
BLUE_DARK     = "#1050E2"   # Deep active cobalt

TEXT_MAIN     = "#0F172A"   # Crisp high-contrast obsidian slate
TEXT_MUTED    = "#5A718F"   # Cool slate gray for subtitles/labels
TEXT_FAINT    = "#8FA8C4"   # Refined placeholder text
WHITE         = "#FFFFFF"

GREEN_ONLINE  = "#10B981"   # Mint green status dot
GREEN_BG      = "#ECFDF5"   # Mint pill background
GREEN_BORDER  = "#A7F3D0"   # Mint pill border
GREEN_TEXT    = "#059669"   # Mint pill text

RED_BTN       = "#EF4444"
RED_BG        = "#FEE2E2"
RED_BORDER    = "#FECACA"

BUBBLE_FSL    = "#DBEAFE"
BUBBLE_FSL_BD = "#93C5FD"
BUBBLE_PART   = "#E0F2FE"
BUBBLE_PART_BD= "#7DD3FC"

DOCK_DARK_BG  = (22, 36, 60, 185)  # Dark translucent glass overlay on video

# Legacy aliases for compatibility
BG_TOP      = BG_LIGHT
BG_BOT      = BG_DEEP
CARD        = CARD_BG
CARD_DARK   = "#0B132B"
SHADOW      = CARD_SHADOW
SHADOW_D    = "#DCEBF9"
ICON_BG     = "#E6F1FC"
ICON_BG_ON  = "#DBEAFE"
TXT_DARK    = TEXT_MAIN
TXT_MID     = TEXT_MUTED
TXT_FAINT   = TEXT_FAINT
ACC_BLUE    = BLUE_PRIMARY
ACC_BLUE2   = BLUE_LIGHT
ACC_CYAN    = BLUE_CYAN
ACC_NAVY    = "#16243C"
ACC_GREEN   = GREEN_ONLINE
ACC_RED     = "#EF4444"
ACC_AMBER   = "#F59E0B"
ACC_PURPLE  = "#8B5CF6"
BMO_GREEN   = "#22C55E"


# ─────────────────────────────────────────────
#  Colour Conversion & Blending
# ─────────────────────────────────────────────
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _blend(c1, c2, t):
    return tuple(int(c1[i]*(1-t) + c2[i]*t) for i in range(3))


def rgb2hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"


# ─────────────────────────────────────────────
#  PIL Drawing Helpers
# ─────────────────────────────────────────────
def draw_glass_card(img: Image.Image, x, y, w, h, r=26,
                    bg_color=CARD_BG, shadow_color=CARD_SHADOW,
                    border_color=CARD_BORDER, shadow_offset=(0, 9),
                    shadow_blur=20, shadow_alpha=115):
    """
    Renders an authentic neumorphic-glassmorphic card:
    - Multi-layered soft diffused ambient sky-blue shadow (floating depth)
    - Crisp specular glass white rim highlight
    - Clean luminous white-to-ice gradient card fill
    """
    pad = shadow_blur * 2
    sw = w + pad * 2
    sh = h + pad * 2

    shadow_img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow_img)
    sc = hex2rgb(shadow_color)

    sx0 = pad
    sy0 = pad + shadow_offset[1]
    sdraw.rounded_rectangle([sx0, sy0, sx0 + w, sy0 + h], radius=r, fill=sc + (shadow_alpha,))
    shadow_blurred = shadow_img.filter(ImageFilter.GaussianBlur(shadow_blur))

    img.paste(shadow_blurred, (x - pad, y - pad), shadow_blurred)

    card_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(card_img)

    bg_top = (255, 255, 255)
    bg_bot = (244, 250, 255)
    for row in range(h):
        t = row / max(h - 1, 1)
        c = _blend(bg_top, bg_bot, t)
        cdraw.line([(0, row), (w, row)], fill=c + (252,))

    mask = Image.new("L", (w, h), 0)
    m_draw = ImageDraw.Draw(mask)
    m_draw.rounded_rectangle([0, 0, w, h], radius=r, fill=255)

    # Crisp specular white rim highlight
    b_col = hex2rgb(border_color)
    cdraw.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, outline=b_col + (255,), width=2)

    img.paste(card_img, (x, y), mask)


def draw_shadow_card(draw: ImageDraw.ImageDraw, x, y, w, h, r=18,
                     bg=CARD, shadow_color=SHADOW, blur=8, offset_y=4):
    """Draw a card with a clean, smooth multi-pass diffused drop shadow."""
    bgc = hex2rgb(bg)
    if shadow_color:
        sc = hex2rgb(shadow_color)
        soft_outer = _blend(sc, hex2rgb(BG_BOT), 0.55)
        _rounded_rect(draw, x - 1, y + offset_y + 2, w + 2, h + 2, r + 2, soft_outer)
        _rounded_rect(draw, x, y + offset_y, w, h, r, sc)
    _rounded_rect(draw, x, y, w, h, r, bgc)


def _rounded_rect(draw, x, y, w, h, r, fill):
    """Fill a rounded rectangle on a PIL ImageDraw."""
    r = max(0, int(r))
    x, y, w, h = int(x), int(y), int(w), int(h)
    draw.rectangle([x + r, y, x + w - r, y + h], fill=fill)
    draw.rectangle([x, y + r, x + w, y + h - r], fill=fill)
    draw.ellipse([x, y, x + 2 * r, y + 2 * r], fill=fill)
    draw.ellipse([x + w - 2 * r, y, x + w, y + 2 * r], fill=fill)
    draw.ellipse([x, y + h - 2 * r, x + 2 * r, y + h], fill=fill)
    draw.ellipse([x + w - 2 * r, y + h - 2 * r, x + w, y + h], fill=fill)


def draw_pill(draw, x, y, w, h, fill):
    r = h // 2
    _rounded_rect(draw, x, y, w, h, r, hex2rgb(fill) if isinstance(fill, str) else fill)


def draw_pill_outline(draw, x, y, w, h, outline, width=2):
    r = h // 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=r,
                           outline=hex2rgb(outline) if isinstance(outline, str) else outline, width=width)


_gradient_pill_cache = {}


def draw_gradient_pill(img, x, y, w, h, c1_hex, c2_hex, outline_col=None):
    if w <= 0 or h <= 0:
        return
    cache_key = (w, h, c1_hex, c2_hex, outline_col)
    cached = _gradient_pill_cache.get(cache_key)
    if cached is not None:
        img.paste(cached[0], (x, y), cached[1])
        return
    r = h // 2
    c1, c2 = hex2rgb(c1_hex), hex2rgb(c2_hex)
    # Vectorized gradient via numpy — replaces slow per-pixel Python loop
    t = np.linspace(0, 1, max(w, 1), dtype=np.float32)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = (c1[0] * (1 - t) + c2[0] * t).astype(np.uint8)
    arr[:, :, 1] = (c1[1] * (1 - t) + c2[1] * t).astype(np.uint8)
    arr[:, :, 2] = (c1[2] * (1 - t) + c2[2] * t).astype(np.uint8)
    arr[:, :, 3] = 255
    grad = Image.fromarray(arr, "RGBA")
    if outline_col:
        gd = ImageDraw.Draw(grad)
        gd.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, outline=hex2rgb(outline_col) + (200,), width=1)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    _gradient_pill_cache[cache_key] = (grad.copy(), mask.copy())
    img.paste(grad, (x, y), mask)


def gradient_pill(draw, x, y, w, h, c1, c2):
    """Horizontal gradient pill."""
    r = h // 2
    grad = Image.new("RGB", (w, h))
    gd = ImageDraw.Draw(grad)
    for i in range(w):
        t = i / max(w - 1, 1)
        gc = _blend(hex2rgb(c1), hex2rgb(c2), t)
        gd.line([(i, 0), (i, h)], fill=gc)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    return grad, mask


# ─────────────────────────────────────────────
#  Font Helpers (PIL + Tkinter)
# ─────────────────────────────────────────────
_font_cache = {}


def _get_font(size=14, bold=False):
    """PIL TrueType font loader with Windows fallbacks, cached."""
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    candidates_bold = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    candidates_reg = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    candidates = candidates_bold if bold else candidates_reg
    for p in candidates:
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


def _get_tk_font(size=14, bold=False):
    weight = "bold" if bold else "normal"
    return ("Segoe UI", max(8, int(size)), weight)


def _draw_text(draw, x, y, text, size=14, fill=TEXT_MAIN, bold=False, anchor="mm"):
    if not text:
        return
    f = _get_font(size, bold)
    try:
        draw.text((x, y), text, font=f, fill=hex2rgb(fill), anchor=anchor)
    except Exception:
        draw.text((x, y), text, font=f, fill=hex2rgb(fill))


def _round_rect_pts(x0, y0, x1, y1, r):
    r = min(max(1, r), abs(x1 - x0) // 2, abs(y1 - y0) // 2)
    return [
        x0 + r, y0,
        x1 - r, y0,
        x1, y0,
        x1, y0 + r,
        x1, y1 - r,
        x1, y1,
        x1 - r, y1,
        x0 + r, y1,
        x0, y1,
        x0, y1 - r,
        x0, y0 + r,
        x0, y0
    ]


# ─────────────────────────────────────────────
#  Ambient Background
# ─────────────────────────────────────────────
def create_ambient_background(W, H):
    """Generates a rich, luminous modern sky-blue neumorphic ambient background."""
    bg_top = (210, 232, 254)   # #D2E8FE - luminous fresh sky blue
    bg_mid = (228, 243, 254)   # #E4F3FE - soft radiant center
    bg_bot = (192, 222, 253)   # #C0DEFD - rich pastel blue

    bg = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / max(H - 1, 1)
        if t < 0.35:
            c = _blend(bg_top, bg_mid, t / 0.35)
        else:
            c = _blend(bg_mid, bg_bot, (t - 0.35) / 0.65)
        draw.line([(0, y), (W, y)], fill=c + (255,))

    # Luminous ambient lighting blooms
    glows = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glows)
    gdraw.ellipse([W - 680, -260, W + 360, 580], fill=(255, 255, 255, 140))
    gdraw.ellipse([-220, -220, 680, 520], fill=(165, 220, 255, 110))
    gdraw.ellipse([-150, 400, 450, 900], fill=(180, 225, 255, 80))
    gdraw.ellipse([-320, H - 560, 520, H + 320], fill=(145, 202, 253, 95))
    gdraw.ellipse([W - 720, H - 460, W + 260, H + 260], fill=(155, 212, 254, 95))
    glows = glows.filter(ImageFilter.GaussianBlur(85))
    bg.alpha_composite(glows)

    # Soft 3D Neumorphic ambient wave curves
    curves = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(curves)
    cdraw.ellipse([W - 760, -190, W + 320, 560], fill=(238, 248, 255, 90))
    cdraw.ellipse([-260, H - 410, 490, H + 230], fill=(232, 245, 255, 80))
    curves = curves.filter(ImageFilter.GaussianBlur(36))
    bg.alpha_composite(curves)

    return bg.convert("RGB")
