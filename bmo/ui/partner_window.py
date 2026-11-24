"""
non_fsl_window.py - Partner Kiosk (layered-canvas rewrite)

What changed vs. the old single-surface version
-----------------------------------------------
* Self-scheduled render loop that honours settings["partner_fps"] (no more
  hard-coded 10 FPS cap in main.py, and it no longer runs *inside* the FSL tick).
* Skips rendering entirely while the window is minimised / hidden.
* Static shell is scaled ONCE (on resize) and shown as a canvas background item.
* Dynamic parts are separate canvas image items (camera, chat, sentence strip,
  palette, caption/badges) that are re-rendered ONLY when their state key changes.
* Camera comes straight from the capture thread (no ML dependency, no PIL copy):
  cv2 resize -> corner fill -> PhotoImage.paste(), only when a new frame arrived.
* Palette / token tiles are cached per (sign, hover) state.
* No more `from main import ...` (that re-imported main.py as a 2nd module:
  duplicate model load + duplicate camera thread). Callbacks are injected instead.
* PERF=1 prints partner timings once per second.
"""
import os
import sys
import time
import collections
import traceback
import functools
import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont

from bmo.coop_bus import coop_hub, load_icon_palette
from bmo.sentence import translate_sentence
from bmo.settings import settings as _settings
from bmo.ui.draw import (
    hex2rgb, get_font as _get_font, draw_text,
    draw_glass_card, draw_gradient_pill, create_ambient_background,
)

_draw_text = functools.partial(draw_text, anchor="mm")

from bmo.config import ASSETS_DIR

# ── Paths ──────────────────────────────────────────────────────────────────
# ASSETS_DIR comes from bmo.config

ENABLE_PERF = os.environ.get("PERF") == "1"
MIRROR_FEED = True            # True = same mirrored view the signer sees. False = face-to-face view.

# ── Color Palette (Project B.M.O) ───────────────────────────────────────────
BG_LIGHT      = "#D2E8FE"
CARD_BG       = "#FFFFFF"
CARD_BORDER   = "#FFFFFF"
CARD_SHADOW   = "#7BAADF"

BLUE_PRIMARY  = "#1D68FE"
BLUE_CYAN     = "#38BDF8"

TEXT_MAIN     = "#0F172A"
TEXT_MUTED    = "#5A718F"
TEXT_FAINT    = "#8FA8C4"
WHITE         = "#FFFFFF"

GREEN_ONLINE  = "#10B981"
GREEN_BG      = "#ECFDF5"
GREEN_BORDER  = "#A7F3D0"
GREEN_TEXT    = "#059669"

RED_BTN       = "#EF4444"
RED_BG        = "#FEE2E2"
RED_BORDER    = "#FECACA"

BUBBLE_FSL    = "#DBEAFE"
BUBBLE_FSL_BD = "#93C5FD"
BUBBLE_PART   = "#E0F2FE"
BUBBLE_PART_BD= "#7DD3FC"

CAM_DARK      = (11, 19, 43)   # #0B132B monitor container colour


# ── Color & Drawing Helpers ────────────────────────────────────────────────
def _get_tk_font(size=14, bold=False):
    return ("Segoe UI", max(8, int(size)), "bold" if bold else "normal")

def _round_rect_pts(x0, y0, x1, y1, r):
    r = min(max(1, r), abs(x1 - x0) // 2, abs(y1 - y0) // 2)
    return [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
            x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]



class _Avg:
    def __init__(self, n=30):
        self.d = collections.deque(maxlen=n)
    def add(self, ms):
        self.d.append(ms)
    def get(self):
        return sum(self.d) / len(self.d) if self.d else 0.0


# ── Non-FSL Partner App ─────────────────────────────────────────────────────
class NonFSLWindow:
    DESIGN_W  = 1920
    DESIGN_H  = 1080
    W         = 1920
    H         = 1080

    PAD       = 28
    TOP_Y     = 16
    TOP_H     = 92

    CONTENT_Y = 124
    CONTENT_H = 550
    LEFT_W    = 1084
    RIGHT_X   = 1132
    RIGHT_W   = 760

    STRIP_Y   = 690
    STRIP_H   = 140

    PALETTE_Y = 846
    PALETTE_H = 218

    # Camera container (dark monitor) inside the left card
    CAM_X = PAD + 18
    CAM_Y = CONTENT_Y + 18
    CAM_W = LEFT_W - 36
    CAM_H = CONTENT_H - 36

    # Right column
    BMO_H      = 104
    CHAT_CARD_Y = CONTENT_Y + BMO_H + 16
    CHAT_CARD_H = CONTENT_H - BMO_H - 16

    # Dynamic layer rectangles (design coords). Each sits INSIDE its card so the
    # rounded corners / drop shadow stay in the static shell.
    CHAT_LX, CHAT_LY = RIGHT_X + 16, CHAT_CARD_Y + 48
    CHAT_LW, CHAT_LH = RIGHT_W - 32, CHAT_CARD_H - 48 - 16

    STRIP_LX, STRIP_LY = PAD + 16, STRIP_Y + 12
    STRIP_LW, STRIP_LH = W - PAD * 2 - 32, STRIP_H - 24

    PAL_LX, PAL_LY = PAD + 16, PALETTE_Y + 12
    PAL_LW, PAL_LH = W - PAD * 2 - 32, PALETTE_H - 24

    @classmethod
    def prebuild(cls):
        t = cls.__new__(cls)
        t._logo_rgba = t._logo_alpha = t._logo_rim = t._logo_rim_mask = t._logo_shadow = None
        t._logo_pad, t._logo_w, t._logo_h = 20, 50, 50
        t._load_logo()
        t.palette = load_icon_palette()
        t._cached_icons = {}
        t._load_palette_icons()
        return {
            "shell": t._pre_render_ui_shell(),
            "palette": t.palette,
            "icons": t._cached_icons,
        }

    def __init__(self, master=None, on_close_callback=None, get_cam_fn=None,
                 get_raw_frame_fn=None, speak_fn=None, ping_fn=None, pre=None):
        """
        get_raw_frame_fn(last_id) -> (frame_id, bgr_ndarray | None)   [preferred, no copy]
        get_cam_fn()              -> PIL.Image | None                  [legacy fallback]
        speak_fn(text), ping_fn() -> injected so we never `import main`.
        """
        if master is None:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel(master)
        if pre is not None:
            self.window.withdraw()

        self.window.title("Project B.M.O — Co-Op Communicator (Partner Kiosk)")
        self.window.configure(bg=BG_LIGHT)

        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        usable_w, usable_h = screen_w - 40, screen_h - 80
        if self.DESIGN_W > usable_w or self.DESIGN_H > usable_h:
            sc = min(usable_w / self.DESIGN_W, usable_h / self.DESIGN_H)
            init_w = max(800, int(self.DESIGN_W * sc))
            init_h = max(450, int(self.DESIGN_H * sc))
        else:
            init_w, init_h = self.DESIGN_W, self.DESIGN_H

        pos_x = max(0, (screen_w - init_w) // 2 + 25)
        pos_y = max(0, (screen_h - init_h) // 2 + 25)
        self.window.geometry(f"{init_w}x{init_h}+{pos_x}+{pos_y}")
        self.window.minsize(800, 450)
        self.window.resizable(True, True)

        self.on_close_callback = on_close_callback
        self.get_cam_fn = get_cam_fn
        self.get_raw_frame_fn = get_raw_frame_fn
        self._speak_fn = speak_fn
        self._ping_fn = ping_fn

        self.win_w, self.win_h = init_w, init_h
        self._recompute_scale()
        self._resize_job = None
        self._layout_ready = False

        # ── Canvas + layered items (z-order = creation order) ──
        self.canvas = tk.Canvas(self.window, width=init_w, height=init_h,
                                bg=BG_LIGHT, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.bg_id         = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.cam_id        = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.badge_live_id = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.badge_tag_id  = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.caption_id    = self.canvas.create_image(0, 0, anchor=tk.NW, state="hidden")
        self.chat_id       = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.strip_id      = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.palette_id    = self.canvas.create_image(0, 0, anchor=tk.NW)
        self.toast_bg_id   = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=BLUE_PRIMARY,
                                                        outline="", state="hidden")
        self.toast_text_id = self.canvas.create_text(0, 0, text="", font=_get_tk_font(15, True),
                                                     fill=WHITE, anchor="center", state="hidden")
        self._tk_refs = {}                 # keep PhotoImages alive
        self._toast_key = None
        self._toast_visible = False

        if pre is None:
            pre = NonFSLWindow.prebuild()
        self.palette = pre["palette"]
        self._cached_icons = pre["icons"]
        self._measure = ImageDraw.Draw(Image.new("L", (1, 1)))
        self.ui_shell_cache = pre["shell"]

        # ── State ──
        self.sentence_tokens = []
        self._chat_history = []
        self.latest_signer_text = ""

        self.notification_text = ""
        self.notification_end = 0.0

        self._send_hover = self._del_hover = self._clr_hover = False
        self._send_click_until = self._del_click_until = self._clr_click_until = 0.0
        self._hovered_palette_key = None
        self._hovered_strip_idx = None

        self._btn_send_rect = self._btn_del_rect = self._btn_clr_rect = (0, 0, 0, 0)
        self._palette_rects = []
        self._strip_item_rects = []

        # Caches
        self._keys = {"chat": None, "strip": None, "pal": None, "cap": None}
        self._tile_cache = {}
        self._mask_cache = {}
        self._send_glow_cache = {}
        self._cam_geo = None
        self._cam_tk = None
        self._cam_tk_size = None
        self._last_frame_id = -1

        # PERF
        self._perf = {n: _Avg() for n in ("total", "cam", "chat", "strip", "pal")}
        self._perf_frames = 0
        self._perf_last_print = time.perf_counter()
        self._last_render_pc = 0.0

        self._apply_layout()
        self._layout_ready = True

        self.window.bind("<Motion>", self._on_motion)
        self.window.bind("<Button-1>", self._on_click)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        coop_hub.register_partner_listener(self._on_signer_message_received)

        self._running = True
        self._next_t = time.perf_counter()
        self.window.after(50, self._tick)

    # ── Settings / scheduling ───────────────────────────────────────────────
    def _target_fps(self):
        try:
            v = int(_settings.get("partner_fps", 30)) if _settings else 30
        except Exception:
            v = 30
        return max(5, min(60, v))

    def _is_visible(self):
        try:
            return self.window.state() != "iconic" and bool(self.window.winfo_viewable())
        except tk.TclError:
            return False

    def _tick(self):
        if not self._running:
            return
        if self._is_visible():
            try:
                self._render(force=True)
            except Exception as e:
                print(f"[Partner Render Error] {e}")
                traceback.print_exc()
            interval = 1.0 / self._target_fps()
        else:
            interval = 0.25            # hidden / minimised: idle poll only
        t1 = time.perf_counter()
        self._next_t += interval
        if self._next_t < t1 - interval:   # fell behind: don't burst-catch-up
            self._next_t = t1
        delay = max(3, int((self._next_t - t1) * 1000))   # >=3 ms so the FSL window can interleave
        try:
            self.window.after(delay, self._tick)
        except tk.TclError:
            pass

    # ── Scale helpers ───────────────────────────────────────────────────────
    def _recompute_scale(self):
        self.ui_scale = min(self.win_w / self.DESIGN_W, self.win_h / self.DESIGN_H)
        self.render_w = max(1, int(self.DESIGN_W * self.ui_scale))
        self.render_h = max(1, int(self.DESIGN_H * self.ui_scale))
        self.ui_ox = (self.win_w - self.render_w) // 2
        self.ui_oy = (self.win_h - self.render_h) // 2

    def _on_canvas_resize(self, event):
        if event.width <= 100 or event.height <= 100:
            return
        if event.width == self.win_w and event.height == self.win_h:
            return
        self.win_w, self.win_h = event.width, event.height
        self._recompute_scale()
        if self._resize_job is not None:
            try:
                self.window.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.window.after(80, self._apply_layout)   # debounce drag-resize

    def _apply_layout(self):
        """(Re)scale the static shell once and invalidate every dynamic layer."""
        self._resize_job = None
        shell = self.ui_shell_cache
        if (self.render_w, self.render_h) != shell.size:
            scaled = shell.resize((self.render_w, self.render_h), Image.Resampling.BILINEAR)
        else:
            scaled = shell
        self._tk_refs["bg"] = ImageTk.PhotoImage(scaled)
        self.canvas.itemconfig(self.bg_id, image=self._tk_refs["bg"])
        self.canvas.coords(self.bg_id, self.ui_ox, self.ui_oy)

        for k in self._keys:
            self._keys[k] = None
        self._cam_geo = None
        self._cam_tk = None
        self._cam_tk_size = None
        self._last_frame_id = -1
        self._build_static_badges()

    def _set_layer(self, name, item_id, pil_img, x, y):
        """Scale a design-res layer to screen res and put it on the canvas item."""
        s = self.ui_scale
        if s != 1.0:
            w = max(1, int(pil_img.width * s))
            h = max(1, int(pil_img.height * s))
            pil_img = pil_img.resize((w, h), Image.Resampling.BILINEAR)
        tk_img = ImageTk.PhotoImage(pil_img)
        self._tk_refs[name] = tk_img
        self.canvas.itemconfig(item_id, image=tk_img)
        self.canvas.coords(item_id, self.ui_ox + int(x * s), self.ui_oy + int(y * s))

    # ── Logo / icons ────────────────────────────────────────────────────────
    def _load_logo(self):
        logo_path = os.path.join(ASSETS_DIR, 'shield_logo.png')
        if not os.path.exists(logo_path):
            logo_path = os.path.join(ASSETS_DIR, 'logo.png')
        if not os.path.exists(logo_path):
            return
        try:
            logo = Image.open(logo_path).convert("RGBA")
            target_h = 50
            aspect = logo.size[0] / max(logo.size[1], 1)
            target_w = int(target_h * aspect)
            logo_resized = logo.resize((target_w, target_h), Image.Resampling.BILINEAR)
            alpha = logo_resized.split()[3]
            rim_mask = alpha.filter(ImageFilter.MaxFilter(7))
            rim = Image.new("RGBA", (target_w, target_h), (60, 100, 180, 120))
            pad = 20
            shadow = Image.new("RGBA", (target_w + pad * 2, target_h + pad * 2), (0, 0, 0, 0))
            shadow.paste(Image.new("RGBA", (target_w, target_h), (15, 30, 60, 65)), (pad + 1, pad + 5), alpha)
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))
            self._logo_rgba, self._logo_alpha = logo_resized, alpha
            self._logo_rim, self._logo_rim_mask = rim, rim_mask
            self._logo_shadow, self._logo_pad = shadow, pad
            self._logo_w, self._logo_h = target_w, target_h
        except Exception as e:
            print(f"[!] Partner Kiosk logo load error: {e}")

    def _load_palette_icons(self):
        for item in self.palette:
            key, p = item["key"], item["icon_path"]
            try:
                im = Image.open(p).convert("RGBA")
                self._cached_icons[key] = {
                    "palette": im.resize((64, 64), Image.Resampling.LANCZOS),
                    "strip":   im.resize((38, 38), Image.Resampling.LANCZOS),
                    "bubble":  im.resize((26, 26), Image.Resampling.LANCZOS),
                }
            except Exception as e:
                print(f"[!] Error caching icon {key}: {e}")

    # ── Static shell (design resolution, drawn once) ────────────────────────
    def _pre_render_ui_shell(self):
        W, H = self.W, self.H
        shell = create_ambient_background(W, H)
        draw = ImageDraw.Draw(shell)

        # 1. Top bar
        bx, by, bw, bh = self.PAD, self.TOP_Y, W - self.PAD * 2, self.TOP_H
        draw_glass_card(shell, bx, by, bw, bh, r=26, shadow_blur=16, shadow_alpha=95)
        cy = by + bh // 2

        logo_x = bx + 28
        glow = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([0, 0, 72, 72], fill=hex2rgb("#93C5FD") + (120,))
        gdraw.ellipse([12, 12, 60, 60], fill=hex2rgb("#38BDF8") + (180,))
        glow = glow.filter(ImageFilter.GaussianBlur(12))
        shell.paste(glow, (logo_x, cy - 36), glow)
        if self._logo_shadow:
            shell.paste(self._logo_shadow, (logo_x - self._logo_pad, cy - self._logo_h // 2 - self._logo_pad),
                        self._logo_shadow)
        if self._logo_rim:
            shell.paste(self._logo_rim, (logo_x, cy - self._logo_h // 2), self._logo_rim_mask)
        if self._logo_rgba:
            shell.paste(self._logo_rgba, (logo_x, cy - self._logo_h // 2), self._logo_alpha)

        title_x = logo_x + 80
        f_title = _get_font(25, bold=True)
        draw.text((title_x, cy - 12), "Project ", font=f_title, fill=hex2rgb(TEXT_MAIN), anchor="lm")
        w_proj = draw.textlength("Project ", font=f_title)
        bmo_x = int(title_x + w_proj)
        draw.text((bmo_x, cy - 12), "B.M.O", font=f_title, fill=hex2rgb(BLUE_PRIMARY), anchor="lm")
        w_bmo = draw.textlength("B.M.O", font=f_title)

        badge_x, badge_y, badge_w, badge_h = int(bmo_x + w_bmo + 16), cy - 23, 80, 24
        draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                               radius=badge_h // 2, fill=hex2rgb(GREEN_BG), outline=hex2rgb(GREEN_BORDER), width=1)
        draw.ellipse([badge_x + 9, badge_y + 8, badge_x + 16, badge_y + 15], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, badge_x + 22, badge_y + 12, "ONLINE", size=10, fill=GREEN_TEXT, bold=True, anchor="lm")
        _draw_text(draw, title_x, cy + 16, "Partner Kiosk • Visual Two-Way Communication",
                   size=13, fill=TEXT_MUTED, anchor="lm")

        switch_w, switch_h = 240, 48
        switch_x, switch_y = W // 2 - switch_w // 2, cy - switch_h // 2
        draw.rounded_rectangle([switch_x, switch_y, switch_x + switch_w, switch_y + switch_h],
                               radius=switch_h // 2, fill=hex2rgb("#DDEBF9"), outline=hex2rgb("#CBDDF2"), width=1)
        draw_gradient_pill(shell, switch_x + 4, switch_y + 4, switch_w // 2, switch_h - 8, BLUE_PRIMARY, BLUE_CYAN)
        _draw_text(draw, switch_x + switch_w // 4 + 2, cy, "Partner", size=13, fill=WHITE, bold=True)
        _draw_text(draw, switch_x + 3 * switch_w // 4, cy, "FSL Linked", size=13, fill="#5A718F")

        fps_w, fps_h = 160, 42
        fps_x, fps_y = bx + bw - fps_w - 28, cy - fps_h // 2
        draw.rounded_rectangle([fps_x, fps_y, fps_x + fps_w, fps_y + fps_h],
                               radius=fps_h // 2, fill=hex2rgb("#F8FBFE"), outline=hex2rgb("#DCE8F6"), width=1)
        draw.ellipse([fps_x + 16, fps_y + 17, fps_x + 24, fps_y + 25], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, fps_x + 32, cy, "2-WAY CO-OP", size=12, fill=GREEN_TEXT, bold=True, anchor="lm")

        # 2. Left card + dark monitor container + idle placeholder
        draw_glass_card(shell, self.PAD, self.CONTENT_Y, self.LEFT_W, self.CONTENT_H,
                        r=26, shadow_blur=18, shadow_alpha=110)
        draw.rounded_rectangle([self.CAM_X, self.CAM_Y, self.CAM_X + self.CAM_W, self.CAM_Y + self.CAM_H],
                               radius=18, fill=CAM_DARK, outline=hex2rgb("#1E293B"), width=1)
        fcx, fcy = self.CAM_X + self.CAM_W // 2, self.CAM_Y + self.CAM_H // 2
        draw.rounded_rectangle([fcx - 36, fcy - 24, fcx + 36, fcy + 24], radius=8,
                               outline=hex2rgb(TEXT_MUTED), width=2)
        draw.ellipse([fcx - 14, fcy - 14, fcx + 14, fcy + 14], outline=hex2rgb(TEXT_MUTED), width=2)
        _draw_text(draw, fcx, fcy + 42, "SIGNER CAMERA FEED LIVE", size=15, fill=TEXT_MUTED, bold=True)

        # 3. Right header card (fully static): BMO companion
        rx, ry, rw, bmo_h = self.RIGHT_X, self.CONTENT_Y, self.RIGHT_W, self.BMO_H
        draw_glass_card(shell, rx, ry, rw, bmo_h, r=26, shadow_blur=16, shadow_alpha=95)
        av_d = 70
        av_x, av_y = rx + 20, ry + (bmo_h - av_d) // 2
        draw_gradient_pill(shell, av_x, av_y, av_d, av_d, "#38BDF8", "#2563EB", outline_col="#93C5FD")
        fc_cx, fc_cy = av_x + av_d // 2, av_y + av_d // 2
        draw.ellipse([fc_cx - 15, fc_cy - 4, fc_cx - 9, fc_cy + 2], fill=hex2rgb(TEXT_MAIN))
        draw.ellipse([fc_cx + 9, fc_cy - 4, fc_cx + 15, fc_cy + 2], fill=hex2rgb(TEXT_MAIN))
        draw.arc([fc_cx - 8, fc_cy + 2, fc_cx + 8, fc_cy + 10], start=20, end=160, fill=hex2rgb(TEXT_MAIN), width=2)
        tx = av_x + av_d + 16
        _draw_text(draw, tx, fc_cy - 12, "BMO Companion", size=19, fill=TEXT_MAIN, bold=True, anchor="lm")
        w_title = draw.textlength("BMO Companion", font=_get_font(19, bold=True))
        b_x, b_y, b_w, b_h = int(tx + w_title + 14), fc_cy - 20, 74, 22
        draw.rounded_rectangle([b_x, b_y, b_x + b_w, b_y + b_h], radius=b_h // 2,
                               fill=hex2rgb(GREEN_BG), outline=hex2rgb(GREEN_BORDER), width=1)
        draw.ellipse([b_x + 8, b_y + 7, b_x + 14, b_y + 13], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, b_x + 20, b_y + 11, "ONLINE", size=9, fill=GREEN_TEXT, bold=True, anchor="lm")
        _draw_text(draw, tx, fc_cy + 14, "Two-way communication history", size=12, fill=TEXT_MUTED, anchor="lm")
        eq_x = rx + rw - 72
        for i, amp in enumerate([4, 9, 15, 22, 13, 8, 4]):
            bxx = eq_x + i * 6
            draw.line([(bxx, fc_cy - amp), (bxx, fc_cy + amp)], fill=hex2rgb(BLUE_PRIMARY), width=3)

        # 4. Chat card (title static, bubbles dynamic)
        chat_y, chat_h = self.CHAT_CARD_Y, self.CHAT_CARD_H
        draw_glass_card(shell, rx, chat_y, rw, chat_h, r=26, shadow_blur=16, shadow_alpha=95)
        draw.ellipse([rx + 24, chat_y + 24, rx + 32, chat_y + 32], fill=hex2rgb(BLUE_PRIMARY))
        _draw_text(draw, rx + 40, chat_y + 28, "CONVERSATION STREAM", size=13, fill=BLUE_PRIMARY, bold=True, anchor="lm")

        # 5. Sentence strip card + recessed track
        st_x, st_y, st_w, st_h = self.PAD, self.STRIP_Y, W - self.PAD * 2, self.STRIP_H
        draw_glass_card(shell, st_x, st_y, st_w, st_h, r=24, shadow_blur=16, shadow_alpha=95)
        row_y, row_h, btn_w, gap = st_y + 50, 66, 175, 14
        send_x = st_x + st_w - btn_w - 36
        clr_x = send_x - 110 - gap
        del_x = clr_x - 110 - gap
        inp_x = st_x + 36
        inp_w = del_x - inp_x - 20
        draw.rounded_rectangle([inp_x, row_y, inp_x + inp_w, row_y + row_h], radius=row_h // 2,
                               fill=hex2rgb("#F3F8FE"), outline=hex2rgb("#CFE2F7"), width=1)

        # 6. Palette card + static header
        pal_x, pal_y, pal_w, pal_h = self.PAD, self.PALETTE_Y, W - self.PAD * 2, self.PALETTE_H
        draw_glass_card(shell, pal_x, pal_y, pal_w, pal_h, r=24, shadow_blur=16, shadow_alpha=95)
        _draw_text(draw, pal_x + 36, pal_y + 26, "SIGN PALETTE", size=14, fill=TEXT_MAIN, bold=True, anchor="lm")
        _draw_text(draw, pal_x + 160, pal_y + 26,
                   f"({len(self.palette)} signs available • Trained & Ready)",
                   size=12, fill=TEXT_MUTED, anchor="lm")
        return shell

    # ── Small static overlays on top of the camera ──────────────────────────
    def _build_static_badges(self):
        # LIVE SIGNER FEED
        bw, bh = 180, 32
        im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
        draw_gradient_pill(im, 0, 0, bw, bh, "#1E293B", "#0F172A")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, bw - 1, bh - 1], radius=bh // 2, outline=hex2rgb("#334155") + (255,), width=1)
        d.ellipse([12, 11, 22, 21], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(d, 30, bh // 2, "LIVE SIGNER FEED", size=11, fill=WHITE, bold=True, anchor="lm")
        self._set_layer("badge_live", self.badge_live_id, im, self.CAM_X + 16, self.CAM_Y + 16)

        # CAM 01 tag
        tw, th = 140, 30
        im2 = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
        draw_gradient_pill(im2, 0, 0, tw, th, "#1E293B", "#0F172A")
        _draw_text(ImageDraw.Draw(im2), tw // 2, th // 2, "CAM 01 • FSL SIGNER", size=10,
                   fill=TEXT_FAINT, bold=True)
        self._set_layer("badge_tag", self.badge_tag_id, im2,
                        self.CAM_X + self.CAM_W - tw - 16, self.CAM_Y + 16)

    # ── Render (called by self._tick; keep cheap) ───────────────────────────
    def _render(self, force=False):
        now_pc = time.perf_counter()
        if not force and (now_pc - self._last_render_pc) < 0.5 / self._target_fps():
            return           # tolerate a leftover external driver calling us
        self._last_render_pc = now_pc
        now = time.time()

        t0 = time.perf_counter()
        self._update_camera()
        t1 = time.perf_counter()
        self._update_caption()
        self._update_chat()
        t2 = time.perf_counter()
        self._update_strip(now)
        t3 = time.perf_counter()
        self._update_palette()
        t4 = time.perf_counter()
        self._update_toast(now)
        t5 = time.perf_counter()

        if ENABLE_PERF:
            p = self._perf
            p["cam"].add((t1 - t0) * 1000)
            p["chat"].add((t2 - t1) * 1000)
            p["strip"].add((t3 - t2) * 1000)
            p["pal"].add((t4 - t3) * 1000)
            p["total"].add((t5 - t0) * 1000)
            self._perf_frames += 1
            if t5 - self._perf_last_print >= 1.0:
                fps = self._perf_frames / (t5 - self._perf_last_print)
                print(f"[PERF-PARTNER] fps={fps:.1f}  render={p['total'].get():.1f}ms  "
                      f"cam={p['cam'].get():.1f}  chat={p['chat'].get():.1f}  "
                      f"strip={p['strip'].get():.1f}  palette={p['pal'].get():.1f}  "
                      f"(target {self._target_fps()} FPS)")
                self._perf_frames = 0
                self._perf_last_print = t5

    # ── Camera layer (every new frame) ──────────────────────────────────────
    def _fetch_frame(self):
        if self.get_raw_frame_fn is not None:
            fid, frame = self.get_raw_frame_fn(self._last_frame_id)
            if frame is None:
                return None
            self._last_frame_id = fid
            return frame
        if self.get_cam_fn is not None:                     # legacy PIL path
            pil = self.get_cam_fn()
            if pil is None:
                return None
            return cv2.cvtColor(np.asarray(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        return None

    def _cam_geometry(self, fw, fh):
        key = (fw, fh, self.ui_scale)
        if self._cam_geo is not None and self._cam_geo["key"] == key:
            return self._cam_geo
        k = min(self.CAM_W / fw, self.CAM_H / fh)
        nw, nh = int(fw * k), int(fh * k)
        ox = self.CAM_X + (self.CAM_W - nw) // 2
        oy = self.CAM_Y + (self.CAM_H - nh) // 2
        s = self.ui_scale
        dw, dh = max(4, int(nw * s)), max(4, int(nh * s))
        r = max(2, min(int(16 * s), dw // 2, dh // 2))
        yy, xx = np.mgrid[0:r, 0:r]
        m = ((xx + 0.5 - r) ** 2 + (yy + 0.5 - r) ** 2) > r * r      # True = outside the rounded corner
        corners = [
            (slice(0, r), slice(0, r), m),
            (slice(0, r), slice(dw - r, dw), m[:, ::-1]),
            (slice(dh - r, dh), slice(0, r), m[::-1, :]),
            (slice(dh - r, dh), slice(dw - r, dw), m[::-1, ::-1]),
        ]
        self._cam_geo = {"key": key, "dw": dw, "dh": dh, "corners": corners, "ox": ox, "oy": oy}
        self.canvas.coords(self.cam_id, self.ui_ox + int(ox * s), self.ui_oy + int(oy * s))
        return self._cam_geo

    def _update_camera(self):
        frame = self._fetch_frame()
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        geo = self._cam_geometry(fw, fh)
        dw, dh = geo["dw"], geo["dh"]

        small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        if MIRROR_FEED:
            small = cv2.flip(small, 1)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        for ys, xs, mk in geo["corners"]:
            view = rgb[ys, xs]
            view[mk] = CAM_DARK
        img = Image.fromarray(rgb)

        if self._cam_tk is None or self._cam_tk_size != (dw, dh):
            self._cam_tk = ImageTk.PhotoImage(img)
            self._cam_tk_size = (dw, dh)
            self.canvas.itemconfig(self.cam_id, image=self._cam_tk)
        else:
            self._cam_tk.paste(img)                 # reuse PhotoImage: no realloc

    # ── Caption (live signer text over the camera) ──────────────────────────
    def _update_caption(self):
        txt = self.latest_signer_text
        if txt == self._keys["cap"]:
            return
        self._keys["cap"] = txt
        if not txt:
            self.canvas.itemconfig(self.caption_id, state="hidden")
            return
        f = _get_font(15, True)
        max_w = self.CAM_W - 40
        disp = f'Signer: "{txt}"'
        if self._measure.textlength(disp, font=f) + 48 > max_w:
            while len(disp) > 8 and self._measure.textlength(disp + '…"', font=f) + 48 > max_w:
                disp = disp[:-1]
            disp = disp.rstrip() + '…"'
        cap_w = min(max_w, int(self._measure.textlength(disp, font=f)) + 48)
        cap_h = 44
        im = Image.new("RGBA", (cap_w, cap_h), (0, 0, 0, 0))
        draw_gradient_pill(im, 0, 0, cap_w, cap_h, "#0F172A", "#1E293B")
        d = ImageDraw.Draw(im)
        d.rounded_rectangle([0, 0, cap_w - 1, cap_h - 1], radius=cap_h // 2,
                            outline=hex2rgb(BLUE_PRIMARY) + (255,), width=1)
        _draw_text(d, 24, cap_h // 2, disp, size=15, fill=WHITE, bold=True, anchor="lm")
        self._set_layer("caption", self.caption_id, im, self.CAM_X + 20, self.CAM_Y + self.CAM_H - cap_h - 16)
        self.canvas.itemconfig(self.caption_id, state="normal")

    # ── Cached tiles ────────────────────────────────────────────────────────
    def _rr_mask(self, w, h, r):
        k = (w, h, r)
        m = self._mask_cache.get(k)
        if m is None:
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
            self._mask_cache[k] = m
        return m

    def _palette_tile(self, item, hov):
        k = ("p", item["key"], hov)
        c = self._tile_cache.get(k)
        if c:
            return c
        w, h = 111, 131
        fill_c = "#E0EEFA" if hov else "#FFFFFF"
        bord_c = BLUE_PRIMARY if hov else "#CBDDF2"
        tile = Image.new("RGB", (w, h), hex2rgb(fill_c))
        d = ImageDraw.Draw(tile)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=18, fill=hex2rgb(fill_c),
                            outline=hex2rgb(bord_c), width=2 if hov else 1)
        ic = self._cached_icons.get(item["key"])
        if ic:
            tile.paste(ic["palette"], ((w - 64) // 2, 14), ic["palette"])
        _draw_text(d, w // 2, h - 23, item["display"], size=12, fill=TEXT_MAIN, bold=True)
        c = (tile, self._rr_mask(w, h, 18))
        self._tile_cache[k] = c
        return c

    def _token_tile(self, item, hov):
        k = ("t", item["key"], hov)
        c = self._tile_cache.get(k)
        if c:
            return c
        w, h = 111, 49
        fill_c = "#DBEAFE" if hov else "#E0EEFA"
        bord_c = BLUE_PRIMARY if hov else "#93C5FD"
        tile = Image.new("RGB", (w, h), hex2rgb(fill_c))
        d = ImageDraw.Draw(tile)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=hex2rgb(fill_c),
                            outline=hex2rgb(bord_c), width=1)
        ic = self._cached_icons.get(item["key"])
        if ic:
            tile.paste(ic["strip"], (8, 5), ic["strip"])
        _draw_text(d, 52, h // 2, item["display"], size=12, fill=TEXT_MAIN, bold=True, anchor="lm")
        _draw_text(d, w - 14, h // 2, "×", size=13, fill=TEXT_MUTED, bold=True)
        c = (tile, self._rr_mask(w, h, 14))
        self._tile_cache[k] = c
        return c

    # ── Chat layer (only when history changes) ──────────────────────────────
    def _update_chat(self):
        key = len(self._chat_history)
        if key == self._keys["chat"]:
            return
        self._keys["chat"] = key
        t0 = time.perf_counter()

        LX, LY, LW, LH = self.CHAT_LX, self.CHAT_LY, self.CHAT_LW, self.CHAT_LH
        img = self.ui_shell_cache.crop((LX, LY, LX + LW, LY + LH))
        draw = ImageDraw.Draw(img)

        if not self._chat_history:
            _draw_text(draw, LW // 2, LH // 2 - 12, "Waiting for conversation...", size=16, fill=TEXT_MUTED, bold=True)
            _draw_text(draw, LW // 2, LH // 2 + 16, "Sent and received sentences will appear here as text bubbles",
                       size=12, fill=TEXT_FAINT)
        else:
            f_txt = _get_font(15, True)
            gap = 10
            # pick the newest messages that fit, keep chronological order
            chosen, used = [], 6
            for msg in reversed(self._chat_history):
                bh = 62 + (28 if msg.get("tokens") else 0)
                if used + bh > LH:
                    break
                chosen.append((msg, bh))
                used += bh + gap
            chosen.reverse()

            cur_y = 6
            for msg, b_h in chosen:
                sender, txt, t_str = msg["sender"], msg["text"], msg["time"]
                tokens = msg.get("tokens", [])
                max_bw = LW - 16
                tw = self._measure.textlength(txt, font=f_txt)
                b_w = int(min(max_bw, max(240, tw + 44)))
                if tw + 28 > b_w:                         # truncate with ellipsis
                    while len(txt) > 4 and self._measure.textlength(txt + "…", font=f_txt) + 28 > b_w:
                        txt = txt[:-1]
                    txt = txt.rstrip() + "…"

                if sender == "signer":
                    bx = 8
                    draw.rounded_rectangle([bx, cur_y, bx + b_w, cur_y + b_h], radius=14,
                                           fill=hex2rgb(BUBBLE_FSL), outline=hex2rgb(BUBBLE_FSL_BD), width=1)
                    _draw_text(draw, bx + 14, cur_y + 14, f"Signer (FSL) • {t_str}", size=10,
                               fill=BLUE_PRIMARY, bold=True, anchor="lm")
                    _draw_text(draw, bx + 14, cur_y + 38, txt, size=15, fill=TEXT_MAIN, bold=True, anchor="lm")
                else:
                    bx = LW - b_w - 8
                    draw.rounded_rectangle([bx, cur_y, bx + b_w, cur_y + b_h], radius=14,
                                           fill=hex2rgb(BUBBLE_PART), outline=hex2rgb(BUBBLE_PART_BD), width=1)
                    _draw_text(draw, bx + b_w - 14, cur_y + 14, f"You (Partner) • {t_str}", size=10,
                               fill=GREEN_TEXT, bold=True, anchor="rm")
                    _draw_text(draw, bx + b_w - 14, cur_y + 38, txt, size=15, fill=TEXT_MAIN, bold=True, anchor="rm")
                    if tokens:
                        px = bx + b_w - 14 - len(tokens) * 28
                        py = cur_y + 54
                        for t in tokens:
                            ic = self._cached_icons.get(t)
                            if ic:
                                img.paste(ic["bubble"], (px, py), ic["bubble"])
                            px += 28
                cur_y += b_h + gap

        self._set_layer("chat", self.chat_id, img, LX, LY)

    # ── Sentence strip layer ────────────────────────────────────────────────
    def _update_strip(self, now):
        key = (tuple(t["key"] for t in self.sentence_tokens), self._hovered_strip_idx,
               self._send_hover, self._del_hover, self._clr_hover,
               now < self._send_click_until, now < self._del_click_until, now < self._clr_click_until)
        if key == self._keys["strip"]:
            return
        self._keys["strip"] = key

        LX, LY, LW, LH = self.STRIP_LX, self.STRIP_LY, self.STRIP_LW, self.STRIP_LH
        img = self.ui_shell_cache.crop((LX, LY, LX + LW, LY + LH))
        draw = ImageDraw.Draw(img)

        st_x, st_y, st_w = self.PAD, self.STRIP_Y, self.W - self.PAD * 2
        hx, hy = st_x + 36 - LX, st_y + 26 - LY
        for i, amp in enumerate([4, 9, 14, 8]):
            xi = hx + i * 5
            draw.line([(xi, hy - amp), (xi, hy + amp)], fill=hex2rgb(BLUE_PRIMARY), width=2)
        _draw_text(draw, hx + 28, hy, "YOUR SENTENCE STRIP", size=13, fill=BLUE_PRIMARY, bold=True, anchor="lm")
        _draw_text(draw, hx + 210, hy, f"({len(self.sentence_tokens)} signs selected)",
                   size=12, fill=TEXT_MUTED, anchor="lm")

        row_y, row_h, btn_w, gap = st_y + 50, 66, 175, 14
        send_x = st_x + st_w - btn_w - 36
        clr_x = send_x - 110 - gap
        del_x = clr_x - 110 - gap
        inp_x = st_x + 36
        inp_w = del_x - inp_x - 20
        ry = row_y - LY

        # tokens
        self._strip_item_rects = []
        toks = self.sentence_tokens
        if not toks:
            _draw_text(draw, inp_x + 28 - LX, ry + row_h // 2,
                       "Tap signs from the palette below to compose your sentence...",
                       size=15, fill=TEXT_FAINT, anchor="lm")
        else:
            step, tok_w, tok_h = 120, 110, 48
            max_fit = max(1, (inp_w - 36) // step)
            start = max(0, len(toks) - max_fit)
            cur_x = inp_x + 18
            tok_y = row_y + (row_h - tok_h) // 2
            for idx in range(start, len(toks)):
                tile, mask = self._token_tile(toks[idx], self._hovered_strip_idx == idx)
                img.paste(tile, (cur_x - LX, tok_y - LY), mask)
                self._strip_item_rects.append((idx, cur_x, tok_y, cur_x + tok_w, tok_y + tok_h))
                cur_x += step

        # Delete
        is_click = now < self._del_click_until
        is_hov = self._del_hover and not is_click
        self._btn_del_rect = (del_x, row_y, del_x + 110, row_y + row_h)
        c1, c2 = ("#E2E8F0", "#CBD5E1") if is_click else (("#F1F5F9", "#E2E8F0") if is_hov else ("#FFFFFF", "#F8FAFC"))
        draw_gradient_pill(img, del_x - LX, ry, 110, row_h, c1, c2, outline_col="#CBD5E1")
        _draw_text(draw, del_x - LX + 55, ry + row_h // 2, "⌫ Delete", size=14, fill=TEXT_MAIN, bold=True)

        # Clear
        is_click = now < self._clr_click_until
        is_hov = self._clr_hover and not is_click
        self._btn_clr_rect = (clr_x, row_y, clr_x + 110, row_y + row_h)
        c1, c2 = ("#FECACA", "#FCA5A5") if is_click else (("#FEE2E2", "#FECACA") if is_hov else (RED_BG, "#FEE2E2"))
        draw_gradient_pill(img, clr_x - LX, ry, 110, row_h, c1, c2, outline_col=RED_BORDER)
        _draw_text(draw, clr_x - LX + 55, ry + row_h // 2, "✕ Clear", size=14, fill=RED_BTN, bold=True)

        # SEND
        is_click = now < self._send_click_until
        is_hov = self._send_hover and not is_click
        self._btn_send_rect = (send_x, row_y, send_x + btn_w, row_y + row_h)
        if is_click:
            sx, sy, sw, sh, sc1, sc2 = send_x + 2, row_y + 2, btn_w - 4, row_h - 4, "#0369A1", "#0284C7"
        elif is_hov:
            sx, sy, sw, sh, sc1, sc2 = send_x - 1, row_y - 1, btn_w + 2, row_h + 2, "#38BDF8", "#0284C7"
        else:
            sx, sy, sw, sh, sc1, sc2 = send_x, row_y, btn_w, row_h, "#0EA5E9", "#0369A1"
        glow_pad = 10
        glow = self._send_glow_cache.get((sw, sh))
        if glow is None:
            glow = Image.new("RGBA", (sw + glow_pad * 2, sh + glow_pad * 2), (0, 0, 0, 0))
            ImageDraw.Draw(glow).rounded_rectangle([glow_pad, glow_pad + 4, glow_pad + sw, glow_pad + sh + 4],
                                                   radius=sh // 2, fill=(14, 165, 233, 85))
            glow = glow.filter(ImageFilter.GaussianBlur(8))
            self._send_glow_cache[(sw, sh)] = glow
        lsx, lsy = sx - LX, sy - LY
        img.paste(glow, (lsx - glow_pad, lsy - glow_pad), glow)
        draw_gradient_pill(img, lsx, lsy, sw, sh, sc1, sc2)
        draw.arc([lsx + 1, lsy + 1, lsx + sw - 2, lsy + sh - 2], start=190, end=350, fill=hex2rgb(WHITE), width=1)
        acx, acy = lsx + 32, lsy + sh // 2
        draw.line([(acx - 10, acy), (acx + 8, acy)], fill=hex2rgb(WHITE), width=3)
        draw.polygon([(acx + 8, acy), (acx + 2, acy - 6), (acx + 2, acy + 6)], fill=hex2rgb(WHITE))
        _draw_text(draw, lsx + 54, lsy + sh // 2, "SEND", size=15, fill=WHITE, bold=True, anchor="lm")

        self._set_layer("strip", self.strip_id, img, LX, LY)

    # ── Palette layer (only on hover change) ────────────────────────────────
    def _update_palette(self):
        key = self._hovered_palette_key
        if self._keys["pal"] is not None and key == self._keys["pal"][0]:
            return
        self._keys["pal"] = (key,)

        LX, LY, LW, LH = self.PAL_LX, self.PAL_LY, self.PAL_LW, self.PAL_LH
        img = self.ui_shell_cache.crop((LX, LY, LX + LW, LY + LH))

        card_w, gap = 110, 16
        start_x, start_y = self.PAD + 36, self.PALETTE_Y + 54
        self._palette_rects = []
        for idx, item in enumerate(self.palette):
            cx = start_x + idx * (card_w + gap)
            if cx + card_w > LX + LW:
                break
            tile, mask = self._palette_tile(item, self._hovered_palette_key == item["key"])
            img.paste(tile, (cx - LX, start_y - LY), mask)
            self._palette_rects.append((item["key"], item, cx, start_y, cx + card_w, start_y + 130))

        self._set_layer("palette", self.palette_id, img, LX, LY)

    # ── Toast (plain canvas items, like the FSL window) ─────────────────────
    def _update_toast(self, now):
        if now < self.notification_end:
            txt = self.notification_text
            key = (txt, self.ui_scale)
            if key != self._toast_key:
                s = self.ui_scale
                nw, nh = max(220, len(txt) * 14 + 52), 48
                nx = self.ui_ox + int((self.W // 2 - nw // 2) * s)
                ny = self.ui_oy + int((self.CONTENT_Y + self.CONTENT_H // 2 - nh // 2) * s)
                nws, nhs = int(nw * s), int(nh * s)
                self.canvas.coords(self.toast_bg_id, *_round_rect_pts(nx, ny, nx + nws, ny + nhs, nhs // 2))
                self.canvas.coords(self.toast_text_id, nx + nws // 2, ny + nhs // 2)
                self.canvas.itemconfig(self.toast_text_id, text=txt, font=_get_tk_font(int(15 * s), True))
                self._toast_key = key
            if not self._toast_visible:
                self.canvas.itemconfig(self.toast_bg_id, state="normal")
                self.canvas.itemconfig(self.toast_text_id, state="normal")
                self._toast_visible = True
        elif self._toast_visible:
            self.canvas.itemconfig(self.toast_bg_id, state="hidden")
            self.canvas.itemconfig(self.toast_text_id, state="hidden")
            self._toast_visible = False
            self._toast_key = None

    def _set_notification(self, msg, duration=2.5):
        self.notification_text = msg
        self.notification_end = time.time() + duration

    # ── Mouse ───────────────────────────────────────────────────────────────
    def _design_xy(self, event):
        return (int((event.x - self.ui_ox) / self.ui_scale),
                int((event.y - self.ui_oy) / self.ui_scale))

    @staticmethod
    def _inside(rect, mx, my):
        x0, y0, x1, y1 = rect
        return x0 <= mx <= x1 and y0 <= my <= y1

    def _on_motion(self, event):
        if self.ui_scale <= 0:
            return
        mx, my = self._design_xy(event)
        self._send_hover = self._inside(self._btn_send_rect, mx, my)
        self._del_hover = self._inside(self._btn_del_rect, mx, my)
        self._clr_hover = self._inside(self._btn_clr_rect, mx, my)

        self._hovered_palette_key = None
        for key, _item, x0, y0, x1, y1 in self._palette_rects:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                self._hovered_palette_key = key
                break

        self._hovered_strip_idx = None
        for idx, x0, y0, x1, y1 in self._strip_item_rects:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                self._hovered_strip_idx = idx
                break

    def _on_click(self, event):
        if self.ui_scale <= 0:
            return
        mx, my = self._design_xy(event)

        if self._inside(self._btn_send_rect, mx, my):
            self._send_click_until = time.time() + 0.18
            self._send_sentence()
            return
        if self._inside(self._btn_del_rect, mx, my):
            self._del_click_until = time.time() + 0.18
            if self.sentence_tokens:
                self.sentence_tokens.pop()
            return
        if self._inside(self._btn_clr_rect, mx, my):
            self._clr_click_until = time.time() + 0.18
            self.sentence_tokens.clear()
            return
        for idx, x0, y0, x1, y1 in self._strip_item_rects:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                if 0 <= idx < len(self.sentence_tokens):
                    self.sentence_tokens.pop(idx)
                self._hovered_strip_idx = None
                return
        for _key, item, x0, y0, x1, y1 in self._palette_rects:
            if x0 <= mx <= x1 and y0 <= my <= y1:
                self.sentence_tokens.append(item)
                return

    # ── Co-op bus ───────────────────────────────────────────────────────────
    def _call_main(self, name, *args):
        """Use injected callbacks, else the running __main__ module. Never re-import main.py."""
        fn = self._speak_fn if name == "speak_text" else self._ping_fn if name == "play_ping_sound" else None
        if fn is None:
            fn = getattr(sys.modules.get("__main__"), name, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception as e:
                print(f"[Partner] {name} failed: {e}")

    def _send_sentence(self):
        if not self.sentence_tokens:
            self._set_notification("Select signs to compose sentence", duration=1.8)
            return
        tokens = [item["key"] for item in self.sentence_tokens]
        translated = translate_sentence(" ".join(tokens))

        coop_hub.send_from_partner(tokens, translated)
        self._call_main("speak_text", translated)

        self._chat_history.append({"sender": "partner", "text": translated,
                                   "tokens": list(tokens), "time": time.strftime("%I:%M %p")})
        self._set_notification(f"SENT: {translated}")
        self.sentence_tokens.clear()

    def _on_signer_message_received(self, text: str):
        self._chat_history.append({"sender": "signer", "text": text, "tokens": [],
                                   "time": time.strftime("%I:%M %p")})
        self.latest_signer_text = text
        self._set_notification(f"SIGNER: {text[:28]}")
        self._call_main("play_ping_sound")

    def _on_close(self):
        self._running = False
        if self.on_close_callback:
            try:
                self.on_close_callback()
            except Exception:
                pass
        try:
            self.window.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    app = NonFSLWindow()
    app.window.mainloop()
