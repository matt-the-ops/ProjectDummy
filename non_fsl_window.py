import os
import math
import time
import collections
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont

from coop_bus import coop_hub, load_icon_palette, get_icon_path_for_sign
from sentence_translator import translate_sentence

from ui_kit import (
    BG_LIGHT, BG_CENTER, BG_DEEP,
    CARD_BG, CARD_BORDER, CARD_RIM_SUB, CARD_SHADOW,
    BLUE_PRIMARY, BLUE_LIGHT, BLUE_CYAN, BLUE_DARK,
    TEXT_MAIN, TEXT_MUTED, TEXT_FAINT, WHITE,
    GREEN_ONLINE, GREEN_BG, GREEN_BORDER, GREEN_TEXT,
    RED_BTN, RED_BG, RED_BORDER,
    BUBBLE_FSL, BUBBLE_FSL_BD, BUBBLE_PART, BUBBLE_PART_BD,
    hex2rgb, _blend,
    _get_font, _draw_text,
    draw_gradient_pill,
    create_ambient_background,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')



def draw_glass_card(img: Image.Image, x, y, w, h, r=26,
                    bg_color=CARD_BG, shadow_color=CARD_SHADOW,
                    border_color=CARD_BORDER, shadow_offset=(0, 9),
                    shadow_blur=20, shadow_alpha=115):
    """Renders authentic neumorphic-glassmorphic card with diffuse sky-blue drop shadow."""
    pad = shadow_blur * 2
    sw, sh = w + pad * 2, h + pad * 2
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
    c_bg = hex2rgb(bg_color)
    cdraw.rounded_rectangle([0, 0, w, h], radius=r, fill=c_bg + (248,))
    c_border = hex2rgb(border_color)
    cdraw.rounded_rectangle([0, 0, w, h], radius=r, outline=c_border + (255,), width=1)
    img.paste(card_img, (x, y), card_img)


# ── Non-FSL Partner App (PIL Canvas Twin) ───────────────────────────────────
class NonFSLWindow:
    """
    Non-FSL Partner Kiosk Window with authentic Neumorphic-Glassmorphic UI.
    - Matches ShieldFSLApp visual design, lighting, cards, and animations.
    - Displays live FSL user camera feed in real time.
    - Features the SHIELD logo with ambient glow and drop shadow.
    - Communicates ONLY with icons (no keyboard / text entry).
    - Taps icons to build sentence strip -> SEND translates & delivers.
    - Receives signer sentences as text bubbles in a companion conversation card.
    """
    DESIGN_W  = 1920
    DESIGN_H  = 1080
    W         = 1920
    H         = 1080

    PAD       = 28
    TOP_Y     = 16
    TOP_H     = 92

    CONTENT_Y = 124
    CONTENT_H = 550
    LEFT_W    = 1084    # Camera Card width (matching FSL app)
    RIGHT_X   = 1132    # Chat/Companion Card X (matching FSL app)
    RIGHT_W   = 760     # Chat/Companion Card width

    STRIP_Y   = 690
    STRIP_H   = 140

    PALETTE_Y = 846
    PALETTE_H = 218

    def __init__(self, master=None, on_close_callback=None, get_cam_fn=None):
        if master is None:
            self.window = tk.Tk()
        else:
            self.window = tk.Toplevel(master)

        self.window.title("Project B.M.O — Co-Op Communicator (Partner Kiosk)")
        self.window.configure(bg=BG_LIGHT)

        # ── Responsive Screen Auto-Fit ──
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        usable_w = screen_w - 40
        usable_h = screen_h - 80

        if self.DESIGN_W > usable_w or self.DESIGN_H > usable_h:
            init_scale = min(usable_w / self.DESIGN_W, usable_h / self.DESIGN_H)
            init_w = max(800, int(self.DESIGN_W * init_scale))
            init_h = max(450, int(self.DESIGN_H * init_scale))
        else:
            init_w = self.DESIGN_W
            init_h = self.DESIGN_H

        pos_x = max(0, (screen_w - init_w) // 2 + 25)
        pos_y = max(0, (screen_h - init_h) // 2 + 25)
        self.window.geometry(f"{init_w}x{init_h}+{pos_x}+{pos_y}")
        self.window.minsize(800, 450)
        self.window.resizable(True, True)
        self.on_close_callback = on_close_callback
        self.get_cam_fn = get_cam_fn

        self.win_w = init_w
        self.win_h = init_h
        self.ui_scale = min(self.win_w / self.DESIGN_W, self.win_h / self.DESIGN_H)
        self.render_w = max(1, int(self.DESIGN_W * self.ui_scale))
        self.render_h = max(1, int(self.DESIGN_H * self.ui_scale))
        self.ui_ox = (self.win_w - self.render_w) // 2
        self.ui_oy = (self.win_h - self.render_h) // 2

        # Single offscreen PIL surface
        self.canvas = tk.Canvas(self.window, width=init_w, height=init_h,
                                bg=BG_LIGHT, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.pil_img  = Image.new("RGB", (self.DESIGN_W, self.DESIGN_H), hex2rgb(BG_LIGHT))
        self.tk_img   = None
        self.image_id = self.canvas.create_image(self.ui_ox, self.ui_oy, anchor=tk.NW)

        # Performance caches
        self._cached_cam_mask = None
        self._cached_cam_mask_key = None

        # Pre-render static UI shell
        self._logo_rgba     = None
        self._logo_alpha    = None
        self._logo_rim      = None
        self._logo_rim_mask = None
        self._logo_shadow   = None
        self._logo_pad      = 20
        self._logo_w        = 50
        self._logo_h        = 50
        self._load_logo()

        self.ui_shell_cache = self._pre_render_ui_shell()

        # Hard Rule: Load available signs strictly trained AND in icon_map.json
        self.palette = load_icon_palette()
        self._cached_icons = {}
        self._load_palette_icons()

        # Sentence Strip & Chat State
        self.sentence_tokens = []  # List of dicts from palette
        self._chat_history   = []  # List of {"sender": "signer"|"partner", "text": str, "tokens": list, "time": str}
        self.latest_signer_text = ""

        # Notification & button states
        self.notification_text = ""
        self.notification_end  = 0.0

        self._send_hover       = False
        self._send_click_until = 0.0
        self._del_hover        = False
        self._del_click_until  = 0.0
        self._clr_hover        = False
        self._clr_click_until  = 0.0
        self._hovered_palette_key = None
        self._hovered_strip_idx   = None

        # Hit-test rects
        self._btn_send_rect    = (0, 0, 0, 0)
        self._btn_del_rect     = (0, 0, 0, 0)
        self._btn_clr_rect     = (0, 0, 0, 0)
        # _hide_btn_rect/_show_main_btn_rect set by _pre_render_ui_shell
        self._palette_rects    = []  # [(key, item, x0, y0, x1, y1)]
        self._strip_item_rects = []  # [(idx, x0, y0, x1, y1)]

        # Visibility / cross-window callbacks
        self._visible = True
        self.main_app = None
        self.is_main_visible_fn = None
        # hit-test rects are populated by _pre_render_ui_shell(); do not overwrite after shell build
        if not hasattr(self, "_hide_btn_rect"):
            self._hide_btn_rect = (0, 0, 0, 0)
        if not hasattr(self, "_show_main_btn_rect"):
            self._show_main_btn_rect = (0, 0, 0, 0)

        # Bind events
        self.window.bind("<Motion>",   self._on_motion)
        self.window.bind("<Button-1>", self._on_click)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

        # Hover states
        self._hide_hover = False
        self._show_main_hover = False

        # Register with Co-Op Bus
        coop_hub.register_partner_listener(self._on_signer_message_received)

        # Always self-schedule the render loop. Standalone (no master) targets ~30 FPS;
        # when embedded alongside the main window it throttles to ~15 FPS so it does not
        # starve the primary GUI thread (both share the one Tk main loop).
        self._running = True
        self._render_fps = 30.0 if master is None else 15.0
        self.window.after(100, self._gui_loop)

    # ── Logo loader (identical to ShieldFSLApp) ─────────────────────────────────
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
            sw, sh = target_w + pad * 2, target_h + pad * 2
            shadow = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
            shadow_solid = Image.new("RGBA", (target_w, target_h), (15, 30, 60, 65))
            shadow.paste(shadow_solid, (pad + 1, pad + 5), alpha)
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=6))

            self._logo_rgba     = logo_resized
            self._logo_alpha    = alpha
            self._logo_rim      = rim
            self._logo_rim_mask = rim_mask
            self._logo_shadow   = shadow
            self._logo_pad      = pad
            self._logo_w        = target_w
            self._logo_h        = target_h
            print("[OK] Partner Kiosk: Logo loaded with drop shadow and rim contour.")
        except Exception as e:
            print(f"[!] Partner Kiosk logo load error: {e}")

    # ── Pre-cache Palette Icons ────────────────────────────────────────────────
    def _load_palette_icons(self):
        for item in self.palette:
            key = item["key"]
            p = item["icon_path"]
            try:
                im = Image.open(p).convert("RGBA")
                self._cached_icons[key] = {
                    "raw": im,
                    "palette": im.resize((64, 64), Image.Resampling.LANCZOS),
                    "strip":   im.resize((38, 38), Image.Resampling.LANCZOS),
                    "bubble":  im.resize((26, 26), Image.Resampling.LANCZOS),
                }
            except Exception as e:
                print(f"[!] Error caching icon {key}: {e}")

    # ── Pre-render Static UI Shell ─────────────────────────────────────────────
    def _pre_render_ui_shell(self):
        W, H = self.W, self.H
        shell = create_ambient_background(W, H)
        draw = ImageDraw.Draw(shell)

        # ── 1. Top Bar Card ──
        bx, by, bw, bh = self.PAD, self.TOP_Y, W - self.PAD * 2, self.TOP_H
        draw_glass_card(shell, bx, by, bw, bh, r=26, shadow_blur=16, shadow_alpha=95)
        cy = by + bh // 2

        # Ambient Glow + Logo
        logo_x = bx + 28
        glow = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([0, 0, 72, 72], fill=hex2rgb("#93C5FD") + (120,))
        gdraw.ellipse([12, 12, 60, 60], fill=hex2rgb("#38BDF8") + (180,))
        glow = glow.filter(ImageFilter.GaussianBlur(12))
        shell.paste(glow, (logo_x, cy - 36), glow)

        if self._logo_shadow:
            shell.paste(self._logo_shadow,
                        (logo_x - self._logo_pad, cy - self._logo_h // 2 - self._logo_pad),
                        self._logo_shadow)
        if self._logo_rim:
            shell.paste(self._logo_rim, (logo_x, cy - self._logo_h // 2), self._logo_rim_mask)
        if self._logo_rgba:
            shell.paste(self._logo_rgba, (logo_x, cy - self._logo_h // 2), self._logo_alpha)

        # Title & Subtitle
        title_x = logo_x + 80
        f_title = _get_font(25, bold=True)
        draw.text((title_x, cy - 12), "Project ", font=f_title, fill=hex2rgb(TEXT_MAIN), anchor="lm")
        w_proj = draw.textlength("Project ", font=f_title)
        bmo_x = int(title_x + w_proj)
        draw.text((bmo_x, cy - 12), "B.M.O", font=f_title, fill=hex2rgb(BLUE_PRIMARY), anchor="lm")
        w_bmo = draw.textlength("B.M.O", font=f_title)

        badge_x = int(bmo_x + w_bmo + 16)
        badge_y = cy - 23
        badge_w, badge_h = 80, 24
        draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                               radius=badge_h // 2, fill=hex2rgb(GREEN_BG),
                               outline=hex2rgb(GREEN_BORDER), width=1)
        draw.ellipse([badge_x + 9, badge_y + 8, badge_x + 16, badge_y + 15], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, badge_x + 22, badge_y + 12, "ONLINE", size=10, fill=GREEN_TEXT, bold=True, anchor="lm")

        _draw_text(draw, title_x, cy + 16, "Partner Kiosk • Visual Two-Way Communication",
                   size=13, fill=TEXT_MUTED, bold=False, anchor="lm")

        # Center Mode Capsule: Partner Mode
        switch_w, switch_h = 240, 48
        switch_x = W // 2 - switch_w // 2
        switch_y = cy - switch_h // 2
        draw.rounded_rectangle([switch_x, switch_y, switch_x + switch_w, switch_y + switch_h],
                               radius=switch_h // 2, fill=hex2rgb("#DDEBF9"),
                               outline=hex2rgb("#CBDDF2"), width=1)
        draw_gradient_pill(shell, switch_x + 4, switch_y + 4, switch_w // 2, switch_h - 8,
                           BLUE_PRIMARY, BLUE_CYAN)
        _draw_text(draw, switch_x + switch_w // 4 + 2, cy, "Partner", size=13, fill=WHITE, bold=True, anchor="mm")
        _draw_text(draw, switch_x + 3 * switch_w // 4, cy, "FSL Linked", size=13, fill="#5A718F", bold=False, anchor="mm")

        # Right: window visibility + status badge
        btn_w_vis = 110
        btn_h_vis = 34
        btn_gap = 10
        fps_w, fps_h = 160, 42
        right_edge = bx + bw - 20
        fps_x = right_edge - fps_w
        fps_y = cy - fps_h // 2

        hide_x = fps_x - btn_gap - btn_w_vis
        hide_y = cy - btn_h_vis // 2
        draw.rounded_rectangle([hide_x, hide_y, hide_x + btn_w_vis, hide_y + btn_h_vis],
                               radius=btn_h_vis // 2, fill=hex2rgb("#FEE2E2"),
                               outline=hex2rgb(RED_BORDER), width=1)
        _draw_text(draw, hide_x + btn_w_vis // 2, cy, "Hide Partner", size=10, fill=RED_BTN, bold=True, anchor="mm")
        self._hide_btn_rect = (hide_x, hide_y, hide_x + btn_w_vis, hide_y + btn_h_vis)

        show_x = hide_x - btn_gap - btn_w_vis
        show_y = hide_y
        draw.rounded_rectangle([show_x, show_y, show_x + btn_w_vis, show_y + btn_h_vis],
                               radius=btn_h_vis // 2, fill=hex2rgb("#E0F2FE"),
                               outline=hex2rgb("#7DD3FC"), width=1)
        _draw_text(draw, show_x + btn_w_vis // 2, cy, "Show Main", size=10, fill=BLUE_DARK, bold=True, anchor="mm")
        self._show_main_btn_rect = (show_x, show_y, show_x + btn_w_vis, show_y + btn_h_vis)

        draw.rounded_rectangle([fps_x, fps_y, fps_x + fps_w, fps_y + fps_h],
                               radius=fps_h // 2, fill=hex2rgb("#F8FBFE"),
                               outline=hex2rgb("#DCE8F6"), width=1)
        draw.ellipse([fps_x + 16, fps_y + 17, fps_x + 24, fps_y + 25], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, fps_x + 32, cy, "2-WAY CO-OP", size=12, fill=GREEN_TEXT, bold=True, anchor="lm")

        # ── 2. Left Main Card: Signer Camera Outer Glass Card ──
        cx, cy_cam, cw, ch_cam = self.PAD, self.CONTENT_Y, self.LEFT_W, self.CONTENT_H
        draw_glass_card(shell, cx, cy_cam, cw, ch_cam, r=26, shadow_blur=18, shadow_alpha=110)

        # ── 3. Right Main Card: BMO Companion Header ──
        rx, ry_bmo, rw = self.RIGHT_X, self.CONTENT_Y, self.RIGHT_W
        bmo_h = 104
        draw_glass_card(shell, rx, ry_bmo, rw, bmo_h, r=26, shadow_blur=16, shadow_alpha=95)

        # ── 4. Right Main Card: Conversation Stream Card ──
        chat_y = ry_bmo + bmo_h + 16
        chat_h = ch_cam - bmo_h - 16
        draw_glass_card(shell, rx, chat_y, rw, chat_h, r=26, shadow_blur=16, shadow_alpha=95)

        # ── 5. Sentence Strip Card ──
        st_x, st_y, st_w, st_h = self.PAD, self.STRIP_Y, W - self.PAD * 2, self.STRIP_H
        draw_glass_card(shell, st_x, st_y, st_w, st_h, r=24, shadow_blur=16, shadow_alpha=95)

        # Strip Track (recessed container)
        row_y = st_y + 50
        row_h = 66
        btn_w = 175
        gap = 14
        send_x = st_x + st_w - btn_w - 36
        clr_x  = send_x - 110 - gap
        del_x  = clr_x - 110 - gap
        inp_x  = st_x + 36
        inp_w  = del_x - inp_x - 20
        draw.rounded_rectangle([inp_x, row_y, inp_x + inp_w, row_y + row_h],
                               radius=row_h // 2, fill=hex2rgb("#F3F8FE"),
                               outline=hex2rgb("#CFE2F7"), width=1)

        # ── 6. Palette Card ──
        pal_x, pal_y, pal_w, pal_h = self.PAD, self.PALETTE_Y, W - self.PAD * 2, self.PALETTE_H
        draw_glass_card(shell, pal_x, pal_y, pal_w, pal_h, r=24, shadow_blur=16, shadow_alpha=95)

        return shell

    # ── Canvas Resize & Resolution Adaption ──────────────
    def _on_canvas_resize(self, event):
        if event.width <= 100 or event.height <= 100:
            return
        if event.width == self.win_w and event.height == self.win_h:
            return
        self.win_w = event.width
        self.win_h = event.height
        self.ui_scale = min(self.win_w / self.DESIGN_W, self.win_h / self.DESIGN_H)
        self.render_w = max(1, int(self.DESIGN_W * self.ui_scale))
        self.render_h = max(1, int(self.DESIGN_H * self.ui_scale))
        self.ui_ox = (self.win_w - self.render_w) // 2
        self.ui_oy = (self.win_h - self.render_h) // 2
        self.canvas.coords(self.image_id, self.ui_ox, self.ui_oy)

    # ── GUI Render Loop (Main Thread @ ~60 FPS) ─────────────────────────────────
    def _gui_loop(self):
        if not self._running:
            return
        if not self._visible:
            return
        t_start = time.time()
        try:
            self._render()
        except Exception as e:
            import traceback
            print(f"[Partner Render Error] {e}")
            traceback.print_exc()
        if not self._running or not self._visible:
            return
        elapsed_ms = int((time.time() - t_start) * 1000)
        target_ms = int(1000.0 / max(1.0, getattr(self, '_render_fps', 30.0)))
        delay = max(1, target_ms - elapsed_ms)  # self-throttled partner render cadence
        self.window.after(delay, self._gui_loop)

    def _draw_topbar_hover(self, img, draw):
        main_visible = self._main_is_visible()

        # Hide Partner: hover => slightly bolder outline (no fill, text stays visible)
        hx0, hy0, hx1, hy1 = getattr(self, "_hide_btn_rect", (0, 0, 0, 0))
        if getattr(self, "_hide_hover", False) and (hx1 - hx0) > 0:
            r = (hy1 - hy0) // 2
            draw.rounded_rectangle([hx0, hy0, hx1, hy1], radius=r,
                                   outline=hex2rgb("#DC2626"), width=2)

        # Show Main: active (main visible) => light indicator; hover while hidden => bolder outline
        sx0, sy0, sx1, sy1 = getattr(self, "_show_main_btn_rect", (0, 0, 0, 0))
        show_hover = getattr(self, "_show_main_hover", False)
        if (sx1 - sx0) > 0 and main_visible:
            r = (sy1 - sy0) // 2
            draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=r,
                                   outline=hex2rgb("#CBD5E1"), width=1)
        elif (sx1 - sx0) > 0 and show_hover and not main_visible:
            r = (sy1 - sy0) // 2
            draw.rounded_rectangle([sx0, sy0, sx1, sy1], radius=r,
                                   outline=hex2rgb("#0284C7"), width=2)

    # ── Dynamic Master Render ──────────────────────────────────────────────────
    def _render(self):
        now = time.time()
        img = self.ui_shell_cache.copy()
        draw = ImageDraw.Draw(img)

        # 1. Signer Live Camera Feed in Left Panel
        self._draw_signer_camera_feed(img, draw, now)

        # 2. Companion & Conversation Stream in Right Panel
        self._draw_companion_and_chat(img, draw, now)

        # 3. Sentence Strip
        self._draw_sentence_strip(img, draw, now)

        # 4. Sign Palette Grid
        self._draw_sign_palette(img, draw, now)

        # 4b. Top-bar hover overlay (buttons are in static shell)
        self._draw_topbar_hover(img, draw)

        # 5. Toast Notification
        self._draw_notification(img, draw, now)

        # Push to Tkinter canvas with adaptive scaling (fast NEAREST scaling)
        if self.render_w != self.DESIGN_W or self.render_h != self.DESIGN_H:
            scaled_img = img.resize((self.render_w, self.render_h), Image.Resampling.NEAREST)
            self.pil_img = scaled_img
            self.tk_img  = ImageTk.PhotoImage(scaled_img)
        else:
            self.pil_img = img
            self.tk_img  = ImageTk.PhotoImage(img)

        self.canvas.itemconfig(self.image_id, image=self.tk_img)
        self.canvas.coords(self.image_id, self.ui_ox, self.ui_oy)

    # ── Render Signer Live Camera ──────────────────────────────────────────────
    def _draw_signer_camera_feed(self, img, draw, now):
        cx, cy, cw, ch = self.PAD, self.CONTENT_Y, self.LEFT_W, self.CONTENT_H
        cam_pad = 18
        vid_x = cx + cam_pad
        vid_y = cy + cam_pad
        vid_w = cw - cam_pad * 2
        vid_h = ch - cam_pad * 2

        # Inner dark monitor container
        draw.rounded_rectangle([vid_x, vid_y, vid_x + vid_w, vid_y + vid_h],
                               radius=18, fill=hex2rgb("#0B132B"),
                               outline=hex2rgb("#1E293B"), width=1)

        cam_frame = self.get_cam_fn() if self.get_cam_fn else None

        if cam_frame:
            # Scale video maintaining aspect ratio
            fw, fh = cam_frame.size
            scale = min(vid_w / max(fw, 1), vid_h / max(fh, 1))
            nw = int(fw * scale)
            nh = int(fh * scale)
            ox = vid_x + (vid_w - nw) // 2
            oy = vid_y + (vid_h - nh) // 2

            resized_cam = cam_frame.resize((nw, nh), Image.Resampling.NEAREST)

            # Mask for smooth rounded corners (cached)
            mask_key = (nw, nh, 16)
            if self._cached_cam_mask_key != mask_key:
                mask = Image.new("L", (nw, nh), 0)
                mdraw = ImageDraw.Draw(mask)
                mdraw.rounded_rectangle([0, 0, nw, nh], radius=16, fill=255)
                self._cached_cam_mask = mask
                self._cached_cam_mask_key = mask_key
            else:
                mask = self._cached_cam_mask

            img.paste(resized_cam, (ox, oy), mask)
        else:
            # Camera Idle Screen
            fcx = vid_x + vid_w // 2
            fcy = vid_y + vid_h // 2
            # Minimal camera outline icon
            draw.rounded_rectangle([fcx - 36, fcy - 24, fcx + 36, fcy + 24],
                                   radius=8, outline=hex2rgb(TEXT_MUTED), width=2)
            draw.ellipse([fcx - 14, fcy - 14, fcx + 14, fcy + 14],
                         outline=hex2rgb(TEXT_MUTED), width=2)
            _draw_text(draw, fcx, fcy + 42, "SIGNER CAMERA FEED LIVE", size=15,
                       fill=TEXT_MUTED, bold=True, anchor="mm")

        # Top-Left Live Badge Overlay
        badge_w, badge_h = 180, 32
        bx0 = vid_x + 16
        by0 = vid_y + 16
        draw_gradient_pill(img, bx0, by0, badge_w, badge_h, "#1E293B", "#0F172A")
        draw.rounded_rectangle([bx0, by0, bx0 + badge_w, by0 + badge_h],
                               radius=badge_h // 2, outline=hex2rgb("#334155"), width=1)
        draw.ellipse([bx0 + 12, by0 + 11, bx0 + 22, by0 + 21], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, bx0 + 30, by0 + badge_h // 2, "LIVE SIGNER FEED",
                   size=11, fill=WHITE, bold=True, anchor="lm")

        # Top-Right Camera ID Badge Overlay
        tag_w, tag_h = 140, 30
        tx0 = vid_x + vid_w - tag_w - 16
        ty0 = vid_y + 16
        draw_gradient_pill(img, tx0, ty0, tag_w, tag_h, "#1E293B", "#0F172A")
        _draw_text(draw, tx0 + tag_w // 2, ty0 + tag_h // 2, "CAM 01 • FSL SIGNER",
                   size=10, fill=TEXT_FAINT, bold=True, anchor="mm")

        # Bottom Caption Overlay (Live spoken/signed text from FSL signer)
        if self.latest_signer_text:
            cap_h = 44
            cap_y = vid_y + vid_h - cap_h - 16
            cap_w = vid_w - 40
            cap_x = vid_x + 20
            draw_gradient_pill(img, cap_x, cap_y, cap_w, cap_h, "#0F172A", "#1E293B")
            draw.rounded_rectangle([cap_x, cap_y, cap_x + cap_w, cap_y + cap_h],
                                   radius=cap_h // 2, outline=hex2rgb(BLUE_PRIMARY), width=1)
            disp_txt = f'Signer: "{self.latest_signer_text}"'
            _draw_text(draw, cap_x + 24, cap_y + cap_h // 2, disp_txt,
                       size=15, fill=WHITE, bold=True, anchor="lm")

    # ── Render BMO Companion Header & Conversation Stream ──────────────────────
    def _draw_companion_and_chat(self, img, draw, now):
        rx, ry, rw, rh = self.RIGHT_X, self.CONTENT_Y, self.RIGHT_W, self.CONTENT_H
        bmo_h = 104

        # ── 1. Top Card: BMO Companion Header ──
        av_d = 70
        av_x = rx + 20
        av_y = ry + (bmo_h - av_d) // 2
        draw_gradient_pill(img, av_x, av_y, av_d, av_d, "#38BDF8", "#2563EB", outline_col="#93C5FD")

        # BMO Face
        fc_cx = av_x + av_d // 2
        fc_cy = av_y + av_d // 2
        draw.ellipse([fc_cx - 15, fc_cy - 4, fc_cx - 9, fc_cy + 2], fill=hex2rgb(TEXT_MAIN))
        draw.ellipse([fc_cx + 9, fc_cy - 4, fc_cx + 15, fc_cy + 2], fill=hex2rgb(TEXT_MAIN))
        draw.arc([fc_cx - 8, fc_cy + 2, fc_cx + 8, fc_cy + 10], start=20, end=160, fill=hex2rgb(TEXT_MAIN), width=2)

        # Labels
        tx = av_x + av_d + 16
        _draw_text(draw, tx, fc_cy - 12, "BMO Companion", size=19, fill=TEXT_MAIN, bold=True, anchor="lm")
        w_title = draw.textlength("BMO Companion", font=_get_font(19, bold=True))

        badge_x = int(tx + w_title + 14)
        badge_y = fc_cy - 20
        badge_w, badge_h = 74, 22
        draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                               radius=badge_h // 2, fill=hex2rgb(GREEN_BG),
                               outline=hex2rgb(GREEN_BORDER), width=1)
        draw.ellipse([badge_x + 8, badge_y + 7, badge_x + 14, badge_y + 13], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, badge_x + 20, badge_y + 11, "ONLINE", size=9, fill=GREEN_TEXT, bold=True, anchor="lm")

        _draw_text(draw, tx, fc_cy + 14, "Two-way communication history", size=12, fill=TEXT_MUTED, bold=False, anchor="lm")

        # Equalizer Waveform
        eq_x = rx + rw - 72
        eq_y = fc_cy
        eq_amps = [4, 9, 15, 22, 13, 8, 4]
        for i, amp in enumerate(eq_amps):
            bx = eq_x + i * 6
            draw.line([(bx, eq_y - amp), (bx, eq_y + amp)], fill=hex2rgb(BLUE_PRIMARY), width=3)

        # ── 2. Bottom Card: Conversation Stream ──
        chat_y = ry + bmo_h + 16
        chat_h = rh - bmo_h - 16

        # Header: • CONVERSATION STREAM
        draw.ellipse([rx + 24, chat_y + 24, rx + 32, chat_y + 32], fill=hex2rgb(BLUE_PRIMARY))
        _draw_text(draw, rx + 40, chat_y + 28, "CONVERSATION STREAM", size=13, fill=BLUE_PRIMARY, bold=True, anchor="lm")

        # Render Speech Bubbles (Latest 4 messages)
        recent_msgs = self._chat_history[-4:] if self._chat_history else []

        if not recent_msgs:
            # Placeholder in center
            cx_mid = rx + rw // 2
            cy_mid = chat_y + chat_h // 2
            _draw_text(draw, cx_mid, cy_mid - 12, "Waiting for conversation...",
                       size=16, fill=TEXT_MUTED, bold=True, anchor="mm")
            _draw_text(draw, cx_mid, cy_mid + 16, "Sent and received sentences will appear here as text bubbles",
                       size=12, fill=TEXT_FAINT, bold=False, anchor="mm")
            return

        cur_y = chat_y + 54
        for msg in recent_msgs:
            sender = msg["sender"]
            txt    = msg["text"]
            t_str  = msg["time"]
            tokens = msg.get("tokens", [])

            b_w = min(rw - 48, max(240, len(txt) * 11 + 60))
            b_h = 62 + (28 if tokens else 0)

            if sender == "signer":
                # Left-aligned Signer Bubble
                bx = rx + 24
                draw.rounded_rectangle([bx, cur_y, bx + b_w, cur_y + b_h],
                                       radius=14, fill=hex2rgb(BUBBLE_FSL),
                                       outline=hex2rgb(BUBBLE_FSL_BD), width=1)
                _draw_text(draw, bx + 14, cur_y + 14, f"Signer (FSL) • {t_str}",
                           size=10, fill=BLUE_PRIMARY, bold=True, anchor="lm")
                _draw_text(draw, bx + 14, cur_y + 38, txt, size=15, fill=TEXT_MAIN, bold=True, anchor="lm")
            else:
                # Right-aligned Partner Bubble
                bx = rx + rw - b_w - 24
                draw.rounded_rectangle([bx, cur_y, bx + b_w, cur_y + b_h],
                                       radius=14, fill=hex2rgb(BUBBLE_PART),
                                       outline=hex2rgb(BUBBLE_PART_BD), width=1)
                _draw_text(draw, bx + b_w - 14, cur_y + 14, f"You (Partner) • {t_str}",
                           size=10, fill=GREEN_TEXT, bold=True, anchor="rm")
                _draw_text(draw, bx + b_w - 14, cur_y + 38, txt, size=15, fill=TEXT_MAIN, bold=True, anchor="rm")

                # Tiny Icon Previews below message text
                if tokens:
                    px = bx + b_w - 14 - len(tokens) * 28
                    py = cur_y + 54
                    for t in tokens:
                        if t in self._cached_icons:
                            b_icon = self._cached_icons[t]["bubble"]
                            img.paste(b_icon, (px, py), b_icon)
                        px += 28

            cur_y += b_h + 10

    # ── Render Sentence Strip ──────────────────────────────────────────────────
    def _draw_sentence_strip(self, img, draw, now):
        st_x, st_y, st_w, st_h = self.PAD, self.STRIP_Y, self.W - self.PAD * 2, self.STRIP_H

        # Header: Audio Equalizer + Title + Counter
        hdr_x = st_x + 36
        hdr_y = st_y + 26
        w_amps = [4, 9, 14, 8]
        for i, amp in enumerate(w_amps):
            xi = hdr_x + i * 5
            draw.line([(xi, hdr_y - amp), (xi, hdr_y + amp)], fill=hex2rgb(BLUE_PRIMARY), width=2)
        _draw_text(draw, hdr_x + 28, hdr_y, "YOUR SENTENCE STRIP", size=13, fill=BLUE_PRIMARY, bold=True, anchor="lm")

        count_str = f"({len(self.sentence_tokens)} signs selected)"
        _draw_text(draw, hdr_x + 210, hdr_y, count_str, size=12, fill=TEXT_MUTED, bold=False, anchor="lm")

        # Inset Pill Container Track
        row_y = st_y + 50
        row_h = 66
        btn_w = 175
        gap = 14
        send_x = st_x + st_w - btn_w - 36
        clr_x  = send_x - 110 - gap
        del_x  = clr_x - 110 - gap
        inp_x  = st_x + 36
        inp_w  = del_x - inp_x - 20

        # Draw Selected Tokens inside track
        self._strip_item_rects.clear()

        if not self.sentence_tokens:
            placeholder = "Tap signs from the palette below to compose your sentence..."
            _draw_text(draw, inp_x + 28, row_y + row_h // 2, placeholder,
                       size=15, fill=TEXT_FAINT, bold=False, anchor="lm")
        else:
            cur_tx = inp_x + 18
            for idx, item in enumerate(self.sentence_tokens):
                key = item["key"]
                disp = item["display"]
                tok_w = 110
                tok_h = 48
                tok_y = row_y + (row_h - tok_h) // 2

                is_hov = (self._hovered_strip_idx == idx)
                fill_c = "#DBEAFE" if is_hov else "#E0EEFA"
                bord_c = BLUE_PRIMARY if is_hov else "#93C5FD"

                draw.rounded_rectangle([cur_tx, tok_y, cur_tx + tok_w, tok_y + tok_h],
                                       radius=14, fill=hex2rgb(fill_c),
                                       outline=hex2rgb(bord_c), width=1)

                if key in self._cached_icons:
                    s_icon = self._cached_icons[key]["strip"]
                    img.paste(s_icon, (cur_tx + 8, tok_y + 5), s_icon)

                _draw_text(draw, cur_tx + 52, tok_y + tok_h // 2, disp,
                           size=12, fill=TEXT_MAIN, bold=True, anchor="lm")

                # Remove 'x' on right
                _draw_text(draw, cur_tx + tok_w - 14, tok_y + tok_h // 2, "×",
                           size=13, fill=TEXT_MUTED, bold=True, anchor="mm")

                self._strip_item_rects.append((idx, cur_tx, tok_y, cur_tx + tok_w, tok_y + tok_h))
                cur_tx += tok_w + 10

        # ── Action Buttons on Right ──
        # 1. Delete Button (⌫ Delete)
        is_del_click = time.time() < self._del_click_until
        is_del_hov   = self._del_hover and not is_del_click
        del_w, del_h = 110, row_h
        self._btn_del_rect = (del_x, row_y, del_x + del_w, row_y + del_h)
        dc1, dc2 = ("#E2E8F0", "#CBD5E1") if is_del_click else (("#F1F5F9", "#E2E8F0") if is_del_hov else ("#FFFFFF", "#F8FAFC"))
        draw_gradient_pill(img, del_x, row_y, del_w, del_h, dc1, dc2, outline_col="#CBD5E1")
        _draw_text(draw, del_x + del_w // 2, row_y + del_h // 2, "⌫ Delete",
                   size=14, fill=TEXT_MAIN, bold=True, anchor="mm")

        # 2. Clear Button (✕ Clear)
        is_clr_click = time.time() < self._clr_click_until
        is_clr_hov   = self._clr_hover and not is_clr_click
        clr_w, clr_h = 110, row_h
        self._btn_clr_rect = (clr_x, row_y, clr_x + clr_w, row_y + clr_h)
        cc1, cc2 = ("#FECACA", "#FCA5A5") if is_clr_click else (("#FEE2E2", "#FECACA") if is_clr_hov else (RED_BG, "#FEE2E2"))
        draw_gradient_pill(img, clr_x, row_y, clr_w, clr_h, cc1, cc2, outline_col=RED_BORDER)
        _draw_text(draw, clr_x + clr_w // 2, row_y + clr_h // 2, "✕ Clear",
                   size=14, fill=RED_BTN, bold=True, anchor="mm")

        # 3. SEND Button (SEND ➔)
        is_send_click = time.time() < self._send_click_until
        is_send_hov   = self._send_hover and not is_send_click
        send_h = row_h
        self._btn_send_rect = (send_x, row_y, send_x + btn_w, row_y + send_h)

        if is_send_click:
            sx, sy, sw, sh = send_x + 2, row_y + 2, btn_w - 4, send_h - 4
            sc1, sc2 = "#0369A1", "#0284C7"
        elif is_send_hov:
            sx, sy, sw, sh = send_x - 1, row_y - 1, btn_w + 2, send_h + 2
            sc1, sc2 = "#38BDF8", "#0284C7"
        else:
            sx, sy, sw, sh = send_x, row_y, btn_w, send_h
            sc1, sc2 = "#0EA5E9", "#0369A1"

        glow_pad = 10
        glow_send = Image.new("RGBA", (sw + glow_pad * 2, sh + glow_pad * 2), (0, 0, 0, 0))
        gsdraw = ImageDraw.Draw(glow_send)
        gsdraw.rounded_rectangle([glow_pad, glow_pad + 4, glow_pad + sw, glow_pad + sh + 4],
                                 radius=sh // 2, fill=(14, 165, 233, 85))
        glow_send = glow_send.filter(ImageFilter.GaussianBlur(8))
        img.paste(glow_send, (sx - glow_pad, sy - glow_pad), glow_send)

        draw_gradient_pill(img, sx, sy, sw, sh, sc1, sc2)
        draw.arc([sx + 1, sy + 1, sx + sw - 2, sy + sh - 2], start=190, end=350, fill=hex2rgb(WHITE), width=1)

        # Send Arrow Icon
        arr_cx = sx + 32
        arr_cy = sy + sh // 2
        draw.line([(arr_cx - 10, arr_cy), (arr_cx + 8, arr_cy)], fill=hex2rgb(WHITE), width=3)
        draw.polygon([(arr_cx + 8, arr_cy), (arr_cx + 2, arr_cy - 6), (arr_cx + 2, arr_cy + 6)], fill=hex2rgb(WHITE))
        _draw_text(draw, sx + 54, sy + sh // 2, "SEND", size=15, fill=WHITE, bold=True, anchor="lm")

    # ── Render Sign Palette Grid ───────────────────────────────────────────────
    def _draw_sign_palette(self, img, draw, now):
        pal_x, pal_y, pal_w, pal_h = self.PAD, self.PALETTE_Y, self.W - self.PAD * 2, self.PALETTE_H

        # Header: Title + Available Count
        _draw_text(draw, pal_x + 36, pal_y + 26, "SIGN PALETTE", size=14, fill=TEXT_MAIN, bold=True, anchor="lm")
        count_str = f"({len(self.palette)} signs available • Trained & Ready)"
        _draw_text(draw, pal_x + 160, pal_y + 26, count_str, size=12, fill=TEXT_MUTED, bold=False, anchor="lm")

        # Horizontal Row of Large Accessible Sign Cards
        card_w = 110
        card_h = 130
        gap = 16
        start_x = pal_x + 36
        start_y = pal_y + 54

        self._palette_rects.clear()

        for idx, item in enumerate(self.palette):
            key = item["key"]
            disp = item["display"]
            cx = start_x + idx * (card_w + gap)
            cy = start_y

            is_hov = (self._hovered_palette_key == key)
            fill_c = "#E0EEFA" if is_hov else "#FFFFFF"
            bord_c = BLUE_PRIMARY if is_hov else "#CBDDF2"
            bord_w = 2 if is_hov else 1

            draw.rounded_rectangle([cx, cy, cx + card_w, cy + card_h],
                                   radius=18, fill=hex2rgb(fill_c),
                                   outline=hex2rgb(bord_c), width=bord_w)

            # Icon Thumbnail
            if key in self._cached_icons:
                icon_im = self._cached_icons[key]["palette"]
                ix = cx + (card_w - 64) // 2
                iy = cy + 14
                img.paste(icon_im, (ix, iy), icon_im)

            # Label below icon
            _draw_text(draw, cx + card_w // 2, cy + card_h - 22, disp,
                       size=12, fill=TEXT_MAIN, bold=True, anchor="mm")

            self._palette_rects.append((key, item, cx, cy, cx + card_w, cy + card_h))

    # ── Notification Toast ─────────────────────────────────────────────────────
    def _draw_notification(self, img, draw, now):
        if now >= self.notification_end:
            return
        alpha = min(1.0, (self.notification_end - now) / 0.35)
        txt   = self.notification_text
        nw    = max(220, len(txt) * 14 + 52)
        nh    = 48
        nx    = self.W // 2 - nw // 2
        ny    = self.CONTENT_Y + self.CONTENT_H // 2 - nh // 2

        a = int(220 * alpha)
        toast = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
        td = ImageDraw.Draw(toast)
        c = hex2rgb(BLUE_PRIMARY)
        td.rounded_rectangle([0, 0, nw, nh], radius=nh // 2, fill=c + (a,))

        bg_roi = img.crop((nx, ny, nx + nw, ny + nh)).convert("RGBA")
        merged = Image.alpha_composite(bg_roi, toast).convert("RGB")
        img.paste(merged, (nx, ny))

        draw2 = ImageDraw.Draw(img)
        _draw_text(draw2, nx + nw // 2, ny + nh // 2, txt, size=16, fill=WHITE, bold=True, anchor="mm")

    def _set_notification(self, msg, duration=2.5):
        self.notification_text = msg
        self.notification_end  = time.time() + duration

    # ── Mouse Interaction Handlers ─────────────────────────────────────────────
    def _on_motion(self, event):
        if not hasattr(self, 'ui_scale') or self.ui_scale <= 0:
            return
        mx = int((event.x - self.ui_ox) / self.ui_scale)
        my = int((event.y - self.ui_oy) / self.ui_scale)

        # Hover over Send
        x0, y0, x1, y1 = self._btn_send_rect
        self._send_hover = (x0 <= mx <= x1 and y0 <= my <= y1)

        # Hover over Delete
        x0, y0, x1, y1 = self._btn_del_rect
        self._del_hover = (x0 <= mx <= x1 and y0 <= my <= y1)

        # Hover over Clear
        x0, y0, x1, y1 = self._btn_clr_rect
        self._clr_hover = (x0 <= mx <= x1 and y0 <= my <= y1)

        # Hover over Palette tiles
        hov_key = None
        for key, item, px0, py0, px1, py1 in self._palette_rects:
            if px0 <= mx <= px1 and py0 <= my <= py1:
                hov_key = key
                break
        self._hovered_palette_key = hov_key

        # Hover over Strip tokens
        hov_strip = None
        for idx, sx0, sy0, sx1, sy1 in self._strip_item_rects:
            if sx0 <= mx <= sx1 and sy0 <= my <= sy1:
                hov_strip = idx
                break
        self._hovered_strip_idx = hov_strip

        # Hover over top-bar visibility buttons
        hx0, hy0, hx1, hy1 = getattr(self, "_hide_btn_rect", (0, 0, 0, 0))
        sx0, sy0, sx1, sy1 = getattr(self, "_show_main_btn_rect", (0, 0, 0, 0))
        self._hide_hover = (hx0 <= mx <= hx1 and hy0 <= my <= hy1)
        self._show_main_hover = (sx0 <= mx <= sx1 and sy0 <= my <= sy1)
        if (self._send_hover or self._del_hover or self._clr_hover or
                self._hovered_palette_key is not None or self._hovered_strip_idx is not None or
                self._hide_hover or self._show_main_hover):
            self.window.configure(cursor="hand2")
        else:
            self.window.configure(cursor="")

    def _on_click(self, event):
        if not hasattr(self, 'ui_scale') or self.ui_scale <= 0:
            return
        mx = int((event.x - self.ui_ox) / self.ui_scale)
        my = int((event.y - self.ui_oy) / self.ui_scale)

        # 0. Window visibility buttons (top bar)
        x0, y0, x1, y1 = getattr(self, "_hide_btn_rect", (0, 0, 0, 0))
        if x0 <= mx <= x1 and y0 <= my <= y1:
            self.hide_self()
            return
        x0, y0, x1, y1 = getattr(self, "_show_main_btn_rect", (0, 0, 0, 0))
        if x0 <= mx <= x1 and y0 <= my <= y1:
            self.show_main()
            return

        # 1. SEND Button Click
        x0, y0, x1, y1 = self._btn_send_rect
        if x0 <= mx <= x1 and y0 <= my <= y1:
            self._send_click_until = time.time() + 0.18
            self._send_sentence()
            return

        # 2. Delete Button Click
        x0, y0, x1, y1 = self._btn_del_rect
        if x0 <= mx <= x1 and y0 <= my <= y1:
            self._del_click_until = time.time() + 0.18
            if self.sentence_tokens:
                self.sentence_tokens.pop()
            return

        # 3. Clear Button Click
        x0, y0, x1, y1 = self._btn_clr_rect
        if x0 <= mx <= x1 and y0 <= my <= y1:
            self._clr_click_until = time.time() + 0.18
            self.sentence_tokens.clear()
            return

        # 4. Token Remove click in sentence strip
        for idx, sx0, sy0, sx1, sy1 in self._strip_item_rects:
            if sx0 <= mx <= sx1 and sy0 <= my <= sy1:
                if 0 <= idx < len(self.sentence_tokens):
                    self.sentence_tokens.pop(idx)
                return

        # 5. Palette Tile Click -> Add to strip
        for key, item, px0, py0, px1, py1 in self._palette_rects:
            if px0 <= mx <= px1 and py0 <= my <= py1:
                self.sentence_tokens.append(item)
                return

    # ── Sending sentence to FSL Window ─────────────────────────────────────────
    def _send_sentence(self):
        if not self.sentence_tokens:
            self._set_notification("Select signs to compose sentence", duration=1.8)
            return

        tokens = [item["key"] for item in self.sentence_tokens]
        gloss = " ".join(tokens)
        translated = translate_sentence(gloss)

        # Deliver to FSL window via CoOp bus
        coop_hub.send_from_partner(tokens, translated)

        # Play TTS speech
        try:
            from main import speak_text
            speak_text(translated)
        except Exception:
            pass

        # Record in local conversation history
        time_str = time.strftime("%I:%M %p")
        self._chat_history.append({
            "sender": "partner",
            "text": translated,
            "tokens": list(tokens),
            "time": time_str,
        })

        self._set_notification(f"SENT: {translated}")
        self.sentence_tokens.clear()

    # ── Receiving message from FSL Window ──────────────────────────────────────
    def _on_signer_message_received(self, text: str):
        time_str = time.strftime("%I:%M %p")
        self._chat_history.append({
            "sender": "signer",
            "text": text,
            "tokens": [],
            "time": time_str,
        })
        self.latest_signer_text = text
        self._set_notification(f"SIGNER: {text[:28]}")
        try:
            from main import play_ping_sound
            play_ping_sound()
        except Exception:
            pass

    # ── Window visibility / idle helpers ─────────────────────────────────────
    def is_window_visible(self):
        return bool(getattr(self, "_visible", True))

    def _main_is_visible(self):
        fn = getattr(self, "is_main_visible_fn", None)
        if callable(fn):
            return bool(fn())
        app = getattr(self, "main_app", None)
        if app is not None and hasattr(app, "is_window_visible"):
            return bool(app.is_window_visible())
        return True

    def hide_self(self):
        if not self._visible:
            return
        if not self._main_is_visible():
            self._set_notification("Keep one window visible")
            print("[Partner] hide refused: main not visible")
            return
        self._visible = False
        try:
            self.window.withdraw()
            print("[Partner] hidden")
        except Exception as e:
            print(f"[Partner] hide error: {e}")

    def show_self(self):
        self._visible = True
        try:
            self.window.deiconify()
            self.window.lift()
        except Exception as e:
            print(f"[Partner] show error: {e}")
        if self._running:
            self.window.after(1, self._gui_loop)
        print("[Partner] shown")

    def show_main(self):
        app = getattr(self, "main_app", None)
        print(f"[Partner] show_main click, main_app={app is not None}")
        if app and hasattr(app, "show_self"):
            app.show_self()
        else:
            self._set_notification("Main window unavailable")

    def _on_close(self):
        self._running = False
        self._visible = False
        if self.on_close_callback:
            self.on_close_callback()
        try:
            self.window.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    app = NonFSLWindow()
    root.mainloop()
