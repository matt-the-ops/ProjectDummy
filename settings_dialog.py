"""
SHIELD FSL - Neumorphic Settings & Keybinds Customization Dialog
Provides user-friendly controls for Graphics, Performance Mode, and Keybind Re-mapping.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from settings_manager import settings, KEYBIND_LABELS, PRESETS, get_gpu_device_info


class SettingsDialog(tk.Toplevel):
    def __init__(self, master, on_change_callback=None):
        super().__init__(master)
        self.master = master
        self.on_change_callback = on_change_callback
        self.title("Project B.M.O — Settings & Performance")
        self.geometry("820x700")
        self.minsize(760, 640)
        self.configure(bg="#0B132B")
        self.transient(master)
        self.grab_set()

        # Center on master
        try:
            mw = master.winfo_width()
            mh = master.winfo_height()
            mx = master.winfo_rootx()
            my = master.winfo_rooty()
            w, h = 820, 700
            x = mx + max(0, (mw - w) // 2)
            y = my + max(0, (mh - h) // 2)
            self.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass

        self.listening_action = None  # Action currently awaiting keypress

        # Try setting icon
        try:
            icon_p = os.path.join(os.path.dirname(__file__), "assets", "shield_logo.png")
            if os.path.exists(icon_p):
                im = Image.open(icon_p).resize((32, 32), Image.Resampling.LANCZOS)
                self._icon_tk = ImageTk.PhotoImage(im)
                self.iconphoto(False, self._icon_tk)
        except Exception:
            pass

        self._build_ui()
        self.bind("<KeyPress>", self._on_keypress)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Header Bar
        header = tk.Frame(self, bg="#111C3A", height=70)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        title_frame = tk.Frame(header, bg="#111C3A")
        title_frame.pack(side=tk.LEFT, padx=24, pady=12)

        lbl_title = tk.Label(title_frame, text="⚙ SETTINGS & PERFORMANCE", font=("Segoe UI", 16, "bold"),
                             fg="#FFFFFF", bg="#111C3A")
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(title_frame, text="Configure visual presets, potato PC optimizations, and keybinds",
                           font=("Segoe UI", 10), fg="#93C5FD", bg="#111C3A")
        lbl_sub.pack(anchor="w")

        btn_close = tk.Button(header, text="✕", font=("Segoe UI", 14, "bold"), fg="#94A3B8", bg="#1E293B",
                              activebackground="#EF4444", activeforeground="#FFFFFF", bd=0, relief="flat",
                              cursor="hand2", padx=12, pady=4, command=self._on_close)
        btn_close.pack(side=tk.RIGHT, padx=20, pady=16)

        # Tab Selector Bar
        tab_bar = tk.Frame(self, bg="#0B132B", height=46)
        tab_bar.pack(fill=tk.X, padx=24, pady=(14, 0))

        self.active_tab = "perf"
        self.btn_tab_perf = tk.Button(tab_bar, text="⚡ Performance & Graphics", font=("Segoe UI", 11, "bold"),
                                      bg="#2563EB", fg="#FFFFFF", bd=0, relief="flat", padx=16, pady=8,
                                      cursor="hand2", command=lambda: self._switch_tab("perf"))
        self.btn_tab_perf.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_tab_keys = tk.Button(tab_bar, text="⌨ Customizable Keybinds", font=("Segoe UI", 11, "bold"),
                                      bg="#1E293B", fg="#94A3B8", bd=0, relief="flat", padx=16, pady=8,
                                      cursor="hand2", command=lambda: self._switch_tab("keys"))
        self.btn_tab_keys.pack(side=tk.LEFT)

        # Content Container
        self.content_frame = tk.Frame(self, bg="#0B132B")
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=12)

        # Build individual tabs
        self.perf_frame = tk.Frame(self.content_frame, bg="#0B132B")
        self.keys_frame = tk.Frame(self.content_frame, bg="#0B132B")

        self._build_perf_tab()
        self._build_keys_tab()

        self._switch_tab("perf")

        # Bottom Bar
        footer = tk.Frame(self, bg="#111C3A", height=60)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        self.lbl_status = tk.Label(footer, text="Settings auto-save on change", font=("Segoe UI", 10),
                                   fg="#10B981", bg="#111C3A")
        self.lbl_status.pack(side=tk.LEFT, padx=24)

        btn_save = tk.Button(footer, text="Save & Done", font=("Segoe UI", 11, "bold"),
                             bg="#2563EB", fg="#FFFFFF", activebackground="#1D4ED8", activeforeground="#FFFFFF",
                             bd=0, relief="flat", cursor="hand2", padx=20, pady=8,
                             command=self._on_close)
        btn_save.pack(side=tk.RIGHT, padx=24, pady=10)

    def _switch_tab(self, tab):
        self.active_tab = tab
        if tab == "perf":
            self.keys_frame.pack_forget()
            self.perf_frame.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_perf.configure(bg="#2563EB", fg="#FFFFFF")
            self.btn_tab_keys.configure(bg="#1E293B", fg="#94A3B8")
        else:
            self.perf_frame.pack_forget()
            self.keys_frame.pack(fill=tk.BOTH, expand=True)
            self.btn_tab_perf.configure(bg="#1E293B", fg="#94A3B8")
            self.btn_tab_keys.configure(bg="#2563EB", fg="#FFFFFF")

    # ── Performance & Graphics Tab ─────────────────────────────────
    def _build_perf_tab(self):
        # 1. Big Hero Master Toggle Card
        hero_card = tk.Frame(self.perf_frame, bg="#152347", bd=1, relief="solid", highlightthickness=0)
        hero_card.pack(fill=tk.X, pady=(0, 14), padx=2)

        hero_inner = tk.Frame(hero_card, bg="#152347")
        hero_inner.pack(fill=tk.X, padx=18, pady=14)

        hero_left = tk.Frame(hero_inner, bg="#152347")
        hero_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        is_perf = settings.get("performance_mode", False)
        tk.Label(hero_left, text="⚡ ULTRA-PERFORMANCE MODE (LOW-END PC)", font=("Segoe UI", 13, "bold"),
                 fg="#F59E0B" if is_perf else "#38BDF8", bg="#152347").pack(anchor="w")

        tk.Label(hero_left,
                 text="Cuts CPU/GPU overhead to run smoothly on low-spec PCs, Intel Celerons, and integrated graphics.\n"
                      "Optimizes MediaPipe resolution, disables heavy 468-point face mesh, and limits kiosk refresh.",
                 font=("Segoe UI", 9), fg="#94A3B8", bg="#152347", justify="left").pack(anchor="w", pady=(4, 0))

        # Toggle Button
        self.btn_perf_toggle = tk.Button(hero_inner, text="⚡ ACTIVE [ON]" if is_perf else "⚡ OFF",
                                         font=("Segoe UI", 12, "bold"),
                                         bg="#10B981" if is_perf else "#334155",
                                         fg="#FFFFFF", activebackground="#059669", activeforeground="#FFFFFF",
                                         bd=0, relief="flat", cursor="hand2", padx=18, pady=10,
                                         command=self._toggle_perf_mode)
        self.btn_perf_toggle.pack(side=tk.RIGHT, padx=10)

        # 1B. GPU Hardware Acceleration Card
        gpu_card = tk.Frame(self.perf_frame, bg="#152347", bd=1, relief="solid", highlightthickness=0)
        gpu_card.pack(fill=tk.X, pady=(0, 14), padx=2)

        gpu_inner = tk.Frame(gpu_card, bg="#152347")
        gpu_inner.pack(fill=tk.X, padx=18, pady=10)

        gpu_left = tk.Frame(gpu_inner, bg="#152347")
        gpu_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        has_gpu, gpu_dev_name = get_gpu_device_info()
        is_gpu = settings.get("use_gpu", True)

        gpu_title_frame = tk.Frame(gpu_left, bg="#152347")
        gpu_title_frame.pack(anchor="w")

        tk.Label(gpu_title_frame, text="⚡ HARDWARE ACCELERATION (GPU)", font=("Segoe UI", 12, "bold"),
                 fg="#10B981" if is_gpu else "#94A3B8", bg="#152347").pack(side=tk.LEFT)

        dev_badge = f"🟢 {gpu_dev_name}" if has_gpu else "⚠️ CPU Fallback"
        tk.Label(gpu_title_frame, text=f" [{dev_badge}]", font=("Segoe UI", 9, "bold"),
                 fg="#38BDF8" if has_gpu else "#F59E0B", bg="#152347").pack(side=tk.LEFT, padx=6)

        tk.Label(gpu_left,
                 text=f"Offloads video pipeline, color conversion, and image tensor resizing to your GPU.\n"
                      f"Dedicated graphics adapter: {gpu_dev_name}",
                 font=("Segoe UI", 9), fg="#94A3B8", bg="#152347", justify="left").pack(anchor="w", pady=(2, 0))

        self.btn_gpu_toggle = tk.Button(gpu_inner, text="⚡ GPU ON" if is_gpu else "⚡ CPU ONLY",
                                        font=("Segoe UI", 11, "bold"),
                                        bg="#10B981" if is_gpu else "#334155",
                                        fg="#FFFFFF", activebackground="#059669", activeforeground="#FFFFFF",
                                        bd=0, relief="flat", cursor="hand2", padx=16, pady=8,
                                        command=self._toggle_gpu_mode)
        self.btn_gpu_toggle.pack(side=tk.RIGHT, padx=10)

        # 2. Preset Selector
        preset_frame = tk.Frame(self.perf_frame, bg="#0B132B")
        preset_frame.pack(fill=tk.X, pady=(0, 14))

        tk.Label(preset_frame, text="HARDWARE PRESETS", font=("Segoe UI", 10, "bold"),
                 fg="#93C5FD", bg="#0B132B").pack(anchor="w", pady=(0, 6))

        p_row = tk.Frame(preset_frame, bg="#0B132B")
        p_row.pack(fill=tk.X)

        curr_preset = settings.get("performance_preset", "balanced")
        self.preset_btns = {}
        preset_defs = [
            ("potato", "🥔 Potato PC (Ultra-Low)", "For slow dual-cores, 240p ML, zero face mesh"),
            ("balanced", "⚖️ Balanced (Recommended)", "Smooth 60 FPS, 320p ML, calibration mesh"),
            ("quality", "✨ High Fidelity", "Full 480p ML, complete face mesh, high FPS"),
        ]
        for pid, label, desc in preset_defs:
            is_active = (curr_preset == pid)
            c = tk.Frame(p_row, bg="#1E293B" if not is_active else "#1D4ED8", padx=2, pady=2, cursor="hand2")
            c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

            b = tk.Button(c, text=f"{label}\n{desc}", font=("Segoe UI", 9, "bold" if is_active else "normal"),
                          bg="#172554" if is_active else "#1E293B", fg="#FFFFFF" if is_active else "#CBD5E1",
                          activebackground="#1E40AF", activeforeground="#FFFFFF", bd=0, relief="flat",
                          cursor="hand2", padx=8, pady=8, justify="center",
                          command=lambda p=pid: self._choose_preset(p))
            b.pack(fill=tk.BOTH, expand=True)
            self.preset_btns[pid] = (c, b)

        # 3. Screen Resolution & Fullscreen Selector
        res_frame = tk.Frame(self.perf_frame, bg="#0B132B")
        res_frame.pack(fill=tk.X, pady=(0, 14))

        tk.Label(res_frame, text="DISPLAY RESOLUTION & FULLSCREEN", font=("Segoe UI", 10, "bold"),
                 fg="#93C5FD", bg="#0B132B").pack(anchor="w", pady=(0, 6))

        r_row = tk.Frame(res_frame, bg="#0B132B")
        r_row.pack(fill=tk.X)

        curr_res = settings.get("screen_resolution", "auto")
        self.res_btns = {}
        res_defs = [
            ("720p", "📺 720p HD", "1280×720"),
            ("900p", "🖥️ 900p HD+", "1600×900"),
            ("1080p", "✨ 1080p FHD", "1920×1080"),
            ("fullscreen", "🔲 Fullscreen", "Borderless"),
            ("auto", "📐 Auto-Fit", "Best for Monitor"),
        ]
        for rid, label, sub in res_defs:
            is_active = (curr_res == rid)
            c = tk.Frame(r_row, bg="#1E293B" if not is_active else "#1D4ED8", padx=2, pady=2, cursor="hand2")
            c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

            b = tk.Button(c, text=f"{label}\n{sub}", font=("Segoe UI", 9, "bold" if is_active else "normal"),
                          bg="#172554" if is_active else "#1E293B", fg="#FFFFFF" if is_active else "#CBD5E1",
                          activebackground="#1E40AF", activeforeground="#FFFFFF", bd=0, relief="flat",
                          cursor="hand2", padx=4, pady=6, justify="center",
                          command=lambda r=rid: self._choose_resolution(r))
            b.pack(fill=tk.BOTH, expand=True)
            self.res_btns[rid] = (c, b)

        # 4. Resolution Scale (ML Processing Resolution)
        scale_frame = tk.Frame(self.perf_frame, bg="#0B132B")
        scale_frame.pack(fill=tk.X, pady=(0, 14))

        tk.Label(scale_frame, text="RESOLUTION SCALE (Affects Lag & Performance)", font=("Segoe UI", 10, "bold"),
                 fg="#F59E0B", bg="#0B132B").pack(anchor="w", pady=(0, 4))

        tk.Label(scale_frame, text="Lower scale = smaller ML processing = less lag", font=("Segoe UI", 8),
                 fg="#94A3B8", bg="#0B132B").pack(anchor="w", pady=(0, 6))

        scale_row = tk.Frame(scale_frame, bg="#0B132B")
        scale_row.pack(fill=tk.X)

        curr_res = settings.get("ml_w", 480)
        self.scale_btns = {}
        # Resolution scale presets - these control ML processing resolution
        scale_defs = [
            (320, 240, " 🥔 Low (320×240)", "Fastest, most lag reduction"),
            (416, 312, " ⚖️ Medium (416×312)", "Balanced quality/performance"),
            (480, 360, " ✨ High (480×360)", "Default quality"),
            (640, 480, " 🚀 Ultra (640×480)", "Maximum quality, more CPU"),
        ]
        for sw, sh, label, desc in scale_defs:
            is_active = (curr_res == sw)
            c = tk.Frame(scale_row, bg="#1E293B" if not is_active else "#1D4ED8", padx=2, pady=2, cursor="hand2")
            c.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)

            b = tk.Button(c, text=f"{label}\n{desc}", font=("Segoe UI", 8, "bold" if is_active else "normal"),
                          bg="#172554" if is_active else "#1E293B", fg="#FFFFFF" if is_active else "#CBD5E1",
                          activebackground="#1E40AF", activeforeground="#FFFFFF", bd=0, relief="flat",
                          cursor="hand2", padx=4, pady=6, justify="center",
                          command=lambda s=sw, t=sh: self._choose_resolution_scale(s, t))
            b.pack(fill=tk.BOTH, expand=True)
            self.scale_btns[sw] = (c, b)

        # 5. Fine-Grained Graphic Options
        opt_box = tk.LabelFrame(self.perf_frame, text="  Granular Graphic & Model Settings  ",
                                font=("Segoe UI", 10, "bold"), fg="#93C5FD", bg="#0B132B", bd=1)
        opt_box.pack(fill=tk.BOTH, expand=True, pady=(0, 6), padx=2)

        # Option A: Face Mesh Drawing
        row_a = tk.Frame(opt_box, bg="#0B132B")
        row_a.pack(fill=tk.X, padx=14, pady=8)
        tk.Label(row_a, text="Face Landmark Rendering:", font=("Segoe UI", 10, "bold"), fg="#E2E8F0", bg="#0B132B", width=25, anchor="w").pack(side=tk.LEFT)
        self.mesh_var = tk.StringVar(value=settings.get("face_mesh_mode", "calibrate_only"))
        for val, txt in [("off", "Fastest (Hands Only)"), ("calibrate_only", "Calibrate Only"), ("full", "Always Full")]:
            rb = tk.Radiobutton(row_a, text=txt, value=val, variable=self.mesh_var, font=("Segoe UI", 9),
                                fg="#93C5FD", bg="#0B132B", activebackground="#0B132B", activeforeground="#FFFFFF",
                                selectcolor="#1E293B", command=self._update_granular)
            rb.pack(side=tk.LEFT, padx=6)

        # Option C: Target FPS
        row_c = tk.Frame(opt_box, bg="#0B132B")
        row_c.pack(fill=tk.X, padx=14, pady=8)
        tk.Label(row_c, text="GUI Framerate Target:", font=("Segoe UI", 10, "bold"), fg="#E2E8F0", bg="#0B132B", width=25, anchor="w").pack(side=tk.LEFT)
        self.fps_var = tk.IntVar(value=settings.get("target_fps", 60))
        for val, txt in [(30, "30 FPS (Low CPU)"), (60, "60 FPS (Ultra-Smooth)")]:
            rb = tk.Radiobutton(row_c, text=txt, value=val, variable=self.fps_var, font=("Segoe UI", 9),
                                fg="#93C5FD", bg="#0B132B", activebackground="#0B132B", activeforeground="#FFFFFF",
                                selectcolor="#1E293B", command=self._update_granular)
            rb.pack(side=tk.LEFT, padx=6)

        # Option D: Partner Kiosk FPS
        row_d = tk.Frame(opt_box, bg="#0B132B")
        row_d.pack(fill=tk.X, padx=14, pady=8)
        tk.Label(row_d, text="Partner Kiosk FPS Cap:", font=("Segoe UI", 10, "bold"), fg="#E2E8F0", bg="#0B132B", width=25, anchor="w").pack(side=tk.LEFT)
        self.kiosk_var = tk.IntVar(value=settings.get("partner_fps", 30))
        for val, txt in [(15, "15 FPS (Eco Mode)"), (20, "20 FPS (Low CPU)"), (30, "30 FPS (Smooth)")]:
            rb = tk.Radiobutton(row_d, text=txt, value=val, variable=self.kiosk_var, font=("Segoe UI", 9),
                                fg="#93C5FD", bg="#0B132B", activebackground="#0B132B", activeforeground="#FFFFFF",
                                selectcolor="#1E293B", command=self._update_granular)
            rb.pack(side=tk.LEFT, padx=6)

    def _toggle_perf_mode(self):
        new_mode = settings.toggle_performance_mode()
        self.btn_perf_toggle.configure(text="⚡ ACTIVE [ON]" if new_mode else "⚡ OFF",
                                       bg="#10B981" if new_mode else "#334155")
        self._refresh_perf_ui()
        self.lbl_status.configure(text=f"Performance Mode {'ENABLED (Potato PC Preset)' if new_mode else 'DISABLED'}")
        if self.on_change_callback:
            self.on_change_callback("performance_mode", new_mode)

    def _toggle_gpu_mode(self):
        curr = settings.get("use_gpu", True)
        new_val = not curr
        settings.set("use_gpu", new_val)
        self._refresh_perf_ui()
        self.lbl_status.configure(text=f"GPU Hardware Acceleration {'ENABLED' if new_val else 'DISABLED (CPU Only)'}")
        if self.on_change_callback:
            self.on_change_callback("use_gpu", new_val)

    def _choose_preset(self, preset_name):
        settings.apply_preset(preset_name)
        self._refresh_perf_ui()
        self.lbl_status.configure(text=f"Applied preset: {preset_name.upper()}")
        if self.on_change_callback:
            self.on_change_callback("preset", preset_name)

    def _choose_resolution_scale(self, width, height):
        settings.set("ml_w", width)
        settings.set("ml_h", height)
        self._refresh_perf_ui()
        self.lbl_status.configure(text=f"Resolution scale set to: {width}×{height}")
        if self.on_change_callback:
            self.on_change_callback("resolution_scale", (width, height))

    def _choose_resolution(self, res_name):
        settings.set("screen_resolution", res_name)
        self._refresh_perf_ui()
        self.lbl_status.configure(text=f"Display resolution set to: {res_name.upper()}")
        if self.on_change_callback:
            self.on_change_callback("screen_resolution", res_name)

    def _update_granular(self):
        # Mesh
        settings.set("face_mesh_mode", self.mesh_var.get())
        # Target FPS
        settings.set("target_fps", self.fps_var.get())
        # Kiosk FPS
        settings.set("partner_fps", self.kiosk_var.get())

        self.lbl_status.configure(text="Graphic settings updated.")
        if self.on_change_callback:
            self.on_change_callback("granular", None)

    def _refresh_perf_ui(self):
        curr_preset = settings.get("performance_preset", "balanced")
        is_perf = settings.get("performance_mode", False)
        self.btn_perf_toggle.configure(text="⚡ ACTIVE [ON]" if is_perf else "⚡ OFF",
                                       bg="#10B981" if is_perf else "#334155")

        is_gpu = settings.get("use_gpu", True)
        if hasattr(self, 'btn_gpu_toggle'):
            self.btn_gpu_toggle.configure(text="⚡ GPU ON" if is_gpu else "⚡ CPU ONLY",
                                          bg="#10B981" if is_gpu else "#334155")

        for pid, (c, b) in self.preset_btns.items():
            active = (pid == curr_preset)
            c.configure(bg="#1D4ED8" if active else "#1E293B")
            b.configure(bg="#172554" if active else "#1E293B",
                        fg="#FFFFFF" if active else "#CBD5E1",
                        font=("Segoe UI", 9, "bold" if active else "normal"))

        curr_res = settings.get("screen_resolution", "auto")
        for rid, (c, b) in getattr(self, "res_btns", {}).items():
            active = (rid == curr_res)
            c.configure(bg="#1D4ED8" if active else "#1E293B")
            b.configure(bg="#172554" if active else "#1E293B",
                        fg="#FFFFFF" if active else "#CBD5E1",
                        font=("Segoe UI", 9, "bold" if active else "normal"))

        # Resolution scale buttons
        curr_ml_w = settings.get("ml_w", 480)
        for sw, (c, b) in getattr(self, "scale_btns", {}).items():
            active = (sw == curr_ml_w)
            c.configure(bg="#1D4ED8" if active else "#1E293B")
            b.configure(bg="#172554" if active else "#1E293B",
                        fg="#FFFFFF" if active else "#CBD5E1",
                        font=("Segoe UI", 8, "bold" if active else "normal"))

        self.mesh_var.set(settings.get("face_mesh_mode", "calibrate_only"))
        self.fps_var.set(settings.get("target_fps", 60))
        self.kiosk_var.set(settings.get("partner_fps", 30))

    # ── Keybinds Customization Tab ──────────────────────────────────
    def _build_keys_tab(self):
        top_row = tk.Frame(self.keys_frame, bg="#0B132B")
        top_row.pack(fill=tk.X, pady=(0, 10))

        tk.Label(top_row, text="CUSTOM KEYBOARD SHORTCUTS", font=("Segoe UI", 11, "bold"),
                 fg="#93C5FD", bg="#0B132B").pack(side=tk.LEFT)

        btn_reset_keys = tk.Button(top_row, text="↺ Reset Defaults", font=("Segoe UI", 9),
                                   bg="#334155", fg="#FFFFFF", activebackground="#475569", activeforeground="#FFFFFF",
                                   bd=0, relief="flat", cursor="hand2", padx=10, pady=4,
                                   command=self._reset_keybinds)
        btn_reset_keys.pack(side=tk.RIGHT)

        # Scrollable Canvas container for keybind list
        canvas = tk.Canvas(self.keys_frame, bg="#0B132B", highlightthickness=0)
        scrollbar = tk.Scrollbar(self.keys_frame, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg="#0B132B")

        self.scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_win = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_win, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.key_buttons = {}
        self._populate_keybind_rows()

    def _populate_keybind_rows(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.key_buttons.clear()

        keybinds = settings.get("keybinds", {})
        for act_id, (label, desc) in KEYBIND_LABELS.items():
            curr_key = keybinds.get(act_id, "").upper()

            row = tk.Frame(self.scroll_frame, bg="#152347", bd=1, relief="solid")
            row.pack(fill=tk.X, pady=4, padx=4)

            text_box = tk.Frame(row, bg="#152347")
            text_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=8)

            tk.Label(text_box, text=label, font=("Segoe UI", 10, "bold"), fg="#FFFFFF", bg="#152347").pack(anchor="w")
            tk.Label(text_box, text=desc, font=("Segoe UI", 8), fg="#94A3B8", bg="#152347").pack(anchor="w")

            btn_key = tk.Button(row, text=f"[ {curr_key} ]", font=("Consolas", 11, "bold"),
                                bg="#1E293B", fg="#38BDF8", activebackground="#2563EB", activeforeground="#FFFFFF",
                                bd=0, relief="flat", cursor="hand2", padx=16, pady=6,
                                command=lambda a=act_id: self._start_listen_key(a))
            btn_key.pack(side=tk.RIGHT, padx=12, pady=8)
            self.key_buttons[act_id] = btn_key

    def _start_listen_key(self, act_id):
        if self.listening_action:
            # Revert previous
            old_k = settings.get_keybind(self.listening_action).upper()
            if self.listening_action in self.key_buttons:
                self.key_buttons[self.listening_action].configure(text=f"[ {old_k} ]", bg="#1E293B", fg="#38BDF8")

        self.listening_action = act_id
        self.key_buttons[act_id].configure(text="[ Press Any Key... ]", bg="#F59E0B", fg="#0F172A")
        self.lbl_status.configure(text=f"Press any key to assign to '{KEYBIND_LABELS[act_id][0]}' (Escape to cancel)")

    def _on_keypress(self, event):
        if not self.listening_action:
            if event.keysym.lower() == "escape":
                self._on_close()
            return

        act = self.listening_action
        k = event.keysym.lower()

        if k == "escape":
            # Cancel rebind
            old_k = settings.get_keybind(act).upper()
            self.key_buttons[act].configure(text=f"[ {old_k} ]", bg="#1E293B", fg="#38BDF8")
            self.listening_action = None
            self.lbl_status.configure(text="Keybind change cancelled.")
            return

        # Check conflict
        existing_action = settings.get_action_for_key(k)
        if existing_action and existing_action != act:
            # Swap or reassign
            settings.set_keybind(existing_action, "")
            if existing_action in self.key_buttons:
                self.key_buttons[existing_action].configure(text="[ NONE ]", bg="#1E293B", fg="#94A3B8")

        settings.set_keybind(act, k)
        self.key_buttons[act].configure(text=f"[ {k.upper()} ]", bg="#10B981", fg="#FFFFFF")
        self.master.after(400, lambda b=self.key_buttons[act], key=k: b.configure(text=f"[ {key.upper()} ]", bg="#1E293B", fg="#38BDF8"))

        self.listening_action = None
        self.lbl_status.configure(text=f"Rebound '{KEYBIND_LABELS[act][0]}' to [{k.upper()}]")
        if self.on_change_callback:
            self.on_change_callback("keybind", (act, k))

    def _reset_keybinds(self):
        settings.settings["keybinds"] = dict(settings.get("keybinds", {}))
        from settings_manager import DEFAULT_SETTINGS
        settings.settings["keybinds"] = dict(DEFAULT_SETTINGS["keybinds"])
        settings.save()
        self._populate_keybind_rows()
        self.lbl_status.configure(text="Keybinds reset to default factory profile.")
        if self.on_change_callback:
            self.on_change_callback("keybind_reset", None)

    def _on_close(self):
        self.grab_release()
        self.destroy()
