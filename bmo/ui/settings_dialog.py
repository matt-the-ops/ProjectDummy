import tkinter as tk
from tkinter import ttk
from bmo.settings import settings, KEYBIND_LABELS, DEFAULT_SETTINGS


class SettingsDialog(tk.Toplevel):
    def __init__(self, master, on_change_callback=None):
        super().__init__(master)
        self.master = master
        self.on_change_callback = on_change_callback
        self.title("Keybinds")
        self.geometry("460x420")
        self.minsize(360, 320)
        self.transient(master)
        self.grab_set()
        self.listening_action = None

        try:
            mw = master.winfo_width()
            mh = master.winfo_height()
            mx = master.winfo_rootx()
            my = master.winfo_rooty()
            x = mx + max(0, (mw - 460) // 2)
            y = my + max(0, (mh - 420) // 2)
            self.geometry(f"460x420+{x}+{y}")
        except Exception:
            pass

        self._build_ui()
        self.bind("<KeyPress>", self._on_keypress)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text="Keybinds", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

        cols = ("action", "key")
        self.tree = ttk.Treeview(outer, columns=cols, show="headings", selectmode="browse", height=10)
        self.tree.heading("action", text="Action")
        self.tree.heading("key", text="Key")
        self.tree.column("action", width=280, stretch=True)
        self.tree.column("key", width=100, minwidth=60, stretch=False)
        sb = ttk.Scrollbar(outer, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.LEFT, fill=tk.Y)

        btn_frame = ttk.Frame(outer, padding=(8, 0, 0, 0))
        btn_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(btn_frame, text="Rebind", command=self._start_rebind).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_frame, text="Reset defaults", command=self._reset_keybinds).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(btn_frame, text="Close", command=self._on_close).pack(fill=tk.X)

        self.lbl_status = ttk.Label(self, text="Click a row then Rebind to change a key.")
        self.lbl_status.pack(side=tk.BOTTOM, anchor="w", padx=12, pady=6)

        self._populate_tree()

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        keybinds = settings.get("keybinds", {})
        for act_id, (label, _desc) in KEYBIND_LABELS.items():
            key = keybinds.get(act_id, "")
            self.tree.insert("", tk.END, iid=act_id, values=(label, key.upper() if key else "—"))

    def _switch_tab(self, tab):
        # Kept for compatibility with maintain.py which calls _switch_tab("keys")
        pass

    def _start_rebind(self):
        sel = self.tree.selection()
        if not sel:
            self.lbl_status.config(text="Select a row first.")
            return
        self.listening_action = sel[0]
        label = KEYBIND_LABELS[self.listening_action][0]
        self.lbl_status.config(text=f"Press a key for '{label}' (Esc = cancel)…")
        self.focus_force()

    def _on_keypress(self, event):
        if not self.listening_action:
            if event.keysym.lower() == "escape":
                self._on_close()
            return

        act = self.listening_action
        k = event.keysym.lower()

        if k in ("shift_l", "shift_r", "control_l", "control_r", "alt_l", "alt_r", "caps_lock"):
            self.lbl_status.config(text=f"Cannot bind modifier key [{k.upper()}].")
            return

        if k == "escape":
            self.listening_action = None
            self.lbl_status.config(text="Rebind cancelled.")
            return

        old_act_key = (settings.get_keybind(act) or "").lower()

        # Swap on conflict
        existing_action = settings.get_action_for_key(k)
        if existing_action and existing_action != act:
            settings.set_keybind(existing_action, old_act_key)

        settings.set_keybind(act, k)
        self.listening_action = None
        self._populate_tree()
        self.lbl_status.config(text=f"Saved. Restart kiosk to apply.")
        if self.on_change_callback:
            self.on_change_callback("keybind", (act, k))

    def _reset_keybinds(self):
        settings.settings["keybinds"] = dict(DEFAULT_SETTINGS["keybinds"])
        settings.save()
        self._populate_tree()
        self.lbl_status.config(text="Reset to defaults. Restart kiosk to apply.")
        if self.on_change_callback:
            self.on_change_callback("keybind_reset", None)

    def _on_close(self):
        self.grab_release()
        self.destroy()
