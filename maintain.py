import cv2, csv, traceback
import os, json, shutil, threading, math
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from PIL import Image, ImageTk
from bmo.settings import settings
from bmo.features.fsl_features import (
    resolve_and_smooth_hands,
    extract_hand_frame,
    get_raw_hand_anchors,
    build_motion_dynamics_and_trajectory,
    extract_face_frame,
)
import mediapipe as mp
mp_holistic, mp_draw = mp.solutions.holistic, mp.solutions.drawing_utils

import pandas as pd

from bmo.config import DATA_DIR, MODELS_DIR, ICON_MAP_FILE, ROOT_DIR, ICONS_DIR

CSV = os.path.join(DATA_DIR, "fsl_dataset.csv")
WORDS = os.path.join(DATA_DIR, "word_dataset")
ICONS = ICON_MAP_FILE


def load_signs():
    """Return label -> (type, sample count)."""
    signs = {}
    if os.path.isfile(CSV):
        df = pd.read_csv(CSV)
        if "label" in df.columns:
            for lbl, n in df["label"].astype(str).value_counts().items():
                lbl = lbl.lower()
                signs[lbl] = ("number" if lbl.isdigit() else "letter", int(n))

    if os.path.isdir(WORDS):
        for name in os.listdir(WORDS):
            path = os.path.join(WORDS, name)
            if os.path.isdir(path):
                n = sum(
                    filename.endswith("_hand.npy")
                    for filename in os.listdir(path)
                    if os.path.isfile(os.path.join(path, filename))
                )
                signs[name.lower()] = ("word", n)
    return signs


def load_icon_map():
    if os.path.isfile(ICON_MAP_FILE):
        try:
            with open(ICON_MAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def save_icon_map(d):
    os.makedirs(os.path.dirname(os.path.abspath(ICON_MAP_FILE)), exist_ok=True)
    tmp = ICON_MAP_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, ICON_MAP_FILE)


def set_icon(label, src):
    if not src.lower().endswith(".png"):
        raise ValueError("Only PNG files are supported")
    with Image.open(src) as img:
        img.verify()
    os.makedirs(ICONS_DIR, exist_ok=True)
    dest_name = f"{label.lower()}.png"
    dest = os.path.join(ICONS_DIR, dest_name)
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy2(src, dest)
    m = load_icon_map()
    for k in list(m):
        if k.lower() == label.lower():
            del m[k]
    m[label.lower()] = f"assets/fsl_icons/{dest_name}"
    save_icon_map(m)


def remove_icon(label):
    lbl = label.lower()
    m = load_icon_map()
    rel = None
    keys = [k for k in m if k.lower() == lbl]
    if keys:
        for k in keys:
            rel = m.pop(k)
        save_icon_map(m)
    for p in (os.path.join(ICONS_DIR, f"{lbl}.png"), os.path.join(ROOT_DIR, rel) if rel else None):
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


def load_icons():
    return {key.lower() for key in load_icon_map()}


ORDER = {"letter": 0, "number": 1, "word": 2}


def delete_sign(label, kind):
    if kind == "word":
        found = False
        for name in os.listdir(WORDS) if os.path.isdir(WORDS) else []:
            path = os.path.realpath(os.path.join(WORDS, name))
            if os.path.basename(path).lower() == label.lower():
                if os.path.dirname(path) != os.path.realpath(WORDS):
                    raise ValueError("bad path")
                shutil.rmtree(path)
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Word '{label}' not found")
    else:
        df = pd.read_csv(CSV)
        df = df[df["label"].astype(str).str.lower() != label.lower()]
        df.to_csv(CSV + ".tmp", index=False)
        os.replace(CSV + ".tmp", CSV)
    remove_icon(label)


def rename_sign(old, new, kind):
    if kind == "word":
        src = os.path.join(WORDS, old)
        dst = os.path.join(WORDS, new)
        if os.path.exists(dst):
            raise ValueError("Name already exists")
        os.rename(src, dst)
    else:
        df = pd.read_csv(CSV)
        m = df["label"].astype(str).str.lower() == old
        df.loc[m, "label"] = new.upper()
        df.to_csv(CSV + ".tmp", index=False)
        os.replace(CSV + ".tmp", CSV)
    try:
        im = load_icon_map()
        changed = False
        old_l = old.lower()
        new_l = new.lower()
        new_png = f"{new_l}.png"
        new_icon_path = os.path.join(ICONS_DIR, new_png)
        new_map_path = f"assets/fsl_icons/{new_png}"

        old_val = None
        for k in list(im):
            if k.lower() == old_l:
                old_val = im.pop(k)
                im[new_l] = new_map_path
                changed = True

        src_candidates = []
        if old_val:
            src_candidates.append(os.path.join(ROOT_DIR, old_val))
        src_candidates.append(os.path.join(ICONS_DIR, f"{old_l}.png"))

        for src in src_candidates:
            if os.path.isfile(src):
                os.makedirs(ICONS_DIR, exist_ok=True)
                if os.path.abspath(src) != os.path.abspath(new_icon_path):
                    if os.path.exists(new_icon_path):
                        os.remove(new_icon_path)
                    os.replace(src, new_icon_path)
                if new_l not in im:
                    im[new_l] = new_map_path
                    changed = True
                break

        if changed:
            save_icon_map(im)
    except (OSError, ValueError):
        pass


def valid_label(kind, s):
    if kind == "letter":
        return len(s) == 1 and s.isalpha(), "Letter: one A-Z character, e.g. A"
    if kind == "number":
        return s.isdigit(), "Number: digits only, e.g. 5"
    return s.replace("_", "").isalnum() and not s.isdigit() and len(s) > 1, \
        "Word: 2+ chars, letters/underscore, not only digits, e.g. hello"


class Recorder(tk.Toplevel):
    def __init__(self, master, on_done, kind=None, label=None):
        super().__init__(master)
        self.title("Record")
        self.on_done = on_done
        self.cap = self.holo = None
        self.pending, self.take, self.left = False, None, 0
        self.recorded = 0
        self.state = [None, None, 0, 0]
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda e: self.on_escape())
        f = self.setup = ttk.Frame(self, padding=16)
        f.pack()
        self.kind = tk.StringVar(value="letter")
        ttk.Label(f, text="1. What are you training?").pack(anchor="w")
        for k in ("letter", "number", "word"):
            ttk.Radiobutton(f, text=k.title() + "s", value=k, variable=self.kind).pack(anchor="w")
        ttk.Label(f, text="2. Sign name (Ex: A, 5, hello)").pack(anchor="w", pady=(10, 0))
        self.sign = ttk.Entry(f); self.sign.pack(fill=tk.X)
        ttk.Label(f, text="3. How many shots?").pack(anchor="w", pady=(10, 0))
        self.shots = tk.Spinbox(f, from_=1, to=200, width=6)
        self.shots.delete(0, tk.END); self.shots.insert(0, "30"); self.shots.pack(anchor="w")
        ttk.Button(f, text="Start", command=self.start).pack(pady=12)
        if kind:
            self.kind.set(kind); self.sign.insert(0, label)

    def start(self):
        self.label = self.sign.get().strip()
        k = self.kind.get()
        ok, hint = valid_label(self.kind.get(), self.label)
        if not ok:
            return messagebox.showerror("Wrong sign name", hint, parent=self)
        self.label = self.label.upper() if k != "word" else self.label.lower()
        try:
            self.left = max(1, int(self.shots.get()))
        except ValueError:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = None
            return messagebox.showerror("Camera", "Camera busy. Close the kiosk.", parent=self)
        self.holo = mp_holistic.Holistic(model_complexity=0, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.setup.destroy()
        self.view = tk.Label(self); self.view.pack()
        self.msg = ttk.Label(self, font=("Segoe UI", 12, "bold")); self.msg.pack(pady=6)
        self.bind("<Return>", self.shoot)
        self.focus_force()
        self.show(); self.loop()

    def finish(self):
        messagebox.showinfo("Done", f"Finished recording '{self.label}'.", parent=self)
        if self.label.lower() not in load_icons():
            if messagebox.askyesno("Icon", f"Assign an icon for '{self.label}'?", parent=self):
                src = filedialog.askopenfilename(
                    parent=self,
                    title=f"Select icon for '{self.label}'",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                )
                if src:
                    try:
                        set_icon(self.label, src)
                    except Exception as e:
                        messagebox.showerror("Error", str(e), parent=self)
        return self.close()

    def show(self):
        if self.left <= 0:
            return self.finish()
        what = "motion" if self.kind.get() == "word" else "pose"
        self.msg.config(text=f"'{self.label}' | {self.left} left | ENTER = record {what} | ESC = done")

    def on_escape(self):
        if getattr(self, "recorded", 0) > 0:
            self.finish()
        else:
            self.close()

    def shoot(self, _=None):
        if self.left <= 0:
            return
        if self.kind.get() == "word":
            self.take = {"hand": [], "anc": [], "face": []}
        else:
            self.pending = True

    def loop(self):
        if not self.cap or self.holo is None:
            return
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            w, h = int(settings.get("ml_w", 320)), int(settings.get("ml_h", 240))
            rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)
            res = self.holo.process(rgb)
            self.state = list(resolve_and_smooth_hands(res, *self.state, max_lost=4, alpha=0.7))
            for lm in (res.left_hand_landmarks, res.right_hand_landmarks):
                if lm:
                    mp_draw.draw_landmarks(rgb, lm, mp_holistic.HAND_CONNECTIONS)
            hand = res.right_hand_landmarks or res.left_hand_landmarks
            if hand:
                if self.pending:
                    self.save_static(hand)
                if self.take is not None:
                    self.add_frame(res)
            elif self.pending:
                self.pending = False
                self.msg.config(text="NO HAND DETECTED. Show your hand, then press ENTER.")
            elif self.take is not None:
                self.take = None
                self.msg.config(text="Hand lost. Take discarded. Press ENTER to retry.")
            if not self.cap:
                return
            img = Image.fromarray(rgb).resize((w * 2, h * 2))
            self.tkimg = ImageTk.PhotoImage(img)
            self.view.config(image=self.tkimg)
        self.after(15, self.loop)

    def save_static(self, hand):
        self.pending = False
        lm = hand.landmark
        s = math.dist((lm[0].x, lm[0].y, lm[0].z), (lm[9].x, lm[9].y, lm[9].z)) or 1.0
        row = [v for p in lm for v in ((p.x - lm[0].x) / s, (p.y - lm[0].y) / s, (p.z - lm[0].z) / s)]
        new = not os.path.isfile(CSV) or os.path.getsize(CSV) == 0
        with open(CSV, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["label"] + [f"{c}_{i}" for i in range(21) for c in "xyz"])
            wr.writerow([self.label] + row)
        self.recorded += 1
        self.left -= 1
        self.show()

    def add_frame(self, res):
        t = self.take
        t["hand"].append(extract_hand_frame(res))
        t["anc"].append(get_raw_hand_anchors(res))
        t["face"].append(extract_face_frame(res))
        self.msg.config(text=f"Recording {len(t['hand'])}/40...")
        if len(t["hand"]) < 40:
            return
        self.take = None
        d = os.path.join(WORDS, self.label); os.makedirs(d, exist_ok=True)
        idx = sum(n.endswith("_hand.npy") for n in os.listdir(d))
        shape = np.array(t["hand"], dtype=np.float32)
        full = np.concatenate([shape, build_motion_dynamics_and_trajectory(t["anc"])], axis=1)
        fseq = np.array(t["face"], dtype=np.float32)
        np.save(os.path.join(d, f"{idx}_hand.npy"), full)
        np.save(os.path.join(d, f"{idx}_face.npy"), fseq.mean(axis=0))
        np.save(os.path.join(d, f"{idx}_face_seq.npy"), fseq)
        self.recorded += 1
        self.left -= 1
        self.show()

    def close(self):
        cap, self.cap = self.cap, None
        if cap:
            cap.release()
        if self.holo:
            self.holo.close()
        self.on_done()
        self.destroy()


MANUAL = """\
BMO MAINTENANCE MANUAL

BEFORE YOU START
- Close the kiosk (main.py). The camera can only be used by one program.
- Back up first: copy the data and models folders.

THE TABLE
Every sign you taught the kiosk is one row: name, type, number of
samples, and whether it has an icon. Use the Show box to filter by
letter, number, or word. Click a column header to sort.

ADD A NEW SIGN (Record)
1. Click Record.
2. Pick what you are training: letter, number, or word.
3. Type the sign name (A, 5, hello). Wrong type = error.
4. Enter how many shots (30+ recommended).
5. Click Start. The camera opens.
6. Show your hand. Wait until the green dots appear on it.
7. Press ENTER to record one shot.
   - Letter/number: hold the pose, press ENTER.
   - Word: press ENTER, then do the full motion. It records 40 frames.
8. Repeat until shots left reaches 0. The window closes by itself.
If you see "NO HAND DETECTED", nothing was saved. Try again.

EDIT A SIGN
Select a row, click Edit samples.
- Yes = delete the old samples and record fresh ones.
- No  = keep the old ones and add more.

RENAME A SIGN
Select a row, click Rename, type the new name.

DELETE A SIGN
Select a row, click Delete. This removes all its samples.

ICONS
Select a row and click Set icon to assign a PNG image to that sign.
Click Remove icon to remove its icon. When recording a new sign that
has no icon, you will also be prompted to choose one when finished.
Icons are stored in assets/fsl_icons and mapped in icon_map.json.

KEYBINDS
Click Keybinds to customize keyboard shortcuts for kiosk actions.
Click any shortcut button and press a key to rebind it. If the key is
already in use, the two actions will swap keys. Modifier keys like
Shift, Ctrl, Alt, and Caps Lock cannot be bound. Restart the kiosk
to apply new keybinds.

TRAIN MODELS (important)
Recording only saves samples. The kiosk does not learn until you:
1. Click Train models. Wait for the Done box (words take a while).
2. Close and restart the kiosk (main.py).
Do this after every record, edit, rename, or delete.

TIPS
- Use the same distance, lighting, and angle as the real kiosk.
- Record about the same number of shots for every sign.
- Vary the position a little between shots.
- If a sign is often wrong, add more shots for it.
"""


class FlowBar(tk.Frame):
    """Lays children left to right, wraps to new rows when width runs out."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.bind("<Configure>", lambda e: self.reflow())

    def reflow(self):
        w = self.winfo_width()
        x = y = row_h = 0
        for c in self.winfo_children():
            cw, ch = c.winfo_reqwidth() + 6, c.winfo_reqheight() + 6
            if x + cw > w and x > 0:
                x, y, row_h = 0, y + row_h, 0
            c.place(x=x, y=y)
            x += cw
            row_h = max(row_h, ch)
        self.configure(height=y + row_h)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BMO Maintenance")
        self.geometry("900x460")
        self.minsize(420, 300)
        cols = ("sign", "type", "samples", "icon")
        self.tree = ttk.Treeview(
            self, columns=cols, show="headings", selectmode="browse"
        )
        for col in cols:
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=100, minwidth=60, stretch=True)
        bar = FlowBar(self)
        bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 10))

        self.filter = tk.StringVar(value="all")
        ttk.Label(bar, text="Show:")
        cb = ttk.Combobox(bar, textvariable=self.filter, width=9, state="readonly",
                          values=["all", "letter", "number", "word"])
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Button(bar, text="Delete", command=self.on_delete)
        ttk.Button(bar, text="Edit samples", command=self.on_edit)
        ttk.Button(bar, text="Rename", command=self.on_rename)
        ttk.Button(bar, text="Set icon", command=self.on_set_icon)
        ttk.Button(bar, text="Remove icon", command=self.on_remove_icon)
        ttk.Button(bar, text="Keybinds", command=self.on_keybinds)
        ttk.Button(bar, text="Refresh", command=self.refresh)
        ttk.Button(bar, text="Record", command=lambda: Recorder(self, self.refresh))
        self.train_btn = ttk.Button(bar, text="Train models", command=self.on_train)
        ttk.Button(bar, text="? Help", command=self.on_help)
        self.status = ttk.Label(bar, text="")

        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        bar.reflow()
        self.refresh()

    def refresh(self):
        self.tree.delete(*self.tree.get_children())
        icons = load_icons()
        want = self.filter.get()
        rows = [(l, k, n) for l, (k, n) in load_signs().items()
                if want == "all" or k == want]
        rows.sort(key=lambda r: (ORDER[r[1]], r[0]))
        for lbl, kind, n in rows:
            self.tree.insert("", tk.END, iid=lbl,
                             values=(lbl, kind, n, "yes" if lbl in icons else "missing"))
        self.status.config(text=f"{len(rows)} signs")

    def on_help(self):
        w = tk.Toplevel(self)
        w.title("Help")
        w.geometry("560x520")
        t = tk.Text(w, wrap="word", font=("Segoe UI", 10), padx=12, pady=10)
        sb = ttk.Scrollbar(w, command=t.yview)
        t.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        t.pack(fill=tk.BOTH, expand=True)
        t.insert("1.0", MANUAL)
        t.config(state="disabled")

    def on_delete(self):
        selected = self.tree.selection()
        if not selected:
            return
        label = selected[0]
        kind = self.tree.set(label, "type")
        if not messagebox.askyesno(
            "Delete", f"Delete '{label}' and all its samples?"
        ):
            return
        try:
            delete_sign(label, kind)
            self.refresh()
            self.status.config(text="Deleted. Retrain needed.")
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))

    def on_edit(self):
        sel = self.tree.selection()
        if not sel:
            return
        lbl, kind = sel[0], self.tree.set(sel[0], "type")
        if not messagebox.askyesno(
            "Edit", f"Delete old '{lbl}' samples and re-record?\n(No = add more samples)"
        ):
            Recorder(self, self.refresh, kind, lbl)
            return
        try:
            delete_sign(lbl, kind)
        except Exception as e:
            return messagebox.showerror("Failed", str(e))
        Recorder(self, self.refresh, kind, lbl)

    def on_rename(self):
        from tkinter import simpledialog
        sel = self.tree.selection()
        if not sel:
            return
        old, kind = sel[0], self.tree.set(sel[0], "type")
        new = simpledialog.askstring("Rename", f"New name for '{old}':", parent=self)
        if not new:
            return
        new = new.strip().lower() if kind == "word" else new.strip().upper()
        ok, hint = valid_label(kind, new)
        if not ok:
            return messagebox.showerror("Wrong name", hint)
        try:
            rename_sign(old, new.lower(), kind)
        except Exception as e:
            return messagebox.showerror("Failed", str(e))
        self.refresh()
        self.status.config(text="Renamed. Retrain + restart kiosk.")

    def on_set_icon(self):
        sel = self.tree.selection()
        if not sel:
            return
        lbl = sel[0]
        path = filedialog.askopenfilename(
            parent=self,
            title=f"Select icon for '{lbl}'",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            set_icon(lbl, path)
            self.refresh()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)

    def on_remove_icon(self):
        sel = self.tree.selection()
        if not sel:
            return
        lbl = sel[0]
        try:
            remove_icon(lbl)
            self.refresh()
        except Exception as e:
            messagebox.showerror("Error", str(e), parent=self)

    def on_keybinds(self):
        dlg = getattr(self, "settings_dialog", None)
        if dlg and dlg.winfo_exists():
            dlg.lift()
            dlg.focus_force()
            return
        from bmo.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        dlg._switch_tab("keys")
        self.settings_dialog = dlg

        def on_destroy(e):
            if e.widget == dlg:
                self.settings_dialog = None
                self.status.config(text="Keybinds saved. Restart kiosk.")

        dlg.bind("<Destroy>", on_destroy, add="+")
        orig_close = dlg._on_close

        def on_close():
            orig_close()
            self.settings_dialog = None
            self.status.config(text="Keybinds saved. Restart kiosk.")

        dlg._on_close = on_close

    def on_train(self):
        self.train_btn.state(["disabled"])
        self.status.config(text="Training...")

        def run():
            def say(t):
                self.after(0, lambda: self.status.config(text=t))
            try:
                say("Training letters...")
                from tools.train_letters import train_letter_model
                train_letter_model()
                say("Training words (slow)...")
                from tools.train_words import train_word_model
                train_word_model()
                msg = "Done. Restart kiosk."
            except Exception as e:
                traceback.print_exc()
                msg = f"Failed: {e}"
            self.after(0, lambda: messagebox.showinfo("Training", msg))
            self.after(0, lambda: (self.status.config(text=msg),
                                   self.train_btn.state(["!disabled"]),
                                   self.refresh()))

        threading.Thread(target=run, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()