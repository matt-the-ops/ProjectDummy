"""Launcher: splash first, background model loading, press-to-start gate."""

import os
import threading
import time
import tkinter as tk


BG = "#D2E8FE"
FG = "#0F172A"
ACC = "#1D68FE"
MUTED = "#5A718F"
PERF = os.environ.get("PERF") == "1"


class Boot:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()

        sp = self.sp = tk.Toplevel(self.root)
        sp.title("Project B.M.O")
        sp.configure(bg=BG)
        sp.resizable(False, False)
        w, h = 520, 300
        sp.geometry(
            f"{w}x{h}+"
            f"{(sp.winfo_screenwidth() - w) // 2}+"
            f"{(sp.winfo_screenheight() - h) // 2}"
        )
        sp.protocol("WM_DELETE_WINDOW", self._quit)
        sp.bind("<Escape>", lambda _event: self._quit())

        tk.Label(
            sp, text="Project B.M.O", font=("Segoe UI", 26, "bold"),
            fg=FG, bg=BG,
        ).pack(pady=(32, 0))
        tk.Label(
            sp, text="Filipino Sign Language Recognition and Translation",
            font=("Segoe UI", 11), fg=MUTED, bg=BG,
        ).pack()

        self.cv = tk.Canvas(sp, width=70, height=70, bg=BG, highlightthickness=0)
        self.cv.pack(pady=16)
        self.arc = self.cv.create_arc(
            8, 8, 62, 62, start=0, extent=100,
            style="arc", outline=ACC, width=6,
        )
        self.msg = tk.Label(
            sp, text="Starting...", font=("Segoe UI", 12, "bold"),
            fg=FG, bg=BG, wraplength=480,
        )
        self.msg.pack()

        self.st = {"msg": "Loading models...", "err": None, "done": False}
        self.phase = "load"
        self.angle = 0
        self.t0 = time.time()
        self.mw = self.app = self.partner = None

        threading.Thread(target=self._load, daemon=True).start()
        self._spin()
        self._poll()

    def _load(self):
        try:
            import bmo.ui.main_window as mw
            self._t("imports")
            self.st["msg"] = "Rendering interface..."
            self.st["shell"] = mw.ShieldFSLApp.prebuild()
            self._t("main shell")
            self.st["msg"] = "Preparing partner window..."
            self.st["pre"] = mw.NonFSLWindow.prebuild()
            self._t("partner shell")
            self.st["mw"] = mw
            self.st["done"] = True
        except BaseException as exc:
            self.st["err"] = f"{type(exc).__name__}: {exc}. See the console."

    def _t(self, label):
        if PERF:
            print(f"[BOOT] {label}: {time.time() - self.t0:.1f}s")

    def _spin(self):
        self.angle = (self.angle + 12) % 360
        self.cv.itemconfig(self.arc, start=self.angle)
        self.sp.after(30, self._spin)

    def _poll(self):
        st = self.st
        if st["err"]:
            self.msg.config(text="Load failed: " + st["err"], fg="#EF4444")
            return

        if self.phase == "load":
            if st["done"]:
                self.mw = st["mw"]
                self.phase = "app"
                self.msg.config(text="Building main window...")
            else:
                self.msg.config(text=st["msg"])
        elif self.phase == "app":
            self.app = self.mw.ShieldFSLApp(
                self.root, shell=st["shell"], autostart=False,
            )
            self.msg.config(text="Building partner window...")
            self.phase = "partner"
        elif self.phase == "partner":
            self.partner = self.mw.NonFSLWindow(
                master=self.root,
                get_cam_fn=self.app.get_cam_frame,
                get_raw_frame_fn=self.mw.cam_capture.read_latest,
                speak_fn=self.mw.speak_text,
                ping_fn=self.mw.play_ping_sound,
                pre=st["pre"],
            )
            self.app.partner_window = self.partner
            try:
                self.partner._render(force=True)
            except Exception:
                pass
            self.root.protocol("WM_DELETE_WINDOW", self._close_all)
            self.partner.on_close_callback = self._close_all
            self.msg.config(text="Warming up tracking...")
            self.phase = "warm"
            self.t_warm = time.time()
            self._t("windows built")
        elif self.phase == "warm":
            if self.app.ml_ready.is_set():
                self.phase = "ready"
                self.msg.config(text="Press any key or click to start")
                self.sp.bind("<Key>", self._start)
                self.sp.bind("<Button-1>", self._start)
                self.sp.focus_force()
                self._t("ready")
            elif time.time() - self.t_warm > 40:
                self.msg.config(
                    text="Tracking failed to start. See the console.",
                    fg="#EF4444",
                )
                return

        self.sp.after(50, self._poll)

    def _start(self, _event=None):
        if self.phase != "ready":
            return
        self.phase = "run"
        self.sp.destroy()
        self.root.state("zoomed")
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.app.start()
        self.t_cal = time.time()
        self._watch()

    def _watch(self):
        locked = (
            not self.mw.is_calibrating
            and self.mw.target_person_lock is not None
        )
        if locked or time.time() - self.t_cal > 60:
            window = self.partner.window
            window.deiconify()
            window.lift()
            self.root.focus_force()
            return
        self.root.after(250, self._watch)

    def _close_all(self):
        try:
            self.partner.window.destroy()
        except Exception:
            pass
        self.app.destroy()
        self.root.destroy()

    def _quit(self):
        if self.app:
            self.app.destroy()
        self.root.destroy()


def main():
    Boot().root.mainloop()
