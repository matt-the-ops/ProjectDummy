"""
SHIELD FSL - Application Settings and Keybindings Manager
Handles persistent configuration for graphics, performance presets, and customizable keybinds.
Optimized for ultra-low spec hardware and smooth operation across devices.
"""

import os
import json
import threading

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

def get_gpu_device_info():
    """Detects available hardware GPU via OpenCV OpenCL / DirectML."""
    try:
        import cv2
        if cv2.ocl.haveOpenCL():
            dev = cv2.ocl.Device.getDefault()
            if dev.available():
                return True, dev.name()
    except Exception:
        pass
    return False, "CPU Fallback"

DEFAULT_SETTINGS = {
    # ── Performance & Graphics ──
    "performance_mode": False,       # Master toggle for low-end PC optimization
    "performance_preset": "balanced", # "potato", "balanced", "quality"
    "use_gpu": True,                 # Hardware acceleration via NVIDIA RTX / OpenCL
    "face_mesh_mode": "calibrate_only", # "off", "calibrate_only", "full"
    "ml_w": 480,                     # ML input width (sharp hand & face capture)
    "ml_h": 360,                     # ML input height
    "target_fps": 60,                # Target GUI refresh rate (30, 60)
    "partner_fps": 30,               # Partner window refresh rate (15, 20, 30)
    "camera_index": 0,               # Active camera device index
    "screen_resolution": "auto",     # "720p", "900p", "1080p", "fullscreen", "auto"
    
    # ── Customizable Keybinds ──
    # Format: action_id -> tkinter keysym string (lowercase)
    "keybinds": {
        "letters_mode": "1",
        "words_mode": "2",
        "toggle_camera": "h",
        "calibrate": "k",
        "switch_camera": "n",
        "speak": "t",
        "send": "return",
        "clear": "c",
        "fullscreen": "f11",
        "perf_mode": "p",
        "settings": "s",
    }
}

KEYBIND_LABELS = {
    "letters_mode": ("Letters Mode", "Switch recognition core to static letter finger-spelling"),
    "words_mode": ("Words Mode", "Switch recognition core to dynamic whole-word signs"),
    "toggle_camera": ("Toggle Camera View", "Hide or reveal the live camera monitor"),
    "calibrate": ("Calibrate Target Person", "Lock tracking strictly to user in front of camera"),
    "switch_camera": ("Next Camera Device", "Cycle to next connected camera index"),
    "speak": ("Speak Translation", "Text-to-speech output of current sentence"),
    "send": ("Send to Partner", "Deliver sentence to Co-Op partner window"),
    "clear": ("Clear Output Buffer", "Erase all recognized text and reset buffers"),
    "fullscreen": ("Toggle Fullscreen", "Expand window to borderless fullscreen"),
    "perf_mode": ("Toggle Performance Mode", "Instant switch between low-end and normal mode"),
    "settings": ("Open Settings Menu", "Open graphic settings and keybinds dialog"),
}

PRESETS = {
    "potato": {
        "performance_mode": True,
        "performance_preset": "potato",
        "use_gpu": True,
        "face_mesh_mode": "off",
        "ml_w": 360,
        "ml_h": 270,
        "target_fps": 30,
        "partner_fps": 20,
    },
    "balanced": {
        "performance_mode": False,
        "performance_preset": "balanced",
        "use_gpu": True,
        "face_mesh_mode": "calibrate_only",
        "ml_w": 480,
        "ml_h": 360,
        "target_fps": 60,
        "partner_fps": 30,
    },
    "quality": {
        "performance_mode": False,
        "performance_preset": "quality",
        "use_gpu": True,
        "face_mesh_mode": "full",
        "ml_w": 640,
        "ml_h": 480,
        "target_fps": 60,
        "partner_fps": 30,
    }
}


class SettingsManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SettingsManager, cls).__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings["keybinds"] = dict(DEFAULT_SETTINGS["keybinds"])
        self.listeners = []
        self.load()

    def load(self):
        if os.path.exists(SETTINGS_FILE):
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Merge loaded data with defaults
                for k, v in data.items():
                    if k == "keybinds" and isinstance(v, dict):
                        self.settings["keybinds"].update(v)
                    else:
                        self.settings[k] = v
                print("[OK] Settings loaded from disk.")
            except Exception as e:
                print(f"[!] Warning: Failed to load settings ({e}), using defaults.")

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=4)
            print("[OK] Settings saved to disk.")
        except Exception as e:
            print(f"[!] Warning: Failed to save settings ({e}).")

    def get(self, key, default=None):
        return self.settings.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))

    def set(self, key, value, notify=True):
        self.settings[key] = value
        if notify:
            self.notify_listeners(key, value)
        self.save()

    def get_keybind(self, action):
        return self.settings.get("keybinds", {}).get(action, DEFAULT_SETTINGS["keybinds"].get(action, ""))

    def set_keybind(self, action, keysym):
        keysym = keysym.lower()
        if "keybinds" not in self.settings:
            self.settings["keybinds"] = dict(DEFAULT_SETTINGS["keybinds"])
        self.settings["keybinds"][action] = keysym
        self.save()
        self.notify_listeners("keybinds", self.settings["keybinds"])

    def get_action_for_key(self, keysym):
        """Map a received keysym string to an action name, if mapped."""
        k = keysym.lower()
        keybinds = self.settings.get("keybinds", {})
        for act, bound_k in keybinds.items():
            if bound_k == k:
                return act
        return None

    def toggle_performance_mode(self):
        """Toggle between performance (potato) mode and balanced mode."""
        curr = self.get("performance_mode", False)
        new_mode = not curr
        if new_mode:
            self.apply_preset("potato")
        else:
            self.apply_preset("balanced")
        return new_mode

    def apply_preset(self, preset_name):
        if preset_name in PRESETS:
            for k, v in PRESETS[preset_name].items():
                self.settings[k] = v
            self.save()
            self.notify_listeners("preset", preset_name)
            return True
        return False

    def reset_to_defaults(self):
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings["keybinds"] = dict(DEFAULT_SETTINGS["keybinds"])
        self.save()
        self.notify_listeners("reset", None)

    def add_listener(self, fn):
        """Register a callback for when settings change: fn(key, value)"""
        if fn not in self.listeners:
            self.listeners.append(fn)

    def remove_listener(self, fn):
        if fn in self.listeners:
            self.listeners.remove(fn)

    def notify_listeners(self, key, value):
        for fn in list(self.listeners):
            try:
                fn(key, value)
            except Exception as e:
                print(f"[!] Error in settings listener: {e}")


# Global singleton instance
settings = SettingsManager()
