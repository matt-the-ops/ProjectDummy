import os
import json
import pickle
import threading
import time
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


# ─── mtime/size-based caches ─────────────────────────────────────────────────
# Single-writer/multi-reader under this lock; invalidation is automatic when
# artifact (mtime_ns, size) signatures change or files appear/disappear.
_cache_lock = threading.Lock()
_trained_cache = {}      # models_dir -> {"sig": (classes_sig, letter_sig), "signs": set}
_icon_map_cache = {}     # map_path -> {"sig": (mtime_ns, size), "data": dict}
_palette_cache = {}      # (map_path, trained_sig, icons_sig) -> list (copy-safe)
_path_cache = {}         # (map_path, map_sig, icon_files_sig) -> {name: full_path}


def _file_sig(path):
    """(mtime_ns, size) tuple, or None when the file is missing."""
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _normalize_icon_map(icon_map):
    """{clean_name -> (original_name, rel_path)} — first entry wins, matching
    the original linear scan's first-match semantics."""
    out = {}
    for raw_name, rel_path in icon_map.items():
        name_clean = str(raw_name).strip().lower()
        if name_clean not in out:
            out[name_clean] = (str(raw_name), rel_path)
    return out


def _compute_trained_signs(models_dir):
    """Uncached trained-sign universe (original get_trained_signs logic)."""
    trained = set()

    # 1. Word classes (TensorFlow / LSTM)
    classes_path = os.path.join(models_dir, 'word_classes.npy')
    if os.path.exists(classes_path):
        try:
            words = np.load(classes_path, allow_pickle=True)
            for w in words:
                trained.add(str(w).strip().lower())
        except Exception as e:
            print(f"[CoOpBus] Could not load word_classes.npy: {e}")

    # 2. Letter classes (RandomForest)
    letter_path = os.path.join(models_dir, 'fsl_model.pkl')
    if os.path.exists(letter_path):
        try:
            with open(letter_path, 'rb') as f:
                letter_data = SafeUnpickler(f).load()
                model = letter_data['model'] if isinstance(letter_data, dict) else letter_data
                classes = getattr(model, 'classes_', None)
                if classes is not None:
                    for c in classes:
                        trained.add(str(c).strip().lower())
        except Exception as e:
            print(f"[CoOpBus] Could not load fsl_model.pkl: {e}")

    return trained


def get_trained_signs(models_dir=MODELS_DIR):
    """
    Returns a set of lowercase trained sign names from word_classes.npy and fsl_model.pkl.
    Cached per models_dir; reloads only when artifact signatures change.
    """
    classes_sig = _file_sig(os.path.join(models_dir, 'word_classes.npy'))
    letter_sig = _file_sig(os.path.join(models_dir, 'fsl_model.pkl'))
    sig = (classes_sig, letter_sig)

    with _cache_lock:
        entry = _trained_cache.get(models_dir)
        if entry is not None and entry["sig"] == sig:
            return set(entry["signs"])

    trained = _compute_trained_signs(models_dir)

    with _cache_lock:
        _trained_cache[models_dir] = {"sig": sig, "signs": trained}
    return set(trained)


def _load_icon_map_cached(map_path):
    """Parsed icon_map.json, cached by (mtime_ns, size). Returns None if missing."""
    sig = _file_sig(map_path)
    if sig is None:
        return None

    with _cache_lock:
        entry = _icon_map_cache.get(map_path)
        if entry is not None and entry["sig"] == sig:
            return entry["data"]

    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[CoOpBus] Error reading {map_path}: {e}")
        return None

    with _cache_lock:
        _icon_map_cache[map_path] = {"sig": sig, "data": data}
    return data


def _palette_for(icon_map, base_dir, trained_signs):
    """Uncached palette build (original Hard Rule logic)."""
    valid_palette = []

    for raw_name, rel_path in icon_map.items():
        name_clean = str(raw_name).strip().lower()
        full_path = os.path.normpath(os.path.join(base_dir, rel_path.replace('/', os.sep)))

        # HARD RULE: Must be trained AND in icon_map.json AND existing icon file
        if (name_clean in trained_signs) and os.path.isfile(full_path):
            display_name = str(raw_name).upper() if len(str(raw_name)) <= 2 else str(raw_name).capitalize()
            valid_palette.append({
                "key": name_clean,
                "display": display_name,
                "icon_path": full_path,
            })
        else:
            # Silently skipped
            pass

    return valid_palette


def load_icon_palette(icon_map_filename="icon_map.json", base_dir=BASE_DIR):
    """
    Enforces the Hard Rule:
    The icon palette is built ONLY from signs the system already knows (word_classes.npy + letters).
    A sign appears only if it is trained AND has an icon in icon_map.json.
    Otherwise it is silently skipped.
    """
    map_path = os.path.join(base_dir, icon_map_filename)
    icon_map = _load_icon_map_cached(map_path)
    if icon_map is None:
        return []

    models_dir = os.path.join(base_dir, 'models')
    trained_signs = get_trained_signs(models_dir)

    # Signature covers trained-sign + icon-file state so a newly added icon or
    # retrained model invalidates the palette automatically.
    trained_sig = _trained_cache.get(models_dir, {}).get("sig")
    icons_sig = tuple(sorted(
        (rel, _file_sig(os.path.normpath(os.path.join(base_dir, rel.replace('/', os.sep)))))
        for rel in icon_map.values()
    ))
    cache_key = (map_path, trained_sig, icons_sig)

    with _cache_lock:
        cached = _palette_cache.get(cache_key)
    if cached is not None:
        return [dict(entry) for entry in cached]

    valid_palette = _palette_for(icon_map, base_dir, trained_signs)

    with _cache_lock:
        _palette_cache[cache_key] = valid_palette
        if len(_palette_cache) > 8:
            _palette_cache.pop(next(iter(_palette_cache)))
    return [dict(entry) for entry in valid_palette]


def _paths_for_map(map_path, icon_map):
    """{clean_name: full_path} for entries whose icon file exists on disk.
    Shared by the initial build and the cached lookup (signature-checked)."""
    base_dir = os.path.dirname(map_path)
    paths = {}
    normalized = _normalize_icon_map(icon_map)
    for name_clean, (_raw_name, rel) in normalized.items():
        full_path = os.path.normpath(os.path.join(base_dir, rel.replace('/', os.sep)))
        if os.path.isfile(full_path):
            paths[name_clean] = full_path
    return paths


def get_icon_path_for_sign(sign_name, icon_map_filename="icon_map.json", base_dir=BASE_DIR):
    """Resolves icon path for a sign name if it exists in icon_map.json and is on disk."""
    map_path = os.path.join(base_dir, icon_map_filename)
    icon_map = _load_icon_map_cached(map_path)
    if icon_map is None:
        return None

    # Signature includes icon file mtimes so appearing/disappearing/updated
    # icons invalidate the cached path dict automatically.
    paths_sig = tuple(sorted(
        (rel, _file_sig(os.path.normpath(os.path.join(base_dir, rel.replace('/', os.sep)))))
        for rel in icon_map.values()
    ))
    map_sig = _file_sig(map_path)
    cache_key = (map_path, map_sig, paths_sig)

    with _cache_lock:
        paths = _path_cache.get(cache_key)
        if paths is None:
            paths = _paths_for_map(map_path, icon_map)
            _path_cache[cache_key] = paths
            if len(_path_cache) > 8:
                _path_cache.pop(next(iter(_path_cache)))

    return paths.get(str(sign_name).strip().lower())


def refresh_coop_caches():
    """Invalidate all coop caches (trained signs, icon map, palette, paths).
    Next helper call reloads from disk. Safe to call at any time."""
    with _cache_lock:
        _trained_cache.clear()
        _icon_map_cache.clear()
        _palette_cache.clear()
        _path_cache.clear()


class CoOpBus:
    """
    Bi-directional message bus between FSL Window and Non-FSL Window in the same process.
    """
    def __init__(self):
        self._fsl_listeners = []       # Callbacks when partner sends to FSL: cb(tokens, text, icon_paths)
        self._partner_listeners = []   # Callbacks when FSL sends to partner: cb(text)
        self._chat_history = []        # Shared conversation history
        self.last_fsl_message = ""
        self.last_partner_message = None

    def register_fsl_listener(self, callback):
        """FSL Window registers to receive incoming partner messages."""
        if callback not in self._fsl_listeners:
            self._fsl_listeners.append(callback)

    def register_partner_listener(self, callback):
        """Non-FSL Partner Window registers to receive incoming signer messages."""
        if callback not in self._partner_listeners:
            self._partner_listeners.append(callback)

    def send_from_fsl(self, text: str):
        """Signer delivers a recognized/translated sentence to the Non-FSL partner."""
        clean_text = text.strip()
        if not clean_text:
            return
        
        self.last_fsl_message = clean_text
        msg_entry = {
            "sender": "signer",
            "text": clean_text,
            "timestamp": time.time()
        }
        self._chat_history.append(msg_entry)

        # Notify non-FSL window
        for cb in list(self._partner_listeners):
            try:
                cb(clean_text)
            except Exception as e:
                print(f"[CoOpBus] Error in partner listener: {e}")

    def send_from_partner(self, tokens: list, sentence_text: str):
        """Partner delivers an icon-composed sentence to the FSL signer."""
        clean_sentence = sentence_text.strip()
        if not tokens and not clean_sentence:
            return

        # Resolve icon paths for previews
        icon_paths = []
        for t in tokens:
            p = get_icon_path_for_sign(t)
            icon_paths.append((t, p))

        msg_entry = {
            "sender": "partner",
            "tokens": list(tokens),
            "text": clean_sentence,
            "icon_paths": icon_paths,
            "timestamp": time.time()
        }
        self.last_partner_message = msg_entry
        self._chat_history.append(msg_entry)

        # Notify FSL window
        for cb in list(self._fsl_listeners):
            try:
                cb(tokens, clean_sentence, icon_paths)
            except Exception as e:
                print(f"[CoOpBus] Error in FSL listener: {e}")

    def get_history(self):
        return list(self._chat_history)


# Global singleton instance for the process
coop_hub = CoOpBus()
