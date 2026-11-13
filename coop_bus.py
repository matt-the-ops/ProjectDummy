import os
import json
import pickle
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


def get_trained_signs(models_dir=MODELS_DIR):
    """
    Returns a set of lowercase trained sign names from word_classes.npy and fsl_model.pkl.
    """
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


def load_icon_palette(icon_map_filename="icon_map.json", base_dir=BASE_DIR):
    """
    Enforces the Hard Rule:
    The icon palette is built ONLY from signs the system already knows (word_classes.npy + letters).
    A sign appears only if it is trained AND has an icon in icon_map.json.
    Otherwise it is silently skipped.
    """
    map_path = os.path.join(base_dir, icon_map_filename)
    if not os.path.exists(map_path):
        return []

    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            icon_map = json.load(f)
    except Exception as e:
        print(f"[CoOpBus] Error reading {map_path}: {e}")
        return []

    trained_signs = get_trained_signs(os.path.join(base_dir, 'models'))
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


def get_icon_path_for_sign(sign_name, icon_map_filename="icon_map.json", base_dir=BASE_DIR):
    """Resolves icon path for a sign name if it exists in icon_map.json and is on disk."""
    map_path = os.path.join(base_dir, icon_map_filename)
    if not os.path.exists(map_path):
        return None
    try:
        with open(map_path, 'r', encoding='utf-8') as f:
            icon_map = json.load(f)
        s_clean = str(sign_name).strip().lower()
        for k, rel in icon_map.items():
            if str(k).strip().lower() == s_clean:
                p = os.path.normpath(os.path.join(base_dir, rel.replace('/', os.sep)))
                if os.path.isfile(p):
                    return p
    except Exception:
        pass
    return None


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
