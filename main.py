import os
import cv2
import math
import pickle
import collections
import time
import numpy as np
import pyttsx3
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model

try:
    import mediapipe.solutions.holistic as mp_holistic
    import mediapipe.solutions.drawing_utils as mp_drawing
except ModuleNotFoundError:
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
#  TTS — BMO Voice (High-pitch robotic child-like)
# ─────────────────────────────────────────────
import asyncio
import io

_tts_lock = threading.Lock()
tts_is_speaking = False
_pygame_initialized = False

def _init_audio():
    global _pygame_initialized
    if not _pygame_initialized:
        try:
            import pygame
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
            _pygame_initialized = True
        except Exception:
            _pygame_initialized = False

def speak_text(text):
    global tts_is_speaking
    clean = text.strip()
    if not clean:
        return

    def _run():
        global tts_is_speaking
        with _tts_lock:
            tts_is_speaking = True
            spoken = False

            # 1. Edge-TTS Natural Child BMO Voice (en-US-AnaNeural, +20Hz pitch, upbeat rate)
            try:
                import edge_tts
                import pygame
                _init_audio()

                async def _synthesize():
                    # AnaNeural is a child voice suited for BMO
                    comm = edge_tts.Communicate(clean, voice="en-US-AnaNeural", pitch="+20Hz", rate="+12%")
                    mp3_data = bytearray()
                    async for chunk in comm.stream():
                        if chunk["type"] == "audio":
                            mp3_data.extend(chunk["data"])
                    return bytes(mp3_data)

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio_bytes = loop.run_until_complete(_synthesize())
                loop.close()

                if audio_bytes:
                    sound_io = io.BytesIO(audio_bytes)
                    pygame.mixer.music.load(sound_io)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.04)
                    spoken = True
            except Exception as e:
                spoken = False

            # 2. Windows SAPI XML high pitch fallback
            if not spoken:
                try:
                    import pythoncom
                    import win32com.client
                    pythoncom.CoInitialize()
                    try:
                        speaker = win32com.client.Dispatch("SAPI.SpVoice")
                        for v in speaker.GetVoices():
                            if "zira" in v.GetDescription().lower():
                                speaker.Voice = v
                                break
                        escaped = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                        xml = f'<pitch absmiddle="12"><rate absspeed="2">{escaped}</rate></pitch>'
                        speaker.Speak(xml, 8)
                        spoken = True
                    finally:
                        pythoncom.CoUninitialize()
                except Exception:
                    pass

            # 3. pyttsx3 fallback
            if not spoken:
                try:
                    engine = pyttsx3.init()
                    for v in engine.getProperty('voices'):
                        if 'zira' in v.name.lower() or 'female' in v.name.lower():
                            engine.setProperty('voice', v.id)
                            break
                    engine.setProperty('rate', 190)
                    engine.say(clean)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    print(f"TTS Error: {e}")

            tts_is_speaking = False

    threading.Thread(target=_run, daemon=True).start()


# ─────────────────────────────────────────────
#  Base Directories
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# ─────────────────────────────────────────────
#  Safe Unpickler
# ─────────────────────────────────────────────
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

# ─────────────────────────────────────────────
#  Load Mode 1: Letter Model
# ─────────────────────────────────────────────
letter_model_path = os.path.join(MODELS_DIR, 'fsl_model.pkl')
try:
    with open(letter_model_path, 'rb') as f:
        letter_data = SafeUnpickler(f).load()
        letter_model = letter_data['model'] if isinstance(letter_data, dict) else letter_data
    print("[OK] Letter model loaded.")
except FileNotFoundError:
    print(f"Error: '{letter_model_path}' not found.")
    exit()

# ─────────────────────────────────────────────
#  Load Mode 2: Dual-Gate Word Models
# ─────────────────────────────────────────────
motion_model = None
face_clf     = None
word_classes = None

motion_model_path = os.path.join(MODELS_DIR, 'word_motion_model.h5')
face_model_path   = os.path.join(MODELS_DIR, 'word_face_model.pkl')
classes_path      = os.path.join(MODELS_DIR, 'word_classes.npy')

if os.path.exists(motion_model_path) and \
   os.path.exists(face_model_path) and \
   os.path.exists(classes_path):
    try:
        motion_model = load_model(motion_model_path)
        with open(face_model_path, 'rb') as f:
            face_data = SafeUnpickler(f).load()
            face_clf  = face_data['model']
        word_classes = np.load(classes_path, allow_pickle=True)
        print(f"[OK] Dual-gate models loaded. Classes: {word_classes}")
        DUAL_GATE_MODE = True
    except Exception as e:
        print(f"[!] Dual-gate load error: {e}")
        DUAL_GATE_MODE = False
else:
    DUAL_GATE_MODE = False
    print("[!] Dual-gate models not found in models/.")
    legacy_word_model = os.path.join(MODELS_DIR, 'word_model.h5')
    if os.path.exists(legacy_word_model) and os.path.exists(classes_path):
        try:
            motion_model = load_model(legacy_word_model)
            word_classes = np.load(classes_path, allow_pickle=True)
            print(f"[OK] Legacy word model loaded. Classes: {word_classes}")
        except Exception as e:
            print(f"[!] Legacy word model error: {e}")
    else:
        print("[!] No word model found. Mode 2 unavailable.")

# ─────────────────────────────────────────────
#  Feature Extraction Helpers
# ─────────────────────────────────────────────
HAND_FEATURES = 132
FACE_FEATURES = 1404

def extract_hand_frame(results):
    out = []
    if results.left_hand_landmarks:
        w = results.left_hand_landmarks.landmark[0]
        for lm in results.left_hand_landmarks.landmark:
            out.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
    else:
        out.extend([0.0] * 63)
    if results.right_hand_landmarks:
        w = results.right_hand_landmarks.landmark[0]
        for lm in results.right_hand_landmarks.landmark:
            out.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
    else:
        out.extend([0.0] * 63)
    return out

def get_raw_wrist(results):
    if results.left_hand_landmarks:
        w = results.left_hand_landmarks.landmark[0]
        lw = np.array([w.x, w.y, w.z], dtype=np.float32)
    else:
        lw = np.zeros(3, dtype=np.float32)
    if results.right_hand_landmarks:
        w = results.right_hand_landmarks.landmark[0]
        rw = np.array([w.x, w.y, w.z], dtype=np.float32)
    else:
        rw = np.zeros(3, dtype=np.float32)
    return lw, rw

def build_wrist_trajectory(wrist_frames):
    arr = np.array([np.concatenate([lw, rw]) for lw, rw in wrist_frames], dtype=np.float32)
    for col in (slice(0, 3), slice(3, 6)):
        block = arr[:, col]
        valid = np.any(block != 0, axis=1)
        if valid.any():
            last = block[valid][0]
            for i in range(len(block)):
                if valid[i]:
                    last = block[i]
                else:
                    block[i] = last
            arr[:, col] = block
    return arr - arr[0]

def normalize_hand_sequence(hand_seq):
    seq = np.asarray(hand_seq, dtype=np.float32)
    if seq.size == 0:
        return seq
    out = seq.copy()
    spans = []
    for off in (0, 63):
        tip_idx = 12 * 3
        tip = out[0, off + tip_idx: off + tip_idx + 3]
        span = np.linalg.norm(tip)
        if span < 1e-6:
            span = 1.0
        spans.append(span)
        out[:, off:off + 63] = out[:, off:off + 63] / span
    if out.shape[1] >= 132:
        out[:, 126:129] = out[:, 126:129] / spans[0]
        out[:, 129:132] = out[:, 129:132] / spans[1]
    return out

def extract_face_frame(results):
    out = []
    if results.face_landmarks:
        nose = results.face_landmarks.landmark[1]
        for lm in results.face_landmarks.landmark:
            out.extend([lm.x - nose.x, lm.y - nose.y, lm.z - nose.z])
    else:
        out.extend([0.0] * 1404)
    return out

# ─────────────────────────────────────────────
#  Soft / Neumorphic UI Palette
# ─────────────────────────────────────────────
def _rgb(r, g, b):
    """Helper: write colors as RGB, store as BGR (OpenCV order)."""
    return (b, g, r)

BG_MAIN        = _rgb(237, 236, 247)   # soft lavender-white app background
CARD_BASE      = _rgb(246, 245, 252)   # raised card fill
CARD_PRESSED   = _rgb(228, 227, 240)   # inset/pressed card fill
TRACK_BG       = _rgb(222, 221, 236)   # progress/toggle track background
SHADOW_DARK    = _rgb(202, 200, 216)   # soft "cast" shadow tone
SHADOW_LIGHT   = _rgb(255, 255, 255)   # soft highlight tone

TEXT_DARK      = _rgb(48,  45,  60)    # primary text
TEXT_MUTED     = _rgb(150, 147, 168)   # secondary/hint text
TEXT_FAINT     = _rgb(190, 188, 208)   # placeholder-level text
TEXT_WHITE     = _rgb(255, 255, 255)

ACCENT_PURPLE  = _rgb(150, 90,  195)
ACCENT_ORANGE  = _rgb(250, 165, 70)
ACCENT_BLUE    = _rgb(70,  130, 240)
ACCENT_BLUE_L  = _rgb(125, 185, 250)
ACCENT_PINK    = _rgb(232, 95,  150)
ACCENT_GREEN   = _rgb(105, 200, 140)
ACCENT_RED     = _rgb(235, 95,  95)
ACCENT_AMBER   = _rgb(245, 175, 70)

# Mode-specific gradient accents
ACCENT_LETTER   = ACCENT_PURPLE
ACCENT_LETTER2  = ACCENT_ORANGE
ACCENT_WORD     = ACCENT_BLUE
ACCENT_WORD2    = ACCENT_BLUE_L
ACCENT_WARN     = ACCENT_AMBER

# Legacy aliases kept so the rest of the file needs no other changes
XBOX_GLOW        = ACCENT_LETTER
XBOX_GLOW_DIM    = TEXT_FAINT
XBOX_GLOW_BRIGHT = TEXT_WHITE
XBOX_WARN        = ACCENT_WARN
XBOX_BLADE_FILL  = CARD_BASE
XBOX_PANEL_EDGE  = SHADOW_DARK
TEXT_DIM         = TEXT_MUTED
PANEL_COLOR      = CARD_BASE

# ─────────────────────────────────────────────
#  Drawing Helpers
# ─────────────────────────────────────────────
def draw_panel(img, x, y, w, h, color=CARD_BASE, alpha=0.7):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    r = max(0, min(r, w // 2, h // 2))
    if thickness == -1:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(img, (cx,cy), r, color, -1)
    else:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
        for cx, cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
            cv2.circle(img, (cx,cy), r, color, thickness)

def put_text_shadow(img, text, org, font, scale, color, thickness=2):
    """Clean anti-aliased text (kept name for compatibility with all call sites)."""
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def draw_neu_raised(img, x, y, w, h, r, color=CARD_BASE, depth=5):
    """Soft neumorphic 'raised' card: dark cast shadow + light highlight, flat fill on top."""
    H, W = img.shape[:2]
    ov = img.copy()
    draw_rounded_rect(ov, min(x+depth, W-1), min(y+depth, H-1), w, h, r, SHADOW_DARK, -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    ov2 = img.copy()
    draw_rounded_rect(ov2, max(x-depth, 0), max(y-depth, 0), w, h, r, SHADOW_LIGHT, -1)
    cv2.addWeighted(ov2, 0.55, img, 0.45, 0, img)
    draw_rounded_rect(img, x, y, w, h, r, color, -1)

def draw_neu_inset(img, x, y, w, h, r, color=CARD_PRESSED):
    """Soft neumorphic 'pressed in' card, for text fields / tracks."""
    draw_rounded_rect(img, x, y, w, h, r, color, -1)
    cv2.line(img, (x+r, y+1), (x+w-r, y+1), SHADOW_DARK, 2, cv2.LINE_AA)
    cv2.line(img, (x+1, y+r), (x+1, y+h-r), SHADOW_DARK, 2, cv2.LINE_AA)
    cv2.line(img, (x+r, y+h-1), (x+w-r, y+h-1), SHADOW_LIGHT, 2, cv2.LINE_AA)
    cv2.line(img, (x+w-1, y+r), (x+w-1, y+h-r), SHADOW_LIGHT, 2, cv2.LINE_AA)

def draw_gradient_rounded_rect(img, x, y, w, h, r, color1, color2, vertical=False):
    """Fill a rounded rect with a linear gradient between two colors."""
    if w <= 0 or h <= 0:
        return
    H, W = img.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    x, y = max(x, 0), max(y, 0)
    w, h = x2 - x, y2 - y
    if w <= 0 or h <= 0:
        return
    grad = np.zeros((h, w, 3), dtype=np.uint8)
    if vertical:
        for i in range(h):
            t = i / max(h - 1, 1)
            grad[i, :] = [int(color1[c]*(1-t) + color2[c]*t) for c in range(3)]
    else:
        for i in range(w):
            t = i / max(w - 1, 1)
            grad[:, i] = [int(color1[c]*(1-t) + color2[c]*t) for c in range(3)]
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_rounded_rect(mask, 0, 0, w, h, r, 255, -1)
    mask3 = cv2.merge([mask, mask, mask])
    roi = img[y:y+h, x:x+w]
    np.copyto(roi, grad, where=(mask3 == 255))

def draw_progress_bar(img, x, y, w, h, value, max_val, bar_color, bg=None):
    """Rounded pill progress bar (kept signature for existing call sites)."""
    if bg is None or bg in ((10,30,10), (50,50,50)):
        bg = TRACK_BG
    r = h // 2
    draw_rounded_rect(img, x, y, w, h, r, bg, -1)
    filled = int(w * min(value / max(max_val, 1e-6), 1.0))
    if filled > 2:
        draw_rounded_rect(img, x, y, max(filled, h), h, r, bar_color, -1)

def draw_ring_progress(img, cx, cy, radius, thickness, value, max_val, color, bg=TRACK_BG):
    """Circular progress ring."""
    cv2.circle(img, (cx, cy), radius, bg, thickness, cv2.LINE_AA)
    frac = max(0.0, min(value / max(max_val, 1e-6), 1.0))
    if frac > 0.003:
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, 360 * frac, color, thickness, cv2.LINE_AA)

def draw_toggle_pill(img, x, y, w, h, on, accent1, accent2, label=None, font=None):
    """iOS-style toggle pill, filled with a gradient when on."""
    r = h // 2
    if on:
        draw_gradient_rounded_rect(img, x, y, w, h, r, accent1, accent2)
    else:
        draw_rounded_rect(img, x, y, w, h, r, TRACK_BG, -1)
    knob_r = h // 2 - 3
    cx = (x + w - h // 2) if on else (x + h // 2)
    cy = y + h // 2
    cv2.circle(img, (cx, cy), knob_r, (255, 255, 255), -1)
    cv2.circle(img, (cx, cy), knob_r, SHADOW_DARK, 1, cv2.LINE_AA)
    if label and font:
        (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
        tx = x + 10 if on else x + w - tw - h // 2 - 6
        cv2.putText(img, label, (tx, y + (h+th)//2), font, 0.4,
                    TEXT_WHITE if on else TEXT_MUTED, 1, cv2.LINE_AA)

def draw_segmented_pill(img, x, y, w, h, n_slots, active_idx, labels, font,
                         accent1, accent2, track=CARD_PRESSED, pad=4):
    """Sliding-pill segmented control (mode selector)."""
    r = h // 2
    draw_neu_inset(img, x, y, w, h, r, track)
    slot_w = w // n_slots
    ax = x + active_idx * slot_w
    draw_gradient_rounded_rect(img, ax + pad, y + pad, slot_w - 2*pad, h - 2*pad,
                                (h - 2*pad)//2, accent1, accent2)
    for i, label in enumerate(labels):
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        tx = x + i*slot_w + (slot_w - tw)//2
        ty = y + (h + th)//2
        col = TEXT_WHITE if i == active_idx else TEXT_MUTED
        cv2.putText(img, label, (tx, ty), font, 0.55, col, 1, cv2.LINE_AA)

def draw_status_chip(img, x, y, w, h, text, color, font):
    r = h // 2
    draw_rounded_rect(img, x, y, w, h, r, _rgb(255,255,255), -1)
    draw_rounded_rect(img, x, y, w, h, r, color, 2)
    (tw, _), _ = cv2.getTextSize(text, font, 0.45, 1)
    cv2.putText(img, text, (x + (w-tw)//2, y + h - h//3), font, 0.45, color, 1, cv2.LINE_AA)

def draw_app_topbar(img, w, title="FSL Translator", accent=ACCENT_PURPLE):
    cv2.circle(img, (22, 20), 6, accent, -1)
    FONT_P = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, title, (38, 26), FONT_P, 0.62, TEXT_DARK, 1, cv2.LINE_AA)
    hint = "SHIELD FSL"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(img, hint, (w - hw - 16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_FAINT, 1, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  BMO Avatar — Glowing Closed/Open Face Images
# ─────────────────────────────────────────────
bmo_face_closed = None
bmo_face_open   = None

closed_path = os.path.join(ASSETS_DIR, 'bmo', 'bmo_closed.png')
open_path   = os.path.join(ASSETS_DIR, 'bmo', 'bmo_open.png')

if os.path.exists(closed_path):
    bmo_face_closed = cv2.imread(closed_path)
if os.path.exists(open_path):
    bmo_face_open = cv2.imread(open_path)

def draw_clean_rounded_border(img, x, y, w, h, r, color, thickness=2):
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)

def draw_bmo_avatar(img, x, y, w, h, is_talking, current_time=0.0, accent=ACCENT_PURPLE, accent2=ACCENT_ORANGE):
    """
    BMO avatar card, neumorphic style: raised card, soft color-wash glow behind
    the face art, rounded white frame around the face.
    - While idle: closed smile face
    - While talking: mouth stays open (static)
    """
    FONT   = cv2.FONT_HERSHEY_SIMPLEX
    FONT_M = cv2.FONT_HERSHEY_DUPLEX

    draw_neu_raised(img, x, y, w, h, 18, CARD_BASE, depth=4)

    face_h = h - 24
    face_w = int(face_h * 1.20)
    fx = x + 18
    fy = y + (h - face_h) // 2

    # Choose face image
    face_img = bmo_face_open if is_talking else bmo_face_closed

    # Soft color-wash glow behind the face (no hard blur needed at this size)
    glow_color = accent2 if is_talking else accent
    ov = img.copy()
    cv2.circle(ov, (fx + face_w // 2, fy + face_h // 2), int(face_h * 0.85), glow_color, -1)
    cv2.addWeighted(ov, 0.22 if is_talking else 0.14, img, 0.78 if is_talking else 0.86, 0, img)

    # White rounded frame behind the art
    draw_rounded_rect(img, fx - 6, fy - 6, face_w + 12, face_h + 12, 16, (255, 255, 255), -1)

    if face_img is not None:
        resized = cv2.resize(face_img, (face_w, face_h), interpolation=cv2.INTER_AREA)
        mask = np.zeros((face_h, face_w), dtype=np.uint8)
        r = 10
        cv2.rectangle(mask, (r, 0), (face_w - r, face_h), 255, -1)
        cv2.rectangle(mask, (0, r), (face_w, face_h - r), 255, -1)
        for cx, cy in [(r, r), (face_w - r, r), (r, face_h - r), (face_w - r, face_h - r)]:
            cv2.circle(mask, (cx, cy), r, 255, -1)
        roi = img[fy:fy+face_h, fx:fx+face_w]
        mask_3c = cv2.merge([mask, mask, mask])
        np.copyto(roi, resized, where=(mask_3c == 255))
    else:
        draw_rounded_rect(img, fx, fy, face_w, face_h, 10, _rgb(255, 235, 245), -1)

    # Status text beside face
    text_x = fx + face_w + 20
    text_y_base = fy + int(face_h * 0.34)

    if is_talking:
        dots = '.' * (int(current_time * 4) % 4)
        put_text_shadow(img, 'Speaking' + dots, (text_x, text_y_base), FONT_M, 0.55, accent2, 1)
        put_text_shadow(img, 'Voice synthesis', (text_x, text_y_base + 24), FONT, 0.38, TEXT_MUTED, 1)
    else:
        put_text_shadow(img, 'BMO', (text_x, text_y_base), FONT_M, 0.65, TEXT_DARK, 1)
        put_text_shadow(img, 'Online', (text_x, text_y_base + 22), FONT, 0.38, TEXT_MUTED, 1)
        put_text_shadow(img, 'Enter = Speak', (text_x, text_y_base + 44), FONT, 0.33, TEXT_FAINT, 1)



# ─────────────────────────────────────────────
#  State -- Mode 1 & 2
# ─────────────────────────────────────────────
LETTER_BUFFER_SIZE  = 12
STABILIZATION_DELAY = 2.5
GLOBAL_COOLDOWN     = 1.8

last_predicted_word = ""
last_word_time      = 0.0
WORD_DEBOUNCE_TIME  = 2.0

prediction_buffer   = collections.deque(maxlen=LETTER_BUFFER_SIZE)
pinky_history       = collections.deque(maxlen=30)
index_history       = collections.deque(maxlen=30)

last_stable_char    = ""
stable_start_time   = time.time()
global_cooldown_end = 0.0

motion_state        = ""
motion_cooldown_frames = 0
MOTION_GRACE_FRAMES = 8

suppress_char       = ""
suppression_end     = 0.0

# Word State (Dual-Gate)
WORD_SEQ_LEN        = 40
MOTION_CONF_THRESH  = 0.45
FACE_CONF_THRESH    = 0.35
MOTION_MARGIN_THRESH = 0.08
MIN_MOTION_ENERGY    = 0.005

hand_sequence_buffer  = collections.deque(maxlen=WORD_SEQ_LEN)
wrist_sequence_buffer = collections.deque(maxlen=WORD_SEQ_LEN)
face_sequence_buffer  = collections.deque(maxlen=WORD_SEQ_LEN)

debug_motion_word = "--"
debug_motion_conf = 0.0
debug_face_word   = "--"
debug_face_conf   = 0.0
gate_status       = "WAITING"

current_mode      = 1
typed_output      = ""
missing_frames    = 0
camera_index      = 0

notification_text = ""
notification_end  = 0.0
NOTIF_DURATION    = 2.0

def set_notification(msg):
    global notification_text, notification_end
    notification_text = msg
    notification_end  = time.time() + NOTIF_DURATION

# ─────────────────────────────────────────────
#  Camera Setup
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(camera_index)
print("\n[STARTED] FSL Dual-Gate Translator")
print("Controls: 1=Letters  2=Words  N=Cam  F=Fullscreen  Enter=Speak  Space  Bksp  C=Clear  Q=Quit\n")

# ─────────────────────────────────────────────
#  Window Setup (Dual-Window Layout)
# ─────────────────────────────────────────────
CAM_WIN   = "SHIELD CAM"
CTRL_WIN  = "SHIELD CONTROL"
CTRL_W, CTRL_H = 520, 800   # control panel canvas size

cv2.namedWindow(CAM_WIN,  cv2.WINDOW_NORMAL)
cv2.namedWindow(CTRL_WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(CTRL_WIN, CTRL_W, CTRL_H)
cv2.moveWindow(CAM_WIN,  0,      0)
cv2.moveWindow(CTRL_WIN, 660,    0)

fullscreen = False


with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1,
) as holistic:

    last_left_hand = None
    last_right_hand = None
    lost_frames_left = 0
    lost_frames_right = 0
    MAX_LOST_FRAMES = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            camera_index = 0
            cap = cv2.VideoCapture(camera_index)
            ret, frame = cap.read()
            if not ret:
                break

        frame        = cv2.flip(frame, 1)
        H, W, _      = frame.shape
        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results      = holistic.process(rgb_frame)
        
        # Smooth hand landmarks (prevent sudden stuttering/disappearing)
        if results.left_hand_landmarks:
            last_left_hand = results.left_hand_landmarks
            lost_frames_left = 0
        else:
            lost_frames_left += 1
            if lost_frames_left <= MAX_LOST_FRAMES:
                results.left_hand_landmarks = last_left_hand
            else:
                last_left_hand = None
                
        if results.right_hand_landmarks:
            last_right_hand = results.right_hand_landmarks
            lost_frames_right = 0
        else:
            lost_frames_right += 1
            if lost_frames_right <= MAX_LOST_FRAMES:
                results.right_hand_landmarks = last_right_hand
            else:
                last_right_hand = None

        # Resolve duplicate hands (when both left and right landmarks lock onto the SAME hand)
        if results.left_hand_landmarks and results.right_hand_landmarks:
            lw = results.left_hand_landmarks.landmark[0]
            rw = results.right_hand_landmarks.landmark[0]
            wrist_dist = math.sqrt((lw.x - rw.x)**2 + (lw.y - rw.y)**2 + (lw.z - rw.z)**2)
            if wrist_dist < 0.12:  # Wrists are virtually on top of each other
                # Discard the stale or synthetic cached hand, or drop the one that was lost
                if lost_frames_left > 0 and lost_frames_right == 0:
                    results.left_hand_landmarks = None
                    last_left_hand = None
                elif lost_frames_right > 0 and lost_frames_left == 0:
                    results.right_hand_landmarks = None
                    last_right_hand = None
                else:
                    # Both detected directly in this frame: check body side relative to shoulders if pose available
                    drop_left = False
                    if results.pose_landmarks:
                        l_sh = results.pose_landmarks.landmark[11]
                        r_sh = results.pose_landmarks.landmark[12]
                        hand_x = (lw.x + rw.x) / 2.0
                        # If hand is on right shoulder's side (closer to right), drop left_hand detection
                        dist_to_left_sh = abs(hand_x - l_sh.x)
                        dist_to_right_sh = abs(hand_x - r_sh.x)
                        drop_left = dist_to_right_sh < dist_to_left_sh
                    else:
                        drop_left = (lw.x > 0.5)

                    if drop_left:
                        results.left_hand_landmarks = None
                        last_left_hand = None
                    else:
                        results.right_hand_landmarks = None
                        last_right_hand = None

        current_time = time.time()
        on_cooldown  = current_time < global_cooldown_end

        primary_hand = None
        if results.right_hand_landmarks:
            primary_hand = results.right_hand_landmarks.landmark
        elif results.left_hand_landmarks:
            primary_hand = results.left_hand_landmarks.landmark

        current_frame_prediction = ""

        # ══════════════════════════════════════════════════════
        #  MODE 2: DUAL-GATE WORD RECOGNITION (Sliding Window)
        # ══════════════════════════════════════════════════════
        if current_mode == 2:
            if motion_model is not None:
                has_hand = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

                if not has_hand:
                    missing_frames += 1
                    # Stop buffering fast when hands disappear from the screen
                    if missing_frames >= 3:
                        hand_sequence_buffer.clear()
                        wrist_sequence_buffer.clear()
                        face_sequence_buffer.clear()
                        last_predicted_word = ""  # Reset duplicate lock once hands leave screen
                        debug_motion_word = "--"
                        debug_motion_conf = 0.0
                        debug_face_word   = "--"
                        debug_face_conf   = 0.0
                        gate_status       = "WAITING"
                else:
                    missing_frames = 0

                    if on_cooldown:
                        # Keep buffer empty during cooldown so retracting/resting hand isn't buffered
                        hand_sequence_buffer.clear()
                        wrist_sequence_buffer.clear()
                        face_sequence_buffer.clear()
                        gate_status = "COOLDOWN"
                    else:
                        hand_sequence_buffer.append(extract_hand_frame(results))
                        wrist_sequence_buffer.append(get_raw_wrist(results))
                        face_sequence_buffer.append(extract_face_frame(results))

                        buf_len = len(hand_sequence_buffer)

                        if buf_len == WORD_SEQ_LEN:
                            shape_arr     = np.array(hand_sequence_buffer, dtype=np.float32)
                            frame_diffs   = np.diff(shape_arr, axis=0)
                            motion_energy = float(np.mean(np.linalg.norm(frame_diffs, axis=1)))

                            if motion_energy < MIN_MOTION_ENERGY:
                                gate_status = "NO MOTION"
                                # Slide window forward by 8 frames instead of wiping clean
                                for _ in range(8):
                                    if hand_sequence_buffer: hand_sequence_buffer.popleft()
                                    if wrist_sequence_buffer: wrist_sequence_buffer.popleft()
                                    if face_sequence_buffer: face_sequence_buffer.popleft()
                            else:
                                traj      = build_wrist_trajectory(list(wrist_sequence_buffer))
                                full_hand = np.concatenate([shape_arr, traj], axis=1)
                                hand_norm = normalize_hand_sequence(full_hand)
                                hand_arr  = np.expand_dims(hand_norm, axis=0)

                                mot_preds  = motion_model.predict(hand_arr, verbose=0)[0]
                                sorted_idx = np.argsort(mot_preds)[::-1]
                                mot_idx    = int(sorted_idx[0])
                                mot_conf   = float(mot_preds[mot_idx])
                                mot_margin = (mot_conf - float(mot_preds[sorted_idx[1]])
                                              if len(sorted_idx) > 1 else mot_conf)

                                debug_motion_word = str(word_classes[mot_idx])
                                debug_motion_conf = mot_conf * 100.0

                                mot_pass = (mot_conf >= MOTION_CONF_THRESH and
                                            mot_margin >= MOTION_MARGIN_THRESH)

                                face_pass = False
                                face_idx = -1
                                face_conf = 0.0

                                if DUAL_GATE_MODE and face_clf is not None:
                                    face_avg  = np.mean(np.array(face_sequence_buffer, dtype=np.float32), axis=0)
                                    face_snap = face_avg.reshape(1, -1)

                                    try:
                                        face_proba = face_clf.predict_proba(face_snap)[0]
                                        face_idx   = int(np.argmax(face_proba))
                                        face_conf  = float(face_proba[face_idx])

                                        debug_face_word = str(word_classes[face_idx])
                                        debug_face_conf = face_conf * 100.0
                                        face_pass = (face_conf >= FACE_CONF_THRESH)
                                    except Exception:
                                        debug_face_word = "--"
                                        debug_face_conf = 0.0

                                # Decision Gate Logic
                                if DUAL_GATE_MODE and face_clf is not None:
                                    if mot_idx == face_idx and mot_pass and face_pass:
                                        gate_status = "AGREE"
                                        accept_pred = True
                                    elif mot_conf >= 0.65 and mot_margin >= MOTION_MARGIN_THRESH:
                                        gate_status = "MOTION-OK"
                                        accept_pred = True
                                    else:
                                        gate_status = "DISAGREE" if mot_idx != face_idx else "LOW CONF"
                                        accept_pred = False
                                else:
                                    if mot_pass:
                                        gate_status = "SINGLE-OK"
                                        accept_pred = True
                                    else:
                                        gate_status = "LOW CONF"
                                        accept_pred = False

                                if accept_pred:
                                    output_word = str(word_classes[mot_idx])
                                    # Prevent duplicate word double-triggering
                                    if output_word == last_predicted_word and (current_time - last_word_time < WORD_DEBOUNCE_TIME):
                                        gate_status = "DUPLICATE"
                                        for _ in range(8):
                                            if hand_sequence_buffer: hand_sequence_buffer.popleft()
                                            if wrist_sequence_buffer: wrist_sequence_buffer.popleft()
                                            if face_sequence_buffer: face_sequence_buffer.popleft()
                                    else:
                                        last_predicted_word = output_word
                                        last_word_time = current_time
                                        typed_output += output_word + " "
                                        global_cooldown_end = current_time + GLOBAL_COOLDOWN
                                        hand_sequence_buffer.clear()
                                        wrist_sequence_buffer.clear()
                                        face_sequence_buffer.clear()
                                        speak_text(output_word)
                                        set_notification("OK: " + output_word)
                                else:
                                    # Slide window smoothly instead of clearing
                                    for _ in range(8):
                                        if hand_sequence_buffer: hand_sequence_buffer.popleft()
                                        if wrist_sequence_buffer: wrist_sequence_buffer.popleft()
                                        if face_sequence_buffer: face_sequence_buffer.popleft()
            else:
                missing_frames += 1
                if missing_frames > 25:
                    hand_sequence_buffer.clear()
                    wrist_sequence_buffer.clear()
                    face_sequence_buffer.clear()
                    debug_motion_word = "--"
                    debug_motion_conf = 0.0
                    debug_face_word   = "--"
                    debug_face_conf   = 0.0
                    gate_status       = "WAITING"

        # ══════════════════════════════════════════════════════
        #  MODE 1: LETTERS + MOTION (J, Z)
        # ══════════════════════════════════════════════════════
        elif current_mode == 1:
            if primary_hand is not None:
                missing_frames = 0
                wrist_x, wrist_y, wrist_z = primary_hand[0].x, primary_hand[0].y, primary_hand[0].z
                scale = math.sqrt(
                    (primary_hand[9].x - wrist_x)**2 +
                    (primary_hand[9].y - wrist_y)**2 +
                    (primary_hand[9].z - wrist_z)**2
                ) or 1.0

                row = []
                for lm in primary_hand:
                    row.extend([(lm.x-wrist_x)/scale, (lm.y-wrist_y)/scale, (lm.z-wrist_z)/scale])

                raw_ml_char = letter_model.predict([row])[0]
                prediction_buffer.append(raw_ml_char)
                base_sign = collections.Counter(prediction_buffer).most_common(1)[0][0]

                pinky_pos = (int(primary_hand[20].x * W), int(primary_hand[20].y * H))
                index_pos = (int(primary_hand[8].x  * W), int(primary_hand[8].y  * H))

                is_pinky_extended = primary_hand[20].y < primary_hand[17].y
                is_index_extended = primary_hand[8].y  < primary_hand[5].y

                j_posture = (base_sign == 'I') and is_pinky_extended
                z_posture = (base_sign == 'D') and is_index_extended

                if j_posture:
                    if motion_state != "J_TRACKING":
                        motion_state = "J_TRACKING"
                        pinky_history.clear()
                        index_history.clear()
                    motion_cooldown_frames = 0
                elif z_posture:
                    if motion_state != "Z_TRACKING":
                        motion_state = "Z_TRACKING"
                        pinky_history.clear()
                        index_history.clear()
                    motion_cooldown_frames = 0
                else:
                    if motion_state != "":
                        motion_cooldown_frames += 1
                        if motion_cooldown_frames > MOTION_GRACE_FRAMES:
                            motion_state = ""
                            pinky_history.clear()
                            index_history.clear()

                MIN_MOVE_SQ = 30
                if motion_state == "J_TRACKING":
                    if (not pinky_history or
                            (pinky_pos[0]-pinky_history[-1][0])**2 +
                            (pinky_pos[1]-pinky_history[-1][1])**2 > MIN_MOVE_SQ):
                        pinky_history.append(pinky_pos)
                if motion_state == "Z_TRACKING":
                    if (not index_history or
                            (index_pos[0]-index_history[-1][0])**2 +
                            (index_pos[1]-index_history[-1][1])**2 > MIN_MOVE_SQ):
                        index_history.append(index_pos)

                motion_triggered, motion_char = False, ""
                if motion_state == "J_TRACKING" and len(pinky_history) >= 8:
                    highest_y = min(p[1] for p in pinky_history)
                    if pinky_history[-1][1] - highest_y > 45:
                        if not on_cooldown:
                            motion_char, motion_triggered = 'J', True
                            suppress_char  = 'I'
                            suppression_end = current_time + 1.2
                        motion_state = ""
                        pinky_history.clear()
                elif motion_state == "Z_TRACKING" and len(index_history) >= 8:
                    min_x = min(p[0] for p in index_history)
                    max_x = max(p[0] for p in index_history)
                    if max_x - min_x > 55:
                        if not on_cooldown:
                            motion_char, motion_triggered = 'Z', True
                            suppress_char  = 'D'
                            suppression_end = current_time + 1.2
                        motion_state = ""
                        index_history.clear()

                letter_triggered, letter_char = False, ""
                if not motion_triggered:
                    candidate = base_sign
                    if current_time < suppression_end and candidate == suppress_char:
                        candidate = ""
                    if candidate:
                        if candidate == last_stable_char:
                            held = current_time - stable_start_time
                            if held >= STABILIZATION_DELAY and not on_cooldown:
                                letter_char, letter_triggered = candidate, True
                        else:
                            last_stable_char  = candidate
                            stable_start_time = current_time
                    current_frame_prediction = last_stable_char

                if motion_triggered and motion_char and not on_cooldown:
                    typed_output        += motion_char
                    global_cooldown_end  = current_time + GLOBAL_COOLDOWN
                    stable_start_time    = current_time + GLOBAL_COOLDOWN
                    speak_text(motion_char)
                    set_notification("OK: " + motion_char + " (motion)")
                elif letter_triggered and letter_char and not on_cooldown:
                    typed_output        += letter_char
                    global_cooldown_end  = current_time + GLOBAL_COOLDOWN
                    stable_start_time    = current_time + GLOBAL_COOLDOWN
                    speak_text(letter_char)
                    set_notification("OK: " + letter_char)
            else:
                missing_frames += 1
                if missing_frames > 10:
                    pinky_history.clear()
                    index_history.clear()
                    prediction_buffer.clear()
                    last_stable_char = ""
                    motion_state     = ""

        # ══════════════════════════════════════════════════════
        #  CAMERA WINDOW — Landmark Drawing + Minimal HUD
        # ══════════════════════════════════════════════════════
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(20, 60, 20), thickness=1))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 255, 140), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(30, 200, 80),  thickness=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(100, 255, 100), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(60,  220, 60),  thickness=2))

        # Minimal cam HUD — soft floating pill bar (translucent dark, since it
        # sits over live video where the light neumorphic palette has no contrast)
        FONT   = cv2.FONT_HERSHEY_SIMPLEX
        FONT_M = cv2.FONT_HERSHEY_DUPLEX
        mode_accent  = ACCENT_LETTER  if current_mode == 1 else ACCENT_WORD
        mode_accent2 = ACCENT_LETTER2 if current_mode == 1 else ACCENT_WORD2

        cam_bar = frame.copy()
        cv2.rectangle(cam_bar, (0, 0), (W, 50), (25, 22, 20), -1)
        cv2.addWeighted(cam_bar, 0.55, frame, 0.45, 0, frame)

        pill_lbl = "LETTERS" if current_mode == 1 else "WORDS"
        (pw, ph), _ = cv2.getTextSize(pill_lbl, FONT_M, 0.55, 1)
        pill_w, pill_h = pw + 34, 30
        draw_gradient_rounded_rect(frame, 12, 10, pill_w, pill_h, pill_h//2, mode_accent, mode_accent2)
        cv2.putText(frame, pill_lbl, (12 + 17, 10 + pill_h - 10), FONT_M, 0.55, (255,255,255), 1, cv2.LINE_AA)

        cam_info = f"CAM {camera_index}  |  F = CTRL FULLSCREEN"
        (ci_w, _), _ = cv2.getTextSize(cam_info, FONT, 0.40, 1)
        cv2.putText(frame, cam_info, (W - ci_w - 12, 30), FONT, 0.40, (210, 210, 215), 1, cv2.LINE_AA)

        # Buffer / progress strip on cam bottom (rounded pill floating above the edge)
        strip_h = 6
        strip_margin = 14
        strip_w = W - 2 * strip_margin
        strip_y = H - strip_h - 10
        if current_mode == 2:
            buf_ratio = len(hand_sequence_buffer) / max(WORD_SEQ_LEN, 1)
            if on_cooldown:
                left = global_cooldown_end - current_time
                ratio = 1 - (left / GLOBAL_COOLDOWN)
                draw_progress_bar(frame, strip_margin, strip_y, strip_w, strip_h,
                                   ratio, 1.0, ACCENT_WARN, bg=(40,38,36))
            else:
                draw_progress_bar(frame, strip_margin, strip_y, strip_w, strip_h,
                                   buf_ratio, 1.0, mode_accent, bg=(40,38,36))
        elif current_mode == 1 and current_frame_prediction and not on_cooldown:
            held = current_time - stable_start_time
            draw_progress_bar(frame, strip_margin, strip_y, strip_w, strip_h,
                               held, STABILIZATION_DELAY, mode_accent, bg=(40,38,36))

        cv2.imshow(CAM_WIN, frame)

        # ══════════════════════════════════════════════════════
        #  CONTROL PANEL WINDOW — Soft / Neumorphic Dashboard
        # ══════════════════════════════════════════════════════
        mode_accent2 = ACCENT_LETTER2 if current_mode == 1 else ACCENT_WORD2

        ctrl = np.zeros((CTRL_H, CTRL_W, 3), dtype=np.uint8)
        ctrl[:] = BG_MAIN

        # ── Top bar ──
        draw_app_topbar(ctrl, CTRL_W, title="FSL Translator", accent=mode_accent)

        # ── Mode selector: sliding-pill segmented control ──
        seg_x, seg_y, seg_w, seg_h = 20, 44, CTRL_W - 40, 50
        draw_segmented_pill(ctrl, seg_x, seg_y, seg_w, seg_h, 2,
                            0 if current_mode == 1 else 1,
                            ["Letters", "Words"], FONT_M,
                            mode_accent, mode_accent2)

        # ── BMO Interactive Talking Avatar ──
        bmo_y = seg_y + seg_h + 14
        bmo_h = 106
        draw_bmo_avatar(ctrl, 12, bmo_y, CTRL_W - 24, bmo_h, tts_is_speaking,
                        current_time, accent=mode_accent, accent2=mode_accent2)

        # ── Current Prediction Card ──
        pred_y = bmo_y + bmo_h + 14
        pred_h = 150
        draw_neu_raised(ctrl, 12, pred_y, CTRL_W - 24, pred_h, 18, CARD_BASE, depth=4)
        put_text_shadow(ctrl, "DETECTING", (30, pred_y + 24), FONT, 0.42, TEXT_MUTED, 1)

        if current_mode == 1:
            sign_display = current_frame_prediction.upper() if current_frame_prediction else "-"

            # Ring behind the big letter shows hold / cooldown progress
            ring_cx, ring_cy, ring_r = 12 + (CTRL_W - 24) // 2, pred_y + 92, 58
            if current_frame_prediction and not on_cooldown and current_time >= suppression_end:
                held = current_time - stable_start_time
                draw_ring_progress(ctrl, ring_cx, ring_cy, ring_r, 8, held, STABILIZATION_DELAY, mode_accent)
                hint_txt, hint_col = "Hold to confirm", mode_accent
            elif on_cooldown:
                left = global_cooldown_end - current_time
                draw_ring_progress(ctrl, ring_cx, ring_cy, ring_r, 8,
                                    GLOBAL_COOLDOWN - left, GLOBAL_COOLDOWN, ACCENT_WARN)
                hint_txt, hint_col = "Cooldown", ACCENT_WARN
            else:
                draw_ring_progress(ctrl, ring_cx, ring_cy, ring_r, 8, 0, 1, mode_accent)
                hint_txt, hint_col = "", TEXT_MUTED

            (sw, sh), _ = cv2.getTextSize(sign_display, FONT_M, 2.4, 4)
            put_text_shadow(ctrl, sign_display, (ring_cx - sw//2, ring_cy + sh//2), FONT_M, 2.4, TEXT_DARK, 4)
            if hint_txt:
                (hw2, _), _ = cv2.getTextSize(hint_txt, FONT, 0.4, 1)
                put_text_shadow(ctrl, hint_txt, (12 + (CTRL_W-24-hw2)//2, pred_y + pred_h - 14), FONT, 0.4, hint_col, 1)

        elif current_mode == 2:
            # Buffer / cooldown ring
            ring_cx, ring_cy, ring_r = 100, pred_y + 92, 44
            if on_cooldown:
                left = global_cooldown_end - current_time
                draw_ring_progress(ctrl, ring_cx, ring_cy, ring_r, 7,
                                    GLOBAL_COOLDOWN - left, GLOBAL_COOLDOWN, ACCENT_WARN)
                ring_lbl, ring_col = "Cooldown", ACCENT_WARN
            else:
                buf_count = len(hand_sequence_buffer)
                draw_ring_progress(ctrl, ring_cx, ring_cy, ring_r, 7, buf_count, WORD_SEQ_LEN, mode_accent)
                ring_lbl, ring_col = f"{buf_count}/{WORD_SEQ_LEN}", mode_accent
            (rw, rh), _ = cv2.getTextSize(ring_lbl, FONT, 0.42, 1)
            put_text_shadow(ctrl, ring_lbl, (ring_cx - rw//2, ring_cy + rh//2), FONT, 0.42, TEXT_DARK, 1)
            put_text_shadow(ctrl, "Buffer", (ring_cx - 22, ring_cy + ring_r + 22), FONT, 0.38, TEXT_MUTED, 1)

            # Last word display
            last_word = debug_motion_word.upper() if debug_motion_word != "--" else "--"
            wc = mode_accent if debug_motion_conf >= MOTION_CONF_THRESH * 100 else TEXT_FAINT
            (ww, wh), _ = cv2.getTextSize(last_word, FONT_M, 1.15, 2)
            wx = 170
            put_text_shadow(ctrl, last_word, (wx, pred_y + 90), FONT_M, 1.15, wc, 2)
            conf_txt = f"{debug_motion_conf:.1f}% confidence"
            put_text_shadow(ctrl, conf_txt, (wx, pred_y + 116), FONT, 0.4, TEXT_MUTED, 1)

        # ── Gate Status Card ──
        gate_y = pred_y + pred_h + 14
        gate_h = 118 if DUAL_GATE_MODE else 66
        draw_neu_raised(ctrl, 12, gate_y, CTRL_W - 24, gate_h, 16, CARD_BASE, depth=4)

        gs_palette = {
            "AGREE":     ACCENT_GREEN,
            "MOTION-OK": ACCENT_GREEN,
            "SINGLE-OK": ACCENT_GREEN,
            "DISAGREE":  ACCENT_RED,
            "LOW CONF":  ACCENT_AMBER,
            "NO MOTION": TEXT_FAINT,
            "WAITING":   TEXT_MUTED,
            "COOLDOWN":  ACCENT_WARN,
            "DUPLICATE": ACCENT_BLUE,
        }
        gs_color = gs_palette.get(gate_status, TEXT_MUTED)

        row_y = gate_y + 28
        put_text_shadow(ctrl, "Hand motion", (30, row_y), FONT, 0.44, TEXT_MUTED, 1)
        m_color = mode_accent if debug_motion_conf >= MOTION_CONF_THRESH*100 else TEXT_FAINT
        put_text_shadow(ctrl, debug_motion_word.upper(), (185, row_y), FONT_M, 0.5, m_color, 1)
        put_text_shadow(ctrl, f"{debug_motion_conf:.0f}%", (CTRL_W - 70, row_y), FONT, 0.44, m_color, 1)

        if DUAL_GATE_MODE:
            row_y2 = row_y + 32
            cv2.line(ctrl, (28, row_y + 12), (CTRL_W - 28, row_y + 12), CARD_PRESSED, 1)
            put_text_shadow(ctrl, "Face signal", (30, row_y2), FONT, 0.44, TEXT_MUTED, 1)
            f_color = mode_accent if debug_face_conf >= FACE_CONF_THRESH*100 else TEXT_FAINT
            put_text_shadow(ctrl, debug_face_word.upper(), (185, row_y2), FONT_M, 0.5, f_color, 1)
            put_text_shadow(ctrl, f"{debug_face_conf:.0f}%", (CTRL_W - 70, row_y2), FONT, 0.44, f_color, 1)

            badge_w = 130
            draw_status_chip(ctrl, 12 + (CTRL_W-24-badge_w)//2, gate_y + gate_h - 34, badge_w, 26,
                              gate_status, gs_color, FONT)
        else:
            badge_w = 130
            draw_status_chip(ctrl, 12 + (CTRL_W-24-badge_w)//2, gate_y + gate_h - 30, badge_w, 24,
                              gate_status, gs_color, FONT)

        # ── Output Text Field (inset) ──
        out_y   = gate_y + gate_h + 14
        out_h   = 90
        draw_neu_inset(ctrl, 12, out_y, CTRL_W - 24, out_h, 16, CARD_PRESSED)
        put_text_shadow(ctrl, "OUTPUT", (28, out_y + 20), FONT, 0.42, TEXT_MUTED, 1)

        # Word-wrap output into up to 2 lines
        max_line_chars = 34
        words_out = typed_output.rstrip()
        if len(words_out) <= max_line_chars:
            lines_out = [words_out]
        else:
            lines_out = ["..." + words_out[-(max_line_chars):]]
        for li, ln in enumerate(lines_out[:2]):
            put_text_shadow(ctrl, ln, (26, out_y + 54 + li * 26), FONT_M, 0.62, TEXT_DARK, 1)

        # Cursor blink
        if int(current_time * 2) % 2 == 0:
            disp_line = lines_out[-1] if lines_out else ""
            (cl_w, _), _ = cv2.getTextSize(disp_line, FONT_M, 0.62, 1)
            cur_x = 26 + cl_w + 3
            cur_y_top = out_y + 38 + (len(lines_out)-1)*26
            cv2.line(ctrl, (cur_x, cur_y_top), (cur_x, cur_y_top + 20), mode_accent, 2)

        # ── Keybind hints ──
        hint_y = CTRL_H - 60
        cv2.line(ctrl, (20, hint_y), (CTRL_W-20, hint_y), CARD_PRESSED, 1)
        hints_lines = [
            "1 Letters   2 Words   N Cam   F Fullscreen",
            "Enter Speak   Space   Bksp   C Clear   Q Quit"
        ]
        for hi, hl in enumerate(hints_lines):
            (hw, _), _ = cv2.getTextSize(hl, FONT, 0.35, 1)
            hx = (CTRL_W - hw) // 2
            cv2.putText(ctrl, hl, (hx, hint_y + 20 + hi * 18), FONT, 0.35, TEXT_FAINT, 1, cv2.LINE_AA)

        # ── Notification toast ──
        if current_time < notification_end:
            toast_alpha = min(1.0, (notification_end - current_time) / 0.35)
            (tw, _), _ = cv2.getTextSize(notification_text, FONT_M, 0.6, 1)
            nw, nh = tw + 48, 40
            nx, ny = (CTRL_W - nw)//2, out_y - nh - 12
            ov2 = ctrl.copy()
            draw_gradient_rounded_rect(ov2, nx, ny, nw, nh, nh//2, mode_accent, mode_accent2)
            cv2.addWeighted(ov2, 0.92*toast_alpha, ctrl, 1 - 0.92*toast_alpha, 0, ctrl)
            put_text_shadow(ctrl, notification_text, (nx + 24, ny + 26), FONT_M, 0.6, (255,255,255), 1)

        cv2.imshow(CTRL_WIN, ctrl)

        # ══════════════════════════════════════════════════════
        #  Key Handling
        # ══════════════════════════════════════════════════════
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('f') or key == ord('F'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(CTRL_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(CTRL_WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(CTRL_WIN, CTRL_W, CTRL_H)
        elif key == ord('1'):
            current_mode = 1
            typed_output = ""
            motion_state = ""
            pinky_history.clear()
            index_history.clear()
            prediction_buffer.clear()
            hand_sequence_buffer.clear()
            last_stable_char = ""
            set_notification("SWITCHED -> LETTERS")
        elif key == ord('2'):
            current_mode = 2
            typed_output = ""
            hand_sequence_buffer.clear()
            debug_motion_word = "--"
            debug_face_word   = "--"
            debug_motion_conf = 0.0
            debug_face_conf   = 0.0
            gate_status       = "WAITING"
            set_notification("SWITCHED -> WORDS")
        elif key == ord('n') or key == ord('N'):
            cap.release()
            camera_index = (camera_index + 1) % 3
            cap = cv2.VideoCapture(camera_index)
            set_notification(f"CAMERA -> {camera_index}")
        elif key == 13:
            speak_text(typed_output)
        elif key == 32:
            typed_output += " "
        elif key in (8, 127):
            typed_output = typed_output[:-1]
        elif key == ord('c') or key == ord('C'):
            typed_output = ""
            set_notification("OUTPUT CLEARED")

cap.release()
cv2.destroyAllWindows()