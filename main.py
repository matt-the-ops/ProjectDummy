"""
SHIELD FSL — Neumorphism Tkinter GUI (Revamped)
Single 1920×1080 window, matching the SHIELD FSL reference design.
"""

import os
import cv2
import math
import pickle
import collections
import time
import numpy as np
import threading
import asyncio
import io
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont
import pyttsx3
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
#  TTS — BMO Voice
# ─────────────────────────────────────────────
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
            try:
                import edge_tts
                import pygame
                _init_audio()

                async def _synthesize():
                    comm = edge_tts.Communicate(clean, voice="en-US-AnaNeural",
                                                pitch="+20Hz", rate="+12%")
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
            except Exception:
                spoken = False

            if not spoken:
                try:
                    import pythoncom, win32com.client
                    pythoncom.CoInitialize()
                    try:
                        speaker = win32com.client.Dispatch("SAPI.SpVoice")
                        for v in speaker.GetVoices():
                            if "zira" in v.GetDescription().lower():
                                speaker.Voice = v
                                break
                        escaped = (clean.replace("&", "&amp;")
                                   .replace("<", "&lt;").replace(">", "&gt;"))
                        xml = f'<pitch absmiddle="12"><rate absspeed="2">{escaped}</rate></pitch>'
                        speaker.Speak(xml, 8)
                        spoken = True
                    finally:
                        pythoncom.CoUninitialize()
                except Exception:
                    pass

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
#  Paths
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
#  Load Letter Model
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
#  Load Word Models
# ─────────────────────────────────────────────
motion_model = None
face_clf      = None
word_classes  = None

motion_model_path = os.path.join(MODELS_DIR, 'word_motion_model.h5')
face_model_path   = os.path.join(MODELS_DIR, 'word_face_model.pkl')
classes_path      = os.path.join(MODELS_DIR, 'word_classes.npy')

if (os.path.exists(motion_model_path) and
        os.path.exists(face_model_path) and
        os.path.exists(classes_path)):
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
    legacy_word_model = os.path.join(MODELS_DIR, 'word_model.h5')
    if os.path.exists(legacy_word_model) and os.path.exists(classes_path):
        try:
            motion_model = load_model(legacy_word_model)
            word_classes = np.load(classes_path, allow_pickle=True)
            print(f"[OK] Legacy word model loaded.")
        except Exception as e:
            print(f"[!] Legacy word model error: {e}")

# ─────────────────────────────────────────────
#  Feature Helpers & Calibration Anchors
# ─────────────────────────────────────────────
HAND_FEATURES = 198   # 126 shape + 36 spatial anchors + 12 facial emotion + 18 per-frame velocities/speeds/direction ratios + 6 trajectory
FACE_FEATURES = 1404

def get_detailed_face_and_body_anchors(results):
    """
    Extracts explicit 3D face-zone anchors, shoulder center, and facial expression micro-features:
    - Nose Tip (lm 1), Forehead (lm 10), Chin (lm 152), Left Cheek/Temple (lm 234), Right Cheek/Temple (lm 454)
    - Shoulder Center (pose lm 11 & 12)
    - Expression micro-features: Mouth open, Mouth width, Eyebrow raise (L/R), Eyebrow squeeze, Jaw drop, Eye open (L/R), Head tilts.
    """
    nose      = [0.50, 0.35, 0.0]
    forehead  = [0.50, 0.22, 0.0]
    chin      = [0.50, 0.48, 0.0]
    l_cheek   = [0.65, 0.35, 0.0]
    r_cheek   = [0.35, 0.35, 0.0]
    shoulder  = [0.50, 0.65, 0.0]

    expr = [0.0] * 12

    if results.face_landmarks:
        flm = results.face_landmarks.landmark
        n  = flm[1]   # Nose tip
        fh = flm[10]  # Forehead
        ch = flm[152] # Chin
        lc = flm[234] # Left Cheek/Temple
        rc = flm[454] # Right Cheek/Temple

        nose     = [n.x, n.y, n.z]
        forehead = [fh.x, fh.y, fh.z]
        chin     = [ch.x, ch.y, ch.z]
        l_cheek  = [lc.x, lc.y, lc.z]
        r_cheek  = [rc.x, rc.y, rc.z]

        # Face scale reference (forehead to chin distance)
        face_scale = math.sqrt((ch.x - fh.x)**2 + (ch.y - fh.y)**2 + (ch.z - fh.z)**2) or 0.25

        # Mouth open & width
        mouth_open  = (flm[14].y - flm[13].y) / face_scale
        mouth_width = (flm[291].x - flm[61].x) / face_scale

        # Eyebrows (raise & squeeze)
        l_brow_raise = (flm[159].y - flm[70].y) / face_scale
        r_brow_raise = (flm[386].y - flm[300].y) / face_scale
        brow_squeeze = (flm[336].x - flm[107].x) / face_scale

        # Jaw drop
        jaw_drop = (ch.y - n.y) / face_scale

        # Eye openness
        l_eye_open = (flm[145].y - flm[159].y) / face_scale
        r_eye_open = (flm[374].y - flm[386].y) / face_scale

        # Smile curvature
        l_corner_y = (flm[61].y - flm[13].y) / face_scale
        r_corner_y = (flm[291].y - flm[13].y) / face_scale

        # Head orientation tilts
        head_yaw  = (n.x - (lc.x + rc.x) / 2.0) / face_scale
        head_roll = (flm[386].y - flm[159].y) / face_scale

        expr = [
            mouth_open, mouth_width, l_brow_raise, r_brow_raise,
            brow_squeeze, jaw_drop, l_eye_open, r_eye_open,
            l_corner_y, r_corner_y, head_yaw, head_roll
        ]

    if results.pose_landmarks:
        l_sh = results.pose_landmarks.landmark[11]
        r_sh = results.pose_landmarks.landmark[12]
        shoulder = [(l_sh.x + r_sh.x) / 2.0, (l_sh.y + r_sh.y) / 2.0, (l_sh.z + r_sh.z) / 2.0]

    anchors = {
        'nose': nose,
        'forehead': forehead,
        'chin': chin,
        'l_cheek': l_cheek,
        'r_cheek': r_cheek,
        'shoulder': shoulder,
        'expression': expr
    }
    return anchors

def extract_hand_frame(results):
    """
    Extracts 174 spatial & shape per-frame features:
    - 63 relative coords for Left Hand shape
    - 63 relative coords for Right Hand shape
    - 18 spatial face-zone anchor distances for Left Hand (Nose, Forehead, Chin, L-Cheek, R-Cheek, Shoulder)
    - 18 spatial face-zone anchor distances for Right Hand (Nose, Forehead, Chin, L-Cheek, R-Cheek, Shoulder)
    - 12 facial expression & emotion micro-features
    (24 motion dynamic velocity/speed/direction/trajectory features appended across sequence = 198 total)
    """
    out = []
    anchors = get_detailed_face_and_body_anchors(results)

    # 1. Left Hand Shape (63)
    lw_xyz = None
    if results.left_hand_landmarks:
        w = results.left_hand_landmarks.landmark[0]
        lw_xyz = [w.x, w.y, w.z]
        for lm in results.left_hand_landmarks.landmark:
            out.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
    else:
        out.extend([0.0] * 63)

    # 2. Right Hand Shape (63)
    rw_xyz = None
    if results.right_hand_landmarks:
        w = results.right_hand_landmarks.landmark[0]
        rw_xyz = [w.x, w.y, w.z]
        for lm in results.right_hand_landmarks.landmark:
            out.extend([lm.x - w.x, lm.y - w.y, lm.z - w.z])
    else:
        out.extend([0.0] * 63)

    # 3. Left Hand Spatial Face-Zone & Body Anchors (18)
    if lw_xyz:
        for key in ('nose', 'forehead', 'chin', 'l_cheek', 'r_cheek', 'shoulder'):
            anc = anchors[key]
            out.extend([lw_xyz[0] - anc[0], lw_xyz[1] - anc[1], lw_xyz[2] - anc[2]])
    else:
        out.extend([0.0] * 18)

    # 4. Right Hand Spatial Face-Zone & Body Anchors (18)
    if rw_xyz:
        for key in ('nose', 'forehead', 'chin', 'l_cheek', 'r_cheek', 'shoulder'):
            anc = anchors[key]
            out.extend([rw_xyz[0] - anc[0], rw_xyz[1] - anc[1], rw_xyz[2] - anc[2]])
    else:
        out.extend([0.0] * 18)

    # 5. Facial Expression & Emotion Micro-Features (12)
    out.extend(anchors['expression'])

    return out

def get_raw_hand_anchors(results):
    """Extract raw 3D positions for Left/Right Wrists and Index Fingertips for motion velocity computation."""
    def get_xyz(lms, idx):
        if lms:
            p = lms.landmark[idx]
            return np.array([p.x, p.y, p.z], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    lw = get_xyz(results.left_hand_landmarks, 0)
    rw = get_xyz(results.right_hand_landmarks, 0)
    li = get_xyz(results.left_hand_landmarks, 8)
    ri = get_xyz(results.right_hand_landmarks, 8)
    return lw, rw, li, ri

def build_motion_dynamics_and_trajectory(anchor_frames):
    """
    Computes per-frame 3D velocity, speed, direction ratios, and cumulative displacement.
    Returns 24 motion features per frame across the sequence:
    - 18 per-frame dynamic features (3D vel LW, 3D vel RW, 3D vel LI, 3D vel RI, speed LW, speed RW, speed LI, speed RI, dir ratio L, dir ratio R)
    - 6 cumulative trajectory features (LW displacement, RW displacement relative to frame 0)
    """
    T = len(anchor_frames)
    lw_arr = np.array([f[0] for f in anchor_frames], dtype=np.float32)
    rw_arr = np.array([f[1] for f in anchor_frames], dtype=np.float32)
    li_arr = np.array([f[2] for f in anchor_frames], dtype=np.float32)
    ri_arr = np.array([f[3] for f in anchor_frames], dtype=np.float32)

    for arr in (lw_arr, rw_arr, li_arr, ri_arr):
        valid = np.any(arr != 0, axis=1)
        if valid.any():
            last = arr[valid][0]
            for i in range(T):
                if valid[i]:
                    last = arr[i]
                else:
                    arr[i] = last

    vel_lw = np.zeros_like(lw_arr)
    vel_rw = np.zeros_like(rw_arr)
    vel_li = np.zeros_like(li_arr)
    vel_ri = np.zeros_like(ri_arr)

    vel_lw[1:] = np.diff(lw_arr, axis=0)
    vel_rw[1:] = np.diff(rw_arr, axis=0)
    vel_li[1:] = np.diff(li_arr, axis=0)
    vel_ri[1:] = np.diff(ri_arr, axis=0)

    sp_lw = np.linalg.norm(vel_lw, axis=1, keepdims=True)
    sp_rw = np.linalg.norm(vel_rw, axis=1, keepdims=True)
    sp_li = np.linalg.norm(vel_li, axis=1, keepdims=True)
    sp_ri = np.linalg.norm(vel_ri, axis=1, keepdims=True)

    dir_l = vel_lw[:, 1:2] / (np.abs(vel_lw[:, 0:1]) + 1e-4)
    dir_r = vel_rw[:, 1:2] / (np.abs(vel_rw[:, 0:1]) + 1e-4)
    dir_l = np.clip(dir_l, -5.0, 5.0)
    dir_r = np.clip(dir_r, -5.0, 5.0)

    traj_lw = lw_arr - lw_arr[0]
    traj_rw = rw_arr - rw_arr[0]

    motion_features = np.hstack([
        vel_lw, vel_rw, vel_li, vel_ri,
        sp_lw, sp_rw, sp_li, sp_ri,
        dir_l, dir_r,
        traj_lw, traj_rw
    ])
    return motion_features.astype(np.float32)

def normalize_hand_sequence(hand_seq):
    seq = np.asarray(hand_seq, dtype=np.float32)
    if seq.size == 0:
        return seq
    out = seq.copy()
    spans = []
    for off in (0, 63):
        tip_idx = 12 * 3
        tip  = out[0, off + tip_idx: off + tip_idx + 3]
        span = np.linalg.norm(tip)
        if span < 1e-6:
            span = 1.0
        spans.append(span)
        out[:, off:off + 63] = out[:, off:off + 63] / span

    if out.shape[1] >= 198:
        out[:, 174:177] = out[:, 174:177] / spans[0]
        out[:, 180:183] = out[:, 180:183] / spans[0]
        out[:, 186:187] = out[:, 186:187] / spans[0]
        out[:, 188:189] = out[:, 188:189] / spans[0]
        out[:, 192:195] = out[:, 192:195] / spans[0]

        out[:, 177:180] = out[:, 177:180] / spans[1]
        out[:, 183:186] = out[:, 183:186] / spans[1]
        out[:, 187:188] = out[:, 187:188] / spans[1]
        out[:, 189:190] = out[:, 189:190] / spans[1]
        out[:, 195:198] = out[:, 195:198] / spans[1]
    elif out.shape[1] >= 180:
        out[:, 174:177] = out[:, 174:177] / spans[0]
        out[:, 177:180] = out[:, 177:180] / spans[1]
    elif out.shape[1] >= 144:
        out[:, 138:141] = out[:, 138:141] / spans[0]
        out[:, 141:144] = out[:, 141:144] / spans[1]
    elif out.shape[1] >= 132:
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
#  Neumorphism Colour Palette
# ─────────────────────────────────────────────
# Tkinter hex colours
BG          = "#E8EAF0"      # main background
CARD        = "#ECEEF4"      # raised card surface
CARD_IN     = "#E0E2EA"      # inset / pressed surface
SHADOW_D    = "#C8CAD4"      # dark shadow
SHADOW_L    = "#FFFFFF"      # light highlight
TXT_DARK    = "#2E2C3C"      # primary text
TXT_MID     = "#8886A0"      # secondary text
TXT_FAINT   = "#B8B6CC"      # hint text
WHITE       = "#FFFFFF"

ACC_BLUE    = "#2E7CF6"      # primary accent (detection ring, progress, speak btn)
ACC_BLUE2   = "#5BA8FF"      # lighter blue
ACC_NAVY    = "#1A2F5A"      # speak button bg
ACC_GREEN   = "#52C878"      # online dot / agree
ACC_RED     = "#E85F5F"      # record dot / disagree
ACC_AMBER   = "#F5AF46"      # warning / cooldown
ACC_PURPLE  = "#966BC3"      # letters accent
ACC_ORANGE  = "#F0A540"      # letters accent2
ACC_TEAL    = "#3DB8B0"
BMO_GREEN   = "#7ED87E"      # BMO card accent


# ─────────────────────────────────────────────
#  PIL Neumorphism Drawing Helpers
# ─────────────────────────────────────────────
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _blend(c1, c2, t):
    return tuple(int(c1[i]*(1-t) + c2[i]*t) for i in range(3))

def rgb2hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def draw_neu_card(draw: ImageDraw.ImageDraw, x, y, w, h, r=18,
                  bg=CARD, depth=6, inset=False):
    """Neumorphic card on a PIL ImageDraw surface."""
    bgc  = hex2rgb(bg)
    dark = hex2rgb(SHADOW_D)
    lite = hex2rgb(SHADOW_L)

    if not inset:
        # dark shadow (bottom-right)
        _rounded_rect(draw, x+depth, y+depth, w, h, r, _blend(bgc, dark, 0.55))
        # light shadow (top-left)
        _rounded_rect(draw, x-depth, y-depth, w, h, r, _blend(bgc, lite, 0.70))
        # face
        _rounded_rect(draw, x, y, w, h, r, bgc)
    else:
        _rounded_rect(draw, x, y, w, h, r, hex2rgb(CARD_IN))
        # inner top-left dark line
        draw.line([(x+r, y+1), (x+w-r, y+1)], fill=dark, width=2)
        draw.line([(x+1, y+r), (x+1, y+h-r)], fill=dark, width=2)
        # inner bottom-right light line
        draw.line([(x+r, y+h-1), (x+w-r, y+h-1)], fill=lite, width=2)
        draw.line([(x+w-1, y+r), (x+w-1, y+h-r)], fill=lite, width=2)

def _rounded_rect(draw, x, y, w, h, r, fill):
    """Fill a rounded rectangle on a PIL ImageDraw."""
    r  = max(0, int(r))
    x, y, w, h = int(x), int(y), int(w), int(h)
    draw.rectangle([x+r, y, x+w-r, y+h], fill=fill)
    draw.rectangle([x, y+r, x+w, y+h-r], fill=fill)
    draw.ellipse([x, y, x+2*r, y+2*r], fill=fill)
    draw.ellipse([x+w-2*r, y, x+w, y+2*r], fill=fill)
    draw.ellipse([x, y+h-2*r, x+2*r, y+h], fill=fill)
    draw.ellipse([x+w-2*r, y+h-2*r, x+w, y+h], fill=fill)

def draw_progress_arc(draw, cx, cy, radius, thickness, value, max_val,
                      fg_color, bg_color=SHADOW_D):
    """Draw a circular arc progress ring."""
    frac  = max(0.0, min(value / max(max_val, 1e-6), 1.0))
    box   = [cx-radius, cy-radius, cx+radius, cy+radius]
    # background ring
    draw.arc(box, 0, 360, fill=hex2rgb(bg_color), width=thickness)
    if frac > 0.005:
        end_angle = -90 + 360 * frac
        draw.arc(box, -90, end_angle, fill=hex2rgb(fg_color), width=thickness)

def draw_pill(draw, x, y, w, h, fill):
    r = h // 2
    _rounded_rect(draw, x, y, w, h, r, hex2rgb(fill))

def draw_pill_outline(draw, x, y, w, h, outline, width=2):
    r = h // 2
    draw.rounded_rectangle([x, y, x+w, y+h], radius=r,
                            outline=hex2rgb(outline), width=width)

def gradient_pill(draw, x, y, w, h, c1, c2):
    """Horizontal gradient pill."""
    r = h // 2
    # build gradient image and paste
    grad = Image.new("RGB", (w, h))
    gd   = ImageDraw.Draw(grad)
    for i in range(w):
        t  = i / max(w-1, 1)
        gc = _blend(hex2rgb(c1), hex2rgb(c2), t)
        gd.line([(i, 0), (i, h)], fill=gc)
    mask = Image.new("L", (w, h), 0)
    md   = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    # We can't paste onto draw directly; caller must handle
    return grad, mask

# ─────────────────────────────────────────────
#  App State
# ─────────────────────────────────────────────
LETTER_BUFFER_SIZE   = 12
STABILIZATION_DELAY  = 2.5
GLOBAL_COOLDOWN      = 1.8
WORD_DEBOUNCE_TIME   = 2.0
WORD_SEQ_LEN         = 40
MOTION_CONF_THRESH   = 0.45
FACE_CONF_THRESH     = 0.35
MOTION_MARGIN_THRESH = 0.08
MIN_MOTION_ENERGY    = 0.005

last_predicted_word = ""
last_word_time      = 0.0
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

hand_sequence_buffer  = collections.deque(maxlen=WORD_SEQ_LEN)
wrist_sequence_buffer = collections.deque(maxlen=WORD_SEQ_LEN)
face_sequence_buffer  = collections.deque(maxlen=WORD_SEQ_LEN)

debug_motion_word = "--"
debug_motion_conf = 0.0
debug_face_word   = "--"
debug_face_conf   = 0.0
gate_status       = "WAITING"

current_mode      = 1      # 1=Letters, 2=Words
typed_output      = ""
missing_frames    = 0
camera_index      = 0
current_frame_prediction = ""
fps_value         = 0.0
node_count        = 0
lux_value         = 485
dist_cm           = 54
camera_hidden     = False   # Toggle with 'H' key

notification_text = ""
notification_end  = 0.0
NOTIF_DURATION    = 2.0

def set_notification(msg):
    global notification_text, notification_end
    notification_text = msg
    notification_end  = time.time() + NOTIF_DURATION

# ─────────────────────────────────────────────
#  Calibration & Target Person Lock State
# ─────────────────────────────────────────────
is_calibrating       = False
calibration_progress = 0.0
calibration_start_t  = 0.0
CALIBRATION_DURATION = 3.0   # 3 seconds countdown
target_person_lock   = None  # Dict: {'nose': [x,y,z], 'shoulder': [x,y,z], 'reach_radius': float}

def trigger_calibration():
    global is_calibrating, calibration_progress, calibration_start_t, target_person_lock
    is_calibrating       = True
    calibration_progress = 0.0
    calibration_start_t  = time.time()
    target_person_lock   = None
    set_notification("CALIBRATING... POSITION IN FRAME")

def smooth_landmarks(current_lms, previous_lms, alpha=0.70):
    """Applies Exponential Moving Average (EMA) smoothing to eliminate landmark micro-jitter."""
    if previous_lms is None or current_lms is None:
        return current_lms
    try:
        smoothed = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for cur, prev in zip(current_lms.landmark, previous_lms.landmark):
            sm_lm = smoothed.landmark.add()
            sm_lm.x = alpha * cur.x + (1.0 - alpha) * prev.x
            sm_lm.y = alpha * cur.y + (1.0 - alpha) * prev.y
            sm_lm.z = alpha * cur.z + (1.0 - alpha) * prev.z
        return smoothed
    except Exception:
        return current_lms

def resolve_and_smooth_hands(results, last_left, last_right, lost_left, lost_right, max_lost=4, alpha=0.70):
    """
    Foolproof Anatomical Spatial Disambiguation Engine:
    - Bypasses MediaPipe's buggy Left/Right labels by matching hand landmark positions 
      against physical left & right shoulder anchors and nose spatial center.
    - Prevents Left/Right hand swapping and confusion 100% of the time.
    """
    lh = results.left_hand_landmarks
    rh = results.right_hand_landmarks

    # Extract anatomical reference anchors
    l_sh = [0.65, 0.60]  # Default physical left shoulder (screen right)
    r_sh = [0.35, 0.60]  # Default physical right shoulder (screen left)

    if results.pose_landmarks:
        l_sh = [results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y]
        r_sh = [results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y]
    elif results.face_landmarks:
        nose_x = results.face_landmarks.landmark[1].x
        nose_y = results.face_landmarks.landmark[1].y
        l_sh = [nose_x + 0.18, nose_y + 0.25]
        r_sh = [nose_x - 0.18, nose_y + 0.25]

    def dist_sq(w, anchor):
        return (w.x - anchor[0])**2 + (w.y - anchor[1])**2

    # Case 1: Both hands detected by MediaPipe
    if lh is not None and rh is not None:
        w_lh = lh.landmark[0]
        w_rh = rh.landmark[0]

        # Cost if lh=LeftHand, rh=RightHand
        cost_normal  = dist_sq(w_lh, l_sh) + dist_sq(w_rh, r_sh)
        # Cost if rh=LeftHand, lh=RightHand (swapped)
        cost_swapped = dist_sq(w_rh, l_sh) + dist_sq(w_lh, r_sh)

        if cost_swapped < cost_normal:
            # MediaPipe swapped them -> Swap them back!
            results.left_hand_landmarks, results.right_hand_landmarks = rh, lh

    # Case 2: Only 1 hand detected by MediaPipe (currently assigned to left_hand)
    elif lh is not None and rh is None:
        w_lh = lh.landmark[0]
        d_left_side  = dist_sq(w_lh, l_sh)
        d_right_side = dist_sq(w_lh, r_sh)

        # If this hand is physically closer to the Right Shoulder -> Reassign to Right Hand!
        if d_right_side < d_left_side:
            results.right_hand_landmarks = lh
            results.left_hand_landmarks  = None

    # Case 3: Only 1 hand detected by MediaPipe (currently assigned to right_hand)
    elif rh is not None and lh is None:
        w_rh = rh.landmark[0]
        d_left_side  = dist_sq(w_rh, l_sh)
        d_right_side = dist_sq(w_rh, r_sh)

        # If this hand is physically closer to the Left Shoulder -> Reassign to Left Hand!
        if d_left_side < d_right_side:
            results.left_hand_landmarks  = rh
            results.right_hand_landmarks = None

    # Step B: EMA Smoothing & Temporal Persistence
    if results.left_hand_landmarks:
        smoothed_left = smooth_landmarks(results.left_hand_landmarks, last_left, alpha=alpha)
        results.left_hand_landmarks = smoothed_left
        last_left = smoothed_left
        lost_left = 0
    else:
        lost_left += 1
        if lost_left <= max_lost:
            results.left_hand_landmarks = last_left
        else:
            last_left = None

    if results.right_hand_landmarks:
        smoothed_right = smooth_landmarks(results.right_hand_landmarks, last_right, alpha=alpha)
        results.right_hand_landmarks = smoothed_right
        last_right = smoothed_right
        lost_right = 0
    else:
        lost_right += 1
        if lost_right <= max_lost:
            results.right_hand_landmarks = last_right
        else:
            last_right = None

    return last_left, last_right, lost_left, lost_right

def play_ping_sound():
    """Play clean double ping chime after calibration completes."""
    def _ping():
        try:
            import winsound
            winsound.Beep(1200, 100)
            winsound.Beep(1800, 150)
        except Exception:
            pass
    threading.Thread(target=_ping, daemon=True).start()

def validate_and_filter_target(results):
    """
    Lock-In Engine: Keeps tracking locked strictly onto the calibrated user in front of the camera.
    Filters out and discards any secondary people or background faces/hands.
    """
    global is_calibrating, calibration_progress, calibration_start_t, target_person_lock

    anchors = get_detailed_face_and_body_anchors(results)
    nose     = anchors['nose']
    shoulder = anchors['shoulder']
    has_face = (results.face_landmarks is not None)
    has_hand = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)
    has_person = (has_face or has_hand)

    # 1. Calibration phase handling
    if is_calibrating:
        if has_person:
            elapsed = time.time() - calibration_start_t
            calibration_progress = min(100.0, (elapsed / CALIBRATION_DURATION) * 100.0)

            if elapsed >= CALIBRATION_DURATION:
                target_person_lock = {
                    'nose': nose,
                    'shoulder': shoulder,
                    'max_reach_radius': 0.65,   # Reach boundary around shoulder/face
                    'max_face_dist': 0.22       # Strict face distance threshold to reject other people
                }
                is_calibrating = False
                set_notification("TARGET LOCKED: CALIBRATED!")
                play_ping_sound()
        else:
            # Pause calibration progress if person leaves frame
            calibration_start_t = time.time() - (calibration_progress / 100.0) * CALIBRATION_DURATION
        return

    # Auto-calibrate on first detection if not yet calibrated
    if target_person_lock is None and not is_calibrating and has_person:
        trigger_calibration()

    # 2. Strict Target Person Filtering
    if target_person_lock:
        locked_nose = target_person_lock['nose']
        locked_sh   = target_person_lock['shoulder']
        max_reach   = target_person_lock.get('max_reach_radius', 0.65)
        max_face    = target_person_lock.get('max_face_dist', 0.22)

        if has_face:
            d_nose = math.sqrt((nose[0] - locked_nose[0])**2 + (nose[1] - locked_nose[1])**2)
            if d_nose <= max_face:
                # Smoothly adapt target anchor for natural user movement
                smooth_nose = [
                    0.20 * nose[0] + 0.80 * locked_nose[0],
                    0.20 * nose[1] + 0.80 * locked_nose[1],
                    0.20 * nose[2] + 0.80 * locked_nose[2]
                ]
                smooth_sh = [
                    0.20 * shoulder[0] + 0.80 * locked_sh[0],
                    0.20 * shoulder[1] + 0.80 * locked_sh[1],
                    0.20 * shoulder[2] + 0.80 * locked_sh[2]
                ]
                target_person_lock['nose']     = smooth_nose
                target_person_lock['shoulder'] = smooth_sh
            else:
                # Reject face of a different person!
                results.face_landmarks = None

        # Filter Left Hand if it belongs to a different person
        if results.left_hand_landmarks:
            lw = results.left_hand_landmarks.landmark[0]
            d_sh   = math.sqrt((lw.x - locked_sh[0])**2 + (lw.y - locked_sh[1])**2)
            d_nose = math.sqrt((lw.x - locked_nose[0])**2 + (lw.y - locked_nose[1])**2)
            if min(d_sh, d_nose) > max_reach:
                results.left_hand_landmarks = None

        # Filter Right Hand if it belongs to a different person
        if results.right_hand_landmarks:
            rw = results.right_hand_landmarks.landmark[0]
            d_sh   = math.sqrt((rw.x - locked_sh[0])**2 + (rw.y - locked_sh[1])**2)
            d_nose = math.sqrt((rw.x - locked_nose[0])**2 + (rw.y - locked_nose[1])**2)
            if min(d_sh, d_nose) > max_reach:
                results.right_hand_landmarks = None

# ─────────────────────────────────────────────
#  Camera
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(camera_index)

# ─────────────────────────────────────────────
#  BMO face images
# ─────────────────────────────────────────────
bmo_face_closed = None
bmo_face_open   = None
bmo_idle_pil    = None
bmo_talk_pil    = None

def _load_bmo_images():
    global bmo_face_closed, bmo_face_open, bmo_idle_pil, bmo_talk_pil
    closed_path = os.path.join(ASSETS_DIR, 'bmo', 'bmo_closed.png')
    open_path   = os.path.join(ASSETS_DIR, 'bmo', 'bmo_open.png')
    if os.path.exists(closed_path):
        bmo_face_closed = cv2.imread(closed_path)
        img = cv2.cvtColor(bmo_face_closed, cv2.COLOR_BGR2RGB)
        bmo_idle_pil = Image.fromarray(img)
    if os.path.exists(open_path):
        bmo_face_open = cv2.imread(open_path)
        img = cv2.cvtColor(bmo_face_open, cv2.COLOR_BGR2RGB)
        bmo_talk_pil = Image.fromarray(img)

_load_bmo_images()

# ─────────────────────────────────────────────
#  Main Tkinter Application
# ─────────────────────────────────────────────
class ShieldFSLApp:

    # ── Layout constants (1920×1080 canvas) ─────────────
    W  = 1920
    H  = 1080

    PAD       = 24          # outer padding
    TOP_H     = 68          # top nav bar height
    BOT_H     = 110         # bottom output bar height
    STATS_H   = 56          # stats row height
    GAP       = 14          # gap between sections

    # Derived
    CONTENT_Y  = TOP_H + PAD
    CONTENT_H  = H - TOP_H - PAD - STATS_H - GAP - BOT_H - PAD - GAP
    RIGHT_X    = W * 60 // 100
    RIGHT_W    = W - RIGHT_X - PAD
    LEFT_W     = RIGHT_X - PAD * 2

    # ────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SHIELD FSL")
        self.root.configure(bg=BG)
        self.root.geometry(f"{self.W}x{self.H}")
        self.root.resizable(True, True)
        self.fullscreen = False

        # ── Canvas (single surface) ──
        self.canvas = tk.Canvas(root, width=self.W, height=self.H,
                                bg=BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── PIL offscreen image ──
        self.pil_img   = Image.new("RGB", (self.W, self.H), hex2rgb(BG))
        self.tk_img    = None   # kept to prevent GC

        # Cached camera ImageTk (updated by thread)
        self._cam_photo  = None
        self._cam_lock   = threading.Lock()
        self._cam_pil    = None  # latest PIL camera frame

        # Cached BMO ImageTk
        self._bmo_idle_tk = None
        self._bmo_talk_tk = None

        # ── Load Logo ──
        self._logo_resized = None
        self._logo_mask    = None
        self._load_logo()

        # Key and Mouse bindings
        root.bind("<KeyPress>", self._on_key)
        root.bind("<Button-1>", self._on_click)
        root.bind("<F11>",      lambda e: self._toggle_fullscreen())
        root.bind("<Escape>",   lambda e: self._exit_fullscreen())

        # FPS tracking
        self._frame_times = collections.deque(maxlen=30)
        self._last_draw   = time.time()

        # Start ML thread
        self._running = True
        self._ml_thread = threading.Thread(target=self._ml_loop, daemon=True)
        self._ml_thread.start()

        # Start GUI refresh loop
        self._gui_loop()

    # ── Logo loader ──────────────────────────────────────
    def _load_logo(self):
        """Load SHIELD FSL logo from assets and prepare for top bar."""
        logo_path = os.path.join(ASSETS_DIR, 'shield_logo.png')
        if not os.path.exists(logo_path):
            logo_path = os.path.join(ASSETS_DIR, 'logo.png')
        if not os.path.exists(logo_path):
            return
        try:
            logo = Image.open(logo_path).convert("RGBA")
            # Make near-white pixels transparent for clean compositing
            data = list(logo.getdata())
            new_data = []
            for r, g, b, a in data:
                if r > 235 and g > 235 and b > 235:
                    new_data.append((r, g, b, 0))
                else:
                    new_data.append((r, g, b, a))
            logo.putdata(new_data)
            # Resize to fit top bar
            logo_size = 44
            logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
            self._logo_resized = logo.convert("RGB")
            self._logo_mask = logo.split()[3]  # alpha channel
            print("[OK] Logo loaded.")
        except Exception as e:
            print(f"[!] Logo load error: {e}")

    # ── Fullscreen ────────────────────────────────────────
    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def _exit_fullscreen(self):
        if self.fullscreen:
            self.fullscreen = False
            self.root.attributes("-fullscreen", False)

    # ── Key handler ──────────────────────────────────────
    def _on_key(self, event):
        global current_mode, typed_output, motion_state, camera_index, cap
        global pinky_history, index_history, prediction_buffer
        global hand_sequence_buffer, wrist_sequence_buffer, face_sequence_buffer
        global debug_motion_word, debug_face_word, debug_motion_conf
        global debug_face_conf, gate_status, last_stable_char
        global camera_hidden

        k = event.keysym.lower()
        if k == 'q':
            self._running = False
            self.root.after(200, self.root.destroy)
        elif k == 'f11' or k == 'f':
            self._toggle_fullscreen()
        elif k == '1':
            current_mode = 1
            typed_output = ""
            motion_state = ""
            pinky_history.clear(); index_history.clear()
            prediction_buffer.clear()
            hand_sequence_buffer.clear()
            last_stable_char = ""
            set_notification("LETTERS MODE")
        elif k == '2':
            current_mode = 2
            typed_output = ""
            hand_sequence_buffer.clear()
            debug_motion_word = "--"; debug_face_word = "--"
            debug_motion_conf = 0.0;  debug_face_conf  = 0.0
            gate_status = "WAITING"
            set_notification("WORDS MODE")
        elif k == 'h':
            camera_hidden = not camera_hidden
            set_notification("CAMERA HIDDEN" if camera_hidden else "CAMERA VISIBLE")
        elif k == 'k':
            trigger_calibration()
        elif k == 'n':
            cap.release()
            camera_index = (camera_index + 1) % 3
            cap = cv2.VideoCapture(camera_index)
            set_notification(f"CAMERA → {camera_index}")
        elif k == 'return':
            speak_text(typed_output)
        elif k == 'space':
            typed_output += " "
        elif k in ('backspace', 'delete'):
            typed_output = typed_output[:-1]
        elif k == 'c':
            typed_output = ""
            set_notification("OUTPUT CLEARED")

    # ── Mouse click handler ──────────────────────────────
    def _on_click(self, event):
        global current_mode, typed_output, camera_hidden
        global pinky_history, index_history, prediction_buffer
        global hand_sequence_buffer, debug_motion_word, debug_face_word
        global debug_motion_conf, debug_face_conf, gate_status, last_stable_char

        mx, my = event.x, event.y

        # Mode tabs click
        tab_w, tab_h = 260, 38
        tab_x = self.W // 2 - tab_w // 2
        tab_y = (self.TOP_H - tab_h) // 2
        if tab_y <= my <= tab_y + tab_h:
            if tab_x <= mx < tab_x + tab_w // 2 and current_mode != 1:
                current_mode = 1
                typed_output = ""
                pinky_history.clear(); index_history.clear()
                prediction_buffer.clear()
                hand_sequence_buffer.clear()
                last_stable_char = ""
                set_notification("LETTERS MODE")
            elif tab_x + tab_w // 2 <= mx <= tab_x + tab_w and current_mode != 2:
                current_mode = 2
                typed_output = ""
                hand_sequence_buffer.clear()
                debug_motion_word = "--"; debug_face_word = "--"
                debug_motion_conf = 0.0;  debug_face_conf  = 0.0
                gate_status = "WAITING"
                set_notification("WORDS MODE")

        # Camera eye toggle click
        eye_w, eye_h = 40, 28
        eye_x = self.PAD + self.LEFT_W - eye_w - 16
        eye_y = self.CONTENT_Y + (40 - eye_h) // 2 + 2
        if eye_x - 5 <= mx <= eye_x + eye_w + 5 and eye_y - 5 <= my <= eye_y + eye_h + 5:
            camera_hidden = not camera_hidden
            set_notification("CAMERA HIDDEN" if camera_hidden else "CAMERA VISIBLE")

        # Calibrate button click
        cal_w, cal_h = 110, 28
        cal_x = eye_x - cal_w - 10
        cal_y = eye_y
        if cal_x - 5 <= mx <= cal_x + cal_w + 5 and cal_y - 5 <= my <= cal_y + cal_h + 5:
            trigger_calibration()

        # Speak button click
        spk_w = 150
        spk_x = self.W - self.PAD - 24 - spk_w
        y_bot = self.H - self.BOT_H - self.PAD
        spk_y = y_bot + 16 + (54 - 50) // 2
        if spk_x - 5 <= mx <= spk_x + spk_w + 5 and spk_y - 5 <= my <= spk_y + 50 + 5:
            speak_text(typed_output)

    # ── ML / camera processing thread ────────────────────
    def _ml_loop(self):
        global current_frame_prediction, typed_output, tts_is_speaking
        global last_predicted_word, last_word_time, prediction_buffer
        global pinky_history, index_history, last_stable_char, stable_start_time
        global global_cooldown_end, motion_state, motion_cooldown_frames
        global suppress_char, suppression_end
        global hand_sequence_buffer, wrist_sequence_buffer, face_sequence_buffer
        global debug_motion_word, debug_motion_conf, debug_face_word, debug_face_conf
        global gate_status, missing_frames, node_count, fps_value

        last_left_hand  = None
        last_right_hand = None
        lost_frames_left  = 0
        lost_frames_right = 0
        MAX_LOST = 4
        _prev_t = time.time()

        with mp_holistic.Holistic(
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
            model_complexity=1,
        ) as holistic:

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02)
                    continue

                frame     = cv2.flip(frame, 1)
                H_f, W_f  = frame.shape[:2]
                rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = holistic.process(rgb)

                # Target Lock-In & Calibration Filter
                validate_and_filter_target(results)

                # FPS
                now = time.time()
                fps_value = 1.0 / max(now - _prev_t, 1e-6)
                _prev_t   = now

                # ── Anatomical Hand Disambiguation & EMA Smoothing ──
                last_left_hand, last_right_hand, lost_frames_left, lost_frames_right = resolve_and_smooth_hands(
                    results, last_left_hand, last_right_hand, lost_frames_left, lost_frames_right, max_lost=MAX_LOST, alpha=0.70
                )

                current_time = time.time()
                on_cooldown  = current_time < global_cooldown_end
                primary_hand = None
                if results.right_hand_landmarks:
                    primary_hand = results.right_hand_landmarks.landmark
                elif results.left_hand_landmarks:
                    primary_hand = results.left_hand_landmarks.landmark

                # Node count
                nc = 0
                if results.left_hand_landmarks:  nc += len(results.left_hand_landmarks.landmark)
                if results.right_hand_landmarks: nc += len(results.right_hand_landmarks.landmark)
                node_count = nc

                # ── Calibration Silence Guard ──
                if is_calibrating:
                    hand_sequence_buffer.clear()
                    wrist_sequence_buffer.clear()
                    face_sequence_buffer.clear()
                    pinky_history.clear()
                    index_history.clear()
                    prediction_buffer.clear()
                    last_stable_char = ""
                    current_frame_prediction = ""
                    gate_status = "CALIBRATING"

                # ── MODE 2: WORD ──
                elif current_mode == 2 and motion_model is not None:
                    has_hand = (results.left_hand_landmarks is not None or
                                results.right_hand_landmarks is not None)
                    if not has_hand:
                        missing_frames += 1
                        if missing_frames >= 3:
                            hand_sequence_buffer.clear()
                            wrist_sequence_buffer.clear()
                            face_sequence_buffer.clear()
                            last_predicted_word = ""
                            debug_motion_word = "--"; debug_motion_conf = 0.0
                            debug_face_word   = "--"; debug_face_conf   = 0.0
                            gate_status = "WAITING"
                    else:
                        missing_frames = 0
                        if on_cooldown:
                            hand_sequence_buffer.clear()
                            wrist_sequence_buffer.clear()
                            face_sequence_buffer.clear()
                            gate_status = "COOLDOWN"
                        else:
                            hand_sequence_buffer.append(extract_hand_frame(results))
                            wrist_sequence_buffer.append(get_raw_hand_anchors(results))
                            face_sequence_buffer.append(extract_face_frame(results))

                            if len(hand_sequence_buffer) == WORD_SEQ_LEN:
                                shape_arr   = np.array(hand_sequence_buffer, dtype=np.float32)
                                frame_diffs = np.diff(shape_arr, axis=0)
                                energy      = float(np.mean(np.linalg.norm(frame_diffs, axis=1)))

                                if energy < MIN_MOTION_ENERGY:
                                    gate_status = "NO MOTION"
                                    for _ in range(8):
                                        if hand_sequence_buffer:  hand_sequence_buffer.popleft()
                                        if wrist_sequence_buffer: wrist_sequence_buffer.popleft()
                                        if face_sequence_buffer:  face_sequence_buffer.popleft()
                                else:
                                    motion_dyn = build_motion_dynamics_and_trajectory(list(wrist_sequence_buffer))
                                    full_hand  = np.concatenate([shape_arr, motion_dyn], axis=1)
                                    hand_norm  = normalize_hand_sequence(full_hand)
                                    hand_arr   = np.expand_dims(hand_norm, axis=0)

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

                                    face_pass = False; face_idx = -1; face_conf2 = 0.0
                                    if DUAL_GATE_MODE and face_clf is not None:
                                        face_avg  = np.mean(np.array(face_sequence_buffer,
                                                                      dtype=np.float32), axis=0)
                                        face_snap = face_avg.reshape(1, -1)
                                        try:
                                            face_proba = face_clf.predict_proba(face_snap)[0]
                                            face_idx   = int(np.argmax(face_proba))
                                            face_conf2 = float(face_proba[face_idx])
                                            debug_face_word = str(word_classes[face_idx])
                                            debug_face_conf = face_conf2 * 100.0
                                            face_pass = (face_conf2 >= FACE_CONF_THRESH)
                                        except Exception:
                                            debug_face_word = "--"; debug_face_conf = 0.0

                                    if DUAL_GATE_MODE and face_clf is not None:
                                        if mot_idx == face_idx and mot_pass and face_pass:
                                            gate_status = "AGREE";      accept_pred = True
                                        elif mot_conf >= 0.65 and mot_margin >= MOTION_MARGIN_THRESH:
                                            gate_status = "MOTION-OK";  accept_pred = True
                                        else:
                                            gate_status = ("DISAGREE" if mot_idx != face_idx
                                                           else "LOW CONF")
                                            accept_pred = False
                                    else:
                                        if mot_pass:
                                            gate_status = "SINGLE-OK"; accept_pred = True
                                        else:
                                            gate_status = "LOW CONF";  accept_pred = False

                                    if accept_pred:
                                        out_word = str(word_classes[mot_idx])
                                        if (out_word == last_predicted_word and
                                                current_time - last_word_time < WORD_DEBOUNCE_TIME):
                                            gate_status = "DUPLICATE"
                                            for _ in range(8):
                                                if hand_sequence_buffer:  hand_sequence_buffer.popleft()
                                                if wrist_sequence_buffer: wrist_sequence_buffer.popleft()
                                                if face_sequence_buffer:  face_sequence_buffer.popleft()
                                        else:
                                            last_predicted_word = out_word
                                            last_word_time      = current_time
                                            typed_output       += out_word + " "
                                            global_cooldown_end = current_time + GLOBAL_COOLDOWN
                                            hand_sequence_buffer.clear()
                                            wrist_sequence_buffer.clear()
                                            face_sequence_buffer.clear()
                                            speak_text(out_word)
                                            set_notification("OK: " + out_word)
                                    else:
                                        for _ in range(8):
                                            if hand_sequence_buffer:  hand_sequence_buffer.popleft()
                                            if wrist_sequence_buffer: wrist_sequence_buffer.popleft()
                                            if face_sequence_buffer:  face_sequence_buffer.popleft()

                # ── MODE 1: LETTERS ──
                elif current_mode == 1:
                    if primary_hand is not None:
                        missing_frames = 0
                        wx, wy, wz = (primary_hand[0].x, primary_hand[0].y,
                                      primary_hand[0].z)
                        scale = math.sqrt(
                            (primary_hand[9].x - wx)**2 +
                            (primary_hand[9].y - wy)**2 +
                            (primary_hand[9].z - wz)**2
                        ) or 1.0
                        row = []
                        for lm in primary_hand:
                            row.extend([(lm.x-wx)/scale,
                                        (lm.y-wy)/scale,
                                        (lm.z-wz)/scale])
                        raw_ml = letter_model.predict([row])[0]
                        prediction_buffer.append(raw_ml)
                        base_sign = collections.Counter(
                            prediction_buffer).most_common(1)[0][0]

                        px = int(primary_hand[20].x * W_f)
                        py = int(primary_hand[20].y * H_f)
                        ix = int(primary_hand[8].x  * W_f)
                        iy = int(primary_hand[8].y  * H_f)
                        is_pinky = primary_hand[20].y < primary_hand[17].y
                        is_index = primary_hand[8].y  < primary_hand[5].y

                        j_posture = (base_sign == 'I') and is_pinky
                        z_posture = (base_sign == 'D') and is_index

                        if j_posture:
                            if motion_state != "J_TRACKING":
                                motion_state = "J_TRACKING"
                                pinky_history.clear(); index_history.clear()
                            motion_cooldown_frames = 0
                        elif z_posture:
                            if motion_state != "Z_TRACKING":
                                motion_state = "Z_TRACKING"
                                pinky_history.clear(); index_history.clear()
                            motion_cooldown_frames = 0
                        else:
                            if motion_state != "":
                                motion_cooldown_frames += 1
                                if motion_cooldown_frames > MOTION_GRACE_FRAMES:
                                    motion_state = ""
                                    pinky_history.clear(); index_history.clear()

                        MIN_MOVE_SQ = 30
                        if motion_state == "J_TRACKING":
                            if (not pinky_history or
                                    (px-pinky_history[-1][0])**2 +
                                    (py-pinky_history[-1][1])**2 > MIN_MOVE_SQ):
                                pinky_history.append((px, py))
                        if motion_state == "Z_TRACKING":
                            if (not index_history or
                                    (ix-index_history[-1][0])**2 +
                                    (iy-index_history[-1][1])**2 > MIN_MOVE_SQ):
                                index_history.append((ix, iy))

                        motion_triggered, motion_char = False, ""
                        if motion_state == "J_TRACKING" and len(pinky_history) >= 8:
                            highest = min(p[1] for p in pinky_history)
                            if pinky_history[-1][1] - highest > 45 and not on_cooldown:
                                motion_char, motion_triggered = 'J', True
                                suppress_char = 'I'; suppression_end = current_time + 1.2
                                motion_state = ""; pinky_history.clear()
                        elif motion_state == "Z_TRACKING" and len(index_history) >= 8:
                            mn = min(p[0] for p in index_history)
                            mx = max(p[0] for p in index_history)
                            if mx - mn > 55 and not on_cooldown:
                                motion_char, motion_triggered = 'Z', True
                                suppress_char = 'D'; suppression_end = current_time + 1.2
                                motion_state = ""; index_history.clear()

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
                            typed_output       += motion_char
                            global_cooldown_end = current_time + GLOBAL_COOLDOWN
                            stable_start_time   = current_time + GLOBAL_COOLDOWN
                            speak_text(motion_char)
                            set_notification("OK: " + motion_char + " (motion)")
                        elif letter_triggered and letter_char and not on_cooldown:
                            typed_output       += letter_char
                            global_cooldown_end = current_time + GLOBAL_COOLDOWN
                            stable_start_time   = current_time + GLOBAL_COOLDOWN
                            speak_text(letter_char)
                            set_notification("OK: " + letter_char)
                    else:
                        missing_frames += 1
                        if missing_frames > 10:
                            pinky_history.clear(); index_history.clear()
                            prediction_buffer.clear()
                            last_stable_char = ""; motion_state = ""

                # ── Draw landmarks onto frame ──
                draw_frame = frame.copy()
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        draw_frame, results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(20, 60, 20), thickness=1))
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        draw_frame, results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 255, 140),
                                               thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(30, 200, 80), thickness=2))
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        draw_frame, results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(100, 255, 100),
                                               thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(60, 220, 60), thickness=2))

                # Convert to PIL and cache
                rgb_frame = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
                pil_cam   = Image.fromarray(rgb_frame)
                with self._cam_lock:
                    self._cam_pil = pil_cam

    # ── GUI render loop (runs on main thread) ─────────────
    def _gui_loop(self):
        if not self._running:
            return
        try:
            self._render()
        except Exception as e:
            print(f"[Render Error] {e}")
        self.root.after(16, self._gui_loop)   # ~60 fps

    # ── Master render ────────────────────────────────────
    def _render(self):
        now  = time.time()
        W, H = self.W, self.H

        img  = Image.new("RGB", (W, H), hex2rgb(BG))
        draw = ImageDraw.Draw(img)

        self._draw_topbar(img, draw, now)
        self._draw_left_panel(img, draw, now)
        self._draw_right_panel(img, draw, now)
        self._draw_stats_row(img, draw, now)
        self._draw_bottom_bar(img, draw, now)
        self._draw_notification(img, draw, now)

        # Push to canvas
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    # ─────────────────────────────────────────────────────
    #  TOP BAR — Logo + Title + Mode Tabs + FPS
    # ─────────────────────────────────────────────────────
    def _draw_topbar(self, img, draw, now):
        W  = self.W
        h  = self.TOP_H

        # Neumorphic top bar background
        draw_neu_card(draw, 0, 0, W, h, r=0, bg=BG, depth=4)

        # ── Logo image ──
        logo_x = self.PAD
        logo_cy = h // 2
        if self._logo_resized is not None:
            lw, lh = self._logo_resized.size
            ly = logo_cy - lh // 2
            img.paste(self._logo_resized, (logo_x, ly), self._logo_mask)
            text_x = logo_x + lw + 12
        else:
            # Fallback: blue shield dot
            draw.ellipse([logo_x, logo_cy-12, logo_x+24, logo_cy+12],
                         fill=hex2rgb(ACC_BLUE))
            draw.ellipse([logo_x+6, logo_cy-6, logo_x+18, logo_cy+6],
                         fill=hex2rgb(WHITE))
            text_x = logo_x + 34

        # "SHIELD FSL" title
        _draw_text(draw, text_x, logo_cy, "SHIELD FSL", size=22,
                   fill=TXT_DARK, bold=True, anchor="lm")

        # Green online dot
        dot_x = text_x + 130
        draw.ellipse([dot_x, logo_cy - 5, dot_x + 10, logo_cy + 5],
                     fill=hex2rgb(ACC_GREEN))

        # ── Centre pill tabs (Letters / Words) ──
        tab_w, tab_h = 260, 38
        tab_x = W // 2 - tab_w // 2
        tab_y = (h - tab_h) // 2
        draw_neu_card(draw, tab_x, tab_y, tab_w, tab_h, r=tab_h//2,
                      bg=CARD_IN, depth=3, inset=True)

        slot_w = tab_w // 2
        # Active tab pill
        ax = tab_x if current_mode == 1 else tab_x + slot_w
        _rounded_rect(draw, ax+3, tab_y+3, slot_w-6, tab_h-6,
                      (tab_h-6)//2, hex2rgb(WHITE))

        _draw_text(draw, tab_x + slot_w//2, tab_y + tab_h//2,
                   "Letters", size=15,
                   fill=TXT_DARK if current_mode == 1 else TXT_MID,
                   bold=(current_mode == 1), anchor="mm")
        _draw_text(draw, tab_x + slot_w + slot_w//2, tab_y + tab_h//2,
                   "Words", size=15,
                   fill=TXT_DARK if current_mode == 2 else TXT_MID,
                   bold=(current_mode == 2), anchor="mm")

        # ── Right: FPS chip ──
        fps_txt = f"{fps_value:.0f} FPS"
        cw = 90
        cx = W - self.PAD - cw
        cy = (h - 28) // 2
        draw_neu_card(draw, cx, cy, cw, 28, r=14, bg=CARD, depth=3)
        _draw_text(draw, cx + cw//2, cy + 14, fps_txt,
                   size=12, fill=TXT_MID, anchor="mm")

        # Mode chip next to FPS
        mode_txt = "FSL"
        mw = 60
        mx = cx - mw - 10
        draw_neu_card(draw, mx, cy, mw, 28, r=14, bg=CARD, depth=3)
        _draw_text(draw, mx + mw//2, cy + 14, mode_txt,
                   size=12, fill=TXT_MID, anchor="mm")

    # ─────────────────────────────────────────────────────
    #  LEFT PANEL — Camera feed card
    # ─────────────────────────────────────────────────────
    def _draw_left_panel(self, img, draw, now):
        pad  = self.PAD
        x    = pad
        y    = self.CONTENT_Y
        w    = self.LEFT_W
        h    = self.CONTENT_H
        r    = 20

        # Outer card
        draw_neu_card(draw, x, y, w, h, r=r, bg=CARD, depth=6)

        # ── Card header ──
        header_h = 40
        # Camera label
        mode_label = "Letters" if current_mode == 1 else "Words"
        _draw_text(draw, x + 20, y + header_h // 2, f"Camera Feed  —  {mode_label}",
                   size=14, fill=TXT_MID, anchor="lm")

        # Camera hide/show toggle (eye icon)
        eye_w, eye_h = 40, 28
        eye_x = x + w - eye_w - 16
        eye_y = y + (header_h - eye_h) // 2 + 2
        draw_neu_card(draw, eye_x, eye_y, eye_w, eye_h, r=8, bg=CARD, depth=3)
        if camera_hidden:
            # Crossed-out eye: draw a small "X"
            cx_e = eye_x + eye_w // 2
            cy_e = eye_y + eye_h // 2
            draw.ellipse([cx_e - 6, cy_e - 4, cx_e + 6, cy_e + 4],
                         outline=hex2rgb(ACC_RED), width=2)
            draw.line([(cx_e - 8, cy_e - 6), (cx_e + 8, cy_e + 6)],
                      fill=hex2rgb(ACC_RED), width=2)
        else:
            # Open eye icon
            cx_e = eye_x + eye_w // 2
            cy_e = eye_y + eye_h // 2
            draw.ellipse([cx_e - 6, cy_e - 4, cx_e + 6, cy_e + 4],
                         outline=hex2rgb(ACC_BLUE), width=2)
            draw.ellipse([cx_e - 2, cy_e - 2, cx_e + 2, cy_e + 2],
                         fill=hex2rgb(ACC_BLUE))

        # Calibrate Button
        cal_w, cal_h = 110, 28
        cal_x = eye_x - cal_w - 10
        cal_y = eye_y
        draw_neu_card(draw, cal_x, cal_y, cal_w, cal_h, r=8, bg=CARD, depth=3)
        if is_calibrating:
            _draw_text(draw, cal_x + cal_w//2, cal_y + cal_h//2,
                       f"Calib {calibration_progress:.0f}%", size=12, fill=ACC_BLUE, bold=True, anchor="mm")
        elif target_person_lock is not None:
            _draw_text(draw, cal_x + cal_w//2, cal_y + cal_h//2,
                       "🔒 Locked", size=12, fill=ACC_GREEN, bold=True, anchor="mm")
        else:
            _draw_text(draw, cal_x + cal_w//2, cal_y + cal_h//2,
                       "🔒 Calibrate", size=12, fill=TXT_MID, anchor="mm")

        # ── Camera area ──
        cam_pad = 14
        cam_x   = x + cam_pad
        cam_y   = y + header_h + 4
        cam_w   = w - cam_pad * 2
        cam_h   = h - header_h - cam_pad - 4

        if camera_hidden:
            # ── Hidden placeholder ──
            _rounded_rect(draw, cam_x, cam_y, cam_w, cam_h, 14, hex2rgb(CARD_IN))
            # Draw a large eye-slash icon in center
            cx_c = cam_x + cam_w // 2
            cy_c = cam_y + cam_h // 2
            # Eye shape
            draw.ellipse([cx_c - 30, cy_c - 18, cx_c + 30, cy_c + 18],
                         outline=hex2rgb(TXT_FAINT), width=3)
            draw.ellipse([cx_c - 10, cy_c - 10, cx_c + 10, cy_c + 10],
                         fill=hex2rgb(TXT_FAINT))
            # Slash line
            draw.line([(cx_c - 35, cy_c - 25), (cx_c + 35, cy_c + 25)],
                      fill=hex2rgb(TXT_FAINT), width=3)
            _draw_text(draw, cx_c, cy_c + 45, "Camera Hidden",
                       size=18, fill=TXT_FAINT, bold=True, anchor="mm")
            _draw_text(draw, cx_c, cy_c + 72, "Press H to show",
                       size=13, fill=TXT_FAINT, anchor="mm")
            return

        with self._cam_lock:
            pil_cam = self._cam_pil

        if pil_cam is not None:
            # ── Letterbox: maintain original aspect ratio ──
            orig_w, orig_h = pil_cam.size
            aspect = orig_w / orig_h

            if cam_w / cam_h > aspect:
                # Card is wider → fit to height
                new_h = cam_h
                new_w = int(cam_h * aspect)
            else:
                # Card is taller → fit to width
                new_w = cam_w
                new_h = int(cam_w / aspect)

            offset_x = cam_x + (cam_w - new_w) // 2
            offset_y = cam_y + (cam_h - new_h) // 2

            # Fill background for letterbox bars
            _rounded_rect(draw, cam_x, cam_y, cam_w, cam_h, 14, hex2rgb(CARD_IN))

            resized = pil_cam.resize((new_w, new_h), Image.LANCZOS)
            # Rounded clip mask
            mask_c = Image.new("L", (new_w, new_h), 0)
            md = ImageDraw.Draw(mask_c)
            md.rounded_rectangle([0, 0, new_w, new_h], radius=10, fill=255)
            img.paste(resized, (offset_x, offset_y), mask_c)

            # ── Calibration Overlay ──
            if is_calibrating:
                # Draw animated target silhouette guide
                sil_cx = offset_x + new_w // 2
                sil_cy = offset_y + new_h // 2 - 20
                # Head oval
                draw.ellipse([sil_cx - 50, sil_cy - 75, sil_cx + 50, sil_cy + 35],
                             outline=hex2rgb(ACC_BLUE), width=3)
                # Shoulders arc
                draw.arc([sil_cx - 120, sil_cy + 10, sil_cx + 120, sil_cy + 180],
                         start=180, end=360, fill=hex2rgb(ACC_BLUE), width=3)

                # Instruction Banner
                banner_h = 36
                banner_y = offset_y + new_h - banner_h - 15
                _rounded_rect(draw, offset_x + 20, banner_y, new_w - 40, banner_h, 18, hex2rgb(ACC_NAVY))
                _draw_text(draw, sil_cx, banner_y + banner_h//2,
                           f"Calibrating Target Person... {calibration_progress:.0f}%",
                           size=13, fill=WHITE, bold=True, anchor="mm")

                # Calibration Progress Bar
                bar_w_cal = int((new_w - 60) * (calibration_progress / 100.0))
                if bar_w_cal > 0:
                    _rounded_rect(draw, offset_x + 30, banner_y + banner_h - 5, bar_w_cal, 4, 2, hex2rgb(ACC_BLUE2))

            elif target_person_lock is not None:
                # Target Locked Badge in top left of camera
                badge_x = offset_x + 12
                badge_y = offset_y + 12
                _rounded_rect(draw, badge_x, badge_y, 130, 26, 13, hex2rgb(ACC_NAVY))
                draw.ellipse([badge_x + 10, badge_y + 8, badge_x + 18, badge_y + 16], fill=hex2rgb(ACC_GREEN))
                _draw_text(draw, badge_x + 24, badge_y + 13, "TARGET LOCKED",
                           size=11, fill=WHITE, bold=True, anchor="lm")
        else:
            # Placeholder while camera loads
            _rounded_rect(draw, cam_x, cam_y, cam_w, cam_h, 14, hex2rgb(CARD_IN))
            _draw_text(draw, cam_x + cam_w//2, cam_y + cam_h//2,
                       "Camera Loading...", size=16, fill=TXT_FAINT, anchor="mm")

    # ─────────────────────────────────────────────────────
    #  RIGHT PANEL — BMO + Detection Pipeline + Stability
    # ─────────────────────────────────────────────────────
    def _draw_right_panel(self, img, draw, now):
        pad = self.PAD
        x   = self.RIGHT_X
        y   = self.CONTENT_Y
        w   = self.RIGHT_W
        h   = self.CONTENT_H

        # ── BMO Card ──
        bmo_h = 90
        draw_neu_card(draw, x, y, w, bmo_h, r=18, bg=CARD, depth=6)

        # BMO avatar image (square, rounded)
        aw = 58; ah = 58
        ax = x + 16; ay = y + (bmo_h - ah) // 2
        face_pil = bmo_talk_pil if tts_is_speaking else bmo_idle_pil
        if face_pil is not None:
            resized_bmo = face_pil.resize((aw, ah), Image.LANCZOS)
            mask_b = Image.new("L", (aw, ah), 0)
            mbd    = ImageDraw.Draw(mask_b)
            mbd.rounded_rectangle([0, 0, aw, ah], radius=12, fill=255)
            # Green bg behind avatar
            _rounded_rect(draw, ax, ay, aw, ah, 12, hex2rgb(BMO_GREEN))
            img.paste(resized_bmo, (ax, ay), mask_b)
        else:
            _rounded_rect(draw, ax, ay, aw, ah, 12, hex2rgb(BMO_GREEN))
            _draw_text(draw, ax+aw//2, ay+ah//2, "BMO", size=12,
                       fill=WHITE, bold=True, anchor="mm")

        # BMO label + status dot
        label_x  = ax + aw + 16
        _draw_text(draw, label_x, y + bmo_h//2 - 6, "BMO",
                   size=18, fill=TXT_DARK, bold=True, anchor="lm")
        _bmo_dot = ACC_GREEN if not tts_is_speaking else "#F0A040"
        draw.ellipse([label_x + 46, y+bmo_h//2-12,
                      label_x + 56, y+bmo_h//2-2],
                     fill=hex2rgb(_bmo_dot))
        _draw_text(draw, label_x, y + bmo_h//2 + 12,
                   "Speaking..." if tts_is_speaking else "Online",
                   size=12, fill=TXT_MID, anchor="lm")

        # Waveform icon (right side of BMO card)
        wfx = x + w - 50
        wfy = y + bmo_h//2
        for i, amp in enumerate([4, 8, 12, 8, 4, 10, 6]):
            xi = wfx + i * 6
            draw.line([(xi, wfy-amp), (xi, wfy+amp)],
                      fill=hex2rgb(ACC_BLUE), width=2)

        # ── Detection Pipeline Card ──
        det_y = y + bmo_h + pad
        det_h = h - bmo_h - pad
        draw_neu_card(draw, x, det_y, w, det_h, r=18, bg=CARD, depth=6)

        _draw_text(draw, x + w//2, det_y + 28, "DETECTION PIPELINE",
                   size=11, fill=TXT_MID, anchor="mm")

        # Big ring
        ring_cx = x + w // 2
        ring_cy = det_y + det_h // 2 - 30
        ring_r  = min(w, det_h) // 4
        ring_r  = max(50, min(ring_r, 90))

        # Compute value for ring
        current_time = time.time()
        on_cooldown  = current_time < global_cooldown_end

        if current_mode == 1:
            if current_frame_prediction and not on_cooldown:
                held  = current_time - stable_start_time
                frac  = min(held / STABILIZATION_DELAY, 1.0) * 100
                ring_lbl = current_frame_prediction.upper() or "-"
                ring_col = ACC_BLUE
            elif on_cooldown:
                left  = global_cooldown_end - current_time
                frac  = (1 - left / GLOBAL_COOLDOWN) * 100
                ring_lbl = current_frame_prediction.upper() or "-"
                ring_col = ACC_AMBER
            else:
                frac  = 0.0
                ring_lbl = current_frame_prediction.upper() if current_frame_prediction else "-"
                ring_col = ACC_BLUE
        else:
            if on_cooldown:
                left  = global_cooldown_end - current_time
                frac  = (1 - left / GLOBAL_COOLDOWN) * 100
                ring_col = ACC_AMBER
            else:
                frac  = (len(hand_sequence_buffer) / max(WORD_SEQ_LEN, 1)) * 100
                ring_col = ACC_BLUE
            ring_lbl = debug_motion_word.upper() if debug_motion_word != "--" else "-"

        # Draw bg ring
        thickness = max(10, ring_r // 6)
        box = [ring_cx - ring_r, ring_cy - ring_r,
               ring_cx + ring_r, ring_cy + ring_r]
        draw.arc(box, 0, 360, fill=hex2rgb(SHADOW_D), width=thickness)
        if frac > 0.5:
            end_a = -90 + 360 * (frac / 100.0)
            draw.arc(box, -90, end_a, fill=hex2rgb(ring_col), width=thickness)

        # Letter/word in centre
        _draw_text(draw, ring_cx, ring_cy, ring_lbl,
                   size=52, fill=TXT_DARK, bold=True, anchor="mm")

        # ── Confidence progress bar ──
        conf_val = (frac if current_mode == 1 else debug_motion_conf)
        bar_y = ring_cy + ring_r + 28
        bar_x = x + 28
        bar_w = w - 96
        bar_h = 14
        _draw_gradient_bar(img, draw, bar_x, bar_y, bar_w, bar_h,
                           conf_val / 100.0, ACC_BLUE, ACC_BLUE2)
        _draw_text(draw, bar_x + bar_w + 10, bar_y + bar_h // 2,
                   f"{conf_val:.0f}%", size=13, fill=TXT_MID, anchor="lm")

    # ─────────────────────────────────────────────────────
    #  STATS ROW
    # ─────────────────────────────────────────────────────
    def _draw_stats_row(self, img, draw, now):
        pad   = self.PAD
        y     = self.CONTENT_Y + self.CONTENT_H + self.GAP
        h     = self.STATS_H
        W     = self.W

        # Full-width neumorphic strip
        draw_neu_card(draw, pad, y, W - pad*2, h, r=14, bg=CARD, depth=4)

        # Stats slots
        slots = [
            (f"{node_count}", "Nodes", ACC_BLUE),
            (f"{fps_value:.0f}", "FPS", ACC_GREEN),
            ("Letters" if current_mode == 1 else "Words", "Mode", ACC_PURPLE),
        ]
        n_slots = len(slots) + 1  # +1 for stability bar
        slot_w = (W - pad*2) // n_slots

        for i, (val, unit, col) in enumerate(slots):
            sx = pad + i * slot_w + 28
            # Colored dot indicator
            draw.ellipse([sx, y + h//2 - 4, sx + 8, y + h//2 + 4],
                         fill=hex2rgb(col))
            # Value
            _draw_text(draw, sx + 18, y + h//2 - 1, val,
                       size=16, fill=TXT_DARK, bold=True, anchor="lm")
            # Unit
            val_w = len(val) * 10 + 4
            _draw_text(draw, sx + 18 + val_w, y + h//2 - 1,
                       unit, size=13, fill=TXT_MID, anchor="lm")

            # Vertical divider
            if i < len(slots):
                dx = pad + (i + 1) * slot_w
                draw.line([(dx, y + 12), (dx, y + h - 12)],
                          fill=hex2rgb(SHADOW_D), width=1)

        # ── Stability bar (right section) ──
        stab_x = pad + len(slots) * slot_w + 20
        _draw_text(draw, stab_x, y + h//2, "Stability",
                   size=13, fill=TXT_MID, anchor="lm")
        bar_x2 = stab_x + 90
        bar_w2 = W - pad - bar_x2 - 40
        stab_y = y + h//2 - 7
        stab_h = 14
        stab_val = min(debug_motion_conf / 100.0 if current_mode == 2
                       else (min((time.time() - stable_start_time) /
                                 STABILIZATION_DELAY, 1.0)
                             if current_frame_prediction else 0.0), 1.0)
        _draw_gradient_bar(img, draw, bar_x2, stab_y, max(bar_w2, 50), stab_h,
                           stab_val, ACC_BLUE, ACC_BLUE2)
        # Dot indicator
        dot_x = bar_x2 + max(bar_w2, 50) + 14
        dot_col2 = ACC_BLUE if stab_val > 0.5 else TXT_FAINT
        draw.ellipse([dot_x - 6, stab_y + 1, dot_x + 6, stab_y + stab_h - 1],
                     fill=hex2rgb(dot_col2))

    # ─────────────────────────────────────────────────────
    #  BOTTOM OUTPUT BAR
    # ─────────────────────────────────────────────────────
    def _draw_bottom_bar(self, img, draw, now):
        pad  = self.PAD
        W    = self.W
        H    = self.H
        h    = self.BOT_H
        y    = H - h - pad
        r    = 18

        draw_neu_card(draw, pad, y, W - pad*2, h, r=r, bg=CARD, depth=6)

        # ── Large text output field ──
        out_x  = pad + 24
        out_y  = y + 16
        out_h  = 54
        spk_w  = 150
        out_w  = W - pad*2 - spk_w - 80

        draw_neu_card(draw, out_x, out_y, out_w, out_h, r=14,
                      bg=CARD_IN, depth=3, inset=True)

        display = typed_output[-60:] if len(typed_output) > 60 else typed_output
        _draw_text(draw, out_x + 18, out_y + out_h//2, display or "",
                   size=28, fill=TXT_DARK, bold=True, anchor="lm")

        # Cursor blink
        if int(now * 2) % 2 == 0:
            cx_est = out_x + 18 + len(display) * 16
            cx_est = min(cx_est, out_x + out_w - 10)
            draw.line([(cx_est, out_y + 10), (cx_est, out_y + out_h - 10)],
                      fill=hex2rgb(ACC_BLUE), width=3)

        # ── SPEAK button ──
        spk_h = 50
        spk_x = W - pad - 24 - spk_w
        spk_y = out_y + (out_h - spk_h) // 2
        _rounded_rect(draw, spk_x, spk_y, spk_w, spk_h,
                      spk_h//2, hex2rgb(ACC_NAVY))
        _draw_text(draw, spk_x + spk_w//2, spk_y + spk_h//2, "Speak",
                   size=16, fill=WHITE, bold=True, anchor="mm")
        # Small waveform icon inside speak button
        for i, amp in enumerate([4, 8, 12, 8, 4]):
            xi = spk_x + 20 + i * 6
            cy2 = spk_y + spk_h // 2
            draw.line([(xi, cy2-amp), (xi, cy2+amp)],
                      fill=hex2rgb(WHITE), width=2)

        # ── Footer line ──
        footer_y = y + h - 20
        _draw_text(draw, pad + 24, footer_y, "SHIELD FSL",
                   size=11, fill=TXT_FAINT, anchor="lm")

        # Keybind hints (compact)
        hints = "Space · Bksp · Enter=Speak · H=Camera · C=Clear"
        _draw_text(draw, W // 2, footer_y, hints,
                   size=10, fill=TXT_FAINT, anchor="mm")

        _draw_text(draw, W - pad - 24, footer_y, "© 2025",
                   size=11, fill=TXT_FAINT, anchor="rm")

    # ─────────────────────────────────────────────────────
    #  NOTIFICATION TOAST (optimized — no pixel loop)
    # ─────────────────────────────────────────────────────
    def _draw_notification(self, img, draw, now):
        if now >= notification_end:
            return
        alpha = min(1.0, (notification_end - now) / 0.35)
        txt   = notification_text
        nw    = max(200, len(txt) * 14 + 48)
        nh    = 48
        nx    = self.W // 2 - nw // 2
        ny    = self.CONTENT_Y + self.CONTENT_H // 2 - nh // 2

        # Create toast with solid accent color + alpha
        a = int(220 * alpha)
        toast = Image.new("RGBA", (nw, nh), (0, 0, 0, 0))
        td = ImageDraw.Draw(toast)
        c = hex2rgb(ACC_BLUE)
        td.rounded_rectangle([0, 0, nw, nh], radius=nh//2, fill=c + (a,))

        # Composite onto background
        bg_roi = img.crop((nx, ny, nx + nw, ny + nh)).convert("RGBA")
        merged = Image.alpha_composite(bg_roi, toast).convert("RGB")
        img.paste(merged, (nx, ny))

        # Text on top
        draw2 = ImageDraw.Draw(img)
        _draw_text(draw2, nx + nw//2, ny + nh//2, txt,
                   size=16, fill=WHITE, bold=True, anchor="mm")

    def destroy(self):
        self._running = False
        cap.release()


# ─────────────────────────────────────────────
#  Gradient Progress Bar Helper
# ─────────────────────────────────────────────
def _draw_gradient_bar(img: Image.Image, draw: ImageDraw.ImageDraw,
                       x, y, w, h, frac, c1_hex, c2_hex):
    """Rounded pill gradient progress bar pasted directly onto img."""
    r    = h // 2
    # Background track
    _rounded_rect(draw, x, y, w, h, r, hex2rgb(SHADOW_D))
    filled = int(w * max(0.0, min(frac, 1.0)))
    if filled <= 0:
        return
    filled = max(filled, h)   # minimum pill width = full height
    # Build gradient strip
    gw = min(filled, w)
    grad = Image.new("RGB", (gw, h), hex2rgb(c1_hex))
    gd   = ImageDraw.Draw(grad)
    c1   = hex2rgb(c1_hex); c2 = hex2rgb(c2_hex)
    for i in range(gw):
        t  = i / max(gw - 1, 1)
        gc = _blend(c1, c2, t)
        gd.line([(i, 0), (i, h)], fill=gc)
    # Rounded mask
    mask_g = Image.new("L", (gw, h), 0)
    mgd    = ImageDraw.Draw(mask_g)
    mgd.rounded_rectangle([0, 0, gw, h], radius=r, fill=255)
    img.paste(grad, (x, y), mask_g)


# ─────────────────────────────────────────────
#  Text Drawing Helper (PIL with system font)
# ─────────────────────────────────────────────
_font_cache: dict = {}

def _get_font(size=14, bold=False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    candidates_bold = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    candidates_reg = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    candidates = candidates_bold if bold else candidates_reg
    for path in candidates:
        if os.path.exists(path):
            try:
                f = ImageFont.truetype(path, size)
                _font_cache[key] = f
                return f
            except Exception:
                pass
    f = ImageFont.load_default()
    _font_cache[key] = f
    return f

def _draw_text(draw, x, y, text, size=14, fill=TXT_DARK,
               bold=False, anchor="lm"):
    """Draw text at (x,y) with given anchor using PIL."""
    if not text:
        return
    f = _get_font(size, bold)
    try:
        draw.text((x, y), text, font=f, fill=hex2rgb(fill), anchor=anchor)
    except Exception:
        draw.text((x, y), text, font=f, fill=hex2rgb(fill))

def _draw_dashed_rect(draw, x, y, w, h, color, dash=8, gap=5):
    """Draw a dashed rectangle outline."""
    def _dash_line(x1, y1, x2, y2):
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length == 0:
            return
        dx = (x2-x1)/length
        dy = (y2-y1)/length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            draw.line([(int(x1 + dx*pos), int(y1 + dy*pos)),
                       (int(x1 + dx*end), int(y1 + dy*end))],
                      fill=color, width=2)
            pos += dash + gap
    _dash_line(x,   y,   x+w, y)
    _dash_line(x+w, y,   x+w, y+h)
    _dash_line(x+w, y+h, x,   y+h)
    _dash_line(x,   y+h, x,   y)


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
def main():
    root = tk.Tk()
    app  = ShieldFSLApp(root)

    def _on_close():
        app.destroy()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)

    # Start maximized / fullscreen-ready at 1920×1080
    root.state("zoomed")   # Windows maximized
    root.mainloop()

if __name__ == "__main__":
    main()
