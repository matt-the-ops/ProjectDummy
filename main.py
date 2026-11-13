"""
SHIELD FSL — Neumorphism Tkinter GUI (Revamped)
Single 1920×1080 window, matching the SHIELD FSL reference design.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import ctypes
try:
    if sys.platform.startswith("win"):
        ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass
import datetime
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
from sentence_translator import translate_sentence
from coop_bus import coop_hub
from non_fsl_window import NonFSLWindow

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
#  Modern Colour Palette (Project B.M.O)
# ─────────────────────────────────────────────
# Tkinter hex colours & Neumorphic Palette (Modernized Sky Blue)
BG_LIGHT      = "#D2E8FE"   # Luminous fresh sky blue canvas
BG_CENTER     = "#E4F3FE"   # Soft center ambient highlight
BG_DEEP       = "#C0DEFD"   # Rich pastel blue

CARD_BG       = "#FFFFFF"   # Luminous frosted glass card body
CARD_BORDER   = "#FFFFFF"   # Specular white rim highlight
CARD_RIM_SUB  = "#D2E4F8"   # Subtle bottom rim line
CARD_SHADOW   = "#7BAADF"   # Rich sky-blue diffused drop shadow (NOT grey!)

BLUE_PRIMARY  = "#1D68FE"   # Vibrant electric cobalt blue (matches Photoshop title & buttons)
BLUE_LIGHT    = "#4F9BFE"   # Bright electric azure
BLUE_CYAN     = "#38BDF8"   # Sky cyan highlight
BLUE_DARK     = "#1050E2"   # Deep active cobalt

TEXT_MAIN     = "#0F172A"   # Crisp high-contrast obsidian slate
TEXT_MUTED    = "#5A718F"   # Cool slate gray for subtitles/labels
TEXT_FAINT    = "#8FA8C4"   # Refined placeholder text
WHITE         = "#FFFFFF"

GREEN_ONLINE  = "#10B981"   # Mint green status dot
GREEN_BG      = "#ECFDF5"   # Mint pill background
GREEN_BORDER  = "#A7F3D0"   # Mint pill border
GREEN_TEXT    = "#059669"   # Mint pill text

DOCK_DARK_BG  = (22, 36, 60, 185)  # Dark translucent glass overlay on video

# Legacy aliases for compatibility
BG_TOP      = BG_LIGHT
BG_BOT      = BG_DEEP
CARD        = CARD_BG
CARD_DARK   = "#0B132B"
SHADOW      = CARD_SHADOW
SHADOW_D    = "#DCEBF9"
ICON_BG     = "#E6F1FC"
ICON_BG_ON  = "#DBEAFE"
TXT_DARK    = TEXT_MAIN
TXT_MID     = TEXT_MUTED
TXT_FAINT   = TEXT_FAINT
ACC_BLUE    = BLUE_PRIMARY
ACC_BLUE2   = BLUE_LIGHT
ACC_CYAN    = BLUE_CYAN
ACC_NAVY    = "#16243C"
ACC_GREEN   = GREEN_ONLINE
ACC_RED     = "#EF4444"
ACC_AMBER   = "#F59E0B"
ACC_PURPLE  = "#8B5CF6"
BMO_GREEN   = "#22C55E"


# ─────────────────────────────────────────────
#  PIL Drawing Helpers
# ─────────────────────────────────────────────
def hex2rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _blend(c1, c2, t):
    return tuple(int(c1[i]*(1-t) + c2[i]*t) for i in range(3))

def rgb2hex(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

def draw_glass_card(img: Image.Image, x, y, w, h, r=26,
                    bg_color=CARD_BG, shadow_color=CARD_SHADOW,
                    border_color=CARD_BORDER, shadow_offset=(0, 9),
                    shadow_blur=20, shadow_alpha=115):
    """
    Renders an authentic neumorphic-glassmorphic card:
    - Multi-layered soft diffused ambient sky-blue shadow (floating depth)
    - Crisp specular glass white rim highlight
    - Clean luminous white-to-ice gradient card fill
    """
    pad = shadow_blur * 2
    sw = w + pad * 2
    sh = h + pad * 2
    
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
    
    bg_top = (255, 255, 255)
    bg_bot = (244, 250, 255)
    for row in range(h):
        t = row / max(h - 1, 1)
        c = _blend(bg_top, bg_bot, t)
        cdraw.line([(0, row), (w, row)], fill=c + (252,))
        
    mask = Image.new("L", (w, h), 0)
    m_draw = ImageDraw.Draw(mask)
    m_draw.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    
    # Crisp specular white rim highlight
    b_col = hex2rgb(border_color)
    cdraw.rounded_rectangle([0, 0, w-1, h-1], radius=r, outline=b_col + (255,), width=2)
    
    img.paste(card_img, (x, y), mask)

def draw_shadow_card(draw: ImageDraw.ImageDraw, x, y, w, h, r=18,
                     bg=CARD, shadow_color=SHADOW, blur=8, offset_y=4):
    """Draw a card with a clean, smooth multi-pass diffused drop shadow."""
    bgc = hex2rgb(bg)
    if shadow_color:
        sc = hex2rgb(shadow_color)
        soft_outer = _blend(sc, hex2rgb(BG_BOT), 0.55)
        _rounded_rect(draw, x - 1, y + offset_y + 2, w + 2, h + 2, r + 2, soft_outer)
        _rounded_rect(draw, x, y + offset_y, w, h, r, sc)
    _rounded_rect(draw, x, y, w, h, r, bgc)

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

def draw_pill(draw, x, y, w, h, fill):
    r = h // 2
    _rounded_rect(draw, x, y, w, h, r, hex2rgb(fill) if isinstance(fill, str) else fill)

def draw_pill_outline(draw, x, y, w, h, outline, width=2):
    r = h // 2
    draw.rounded_rectangle([x, y, x+w, y+h], radius=r,
                            outline=hex2rgb(outline) if isinstance(outline, str) else outline, width=width)

_gradient_pill_cache = {}

def draw_gradient_pill(img, x, y, w, h, c1_hex, c2_hex, outline_col=None):
    if w <= 0 or h <= 0:
        return
    cache_key = (w, h, c1_hex, c2_hex, outline_col)
    cached = _gradient_pill_cache.get(cache_key)
    if cached is not None:
        img.paste(cached[0], (x, y), cached[1])
        return
    r = h // 2
    c1, c2 = hex2rgb(c1_hex), hex2rgb(c2_hex)
    # Vectorized gradient via numpy — replaces slow per-pixel Python loop
    t = np.linspace(0, 1, max(w, 1), dtype=np.float32)
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[:, :, 0] = (c1[0] * (1 - t) + c2[0] * t).astype(np.uint8)
    arr[:, :, 1] = (c1[1] * (1 - t) + c2[1] * t).astype(np.uint8)
    arr[:, :, 2] = (c1[2] * (1 - t) + c2[2] * t).astype(np.uint8)
    arr[:, :, 3] = 255
    grad = Image.fromarray(arr, "RGBA")
    if outline_col:
        gd = ImageDraw.Draw(grad)
        gd.rounded_rectangle([0, 0, w-1, h-1], radius=r, outline=hex2rgb(outline_col) + (200,), width=1)
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    _gradient_pill_cache[cache_key] = (grad.copy(), mask.copy())
    img.paste(grad, (x, y), mask)

def gradient_pill(draw, x, y, w, h, c1, c2):
    """Horizontal gradient pill."""
    r = h // 2
    grad = Image.new("RGB", (w, h))
    gd   = ImageDraw.Draw(grad)
    for i in range(w):
        t  = i / max(w-1, 1)
        gc = _blend(hex2rgb(c1), hex2rgb(c2), t)
        gd.line([(i, 0), (i, h)], fill=gc)
    mask = Image.new("L", (w, h), 0)
    md   = ImageDraw.Draw(mask)
    md.rounded_rectangle([0, 0, w, h], radius=r, fill=255)
    return grad, mask

def _get_tk_font(size=14, bold=False):
    weight = "bold" if bold else "normal"
    return ("Segoe UI", max(8, int(size)), weight)

def _round_rect_pts(x0, y0, x1, y1, r):
    r = min(max(1, r), abs(x1 - x0) // 2, abs(y1 - y0) // 2)
    return [
        x0 + r, y0,
        x1 - r, y0,
        x1, y0,
        x1, y0 + r,
        x1, y1 - r,
        x1, y1,
        x1 - r, y1,
        x0 + r, y1,
        x0, y1,
        x0, y1 - r,
        x0, y0 + r,
        x0, y0
    ]

def create_ambient_background(W, H):
    """Generates a rich, luminous modern sky-blue neumorphic ambient background."""
    bg_top = (210, 232, 254)   # #D2E8FE - luminous fresh sky blue
    bg_mid = (228, 243, 254)   # #E4F3FE - soft radiant center
    bg_bot = (192, 222, 253)   # #C0DEFD - rich pastel blue

    bg = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(bg)
    for y in range(H):
        t = y / max(H - 1, 1)
        if t < 0.35:
            c = _blend(bg_top, bg_mid, t / 0.35)
        else:
            c = _blend(bg_mid, bg_bot, (t - 0.35) / 0.65)
        draw.line([(0, y), (W, y)], fill=c + (255,))

    # Luminous ambient lighting blooms
    glows = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glows)
    # Top-right bright radiant bloom
    gdraw.ellipse([W - 680, -260, W + 360, 580], fill=(255, 255, 255, 140))
    # Top-left ambient soft cyan
    gdraw.ellipse([-220, -220, 680, 520], fill=(165, 220, 255, 110))
    # Center-left subtle sky blue
    gdraw.ellipse([-150, 400, 450, 900], fill=(180, 225, 255, 80))
    # Bottom-left rich blue wave
    gdraw.ellipse([-320, H - 560, 520, H + 320], fill=(145, 202, 253, 95))
    # Bottom-right cerulean
    gdraw.ellipse([W - 720, H - 460, W + 260, H + 260], fill=(155, 212, 254, 95))
    glows = glows.filter(ImageFilter.GaussianBlur(85))
    bg.alpha_composite(glows)

    # Soft 3D Neumorphic ambient wave curves (smooth layered depth)
    curves = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(curves)
    # Smooth curved raised surface top-right
    cdraw.ellipse([W - 760, -190, W + 320, 560], fill=(238, 248, 255, 90))
    # Smooth curved raised surface bottom-left
    cdraw.ellipse([-260, H - 410, 490, H + 230], fill=(232, 245, 255, 80))
    curves = curves.filter(ImageFilter.GaussianBlur(36))
    bg.alpha_composite(curves)

    return bg.convert("RGB")


# ─────────────────────────────────────────────
#  App State
# ─────────────────────────────────────────────
LETTER_BUFFER_SIZE   = 12
STABILIZATION_DELAY  = 2.5
GLOBAL_COOLDOWN      = 1.8
WORD_COOLDOWN        = 0.8     # Snappy real-time inter-word cooldown
WORD_DEBOUNCE_TIME   = 1.0     # Lowered for faster consecutive signing
TRANSLATE_ABSENCE_DELAY = 3.0  # Seconds with no hands on screen before auto-translating sentence
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
translated_output = ""
sentence_translated = False
hands_absent_start_time = None
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
#  High-Speed Threaded Camera Capture (30-60 FPS)
# ─────────────────────────────────────────────
class CameraCapture:
    """Threaded camera grabber that continuously fetches frames at 30-60 FPS without blocking ML/GUI loops."""
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.fps_tracker = collections.deque(maxlen=30)
        self.cam_fps = 0.0
        self.thread = None
        self.start()

    def _init_camera(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened() and backend != cv2.CAP_ANY:
                cap = cv2.VideoCapture(self.camera_index)

            if cap.isOpened():
                # Set MJPG & 640x480 at high framerate with 1-frame buffer for low latency
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 60)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            self.cap = cap

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()

    def _capture_loop(self):
        self._init_camera()
        while self.running:
            with self.lock:
                cap_ref = self.cap
            if cap_ref is None or not cap_ref.isOpened():
                time.sleep(0.04)
                continue
            ret, frame = cap_ref.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            now = time.time()
            self.fps_tracker.append(now)
            if len(self.fps_tracker) > 1:
                self.cam_fps = (len(self.fps_tracker) - 1) / max(self.fps_tracker[-1] - self.fps_tracker[0], 1e-4)

            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame.copy()
            return False, None

    def switch_camera(self, new_index):
        with self.lock:
            self.camera_index = new_index
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
        threading.Thread(target=self._init_camera, daemon=True).start()

    def release(self):
        self.running = False
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def stop(self):
        self.release()

cam_capture = CameraCapture(camera_index)
cap = cam_capture  # Seamless drop-in replacement

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

    # ── Layout design constants (1920×1080 baseline matching reference) ─────────────
    DESIGN_W   = 1920
    DESIGN_H   = 1080
    W          = 1920
    H          = 1080

    PAD        = 28          # outer padding
    TOP_Y      = 16
    TOP_H      = 92          # floating top bar height
    
    CONTENT_Y  = 124
    CONTENT_H  = 592
    LEFT_W     = 1084        # camera card width
    RIGHT_X    = 1132        # companion panel X
    RIGHT_W    = 760         # companion panel width
    
    METRICS_Y  = 732
    METRICS_H  = 122
    
    BOT_Y      = 870
    BOT_H      = 156

    # ────────────────────────────────────────────────────
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Project B.M.O")
        self.root.configure(bg=BG_TOP)

        # ── Responsive Resolution Auto-Fit ──
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        usable_w = screen_w - 40
        usable_h = screen_h - 80

        if self.DESIGN_W > usable_w or self.DESIGN_H > usable_h:
            init_scale = min(usable_w / self.DESIGN_W, usable_h / self.DESIGN_H)
            init_w = max(800, int(self.DESIGN_W * init_scale))
            init_h = max(450, int(self.DESIGN_H * init_scale))
        else:
            init_w = self.DESIGN_W
            init_h = self.DESIGN_H

        pos_x = max(0, (screen_w - init_w) // 2)
        pos_y = max(0, (screen_h - init_h) // 2)
        self.root.geometry(f"{init_w}x{init_h}+{pos_x}+{pos_y}")
        self.root.minsize(800, 450)
        self.root.resizable(True, True)
        self.fullscreen = False

        self.win_w = init_w
        self.win_h = init_h
        self.ui_scale = min(self.win_w / self.DESIGN_W, self.win_h / self.DESIGN_H)
        self.render_w = max(1, int(self.DESIGN_W * self.ui_scale))
        self.render_h = max(1, int(self.DESIGN_H * self.ui_scale))
        self.ui_ox = (self.win_w - self.render_w) // 2
        self.ui_oy = (self.win_h - self.render_h) // 2

        # ── Canvas (High-Speed Layered Surface @ 60+ FPS) ──
        self.canvas = tk.Canvas(root, width=init_w, height=init_h,
                                bg=BG_TOP, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # 1. Background image (full static UI shell)
        self.bg_tk = None
        self.bg_image_id = self.canvas.create_image(self.ui_ox, self.ui_oy, anchor=tk.NW)

        # 2. Camera viewport image
        self.cam_tk = None
        self.cam_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        # 3. Right panel dynamic content image
        self.right_tk = None
        self.right_image_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        # 4. Top bar dynamic items: Mode slider knob & labels + FPS chip
        self.slider_knob_id = self.canvas.create_polygon(0, 0, 0, 0, 0, 0, fill="#2563EB", outline="#60A5FA", width=1)
        self.lbl_letters_id = self.canvas.create_text(0, 0, text="Letters", font=("Segoe UI", 12, "bold"), fill=WHITE, anchor="center")
        self.lbl_words_id   = self.canvas.create_text(0, 0, text="Words", font=("Segoe UI", 12), fill="#5A718F", anchor="center")
        self.fps_txt_id     = self.canvas.create_text(0, 0, text="60 FPS", font=("Segoe UI", 13, "bold"), fill=TEXT_MAIN, anchor="w")

        # 5. Stats row items (4 metric values + RMS progress bar)
        self.stat_text_ids = [
            self.canvas.create_text(0, 0, text="0 Nodes", font=("Segoe UI", 20, "bold"), fill=TEXT_MAIN, anchor="w"),
            self.canvas.create_text(0, 0, text="0 Lux", font=("Segoe UI", 20, "bold"), fill=TEXT_MAIN, anchor="w"),
            self.canvas.create_text(0, 0, text="0 cm", font=("Segoe UI", 20, "bold"), fill=TEXT_MAIN, anchor="w"),
            self.canvas.create_text(0, 0, text="0.00 RMS", font=("Segoe UI", 20, "bold"), fill=TEXT_MAIN, anchor="w"),
        ]
        self.rms_bar_bg_id = self.canvas.create_rectangle(0, 0, 0, 0, fill="#DCEBF9", outline="")
        self.rms_bar_id    = self.canvas.create_rectangle(0, 0, 0, 0, fill="#3B82F6", outline="")

        # 6. Bottom bar dynamic items: typed sentence + cursor
        self.typed_text_id  = self.canvas.create_text(0, 0, text="", font=("Segoe UI", 22, "bold"), fill=TEXT_MAIN, anchor="w")
        self.cursor_line_id = self.canvas.create_line(0, 0, 0, 0, fill=BLUE_PRIMARY, width=3)

        # 7. Toast notification items
        self.toast_bg_id   = self.canvas.create_polygon(0, 0, 0, 0, fill=BLUE_PRIMARY, outline="", state="hidden")
        self.toast_text_id = self.canvas.create_text(0, 0, text="", font=("Segoe UI", 15, "bold"), fill=WHITE, anchor="center", state="hidden")

        # Hit-test rects for the 3 functional camera control buttons, keyed by name
        self._cam_btn_rects = {}

        # UI Shell Cache: Pre-render static elements
        self.ui_shell_cache = self._pre_render_ui_shell()
        self._update_layout_geometry()

        # Render caches for 60 FPS performance
        self._cached_cam_mask = None
        self._cached_cam_mask_key = None
        self._cached_spk_glow = None
        self._cached_send_glow = None
        self._cached_cam_dock = None
        # Camera HUD overlay cache (static badges, buttons — only redrawn on state change)
        self._cached_hud_overlay = None
        self._cached_hud_key = None

        # Right panel dirty-flag cache: avoid re-rendering unless content changed
        self._right_panel_dirty = True
        self._right_last_update = 0.0
        self._right_cache_key   = None   # (tts_is_speaking, received_msg_id, active_pred)
        self._RIGHT_PANEL_MIN_INTERVAL = 0.08  # update right panel at max 12 FPS

        # Speak and Send button interaction states
        self._spk_hover        = False   # True while cursor is over the button
        self._spk_click_until  = 0.0     # time.time() deadline for click-flash
        self._send_hover       = False   # True while cursor is over send button
        self._send_click_until = 0.0     # time.time() deadline for click-flash

        # Co-Op 2-Way Partner state
        self.partner_window   = None
        self.received_message = None     # {"tokens": [...], "text": "...", "icon_paths": [...], "previews": [...], "time": ...}
        coop_hub.register_fsl_listener(self._on_message_from_partner)

        # Cached camera ImageTk (updated by thread)
        self._cam_photo  = None
        self._cam_lock   = threading.Lock()
        self._cam_pil    = None  # latest PIL camera frame

        # Cached BMO ImageTk
        self._bmo_idle_tk = None
        self._bmo_talk_tk = None

        # ── Load Logo ──
        self._logo_rgba     = None
        self._logo_alpha    = None
        self._logo_rim      = None
        self._logo_rim_mask = None
        self._logo_shadow   = None
        self._logo_pad      = 20
        self._logo_w        = 50
        self._logo_h        = 50
        self._load_logo()

        # Key and Mouse bindings
        root.bind("<KeyPress>",  self._on_key)
        root.bind("<Button-1>",  self._on_click)
        root.bind("<Motion>",    self._on_motion)
        root.bind("<F11>",       lambda e: self._toggle_fullscreen())
        root.bind("<Escape>",    lambda e: self._exit_fullscreen())

        # FPS tracking
        self._frame_times = collections.deque(maxlen=30)
        self._last_draw   = time.time()

        # Start ML thread
        self._running = True
        self._ml_thread = threading.Thread(target=self._ml_loop, daemon=True)
        self._ml_thread.start()

        # Start GUI refresh loop (deferred so mainloop() starts first)
        self.root.after(50, self._gui_loop)

    # ── Co-Op 2-Way Message Handlers ───────────────────────
    def _on_message_from_partner(self, tokens, text, icon_paths):
        """Called by CoOp bus when partner sends a message from Non-FSL window."""
        self.root.after(0, lambda: self._handle_partner_message(tokens, text, icon_paths))

    def _handle_partner_message(self, tokens, text, icon_paths):
        """Processes incoming message, caches sign previews, and triggers alerts."""
        previews = []
        for token, p in icon_paths:
            if p and os.path.exists(p):
                try:
                    im = Image.open(p).convert("RGBA")
                    previews.append((token, im))
                except Exception as e:
                    print(f"[!] Error loading preview icon for {token}: {e}")

        self.received_message = {
            "tokens": list(tokens),
            "text": text,
            "icon_paths": list(icon_paths),
            "previews": previews,
            "time": time.time(),
        }
        play_ping_sound()
        set_notification(f"PARTNER: {text[:28]}")

    def _send_fsl_output(self):
        """Delivers current FSL sentence to Non-FSL partner window, speaks it, and clears."""
        global typed_output, translated_output, sentence_translated, hands_absent_start_time
        if not sentence_translated and typed_output.strip():
            translated_output = translate_sentence(typed_output)
            sentence_translated = True
        final_text = (translated_output or typed_output).strip()
        if final_text:
            coop_hub.send_from_fsl(final_text)
            speak_text(final_text)
            set_notification("SENT TO PARTNER")
            typed_output = ""
            translated_output = ""
            sentence_translated = False
            hands_absent_start_time = None

    def get_cam_frame(self):
        """Thread-safe copy of the latest live PIL camera frame."""
        with self._cam_lock:
            if self._cam_pil is not None:
                return self._cam_pil.copy()
        return None

    # ── Logo loader ──────────────────────────────────────
    def _load_logo(self):
        """Load SHIELD logo preserving both white and blue figures, with rim contour and soft drop shadow."""
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
            logo_resized = logo.resize((target_w, target_h), Image.BILINEAR)
            alpha = logo_resized.split()[3]

            # Soft rim contour so the white figure pops cleanly against light backgrounds
            # Wider MaxFilter gives a fuller outline; higher alpha ensures visibility on white bg
            rim_mask = alpha.filter(ImageFilter.MaxFilter(7))
            rim = Image.new("RGBA", (target_w, target_h), (60, 100, 180, 120))

            # Soft diffused drop shadow — lower alpha + wider blur = floating, non-intrusive shadow
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
            print("[OK] Logo loaded with drop shadow and rim contour.")
        except Exception as e:
            print(f"[!] Logo load error: {e}")


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
        self._update_layout_geometry()

    def _update_layout_geometry(self):
        """Update background scaling and positions of all canvas items when window resizes."""
        if not hasattr(self, 'canvas') or not hasattr(self, 'bg_image_id'):
            return
        s = self.ui_scale
        ox, oy = self.ui_ox, self.ui_oy
        W, H = self.W, self.H

        if self.ui_shell_cache is None:
            self.ui_shell_cache = self._pre_render_ui_shell()

        # High-quality scale background shell ONCE on resize
        scaled_shell = self.ui_shell_cache.resize((self.render_w, self.render_h), Image.Resampling.BILINEAR)
        self.bg_tk = ImageTk.PhotoImage(scaled_shell)
        self.canvas.itemconfig(self.bg_image_id, image=self.bg_tk)
        self.canvas.coords(self.bg_image_id, ox, oy)

        # Top Bar Items
        cy = self.TOP_Y + self.TOP_H // 2
        fps_w, fps_h = 120, 42
        fps_x = self.PAD + (W - self.PAD * 2) - fps_w - 28
        self.canvas.coords(self.fps_txt_id, ox + int((fps_x + 32) * s), oy + int(cy * s))
        self.canvas.itemconfig(self.fps_txt_id, font=_get_tk_font(int(13 * s), bold=True))

        switch_w, switch_h = 210, 48
        switch_x = W // 2 - switch_w // 2
        self.canvas.coords(self.lbl_letters_id, ox + int((switch_x + switch_w // 4) * s), oy + int(cy * s))
        self.canvas.coords(self.lbl_words_id, ox + int((switch_x + 3 * switch_w // 4) * s), oy + int(cy * s))
        self.canvas.itemconfig(self.lbl_letters_id, font=_get_tk_font(int(12 * s), bold=True))
        self.canvas.itemconfig(self.lbl_words_id, font=_get_tk_font(int(12 * s), bold=True))

        # Stats Row Items
        mx, my, mw, mh = self.PAD, self.METRICS_Y, W - self.PAD * 2, self.METRICS_H
        slot_w = mw // 4
        cy_met = my + mh // 2
        for i in range(4):
            sx = mx + i * slot_w + 32 + 58 + 18
            self.canvas.coords(self.stat_text_ids[i], ox + int(sx * s), oy + int((cy_met + 14) * s))
            self.canvas.itemconfig(self.stat_text_ids[i], font=_get_tk_font(int(20 * s), bold=True))

        # Bottom Bar Items
        bx_bot, by_bot, bw_bot, bh_bot = self.PAD, self.BOT_Y, W - self.PAD * 2, self.BOT_H
        row_y = by_bot + 62
        row_h = 60
        inp_x = bx_bot + 36 + 28
        inp_cy = row_y + row_h // 2
        self.canvas.coords(self.typed_text_id, ox + int(inp_x * s), oy + int(inp_cy * s))
        self.canvas.itemconfig(self.typed_text_id, font=_get_tk_font(int(22 * s), bold=True))

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
        global translated_output, sentence_translated, hands_absent_start_time

        k = event.keysym.lower()
        if k == 'q':
            self._running = False
            self.root.after(200, self.root.destroy)
        elif k == 'f11' or k == 'f':
            self._toggle_fullscreen()
        elif k == '1':
            current_mode = 1
            typed_output = ""
            translated_output = ""
            sentence_translated = False
            hands_absent_start_time = None
            motion_state = ""
            pinky_history.clear(); index_history.clear()
            prediction_buffer.clear()
            hand_sequence_buffer.clear()
            last_stable_char = ""
            set_notification("LETTERS MODE")
        elif k == '2':
            current_mode = 2
            typed_output = ""
            translated_output = ""
            sentence_translated = False
            hands_absent_start_time = None
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
            camera_index = (camera_index + 1) % 3
            cam_capture.switch_camera(camera_index)
            set_notification(f"CAMERA → {camera_index}")
        elif k == 'return':
            self._send_fsl_output()
        elif k == 't':
            if typed_output.strip():
                translated_output = translate_sentence(typed_output)
                sentence_translated = True
                set_notification(f"TRANSLATED: {translated_output}")
                speak_text(translated_output)
        elif k == 'space':
            typed_output += " "
            sentence_translated = False
        elif k in ('backspace', 'delete'):
            typed_output = typed_output[:-1]
            sentence_translated = False
            translated_output = ""
        elif k == 'c':
            typed_output = ""
            translated_output = ""
            sentence_translated = False
            hands_absent_start_time = None
            set_notification("OUTPUT CLEARED")

    # ── Mouse motion handler (hover tracking) ───────────
    def _on_motion(self, event):
        if not hasattr(self, 'ui_scale') or self.ui_scale <= 0:
            return
        mx = int((event.x - self.ui_ox) / self.ui_scale)
        my = int((event.y - self.ui_oy) / self.ui_scale)
        row_y = self.BOT_Y + 62
        row_h = 60
        btn_w = 175
        gap = 14
        send_x = self.PAD + (self.W - self.PAD * 2) - btn_w - 36
        spk_x = send_x - btn_w - gap
        self._spk_hover = (spk_x <= mx <= spk_x + btn_w and row_y <= my <= row_y + row_h)
        self._send_hover = (send_x <= mx <= send_x + btn_w and row_y <= my <= row_y + row_h)

    # ── Mouse click handler ──────────────────────────────
    def _on_click(self, event):
        global current_mode, typed_output, camera_hidden
        global pinky_history, index_history, prediction_buffer
        global hand_sequence_buffer, debug_motion_word, debug_face_word
        global debug_motion_conf, debug_face_conf, gate_status, last_stable_char
        global translated_output, sentence_translated, hands_absent_start_time

        if not hasattr(self, 'ui_scale') or self.ui_scale <= 0:
            return
        mx = int((event.x - self.ui_ox) / self.ui_scale)
        my = int((event.y - self.ui_oy) / self.ui_scale)

        # Mode slider capsule click
        switch_w, switch_h = 210, 48
        switch_x = self.W // 2 - switch_w // 2
        cy_top = self.TOP_Y + self.TOP_H // 2
        switch_y = cy_top - switch_h // 2
        if switch_y <= my <= switch_y + switch_h and switch_x <= mx <= switch_x + switch_w:
            if mx < switch_x + switch_w // 2 and current_mode != 1:
                current_mode = 1
                typed_output = ""
                translated_output = ""
                sentence_translated = False
                hands_absent_start_time = None
                pinky_history.clear(); index_history.clear()
                prediction_buffer.clear()
                hand_sequence_buffer.clear()
                last_stable_char = ""
                set_notification("LETTERS MODE")
            elif mx >= switch_x + switch_w // 2 and current_mode != 2:
                current_mode = 2
                typed_output = ""
                translated_output = ""
                sentence_translated = False
                hands_absent_start_time = None
                hand_sequence_buffer.clear()
                debug_motion_word = "--"; debug_face_word = "--"
                debug_motion_conf = 0.0;  debug_face_conf  = 0.0
                gate_status = "WAITING"
                set_notification("WORDS MODE")

        # Camera control row: switch camera / toggle visibility / calibrate.
        # Hit-tests the exact rects the buttons were last drawn at.
        global camera_index, cap
        for key, (bx0, by0, bx1, by1) in self._cam_btn_rects.items():
            if bx0 <= mx <= bx1 and by0 <= my <= by1:
                if key == "eye":
                    camera_hidden = not camera_hidden
                    set_notification("CAMERA HIDDEN" if camera_hidden else "CAMERA VISIBLE")
                elif key == "shield" or key == "lock":
                    trigger_calibration()
                elif key == "cam":
                    camera_index = (camera_index + 1) % 3
                    cam_capture.switch_camera(camera_index)
                    set_notification(f"CAMERA → {camera_index}")
                break

        # Action Buttons in bottom bar: SPEAK and SEND
        row_y = self.BOT_Y + 62
        row_h = 60
        btn_w = 175
        gap = 14
        send_x = self.PAD + (self.W - self.PAD * 2) - btn_w - 36
        spk_x = send_x - btn_w - gap

        # Speak button click
        if spk_x - 5 <= mx <= spk_x + btn_w + 5 and row_y - 5 <= my <= row_y + row_h + 5:
            self._spk_click_until = time.time() + 0.18   # 180 ms click-flash
            if not sentence_translated and typed_output.strip():
                translated_output = translate_sentence(typed_output)
                sentence_translated = True
            speak_text(translated_output or typed_output)

        # SEND button click
        elif send_x - 5 <= mx <= send_x + btn_w + 5 and row_y - 5 <= my <= row_y + row_h + 5:
            self._send_click_until = time.time() + 0.18  # 180 ms click-flash
            self._send_fsl_output()

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
        global lux_value, dist_cm
        global translated_output, sentence_translated, hands_absent_start_time

        last_left_hand  = None
        last_right_hand = None
        lost_frames_left  = 0
        lost_frames_right = 0
        MAX_LOST = 4
        _prev_t = time.time()

        with mp_holistic.Holistic(
            min_detection_confidence=0.50,
            min_tracking_confidence=0.50,
            model_complexity=0,
        ) as holistic:

            while self._running:
                ret, frame = cam_capture.read()
                if not ret or frame is None:
                    time.sleep(0.005)
                    continue

                frame     = cv2.flip(frame, 1)
                H_f, W_f  = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process MediaPipe at downscaled resolution for high FPS on low-end machines
                # 320x240 retains landmark accuracy while cutting processing overhead by 75%
                mp_w, mp_h = 320, 240
                rgb_small = cv2.resize(rgb, (mp_w, mp_h), interpolation=cv2.INTER_LINEAR)
                results   = holistic.process(rgb_small)

                # Target Lock-In & Calibration Filter
                validate_and_filter_target(results)

                # FPS
                now = time.time()
                fps_value = 1.0 / max(now - _prev_t, 1e-6)
                _prev_t   = now

                # ── Ambient Lux estimate from frame brightness (subsampled for speed) ──
                mean_brightness = float(np.mean(rgb_small[::4, ::4]))
                raw_lux = (mean_brightness / 255.0) ** 1.8 * 2000.0
                lux_value = lux_value * 0.85 + raw_lux * 0.15


                # ── Focal Distance estimate from face landmark span ──
                # Uses the eye-to-eye width in normalised coords; a real webcam at
                # typical 60° FOV and average inter-pupillary distance (~63 mm)
                # gives a rough cm estimate.
                FOCAL_REF_PX   = 0.30   # normalised IPD when face is ~50 cm away
                REAL_IPD_CM    = 6.3
                FOCAL_REF_DIST = 50.0
                if results.face_landmarks:
                    lm = results.face_landmarks.landmark
                    # Landmark 33 = left eye outer, 263 = right eye outer
                    le = lm[33]; re = lm[263]
                    ipd_norm = abs(re.x - le.x)
                    if ipd_norm > 0.01:
                        dist_cm = dist_cm * 0.85 + (FOCAL_REF_DIST * FOCAL_REF_PX / ipd_norm) * 0.15

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
                                            if sentence_translated:
                                                typed_output        = out_word + " "
                                                translated_output   = ""
                                                sentence_translated = False
                                            else:
                                                typed_output       += out_word + " "
                                            global_cooldown_end = current_time + WORD_COOLDOWN
                                            hands_absent_start_time = None
                                            hand_sequence_buffer.clear()
                                            wrist_sequence_buffer.clear()
                                            face_sequence_buffer.clear()
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

                        if sentence_translated:
                            typed_output        = ""
                            translated_output   = ""
                            sentence_translated = False

                        if motion_triggered and motion_char and not on_cooldown:
                            typed_output       += motion_char
                            global_cooldown_end = current_time + GLOBAL_COOLDOWN
                            stable_start_time   = current_time + GLOBAL_COOLDOWN
                            set_notification("OK: " + motion_char + " (motion)")
                        elif letter_triggered and letter_char and not on_cooldown:
                            typed_output       += letter_char
                            global_cooldown_end = current_time + GLOBAL_COOLDOWN
                            stable_start_time   = current_time + GLOBAL_COOLDOWN
                            set_notification("OK: " + letter_char)
                    else:
                        missing_frames += 1
                        if missing_frames > 10:
                            pinky_history.clear(); index_history.clear()
                            prediction_buffer.clear()
                            last_stable_char = ""; motion_state = ""

                # ── Auto-Sentence Translation when hands not on screen for 3 seconds ──
                if current_mode == 2:
                    hands_on_screen = (results.left_hand_landmarks is not None or
                                       results.right_hand_landmarks is not None)
                else:
                    hands_on_screen = (primary_hand is not None)

                if hands_on_screen:
                    hands_absent_start_time = None
                else:
                    if typed_output.strip() and not sentence_translated:
                        if hands_absent_start_time is None:
                            hands_absent_start_time = current_time
                        elif (current_time - hands_absent_start_time) >= TRANSLATE_ABSENCE_DELAY:
                            translated_output = translate_sentence(typed_output)
                            sentence_translated = True
                            hands_absent_start_time = None
                            set_notification(f"TRANSLATED: {translated_output}")
                            speak_text(translated_output)
                    else:
                        hands_absent_start_time = None

                # ── Draw landmarks onto frame ──
                # ZERO-COPY DRAWING: Draw landmarks directly on the RGB buffer used for display
                draw_frame = rgb
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

                # Convert to PIL and cache (store as numpy for speed; GUI thread converts)
                pil_cam = Image.fromarray(draw_frame)
                with self._cam_lock:
                    self._cam_pil = pil_cam

    # ── GUI render loop (runs on main thread @ ~60 FPS) ─────────────
    def _gui_loop(self):
        if not self._running:
            return
        t_start = time.time()
        try:
            self._render()
            if self.partner_window and getattr(self.partner_window, '_running', False):
                # Throttle partner window: update every ~100ms (10 FPS) to free up primary window
                now = time.time()
                last_p_render = getattr(self.partner_window, '_last_render_t', 0.0)
                if now - last_p_render >= 0.10:
                    self.partner_window._render()
                    self.partner_window._last_render_t = now
        except Exception as e:
            print(f"[Render Error] {e}")
        elapsed_ms = int((time.time() - t_start) * 1000)
        delay = max(1, 16 - elapsed_ms)
        self.root.after(delay, self._gui_loop)   # ~60 fps target

    # ── Master render ────────────────────────────────────
    def _pre_render_ui_shell(self):
        """Pre-renders all static GUI elements into a single cache image matching the Photoshop design."""
        W, H = self.W, self.H
        shell = create_ambient_background(W, H)
        draw = ImageDraw.Draw(shell)

        # ── 1. Top Bar Card ──
        bx, by, bw, bh = self.PAD, self.TOP_Y, W - self.PAD * 2, self.TOP_H
        draw_glass_card(shell, bx, by, bw, bh, r=26, shadow_blur=16, shadow_alpha=95)
        cy = by + bh // 2

        # Left: Ambient glow + Title
        logo_x = bx + 28
        glow = Image.new("RGBA", (72, 72), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow)
        gdraw.ellipse([0, 0, 72, 72], fill=hex2rgb("#93C5FD") + (120,))
        gdraw.ellipse([12, 12, 60, 60], fill=hex2rgb("#38BDF8") + (180,))
        glow = glow.filter(ImageFilter.GaussianBlur(12))
        shell.paste(glow, (logo_x, cy - 36), glow)

        title_x = logo_x + 80
        f_title = _get_font(25, bold=True)
        draw.text((title_x, cy - 12), "Project ", font=f_title, fill=hex2rgb(TEXT_MAIN), anchor="lm")
        w_proj = draw.textlength("Project ", font=f_title)

        bmo_x = int(title_x + w_proj)
        draw.text((bmo_x, cy - 12), "B.M.O", font=f_title, fill=hex2rgb(BLUE_PRIMARY), anchor="lm")
        w_bmo = draw.textlength("B.M.O", font=f_title)

        # Online badge pill
        badge_x = int(bmo_x + w_bmo + 16)
        badge_y = cy - 23
        badge_w, badge_h = 80, 24
        draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                               radius=badge_h // 2, fill=hex2rgb(GREEN_BG),
                               outline=hex2rgb(GREEN_BORDER), width=1)
        draw.ellipse([badge_x + 9, badge_y + 8, badge_x + 16, badge_y + 15], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, badge_x + 22, badge_y + 12, "ONLINE", size=10, fill=GREEN_TEXT, bold=True, anchor="lm")

        # Subtitle
        _draw_text(draw, title_x, cy + 16, "Filipino Sign Language Neural Core v2.4", size=13, fill=TEXT_MUTED, bold=False, anchor="lm")

        # Center: Inset Mode Slider Track
        switch_w, switch_h = 210, 48
        switch_x = W // 2 - switch_w // 2
        switch_y = cy - switch_h // 2
        draw.rounded_rectangle([switch_x, switch_y, switch_x + switch_w, switch_y + switch_h],
                               radius=switch_h // 2, fill=hex2rgb("#DDEBF9"),
                               outline=hex2rgb("#CBDDF2"), width=1)
        _draw_text(draw, switch_x + switch_w // 4, cy, "Letters", size=13, fill="#5A718F", bold=False, anchor="mm")
        _draw_text(draw, switch_x + 3 * switch_w // 4, cy, "Words", size=13, fill="#5A718F", bold=False, anchor="mm")

        # Right: FPS Counter Pill Badge Outline
        fps_w, fps_h = 120, 42
        fps_x = bx + bw - fps_w - 28
        fps_y = cy - fps_h // 2
        draw.rounded_rectangle([fps_x, fps_y, fps_x + fps_w, fps_y + fps_h],
                               radius=fps_h // 2, fill=hex2rgb("#F8FBFE"),
                               outline=hex2rgb("#DCE8F6"), width=1)
        draw.ellipse([fps_x + 16, fps_y + 17, fps_x + 24, fps_y + 25], fill=hex2rgb(GREEN_ONLINE))

        # ── 2. Left Panel Card (Camera Outer Glass Card) ──
        cx, cy_cam, cw, ch_cam = self.PAD, self.CONTENT_Y, self.LEFT_W, self.CONTENT_H
        draw_glass_card(shell, cx, cy_cam, cw, ch_cam, r=26, shadow_blur=18, shadow_alpha=110)

        # ── 3. Right Panel: BMO Header Card ──
        rx, ry_bmo, rw = self.RIGHT_X, self.CONTENT_Y, self.RIGHT_W
        bmo_h = 114
        draw_glass_card(shell, rx, ry_bmo, rw, bmo_h, r=26, shadow_blur=16, shadow_alpha=95)

        # ── 4. Right Panel: Inference Panel Card ──
        det_y = ry_bmo + bmo_h + 18
        det_h = ch_cam - bmo_h - 18
        draw_glass_card(shell, rx, det_y, rw, det_h, r=26, shadow_blur=16, shadow_alpha=95)

        # Top-Left: • INFERENCE ACTIVE
        draw.ellipse([rx + 28, det_y + 28, rx + 36, det_y + 36], fill=hex2rgb(BLUE_PRIMARY))
        _draw_text(draw, rx + 44, det_y + 32, "INFERENCE ACTIVE", size=13, fill=BLUE_PRIMARY, bold=True, anchor="lm")

        # Center: Neumorphic Squircle Background
        sq_size = 90
        sq_x = rx + (rw - sq_size) // 2
        sq_y = det_y + (det_h - sq_size) // 2 - 20
        draw.rounded_rectangle([sq_x, sq_y, sq_x + sq_size, sq_y + sq_size],
                               radius=24, fill=hex2rgb("#E0EEFA"),
                               outline=hex2rgb("#FFFFFF"), width=2)

        # ── 5. Metrics Row Card ──
        mx, my, mw, mh = self.PAD, self.METRICS_Y, W - self.PAD * 2, self.METRICS_H
        draw_glass_card(shell, mx, my, mw, mh, r=24, shadow_blur=16, shadow_alpha=95)

        slot_w = mw // 4
        cy_met = my + mh // 2

        items = [
            ("NEURAL LANDMARKS", "nodes"),
            ("AMBIENT LUX", "lux"),
            ("FOCAL DISTANCE", "dist"),
            ("KINEMATIC STABILITY", "ecg"),
        ]

        for i, (label, itype) in enumerate(items):
            sx = mx + i * slot_w + 32
            dome_d = 58
            dome_x = sx
            dome_y = cy_met - dome_d // 2

            # Neumorphic Dome Background
            draw.ellipse([dome_x, dome_y, dome_x + dome_d, dome_y + dome_d],
                         fill=hex2rgb("#E3F0FC"), outline=hex2rgb("#FFFFFF"), width=2)

            ic_cx, ic_cy = dome_x + dome_d // 2, dome_y + dome_d // 2
            if itype == "nodes":
                draw.rounded_rectangle([ic_cx - 10, ic_cy - 10, ic_cx + 10, ic_cy + 10], radius=4,
                                       outline=hex2rgb(BLUE_PRIMARY), width=2)
                draw.ellipse([ic_cx - 3, ic_cy - 3, ic_cx + 3, ic_cy + 3], fill=hex2rgb(BLUE_PRIMARY))
            elif itype == "lux":
                draw.ellipse([ic_cx - 6, ic_cy - 6, ic_cx + 6, ic_cy + 6], outline=hex2rgb(BLUE_PRIMARY), width=2)
                for deg in range(0, 360, 45):
                    rad = math.radians(deg)
                    x1 = ic_cx + int(9 * math.cos(rad))
                    y1 = ic_cy + int(9 * math.sin(rad))
                    x2 = ic_cx + int(13 * math.cos(rad))
                    y2 = ic_cy + int(13 * math.sin(rad))
                    draw.line([(x1, y1), (x2, y2)], fill=hex2rgb(BLUE_PRIMARY), width=2)
            elif itype == "dist":
                draw.ellipse([ic_cx - 9, ic_cy - 9, ic_cx + 9, ic_cy + 9], outline=hex2rgb(BLUE_PRIMARY), width=2)
                draw.ellipse([ic_cx - 3, ic_cy - 3, ic_cx + 3, ic_cy + 3], fill=hex2rgb(BLUE_PRIMARY))
                draw.line([(ic_cx - 13, ic_cy), (ic_cx - 10, ic_cy)], fill=hex2rgb(BLUE_PRIMARY), width=2)
                draw.line([(ic_cx + 10, ic_cy), (ic_cx + 13, ic_cy)], fill=hex2rgb(BLUE_PRIMARY), width=2)
            elif itype == "ecg":
                pts = [(ic_cx - 14, ic_cy), (ic_cx - 8, ic_cy), (ic_cx - 4, ic_cy - 8),
                       (ic_cx + 1, ic_cy + 9), (ic_cx + 5, ic_cy - 4), (ic_cx + 9, ic_cy), (ic_cx + 14, ic_cy)]
                draw.line(pts, fill=hex2rgb(BLUE_PRIMARY), width=2)

            tx = dome_x + dome_d + 18
            _draw_text(draw, tx, cy_met - 14, label, size=11, fill=TEXT_MUTED, bold=True, anchor="lm")

            if i < 3:
                div_x = mx + (i + 1) * slot_w
                draw.line([(div_x, cy_met - 30), (div_x, cy_met + 30)], fill=hex2rgb("#DDEAF7"), width=1)

        # ── 6. Bottom Bar Card ──
        bx_bot, by_bot, bw_bot, bh_bot = self.PAD, self.BOT_Y, W - self.PAD * 2, self.BOT_H
        draw_glass_card(shell, bx_bot, by_bot, bw_bot, bh_bot, r=26, shadow_blur=18, shadow_alpha=110)

        # Header: Audio Equalizer Waveform + FSL OUTPUT
        hdr_x = bx_bot + 36
        hdr_y = by_bot + 28
        w_amps = [4, 9, 14, 8]
        for i, amp in enumerate(w_amps):
            xi = hdr_x + i * 5
            draw.line([(xi, hdr_y - amp), (xi, hdr_y + amp)], fill=hex2rgb(BLUE_PRIMARY), width=2)
        _draw_text(draw, hdr_x + 28, hdr_y, "FSL OUTPUT", size=14, fill=BLUE_PRIMARY, bold=True, anchor="lm")

        # Inset Pill Container Track (recessed neumorphic text field)
        row_y = by_bot + 62
        row_h = 60
        btn_w = 175
        gap = 14
        send_x = bx_bot + bw_bot - btn_w - 36
        spk_x = send_x - btn_w - gap
        inp_x = bx_bot + 36
        inp_w = spk_x - inp_x - 20
        inp_h = row_h
        draw.rounded_rectangle([inp_x, row_y, inp_x + inp_w, row_y + inp_h],
                               radius=inp_h // 2, fill=hex2rgb("#F3F8FE"),
                               outline=hex2rgb("#CFE2F7"), width=1)

        # Pre-render default Speak and Send buttons onto static shell
        draw_gradient_pill(shell, spk_x, row_y, btn_w, row_h, "#1D68FE", "#38BDF8")
        draw.arc([spk_x + 1, row_y + 1, spk_x + btn_w - 2, row_y + row_h - 2], start=190, end=350, fill=hex2rgb(WHITE), width=1)
        spk_ic_x = spk_x + 26
        spk_ic_y = row_y + row_h // 2
        draw.polygon([(spk_ic_x - 8, spk_ic_y - 4), (spk_ic_x - 3, spk_ic_y - 4),
                      (spk_ic_x + 3, spk_ic_y - 9), (spk_ic_x + 3, spk_ic_y + 9),
                      (spk_ic_x - 3, spk_ic_y + 4), (spk_ic_x - 8, spk_ic_y + 4)], fill=hex2rgb(WHITE))
        draw.arc([spk_ic_x + 3, spk_ic_y - 7, spk_ic_x + 11, spk_ic_y + 7], start=-60, end=60, fill=hex2rgb(WHITE), width=2)
        draw.arc([spk_ic_x + 6, spk_ic_y - 11, spk_ic_x + 16, spk_ic_y + 11], start=-60, end=60, fill=hex2rgb(WHITE), width=2)
        _draw_text(draw, spk_x + 52, row_y + row_h // 2, "SPEAK", size=15, fill=WHITE, bold=True, anchor="lm")

        draw_gradient_pill(shell, send_x, row_y, btn_w, row_h, "#0EA5E9", "#0369A1")
        draw.arc([send_x + 1, row_y + 1, send_x + btn_w - 2, row_y + row_h - 2], start=190, end=350, fill=hex2rgb(WHITE), width=1)
        arr_cx = send_x + 30
        arr_cy = row_y + row_h // 2
        draw.line([(arr_cx - 10, arr_cy), (arr_cx + 8, arr_cy)], fill=hex2rgb(WHITE), width=3)
        draw.polygon([(arr_cx + 8, arr_cy), (arr_cx + 2, arr_cy - 6), (arr_cx + 2, arr_cy + 6)], fill=hex2rgb(WHITE))
        _draw_text(draw, send_x + 52, row_y + row_h // 2, "SEND", size=15, fill=WHITE, bold=True, anchor="lm")

        return shell

    def _render(self):
        now  = time.time()
        W, H = self.W, self.H
        s    = self.ui_scale
        ox, oy = self.ui_ox, self.ui_oy

        # 1. Update GUI rendering FPS tracker
        self._frame_times.append(now)
        if len(self._frame_times) > 1:
            self.gui_fps = (len(self._frame_times) - 1) / max(self._frame_times[-1] - self._frame_times[0], 1e-4)

        display_fps = self.gui_fps if hasattr(self, 'gui_fps') and self.gui_fps > 0 else fps_value
        self.canvas.itemconfig(self.fps_txt_id, text=f"{display_fps:.0f} FPS")

        # 2. Dynamic Slider Knob
        cy_top = self.TOP_Y + self.TOP_H // 2
        switch_w, switch_h = 210, 48
        switch_x = W // 2 - switch_w // 2
        switch_y = cy_top - switch_h // 2
        knob_pad = 4
        knob_w = switch_w // 2 - knob_pad
        knob_h = switch_h - knob_pad * 2
        knob_y = switch_y + knob_pad
        knob_x = switch_x + knob_pad if current_mode == 1 else switch_x + switch_w // 2

        kx0 = ox + int(knob_x * s)
        ky0 = oy + int(knob_y * s)
        kx1 = kx0 + int(knob_w * s)
        ky1 = ky0 + int(knob_h * s)
        kr  = int(knob_h * s) // 2
        self.canvas.coords(self.slider_knob_id, *_round_rect_pts(kx0, ky0, kx1, ky1, kr))
        self.canvas.itemconfig(self.lbl_letters_id, fill=WHITE if current_mode == 1 else "#5A718F")
        self.canvas.itemconfig(self.lbl_words_id, fill=WHITE if current_mode == 2 else "#5A718F")

        # 3. Update Camera Viewport Sub-image
        self._draw_left_panel(now)

        # 4. Update Right Companion Panel Sub-image (throttled + dirty-flag for max FPS)
        active_pred = current_frame_prediction if current_mode == 1 else debug_motion_word
        right_key = (tts_is_speaking, id(self.received_message), active_pred, current_mode)
        time_since_right = now - self._right_last_update
        if right_key != self._right_cache_key or time_since_right >= self._RIGHT_PANEL_MIN_INTERVAL:
            self._draw_right_panel(now)
            self._right_cache_key = right_key
            self._right_last_update = now

        # 5. Update Stats Row Texts & Kinematic Stability Bar
        mx, my, mw, mh = self.PAD, self.METRICS_Y, self.W - self.PAD * 2, self.METRICS_H
        slot_w = mw // 4
        cy_met = my + mh // 2

        stab_val = min(debug_motion_conf / 100.0 if current_mode == 2
                       else (min((time.time() - stable_start_time) / STABILIZATION_DELAY, 1.0)
                             if current_frame_prediction else 0.0), 1.0)
        rms_val = max(0.0, 0.1 - (stab_val * 0.1))

        self.canvas.itemconfig(self.stat_text_ids[0], text=f"{node_count} Nodes")
        self.canvas.itemconfig(self.stat_text_ids[1], text=f"{int(lux_value)} Lux")
        self.canvas.itemconfig(self.stat_text_ids[2], text=f"{int(dist_cm)} cm")
        self.canvas.itemconfig(self.stat_text_ids[3], text=f"{rms_val:.2f} RMS")

        bar_x = ox + int((mx + 3 * slot_w + 32 + 58 + 18 + 120) * s)
        bar_w = int(120 * s)
        bar_h = int(10 * s)
        bar_y = oy + int((cy_met + 9) * s)
        if bar_w > 20:
            fill_w = int(bar_w * max(0.05, min(1.0, 1.0 - (rms_val / 0.15))))
            self.canvas.coords(self.rms_bar_bg_id, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h)
            self.canvas.coords(self.rms_bar_id, bar_x, bar_y, bar_x + fill_w, bar_y + bar_h)

        # 6. Update Bottom Bar Typed Text & Cursor
        bx_bot, by_bot, bw_bot, bh_bot = self.PAD, self.BOT_Y, self.W - self.PAD * 2, self.BOT_H
        row_y = by_bot + 62
        row_h = 60
        inp_x = bx_bot + 36 + 28
        inp_cy = row_y + row_h // 2
        display_text = translated_output or typed_output

        if not display_text:
            self.canvas.itemconfig(self.typed_text_id, text="Sign to the camera... Press SEND to deliver to partner.", fill=TEXT_FAINT)
            self.canvas.itemconfig(self.cursor_line_id, state="hidden")
        else:
            self.canvas.itemconfig(self.typed_text_id, text=display_text, fill=TEXT_MAIN)
            if int(now * 2) % 2 == 0:
                f = _get_font(int(22 * s), bold=True)
                tmp = Image.new("L", (1, 1))
                td = ImageDraw.Draw(tmp)
                text_len = td.textlength(display_text, font=f)
                cur_x = ox + int(inp_x * s) + int(text_len) + 4
                cur_y0 = oy + int((inp_cy - 16) * s)
                cur_y1 = oy + int((inp_cy + 16) * s)
                self.canvas.coords(self.cursor_line_id, cur_x, cur_y0, cur_x, cur_y1)
                self.canvas.itemconfig(self.cursor_line_id, state="normal")
            else:
                self.canvas.itemconfig(self.cursor_line_id, state="hidden")

        # 7. Notification Toast
        if now < notification_end:
            txt = notification_text
            nw  = max(200, len(txt) * 14 + 48)
            nh  = 48
            nx  = ox + int((self.W // 2 - nw // 2) * s)
            ny  = oy + int((self.CONTENT_Y + self.CONTENT_H // 2 - nh // 2) * s)
            nw_s = int(nw * s)
            nh_s = int(nh * s)
            nr   = nh_s // 2
            self.canvas.coords(self.toast_bg_id, *_round_rect_pts(nx, ny, nx + nw_s, ny + nh_s, nr))
            self.canvas.coords(self.toast_text_id, nx + nw_s // 2, ny + nh_s // 2)
            self.canvas.itemconfig(self.toast_text_id, text=txt, font=_get_tk_font(int(15 * s), bold=True))
            self.canvas.itemconfig(self.toast_bg_id, state="normal")
            self.canvas.itemconfig(self.toast_text_id, state="normal")
        else:
            self.canvas.itemconfig(self.toast_bg_id, state="hidden")
            self.canvas.itemconfig(self.toast_text_id, state="hidden")

    def _draw_left_panel(self, now):
        cx, cy, cw, ch = self.PAD, self.CONTENT_Y, self.LEFT_W, self.CONTENT_H
        pad = 12
        vx, vy, vw, vh = cx + pad, cy + pad, cw - pad * 2, ch - pad * 2
        vr = 20

        with self._cam_lock:
            pil_cam = self._cam_pil

        if camera_hidden:
            sub_img = Image.new("RGBA", (vw, vh), (11, 19, 43, 255))
            s_draw = ImageDraw.Draw(sub_img)
            _rounded_rect(s_draw, 0, 0, vw, vh, vr, hex2rgb("#0B132B"))
            _draw_text(s_draw, vw // 2, vh // 2 - 14, "Camera Feed Hidden", size=20, fill=TEXT_MUTED, bold=True, anchor="mm")
            _draw_text(s_draw, vw // 2, vh // 2 + 18, "Press H to show", size=14, fill=TEXT_FAINT, bold=False, anchor="mm")
            offset_x, offset_y, new_w, new_h = vx, vy, vw, vh
            cam_crop = sub_img
        elif pil_cam is not None:
            orig_w, orig_h = pil_cam.size
            aspect = orig_w / max(orig_h, 1)
            if vw / vh > aspect:
                new_h = vh
                new_w = int(vh * aspect)
            else:
                new_w = vw
                new_h = int(vw / aspect)

            resized = pil_cam.resize((new_w, new_h), Image.Resampling.NEAREST)
            offset_x = vx + (vw - new_w) // 2
            offset_y = vy + (vh - new_h) // 2

            # Cached mask for high-speed rendering
            mask_key = (new_w, new_h, vr)
            if self._cached_cam_mask_key != mask_key:
                mask = Image.new("L", (new_w, new_h), 0)
                m_draw = ImageDraw.Draw(mask)
                m_draw.rounded_rectangle([0, 0, new_w, new_h], radius=vr, fill=255)
                self._cached_cam_mask = mask
                self._cached_cam_mask_key = mask_key
            else:
                mask = self._cached_cam_mask

            cam_crop = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            cam_crop.paste(resized, (0, 0), mask)
            draw = ImageDraw.Draw(cam_crop)
            draw.rounded_rectangle([0, 0, new_w - 1, new_h - 1],
                                   radius=vr, outline=hex2rgb("#60A5FA"), width=1)
        else:
            sub_img = Image.new("RGBA", (vw, vh), (11, 19, 43, 255))
            s_draw = ImageDraw.Draw(sub_img)
            _rounded_rect(s_draw, 0, 0, vw, vh, vr, hex2rgb("#0B132B"))
            _draw_text(s_draw, vw // 2, vh // 2, "Connecting Camera...", size=20, fill=TEXT_MUTED, bold=True, anchor="mm")
            offset_x, offset_y, new_w, new_h = vx, vy, vw, vh
            cam_crop = sub_img

        # ── Overlay HUD (cached unless state changes) ──────────────────────────
        # Key: button states + size + calibration state
        hud_key = (new_w, new_h, camera_hidden, bool(target_person_lock), is_calibrating,
                   int(calibration_progress))
        if self._cached_hud_overlay is None or self._cached_hud_key != hud_key:
            hud = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
            h_draw = ImageDraw.Draw(hud)

            # 1. Top-Left Overlay Dock: 📹 CAM-01 •
            tl_w, tl_h = 160, 36
            tl_x, tl_y = 18, 18
            h_draw.rounded_rectangle([tl_x, tl_y, tl_x + tl_w, tl_y + tl_h], radius=tl_h // 2,
                                     fill=(15, 23, 42, 215), outline=hex2rgb("#475569"), width=1)
            ic_cx, ic_cy = tl_x + 24, tl_y + tl_h // 2
            h_draw.rounded_rectangle([ic_cx - 9, ic_cy - 6, ic_cx + 9, ic_cy + 6], radius=2, outline=hex2rgb(WHITE), width=2)
            h_draw.polygon([(ic_cx + 10, ic_cy - 4), (ic_cx + 15, ic_cy - 6), (ic_cx + 15, ic_cy + 6), (ic_cx + 10, ic_cy + 4)], fill=hex2rgb(WHITE))
            _draw_text(h_draw, tl_x + 48, tl_y + tl_h // 2, "CAM-01", size=12, fill=WHITE, bold=True, anchor="lm")
            h_draw.ellipse([tl_x + 115, tl_y + tl_h // 2 - 4, tl_x + 123, tl_y + tl_h // 2 + 4], fill=hex2rgb(GREEN_ONLINE))

            # 2. Bottom-Center Floating Control Dock: 3 Buttons
            self._draw_camera_controls_onto_crop(hud, h_draw, new_w, new_h, offset_x, offset_y)

            # 3. Bottom-Right Overlay: CAM-01 • Resolution
            br_w, br_h = 160, 32
            br_x = new_w - br_w - 18
            br_y = new_h - br_h - 22
            h_draw.rounded_rectangle([br_x, br_y, br_x + br_w, br_y + br_h], radius=br_h // 2,
                                     fill=(15, 23, 42, 215), outline=hex2rgb("#475569"), width=1)
            cam_info_text = f"CAM-01 • {self.render_w}×{self.render_h}"
            _draw_text(h_draw, br_x + br_w // 2, br_y + br_h // 2, cam_info_text, size=11, fill=WHITE, bold=True, anchor="mm")

            # 4. Calibration Progress Overlay
            if is_calibrating:
                sil_cx = new_w // 2
                sil_cy = new_h // 2 - 20
                h_draw.ellipse([sil_cx - 50, sil_cy - 75, sil_cx + 50, sil_cy + 35], outline=hex2rgb(BLUE_PRIMARY), width=3)
                h_draw.arc([sil_cx - 120, sil_cy + 10, sil_cx + 120, sil_cy + 180], start=180, end=360, fill=hex2rgb(BLUE_PRIMARY), width=3)
                banner_h = 36
                banner_y = new_h - banner_h - 95
                _rounded_rect(h_draw, 60, banner_y, new_w - 120, banner_h, 18, hex2rgb(CARD_DARK))
                _draw_text(h_draw, sil_cx, banner_y + banner_h // 2, f"Calibrating Target Person... {calibration_progress:.0f}%", size=13, fill=WHITE, bold=True, anchor="mm")
                bar_w_cal = int((new_w - 140) * (calibration_progress / 100.0))
                if bar_w_cal > 0:
                    _rounded_rect(h_draw, 70, banner_y + banner_h - 5, bar_w_cal, 4, 2, hex2rgb(BLUE_LIGHT))

            self._cached_hud_overlay = hud
            self._cached_hud_key = hud_key

        # Composite: paste HUD overlay onto camera frame
        if not camera_hidden and pil_cam is not None:
            cam_crop.paste(self._cached_hud_overlay, (0, 0), self._cached_hud_overlay)

        # Scale cam_crop to match ui_scale for canvas
        scaled_w = max(1, int(new_w * self.ui_scale))
        scaled_h = max(1, int(new_h * self.ui_scale))
        if scaled_w != new_w or scaled_h != new_h:
            cam_crop = cam_crop.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)

        self.cam_tk = ImageTk.PhotoImage(cam_crop)
        self.canvas.itemconfig(self.cam_image_id, image=self.cam_tk)
        self.canvas.coords(self.cam_image_id, self.ui_ox + int(offset_x * self.ui_scale), self.ui_oy + int(offset_y * self.ui_scale))

    def _draw_camera_controls_onto_crop(self, img, draw, new_w, new_h, offset_x, offset_y):
        dock_w, dock_h = 240, 56
        dock_x = (new_w - dock_w) // 2
        dock_y = new_h - dock_h - 22

        draw.rounded_rectangle([dock_x, dock_y, dock_x + dock_w, dock_y + dock_h], radius=dock_h // 2,
                               fill=(20, 35, 60, 205), outline=hex2rgb("#3B82F6"), width=1)

        self._cam_btn_rects = {}
        btn_r = 20
        gap = (dock_w - 6 * btn_r) // 4
        specs = [
            ("cam", False),
            ("eye", not camera_hidden),
            ("shield", target_person_lock is not None or is_calibrating),
        ]

        for i, (b_type, is_active) in enumerate(specs):
            bx = dock_x + gap + i * (2 * btn_r + gap) + btn_r
            by = dock_y + dock_h // 2
            global_bx = offset_x + bx
            global_by = offset_y + by
            self._cam_btn_rects[b_type] = (global_bx - btn_r, global_by - btn_r, global_bx + btn_r, global_by + btn_r)

            b_fill = "#2563EB" if is_active else "#1E293B"
            draw.ellipse([bx - btn_r, by - btn_r, bx + btn_r, by + btn_r],
                         fill=hex2rgb(b_fill), outline=hex2rgb("#60A5FA" if is_active else "#475569"), width=1)

            if b_type == "cam":
                draw.rounded_rectangle([bx - 8, by - 5, bx + 8, by + 5], radius=2, outline=hex2rgb(WHITE), width=2)
                draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], outline=hex2rgb(WHITE), width=2)
            elif b_type == "eye":
                draw.ellipse([bx - 9, by - 6, bx + 9, by + 6], outline=hex2rgb(WHITE), width=2)
                draw.ellipse([bx - 3, by - 3, bx + 3, by + 3], fill=hex2rgb(WHITE))
                if camera_hidden:
                    draw.line([(bx - 8, by - 7), (bx + 8, by + 7)], fill=hex2rgb("#EF4444"), width=2)
            elif b_type == "shield":
                draw.polygon([(bx - 8, by - 7), (bx + 8, by - 7), (bx + 8, by + 3), (bx, by + 9), (bx - 8, by + 3)],
                             outline=hex2rgb(WHITE), width=2)

    def _draw_right_panel(self, now):
        rx, ry, rw, rh = self.RIGHT_X, self.CONTENT_Y, self.RIGHT_W, self.CONTENT_H
        bmo_h = 114

        r_img = Image.new("RGBA", (rw, rh), (255, 255, 255, 0))
        draw = ImageDraw.Draw(r_img)

        # ── 1. Top Card: BMO Companion Header ──
        av_pad = 18
        av_d = 76
        av_x = av_pad
        av_y = (bmo_h - av_d) // 2

        draw_gradient_pill(r_img, av_x, av_y, av_d, av_d, "#38BDF8", "#2563EB", outline_col="#93C5FD")

        # BMO Face
        fc_cx = av_x + av_d // 2
        fc_cy = av_y + av_d // 2
        eye_offset = 12
        draw.ellipse([fc_cx - eye_offset - 3, fc_cy - 5, fc_cx - eye_offset + 3, fc_cy + 1], fill=hex2rgb(TEXT_MAIN))
        draw.ellipse([fc_cx + eye_offset - 3, fc_cy - 5, fc_cx + eye_offset + 3, fc_cy + 1], fill=hex2rgb(TEXT_MAIN))

        if tts_is_speaking:
            draw.ellipse([fc_cx - 4, fc_cy + 4, fc_cx + 4, fc_cy + 12], fill=hex2rgb(TEXT_MAIN))
        else:
            draw.arc([fc_cx - 7, fc_cy + 2, fc_cx + 7, fc_cy + 10], start=20, end=160, fill=hex2rgb(TEXT_MAIN), width=2)

        # Labels
        tx = av_x + av_d + 18
        _draw_text(draw, tx, fc_cy - 12, "BMO Companion", size=20, fill=TEXT_MAIN, bold=True, anchor="lm")
        w_bmotitle = draw.textlength("BMO Companion", font=_get_font(20, bold=True))

        badge_x = int(tx + w_bmotitle + 14)
        badge_y = fc_cy - 22
        badge_w, badge_h = 76, 22
        draw.rounded_rectangle([badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
                               radius=badge_h // 2, fill=hex2rgb(GREEN_BG),
                               outline=hex2rgb(GREEN_BORDER), width=1)
        draw.ellipse([badge_x + 8, badge_y + 7, badge_x + 14, badge_y + 13], fill=hex2rgb(GREEN_ONLINE))
        _draw_text(draw, badge_x + 20, badge_y + 11, "ONLINE", size=9, fill=GREEN_TEXT, bold=True, anchor="lm")

        sub_text = "Speaking translation..." if tts_is_speaking else (
            "Partner sent you a message!" if (self.received_message and (now - self.received_message.get("time", 0)) < 15.0)
            else "Listening for your signs..."
        )
        _draw_text(draw, tx, fc_cy + 14, sub_text, size=13, fill=TEXT_MUTED, bold=False, anchor="lm")

        # Audio Waveform Equalizer
        eq_x = rw - 70
        eq_y = fc_cy
        eq_amps = [5, 12, 18, 24, 15, 9, 4] if tts_is_speaking else [4, 8, 12, 18, 11, 7, 3]
        for i, amp in enumerate(eq_amps):
            bar_x = eq_x + i * 6
            draw.line([(bar_x, eq_y - amp), (bar_x, eq_y + amp)], fill=hex2rgb(BLUE_PRIMARY), width=3)

        # ── 2. Bottom Card: Inference & Incoming Partner Messages ──
        det_y = bmo_h + 18
        det_h = rh - bmo_h - 18

        if self.received_message:
            # Header Badge
            draw.rounded_rectangle([24, det_y + 16, 224, det_y + 42],
                                   radius=13, fill=hex2rgb(GREEN_BG),
                                   outline=hex2rgb(GREEN_BORDER), width=1)
            draw.ellipse([34, det_y + 25, 42, det_y + 33], fill=hex2rgb(GREEN_ONLINE))
            _draw_text(draw, 48, det_y + 29, "CO-OP PARTNER MESSAGE", size=10, fill=GREEN_TEXT, bold=True, anchor="lm")

            # Caption Sentence Box
            cap_y = det_y + 50
            cap_h = 72
            draw.rounded_rectangle([24, cap_y, rw - 24, cap_y + cap_h],
                                   radius=14, fill=hex2rgb("#F8FBFE"),
                                   outline=hex2rgb("#CFE2F7"), width=1)

            cap_text = self.received_message.get("text", "")
            disp_cap = cap_text[:39] + "..." if len(cap_text) > 42 else cap_text
            _draw_text(draw, 40, cap_y + cap_h // 2, f'"{disp_cap}"', size=20, fill=TEXT_MAIN, bold=True, anchor="lm")

            # Sign Previews Section
            prv_lbl_y = cap_y + cap_h + 18
            _draw_text(draw, 28, prv_lbl_y, "SIGN PREVIEWS", size=11, fill=BLUE_PRIMARY, bold=True, anchor="lm")

            previews = self.received_message.get("previews", [])
            if previews:
                prv_y = prv_lbl_y + 14
                card_w = min(105, int((rw - 56) / max(len(previews), 1)) - 10)
                card_h = 104
                for idx, (token, icon_im) in enumerate(previews[:6]):
                    cx = 24 + idx * (card_w + 10)
                    draw.rounded_rectangle([cx, prv_y, cx + card_w, prv_y + card_h],
                                           radius=14, fill=hex2rgb("#FFFFFF"),
                                           outline=hex2rgb("#CBDDF2"), width=1)
                    try:
                        thumb = icon_im.resize((52, 52), Image.Resampling.BILINEAR)
                        if thumb.mode == "RGBA":
                            r_img.paste(thumb, (cx + (card_w - 52) // 2, prv_y + 8), thumb)
                        else:
                            r_img.paste(thumb, (cx + (card_w - 52) // 2, prv_y + 8))
                    except Exception:
                        pass
                    _draw_text(draw, cx + card_w // 2, prv_y + 78, token.upper(), size=11, fill=TEXT_MAIN, bold=True, anchor="mm")
            else:
                _draw_text(draw, 28, prv_lbl_y + 24, "(No sign previews mapped for this sentence)", size=12, fill=TEXT_MUTED)

            # Bottom Divider & Live Signer Status
            div_y = det_y + det_h - 70
            draw.line([(24, div_y), (rw - 24, div_y)], fill=hex2rgb("#DDEAF7"), width=1)
            active_pred = current_frame_prediction if current_mode == 1 else (debug_motion_word if debug_motion_word != "--" else "")
            live_text = f"LIVE RECOGNITION: {active_pred.upper()}" if active_pred else "LIVE RECOGNITION: Ready (Signing active)"
            _draw_text(draw, 32, div_y + 35, live_text, size=14, fill=BLUE_PRIMARY if active_pred else TEXT_MUTED, bold=True, anchor="lm")

        else:
            # Default Live Inference View
            sq_size = 90
            sq_x = (rw - sq_size) // 2
            sq_y = det_y + (det_h - sq_size) // 2 - 20
            w_cx = sq_x + sq_size // 2
            w_cy = sq_y + sq_size // 2

            active_mult = 1.3 if (current_frame_prediction or debug_motion_word not in ("--", "")) else 1.0
            w_amps = [int(6 * active_mult), int(12 * active_mult), int(20 * active_mult), int(14 * active_mult), int(7 * active_mult)]
            for i, amp in enumerate(w_amps):
                xi = w_cx - 16 + i * 8
                draw.line([(xi, w_cy - amp), (xi, w_cy + amp)], fill=hex2rgb(BLUE_PRIMARY), width=4)

            status_y = sq_y + sq_size + 36
            active_pred = current_frame_prediction if current_mode == 1 else (debug_motion_word if debug_motion_word != "--" else "")
            if active_pred:
                _draw_text(draw, rw // 2, status_y, f"Recognized: {active_pred.upper()}", size=18, fill=TEXT_MAIN, bold=True, anchor="mm")
            else:
                _draw_text(draw, rw // 2, status_y, "Waiting for your sign...", size=15, fill=TEXT_MUTED, bold=False, anchor="mm")
                _draw_text(draw, rw // 2, status_y + 28, "Partner messages will appear here as captions & sign previews", size=12, fill=TEXT_FAINT, anchor="mm")

        scaled_w = max(1, int(rw * self.ui_scale))
        scaled_h = max(1, int(rh * self.ui_scale))
        if scaled_w != rw or scaled_h != rh:
            r_img = r_img.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)

        self.right_tk = ImageTk.PhotoImage(r_img)
        self.canvas.itemconfig(self.right_image_id, image=self.right_tk)
        self.canvas.coords(self.right_image_id, self.ui_ox + int(rx * self.ui_scale), self.ui_oy + int(ry * self.ui_scale))

    def destroy(self):
        self._running = False
        cam_capture.stop()
        try:
            if sys.platform.startswith("win"):
                ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass


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

def _draw_gradient_text(img, x, y, text, size=14, c1_hex=ACC_BLUE, c2_hex=ACC_CYAN,
                        bold=True, anchor="lm"):
    """Draw text filled with a horizontal gradient by rendering it as a mask."""
    if not text:
        return
    f = _get_font(size, bold)
    tmp = Image.new("L", (1, 1), 0)
    tmp_draw = ImageDraw.Draw(tmp)
    bbox = tmp_draw.textbbox((0, 0), text, font=f, anchor=anchor)
    tw = max(1, bbox[2] - bbox[0])
    th = max(1, bbox[3] - bbox[1])

    mask = Image.new("L", (tw, th), 0)
    md = ImageDraw.Draw(mask)
    md.text((-bbox[0], -bbox[1]), text, font=f, fill=255, anchor=anchor)

    grad = Image.new("RGB", (tw, th))
    gd = ImageDraw.Draw(grad)
    c1, c2 = hex2rgb(c1_hex), hex2rgb(c2_hex)
    for i in range(tw):
        t = i / max(tw - 1, 1)
        gd.line([(i, 0), (i, th)], fill=_blend(c1, c2, t))

    img.paste(grad, (int(x + bbox[0]), int(y + bbox[1])), mask)


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

    # Launch Non-FSL Partner Window in the same process with live camera access
    partner_app = NonFSLWindow(master=root, get_cam_fn=app.get_cam_frame)
    app.partner_window = partner_app

    def _on_close():
        try:
            partner_app.window.destroy()
        except Exception:
            pass
        app.destroy()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    partner_app.on_close_callback = _on_close

    # Start maximized / fullscreen-ready at 1920×1080
    root.state("zoomed")   # Windows maximized
    root.deiconify()
    root.lift()
    root.focus_force()

    if partner_app:
        partner_app.window.deiconify()
        partner_app.window.lift()

    root.mainloop()

if __name__ == "__main__":
    main()
