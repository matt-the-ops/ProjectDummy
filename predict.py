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
#  TTS (Non-blocking)
# ─────────────────────────────────────────────
_tts_lock = threading.Lock()

def speak_text(text):
    if not text.strip():
        return
    def _run():
        with _tts_lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"TTS Error: {e}")
    threading.Thread(target=_run, daemon=True).start()

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
try:
    with open('fsl_model.pkl', 'rb') as f:
        letter_data = SafeUnpickler(f).load()
        letter_model = letter_data['model'] if isinstance(letter_data, dict) else letter_data
    print("[OK] Letter model loaded.")
except FileNotFoundError:
    print("Error: 'fsl_model.pkl' not found.")
    exit()

# ─────────────────────────────────────────────
#  Load Mode 2: Dual-Gate Word Models
# ─────────────────────────────────────────────
motion_model = None
face_clf     = None
word_classes = None

if os.path.exists('word_motion_model.h5') and \
   os.path.exists('word_face_model.pkl') and \
   os.path.exists('word_classes.npy'):
    try:
        motion_model = load_model('word_motion_model.h5')
        with open('word_face_model.pkl', 'rb') as f:
            face_data = SafeUnpickler(f).load()
            face_clf  = face_data['model']
        word_classes = np.load('word_classes.npy', allow_pickle=True)
        print(f"[OK] Dual-gate models loaded. Classes: {word_classes}")
        DUAL_GATE_MODE = True
    except Exception as e:
        print(f"[!] Dual-gate load error: {e}")
        DUAL_GATE_MODE = False
else:
    DUAL_GATE_MODE = False
    print("[!] Dual-gate models not found.")
    if os.path.exists('word_model.h5') and os.path.exists('word_classes.npy'):
        try:
            motion_model = load_model('word_model.h5')
            word_classes = np.load('word_classes.npy', allow_pickle=True)
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
#  Drawing Helpers
# ─────────────────────────────────────────────
PANEL_COLOR   = (20,  20,  20)
ACCENT_LETTER = (0,  220, 100)
ACCENT_WORD   = (140,  60, 255)
ACCENT_MOTION = (0,  200, 255)
ACCENT_WARN   = (0,   90, 255)
TEXT_PRIMARY  = (240, 240, 240)
TEXT_DIM      = (140, 140, 140)

def draw_panel(img, x, y, w, h, color=PANEL_COLOR, alpha=0.55):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1):
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
    cv2.putText(img, text, (org[0]+1, org[1]+1), font, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def draw_progress_bar(img, x, y, w, h, value, max_val, bar_color, bg=(50,50,50)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    filled = int(w * min(value / max(max_val, 1), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+h), bar_color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (80,80,80), 1)

# ─────────────────────────────────────────────
#  State -- Mode 1 & 2
# ─────────────────────────────────────────────
LETTER_BUFFER_SIZE  = 12
STABILIZATION_DELAY = 2.5
GLOBAL_COOLDOWN     = 1.2

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
print("Controls: 1=Letters  2=Words  N=Cam  Enter=Speak  Space  Bksp  C=Clear  Q=Quit\n")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,
) as holistic:

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
        # ══════════════════════════════════════════════════════
        #  MODE 2: DUAL-GATE WORD RECOGNITION (Sliding Window)
        # ══════════════════════════════════════════════════════
        if current_mode == 2:
            if motion_model is not None:
                has_hand = (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None)

                if has_hand:
                    missing_frames = 0
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
                else:
                    hand_sequence_buffer.append(extract_hand_frame(results))
                    wrist_sequence_buffer.append(get_raw_wrist(results))
                    face_sequence_buffer.append(extract_face_frame(results))

                    buf_len = len(hand_sequence_buffer)

                    if buf_len == WORD_SEQ_LEN and not on_cooldown:
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
        #  Landmark Drawing
        # ══════════════════════════════════════════════════════
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(60,60,60), thickness=1))
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,255,140), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,180,80),   thickness=2))
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,100,80), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(200,50,30),  thickness=2))

        # ══════════════════════════════════════════════════════
        #  HUD / UI Rendering
        # ══════════════════════════════════════════════════════
        mode_accent = ACCENT_LETTER if current_mode == 1 else ACCENT_WORD
        FONT        = cv2.FONT_HERSHEY_SIMPLEX
        FONT_MONO   = cv2.FONT_HERSHEY_DUPLEX

        draw_panel(frame, 0, 0, W, 58)
        mode_label = "[ LETTERS ]" if current_mode == 1 else "[  WORDS  ]"
        put_text_shadow(frame, mode_label, (14, 38), FONT_MONO, 0.85, mode_accent, 2)

        cam_txt = "Cam " + str(camera_index)
        (cw, _), _ = cv2.getTextSize(cam_txt, FONT, 0.55, 1)
        put_text_shadow(frame, cam_txt, (W-cw-14, 36), FONT, 0.55, TEXT_DIM, 1)

        if current_mode == 1:
            panel_x, panel_y, panel_w, panel_h = 10, 65, 200, 130
            draw_panel(frame, panel_x, panel_y, panel_w, panel_h)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), mode_accent, 1)

            sign_display = current_frame_prediction if current_frame_prediction else "?"
            put_text_shadow(frame, sign_display, (panel_x+14, panel_y+70), FONT_MONO, 2.8, mode_accent, 5)

            if current_frame_prediction and not on_cooldown and current_time >= suppression_end:
                held = current_time - stable_start_time
                draw_progress_bar(frame, panel_x+10, panel_y+100, panel_w-20, 10, held, STABILIZATION_DELAY, ACCENT_LETTER)
            elif on_cooldown:
                left = global_cooldown_end - current_time
                draw_progress_bar(frame, panel_x+10, panel_y+100, panel_w-20, 10, GLOBAL_COOLDOWN-left, GLOBAL_COOLDOWN, ACCENT_WARN)

        elif current_mode == 2:
            panel_x, panel_y, panel_w, panel_h = 10, 65, 280, 200
            draw_panel(frame, panel_x, panel_y, panel_w, panel_h)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), mode_accent, 1)

            buf_count = len(hand_sequence_buffer)
            put_text_shadow(frame, "BUFFER", (panel_x+10, panel_y+22), FONT, 0.48, TEXT_DIM, 1)
            draw_progress_bar(frame, panel_x+10, panel_y+28, panel_w-20, 10, buf_count, WORD_SEQ_LEN, ACCENT_WORD)

            put_text_shadow(frame, "HAND MOTION", (panel_x+10, panel_y+76), FONT, 0.48, TEXT_DIM, 1)
            m_color = ACCENT_WORD if debug_motion_conf >= MOTION_CONF_THRESH*100 else (100,100,100)
            put_text_shadow(frame, debug_motion_word, (panel_x+10, panel_y+100), FONT_MONO, 0.85, m_color, 2)
            put_text_shadow(frame, str(round(debug_motion_conf,1))+"%", (panel_x+130, panel_y+100), FONT, 0.48, m_color, 1)

            if DUAL_GATE_MODE:
                cv2.line(frame, (panel_x+8, panel_y+112), (panel_x+panel_w-8, panel_y+112), (60,60,60), 1)
                put_text_shadow(frame, "FACE SIGNATURE", (panel_x+10, panel_y+130), FONT, 0.48, TEXT_DIM, 1)
                f_color = ACCENT_WORD if debug_face_conf >= FACE_CONF_THRESH*100 else (100,100,100)
                put_text_shadow(frame, debug_face_word, (panel_x+10, panel_y+155), FONT_MONO, 0.85, f_color, 2)
                put_text_shadow(frame, str(round(debug_face_conf,1))+"%", (panel_x+130, panel_y+155), FONT, 0.48, f_color, 1)

                gs_colors = {
                    "AGREE":     (0, 220, 100),
                    "MOTION-OK": (0, 220, 100),
                    "SINGLE-OK": (0, 220, 100),
                    "DISAGREE":  (0,  60, 220),
                    "LOW CONF":  (0,  90, 255),
                    "NO MOTION": (140, 140, 140),
                    "WAITING":   (100, 100, 100),
                }
                gs_color = gs_colors.get(gate_style := gate_status, TEXT_DIM)
                bx2, by2 = panel_x+10, panel_y+168
                draw_rounded_rect(frame, bx2, by2, panel_w-20, 24, 5, (30,30,30))
                cv2.rectangle(frame, (bx2,by2), (bx2+panel_w-20, by2+24), gs_color, 1)
                (gw,_),_ = cv2.getTextSize(gate_status, FONT, 0.5, 1)
                gtx = bx2 + (panel_w-20-gw)//2
                put_text_shadow(frame, gate_status, (gtx, by2+17), FONT, 0.5, gs_color, 1)

        out_box_y = H - 100
        draw_panel(frame, 0, out_box_y, W, 58)
        cv2.line(frame, (0, out_box_y), (W, out_box_y), mode_accent, 2)
        put_text_shadow(frame, "OUTPUT", (12, out_box_y+18), FONT, 0.42, TEXT_DIM, 1)

        max_chars = 38
        disp_out = typed_output if len(typed_output) <= max_chars else "..." + typed_output[-max_chars:]
        put_text_shadow(frame, disp_out, (12, out_box_y+48), FONT_MONO, 0.9, TEXT_PRIMARY, 2)

        hints_y = H - 36
        draw_panel(frame, 0, hints_y, W, 36)
        hints = "1:Letters  2:Words  N:Cam  Enter:Speak  Space  Bksp  C:Clear  Q:Quit"
        put_text_shadow(frame, hints, (10, hints_y+22), FONT, 0.36, TEXT_DIM, 1)

        if current_time < notification_end:
            alpha = min(1.0, (notification_end - current_time) / 0.35)
            nx, ny, nw, nh = W//2-150, 70, 300, 44
            overlay = frame.copy()
            draw_rounded_rect(overlay, nx, ny, nw, nh, 10, (20,20,20))
            cv2.addWeighted(overlay, 0.80*alpha, frame, 1-0.80*alpha, 0, frame)
            cv2.rectangle(frame, (nx,ny), (nx+nw, ny+nh), mode_accent, 1)
            (tw,_),_ = cv2.getTextSize(notification_text, FONT_MONO, 0.72, 2)
            tx = nx + (nw-tw)//2
            put_text_shadow(frame, notification_text, (tx, ny+30), FONT_MONO, 0.72, mode_accent, 2)

        cv2.imshow("FSL Dual-Gate Translator", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('1'):
            current_mode = 1
            typed_output = ""
            motion_state = ""
            pinky_history.clear()
            index_history.clear()
            prediction_buffer.clear()
            hand_sequence_buffer.clear()
            last_stable_char = ""
            set_notification("Switched -> LETTERS")
        elif key == ord('2'):
            current_mode = 2
            typed_output = ""
            hand_sequence_buffer.clear()
            debug_motion_word = "--"
            debug_face_word   = "--"
            debug_motion_conf = 0.0
            debug_face_conf   = 0.0
            gate_status       = "WAITING"
            set_notification("Switched -> WORDS")
        elif key == ord('n') or key == ord('N'):
            cap.release()
            camera_index = (camera_index + 1) % 3
            cap = cv2.VideoCapture(camera_index)
            set_notification("Camera -> " + str(camera_index))
        elif key == 13:
            speak_text(typed_output)
        elif key == 32:
            typed_output += " "
        elif key in (8, 127):
            typed_output = typed_output[:-1]
        elif key == ord('c') or key == ord('C'):
            typed_output = ""
            set_notification("Output cleared")

cap.release()
cv2.destroyAllWindows()