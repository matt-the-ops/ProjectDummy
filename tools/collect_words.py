import os
import cv2
import json
import shutil
import time
import math
import numpy as np
import mediapipe as mp

try:
    import mediapipe.solutions.holistic as mp_holistic
    import mediapipe.solutions.drawing_utils as mp_drawing
except ModuleNotFoundError:
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmo.config import DATA_DIR, MODELS_DIR, ASSETS_DIR, ROOT_DIR, ICONS_DIR, ICON_MAP_FILE

DATA_PATH      = os.path.join(DATA_DIR, 'word_dataset')
ICON_MAP_FILE  = ICON_MAP_FILE
ICONS_DIR      = ICONS_DIR

SEQUENCE_LEN   = 40       # hand motion frames
HAND_FEATURES  = 198      # 126 shape + 36 spatial anchors + 12 facial emotion + 18 per-frame velocities/speeds/direction ratios + 6 trajectory
FACE_FEATURES  = 1404     # 468 face landmarks * 3

os.makedirs(DATA_PATH, exist_ok=True)

# ─── Feature Helpers ────────────────────────────────────────────────────────

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bmo.features.fsl_features import (
    get_detailed_face_and_body_anchors,
    extract_hand_frame,
    get_raw_hand_anchors,
    build_motion_dynamics_and_trajectory,
    extract_face_frame,
    smooth_landmarks,
    resolve_and_smooth_hands,
)










# ─── Dataset Management ─────────────────────────────────────────────────────

def remove_outdated_models():
    for f in ['word_motion_model.h5', 'word_face_model.pkl', 'word_classes.npy']:
        p = os.path.join(MODELS_DIR, f)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f"[!] Removed stale '{f}'. Retrain required.")
            except Exception as e:
                print(f"[!] Could not remove '{f}': {e}")



# --- Icon Map Management -----------------------------------------------------

def load_icon_map():
    """Load icon_map.json, return as dict. Returns {} if missing."""
    if os.path.exists(ICON_MAP_FILE):
        try:
            with open(ICON_MAP_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"[!] Could not read icon_map.json: {e}")
    return {}

def save_icon_map(icon_map):
    """Save dict back to icon_map.json with pretty formatting."""
    try:
        with open(ICON_MAP_FILE, 'w') as f:
            json.dump(icon_map, f, indent=2)
    except Exception as e:
        print(f"[!] Could not save icon_map.json: {e}")

def manage_word_icon(word_name):
    """
    Interactive prompt to assign, replace, or remove an icon PNG for a word.
    Called automatically after recording a new word.
    """
    os.makedirs(ICONS_DIR, exist_ok=True)
    icon_map = load_icon_map()
    word_key = word_name.lower()
    existing = icon_map.get(word_key)

    print(f"\n--- Icon Manager for '{word_name}' ---")
    if existing:
        print(f"  Current icon : {existing}")
    else:
        print("  No icon assigned yet.")

    print("  1. Add / Replace icon PNG")
    print("  2. Remove icon")
    print("  3. Skip (keep as is)")

    choice = input("  Select (1-3): ").strip()

    if choice == '1':
        src = input("  Full path to your PNG file: ").strip().strip('"').strip("'")
        if not src or not os.path.isfile(src):
            print("  [!] File not found. Skipping icon assignment.")
            return
        if not src.lower().endswith('.png'):
            print("  [!] Only PNG files are supported. Skipping.")
            return
        dest_name = f"{word_key}.png"
        dest_path = os.path.join(ICONS_DIR, dest_name)
        try:
            import shutil as _shutil
            _shutil.copy2(src, dest_path)
            icon_map[word_key] = f"assets/fsl_icons/{dest_name}"
            save_icon_map(icon_map)
            print(f"  [OK] Icon saved ? {dest_path}")
            print(f"  [OK] icon_map.json updated: '{word_key}' ? 'assets/fsl_icons/{dest_name}'")
        except Exception as e:
            print(f"  [!] Failed to copy icon: {e}")

    elif choice == '2':
        if word_key in icon_map:
            old_icon = icon_map.pop(word_key)
            save_icon_map(icon_map)
            print(f"  [OK] Removed icon entry for '{word_key}' from icon_map.json.")
            # Optionally delete the file too
            old_file = os.path.join(ROOT_DIR, old_icon)
            if os.path.exists(old_file):
                del_file = input(f"  Also delete the image file '{old_icon}'? (y/n): ").strip().lower()
                if del_file == 'y':
                    try:
                        os.remove(old_file)
                        print(f"  [OK] Deleted icon file: {old_file}")
                    except Exception as e:
                        print(f"  [!] Could not delete file: {e}")
        else:
            print(f"  [!] No icon entry found for '{word_key}' in icon_map.json.")

    else:
        print("  [~] Icon unchanged.")

def reset_single_word(word_name):
    target_dir = os.path.join(DATA_PATH, word_name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"[OK] Deleted word folder: '{word_name}'")
        remove_outdated_models()
    else:
        print(f"[!] Word '{word_name}' not found.")


def reset_all_words():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH, exist_ok=True)
        print("[OK] Reset ALL word datasets.")
        remove_outdated_models()

# ─── Landmark Smoothing & Disambiguation Helpers ─────────────────────────────




# ─── Recording ──────────────────────────────────────────────────────────────

def record_word(word_name, num_samples=20):
    word_dir = os.path.join(DATA_PATH, word_name)
    os.makedirs(word_dir, exist_ok=True)

    existing = [f for f in os.listdir(word_dir) if f.endswith('_hand.npy')]
    start_counter = len(existing)

    cap = cv2.VideoCapture(0)
    print(f"\n--- Recording '{word_name}' ({num_samples} samples) ---")
    print("SPACE = start recording | Q = stop\n")

    FONT = cv2.FONT_HERSHEY_DUPLEX

    with mp_holistic.Holistic(min_detection_confidence=0.55,
                               min_tracking_confidence=0.55,
                               model_complexity=1) as holistic:
        sample_count = 0
        last_left_hand = None
        last_right_hand = None
        lost_frames_left = 0
        lost_frames_right = 0
        MAX_LOST_FRAMES = 4

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Anatomical Left/Right Disambiguation & EMA Smoothing
            last_left_hand, last_right_hand, lost_frames_left, lost_frames_right = resolve_and_smooth_hands(
                results, last_left_hand, last_right_hand, lost_frames_left, lost_frames_right, max_lost=MAX_LOST_FRAMES, alpha=0.70
            )

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80,255,140), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,180,80), thickness=2))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,100,80), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(200,50,30), thickness=2))
            if results.face_landmarks:
                mp_drawing.draw_landmarks(frame, results.face_landmarks,
                    mp_holistic.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(60,60,60), thickness=1))

            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, 70), (20,20,20), -1)
            cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

            title_txt = f"WORD: {word_name.upper()}  [{start_counter + sample_count}/{start_counter + num_samples}]"
            cv2.putText(frame, title_txt, (14, 28), FONT, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Press SPACE to start recording sample | Q to cancel",
                        (14, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

            cv2.imshow("FSL Word Collector", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q') or k == ord('Q'):
                break

            if k == 32:  # SPACE
                for cd in range(3, 0, -1):
                    t_end = time.time() + 0.95
                    while time.time() < t_end:
                        ret2, f2 = cap.read()
                        if not ret2:
                            break
                        f2 = cv2.flip(f2, 1)
                        H2, W2 = f2.shape[:2]
                        cv2.circle(f2, (W2//2, H2//2), 70, (0,140,255), -1)
                        cv2.putText(f2, str(cd), (W2//2 - 20, H2//2 + 25),
                                    FONT, 2.5, (255,255,255), 5, cv2.LINE_AA)
                        cv2.imshow("FSL Word Collector", f2)
                        cv2.waitKey(15)

                hand_seq   = []
                anchor_seq = []
                face_list  = []
                ok         = True

                for frame_idx in range(SEQUENCE_LEN):
                    ret3, f3 = cap.read()
                    if not ret3:
                        ok = False
                        break
                    f3 = cv2.flip(f3, 1)
                    H3, W3 = f3.shape[:2]

                    rgb3 = cv2.cvtColor(f3, cv2.COLOR_BGR2RGB)
                    res3 = holistic.process(rgb3)

                    # Anatomical Left/Right Disambiguation & EMA Smoothing
                    last_left_hand, last_right_hand, lost_frames_left, lost_frames_right = resolve_and_smooth_hands(
                        res3, last_left_hand, last_right_hand, lost_frames_left, lost_frames_right, max_lost=MAX_LOST_FRAMES, alpha=0.70
                    )

                    hand_seq.append(extract_hand_frame(res3))
                    anchor_seq.append(get_raw_hand_anchors(res3))
                    face_list.append(extract_face_frame(res3))

                    if res3.left_hand_landmarks:
                        mp_drawing.draw_landmarks(f3, res3.left_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(80,255,140), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0,180,80), thickness=2))
                    if res3.right_hand_landmarks:
                        mp_drawing.draw_landmarks(f3, res3.right_hand_landmarks,
                            mp_holistic.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(255,100,80), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(200,50,30), thickness=2))
                    if res3.face_landmarks:
                        mp_drawing.draw_landmarks(f3, res3.face_landmarks,
                            mp_holistic.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(60,60,60), thickness=1))

                    ov3 = f3.copy()
                    cv2.rectangle(ov3, (0,0), (W3, 58), (15,15,15), -1)
                    cv2.addWeighted(ov3, 0.60, f3, 0.40, 0, f3)
                    rec_txt = f"RECORDING  {frame_idx+1}/{SEQUENCE_LEN}"
                    cv2.putText(f3, rec_txt, (14, 36), FONT, 0.8, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(f3, rec_txt, (14, 36), FONT, 0.8, (0,0,255), 2, cv2.LINE_AA)

                    bar_w = int((W3 - 28) * (frame_idx + 1) / SEQUENCE_LEN)
                    cv2.rectangle(f3, (14, H3-18), (W3-14, H3-8), (50,50,50), -1)
                    cv2.rectangle(f3, (14, H3-18), (14+bar_w, H3-8), (0,0,220), -1)

                    cv2.imshow("FSL Word Collector", f3)
                    cv2.waitKey(20)

                if ok and len(hand_seq) == SEQUENCE_LEN:
                    idx    = start_counter + sample_count
                    h_path = os.path.join(word_dir, f"{idx}_hand.npy")
                    f_path = os.path.join(word_dir, f"{idx}_face.npy")

                    motion_dyn = build_motion_dynamics_and_trajectory(anchor_seq)
                    full_hand  = np.concatenate(
                        [np.array(hand_seq, dtype=np.float32), motion_dyn], axis=1)

                    np.save(h_path, full_hand.astype(np.float32))
                    np.save(f_path, np.mean(np.array(face_list, dtype=np.float32), axis=0))
                    np.save(os.path.join(word_dir, f"{idx}_face_seq.npy"),
                            np.array(face_list, dtype=np.float32))

                    print(f"  [OK] Saved sample {idx}")
                    sample_count += 1

                    ret4, f4 = cap.read()
                    if ret4:
                        f4 = cv2.flip(f4, 1)
                        H4, W4 = f4.shape[:2]
                        ov4 = f4.copy()
                        cv2.rectangle(ov4, (0,0), (W4,H4), (0,120,0), -1)
                        cv2.addWeighted(ov4, 0.25, f4, 0.75, 0, f4)
                        done_txt = f"Saved! ({sample_count}/{num_samples})"
                        cv2.putText(f4, done_txt, (14, H4//2), FONT, 1.2, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(f4, done_txt, (14, H4//2), FONT, 1.2, (0,255,100), 3, cv2.LINE_AA)
                        cv2.imshow("FSL Word Collector", f4)
                        cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()
    remove_outdated_models()
    print(f"\n[OK] Done recording '{word_name}'. Total samples: {start_counter + sample_count}")

def main():
    while True:
        words = [d for d in os.listdir(DATA_PATH)
                 if os.path.isdir(os.path.join(DATA_PATH, d))]
        counts = {}
        for w in words:
            wd = os.path.join(DATA_PATH, w)
            counts[w] = len([f for f in os.listdir(wd) if f.endswith('_hand.npy')])

        print("\n==========================================")
        print("  FSL WORD DATASET COLLECTOR  (Dual-Gate)")
        print("==========================================")
        if words:
            for w, c in counts.items():
                print(f"  {w:20s}  {c} samples")
        else:
            print("  No words recorded yet.")
        print("\n1. Record / Add samples for a word")
        print("2. Delete a specific word")
        print("3. Delete ALL words")
        print("4. Exit")

        choice = input("\nSelect (1-4): ").strip()

        if choice == '1':
            wn = input("Word label (e.g. hello, thank_you): ").strip().lower()
            if wn:
                ns = input("Number of samples (default 20): ").strip()
                ns = int(ns) if ns.isdigit() else 20
                record_word(wn, ns)
                manage_word_icon(wn)
        elif choice == '2':
            wn = input("Word to DELETE: ").strip().lower()
            if wn:
                reset_single_word(wn)
                # Auto-remove icon entry when word is deleted
                _imap = load_icon_map()
                if wn in _imap:
                    _imap.pop(wn)
                    save_icon_map(_imap)
                    print(f"[OK] Removed icon entry for '{wn}' from icon_map.json.")
        elif choice == '3':
            c = input("Delete ALL words? (y/n): ").strip().lower()
            if c == 'y':
                reset_all_words()
        elif choice == '4':
            break
        else:
            print("Invalid option.")

if __name__ == '__main__':
    main()
