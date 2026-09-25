import os
import cv2
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

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(ROOT_DIR, 'data', 'word_dataset')
MODELS_DIR     = os.path.join(ROOT_DIR, 'models')

SEQUENCE_LEN   = 40       # hand motion frames
HAND_FEATURES  = 132      # 126 shape (21 lm * 3 * 2 hands) + 6 wrist-trajectory (3 * 2 hands)
FACE_FEATURES  = 1404     # 468 face landmarks * 3

os.makedirs(DATA_PATH, exist_ok=True)

# ─── Feature Helpers ────────────────────────────────────────────────────────

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


def extract_face_frame(results):
    out = []
    if results.face_landmarks:
        nose = results.face_landmarks.landmark[1]
        for lm in results.face_landmarks.landmark:
            out.extend([lm.x - nose.x, lm.y - nose.y, lm.z - nose.z])
    else:
        out.extend([0.0] * 1404)
    return out

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

    with mp_holistic.Holistic(min_detection_confidence=0.7,
                               min_tracking_confidence=0.7,
                               model_complexity=1) as holistic:
        sample_count = 0
        last_left_hand = None
        last_right_hand = None
        lost_frames_left = 0
        lost_frames_right = 0
        MAX_LOST_FRAMES = 5

        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

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

            if results.left_hand_landmarks and results.right_hand_landmarks:
                lw = results.left_hand_landmarks.landmark[0]
                rw = results.right_hand_landmarks.landmark[0]
                wrist_dist = math.sqrt((lw.x - rw.x)**2 + (lw.y - rw.y)**2 + (lw.z - rw.z)**2)
                if wrist_dist < 0.12:
                    if lost_frames_left > 0 and lost_frames_right == 0:
                        results.left_hand_landmarks = None
                        last_left_hand = None
                    elif lost_frames_right > 0 and lost_frames_left == 0:
                        results.right_hand_landmarks = None
                        last_right_hand = None
                    else:
                        drop_left = (lw.x > 0.5)
                        if drop_left:
                            results.left_hand_landmarks = None
                            last_left_hand = None
                        else:
                            results.right_hand_landmarks = None
                            last_right_hand = None

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
                wrist_seq  = []
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

                    if res3.left_hand_landmarks:
                        last_left_hand = res3.left_hand_landmarks
                        lost_frames_left = 0
                    else:
                        lost_frames_left += 1
                        if lost_frames_left <= MAX_LOST_FRAMES:
                            res3.left_hand_landmarks = last_left_hand
                        else:
                            last_left_hand = None

                    if res3.right_hand_landmarks:
                        last_right_hand = res3.right_hand_landmarks
                        lost_frames_right = 0
                    else:
                        lost_frames_right += 1
                        if lost_frames_right <= MAX_LOST_FRAMES:
                            res3.right_hand_landmarks = last_right_hand
                        else:
                            last_right_hand = None

                    if res3.left_hand_landmarks and res3.right_hand_landmarks:
                        lw = res3.left_hand_landmarks.landmark[0]
                        rw = res3.right_hand_landmarks.landmark[0]
                        wrist_dist = math.sqrt((lw.x - rw.x)**2 + (lw.y - rw.y)**2 + (lw.z - rw.z)**2)
                        if wrist_dist < 0.12:
                            if lost_frames_left > 0 and lost_frames_right == 0:
                                res3.left_hand_landmarks = None
                                last_left_hand = None
                            elif lost_frames_right > 0 and lost_frames_left == 0:
                                res3.right_hand_landmarks = None
                                last_right_hand = None
                            else:
                                drop_left = (lw.x > 0.5)
                                if drop_left:
                                    res3.left_hand_landmarks = None
                                    last_left_hand = None
                                else:
                                    res3.right_hand_landmarks = None
                                    last_right_hand = None

                    hand_seq.append(extract_hand_frame(res3))
                    wrist_seq.append(get_raw_wrist(res3))
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

                    traj = build_wrist_trajectory(wrist_seq)
                    full_hand = np.concatenate(
                        [np.array(hand_seq, dtype=np.float32), traj], axis=1)

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
        elif choice == '2':
            wn = input("Word to DELETE: ").strip().lower()
            if wn:
                reset_single_word(wn)
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
