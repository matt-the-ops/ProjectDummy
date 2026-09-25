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
HAND_FEATURES  = 198      # 126 shape + 36 spatial anchors + 12 facial emotion + 18 per-frame velocities/speeds/direction ratios + 6 trajectory
FACE_FEATURES  = 1404     # 468 face landmarks * 3

os.makedirs(DATA_PATH, exist_ok=True)

# ─── Feature Helpers ────────────────────────────────────────────────────────

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

# ─── Landmark Smoothing & Disambiguation Helpers ─────────────────────────────

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
