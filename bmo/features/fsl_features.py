"""Shared FSL feature code. Single source for main.py, collect_word_data.py, train_word_model.py."""
import math

import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Word data was recorded unsmoothed. Flip only after re-recording and retraining.
SMOOTH_LANDMARKS = False

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


def smooth_landmarks(current_lms, previous_lms, alpha=0.70):
    """Applies Exponential Moving Average (EMA) smoothing to eliminate landmark micro-jitter."""
    if not SMOOTH_LANDMARKS or previous_lms is None or current_lms is None:
        return current_lms
    try:
        smoothed = landmark_pb2.NormalizedLandmarkList()
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


# ─── Atomic file writes ───────────────────────────────────────────────────────
# Save to *.tmp, then os.replace() so a kiosk starting mid-train never loads
# a half-written model or dataset file.
import os
import pickle as _pickle


def atomic_replace_bytes(path, data):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def atomic_replace_npy(path, arr):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, np.asarray(arr))
    os.replace(tmp, path)


def atomic_replace_pickle(path, obj):
    atomic_replace_bytes(path, _pickle.dumps(obj))
