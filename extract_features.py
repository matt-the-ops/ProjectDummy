import numpy as np
import mediapipe as mp

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_landmarks(results):
    """
    Extracts flattened spatial coordinates for left hand, right hand, and face keypoints.
    - Left Hand: 21 landmarks * 3 (x, y, z) = 63 values
    - Right Hand: 21 landmarks * 3 (x, y, z) = 63 values
    - Face: 468 landmarks * 3 (x, y, z) = 1404 values
    Returns a single 1D NumPy array combining all features.
    """
    # 1. Left Hand Landmarks (63 floats)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res.x, res.y, res.z in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)

    # 2. Right Hand Landmarks (63 floats)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res.x, res.y, res.z in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    # 3. Facial Expression Landmarks (1404 floats)
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res.x, res.y, res.z in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)

    # Combined Feature Vector Length: 1530 floats per frame
    return np.concatenate([lh, rh, face])