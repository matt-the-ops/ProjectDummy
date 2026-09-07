import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define hand joint connections manually for the Tasks API
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky finger
    (0, 17)                                    # Palm base
]

# Initialize Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Tracking skeleton and extracting coordinates. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # 1. Extract raw normalized keypoints (63 float values: x, y, z for 21 joints)
            landmark_features = []
            pixel_coords = []

            for lm in hand_landmarks:
                landmark_features.extend([lm.x, lm.y, lm.z])
                px, py = int(lm.x * w), int(lm.y * h)
                pixel_coords.append((px, py))

            # 2. Draw skeleton connecting lines
            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (255, 255, 255), 2)

            # 3. Draw joint dots
            for px, py in pixel_coords:
                cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

            # Display keypoint count on frame
            cv2.putText(frame, f"Features Captured: {len(landmark_features)}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("FSL Prototype - Landmark Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()