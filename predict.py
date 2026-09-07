import cv2
import pickle
import math
from collections import deque, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Hand skeleton line indices
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),      # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),    # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky finger
    (0, 17)                                    # Palm base
]

with open('fsl_model.pkl', 'rb') as f:
    model = pickle.load(f)

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

prediction_history = deque(maxlen=10)
cap = cv2.VideoCapture(0)

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
        landmarks = detection_result.hand_landmarks[0]
        wrist_x, wrist_y, wrist_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

        scale = math.sqrt((landmarks[9].x - wrist_x)**2 + (landmarks[9].y - wrist_y)**2 + (landmarks[9].z - wrist_z)**2)
        if scale == 0:
            scale = 1.0

        row = []
        pixel_coords = []
        for lm in landmarks:
            row.extend([(lm.x - wrist_x) / scale, (lm.y - wrist_y) / scale, (lm.z - wrist_z) / scale])
            px, py = int(lm.x * w), int(lm.y * h)
            pixel_coords.append((px, py))

        # Draw skeleton lines
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, pixel_coords[start_idx], pixel_coords[end_idx], (255, 255, 255), 2)

        # Draw landmark dots
        for px, py in pixel_coords:
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        raw_prediction = model.predict([row])[0]
        prediction_history.append(raw_prediction)
        smoothed_prediction = Counter(prediction_history).most_common(1)[0][0]

        cv2.putText(frame, f"Predicted Sign: {smoothed_prediction}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("FSL Prototype - Real-Time Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()