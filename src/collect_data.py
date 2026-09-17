import cv2
import csv
import os
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from project_paths import asset_path, data_path

CSV_FILE = str(data_path('fsl_dataset.csv'))

# Create CSV header if file doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')]
        writer.writerow(header)

base_options = python.BaseOptions(model_asset_path=str(asset_path('hand_landmarker.task')))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
print("Hold an FSL sign and press any letter/number key to save frames.")
print("Press 'ESC' key to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    # Capture keypress once per frame
    key = cv2.waitKey(1) & 0xFF

    # Press ESC key (27) to exit script
    if key == 27:
        break

    if detection_result.hand_landmarks:
        landmarks = detection_result.hand_landmarks[0]
        wrist_x, wrist_y, wrist_z = landmarks[0].x, landmarks[0].y, landmarks[0].z

        # Scale normalization (wrist to middle knuckle)
        scale = math.sqrt((landmarks[9].x - wrist_x)**2 + (landmarks[9].y - wrist_y)**2 + (landmarks[9].z - wrist_z)**2)
        if scale == 0:
            scale = 1.0

        row = []
        for lm in landmarks:
            row.extend([(lm.x - wrist_x) / scale, (lm.y - wrist_y) / scale, (lm.z - wrist_z) / scale])

        for lm in landmarks:
            px, py = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        # Save data row if a valid key was pressed
        if key != 255:
            label = chr(key).upper()
            with open(CSV_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([label] + row)
            print(f"Saved 1 frame for label: {label}")

    cv2.imshow("FSL Data Collector", frame)

cap.release()
cv2.destroyAllWindows()