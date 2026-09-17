import os
import cv2
import shutil
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from project_paths import asset_path, data_path, model_path

DATA_PATH = str(data_path('word_dataset'))
MODEL_PATH = str(model_path('word_model.pkl'))
SEQUENCE_LENGTH = 40  # 40 frames per word sample

# Ensure dataset directory exists
os.makedirs(DATA_PATH, exist_ok=True)

def remove_outdated_model():
    """Deletes the old trained model so predict.py doesn't show old words."""
    if os.path.exists(MODEL_PATH):
        try:
            os.remove(MODEL_PATH)
            print(f"[!] Removed stale '{MODEL_PATH}'. You will need to retrain via train_words.py.")
        except Exception as e:
            print(f"[!] Could not remove '{MODEL_PATH}': {e}")

def reset_single_word(word_name):
    """Deletes a specific word directory and invalidates the word model."""
    target_dir = os.path.join(DATA_PATH, word_name)
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"[✓] Successfully deleted word folder: '{word_name}'")
        remove_outdated_model()
    else:
        print(f"[!] Word '{word_name}' not found in '{DATA_PATH}'.")

def reset_all_words():
    """Wipes out the entire word dataset and deletes the word model."""
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH, exist_ok=True)
        print("[✓] Successfully reset ALL word datasets.")
        remove_outdated_model()
    else:
        print("[!] Dataset directory does not exist.")

def record_word(word_name, num_samples=15):
    """Records motion sequences for a specific word using MediaPipe."""
    word_dir = os.path.join(DATA_PATH, word_name)
    os.makedirs(word_dir, exist_ok=True)

    # Count existing sequences
    existing_files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
    start_counter = len(existing_files)

    base_options = python.BaseOptions(model_asset_path=str(asset_path('hand_landmarker.task')))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    print(f"\n--- Recording '{word_name}' ({num_samples} sequences) ---")
    print("Press SPACE to start recording a sequence. Press 'q' to stop.")

    sample_count = 0
    while sample_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        cv2.putText(frame, f"Word: '{word_name}' | Captured: {sample_count}/{num_samples}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to record next sample | 'q' to quit", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Record Word Dataset", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == 32:  # SPACE KEY pressed
            sequence = []
            print(f"  -> Recording sample {sample_count + 1}/{num_samples}... Get ready!")
            
            # 3-second countdown visual
            for c in range(3, 0, -1):
                temp_ret, temp_frame = cap.read()
                if temp_ret:
                    temp_frame = cv2.flip(temp_frame, 1)
                    cv2.putText(temp_frame, f"Starting in {c}...", (w//2 - 100, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("Record Word Dataset", temp_frame)
                    cv2.waitKey(1000)

            # Record sequence frames
            for frame_idx in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)

                combined_landmarks = []
                if detection_result.hand_landmarks:
                    hands = detection_result.hand_landmarks
                    primary_hand = hands[0]
                    wrist_x, wrist_y, wrist_z = primary_hand[0].x, primary_hand[0].y, primary_hand[0].z

                    for hand_idx in range(2):
                        if hand_idx < len(hands):
                            for lm in hands[hand_idx]:
                                combined_landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                        else:
                            combined_landmarks.extend([0.0] * 63)
                else:
                    combined_landmarks = [0.0] * 126

                sequence.append(combined_landmarks)

                cv2.putText(frame, f"RECORDING ({frame_idx + 1}/{SEQUENCE_LENGTH})", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Record Word Dataset", frame)
                cv2.waitKey(30)

            if len(sequence) == SEQUENCE_LENGTH:
                file_name = f"{start_counter + sample_count}.npy"
                np.save(os.path.join(word_dir, file_name), np.array(sequence))
                print(f"  [✓] Saved sample: {file_name}")
                sample_count += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    while True:
        existing_words = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
        print("\n==========================================")
        print("  FSL WORD DATASET COLLECTOR & MANAGER")
        print("==========================================")
        print(f"Current Recorded Words: {existing_words if existing_words else 'None'}")
        print("1. Record / Add a Word")
        print("2. Reset / Delete a Specific Word")
        print("3. Reset ALL Words (Clear Dataset)")
        print("4. Exit")
        
        choice = input("Select an option (1-4): ").strip()

        if choice == '1':
            word_name = input("Enter the word label to record (e.g., 'hello', 'thank_you'): ").strip().lower()
            if word_name:
                num_samples = input("Number of samples to capture (default 15): ").strip()
                num_samples = int(num_samples) if num_samples.isdigit() else 15
                record_word(word_name, num_samples)
        elif choice == '2':
            word_name = input("Enter the word label you want to DELETE: ").strip().lower()
            if word_name:
                reset_single_word(word_name)
        elif choice == '3':
            confirm = input("Are you sure you want to DELETE ALL WORDS? (y/n): ").strip().lower()
            if confirm == 'y':
                reset_all_words()
        elif choice == '4':
            print("Exiting collector menu.")
            break
        else:
            print("Invalid option. Please enter 1, 2, 3, or 4.")

if __name__ == '__main__':
    main()