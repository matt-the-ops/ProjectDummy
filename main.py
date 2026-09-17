import cv2
import math
import pickle
import collections
import time
import numpy as np
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from project_paths import asset_path, model_path

def speak_text(text):
    """Re-initializes a fresh TTS engine instance per call to prevent event loop lockups."""
    if text.strip():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)  # Speed of speech
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

# Load Models
try:
    with open(model_path('fsl_model.pkl'), 'rb') as f:
        letter_data = pickle.load(f)
        # Check if loaded object is a dictionary or direct model
        if isinstance(letter_data, dict) and 'model' in letter_data:
            letter_model = letter_data['model']
        else:
            letter_model = letter_data
except FileNotFoundError:
    print("Error: 'fsl_model.pkl' not found. Please run your letter training script first.")
    exit()

try:
    with open(model_path('word_model.pkl'), 'rb') as f:
        model_data = pickle.load(f)
        if isinstance(model_data, dict) and 'model' in model_data:
            word_model = model_data['model']
            word_classes = model_data['classes']
        else:
            word_model = model_data
            word_classes = getattr(word_model, 'classes_', None)
except FileNotFoundError:
    print("Warning: 'word_model.pkl' not found. Mode 2 (Words) will not work until you train it.")
    word_model = None

# MediaPipe Setup for UP TO 2 HANDS
base_options = python.BaseOptions(model_asset_path=str(asset_path('hand_landmarker.task')))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# State Buffers & Tracking
prediction_buffer = collections.deque(maxlen=10) 
pinky_history = collections.deque(maxlen=25)     
index_history = collections.deque(maxlen=25)     
word_sequence_buffer = collections.deque(maxlen=40) # Rolling buffer for Mode 2 words

last_stable_char = ""
stable_start_time = time.time()
STABILIZATION_DELAY = 3.0  

# Cooldown Settings
GLOBAL_COOLDOWN = 1.5      
global_cooldown_end = 0.0

# Motion Locks (Mode 1)
active_motion_mode = ""  
motion_timeout = 0         
suppress_char = ""
suppression_end = 0.0

current_mode = 1  # 1 = Letters, 2 = Words
typed_output = ""        
last_appended_sign = ""    
missing_frames = 0       

# Camera Selection Setup
camera_index = 0
cap = cv2.VideoCapture(camera_index)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         
    (0, 5), (5, 6), (6, 7), (7, 8),         
    (5, 9), (9, 10), (10, 11), (11, 12),    
    (9, 13), (13, 14), (14, 15), (15, 16),  
    (13, 17), (17, 18), (18, 19), (19, 20), 
    (0, 17)                                 
]

print("Starting FSL Real-Time Translator...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Failed to read from camera index {camera_index}. Reverting to 0.")
        cap.release()
        camera_index = 0
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not access any webcam.")
            break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    current_frame_prediction = ""
    current_time = time.time()
    on_cooldown = current_time < global_cooldown_end

    if detection_result.hand_landmarks:
        missing_frames = 0 
        hands = detection_result.hand_landmarks
        primary_hand = hands[0] 

        pinky_pos = (int(primary_hand[20].x * w), int(primary_hand[20].y * h))
        index_pos = (int(primary_hand[8].x * w), int(primary_hand[8].y * h))

        # --- MODE 2: DYNAMIC WORD RECOGNITION PIPELINE ---
        if current_mode == 2 and word_model is not None:
            wrist_x, wrist_y, wrist_z = primary_hand[0].x, primary_hand[0].y, primary_hand[0].z
            combined_landmarks = []
            for hand_idx in range(2):
                if hand_idx < len(hands):
                    landmarks = hands[hand_idx]
                    for lm in landmarks:
                        combined_landmarks.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                else:
                    combined_landmarks.extend([0.0] * 63)
            
            word_sequence_buffer.append(combined_landmarks)

            if len(word_sequence_buffer) == 40 and not on_cooldown:
                flattened_seq = np.array(word_sequence_buffer).flatten().reshape(1, -1)
                predicted_word = word_model.predict(flattened_seq)[0]
                
                # Append word and trigger TTS for every recognized word
                typed_output += str(predicted_word) + " "
                speak_text(str(predicted_word))
                
                global_cooldown_end = current_time + GLOBAL_COOLDOWN
                word_sequence_buffer.clear() 

        # --- MODE 1: LETTER-BY-LETTER & MOTION PIPELINE ---
        elif current_mode == 1:
            wrist_x, wrist_y, wrist_z = primary_hand[0].x, primary_hand[0].y, primary_hand[0].z
            scale = math.sqrt((primary_hand[9].x - wrist_x)**2 + (primary_hand[9].y - wrist_y)**2 + (primary_hand[9].z - wrist_z)**2)
            if scale == 0: scale = 1.0

            row = []
            for lm in primary_hand:
                row.extend([(lm.x - wrist_x) / scale, (lm.y - wrist_y) / scale, (lm.z - wrist_z) / scale])

            raw_ml_char = letter_model.predict([row])[0]
            prediction_buffer.append(raw_ml_char)
            base_sign = collections.Counter(prediction_buffer).most_common(1)[0][0]

            is_pinky_extended = primary_hand[20].y < primary_hand[17].y
            is_index_extended = primary_hand[8].y < primary_hand[5].y

            if base_sign == 'I' and is_pinky_extended:
                if active_motion_mode != "J_READY":
                    active_motion_mode = "J_READY"
                    pinky_history.clear()
                    index_history.clear()
                motion_timeout = 0  
            elif base_sign == 'D' and is_index_extended:
                if active_motion_mode != "Z_READY":
                    active_motion_mode = "Z_READY"
                    pinky_history.clear()
                    index_history.clear()
                motion_timeout = 0  
            else:
                if active_motion_mode != "":
                    motion_timeout += 1
                    if motion_timeout > 10:  
                        active_motion_mode = ""
                        pinky_history.clear()
                        index_history.clear()

            if active_motion_mode == "J_READY":
                if not pinky_history or (pinky_pos[0] - pinky_history[-1][0])**2 + (pinky_pos[1] - pinky_history[-1][1])**2 > 25:
                    pinky_history.append(pinky_pos)
            if active_motion_mode == "Z_READY":
                if not index_history or (index_pos[0] - index_history[-1][0])**2 + (index_pos[1] - index_history[-1][1])**2 > 25:
                    index_history.append(index_pos)

            output_char = ""
            trigger_immediate = False

            if active_motion_mode == "J_READY" and len(pinky_history) >= 8:
                highest_y = min(p[1] for p in pinky_history)
                current_y = pinky_history[-1][1]
                if current_y - highest_y > 45:
                    if not on_cooldown:
                        output_char = 'J'
                        trigger_immediate = True
                        suppress_char = 'I'                      
                        suppression_end = current_time + 1.0    
                    active_motion_mode = ""
                    pinky_history.clear()

            elif active_motion_mode == "Z_READY" and len(index_history) >= 8:
                min_x = min(p[0] for p in index_history)
                max_x = max(p[0] for p in index_history)
                if max_x - min_x > 55:
                    if not on_cooldown:
                        output_char = 'Z'
                        trigger_immediate = True
                        suppress_char = 'D'                      
                        suppression_end = current_time + 1.0    
                    active_motion_mode = ""
                    index_history.clear()

            if not trigger_immediate:
                output_char = base_sign
                
                if current_time < suppression_end and output_char == suppress_char:
                    output_char = "" 

                if output_char == last_stable_char:
                    if output_char != "" and (current_time - stable_start_time) >= STABILIZATION_DELAY:
                        if not on_cooldown:
                            trigger_immediate = True
                else:
                    last_stable_char = output_char
                    stable_start_time = current_time

            if trigger_immediate and output_char != "" and not on_cooldown:
                typed_output += output_char
                speak_text(output_char) # TTS speaks confirmed letter
                
                global_cooldown_end = current_time + GLOBAL_COOLDOWN
                stable_start_time = current_time + GLOBAL_COOLDOWN  
                last_appended_sign = output_char
                
            current_frame_prediction = last_stable_char

        for hand_landmarks in hands:
            for connection in HAND_CONNECTIONS:
                p1 = hand_landmarks[connection[0]]
                p2 = hand_landmarks[connection[1]]
                cv2.line(frame, (int(p1.x * w), int(p1.y * h)), (int(p2.x * w), int(p2.y * h)), (255, 255, 255), 2)
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)

        if current_mode == 1:
            if active_motion_mode == "J_READY":
                for i in range(1, len(pinky_history)):
                    cv2.line(frame, pinky_history[i - 1], pinky_history[i], (255, 255, 0), 4)
            elif active_motion_mode == "Z_READY":
                for i in range(1, len(index_history)):
                    cv2.line(frame, index_history[i - 1], index_history[i], (255, 0, 0), 4)

    else:
        missing_frames += 1
        if missing_frames > 10:
            pinky_history.clear()
            index_history.clear()
            prediction_buffer.clear()
            last_stable_char = ""
            active_motion_mode = ""
            motion_timeout = 0
            if current_mode == 2:
                word_sequence_buffer.clear()

    # UI Overlay Setup
    mode_titles = {1: "Mode 1: LETTERS", 2: "Mode 2: WORDS"}
    cv2.putText(frame, mode_titles[current_mode], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    cv2.putText(frame, f"Cam: {camera_index}", (520, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    if on_cooldown:
        cooldown_left = global_cooldown_end - current_time
        cv2.putText(frame, f"Cooldown: {cooldown_left:.1f}s", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif current_mode == 1 and current_frame_prediction != "" and current_time >= suppression_end:
        time_left = max(0.0, STABILIZATION_DELAY - (current_time - stable_start_time))
        cv2.putText(frame, f"Hold for {time_left:.1f}s", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    elif current_mode == 2:
        cv2.putText(frame, f"Buffer: {len(word_sequence_buffer)}/40", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)

    cv2.putText(frame, f"Sign: {current_frame_prediction}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Output: {typed_output}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
    cv2.putText(frame, "Keys: 1/2(Mode)|N(Cam)|Enter(Speak)|Space|Bksp|C(Clr)", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("FSL Real-Time Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('1'): 
        current_mode = 1; typed_output = ""
    elif key == ord('2'): 
        current_mode = 2; typed_output = ""; word_sequence_buffer.clear()
        print("Switched to Mode 2: Words. Perform your dynamic sign continuously.")
    elif key == ord('n') or key == ord('N'):
        cap.release()
        camera_index = (camera_index + 1) % 3
        print(f"Switching to camera index: {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
    elif key == 13: # ENTER key speaks the entire accumulated sentence aloud
        speak_text(typed_output)
    elif key == 32: 
        typed_output += " " 
    elif key == 8 or key == 127: 
        typed_output = typed_output[:-1]
    elif key == ord('c') or key == ord('C'): 
        typed_output = ""

cap.release()
cv2.destroyAllWindows()