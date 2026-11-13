import os
import cv2
import csv
import json
import time
import math
import numpy as np
import pandas as pd
import mediapipe as mp

try:
    import mediapipe.solutions.hands as mp_hands
    import mediapipe.solutions.drawing_utils as mp_drawing
    import mediapipe.solutions.drawing_styles as mp_drawing_styles
except ModuleNotFoundError:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

# Root directory reference
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
CSV_FILE   = os.path.join(DATA_DIR, 'fsl_dataset.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
ICONS_DIR  = os.path.join(ROOT_DIR, 'assets', 'fsl_icons')
ICON_MAP_FILE = os.path.join(ROOT_DIR, 'icon_map.json')
MODEL_FILE = os.path.join(MODELS_DIR, 'fsl_model.pkl')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_HEADER = ['label'] + [f'{coord}_{i}' for i in range(21) for coord in ('x', 'y', 'z')]

# ─── Dataset & Model Management ─────────────────────────────────────────────

def ensure_csv_exists():
    """Ensure the dataset CSV file exists with the standard header."""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        with open(CSV_FILE, mode='w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

def remove_outdated_models():
    """Removes stale letter model whenever the dataset changes so retrain is forced."""
    if os.path.exists(MODEL_FILE):
        try:
            os.remove(MODEL_FILE)
            print(f"[!] Removed outdated letter model '{os.path.basename(MODEL_FILE)}'. Retrain required.")
        except Exception as e:
            print(f"[!] Could not remove '{MODEL_FILE}': {e}")

def get_letter_counts():
    """Returns a dict of letter labels and their sample counts, sorted alphabetically."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty or 'label' not in df.columns:
            return {}, 0
        counts = df['label'].astype(str).value_counts().to_dict()
        sorted_counts = {k: counts[k] for k in sorted(counts.keys())}
        return sorted_counts, len(df)
    except Exception as e:
        print(f"[!] Error reading dataset: {e}")
        return {}, 0

def delete_single_letter(letter_name):
    """Deletes all samples corresponding to a specific letter."""
    ensure_csv_exists()
    letter_name = letter_name.strip().upper()
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty or 'label' not in df.columns:
            print("[!] Dataset is empty.")
            return

        initial_count = len(df)
        df_filtered = df[df['label'].astype(str).str.upper() != letter_name]
        removed_count = initial_count - len(df_filtered)

        if removed_count == 0:
            print(f"[!] Letter '{letter_name}' not found in dataset.")
            return

        df_filtered.to_csv(CSV_FILE, index=False)
        print(f"[OK] Successfully deleted letter '{letter_name}' ({removed_count} samples removed).")
        remove_outdated_models()
    except Exception as e:
        print(f"[!] Error deleting letter '{letter_name}': {e}")

def redo_single_letter(letter_name, num_samples=30):
    """
    Clears existing samples for the chosen letter, then immediately
    starts recording fresh samples to teach the letter.
    """
    letter_name = letter_name.strip().upper()
    print(f"\n[*] Preparing to redo signs for letter '{letter_name}'...")
    delete_single_letter(letter_name)
    print(f"[*] Starting fresh recording for '{letter_name}' ({num_samples} samples)...")
    record_letter(letter_name, num_samples)

def delete_last_sample():
    """Removes the most recently added sample from the dataset."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            print("[!] Dataset is already empty.")
            return
        last_label = str(df.iloc[-1]['label'])
        df = df.iloc[:-1]
        df.to_csv(CSV_FILE, index=False)
        print(f"[OK] Undid / deleted last sample (Letter: '{last_label}'). Remaining samples: {len(df)}.")
        remove_outdated_models()
    except Exception as e:
        print(f"[!] Error undoing last sample: {e}")

def reset_all_letters():
    """Wipes all letter data from the CSV."""
    with open(CSV_FILE, mode='w', newline='') as f:
        csv.writer(f).writerow(CSV_HEADER)
    print("[OK] Reset ALL letter datasets to empty.")
    remove_outdated_models()

# ─── Landmark Normalization Helper ──────────────────────────────────────────

def extract_normalized_hand(landmarks):
    """
    Extracts 63 scale-normalized 3D coordinates relative to the wrist (landmark 0),
    scaled by wrist-to-middle-knuckle (landmark 9) distance.
    Matches exact training & inference format expected by train_letter_model and main.py.
    """
    wrist = landmarks[0]
    wx, wy, wz = wrist.x, wrist.y, wrist.z

    # Scale normalization (wrist to middle knuckle landmark 9)
    scale = math.sqrt(
        (landmarks[9].x - wx)**2 +
        (landmarks[9].y - wy)**2 +
        (landmarks[9].z - wz)**2
    )
    if scale == 0:
        scale = 1.0

    row = []
    for lm in landmarks:
        row.extend([
            (lm.x - wx) / scale,
            (lm.y - wy) / scale,
            (lm.z - wz) / scale
        ])
    return row

# ─── Live Camera Data Collector ─────────────────────────────────────────────

def record_letter(letter_name, num_samples=30):
    """
    Interactive camera interface to teach and collect letter samples.
    Features:
    - Real-time mirrored hand landmark rendering with connections.
    - SPACE: Capture 1 sample.
    - C: Auto-Burst mode with 3..2..1 countdown.
    - Z / Backspace: Undo last sample in this session.
    - R: Redo/reset this session for this letter.
    - Q / ESC: Finish recording and save.
    """
    ensure_csv_exists()
    letter_name = letter_name.strip().upper()

    counts, _ = get_letter_counts()
    initial_existing = counts.get(letter_name, 0)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Error: Could not open camera device 0.")
        return

    print(f"\n==========================================")
    print(f"  RECORDING LETTER: '{letter_name}'")
    print(f"==========================================")
    print(f"Existing samples: {initial_existing} | Target new samples: {num_samples}")
    print("Controls:")
    print("  SPACE       : Capture 1 sample")
    print("  C           : Auto-Burst capture (10 samples with 3s countdown)")
    print("  Z/Backspace : Undo last captured sample")
    print("  R           : Redo current session (wipe session samples)")
    print("  Q / ESC     : Finish & return to menu\n")

    FONT = cv2.FONT_HERSHEY_DUPLEX
    session_rows = []     # Tracks rows saved during this session for easy session-undo/redo
    session_count = 0
    toast_msg = ""
    toast_color = (0, 255, 100)
    toast_expiry = 0.0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.60,
        min_tracking_confidence=0.60
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            hand_detected = False
            current_row = None

            if results.multi_hand_landmarks:
                hand_detected = True
                hand_lms = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 180, 80), thickness=2)
                )
                current_row = extract_normalized_hand(hand_lms.landmark)

            # Top translucent HUD overlay
            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, 70), (18, 18, 18), -1)
            cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)

            total_now = initial_existing + session_count
            target_total = initial_existing + num_samples

            title_txt = f"LETTER: {letter_name}  [{total_now}/{target_total}]  (Session: +{session_count})"
            cv2.putText(frame, title_txt, (14, 26), FONT, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

            help_txt = "SPACE: Add 1 | C: Auto-Burst | Z: Undo | R: Redo Session | Q: Done"
            cv2.putText(frame, help_txt, (14, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            # Status Badge on Top Right
            if hand_detected:
                status_txt = "[HAND READY]"
                badge_color = (0, 230, 100)
            else:
                status_txt = "[NO HAND DETECTED]"
                badge_color = (0, 140, 255)
            cv2.putText(frame, status_txt, (W - 220, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, badge_color, 2, cv2.LINE_AA)

            # Bottom Progress Bar
            progress = min(1.0, max(0.0, session_count / max(1, num_samples)))
            bar_w = int((W - 28) * progress)
            cv2.rectangle(frame, (14, H - 20), (W - 14, H - 10), (50, 50, 50), -1)
            cv2.rectangle(frame, (14, H - 20), (14 + bar_w, H - 10), (0, 210, 90), -1)

            # Temporary notification toast
            now = time.time()
            if now < toast_expiry and toast_msg:
                t_ov = frame.copy()
                cv2.rectangle(t_ov, (W // 2 - 170, H // 2 - 30), (W // 2 + 170, H // 2 + 25), (10, 10, 10), -1)
                cv2.addWeighted(t_ov, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (W // 2 - 170, H // 2 - 30), (W // 2 + 170, H // 2 + 25), toast_color, 2)
                cv2.putText(frame, toast_msg, (W // 2 - 150, H // 2 + 8), FONT, 0.70, toast_color, 2, cv2.LINE_AA)

            cv2.imshow("FSL Letters Data Collector", frame)
            k = cv2.waitKey(10) & 0xFF

            # Exit
            if k == ord('q') or k == ord('Q') or k == 27:
                break

            # 1. Manual Single-Sample Capture (SPACE)
            if k == 32:  # SPACE
                if hand_detected and current_row:
                    with open(CSV_FILE, mode='a', newline='') as f:
                        csv.writer(f).writerow([letter_name] + current_row)
                    session_rows.append(current_row)
                    session_count += 1
                    toast_msg = f"Saved! ({session_count}/{num_samples})"
                    toast_color = (0, 255, 100)
                    toast_expiry = now + 0.9
                    print(f"  [+] Saved sample {session_count} for '{letter_name}'")
                else:
                    toast_msg = "No Hand Detected!"
                    toast_color = (0, 100, 255)
                    toast_expiry = now + 1.0

            # 2. Auto-Burst Mode (C)
            elif k == ord('c') or k == ord('C'):
                # 3-second countdown
                burst_cancelled = False
                for cd in range(3, 0, -1):
                    t_end = time.time() + 0.95
                    while time.time() < t_end:
                        r_cd, f_cd = cap.read()
                        if not r_cd:
                            break
                        f_cd = cv2.flip(f_cd, 1)
                        H_cd, W_cd = f_cd.shape[:2]
                        cv2.circle(f_cd, (W_cd // 2, H_cd // 2), 65, (0, 140, 255), -1)
                        cv2.putText(f_cd, str(cd), (W_cd // 2 - 18, H_cd // 2 + 22),
                                    FONT, 2.2, (255, 255, 255), 4, cv2.LINE_AA)
                        cv2.putText(f_cd, f"HOLD SIGN '{letter_name}'!", (W_cd // 2 - 130, H_cd // 2 - 80),
                                    FONT, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow("FSL Letters Data Collector", f_cd)
                        key_cd = cv2.waitKey(15) & 0xFF
                        if key_cd == ord('q') or key_cd == 27:
                            burst_cancelled = True
                            break
                    if burst_cancelled:
                        break

                if not burst_cancelled:
                    burst_target = min(10, num_samples - session_count) if session_count < num_samples else 10
                    burst_saved = 0
                    for _ in range(burst_target):
                        r_b, f_b = cap.read()
                        if not r_b:
                            break
                        f_b = cv2.flip(f_b, 1)
                        rgb_b = cv2.cvtColor(f_b, cv2.COLOR_BGR2RGB)
                        res_b = hands.process(rgb_b)

                        if res_b.multi_hand_landmarks:
                            h_b = res_b.multi_hand_landmarks[0]
                            mp_drawing.draw_landmarks(
                                f_b, h_b, mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                                mp_drawing.DrawingSpec(color=(0, 180, 80), thickness=2)
                            )
                            row_b = extract_normalized_hand(h_b.landmark)
                            with open(CSV_FILE, mode='a', newline='') as f:
                                csv.writer(f).writerow([letter_name] + row_b)
                            session_rows.append(row_b)
                            session_count += 1
                            burst_saved += 1

                        ov_b = f_b.copy()
                        cv2.rectangle(ov_b, (0, 0), (W, 60), (15, 15, 15), -1)
                        cv2.addWeighted(ov_b, 0.65, f_b, 0.35, 0, f_b)
                        cv2.putText(f_b, f"BURST RECORDING: {burst_saved}/{burst_target}",
                                    (14, 38), FONT, 0.8, (0, 220, 255), 2, cv2.LINE_AA)
                        cv2.imshow("FSL Letters Data Collector", f_b)
                        cv2.waitKey(100)

                    toast_msg = f"Burst complete! (+{burst_saved})"
                    toast_color = (0, 255, 200)
                    toast_expiry = time.time() + 1.2
                    print(f"  [+] Auto-burst finished: added {burst_saved} samples for '{letter_name}'")

            # 3. Undo Last Sample (Z or Backspace)
            elif k == ord('z') or k == ord('Z') or k == 8:
                if session_count > 0:
                    delete_last_sample()
                    session_rows.pop()
                    session_count -= 1
                    toast_msg = f"Undid last sample! ({session_count})"
                    toast_color = (0, 180, 255)
                    toast_expiry = now + 1.0
                else:
                    toast_msg = "Nothing to undo in this session!"
                    toast_color = (0, 100, 255)
                    toast_expiry = now + 1.0

            # 4. Redo Session (R)
            elif k == ord('r') or k == ord('R'):
                if session_count > 0:
                    try:
                        df = pd.read_csv(CSV_FILE)
                        if len(df) >= session_count:
                            df = df.iloc[:-session_count]
                            df.to_csv(CSV_FILE, index=False)
                            session_rows.clear()
                            session_count = 0
                            toast_msg = "Session reset! Ready to re-record."
                            toast_color = (255, 180, 0)
                            toast_expiry = now + 1.5
                            print(f"  [*] Redid session: removed all {session_count} session samples for '{letter_name}'.")
                    except Exception as e:
                        print(f"[!] Error resetting session: {e}")
                else:
                    toast_msg = "Session is already empty."
                    toast_color = (200, 200, 200)
                    toast_expiry = now + 1.0

    cap.release()
    cv2.destroyAllWindows()

    if session_count > 0:
        remove_outdated_models()

    counts, _ = get_letter_counts()
    new_total = counts.get(letter_name, 0)
    print(f"\n[OK] Finished recording for '{letter_name}'. Total samples for this letter: {new_total}\n")

# ─── Training Integration ───────────────────────────────────────────────────

def run_trainer():
    """Runs the letter trainer model directly."""
    try:
        from Letters_Learning.train_letter_model import train_letter_model
    except ImportError:
        try:
            from train_letter_model import train_letter_model
        except ImportError:
            train_letter_model = None

    if train_letter_model:
        train_letter_model()
    else:
        print("[!] Could not import 'train_letter_model'. Please run it directly via 'python Letters_Learning/train_letter_model.py'")

# ─── Main Menu ──────────────────────────────────────────────────────────────

def main():
    while True:
        counts, total_samples = get_letter_counts()
        model_exists = os.path.exists(MODEL_FILE)

        print("\n==========================================")
        print("   FSL LETTER DATASET COLLECTOR & TRAINER  ")
        print("==========================================")
        if counts:
            for lbl, c in counts.items():
                print(f"    Letter '{lbl}': {c:4d} samples")
            print("  ----------------------------------------")
            print(f"  TOTAL: {len(counts)} letters, {total_samples} samples")
        else:
            print("  No letters recorded yet in dataset.")

        model_status = "Ready" if model_exists else "Not Trained / Retrain Needed"
        print(f"  Model Status: {model_status}")
        print("==========================================")
        print("1. Record / Add samples for a letter")
        print("2. Redo signs for a letter (Reset & Re-record)")
        print("3. Delete a specific letter")
        print("4. Delete last recorded sample (Undo)")
        print("5. Delete ALL letters (Reset dataset)")
        print("6. Train Letter Model (Random Forest)")
        print("7. Exit")

        choice = input("\nSelect (1-7): ").strip()

        if choice == '1':
            lbl = input("Letter to teach / record (e.g. A, B, C): ").strip().upper()
            if lbl:
                ns = input("Number of samples to record (default 30): ").strip()
                ns = int(ns) if ns.isdigit() and int(ns) > 0 else 30
                record_letter(lbl, ns)
                manage_letter_icon(lbl)

        elif choice == '2':
            if not counts:
                print("[!] No letters in dataset to redo.")
                continue
            lbl = input("Letter to REDO from scratch: ").strip().upper()
            if lbl:
                if lbl in counts:
                    confirm = input(f"Are you sure you want to delete all {counts[lbl]} samples for '{lbl}' and re-record? (y/n): ").strip().lower()
                    if confirm == 'y':
                        ns = input("Number of new samples to record (default 30): ").strip()
                        ns = int(ns) if ns.isdigit() and int(ns) > 0 else 30
                        redo_single_letter(lbl, ns)
                else:
                    print(f"[!] Letter '{lbl}' not found in current dataset.")

        elif choice == '3':
            if not counts:
                print("[!] No letters in dataset to delete.")
                continue
            lbl = input("Letter to DELETE: ").strip().upper()
            if lbl:
                if lbl in counts:
                    confirm = input(f"Confirm deleting all {counts[lbl]} samples for '{lbl}'? (y/n): ").strip().lower()
                    if confirm == 'y':
                        delete_single_letter(lbl)
                        # Auto-remove icon entry when letter is deleted
                        _imap = load_icon_map()
                        _mkey = lbl.lower()
                        if _mkey in _imap:
                            _imap.pop(_mkey)
                            save_icon_map(_imap)
                            print(f"[OK] Removed icon entry for '{lbl}' from icon_map.json.")
                else:
                    print(f"[!] Letter '{lbl}' not found in current dataset.")

        elif choice == '4':
            delete_last_sample()

        elif choice == '5':
            confirm = input("Are you sure you want to delete ALL letter datasets? (y/n): ").strip().lower()
            if confirm == 'y':
                reset_all_letters()

        elif choice == '6':
            run_trainer()

        elif choice == '7':
            print("Exiting Letters Data Collector. Goodbye!")
            break
        else:
            print("[!] Invalid option. Please enter a number from 1 to 7.")

if __name__ == '__main__':
    main()
