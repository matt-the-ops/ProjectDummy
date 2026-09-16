import os
import csv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

CSV_FILE = 'fsl_dataset.csv'
MODEL_FILE = 'fsl_model.pkl'

def train_letter_model():
    print("==========================================")
    print("    FSL LETTER MODEL TRAINER (Mode 1)     ")
    print("==========================================\n")

    if not os.path.exists(CSV_FILE):
        print(f"[!] Error: Dataset '{CSV_FILE}' not found. Run collect_data.py to collect letter samples.")
        return

    X = []
    y = []

    with open(CSV_FILE, mode='r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 64:
                continue
            label = row[0]
            try:
                features = [float(val) for val in row[1:64]]
                X.append(features)
                y.append(label)
            except ValueError:
                continue

    if not X:
        print("[!] Error: No valid rows found in CSV.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    labels, counts = np.unique(y, return_counts=True)
    print(f"[*] Loaded {len(X)} samples across {len(labels)} classes.")
    for lbl, count in zip(labels, counts):
        print(f"    - Class '{lbl}': {count} samples")

    if len(labels) < 2:
        print("[!] Error: Need at least 2 letter classes to train a classifier.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\n[+] Test Accuracy: {acc * 100:.2f}%\n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model}, f)

    print(f"[OK] Successfully saved trained letter model to '{MODEL_FILE}'.")

if __name__ == '__main__':
    train_letter_model()