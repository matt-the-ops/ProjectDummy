import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmo.config import DATA_DIR, MODELS_DIR, LETTER_CSV_FILE, LETTER_MODEL_FILE

CSV_FILE = LETTER_CSV_FILE
MODEL_FILE = LETTER_MODEL_FILE

from bmo.features.fsl_features import atomic_replace_pickle

def train_letter_model():
    print("==========================================")
    print("    FSL LETTER MODEL TRAINER (Mode 1)     ")
    print("==========================================\n")

    if not os.path.exists(CSV_FILE):
        print(f"[!] Error: Dataset '{CSV_FILE}' not found. Run collect_letter_data.py to collect letter samples.")
        return

    # Load dataset using pandas
    df = pd.read_csv(CSV_FILE)
    if df.empty or len(df.columns) < 64:
        print("[!] Error: No valid rows/columns found in CSV.")
        return

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:64].values.astype(np.float32)

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

    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    atomic_replace_pickle(MODEL_FILE, {'model': model})

    print(f"[OK] Successfully saved trained letter model to '{MODEL_FILE}'.")

if __name__ == '__main__':
    train_letter_model()
