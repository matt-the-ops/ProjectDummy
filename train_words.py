import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join('word_dataset')
MODEL_PATH = 'word_model.pkl'

def train():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Folder '{DATA_PATH}' does not exist.")
        return

    words = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
    if not words:
        print("Error: No word directories found in 'word_dataset/'. Record words first!")
        return

    X, y = [], []
    for word in words:
        word_dir = os.path.join(DATA_PATH, word)
        files = [f for f in os.listdir(word_dir) if f.endswith('.npy')]
        for file in files:
            sequence_data = np.load(os.path.join(word_dir, file))
            X.append(sequence_data.flatten())
            y.append(word)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save ONLY to word_model.pkl
    payload = {'model': model, 'classes': model.classes_}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)

    print(f"[✓] Successfully saved word model to '{MODEL_PATH}'!")

if __name__ == '__main__':
    train()