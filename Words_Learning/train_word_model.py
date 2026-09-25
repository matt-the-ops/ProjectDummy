import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Root directory reference
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'word_dataset')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
MOTION_MODEL_PATH = os.path.join(MODELS_DIR, 'word_motion_model.h5')
FACE_MODEL_PATH = os.path.join(MODELS_DIR, 'word_face_model.pkl')
CLASSES_PATH = os.path.join(MODELS_DIR, 'word_classes.npy')

os.makedirs(MODELS_DIR, exist_ok=True)

SEQUENCE_LEN = 40
HAND_FEATURES = 132   # 126 shape + 6 wrist-trajectory
FACE_FEATURES = 1404
AUGMENT_COUNT = 4

# ── Safe Unpickler ──────────────────────────────────────────────────────
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

# ── Hand normalization (scale-invariant) ────────────────────────────────
def normalize_hand_sequence(hand_seq):
    seq = np.asarray(hand_seq, dtype=np.float32)
    if seq.size == 0:
        return seq
    out = seq.copy()
    spans = []
    for off in (0, 63):
        tip_idx = 12 * 3          # landmark 12 (middle fingertip)
        tip = out[0, off + tip_idx : off + tip_idx + 3]
        span = np.linalg.norm(tip)
        if span < 1e-6:
            span = 1.0
        spans.append(span)
        out[:, off:off + 63] = out[:, off:off + 63] / span
    if out.shape[1] >= 132:
        out[:, 126:129] = out[:, 126:129] / spans[0]
        out[:, 129:132] = out[:, 129:132] / spans[1]
    return out

def _rot_scale(block, cos, sin, scale):
    xy = block[..., :2]
    z  = block[..., 2:3]
    rot = np.stack([xy[..., 0] * cos - xy[..., 1] * sin,
                     xy[..., 0] * sin + xy[..., 1] * cos], axis=-1)
    return np.concatenate([rot * scale, z], axis=-1)

def augment_hand_sequence(hand_seq, rng, angle_deg=8, scale_range=(0.9, 1.1)):
    seq = np.asarray(hand_seq, dtype=np.float32)
    T = seq.shape[0]
    angle = np.deg2rad(rng.uniform(-angle_deg, angle_deg))
    cos, sin = np.cos(angle), np.sin(angle)
    scale = rng.uniform(*scale_range)
    out = seq.copy()
    for off in (0, 63):
        block = out[:, off:off + 63].reshape(T, 21, 3)
        block = _rot_scale(block, cos, sin, scale)
        out[:, off:off + 63] = block.reshape(T, 63)
    if out.shape[1] >= 132:
        for off in (126, 129):
            block = out[:, off:off + 3].reshape(T, 1, 3)
            block = _rot_scale(block, cos, sin, scale)
            out[:, off:off + 3] = block.reshape(T, 3)
    return out.astype(np.float32)

# ── Dataset Loader ──────────────────────────────────────────────────────
def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"'{DATA_PATH}' does not exist. Run collect_word_data.py first.")

    words = sorted([d for d in os.listdir(DATA_PATH)
                    if os.path.isdir(os.path.join(DATA_PATH, d))])
    if len(words) < 2:
        raise ValueError(f"Need at least 2 word classes, found: {words}")
    print(f"[*] Word classes: {words}")

    hand_seqs, face_vecs, labels = [], [], []
    skipped = 0

    for word in words:
        wdir = os.path.join(DATA_PATH, word)
        hand_files = sorted([f for f in os.listdir(wdir) if f.endswith('_hand.npy')])

        for hf in hand_files:
            idx = hf.replace('_hand.npy', '')
            hp = os.path.join(wdir, hf)
            fp = os.path.join(wdir, idx + '_face.npy')

            hand_arr = np.load(hp)
            if not os.path.exists(fp):
                print(f"  [!] Missing face file for {hf}, skipping.")
                skipped += 1
                continue
            face_arr = np.load(fp)

            if hand_arr.shape != (SEQUENCE_LEN, HAND_FEATURES):
                print(f"  [!] Bad hand shape {hand_arr.shape} in {hf}, skipping.")
                skipped += 1
                continue
            if face_arr.shape != (FACE_FEATURES,):
                print(f"  [!] Bad face shape {face_arr.shape} in {hf}, skipping.")
                skipped += 1
                continue

            hand_seqs.append(hand_arr)
            face_vecs.append(face_arr)
            labels.append(word)

    if not hand_seqs:
        raise ValueError("No valid samples found. Re-record with collect_word_data.py.")

    hand_seqs = np.array([normalize_hand_sequence(h) for h in hand_seqs], dtype=np.float32)
    face_vecs = np.array(face_vecs, dtype=np.float32)
    labels_arr = np.array(labels)

    print(f"[*] Loaded {len(hand_seqs)} samples ({skipped} skipped).")
    print(f"[*] Hand shape: {hand_seqs.shape}, Face shape: {face_vecs.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(labels_arr)
    np.save(CLASSES_PATH, le.classes_)
    print(f"[OK] Saved class list: {le.classes_}")

    return hand_seqs, face_vecs, y_enc, le.classes_

# ── Motion Model (LSTM on hand sequence) ────────────────────────────────
def build_motion_model(num_classes):
    model = Sequential([
        Input(shape=(SEQUENCE_LEN, HAND_FEATURES)),
        LSTM(64,  return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(128, return_sequences=True, activation='tanh'),
        BatchNormalization(),
        Dropout(0.4),
        LSTM(64,  return_sequences=False, activation='tanh'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model

def train_motion_model(X_hand, y, classes):
    print("\n=== Training HAND MOTION Model (LSTM) ===")
    nc = len(classes)

    rng = np.random.RandomState(42)
    X_aug, y_aug = [], []
    for seq, lab in zip(X_hand, y):
        X_aug.append(seq)
        y_aug.append(lab)
        for _ in range(AUGMENT_COUNT):
            X_aug.append(augment_hand_sequence(seq, rng))
            y_aug.append(lab)
    X_hand_aug = np.array(X_aug, dtype=np.float32)
    y_aug = np.array(y_aug)
    print(f"[*] Augmented: {len(X_hand)} -> {len(X_hand_aug)} samples")

    try:
        Xtr, Xte, ytr, yte = train_test_split(X_hand_aug, y_aug, test_size=0.2,
                                               random_state=42, stratify=y_aug)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X_hand_aug, y_aug, test_size=0.2,
                                               random_state=42)

    model = build_motion_model(nc)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
    ]
    model.fit(Xtr, ytr, epochs=200, batch_size=max(4, len(Xtr) // 8),
              validation_data=(Xte, yte), callbacks=callbacks, verbose=1)

    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\n[Motion Model] Test accuracy: {acc*100:.1f}%  |  Loss: {loss:.4f}")
    preds = np.argmax(model.predict(Xte, verbose=0), axis=1)
    print(classification_report(yte, preds, target_names=classes))

    model.save(MOTION_MODEL_PATH)
    print(f"[OK] Saved -> '{MOTION_MODEL_PATH}'")
    return model

# ── Face Model (PCA + LogisticRegression on face signature) ────────────
def train_face_model(X_face, y, classes):
    print("\n=== Training FACE SIGNATURE Model (PCA+LogReg) ===")

    try:
        Xtr, Xte, ytr, yte = train_test_split(X_face, y, test_size=0.2,
                                               random_state=42, stratify=y)
    except ValueError:
        Xtr, Xte, ytr, yte = train_test_split(X_face, y, test_size=0.2,
                                               random_state=42)

    n = max(1, min(30, Xtr.shape[0] - 1))

    face_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n, random_state=42)),
        ('clf', LogisticRegression(C=0.3, max_iter=3000,
                                    class_weight='balanced', solver='lbfgs')),
    ])
    face_pipeline.fit(Xtr, ytr)
    acc = face_pipeline.score(Xte, yte)
    print(f"[Face Model] Test accuracy: {acc*100:.1f}%")
    preds = face_pipeline.predict(Xte)
    print(classification_report(yte, preds, target_names=classes))

    payload = {'model': face_pipeline, 'classes': classes}
    with open(FACE_MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)
    print(f"[OK] Saved -> '{FACE_MODEL_PATH}'")
    return face_pipeline

def main():
    print("==================================================")
    print("   WORDS LEARNING: DUAL-GATE MODEL TRAINER        ")
    print("==================================================\n")
    X_hand, X_face, y, classes = load_dataset()
    print(f"Dataset: X_hand={X_hand.shape}, X_face={X_face.shape}, y={y.shape}\n")

    train_motion_model(X_hand, y, classes)
    train_face_model(X_face, y, classes)

    print("\n[ALL DONE] Both models trained and saved.")
    print(f"  Motion model : {MOTION_MODEL_PATH}")
    print(f"  Face model   : {FACE_MODEL_PATH}")
    print(f"  Classes      : {classes}")
    print("\nRun python main.py to start the translator.")

if __name__ == '__main__':
    main()
