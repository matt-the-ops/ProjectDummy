# Project BMO: Filipino Sign Language (FSL) Prototype Kiosk

Real-time Filipino Sign Language (FSL) translation kiosk powered by MediaPipe, OpenCV, Scikit-Learn, and TensorFlow LSTM models with BMO character persona & Text-to-Speech (TTS).

---

## 📁 Project Organization

```text
ProjectDummy/
├── main.py                     # Main application entry point (live translator GUI)
├── Letters_Learning/           # Mode 1: Letters (Alphabet & Static Signs)
│   ├── collect_letter_data.py  # Collect single-hand landmark keypoint samples
│   └── train_letter_model.py   # Train Random Forest classifier (fsl_model.pkl)
├── Words_Learning/             # Mode 2: Words (2-Hand Dynamic Sequences)
│   ├── collect_word_data.py    # Dual-gate hand motion + face data recorder
│   └── train_word_model.py     # Train LSTM motion model + Face signature model
├── models/                     # Saved trained models & MediaPipe task file
│   ├── fsl_model.pkl           # Letters model
│   ├── word_motion_model.h5    # Dynamic hand motion LSTM
│   ├── word_face_model.pkl     # Facial expression classifier
│   ├── word_classes.npy        # Word class labels
│   ├── hand_landmarker.task    # MediaPipe hand landmarker
│   └── BMO Voice/              # BMO RVC model weights (.pth and .index)
├── data/                       # Datasets
│   ├── fsl_dataset.csv         # Letters dataset
│   └── word_dataset/           # Word sequence frames (.npy)
├── assets/                     # Media & design assets
│   ├── bmo/                    # BMO face avatars (closed, open, idle, talk)
│   ├── Logo/                   # Project logos
│   └── Neumorphism References/ # UI mockups and visual references
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run the Translator

Activate your virtual environment and start `main.py`:

```bash
# In PowerShell:
.\venv\Scripts\Activate.ps1
python main.py
```

### On-Screen Controls & Keybindings
- **`1`** : Switch to **Mode 1 (Letters)**
- **`2`** : Switch to **Mode 2 (Words)**
- **`N`** : Toggle Camera Device Index (0, 1, 2)
- **`F`** : Toggle Fullscreen Control Panel
- **`Enter`** : Speak full typed sentence via BMO Voice (TTS)
- **`Space`** : Add space
- **`Backspace`** : Delete last character
- **`C`** : Clear text output
- **`Q`** : Quit Application

---

## 🎓 Training & Data Collection

### Working with Letters (Mode 1)
To collect letter keypoints:
```bash
python Letters_Learning/collect_letter_data.py
```
To retrain the letters model:
```bash
python Letters_Learning/train_letter_model.py
```

### Working with Words (Mode 2)
To record or manage dynamic word gestures:
```bash
python Words_Learning/collect_word_data.py
```
To retrain the dual-gate motion and face models:
```bash
python Words_Learning/train_word_model.py
```