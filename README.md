# Project SHIELD: Real-Time Filipino Sign Language (FSL) Kiosk Translator

An automated, real-time Filipino Sign Language (FSL) translation kiosk powered by Computer Vision and Machine Learning. Project SHIELD supports both static letter/motion-based fingerspelling and dynamic 2-hand word gesture recognition with integrated Text-to-Speech (TTS).

---

## 🌟 Key Features

* **Dual-Mode Recognition Engine**:
  * **Mode 1 (Letters & Motion)**: Single-hand static fingerspelling with specialized spatial tracking for dynamic letters (`J` and `Z`) and stabilization cooldowns to prevent double-triggering.
  * **Mode 2 (Dynamic Words)**: Multi-hand sequence tracking (40-frame rolling buffer) normalized relative to wrist positions to recognize full FSL signs.
* **Text-to-Speech (TTS) Integration**: Built-in voice playback using `pyttsx3` for immediate letter/word feedback and full sentence readout.
* **Dataset & Model Management**: Interactive CLI tool to capture, manage, delete individual words, or clear datasets with automatic model cache purging.
* **Multi-Camera Toggle**: Switch active camera inputs on the fly.

---

## 🛠️ Requirements & Setup on a New PC

### Prerequisites
* **Python**: Version 3.10 or 3.11 recommended.
* **Webcam**: Integrated or USB camera.

### Installation Steps

1. **Clone or Copy Project Directory**  
   Ensure all script files and project dependencies are in your project root folder.

2. **Set Up a Virtual Environment**  
   Open your terminal (PowerShell / Command Prompt) in the project root:
   ```bash
   python -m venv venv

# Activate the Virtual Environment

Windows (PowerShell):

PowerShell
.\venv\Scripts\Activate.ps1
Windows (CMD):

DOS
.\venv\Scripts\activate.bat

# Install Dependencies via requirements.txt

Bash
pip install -r requirements.txt
Download MediaPipe Model Asset

Ensure the hand_landmarker.task file is located in the root project folder alongside your scripts.

System Usage & Workflow
1. Data Collection & Dataset Management
For Words (Mode 2):
Run the interactive word manager:

Bash
python collect_data_words.py
Option 1 (Record/Add a Word): Enter the word label and number of sequence samples (default 15). Press SPACE to initiate recording for each sample after the 3-second countdown.

Option 2 (Delete a Specific Word): Deletes the target word dataset folder and purges outdated word_model.pkl.

Option 3 (Reset ALL Words): Wipes the entire word_dataset/ directory and removes old model cache.

2. Model Training
Train your models whenever you add, modify, or reset datasets.

Train Letter Model (Mode 1):

Bash
python train_model.py
Generates fsl_model.pkl.

Train Word Model (Mode 2):

Bash
python train_words.py
Generates word_model.pkl.

3. Running the TranslatorLaunch the main live recognition application:

python predict.py


On-Screen Controls & Keybindings:

1         - Switch to Mode 1 (Letters)
2         - Switch to Mode 2 (Words)
N         - Toggle Camera Device Index (0, 1, 2)
ENTER     - Speak full accumulated output sentence via TTS
SPACE     - Insert space character into output string
BACKSPACE - Delete last character in output string
C         - Clear output text buffer
Q         - Exit Application