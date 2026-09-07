```markdown
# Real-Time Filipino Sign Language (FSL) Translator (WIP)

A real-time FSL recognition system built using Python, MediaPipe 3D Landmark Tracking, and Scikit-Learn Machine Learning.

---

## Prerequisites & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/matt-the-ops/ProjectDummy.git](https://github.com/matt-the-ops/ProjectDummy.git)
cd ProjectDummy

```

### 2. Activate Virtual Environment

* **Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## How to Run

### Step 1: Test Camera & Landmark Tracking (Optional)

Verifies webcam access and MediaPipe 3D landmark extraction.

```bash
python hand_tracker.py

```

### Step 2: Collect Gesture Data

```bash
python collect_data.py

```

**What to Expect:**

* The webcam window opens.
* Hold an FSL sign (e.g., **A**) in front of the lens.
* Tap the corresponding keyboard key (e.g., `a`) 50–100 times while slightly moving your hand to capture angle variations.
* Repeat for other letters or words (e.g., hold **B** and press `b`).
* Press `q` to exit. Terminal outputs confirm saved frames, which update `fsl_dataset.csv`.

### Step 3: Train the Model

```bash
python train_model.py

```

**What to Expect:**

* Processes `fsl_dataset.csv` and trains the Random Forest classifier.
* Outputs the model accuracy score and creates `fsl_model.pkl`.

### Step 4: Run Real-Time Recognition

```bash
python predict.py

```

**What to Expect:**

* Opens webcam feed showing hand skeleton lines, joint points, and live predicted sign labels in the top-left corner.
* Press `q` to exit.

```

```
