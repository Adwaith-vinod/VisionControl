# 🖐️ VisionControl

### Control your video player with nothing but your hand.

**VisionControl** is an AI-powered, computer-vision-based video player controller that lets you interact with videos using natural hand gestures captured through a webcam.

Instead of reaching for a keyboard, mouse, or remote, VisionControl uses **hand landmarks + machine learning + real-time computer vision** to understand your gestures and control video playback.

> 🖐️ Your hand becomes the remote.

---

## ✨ Features

| Gesture / Movement | Action | State / Behavior |
| :--- | :--- | :--- |
| ✊ **Fist** | ⏸️ **Pause Video** | Instant playback pause, seek engine idle |
| 🖐️ **Open Palm** | ▶️ **Play Video** | Resumes playback, enables pinch volume |
| ✌️ **Two Fingers (Swipe Right)** | ⏩ **Seek Forward** | Continuous fast-forward scrubbing |
| ✌️ **Two Fingers (Swipe Left)** | ⏪ **Seek Backward** | Continuous rewind scrubbing |
| 🤏 **Thumb ↔ Index Pinch** | 🔊 **System Volume** | Scale-independent volume (active on Open Palm) |
| 📏 **Distance Invariant** | 🎯 **Consistent Control** | Volume & gesture recognition scale across camera distances |

### Current Capabilities

* 🎥 Real-time webcam hand tracking with MediaPipe
* 🧠 Custom-trained 3-class gesture classifier (`fist`, `open`, `two_finger`)
* ⚡ State-isolated gesture execution (no accidental cross-triggers)
* ⏩ Two-finger continuous hand scrubbing with momentum & decay
* 🔊 Smart scale-invariant Windows volume control
* 🛡️ Software-decoded VLC pipeline with zero frame-drop errors during scrubs
* 🧹 Temporal voting & grace-period debounce for stable gesture transitions
* 📊 Real-time on-screen HUD (volume bar, scrub bar, velocity stats, gesture status)

---

# 🧠 How It Works

VisionControl combines **computer vision, machine learning, and media control** into a streamlined, state-gated pipeline.

```text
                 📷 Webcam
                    │
                    ▼
              OpenCV Capture
                    │
                    ▼
          ┌─────────────────────┐
          │  MediaPipe Hands    │
          │  21 Hand Landmarks  │
          └─────────────────────┘
                    │
                    ▼
          Landmark Normalization
          (Wrist-relative + Scale)
                    │
                    ▼
          ML Random Forest Classifier
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
    ✊ Fist      🖐️ Open     ✌️ Two Fingers
       │            │            │
       ▼            ▼            ▼
     PAUSE      PLAY VIDEO   SEEK / SCRUB
                    │            │
                    ▼            ▼
               Pinch Volume   Horizontal
                 Control       Movement
```

---

# 🤖 Machine Learning Pipeline

The project does **not** directly train a heavy CNN on raw webcam images. Instead, MediaPipe first extracts the hand's 21 3D landmarks (X, Y, Z coordinates).

```text
21 landmarks × 3 coordinates = 63 numerical features
```

### Feature Normalization

To ensure distance invariance (whether your hand is 30 cm or 1.5 m from the camera), features are:
1. **Zero-centered** relative to the wrist landmark (`landmark[0]`).
2. **Scale-normalized** by the hand size (Euclidean distance from wrist `landmark[0]` to middle MCP `landmark[9]`).

```text
Webcam Frame
     ↓
21 Hand Landmarks
     ↓
Wrist-Relative Coordinates (x - x_wrist, y - y_wrist, z - z_wrist)
     ↓
Scale Normalization (features / hand_size)
     ↓
63 Normalized Features
     ↓
Random Forest Classifier (100 Trees)
     ↓
Gesture Class: ['fist', 'open', 'two_finger']
     ↓
Temporal Majority Voting (Queue of 7 Frames, Threshold = 5)
```

The classifier accurately recognizes:
* `fist` &rarr; Pauses video
* `open` &rarr; Plays video & enables volume pinch
* `two_finger` &rarr; Engages video seeking/scrubbing mode

---

# ⏩ Two-Finger Continuous Seeking

VisionControl uses **horizontal movement of two raised fingers** to scrub video position smoothly.

Instead of treating a swipe as a crude trigger (e.g. *"swipe detected &rarr; jump 10s"*), the system behaves like a **virtual scrubbing wheel**:

```text
       Two Fingers Detected
                ↓
    Index Fingertip Tracking
                ↓
        Position Smoothing
                ↓
       Movement Velocity (dt)
                ↓
     Robust Median Velocity
                ↓
     Exponential Velocity Filter
                ↓
      Dynamic Seek Acceleration
                ↓
      VLC Time Update (set_time)
```

### ✌️ Scrubbing Controls

* **✌️ Swipe Right:** Continuously advances the video forward at a rate proportional to your hand's speed.
* **✌️ Swipe Left:** Continuously rewinds the video backward at a rate proportional to your hand's speed.
* **Momentum & Friction:** When your hand stops moving, seeking decelerates smoothly via `MOMENTUM_DECAY` rather than cutting off abruptly.
* **Flicker Debounce:** A 400ms grace period prevents minor angle shifts during a swipe from interrupting the scrub or accidentally triggering play/pause.
* **Playback Synchronization:** When stationary, the seek target continuously synchronizes with live playback so scrubbing always begins from the current scene.

---

# 🔊 Smart Volume Control

Volume is handled independently from gesture classification:

Instead of using an ML model to guess target percentages, VisionControl directly measures the 3D distance between:
```text
👍 Thumb tip (Landmark 4)  ↔  ☝️ Index tip (Landmark 8)
```

### Scale Normalization

```text
Normalized Pinch Distance = (Thumb–Index Distance) / Hand Size
```

```text
Hand Close (Large Landmarks)   ──┐
                                 ├── Normalized Distance ──► Consistent Volume
Hand Far (Small Landmarks)     ──┘
```

* **Isolated to Open Hand:** Volume adjustments are active **only** when showing an open palm (`open`). During two-finger scrubbing or a fist, volume is frozen to prevent accidental swings.
* **PyCaw Direct Integration:** Directly interfaces with the Windows Core Audio Endpoint Volume API for instantaneous hardware-level volume control.

---

# 🧩 Project Architecture

```text
VisionControl/
│
├── dataset/
│   └── gestures.csv            # Training dataset (fist, open, two_finger)
├── models/
│   ├── gesture_model.pkl       # Trained Random Forest classifier
│   └── hand_landmarker.task    # MediaPipe Hand Landmarker model
│
├── gesture_player.py           # Main application with full gesture pipeline
├── main.py                     # Interactive dataset collection tool
├── train_model.py              # Model training and evaluation script
│
├── Avengers.mp4                # Demo video
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git configuration
└── README.md                   # Documentation
```

### Core Components

| Component | Description |
| :--- | :--- |
| [`gesture_player.py`](file:///c:/Users/adwai/OneDrive/Desktop/Projects/VisionControl/gesture_player.py) | **Main application** &mdash; integrates MediaPipe, ML classification, swipe seek engine, PyCaw audio, and VLC media player. |
| [`main.py`](file:///c:/Users/adwai/OneDrive/Desktop/Projects/VisionControl/main.py) | **Data collector** &mdash; records normalized landmark features from webcam into `dataset/gestures.csv` with keyboard controls. |
| [`train_model.py`](file:///c:/Users/adwai/OneDrive/Desktop/Projects/VisionControl/train_model.py) | **Model trainer** &mdash; splits `gestures.csv`, trains a 100-tree Random Forest, evaluates metrics, and exports `gesture_model.pkl`. |

---

# 🛠️ Tech Stack

* **Computer Vision:** OpenCV, MediaPipe Tasks Vision (`hand_landmarker.task`)
* **Machine Learning:** Scikit-learn (Random Forest), NumPy, Pandas, Joblib
* **Media Control:** VLC Media Player (`python-vlc` with software decoding flags)
* **Audio Control:** PyCaw (`CoreAudio` Windows Audio Session API)
* **Environment:** Python 3.10+ on Windows

---

# 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Adwaith-vinod/VisionControl
cd VisionControl
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Prerequisite:** Make sure [VLC Media Player](https://www.videolan.org/vlc/) (64-bit) is installed on your system.

### 4. Add a Video

Place your video file in the project folder and configure the path in [`gesture_player.py`](file:///c:/Users/adwai/OneDrive/Desktop/Projects/VisionControl/gesture_player.py):

```python
VIDEO_PATH = "Avengers.mp4"
```

---

# 🧪 Training Your Own Gesture Model

VisionControl allows you to easily collect your own dataset and train a custom model from scratch.

### Step 1 — Collect Training Data

Run the interactive data collector:

```bash
python main.py
```

* Position your hand in front of the camera.
* Press **`F`** &rarr; Select `fist`
* Press **`O`** &rarr; Select `open`
* Press **`T`** (or **`2`**) &rarr; Select `two_finger`
* Press **`S`** &rarr; Start / Stop recording (collect ~2,000 samples per class)
* Press **`Q`** &rarr; Save and quit

Collected landmarks are automatically saved to `dataset/gestures.csv`.

### Step 2 — Train the Classifier

```bash
python train_model.py
```

This script:
1. Loads `dataset/gestures.csv`.
2. Splits data into 80% train and 20% test sets (stratified).
3. Trains a `RandomForestClassifier(n_estimators=100)`.
4. Prints evaluation metrics (Accuracy, Confusion Matrix, Classification Report).
5. Exports the trained model to `models/gesture_model.pkl`.

---

# ▶️ Run VisionControl

Launch the main integrated application:

```bash
python gesture_player.py
```

### On-Screen Controls

```text
✊ Fist                 → ⏸️ Pause Video
🖐️ Open Palm            → ▶️ Play Video
✌️ Two Fingers (Swipe)  → ⏩ Scrub Forward / ⏪ Scrub Backward
🤏 Thumb-Index Pinch    → 🔊 Adjust Windows Volume
Q                       → Quit Application
```

---

# 🎯 Interaction Model

VisionControl prevents gesture conflicts by isolating states:

```text
┌─────────────────┬──────────────────┬──────────────────┐
│  Current Mode   │  Active Engine   │  Inactive Engine │
├─────────────────┼──────────────────┼──────────────────┤
│ ✊ Fist         │ VLC Pause        │ Seeking & Volume │
│ 🖐️ Open Palm    │ VLC Play + Volume│ Seeking          │
│ ✌️ Two Fingers  │ Swipe Scrubbing  │ Volume & Playback│
└─────────────────┴──────────────────┴──────────────────┘
```

* **Zero Seeking Glitches:** The video seeking engine is strictly locked to `two_finger`. Your open palm or fist will never trigger unwanted video jumping.
* **Volume Isolation:** Volume adjustment only responds when an open palm is shown, avoiding volume spikes during seeking.
* **VLC Software Decoding:** Configured with `--avcodec-hw=none` to eliminate H.264 GPU accelerator crashes during high-speed scrubbing.

---

# 🗺️ Roadmap

### ✅ Completed

* [x] Real-time 21 hand landmark tracking with MediaPipe
* [x] Scale- and distance-invariant feature engineering
* [x] 3-class dataset collection (`fist`, `open`, `two_finger`)
* [x] Random Forest model training & evaluation pipeline (>99% accuracy)
* [x] State-isolated Play/Pause and Volume control
* [x] Two-finger dynamic scrubbing engine with momentum & friction
* [x] 400ms gesture grace period to prevent transition flickers
* [x] Direct Windows master volume control via PyCaw
* [x] Stable VLC scrubbing with software decode integration
* [x] Full HUD overlay (volume bar, scrub indicator, gesture readout)

### 🚧 In Development

* [ ] Fullscreen toggle gesture
* [ ] Audio mute / unmute gesture
* [ ] Next / Previous chapter or playlist track
* [ ] Configurable video path via CLI argument (`python gesture_player.py --video path/to/video.mp4`)
* [ ] Multi-platform audio support (Linux / macOS)

---

# 👨‍💻 Author

**Adwaith Vinod**  
B.Tech Computer Science Engineering  

Interested in Artificial Intelligence, Machine Learning, Computer Vision, and Software Engineering.

---

## ⭐ VisionControl
> **See it. Understand it. Control it.**  
> *Your hand. Your gestures. Your video.*
