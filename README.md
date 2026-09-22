# 🖐️ VisionControl

### Control your video player with nothing but your hand.

**VisionControl** is an AI-powered, computer-vision-based video player controller that lets you interact with videos using natural hand gestures captured through a webcam.

Instead of reaching for a keyboard, mouse, or remote, VisionControl uses **hand landmarks + machine learning + real-time computer vision** to understand your gestures and control video playback.

> 🖐️ Your hand becomes the remote.

---

## ✨ Features

| Gesture / Movement               | Action                                                  |
| -------------------------------- | ------------------------------------------------------- |
| ✊ **Fist**                       | ⏸️ Pause video                                          |
| 🖐️ **Open Hand**                | ▶️ Play video                                           |
| 🖐️ **Swipe Hand Right**         | ⏩ Seek forward                                          |
| 🖐️ **Swipe Hand Left**          | ⏪ Seek backward                                         |
| 🤏 **Thumb ↔ Index Distance**    | 🔊 Control system volume                                |
| 📏 **Hand Distance Independent** | Consistent volume control at different camera distances |

### Current Capabilities

* 🎥 Real-time webcam hand tracking
* 🧠 Custom-trained gesture classification model
* ✊ Fist/Open-hand recognition
* ⏯️ Gesture-based Play/Pause
* 🖐️ Hand-swipe-based continuous seeking
* 🔊 Real-time volume control
* 📏 Scale-independent volume control
* 🎯 Landmark-based feature extraction
* 🧹 Prediction smoothing for more stable gestures
* 🎞️ VLC media player integration
* 🪟 Windows system-volume control
* ⚡ Real-time processing

---

# 🧠 How It Works

VisionControl combines **computer vision, machine learning, and media control** into one pipeline.

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
             Feature Extraction
                    │
                    ▼
          Landmark Normalization
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
       ML Classifier    Measurements
             │             │
             ▼             ▼
        Fist / Open    Finger Distance
             │             │
             ▼             ▼
        Play / Pause     Volume
             │
             │
             └──────────┬──────────┐
                        │          │
                        ▼          ▼
                  Hand Movement   VLC
                        │
                        ▼
                  Video Seeking
```

---

# 🤖 Machine Learning Pipeline

The project does **not** directly train a model on raw webcam images.

Instead, MediaPipe first extracts the hand's 21 landmarks.

Each landmark contains:

```text
X coordinate
Y coordinate
Z coordinate
```

Therefore:

```text
21 landmarks × 3 coordinates
        =
63 numerical features
```

These features are then normalized relative to the hand.

```text
Webcam Frame
     ↓
21 Hand Landmarks
     ↓
Relative Coordinates
     ↓
Scale Normalization
     ↓
63 Features
     ↓
Random Forest
     ↓
Gesture Prediction
```

The current classifier recognizes:

* `fist`
* `open`

---

# ⏩ Hand-Swipe-Based Continuous Seeking

VisionControl uses **hand movement** to control the video position.

Instead of treating a swipe as a single event such as:

> "Swipe detected → jump 10 seconds"

the system continuously analyzes the movement of the hand.

```text
          Hand Movement
                ↓
        Position Smoothing
                ↓
       Movement Calculation
                ↓
        Velocity Estimation
                ↓
        Velocity Smoothing
                ↓
          Seek Rate
                ↓
        Video Position
```

### 🖐️ Swipe Right

Moving your hand toward the **right** continuously moves the video forward.

```text
🖐️  →  →  →  →  →  ⏩
             Video →
```

### 🖐️ Swipe Left

Moving your hand toward the **left** continuously moves the video backward.

```text
⏪  ←  ←  ←  ←  ←  🖐️
      ← Video
```

The seeking system incorporates:

* position smoothing
* movement detection
* velocity calculation
* velocity smoothing
* dead-zone handling
* acceleration
* momentum
* boundary detection
* controlled VLC updates

This makes the hand behave more like a **virtual video-scrubbing controller** rather than a simple one-time swipe gesture.

---

# 🔊 Smart Volume Control

Volume is handled differently from gesture classification.

We don't need an ML model to determine whether the volume should be 30%, 50%, or 90%.

Instead, VisionControl measures the distance between:

```text
👍 Thumb tip
      ↕
☝️ Index fingertip
```

However, absolute distance changes when the hand moves closer to or farther from the camera.

So VisionControl normalizes the distance using hand size:

```text
Normalized Pinch Distance
=
Thumb–Index Distance
/
Hand Size
```

This makes volume control more consistent at different camera distances.

```text
Hand Close
    ↓
Large landmarks
    ↓
Normalized distance
    ↓
Volume


Hand Far
    ↓
Small landmarks
    ↓
Normalized distance
    ↓
Similar volume
```

---

# 🧩 Project Architecture

```text
VisionControl/
│
├── main.py
├── prepare_dataset.py
├── train_model.py
├── predict_gesture.py
├── volume_control.py
├── windows_volume.py
├── gesture_player.py
├── vlc_test.py
│
├── dataset/
│   └── gestures.csv
│
├── models/
│   ├── gesture_model.pkl
│   └── hand_landmarker.task
│
├── .gitignore
└── README.md
```

### Main Components

| File                 | Purpose                                   |
| -------------------- | ----------------------------------------- |
| `main.py`            | Collect hand-gesture training data        |
| `prepare_dataset.py` | Prepare and split the dataset             |
| `train_model.py`     | Train and evaluate the gesture classifier |
| `predict_gesture.py` | Test real-time gesture prediction         |
| `volume_control.py`  | Handle finger-distance-based volume logic |
| `windows_volume.py`  | Interface with Windows system volume      |
| `gesture_player.py`  | Main integrated VisionControl application |
| `vlc_test.py`        | Test VLC video playback                   |

---

# 🛠️ Tech Stack

### Computer Vision

* **OpenCV**
* **MediaPipe**

### Machine Learning

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Random Forest**
* **Joblib**

### Media Control

* **VLC**
* **python-vlc**
* **Pycaw**

### Development

* **Git**
* **GitHub**
* **VS Code**

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/VisionControl.git
cd VisionControl
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install opencv-python mediapipe pandas scikit-learn joblib python-vlc pycaw
```

Make sure **VLC Media Player** is installed on your system.

---

# 🎥 Add a Video

Place your video inside the project directory:

```text
VisionControl/
│
├── Avengers.mp4
├── gesture_player.py
└── ...
```

The current application expects:

```python
VIDEO_PATH = "Avengers.mp4"
```

You can change this variable to another video.

> Video files are excluded from Git using `.gitignore` to avoid unnecessarily large repository sizes.

---

# 🧪 Training Your Own Gesture Model

One of the main goals of VisionControl is to experiment with **your own machine-learning model** rather than relying on a pre-trained gesture classifier.

### Step 1 — Collect Data

```bash
python main.py
```

Collect examples for:

```text
fist
open
```

The collected landmark features are stored in:

```text
dataset/gestures.csv
```

### Step 2 — Prepare the Dataset

```bash
python prepare_dataset.py
```

### Step 3 — Train the Model

```bash
python train_model.py
```

The trained model is saved as:

```text
models/gesture_model.pkl
```

### Step 4 — Test Real-Time Prediction

```bash
python predict_gesture.py
```

---

# ▶️ Run VisionControl

Once the model and MediaPipe hand-landmarker model are available:

```bash
python gesture_player.py
```

You should see the VisionControl webcam interface.

```text
✊ → Pause
🖐️ → Play
🖐️ → Right → Seek Forward
🖐️ → Left  → Seek Backward
🤏 → Volume
```

Press:

```text
Q
```

to exit.

---

# 🎯 Interaction Model

VisionControl separates different types of interactions instead of forcing everything into one ML model.

### Discrete Gestures

```text
✊ Fist
   ↓
Classifier
   ↓
Pause
```

```text
🖐️ Open Hand
   ↓
Classifier
   ↓
Play
```

### Continuous Volume

```text
Thumb ↔ Index Distance
          ↓
   Normalized Distance
          ↓
        Volume
```

### Hand Movement

```text
Hand Position
      ↓
Movement
      ↓
Velocity
      ↓
Direction + Speed
      ↓
Video Seeking
```

This allows each interaction to use the technique that best fits the problem.

---

# 🗺️ Roadmap

### ✅ Completed

* [x] Webcam hand tracking
* [x] 21 hand landmark extraction
* [x] Landmark feature engineering
* [x] Scale normalization
* [x] Dataset collection
* [x] Fist/Open gesture dataset
* [x] Random Forest training
* [x] Model evaluation
* [x] Real-time gesture prediction
* [x] Prediction smoothing
* [x] VLC integration
* [x] Gesture-based Play/Pause
* [x] Continuous hand-swipe seeking
* [x] Seek smoothing
* [x] Seek momentum
* [x] Seek boundary handling
* [x] Finger-distance volume control
* [x] Scale-independent volume control
* [x] Windows system-volume control

### 🚧 In Development

* [ ] Swipe gesture refinement
* [ ] Subtitle control
* [ ] Audio-track switching
* [ ] Playback speed control
* [ ] Fullscreen control
* [ ] Mute gesture
* [ ] Previous/Next video
* [ ] Brightness control

### 🔮 Future ML Development

* [ ] Expand gesture dataset
* [ ] Compare Random Forest vs SVM
* [ ] Train a custom neural network
* [ ] Dynamic gesture recognition
* [ ] Temporal sequence modeling
* [ ] LSTM/GRU experiments
* [ ] Transformer-based gesture recognition
* [ ] Custom gesture vocabulary
* [ ] Personalized gesture calibration

---

# 🔬 Future Architecture

The long-term goal is to evolve VisionControl into a more advanced gesture-recognition platform.

```text
                 Webcam
                   │
                   ▼
             Hand Detection
                   │
                   ▼
            Landmark Sequence
                   │
          ┌────────┴─────────┐
          │                  │
          ▼                  ▼
     Static Gesture      Dynamic Gesture
       Classifier          Model
          │                  │
          └────────┬─────────┘
                   ▼
             Gesture Engine
                   │
                   ▼
             Command System
                   │
                   ▼
             Media Player
```

---

# 🧠 Learning Goals

VisionControl is a hands-on learning project focused on understanding how computer vision and machine learning systems are built from individual components.

The project explores:

* Computer vision
* Hand landmark detection
* Feature engineering
* Dataset creation
* Data preprocessing
* Machine learning
* Model evaluation
* Real-time inference
* Signal smoothing
* Temporal movement analysis
* Human-computer interaction
* Media automation

The goal is not simply to build a gesture-controlled video player, but to understand the engineering decisions behind every component.

---

# 🔐 Privacy

VisionControl is designed around local processing.

The webcam feed is processed locally for hand detection and gesture recognition.

No cloud-based video processing is required for the current system.

---

# ⚡ Performance

The application is designed for real-time interaction using:

* lightweight landmark representations
* local ML inference
* smoothed predictions
* controlled VLC updates
* direct Windows audio control

The system avoids sending every webcam frame through a large neural network, keeping the architecture lightweight.

---

# 🤝 Contributing

This project is primarily a personal learning project, but ideas, improvements, and experiments are welcome.

If you want to contribute:

```bash
git checkout -b feature/your-feature
```

Make your changes, test them, and submit a pull request.

---

# 📜 License

This project is currently intended as a personal learning and development project.

A formal open-source license can be added in the future.

---

# 👨‍💻 Author

**Adwaith Vinod**

B.Tech Computer Science Engineering

Interested in:

* Artificial Intelligence
* Machine Learning
* Computer Vision
* AI Engineering
* Software Development

---

## ⭐ VisionControl

> **See it. Understand it. Control it.**

**Your hand. Your gestures. Your video.**
