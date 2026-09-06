# Gesture Recognition — Live

A real-time hand gesture recognition system that runs entirely in the browser. It combines **rule-based static gesture detection** with a **trained LSTM model** for dynamic, motion-based gestures — all powered by MediaPipe hand tracking and deployed live via Streamlit.

## Live Demo

Try it right now, no installation needed:

**[gesture-recognition-1312-v2.streamlit.app](https://gesture-recognition-1312-v2.streamlit.app/)**

Click **Start**, allow camera access, and try the gestures listed below.

## Features

- **Real-time hand tracking** — 21-point hand landmark detection using Google's MediaPipe Tasks API
- **Static gesture recognition** — instant, rule-based classification using hand landmark geometry (no ML training needed)
- **Dynamic gesture recognition** — motion-based gestures (ASL-inspired) recognized using a custom-trained LSTM neural network
- **Multi-user support** — each browser session gets its own isolated hand-tracking and prediction state, so multiple people can use the app simultaneously without interfering with each other
- **Runs in the browser** — no installation required for end users; webcam access via WebRTC

## Gestures Supported

**Static (instant, single-frame):**
| Gesture | Hand Shape |
|---|---|
| Fist | All fingers curled |
| Open Palm | All fingers extended |
| Peace Sign | Index + middle extended |
| Thumbs Up | Only thumb extended |
| Pointing | Only index extended |
| Call Me | Thumb + pinky extended |
| *(+ additional combinations mapped in `gesture_logic.py`)* | |

**Dynamic (motion-based, ~1 second):**
| Gesture | Description |
|---|---|
| Hello | Open hand motion near the face, moving outward |
| Bye | Open palm, oscillating side-to-side wave |
| Help | Real ASL sign — thumbs-up fist resting on an open upward-facing palm, both hands lifting together |

## Architecture

```
Webcam (Browser)
      │
      ▼
WebRTC (streamlit-webrtc) ──── streams video frames to server
      │
      ▼
MediaPipe HandLandmarker ──── extracts 21 landmarks per hand
      │
      ├──► Static Rule Engine ──── geometric finger-state logic → instant label
      │
      └──► Rolling 30-frame Buffer ──► LSTM Model ──── motion classification → label
                                              │
                                              ▼
                                  Confidence threshold + cooldown
                                              │
                                              ▼
                                    Label rendered on video feed
```

### Why two detection systems?

- **Static gestures** are a single-frame shape — no need for a trained model, geometric rules are faster, more interpretable, and need zero training data.
- **Dynamic gestures** (Hello, Bye, Help) depend on *motion over time*, which single-frame rules cannot capture. These use a small LSTM trained on self-recorded, normalized landmark sequences.

## Tech Stack

| Layer | Tool |
|---|---|
| Hand tracking | MediaPipe Tasks API (`HandLandmarker`) |
| Dynamic gesture model | TensorFlow / Keras (LSTM) |
| Video capture (browser) | `streamlit-webrtc` + WebRTC |
| Web framework | Streamlit |
| Image processing | OpenCV (headless build, for server deployment) |
| Dependency management | `uv` |
| Deployment | Streamlit Community Cloud |

## Project Structure

```
gesture-recognition/
├── app.py                  # Main Streamlit web app (single-session version)
├── app_multi.py            # Multi-user web app (per-session isolated state) — used for live deployment
├── data_utils.py           # Landmark normalization + feature extraction (shared by training & inference)
├── gesture_logic.py        # Static gesture rules (finger-state geometry)
├── record_gesture_data.py  # Script used to record training sequences for dynamic gestures
├── train_model.py          # Trains the LSTM on recorded gesture sequences
├── gesture_model.keras     # Trained LSTM model (committed — required for deployment)
├── label_mapping.json      # Maps model output indices to gesture names
├── packages.txt            # System-level (apt) dependencies required for deployment
├── pyproject.toml          # Python dependencies (managed by uv)
└── gesture_data/           # Recorded training sequences (not committed — see .gitignore)
```

## Running Locally

**Prerequisites:** Python 3.11+, [uv](https://docs.astral.sh/uv/) installed.

```bash
# Clone the repo
git clone https://github.com/Afraaz-1312/Gesture-Recognition.git
cd Gesture-Recognition

# Install dependencies
uv sync

# Run the app
uv run streamlit run app_multi.py
```

This opens the app at `http://localhost:8501` in your browser.

> **Note:** the project uses `opencv-python-headless` (required for cloud deployment, since server environments have no display). This works fine for the Streamlit app. If you want to run the older desktop test scripts (`test_hands.py`, `main.py`) which use `cv2.imshow()` to open a native window, swap to the GUI-capable build first:
> ```bash
> uv remove opencv-python-headless
> uv add opencv-python
> ```

## Training the Dynamic Gesture Model

The LSTM model was trained on self-recorded gesture sequences for full personalization (see architecture notes above for reasoning). To retrain or add new gestures:

1. Record samples for each gesture class:
   ```bash
   uv run record_gesture_data.py <GestureName>
   ```
   Press `r` to record a 1-second (30-frame) sample, `q` to quit. Aim for 30-40 samples per class, including a `None` class for idle/non-gesture movement.

2. Train the model:
   ```bash
   uv run train_model.py
   ```
   This outputs `gesture_model.keras` and `label_mapping.json`.

## Deployment Notes

Deployed on Streamlit Community Cloud. A few environment-specific fixes were required to get MediaPipe + OpenCV running in a headless Linux container — documented here in case you fork this and hit the same issues:

- `opencv-python` → `opencv-python-headless` (no GUI dependencies needed/available in a server environment)
- `packages.txt` includes system libraries MediaPipe's native binary requires at runtime: `libgl1`, `libglib2.0-0t64`, `libgomp1`, `libegl1`, `libgles2`, `libsm6`, `libxext6`, `libxrender1`
- MediaPipe/TensorFlow model loading and WebRTC video processing are wrapped in a per-session class (`VideoProcessorBase`) rather than shared global state, so multiple concurrent users don't interfere with each other's hand-tracking state

## Known Limitations

- Dynamic gesture model was trained on one person's hand/signing style — accuracy may vary for other users, lighting conditions, or camera angles
- "Help" gesture requires both hands visible in frame simultaneously
- Free-tier cloud hosting may sleep after inactivity — first load after idle time can be slow

