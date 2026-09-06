import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys
from data_utils import build_frame_features

SEQUENCE_LENGTH = 30
DATA_DIR = "gesture_data"
MODEL_PATH = "hand_landmarker.task"

if len(sys.argv) != 2:
    print("Usage: uv run record_gesture_data.py <label>")
    print("Example: uv run record_gesture_data.py Hello")
    sys.exit(1)

label = sys.argv[1]
label_dir = os.path.join(DATA_DIR, label)
os.makedirs(label_dir, exist_ok=True)
existing = len(os.listdir(label_dir))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
start_time = time.time()

print(f"Recording label: '{label}'. {existing} samples already saved.")
print("Press 'r' to record a sequence, 'q' to quit.")

recording = False
buffer = []
sample_count = existing

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if recording:
        features = build_frame_features(result)
        buffer.append(features)
        cv2.putText(frame, f"RECORDING {len(buffer)}/{SEQUENCE_LENGTH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if len(buffer) == SEQUENCE_LENGTH:
            save_path = os.path.join(label_dir, f"seq_{sample_count}.npy")
            np.save(save_path, np.array(buffer))
            print(f"Saved sample {sample_count} for '{label}'")
            sample_count += 1
            buffer = []
            recording = False
    else:
        cv2.putText(frame, f"Label: {label} | Samples: {sample_count} | Press 'r' to record",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and not recording:
        recording = True
        buffer = []
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()