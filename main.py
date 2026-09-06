import cv2
import mediapipe as mp
import numpy as np
import time
import json
from collections import deque
from tensorflow import keras
from data_utils import build_frame_features
from gesture_logic import get_finger_states, classify_gesture

SEQUENCE_LENGTH = 30
MODEL_PATH = "hand_landmarker.task"
DYNAMIC_MODEL_PATH = "gesture_model.keras"
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_FRAMES = 30  # ~1 second at 30fps, prevents repeat-fire

# --- Load dynamic gesture model + label mapping ---
dynamic_model = keras.models.load_model(DYNAMIC_MODEL_PATH)
with open("label_mapping.json") as f:
    label_to_index = json.load(f)
index_to_label = {v: k for k, v in label_to_index.items()}

# --- MediaPipe setup ---
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

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
start_time = time.time()

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
cooldown = 0
last_dynamic_label = ""

print("Full gesture recognition running. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = frame.shape

    # --- Draw hand skeleton + run static gesture logic ---
    static_label = ""
    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            handedness_label = result.handedness[idx][0].category_name

            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
                x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Only run static classification when exactly one hand (avoids weirdness with two-hand static rules)
        if len(result.hand_landmarks) == 1:
            finger_states = get_finger_states(result.hand_landmarks[0], result.handedness[0][0].category_name)
            static_label = classify_gesture(finger_states)

    # --- Dynamic gesture model ---
    features = build_frame_features(result)
    frame_buffer.append(features)

    dynamic_label = ""
    if len(frame_buffer) == SEQUENCE_LENGTH and cooldown == 0:
        sequence = np.expand_dims(np.array(frame_buffer), axis=0)  # shape (1, 30, 126)
        predictions = dynamic_model.predict(sequence, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = index_to_label[predicted_idx]

        if predicted_label != "None" and confidence >= CONFIDENCE_THRESHOLD:
            dynamic_label = f"{predicted_label} ({confidence:.0%})"
            last_dynamic_label = dynamic_label
            cooldown = COOLDOWN_FRAMES

    if cooldown > 0:
        cooldown -= 1
        dynamic_label = last_dynamic_label

    # --- Display ---
    display_text = dynamic_label if dynamic_label else static_label
    cv2.putText(frame, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()