import cv2
import mediapipe as mp
import numpy as np
import urllib.request
import os
import time
from gesture_logic import get_finger_states, classify_gesture

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# Download the model once, locally, if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

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

# Standard 21-point hand connections (since drawing_utils no longer ships with mp.solutions)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (5,9),(9,10),(10,11),(11,12),     # middle
    (9,13),(13,14),(14,15),(15,16),   # ring
    (13,17),(17,18),(18,19),(19,20),  # pinky
    (0,17)                            # palm base
]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Hand tracking started. Press 'q' to quit.")
start_time = time.time()

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

    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            handedness_label = result.handedness[idx][0].category_name  # "Left" or "Right"

            # Draw connections
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
                x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw landmark points
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Gesture classification
            finger_states = get_finger_states(hand_landmarks, handedness_label)
            gesture = classify_gesture(finger_states)

                        # Debug overlay: show raw finger states
            debug_text = f"T:{int(finger_states[0])} I:{int(finger_states[1])} M:{int(finger_states[2])} R:{int(finger_states[3])} P:{int(finger_states[4])}"
            cv2.putText(frame, debug_text, (10, 30 + idx * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display the gesture label above the wrist
            wrist_x, wrist_y = int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h)
            cv2.putText(frame, gesture, (wrist_x - 50, wrist_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()