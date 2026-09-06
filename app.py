import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
import urllib.request
from collections import deque
from tensorflow import keras
from data_utils import build_frame_features
from gesture_logic import get_finger_states, classify_gesture

st.title("Gesture Recognition — Live")

SEQUENCE_LENGTH = 30
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DYNAMIC_MODEL_PATH = "gesture_model.keras"
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_FRAMES = 30

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

@st.cache_resource
def load_dynamic_model():
    model = keras.models.load_model(DYNAMIC_MODEL_PATH)
    with open("label_mapping.json") as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    return model, index_to_label

@st.cache_resource
def load_landmarker():
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
    return HandLandmarker.create_from_options(options)

dynamic_model, index_to_label = load_dynamic_model()
landmarker = load_landmarker()

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

start_time = time.time()
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
state = {"cooldown": 0, "last_dynamic_label": ""}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = img.shape
    static_label = ""

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
                x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

        if len(result.hand_landmarks) == 1:
            finger_states = get_finger_states(result.hand_landmarks[0], result.handedness[0][0].category_name)
            static_label = classify_gesture(finger_states)

    features = build_frame_features(result)
    frame_buffer.append(features)

    #cv2.putText(img, f"Buffer: {len(frame_buffer)}/{SEQUENCE_LENGTH} | Cooldown: {state['cooldown']}",
                #(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    dynamic_label = ""
    if len(frame_buffer) == SEQUENCE_LENGTH and state["cooldown"] == 0:
        sequence = np.expand_dims(np.array(frame_buffer), axis=0)
        predictions = dynamic_model.predict(sequence, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx]
        predicted_label = index_to_label[predicted_idx]

        if predicted_label != "None" and confidence >= CONFIDENCE_THRESHOLD:
            dynamic_label = f"{predicted_label} ({confidence:.0%})"
            state["last_dynamic_label"] = dynamic_label
            state["cooldown"] = COOLDOWN_FRAMES

    if state["cooldown"] > 0:
        state["cooldown"] -= 1
        dynamic_label = state["last_dynamic_label"]

    display_text = dynamic_label if dynamic_label else static_label
    cv2.putText(img, display_text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="gesture-recognition",
    video_frame_callback=video_frame_callback,
)