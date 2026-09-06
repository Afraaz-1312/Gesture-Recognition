import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import time
import os
import urllib.request

st.title("Hand Tracking Test (Web)")

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

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

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    h, w, _ = img.shape
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for start_idx, end_idx in HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h)
                x2, y2 = int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="hand-tracking-test",
    video_frame_callback=video_frame_callback,
)