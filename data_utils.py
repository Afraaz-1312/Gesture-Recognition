import numpy as np


def normalize_hand(hand_landmarks):
    """
    Converts 21 MediaPipe landmarks into a 63-length feature vector,
    normalized to be position- and scale-invariant:
    - Translated so the wrist (landmark 0) is the origin
    - Scaled by the wrist-to-middle-finger-MCP distance (proxy for hand size)
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = coords[0]
    coords = coords - wrist  # translate: wrist becomes origin

    scale_ref = np.linalg.norm(coords[9])  # middle finger MCP, post-translation
    if scale_ref < 1e-6:
        scale_ref = 1e-6  # avoid divide-by-zero on degenerate frames

    coords = coords / scale_ref
    return coords.flatten()  # shape (63,)


def build_frame_features(result):
    """
    Takes one frame's HandLandmarker result, returns a fixed-length
    126-feature vector: [Left hand 63 features, Right hand 63 features].
    Missing hand(s) are zero-padded.
    """
    left = np.zeros(63)
    right = np.zeros(63)

    if result.hand_landmarks:
        for idx, hand_landmarks in enumerate(result.hand_landmarks):
            label = result.handedness[idx][0].category_name
            features = normalize_hand(hand_landmarks)
            if label == "Left":
                left = features
            elif label == "Right":
                right = features

    return np.concatenate([left, right])  # shape (126,)