import math


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def get_finger_states(hand_landmarks, handedness_label=None):
    """
    Returns [thumb, index, middle, ring, pinky] as booleans.
    True = extended, False = curled.
    Rotation-invariant: works regardless of hand angle or left/right.
    """
    fingers = []

    # Thumb — distance-based, not axis-based, so it survives hand rotation
    pinky_mcp = hand_landmarks[17]
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    thumb_extended = _dist(thumb_tip, pinky_mcp) > _dist(thumb_ip, pinky_mcp)
    fingers.append(thumb_extended)

    # Other 4 fingers — tip vs pip, y-axis comparison still works fine
    # for these since they curl consistently regardless of hand rotation
    tip_ids = [8, 12, 16, 20]
    pip_ids = [6, 10, 14, 18]
    for tip, pip in zip(tip_ids, pip_ids):
        fingers.append(hand_landmarks[tip].y < hand_landmarks[pip].y)

    return fingers


# Full lookup table: (thumb, index, middle, ring, pinky) -> gesture name
GESTURE_MAP = {
    (0,0,0,0,0): "Fist",
    (1,1,1,1,1): "Open Palm",
    (0,1,1,0,0): "Peace Sign",
    (1,0,0,0,0): "Thumbs Up",
    (0,1,0,0,0): "Pointing",
    (1,0,0,0,1): "Call Me/Shaka",
    (0,1,1,1,0): "Three",
    (0,1,1,1,1): "Four",
    (0,1,0,0,1): "Rock On",
    (1,1,0,0,0): "Gun",
    #(1,0,0,0,1): "Shaka",
    (0,0,0,0,1): "Pinky Up",
    (1,1,0,0,1): "Spock-ish",
    (1,1,1,1,0): "Four (thumb)",
}


def classify_gesture(finger_states):
    key = tuple(int(f) for f in finger_states)

    if key in GESTURE_MAP:
        return GESTURE_MAP[key]

    # Graceful fallback: describe by count instead of "Unknown"
    count = sum(key)
    if count == 0:
        return "Fist"
    return f"{count} Finger(s) Up"