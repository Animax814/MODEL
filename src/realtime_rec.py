import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import time

# ------------------------
# Load trained model
# ------------------------
model = load_model("../models/sign_model1.h5")

data = np.load("../models/dataset.npz", allow_pickle=True)
classes = data["classes"]
print("[INFO] Classes:", classes)

# ------------------------
# Speech Setup
# ------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 170)
last_spoken = None

# ------------------------
# MediaPipe Setup
# ------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

SEQ_LEN = 30
sequence = []

prev_time = time.time()

# ------------------------
# Gradient Background
# ------------------------
def apply_gradient(frame):
    h, w, _ = frame.shape
    gradient = np.zeros_like(frame, dtype=np.uint8)

    top_color = np.array([230, 235, 240], dtype=np.uint8)   # light cool
    bottom_color = np.array([120, 130, 150], dtype=np.uint8)  # slate blue

    for y in range(h):
        alpha = y / h
        color = (1 - alpha) * top_color + alpha * bottom_color
        gradient[y, :] = color

    return cv2.addWeighted(gradient, 0.45, frame, 0.55, 0)

# ------------------------
# Extract landmarks
# ------------------------
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    all_hands = []
    for hand_landmarks in results.multi_hand_landmarks:
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        all_hands.append(coords)

    if len(all_hands) == 1:
        all_hands.append([0] * 63)

    return np.array(all_hands).flatten()

# ------------------------
# Normalize
# ------------------------
def normalize(sample):
    sample = sample - np.mean(sample, axis=0)
    max_val = np.max(np.abs(sample))
    if max_val > 0:
        sample = sample / max_val
    return sample

# ------------------------
# REAL-TIME LOOP
# ------------------------
cap = cv2.VideoCapture(0)
print("[INFO] Starting real-time sign recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🎨 Apply Gradient Background
    frame = apply_gradient(frame)

    # ------------------------
    # Glass Top Bar
    # ------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (255, 255, 255), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # Title (shadow + text)
    cv2.putText(frame, "Real-Time Sign Recognition",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (30, 30, 30), 3)
    cv2.putText(frame, "Real-Time Sign Recognition",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (60, 60, 60), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (frame.shape[1] - 160, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (60, 60, 60), 2)

    # ------------------------
    # Hand Processing
    # ------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    landmarks = extract_landmarks(results)

    if landmarks is not None:
        sequence.append(landmarks)
        if len(sequence) > SEQ_LEN:
            sequence.pop(0)

        if len(sequence) == SEQ_LEN:
            sample = normalize(np.array(sequence))
            sample = np.expand_dims(sample, axis=0)

            preds = model.predict(sample, verbose=0)[0]
            class_id = np.argmax(preds)
            percent = preds[class_id] * 100
            label = classes[class_id]

            # Color coding
            color = (0, 200, 0) if percent >= 80 else (0, 180, 255) if percent >= 50 else (0, 0, 255)

            # Speech
            if 70 <= percent <= 90 and label != last_spoken:
                engine.say(label)
                engine.runAndWait()
                last_spoken = label

            # ------------------------
            # Confidence Bar
            # ------------------------
            bar_x, bar_y, bar_w, bar_h = 15, 65, 260, 18

            cv2.rectangle(frame, (bar_x-1, bar_y-1),
                          (bar_x+bar_w+1, bar_y+bar_h+1),
                          (100, 100, 100), 1)

            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x+bar_w, bar_y+bar_h),
                          (200, 200, 200), -1)

            fill = int(bar_w * percent / 100)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x+fill, bar_y+bar_h),
                          color, -1)

            cv2.putText(frame, f"{label} ({percent:.1f}%)",
                        (290, 82), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (40, 40, 40), 2)
    else:
        sequence = []

    cv2.imshow("Real-Time Sign Recognition UI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()