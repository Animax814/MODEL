import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Creates folder if not exists
def create_label_folder(base_path, label):
    label_path = os.path.join(base_path, label)
    os.makedirs(label_path, exist_ok=True)
    return label_path

# Extract 21 landmarks per hand
def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None

    all_hands = []

    for hand_landmarks in results.multi_hand_landmarks:
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        all_hands.append(coords)

    # Pad to ALWAYS two hands (2 × 63 = 126 values)
    if len(all_hands) == 1:
        all_hands.append([0] * 63)

    return np.array(all_hands).flatten()

# Main data recording function
def record_samples(label, save_dir="../data", seq_length=30, num_samples=40):

    label_path = create_label_folder(save_dir, label)

    cap = cv2.VideoCapture(0)
    print(f"[INFO] Recording samples for label: {label}")

    sample_count = 0

    while sample_count < num_samples:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Draw hand skeleton
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        cv2.putText(frame,
                    f"Press 'S' to record sample {sample_count+1}/{num_samples}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        cv2.imshow("Data Collection - Hand Skeleton", frame)
        key = cv2.waitKey(1)

        if key == ord('s'):
            seq = []
            print(f"[INFO] Recording sample {sample_count+1}/{num_samples}...")

            while len(seq) < seq_length:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Draw skeleton during recording
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
                    seq.append(landmarks)
                else:
                    # If hand not detected → skip frame instead of saving noise
                    continue

                cv2.imshow("Data Collection - Hand Skeleton", frame)
                cv2.waitKey(1)

            seq = np.array(seq)
            file_path = os.path.join(label_path, f"{label}_{sample_count}.npy")
            np.save(file_path, seq)

            print(f"[SAVED] {file_path}")
            sample_count += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Data collection complete.")

# Run example
if __name__ == "__main__":
    record_samples("little bit", num_samples=40)
