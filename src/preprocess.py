import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "../data"
SEQ_LEN = 30  # number of frames per sample

def load_dataset():
    X = []
    y = []

    labels = sorted(os.listdir(DATA_DIR))
    print("[INFO] Found labels:", labels)

    for label in labels:
        label_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".npy"):
                seq = np.load(os.path.join(label_path, file))

                # Ensure correct shape (30, 126)
                if seq.shape == (SEQ_LEN, 126):
                    X.append(seq)
                    y.append(label)

    return np.array(X), np.array(y)

def normalize_landmarks(X):
    X_norm = X.copy()

    # Normalize per-sample
    for i in range(len(X_norm)):
        sample = X_norm[i]

        # subtract mean
        sample = sample - np.mean(sample, axis=0)

        # scale to unit length
        max_val = np.max(np.abs(sample))
        if max_val > 0:
            sample = sample / max_val

        X_norm[i] = sample

    return X_norm

def preprocess():
    print("[INFO] Loading samples...")
    X, y = load_dataset()

    print("[INFO] Normalizing skeleton data...")
    X = normalize_landmarks(X)

    print("[INFO] Encoding labels...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("[INFO] Saving dataset...")
    np.savez("../models/dataset.npz", X=X, y=y_encoded, classes=encoder.classes_)

    print("[DONE] Dataset saved to models/dataset.npz")
    print("Classes:", encoder.classes_)

if __name__ == "__main__":
    preprocess()
