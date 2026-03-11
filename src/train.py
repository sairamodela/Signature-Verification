"""
Signature Verification System - Model Training
Dataset: Custom Signature Dataset
Algorithm: Random Forest Classifier
Author: Sairam Odela
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
from features import extract_features

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
MODEL_DIR    = os.path.join(BASE_DIR, 'models')
DATA_DIR     = os.path.join(BASE_DIR, 'data')
FEATURES_DIR = os.path.join(BASE_DIR, 'features')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


def load_dataset(data_dir: str):
    """
    Expects the following folders in data_dir:
        genuine/   ← genuine signature images (.png / .jpg)
        forged/    ← forged  signature images (.png / .jpg)
    Returns X (feature matrix) and y (labels: 1=genuine, 0=forged).
    """
    X, y = [], []
    classes = {'genuine': 1, 'forged': 0}

    for folder_name, label_val in classes.items():
        folder = os.path.join(data_dir, folder_name)
        if not os.path.exists(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                continue
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feats = extract_features(img)
            X.append(feats)
            y.append(label_val)
        print(f"  Loaded {folder_name}: {len([v for v in y if v == label_val])} images")

    # Save extracted features to CSV
    if len(X) > 0:
        df = pd.DataFrame(X)
        df['label'] = y
        features_csv_path = os.path.join(FEATURES_DIR, 'extracted_features.csv')
        df.to_csv(features_csv_path, index=False)
        print(f"  Features saved to {features_csv_path}")

    return np.array(X), np.array(y)


def train():
    print("=" * 55)
    print("  Signature Verification — Training Pipeline")
    print("=" * 55)

    print("\n[1/4] Loading dataset …")
    X, y = load_dataset(DATA_DIR)
    if len(X) == 0:
        raise RuntimeError(
            "No images found. Check data/genuine/ and data/forged/ folders."
        )
    print(f"  Total samples : {len(X)}")
    print(f"  Feature size  : {X.shape[1]}")

    print("\n[2/4] Preprocessing …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    print("\n[3/4] Training Random Forest …")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    print("\n[4/4] Evaluating …")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {acc * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Forged', 'Genuine']))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save artefacts
    model_path  = os.path.join(MODEL_DIR, 'signature_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(clf,    model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n  Model  saved → {model_path}")
    print(f"  Scaler saved → {scaler_path}")
    print("\nTraining complete!")


if __name__ == '__main__':
    train()
