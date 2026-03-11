"""
Signature Verification System - Prediction Module
Author: Sairam Odela
"""

import os
import numpy as np
import joblib
import cv2
from features import extract_features

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_model():
    model_path  = os.path.join(MODEL_DIR, 'signature_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run train.py first.")
    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler


def predict_signature(image_path: str) -> dict:
    """
    Predict whether a signature is Genuine or Forged.

    Returns:
        {
          'label'      : 'Genuine' | 'Forged',
          'confidence' : float (0–100),
          'genuine_prob': float,
          'forged_prob' : float,
        }
    """
    clf, scaler = load_model()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    feats  = extract_features(img).reshape(1, -1)
    scaled = scaler.transform(feats)

    proba       = clf.predict_proba(scaled)[0]   # [P(forged), P(genuine)]
    forged_prob = proba[0]
    genuine_prob = proba[1]
    label       = 'Genuine' if genuine_prob >= 0.5 else 'Forged'
    confidence  = max(genuine_prob, forged_prob) * 100

    return {
        'label'       : label,
        'confidence'  : round(confidence, 2),
        'genuine_prob': round(genuine_prob * 100, 2),
        'forged_prob' : round(forged_prob  * 100, 2),
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_signature_image>")
        sys.exit(1)
    result = predict_signature(sys.argv[1])
    print(f"\nPrediction  : {result['label']}")
    print(f"Confidence  : {result['confidence']}%")
    print(f"Genuine Prob: {result['genuine_prob']}%")
    print(f"Forged Prob : {result['forged_prob']}%")
