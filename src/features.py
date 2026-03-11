"""
Feature Extraction for Signature Verification
Extracts handcrafted image features from grayscale signature images.
Author: Sairam Odela
"""

import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.measure import regionprops, label as sk_label


def preprocess(img: np.ndarray, size: tuple = (128, 256)) -> np.ndarray:
    """Resize, denoise, and binarise a grayscale signature image."""
    img = cv2.resize(img, size)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def geometric_features(binary: np.ndarray) -> np.ndarray:
    """Aspect ratio, pixel density, centroid position, bounding-box ratio."""
    h, w = binary.shape
    pixel_density = np.sum(binary > 0) / (h * w)

    coords = np.argwhere(binary > 0)
    if len(coords) == 0:
        return np.zeros(6)

    cy, cx = coords.mean(axis=0)
    cy_norm, cx_norm = cy / h, cx / w

    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    bb_h = (r_max - r_min + 1) / h
    bb_w = (c_max - c_min + 1) / w
    aspect_ratio = bb_w / (bb_h + 1e-6)

    return np.array([pixel_density, cy_norm, cx_norm, bb_h, bb_w, aspect_ratio])


def hog_features(gray: np.ndarray) -> np.ndarray:
    """HOG descriptor — captures edge/gradient patterns."""
    resized = cv2.resize(gray, (128, 256))
    feats = hog(
        resized,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return feats


def lbp_features(gray: np.ndarray, P: int = 8, R: float = 1.0) -> np.ndarray:
    """LBP histogram — captures local texture patterns."""
    resized = cv2.resize(gray, (128, 256))
    lbp = local_binary_pattern(resized, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    return hist


def projection_features(binary: np.ndarray, bins: int = 32) -> np.ndarray:
    """Horizontal and vertical ink projections."""
    h_proj = binary.sum(axis=1).astype(float)
    v_proj = binary.sum(axis=0).astype(float)
    h_hist = np.interp(np.linspace(0, len(h_proj) - 1, bins),
                       np.arange(len(h_proj)), h_proj)
    v_hist = np.interp(np.linspace(0, len(v_proj) - 1, bins),
                       np.arange(len(v_proj)), v_proj)
    # Normalise
    h_hist /= (h_hist.max() + 1e-6)
    v_hist /= (v_hist.max() + 1e-6)
    return np.concatenate([h_hist, v_hist])


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Master feature extractor.
    Input : grayscale image (numpy array).
    Output: 1-D feature vector.
    """
    binary = preprocess(img)
    gray   = cv2.resize(img, (128, 256))

    geo  = geometric_features(binary)          #   6 features
    hog_ = hog_features(gray)                  # 1764 features
    lbp_ = lbp_features(gray)                  #  10 features
    proj = projection_features(binary)         #  64 features

    return np.concatenate([geo, hog_, lbp_, proj])
