#!/usr/bin/env python3
"""
src/model_manager.py - Zero-Dataset Cold-Start Model Manager & LBP Engine

Classical Computer Vision Pipeline:
- CLAHE Preprocessing & Canonical 100x100 Normalization
- Uniform Local Binary Patterns (LBP, R=1, P=8)
- 8x8 Block-Partitioned 59-bin Histogram Concatenation (3,776-dim feature vector)
- NearestNeighbors (n_neighbors=1, metric='euclidean') with Intra-Class Rejection Thresholding
- Cold-Start Fault Tolerance (Graceful False return if dataset is empty)
"""

import os
import pickle
import datetime
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Canonical paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "model.pkl"
ALT_MODEL_PATH = BASE_DIR / "src" / "model.pkl"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

DEFAULT_FALLBACK_THRESHOLD = 0.28

def _build_uniform_lbp_lookup() -> np.ndarray:
    """
    Constructs a lookup table mapping 256 8-bit patterns to 59 uniform LBP bins.
    Uniform patterns (transitions <= 2) are assigned individual bins 0..57.
    All non-uniform patterns are pooled into bin 58.
    """
    table = np.zeros(256, dtype=np.int32)
    current_label = 0
    for v in range(256):
        bits = [(v >> i) & 1 for i in range(8)]
        transitions = sum(bits[i] != bits[(i + 1) % 8] for i in range(8))
        if transitions <= 2:
            table[v] = current_label
            current_label += 1
        else:
            table[v] = 58
    return table

LOOKUP_TABLE = _build_uniform_lbp_lookup()


def preprocess_face(face_crop: np.ndarray) -> np.ndarray:
    """
    Standardizes face crop:
    1. Converts BGR/RGB to Grayscale.
    2. Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
       with clipLimit=2.0 and tileGridSize=(8, 8).
    3. Resizes to canonical 100x100 pixels.
    """
    if face_crop is None or face_crop.size == 0:
        raise ValueError("Invalid or empty face crop received.")

    # 1. Grayscale conversion
    if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    elif len(face_crop.shape) == 3 and face_crop.shape[2] == 4:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGRA2GRAY)
    else:
        gray = face_crop.copy()

    # 2. Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # 3. Canonical 100x100 resize
    canonical = cv2.resize(equalized, (100, 100), interpolation=cv2.INTER_AREA)
    return canonical


def extract_lbp_feature(face_100x100: np.ndarray) -> np.ndarray:
    """
    Extracts 3,776-dimensional Uniform LBP feature vector:
    - Padded circular neighborhood evaluation (R=1, P=8)
    - Uniform LBP pattern indexing (59 bins)
    - Spatial 8x8 block partitioning (64 cells)
    - 59-bin normalized histogram per block
    - Concatenation: 64 blocks * 59 bins = 3,776 features
    """
    if face_100x100.shape != (100, 100):
        face_100x100 = cv2.resize(face_100x100, (100, 100), interpolation=cv2.INTER_AREA)

    # Pad with border reflection for boundary invariance
    padded = cv2.copyMakeBorder(face_100x100, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = padded[1:-1, 1:-1]

    # 8 neighbors with radius R=1
    neighbors = [
        padded[1:-1, 2:],   # (0, 1)
        padded[:-2, 2:],    # (-1, 1)
        padded[:-2, 1:-1],  # (-1, 0)
        padded[:-2, :-2],   # (-1, -1)
        padded[1:-1, :-2],  # (0, -1)
        padded[2:, :-2],    # (1, -1)
        padded[2:, 1:-1],   # (1, 0)
        padded[2:, 2:],     # (1, 1)
    ]

    lbp_code = np.zeros_like(center, dtype=np.uint8)
    for i, n in enumerate(neighbors):
        lbp_code |= ((n >= center).astype(np.uint8) << i)

    # Map to 59 uniform bins
    mapped_lbp = LOOKUP_TABLE[lbp_code]

    # Partition 100x100 into 8x8 grid (64 blocks)
    h_steps = np.linspace(0, 100, 9, dtype=int)
    w_steps = np.linspace(0, 100, 9, dtype=int)

    block_histograms = []
    for i in range(8):
        for j in range(8):
            cell = mapped_lbp[h_steps[i]:h_steps[i+1], w_steps[j]:w_steps[j+1]]
            hist = np.bincount(cell.ravel(), minlength=59).astype(np.float32)
            # Normalize histogram to sum to 1.0 with epsilon
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist /= hist_sum
            block_histograms.append(hist)

    feature_vector = np.concatenate(block_histograms).astype(np.float32)
    return feature_vector


def compute_intra_class_distances(features: np.ndarray, labels: List[str]) -> List[float]:
    """
    Computes pairwise Euclidean distances between feature vectors of the same enrolled subject.
    """
    unique_labels = sorted(list(set(labels)))
    intra_distances = []

    for label in unique_labels:
        idx = [i for i, lbl in enumerate(labels) if lbl == label]
        if len(idx) < 2:
            continue
        subj_feats = features[idx]
        for a in range(len(subj_feats)):
            for b in range(a + 1, len(subj_feats)):
                dist = float(np.linalg.norm(subj_feats[a] - subj_feats[b]))
                intra_distances.append(dist)

    return intra_distances


def train_or_update_model(
    dataset_dir: Optional[Path] = None,
    save_path: Optional[Path] = None
) -> bool:
    """
    Scans dataset directory, extracts LBP features, fits distance-gated kNN,
    computes rejection threshold, and serializes model to model.pkl.

    Cold-Start Tolerance:
    If dataset directory is missing, empty, or contains zero valid image crops,
    gracefully returns False without raising exceptions.
    """
    d_dir = dataset_dir or DATASET_DIR
    out_path = save_path or MODEL_PATH

    if not d_dir.exists() or not d_dir.is_dir():
        print(f"[ModelManager] Dataset directory '{d_dir}' does not exist. Cold-start graceful exit.")
        return False

    student_dirs = [d for d in d_dir.iterdir() if d.is_dir()]
    if not student_dirs:
        print("[ModelManager] Zero student folders found in dataset. Cold-start graceful exit.")
        return False

    features = []
    labels = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for student_dir in student_dirs:
        student_name = student_dir.name
        img_files = [f for f in student_dir.iterdir() if f.suffix.lower() in valid_extensions]

        for img_file in img_files:
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                preprocessed = preprocess_face(img)
                feat = extract_lbp_feature(preprocessed)
                features.append(feat)
                labels.append(student_name)
            except Exception as e:
                print(f"[ModelManager] Warning: Failed to process {img_file}: {e}")

    if not features:
        print("[ModelManager] Zero valid face crops found. Cold-start graceful exit.")
        return False

    features_array = np.array(features, dtype=np.float32)

    # Fit 1-NN model with Euclidean metric
    knn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    knn.fit(features_array)

    # Compute empirical rejection threshold
    intra_dists = compute_intra_class_distances(features_array, labels)
    if intra_dists and len(intra_dists) >= 2:
        mean_dist = float(np.mean(intra_dists))
        std_dist = float(np.std(intra_dists))
        threshold = float(mean_dist + 2.5 * std_dist)
        print(f"[ModelManager] Empirical Threshold calculated: {threshold:.4f} (mean={mean_dist:.4f}, std={std_dist:.4f})")
    else:
        threshold = DEFAULT_FALLBACK_THRESHOLD
        print(f"[ModelManager] Insufficient intra-class pairs. Using calibrated fallback threshold: {threshold:.4f}")

    payload: Dict[str, Any] = {
        "knn": knn,
        "features": features_array,
        "labels": labels,
        "classes": sorted(list(set(labels))),
        "threshold": threshold,
        "trained_at": datetime.datetime.now().isoformat(),
        "total_samples": len(labels)
    }

    # Save to primary and src locations
    for p in [out_path, ALT_MODEL_PATH]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "wb") as f:
                pickle.dump(payload, f)
            print(f"[ModelManager] Model serialized successfully to {p}")
        except Exception as e:
            print(f"[ModelManager] Error saving model to {p}: {e}")

    return True


def load_model(model_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """
    Safely loads serialized model. Returns None if missing or corrupted.
    """
    paths_to_check = [model_path] if model_path else [MODEL_PATH, ALT_MODEL_PATH]

    for p in paths_to_check:
        if p and p.exists() and p.is_file():
            try:
                with open(p, "rb") as f:
                    model_dict = pickle.load(f)
                return model_dict
            except Exception as e:
                print(f"[ModelManager] Failed loading model from {p}: {e}")

    return None


def predict_face(
    face_crop: np.ndarray,
    model_dict: Optional[Dict[str, Any]] = None
) -> Tuple[str, float, bool]:
    """
    Inference helper:
    Takes raw face crop, runs preprocessing and LBP feature extraction,
    queries kNN, and performs distance threshold verification.

    Returns:
    (predicted_label, distance, is_verified)
    If distance > threshold, returns ("Unknown", distance, False).
    """
    if model_dict is None:
        model_dict = load_model()

    if model_dict is None or "knn" not in model_dict:
        return ("Unknown", 999.0, False)

    preprocessed = preprocess_face(face_crop)
    feat = extract_lbp_feature(preprocessed).reshape(1, -1)

    knn = model_dict["knn"]
    labels = model_dict["labels"]
    threshold = model_dict.get("threshold", DEFAULT_FALLBACK_THRESHOLD)

    distances, indices = knn.kneighbors(feat, n_neighbors=1)
    distance = float(distances[0][0])
    matched_idx = int(indices[0][0])
    candidate_label = labels[matched_idx]

    if distance <= threshold:
        return (candidate_label, distance, True)
    else:
        return ("Unknown", distance, False)


if __name__ == "__main__":
    print("[ModelManager Test Execution]")
    result = train_or_update_model()
    print(f"Cold-start execution result (expected False on empty dataset): {result}")
