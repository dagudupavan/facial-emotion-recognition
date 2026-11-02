"""Shared preprocessing utilities for FER-style 48x48 grayscale models."""
import cv2
import numpy as np

IMAGE_SIZE = (48, 48)

def preprocess_face(face_bgr_or_gray):
    """Return one normalized (1, 48, 48, 1) float32 model input."""
    if face_bgr_or_gray is None or face_bgr_or_gray.size == 0:
        raise ValueError("Face crop is empty.")
    if len(face_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_bgr_or_gray
    resized = cv2.resize(gray, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized[np.newaxis, ..., np.newaxis]
