"""
MediaPipe landmark indices and helper accessors.
Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""

from enum import IntEnum
import numpy as np


class LM(IntEnum):
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


def get_point(landmarks, idx: LM, image_w: int, image_h: int) -> np.ndarray:
    """Return (x, y) pixel coords for a landmark."""
    lm = landmarks[int(idx)]
    return np.array([lm.x * image_w, lm.y * image_h])


def get_point_3d(landmarks, idx: LM) -> np.ndarray:
    """Return (x, y, z) normalized coords for a landmark."""
    lm = landmarks[int(idx)]
    return np.array([lm.x, lm.y, lm.z])


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2


def angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """
    Angle at `vertex` formed by vectors vertex->a and vertex->b.
    Returns degrees in [0, 180].
    """
    va = a - vertex
    vb = b - vertex
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    cos_theta = np.clip(np.dot(va, vb) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def visibility(landmarks, idx: LM) -> float:
    """Return visibility score [0,1] for a landmark."""
    return landmarks[int(idx)].visibility
