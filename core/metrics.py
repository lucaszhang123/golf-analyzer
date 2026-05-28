"""
Extract biomechanical metrics from MediaPipe pose landmarks.

ROTATION MEASUREMENT — face-on camera approach:
  arctan2 of the 2D shoulder line is unreliable for face-on video because
  the angle barely changes as the golfer turns (shoulders appear nearly
  horizontal throughout). Instead we use TWO signals combined:

  1. SHOULDER WIDTH RATIO (2D pixels):
     At address facing camera, shoulders appear at maximum apparent width.
     As the golfer turns into the backswing, the trail shoulder moves away
     from camera and apparent pixel width shrinks. This is reliable and
     directly observable from a face-on view.
     rotation_proxy = 1 - (current_width / address_width)
     Scaled to degrees: 0 = square, ~90 = fully turned.

  2. MEDIAPIPE Z-COORDINATE (3D depth):
     MediaPipe provides a z estimate for each landmark (less accurate than
     x/y but useful as a secondary signal). The trail shoulder z increases
     (moves away from camera) as the golfer turns.

  We combine both into a single rotation estimate, normalized so that
  address = 0° and full turn = approximately 90°.

  For hips: same approach using hip landmark widths and z-coords.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .landmarks import LM, get_point, get_point_3d, midpoint, angle_between, visibility


@dataclass
class SwingMetrics:
    # Rotation — estimated degrees from address position (0 = square)
    shoulder_rotation_deg: Optional[float] = None
    hip_rotation_deg: Optional[float] = None
    x_factor_deg: Optional[float] = None

    # Spine / posture
    spine_tilt_deg: Optional[float] = None
    spine_angle_deg: Optional[float] = None

    # Arms
    lead_elbow_angle_deg: Optional[float] = None
    trail_elbow_angle_deg: Optional[float] = None
    wrist_hinge_deg: Optional[float] = None

    # Weight / balance
    weight_shift: Optional[float] = None
    hip_sway_px: Optional[float] = None

    # Head
    head_movement_px: Optional[float] = None

    # Knees
    lead_knee_angle_deg: Optional[float] = None
    trail_knee_angle_deg: Optional[float] = None

    # Hand height — y pixel coordinate of midpoint between both wrists
    # Lower value = higher physical position (y increases downward in image coords)
    hand_height_y: Optional[float] = None

    # Raw width values stored for normalization in analyzer
    _shoulder_width_px: Optional[float] = None
    _hip_width_px: Optional[float] = None
    _shoulder_z_diff: Optional[float] = None  # trail_z - lead_z
    _hip_z_diff: Optional[float] = None

    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def extract_metrics(
    landmarks,
    image_w: int,
    image_h: int,
    handedness: str = "right",
) -> SwingMetrics:

    def pt(idx: LM) -> np.ndarray:
        return get_point(landmarks, idx, image_w, image_h)

    def pt3(idx: LM) -> np.ndarray:
        return get_point_3d(landmarks, idx)

    m = SwingMetrics()

    if handedness == "right":
        lead_shoulder, trail_shoulder = LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER
        lead_hip, trail_hip = LM.LEFT_HIP, LM.RIGHT_HIP
        lead_knee, trail_knee = LM.LEFT_KNEE, LM.RIGHT_KNEE
        lead_ankle, trail_ankle = LM.LEFT_ANKLE, LM.RIGHT_ANKLE
        lead_elbow, trail_elbow = LM.LEFT_ELBOW, LM.RIGHT_ELBOW
        lead_wrist, trail_wrist = LM.LEFT_WRIST, LM.RIGHT_WRIST
    else:
        lead_shoulder, trail_shoulder = LM.RIGHT_SHOULDER, LM.LEFT_SHOULDER
        lead_hip, trail_hip = LM.RIGHT_HIP, LM.LEFT_HIP
        lead_knee, trail_knee = LM.RIGHT_KNEE, LM.LEFT_KNEE
        lead_ankle, trail_ankle = LM.RIGHT_ANKLE, LM.LEFT_ANKLE
        lead_elbow, trail_elbow = LM.RIGHT_ELBOW, LM.LEFT_ELBOW
        lead_wrist, trail_wrist = LM.RIGHT_WRIST, LM.LEFT_WRIST

    key_lms = [lead_shoulder, trail_shoulder, lead_hip, trail_hip,
               lead_knee, trail_knee, lead_elbow, lead_wrist]
    m.confidence = float(np.mean([visibility(landmarks, l) for l in key_lms]))

    if m.confidence < 0.4:
        return m

    # 2D pixel positions
    ls = pt(lead_shoulder)
    ts = pt(trail_shoulder)
    lh = pt(lead_hip)
    th = pt(trail_hip)

    # 3D normalized positions (includes z depth)
    ls3 = pt3(lead_shoulder)
    ts3 = pt3(trail_shoulder)
    lh3 = pt3(lead_hip)
    th3 = pt3(trail_hip)

    # --- Rotation via width ratio + z-depth (face-on camera) ---
    # Width in pixels: how wide do shoulders/hips appear in the image
    shoulder_width_px = float(abs(ts[0] - ls[0]))
    hip_width_px = float(abs(th[0] - lh[0]))

    # Z-depth difference: trail landmark moves away from camera as golfer turns
    # In MediaPipe, z is negative = closer to camera, positive = further
    shoulder_z_diff = float(ts3[2] - ls3[2])  # trail - lead z
    hip_z_diff = float(th3[2] - lh3[2])

    # Store raw values — normalization happens in analyzer after address is identified
    m._shoulder_width_px = shoulder_width_px
    m._hip_width_px = hip_width_px
    m._shoulder_z_diff = shoulder_z_diff
    m._hip_z_diff = hip_z_diff

    # Rotation will be filled in by analyzer after normalization
    # (set to None here; analyzer computes relative values)
    m.shoulder_rotation_deg = None
    m.hip_rotation_deg = None
    m.x_factor_deg = None

    # --- Spine tilt (lateral lean, visible from face-on) ---
    mid_shoulder = midpoint(ls, ts)
    mid_hip = midpoint(lh, th)
    spine_vec = mid_shoulder - mid_hip
    m.spine_tilt_deg = float(np.degrees(np.arctan2(spine_vec[0], -spine_vec[1])))

    # --- Spine angle (forward bend, uses 3D z) ---
    mid_s3 = midpoint(ls3, ts3)
    mid_h3 = midpoint(lh3, th3)
    spine3 = mid_s3 - mid_h3
    vertical = np.array([0, -1, 0])
    cos_a = np.clip(np.dot(spine3, vertical) / (np.linalg.norm(spine3) + 1e-9), -1, 1)
    m.spine_angle_deg = float(np.degrees(np.arccos(cos_a)))

    # --- Elbow angles ---
    le = pt(lead_elbow)
    lw = pt(lead_wrist)
    m.lead_elbow_angle_deg = angle_between(ls, le, lw)

    te = pt(trail_elbow)
    tw = pt(trail_wrist)
    m.trail_elbow_angle_deg = angle_between(ts, te, tw)

    # --- Hand height (y pixel position of wrist midpoint) ---
    hand_mid = midpoint(lw, tw)
    m.hand_height_y = float(hand_mid[1])  # y increases downward; lower = higher physically

    # --- Wrist hinge ---
    m.wrist_hinge_deg = angle_between(le, lw, hand_mid)

    # --- Knee angles ---
    la = pt(lead_ankle)
    m.lead_knee_angle_deg = angle_between(lh, pt(lead_knee), la)
    ta = pt(trail_ankle)
    m.trail_knee_angle_deg = angle_between(th, pt(trail_knee), ta)

    # --- Weight shift ---
    lead_foot_x = pt(lead_ankle)[0]
    trail_foot_x = pt(trail_ankle)[0]
    hip_center_x = midpoint(lh, th)[0]
    foot_span = trail_foot_x - lead_foot_x
    if abs(foot_span) > 1:
        m.weight_shift = float(2 * (hip_center_x - lead_foot_x) / foot_span - 1)
        if handedness == "right":
            m.weight_shift = -m.weight_shift

    return m
