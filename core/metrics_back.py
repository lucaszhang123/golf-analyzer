"""
Back view view metrics extraction.

From a back view camera (positioned behind the golfer, looking toward target):

WHAT WE CAN MEASURE:
  - Spine angle (forward bend from vertical) — best angle for this
  - Spine angle maintenance through downswing
  - Hip hinge depth at address
  - Lead arm plane at top (above/on/below shoulder line)
  - Knee flex — both knees visible in profile
  - Head position relative to ball (forward press / hanging back)
  - Forward shaft lean at impact
  - Hip slide vs hip turn in profile

WHAT CHANGES vs FACE-ON:
  - Rotation (shoulder/hip turn) is NOT reliable from DTL — golfer faces away
  - Weight shift IS visible as lateral movement toward target
  - Spine tilt (lateral) is NOT visible — spine is in profile
  - Spine angle (forward bend) IS the primary spine metric

COORDINATE CONVENTION (DTL camera):
  - Golfer faces AWAY from camera (toward target)
  - x-axis: right = toward target (lead side), left = away from target (trail side)
  - y-axis: up = decreasing y (standard image coords)
  - The lead shoulder appears on the LEFT side of frame (closer to target)
  - The trail shoulder appears on the RIGHT side of frame

All angles in degrees.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .landmarks import LM, get_point, get_point_3d, midpoint, angle_between, visibility


@dataclass
class BackMetrics:
    """Biomechanical metrics extracted from a back view frame."""

    # --- Spine / posture (primary DTL metrics) ---
    spine_angle_deg: Optional[float] = None        # forward bend from vertical (0=upright, 30-45=ideal at address)
    hip_hinge_deg: Optional[float] = None          # forward bend of hips specifically

    # --- Spine angle maintenance ---
    spine_angle_change: Optional[float] = None     # relative to address (filled after normalization)

    # --- Arm plane ---
    lead_arm_angle_deg: Optional[float] = None     # angle of lead arm from horizontal at top
    trail_arm_angle_deg: Optional[float] = None

    # --- Knee flex (profile view) ---
    lead_knee_flex_deg: Optional[float] = None     # 180 = straight, 150 = good flex
    trail_knee_flex_deg: Optional[float] = None

    # --- Head position ---
    head_forward_px: Optional[float] = None        # how far head is in front of hips (forward press)
    head_movement_px: Optional[float] = None       # displacement from address

    # --- Hip movement ---
    hip_slide_px: Optional[float] = None           # lateral slide toward target (px)
    hip_depth_px: Optional[float] = None           # forward/backward hip movement

    # --- Wrist / hand position ---
    hand_height_y: Optional[float] = None          # for phase detection (same as face-on)
    wrist_hinge_deg: Optional[float] = None

    # --- Shoulder plane ---
    shoulder_plane_deg: Optional[float] = None     # angle of shoulder line from horizontal (DTL)

    # --- Raw storage for normalization ---
    _spine_angle_raw: Optional[float] = None
    _hip_x_raw: Optional[float] = None
    _head_x_raw: Optional[float] = None
    _head_y_raw: Optional[float] = None

    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def extract_back_metrics(
    landmarks,
    image_w: int,
    image_h: int,
    handedness: str = "right",
) -> BackMetrics:
    """
    Extract back view metrics from MediaPipe landmarks.

    For a RH golfer filmed from behind (DTL):
      - Lead side (left hand) appears on LEFT of frame
      - Trail side (right hand) appears on RIGHT of frame
      - Golfer faces AWAY from camera
    """

    def pt(idx: LM) -> np.ndarray:
        return get_point(landmarks, idx, image_w, image_h)

    def pt3(idx: LM) -> np.ndarray:
        return get_point_3d(landmarks, idx)

    m = BackMetrics()

    # For RH golfer DTL: lead=left, trail=right (same landmark assignment)
    if handedness == "right":
        lead_shoulder, trail_shoulder = LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER
        lead_hip,      trail_hip      = LM.LEFT_HIP,       LM.RIGHT_HIP
        lead_knee,     trail_knee     = LM.LEFT_KNEE,      LM.RIGHT_KNEE
        lead_ankle,    trail_ankle    = LM.LEFT_ANKLE,     LM.RIGHT_ANKLE
        lead_elbow,    trail_elbow    = LM.LEFT_ELBOW,     LM.RIGHT_ELBOW
        lead_wrist,    trail_wrist    = LM.LEFT_WRIST,     LM.RIGHT_WRIST
    else:
        lead_shoulder, trail_shoulder = LM.RIGHT_SHOULDER, LM.LEFT_SHOULDER
        lead_hip,      trail_hip      = LM.RIGHT_HIP,      LM.LEFT_HIP
        lead_knee,     trail_knee     = LM.RIGHT_KNEE,     LM.LEFT_KNEE
        lead_ankle,    trail_ankle    = LM.RIGHT_ANKLE,    LM.LEFT_ANKLE
        lead_elbow,    trail_elbow    = LM.RIGHT_ELBOW,    LM.LEFT_ELBOW
        lead_wrist,    trail_wrist    = LM.RIGHT_WRIST,    LM.LEFT_WRIST

    key_lms = [lead_shoulder, trail_shoulder, lead_hip, trail_hip,
               lead_knee, trail_knee]
    m.confidence = float(np.mean([visibility(landmarks, l) for l in key_lms]))

    if m.confidence < 0.35:
        return m

    # 2D pixel positions
    ls  = pt(lead_shoulder)
    ts  = pt(trail_shoulder)
    lh  = pt(lead_hip)
    th  = pt(trail_hip)
    lk  = pt(lead_knee)
    tk  = pt(trail_knee)
    la  = pt(lead_ankle)
    ta  = pt(trail_ankle)
    le  = pt(lead_elbow)
    te  = pt(trail_elbow)
    lw  = pt(lead_wrist)
    tw  = pt(trail_wrist)
    nose= pt(LM.NOSE)

    mid_shoulder = midpoint(ls, ts)
    mid_hip      = midpoint(lh, th)

    # --- Spine angle (forward bend from vertical) ---
    # In DTL view, the spine runs from mid_hip to mid_shoulder
    # Angle from vertical: 0 = upright, positive = bending forward
    spine_vec = mid_shoulder - mid_hip
    # Vertical in image = (0, -1) (upward)
    vert = np.array([0.0, -1.0])
    norm = np.linalg.norm(spine_vec)
    if norm > 1:
        cos_a = np.clip(np.dot(spine_vec / norm, vert), -1, 1)
        m.spine_angle_deg = float(np.degrees(np.arccos(cos_a)))
        m._spine_angle_raw = m.spine_angle_deg

    # --- Hip hinge depth ---
    # How much the hips are pushed back / forward bent
    # Measured as horizontal offset of mid_hip relative to mid_ankle
    mid_ankle = midpoint(la, ta)
    hip_forward = float(mid_hip[0] - mid_ankle[0])  # + = hip forward of feet
    m.hip_hinge_deg = hip_forward  # store as pixel offset (normalized later)
    m._hip_x_raw = float(mid_hip[0])

    # --- Lead arm plane at top ---
    # Angle of lead arm (shoulder→elbow→wrist line) from horizontal
    lead_arm_vec = lw - ls
    la_norm = np.linalg.norm(lead_arm_vec)
    if la_norm > 1:
        m.lead_arm_angle_deg = float(
            np.degrees(np.arctan2(-lead_arm_vec[1], lead_arm_vec[0]))
        )

    # Trail arm
    trail_arm_vec = tw - ts
    ta_norm = np.linalg.norm(trail_arm_vec)
    if ta_norm > 1:
        m.trail_arm_angle_deg = float(
            np.degrees(np.arctan2(-trail_arm_vec[1], trail_arm_vec[0]))
        )

    # --- Knee flex (profile — both knees visible in DTL) ---
    m.lead_knee_flex_deg  = angle_between(lh, lk, la)
    m.trail_knee_flex_deg = angle_between(th, tk, ta)

    # --- Head position ---
    m._head_x_raw = float(nose[0])
    m._head_y_raw = float(nose[1])
    # Forward of hips
    m.head_forward_px = float(nose[0] - mid_hip[0])

    # --- Shoulder plane angle ---
    # arctan2 returns -180 to +180. We normalize to -90 to +90
    # because shoulder tilt is symmetric — a line pointing left at
    # 163° is the same tilt as one pointing right at -17°.
    # We also ensure the vector always points trail→lead so the
    # sign is consistent: positive = trail shoulder higher than lead.
    shld_vec = ts - ls
    if np.linalg.norm(shld_vec) > 1:
        raw_angle = float(np.degrees(np.arctan2(-shld_vec[1], shld_vec[0])))
        # Normalize to (-90, 90] — fold angles outside this range
        if raw_angle > 90:
            raw_angle = raw_angle - 180
        elif raw_angle < -90:
            raw_angle = raw_angle + 180
        m.shoulder_plane_deg = raw_angle

    # --- Wrist hinge ---
    hand_mid = midpoint(lw, tw)
    m.wrist_hinge_deg = angle_between(le, lw, hand_mid)

    # --- Hand height (for phase detection) ---
    m.hand_height_y = float(hand_mid[1])

    return m