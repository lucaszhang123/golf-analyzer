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

NEW LANDMARKS (v2):
  EARS (7, 8):
    The ear-to-ear line gives a direct read on head rotation that the nose
    alone cannot. As the golfer turns in the backswing the trail ear drops
    and lead ear rises. We compute:
      - head_rotation_deg: angle of ear line from horizontal (0 = level)
      - head_lateral_px: lateral drift of the ear midpoint from address
        (separates drift from rotation — two distinct fault signals)

  HAND LANDMARKS (17-22 — pinky, index, thumb):
    Palm plane: triangle (wrist, index, pinky) defines the palm orientation.
    From face-on this measures lead wrist cup/bow:
      - lead_wrist_cup_deg > 0 = cupped (extended, opens face)
      - lead_wrist_cup_deg < 0 = bowed (flexed, closes face, DJ position)
    Shaft angle proxy: lead wrist → lead index finger line angle from
    vertical at the top of the backswing. Approximates where the club points.
    Wrist hinge direction: radial vs ulnar deviation via thumb position.
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

    # Head — nose-based (kept for back-compat)
    head_movement_px: Optional[float] = None

    # Head — ear-based (new, more informative)
    head_rotation_deg: Optional[float] = None   # ear-line angle from horizontal
                                                 # 0=level, +=trail ear drops (correct BS turn)
                                                 # negative=lead ear drops (reverse pivot)
    head_lateral_px: Optional[float] = None     # lateral drift of ear midpoint from address
                                                 # separates drift from rotation

    # Knees
    lead_knee_angle_deg: Optional[float] = None
    trail_knee_angle_deg: Optional[float] = None

    # Hand height — y pixel coordinate of midpoint between both wrists
    # Lower value = higher physical position (y increases downward in image coords)
    hand_height_y: Optional[float] = None

    # Lead wrist cup/bow — from palm plane (wrist + index + pinky triangle)
    # Positive = cupped (extended) — opens club face, common fault
    # Negative = bowed (flexed) — closes face, DJ/Rahm position
    # None if hand landmarks not visible
    lead_wrist_cup_deg: Optional[float] = None

    # Shaft angle proxy at top — angle of lead wrist→index line from vertical
    # Gives approximate club direction at top of backswing
    shaft_angle_proxy_deg: Optional[float] = None

    # Raw values stored for normalization in analyzer (private, not serialized)
    _shoulder_width_px: Optional[float] = None
    _hip_width_px: Optional[float] = None
    _shoulder_z_diff: Optional[float] = None   # trail_z - lead_z
    _hip_z_diff: Optional[float] = None
    _ear_mid_x_raw: Optional[float] = None     # ear midpoint x for address baseline
    _ear_line_raw: Optional[float] = None      # raw ear-line angle for address baseline

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

    def vis(idx: LM) -> float:
        return visibility(landmarks, idx)

    m = SwingMetrics()

    if handedness == "right":
        lead_shoulder,  trail_shoulder  = LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER
        lead_hip,       trail_hip       = LM.LEFT_HIP,       LM.RIGHT_HIP
        lead_knee,      trail_knee      = LM.LEFT_KNEE,      LM.RIGHT_KNEE
        lead_ankle,     trail_ankle     = LM.LEFT_ANKLE,     LM.RIGHT_ANKLE
        lead_elbow,     trail_elbow     = LM.LEFT_ELBOW,     LM.RIGHT_ELBOW
        lead_wrist,     trail_wrist     = LM.LEFT_WRIST,     LM.RIGHT_WRIST
        lead_ear,       trail_ear       = LM.LEFT_EAR,       LM.RIGHT_EAR
        lead_index,     trail_index     = LM.LEFT_INDEX,     LM.RIGHT_INDEX
        lead_pinky,     trail_pinky     = LM.LEFT_PINKY,     LM.RIGHT_PINKY
        lead_thumb,     trail_thumb     = LM.LEFT_THUMB,     LM.RIGHT_THUMB
    else:
        lead_shoulder,  trail_shoulder  = LM.RIGHT_SHOULDER, LM.LEFT_SHOULDER
        lead_hip,       trail_hip       = LM.RIGHT_HIP,      LM.LEFT_HIP
        lead_knee,      trail_knee      = LM.RIGHT_KNEE,     LM.LEFT_KNEE
        lead_ankle,     trail_ankle     = LM.RIGHT_ANKLE,    LM.LEFT_ANKLE
        lead_elbow,     trail_elbow     = LM.RIGHT_ELBOW,    LM.LEFT_ELBOW
        lead_wrist,     trail_wrist     = LM.RIGHT_WRIST,    LM.LEFT_WRIST
        lead_ear,       trail_ear       = LM.RIGHT_EAR,      LM.LEFT_EAR
        lead_index,     trail_index     = LM.RIGHT_INDEX,    LM.LEFT_INDEX
        lead_pinky,     trail_pinky     = LM.RIGHT_PINKY,    LM.LEFT_PINKY
        lead_thumb,     trail_thumb     = LM.RIGHT_THUMB,    LM.LEFT_THUMB

    # Core confidence: primary structural landmarks only
    key_lms = [lead_shoulder, trail_shoulder, lead_hip, trail_hip,
               lead_knee, trail_knee, lead_elbow, lead_wrist]
    m.confidence = float(np.mean([vis(l) for l in key_lms]))

    if m.confidence < 0.4:
        return m

    # ── 2D pixel positions ──────────────────────────────────────────────────
    ls = pt(lead_shoulder)
    ts = pt(trail_shoulder)
    lh = pt(lead_hip)
    th = pt(trail_hip)
    le = pt(lead_elbow)
    lw = pt(lead_wrist)
    te = pt(trail_elbow)
    tw = pt(trail_wrist)

    # ── 3D positions (z depth) ──────────────────────────────────────────────
    ls3 = pt3(lead_shoulder)
    ts3 = pt3(trail_shoulder)
    lh3 = pt3(lead_hip)
    th3 = pt3(trail_hip)

    # ── Rotation (shoulder width ratio + z-depth blend) ─────────────────────
    m._shoulder_width_px = float(abs(ts[0] - ls[0]))
    m._hip_width_px      = float(abs(th[0] - lh[0]))
    m._shoulder_z_diff   = float(ts3[2] - ls3[2])
    m._hip_z_diff        = float(th3[2] - lh3[2])
    # Filled in by analyzer after address normalization:
    m.shoulder_rotation_deg = None
    m.hip_rotation_deg      = None
    m.x_factor_deg          = None

    # ── Spine tilt (lateral lean from face-on) ───────────────────────────────
    mid_shoulder = midpoint(ls, ts)
    mid_hip      = midpoint(lh, th)
    spine_vec    = mid_shoulder - mid_hip
    m.spine_tilt_deg = float(np.degrees(np.arctan2(spine_vec[0], -spine_vec[1])))

    # ── Spine angle (forward bend via 3D z) ─────────────────────────────────
    mid_s3   = midpoint(ls3, ts3)
    mid_h3   = midpoint(lh3, th3)
    spine3   = mid_s3 - mid_h3
    vertical = np.array([0, -1, 0])
    cos_a    = np.clip(np.dot(spine3, vertical) / (np.linalg.norm(spine3) + 1e-9), -1, 1)
    m.spine_angle_deg = float(np.degrees(np.arccos(cos_a)))

    # ── Elbow angles ─────────────────────────────────────────────────────────
    m.lead_elbow_angle_deg  = angle_between(ls, le, lw)
    m.trail_elbow_angle_deg = angle_between(ts, te, tw)

    # ── Hand height ───────────────────────────────────────────────────────────
    hand_mid       = midpoint(lw, tw)
    m.hand_height_y = float(hand_mid[1])

    # ── Wrist hinge (existing — elbow/wrist/hand_mid angle) ──────────────────
    m.wrist_hinge_deg = angle_between(le, lw, hand_mid)

    # ── Knee angles ───────────────────────────────────────────────────────────
    la = pt(lead_ankle)
    ta = pt(trail_ankle)
    m.lead_knee_angle_deg  = angle_between(lh, pt(lead_knee),  la)
    m.trail_knee_angle_deg = angle_between(th, pt(trail_knee), ta)

    # ── Weight shift ──────────────────────────────────────────────────────────
    foot_span = pt(trail_ankle)[0] - pt(lead_ankle)[0]
    if abs(foot_span) > 1:
        m.weight_shift = float(2 * (mid_hip[0] - pt(lead_ankle)[0]) / foot_span - 1)
        if handedness == "right":
            m.weight_shift = -m.weight_shift

    # ── EAR-BASED HEAD METRICS ────────────────────────────────────────────────
    # Only computed when both ears are reasonably visible.
    ear_vis_threshold = 0.45
    l_ear_vis = vis(lead_ear)
    t_ear_vis = vis(trail_ear)

    if l_ear_vis > ear_vis_threshold and t_ear_vis > ear_vis_threshold:
        le_pt = pt(lead_ear)
        te_pt = pt(trail_ear)

        # Ear-line angle from horizontal.
        # In image coords y increases downward, so a positive angle means
        # the trail ear is lower than the lead ear — which is what happens
        # when the golfer turns into the backswing correctly.
        ear_vec = te_pt - le_pt   # lead→trail direction
        m.head_rotation_deg = float(
            np.degrees(np.arctan2(ear_vec[1], abs(ear_vec[0]) + 1e-9))
        )

        # Raw ear midpoint x stored for address baseline (filled by analyzer)
        ear_mid_x = float((le_pt[0] + te_pt[0]) / 2)
        m._ear_mid_x_raw  = ear_mid_x
        m._ear_line_raw   = m.head_rotation_deg
        # head_lateral_px is delta from address — filled by analyzer

    # ── HAND LANDMARK METRICS ─────────────────────────────────────────────────
    # High individual visibility threshold required for ALL three finger landmarks.
    # At 30fps or below the finger tips are unreliable — the palm triangle
    # degenerates and produces saturated ±45° readings.  We require:
    #   (a) All three landmarks above 0.70 visibility (MediaPipe is confident)
    #   (b) The finger landmarks are spatially separated enough from the wrist
    #       (palm_span > 20px) — rules out collapsed / stacked readings
    # If these conditions aren't met the fields stay None rather than returning
    # a saturated junk value.
    hand_vis_threshold = 0.70   # raised from 0.40 — fingers need high conf

    lw_vis  = vis(lead_wrist)
    li_vis  = vis(lead_index)
    lp_vis  = vis(lead_pinky)
    lt_vis  = vis(lead_thumb)

    if lw_vis > hand_vis_threshold and li_vis > hand_vis_threshold and lp_vis > hand_vis_threshold:
        lw_pt = lw                     # already computed above
        li_pt = pt(lead_index)
        lp_pt = pt(lead_pinky)

        # Palm plane normal (2D approximation).
        # Vectors from wrist to index and wrist to pinky define the palm plane.
        # The signed angle between them (via cross product z-component) tells us
        # cup vs bow relative to the forearm axis.
        wi = li_pt - lw_pt   # wrist → index
        wp = lp_pt - lw_pt   # wrist → pinky

        # In face-on view, for a right-handed golfer:
        #   index is above pinky at address (both fingers extend toward camera)
        #   cupped wrist = index drops toward pinky (palm faces down more)
        #   bowed wrist  = index rises away from pinky (palm faces target more)
        # We measure the angle of the palm normal from the lead arm direction.
        lead_arm_vec = lw_pt - le_pt
        lead_arm_len = np.linalg.norm(lead_arm_vec)

        palm_span = np.linalg.norm(wi) + np.linalg.norm(wp) + 1e-9

        # Reject if finger landmarks are collapsed onto the wrist —
        # this is the main cause of saturated ±45° readings.
        if lead_arm_len > 5 and palm_span > 20.0:
            # Palm "up" vector = cross product direction (z-component in 2D)
            # Positive cross-product z = index is above forearm line = bowed
            # Negative = index below forearm line = cupped
            cross_z = wi[0] * wp[1] - wi[1] * wp[0]

            raw_cup = -cross_z / palm_span * 180.0
            # Only write the value if it's not saturated (within ±40°)
            # A saturated reading means the geometry is still degenerate
            if abs(raw_cup) < 40.0:
                m.lead_wrist_cup_deg = float(raw_cup)

        # Shaft angle proxy: only when wrist→index vector is meaningful length
        wi_norm = np.linalg.norm(wi)
        if wi_norm > 20.0:
            m.shaft_angle_proxy_deg = float(
                np.degrees(np.arctan2(wi[0], -wi[1]))
            )

    return m