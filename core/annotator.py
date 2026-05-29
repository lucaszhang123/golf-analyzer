"""
Draw pose skeleton and metric overlays onto video frames.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Tuple

from .landmarks import LM, get_point
from .metrics import SwingMetrics


# MediaPipe pose connections (pairs of landmark indices to draw as lines)
POSE_CONNECTIONS = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
    (LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
    (LM.LEFT_WRIST, LM.RIGHT_WRIST),  # club line approximation
]

# Colors (BGR)
COLOR_SKELETON = (100, 220, 100)
COLOR_JOINT = (255, 255, 255)
COLOR_HIGHLIGHT = (0, 200, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (0, 0, 0)
COLOR_KEY_FRAME = (50, 200, 255)
COLOR_WARNING = (50, 100, 255)


class VideoAnnotator:

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks,
        alpha: float = 0.85,
    ) -> np.ndarray:
        """Draw pose skeleton on a copy of the frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        if landmarks is None:
            return out

        lms = landmarks.pose_landmarks
        if lms is None:
            return out

        pts = {}
        for lm_idx in LM:
            p = get_point(lms.landmark, lm_idx, w, h)
            pts[lm_idx] = (int(p[0]), int(p[1]))
            vis = lms.landmark[int(lm_idx)].visibility
            if vis > 0.15:   # low threshold — draw even uncertain joints
                cv2.circle(out, pts[lm_idx], 4, COLOR_JOINT, -1, cv2.LINE_AA)

        for a, b in POSE_CONNECTIONS:
            if a in pts and b in pts:
                va = lms.landmark[int(a)].visibility
                vb = lms.landmark[int(b)].visibility
                if va > 0.10 and vb > 0.10:   # low — prefer a line to a gap
                    cv2.line(out, pts[a], pts[b], COLOR_SKELETON, 2, cv2.LINE_AA)

        return out

    def draw_metrics_overlay(
        self,
        frame: np.ndarray,
        metrics,
        position_name: Optional[str] = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Draw metrics panel. Handles both SwingMetrics (face-on) and DTLMetrics."""
        out = frame.copy()
        h, w = out.shape[:2]

        lines = []
        if position_name:
            lines.append((">> " + position_name.upper().replace("_", " "), COLOR_KEY_FRAME))

        def fmt(val, unit="°"):
            return f"{val:.1f}{unit}" if val is not None else "—"

        if hasattr(metrics, 'shoulder_rotation_deg'):
            # Face-on SwingMetrics
            lines += [
                (f"Shoulder turn:  {fmt(metrics.shoulder_rotation_deg)}", COLOR_TEXT),
                (f"Hip turn:       {fmt(metrics.hip_rotation_deg)}", COLOR_TEXT),
                (f"X-Factor:       {fmt(metrics.x_factor_deg)}", COLOR_TEXT),
                (f"Spine tilt:     {fmt(metrics.spine_tilt_deg)}", COLOR_TEXT),
                (f"Lead elbow:     {fmt(metrics.lead_elbow_angle_deg)}", COLOR_TEXT),
                (f"Wrist hinge:    {fmt(metrics.wrist_hinge_deg)}", COLOR_TEXT),
                (f"Weight shift:   {fmt(metrics.weight_shift, '')}", COLOR_TEXT),
                (f"Lead knee:      {fmt(metrics.lead_knee_angle_deg)}", COLOR_TEXT),
            ]
        else:
            # DTLMetrics — down-the-line
            lines += [
                (f"Spine angle:    {fmt(metrics.spine_angle_deg)}", COLOR_TEXT),
                (f"Lead arm:       {fmt(metrics.lead_arm_angle_deg)}", COLOR_TEXT),
                (f"Shld plane:     {fmt(metrics.shoulder_plane_deg)}", COLOR_TEXT),
                (f"Lead knee:      {fmt(metrics.lead_knee_flex_deg)}", COLOR_TEXT),
                (f"Trail knee:     {fmt(metrics.trail_knee_flex_deg)}", COLOR_TEXT),
                (f"Hip slide:      {fmt(getattr(metrics, 'hip_slide_px', None), 'px')}", COLOR_TEXT),
                (f"Head fwd:       {fmt(getattr(metrics, 'head_forward_px', None), 'px')}", COLOR_TEXT),
            ]

        lines += [
            (f"Confidence:     {metrics.confidence:.0%}", COLOR_TEXT),
            (f"Frame: {frame_idx}", COLOR_TEXT),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        line_h = 18
        pad = 8
        panel_w = 230
        panel_h = len(lines) * line_h + pad * 2

        overlay = out.copy()
        cv2.rectangle(overlay, (4, 4), (4 + panel_w, 4 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        for i, (text, color) in enumerate(lines):
            y = 4 + pad + i * line_h + line_h // 2
            cv2.putText(out, text, (10, y), font, font_scale, color, thickness, cv2.LINE_AA)

        return out

    def draw_spine_line(
        self,
        frame: np.ndarray,
        landmarks,
    ) -> np.ndarray:
        """Draw a line representing the spine angle."""
        out = frame.copy()
        h, w = out.shape[:2]

        if landmarks is None or landmarks.pose_landmarks is None:
            return out

        lms = landmarks.pose_landmarks.landmark

        def pt(idx):
            return get_point(lms, idx, w, h)

        from .landmarks import midpoint
        mid_s = midpoint(pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER))
        mid_h = midpoint(pt(LM.LEFT_HIP), pt(LM.RIGHT_HIP))

        p1 = (int(mid_h[0]), int(mid_h[1]))
        p2 = (int(mid_s[0]), int(mid_s[1]))
        cv2.line(out, p1, p2, COLOR_HIGHLIGHT, 2, cv2.LINE_AA)
        cv2.circle(out, p1, 5, COLOR_HIGHLIGHT, -1)

        return out

    def draw_keyframe_banner(
        self,
        frame: np.ndarray,
        label: str,
    ) -> np.ndarray:
        """Draw a colored banner at the top of the frame for a key position."""
        out = frame.copy()
        h, w = out.shape[:2]

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (30, 120, 200), -1)
        cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

        text = label.upper().replace("_", " ")
        font = cv2.FONT_HERSHEY_SIMPLEX
        tw, _ = cv2.getTextSize(text, font, 0.7, 2)[0], None
        cv2.putText(out, text, (w // 2 - 70, 22), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return out