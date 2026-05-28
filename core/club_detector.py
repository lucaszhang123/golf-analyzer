#doesnt work right now

"""
Club shaft detection using a hybrid approach:

1. Sample a TINY patch (30x30px) right at the grip point
2. In that patch, run Canny + Hough — the shaft MUST pass through here
   so whatever line exists in this tiny area IS the shaft
3. Extend that direction across the full frame
4. Fallback: use wrist-to-estimated-ball-position geometry

Key insight: in a 30x30px box centered on the grip, the shaft is the
ONLY significant line. There is no background confusion at that scale.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque


@dataclass
class ClubDetection:
    grip_point: Optional[Tuple[float, float]] = None
    shaft_angle_deg: Optional[float] = None
    line_p1: Optional[Tuple[int, int]] = None
    line_p2: Optional[Tuple[int, int]] = None
    confidence: float = 0.0

    @property
    def detected(self) -> bool:
        return self.confidence > 0.1 and self.line_p1 is not None


class ClubDetector:

    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self._history: List[ClubDetection] = []
        self._frame_buffer = deque(maxlen=3)
        self._last_angle: Optional[float] = None

    def detect(
        self,
        frame: np.ndarray,
        left_wrist: Optional[Tuple[float, float]],
        right_wrist: Optional[Tuple[float, float]],
        handedness: str = "right",
    ) -> ClubDetection:

        h, w = frame.shape[:2]
        self._frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if left_wrist is None or right_wrist is None:
            self._update_history(ClubDetection())
            return self._smooth()

        grip_x = (left_wrist[0] + right_wrist[0]) / 2
        grip_y = (left_wrist[1] + right_wrist[1]) / 2
        grip = (float(grip_x), float(grip_y))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # === METHOD 1: Tiny patch at grip ===
        angle_from_patch = self._detect_from_patch(gray, grip_x, grip_y, patch_size=40)

        # === METHOD 2: Slightly larger patch with motion if available ===
        angle_from_motion = None
        if len(self._frame_buffer) == 3:
            angle_from_motion = self._detect_from_motion(grip_x, grip_y, w, h)

        # === METHOD 3: Geometric fallback using lead arm direction ===
        # The shaft roughly continues the line of the lead forearm
        # Lead arm: for RH golfer = left elbow → left wrist direction
        angle_from_geometry = self._estimate_from_arm(
            left_wrist, right_wrist, h, handedness
        )

        # === Combine: prefer patch detection, use geometry as fallback ===
        final_angle, confidence = self._combine_angles(
            angle_from_patch,
            angle_from_motion,
            angle_from_geometry,
            self._last_angle,
        )

        if final_angle is None:
            self._update_history(ClubDetection(grip_point=grip, confidence=0.05))
            return self._smooth()

        rad = np.radians(final_angle)
        dx = float(np.sin(rad))
        dy = float(np.cos(rad))

        p1 = self._clip_to_frame(grip_x, grip_y,  dx,  dy, w, h)
        p2 = self._clip_to_frame(grip_x, grip_y, -dx, -dy, w, h)

        self._last_angle = final_angle

        result = ClubDetection(
            grip_point=grip,
            shaft_angle_deg=final_angle,
            line_p1=p1,
            line_p2=p2,
            confidence=confidence,
        )
        self._update_history(result)
        return self._smooth()

    def _detect_from_patch(self, gray, gx, gy, patch_size=40) -> Optional[float]:
        """
        Extract a tiny patch right at the grip and find the dominant line.
        At this scale, the shaft is the only significant edge.
        """
        h, w = gray.shape
        half = patch_size // 2
        px1 = max(0, int(gx) - half)
        px2 = min(w, int(gx) + half)
        py1 = max(0, int(gy) - half)
        py2 = min(h, int(gy) + half)

        patch = gray[py1:py2, px1:px2]
        if patch.size == 0:
            return None

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        patch = clahe.apply(patch)

        # Strong edge detection
        edges = cv2.Canny(patch, 15, 50)

        # Hough on tiny patch — very low thresholds
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=4,
            minLineLength=max(5, patch_size // 5),
            maxLineGap=8,
        )

        if lines is None:
            return None

        # In this tiny patch, pick the longest non-horizontal line
        best_len = 0
        best_angle = None

        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = np.sqrt(dx**2 + dy**2)

            # Must not be near-horizontal (within 20°)
            angle_h = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
            if angle_h < 20:
                continue

            if length > best_len:
                best_len = length
                best_angle = float(np.degrees(np.arctan2(dx, dy)))

        return best_angle

    def _detect_from_motion(self, gx, gy, w, h) -> Optional[float]:
        """Frame differencing in a moderate region around grip."""
        f0, f1, f2 = self._frame_buffer[0], self._frame_buffer[1], self._frame_buffer[2]

        diff1 = cv2.absdiff(f2, f1)
        diff2 = cv2.absdiff(f1, f0)
        _, m1 = cv2.threshold(diff1, 12, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(diff2, 12, 255, cv2.THRESH_BINARY)
        motion = cv2.bitwise_and(m1, m2)

        pad = int(min(w, h) * 0.15)
        rx1 = max(0, int(gx) - pad)
        rx2 = min(w, int(gx) + pad)
        ry1 = max(0, int(gy) - pad)
        ry2 = min(h, int(gy) + pad)

        region = motion[ry1:ry2, rx1:rx2]
        if region.sum() < 500:  # not enough motion
            return None

        local_gx = gx - rx1
        local_gy = gy - ry1
        rh, rw = region.shape

        lines = cv2.HoughLinesP(
            region, rho=1, theta=np.pi/180, threshold=8,
            minLineLength=max(10, int(min(rh, rw)*0.08)),
            maxLineGap=15,
        )
        if lines is None:
            return None

        grip_thresh = min(rh, rw) * 0.25
        best_score = -1
        best_angle = None

        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx, dy = float(x2-x1), float(y2-y1)
            length = np.sqrt(dx**2 + dy**2)
            angle_h = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
            if angle_h < 15 or length < 8:
                continue

            a, b = dy, -dx
            c = dx*y1 - dy*x1
            dist = abs(a*local_gx + b*local_gy + c) / (np.sqrt(a**2+b**2)+1e-9)
            if dist > grip_thresh:
                continue

            if length > best_score:
                best_score = length
                best_angle = float(np.degrees(np.arctan2(dx, dy)))

        return best_angle

    def _estimate_from_arm(self, lw, rw, h, handedness) -> Optional[float]:
        """
        Geometric fallback: shaft roughly follows the lead arm direction,
        continuing past the wrist downward toward the ball.

        For a RH golfer at address: shaft angle ~45-55° from vertical
        tilted toward the trail side. We use the wrist height relative
        to frame height to estimate a plausible angle.
        """
        if lw is None or rw is None:
            return None

        # The two wrists define the grip orientation
        # Shaft continues in the direction from trail wrist → lead wrist → extended
        # For RH: trail=right wrist, lead=left wrist
        if handedness == "right":
            trail_w = rw
            lead_w = lw
        else:
            trail_w = lw
            lead_w = rw

        # Direction from trail → lead wrist
        dx = float(lead_w[0] - trail_w[0])
        dy = float(lead_w[1] - trail_w[1])

        # Shaft continues past lead wrist in same direction
        # (from grip end toward club head)
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1:
            # Wrists at same point — use typical address angle
            return 35.0  # ~35° from vertical, typical iron address

        angle = float(np.degrees(np.arctan2(dx, dy)))
        return angle

    def _combine_angles(self, patch, motion, geometry, last):
        """
        Weighted combination of angle estimates.
        Patch is most trusted (direct detection at grip).
        Motion is second (moving pixels = shaft).
        Geometry is fallback (always available but least accurate).
        """
        candidates = []

        if patch is not None:
            # Patch detection: high confidence, weight 3
            candidates.append((patch, 3.0, 0.7))
        if motion is not None:
            # Motion detection: medium confidence, weight 2
            candidates.append((motion, 2.0, 0.5))
        if last is not None and patch is None and motion is None:
            # Temporal: use last known angle when nothing else works
            candidates.append((last, 1.5, 0.4))
        if geometry is not None and not candidates:
            # Geometry fallback: lowest confidence
            candidates.append((geometry, 1.0, 0.2))

        if not candidates:
            return None, 0.0

        # Weighted average of angles (handle wrap-around)
        total_weight = sum(w for _, w, _ in candidates)
        avg_angle = sum(a * w for a, w, _ in candidates) / total_weight
        avg_conf = max(c for _, _, c in candidates)

        # Boost confidence if patch + motion agree
        if patch is not None and motion is not None:
            diff = abs(patch - motion)
            diff = min(diff, 180 - diff)
            if diff < 15:
                avg_conf = min(1.0, avg_conf + 0.2)

        return avg_angle, avg_conf

    def _clip_to_frame(self, gx, gy, dx, dy, w, h):
        ts = []
        if abs(dx) > 1e-9:
            ts += [(0 - gx) / dx, (w - 1 - gx) / dx]
        if abs(dy) > 1e-9:
            ts += [(0 - gy) / dy, (h - 1 - gy) / dy]
        pos = [t for t in ts if t > 1e-3]
        t = min(pos) if pos else max(w, h)
        return (int(np.clip(gx + dx * t, 0, w - 1)),
                int(np.clip(gy + dy * t, 0, h - 1)))

    def _update_history(self, result):
        self._history.append(result)
        if len(self._history) > self.smoothing_window:
            self._history.pop(0)

    def _smooth(self):
        if not self._history:
            return ClubDetection()
        valid = [d for d in self._history if d.detected]
        if not valid:
            return self._history[-1]

        angles = [d.shaft_angle_deg for d in valid if d.shaft_angle_deg is not None]
        gxs = [d.grip_point[0] for d in valid if d.grip_point]
        gys = [d.grip_point[1] for d in valid if d.grip_point]
        conf = float(np.mean([d.confidence for d in valid]))

        avg_angle = float(np.mean(angles)) if angles else 0.0
        avg_gx = float(np.mean(gxs))
        avg_gy = float(np.mean(gys))

        rad = np.radians(avg_angle)
        dx, dy = float(np.sin(rad)), float(np.cos(rad))

        last = valid[-1]
        fw = max(abs((last.line_p1 or (0,0))[0] - (last.line_p2 or (640,0))[0]) + 200, 640)
        fh = max(abs((last.line_p1 or (0,0))[1] - (last.line_p2 or (0,960))[1]) + 200, 960)

        p1 = self._clip_to_frame(avg_gx, avg_gy,  dx,  dy, fw, fh)
        p2 = self._clip_to_frame(avg_gx, avg_gy, -dx, -dy, fw, fh)

        return ClubDetection(
            grip_point=(avg_gx, avg_gy),
            shaft_angle_deg=avg_angle,
            line_p1=p1, line_p2=p2,
            confidence=conf,
        )

    def close(self):
        pass


def draw_club(
    frame: np.ndarray,
    detection: ClubDetection,
    color: Tuple[int, int, int] = (0, 220, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = frame.copy()

    if detection.grip_point:
        grip = (int(detection.grip_point[0]), int(detection.grip_point[1]))
        cv2.circle(out, grip, 7, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, grip, 5, color, -1, cv2.LINE_AA)

    if not detection.detected or detection.line_p1 is None:
        return out

    p1, p2 = detection.line_p1, detection.line_p2
    alpha = float(np.clip(detection.confidence, 0.5, 1.0))
    overlay = out.copy()
    cv2.line(overlay, p1, p2, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    return out