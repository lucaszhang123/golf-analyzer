"""
landmark_reconciler.py — Per-frame landmark source selection across views.

Given aligned face and back metric streams, decide for each frame and each
of the 33 MediaPipe landmarks which view should be the authoritative
source based on per-landmark visibility scores.

This module DOES NOT merge 2D coordinates from the two views into a
single skeleton — face-on and DTL are perpendicular angles, so their
pixel coordinates describe the same joint from different sides and
cannot be averaged meaningfully without true 3D reconstruction (which
requires calibrated cameras we don't have).

What it DOES produce, per face frame:
  - For each of the 33 landmarks, the chosen view (face or back)
  - The (x, y) position of that landmark in the chosen view's coords
  - Both visibility scores for diagnostic transparency
  - A coverage summary (how many landmarks came from each view)

Body-relative grounding:
  Each chosen position is also expressed as an offset from the
  hip-midpoint of that view's frame. This puts both views in a
  body-relative coordinate system so downstream tools can use
  whichever frame of reference they want.

USAGE:
    from core.landmark_reconciler import LandmarkReconciler
    from core.phase_align import PhaseAligner

    aligner = PhaseAligner(face_metrics=fm, back_metrics=bm)
    reconciler = LandmarkReconciler(
        face_landmarks=face_lm_list,   # list of MediaPipe result objects
        back_landmarks=back_lm_list,
        aligner=aligner,
        face_w=fw, face_h=fh,
        back_w=bw, back_h=bh,
    )

    # Get per-frame reconciliation as a list of ReconciledFrame
    frames = reconciler.reconcile_all()

    # Or save as JSON
    reconciler.save_json("output/lucas/reconciled_landmarks.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

from .landmarks import LM


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ReconciledLandmark:
    """One landmark's reconciled value for one frame."""
    landmark_idx: int
    landmark_name: str
    chosen_view: str          # "face" | "back" | "none"
    chosen_x_px: Optional[float]
    chosen_y_px: Optional[float]
    chosen_x_body: Optional[float]   # x relative to hip midpoint (chosen view)
    chosen_y_body: Optional[float]
    face_vis: float
    back_vis: float
    face_x_px: Optional[float]
    face_y_px: Optional[float]
    back_x_px: Optional[float]
    back_y_px: Optional[float]


@dataclass
class ReconciledFrame:
    """All 33 landmarks reconciled for one face frame."""
    face_frame_idx: int
    back_frame_idx: Optional[int]
    landmarks: List[ReconciledLandmark] = field(default_factory=list)

    # Per-frame coverage stats
    face_count: int = 0
    back_count: int = 0
    none_count: int = 0

    # Hip midpoint positions used as body-relative origins
    face_hip_mid_x_px: Optional[float] = None
    face_hip_mid_y_px: Optional[float] = None
    back_hip_mid_x_px: Optional[float] = None
    back_hip_mid_y_px: Optional[float] = None


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

# Landmarks where face view is preferred (frontal joints — chest, lead side)
# when visibility scores are close. Back view is preferred for trail-side
# joints that are typically occluded face-on at the top of backswing.
_FACE_PREFERRED = {
    LM.NOSE, LM.LEFT_EYE if hasattr(LM, 'LEFT_EYE') else LM.NOSE,
    LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST,
    LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE,
}
_BACK_PREFERRED = {
    LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
    LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE,
}

# Visibility tie-break margin — if both views are within this margin,
# defer to the preferred-view sets above.
_TIE_BREAK_MARGIN = 0.10


class LandmarkReconciler:
    """
    Per-frame landmark source selection from face + back MediaPipe results.

    Parameters
    ----------
    face_landmarks : list of MediaPipe pose result objects (one per face frame)
    back_landmarks : list of MediaPipe pose result objects (one per back frame)
    aligner        : a PhaseAligner instance (or None — single view = pass-through)
    face_w, face_h : face video dimensions in pixels
    back_w, back_h : back video dimensions in pixels
    min_visibility : per-landmark visibility floor below which we report "none"
    """

    def __init__(
        self,
        face_landmarks: list,
        back_landmarks: list,
        aligner=None,
        face_w: int = 1920,
        face_h: int = 1080,
        back_w: int = 1920,
        back_h: int = 1080,
        min_visibility: float = 0.30,
    ):
        self.face_lms = face_landmarks or []
        self.back_lms = back_landmarks or []
        self.aligner  = aligner
        self.fw, self.fh = face_w, face_h
        self.bw, self.bh = back_w, back_h
        self.min_visibility = min_visibility

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_xy_vis(self, result, lm_idx: int, w: int, h: int):
        """Return (x_px, y_px, visibility) or (None, None, 0)."""
        if result is None or result.pose_landmarks is None:
            return None, None, 0.0
        lm = result.pose_landmarks.landmark[lm_idx]
        return float(lm.x * w), float(lm.y * h), float(lm.visibility)

    def _hip_mid(self, result, w: int, h: int):
        """Return hip midpoint (x, y) in pixels, or (None, None)."""
        if result is None or result.pose_landmarks is None:
            return None, None
        lms = result.pose_landmarks.landmark
        lh, rh = lms[int(LM.LEFT_HIP)], lms[int(LM.RIGHT_HIP)]
        if lh.visibility < 0.30 or rh.visibility < 0.30:
            return None, None
        mx = (lh.x + rh.x) * 0.5 * w
        my = (lh.y + rh.y) * 0.5 * h
        return mx, my

    def _choose_view(self, lm_idx: int, face_vis: float, back_vis: float) -> str:
        """
        Pick which view to use for this landmark.
        Returns 'face', 'back', or 'none'.
        """
        face_ok = face_vis >= self.min_visibility
        back_ok = back_vis >= self.min_visibility

        if not face_ok and not back_ok:
            return "none"
        if face_ok and not back_ok:
            return "face"
        if back_ok and not face_ok:
            return "back"

        # Both visible — compare scores
        diff = face_vis - back_vis
        if abs(diff) <= _TIE_BREAK_MARGIN:
            # Close to a tie — defer to the preferred-view sets
            try:
                lm_enum = LM(lm_idx)
            except ValueError:
                lm_enum = None
            if lm_enum in _BACK_PREFERRED:
                return "back"
            if lm_enum in _FACE_PREFERRED:
                return "face"
            # No preference — go with higher visibility (any difference wins)
            return "face" if diff >= 0 else "back"

        return "face" if diff > 0 else "back"

    # ------------------------------------------------------------------
    # Per-frame reconciliation
    # ------------------------------------------------------------------

    def reconcile_frame(self, face_frame_idx: int) -> ReconciledFrame:
        """Reconcile all 33 landmarks for a single face frame."""
        face_res = (self.face_lms[face_frame_idx]
                    if 0 <= face_frame_idx < len(self.face_lms) else None)

        back_frame_idx = None
        back_res = None
        if self.aligner is not None and self.aligner.is_aligned():
            back_frame_idx = self.aligner.face_to_back(face_frame_idx)
            if back_frame_idx is not None and 0 <= back_frame_idx < len(self.back_lms):
                back_res = self.back_lms[back_frame_idx]

        face_hip_x, face_hip_y = self._hip_mid(face_res, self.fw, self.fh)
        back_hip_x, back_hip_y = self._hip_mid(back_res, self.bw, self.bh)

        frame = ReconciledFrame(
            face_frame_idx=face_frame_idx,
            back_frame_idx=back_frame_idx,
            face_hip_mid_x_px=face_hip_x,
            face_hip_mid_y_px=face_hip_y,
            back_hip_mid_x_px=back_hip_x,
            back_hip_mid_y_px=back_hip_y,
        )

        for lm_idx in range(33):
            fx, fy, fv = self._get_xy_vis(face_res, lm_idx, self.fw, self.fh)
            bx, by, bv = self._get_xy_vis(back_res, lm_idx, self.bw, self.bh)
            chosen = self._choose_view(lm_idx, fv, bv)

            if chosen == "face":
                cx, cy = fx, fy
                hip_x, hip_y = face_hip_x, face_hip_y
                frame.face_count += 1
            elif chosen == "back":
                cx, cy = bx, by
                hip_x, hip_y = back_hip_x, back_hip_y
                frame.back_count += 1
            else:
                cx, cy = None, None
                hip_x, hip_y = None, None
                frame.none_count += 1

            # Body-relative coords: offset from chosen view's hip midpoint
            cx_body = (cx - hip_x) if (cx is not None and hip_x is not None) else None
            cy_body = (cy - hip_y) if (cy is not None and hip_y is not None) else None

            try:
                name = LM(lm_idx).name
            except ValueError:
                name = f"LM_{lm_idx}"

            frame.landmarks.append(ReconciledLandmark(
                landmark_idx=lm_idx,
                landmark_name=name,
                chosen_view=chosen,
                chosen_x_px=cx,
                chosen_y_px=cy,
                chosen_x_body=cx_body,
                chosen_y_body=cy_body,
                face_vis=fv,
                back_vis=bv,
                face_x_px=fx,
                face_y_px=fy,
                back_x_px=bx,
                back_y_px=by,
            ))

        return frame

    def reconcile_all(self) -> List[ReconciledFrame]:
        """Reconcile every face frame."""
        return [self.reconcile_frame(i) for i in range(len(self.face_lms))]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_json(self, out_path: str, verbose: bool = True) -> dict:
        """
        Reconcile every frame and write a JSON file with the per-frame,
        per-landmark choices.
        """
        all_frames = self.reconcile_all()

        # Aggregate stats
        total_face = sum(f.face_count for f in all_frames)
        total_back = sum(f.back_count for f in all_frames)
        total_none = sum(f.none_count for f in all_frames)
        total_lm   = total_face + total_back + total_none

        # Per-landmark source distribution
        per_lm_counts: Dict[str, Dict[str, int]] = {}
        for f in all_frames:
            for lm in f.landmarks:
                d = per_lm_counts.setdefault(lm.landmark_name, {"face": 0, "back": 0, "none": 0})
                d[lm.chosen_view] += 1

        payload = {
            "metadata": {
                "n_face_frames":   len(self.face_lms),
                "n_back_frames":   len(self.back_lms),
                "aligner_present": self.aligner is not None,
                "aligner_active":  (self.aligner is not None
                                    and self.aligner.is_aligned()),
                "alignment_confidence": (self.aligner.quality.confidence
                                         if self.aligner is not None
                                         and self.aligner.quality is not None
                                         else None),
                "face_dimensions": {"w": self.fw, "h": self.fh},
                "back_dimensions": {"w": self.bw, "h": self.bh},
                "min_visibility":  self.min_visibility,
            },
            "totals": {
                "total_landmark_picks": total_lm,
                "from_face":           total_face,
                "from_back":           total_back,
                "no_view":             total_none,
                "face_pct":            round(100 * total_face / max(total_lm, 1), 1),
                "back_pct":            round(100 * total_back / max(total_lm, 1), 1),
                "none_pct":            round(100 * total_none / max(total_lm, 1), 1),
            },
            "per_landmark_counts": per_lm_counts,
            "frames": [
                {
                    "face_frame_idx": f.face_frame_idx,
                    "back_frame_idx": f.back_frame_idx,
                    "face_count":     f.face_count,
                    "back_count":     f.back_count,
                    "none_count":     f.none_count,
                    "face_hip_mid_px": [f.face_hip_mid_x_px, f.face_hip_mid_y_px],
                    "back_hip_mid_px": [f.back_hip_mid_x_px, f.back_hip_mid_y_px],
                    "landmarks": [asdict(lm) for lm in f.landmarks],
                }
                for f in all_frames
            ],
        }

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        if verbose:
            print(f"\nLandmark reconciliation written to: {out_path}")
            print(f"  Frames reconciled: {len(all_frames)}")
            print(f"  Source distribution:")
            print(f"    Face view : {total_face} picks ({payload['totals']['face_pct']}%)")
            print(f"    Back view : {total_back} picks ({payload['totals']['back_pct']}%)")
            print(f"    No view   : {total_none} picks ({payload['totals']['none_pct']}%)")

        return payload
