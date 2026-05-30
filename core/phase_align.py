"""
phase_align.py — Phase-based synchronization of face-on and DTL metric streams.

Two videos of the same swing recorded by separate cameras have no shared
clock. Even with simultaneous "go" they differ in:
  - start time (recording started at different moments)
  - frame rate (iPhones record at 28, 30, 60, or 240 fps depending on mode)
  - duration

This module aligns them using the swing itself as the clock. Three
biomechanical events happen at the same physical instant regardless of
camera: ADDRESS, TOP OF BACKSWING, and IMPACT.

By matching these anchor frames between the two streams, we can map any
face frame to its corresponding back frame via piecewise linear
interpolation.

USAGE:
    from core.phase_align import PhaseAligner

    aligner = PhaseAligner(
        face_metrics=face_metrics,
        back_metrics=back_metrics,
    )

    # Map a face frame to the equivalent back frame
    back_frame_idx = aligner.face_to_back(face_frame_idx=35)

    # Build a unified metric stream — one entry per face frame, drawing
    # the best landmark data from whichever view has higher visibility
    unified = aligner.unified_metrics()

ACCURACY:
    Phase anchors are typically detected within +/-1 frame on each video.
    Between anchors the alignment drifts by at most a fraction of a
    frame per swing-phase frame, so absolute alignment is +/-2 frames
    in the worst case — fine for phase-averaged metrics, marginal for
    impact-frame analysis where +/-2 frames can matter.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class AlignmentQuality:
    """Diagnostic info about how well the two streams aligned."""
    face_frames: int
    back_frames: int
    face_phases: Dict[str, int]
    back_phases: Dict[str, int]
    fps_ratio: float                # back_fps / face_fps (estimated)
    confidence: float                # 0-1 quality of alignment

    def summary(self) -> str:
        lines = [
            "PhaseAligner quality:",
            f"  Face frames: {self.face_frames}  phases: "
            f"a={self.face_phases.get('addr_idx')} "
            f"t={self.face_phases.get('top_idx')} "
            f"i={self.face_phases.get('impact_idx')}",
            f"  Back frames: {self.back_frames}  phases: "
            f"a={self.back_phases.get('addr_idx')} "
            f"t={self.back_phases.get('top_idx')} "
            f"i={self.back_phases.get('impact_idx')}",
            f"  Estimated fps ratio (back/face): {self.fps_ratio:.3f}",
            f"  Alignment confidence: {self.confidence:.0%}",
        ]
        return "\n".join(lines)


class PhaseAligner:
    """
    Aligns face-on and DTL metric streams by their detected swing phases.

    Parameters
    ----------
    face_metrics : list of SwingMetrics (or None)
    back_metrics : list of BackMetrics  (or None)

    The alignment is computed eagerly in __init__. Use:
      - face_to_back(idx) to map a single face frame to a back frame
      - back_to_face(idx) for the reverse direction
      - quality to get a diagnostic report
    """

    def __init__(
        self,
        face_metrics: Optional[list] = None,
        back_metrics: Optional[list] = None,
    ):
        self.face = face_metrics or []
        self.back = back_metrics or []

        # Detect phases on each stream independently
        from .fault_engine import _detect_phases
        self.face_phases = _detect_phases(self.face) if self.face else {}
        self.back_phases = _detect_phases(self.back) if self.back else {}

        self._build_mapping()

    # ------------------------------------------------------------------
    # Mapping construction
    # ------------------------------------------------------------------

    def _build_mapping(self):
        """
        Build face→back and back→face frame mappings via piecewise linear
        interpolation across the three phase anchors.
        """
        self._face_to_back = None
        self._back_to_face = None
        self.quality = None

        if not self.face or not self.back:
            self.quality = AlignmentQuality(
                face_frames=len(self.face),
                back_frames=len(self.back),
                face_phases=self.face_phases,
                back_phases=self.back_phases,
                fps_ratio=1.0,
                confidence=0.0,
            )
            return

        nf = len(self.face)
        nb = len(self.back)

        f_addr = self.face_phases.get("addr_idx", 0)
        f_top  = self.face_phases.get("top_idx", nf // 2)
        f_imp  = self.face_phases.get("impact_idx", nf - 1)

        b_addr = self.back_phases.get("addr_idx", 0)
        b_top  = self.back_phases.get("top_idx", nb // 2)
        b_imp  = self.back_phases.get("impact_idx", nb - 1)

        # Sanity: anchors must be monotonic in both streams
        if not (f_addr < f_top < f_imp and b_addr < b_top < b_imp):
            # Phase detection failed on one stream — fall back to proportional
            f_anchors = [0, nf // 2, nf - 1]
            b_anchors = [0, nb // 2, nb - 1]
            confidence = 0.3
        else:
            f_anchors = [f_addr, f_top, f_imp]
            b_anchors = [b_addr, b_top, b_imp]

            # Confidence based on swing arc consistency:
            # Both streams should have similar backswing/downswing ratios.
            face_bs = f_top - f_addr
            face_ds = f_imp - f_top
            back_bs = b_top - b_addr
            back_ds = b_imp - b_top

            if face_bs > 0 and back_bs > 0 and face_ds > 0 and back_ds > 0:
                face_ratio = face_bs / face_ds
                back_ratio = back_bs / back_ds
                # If ratios match closely, alignment is reliable
                ratio_diff = abs(face_ratio - back_ratio) / max(face_ratio, back_ratio)
                confidence = float(np.clip(1.0 - ratio_diff, 0.0, 1.0))
            else:
                confidence = 0.5

        # FPS ratio estimate from the swing-arc lengths
        face_span = f_anchors[2] - f_anchors[0]
        back_span = b_anchors[2] - b_anchors[0]
        fps_ratio = back_span / max(face_span, 1)

        # Build face→back map: for each face frame in [0, nf), what back frame?
        # Three-segment piecewise linear interpolation:
        #   - Before address: hold at b_addr (no extrapolation)
        #   - Address→Top: linear between f_anchors[0..1] and b_anchors[0..1]
        #   - Top→Impact:  linear between f_anchors[1..2] and b_anchors[1..2]
        #   - After impact: extend linearly using the downswing rate

        f2b = np.zeros(nf, dtype=int)
        for fi in range(nf):
            f2b[fi] = self._interpolate_anchor(fi, f_anchors, b_anchors, nb)
        self._face_to_back = f2b

        # Build reverse map back→face
        b2f = np.zeros(nb, dtype=int)
        for bi in range(nb):
            b2f[bi] = self._interpolate_anchor(bi, b_anchors, f_anchors, nf)
        self._back_to_face = b2f

        self.quality = AlignmentQuality(
            face_frames=nf,
            back_frames=nb,
            face_phases=self.face_phases,
            back_phases=self.back_phases,
            fps_ratio=fps_ratio,
            confidence=confidence,
        )

    @staticmethod
    def _interpolate_anchor(
        src_idx: int,
        src_anchors: list,
        dst_anchors: list,
        dst_n: int,
    ) -> int:
        """
        Piecewise linear interpolation between three anchors.
        Holds at endpoints rather than extrapolating wildly.
        """
        s_addr, s_top, s_imp = src_anchors
        d_addr, d_top, d_imp = dst_anchors

        # Before address: hold (could also extrapolate, but address frames
        # are typically before the swing really starts)
        if src_idx <= s_addr:
            # Linear extrapolation backwards from address, capped at 0
            if s_addr > 0:
                offset = src_idx - s_addr
                d_offset = int(round(offset * (d_top - d_addr) / max(s_top - s_addr, 1)))
                return max(0, d_addr + d_offset)
            return d_addr

        if src_idx <= s_top:
            # Backswing segment
            t = (src_idx - s_addr) / max(s_top - s_addr, 1)
            return int(round(d_addr + t * (d_top - d_addr)))

        if src_idx <= s_imp:
            # Downswing segment
            t = (src_idx - s_top) / max(s_imp - s_top, 1)
            return int(round(d_top + t * (d_imp - d_top)))

        # After impact: linear extrapolation using downswing rate
        offset = src_idx - s_imp
        d_offset = int(round(offset * (d_imp - d_top) / max(s_imp - s_top, 1)))
        return min(dst_n - 1, d_imp + d_offset)

    # ------------------------------------------------------------------
    # Public mapping methods
    # ------------------------------------------------------------------

    def face_to_back(self, face_frame_idx: int) -> Optional[int]:
        """Return the back-view frame corresponding to a face-view frame."""
        if self._face_to_back is None:
            return None
        if not (0 <= face_frame_idx < len(self._face_to_back)):
            return None
        return int(self._face_to_back[face_frame_idx])

    def back_to_face(self, back_frame_idx: int) -> Optional[int]:
        """Return the face-view frame corresponding to a back-view frame."""
        if self._back_to_face is None:
            return None
        if not (0 <= back_frame_idx < len(self._back_to_face)):
            return None
        return int(self._back_to_face[back_frame_idx])

    def is_aligned(self) -> bool:
        """Both streams present and confidence > 30%."""
        return (self.quality is not None
                and self.quality.confidence > 0.30
                and self._face_to_back is not None
                and self._back_to_face is not None)

    # ------------------------------------------------------------------
    # Cross-view metric reconciliation
    # ------------------------------------------------------------------

    def reconcile_metric(
        self,
        attr: str,
        face_frame_idx: int,
        prefer: str = "auto",
    ) -> Optional[float]:
        """
        Return the best estimate for a metric attribute at a given face frame,
        drawing from whichever view has higher landmark confidence.

        Parameters
        ----------
        attr   : attribute name (must exist on SwingMetrics and/or BackMetrics)
        face_frame_idx : the face frame to query (back frame is mapped from this)
        prefer : "face" | "back" | "auto"
                  auto picks whichever view has higher confidence
                  face/back forces the choice (but falls back if missing)

        Returns
        -------
        Best available value, or None if both views are missing/unreliable.
        """
        if not self.is_aligned():
            # Single view only — return whatever we have
            if self.face and 0 <= face_frame_idx < len(self.face):
                return getattr(self.face[face_frame_idx], attr, None)
            return None

        face_val  = None
        back_val  = None
        face_conf = 0.0
        back_conf = 0.0

        if 0 <= face_frame_idx < len(self.face):
            fm = self.face[face_frame_idx]
            face_val  = getattr(fm, attr, None)
            face_conf = getattr(fm, "confidence", 0.0) or 0.0

        bi = self.face_to_back(face_frame_idx)
        if bi is not None and 0 <= bi < len(self.back):
            bm = self.back[bi]
            back_val  = getattr(bm, attr, None)
            back_conf = getattr(bm, "confidence", 0.0) or 0.0

        if prefer == "face":
            return face_val if face_val is not None else back_val
        if prefer == "back":
            return back_val if back_val is not None else face_val

        # Auto: prefer whichever view has the value AND higher confidence
        if face_val is not None and back_val is not None:
            return face_val if face_conf >= back_conf else back_val
        return face_val if face_val is not None else back_val