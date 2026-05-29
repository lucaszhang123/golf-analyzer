"""
DTL (Down-the-Line) Swing Analyzer — identical pipeline to face-on analyzer
but extracts DTL-specific metrics and produces DTL-specific outputs.

Usage:
    from core.analyzer_dtl import BackAnalyzer
    result = BackAnalyzer(handedness="right").analyze("swing_dtl.mov")
"""

import cv2
import numpy as np
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .metrics_back import BackMetrics, extract_back_metrics
from .annotator import VideoAnnotator
from .stick_figure import (
    draw_stick_figure, draw_metrics_minimal, draw_frame_label,
    POSE_CONNECTIONS, SEGMENT_COLORS
)
from .landmarks import LM, get_point
from .pose_filter import PoseFilter

MODEL_PATH = Path.home() / ".golf_analyzer" / "pose_landmarker_full.task"
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


def ensure_model():
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model (~7MB)...")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r, open(MODEL_PATH, "wb") as f:
        f.write(r.read())
    print("  Done.")
    return str(MODEL_PATH)


class _PoseWrapper:
    def __init__(self, model_path):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lm = mp_vision.PoseLandmarker.create_from_options(opts)
        self._ts = 0

    def process(self, rgb):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(img, self._ts)
        self._ts += 33
        return _Wrap(res)

    def close(self):
        self._lm.close()


class _Wrap:
    def __init__(self, r):
        self.pose_landmarks = _LMWrap(r.pose_landmarks[0]) if r.pose_landmarks else None

class _LMWrap:
    def __init__(self, lm):
        self.landmark = lm


@dataclass
class BackSnapshot:
    frame_idx: int
    timestamp_sec: float
    pct: int
    metrics: BackMetrics
    image_path: Optional[str] = None


@dataclass
class BackReport:
    """Fault report for DTL analysis."""
    faults: list = field(default_factory=list)
    address_frame: int = 0
    top_frame: int = 0
    impact_frame: int = 0

    def summary(self) -> str:
        lines = [
            "",
            "Back View Fault Detection Report",
            "==========================",
            f"Benchmark : Elite / PGA Tour standard (back view)",
            f"Phases    : address={self.address_frame}  top={self.top_frame}  impact={self.impact_frame}",
            "",
        ]
        if not self.faults:
            lines.append("No significant DTL faults detected.")
            return "\n".join(lines)

        phase_order = ["address", "backswing", "backswing (ascent)",
                       "downswing", "downswing (descent)", "impact", "follow_through"]
        grouped = {}
        for f in self.faults:
            grouped.setdefault(f.phase, []).append(f)

        lines.append(f"Faults found: {len(self.faults)}\n")
        num = 1
        for phase in phase_order:
            if phase not in grouped:
                continue
            label = phase.upper().replace("_"," ").replace("(","— ").replace(")","")
            lines.append(f"── {label} ──────────────────────")
            for f in grouped[phase]:
                val = f"{f.measured_value:.1f}" if f.measured_value is not None else "N/A"
                lines += [
                    f"{num}. [{f.severity_label.upper()}] {f.display_name}  (severity {f.severity:.2f})",
                    f"   Measured   : {val}   Elite: {f.elite_benchmark}",
                    f"   Detail     : {f.description}",
                    f"   Root cause : {f.root_cause}",
                    f"   Ball flight: {f.ball_flight}",
                    f"   Source     : {f.source}",
                    "",
                ]
                num += 1
        return "\n".join(lines)


@dataclass
class BackResult:
    video_path: str
    fps: float
    total_frames: int
    handedness: str
    snapshots: List[BackSnapshot] = field(default_factory=list)
    report: Optional[BackReport] = None
    annotated_video_path: Optional[str] = None
    stick_video_path: Optional[str] = None


class BackAnalyzer:
    """Down-the-line swing analyzer — same pipeline as face-on."""

    def __init__(self, handedness="right", num_snapshots=None):
        self.handedness    = handedness
        self.num_snapshots = num_snapshots
        self.annotator     = VideoAnnotator()
        self._pose         = _PoseWrapper(ensure_model())
        self._last_landmarks = []

    def analyze(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        stick_path: Optional[str] = None,
        stick_frames_dir: Optional[str] = None,
        save_snapshot_images: bool = True,
        show_progress: bool = True,
        out_dir: str = "output/back",
        run_faults: bool = True,
    ) -> BackResult:

        cap = cv2.VideoCapture(video_path)
        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        image_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = BackResult(
            video_path=video_path, fps=fps,
            total_frames=total, handedness=self.handedness
        )

        # Snapshot indices
        n = self.num_snapshots
        if n is None or n >= total:
            snap_idx: Set[int] = set(range(total))
        else:
            snap_idx = set(int(round(i*(total-1)/(n-1))) for i in range(n))

        if show_progress:
            print(f"Back: Processing {total} frames at {fps:.1f} fps...")

        all_landmarks = []
        raw_frames    = []
        all_metrics   = []

        # Address baseline storage
        addr_spine = addr_hip_x = addr_head_x = addr_head_y = None
        frame_idx  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pr  = self._pose.process(rgb)

            all_landmarks.append(pr)
            raw_frames.append(frame)

            if pr.pose_landmarks:
                m = extract_back_metrics(
                    pr.pose_landmarks.landmark,
                    image_w, image_h, self.handedness
                )
            else:
                m = BackMetrics(confidence=0.0)

            # Set address baseline from first good frame
            if (addr_spine is None and m.confidence > 0.65
                    and m._spine_angle_raw is not None
                    and frame_idx < total // 3):
                addr_spine  = m._spine_angle_raw
                addr_hip_x  = m._hip_x_raw
                addr_head_x = m._head_x_raw
                addr_head_y = m._head_y_raw

            # Compute delta metrics
            if addr_spine is not None and m._spine_angle_raw is not None:
                m.spine_angle_change = m._spine_angle_raw - addr_spine
            if addr_hip_x is not None and m._hip_x_raw is not None:
                m.hip_slide_px = float(m._hip_x_raw - addr_hip_x)
            if addr_head_x is not None and m._head_x_raw is not None:
                dx = m._head_x_raw - addr_head_x
                dy = (m._head_y_raw or 0) - (addr_head_y or 0)
                m.head_movement_px = float(np.sqrt(dx**2 + dy**2))

            all_metrics.append(m)

            frame_idx += 1
            if show_progress and frame_idx % 30 == 0:
                print(f"  {100*frame_idx//total}% ({frame_idx}/{total})")

        cap.release()

        # --- Kinematic constraint filtering ---
        # Clean landmark trajectories before address baseline and delta metrics
        # are computed, so all downstream numbers come from corrected positions.
        if show_progress:
            print("  Running kinematic pose filter (back view)...")
        pf = PoseFilter(image_w=image_w, image_h=image_h, fps=fps, verbose=show_progress)
        all_landmarks, filter_quality = pf.filter_with_quality(all_landmarks)

        # Re-extract all metrics from cleaned landmarks, then recompute deltas.
        addr_spine = addr_hip_x = addr_head_x = addr_head_y = None
        all_metrics = []
        for fi, lm in enumerate(all_landmarks):
            if lm and lm.pose_landmarks:
                m = extract_back_metrics(
                    lm.pose_landmarks.landmark,
                    image_w, image_h, self.handedness
                )
            else:
                m = BackMetrics(confidence=0.0)

            # Establish address baseline from first high-confidence frame
            if (addr_spine is None and m.confidence > 0.65
                    and m._spine_angle_raw is not None
                    and fi < total // 3):
                addr_spine  = m._spine_angle_raw
                addr_hip_x  = m._hip_x_raw
                addr_head_x = m._head_x_raw
                addr_head_y = m._head_y_raw

            # Compute delta metrics relative to address
            if addr_spine is not None and m._spine_angle_raw is not None:
                m.spine_angle_change = m._spine_angle_raw - addr_spine
            if addr_hip_x is not None and m._hip_x_raw is not None:
                m.hip_slide_px = float(m._hip_x_raw - addr_hip_x)
            if addr_head_x is not None and m._head_x_raw is not None:
                dx = m._head_x_raw - addr_head_x
                dy = (m._head_y_raw or 0) - (addr_head_y or 0)
                m.head_movement_px = float(np.sqrt(dx**2 + dy**2))

            all_metrics.append(m)

        self._last_landmarks = all_landmarks
        self._last_metrics   = all_metrics

        # --- Fault detection (optional) ---
        if run_faults:
            if show_progress:
                print("  Running DTL fault detection...")

            from .syndrome_engine import SyndromeEngine
            engine = SyndromeEngine(all_metrics, view="back")
            phases = engine._phases
            addr_f = phases.get("addr_idx", 0)
            top_f  = phases.get("top_idx",  0)
            imp_f  = phases.get("impact_idx", 0)
            deduped = engine.detected()

            result.report = BackReport(
                faults=deduped,
                address_frame=addr_f,
                top_frame=top_f,
                impact_frame=imp_f,
            )

            if show_progress:
                print(result.report.summary())

        # --- Snapshot images ---
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        for i in sorted(snap_idx):
            if i >= len(all_metrics):
                continue
            pct  = int(round(100*i/max(total-1,1)))
            snap = BackSnapshot(
                frame_idx=i, timestamp_sec=round(i/fps,2),
                pct=pct, metrics=all_metrics[i]
            )
            if save_snapshot_images:
                img = raw_frames[i]
                lm  = all_landmarks[i]
                ann = self.annotator.draw_skeleton(img, lm)
                ann = self.annotator.draw_spine_line(ann, lm)
                ann = self.annotator.draw_metrics_overlay(
                    ann, all_metrics[i], f"Back {pct}%", i
                )
                ip  = str(snap_dir / f"back_snapshot_{pct:03d}pct_frame{i:04d}.png")
                cv2.imwrite(ip, ann)
                snap.image_path = ip

            result.snapshots.append(snap)

        # --- Annotated video ---
        if output_path:
            self._write_video(output_path, raw_frames, all_landmarks,
                              all_metrics, snap_idx, image_w, image_h, fps)
            result.annotated_video_path = output_path

        # --- Stick figure video ---
        if stick_path:
            self._write_stick_video(
                stick_path, raw_frames, all_landmarks, all_metrics,
                snap_idx, image_w, image_h, fps,
                frames_dir=stick_frames_dir
            )
            result.stick_video_path = stick_path

        return result

    def _write_video(self, path, frames, lms, metrics, snap_idx, w, h, fps):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        wr = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not wr.isOpened():
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = max(len(frames)-1, 1)
        for i, (frame, lm, m) in enumerate(zip(frames, lms, metrics)):
            ann = self.annotator.draw_skeleton(frame, lm)
            ann = self.annotator.draw_spine_line(ann, lm)
            pct = int(round(100*i/n))
            ann = self.annotator.draw_metrics_overlay(ann, m, f"Back {pct}%" if i in snap_idx else None, i)
            wr.write(ann)
        wr.release()

    def _write_stick_video(self, path, frames, lms, metrics,
                           snap_idx, w, h, fps, frames_dir=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if frames_dir:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out_w  = w * 2
        wr = cv2.VideoWriter(path, fourcc, 8.0, (out_w, h))
        if not wr.isOpened():
            wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (out_w, h))

        total = max(len(frames)-1, 1)
        for i, (frame, lm, m) in enumerate(zip(frames, lms, metrics)):
            pct = int(round(100*i/total))

            # Annotated original
            orig = frame.copy()
            if lm and lm.pose_landmarks:
                lmks = lm.pose_landmarks.landmark
                for (a, b) in POSE_CONNECTIONS:
                    if lmks[int(a)].visibility > 0.4 and lmks[int(b)].visibility > 0.4:
                        pa = get_point(lmks, a, w, h)
                        pb = get_point(lmks, b, w, h)
                        color = SEGMENT_COLORS.get((a,b), SEGMENT_COLORS.get((b,a),(100,220,100)))
                        cv2.line(orig, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])),
                                 color, 2, cv2.LINE_AA)
            cv2.putText(orig, "Original (Back)", (8, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1, cv2.LINE_AA)

            # Stick figure
            stick = np.zeros((h, w, 3), dtype=np.uint8)
            draw_stick_figure(stick, lm, w, h)
            draw_frame_label(stick, pct, i, w, h)
            if m:
                self._draw_back_metrics(stick, m, i, pct)
            cv2.putText(stick, "Stick (Back)", (w-90, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1, cv2.LINE_AA)

            out_frame = np.hstack([orig, stick])
            wr.write(out_frame)

            if frames_dir and i in snap_idx:
                cv2.imwrite(
                    str(Path(frames_dir) / f"back_stick_{pct:03d}pct_f{i:04d}.png"),
                    stick
                )

        wr.release()

    def _draw_back_metrics(self, canvas, m: BackMetrics, frame_idx, pct):
        font  = cv2.FONT_HERSHEY_SIMPLEX
        sc    = 0.38
        color = (140, 140, 140)
        thick = 1
        y     = 18

        def f(v, u="°"):
            return f"{v:.1f}{u}" if v is not None else "--"

        lines = [
            f"Frame {frame_idx}  ({pct}%)",
            f"Spine angle : {f(m.spine_angle_deg)}",
            f"Lead arm    : {f(m.lead_arm_angle_deg)}",
            f"Lead knee   : {f(m.lead_knee_flex_deg)}",
            f"Trail knee  : {f(m.trail_knee_flex_deg)}",
            f"Hip slide   : {f(m.hip_slide_px, 'px')}",
            f"Head fwd    : {f(m.head_forward_px, 'px')}",
            f"Shld plane  : {f(m.shoulder_plane_deg)}",
            f"Conf        : {m.confidence:.0%}",
        ]
        for line in lines:
            cv2.putText(canvas, line, (8, y), font, sc, color, thick, cv2.LINE_AA)
            y += 15

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()