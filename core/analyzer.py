"""
SwingAnalyzer: main orchestrator for the golf swing analysis pipeline.

By default processes every frame in the video. Use num_snapshots to limit
to a smaller evenly-spaced sample for longer videos.
"""

import cv2
import numpy as np
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .metrics import SwingMetrics, extract_metrics
from .club_detector import ClubDetector, draw_club
from .stick_figure import StickFigureRenderer
from .annotator import VideoAnnotator
from .landmarks import LM, get_point

MODEL_PATH = Path.home() / ".golf_analyzer" / "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


def ensure_model():
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model (~7MB) to {MODEL_PATH} ...")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
        print("  Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download pose model: {e}\n"
            f"Please manually download from:\n  {MODEL_URL}\n"
            f"and save it to:\n  {MODEL_PATH}"
        )
    return str(MODEL_PATH)


@dataclass
class SwingSnapshot:
    """Metrics + image for one frame."""
    frame_idx: int
    timestamp_sec: float
    pct: int                    # 0-100, percentage through the swing
    metrics: SwingMetrics
    image_path: Optional[str] = None


@dataclass
class AnalysisResult:
    video_path: str
    fps: float
    total_frames: int
    handedness: str
    snapshots: List[SwingSnapshot] = field(default_factory=list)
    annotated_video_path: Optional[str] = None

    def summary(self) -> str:
        lines = [
            "Golf Swing Analysis",
            "===================",
            f"Video:      {self.video_path}",
            f"Handedness: {self.handedness}",
            f"Frames:     {self.total_frames}  ({self.fps:.1f} fps)",
            f"Duration:   {self.total_frames / self.fps:.2f}s",
            f"Snapshots:  {len(self.snapshots)}",
            "",
            f"{'%':>4}  {'Frame':>5}  {'Time':>5}  {'Shld°':>6}  {'Hip°':>6}  {'X-Fac':>6}  {'Spine':>6}  {'Elbow':>6}  {'Wrist':>6}  {'Wt':>5}  {'Conf':>5}",
            "-" * 90,
        ]
        for s in self.snapshots:
            m = s.metrics
            def f(v): return f"{v:6.1f}" if v is not None else "   N/A"
            lines.append(
                f"{s.pct:3}%  {s.frame_idx:5}  {s.timestamp_sec:4.2f}s"
                f"  {f(m.shoulder_rotation_deg)}"
                f"  {f(m.hip_rotation_deg)}"
                f"  {f(m.x_factor_deg)}"
                f"  {f(m.spine_tilt_deg)}"
                f"  {f(m.lead_elbow_angle_deg)}"
                f"  {f(m.wrist_hinge_deg)}"
                f"  {f(m.weight_shift)}"
                f"  {s.metrics.confidence:4.0%}"
            )
        return "\n".join(lines)

    def key_metrics_dict(self) -> list:
        """Return all snapshots as a list of dicts (ready to pass to an LLM)."""
        return [
            {
                "pct_through_swing": s.pct,
                "frame": s.frame_idx,
                "time_sec": round(s.timestamp_sec, 2),
                "shoulder_rotation_deg": s.metrics.shoulder_rotation_deg,
                "hip_rotation_deg": s.metrics.hip_rotation_deg,
                "x_factor_deg": s.metrics.x_factor_deg,
                "spine_tilt_deg": s.metrics.spine_tilt_deg,
                "lead_elbow_angle_deg": s.metrics.lead_elbow_angle_deg,
                "wrist_hinge_deg": s.metrics.wrist_hinge_deg,
                "weight_shift": s.metrics.weight_shift,
                "lead_knee_angle_deg": s.metrics.lead_knee_angle_deg,
                "head_movement_px": s.metrics.head_movement_px,
                "hip_sway_px": s.metrics.hip_sway_px,
            }
            for s in self.snapshots
        ]


class _PoseWrapper:
    def __init__(self, model_path: str):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._frame_ts_ms = 0

    def process(self, rgb_frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        self._frame_ts_ms += 33
        return _ResultWrapper(result)

    def close(self):
        self._landmarker.close()


class _ResultWrapper:
    def __init__(self, tasks_result):
        if tasks_result.pose_landmarks:
            self.pose_landmarks = _LandmarkListWrapper(tasks_result.pose_landmarks[0])
        else:
            self.pose_landmarks = None


class _LandmarkListWrapper:
    def __init__(self, landmark_list):
        self.landmark = landmark_list


class SwingAnalyzer:
    """
    End-to-end golf swing analyzer.

    Parameters
    ----------
    handedness    : "right" or "left"
    num_snapshots : how many frames to save images + include in summary table.
                    None (default) = every frame. Use e.g. 10 for long videos.
    """

    def __init__(
        self,
        handedness: str = "right",
        num_snapshots: Optional[int] = None,
    ):
        self.handedness = handedness
        self.num_snapshots = num_snapshots
        self.annotator = VideoAnnotator()
        self.club_detector = ClubDetector(smoothing_window=5)
        model_path = ensure_model()
        self._pose = _PoseWrapper(model_path)

    def analyze(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_snapshot_images: bool = True,
        show_progress: bool = True,
        out_dir: str = "output/face",
    ) -> AnalysisResult:

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        image_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        image_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = AnalysisResult(
            video_path=video_path,
            fps=fps,
            total_frames=total_frames,
            handedness=self.handedness,
        )

        if show_progress:
            print(f"Processing {total_frames} frames at {fps:.1f} fps...")

        # Decide which frames to snapshot
        n = self.num_snapshots
        if n is None or n >= total_frames:
            # Use every frame
            snapshot_indices = set(range(total_frames))
        else:
            snapshot_indices = set(
                int(round(i * (total_frames - 1) / (n - 1)))
                for i in range(n)
            )

        # --- Pass 1: extract pose + metrics for every frame ---
        all_landmarks = []
        raw_frames = []
        all_metrics = []
        self._club_detections = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self._pose.process(rgb)
            all_landmarks.append(pose_result)
            raw_frames.append(frame)

            # --- Club detection ---
            club_det = None
            if pose_result.pose_landmarks:
                lms = pose_result.pose_landmarks.landmark
                from .landmarks import LM, get_point
                lw = tuple(get_point(lms, LM.LEFT_WRIST, image_w, image_h))
                rw = tuple(get_point(lms, LM.RIGHT_WRIST, image_w, image_h))
                club_det = self.club_detector.detect(frame, lw, rw, self.handedness)
            else:
                from .club_detector import ClubDetection
                club_det = ClubDetection()
            # store for video writing
            if not hasattr(self, '_club_detections'):
                self._club_detections = []
            self._club_detections.append(club_det)

            if pose_result.pose_landmarks:
                m = extract_metrics(
                    pose_result.pose_landmarks.landmark,
                    image_w, image_h,
                    handedness=self.handedness,
                )
            else:
                m = SwingMetrics(confidence=0.0)

            all_metrics.append(m)
            frame_idx += 1

            if show_progress and frame_idx % 30 == 0:
                print(f"  {100 * frame_idx // total_frames}% ({frame_idx}/{total_frames})")

        cap.release()

        # --- Normalize rotation relative to address (first good frame) ---
        self._normalize_rotation(all_metrics, all_landmarks, image_w, image_h)

        # --- Build snapshots ---
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)

        for i in sorted(snapshot_indices):
            if i >= len(all_metrics):
                continue
            pct = int(round(100 * i / max(total_frames - 1, 1)))
            snap = SwingSnapshot(
                frame_idx=i,
                timestamp_sec=round(i / fps, 2),
                pct=pct,
                metrics=all_metrics[i],
            )

            if save_snapshot_images:
                img = raw_frames[i]
                lm = all_landmarks[i]
                annotated = self.annotator.draw_skeleton(img, lm)
                annotated = self.annotator.draw_spine_line(annotated, lm)
                # Draw club if detected
                if hasattr(self, '_club_detections') and i < len(self._club_detections):
                    annotated = draw_club(annotated, self._club_detections[i])
                annotated = self.annotator.draw_metrics_overlay(
                    annotated, all_metrics[i],
                    position_name=f"{pct}% through swing",
                    frame_idx=i,
                )
                annotated = self.annotator.draw_keyframe_banner(annotated, f"{pct}%")
                img_path = str(snap_dir / f"snapshot_{pct:03d}pct_frame{i:04d}.png")
                cv2.imwrite(img_path, annotated)
                snap.image_path = img_path

            result.snapshots.append(snap)

        if show_progress:
            print(f"  Saved {len(result.snapshots)} snapshots")

        # --- Write full annotated video ---
        if output_path:
            if show_progress:
                print(f"Writing annotated video -> {output_path}")
            self._write_annotated_video(
                output_path, raw_frames, all_landmarks,
                all_metrics, snapshot_indices, image_w, image_h, fps
            )
            result.annotated_video_path = output_path

        # Store for external access (e.g. stick figure rendering, metrics dump)
        self._last_landmarks = all_landmarks
        self._last_snapshot_indices = snapshot_indices
        self._last_metrics = all_metrics

        if show_progress:
            print("Done.\n")
            print(result.summary())

        return result

    def _normalize_rotation(self, all_metrics, all_landmarks, w, h):
        """
        Compute rotation angles using shoulder/hip WIDTH RATIO approach.

        From a face-on camera, as the golfer turns away, the apparent pixel
        width of the shoulder line shrinks. We convert this shrinkage to degrees
        using: rotation = arccos(current_width / address_width) * (180/pi)
        This gives 0° at address and approaches 90° at full turn.

        We also use MediaPipe's z-coordinate as a secondary signal and blend
        both into the final estimate for stability.
        """
        cutoff = max(1, int(len(all_metrics) * 0.4))
        base_nose = base_hip_x = None

        # Find address frame: first stable high-confidence frame in first 40%
        addr_shoulder_w = addr_hip_w = None
        addr_shoulder_z = addr_hip_z = None

        for m, lm in zip(all_metrics[:cutoff], all_landmarks[:cutoff]):
            if m.confidence > 0.65 and m._shoulder_width_px is not None:
                addr_shoulder_w = m._shoulder_width_px
                addr_hip_w = m._hip_width_px
                addr_shoulder_z = m._shoulder_z_diff
                addr_hip_z = m._hip_z_diff
                if lm and lm.pose_landmarks:
                    lms = lm.pose_landmarks.landmark
                    base_nose = get_point(lms, LM.NOSE, w, h)
                    lhip = get_point(lms, LM.LEFT_HIP, w, h)
                    rhip = get_point(lms, LM.RIGHT_HIP, w, h)
                    base_hip_x = float((lhip[0] + rhip[0]) / 2)
                break

        if addr_shoulder_w is None or addr_shoulder_w < 5:
            return

        for m, lm in zip(all_metrics, all_landmarks):
            if m._shoulder_width_px is None:
                continue

            # --- Shoulder rotation ---
            # Width ratio: 1.0 = square, 0.0 = fully turned (90°)
            s_ratio = np.clip(m._shoulder_width_px / addr_shoulder_w, 0.01, 1.0)
            # arccos converts ratio to angle: 0 ratio = 90°, 1.0 ratio = 0°
            s_width_deg = float(np.degrees(np.arccos(s_ratio)))

            # Z-depth signal: trail shoulder moves away (positive z change = turning)
            # Normalize z diff relative to address z diff
            if addr_shoulder_z is not None and m._shoulder_z_diff is not None:
                z_change = m._shoulder_z_diff - addr_shoulder_z
                # z_change of ~0.3 ≈ 90° turn empirically; scale to degrees
                s_z_deg = float(np.clip(z_change * 300.0, 0, 90))
            else:
                s_z_deg = s_width_deg

            # Blend: weight width-ratio more (more reliable from face-on)
            m.shoulder_rotation_deg = round(0.65 * s_width_deg + 0.35 * s_z_deg, 1)

            # --- Hip rotation ---
            if addr_hip_w and addr_hip_w > 5:
                h_ratio = np.clip(m._hip_width_px / addr_hip_w, 0.01, 1.0)
                h_width_deg = float(np.degrees(np.arccos(h_ratio)))

                if addr_hip_z is not None and m._hip_z_diff is not None:
                    hz_change = m._hip_z_diff - addr_hip_z
                    h_z_deg = float(np.clip(hz_change * 300.0, 0, 90))
                else:
                    h_z_deg = h_width_deg

                m.hip_rotation_deg = round(0.65 * h_width_deg + 0.35 * h_z_deg, 1)
            else:
                m.hip_rotation_deg = m.shoulder_rotation_deg * 0.47  # fallback ratio

            # --- X-factor ---
            m.x_factor_deg = round(max(0.0, m.shoulder_rotation_deg - m.hip_rotation_deg), 1)

            # --- Head movement and hip sway ---
            if lm and lm.pose_landmarks and base_nose is not None:
                lms = lm.pose_landmarks.landmark
                nose = get_point(lms, LM.NOSE, w, h)
                lhip = get_point(lms, LM.LEFT_HIP, w, h)
                rhip = get_point(lms, LM.RIGHT_HIP, w, h)
                cur_hip_x = float((lhip[0] + rhip[0]) / 2)
                m.head_movement_px = float(np.linalg.norm(nose - base_nose))
                m.hip_sway_px = float(cur_hip_x - base_hip_x)

    def _write_annotated_video(self, output_path, frames, all_landmarks,
                                all_metrics, snapshot_indices, w, h, fps):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        n = max(len(frames) - 1, 1)

        for i, (frame, lm, m) in enumerate(zip(frames, all_landmarks, all_metrics)):
            annotated = self.annotator.draw_skeleton(frame, lm)
            annotated = self.annotator.draw_spine_line(annotated, lm)
            if hasattr(self, '_club_detections') and i < len(self._club_detections):
                annotated = draw_club(annotated, self._club_detections[i])
            pct = int(round(100 * i / n))
            label = f"{pct}%" if i in snapshot_indices else None
            annotated = self.annotator.draw_metrics_overlay(annotated, m, label, i)
            if i in snapshot_indices:
                annotated = self.annotator.draw_keyframe_banner(annotated, f"{pct}% through swing")
            writer.write(annotated)
        writer.release()

    def close(self):
        self._pose.close()
        self.club_detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()