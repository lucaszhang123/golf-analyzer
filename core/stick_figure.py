"""
Stick figure renderer — draws the pose skeleton on a pure black background.

Outputs:
  - A slow-motion stick figure video (slowed to 8fps so you can see each frame)
  - Side-by-side with original, or stick figure only
  - Individual PNG frames saved to output/stick_frames/
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Set

from .landmarks import LM, get_point
from .annotator import POSE_CONNECTIONS


# Joint colors by body region (BGR)
JOINT_COLORS = {
    LM.NOSE:             (200, 200, 200),
    LM.LEFT_SHOULDER:    (255, 200, 100),
    LM.RIGHT_SHOULDER:   (255, 200, 100),
    LM.LEFT_ELBOW:       (255, 180,  80),
    LM.RIGHT_ELBOW:      (255, 180,  80),
    LM.LEFT_WRIST:       (255, 160,  60),
    LM.RIGHT_WRIST:      (255, 160,  60),
    LM.LEFT_HIP:         (100, 255, 160),
    LM.RIGHT_HIP:        (100, 255, 160),
    LM.LEFT_KNEE:        ( 80, 220, 130),
    LM.RIGHT_KNEE:       ( 80, 220, 130),
    LM.LEFT_ANKLE:       ( 60, 200, 100),
    LM.RIGHT_ANKLE:      ( 60, 200, 100),
    LM.LEFT_FOOT_INDEX:  ( 50, 180,  90),
    LM.RIGHT_FOOT_INDEX: ( 50, 180,  90),
    LM.LEFT_HEEL:        ( 50, 180,  90),
    LM.RIGHT_HEEL:       ( 50, 180,  90),
}

SEGMENT_COLORS = {
    (LM.LEFT_SHOULDER,  LM.RIGHT_SHOULDER): (255, 200, 100),
    (LM.LEFT_SHOULDER,  LM.LEFT_ELBOW):     (255, 180,  80),
    (LM.LEFT_ELBOW,     LM.LEFT_WRIST):     (255, 160,  60),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW):    (255, 180,  80),
    (LM.RIGHT_ELBOW,    LM.RIGHT_WRIST):    (255, 160,  60),
    (LM.LEFT_SHOULDER,  LM.LEFT_HIP):       (200, 200, 200),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP):      (200, 200, 200),
    (LM.LEFT_HIP,       LM.RIGHT_HIP):      (100, 255, 160),
    (LM.LEFT_HIP,       LM.LEFT_KNEE):      ( 80, 220, 130),
    (LM.LEFT_KNEE,      LM.LEFT_ANKLE):     ( 60, 200, 100),
    (LM.RIGHT_HIP,      LM.RIGHT_KNEE):     ( 80, 220, 130),
    (LM.RIGHT_KNEE,     LM.RIGHT_ANKLE):    ( 60, 200, 100),
    (LM.LEFT_ANKLE,     LM.LEFT_FOOT_INDEX):(  50, 180,  90),
    (LM.RIGHT_ANKLE,    LM.RIGHT_FOOT_INDEX):( 50, 180,  90),
    (LM.LEFT_WRIST,     LM.RIGHT_WRIST):    ( 50, 220, 255),
}

SPINE_COLOR  = (80, 150, 255)


def draw_stick_figure(
    canvas: np.ndarray,
    landmarks,
    image_w: int,
    image_h: int,
    joint_radius: int = 6,
    line_thickness: int = 3,
) -> np.ndarray:
    if landmarks is None or landmarks.pose_landmarks is None:
        return canvas

    lms = landmarks.pose_landmarks.landmark

    def pt(idx: LM) -> Optional[Tuple[int, int]]:
        lm = lms[int(idx)]
        if lm.visibility < 0.35:
            return None
        return (int(lm.x * image_w), int(lm.y * image_h))

    # Connections
    for (a, b) in POSE_CONNECTIONS:
        pa, pb = pt(a), pt(b)
        if pa is None or pb is None:
            continue
        color = SEGMENT_COLORS.get((a, b), SEGMENT_COLORS.get((b, a), (150, 150, 150)))
        cv2.line(canvas, pa, pb, color, line_thickness, cv2.LINE_AA)

    # Spine
    ls, rs = pt(LM.LEFT_SHOULDER), pt(LM.RIGHT_SHOULDER)
    lh, rh = pt(LM.LEFT_HIP),      pt(LM.RIGHT_HIP)
    if ls and rs and lh and rh:
        mid_s = ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2)
        mid_h = ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2)
        cv2.line(canvas, mid_h, mid_s, SPINE_COLOR, line_thickness, cv2.LINE_AA)

    # Joints (on top)
    for lm_idx in LM:
        p = pt(lm_idx)
        if p is None:
            continue
        color = JOINT_COLORS.get(lm_idx, (180, 180, 180))
        cv2.circle(canvas, p, joint_radius,     (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p, joint_radius - 1, color,     -1, cv2.LINE_AA)

    return canvas


def draw_metrics_minimal(canvas, metrics, frame_idx, pct, total):
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.40
    color  = (140, 140, 140)
    thick  = 1
    y = 18

    def f(v, u="°"):
        return f"{v:.1f}{u}" if v is not None else "--"

    lines = [
        f"Frame {frame_idx}/{total}  ({pct}%)",
        f"Shoulder : {f(metrics.shoulder_rotation_deg)}",
        f"Hip      : {f(metrics.hip_rotation_deg)}",
        f"X-Factor : {f(metrics.x_factor_deg)}",
        f"Spine    : {f(metrics.spine_tilt_deg)}",
        f"Weight   : {f(metrics.weight_shift, '')}",
        f"Conf     : {metrics.confidence:.0%}",
    ]
    for line in lines:
        cv2.putText(canvas, line, (8, y), font, fscale, color, thick, cv2.LINE_AA)
        y += 15

    return canvas


def draw_frame_label(canvas, pct, frame_idx, w, h):
    """Large frame counter at top center."""
    label = f"{pct}%  frame {frame_idx}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    tw, _ = cv2.getTextSize(label, font, 0.6, 1)[0], None
    cv2.putText(canvas, label, (w//2 - 60, h - 12),
                font, 0.55, (80, 80, 80), 1, cv2.LINE_AA)


class StickFigureRenderer:

    def render(
        self,
        video_path: str,
        all_landmarks: list,
        all_metrics: list,
        output_path: str,
        snapshot_indices: Set[int],
        side_by_side: bool = True,
        save_frames: bool = True,
        frames_dir: Optional[str] = None,
        playback_fps: float = 8.0,   # slow enough to see each frame clearly
    ) -> str:

        cap = cv2.VideoCapture(video_path)
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w = orig_w * 2 if side_by_side else orig_w
        out_h = orig_h

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Try H.264 first (plays on Mac/iPhone natively), fall back to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(output_path, fourcc, playback_fps, (out_w, out_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, playback_fps, (out_w, out_h))

        if save_frames and frames_dir:
            Path(frames_dir).mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            lm = all_landmarks[frame_idx] if frame_idx < len(all_landmarks) else None
            m  = all_metrics[frame_idx]   if frame_idx < len(all_metrics)   else None
            pct = int(round(100 * frame_idx / max(total - 1, 1)))

            # --- Stick figure canvas ---
            stick = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            draw_stick_figure(stick, lm, orig_w, orig_h)
            if m:
                draw_metrics_minimal(stick, m, frame_idx, pct, total)
            draw_frame_label(stick, pct, frame_idx, orig_w, orig_h)

            if side_by_side:
                # Draw colored skeleton on original
                orig_ann = frame.copy()
                if lm and lm.pose_landmarks:
                    lms = lm.pose_landmarks.landmark
                    for (a, b) in POSE_CONNECTIONS:
                        if lms[int(a)].visibility > 0.4 and lms[int(b)].visibility > 0.4:
                            pa = get_point(lms, a, orig_w, orig_h)
                            pb = get_point(lms, b, orig_w, orig_h)
                            color = SEGMENT_COLORS.get(
                                (a, b), SEGMENT_COLORS.get((b, a), (100, 220, 100)))
                            cv2.line(orig_ann,
                                     (int(pa[0]), int(pa[1])),
                                     (int(pb[0]), int(pb[1])),
                                     color, 2, cv2.LINE_AA)

                # Label sides
                cv2.putText(orig_ann, "Original", (8, orig_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1, cv2.LINE_AA)
                cv2.putText(stick, "Stick figure", (orig_w - 100, orig_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1, cv2.LINE_AA)

                out_frame = np.hstack([orig_ann, stick])
            else:
                out_frame = stick

            writer.write(out_frame)

            # Save PNG for this frame
            if save_frames and frames_dir and frame_idx in snapshot_indices:
                png = str(Path(frames_dir) / f"stick_{pct:03d}pct_f{frame_idx:04d}.png")
                cv2.imwrite(png, stick)

            frame_idx += 1

        cap.release()
        writer.release()
        return output_path