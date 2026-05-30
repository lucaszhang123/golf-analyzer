"""
reconciler_visualizer.py — Render a side-by-side video showing which view
contributed each landmark, frame by frame.

The reconciliation choices live in reconciled_landmarks.json. Each face
frame has a corresponding back frame (via PhaseAligner) and per-landmark
source decisions. This module overlays those decisions on top of the
original videos.

VISUAL ENCODING:
  Each joint dot is colored by source:
    GREEN  — this view was chosen as the source for this landmark
    GRAY   — the OTHER view was chosen (this view's reading is being discarded)
    RED    — neither view was confident enough (no_view)

  Skeleton lines:
    GREEN if both endpoints are from this view
    YELLOW if endpoints are split (one from each view) — the cross-view fix in action
    GRAY if both came from the other view
    RED if either endpoint has no source

USAGE:
    from core.reconciler_visualizer import render_reconciliation

    render_reconciliation(
        face_video_path="output/lucas/face/swing_analyzed.mp4",
        back_video_path="output/lucas/back/swing_back.mp4",
        reconciled_json="output/lucas/reconciled_landmarks.json",
        output_path="output/lucas/reconciled.mp4",
    )
"""

from __future__ import annotations

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# Joint connections to draw — same as MediaPipe pose graph subset
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    (11, 12),                       # shoulders
    (11, 13), (13, 15),             # left arm
    (12, 14), (14, 16),             # right arm
    (11, 23), (12, 24),             # torso sides
    (23, 24),                       # hips
    (23, 25), (25, 27),             # left leg
    (24, 26), (26, 28),             # right leg
    (27, 31), (28, 32),             # feet
    (15, 16),                       # wrist line (club proxy)
]

# Colors (BGR)
COLOR_CHOSEN     = (80, 220, 100)    # green — this view is the source
COLOR_DISCARDED  = (120, 120, 120)   # gray — other view chosen
COLOR_NONE       = (60, 60, 220)     # red — neither view confident
COLOR_SPLIT      = (80, 220, 220)    # yellow — endpoints from different views
COLOR_TEXT       = (240, 240, 240)
COLOR_BG         = (15, 15, 15)


def _safe_get_landmark(frame_data: dict, lm_idx: int) -> Optional[dict]:
    """Find a landmark by index in a frame's landmarks list."""
    for lm in frame_data.get("landmarks", []):
        if lm["landmark_idx"] == lm_idx:
            return lm
    return None


def _draw_panel(
    canvas: np.ndarray,
    bg_frame: Optional[np.ndarray],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    landmarks: List[dict],
    view: str,
    label: str,
    frame_idx: int,
    show_bg: bool = True,
) -> None:
    """
    Draw one view's panel onto the canvas at the given offset.

    bg_frame is the original video frame at this moment (or None for blank).
    landmarks list comes from reconciled JSON — each has chosen_view plus
    both views' pixel positions.
    """
    # Place background
    if show_bg and bg_frame is not None:
        # Resize bg_frame to panel size, dim it so overlay is readable
        resized = cv2.resize(bg_frame, (panel_w, panel_h))
        dimmed  = cv2.addWeighted(resized, 0.35, np.zeros_like(resized), 0.65, 0)
        canvas[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = dimmed
    else:
        canvas[panel_y:panel_y+panel_h, panel_x:panel_x+panel_w] = COLOR_BG

    # Coordinate scaling — the JSON stored pixel coords in original video dims
    # We need scale factors to map onto our panel
    # We use the first landmark with a non-None face/back position to infer original size
    if not landmarks:
        return

    # Get the view's original dimensions from the first landmark with data
    orig_w = orig_h = None
    pos_key_x = f"{view}_x_px"
    pos_key_y = f"{view}_y_px"

    # We need original w/h — but it's stored at file level. For now infer from
    # bg_frame size if available, else fall back to scanning landmarks for max.
    if bg_frame is not None:
        orig_h, orig_w = bg_frame.shape[:2]
    else:
        # Fall back: find max coord across all landmarks
        xs = [lm.get(pos_key_x) for lm in landmarks if lm.get(pos_key_x) is not None]
        ys = [lm.get(pos_key_y) for lm in landmarks if lm.get(pos_key_y) is not None]
        if xs and ys:
            orig_w, orig_h = max(max(xs), 1), max(max(ys), 1)
        else:
            return

    sx = panel_w / orig_w
    sy = panel_h / orig_h

    # Build a lookup of landmark_idx → (px, py, color, chosen)
    point_info: Dict[int, dict] = {}
    for lm in landmarks:
        x = lm.get(pos_key_x)
        y = lm.get(pos_key_y)
        if x is None or y is None:
            continue
        px = int(x * sx) + panel_x
        py = int(y * sy) + panel_y
        chosen = lm.get("chosen_view", "none")

        if chosen == view:
            color = COLOR_CHOSEN
        elif chosen == "none":
            color = COLOR_NONE
        else:
            color = COLOR_DISCARDED

        point_info[lm["landmark_idx"]] = {
            "px": px, "py": py, "color": color, "chosen": chosen,
            "vis": lm.get(f"{view}_vis", 0.0),
        }

    # Draw connections — color by which views the endpoints came from
    for a, b in POSE_CONNECTIONS:
        pa, pb = point_info.get(a), point_info.get(b)
        if pa is None or pb is None:
            continue
        ca, cb = pa["chosen"], pb["chosen"]
        if ca == "none" or cb == "none":
            line_color = COLOR_NONE
        elif ca == cb:
            line_color = COLOR_CHOSEN if ca == view else COLOR_DISCARDED
        else:
            line_color = COLOR_SPLIT
        cv2.line(canvas, (pa["px"], pa["py"]), (pb["px"], pb["py"]),
                 line_color, 2, cv2.LINE_AA)

    # Draw joint dots
    for info in point_info.values():
        cv2.circle(canvas, (info["px"], info["py"]), 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (info["px"], info["py"]), 4, info["color"], -1, cv2.LINE_AA)

    # Panel label
    cv2.putText(canvas, f"{label} view (f{frame_idx})",
                (panel_x + 10, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)


def _draw_legend(canvas: np.ndarray, x: int, y: int) -> None:
    """Color legend in the bottom strip."""
    items = [
        (COLOR_CHOSEN,    "Sourced from this view"),
        (COLOR_DISCARDED, "Other view was used"),
        (COLOR_SPLIT,     "Split connection"),
        (COLOR_NONE,      "Neither view confident"),
    ]
    for i, (color, label) in enumerate(items):
        cx = x + i * 260
        cv2.circle(canvas, (cx, y), 7, color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (cx + 14, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)


def _draw_stats_strip(canvas: np.ndarray, x: int, y: int, w: int,
                      face_count: int, back_count: int, none_count: int,
                      alignment_conf: Optional[float]) -> None:
    """Per-frame contribution counts."""
    total = face_count + back_count + none_count
    if total == 0:
        return
    face_pct = 100 * face_count / total
    back_pct = 100 * back_count / total

    cv2.rectangle(canvas, (x, y), (x + w, y + 22), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * face_pct / 100), y + 22),
                  COLOR_CHOSEN, -1)
    cv2.rectangle(canvas, (x + int(w * face_pct / 100), y),
                  (x + int(w * (face_pct + back_pct) / 100), y + 22),
                  (200, 140, 60), -1)
    # Remaining is none — leave dark

    text = f"Face: {face_count}/{total} ({face_pct:.0f}%)   Back: {back_count}/{total} ({back_pct:.0f}%)"
    if alignment_conf is not None:
        text += f"   Align: {alignment_conf:.0%}"
    cv2.putText(canvas, text, (x + 8, y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_reconciliation(
    face_video_path: str,
    back_video_path: str,
    reconciled_json: str,
    output_path: str,
    panel_w: int = 720,
    panel_h: int = 1280,
    playback_fps: float = 12.0,
    show_video_bg: bool = True,
    verbose: bool = True,
) -> str:
    """
    Render a side-by-side visualization of cross-view reconciliation.

    Parameters
    ----------
    face_video_path : path to the original face video
    back_video_path : path to the original back video
    reconciled_json : path to reconciled_landmarks.json from main.py
    output_path     : where to write the resulting .mp4
    panel_w         : width of each side panel in pixels
    panel_h         : height of each side panel in pixels
    playback_fps    : output playback speed (slow for legibility)
    show_video_bg   : overlay original video underneath the skeleton
    """
    with open(reconciled_json) as f:
        recon = json.load(f)

    meta   = recon.get("metadata", {})
    align_conf = meta.get("alignment_confidence")
    frames = recon["frames"]

    if not frames:
        raise ValueError("No frames in reconciled JSON")

    cap_face = cv2.VideoCapture(face_video_path) if face_video_path else None
    cap_back = cv2.VideoCapture(back_video_path) if back_video_path else None

    # Pre-read all frames into memory keyed by frame index (videos are short)
    def read_all(cap):
        if cap is None or not cap.isOpened():
            return {}
        out = {}
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            out[i] = fr
            i += 1
        cap.release()
        return out

    if verbose:
        print(f"  Reading face video frames...")
    face_frames = read_all(cap_face)
    if verbose:
        print(f"  Reading back video frames...")
    back_frames = read_all(cap_back)
    if verbose:
        print(f"    face frames loaded: {len(face_frames)}")
        print(f"    back frames loaded: {len(back_frames)}")

    # Canvas layout
    margin = 20
    legend_h = 50
    stats_h  = 32
    canvas_w = panel_w * 2 + margin * 3
    canvas_h = panel_h + margin * 2 + legend_h + stats_h

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, playback_fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 playback_fps, (canvas_w, canvas_h))

    if verbose:
        print(f"  Writing {len(frames)} frames at {playback_fps:.0f} fps...")

    for fi, frame_data in enumerate(frames):
        canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

        face_idx = frame_data["face_frame_idx"]
        back_idx = frame_data["back_frame_idx"]
        landmarks = frame_data.get("landmarks", [])

        face_bg = face_frames.get(face_idx)
        back_bg = back_frames.get(back_idx) if back_idx is not None else None

        # Left panel: face view
        _draw_panel(canvas, face_bg,
                    margin, margin, panel_w, panel_h,
                    landmarks, view="face",
                    label="FACE", frame_idx=face_idx,
                    show_bg=show_video_bg)

        # Right panel: back view
        _draw_panel(canvas, back_bg,
                    margin * 2 + panel_w, margin, panel_w, panel_h,
                    landmarks, view="back",
                    label="BACK",
                    frame_idx=back_idx if back_idx is not None else -1,
                    show_bg=show_video_bg)

        # Stats strip
        _draw_stats_strip(
            canvas,
            margin, margin + panel_h + 8,
            canvas_w - 2 * margin,
            frame_data.get("face_count", 0),
            frame_data.get("back_count", 0),
            frame_data.get("none_count", 0),
            align_conf,
        )

        # Legend
        _draw_legend(canvas, margin, margin + panel_h + 8 + stats_h + 22)

        writer.write(canvas)

        if verbose and (fi + 1) % 10 == 0:
            print(f"    rendered {fi + 1}/{len(frames)} frames")

    writer.release()
    if verbose:
        print(f"\n  Visualization saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="Visualize cross-view landmark reconciliation as a side-by-side video."
    )
    p.add_argument("session", help="Path to session folder (e.g. output/lucas)")
    p.add_argument("--out", default=None,
                   help="Output mp4 path (default: <session>/reconciled.mp4)")
    p.add_argument("--no-bg", action="store_true",
                   help="Hide original video, show skeleton on black")
    p.add_argument("--fps", type=float, default=12.0,
                   help="Playback speed (default 12 — slow for legibility)")
    p.add_argument("--panel-w", type=int, default=720)
    p.add_argument("--panel-h", type=int, default=1280)
    args = p.parse_args()

    session = Path(args.session)
    recon_json = session / "reconciled_landmarks.json"
    if not recon_json.exists():
        print(f"Error: {recon_json} not found. Run main.py with both --face and --back first.")
        raise SystemExit(1)

    # Locate the captured videos
    face_videos = list((session / "face").glob("*_analyzed.mp4")) if (session / "face").exists() else []
    back_videos = list((session / "back").glob("*_back.mp4"))     if (session / "back").exists() else []

    face_vid = str(face_videos[0]) if face_videos else None
    back_vid = str(back_videos[0]) if back_videos else None

    if face_vid is None and back_vid is None:
        print("Warning: no annotated videos found in session — rendering on black background.")
        args.no_bg = True

    out_path = args.out or str(session / "reconciled.mp4")

    render_reconciliation(
        face_video_path=face_vid,
        back_video_path=back_vid,
        reconciled_json=str(recon_json),
        output_path=out_path,
        panel_w=args.panel_w,
        panel_h=args.panel_h,
        playback_fps=args.fps,
        show_video_bg=not args.no_bg,
        verbose=True,
    )
