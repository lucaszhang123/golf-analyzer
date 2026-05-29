"""
Golf Swing Analyzer — capture step.

Extracts pose landmarks, metrics, snapshot images, and stick figure videos
from one or both swing-view videos. NO fault detection runs here; this step
is purely about producing the artifacts that fault analysis will consume.

Run fault detection separately with `faults.py` on the session folder this
script creates.

Usage:
    # Recommended — capture both views into one session at the same time
    python3 main.py --face swing.mov --back swing_back.mov --club 7i

    # Custom session folder name (otherwise auto sessionN)
    python3 main.py --face swing.mov --back swing_back.mov --name myround

    # One view only
    python3 main.py --face swing.mov
    python3 main.py --back swing_back.mov

    # Skip the annotated video (faster — stick + metrics only)
    python3 main.py --face swing.mov --back swing_back.mov --no-video

    # Stick-only (no original side-by-side)
    python3 main.py --face swing.mov --back swing_back.mov --stick-only

Then analyze faults:
    python3 faults.py output/<session>/

Clubs: driver, 3w, 5w, 7w, hybrid, 2i-9i, pw, gw, sw, lw
"""

import argparse
import json
import sys
from pathlib import Path

from core.fault_rules import CLUB_GROUPS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture pose data, snapshots, and stick figure videos "
                    "for face and/or back view swing videos. "
                    "Fault detection is handled by faults.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--face", "-f", default=None,
        help="Path to the face-on (front) swing video")
    parser.add_argument("--back", "-b", default=None,
        help="Path to the back / down-the-line swing video")

    parser.add_argument("--name", default=None,
        help="Session folder name under output/. "
             "If omitted, the next available output/sessionN/ is used.")
    parser.add_argument("--club", default="7i",
        help=f"Club used (recorded into session.json for fault analysis). "
             f"Options: {', '.join(sorted(CLUB_GROUPS.keys()))}")
    parser.add_argument("--hand", choices=["right", "left"], default="right")

    parser.add_argument("--snapshots", type=int, default=None,
        help="Limit to N evenly-spaced snapshot frames (default: every frame)")
    parser.add_argument("--no-video", action="store_true",
        help="Skip the annotated MP4 output (faster)")
    parser.add_argument("--stick", action="store_true",
        help="Render side-by-side stick figure video (default behavior — kept "
             "for back-compat; stick is always on)")
    parser.add_argument("--stick-only", action="store_true",
        help="Stick figure on black only, no original alongside")
    parser.add_argument("--no-stick", action="store_true",
        help="Skip stick figure rendering")
    return parser.parse_args()


def resolve_inputs(args):
    """Return (face_path, back_path) as Paths or None."""
    face = Path(args.face) if args.face else None
    back = Path(args.back) if args.back else None

    if face is None and back is None:
        print("Error: no input video. Pass --face <path> and/or --back <path>.",
              file=sys.stderr)
        sys.exit(2)

    for label, p in (("face", face), ("back", back)):
        if p is not None and not p.exists():
            print(f"Error: {label} video {p} not found", file=sys.stderr)
            sys.exit(1)

    return face, back


def resolve_session_dir(name):
    """Return output/<name> if name given, else the next free output/sessionN/."""
    out_root = Path("output")
    out_root.mkdir(parents=True, exist_ok=True)
    if name:
        session = out_root / name
    else:
        i = 1
        while (out_root / f"session{i}").exists():
            i += 1
        session = out_root / f"session{i}"
    session.mkdir(parents=True, exist_ok=True)
    return session


def dump_metrics(view, video_path, fps, total_frames, handedness, club,
                 all_metrics, snapshots, out_file):
    """Serialize per-frame metrics + run metadata to JSON."""
    payload = {
        "view": view,
        "video_path": str(video_path),
        "fps": float(fps),
        "total_frames": int(total_frames),
        "handedness": handedness,
        "club": club,
        "frame_metrics": [m.to_dict() for m in all_metrics],
        "snapshots": [
            {
                "frame_idx": s.frame_idx,
                "timestamp_sec": s.timestamp_sec,
                "pct": s.pct,
                "image_path": s.image_path,
            }
            for s in snapshots
        ],
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(o):
    # numpy scalars / anything else not natively JSON-serializable
    try:
        return float(o)
    except (TypeError, ValueError):
        return str(o)


def capture_face(args, video_path, session_dir, stick_mode):
    """Run face view pose extraction. NO fault detection."""
    from core.analyzer import SwingAnalyzer

    out_dir = session_dir / "face"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = None if args.no_video else str(
        out_dir / f"{video_path.stem}_analyzed.mp4"
    )

    analyzer = SwingAnalyzer(handedness=args.hand, num_snapshots=args.snapshots)
    with analyzer:
        result = analyzer.analyze(
            str(video_path),
            output_path=output_path,
            save_snapshot_images=True,
            show_progress=True,
            out_dir=str(out_dir),
        )

        if stick_mode != "off":
            from core.stick_figure import StickFigureRenderer
            stick_path = str(out_dir / f"{video_path.stem}_stick.mp4")
            frames_dir = str(out_dir / "stick_frames")
            print("\nRendering face stick figure...")
            StickFigureRenderer().render(
                video_path=str(video_path),
                all_landmarks=getattr(analyzer, "_last_landmarks", []),
                all_metrics=getattr(analyzer, "_last_metrics", []),
                output_path=stick_path,
                snapshot_indices=set(range(len(result.snapshots))),
                side_by_side=(stick_mode != "only"),
                save_frames=True,
                frames_dir=frames_dir,
            )
            print(f"Stick video : {stick_path}")
            print(f"Stick frames: {frames_dir}/")

        all_metrics = getattr(analyzer, "_last_metrics", [s.metrics for s in result.snapshots])

    dump_metrics(
        view="face",
        video_path=video_path,
        fps=result.fps,
        total_frames=result.total_frames,
        handedness=result.handedness,
        club=args.club,
        all_metrics=all_metrics,
        snapshots=result.snapshots,
        out_file=out_dir / "metrics.json",
    )

    print(f"\nFace view outputs:")
    print(f"  Directory   : {out_dir}/")
    if result.annotated_video_path:
        print(f"  Annotated   : {result.annotated_video_path}")
    print(f"  Metrics     : {out_dir / 'metrics.json'}")

    return result


def capture_back(args, video_path, session_dir, stick_mode):
    """Run back view pose extraction. NO fault detection."""
    from core.analyzer_back import BackAnalyzer

    out_dir = session_dir / "back"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = None if args.no_video else str(
        out_dir / f"{video_path.stem}_back.mp4"
    )
    stick_path = None
    frames_dir = None
    if stick_mode != "off":
        stick_path = str(out_dir / f"{video_path.stem}_back_stick.mp4")
        frames_dir = str(out_dir / "stick_frames")

    with BackAnalyzer(handedness=args.hand, num_snapshots=args.snapshots) as analyzer:
        result = analyzer.analyze(
            str(video_path),
            output_path=output_path,
            stick_path=stick_path,
            stick_frames_dir=frames_dir,
            save_snapshot_images=True,
            show_progress=True,
            out_dir=str(out_dir),
            run_faults=False,
        )
        all_metrics = getattr(analyzer, "_last_metrics", [s.metrics for s in result.snapshots])

    dump_metrics(
        view="back",
        video_path=video_path,
        fps=result.fps,
        total_frames=result.total_frames,
        handedness=result.handedness,
        club=args.club,
        all_metrics=all_metrics,
        snapshots=result.snapshots,
        out_file=out_dir / "metrics.json",
    )

    print(f"\nBack view outputs:")
    print(f"  Directory   : {out_dir}/")
    if result.annotated_video_path:
        print(f"  Annotated   : {result.annotated_video_path}")
    if result.stick_video_path:
        print(f"  Stick       : {result.stick_video_path}")
    if frames_dir:
        print(f"  Stick frames: {frames_dir}/")
    print(f"  Metrics     : {out_dir / 'metrics.json'}")

    return result


def write_session_manifest(session_dir, args, face_path, back_path):
    """A small session.json so faults.py knows what to load."""
    manifest = {
        "name": session_dir.name,
        "club": args.club,
        "handedness": args.hand,
        "face_video": str(face_path) if face_path else None,
        "back_video": str(back_path) if back_path else None,
        "face_metrics": "face/metrics.json" if face_path else None,
        "back_metrics": "back/metrics.json" if back_path else None,
    }
    with open(session_dir / "session.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    args = parse_args()
    face_path, back_path = resolve_inputs(args)
    session_dir = resolve_session_dir(args.name)

    # Stick mode
    if args.no_stick:
        stick_mode = "off"
    elif args.stick_only:
        stick_mode = "only"
    else:
        stick_mode = "side"  # default: side-by-side

    print("Golf Swing Analyzer — capture")
    print(f"Session : {session_dir}/")
    if face_path:
        print(f"Face    : {face_path}")
    if back_path:
        print(f"Back    : {back_path}")
    print(f"Club    : {args.club}  |  Hand: {args.hand}")
    print("(No fault detection in this step — run `python3 faults.py "
          f"{session_dir}/` to analyze.)\n")

    if face_path:
        print("=" * 60)
        print("FACE VIEW — capture")
        print("=" * 60)
        capture_face(args, face_path, session_dir, stick_mode)

    if back_path:
        print("\n" + "=" * 60)
        print("BACK VIEW — capture")
        print("=" * 60)
        capture_back(args, back_path, session_dir, stick_mode)
    write_session_manifest(session_dir, args, face_path, back_path)

    print("\n" + "=" * 60)
    print(f"Capture complete. Session: {session_dir}/")
    print(f"Next step: python3 faults.py {session_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#python3 main.py --face swing.mov --back swing_back.mov --club 7i --name lucas
#python3 main.py --face rory.mov --back rory_back.mov --club 7i --name rory

#python3 faults.py output/lucas/ --save-json
#python3 faults.py output/rory/ --save-json