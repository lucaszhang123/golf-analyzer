"""
Golf Swing Analyzer

Usage:
    # Face view analysis (default)
    python3 main.py swing.mov --club 7i

    # Face view stick figure
    python3 main.py swing.mov --stick --no-feedback

    # Back view analysis
    python3 main.py swing_back.mov --back

    # Back view stick figure
    python3 main.py swing_back.mov --back --stick

    # With LLM feedback
    python3 main.py swing.mov --club 7i --api-key sk-ant-...

Clubs: driver, 3w, 5w, 7w, hybrid, 2i-9i, pw, gw, sw, lw
"""

import argparse
import os
import sys
from pathlib import Path

from core.fault_rules import CLUB_GROUPS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--club", default="7i",
        help=f"Club used. Options: {', '.join(sorted(CLUB_GROUPS.keys()))}")
    parser.add_argument("--snapshots", type=int, default=None)
    parser.add_argument("--no-video",    action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--api-key",     default=None)
    parser.add_argument("--max-faults",  type=int, default=3)
    parser.add_argument("--stick",       action="store_true",
        help="Side-by-side stick figure video on black background")
    parser.add_argument("--stick-only",  action="store_true",
        help="Stick figure only (no original alongside)")
    parser.add_argument("--back",        action="store_true",
        help="Analyze as a back view video (different metrics)")
    parser.add_argument("--name", default=None,
        help="Session name — groups face and back outputs into output/<name>/face/ and output/<name>/back/")
    return parser.parse_args()


def run_face(args, video_path):
    """Run face view analysis pipeline."""
    from core.analyzer import SwingAnalyzer
    from core.fault_detector import FaultDetector
    from core.feedback import FeedbackGenerator

    # All face view outputs go into output/<name>/face/ or output/face/
    base = Path("output") / args.name if args.name else Path("output")
    out_dir = base / "face"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = None if args.no_video else (
        args.output or str(out_dir / f"{video_path.stem}_analyzed.mp4")
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

    print("\nRunning fault detection...")
    all_metrics = [s.metrics for s in result.snapshots]
    report = FaultDetector().detect(all_metrics, club=args.club)
    print(report.summary())

    if not args.no_feedback:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("\nNo API key — skipping LLM feedback.")
            print("Run with --api-key sk-ant-... or export ANTHROPIC_API_KEY=...")
        else:
            print("\nGenerating coaching feedback...")
            feedback = FeedbackGenerator(api_key=api_key).generate(
                report, club=args.club, max_faults=args.max_faults
            )
            feedback.print_report()

    if args.stick or args.stick_only:
        from core.stick_figure import StickFigureRenderer
        stick_path = str(out_dir / f"{video_path.stem}_stick.mp4")
        frames_dir = str(out_dir / "stick_frames")
        print(f"\nRendering face stick figure...")
        StickFigureRenderer().render(
            video_path=str(video_path),
            all_landmarks=getattr(analyzer, '_last_landmarks', []),
            all_metrics=all_metrics,
            output_path=stick_path,
            snapshot_indices=set(range(len(result.snapshots))),
            side_by_side=not args.stick_only,
            save_frames=True,
            frames_dir=frames_dir,
        )
        print(f"Stick video : {stick_path}")
        print(f"Stick frames: {frames_dir}/")

    print(f"\nAll face view outputs in: {out_dir}/")
    if result.annotated_video_path:
        print(f"  Annotated video : {result.annotated_video_path}")

    return result, report


def run_back(args, video_path):
    """Run back view analysis pipeline."""
    from core.analyzer_back import BackAnalyzer

    # All back view outputs go into output/<name>/back/ or output/back/
    base = Path("output") / args.name if args.name else Path("output")
    out_dir = base / "back"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = None if args.no_video else (
        args.output or str(out_dir / f"{video_path.stem}_back.mp4")
    )
    stick_path = str(out_dir / f"{video_path.stem}_back_stick.mp4") if (args.stick or args.stick_only) else None
    frames_dir = str(out_dir / "stick_frames") if stick_path else None

    with BackAnalyzer(handedness=args.hand, num_snapshots=args.snapshots) as analyzer:
        result = analyzer.analyze(
            str(video_path),
            output_path=output_path,
            stick_path=stick_path,
            stick_frames_dir=frames_dir,
            save_snapshot_images=True,
            show_progress=True,
            out_dir=str(out_dir),
        )

    if result.report:
        print(result.report.summary())

    print(f"\nAll back view outputs in: {out_dir}/")
    if result.annotated_video_path:
        print(f"  Annotated video : {result.annotated_video_path}")
    if result.stick_video_path:
        print(f"  Stick video     : {result.stick_video_path}")
    if frames_dir:
        print(f"  Stick frames    : {frames_dir}/")

    return result


def main():
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: {video_path} not found", file=sys.stderr)
        sys.exit(1)

    view = "Back view" if args.back else "Face view"
    print(f"Golf Swing Analyzer")
    print(f"Input : {video_path}  |  Club: {args.club}  |  Hand: {args.hand}  |  View: {view}")
    print()

    if args.back:
        return run_back(args, video_path)
    else:
        return run_face(args, video_path)


if __name__ == "__main__":
    main()