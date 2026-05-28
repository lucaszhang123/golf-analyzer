"""
Golf Swing Analyzer — fault detection step.

Loads a session captured by `main.py` (which writes per-frame metrics to
`output/<session>/face/metrics.json` and `output/<session>/back/metrics.json`)
and runs fault detection on whichever views are present. Optionally calls
the LLM for coaching feedback.

Usage:
    # Analyze both views from a session folder
    python3 faults.py output/session1/

    # Override the club (defaults to whatever was recorded at capture time)
    python3 faults.py output/session1/ --club 7i

    # With LLM coaching feedback
    python3 faults.py output/session1/ --api-key sk-ant-...
    # or:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 faults.py output/session1/

    # Analyze just one view (e.g. if capture only ran face)
    python3 faults.py output/session1/ --only face

    # Write report JSON for each view (faults_face.json / faults_back.json)
    python3 faults.py output/session1/ --save-json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from core.metrics import SwingMetrics
from core.metrics_back import BackMetrics
from core.fault_detector import FaultDetector, FaultReport
from core.feedback import FeedbackGenerator


def parse_args():
    p = argparse.ArgumentParser(
        description="Run fault detection and (optional) AI coaching feedback "
                    "on a session captured by main.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session", help="Path to a session folder (e.g. output/session1)")
    p.add_argument("--only", choices=["face", "back", "both"], default="both",
                   help="Limit analysis to one view (default: both, if present)")
    p.add_argument("--club", default=None,
                   help="Override club. Defaults to whatever was captured in session.json.")
    p.add_argument("--no-feedback", action="store_true",
                   help="Skip LLM coaching feedback")
    p.add_argument("--api-key", default=None,
                   help="Anthropic API key (else read from $ANTHROPIC_API_KEY)")
    p.add_argument("--max-faults", type=int, default=3,
                   help="Max faults to send to the LLM for detailed feedback")
    p.add_argument("--save-json", action="store_true",
                   help="Write faults_face.json / faults_back.json into the session folder")
    return p.parse_args()


def load_session(session_path: Path):
    """Read session.json if present; otherwise infer from folder layout."""
    manifest_path = session_path / "session.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "name": session_path.name,
            "club": "7i",
            "handedness": "right",
            "face_metrics": "face/metrics.json" if (session_path / "face" / "metrics.json").exists() else None,
            "back_metrics": "back/metrics.json" if (session_path / "back" / "metrics.json").exists() else None,
        }
    return manifest


def load_metrics(metrics_file: Path, view: str):
    """Load per-frame metrics from JSON, reconstructing the dataclass objects."""
    with open(metrics_file) as f:
        payload = json.load(f)

    Cls = SwingMetrics if view == "face" else BackMetrics
    valid_fields = {fld for fld in Cls.__dataclass_fields__ if not fld.startswith("_")}

    all_metrics = []
    for d in payload["frame_metrics"]:
        kwargs = {k: v for k, v in d.items() if k in valid_fields}
        all_metrics.append(Cls(**kwargs))

    return payload, all_metrics


def analyze_view(view: str, metrics_file: Path, club: str,
                 do_feedback: bool, api_key: Optional[str], max_faults: int,
                 save_json: bool, session_dir: Path) -> Optional[FaultReport]:
    print("\n" + "=" * 60)
    print(f"{view.upper()} VIEW — fault detection")
    print("=" * 60)

    if not metrics_file.exists():
        print(f"  No metrics file at {metrics_file} — skipping.")
        return None

    payload, all_metrics = load_metrics(metrics_file, view)
    print(f"  Loaded {len(all_metrics)} frames from {metrics_file}")
    print(f"  Source video : {payload.get('video_path', 'unknown')}")
    print(f"  fps          : {payload.get('fps', 'unknown')}")

    report = FaultDetector().detect(all_metrics, club=club, view=view)
    print(report.summary())

    if save_json:
        out_path = session_dir / f"faults_{view}.json"
        with open(out_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"  Wrote {out_path}")

    if do_feedback and report.faults:
        if not api_key:
            print("\n  No API key — skipping LLM feedback.")
            print("  Pass --api-key sk-ant-... or export ANTHROPIC_API_KEY=...")
        else:
            print("\n  Generating coaching feedback...")
            feedback = FeedbackGenerator(api_key=api_key).generate(
                report, club=club, max_faults=max_faults
            )
            feedback.print_report()

    return report


def main():
    args = parse_args()

    session_dir = Path(args.session)
    if not session_dir.exists() or not session_dir.is_dir():
        print(f"Error: session folder {session_dir} not found", file=sys.stderr)
        sys.exit(1)

    manifest = load_session(session_dir)
    club = args.club or manifest.get("club") or "7i"

    print("Golf Swing Analyzer — fault detection")
    print(f"Session : {session_dir}/")
    print(f"Club    : {club}")
    print(f"Hand    : {manifest.get('handedness', 'right')}")

    do_feedback = not args.no_feedback
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "") or None

    views_to_run = []
    if args.only in ("face", "both") and manifest.get("face_metrics"):
        views_to_run.append("face")
    if args.only in ("back", "both") and manifest.get("back_metrics"):
        views_to_run.append("back")

    if not views_to_run:
        print(f"\nNothing to analyze. No metrics.json found for the requested view(s).",
              file=sys.stderr)
        sys.exit(2)

    reports = {}
    for view in views_to_run:
        rel = manifest.get(f"{view}_metrics") or f"{view}/metrics.json"
        metrics_file = session_dir / rel
        reports[view] = analyze_view(
            view=view,
            metrics_file=metrics_file,
            club=club,
            do_feedback=do_feedback,
            api_key=api_key,
            max_faults=args.max_faults,
            save_json=args.save_json,
            session_dir=session_dir,
        )

    print("\n" + "=" * 60)
    print(f"Fault analysis complete: {session_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
