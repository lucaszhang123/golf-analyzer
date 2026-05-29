"""
Golf Swing Analyzer — fault detection step.

Usage:
    python3 faults.py output/session1/
    python3 faults.py output/session1/ --club 7i --api-key sk-ant-...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from core.metrics import SwingMetrics
from core.metrics_back import BackMetrics
from core.fault_engine import run_fault_detection
from core.feedback import FeedbackGenerator


def parse_args():
    p = argparse.ArgumentParser(
        description="Run fault detection on a session captured by main.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("session")
    p.add_argument("--only", choices=["face", "back", "both"], default="both")
    p.add_argument("--club", default=None)
    p.add_argument("--no-feedback", action="store_true")
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-faults", type=int, default=3)
    p.add_argument("--save-json", action="store_true")
    return p.parse_args()


def load_session(session_path: Path):
    manifest_path = session_path / "session.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {
        "name": session_path.name,
        "club": "7i",
        "handedness": "right",
        "face_metrics": "face/metrics.json" if (session_path / "face" / "metrics.json").exists() else None,
        "back_metrics": "back/metrics.json" if (session_path / "back" / "metrics.json").exists() else None,
    }


def load_metrics(metrics_file: Path, view: str):
    with open(metrics_file) as f:
        payload = json.load(f)
    Cls = SwingMetrics if view == "face" else BackMetrics
    valid = {k for k in Cls.__dataclass_fields__ if not k.startswith("_")}
    metrics = [Cls(**{k: v for k, v in d.items() if k in valid})
               for d in payload["frame_metrics"]]
    return payload, metrics


def _try_load(view: str, manifest: dict, session_dir: Path, requested: bool):
    if not requested:
        return None, None
    rel = manifest.get(f"{view}_metrics") or f"{view}/metrics.json"
    path = session_dir / rel
    if not path.exists():
        print(f"  {view.upper()}: no metrics.json at {path} — skipping.")
        return None, None
    payload, metrics = load_metrics(path, view)
    print(f"  {view.upper()}: {len(metrics)} frames  fps={payload.get('fps','?')}")
    return payload, metrics


def main():
    args = parse_args()
    session_dir = Path(args.session)
    if not session_dir.exists():
        print(f"Error: {session_dir} not found", file=sys.stderr)
        sys.exit(1)

    manifest    = load_session(session_dir)
    club        = args.club or manifest.get("club") or "7i"
    do_feedback = not args.no_feedback
    api_key     = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or None

    print("Golf Swing Analyzer — fault detection")
    print(f"Session : {session_dir}/")
    print(f"Club    : {club}")
    print(f"Hand    : {manifest.get('handedness', 'right')}")

    print("\nLoading metrics...")
    _, face_metrics = _try_load("face", manifest, session_dir, args.only in ("face", "both"))
    _, back_metrics = _try_load("back", manifest, session_dir, args.only in ("back", "both"))

    if face_metrics is None and back_metrics is None:
        print("Nothing to analyze.", file=sys.stderr)
        sys.exit(2)

    print("\n" + "=" * 60)
    print("FAULT DETECTION")
    print("=" * 60)

    report = run_fault_detection(
        face_metrics=face_metrics,
        back_metrics=back_metrics,
        verbose=True,
    )

    if args.save_json:
        out = session_dir / "faults.json"
        with open(out, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        print(f"\n  Wrote {out}")

    if do_feedback and report.faults:
        if not api_key:
            print("\n  No API key — skipping LLM feedback.")
        else:
            print("\n  Generating coaching feedback...")
            feedback = FeedbackGenerator(api_key=api_key).generate(
                report, club=club, max_faults=args.max_faults
            )
            feedback.print_report()

    print("\n" + "=" * 60)
    print(f"Complete: {session_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()