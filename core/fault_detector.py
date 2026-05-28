"""
FaultDetector: orchestrates the syndrome engine for face and back view.
Replaces the old threshold-based single-metric detection.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .syndrome_engine import SyndromeEngine, SyndromeResult


@dataclass
class FaultReport:
    club: str
    club_group: str
    view: str                          # "face" or "back"
    address_frame: int
    top_frame: int
    impact_frame: int
    faults: List[SyndromeResult] = field(default_factory=list)

    def summary(self) -> str:
        view_label = "Face View" if self.view == "face" else "Back View"
        lines = [
            "",
            f"Fault Detection Report — {view_label}",
            "=" * 45,
            f"Benchmark    : Elite / PGA Tour + TPI standards",
            f"Club         : {self.club}  ({self.club_group})",
            f"Detection    : Syndrome-based (multi-signal agreement required)",
            f"Phases       : address={self.address_frame}  top={self.top_frame}  impact={self.impact_frame}",
            "",
        ]

        if not self.faults:
            lines.append("No significant faults detected.")
            return "\n".join(lines)

        # Group by phase
        phase_order = [
            "address", "backswing", "backswing (ascent)",
            "transition", "downswing", "downswing (descent)",
            "impact", "follow_through"
        ]
        grouped = {}
        for f in self.faults:
            grouped.setdefault(f.phase, []).append(f)

        lines.append(f"Faults detected: {len(self.faults)}  "
                     f"(reliability >= 50% required)\n")

        num = 1
        for phase in phase_order:
            if phase not in grouped:
                continue
            label = phase.upper().replace("_"," ").replace("(","— ").replace(")","")
            lines.append(f"── {label} ──────────────────────────────")
            for f in grouped[phase]:
                lines += [
                    f"{num}. [{f.severity_label.upper()}] {f.display_name}",
                    f"   Reliability : {f.reliability:.0%}  "
                    f"({len(f.triggered_signals)}/{len(f.signals)} signals triggered)",
                    f"   Signals     :",
                ]
                for s in f.signals:
                    tick = "✓" if s.triggered else "✗"
                    val_str = f"{s.value:.1f}" if s.value is not None else "N/A"
                    lines.append(f"      {tick} {s.name}: {s.description} (conf: {s.confidence:.0%})")
                lines += [
                    f"   Description : {f.description}",
                    f"   Root cause  : {f.root_cause}",
                    f"   Ball flight : {f.ball_flight}",
                    f"   Source      : {f.source}",
                    "",
                ]
                num += 1

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "benchmark": "TPI + elite PGA Tour standards",
            "detection_method": "syndrome_based_multi_signal",
            "view": self.view,
            "club": self.club,
            "swing_phases": {
                "address_frame": self.address_frame,
                "top_frame":     self.top_frame,
                "impact_frame":  self.impact_frame,
            },
            "faults": [f.to_dict() for f in self.faults],
            "top_fault": self.faults[0].to_dict() if self.faults else None,
            "fault_count": len(self.faults),
        }


class FaultDetector:

    def detect(
        self,
        all_metrics: list,
        club: str = "7i",
        view: str = "face",
    ) -> FaultReport:

        if not all_metrics:
            return FaultReport(club, "iron_mid", view, 0, 0, 0, [])

        club_group = self._club_group(club)

        # Run syndrome engine
        engine = SyndromeEngine(all_metrics, view=view)
        phases = engine._phases

        print(f"  View   : {view}")
        print(f"  Club   : {club} ({club_group})")
        print(f"  Phases : address={phases.get('addr_idx',0)}  "
              f"top={phases.get('top_idx',0)}  "
              f"impact={phases.get('impact_idx',0)}")
        print(f"  Running {len(engine.run_all.__self__.__class__.__dict__)} syndrome checks...")

        detected = engine.detected()

        print(f"  Detected: {len(detected)} syndrome(s) with reliability >= 50%")

        return FaultReport(
            club=club,
            club_group=club_group,
            view=view,
            address_frame=phases.get("addr_idx", 0),
            top_frame=phases.get("top_idx", 0),
            impact_frame=phases.get("impact_idx", 0),
            faults=detected,
        )

    def _club_group(self, club: str) -> str:
        groups = {
            "driver": "driver",
            "3w": "iron_long", "5w": "iron_long", "7w": "iron_long",
            "hybrid": "iron_long", "2i": "iron_long", "3i": "iron_long",
            "4i": "iron_mid", "5i": "iron_mid", "6i": "iron_mid", "7i": "iron_mid",
            "8i": "iron_short", "9i": "iron_short",
            "pw": "iron_short", "gw": "iron_short", "sw": "iron_short", "lw": "iron_short",
        }
        return groups.get(club.lower().replace(" ","").replace("-",""), "iron_mid")