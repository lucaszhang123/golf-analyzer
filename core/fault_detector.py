"""
FaultDetector: runs all fault checks against the full metric time-series
and returns a prioritized FaultReport. Compares against a single elite benchmark.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from .metrics import SwingMetrics
from .fault_rules import (
    FAULT_CHECKS, FaultResult,
    _arrays, _smooth, _find_backswing_window, _find_impact_idx,
    get_club_group, get_club_adjustments,
)


@dataclass
class FaultReport:
    club: str
    club_group: str
    address_frame: int
    top_frame: int
    impact_frame: int
    faults: List[FaultResult] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "",
            "Fault Detection Report",
            "======================",
            f"Benchmark    : Elite / PGA Tour standard",
            f"Club         : {self.club}  ({self.club_group})",
            f"Swing phases : address={self.address_frame}  top={self.top_frame}  impact={self.impact_frame}",
            "",
        ]

        if not self.faults:
            lines.append("No significant faults detected vs elite benchmark.")
            return "\n".join(lines)

        # Group by phase category
        phase_order = ["address", "backswing", "backswing (ascent)", "transition",
                       "downswing", "downswing (descent)", "impact", "follow_through"]

        grouped = {}
        for f in self.faults:
            grouped.setdefault(f.phase, []).append(f)

        lines.append(f"Faults found: {len(self.faults)}\n")

        fault_num = 1
        for phase in phase_order:
            if phase not in grouped:
                continue
            phase_label = phase.upper().replace("_", " ").replace("(", "— ").replace(")", "")
            lines.append(f"── {phase_label} ─────────────────────────")
            for f in grouped[phase]:
                val = f"{f.measured_value:.1f}" if f.measured_value is not None else "N/A"
                lines += [
                    f"{fault_num}. [{f.severity_label.upper()}] {f.display_name}  (severity {f.severity:.2f})",
                    f"   Measured   : {val}   Elite: {f.elite_benchmark}",
                    f"   Detail     : {f.description}",
                    f"   Root cause : {f.root_cause}",
                    f"   Ball flight: {f.ball_flight}",
                    f"   Source     : {f.source}",
                    "",
                ]
                fault_num += 1

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "benchmark": "elite_pga_tour",
            "club": self.club,
            "club_group": self.club_group,
            "swing_phases": {
                "address_frame": self.address_frame,
                "top_frame": self.top_frame,
                "impact_frame": self.impact_frame,
            },
            "faults": [f.to_dict() for f in self.faults],
            "top_fault": self.faults[0].to_dict() if self.faults else None,
            "fault_count": len(self.faults),
        }


class FaultDetector:

    def detect(
        self,
        all_metrics: List[SwingMetrics],
        club: str = "7i",
    ) -> FaultReport:

        if not all_metrics:
            return FaultReport(club, "iron_mid", 0, 0, 0, [])

        club_group = get_club_group(club)
        adjustments = get_club_adjustments(club)

        a = _arrays(all_metrics)
        addr, top = _find_backswing_window(a)
        impact = _find_impact_idx(a, top)

        peak_shoulder = float(np.max(_smooth(a["shoulder"], w=3)))
        print(f"  Club: {club} ({club_group})")
        print(f"  Peak shoulder rotation: {peak_shoulder:.1f}°")
        print(f"  Phases: address={addr}, top={top}, impact={impact}")

        faults = []
        for check_fn in FAULT_CHECKS:
            try:
                result = check_fn(all_metrics, adjustments)
                if result is not None and result.severity > 0.05:
                    faults.append(result)
            except Exception as e:
                print(f"  Warning: {check_fn.__name__} failed: {e}")

        # Deduplicate: if two faults share the same root biomechanical issue,
        # keep only the higher-severity one
        DUPLICATE_GROUPS = [
            # Early extension detected by both general and descent-specific checks
            {"early_extension", "descent_early_extension"},
            # Weight transfer detected by both general and descent checks
            {"poor_weight_transfer", "descent_weight_stall", "descent_weight_reversal"},
            # Kinematic sequence detected by both checks
            {"poor_kinematic_sequence", "descent_sequence_fault"},
        ]
        seen_groups = []
        deduped = []
        for f in faults:
            in_group = False
            for group in DUPLICATE_GROUPS:
                if f.name in group:
                    if group in seen_groups:
                        in_group = True  # already have a fault from this group
                        break
                    else:
                        seen_groups.append(group)
                        break
            if not in_group:
                deduped.append(f)
        faults = deduped

        faults.sort(key=lambda f: f.severity, reverse=True)

        return FaultReport(
            club=club,
            club_group=club_group,
            address_frame=addr,
            top_frame=top,
            impact_frame=impact,
            faults=faults,
        )
