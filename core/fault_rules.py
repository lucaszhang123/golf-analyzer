"""
Research-grounded fault rules comparing against a single ELITE benchmark.

No skill tiers. Every golfer is compared to professional standards.
Severity 0-1 reflects how far the measurement deviates from elite norms,
so even small deviations show up — giving feedback on anything improvable.

ELITE BENCHMARK SOURCES:
  [M2011]  Meister et al. 2011 "Rotational Biomechanics of the Elite Golf Swing"
           J. Applied Biomechanics 27:242-251
           -> Peak X-factor: 56° (SD ~4°), shoulder: ~95-100°, hip: ~45°
           -> Downswing initiated by hip reversal BEFORE shoulder reversal

  [F2018]  Fleisig, "Biomechanics of Golf" (via Cochran/Sean Cochran)
           -> Lead foot: 80-95% weight at impact
           -> Weight: ~80% trail side at top of backswing

  [R1985]  Richards et al. 1985 "Weight Transfer Patterns During the Golf Swing"
           -> Weight transfer timing with trunk rotation critical for club head velocity

  [N2004]  Novosel "Tour Tempo" (validated Yale University study)
           -> Backswing:downswing ratio = 3:1 across virtually ALL elite golfers
           -> Amateurs typically 4:1 to 5:1 (backswing too slow relative to downswing)

  [K2022]  Murakami et al. via "Golf Swing Biomechanics: Systematic Review" MDPI 2022
           -> Lead knee flexion: 18° address, 33° top, 25° impact
           -> Trail knee flexion: 17° address, 24° top, 22° impact

  [TPI]    Titleist Performance Institute research
           -> Early extension: hips thrust toward ball in downswing
           -> Over the top / casting: arms dominate downswing initiation
           -> Chicken wing: 35% chance if wrist mobility limited

  [HC]     HackMotion sensor data (1M+ swings analyzed)
           -> Trail wrist: 10-15° more extended at impact than address
           -> Lead wrist should move toward flexion (not extension) through impact

  [SLT]    Swing Lab Theory / Adam Young Golf biomechanics coaching
           -> Reverse spine angle: upper body tilts toward target in backswing
           -> Spine angle change >8° from address to impact = early extension

CLUB GROUPS & ADJUSTMENTS:
  driver:      wider stance, more trail-side tilt at address, ascending AoA
               slightly less weight transfer to lead side expected at impact
  iron_long:   (fairway wood, hybrid, 2-3i) — long iron behavior
  iron_mid:    (4-7i) — baseline benchmarks
  iron_short:  (8i-LW) — shorter swing, more lead-side weight at impact
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .metrics import SwingMetrics


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FaultResult:
    name: str
    display_name: str
    description: str
    ball_flight: str
    root_cause: str
    severity: float            # 0.0 - 1.0
    severity_label: str        # "mild" / "moderate" / "severe"
    phase: str
    measured_value: Optional[float] = None
    elite_benchmark: Optional[str] = None
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "fault": self.name,
            "display_name": self.display_name,
            "severity": round(self.severity, 2),
            "severity_label": self.severity_label,
            "phase": self.phase,
            "description": self.description,
            "root_cause": self.root_cause,
            "ball_flight_effect": self.ball_flight,
            "measured_value": round(self.measured_value, 1) if self.measured_value is not None else None,
            "elite_benchmark": self.elite_benchmark,
        }


def _label(s: float) -> str:
    if s < 0.33:
        return "mild"
    elif s < 0.66:
        return "moderate"
    return "severe"


# ---------------------------------------------------------------------------
# Curve analysis helpers
# ---------------------------------------------------------------------------

def _arrays(metrics: List[SwingMetrics]) -> dict:
    n = len(metrics)
    return {
        "shoulder":      np.array([m.shoulder_rotation_deg or 0.0 for m in metrics]),
        "hip":           np.array([m.hip_rotation_deg or 0.0 for m in metrics]),
        "xfactor":       np.array([m.x_factor_deg or 0.0 for m in metrics]),
        "spine":         np.array([m.spine_tilt_deg or 0.0 for m in metrics]),
        "weight":        np.array([m.weight_shift or 0.0 for m in metrics]),
        "elbow":         np.array([m.lead_elbow_angle_deg or 0.0 for m in metrics]),
        "wrist":         np.array([m.wrist_hinge_deg or 0.0 for m in metrics]),
        "head":          np.array([m.head_movement_px or 0.0 for m in metrics]),
        "hip_sway":      np.array([m.hip_sway_px or 0.0 for m in metrics]),
        "lead_knee":     np.array([m.lead_knee_angle_deg or 0.0 for m in metrics]),
        "trail_knee":    np.array([m.trail_knee_angle_deg or 0.0 for m in metrics]),
        "hand_height_y": np.array([m.hand_height_y or 0.0 for m in metrics]),
        "conf":          np.array([m.confidence for m in metrics]),
        "n":             n,
    }


def _smooth(arr: np.ndarray, w: int = 3) -> np.ndarray:
    if w < 2 or len(arr) < w:
        return arr.copy()
    return np.convolve(arr, np.ones(w) / w, mode="same")


def _find_backswing_window(a: dict) -> Tuple[int, int]:
    """
    Returns (address_idx, top_idx).

    ADDRESS: first stable low-movement window in first 30% of frames.

    TOP: frame where hands are at their HIGHEST vertical position
    (minimum y value in image coords where y increases downward).
    This is reliable for any camera angle — hands are always highest
    at the top of the backswing.
    """
    n = a["n"]
    conf = a["conf"]

    # Address: most stable weight-shift window in first 30%
    cutoff = max(3, n // 3)
    win = max(2, n // 20)
    addr = 0
    best_var = float("inf")
    weight = _smooth(a["weight"], w=3)
    for i in range(cutoff - win):
        v = float(np.var(weight[i:i+win]))
        c = float(np.mean(conf[i:i+win]))
        if v < best_var and c > 0.5:
            best_var = v
            addr = i + win // 2

    # Top: minimum hand y-value (highest physical hand position)
    hand_height = a.get("hand_height_y")
    if hand_height is not None and np.any(hand_height > 0):
        search_start = addr + 1
        search_end = min(n, addr + int((n - addr) * 0.85))
        region = hand_height[search_start:search_end]
        conf_region = conf[search_start:search_end]
        # Replace low-confidence frames with max y so they don't win
        weighted_region = np.where(conf_region > 0.4, region, np.max(region) + 1)
        top = search_start + int(np.argmin(weighted_region))
    else:
        # Fallback: shoulder rotation peak
        shoulder = _smooth(a["shoulder"], w=max(3, n // 15))
        search_end = min(n - 1, addr + int((n - addr) * 0.80))
        weighted = shoulder[addr+1:search_end] * conf[addr+1:search_end]
        top = addr + 1 + int(np.argmax(weighted))

    return addr, top


def _find_impact_idx(a: dict, top: int) -> int:
    """
    Impact: after top, hands return to approximately address height.
    Falls back to weight shift signal.
    """
    n = a["n"]
    conf = a["conf"]

    hand_height = a.get("hand_height_y")
    if hand_height is not None and np.any(hand_height > 0):
        addr_hand_y = float(np.mean(hand_height[:max(1, top // 4)]))
        top_hand_y = float(hand_height[top])
        arc_range = addr_hand_y - top_hand_y
        if arc_range > 5:
            threshold_y = top_hand_y + arc_range * 0.75
            for i in range(top + 1, n):
                if hand_height[i] >= threshold_y and conf[i] > 0.4:
                    return i

    weight = _smooth(a["weight"], w=3)
    top_weight = weight[top]
    for i in range(top + 1, n):
        if weight[i] > top_weight + 0.15 and conf[i] > 0.4:
            return i

    return top + max(1, int((n - top) * 0.60))


# ---------------------------------------------------------------------------
# FAULT CHECKS — all compare against single elite benchmark
# ---------------------------------------------------------------------------

def check_shoulder_turn(metrics, adjustments=None):
    """
    Elite: ~95-100° shoulder turn at top [M2011, F2018].
    We measure the peak value in the backswing window.
    """
    a = _arrays(metrics)
    shoulder = _smooth(a["shoulder"], w=3)
    _, top = _find_backswing_window(a)

    peak = float(np.max(shoulder[max(0, top-3):min(a["n"], top+4)]))
    elite_target = 95.0
    adj = adjustments.get("shoulder_min", 1.0) if adjustments else 1.0
    effective_target = elite_target * adj

    if peak >= effective_target * 0.92:  # within 8% of elite = no fault
        return None

    severity = min(1.0, (effective_target - peak) / effective_target)
    return FaultResult(
        name="restricted_shoulder_turn",
        display_name="Restricted Shoulder Turn",
        description=f"Peak shoulder turn of {peak:.1f}° vs elite benchmark of ~{effective_target:.0f}°. "
                    "Elite golfers achieve 95-100° of shoulder turn at the top (Fleisig, Meister 2011). "
                    "A restricted turn limits stored elastic energy and forces arm-dominated compensation.",
        root_cause="Limited thoracic spine mobility, early hip over-rotation, lead arm tension, or trying to 'keep the head still' by restricting the turn.",
        ball_flight="Loss of distance, pushes, pull hooks from arms overworking",
        severity=severity, severity_label=_label(severity), phase="backswing",
        measured_value=peak, elite_benchmark=f"~{effective_target:.0f}° for this club",
        source="Meister et al. 2011; Fleisig Biomechanics of Golf",
    )


def check_hip_turn_at_top(metrics, adjustments=None):
    """
    Elite: ~45° hip turn at top [M2011, F2018].
    Too little = restricted, too much = loss of X-factor coil.
    """
    a = _arrays(metrics)
    hip = _smooth(a["hip"], w=3)
    _, top = _find_backswing_window(a)

    peak_hip = float(np.max(hip[max(0, top-3):min(a["n"], top+4)]))
    elite_target = 45.0
    adj = adjustments.get("shoulder_min", 1.0) if adjustments else 1.0
    effective_target = elite_target * adj

    # Too little hip turn
    if peak_hip < effective_target * 0.60:
        severity = min(1.0, (effective_target * 0.60 - peak_hip) / (effective_target * 0.60))
        return FaultResult(
            name="restricted_hip_turn",
            display_name="Restricted Hip Turn",
            description=f"Hip turn at top is {peak_hip:.1f}° vs elite ~{effective_target:.0f}°. "
                        "Insufficient hip rotation restricts the body's ability to store and release energy, "
                        "and forces excessive lateral sway as a compensation.",
            root_cause="Limited hip internal/external rotation mobility, or active resistance of hip turn (common coaching misconception).",
            ball_flight="Loss of distance, steep downswing, pulls",
            severity=severity, severity_label=_label(severity), phase="backswing",
            measured_value=peak_hip, elite_benchmark=f"~{effective_target:.0f}°",
            source="Meister et al. 2011; Fleisig Biomechanics of Golf",
        )

    # Too much hip turn (over-rotation killing X-factor)
    if peak_hip > effective_target * 1.50:
        severity = min(1.0, (peak_hip - effective_target * 1.50) / effective_target)
        return FaultResult(
            name="over_rotation_hips",
            display_name="Excessive Hip Turn (Over-Rotation)",
            description=f"Hip turn at top is {peak_hip:.1f}°, significantly exceeding the ~{effective_target:.0f}° elite benchmark. "
                        "Excessive hip turn eliminates the coil between hips and shoulders, reducing X-factor and stored elastic energy.",
            root_cause="No resistance in trail hip — the hip slides or sways rather than rotating around a fixed axis.",
            ball_flight="Lack of power, fat shots, over-the-top downswing",
            severity=severity, severity_label=_label(severity), phase="backswing",
            measured_value=peak_hip, elite_benchmark=f"~{effective_target:.0f}°",
            source="Meister et al. 2011",
        )
    return None


def check_x_factor(metrics, adjustments=None):
    """
    Elite: peak X-factor = 56° (SD ~4°) [M2011].
    Peak occurs at START of downswing as hips reverse before shoulders.
    """
    a = _arrays(metrics)
    xf = _smooth(a["xfactor"], w=3)
    n = a["n"]

    win_s = max(0, int(n * 0.15))
    win_e = min(n, int(n * 0.75))
    peak_xf = float(np.max(xf[win_s:win_e]))

    adj = adjustments.get("xf_min", 1.0) if adjustments else 1.0
    elite_target = 56.0 * adj

    if peak_xf >= elite_target * 0.86:
        return None

    severity = min(1.0, (elite_target - peak_xf) / elite_target)
    return FaultResult(
        name="low_x_factor",
        display_name="Low X-Factor (Hip-Shoulder Separation)",
        description=f"Peak X-factor {peak_xf:.1f}° vs elite mean of {elite_target:.0f}° (SD ~4°). "
                    "Stanford research (Meister 2011) found X-factor correlates r=0.94 with club head speed. "
                    "The X-factor peaks at the START of the downswing — hips reverse while shoulders still coil.",
        root_cause="Hips and shoulders rotating as one unit without separation. Often due to insufficient hip mobility or initiating downswing with shoulders.",
        ball_flight="Reduced distance, weak contact, inability to compress ball",
        severity=severity, severity_label=_label(severity), phase="transition",
        measured_value=peak_xf, elite_benchmark=f"~{elite_target:.0f}° (pros: 52-60°)",
        source="Meister et al. 2011 J. Applied Biomechanics",
    )


def check_kinematic_sequence(metrics, adjustments=None):
    """
    Elite: hip rotation reverses BEFORE shoulder rotation in downswing [M2011].
    Confirmed in 100% of professional golfers studied.
    """
    a = _arrays(metrics)
    shoulder = _smooth(a["shoulder"], w=3)
    hip = _smooth(a["hip"], w=3)
    _, top = _find_backswing_window(a)
    n = a["n"]

    def first_decline(arr, from_idx, lookahead=4):
        for i in range(from_idx, min(n - lookahead, from_idx + int(n * 0.45))):
            if arr[i] > arr[i + lookahead] + 1.0:
                return i
        return None

    hip_rev = first_decline(hip, top)
    shld_rev = first_decline(shoulder, top)

    if hip_rev is None or shld_rev is None:
        return None

    lag = shld_rev - hip_rev
    if lag >= 0:
        return None

    severity = min(1.0, abs(lag) / max(4, int(n * 0.10)))
    return FaultResult(
        name="poor_kinematic_sequence",
        display_name="Poor Kinematic Sequence (Over the Top)",
        description=f"Shoulder rotation reverses {abs(lag)} frame(s) before hips in the downswing. "
                    "Meister et al. (2011) confirmed in ALL professional golfers that the downswing is initiated "
                    "by pelvic reversal, followed by upper torso. Shoulder-first = over-the-top path.",
        root_cause="Initiating the downswing with hands/shoulders instead of lower body. Often a compensation for restricted hip mobility.",
        ball_flight="Pull, pull-slice, steep AoA causing fat/thin shots",
        severity=severity, severity_label=_label(severity), phase="transition",
        measured_value=float(lag),
        elite_benchmark="Hip rotation should reverse before or simultaneously with shoulders",
        source="Meister et al. 2011 J. Applied Biomechanics",
    )


def check_swing_tempo(metrics, adjustments=None):
    """
    Elite: backswing:downswing ratio = 3:1 [Novosel 2004, Yale validation].
    Amateurs typically 4:1-5:1 (backswing too slow).
    We detect top and impact frames to estimate the ratio.
    """
    a = _arrays(metrics)
    addr, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    backswing_frames = top - addr
    downswing_frames = impact - top

    if backswing_frames < 2 or downswing_frames < 1:
        return None

    ratio = backswing_frames / downswing_frames
    elite_ratio = 3.0

    # Too slow backswing (ratio >> 3): most common amateur fault
    if ratio > 4.2:
        severity = min(1.0, (ratio - 3.0) / 4.0)
        return FaultResult(
            name="slow_tempo",
            display_name="Slow Tempo (Rushed Transition)",
            description=f"Backswing:downswing ratio is {ratio:.1f}:1 vs elite 3:1. "
                        "Novosel (Tour Tempo, 2004) found virtually all elite golfers share a 3:1 ratio, "
                        "validated by Yale University research. A ratio above 4:1 indicates the backswing "
                        "is too slow relative to the downswing, causing a jerky transition that breaks the kinematic sequence.",
            root_cause="Over-deliberate backswing followed by an aggressive 'hit' at the top. The body cannot transition smoothly.",
            ball_flight="Inconsistency, over-the-top, loss of lag",
            severity=severity, severity_label=_label(severity), phase="transition",
            measured_value=ratio,
            elite_benchmark="3:1 backswing-to-downswing ratio",
            source="Novosel 2004 'Tour Tempo'; Yale University validation study",
        )

    # Too fast backswing (ratio << 3): less common but causes poor loading
    if ratio < 2.0:
        severity = min(1.0, (3.0 - ratio) / 2.0)
        return FaultResult(
            name="fast_backswing",
            display_name="Too-Fast Backswing",
            description=f"Backswing:downswing ratio is {ratio:.1f}:1 vs elite 3:1. "
                        "A rushed backswing doesn't allow enough time to fully load the trail side "
                        "and sequence the downswing properly.",
            root_cause="Anxiety or tension causing a snatch takeaway. Lack of swing routine.",
            ball_flight="Inconsistent contact, loss of power, pulls",
            severity=severity, severity_label=_label(severity), phase="backswing",
            measured_value=ratio,
            elite_benchmark="3:1 backswing-to-downswing ratio",
            source="Novosel 2004 'Tour Tempo'",
        )

    return None


def check_weight_loading_backswing(metrics, adjustments=None):
    """
    Elite: ~80% weight on trail foot at top of backswing [R1985, F2018].
    Our scale: -0.5 to -0.8 = good trail loading. Positive = reverse pivot.
    """
    a = _arrays(metrics)
    weight = _smooth(a["weight"], w=3)
    addr, top = _find_backswing_window(a)

    mid = (addr + top) // 2
    backswing_weight = float(np.mean(weight[mid:top+1]))

    adj = adjustments.get("wt_top", 1.0) if adjustments else 1.0

    # Reverse pivot: weight going lead side on backswing
    if backswing_weight > 0.15 * adj:
        severity = min(1.0, backswing_weight / (0.7 * adj))
        return FaultResult(
            name="reverse_pivot",
            display_name="Reverse Pivot",
            description=f"Weight shift during backswing is {backswing_weight:.2f} (lead side). "
                        "Research shows elite golfers load ~80% onto the trail foot at top of backswing. "
                        "A reverse pivot (weight going forward on backswing) eliminates power storage and causes a steep downswing.",
            root_cause="Upper body tilting toward target in backswing (reverse spine angle). "
                       "Often from trying to 'keep the head still' by shifting weight the wrong direction.",
            ball_flight="Fat shots, pulls, slices, steep AoA",
            severity=severity, severity_label=_label(severity), phase="backswing",
            measured_value=backswing_weight,
            elite_benchmark="~-0.5 to -0.8 (60-80% trail side loaded)",
            source="Richards et al. 1985; Fleisig Biomechanics of Golf",
        )

    # Insufficient trail loading (still centered or barely shifted)
    if backswing_weight > -0.15 * adj:
        severity = min(0.6, abs(backswing_weight + 0.15) / 0.35)
        return FaultResult(
            name="insufficient_trail_loading",
            display_name="Insufficient Trail Side Loading",
            description=f"Weight barely shifts to trail side ({backswing_weight:.2f}) during backswing. "
                        "Elite golfers load 60-80% onto the trail foot, creating a coiled spring ready to unload.",
            root_cause="Excessive lateral hip stability keeping weight centered, or short backswing preventing full load.",
            ball_flight="Loss of distance, steep downswing",
            severity=severity, severity_label=_label(severity), phase="backswing",
            measured_value=backswing_weight,
            elite_benchmark="-0.5 to -0.8 (trail side loaded)",
            source="Richards et al. 1985",
        )

    return None


def check_weight_transfer_impact(metrics, adjustments=None):
    """
    Elite: 80-95% weight on lead foot at impact [F2018, R1985].
    Our scale target: > 0.55 at impact.
    """
    a = _arrays(metrics)
    weight = _smooth(a["weight"], w=3)
    _, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)

    win_s = max(0, impact - 3)
    win_e = min(a["n"], impact + 4)
    impact_weight = float(np.max(weight[win_s:win_e]))

    adj = adjustments.get("wt_impact", 1.0) if adjustments else 1.0
    elite_target = 0.65 * adj

    if impact_weight >= elite_target:
        return None

    severity = min(1.0, (elite_target - impact_weight) / elite_target)
    return FaultResult(
        name="poor_weight_transfer",
        display_name="Insufficient Weight Transfer to Impact",
        description=f"Weight shift at impact: {impact_weight:.2f} vs elite target of >{elite_target:.2f}. "
                    "Fleisig's research shows elite golfers support 80-95% of weight on the lead foot at impact. "
                    "Hanging back creates a scooping motion and weak, inconsistent contact.",
        root_cause="Reverse pivot habit, fear of hitting fat, poor hip sequencing, or trying to lift the ball.",
        ball_flight="Fat shots, thin shots, loss of distance, pushes right",
        severity=severity, severity_label=_label(severity), phase="impact",
        measured_value=impact_weight, elite_benchmark=f">{elite_target:.2f} (80-95% lead foot)",
        source="Fleisig Biomechanics of Golf; Richards et al. 1985",
    )


def check_early_extension(metrics, adjustments=None):
    """
    Elite: spine angle maintained from backswing through impact [TPI, SLT].
    >8° change = fault. Loss of posture = hips thrust toward ball.
    """
    a = _arrays(metrics)
    spine = _smooth(a["spine"], w=3)
    addr, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    addr_tilt = float(np.mean(spine[:max(1, top // 3)]))
    win_s = max(top, impact - 3)
    win_e = min(n, impact + 4)
    impact_tilt = float(np.mean(spine[win_s:win_e]))

    tilt_change = abs(impact_tilt) - abs(addr_tilt)
    adj = adjustments.get("spine", 1.0) if adjustments else 1.0
    threshold = 8.0 * adj

    if tilt_change <= threshold:
        return None

    severity = min(1.0, (tilt_change - threshold) / 18.0)
    return FaultResult(
        name="early_extension",
        display_name="Early Extension (Loss of Posture)",
        description=f"Spine tilt changed {tilt_change:.1f}° from backswing to impact (threshold: {threshold:.0f}°). "
                    "TPI research identifies early extension as one of the most common faults — "
                    "the hips thrust toward the ball causing the upper body to stand up through impact.",
        root_cause="Limited hip mobility forcing the body to extend to create rotation space. "
                   "Weak core stability, or trying to 'help' the ball into the air.",
        ball_flight="Thin strikes, blocks right, hooks from compensatory flip",
        severity=severity, severity_label=_label(severity), phase="downswing",
        measured_value=tilt_change, elite_benchmark=f"<{threshold:.0f}° spine angle change",
        source="TPI research; Swing Lab Theory biomechanics",
    )


def check_head_stability(metrics, adjustments=None):
    """
    Elite: head moves minimally from address through impact.
    Lateral movement = sway. Vertical = early extension.
    """
    a = _arrays(metrics)
    head = a["head"]
    _, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)

    max_head = float(np.max(head[:impact+1]))
    adj = adjustments.get("head", 1.0) if adjustments else 1.0
    threshold = 40.0 * adj

    if max_head <= threshold:
        return None

    severity = min(1.0, (max_head - threshold) / 70.0)
    return FaultResult(
        name="excessive_head_movement",
        display_name="Excessive Head Movement",
        description=f"Head moved {max_head:.0f}px from address through impact (threshold: {threshold:.0f}px). "
                    "The swing center should remain stable — lateral movement indicates sway, "
                    "vertical movement indicates early extension.",
        root_cause="Lateral: hip sway during backswing. Vertical: early extension / standing up through impact.",
        ball_flight="Inconsistent contact, heel/toe strikes, difficulty controlling low point",
        severity=severity, severity_label=_label(severity), phase="backswing",
        measured_value=max_head, elite_benchmark=f"<{threshold:.0f}px from address",
        source="General biomechanics coaching consensus; TPI research",
    )


def check_hip_sway(metrics, adjustments=None):
    """
    Elite: hips rotate around stable spine axis, minimal lateral slide.
    Sway detected from hip_sway_px during backswing window.
    """
    a = _arrays(metrics)
    sway = _smooth(a["hip_sway"], w=3)
    addr, top = _find_backswing_window(a)

    region = sway[addr:top+1]
    if len(region) == 0:
        return None

    max_sway = float(np.max(np.abs(region)))
    threshold = 22.0

    if max_sway <= threshold:
        return None

    severity = min(1.0, (max_sway - threshold) / 45.0)
    return FaultResult(
        name="hip_sway",
        display_name="Hip Sway (Lateral Slide)",
        description=f"Hips slid {max_sway:.0f}px laterally during backswing (threshold: {threshold:.0f}px). "
                    "Elite golfers rotate the hips around a stable spine axis. Lateral sway "
                    "prevents proper weight loading and destabilizes the downswing path.",
        root_cause="Lack of trail hip internal rotation mobility, or shifting weight sideways rather than rotating.",
        ball_flight="Reverse pivot effects, fat shots, steep downswing, loss of distance",
        severity=severity, severity_label=_label(severity), phase="backswing",
        measured_value=max_sway, elite_benchmark="<22px lateral hip movement",
        source="Biomechanical coaching consensus; TPI screening",
    )


def check_lead_knee_flex(metrics, adjustments=None):
    """
    Elite lead knee flexion: 18° address, 33° top, 25° impact [K2022].
    Straightening lead knee at top = loss of flex/athletic position.
    """
    a = _arrays(metrics)
    knee = _smooth(a["lead_knee"], w=3)
    addr, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    # Address: should be ~150-165° (slightly flexed — 180° = straight)
    addr_knee = float(np.mean(knee[:max(1, addr+3)]))

    # Top: knee should flex MORE (increase bend) — angle decreases toward ~147°
    top_knee = float(np.mean(knee[max(0, top-2):min(n, top+3)]))

    # If knee becomes MORE straight at top (angle increases), fault
    # Our angle is measured differently — we check that knee doesn't straighten
    if addr_knee < 10 or top_knee < 10:
        return None  # Missing data

    # Lead knee should flex (angle decrease) or stay same going to top
    knee_change = top_knee - addr_knee  # positive = straightening = bad
    if knee_change <= 5:
        return None

    severity = min(1.0, knee_change / 30.0)
    return FaultResult(
        name="lead_knee_straightening",
        display_name="Lead Knee Straightening in Backswing",
        description=f"Lead knee angle increased {knee_change:.1f}° during backswing (straightening). "
                    "Research (Murakami et al. via MDPI 2022 systematic review) shows elite golfers "
                    "maintain or increase lead knee flex from 18° at address to 33° at top of backswing. "
                    "Straightening the lead knee reduces stability and athletic coil.",
        root_cause="Insufficient hip flexibility causing the knee to straighten as compensation. "
                   "Or weight shifting toward the lead side (reverse pivot).",
        ball_flight="Loss of power, reverse pivot, inconsistent contact",
        severity=severity, severity_label=_label(severity), phase="backswing",
        measured_value=knee_change, elite_benchmark="Lead knee should flex (not straighten) into backswing",
        source="Murakami et al. via MDPI Sports 2022 systematic review",
    )


def check_lead_arm_structure(metrics, adjustments=None):
    """
    Elite: lead arm relatively straight through impact (>150°).
    Chicken wing = elbow collapses, reduces width and opens face.
    """
    a = _arrays(metrics)
    elbow = _smooth(a["elbow"], w=3)
    _, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    win_s = max(top, impact - 4)
    win_e = min(n, impact + 5)
    if win_e <= win_s:
        return None

    min_angle = float(np.min(elbow[win_s:win_e]))
    if min_angle < 10:
        return None

    elite_target = 155.0

    if min_angle >= elite_target * 0.92:
        return None

    severity = min(1.0, (elite_target - min_angle) / 55.0)
    return FaultResult(
        name="chicken_wing",
        display_name="Chicken Wing (Lead Arm Collapse)",
        description=f"Lead elbow angle near impact: {min_angle:.1f}° vs elite benchmark >150°. "
                    "TPI research shows wrist mobility limitations cause 35% chance of chicken wing. "
                    "A collapsed lead arm reduces swing width, opens the club face, and transfers power inefficiently.",
        root_cause="Trail arm dominance pushing through impact. Flipping/scooping motion. "
                   "Compensation for an inside-out swing path.",
        ball_flight="Weak slices, pulls, reduced distance, inconsistent contact",
        severity=severity, severity_label=_label(severity), phase="impact",
        measured_value=min_angle, elite_benchmark=">150° through impact",
        source="TPI research; TrackMan University; HackMotion data",
    )


def check_wrist_hinge_at_top(metrics, adjustments=None):
    """
    Elite: wrists should be fully hinged at top of backswing.
    Our wrist_hinge_deg measures angle at lead wrist — smaller angle = more hinge.
    Wrists fully cocked at top = max lag potential.
    """
    a = _arrays(metrics)
    wrist = _smooth(a["wrist"], w=3)
    _, top = _find_backswing_window(a)
    n = a["n"]

    top_wrist = float(np.mean(wrist[max(0, top-2):min(n, top+3)]))
    if top_wrist < 5:
        return None

    # Ideal: wrists fully hinged at top — angle significantly less than at address
    addr_wrist = float(np.mean(wrist[:max(1, 5)]))

    if addr_wrist < 10 or top_wrist < 10:
        return None

    # Insufficient hinge: wrist angle at top not much different from address
    hinge_amount = addr_wrist - top_wrist  # positive = more hinge = good
    if hinge_amount >= 15:
        return None

    severity = min(1.0, (15 - hinge_amount) / 30.0)
    return FaultResult(
        name="insufficient_wrist_hinge",
        display_name="Insufficient Wrist Hinge at Top",
        description=f"Wrist hinge change from address to top: {hinge_amount:.1f}° (target: >15°). "
                    "Keiser University Golf research notes the wrists should reach maximum load "
                    "(fully cocked and hinged) at the top of the backswing to maximize lag and club head speed on the downswing.",
        root_cause="Tight wrists/forearms, 'one-piece takeaway' taken too far, or insufficient wrist mobility.",
        ball_flight="Loss of lag, reduced club head speed, loss of distance",
        severity=severity, severity_label=_label(severity), phase="backswing",
        measured_value=hinge_amount, elite_benchmark=">15° wrist hinge increase from address to top",
        source="Keiser University College of Golf; HackMotion analysis",
    )


def check_spine_tilt_at_address(metrics, adjustments=None):
    """
    Elite: spine tilted away from target at address (5-15° depending on club).
    [Swing Align coaching; multiple sources]. Neutral or toward-target = fault.
    """
    a = _arrays(metrics)
    spine = _smooth(a["spine"], w=3)
    addr, _ = _find_backswing_window(a)

    addr_tilt = float(np.mean(spine[max(0, addr-2):addr+3]))
    adj = adjustments.get("spine", 1.0) if adjustments else 1.0
    elite_min = 2.0 * adj

    # Spine tilt toward target (positive in our convention) = fault
    if addr_tilt < elite_min:
        severity = min(0.7, abs(addr_tilt - elite_min) / 10.0)
        return FaultResult(
            name="poor_address_spine_tilt",
            display_name="Poor Spine Tilt at Address",
            description=f"Spine tilt at address is {addr_tilt:.1f}° — should be tilted away from target. "
                        "Swing Align research shows spine should tilt 5-15° away from target at address "
                        "to promote proper rotation and correct angle of attack.",
            root_cause="Ball position too far forward, weight distribution issue, or setup habit.",
            ball_flight="Inconsistent angle of attack, thin shots, loss of power",
            severity=severity, severity_label=_label(severity), phase="address",
            measured_value=addr_tilt, elite_benchmark="5-15° tilt away from target",
            source="Swing Align coaching research; biomechanical coaching consensus",
        )
    return None


def check_follow_through_completion(metrics, adjustments=None):
    """
    Elite: full rotation through to finish — shoulders past parallel,
    weight fully on lead side, balanced finish.
    Incomplete follow-through indicates deceleration before impact.
    """
    a = _arrays(metrics)
    shoulder = _smooth(a["shoulder"], w=3)
    weight = _smooth(a["weight"], w=3)
    n = a["n"]

    # Check last 15% of swing for finish position
    finish_start = int(n * 0.85)
    finish_shoulder = float(np.mean(shoulder[finish_start:]))
    finish_weight = float(np.mean(weight[finish_start:]))

    faults = []

    # Incomplete rotation at finish
    if finish_shoulder < 20.0:
        severity = min(1.0, (20.0 - finish_shoulder) / 20.0)
        faults.append(FaultResult(
            name="incomplete_follow_through",
            display_name="Incomplete Follow Through",
            description=f"Shoulder rotation at finish: {finish_shoulder:.1f}° (target: >20°). "
                        "An incomplete follow-through indicates deceleration before or at impact. "
                        "Elite golfers accelerate through the ball — the follow-through is a result, not a goal.",
            root_cause="Decelerating through impact, fear of over-swinging, or poor balance preventing full rotation.",
            ball_flight="Short shots, pulled shots, loss of distance",
            severity=severity, severity_label=_label(severity), phase="follow_through",
            measured_value=finish_shoulder, elite_benchmark=">20° shoulder rotation at finish",
            source="Biomechanical coaching consensus",
        ))

    # Weight not on lead side at finish
    if finish_weight < 0.4:
        severity = min(1.0, (0.4 - finish_weight) / 0.5)
        faults.append(FaultResult(
            name="weight_not_on_lead_finish",
            display_name="Weight Not on Lead Side at Finish",
            description=f"Weight shift at finish: {finish_weight:.2f} (target: >0.4). "
                        "Elite golfers finish with virtually all weight on the lead foot. "
                        "Remaining on the trail side indicates poor weight transfer through the swing.",
            root_cause="Reverse pivot habit, insufficient hip drive, or hanging back to try to 'help' the ball.",
            ball_flight="Fat shots, loss of distance, pushes",
            severity=severity, severity_label=_label(severity), phase="follow_through",
            measured_value=finish_weight, elite_benchmark=">0.5 lead side at finish",
            source="Fleisig; Richards et al. 1985",
        ))

    return faults[0] if faults else None


# ---------------------------------------------------------------------------
# Club selection
# ---------------------------------------------------------------------------

CLUB_GROUPS = {
    "driver": "driver",
    "3w": "iron_long", "5w": "iron_long", "7w": "iron_long",
    "hybrid": "iron_long", "2i": "iron_long", "3i": "iron_long",
    "4i": "iron_mid", "5i": "iron_mid", "6i": "iron_mid", "7i": "iron_mid",
    "8i": "iron_short", "9i": "iron_short",
    "pw": "iron_short", "gw": "iron_short", "sw": "iron_short", "lw": "iron_short",
}

CLUB_ADJUSTMENTS = {
    "driver":     {"shoulder_min": 1.05, "xf_min": 1.00, "wt_top": 0.90, "wt_impact": 0.88, "spine": 1.15, "head": 1.10},
    "iron_long":  {"shoulder_min": 1.00, "xf_min": 0.97, "wt_top": 0.95, "wt_impact": 0.97, "spine": 1.00, "head": 1.00},
    "iron_mid":   {"shoulder_min": 0.97, "xf_min": 0.95, "wt_top": 1.00, "wt_impact": 1.00, "spine": 0.95, "head": 0.95},
    "iron_short": {"shoulder_min": 0.90, "xf_min": 0.88, "wt_top": 1.00, "wt_impact": 1.05, "spine": 0.90, "head": 0.90},
}


def get_club_group(club: str) -> str:
    return CLUB_GROUPS.get(club.lower().replace(" ", "").replace("-", ""), "iron_mid")


def get_club_adjustments(club: str) -> dict:
    return CLUB_ADJUSTMENTS.get(get_club_group(club), CLUB_ADJUSTMENTS["iron_mid"])



# ---------------------------------------------------------------------------
# ASCENT MECHANICS — how each metric evolves during the backswing
# ---------------------------------------------------------------------------

def check_ascent_rotation_rate(metrics, adjustments=None):
    """
    Backswing: shoulder rotation should build progressively and continuously.
    Stalling or reversing mid-backswing = loss of coil and poor loading.
    Detect: any significant dip in shoulder rotation curve during ascent.
    """
    a = _arrays(metrics)
    shoulder = _smooth(a["shoulder"], w=3)
    addr, top = _find_backswing_window(a)

    if top <= addr + 2:
        return None

    backswing = shoulder[addr:top+1]
    n_bs = len(backswing)
    if n_bs < 4:
        return None

    # Check for any reversal mid-backswing (rotation dropping before top)
    max_so_far = backswing[0]
    max_reversal = 0.0
    for i in range(1, n_bs - 1):
        if backswing[i] > max_so_far:
            max_so_far = backswing[i]
        elif max_so_far > 5.0:
            drop = max_so_far - backswing[i]
            max_reversal = max(max_reversal, drop)

    if max_reversal < 6.0:
        return None

    severity = min(1.0, max_reversal / 25.0)
    return FaultResult(
        name="ascent_rotation_stall",
        display_name="Backswing Rotation Stall",
        description=f"Shoulder rotation dropped {max_reversal:.1f}° mid-backswing before resuming. "
                    "A proper backswing builds rotation continuously and progressively. "
                    "Any mid-backswing stall or reversal breaks the kinetic chain and disrupts timing.",
        root_cause="Tension in lead arm/shoulder causing a pause or re-routing mid-swing. "
                   "Or a two-part takeaway where the club is picked up separately from the body turn.",
        ball_flight="Inconsistent timing, over-the-top compensation, loss of power",
        severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
        measured_value=max_reversal,
        elite_benchmark="Rotation should build continuously with no mid-backswing reversal",
        source="Biomechanical kinematic continuity principle; TPI",
    )


def check_ascent_weight_direction(metrics, adjustments=None):
    """
    Backswing ascent: weight should move continuously toward trail side.
    If weight oscillates or moves toward lead side mid-backswing, that's
    a loading fault (early weight shift reversal = over-the-top trigger).
    """
    a = _arrays(metrics)
    weight = _smooth(a["weight"], w=3)
    addr, top = _find_backswing_window(a)

    if top <= addr + 2:
        return None

    backswing_w = weight[addr:top+1]
    n_bs = len(backswing_w)
    if n_bs < 4:
        return None

    # Check: does weight move in the right direction (toward trail = negative)?
    # Find if weight moves toward lead (positive) before top — early shift
    early_shift_idx = None
    for i in range(1, n_bs - 2):
        # Weight moving strongly lead-ward mid-backswing
        if backswing_w[i] > backswing_w[0] + 0.15:
            early_shift_idx = i
            break

    if early_shift_idx is None:
        return None

    shift_amount = float(backswing_w[early_shift_idx] - backswing_w[0])
    pct_through = int(100 * early_shift_idx / n_bs)
    severity = min(1.0, shift_amount / 0.5)

    return FaultResult(
        name="early_weight_shift",
        display_name="Early Weight Shift to Lead (Ascent)",
        description=f"Weight shifted {shift_amount:.2f} toward lead side at {pct_through}% through the backswing. "
                    "During ascent, weight should progressively load the trail side. "
                    "An early lead-side shift mid-backswing triggers an over-the-top downswing — "
                    "the body starts coming forward before the club reaches the top.",
        root_cause="Rushing the transition — the lower body begins its forward move before the backswing completes. "
                   "Often caused by poor sequencing or anxiety at the top.",
        ball_flight="Over-the-top pull, steep descending path, pulls and slices",
        severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
        measured_value=shift_amount,
        elite_benchmark="Weight should load trail side continuously through the full backswing",
        source="Richards et al. 1985; Meister et al. 2011",
    )


def check_ascent_spine_stability(metrics, adjustments=None):
    """
    Backswing ascent: spine tilt should remain stable or tilt slightly away
    from target (trail side). Tilting toward target = reverse spine angle.
    Excessive lateral movement during ascent destabilizes the arc.
    """
    a = _arrays(metrics)
    spine = _smooth(a["spine"], w=3)
    addr, top = _find_backswing_window(a)

    addr_tilt = float(np.mean(spine[max(0, addr-1):addr+2]))
    top_tilt = float(np.mean(spine[max(0, top-2):top+3]))

    tilt_change = top_tilt - addr_tilt

    # Reverse spine angle: tilting toward target (positive change in our convention)
    if tilt_change > 8.0:
        severity = min(1.0, (tilt_change - 8.0) / 20.0)
        return FaultResult(
            name="reverse_spine_angle",
            display_name="Reverse Spine Angle (Ascent)",
            description=f"Spine tilted {tilt_change:.1f}° toward target during backswing. "
                        "Elite golfers maintain or slightly increase their away-from-target tilt through the backswing. "
                        "Tilting toward the target (reverse spine angle) is a primary cause of "
                        "over-the-top swing paths and early extension on the downswing.",
            root_cause="Lateral sway of the upper body toward the target, or active resistance of the shoulder turn "
                       "causing the left shoulder to drop instead of rotate.",
            ball_flight="Over-the-top pull, slices, steep angle of attack, back pain risk",
            severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
            measured_value=tilt_change,
            elite_benchmark="Spine tilt should remain stable or increase away from target in backswing",
            source="Swing Lab Theory; TPI; Berman Golf biomechanics",
        )

    # Excessive sway away from target
    if tilt_change < -15.0:
        severity = min(1.0, (abs(tilt_change) - 15.0) / 20.0)
        return FaultResult(
            name="excessive_lateral_tilt",
            display_name="Excessive Lateral Tilt Away from Target (Ascent)",
            description=f"Spine tilted {abs(tilt_change):.1f}° away from target during backswing — beyond normal range. "
                        "While some away-from-target tilt is correct, excessive lateral bend destabilizes "
                        "the spine and makes consistent return to impact difficult.",
            root_cause="Lateral sway toward trail side (hip sway), or over-emphasizing 'staying behind the ball'.",
            ball_flight="Fat shots, blocked shots, inconsistent contact",
            severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
            measured_value=tilt_change,
            elite_benchmark="<15° away-from-target tilt change during backswing",
            source="Biomechanical coaching consensus; Swing Align research",
        )

    return None


# ---------------------------------------------------------------------------
# DESCENT MECHANICS — how each metric evolves during the downswing
# ---------------------------------------------------------------------------

def check_descent_hip_leads_shoulder(metrics, adjustments=None):
    """
    Descent: hips must begin rotating back BEFORE shoulders — the kinematic
    sequence. We check the RATE of change: hip rotation velocity should
    peak before shoulder rotation velocity in the downswing.
    More nuanced than just checking reversal frame.
    [Meister 2011: confirmed in 100% of pros]
    """
    a = _arrays(metrics)
    shoulder = _smooth(a["shoulder"], w=3)
    hip = _smooth(a["hip"], w=3)
    _, top = _find_backswing_window(a)
    n = a["n"]

    if top >= n - 3:
        return None

    downswing = slice(top, min(n, top + int((n - top) * 0.7)))
    s_ds = shoulder[downswing]
    h_ds = hip[downswing]

    if len(s_ds) < 4:
        return None

    # Velocity (frame-to-frame change) during downswing
    s_vel = np.diff(s_ds)
    h_vel = np.diff(h_ds)

    # For correct sequence: hip velocity should go negative (unwinding) first
    # Find first frame where each starts clearly declining
    def first_decline_frame(vel, threshold=-0.5):
        for i, v in enumerate(vel):
            if v < threshold:
                return i
        return None

    h_decline = first_decline_frame(h_vel)
    s_decline = first_decline_frame(s_vel)

    if h_decline is None or s_decline is None:
        return None

    lag = s_decline - h_decline  # positive = hip declines first (correct)

    if lag >= 0:
        return None

    # Shoulders declining before hips
    severity = min(1.0, abs(lag) / max(3, int(len(s_ds) * 0.25)))
    return FaultResult(
        name="descent_sequence_fault",
        display_name="Incorrect Descent Sequence (Shoulders Before Hips)",
        description=f"Shoulders begin unwinding {abs(lag)} frame(s) before hips in the downswing. "
                    "Meister et al. (2011) confirmed in ALL professional golfers that the downswing "
                    "is initiated by pelvic rotation reversal, followed by upper torso. "
                    "Shoulder-first descent sends the club over the target line.",
        root_cause="Initiating the downswing with the upper body — 'hitting from the top'. "
                   "Often caused by anxiety, poor hip mobility, or trying to generate power with the arms.",
        ball_flight="Over-the-top pull, pull-slice, steep angle of attack",
        severity=severity, severity_label=_label(severity), phase="downswing (descent)",
        measured_value=float(lag),
        elite_benchmark="Hip unwind must precede shoulder unwind",
        source="Meister et al. 2011 J. Applied Biomechanics",
    )


def check_descent_weight_acceleration(metrics, adjustments=None):
    """
    Descent: weight transfer to lead side should accelerate through
    the downswing — not stall or decelerate before impact.
    We check that the rate of weight transfer is positive and increasing.
    """
    a = _arrays(metrics)
    weight = _smooth(a["weight"], w=3)
    _, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    if impact <= top + 2:
        return None

    downswing_w = weight[top:impact+1]
    if len(downswing_w) < 4:
        return None

    # Check weight at key points through descent
    quarter = len(downswing_w) // 4
    w_start = float(downswing_w[0])
    w_mid = float(np.mean(downswing_w[quarter:quarter*2]))
    w_end = float(np.mean(downswing_w[-quarter:]))

    # Weight should be moving lead-ward throughout (w_end > w_mid > w_start)
    # Fault: weight stalls or reverses mid-descent
    stall = w_mid - w_start  # should be positive
    completion = w_end - w_mid  # should also be positive

    if stall >= 0 and completion >= 0:
        return None

    if stall < 0:
        # Weight actually going BACK to trail during descent
        severity = min(1.0, abs(stall) / 0.4)
        return FaultResult(
            name="descent_weight_reversal",
            display_name="Weight Reversal in Descent",
            description=f"Weight moved {abs(stall):.2f} back toward trail side during the downswing. "
                        "Elite golfers continuously transfer weight toward the lead side from top to impact. "
                        "Any reversal during descent breaks the kinetic chain and causes a casting motion.",
            root_cause="Hanging back — trying to 'stay behind the ball' too aggressively, "
                       "or the upper body firing before the lower body has cleared.",
            ball_flight="Fat shots, blocks, loss of distance",
            severity=severity, severity_label=_label(severity), phase="downswing (descent)",
            measured_value=stall,
            elite_benchmark="Weight should continuously increase toward lead side through descent",
            source="Richards et al. 1985; Fleisig Biomechanics of Golf",
        )

    if completion < -0.1:
        # Weight transfer stalls in second half of descent
        severity = min(0.7, abs(completion) / 0.3)
        return FaultResult(
            name="descent_weight_stall",
            display_name="Weight Transfer Stalls Late in Descent",
            description=f"Weight transfer rate drops off {abs(completion):.2f} in the second half of the downswing. "
                        "Weight should accelerate continuously to the lead side through impact — "
                        "any late stall means the hips have stopped clearing and the arms are taking over.",
            root_cause="Hips stopping rotation too early, or the upper body and arms taking over before impact.",
            ball_flight="Flipping at impact, hooks, loss of compression",
            severity=severity, severity_label=_label(severity), phase="downswing (descent)",
            measured_value=completion,
            elite_benchmark="Weight transfer rate should be consistent through full descent",
            source="Richards et al. 1985",
        )

    return None


def check_descent_spine_maintenance(metrics, adjustments=None):
    """
    Descent: spine angle should be maintained from top through impact.
    Any significant loss (early extension) = hips thrusting toward ball.
    This is the descent-specific version — checks the downswing window only.
    [TPI: most common fault in high-handicap amateurs]
    """
    a = _arrays(metrics)
    spine = _smooth(a["spine"], w=3)
    _, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    top_tilt = float(np.mean(spine[max(0, top-1):top+2]))

    win_s = max(top, impact - 2)
    win_e = min(n, impact + 3)
    impact_tilt = float(np.mean(spine[win_s:win_e]))

    tilt_change = abs(impact_tilt) - abs(top_tilt)
    adj = adjustments.get("spine", 1.0) if adjustments else 1.0
    threshold = 7.0 * adj

    if tilt_change <= threshold:
        return None

    severity = min(1.0, (tilt_change - threshold) / 15.0)
    return FaultResult(
        name="descent_early_extension",
        display_name="Early Extension in Descent (Loss of Posture)",
        description=f"Spine angle changed {tilt_change:.1f}° from top of backswing to impact during the descent. "
                    "TPI research identifies early extension as one of the most common amateur faults — "
                    "the hips thrust toward the ball, the spine straightens, and the club is rerouted. "
                    "Elite golfers maintain their spine angle from top through impact.",
        root_cause="Limited hip mobility preventing rotation without extension. "
                   "Weak core stability, or the body compensating for an over-the-top path.",
        ball_flight="Thin shots, blocks right, hooks from compensatory flip at impact",
        severity=severity, severity_label=_label(severity), phase="downswing (descent)",
        measured_value=tilt_change,
        elite_benchmark=f"Spine angle change should be <{threshold:.0f}° from top to impact",
        source="TPI research; Swing Lab Theory biomechanics",
    )


def check_descent_elbow_lag(metrics, adjustments=None):
    """
    Descent: lead elbow should remain extended (straight) through the
    hitting zone. Collapse = chicken wing = club face opens.
    Check the minimum lead elbow angle during the descent window.
    """
    a = _arrays(metrics)
    elbow = _smooth(a["elbow"], w=3)
    _, top = _find_backswing_window(a)
    impact = _find_impact_idx(a, top)
    n = a["n"]

    win_s = max(top, impact - 4)
    win_e = min(n, impact + 5)
    if win_e <= win_s:
        return None

    min_angle = float(np.min(elbow[win_s:win_e]))
    if min_angle < 10:
        return None

    elite_target = 155.0
    adj = adjustments.get("spine", 1.0) if adjustments else 1.0
    effective = elite_target * (adj * 0.97)

    if min_angle >= effective * 0.92:
        return None

    severity = min(1.0, (effective - min_angle) / 55.0)
    return FaultResult(
        name="descent_chicken_wing",
        display_name="Lead Arm Collapse in Descent (Chicken Wing)",
        description=f"Lead elbow angle drops to {min_angle:.1f}° during the descent hitting zone (elite: >150°). "
                    "The lead arm should remain extended through impact to maintain swing width and "
                    "control the club face. Collapse indicates the trail arm is dominating or the "
                    "golfer is flipping the wrists.",
        root_cause="Trail arm pushing rather than pulling through impact. "
                   "Flipping/scooping to try to help the ball into the air. "
                   "TPI: 35% chance of chicken wing with limited wrist mobility.",
        ball_flight="Weak slices, pulls, reduced distance, inconsistent contact",
        severity=severity, severity_label=_label(severity), phase="downswing (descent)",
        measured_value=min_angle,
        elite_benchmark=">150° lead elbow angle through impact",
        source="TPI research; HackMotion data; TrackMan University",
    )


# Master fault check list — tempo removed, ascent/descent mechanics added
FAULT_CHECKS = [
    # Address
    check_spine_tilt_at_address,

    # Ascent (backswing) mechanics
    check_shoulder_turn,
    check_hip_turn_at_top,
    check_x_factor,
    check_ascent_rotation_rate,
    check_ascent_weight_direction,
    check_ascent_spine_stability,
    check_weight_loading_backswing,
    check_hip_sway,
    check_lead_knee_flex,
    check_wrist_hinge_at_top,

    # Descent (downswing) mechanics
    check_kinematic_sequence,
    check_descent_hip_leads_shoulder,
    check_descent_weight_acceleration,
    check_descent_spine_maintenance,
    check_descent_elbow_lag,
    check_weight_transfer_impact,
    check_early_extension,
    check_head_stability,

    # Follow through
    check_follow_through_completion,
]
