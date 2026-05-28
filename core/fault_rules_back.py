"""
Back view fault rules — research-grounded, elite benchmark comparison.

BACK VIEW REVEALS:
  - Spine angle at address (forward bend) — must be 30-45° [TPI, Cochran]
  - Spine angle maintenance through swing — biggest DTL fault indicator
  - Lead arm plane at top — above/on/below shoulder plane [BioSwing Dynamics]
  - Knee flex maintenance — especially trail knee [Murakami 2022]
  - Over-the-top path — shoulder plane steepening in downswing
  - Hip slide vs turn — lateral movement toward target visible in profile
  - Forward shaft lean at impact — hands ahead of ball [TrackMan research]
  - Head movement forward/backward — "forward press" or "hanging back"

ELITE BENCHMARKS (DTL):
  [TPI/Cochran]  Spine angle at address: 30-45° forward bend
  [Murakami 2022] Trail knee flex: 17° at address, 24° at top, 22° at impact
                  Lead knee flex: 18° at address, 33° at top, 25° at impact
                  (our angles measure joint angle, so 180-x = flex amount)
  [BioSwing]     Lead arm at top: on or above shoulder plane = "on plane"
  [Swing Lab]    Spine angle change address→impact: <8° (same as face-on)
  [TrackMan Uni] Forward shaft lean at impact: hands 4-8° ahead of ball line
  [Fleisig]      Hip slide toward target at impact: 2-4 inches (50-100px typical)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .metrics_back import BackMetrics


@dataclass
class BackFaultResult:
    name: str
    display_name: str
    description: str
    ball_flight: str
    root_cause: str
    severity: float
    severity_label: str
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
    if s < 0.33: return "mild"
    elif s < 0.66: return "moderate"
    return "severe"


def _smooth(arr: np.ndarray, w: int = 3) -> np.ndarray:
    if w < 2 or len(arr) < w:
        return arr.copy()
    return np.convolve(arr, np.ones(w)/w, mode="same")


def _arrays(metrics: List[BackMetrics]) -> dict:
    return {
        "spine":       np.array([m.spine_angle_deg or 0.0 for m in metrics]),
        "lead_arm":    np.array([m.lead_arm_angle_deg or 0.0 for m in metrics]),
        "trail_arm":   np.array([m.trail_arm_angle_deg or 0.0 for m in metrics]),
        "lead_knee":   np.array([m.lead_knee_flex_deg or 0.0 for m in metrics]),
        "trail_knee":  np.array([m.trail_knee_flex_deg or 0.0 for m in metrics]),
        "head_fwd":    np.array([m.head_forward_px or 0.0 for m in metrics]),
        "head_move":   np.array([m.head_movement_px or 0.0 for m in metrics]),
        "hip_slide":   np.array([m.hip_slide_px or 0.0 for m in metrics]),
        "shld_plane":  np.array([m.shoulder_plane_deg or 0.0 for m in metrics]),
        "hand_height": np.array([m.hand_height_y or 0.0 for m in metrics]),
        "wrist":       np.array([m.wrist_hinge_deg or 0.0 for m in metrics]),
        "conf":        np.array([m.confidence for m in metrics]),
        "n":           len(metrics),
    }


def _find_phases(a: dict) -> Tuple[int, int, int]:
    """Find address, top, impact from hand height (same logic as face-on)."""
    n = a["n"]
    hand_h = a["hand_height"]
    conf   = a["conf"]

    # Address: first stable window in first 30%
    cutoff = max(3, n // 3)
    win    = max(2, n // 20)
    addr   = 0
    best_v = float("inf")
    spine  = _smooth(a["spine"], w=3)
    for i in range(cutoff - win):
        v = float(np.var(spine[i:i+win]))
        c = float(np.mean(conf[i:i+win]))
        if v < best_v and c > 0.5:
            best_v = v
            addr   = i + win // 2

    # Top: minimum hand y (highest hands) between addr and 85% through
    search_s = addr + 1
    search_e = min(n, addr + int((n - addr) * 0.85))
    if np.any(hand_h[search_s:search_e] > 0):
        region  = hand_h[search_s:search_e]
        c_reg   = conf[search_s:search_e]
        w_region= np.where(c_reg > 0.4, region, np.max(region) + 1)
        top     = search_s + int(np.argmin(w_region))
    else:
        top = search_s + (search_e - search_s) // 2

    # Impact: hands return to near-address height
    if np.any(hand_h > 0):
        addr_h = float(np.mean(hand_h[:max(1, top//4)]))
        top_h  = float(hand_h[top])
        arc    = addr_h - top_h
        if arc > 5:
            thresh = top_h + arc * 0.75
            for i in range(top + 1, n):
                if hand_h[i] >= thresh and conf[i] > 0.4:
                    return addr, top, i

    impact = top + max(1, int((n - top) * 0.60))
    return addr, top, impact


# ---------------------------------------------------------------------------
# FAULT CHECKS — DTL specific
# ---------------------------------------------------------------------------

def check_back_spine_angle_address(metrics, adjustments=None):
    """
    Elite: 30-45° forward bend at address [TPI, Cochran/Faeherty].
    Too upright = loss of athletic posture.
    Too bent = restricts rotation, back pain risk.
    """
    a = _arrays(metrics)
    spine = _smooth(a["spine"], w=3)
    addr, _, _ = _find_phases(a)

    addr_spine = float(np.mean(spine[max(0, addr-2):addr+3]))
    elite_min, elite_max = 28.0, 48.0

    if addr_spine < elite_min:
        severity = min(1.0, (elite_min - addr_spine) / elite_min)
        return BackFaultResult(
            name="back_upright_address",
            display_name="Too Upright at Address",
            description=f"Spine angle at address is {addr_spine:.1f}° — too upright. "
                        "TPI and Cochran research show elite golfers bend forward 30-45° at address. "
                        "Standing too upright restricts hip and shoulder rotation.",
            root_cause="Standing too close to the ball, or not hinging forward from the hips.",
            ball_flight="Flat swing plane, pulls, loss of power",
            severity=severity, severity_label=_label(severity), phase="address",
            measured_value=addr_spine, elite_benchmark="30-45° forward bend at address",
            source="TPI Biomechanics; Cochran & Faeherty",
        )

    if addr_spine > elite_max:
        severity = min(1.0, (addr_spine - elite_max) / 25.0)
        return BackFaultResult(
            name="back_hunched_address",
            display_name="Too Much Forward Bend at Address",
            description=f"Spine angle at address is {addr_spine:.1f}° — excessive forward bend. "
                        "More than 45° of forward bend restricts rotation and increases injury risk.",
            root_cause="Standing too far from ball, rounded upper back, or collapsing through setup.",
            ball_flight="Steep swing, chops, back strain over time",
            severity=severity, severity_label=_label(severity), phase="address",
            measured_value=addr_spine, elite_benchmark="30-45° forward bend",
            source="TPI Biomechanics; Cochran & Faeherty",
        )
    return None


def check_back_spine_angle_maintenance(metrics, adjustments=None):
    """
    Elite: spine angle from address maintained through impact — <8° change.
    Loss of posture (standing up) or reverse pivot (tilting forward) both flagged.
    [TPI, Swing Lab Theory]
    """
    a = _arrays(metrics)
    spine = _smooth(a["spine"], w=3)
    addr, top, impact = _find_phases(a)

    addr_spine   = float(np.mean(spine[max(0,addr-1):addr+2]))
    impact_spine = float(np.mean(spine[max(0,impact-2):min(a["n"],impact+3)]))
    change = abs(impact_spine - addr_spine)

    if change <= 8.0:
        return None

    if impact_spine < addr_spine:
        desc = f"Spine angle INCREASED {change:.1f}° from address to impact — standing up (early extension). "
        fault_name = "back_early_extension"
        display = "Early Extension — Standing Up Through Impact"
    else:
        desc = f"Spine angle DECREASED {change:.1f}° from address to impact — losing forward bend. "
        fault_name = "back_spine_loss"
        display = "Loss of Spine Angle Through Impact"

    severity = min(1.0, (change - 8.0) / 20.0)
    return BackFaultResult(
        name=fault_name,
        display_name=display,
        description=desc + "Elite golfers maintain their spine angle from address through impact. "
                    "TPI research identifies loss of posture as one of the most common amateur faults.",
        root_cause="Limited hip mobility forcing the body to extend. Weak core. "
                   "Trying to 'help' the ball into the air by flipping or standing up.",
        ball_flight="Thin shots, blocks, hooks from compensatory flip at impact",
        severity=severity, severity_label=_label(severity), phase="downswing (descent)",
        measured_value=change, elite_benchmark="<8° spine angle change address→impact",
        source="TPI research; Swing Lab Theory",
    )


def check_back_lead_arm_plane(metrics, adjustments=None):
    """
    At top of backswing, lead arm should be on or above the shoulder plane.
    Below = laid off / too flat. Way above = too steep / across the line.
    [BioSwing Dynamics — Adams/Tischler]
    """
    a = _arrays(metrics)
    lead_arm = _smooth(a["lead_arm"], w=3)
    shld_plane = _smooth(a["shld_plane"], w=3)
    _, top, _ = _find_phases(a)
    n = a["n"]

    arm_at_top   = float(np.mean(lead_arm[max(0,top-2):min(n,top+3)]))
    shld_at_top  = float(np.mean(shld_plane[max(0,top-2):min(n,top+3)]))

    if arm_at_top == 0 or shld_at_top == 0:
        return None

    diff = arm_at_top - shld_at_top  # positive = arm above shoulder plane (good)

    # Laid off: arm significantly BELOW shoulder plane
    if diff < -15:
        severity = min(1.0, abs(diff + 15) / 25.0)
        return BackFaultResult(
            name="back_laid_off",
            display_name="Laid Off at Top (Arm Below Plane)",
            description=f"Lead arm is {abs(diff):.1f}° below the shoulder plane at the top of backswing. "
                        "BioSwing Dynamics research shows the lead arm should be on or above the shoulder plane. "
                        "Being 'laid off' promotes an inside-out swing path and blocked/hooked shots.",
            root_cause="Over-rotation of the forearms in the takeaway, or collapsing lead arm.",
            ball_flight="Blocks right, hooks, push-draws that become unpredictable",
            severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
            measured_value=diff, elite_benchmark="Lead arm on or above shoulder plane at top",
            source="BioSwing Dynamics — Adams & Tischler; GolfWRX biomechanics",
        )

    # Across the line: arm significantly ABOVE shoulder plane
    if diff > 25:
        severity = min(1.0, (diff - 25) / 30.0)
        return BackFaultResult(
            name="back_across_the_line",
            display_name="Across the Line at Top (Too Steep)",
            description=f"Lead arm is {diff:.1f}° above shoulder plane at top — club pointing right of target. "
                        "An 'across the line' position promotes an over-the-top downswing path.",
            root_cause="Overactive hands in the takeaway lifting the club rather than turning.",
            ball_flight="Over-the-top pull, pull-slice, steep angle of attack",
            severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
            measured_value=diff, elite_benchmark="Lead arm within 25° above shoulder plane",
            source="BioSwing Dynamics — Adams & Tischler",
        )
    return None


def check_back_trail_knee_flex(metrics, adjustments=None):
    """
    Elite trail knee flex: ~17° at address, increases to 24° at top [Murakami 2022].
    In our angle convention (180=straight): address ~163°, should NOT straighten at top.
    Straightening = loss of power and stability.
    """
    a = _arrays(metrics)
    trail_knee = _smooth(a["trail_knee"], w=3)
    addr, top, _ = _find_phases(a)
    n = a["n"]

    addr_flex = float(np.mean(trail_knee[max(0,addr-1):addr+3]))
    top_flex  = float(np.mean(trail_knee[max(0,top-2):min(n,top+3)]))

    if addr_flex < 10 or top_flex < 10:
        return None

    # Trail knee should FLEX more (angle decrease) or stay same at top
    change = top_flex - addr_flex  # positive = straightening = bad
    if change <= 8:
        return None

    severity = min(1.0, change / 30.0)
    return BackFaultResult(
        name="back_trail_knee_straighten",
        display_name="Trail Knee Straightening in Backswing",
        description=f"Trail knee straightened {change:.1f}° during the backswing. "
                    "Murakami et al. (MDPI 2022 systematic review) shows elite golfers maintain or "
                    "increase trail knee flex from address to the top. "
                    "Straightening the trail knee causes a lateral sway (reverse pivot).",
        root_cause="Insufficient hip internal rotation mobility on the trail side, "
                   "or actively pushing weight to the lead foot on the backswing.",
        ball_flight="Reverse pivot, fat shots, loss of coil and power",
        severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
        measured_value=change, elite_benchmark="Trail knee should maintain or increase flex into backswing",
        source="Murakami et al. via MDPI Sports 2022 systematic review",
    )


def check_back_lead_knee_flex(metrics, adjustments=None):
    """
    Elite lead knee flex at top: ~33° [Murakami 2022].
    In our convention: ~147° angle. Straightening = loss of athletic position.
    """
    a = _arrays(metrics)
    lead_knee = _smooth(a["lead_knee"], w=3)
    addr, top, _ = _find_phases(a)
    n = a["n"]

    addr_flex = float(np.mean(lead_knee[max(0,addr-1):addr+3]))
    top_flex  = float(np.mean(lead_knee[max(0,top-2):min(n,top+3)]))

    if addr_flex < 10 or top_flex < 10:
        return None

    change = top_flex - addr_flex  # positive = straightening
    if change <= 8:
        return None

    severity = min(1.0, change / 28.0)
    return BackFaultResult(
        name="back_lead_knee_straighten",
        display_name="Lead Knee Straightening in Backswing",
        description=f"Lead knee straightened {change:.1f}° during the backswing. "
                    "Research shows the lead knee should flex (bend more) from 18° at address "
                    "to 33° at the top to maintain a stable, athletic base.",
        root_cause="Weight shifting toward the lead side (reverse pivot), or "
                   "insufficient hip mobility causing the leg to brace straight.",
        ball_flight="Unstable base, inconsistent contact, loss of power",
        severity=severity, severity_label=_label(severity), phase="backswing (ascent)",
        measured_value=change, elite_benchmark="Lead knee should flex (not straighten) into backswing",
        source="Murakami et al. via MDPI Sports 2022",
    )


def check_back_shoulder_plane_descent(metrics, adjustments=None):
    """
    Over the top: shoulder plane STEEPENS on the downswing compared to backswing.
    If shoulder plane angle increases (more vertical) from top to impact window,
    the club is attacking on an outside-in path.
    [Meister 2011; BioSwing Dynamics]
    """
    a = _arrays(metrics)
    shld = _smooth(a["shld_plane"], w=3)
    _, top, impact = _find_phases(a)
    n = a["n"]

    top_plane    = float(np.mean(shld[max(0,top-2):min(n,top+3)]))
    descent_plane= float(np.mean(shld[top:min(n,top+max(2,int((impact-top)*0.4)))]))

    if top_plane == 0:
        return None

    steepening = descent_plane - top_plane  # positive = getting steeper = OTT

    if steepening <= 5:
        return None

    severity = min(1.0, steepening / 20.0)
    return BackFaultResult(
        name="back_over_the_top",
        display_name="Over the Top (Shoulder Plane Steepens in Descent)",
        description=f"Shoulder plane steepened {steepening:.1f}° from top into the downswing. "
                    "When the shoulders lead the downswing and the plane gets steeper, "
                    "the club approaches the ball from outside the target line. "
                    "Meister et al. (2011) confirmed elite golfers shallow the plane in the downswing.",
        root_cause="Initiating the downswing with the upper body instead of the lower body. "
                   "Casting from the top.",
        ball_flight="Pull, pull-slice, steep divots pointing left of target",
        severity=severity, severity_label=_label(severity), phase="downswing (descent)",
        measured_value=steepening, elite_benchmark="Shoulder plane should shallow or maintain in descent",
        source="Meister et al. 2011; BioSwing Dynamics",
    )


def check_back_head_movement(metrics, adjustments=None):
    """
    Head should stay relatively stable in DTL view (minimal forward/backward movement).
    Forward head movement = lunging. Backward = hanging back.
    """
    a = _arrays(metrics)
    head_fwd = _smooth(a["head_fwd"], w=3)
    head_move= a["head_move"]
    _, top, impact = _find_phases(a)

    max_move = float(np.max(np.abs(head_move[:impact+1]))) if impact > 0 else 0.0
    threshold = 45.0

    if max_move <= threshold:
        return None

    severity = min(1.0, (max_move - threshold) / 60.0)
    return BackFaultResult(
        name="back_head_movement",
        display_name="Excessive Head Movement (DTL)",
        description=f"Head moved {max_move:.0f}px from address through impact. "
                    "In the down-the-line view, forward head movement indicates 'lunging' toward the ball. "
                    "Backward movement indicates hanging back. Both disrupt the swing center.",
        root_cause="Lateral sway, early extension, or loss of balance through the swing.",
        ball_flight="Inconsistent contact, fat/thin shots, directional inconsistency",
        severity=severity, severity_label=_label(severity), phase="backswing",
        measured_value=max_move, elite_benchmark=f"<{threshold:.0f}px head movement",
        source="General biomechanics consensus; TPI research",
    )


def check_back_hip_slide(metrics, adjustments=None):
    """
    Some hip slide toward target on downswing is correct (2-4 inches / ~50-100px).
    Excessive slide = no hip rotation, blocks shots.
    No slide = spinning hips, inconsistent contact.
    """
    a = _arrays(metrics)
    hip_slide = _smooth(a["hip_slide"], w=3)
    _, top, impact = _find_phases(a)
    n = a["n"]

    if impact <= top:
        return None

    # Max slide during downswing
    ds_slide = hip_slide[top:min(n, impact+3)]
    if len(ds_slide) == 0:
        return None

    max_slide = float(np.max(np.abs(ds_slide)))

    # Too much slide (>150px — pure lateral, no rotation)
    if max_slide > 150:
        severity = min(1.0, (max_slide - 150) / 100.0)
        return BackFaultResult(
            name="back_excessive_hip_slide",
            display_name="Excessive Hip Slide Toward Target",
            description=f"Hips slid {max_slide:.0f}px laterally toward target in downswing (too much). "
                        "Elite golfers slide 2-4 inches then rotate. Excessive slide means the hips "
                        "stop rotating and the upper body has to compensate.",
            root_cause="Initiating the downswing with a lateral bump instead of rotational sequence.",
            ball_flight="Blocks, pushes, loss of power, inability to square the face",
            severity=severity, severity_label=_label(severity), phase="downswing (descent)",
            measured_value=max_slide, elite_benchmark="50-100px lateral slide then rotate",
            source="Fleisig Biomechanics of Golf; TPI",
        )
    return None


BACK_FAULT_CHECKS = [
    check_back_spine_angle_address,
    check_back_spine_angle_maintenance,
    check_back_lead_arm_plane,
    check_back_trail_knee_flex,
    check_back_lead_knee_flex,
    check_back_shoulder_plane_descent,
    check_back_head_movement,
    check_back_hip_slide,
]
