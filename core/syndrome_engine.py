"""
Syndrome-based swing fault detection engine.

DESIGN PHILOSOPHY:
==================
1. SYNDROMES NOT METRICS
   Every fault requires 2-3 independent signals to agree.
   A single metric out of range = noise. Multiple signals = real fault.

2. CONFIDENCE GATING
   Signals from low-confidence frames (<70%) are excluded entirely.
   If a phase has insufficient high-confidence frames, the syndrome
   is marked UNDETECTABLE rather than guessing.

3. RELIABILITY SCORE
   Each detected syndrome gets a reliability score (0-1) based on:
   - How many signals agreed (more = higher)
   - Average confidence during the relevant phase
   - Magnitude of deviation from normal
   Only syndromes with reliability >= 0.50 are reported.

4. HONEST ABOUT LIMITATIONS
   Rotation metrics from a single 2D camera are unreliable.
   We prefer displacement-based signals (pixel movement of joints)
   over angle-based signals where possible, since displacement is
   directly observable and camera-angle-independent.

5. PHASE AWARENESS
   Each syndrome specifies which phase of the swing it applies to.
   Signals are only evaluated within the correct phase window.

SYNDROMES IMPLEMENTED (TPI-based):
===================================
Each fault below maps to the TPI fault matrix and is registered in
FACE_SYNDROMES or BACK_SYNDROMES (or both) at the bottom of this file.

Face view (FACE_SYNDROMES):
  SWAY            - hip slides trail during backswing
  SLIDE           - excessive hip slide toward target in downswing
  HANGING_BACK    - weight fails to transfer to lead side
  REVERSE_PIVOT   - weight moves lead side during backswing
  EARLY_EXTENSION - hips/spine extend toward ball in downswing
  OVER_THE_TOP    - shoulders initiate downswing before hips
  CASTING         - wrist hinge lost early in downswing
  CHICKEN_WING    - lead elbow collapses through impact
  FLYING_ELBOW    - trail elbow excessively elevated at top
  LOSS_OF_POSTURE - significant spine angle change during swing
  REVERSE_SPINE   - spine tilts toward target in backswing
  FORWARD_LUNGE   - upper body lunges toward target in transition
  LATE_BUCKLE     - lead knee buckles after impact

Back view (BACK_SYNDROMES):
  BACK_C_POSTURE       - rounded upper back at address
  BACK_LOSS_OF_POSTURE - spine angle changes from address through impact
  BACK_FLAT_SHOULDER   - shoulder plane too horizontal at top
  BACK_FLYING_ELBOW    - trail elbow elevated at top
  BACK_EARLY_EXTENSION - hips thrust toward ball (DTL confirmation)
  BACK_OVER_THE_TOP    - shoulder plane steepens in downswing
  BACK_SWAY            - excessive lateral hip slide away from target
  BACK_SLIDE           - excessive lateral hip slide toward target
  BACK_HANGING_BACK    - hips stay back; lead knee fails to post up
  BACK_REVERSE_PIVOT   - spine tilts toward target / hips drift forward
  BACK_REVERSE_SPINE   - spine tilts toward target in backswing
  BACK_CASTING         - wrist hinge / lead arm released early
  BACK_CHICKEN_WING    - lead arm breakdown through impact
  BACK_FORWARD_LUNGE   - head moves forward of hips in transition
  BACK_LATE_BUCKLE     - lead knee buckles post-impact

Not implemented (per TPI matrix, low/no measurability from a single 2D camera):
  S-POSTURE  - requires 3D lumbar curve measurement
  C-POSTURE from face view - not measurable without DTL angle

Sources:
  TPI Swing Characteristics: mytpi.com/improve-my-game/swing-characteristics
  Meister et al. 2011 J. Applied Biomechanics
  Fleisig Biomechanics of Golf
  TPI/Gulgin study: Correlation of TPI Level 1 screens and swing faults
  Swing Lab Theory biomechanics
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Tuple
from enum import Enum
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Confidence(Enum):
    HIGH   = "high"       # reliability >= 0.75
    MEDIUM = "medium"     # reliability 0.50-0.74
    LOW    = "low"        # reliability < 0.50 (not reported)


@dataclass
class Signal:
    """One piece of evidence for or against a syndrome."""
    name: str                    # what this signal is
    triggered: bool              # did this signal fire?
    value: Optional[float]       # measured value
    threshold: Optional[float]   # threshold that was checked
    confidence: float            # 0-1, how reliable was this measurement
    description: str             # plain language description of what fired


@dataclass
class SyndromeResult:
    """Result of evaluating one syndrome."""
    name: str
    display_name: str
    detected: bool
    reliability: float           # 0-1
    confidence_level: Confidence
    phase: str
    signals: List[Signal]        # all signals evaluated
    triggered_signals: List[Signal]  # signals that fired
    description: str             # plain language explanation
    ball_flight: str
    root_cause: str
    source: str

    # For the fault report
    severity: float              # same as reliability for compatibility
    severity_label: str
    measured_value: Optional[float] = None
    elite_benchmark: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fault": self.name,
            "display_name": self.display_name,
            "detected": self.detected,
            "reliability": round(self.reliability, 2),
            "confidence_level": self.confidence_level.value,
            "signals_triggered": len(self.triggered_signals),
            "signals_checked": len(self.signals),
            "phase": self.phase,
            "description": self.description,
            "root_cause": self.root_cause,
            "ball_flight_effect": self.ball_flight,
            "severity": round(self.severity, 2),
            "severity_label": self.severity_label,
        }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class SyndromeEngine:
    """
    Evaluates swing syndromes from metric time series.

    Usage:
        engine = SyndromeEngine(metrics, view="face")
        results = engine.run_all()
        detected = [r for r in results if r.detected]
    """

    # Minimum number of signals that must agree for detection
    MIN_SIGNALS_REQUIRED = 2
    # Minimum reliability to report a syndrome
    MIN_RELIABILITY = 0.50
    # Minimum MediaPipe confidence to use a frame's data
    MIN_FRAME_CONFIDENCE = 0.65

    def __init__(self, metrics: list, view: str = "face"):
        """
        Parameters
        ----------
        metrics : list of SwingMetrics or BackMetrics
        view    : "face" or "back"
        """
        self.metrics = metrics
        self.view = view
        self.n = len(metrics)
        self._phases = self._detect_phases()

    def run_all(self) -> List[SyndromeResult]:
        """Run all applicable syndromes and return detected ones."""
        if self.view == "face":
            checks = FACE_SYNDROMES
        else:
            checks = BACK_SYNDROMES

        results = []
        for syndrome_fn in checks:
            try:
                result = syndrome_fn(self)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  Warning: {syndrome_fn.__name__} failed: {e}")

        # Sort by reliability descending
        results.sort(key=lambda r: r.reliability, reverse=True)
        return results

    def detected(self) -> List[SyndromeResult]:
        """Return only detected syndromes with sufficient reliability."""
        return [r for r in self.run_all() if r.detected]

    # ------------------------------------------------------------------
    # Phase detection — based on hand height (most reliable signal)
    # ------------------------------------------------------------------

    def _detect_phases(self) -> Dict[str, Tuple[int, int]]:
        """
        Returns frame index ranges for each phase:
          address:      (0, addr_end)
          backswing:    (addr_end, top)
          downswing:    (top, impact)
          follow_through:(impact, n-1)
        """
        n = self.n
        conf = np.array([m.confidence for m in self.metrics])

        # Hand height: lower y value = higher hands
        hand_y = np.array([
            getattr(m, 'hand_height_y', 0) or 0.0
            for m in self.metrics
        ])

        # Address: first stable window in first 30%
        cutoff = max(3, n // 3)
        win    = max(2, n // 20)
        addr   = 0
        best_v = float("inf")

        # Use weight shift stability for address
        weight = np.array([getattr(m, 'weight_shift', 0) or 0.0 for m in self.metrics])
        weight_s = self._smooth(weight, 3)

        for i in range(max(0, cutoff - win)):
            v = float(np.var(weight_s[i:i+win]))
            c = float(np.mean(conf[i:i+win]))
            if v < best_v and c > 0.5:
                best_v = v
                addr   = i + win // 2

        # Top: minimum hand y (highest hands) between addr and 85%
        s_start = addr + 1
        s_end   = min(n, addr + int((n - addr) * 0.85))
        top     = addr + 1

        if np.any(hand_y[s_start:s_end] > 0):
            region  = hand_y[s_start:s_end]
            c_reg   = conf[s_start:s_end]
            masked  = np.where(c_reg > 0.4, region, np.max(region) + 1)
            top     = s_start + int(np.argmin(masked))

        # Impact: hands return to ~address height
        impact = top + max(1, int((n - top) * 0.60))
        if np.any(hand_y > 0):
            addr_h  = float(np.mean(hand_y[:max(1, top//4)] + 1e-9))
            top_h   = float(hand_y[top])
            arc     = addr_h - top_h
            if arc > 5:
                thresh = top_h + arc * 0.75
                for i in range(top + 1, n):
                    if hand_y[i] >= thresh and conf[i] > 0.4:
                        impact = i
                        break

        return {
            "address":       (0,      max(0, addr)),
            "backswing":     (addr,   top),
            "downswing":     (top,    impact),
            "follow_through":(impact, n - 1),
            "addr_idx":      addr,
            "top_idx":       top,
            "impact_idx":    impact,
        }

    # ------------------------------------------------------------------
    # Helper methods for syndrome checks
    # ------------------------------------------------------------------

    def _smooth(self, arr: np.ndarray, w: int = 3) -> np.ndarray:
        if w < 2 or len(arr) < w:
            return arr.copy()
        return np.convolve(arr, np.ones(w)/w, mode="same")

    def _phase_metrics(self, phase: str, attr: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get smoothed values and confidence for a metric in a phase window.
        Returns (values, confidences) — only high-confidence frames.
        """
        start, end = self.phases_range(phase)
        vals  = np.array([getattr(m, attr, None) or 0.0 for m in self.metrics[start:end]])
        confs = np.array([m.confidence for m in self.metrics[start:end]])
        vals_smooth = self._smooth(vals, 3)
        return vals_smooth, confs

    def phases_range(self, phase: str) -> Tuple[int, int]:
        p = self._phases
        if phase == "address":       return p["address"]
        if phase == "backswing":     return p["backswing"]
        if phase == "downswing":     return p["downswing"]
        if phase == "follow_through":return p["follow_through"]
        return (0, self.n)

    def _get_attr_at(self, attr: str, frame_idx: int) -> Optional[float]:
        if 0 <= frame_idx < self.n:
            return getattr(self.metrics[frame_idx], attr, None)
        return None

    def _high_conf_mean(self, vals: np.ndarray, confs: np.ndarray) -> Optional[float]:
        """Mean of values where confidence is above threshold."""
        mask = confs >= self.MIN_FRAME_CONFIDENCE
        if mask.sum() < 2:
            return None
        return float(np.mean(vals[mask]))

    def _high_conf_max(self, vals: np.ndarray, confs: np.ndarray) -> Optional[float]:
        mask = confs >= self.MIN_FRAME_CONFIDENCE
        if mask.sum() < 2:
            return None
        return float(np.max(vals[mask]))

    def _high_conf_min(self, vals: np.ndarray, confs: np.ndarray) -> Optional[float]:
        mask = confs >= self.MIN_FRAME_CONFIDENCE
        if mask.sum() < 2:
            return None
        return float(np.min(vals[mask]))

    def _phase_conf(self, phase: str) -> float:
        """Average confidence during a phase."""
        start, end = self.phases_range(phase)
        if end <= start:
            return 0.0
        confs = [m.confidence for m in self.metrics[start:end]]
        return float(np.mean(confs)) if confs else 0.0

    def _build_result(
        self,
        name: str,
        display_name: str,
        phase: str,
        signals: List[Signal],
        description: str,
        ball_flight: str,
        root_cause: str,
        source: str,
        measured_value: Optional[float] = None,
        elite_benchmark: Optional[str] = None,
    ) -> SyndromeResult:
        """
        Build a SyndromeResult from a list of signals.
        Detection requires MIN_SIGNALS_REQUIRED to trigger.
        """
        triggered = [s for s in signals if s.triggered]
        n_triggered = len(triggered)
        n_total = len(signals)

        # Signal agreement ratio
        agreement = n_triggered / max(n_total, 1)

        # Average confidence of triggered signals
        if triggered:
            avg_conf = float(np.mean([s.confidence for s in triggered]))
        else:
            avg_conf = float(np.mean([s.confidence for s in signals])) if signals else 0.0

        # Reliability: signal agreement × confidence × magnitude
        if n_triggered < self.MIN_SIGNALS_REQUIRED:
            reliability = agreement * avg_conf * 0.5  # below detection threshold
            detected = False
        else:
            reliability = min(1.0, agreement * avg_conf * (1 + 0.2 * (n_triggered - self.MIN_SIGNALS_REQUIRED)))
            detected = reliability >= self.MIN_RELIABILITY

        # Confidence level
        if reliability >= 0.75:
            conf_level = Confidence.HIGH
        elif reliability >= 0.50:
            conf_level = Confidence.MEDIUM
        else:
            conf_level = Confidence.LOW

        # Severity label
        if reliability >= 0.75:
            sev_label = "severe"
        elif reliability >= 0.50:
            sev_label = "moderate"
        else:
            sev_label = "mild"

        return SyndromeResult(
            name=name,
            display_name=display_name,
            detected=detected,
            reliability=round(reliability, 3),
            confidence_level=conf_level,
            phase=phase,
            signals=signals,
            triggered_signals=triggered,
            description=description,
            ball_flight=ball_flight,
            root_cause=root_cause,
            source=source,
            severity=round(reliability, 3),
            severity_label=sev_label,
            measured_value=measured_value,
            elite_benchmark=elite_benchmark,
        )


# ---------------------------------------------------------------------------
# FACE VIEW SYNDROMES
# ---------------------------------------------------------------------------

def _face_sway(engine: SyndromeEngine) -> SyndromeResult:
    """
    SWAY: Excessive lower body lateral movement AWAY from target in backswing.
    TPI: "Any excessive lower body lateral movement away from the target during backswing."

    Signals:
      1. Hip sways trail (hip_sway_px moves significantly trail-ward)
      2. Weight does NOT load trail side properly (stays centered)
      3. Head moves significantly (head follows the sway)
    """
    bs_hip,  bs_conf  = engine._phase_metrics("backswing", "hip_sway_px")
    bs_wt,   _        = engine._phase_metrics("backswing", "weight_shift")
    bs_head, hd_conf  = engine._phase_metrics("backswing", "head_movement_px")

    max_hip_sway = engine._high_conf_max(np.abs(bs_hip), bs_conf)
    avg_weight   = engine._high_conf_mean(bs_wt, bs_conf)
    max_head     = engine._high_conf_max(bs_head, hd_conf)
    phase_conf   = engine._phase_conf("backswing")

    signals = []

    # Signal 1: Hip sways trail (positive = trail in our convention)
    sway_thresh = 28.0
    s1_val  = max_hip_sway
    s1_fire = s1_val is not None and s1_val > sway_thresh
    signals.append(Signal(
        name="hip_lateral_trail",
        triggered=s1_fire,
        value=s1_val,
        threshold=sway_thresh,
        confidence=phase_conf,
        description=f"Hip moved {s1_val:.0f}px laterally trail-ward (threshold: {sway_thresh}px)" if s1_val else "No data"
    ))

    # Signal 2: Weight NOT loading trail (should be negative at top, positive = sway fault)
    wt_thresh = 0.05  # weight should be negative (trail loaded), if > 0.05 = fault
    s2_val  = avg_weight
    s2_fire = s2_val is not None and s2_val > wt_thresh
    signals.append(Signal(
        name="weight_not_trail_loaded",
        triggered=s2_fire,
        value=s2_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight staying lead-side ({s2_val:.2f}) during backswing — not loading trail" if s2_val else "No data"
    ))

    # Signal 3: Head moving with the body (sway pulls head)
    head_thresh = 40.0
    s3_val  = max_head
    s3_fire = s3_val is not None and s3_val > head_thresh
    signals.append(Signal(
        name="head_moving_with_sway",
        triggered=s3_fire,
        value=s3_val,
        threshold=head_thresh,
        confidence=phase_conf,
        description=f"Head moved {s3_val:.0f}px — consistent with sway pattern" if s3_val else "No data"
    ))

    return engine._build_result(
        name="sway",
        display_name="Sway (Hip Slides Trail in Backswing)",
        phase="backswing",
        signals=signals,
        description="The lower body is sliding away from the target during the backswing instead of rotating. "
                    "This prevents proper weight loading onto the trail foot and destabilizes the downswing path.",
        ball_flight="Fat shots, steep downswing, loss of power, thin shots from reverse weight shift",
        root_cause="Limited trail hip internal rotation mobility forces lateral slide instead of rotation. "
                   "Also caused by weak glutes unable to resist the lateral force.",
        source="TPI Swing Characteristics; Back Nine PT biomechanics; Fleisig",
        measured_value=max_hip_sway,
        elite_benchmark="<28px lateral hip movement during backswing",
    )


def _face_slide(engine: SyndromeEngine) -> SyndromeResult:
    """
    SLIDE: Excessive lower body lateral movement TOWARD target in downswing.
    TPI: "Any excessive lateral movement forward of the lead hip toward the target during the downswing."
    Some slide is CORRECT (2-4 inches) — only flag excessive slide.

    Signals:
      1. Hip slides more than threshold toward target
      2. Hip rotation (shoulder-hip separation) remains LOW despite slide (hips not rotating)
      3. Weight transfer stalls — doesn't accelerate to lead side
    """
    ds_hip,  ds_conf = engine._phase_metrics("downswing", "hip_sway_px")
    ds_xf,   xf_conf = engine._phase_metrics("downswing", "x_factor_deg")
    ds_wt,   wt_conf = engine._phase_metrics("downswing", "weight_shift")

    max_slide  = engine._high_conf_max(np.abs(ds_hip), ds_conf)
    min_xf     = engine._high_conf_min(ds_xf, xf_conf)
    impact_wt  = engine._high_conf_max(ds_wt, wt_conf)
    phase_conf = engine._phase_conf("downswing")

    signals = []

    # Signal 1: Excessive lateral slide toward target
    slide_thresh = 80.0  # pixels — ~2-4 inches is normal, >4 inches is excessive
    s1_val  = max_slide
    s1_fire = s1_val is not None and s1_val > slide_thresh
    signals.append(Signal(
        name="excessive_hip_slide",
        triggered=s1_fire,
        value=s1_val,
        threshold=slide_thresh,
        confidence=phase_conf,
        description=f"Hip slid {s1_val:.0f}px toward target — exceeds normal 2-4 inch range" if s1_val else "No data"
    ))

    # Signal 2: Low X-factor during downswing (hips sliding not rotating)
    xf_thresh = 15.0
    s2_val  = min_xf
    s2_fire = s2_val is not None and s2_val < xf_thresh
    signals.append(Signal(
        name="low_xfactor_with_slide",
        triggered=s2_fire,
        value=s2_val,
        threshold=xf_thresh,
        confidence=phase_conf,
        description=f"X-Factor drops to {s2_val:.1f}° during downswing — hips sliding not rotating" if s2_val else "No data"
    ))

    # Signal 3: Weight transfer inconsistent (slide without rotation = unstable)
    wt_thresh = 0.25
    s3_val  = impact_wt
    s3_fire = s3_val is not None and s3_val < wt_thresh
    signals.append(Signal(
        name="incomplete_weight_transfer",
        triggered=s3_fire,
        value=s3_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight shift at impact {s3_val:.2f} — incomplete transfer despite lateral movement" if s3_val else "No data"
    ))

    return engine._build_result(
        name="slide",
        display_name="Slide (Excessive Hip Slide Toward Target)",
        phase="downswing",
        signals=signals,
        description="The hips are sliding excessively toward the target instead of rotating through impact. "
                    "Some lateral movement (2-4 inches) is correct — this is beyond that range. "
                    "Slide makes it difficult to square the face and transfers momentum to lateral motion instead of rotation.",
        ball_flight="Blocked shots right, pushes, loss of distance, inconsistent contact",
        root_cause="Limited lead hip internal rotation — the body can't rotate so it slides instead. "
                   "TPI research identifies lead hip internal rotation as the primary physical cause of slide.",
        source="TPI Swing Characteristics; Golf Fitness Association of America; Fleisig",
        measured_value=max_slide,
        elite_benchmark="<80px lateral hip slide (2-4 inch normal range)",
    )


def _face_hanging_back(engine: SyndromeEngine) -> SyndromeResult:
    """
    HANGING BACK: Weight fails to transfer to lead side in downswing.
    TPI: "Complete absence of forward weight shift in the downswing."

    Signals:
      1. Weight shift at impact is too low (not enough lead side)
      2. Weight shift RATE is slow or stalls mid-downswing
      3. Hip sway stays trail-side through impact
    """
    ds_wt,   wt_conf = engine._phase_metrics("downswing", "weight_shift")
    ds_hip,  hp_conf = engine._phase_metrics("downswing", "hip_sway_px")
    phase_conf = engine._phase_conf("downswing")

    impact_wt  = engine._high_conf_max(ds_wt, wt_conf)
    max_hip_trail = engine._high_conf_min(ds_hip, hp_conf)  # negative = trail side

    # Weight transfer rate — check if it stalls
    wt_rate_stall = False
    wt_rate_val   = None
    if len(ds_wt) >= 4:
        mid = len(ds_wt) // 2
        first_half_gain  = float(np.mean(ds_wt[mid:]) - np.mean(ds_wt[:mid]))
        wt_rate_val = first_half_gain
        wt_rate_stall = first_half_gain < 0.05  # not gaining lead-side weight

    signals = []

    # Signal 1: Low weight at impact
    wt_thresh = 0.20
    s1_val  = impact_wt
    s1_fire = s1_val is not None and s1_val < wt_thresh
    signals.append(Signal(
        name="low_impact_weight",
        triggered=s1_fire,
        value=s1_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight shift at impact: {s1_val:.2f} (target: >{wt_thresh})" if s1_val else "No data"
    ))

    # Signal 2: Weight transfer rate stalls
    signals.append(Signal(
        name="weight_transfer_stall",
        triggered=wt_rate_stall,
        value=wt_rate_val,
        threshold=0.05,
        confidence=phase_conf,
        description=f"Weight transfer rate in downswing: {wt_rate_val:.2f} — stalling" if wt_rate_val else "No data"
    ))

    # Signal 3: Hips stay trail-side during downswing
    hip_thresh = -15.0
    s3_val  = max_hip_trail
    s3_fire = s3_val is not None and s3_val < hip_thresh
    signals.append(Signal(
        name="hips_trail_in_downswing",
        triggered=s3_fire,
        value=s3_val,
        threshold=hip_thresh,
        confidence=phase_conf,
        description=f"Hips remain trail-side ({s3_val:.0f}px) through downswing" if s3_val else "No data"
    ))

    return engine._build_result(
        name="hanging_back",
        display_name="Hanging Back (No Weight Transfer)",
        phase="downswing",
        signals=signals,
        description="Weight is not transferring to the lead side during the downswing. "
                    "Elite golfers have 80-95% of weight on the lead foot at impact. "
                    "Hanging back creates a scooping, lifting motion and dramatically reduces power.",
        ball_flight="Fat shots, thin shots, pushes, loss of distance, weak contact",
        root_cause="Fear of hitting behind the ball, reverse pivot habit, poor hip sequencing. "
                   "Often a compensation — the body is trying to 'help' the ball into the air.",
        source="TPI Swing Characteristics; Fleisig Biomechanics; Richards et al. 1985",
        measured_value=impact_wt,
        elite_benchmark=">0.5 weight shift at impact (80-95% lead foot)",
    )


def _face_reverse_pivot(engine: SyndromeEngine) -> SyndromeResult:
    """
    REVERSE PIVOT: Weight moves toward LEAD side in backswing (opposite of correct).
    TPI-related: incorrect weight loading direction.

    Signals:
      1. Weight shift moves lead-ward during backswing
      2. Spine tilts toward target during backswing
      3. Hip does not load trail side
    """
    bs_wt,   wt_conf = engine._phase_metrics("backswing", "weight_shift")
    bs_sp,   sp_conf = engine._phase_metrics("backswing", "spine_tilt_deg")
    bs_hip,  hp_conf = engine._phase_metrics("backswing", "hip_sway_px")
    phase_conf = engine._phase_conf("backswing")

    # Weight at top should be trail-side (negative)
    wt_at_top    = engine._high_conf_mean(bs_wt[-max(1,len(bs_wt)//3):], wt_conf[-max(1,len(wt_conf)//3):])
    spine_change = None
    addr_sp = engine._get_attr_at("spine_tilt_deg", engine._phases.get("addr_idx", 0))

    if addr_sp is not None:
        top_sp = engine._high_conf_mean(bs_sp[-max(1,len(bs_sp)//3):], sp_conf[-max(1,len(sp_conf)//3):])
        if top_sp is not None:
            spine_change = top_sp - addr_sp  # positive = tilting toward target

    hip_trail = engine._high_conf_min(bs_hip, hp_conf)

    signals = []

    # Signal 1: Weight moving to lead side in backswing
    wt_thresh = 0.10
    s1_val  = wt_at_top
    s1_fire = s1_val is not None and s1_val > wt_thresh
    signals.append(Signal(
        name="lead_weight_in_backswing",
        triggered=s1_fire,
        value=s1_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight shifts lead ({s1_val:.2f}) during backswing — should load trail" if s1_val else "No data"
    ))

    # Signal 2: Spine tilting toward target
    sp_thresh = 5.0
    s2_val  = spine_change
    s2_fire = s2_val is not None and s2_val > sp_thresh
    signals.append(Signal(
        name="spine_tilts_target",
        triggered=s2_fire,
        value=s2_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine tilts {s2_val:.1f}° toward target in backswing" if s2_val else "No data"
    ))

    # Signal 3: Hip not loading trail
    hip_thresh = -15.0
    s3_val  = hip_trail
    s3_fire = s3_val is None or s3_val > hip_thresh  # not going trail
    signals.append(Signal(
        name="no_trail_hip_load",
        triggered=s3_fire,
        value=s3_val,
        threshold=hip_thresh,
        confidence=phase_conf,
        description=f"Hip doesn't load trail side (sway: {s3_val:.0f}px)" if s3_val else "No data"
    ))

    return engine._build_result(
        name="reverse_pivot",
        display_name="Reverse Pivot",
        phase="backswing",
        signals=signals,
        description="Weight is moving toward the lead side during the backswing — the opposite of correct. "
                    "Research shows elite golfers load 60-80% onto the trail foot at the top. "
                    "A reverse pivot eliminates stored energy and forces a steep, powerless downswing.",
        ball_flight="Fat shots, pulls, slices, steep angle of attack, loss of distance",
        root_cause="Upper body tilting toward target instead of rotating — 'reverse spine angle'. "
                   "Often caused by trying to keep the head perfectly still while restricting shoulder turn.",
        source="TPI Swing Characteristics; Richards et al. 1985; Fleisig",
        measured_value=wt_at_top,
        elite_benchmark="Weight should be <-0.2 (trail-loaded) at top of backswing",
    )


def _face_early_extension(engine: SyndromeEngine) -> SyndromeResult:
    """
    EARLY EXTENSION: Hips/spine straighten toward ball in downswing.
    TPI: #1 most common fault. Hips thrust forward, body stands up.

    Signals:
      1. Spine angle decreases (body standing up) from top to impact
      2. Hip sway moves toward ball (forward thrust)
      3. Weight transfer stalls DESPITE hip movement (hips go to ball not rotating)
    """
    addr_idx = engine._phases.get("addr_idx", 0)
    top_idx  = engine._phases.get("top_idx",  engine.n // 2)
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)

    addr_sp  = engine._get_attr_at("spine_tilt_deg", addr_idx)
    top_sp   = engine._get_attr_at("spine_tilt_deg", top_idx)
    imp_sp   = engine._get_attr_at("spine_tilt_deg", imp_idx)

    ds_hip,  hp_conf = engine._phase_metrics("downswing", "hip_sway_px")
    ds_wt,   wt_conf = engine._phase_metrics("downswing", "weight_shift")
    phase_conf = engine._phase_conf("downswing")

    # Spine angle change: from top to impact
    spine_change = None
    if top_sp is not None and imp_sp is not None:
        spine_change = abs(imp_sp) - abs(top_sp)  # negative = standing up

    # Hip thrust: forward movement during downswing
    hip_thrust = engine._high_conf_max(ds_hip, hp_conf)

    # Weight stall
    impact_wt = engine._high_conf_max(ds_wt, wt_conf)

    signals = []

    # Signal 1: Spine angle change (body standing up)
    sp_thresh = 8.0
    s1_val  = spine_change
    s1_fire = s1_val is not None and abs(s1_val) > sp_thresh
    signals.append(Signal(
        name="spine_angle_change",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=engine.metrics[top_idx].confidence if top_idx < engine.n else 0.5,
        description=f"Spine angle changed {abs(s1_val):.1f}° from top to impact" if s1_val else "No data"
    ))

    # Signal 2: Hip thrust toward ball
    hip_thresh = 30.0
    s2_val  = hip_thrust
    s2_fire = s2_val is not None and s2_val > hip_thresh
    signals.append(Signal(
        name="hip_thrust_forward",
        triggered=s2_fire,
        value=s2_val,
        threshold=hip_thresh,
        confidence=phase_conf,
        description=f"Hips thrust {s2_val:.0f}px toward ball in downswing" if s2_val else "No data"
    ))

    # Signal 3: Weight doesn't transfer cleanly (hip thrust ≠ weight transfer)
    wt_thresh = 0.30
    s3_val  = impact_wt
    s3_fire = s3_val is not None and s3_val < wt_thresh
    signals.append(Signal(
        name="weight_not_transferred",
        triggered=s3_fire,
        value=s3_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight at impact {s3_val:.2f} — insufficient despite hip movement" if s3_val else "No data"
    ))

    return engine._build_result(
        name="early_extension",
        display_name="Early Extension (Loss of Posture)",
        phase="downswing",
        signals=signals,
        description="The hips are thrusting toward the ball and the spine is straightening during the downswing. "
                    "TPI research identifies early extension as the most common fault in amateur golfers. "
                    "It prevents the arms from dropping into the 'slot' and forces compensatory timing.",
        ball_flight="Thin shots, blocks right, hooks from flipping hands, inconsistent contact",
        root_cause="Limited hip mobility preventing rotation without extension. "
                   "Weak glutes and core unable to resist the forward thrust. "
                   "Tight hip flexors make it impossible to maintain posture while rotating.",
        source="TPI research; Swing Lab Theory; Back Nine PT; Meister et al. 2011",
        measured_value=spine_change,
        elite_benchmark="<8° spine angle change from top to impact",
    )


def _face_over_the_top(engine: SyndromeEngine) -> SyndromeResult:
    """
    OVER THE TOP: Club/shoulders approach from outside the swing plane.
    Meister 2011: ALL pros initiate downswing with hip reversal BEFORE shoulders.
    Over the top = shoulders reverse BEFORE hips.

    Signals:
      1. Shoulder rotation starts declining before hip rotation in downswing
      2. Weight shifts lead early in backswing (transition rushed)
      3. Spine tilts toward target in downswing (upper body lunges)
    """
    top_idx = engine._phases.get("top_idx", engine.n // 2)
    imp_idx = engine._phases.get("impact_idx", engine.n * 3 // 4)
    n       = engine.n

    # Get rotation curves
    shld  = np.array([getattr(m, 'shoulder_rotation_deg', 0) or 0.0 for m in engine.metrics])
    hip   = np.array([getattr(m, 'hip_rotation_deg',      0) or 0.0 for m in engine.metrics])
    conf  = np.array([m.confidence for m in engine.metrics])

    shld_s = engine._smooth(shld, 3)
    hip_s  = engine._smooth(hip,  3)

    # Find when each starts declining after top
    def first_decline(arr, from_idx, lookahead=3):
        for i in range(from_idx, min(n - lookahead, from_idx + int(n * 0.4))):
            if arr[i] > arr[i + lookahead] + 1.5 and conf[i] > 0.5:
                return i
        return None

    shld_rev = first_decline(shld_s, top_idx)
    hip_rev  = first_decline(hip_s,  top_idx)

    # Weight shift in early downswing
    ds_wt,  wt_conf = engine._phase_metrics("downswing", "weight_shift")
    bs_wt,  bs_conf = engine._phase_metrics("backswing",  "weight_shift")
    phase_conf = engine._phase_conf("downswing")

    # Early lead-side weight in backswing
    early_lead = engine._high_conf_max(bs_wt[-max(1,len(bs_wt)//3):], bs_conf[-max(1,len(bs_conf)//3):])

    # Spine tilt in downswing
    ds_sp, sp_conf_arr = engine._phase_metrics("downswing", "spine_tilt_deg")
    spine_in_ds = engine._high_conf_mean(ds_sp, sp_conf_arr)

    addr_sp = engine._get_attr_at("spine_tilt_deg", engine._phases.get("addr_idx", 0))
    spine_change_ds = None
    if addr_sp is not None and spine_in_ds is not None:
        spine_change_ds = spine_in_ds - addr_sp

    signals = []

    # Signal 1: Sequence fault — shoulders before hips
    sequence_lag = None
    if shld_rev is not None and hip_rev is not None:
        sequence_lag = shld_rev - hip_rev  # negative = shoulders first (bad)
    s1_fire = sequence_lag is not None and sequence_lag < 0
    signals.append(Signal(
        name="shoulder_before_hip",
        triggered=s1_fire,
        value=sequence_lag,
        threshold=0,
        confidence=phase_conf,
        description=f"Shoulders reverse {abs(sequence_lag):.0f} frames before hips in downswing" if sequence_lag is not None else "Sequence unclear"
    ))

    # Signal 2: Early weight shift to lead in backswing transition
    s2_thresh = 0.15
    s2_val  = early_lead
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="early_lead_weight",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=engine._phase_conf("backswing"),
        description=f"Weight shifts lead ({s2_val:.2f}) before top of backswing" if s2_val else "No data"
    ))

    # Signal 3: Spine tilts toward target in downswing
    sp_thresh = 5.0
    s3_val  = spine_change_ds
    s3_fire = s3_val is not None and s3_val > sp_thresh
    signals.append(Signal(
        name="spine_target_tilt_ds",
        triggered=s3_fire,
        value=s3_val,
        threshold=sp_thresh,
        confidence=engine._phase_conf("downswing"),
        description=f"Spine tilts {s3_val:.1f}° toward target in downswing — upper body leading" if s3_val else "No data"
    ))

    return engine._build_result(
        name="over_the_top",
        display_name="Over the Top",
        phase="downswing",
        signals=signals,
        description="The downswing is initiated by the upper body instead of the lower body. "
                    "Meister et al. (2011) confirmed that ALL professional golfers initiate the downswing "
                    "with hip reversal BEFORE shoulder reversal. When shoulders lead, the club approaches "
                    "from outside the target line — the #1 cause of slicing.",
        ball_flight="Pull, pull-slice, steep divots pointing left, over-the-top path",
        root_cause="Initiating the downswing with hands and shoulders — 'hitting from the top'. "
                   "Often caused by poor hip mobility, transition anxiety, or trying to generate power with the arms.",
        source="Meister et al. 2011 J. Applied Biomechanics; TPI; BioSwing Dynamics",
        measured_value=sequence_lag,
        elite_benchmark="Hip rotation must reverse before shoulder rotation",
    )


def _face_casting(engine: SyndromeEngine) -> SyndromeResult:
    """
    CASTING/SCOOPING: Early release of wrist angles in downswing.
    TPI: "Any premature release of the wrist angles on the downswing."

    Signals:
      1. Wrist hinge lost early in downswing (angle returns to address before impact)
      2. Lead elbow starts bending before impact
      3. Weight doesn't transfer — arms and wrists fire without body
    """
    top_idx = engine._phases.get("top_idx", engine.n // 2)
    imp_idx = engine._phases.get("impact_idx", engine.n * 3 // 4)

    ds_wrist, wr_conf = engine._phase_metrics("downswing", "wrist_hinge_deg")
    bs_wrist, bs_wr   = engine._phase_metrics("backswing",  "wrist_hinge_deg")
    ds_elbow, el_conf = engine._phase_metrics("downswing", "lead_elbow_angle_deg")
    ds_wt,    wt_conf = engine._phase_metrics("downswing", "weight_shift")
    phase_conf = engine._phase_conf("downswing")

    # Wrist hinge at top
    wrist_at_top = engine._get_attr_at("wrist_hinge_deg", top_idx)
    # Address wrist angle
    wrist_at_addr = engine._get_attr_at("wrist_hinge_deg", engine._phases.get("addr_idx", 0))

    # Early wrist release: hinge returns to address angle before halfway through downswing
    early_release = False
    release_pct   = None
    if wrist_at_top is not None and wrist_at_addr is not None and len(ds_wrist) > 3:
        # Total hinge = top - addr (hinge is more angle at top)
        hinge_range = abs(wrist_at_top - wrist_at_addr)
        halfway_ds  = len(ds_wrist) // 2
        wrist_early = engine._high_conf_mean(ds_wrist[:halfway_ds], wr_conf[:halfway_ds])
        if wrist_early is not None and hinge_range > 5:
            # If wrist already close to address angle in first half of downswing = casting
            returned_pct = abs(wrist_early - wrist_at_top) / hinge_range
            release_pct  = returned_pct
            early_release = returned_pct > 0.60  # hinge 60% released in first half of downswing

    # Lead elbow collapse
    min_elbow  = engine._high_conf_min(ds_elbow, el_conf)
    addr_elbow = engine._get_attr_at("lead_elbow_angle_deg", engine._phases.get("addr_idx", 0))
    elbow_drop = None
    if min_elbow is not None and addr_elbow is not None:
        elbow_drop = addr_elbow - min_elbow  # positive = arm bending more

    # Weight stall
    impact_wt = engine._high_conf_max(ds_wt, wt_conf)

    signals = []

    # Signal 1: Early wrist release
    signals.append(Signal(
        name="early_wrist_release",
        triggered=early_release,
        value=release_pct,
        threshold=0.60,
        confidence=phase_conf,
        description=f"{release_pct*100:.0f}% of wrist hinge released in first half of downswing" if release_pct else "No data"
    ))

    # Signal 2: Lead elbow collapses
    el_thresh = 15.0
    s2_val  = elbow_drop
    s2_fire = s2_val is not None and s2_val > el_thresh
    signals.append(Signal(
        name="lead_elbow_collapse",
        triggered=s2_fire,
        value=s2_val,
        threshold=el_thresh,
        confidence=phase_conf,
        description=f"Lead elbow drops {s2_val:.1f}° — arm collapsing with cast" if s2_val else "No data"
    ))

    # Signal 3: Weight doesn't transfer (arms fire without body)
    wt_thresh = 0.25
    s3_val  = impact_wt
    s3_fire = s3_val is not None and s3_val < wt_thresh
    signals.append(Signal(
        name="body_not_transferring",
        triggered=s3_fire,
        value=s3_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight only {s3_val:.2f} at impact — arms firing without body" if s3_val else "No data"
    ))

    return engine._build_result(
        name="casting",
        display_name="Casting / Early Release",
        phase="downswing",
        signals=signals,
        description="Wrist angles are releasing too early in the downswing — like a fisherman casting a rod. "
                    "This destroys lag and club head speed. The wrists should maintain their angle until "
                    "the hands reach hip height in the downswing, then release naturally through the ball.",
        ball_flight="Weak, high shots, loss of compression, thin contact, scooped divots",
        root_cause="Trying to help the ball into the air by flipping the wrists. "
                   "Initiating the downswing with the hands instead of the lower body. "
                   "Fear of hitting the ground causes early release as a compensating lift.",
        source="TPI Swing Characteristics; Fleisig Biomechanics; TrackMan University",
        measured_value=release_pct,
        elite_benchmark="Wrist angles maintained until hands reach hip height in downswing",
    )


def _face_chicken_wing(engine: SyndromeEngine) -> SyndromeResult:
    """
    CHICKEN WING: Breakdown of lead elbow through impact.
    TPI: "Loss of extension or breakdown of the lead elbow through the impact area."

    Signals:
      1. Lead elbow angle drops significantly around impact
      2. Lead elbow is MORE bent at impact than at address (net bend)
      3. Wrist hinge lost (usually accompanies chicken wing)
    """
    imp_idx    = engine._phases.get("impact_idx", engine.n * 3 // 4)
    addr_idx   = engine._phases.get("addr_idx",   0)
    phase_conf = engine._phase_conf("downswing")

    # Lead elbow window around impact
    win_s = max(0, imp_idx - 4)
    win_e = min(engine.n, imp_idx + 5)
    elbow_win  = np.array([getattr(m, 'lead_elbow_angle_deg', 0) or 0.0 for m in engine.metrics[win_s:win_e]])
    conf_win   = np.array([m.confidence for m in engine.metrics[win_s:win_e]])

    min_elbow  = engine._high_conf_min(elbow_win, conf_win)
    addr_elbow = engine._get_attr_at("lead_elbow_angle_deg", addr_idx)

    # Net bend from address
    net_bend = None
    if min_elbow is not None and addr_elbow is not None and addr_elbow > 10:
        net_bend = addr_elbow - min_elbow

    # Wrist at impact
    wrist_imp  = engine._get_attr_at("wrist_hinge_deg", imp_idx)
    wrist_addr = engine._get_attr_at("wrist_hinge_deg", addr_idx)
    wrist_lost = None
    if wrist_imp is not None and wrist_addr is not None:
        wrist_lost = wrist_imp - wrist_addr  # positive = wrist more extended = scooping

    signals = []

    # Signal 1: Lead elbow well below 150° at impact
    el_thresh = 148.0
    s1_val  = min_elbow
    s1_fire = s1_val is not None and s1_val > 0 and s1_val < el_thresh
    signals.append(Signal(
        name="lead_elbow_below_threshold",
        triggered=s1_fire,
        value=s1_val,
        threshold=el_thresh,
        confidence=phase_conf,
        description=f"Lead elbow {s1_val:.1f}° near impact (threshold: >{el_thresh}°)" if s1_val else "No data"
    ))

    # Signal 2: More bent at impact than address (net bend)
    nb_thresh = 12.0
    s2_val  = net_bend
    s2_fire = s2_val is not None and s2_val > nb_thresh
    signals.append(Signal(
        name="net_elbow_bend",
        triggered=s2_fire,
        value=s2_val,
        threshold=nb_thresh,
        confidence=phase_conf,
        description=f"Lead elbow {s2_val:.1f}° more bent at impact than address" if s2_val else "No data"
    ))

    # Signal 3: Wrist scooping (hinge lost = angle increasing at impact)
    wrist_thresh = 8.0
    s3_val  = wrist_lost
    s3_fire = s3_val is not None and s3_val > wrist_thresh
    signals.append(Signal(
        name="wrist_scooping",
        triggered=s3_fire,
        value=s3_val,
        threshold=wrist_thresh,
        confidence=phase_conf,
        description=f"Wrist angle {s3_val:.1f}° more extended at impact — scooping" if s3_val else "No data"
    ))

    return engine._build_result(
        name="chicken_wing",
        display_name="Chicken Wing (Lead Arm Collapse)",
        phase="impact",
        signals=signals,
        description="The lead elbow is bending through the impact zone instead of staying extended. "
                    "TPI identifies this as a breakdown that reduces swing width, opens the club face, "
                    "and transfers power inefficiently. Usually paired with casting.",
        ball_flight="Weak slices, pulls, reduced distance, inconsistent contact",
        root_cause="Trail arm dominance pushing through impact. "
                   "Scooping/flipping motion trying to help the ball up. "
                   "TPI: 35% chance of chicken wing if wrist mobility is limited.",
        source="TPI Swing Characteristics; HackMotion; TrackMan University",
        measured_value=min_elbow,
        elite_benchmark="Lead elbow >150° through impact",
    )


def _face_flying_elbow(engine: SyndromeEngine) -> SyndromeResult:
    """
    FLYING ELBOW: Trail elbow excessively elevated at top of backswing.
    TPI: "An extremely elevated trail elbow during the backswing."

    Signals:
      1. Trail elbow angle is much higher than lead elbow at top
      2. Trail elbow angle significantly different from address position
      3. Shoulder plane is flatter than expected (accompanies flying elbow)
    """
    top_idx    = engine._phases.get("top_idx", engine.n // 2)
    addr_idx   = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("backswing")

    trail_elbow_top  = engine._get_attr_at("trail_elbow_angle_deg", top_idx)
    lead_elbow_top   = engine._get_attr_at("lead_elbow_angle_deg",  top_idx)
    trail_elbow_addr = engine._get_attr_at("trail_elbow_angle_deg", addr_idx)

    elbow_diff       = None
    if trail_elbow_top is not None and lead_elbow_top is not None:
        elbow_diff = trail_elbow_top - lead_elbow_top  # positive = trail much more bent

    elbow_change = None
    if trail_elbow_top is not None and trail_elbow_addr is not None:
        elbow_change = abs(trail_elbow_addr - trail_elbow_top)

    signals = []

    # Signal 1: Trail elbow much more elevated (bent) than lead
    diff_thresh = 25.0
    s1_val  = elbow_diff
    s1_fire = s1_val is not None and abs(s1_val) > diff_thresh
    signals.append(Signal(
        name="trail_elbow_elevated",
        triggered=s1_fire,
        value=s1_val,
        threshold=diff_thresh,
        confidence=phase_conf,
        description=f"Trail elbow {abs(s1_val):.1f}° more elevated than lead at top" if s1_val else "No data"
    ))

    # Signal 2: Large change from address
    change_thresh = 40.0
    s2_val  = elbow_change
    s2_fire = s2_val is not None and s2_val > change_thresh
    signals.append(Signal(
        name="trail_elbow_change",
        triggered=s2_fire,
        value=s2_val,
        threshold=change_thresh,
        confidence=phase_conf,
        description=f"Trail elbow changed {s2_val:.1f}° from address to top" if s2_val else "No data"
    ))

    return engine._build_result(
        name="flying_elbow",
        display_name="Flying Trail Elbow at Top",
        phase="backswing",
        signals=signals,
        description="The trail elbow is excessively elevated at the top of the backswing. "
                    "TPI identifies this as causing a disconnected, across-the-line position "
                    "that promotes an over-the-top downswing path.",
        ball_flight="Over-the-top pull, slice, steep angle of attack",
        root_cause="Arms lifting rather than rotating in the backswing. "
                   "Limited shoulder external rotation, or overactive hands in the takeaway.",
        source="TPI Swing Characteristics; BioSwing Dynamics",
        measured_value=trail_elbow_top,
        elite_benchmark="Trail elbow should be within 25° of lead elbow angle at top",
    )


def _face_loss_of_posture(engine: SyndromeEngine) -> SyndromeResult:
    """
    LOSS OF POSTURE: Any significant alteration from setup angles during the swing.
    TPI: Most common fault. Spine angle changes beyond normal range.

    Signals:
      1. Spine tilt changes significantly from address to any key position
      2. Change is consistent across multiple frames (not just one noisy frame)
      3. Head moves up (standing up) or down (squatting)
    """
    addr_idx   = engine._phases.get("addr_idx",   0)
    top_idx    = engine._phases.get("top_idx",    engine.n // 2)
    imp_idx    = engine._phases.get("impact_idx", engine.n * 3 // 4)
    phase_conf = (engine._phase_conf("backswing") + engine._phase_conf("downswing")) / 2

    addr_sp  = engine._get_attr_at("spine_tilt_deg", addr_idx)
    top_sp   = engine._get_attr_at("spine_tilt_deg", top_idx)
    imp_sp   = engine._get_attr_at("spine_tilt_deg", imp_idx)

    # Max change across the swing
    max_change = 0.0
    phase_name = "backswing"
    if addr_sp is not None:
        for idx, name in [(top_idx, "backswing"), (imp_idx, "downswing")]:
            val = engine._get_attr_at("spine_tilt_deg", idx)
            if val is not None:
                chg = abs(abs(val) - abs(addr_sp))
                if chg > max_change:
                    max_change = chg
                    phase_name = name

    # Head vertical movement
    addr_conf  = engine.metrics[addr_idx].confidence if addr_idx < engine.n else 0
    head_move  = engine._get_attr_at("head_movement_px", imp_idx)

    signals = []

    # Signal 1: Spine angle change > 10°
    sp_thresh = 10.0
    s1_val  = max_change if max_change > 0 else None
    s1_fire = s1_val is not None and s1_val > sp_thresh
    signals.append(Signal(
        name="spine_angle_change",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine tilt changed {s1_val:.1f}° from address during {phase_name}" if s1_val else "No data"
    ))

    # Signal 2: Head moving significantly (follows posture change)
    head_thresh = 50.0
    s2_val  = head_move
    s2_fire = s2_val is not None and s2_val > head_thresh
    signals.append(Signal(
        name="head_movement",
        triggered=s2_fire,
        value=s2_val,
        threshold=head_thresh,
        confidence=engine.metrics[imp_idx].confidence if imp_idx < engine.n else 0.5,
        description=f"Head moved {s2_val:.0f}px from address — consistent with posture change" if s2_val else "No data"
    ))

    return engine._build_result(
        name="loss_of_posture",
        display_name="Loss of Posture",
        phase=phase_name,
        signals=signals,
        description="Spine angles are significantly different from the address position during the swing. "
                    "TPI defines this as any significant alteration from setup angles. "
                    "Loss of posture makes consistent ball striking nearly impossible.",
        ball_flight="Inconsistent contact, mis-hits, fat/thin shots, directional issues",
        root_cause="Limited core stability, poor hip mobility, or a compensation for another fault "
                   "(e.g. loss of posture as a result of early extension or sway).",
        source="TPI Swing Characteristics — #1 most common fault",
        measured_value=max_change,
        elite_benchmark="<10° spine angle change from address through the swing",
    )


def _face_reverse_spine(engine: SyndromeEngine) -> SyndromeResult:
    """
    REVERSE SPINE ANGLE: Upper body tilts excessively toward target in backswing.
    TPI: "#1 cause of lower back pain in golf."

    Signals:
      1. Spine tilts toward target during backswing (not away from it)
      2. Shoulder tilts more than expected toward target
      3. Weight moves lead side (reverse pivot companion)
    """
    bs_sp,   sp_conf = engine._phase_metrics("backswing", "spine_tilt_deg")
    addr_idx = engine._phases.get("addr_idx", 0)
    addr_sp  = engine._get_attr_at("spine_tilt_deg", addr_idx)
    phase_conf = engine._phase_conf("backswing")

    # Spine tilt at top of backswing
    top_sp_val = engine._high_conf_mean(bs_sp[-max(1,len(bs_sp)//3):],
                                         sp_conf[-max(1,len(sp_conf)//3):])
    spine_change = None
    if addr_sp is not None and top_sp_val is not None:
        spine_change = top_sp_val - addr_sp  # positive = toward target = bad

    bs_wt, wt_conf = engine._phase_metrics("backswing", "weight_shift")
    wt_top = engine._high_conf_mean(bs_wt[-max(1,len(bs_wt)//3):],
                                     wt_conf[-max(1,len(wt_conf)//3):])

    signals = []

    # Signal 1: Spine tilts toward target in backswing
    sp_thresh = 7.0
    s1_val  = spine_change
    s1_fire = s1_val is not None and s1_val > sp_thresh
    signals.append(Signal(
        name="spine_toward_target",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine tilts {s1_val:.1f}° toward target in backswing" if s1_val else "No data"
    ))

    # Signal 2: Weight goes lead side (companion to reverse spine)
    wt_thresh = 0.10
    s2_val  = wt_top
    s2_fire = s2_val is not None and s2_val > wt_thresh
    signals.append(Signal(
        name="lead_weight_reverse_spine",
        triggered=s2_fire,
        value=s2_val,
        threshold=wt_thresh,
        confidence=phase_conf,
        description=f"Weight shifts lead ({s2_val:.2f}) during backswing — classic reverse spine companion" if s2_val else "No data"
    ))

    return engine._build_result(
        name="reverse_spine_angle",
        display_name="Reverse Spine Angle",
        phase="backswing",
        signals=signals,
        description="The upper body is tilting excessively toward the target during the backswing. "
                    "TPI identifies reverse spine angle as the #1 cause of lower back pain in golf. "
                    "It makes it nearly impossible to generate proper rotation and forces an over-the-top downswing.",
        ball_flight="Over-the-top pull, slices, steep angle of attack, back pain",
        root_cause="Limited thoracic spine mobility, weak core stability, "
                   "or actively restricting the shoulder turn causing the body to tilt instead of rotate.",
        source="TPI Swing Characteristics; Back Nine PT; Swing Lab Theory",
        measured_value=spine_change,
        elite_benchmark="Spine should maintain or tilt slightly away from target in backswing",
    )


def _face_forward_lunge(engine: SyndromeEngine) -> SyndromeResult:
    """
    FORWARD LUNGE: Upper body lunges toward target excessively in transition/downswing.
    TPI: "Aggressive lunge toward target with upper body during transition."
    Normal: ~2-3 inch upper body lateral move. A lunge exceeds this significantly.

    Signals:
      1. Head moves significantly toward target in downswing
      2. Upper body moves more than lower body toward target
      3. Weight shift happens too fast (lunge vs controlled transfer)
    """
    top_idx  = engine._phases.get("top_idx",    engine.n // 2)
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)

    head_at_top = engine._get_attr_at("head_movement_px", top_idx)
    head_at_imp = engine._get_attr_at("head_movement_px", imp_idx)

    ds_wt,  wt_conf = engine._phase_metrics("downswing", "weight_shift")
    phase_conf = engine._phase_conf("downswing")

    # Rapid weight shift rate (lunge = weight shoots to lead immediately)
    lunge_rate = None
    if len(ds_wt) >= 3:
        first_third = max(1, len(ds_wt) // 3)
        early_wt    = engine._high_conf_mean(ds_wt[:first_third], wt_conf[:first_third])
        final_wt    = engine._high_conf_mean(ds_wt[-first_third:], wt_conf[-first_third:])
        if early_wt is not None and final_wt is not None:
            # Rate = early gain (lunge = most weight shift in first third of downswing)
            lunge_rate = early_wt  # if already high early = lunge

    head_surge = None
    if head_at_top is not None and head_at_imp is not None:
        head_surge = head_at_imp - head_at_top

    signals = []

    # Signal 1: Head moves significantly in downswing (lunging pulls head)
    head_thresh = 35.0
    s1_val  = head_surge
    s1_fire = s1_val is not None and s1_val > head_thresh
    signals.append(Signal(
        name="head_surges_forward",
        triggered=s1_fire,
        value=s1_val,
        threshold=head_thresh,
        confidence=phase_conf,
        description=f"Head moves {s1_val:.0f}px toward target from top to impact" if s1_val else "No data"
    ))

    # Signal 2: Weight reaches high level very early in downswing
    lunge_thresh = 0.35
    s2_val  = lunge_rate
    s2_fire = s2_val is not None and s2_val > lunge_thresh
    signals.append(Signal(
        name="rapid_early_weight_surge",
        triggered=s2_fire,
        value=s2_val,
        threshold=lunge_thresh,
        confidence=phase_conf,
        description=f"Weight reaches {s2_val:.2f} in early downswing — lunging pattern" if s2_val else "No data"
    ))

    return engine._build_result(
        name="forward_lunge",
        display_name="Forward Lunge",
        phase="downswing",
        signals=signals,
        description="The upper body is aggressively lunging toward the target at the start of the downswing. "
                    "TPI defines this as upper body having more forward movement than the lower body in transition. "
                    "A controlled lateral weight shift of 2-4 inches is correct — lunging exceeds this.",
        ball_flight="Over-the-top pull, loss of lag, thin contact from changing the angle of attack",
        root_cause="Poor transition sequence — upper body fires before lower body establishes position. "
                   "Often paired with over-the-top.",
        source="TPI Swing Characteristics; Postureffect",
        measured_value=head_surge,
        elite_benchmark="Upper body should trail lower body in lateral movement",
    )


# ---------------------------------------------------------------------------
# BACK VIEW SYNDROMES
# ---------------------------------------------------------------------------

def _back_early_extension(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW EARLY EXTENSION: Spine straightens and hips thrust toward ball.
    Best visible from back view — spine angle change is most reliable here.

    Signals:
      1. Spine angle changes significantly from top to impact (standing up)
      2. Hip slides forward (thrust toward ball) — visible in back view
      3. Shoulder plane gets steeper in downswing
    """
    addr_idx = engine._phases.get("addr_idx", 0)
    top_idx  = engine._phases.get("top_idx",  engine.n // 2)
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)
    phase_conf = engine._phase_conf("downswing")

    top_spine  = engine._get_attr_at("spine_angle_deg",  top_idx)
    imp_spine  = engine._get_attr_at("spine_angle_deg",  imp_idx)
    addr_spine = engine._get_attr_at("spine_angle_deg",  addr_idx)

    spine_change = None
    if top_spine is not None and imp_spine is not None:
        spine_change = abs(imp_spine) - abs(top_spine)  # negative = standing up

    hip_thrust = engine._get_attr_at("hip_slide_px", imp_idx)

    ds_shld, sp_conf = engine._phase_metrics("downswing", "shoulder_plane_deg")
    bs_shld, _       = engine._phase_metrics("backswing",  "shoulder_plane_deg")
    top_plane  = engine._high_conf_mean(bs_shld[-max(1,len(bs_shld)//3):], _[-max(1,len(_)//3):])
    early_ds   = engine._high_conf_mean(ds_shld[:max(1,len(ds_shld)//3)], sp_conf[:max(1,len(sp_conf)//3)])
    plane_steep = None
    if top_plane is not None and early_ds is not None:
        plane_steep = early_ds - top_plane

    signals = []

    # Signal 1: Spine straightens significantly
    sp_thresh = 7.0
    s1_val  = spine_change
    s1_fire = s1_val is not None and abs(s1_val) > sp_thresh
    signals.append(Signal(
        name="spine_straightens",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine changed {abs(s1_val):.1f}° from top to impact" if s1_val else "No data"
    ))

    # Signal 2: Hip thrust toward ball
    hip_thresh = 40.0
    s2_val  = hip_thrust
    s2_fire = s2_val is not None and abs(s2_val) > hip_thresh
    signals.append(Signal(
        name="hip_thrusts_forward",
        triggered=s2_fire,
        value=s2_val,
        threshold=hip_thresh,
        confidence=phase_conf,
        description=f"Hip slides {abs(s2_val):.0f}px toward ball" if s2_val else "No data"
    ))

    # Signal 3: Shoulder plane steepens
    plane_thresh = 5.0
    s3_val  = plane_steep
    s3_fire = s3_val is not None and s3_val > plane_thresh
    signals.append(Signal(
        name="plane_steepens",
        triggered=s3_fire,
        value=s3_val,
        threshold=plane_thresh,
        confidence=engine._phase_conf("downswing"),
        description=f"Shoulder plane steepens {s3_val:.1f}° in early downswing" if s3_val else "No data"
    ))

    return engine._build_result(
        name="back_early_extension",
        display_name="Early Extension (Back View Confirmation)",
        phase="downswing",
        signals=signals,
        description="Hips are thrusting toward the ball and the spine is straightening in the downswing. "
                    "The back view provides the clearest picture of spine angle maintenance — "
                    "elite golfers hold their address angles from the top through impact.",
        ball_flight="Thin shots, blocks, hooks from compensatory flip",
        root_cause="Limited hip mobility, weak glutes/core, tight hip flexors",
        source="TPI research; Swing Lab Theory; Back Nine PT",
        measured_value=spine_change,
        elite_benchmark="<7° spine angle change from top to impact",
    )


def _back_reverse_spine(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW REVERSE SPINE ANGLE: Upper body tilts toward target in backswing.
    Back view is excellent for seeing this — spine curves toward camera.

    Signals:
      1. Spine angle increases in backswing (less forward bend = leaning back)
      2. Head moves forward of hips in backswing
      3. Shoulder plane angle changes toward horizontal
    """
    addr_idx   = engine._phases.get("addr_idx", 0)
    bs_sp, sp_conf = engine._phase_metrics("backswing", "spine_angle_deg")
    addr_sp = engine._get_attr_at("spine_angle_deg", addr_idx)
    phase_conf = engine._phase_conf("backswing")

    top_sp = engine._high_conf_mean(bs_sp[-max(1,len(bs_sp)//3):],
                                     sp_conf[-max(1,len(sp_conf)//3):])
    spine_change = None
    if addr_sp is not None and top_sp is not None:
        # Spine angle increasing = less forward bend = leaning toward target
        spine_change = top_sp - addr_sp  # negative = leaning back = reverse spine

    bs_head_fwd, hd_conf = engine._phase_metrics("backswing", "head_forward_px")
    addr_head_fwd = engine._get_attr_at("head_forward_px", addr_idx)
    head_fwd_top  = engine._high_conf_mean(bs_head_fwd[-max(1,len(bs_head_fwd)//3):],
                                            hd_conf[-max(1,len(hd_conf)//3):])
    head_change   = None
    if addr_head_fwd is not None and head_fwd_top is not None:
        head_change = head_fwd_top - addr_head_fwd

    signals = []

    # Signal 1: Spine angle increases (less forward bend)
    sp_thresh = 8.0
    s1_val  = spine_change
    # For back view: spine_angle decreasing = standing up more
    s1_fire = s1_val is not None and abs(s1_val) > sp_thresh
    signals.append(Signal(
        name="spine_angle_increases",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine angle changed {s1_val:.1f}° in backswing — leaning" if s1_val else "No data"
    ))

    # Signal 2: Head moves forward of hips (reverse spine signature)
    head_thresh = 25.0
    s2_val  = head_change
    s2_fire = s2_val is not None and s2_val > head_thresh
    signals.append(Signal(
        name="head_forward_of_hips",
        triggered=s2_fire,
        value=s2_val,
        threshold=head_thresh,
        confidence=phase_conf,
        description=f"Head moves {s2_val:.0f}px more forward than hips in backswing" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_reverse_spine",
        display_name="Reverse Spine Angle (Back View)",
        phase="backswing",
        signals=signals,
        description="The spine is tilting toward the target during the backswing. "
                    "TPI identifies this as the #1 cause of lower back pain in golf. "
                    "The back view clearly shows the spine curving toward the camera.",
        ball_flight="Over-the-top, slices, back pain, steep downswing",
        root_cause="Limited thoracic spine mobility, weak core, or restricting the shoulder turn",
        source="TPI Swing Characteristics; Back Nine PT",
        measured_value=spine_change,
        elite_benchmark="Spine angle should remain stable or increase forward bend in backswing",
    )


def _back_flat_shoulder_plane(engine: SyndromeEngine) -> SyndromeResult:
    """
    FLAT SHOULDER PLANE: Shoulders turn too horizontally in backswing.
    TPI: "Shoulders turn on a more horizontal plane than the axis of original spine angle."
    Best visible from back/DTL view.

    Signals:
      1. Shoulder plane angle too horizontal (near 0°) at top of backswing
      2. Spine angle flattens in backswing (companion fault)
      3. Trail arm overly elevated (flying elbow)
    """
    top_idx    = engine._phases.get("top_idx", engine.n // 2)
    addr_idx   = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("backswing")

    shld_at_top  = engine._get_attr_at("shoulder_plane_deg", top_idx)
    shld_at_addr = engine._get_attr_at("shoulder_plane_deg", addr_idx)
    spine_at_addr = engine._get_attr_at("spine_angle_deg",   addr_idx)
    spine_at_top  = engine._get_attr_at("spine_angle_deg",   top_idx)

    # Ideal: shoulders turn perpendicular to spine axis
    # If spine is ~35° forward, shoulder plane should be ~35° from horizontal
    # Flat = shoulder plane < expected (based on spine angle)
    plane_deficit = None
    if shld_at_top is not None and spine_at_addr is not None:
        ideal_plane = 90.0 - spine_at_addr  # perpendicular to spine
        plane_deficit = ideal_plane - abs(shld_at_top)  # positive = too flat

    spine_flattens = None
    if spine_at_addr is not None and spine_at_top is not None:
        spine_flattens = spine_at_addr - spine_at_top  # positive = standing up = flat plane companion

    signals = []

    # Signal 1: Shoulder plane too horizontal
    flat_thresh = 15.0
    s1_val  = plane_deficit
    s1_fire = s1_val is not None and s1_val > flat_thresh
    signals.append(Signal(
        name="shoulder_plane_too_flat",
        triggered=s1_fire,
        value=s1_val,
        threshold=flat_thresh,
        confidence=phase_conf,
        description=f"Shoulder plane {s1_val:.1f}° flatter than spine-perpendicular ideal" if s1_val else "No data"
    ))

    # Signal 2: Spine angle flattens (standing up = flat plane)
    sf_thresh = 8.0
    s2_val  = spine_flattens
    s2_fire = s2_val is not None and s2_val > sf_thresh
    signals.append(Signal(
        name="spine_flattens_with_plane",
        triggered=s2_fire,
        value=s2_val,
        threshold=sf_thresh,
        confidence=phase_conf,
        description=f"Spine angle decreases {s2_val:.1f}° — body standing up causing flat plane" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_flat_shoulder_plane",
        display_name="Flat Shoulder Plane (Back View)",
        phase="backswing",
        signals=signals,
        description="The shoulders are rotating on a more horizontal plane than the spine axis dictates. "
                    "TPI: In an ideal swing, shoulders turn perpendicular to the tilt of the spine. "
                    "A flat shoulder plane often accompanies early extension and loss of posture.",
        ball_flight="Steep downswing, over-the-top path, inconsistent contact",
        root_cause="Loss of forward spine angle during the backswing (standing up). "
                   "Limited torso-pelvis separation, shortened lat flexibility.",
        source="TPI Swing Characteristics; Flat Shoulder Plane; Back Nine PT",
        measured_value=shld_at_top,
        elite_benchmark="Shoulder plane should be perpendicular to spine tilt (~35-45° from horizontal)",
    )


def _back_sway(engine: SyndromeEngine) -> SyndromeResult:
    """BACK VIEW SWAY: Hip slides away from target in backswing."""
    bs_hip, hp_conf = engine._phase_metrics("backswing", "hip_slide_px")
    phase_conf = engine._phase_conf("backswing")

    max_sway = engine._high_conf_max(np.abs(bs_hip), hp_conf)
    # Check direction — sway = trail-ward (depends on convention)
    mean_sway_dir = engine._high_conf_mean(bs_hip, hp_conf)

    signals = []

    sway_thresh = 35.0
    s1_val  = max_sway
    s1_fire = s1_val is not None and s1_val > sway_thresh
    signals.append(Signal(
        name="hip_slides_trail",
        triggered=s1_fire,
        value=s1_val,
        threshold=sway_thresh,
        confidence=phase_conf,
        description=f"Hip slides {s1_val:.0f}px laterally in backswing" if s1_val else "No data"
    ))

    bs_sp, sp_conf = engine._phase_metrics("backswing", "spine_angle_deg")
    spine_change = engine._high_conf_mean(bs_sp, sp_conf)
    addr_sp = engine._get_attr_at("spine_angle_deg", engine._phases.get("addr_idx", 0))
    sp_val = None
    if addr_sp and spine_change:
        sp_val = abs(spine_change - addr_sp)

    sp_thresh = 6.0
    s2_val  = sp_val
    s2_fire = s2_val is not None and s2_val > sp_thresh
    signals.append(Signal(
        name="spine_changes_with_sway",
        triggered=s2_fire,
        value=s2_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine angle changes {s2_val:.1f}° consistent with lateral slide" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_sway",
        display_name="Sway (Back View)",
        phase="backswing",
        signals=signals,
        description="Lower body is sliding away from target during backswing rather than rotating. "
                    "The back view clearly shows this as lateral movement of the trail hip.",
        ball_flight="Reverse weight shift, fat shots, steep downswing",
        root_cause="Limited trail hip internal rotation, weak glutes",
        source="TPI Swing Characteristics; Golf Fitness Association of America",
        measured_value=max_sway,
        elite_benchmark="<35px lateral hip movement in backswing",
    )


def _back_slide(engine: SyndromeEngine) -> SyndromeResult:
    """BACK VIEW SLIDE: Excessive hip slide toward target in downswing."""
    ds_hip, hp_conf = engine._phase_metrics("downswing", "hip_slide_px")
    phase_conf = engine._phase_conf("downswing")

    max_slide = engine._high_conf_max(np.abs(ds_hip), hp_conf)

    ds_sp, sp_conf = engine._phase_metrics("downswing", "shoulder_plane_deg")
    top_shld = engine._get_attr_at("shoulder_plane_deg", engine._phases.get("top_idx", 0))
    early_ds = engine._high_conf_mean(ds_sp[:max(1, len(ds_sp)//3)],
                                       sp_conf[:max(1, len(sp_conf)//3)])
    plane_steep = None
    if top_shld and early_ds:
        plane_steep = early_ds - top_shld

    signals = []

    slide_thresh = 80.0
    s1_val  = max_slide
    s1_fire = s1_val is not None and s1_val > slide_thresh
    signals.append(Signal(
        name="excessive_forward_slide",
        triggered=s1_fire,
        value=s1_val,
        threshold=slide_thresh,
        confidence=phase_conf,
        description=f"Hip slides {s1_val:.0f}px toward target in downswing" if s1_val else "No data"
    ))

    ps_thresh = 5.0
    s2_val  = plane_steep
    s2_fire = s2_val is not None and s2_val > ps_thresh
    signals.append(Signal(
        name="plane_steepens_with_slide",
        triggered=s2_fire,
        value=s2_val,
        threshold=ps_thresh,
        confidence=phase_conf,
        description=f"Shoulder plane steepens {s2_val:.1f}° — upper body compensating for slide" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_slide",
        display_name="Slide (Back View)",
        phase="downswing",
        signals=signals,
        description="Hips sliding excessively toward target in downswing rather than rotating. "
                    "The back view clearly shows this lateral movement pattern. "
                    "Some slide (2-4 inches) is normal — this is beyond that range.",
        ball_flight="Blocks, pushes, loss of rotation power",
        root_cause="Limited lead hip internal rotation mobility",
        source="TPI Swing Characteristics; Golf Fitness Association of America",
        measured_value=max_slide,
        elite_benchmark="<80px lateral slide toward target",
    )


def _back_over_the_top(engine: SyndromeEngine) -> SyndromeResult:
    """BACK VIEW OVER THE TOP: Shoulder plane steepens in early downswing."""
    top_idx = engine._phases.get("top_idx", engine.n // 2)
    n = engine.n

    bs_shld, bs_conf = engine._phase_metrics("backswing",  "shoulder_plane_deg")
    ds_shld, ds_conf = engine._phase_metrics("downswing",  "shoulder_plane_deg")

    top_plane   = engine._high_conf_mean(bs_shld[-max(1,len(bs_shld)//4):],
                                          bs_conf[-max(1,len(bs_conf)//4):])
    early_ds    = engine._high_conf_mean(ds_shld[:max(1,len(ds_shld)//3)],
                                          ds_conf[:max(1,len(ds_conf)//3)])
    steepening  = None
    if top_plane is not None and early_ds is not None:
        steepening = early_ds - top_plane

    bs_head, hd_conf = engine._phase_metrics("backswing", "head_movement_px")
    ds_head, _       = engine._phase_metrics("downswing", "head_movement_px")
    head_surge_ds = engine._high_conf_mean(ds_head[:max(1,len(ds_head)//3)], _[:max(1,len(_)//3)])
    head_at_top   = engine._high_conf_mean(bs_head[-max(1,len(bs_head)//3):], hd_conf[-max(1,len(hd_conf)//3):])
    head_surge_val = None
    if head_surge_ds and head_at_top:
        head_surge_val = head_surge_ds - head_at_top

    phase_conf = engine._phase_conf("downswing")
    signals    = []

    steep_thresh = 6.0
    s1_val  = steepening
    s1_fire = s1_val is not None and s1_val > steep_thresh
    signals.append(Signal(
        name="plane_steepens_downswing",
        triggered=s1_fire,
        value=s1_val,
        threshold=steep_thresh,
        confidence=phase_conf,
        description=f"Shoulder plane steepens {s1_val:.1f}° in early downswing" if s1_val else "No data"
    ))

    head_thresh = 30.0
    s2_val  = head_surge_val
    s2_fire = s2_val is not None and s2_val > head_thresh
    signals.append(Signal(
        name="head_surges_in_transition",
        triggered=s2_fire,
        value=s2_val,
        threshold=head_thresh,
        confidence=phase_conf,
        description=f"Head moves {s2_val:.0f}px forward in transition — upper body leading" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_over_the_top",
        display_name="Over the Top (Back View)",
        phase="downswing",
        signals=signals,
        description="The shoulder plane is steepening in the early downswing — the club is being thrown "
                    "outside the intended swing plane. The back view shows this as the shoulder line "
                    "getting more horizontal (steeper) rather than shallowing into the slot.",
        ball_flight="Pull, pull-slice, steep divots, over-the-top path",
        root_cause="Upper body initiating downswing before lower body establishes position",
        source="TPI; Meister et al. 2011; BioSwing Dynamics",
        measured_value=steepening,
        elite_benchmark="Shoulder plane should shallow or maintain angle in early downswing",
    )


def _back_loss_of_posture(engine: SyndromeEngine) -> SyndromeResult:
    """BACK VIEW LOSS OF POSTURE: Spine angle changes significantly from address."""
    addr_idx = engine._phases.get("addr_idx", 0)
    top_idx  = engine._phases.get("top_idx",  engine.n // 2)
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)

    addr_sp = engine._get_attr_at("spine_angle_deg", addr_idx)
    top_sp  = engine._get_attr_at("spine_angle_deg", top_idx)
    imp_sp  = engine._get_attr_at("spine_angle_deg", imp_idx)

    max_change = 0.0
    worst_phase = "backswing"
    if addr_sp is not None:
        for idx, name in [(top_idx, "backswing"), (imp_idx, "downswing")]:
            val = engine._get_attr_at("spine_angle_deg", idx)
            if val is not None:
                chg = abs(val - addr_sp)
                if chg > max_change:
                    max_change = chg
                    worst_phase = name

    phase_conf = engine._phase_conf(worst_phase)
    head_move  = engine._get_attr_at("head_movement_px", imp_idx)

    signals = []

    sp_thresh = 10.0
    s1_val  = max_change if max_change > 0 else None
    s1_fire = s1_val is not None and s1_val > sp_thresh
    signals.append(Signal(
        name="spine_angle_change_back",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=phase_conf,
        description=f"Spine angle changes {s1_val:.1f}° from address" if s1_val else "No data"
    ))

    hm_thresh = 45.0
    s2_val  = head_move
    s2_fire = s2_val is not None and s2_val > hm_thresh
    signals.append(Signal(
        name="head_moves_with_posture",
        triggered=s2_fire,
        value=s2_val,
        threshold=hm_thresh,
        confidence=phase_conf,
        description=f"Head moves {s2_val:.0f}px — consistent with posture change" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_loss_of_posture",
        display_name="Loss of Posture (Back View)",
        phase=worst_phase,
        signals=signals,
        description="Spine angles have changed significantly from the address position. "
                    "The back view is excellent for measuring forward spine angle — "
                    "loss of posture here confirms a fundamental setup-angle change.",
        ball_flight="Inconsistent contact, fat/thin shots",
        root_cause="Poor core stability, hip mobility limitations, compensation patterns",
        source="TPI Swing Characteristics — #1 most common fault",
        measured_value=max_change,
        elite_benchmark="<10° spine angle change from address",
    )


def _back_c_posture(engine: SyndromeEngine) -> SyndromeResult:
    """
    C-POSTURE: Rounded upper back at address (shoulders slumped forward).
    TPI: "Shoulders slumped forward, definitive roundedness to thoracic spine."
    Detectable from back view as reduced spine angle at address.

    Signals:
      1. Spine angle at address is too low (upright/rounded) — limited forward bend
      2. Shoulder plane angle at address is too flat (rounded shoulders)
    """
    addr_idx = engine._phases.get("addr_idx", 0)
    addr_conf = engine.metrics[addr_idx].confidence if addr_idx < engine.n else 0

    spine_at_addr = engine._get_attr_at("spine_angle_deg",   addr_idx)
    shld_at_addr  = engine._get_attr_at("shoulder_plane_deg", addr_idx)

    signals = []

    # C-posture: spine angle too small at address (not enough forward bend)
    # Combined with rounded appearance — spine angle < 20° = C-posture candidate
    sp_thresh = 20.0
    s1_val  = spine_at_addr
    s1_fire = s1_val is not None and s1_val < sp_thresh and s1_val > 0
    signals.append(Signal(
        name="insufficient_spine_angle",
        triggered=s1_fire,
        value=s1_val,
        threshold=sp_thresh,
        confidence=addr_conf,
        description=f"Spine angle at address only {s1_val:.1f}° — insufficient forward bend (C-posture)" if s1_val else "No data"
    ))

    # Shoulder plane too flat at address
    sp2_thresh = 10.0
    s2_val  = shld_at_addr
    s2_fire = s2_val is not None and abs(s2_val) < sp2_thresh
    signals.append(Signal(
        name="flat_shoulder_at_address",
        triggered=s2_fire,
        value=s2_val,
        threshold=sp2_thresh,
        confidence=addr_conf,
        description=f"Shoulder plane flat at address ({s2_val:.1f}°) — rounded shoulder pattern" if s2_val else "No data"
    ))

    return engine._build_result(
        name="back_c_posture",
        display_name="C-Posture at Address (Back View)",
        phase="address",
        signals=signals,
        description="Shoulders are slumped forward with rounded thoracic spine at address. "
                    "TPI identifies C-posture as causing restricted rotation and increased injury risk. "
                    "The back view clearly shows the roundedness of the upper back.",
        ball_flight="Restricted backswing, loss of rotation, compensatory shoulder plane issues",
        root_cause="Upper Crossed Syndrome — tight chest/lats, weak mid-back muscles. "
                   "Poor posture habits, clubs too short, lack of proper setup instruction.",
        source="TPI Swing Characteristics C-Posture; mytpi.com",
        measured_value=spine_at_addr,
        elite_benchmark="Spine angle 30-45° at address, not rounded/slumped",
    )


# ---------------------------------------------------------------------------
# FACE — LATE BUCKLE
# ---------------------------------------------------------------------------

def _face_late_buckle(engine: SyndromeEngine) -> SyndromeResult:
    """
    LATE BUCKLE: Lead knee buckles (loses extension) after impact in follow-through.
    TPI: "Lead knee collapses/buckles instead of post-up extension through impact."

    Signals:
      1. Lead knee angle DECREASES (more bent) post-impact vs at impact
      2. Lead knee at impact already shows insufficient extension vs address
      3. Trail knee fails to extend (companion — no proper drive)
    """
    imp_idx = engine._phases.get("impact_idx", engine.n * 3 // 4)
    addr_idx = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("follow_through")

    addr_lead = engine._get_attr_at("lead_knee_angle_deg", addr_idx)
    imp_lead  = engine._get_attr_at("lead_knee_angle_deg", imp_idx)

    # Post-impact window: a few frames after impact
    ft_start, ft_end = engine.phases_range("follow_through")
    win_s = max(imp_idx, ft_start)
    win_e = min(engine.n, win_s + max(3, (ft_end - win_s) // 2))
    post_lead = np.array([
        getattr(m, "lead_knee_angle_deg", 0) or 0.0
        for m in engine.metrics[win_s:win_e]
    ])
    post_conf = np.array([m.confidence for m in engine.metrics[win_s:win_e]])

    # The lead knee should EXTEND (angle approach 180°) through follow-through.
    # A buckle = minimum post-impact angle DROPS below impact value.
    min_post_lead = engine._high_conf_min(post_lead, post_conf) if len(post_lead) else None
    buckle_amount = None
    if min_post_lead is not None and imp_lead is not None and imp_lead > 0:
        buckle_amount = imp_lead - min_post_lead  # positive = bending more post-impact

    # Trail knee — also should be extending; if both fail, stronger signal
    addr_trail = engine._get_attr_at("trail_knee_angle_deg", addr_idx)
    post_trail = np.array([
        getattr(m, "trail_knee_angle_deg", 0) or 0.0
        for m in engine.metrics[win_s:win_e]
    ])
    max_post_trail = engine._high_conf_max(post_trail, post_conf) if len(post_trail) else None
    trail_extension = None
    if max_post_trail is not None and addr_trail is not None:
        # Should be greater than at address by impact follow-through
        trail_extension = max_post_trail - addr_trail

    # Impact extension deficit
    impact_deficit = None
    if addr_lead is not None and imp_lead is not None:
        # Elite: lead knee opens to ~155° at impact (per Murakami et al.)
        # Deficit = how far short of 155°
        impact_deficit = 155.0 - imp_lead

    signals = []

    # Signal 1: lead knee actively buckles post-impact
    s1_thresh = 8.0
    s1_val  = buckle_amount
    s1_fire = s1_val is not None and s1_val > s1_thresh
    signals.append(Signal(
        name="lead_knee_buckles_post_impact",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"Lead knee bends {s1_val:.1f}° MORE after impact than at impact" if s1_val is not None else "No data"
    ))

    # Signal 2: insufficient lead knee extension at impact
    s2_thresh = 15.0
    s2_val  = impact_deficit
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="insufficient_lead_extension_at_impact",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Lead knee {s2_val:.1f}° short of elite extension at impact" if s2_val is not None else "No data"
    ))

    # Signal 3: trail knee fails to extend through follow-through
    s3_thresh = 2.0  # should gain some extension vs address
    s3_val  = trail_extension
    s3_fire = s3_val is not None and s3_val < s3_thresh
    signals.append(Signal(
        name="trail_knee_no_drive",
        triggered=s3_fire,
        value=s3_val,
        threshold=s3_thresh,
        confidence=phase_conf,
        description=f"Trail knee gains only {s3_val:.1f}° extension — no drive" if s3_val is not None else "No data"
    ))

    return engine._build_result(
        name="late_buckle",
        display_name="Late Knee Buckle (Post-Impact)",
        phase="follow_through",
        signals=signals,
        description="The lead knee is collapsing/buckling AFTER impact instead of extending into "
                    "the classic post-up. TPI: elite golfers straighten the lead leg through impact "
                    "to use the ground reaction force. A late buckle wastes vertical force and creates "
                    "an inconsistent low point.",
        ball_flight="Loss of distance, weak contact, inconsistent strike pattern",
        root_cause="Weak glutes/quads, poor frontal-plane stability, anterior knee pain compensation. "
                   "Often paired with hanging back (no weight to push against).",
        source="TPI Swing Characteristics; Murakami et al. (MDPI 2022); Smart Golf Performance",
        measured_value=buckle_amount,
        elite_benchmark="Lead knee should EXTEND to ~155-170° at impact, hold through follow-through",
    )


# ---------------------------------------------------------------------------
# BACK — HANGING BACK
# ---------------------------------------------------------------------------

def _back_hanging_back(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW HANGING BACK: Hips stay back (trail-side) through impact.
    DTL confirmation — hip_slide_px fails to move toward target in downswing.

    Signals:
      1. Hip slide at impact is near-zero or still trail-side
      2. Spine angle still leans away from target at impact (compensation)
      3. Lead knee fails to extend at impact (no post-up because no weight on it)
    """
    addr_idx = engine._phases.get("addr_idx", 0)
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)
    phase_conf = engine._phase_conf("downswing")

    hip_at_imp = engine._get_attr_at("hip_slide_px", imp_idx)
    # hip_slide_px convention in metrics_back.py: positive = toward target.
    # If still ~0 or negative at impact, hips are hanging back.

    addr_sp = engine._get_attr_at("spine_angle_deg", addr_idx)
    imp_sp  = engine._get_attr_at("spine_angle_deg", imp_idx)
    spine_change = None
    if addr_sp is not None and imp_sp is not None:
        spine_change = imp_sp - addr_sp  # large change can indicate lean-back compensation

    addr_knee = engine._get_attr_at("lead_knee_flex_deg", addr_idx)
    imp_knee  = engine._get_attr_at("lead_knee_flex_deg", imp_idx)
    # In metrics_back.py 180° = straight, lower = more flexed.
    # Hanging back -> lead knee fails to extend -> imp_knee NOT increasing vs addr_knee
    knee_extension = None
    if addr_knee is not None and imp_knee is not None:
        knee_extension = imp_knee - addr_knee  # positive = extending

    signals = []

    # Signal 1: hips fail to slide toward target
    s1_thresh = 25.0  # px toward target
    s1_val  = hip_at_imp
    s1_fire = s1_val is not None and s1_val < s1_thresh
    signals.append(Signal(
        name="hips_stay_back",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"Hip slide at impact only {s1_val:.0f}px toward target" if s1_val is not None else "No data"
    ))

    # Signal 2: spine angle changes a lot at impact (lean-back compensation)
    s2_thresh = 8.0
    s2_val  = spine_change
    s2_fire = s2_val is not None and abs(s2_val) > s2_thresh
    signals.append(Signal(
        name="lean_back_compensation",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Spine angle changed {s2_val:.1f}° at impact — leaning away" if s2_val is not None else "No data"
    ))

    # Signal 3: lead knee fails to extend
    s3_thresh = 3.0  # min extension gain expected (deg)
    s3_val  = knee_extension
    s3_fire = s3_val is not None and s3_val < s3_thresh
    signals.append(Signal(
        name="lead_knee_no_post_up",
        triggered=s3_fire,
        value=s3_val,
        threshold=s3_thresh,
        confidence=phase_conf,
        description=f"Lead knee extension only {s3_val:.1f}° at impact — no post-up" if s3_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_hanging_back",
        display_name="Hanging Back (Back View)",
        phase="downswing",
        signals=signals,
        description="From the back view, the hips never move toward the target in the downswing — "
                    "the body 'hangs back' on the trail side through impact. Elite golfers shift "
                    "their pressure aggressively to the lead foot in the downswing.",
        ball_flight="Fat, thin, weak shots; pushes; loss of distance",
        root_cause="Fear of hitting fat, reverse pivot pattern, weak glutes/quads",
        source="TPI Swing Characteristics; Fleisig Biomechanics",
        measured_value=hip_at_imp,
        elite_benchmark=">25px hip translation toward target at impact (DTL view)",
    )


# ---------------------------------------------------------------------------
# BACK — REVERSE PIVOT
# ---------------------------------------------------------------------------

def _back_reverse_pivot(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW REVERSE PIVOT: Spine tilts TOWARD the target during the backswing
    (opposite of correct trail-side load). Best confirmed from DTL by spine angle
    and head moving toward the target.

    Signals:
      1. Spine angle change moves toward the target during backswing
      2. Hip slide goes toward the target in backswing (no trail load)
      3. Head moves toward the target in backswing (instead of behind ball)
    """
    addr_idx = engine._phases.get("addr_idx", 0)
    top_idx  = engine._phases.get("top_idx",  engine.n // 2)
    phase_conf = engine._phase_conf("backswing")

    addr_sp = engine._get_attr_at("spine_angle_deg", addr_idx)
    top_sp  = engine._get_attr_at("spine_angle_deg", top_idx)
    spine_change = None
    if addr_sp is not None and top_sp is not None:
        # In DTL metrics_back: spine_angle_deg = forward bend from vertical.
        # Reverse pivot reads as a substantial decrease in forward bend AND
        # measurable lateral lean — here we use magnitude as a proxy.
        spine_change = top_sp - addr_sp

    hip_at_top = engine._get_attr_at("hip_slide_px", top_idx)
    # If positive (toward target) in backswing, that's a reverse pivot signature.

    head_at_top = engine._get_attr_at("head_movement_px", top_idx)
    # head_movement_px is overall displacement — large at top during backswing is suspect
    # combined with spine + hip direction.

    signals = []

    # Signal 1: spine tilts toward target (forward bend decreases noticeably)
    s1_thresh = 6.0
    s1_val  = spine_change
    s1_fire = s1_val is not None and s1_val < -s1_thresh  # less forward bend = leaning back/target-side
    signals.append(Signal(
        name="spine_tilts_target_back",
        triggered=s1_fire,
        value=s1_val,
        threshold=-s1_thresh,
        confidence=phase_conf,
        description=f"Spine angle changes {s1_val:.1f}° (less forward bend) at top — leaning toward target" if s1_val is not None else "No data"
    ))

    # Signal 2: hips slide TOWARD target in backswing (wrong direction)
    s2_thresh = 10.0
    s2_val  = hip_at_top
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="hips_slide_target_in_backswing",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Hips already {s2_val:.0f}px toward target at top — should load trail" if s2_val is not None else "No data"
    ))

    # Signal 3: head moves a lot at top (proxy — large displacement suggests sway/reverse)
    s3_thresh = 35.0
    s3_val  = head_at_top
    s3_fire = s3_val is not None and s3_val > s3_thresh
    signals.append(Signal(
        name="head_moves_in_backswing",
        triggered=s3_fire,
        value=s3_val,
        threshold=s3_thresh,
        confidence=phase_conf,
        description=f"Head displaced {s3_val:.0f}px at top — consistent with reverse pivot" if s3_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_reverse_pivot",
        display_name="Reverse Pivot (Back View)",
        phase="backswing",
        signals=signals,
        description="Body is loading the LEAD side during the backswing — opposite of correct. "
                    "From DTL the spine tilts toward the target and the hips drift forward instead "
                    "of staying centered or sliding back. This eliminates stored power.",
        ball_flight="Steep downswing, pulls, slices, fat contact, distance loss",
        root_cause="Trying to keep the head still while restricting trail-side load; weak trail glute",
        source="TPI Swing Characteristics; Richards et al. 1985",
        measured_value=spine_change,
        elite_benchmark="Spine should hold or increase forward bend, hips load trail-side at top",
    )


# ---------------------------------------------------------------------------
# BACK — FLYING ELBOW
# ---------------------------------------------------------------------------

def _back_flying_elbow(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW FLYING ELBOW: Trail elbow excessively elevated at top of backswing.
    Visible from DTL as trail arm angle separating from lead arm angle.

    Signals:
      1. Trail arm angle much higher than lead arm angle at top
      2. Trail arm angle elevated relative to address baseline
    """
    top_idx  = engine._phases.get("top_idx", engine.n // 2)
    addr_idx = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("backswing")

    trail_top  = engine._get_attr_at("trail_arm_angle_deg", top_idx)
    lead_top   = engine._get_attr_at("lead_arm_angle_deg",  top_idx)
    trail_addr = engine._get_attr_at("trail_arm_angle_deg", addr_idx)

    arm_diff = None
    if trail_top is not None and lead_top is not None:
        arm_diff = abs(trail_top - lead_top)

    trail_change = None
    if trail_top is not None and trail_addr is not None:
        trail_change = abs(trail_top - trail_addr)

    signals = []

    # Signal 1: trail arm separated from lead arm at top
    s1_thresh = 30.0
    s1_val  = arm_diff
    s1_fire = s1_val is not None and s1_val > s1_thresh
    signals.append(Signal(
        name="trail_arm_separated_from_lead",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"Trail arm {s1_val:.1f}° away from lead arm at top — disconnected" if s1_val is not None else "No data"
    ))

    # Signal 2: trail arm change from address is excessive
    s2_thresh = 80.0
    s2_val  = trail_change
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="trail_arm_overswing",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Trail arm rotated {s2_val:.1f}° from address — overswung/elevated" if s2_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_flying_elbow",
        display_name="Flying Trail Elbow (Back View)",
        phase="backswing",
        signals=signals,
        description="From DTL the trail elbow flies away from the body at the top — the trail arm "
                    "angle separates significantly from the lead arm. TPI: this disconnects the arms "
                    "from the body and promotes an over-the-top downswing.",
        ball_flight="Over-the-top pulls, slices, steep angle of attack",
        root_cause="Limited shoulder external rotation, arms lifting instead of rotating, lat tightness",
        source="TPI Swing Characteristics; BioSwing Dynamics",
        measured_value=arm_diff,
        elite_benchmark="Trail arm should track within ~25-30° of lead arm angle at top",
    )


# ---------------------------------------------------------------------------
# BACK — CASTING / SCOOPING
# ---------------------------------------------------------------------------

def _back_casting(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW CASTING / SCOOPING: Wrist angles released early in downswing.
    DTL confirmation — lead arm collapses + wrist hinge returns to address too early.

    Signals:
      1. Wrist hinge released early in downswing (returns to address before midway)
      2. Lead arm collapses (lead_arm_angle_deg decreases sharply in early downswing)
    """
    top_idx = engine._phases.get("top_idx", engine.n // 2)
    addr_idx = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("downswing")

    wrist_top  = engine._get_attr_at("wrist_hinge_deg", top_idx)
    wrist_addr = engine._get_attr_at("wrist_hinge_deg", addr_idx)
    ds_wrist, wr_conf = engine._phase_metrics("downswing", "wrist_hinge_deg")

    release_pct = None
    early_release = False
    if wrist_top is not None and wrist_addr is not None and len(ds_wrist) > 3:
        hinge_range = abs(wrist_top - wrist_addr)
        halfway     = len(ds_wrist) // 2
        wrist_early = engine._high_conf_mean(ds_wrist[:halfway], wr_conf[:halfway])
        if wrist_early is not None and hinge_range > 5:
            release_pct = abs(wrist_early - wrist_top) / hinge_range
            early_release = release_pct > 0.60

    # Lead arm collapse: lead_arm_angle_deg should remain extended through early downswing
    lead_top = engine._get_attr_at("lead_arm_angle_deg", top_idx)
    ds_lead, ld_conf = engine._phase_metrics("downswing", "lead_arm_angle_deg")
    halfway_idx = max(1, len(ds_lead) // 2)
    early_lead = engine._high_conf_mean(ds_lead[:halfway_idx], ld_conf[:halfway_idx])
    lead_collapse = None
    if lead_top is not None and early_lead is not None:
        lead_collapse = abs(lead_top - early_lead)

    signals = []

    s1_thresh = 0.60
    s1_val  = release_pct
    s1_fire = s1_val is not None and s1_val > s1_thresh
    signals.append(Signal(
        name="wrist_released_early_back",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"{s1_val*100:.0f}% of wrist hinge released in first half of downswing" if s1_val is not None else "No data"
    ))

    s2_thresh = 25.0
    s2_val  = lead_collapse
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="lead_arm_collapse_back",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Lead arm angle shifted {s2_val:.1f}° early in downswing — collapsing" if s2_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_casting",
        display_name="Casting / Early Release (Back View)",
        phase="downswing",
        signals=signals,
        description="From DTL, the wrists are unhinging and the lead arm is breaking down too early "
                    "in the downswing. Lag is being burned at the top of the descent instead of held "
                    "into the strike zone.",
        ball_flight="Weak contact, high spin, scooped divots, distance loss",
        root_cause="Initiating downswing with hands rather than body; instinct to lift the ball",
        source="TPI Swing Characteristics; HackMotion; Fleisig",
        measured_value=release_pct,
        elite_benchmark="Wrist hinge maintained until hands reach hip height in downswing",
    )


# ---------------------------------------------------------------------------
# BACK — CHICKEN WING
# ---------------------------------------------------------------------------

def _back_chicken_wing(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW CHICKEN WING: Lead elbow / lead arm breaks down through impact.
    From DTL we read lead_arm_angle_deg drop around impact compared to address.

    Signals:
      1. Lead arm angle near impact is well off the address baseline
      2. Wrist hinge added through impact (scooping accompanies chicken wing)
    """
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)
    addr_idx = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("downswing")

    # Window around impact
    win_s = max(0, imp_idx - 4)
    win_e = min(engine.n, imp_idx + 5)
    lead_win = np.array([
        getattr(m, "lead_arm_angle_deg", 0) or 0.0
        for m in engine.metrics[win_s:win_e]
    ])
    win_conf = np.array([m.confidence for m in engine.metrics[win_s:win_e]])

    addr_lead = engine._get_attr_at("lead_arm_angle_deg", addr_idx)
    impact_lead = engine._high_conf_mean(lead_win, win_conf) if len(lead_win) else None
    lead_deviation = None
    if addr_lead is not None and impact_lead is not None:
        lead_deviation = abs(impact_lead - addr_lead)

    wrist_imp  = engine._get_attr_at("wrist_hinge_deg", imp_idx)
    wrist_addr = engine._get_attr_at("wrist_hinge_deg", addr_idx)
    wrist_added = None
    if wrist_imp is not None and wrist_addr is not None:
        # Hinge added through impact = scooping
        wrist_added = wrist_imp - wrist_addr

    signals = []

    s1_thresh = 25.0
    s1_val  = lead_deviation
    s1_fire = s1_val is not None and s1_val > s1_thresh
    signals.append(Signal(
        name="lead_arm_breakdown_at_impact",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"Lead arm angle deviates {s1_val:.1f}° from address at impact" if s1_val is not None else "No data"
    ))

    s2_thresh = 8.0
    s2_val  = wrist_added
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="wrist_scooping_back",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Wrist hinge {s2_val:.1f}° added at impact — scooping" if s2_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_chicken_wing",
        display_name="Chicken Wing (Back View)",
        phase="impact",
        signals=signals,
        description="From DTL the lead arm breaks down through impact instead of staying long. "
                    "This narrows the arc, opens the face, and saps power. Frequently paired with "
                    "casting/scooping.",
        ball_flight="Weak slices, pulls, inconsistent contact, distance loss",
        root_cause="Trail-arm dominance; instinct to lift the ball; restricted wrist mobility",
        source="TPI Swing Characteristics; HackMotion",
        measured_value=lead_deviation,
        elite_benchmark="Lead arm should remain near full extension through impact (within ~10° of address)",
    )


# ---------------------------------------------------------------------------
# BACK — FORWARD LUNGE
# ---------------------------------------------------------------------------

def _back_forward_lunge(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW FORWARD LUNGE: Upper body lunges toward target / head moves forward
    of hips in transition (the DTL "head fwd of hips" signature from TPI table).

    Signals:
      1. Head moves forward (toward target) significantly from top to impact
      2. Head_forward_px increases vs address (head ahead of hip line)
    """
    top_idx = engine._phases.get("top_idx", engine.n // 2)
    imp_idx = engine._phases.get("impact_idx", engine.n * 3 // 4)
    addr_idx = engine._phases.get("addr_idx", 0)
    phase_conf = engine._phase_conf("downswing")

    # head_movement_px is overall displacement (not directional) — use it for magnitude.
    head_top = engine._get_attr_at("head_movement_px", top_idx)
    head_imp = engine._get_attr_at("head_movement_px", imp_idx)
    head_surge = None
    if head_top is not None and head_imp is not None:
        head_surge = head_imp - head_top

    # head_forward_px = how far head is in front of hips (forward press). Increasing = lunging.
    addr_hf = engine._get_attr_at("head_forward_px", addr_idx)
    imp_hf  = engine._get_attr_at("head_forward_px", imp_idx)
    head_forward_gain = None
    if addr_hf is not None and imp_hf is not None:
        head_forward_gain = imp_hf - addr_hf

    signals = []

    s1_thresh = 30.0
    s1_val  = head_surge
    s1_fire = s1_val is not None and s1_val > s1_thresh
    signals.append(Signal(
        name="head_surges_top_to_impact",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"Head moves {s1_val:.0f}px from top to impact — lunging forward" if s1_val is not None else "No data"
    ))

    s2_thresh = 25.0
    s2_val  = head_forward_gain
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="head_forward_of_hips",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Head {s2_val:.0f}px more forward of hip line at impact vs address" if s2_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_forward_lunge",
        display_name="Forward Lunge (Back View)",
        phase="downswing",
        signals=signals,
        description="From DTL the head moves ahead of the hip line in transition — the upper body "
                    "lunges toward the target instead of the lower body leading. TPI flags this as "
                    "an upper-body-driven downswing.",
        ball_flight="Pulls, over-the-top, thin contact, loss of lag",
        root_cause="Upper body initiates downswing before the lower body; weak transition sequencing",
        source="TPI Swing Characteristics; Postureffect",
        measured_value=head_surge,
        elite_benchmark="Head stays behind ball; lower body leads the downswing",
    )


# ---------------------------------------------------------------------------
# BACK — LATE BUCKLE
# ---------------------------------------------------------------------------

def _back_late_buckle(engine: SyndromeEngine) -> SyndromeResult:
    """
    BACK VIEW LATE BUCKLE: Lead knee buckles after impact in follow-through.
    Uses lead_knee_flex_deg from BackMetrics (180° = straight; lower = more flexed).

    Signals:
      1. Lead knee flex decreases (knee bends MORE) post-impact vs at impact
      2. Lead knee already failed to extend at impact (deficit vs address)
    """
    addr_idx = engine._phases.get("addr_idx", 0)
    imp_idx  = engine._phases.get("impact_idx", engine.n * 3 // 4)
    phase_conf = engine._phase_conf("follow_through")

    addr_lead = engine._get_attr_at("lead_knee_flex_deg", addr_idx)
    imp_lead  = engine._get_attr_at("lead_knee_flex_deg", imp_idx)

    ft_start, ft_end = engine.phases_range("follow_through")
    win_s = max(imp_idx, ft_start)
    win_e = min(engine.n, win_s + max(3, (ft_end - win_s) // 2))
    post_lead = np.array([
        getattr(m, "lead_knee_flex_deg", 0) or 0.0
        for m in engine.metrics[win_s:win_e]
    ])
    post_conf = np.array([m.confidence for m in engine.metrics[win_s:win_e]])

    min_post = engine._high_conf_min(post_lead, post_conf) if len(post_lead) else None
    buckle_amount = None
    if min_post is not None and imp_lead is not None and imp_lead > 0:
        buckle_amount = imp_lead - min_post  # positive = bending MORE post-impact

    impact_deficit = None
    if addr_lead is not None and imp_lead is not None:
        # Expect lead knee to extend at impact (imp > addr). If imp <= addr it's not extending.
        impact_deficit = addr_lead - imp_lead  # positive if knee bent more at impact than address (bad)

    signals = []

    s1_thresh = 8.0
    s1_val  = buckle_amount
    s1_fire = s1_val is not None and s1_val > s1_thresh
    signals.append(Signal(
        name="lead_knee_buckles_post_impact_back",
        triggered=s1_fire,
        value=s1_val,
        threshold=s1_thresh,
        confidence=phase_conf,
        description=f"Lead knee bends {s1_val:.1f}° MORE after impact than at impact" if s1_val is not None else "No data"
    ))

    s2_thresh = 3.0
    s2_val  = impact_deficit
    s2_fire = s2_val is not None and s2_val > s2_thresh
    signals.append(Signal(
        name="no_post_up_extension",
        triggered=s2_fire,
        value=s2_val,
        threshold=s2_thresh,
        confidence=phase_conf,
        description=f"Lead knee {s2_val:.1f}° more flexed at impact than address — no post-up" if s2_val is not None else "No data"
    ))

    return engine._build_result(
        name="back_late_buckle",
        display_name="Late Knee Buckle (Back View)",
        phase="follow_through",
        signals=signals,
        description="From DTL the lead knee continues to flex AFTER impact instead of extending into "
                    "the classic post-up. Elite golfers straighten the lead leg through impact to "
                    "use ground reaction force.",
        ball_flight="Loss of distance, weak contact, inconsistent low point",
        root_cause="Weak glutes/quads, poor frontal-plane stability, hanging-back pattern",
        source="TPI Swing Characteristics; Murakami et al. (MDPI 2022)",
        measured_value=buckle_amount,
        elite_benchmark="Lead knee should EXTEND through impact (less flex than address) and hold",
    )


# ---------------------------------------------------------------------------
# Syndrome registries
# ---------------------------------------------------------------------------

FACE_SYNDROMES = [
    _face_sway,
    _face_slide,
    _face_hanging_back,
    _face_reverse_pivot,
    _face_early_extension,
    _face_over_the_top,
    _face_casting,
    _face_chicken_wing,
    _face_flying_elbow,
    _face_loss_of_posture,
    _face_reverse_spine,
    _face_forward_lunge,
    _face_late_buckle,
]

BACK_SYNDROMES = [
    _back_early_extension,
    _back_reverse_spine,
    _back_flat_shoulder_plane,
    _back_sway,
    _back_slide,
    _back_over_the_top,
    _back_loss_of_posture,
    _back_c_posture,
    _back_hanging_back,
    _back_reverse_pivot,
    _back_flying_elbow,
    _back_casting,
    _back_chicken_wing,
    _back_forward_lunge,
    _back_late_buckle,
]
