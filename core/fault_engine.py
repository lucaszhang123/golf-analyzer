"""
fault_engine.py — Simple, direct swing fault detection.

Six faults, each backed by large reliable body landmarks.
No syndrome complexity, no multi-signal voting — each fault is a
straightforward geometric check against the address baseline.

FAULTS IMPLEMENTED:
  1. SPINE_ANGLE_CHANGE  — spine tilts/straightens from address through swing
                           (face-on: lateral tilt; DTL: forward bend)
  2. REVERSE_SPINE       — spine tilts TOWARD target in backswing (face-on)
  3. HIP_SWAY_SLIDE      — hips drift excessively laterally (face-on)
  4. EARLY_EXTENSION     — hips thrust toward ball in downswing (DTL)
  5. HEAD_MOVEMENT       — head drifts laterally or rotates incorrectly
  6. FLAT_SHOULDER_PLANE — shoulders turn too horizontally at top (DTL)

DESIGN:
  - Every fault is measured relative to the golfer's OWN address position.
    No assumed pixel sizes or resolution-dependent constants.
  - Thresholds are expressed as multiples of body-proportion measurements
    (e.g. hip width, body height) so they scale automatically.
  - Each fault returns a severity 0.0-1.0 and a plain description.
  - Phase indices (address, top, impact) are passed in from the caller.

USAGE:
    from core.fault_engine import FaultEngine, FaultResult

    engine = FaultEngine(
        face_metrics=face_metrics,   # List[SwingMetrics] or None
        back_metrics=back_metrics,   # List[BackMetrics]  or None
    )
    results = engine.detect()        # List[FaultResult]
    print(engine.summary())
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

try:
    from .phase_align import PhaseAligner
    _ALIGNER_AVAILABLE = True
except ImportError:
    PhaseAligner = None
    _ALIGNER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FaultResult:
    name: str
    display_name: str
    detected: bool
    severity: float          # 0.0 – 1.0
    severity_label: str      # "mild" | "moderate" | "severe"
    phase: str               # when in the swing this occurs
    measured: Optional[float]
    threshold: Optional[float]
    description: str
    cause: str
    ball_flight: str
    view: str                # "face" | "back" | "both"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _label(s: float) -> str:
    if s < 0.40: return "mild"
    if s < 0.70: return "moderate"
    return "severe"


# ---------------------------------------------------------------------------
# Phase detection — same logic as syndrome_engine but self-contained
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray, w: int = 3) -> np.ndarray:
    if w < 2 or len(arr) < w:
        return arr.copy()
    return np.convolve(arr, np.ones(w) / w, mode="same")


def _phases_to_dict(p: dict) -> dict:
    """Serialize phase dict for JSON output — split ranges into [start, end]."""
    out = {
        "address_frame":        p.get("addr_idx"),
        "top_frame":            p.get("top_idx"),
        "impact_frame":         p.get("impact_idx"),
        "backswing_range":      list(p.get("backswing_range",      (0, 0))),
        "downswing_range":      list(p.get("downswing_range",      (0, 0))),
        "followthrough_range":  list(p.get("followthrough_range",  (0, 0))),
        "impact_signals":       p.get("_impact_signals"),
    }
    return out


def _detect_phases(metrics: list) -> Dict[str, int]:
    """
    Returns dict with addr_idx, top_idx, impact_idx plus phase RANGES:
      address_range, backswing_range, downswing_range, followthrough_range
    Each range is (start_frame, end_frame_exclusive).

    Address is a single frame (the most stable frame in the first third).
    Backswing spans (address+1, top].
    Top is a single frame.
    Downswing spans (top+1, impact].
    Impact is a single frame.
    Follow-through spans (impact+1, end).

    IMPACT DETECTION (multi-signal):
      Three independent signals vote on the impact frame. If at least 2
      agree within +/-3 frames, that consensus is impact. Otherwise the
      rotation-only estimate is the fallback.

      Signal A — shoulder rotation drops to <55% of its top value
                 (shoulders unwinding through the ball)
      Signal B — hand velocity peak and subsequent rapid deceleration
                 (clubhead/hands at max speed at impact)
      Signal C — hand height returns to within 60% of the address→top arc
                 (hands back down near where they started)
    """
    n = len(metrics)
    if n == 0:
        return {
            "addr_idx": 0, "top_idx": 0, "impact_idx": 0,
            "address_range": (0, 1),
            "backswing_range": (0, 0),
            "downswing_range": (0, 0),
            "followthrough_range": (0, 0),
        }

    conf     = np.array([m.confidence for m in metrics])
    hand_y   = np.array([getattr(m, "hand_height_y", 0) or 0.0 for m in metrics])
    weight   = np.array([getattr(m, "weight_shift",  0) or 0.0 for m in metrics])
    shoulder = np.array([getattr(m, "shoulder_rotation_deg", 0) or 0.0 for m in metrics])
    weight_s = _smooth(weight, 3)
    shld_s   = _smooth(shoulder, max(3, n // 15))
    hand_s   = _smooth(hand_y, max(3, n // 20))

    # ─── ADDRESS ────────────────────────────────────────────────────────────
    cutoff = max(3, n // 3)
    win    = max(2, n // 20)
    addr   = 0
    best_v = float("inf")
    for i in range(max(0, cutoff - win)):
        v = float(np.var(weight_s[i:i+win]))
        c = float(np.mean(conf[i:i+win]))
        if v < best_v and c > 0.5:
            best_v = v
            addr   = i + win // 2

    # ─── TOP OF BACKSWING (multi-signal) ────────────────────────────────────
    s_start = addr + 1
    s_end   = min(n, addr + int((n - addr) * 0.80))

    top_hand = addr + 1
    if s_end > s_start and np.any(hand_y[s_start:s_end] > 0):
        region = hand_y[s_start:s_end]
        c_reg  = conf[s_start:s_end]
        masked = np.where(c_reg > 0.4, region, np.max(region) + 1)
        top_hand = s_start + int(np.argmin(masked))

    top_rot = addr + 1
    if s_end > addr + 1:
        weighted = shld_s[addr+1:s_end] * conf[addr+1:s_end]
        top_rot  = addr + 1 + int(np.argmax(weighted))

    addr_h   = float(np.mean(hand_y[:max(1, addr+1)]))
    hand_arc = max(0.0, addr_h - float(hand_y[top_hand])) if np.any(hand_y > 0) else 0

    if hand_arc > 100:
        top = top_hand
    elif hand_arc > 50 and abs(top_hand - top_rot) < max(3, n // 5):
        top = top_hand
    else:
        top = top_rot
    top = min(top, int(n * 0.75))

    # ─── IMPACT (multi-signal — three independent estimates) ─────────────────
    # Search window: after top, before end (capped at 95%)
    search_lo = top + 1
    search_hi = max(top + 2, int(n * 0.95))

    # Signal A — shoulder rotation drops to < 55% of value at top
    rot_at_top = float(shld_s[top]) if top < n else 0.0
    impact_rot = None
    if rot_at_top > 5:
        thresh = rot_at_top * 0.55
        for i in range(search_lo, search_hi):
            if shld_s[i] < thresh and conf[i] > 0.4:
                impact_rot = i
                break

    # Signal B — hand velocity peak + deceleration
    # Velocity = frame-to-frame change in hand_y (vertical) — magnitude only.
    # Hands accelerate through downswing, peak near impact, decelerate after.
    impact_vel = None
    if search_hi > search_lo + 2:
        vy = np.abs(np.diff(hand_s))  # length n-1
        vy_window = vy[max(0, search_lo-1):search_hi]
        if len(vy_window) > 3:
            peak_off = int(np.argmax(vy_window))
            peak_frame = max(0, search_lo - 1) + peak_off
            # After peak, look for first frame where velocity drops below
            # 60% of peak value — that's the deceleration point ~= impact
            peak_val = float(vy_window[peak_off])
            if peak_val > 1.0:  # at least some motion
                for j in range(peak_frame + 1, min(n - 1, search_hi)):
                    if vy[j] < peak_val * 0.6 and conf[j] > 0.4:
                        impact_vel = j
                        break
                if impact_vel is None:
                    # No clear drop — use peak itself as impact estimate
                    impact_vel = peak_frame + 1

    # Signal C — hand height returns 60% of the way from top to address
    impact_hand = None
    top_h = float(hand_s[top]) if top < n else 0.0
    arc_size = addr_h - top_h
    if arc_size > 20:  # only if hands actually moved
        target_h = top_h + arc_size * 0.60
        for i in range(search_lo, search_hi):
            if hand_s[i] >= target_h and conf[i] > 0.4:
                impact_hand = i
                break

    # ─── CONSENSUS ───────────────────────────────────────────────────────────
    candidates = [c for c in (impact_rot, impact_vel, impact_hand) if c is not None]

    if len(candidates) >= 2:
        # Look for any pair within +/-3 frames of each other.
        # If found, average them. If all three agree, use the median.
        candidates.sort()
        if len(candidates) == 3 and (candidates[2] - candidates[0]) <= 6:
            impact = int(round(np.median(candidates)))
        else:
            best_pair = None
            best_spread = 999
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    spread = candidates[j] - candidates[i]
                    if spread <= 3 and spread < best_spread:
                        best_pair = (candidates[i], candidates[j])
                        best_spread = spread
            if best_pair:
                impact = int(round((best_pair[0] + best_pair[1]) / 2))
            else:
                # No consensus — fall back to rotation signal (most reliable single)
                impact = impact_rot if impact_rot is not None else candidates[0]
    elif len(candidates) == 1:
        impact = candidates[0]
    else:
        # No signal fired — geometric fallback
        impact = top + max(1, int((n - top) * 0.50))

    impact = min(max(impact, top + 1), int(n * 0.95))

    # ─── PHASE RANGES ────────────────────────────────────────────────────────
    return {
        "addr_idx":             addr,
        "top_idx":              top,
        "impact_idx":           impact,
        "address_range":        (max(0, addr-2), min(n, addr+3)),
        "backswing_range":      (addr + 1, top + 1),
        "downswing_range":      (top + 1, impact + 1),
        "followthrough_range":  (impact + 1, n),
        "_impact_signals": {
            "rotation": impact_rot,
            "velocity": impact_vel,
            "hand_return": impact_hand,
        },
    }


# ---------------------------------------------------------------------------
# Helper: get smoothed attribute array with confidence mask
# ---------------------------------------------------------------------------

def _get(metrics, attr, default=0.0):
    vals = []
    for m in metrics:
        v = getattr(m, attr, None)
        vals.append(float(v) if v is not None else default)
    return np.array(vals)


def _get_or_nan(metrics, attr):
    # Returns NaN for None — lets callers skip truly missing frames
    vals = []
    for m in metrics:
        v = getattr(m, attr, None)
        vals.append(float(v) if v is not None else float("nan"))
    return np.array(vals)


def _phase_mean(arr, conf, start, end, min_conf=0.55):
    """Mean of arr[start:end] for frames above min_conf."""
    if end <= start:
        return None
    window = arr[start:end]
    c_win  = conf[start:end]
    mask   = c_win >= min_conf
    if mask.sum() < 2:
        return None
    return float(np.mean(window[mask]))


def _phase_max(arr, conf, start, end, min_conf=0.55):
    if end <= start:
        return None
    mask = conf[start:end] >= min_conf
    if mask.sum() < 2:
        return None
    return float(np.max(arr[start:end][mask]))


def _at(arr, idx):
    """Value at index, or None if out of range."""
    if 0 <= idx < len(arr):
        v = arr[idx]
        return float(v) if v is not None else None
    return None


def _window_mean(arr, conf, idx, half=2, min_conf=0.55):
    """
    Median of arr in a +/- half window around idx, using frames above
    min_conf. Makes a phase baseline (address/top/impact) robust to a
    single bad-pose frame. Widens the window if no confident frames found.
    """
    n = len(arr)
    if n == 0 or not (0 <= idx < n):
        return None
    for h in (half, half + 2, half + 4):
        lo = max(0, idx - h)
        hi = min(n, idx + h + 1)
        window = arr[lo:hi]
        cwin   = conf[lo:hi]
        mask   = cwin >= min_conf
        if mask.sum() >= 1:
            return float(np.median(window[mask]))
    return None


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class FaultEngine:
    """
    Detects six swing faults from face-on and/or DTL metrics.

    Parameters
    ----------
    face_metrics : List[SwingMetrics] or None
    back_metrics : List[BackMetrics]  or None
    """

    # Detection threshold: minimum severity to report a fault
    MIN_SEVERITY = 0.25

    def __init__(
        self,
        face_metrics: Optional[list] = None,
        back_metrics: Optional[list] = None,
    ):
        self.face = face_metrics or []
        self.back = back_metrics or []

        self.fp = _detect_phases(self.face) if self.face else {"addr_idx": 0, "top_idx": 0, "impact_idx": 0}
        self.bp = _detect_phases(self.back) if self.back else {"addr_idx": 0, "top_idx": 0, "impact_idx": 0}

        # Pre-compute confidence arrays
        self.fc = _get(self.face, "confidence") if self.face else np.array([])
        self.bc = _get(self.back, "confidence") if self.back else np.array([])

        # Build cross-view aligner when both streams are present.
        # Allows individual fault checks to opt into reconcile_metric()
        # for cross-view validation without disrupting single-view logic.
        self.aligner = None
        if _ALIGNER_AVAILABLE and self.face and self.back:
            try:
                self.aligner = PhaseAligner(
                    face_metrics=self.face,
                    back_metrics=self.back,
                )
            except Exception:
                self.aligner = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self) -> List[FaultResult]:
        results = []
        checks = [
            self._spine_angle_change,
            self._spine_angle_change_downswing,
            self._reverse_spine,
            self._hip_sway_slide,
            self._early_extension,
            self._hip_shift_forward,
            self._head_movement,
            self._low_x_factor,
            self._insufficient_shoulder_turn,
            self._lead_knee_extension,
        ]
        for check in checks:
            try:
                r = check()
                if r is not None and r.severity >= self.MIN_SEVERITY:
                    results.append(r)
            except Exception as e:
                pass  # never crash on a single fault check

        results.sort(key=lambda r: -r.severity)
        return results

    def summary(self) -> str:
        results = self.detect()
        phases_f = self.fp
        phases_b = self.bp

        def _ph_line(p, label):
            if not p or p.get("addr_idx", 0) == p.get("impact_idx", 0):
                return f"{label} : no swing detected"
            sig = p.get("_impact_signals", {})
            sig_str = (f"  (impact signals: rot={sig.get('rotation')}, "
                       f"vel={sig.get('velocity')}, hand={sig.get('hand_return')})"
                       if sig else "")
            return (
                f"{label} | "
                f"address f{p['addr_idx']}  |  "
                f"backswing f{p['backswing_range'][0]}-{p['backswing_range'][1]-1}  |  "
                f"top f{p['top_idx']}  |  "
                f"downswing f{p['downswing_range'][0]}-{p['downswing_range'][1]-1}  |  "
                f"impact f{p['impact_idx']}  |  "
                f"follow-through f{p['followthrough_range'][0]}-{p['followthrough_range'][1]-1}"
                + sig_str
            )

        lines = [
            "",
            "Swing Fault Report",
            "=" * 50,
            _ph_line(phases_f, "Face"),
            _ph_line(phases_b, "Back"),
            "",
        ]

        if self.aligner is not None and self.aligner.is_aligned():
            q = self.aligner.quality
            lines.append(
                f"Cross-view alignment: confidence={q.confidence:.0%}  "
                f"fps_ratio(back/face)={q.fps_ratio:.2f}"
            )
            lines.append("")

        if not results:
            lines.append("No significant faults detected.")
            return "\n".join(lines)

        lines.append(f"Faults detected: {len(results)}\n")
        for i, r in enumerate(results, 1):
            val = f"{r.measured:.1f}" if r.measured is not None else "N/A"
            thr = f"{r.threshold:.1f}" if r.threshold is not None else "N/A"
            lines += [
                f"{i}. [{r.severity_label.upper()}] {r.display_name}",
                f"   Phase      : {r.phase}",
                f"   Measured   : {val}  Threshold: {thr}",
                f"   View       : {r.view}",
                f"   Detail     : {r.description}",
                f"   Cause      : {r.cause}",
                f"   Ball flight: {r.ball_flight}",
                "",
            ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "faults":      [r.to_dict() for r in self.detect()],
            "phases_face": _phases_to_dict(self.fp),
            "phases_back": _phases_to_dict(self.bp),
        }

    def phase_range(self, name: str, view: str = "face") -> tuple:
        """
        Return (start, end_exclusive) frame range for a named phase.
        name : "address" | "backswing" | "downswing" | "impact" | "follow_through"
        view : "face" | "back"
        """
        p = self.fp if view == "face" else self.bp
        key_map = {
            "address":        "address_range",
            "backswing":      "backswing_range",
            "downswing":      "downswing_range",
            "follow_through": "followthrough_range",
        }
        if name in key_map:
            return p.get(key_map[name], (0, 0))
        if name == "impact":
            idx = p.get("impact_idx", 0)
            return (max(0, idx-2), idx + 3)
        return (0, 0)

    # ------------------------------------------------------------------
    # Cross-view reconciliation helper (uses PhaseAligner when available)
    # ------------------------------------------------------------------
    def reconcile(self, attr: str, face_frame_idx: int, prefer: str = "auto"):
        """
        Return the best estimate for `attr` at the given face frame,
        drawing from whichever view has higher landmark confidence at
        the corresponding biomechanical moment. Falls back to face-only
        when alignment is unavailable.
        """
        if self.aligner is not None and self.aligner.is_aligned():
            return self.aligner.reconcile_metric(attr, face_frame_idx, prefer=prefer)
        # Single-view fallback
        if self.face and 0 <= face_frame_idx < len(self.face):
            return getattr(self.face[face_frame_idx], attr, None)
        return None

    # ------------------------------------------------------------------
    # Fault 1a: Spine angle change — BACKSWING
    # Spine should stay close to address angle through the backswing.
    # Face-on: lateral tilt (spine_tilt_deg).
    # DTL: forward bend (spine_angle_deg) at top vs address.
    # ------------------------------------------------------------------
    def _spine_angle_change(self) -> Optional[FaultResult]:
        results = []

        # --- Face-on: lateral tilt in backswing (address → top) ---
        if len(self.face) >= 10:
            addr = self.fp["addr_idx"]
            top  = self.fp["top_idx"]
            tilt = _smooth(_get(self.face, "spine_tilt_deg"), 3)
            conf = self.fc
            addr_val = _window_mean(tilt, conf, addr, half=2)
            if addr_val is not None:
                top_val = _window_mean(tilt, conf, top, half=2)
                if top_val is not None:
                    change = abs(top_val - addr_val)
                    threshold = 10.0
                    if change > threshold:
                        sev = min(1.0, (change - threshold) / 20.0)
                        results.append(("face", change, threshold, sev,
                                       "Lateral spine tilt changed during backswing"))

        # --- DTL: forward bend at top vs address ---
        if len(self.back) >= 10:
            addr = self.bp["addr_idx"]
            top  = self.bp["top_idx"]
            spine = _smooth(_get(self.back, "spine_angle_deg"), 3)
            conf  = self.bc
            addr_val = _window_mean(spine, conf, addr, half=2)
            if addr_val is not None:
                top_val = _window_mean(spine, conf, top, half=2)
                if top_val is not None:
                    change = abs(top_val - addr_val)
                    threshold = 8.0
                    if change > threshold:
                        sev = min(1.0, (change - threshold) / 15.0)
                        results.append(("back", change, threshold, sev,
                                       "Forward spine bend changed during backswing"))

        if not results:
            return None

        results.sort(key=lambda x: -x[3])
        view, measured, threshold, sev, detail = results[0]

        # Determine direction for face-on (lateral tilt) vs DTL (forward bend)
        if view == "face":
            # spine_tilt_deg: positive = tilting toward target, negative = away from target
            addr_t = _at(_smooth(_get(self.face, "spine_tilt_deg"), 3), self.fp["addr_idx"])
            top_t  = _phase_mean(_smooth(_get(self.face, "spine_tilt_deg"), 3), self.fc,
                                 max(0, self.fp["top_idx"]-2),
                                 min(len(self.face), self.fp["top_idx"]+3))
            if addr_t is not None and top_t is not None:
                direction = "toward the target" if top_t > addr_t else "away from the target"
            else:
                direction = "laterally"
            desc = (f"Spine tilted {measured:.1f}° {direction} during the backswing "
                    f"(changed from address position by {measured:.1f}°, threshold: {threshold:.0f}°). "
                    "The spine should stay in its setup tilt throughout the backswing.")
        else:
            addr_s = _at(_smooth(_get(self.back, "spine_angle_deg"), 3), self.bp["addr_idx"])
            top_s  = _phase_mean(_smooth(_get(self.back, "spine_angle_deg"), 3), self.bc,
                                 max(0, self.bp["top_idx"]-2),
                                 min(len(self.back), self.bp["top_idx"]+3))
            if addr_s is not None and top_s is not None:
                direction = "upright (less forward bend)" if top_s < addr_s else "more forward (increased bend)"
            else:
                direction = "away from address position"
            desc = (f"Spine moved {measured:.1f}° {direction} during the backswing "
                    f"(threshold: {threshold:.0f}°). "
                    "Forward bend should stay constant from address to top of backswing.")

        return FaultResult(
            name="spine_angle_change_backswing",
            display_name="Loss of Spine Angle (Backswing)",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="backswing",
            measured=round(measured, 1),
            threshold=threshold,
            description=desc,
            cause="Limited hip mobility forces the spine to compensate. "
                  "Often paired with reverse pivot.",
            ball_flight="Fat/thin shots, inconsistent contact",
            view=view,
        )

    # ------------------------------------------------------------------
    # Fault 1b: Spine angle change — DOWNSWING
    # Spine should return to address angle at impact.
    # DTL: forward bend at impact vs address.
    # Face-on: lateral tilt at impact vs address.
    # ------------------------------------------------------------------
    def _spine_angle_change_downswing(self) -> Optional[FaultResult]:
        results = []

        # --- Face-on: lateral tilt at impact ---
        if len(self.face) >= 10:
            addr = self.fp["addr_idx"]
            imp  = self.fp["impact_idx"]
            tilt = _smooth(_get(self.face, "spine_tilt_deg"), 3)
            conf = self.fc
            addr_val = _window_mean(tilt, conf, addr, half=2)
            if addr_val is not None:
                imp_val = _window_mean(tilt, conf, imp, half=2)
                if imp_val is not None:
                    change = abs(imp_val - addr_val)
                    threshold = 10.0
                    if change > threshold:
                        sev = min(1.0, (change - threshold) / 20.0)
                        results.append(("face", change, threshold, sev,
                                       "Lateral spine tilt changed at impact vs address"))

        # --- DTL: forward bend at impact vs address ---
        if len(self.back) >= 10:
            addr = self.bp["addr_idx"]
            imp  = self.bp["impact_idx"]
            spine = _smooth(_get(self.back, "spine_angle_deg"), 3)
            conf  = self.bc
            addr_val = _window_mean(spine, conf, addr, half=2)
            if addr_val is not None:
                imp_val = _window_mean(spine, conf, imp, half=2)
                if imp_val is not None:
                    change = abs(imp_val - addr_val)
                    threshold = 8.0
                    if change > threshold:
                        sev = min(1.0, (change - threshold) / 15.0)
                        results.append(("back", change, threshold, sev,
                                       "Forward spine bend changed at impact vs address"))

        if not results:
            return None

        results.sort(key=lambda x: -x[3])
        view, measured, threshold, sev, detail = results[0]

        if view == "face":
            addr_t = _at(_smooth(_get(self.face, "spine_tilt_deg"), 3), self.fp["addr_idx"])
            imp_t  = _phase_mean(_smooth(_get(self.face, "spine_tilt_deg"), 3), self.fc,
                                 max(0, self.fp["impact_idx"]-2),
                                 min(len(self.face), self.fp["impact_idx"]+3))
            if addr_t is not None and imp_t is not None:
                direction = "toward the target" if imp_t > addr_t else "away from the target"
            else:
                direction = "laterally"
            desc = (f"At impact, spine has tilted {measured:.1f}° {direction} "
                    f"compared to address (threshold: {threshold:.0f}°). "
                    "Spine should return to its address tilt at impact.")
        else:
            addr_s = _at(_smooth(_get(self.back, "spine_angle_deg"), 3), self.bp["addr_idx"])
            imp_s  = _phase_mean(_smooth(_get(self.back, "spine_angle_deg"), 3), self.bc,
                                 max(0, self.bp["impact_idx"]-2),
                                 min(len(self.back), self.bp["impact_idx"]+3))
            if addr_s is not None and imp_s is not None:
                direction = "upright (standing up through the ball)" if imp_s < addr_s else "more forward (over-reaching)"
            else:
                direction = "away from address position"
            desc = (f"At impact, spine is {measured:.1f}° {direction} "
                    f"compared to address (threshold: {threshold:.0f}°). "
                    "Forward bend should be maintained from address through impact.")

        return FaultResult(
            name="spine_angle_change_downswing",
            display_name="Loss of Spine Angle (Downswing / Impact)",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="downswing",
            measured=round(measured, 1),
            threshold=threshold,
            description=desc,
            cause="Early extension or hanging back causes the spine to stand up "
                  "or lunge forward through impact.",
            ball_flight="Thin shots, blocks, inconsistent low point",
            view=view,
        )

    # ------------------------------------------------------------------
    # Fault 2: Reverse spine angle (face-on)
    # Spine tilts TOWARD the target in the backswing.
    # Measured as: spine_tilt_deg increases (positive = toward target)
    # from address to top of backswing.
    # ------------------------------------------------------------------
    def _reverse_spine(self) -> Optional[FaultResult]:
        if len(self.face) < 10:
            return None

        addr = self.fp["addr_idx"]
        top  = self.fp["top_idx"]
        tilt = _smooth(_get(self.face, "spine_tilt_deg"), 3)
        conf = self.fc

        addr_val = _window_mean(tilt, conf, addr, half=2)
        if addr_val is None:
            return None

        # Average tilt in the last third of backswing (near top)
        bs_start = max(addr + 1, top - max(2, (top - addr) // 3))
        top_val  = _phase_mean(tilt, conf, bs_start, top + 1)
        if top_val is None:
            return None

        # Positive = tilting toward target = reverse spine
        change = top_val - addr_val
        threshold = 6.0  # degrees toward target

        if change <= threshold:
            return None

        sev = min(1.0, (change - threshold) / 15.0)
        return FaultResult(
            name="reverse_spine",
            display_name="Reverse Spine Angle",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="backswing",
            measured=round(change, 1),
            threshold=threshold,
            description=f"Spine tilted {change:.1f}° toward the target (lead side) during the backswing "
                        f"(threshold: {threshold:.0f}°). At the top of the backswing the spine "
                        "should be neutral or tilting slightly away from the target (trail side), "
                        "not leaning toward it.",
            cause="Trying to keep head perfectly still while restricting shoulder turn. "
                  "Limited thoracic mobility. TPI identifies this as the #1 cause of back pain in golf.",
            ball_flight="Over-the-top pull, steep downswing, slices",
            view="face",
        )

    # ------------------------------------------------------------------
    # Fault 3: Hip sway / slide (face-on)
    # Sway: hips drift trail-ward in backswing (positive hip_sway_px = trail)
    # Slide: hips drift target-ward in downswing without rotating
    #
    # Threshold is a fraction of hip_width at address (body-proportional).
    # ------------------------------------------------------------------
    def _hip_sway_slide(self) -> Optional[FaultResult]:
        if len(self.face) < 10:
            return None

        addr = self.fp["addr_idx"]
        top  = self.fp["top_idx"]
        imp  = self.fp["impact_idx"]
        sway = _get(self.face, "hip_sway_px")
        conf = self.fc

        # Body-proportional threshold: use shoulder width at address as reference
        # Hip width isn't directly available but shoulder width proxy works
        # ~20% of shoulder pixel width = reasonable sway limit
        # We use a fixed pixel fallback if shoulder width unknown
        raw_metrics = self.face
        addr_shld_w = None
        if addr < len(raw_metrics):
            m = raw_metrics[addr]
            w = getattr(m, "_shoulder_width_px", None)
            if w and w > 20:
                addr_shld_w = w

        sway_threshold = addr_shld_w * 0.30 if addr_shld_w else 35.0

        results = []

        # --- Sway: backswing trail drift ---
        bs_sway = _phase_mean(sway, conf, addr, top)
        if bs_sway is not None and bs_sway > sway_threshold:
            sev = min(1.0, (bs_sway - sway_threshold) / (sway_threshold * 1.5))
            results.append(("sway", bs_sway, sev, "backswing",
                           "Hips slide away from target in backswing instead of rotating"))

        # --- Slide: downswing excessive target-ward drift ---
        ds_sway = _get(self.face, "hip_sway_px")
        ds_max_lead = _phase_max(-ds_sway, conf, top, imp)  # negative = lead direction
        slide_threshold = sway_threshold * 2.0  # slide can be a bit more than sway

        if ds_max_lead is not None and ds_max_lead > slide_threshold:
            sev = min(1.0, (ds_max_lead - slide_threshold) / (slide_threshold * 1.5))
            results.append(("slide", ds_max_lead, sev, "downswing",
                           "Hips slide excessively toward target instead of rotating"))

        if not results:
            return None

        results.sort(key=lambda x: -x[2])
        fault_type, measured, sev, phase, detail = results[0]

        display = "Hip Sway" if fault_type == "sway" else "Hip Slide"
        return FaultResult(
            name=f"hip_{fault_type}",
            display_name=display,
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase=phase,
            measured=round(measured, 1),
            threshold=round(sway_threshold if fault_type == "sway" else slide_threshold, 1),
            description=(
                f"{detail}. "
                + (f"Hips drifted {measured:.0f}px away from the target (trail side) "
                   "instead of rotating in place during the backswing."
                   if fault_type == "sway" else
                   f"Hips slid {measured:.0f}px toward the target (lead side) "
                   "through the downswing without rotating — a pure lateral push "
                   "rather than a hip turn.")
            ),
            cause="Limited hip rotation mobility. The body slides laterally instead of rotating "
                  "because the hips can't turn freely.",
            ball_flight="Fat shots, blocks, loss of power, inconsistent low point" if fault_type == "sway"
                        else "Pushes, blocks right, loss of distance",
            view="face",
        )

    # ------------------------------------------------------------------
    # Fault 4: Early extension — DTL (hips thrust toward ball)
    # Two signals:
    #   a) hip_slide_px increases toward ball (forward) in downswing
    #   b) spine_angle_deg decreases (less forward bend) from top to impact
    # ------------------------------------------------------------------
    def _early_extension(self) -> Optional[FaultResult]:
        if len(self.back) < 10:
            return None

        top = self.bp["top_idx"]
        imp = self.bp["impact_idx"]
        conf = self.bc

        spine_raw = _get_or_nan(self.back, "spine_angle_deg")
        if np.nansum(~np.isnan(spine_raw)) < 5:
            return None
        spine = _smooth(np.nan_to_num(spine_raw, nan=0.0), 3)
        top_spine = _window_mean(spine, conf, top, half=2)
        imp_spine = _window_mean(spine, conf, imp, half=2)

        if top_spine is None or imp_spine is None:
            return None

        # Spine decreasing = standing up = early extension
        spine_loss = top_spine - imp_spine  # positive = standing up
        threshold  = 8.0  # degrees

        if spine_loss <= threshold:
            return None

        sev = min(1.0, (spine_loss - threshold) / 15.0)
        return FaultResult(
            name="early_extension",
            display_name="Early Extension",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="downswing",
            measured=round(spine_loss, 1),
            threshold=threshold,
            description=(f"Spine straightened {spine_loss:.1f}° from the top of the backswing to impact "
                        f"(threshold: {threshold:.0f}°). The hips are thrusting forward toward the ball "
                        "and the upper body is standing upright — the golfer is losing their address "
                        "forward bend through the hitting zone."),
            cause="Limited hip mobility prevents rotation, so the body thrusts forward instead. "
                  "TPI identifies early extension as the most common amateur fault.",
            ball_flight="Thin shots, blocks, hooks from compensatory flip at impact",
            view="back",
        )

    # ------------------------------------------------------------------
    # Fault 5: Head movement (face-on + DTL)
    # Face-on: lateral drift of nose from address
    # DTL: forward drift and ear rotation
    # Threshold: body-proportional (fraction of body height)
    # ------------------------------------------------------------------
    def _head_movement(self) -> Optional[FaultResult]:
        results = []

        # --- Face-on: lateral head drift (backswing only) ---
        # Only measure from address to impact — follow-through naturally
        # moves the head a lot and should not be counted as a fault.
        if len(self.face) >= 10:
            conf = self.fc
            head = _smooth(_get(self.face, "head_movement_px"), 3)
            addr = self.fp["addr_idx"]
            imp  = self.fp["impact_idx"]

            # Body height proxy: use hand height range
            hy = _get(self.face, "hand_height_y")
            body_h = float(np.max(hy) - np.min(hy)) if np.any(hy > 0) else 400.0
            threshold = max(30.0, body_h * 0.08)  # 8% of body height

            # Only look at backswing + downswing (address → impact)
            swing_head = head[addr:imp+1]
            swing_conf = conf[addr:imp+1]
            good = swing_conf >= 0.55
            if good.sum() >= 3:
                max_head = float(np.percentile(swing_head[good], 90))
                if max_head > threshold:
                    sev = min(1.0, (max_head - threshold) / (threshold * 2))
                    results.append(("face", max_head, threshold, sev,
                                   "Head moves excessively from its address position"))

        # --- DTL: ear-based head rotation ---
        if len(self.back) >= 10:
            conf = self.bc
            addr = self.bp["addr_idx"]
            top  = self.bp["top_idx"]
            head_fwd = _smooth(_get(self.back, "head_forward_px"), 3)

            addr_hf = _window_mean(head_fwd, conf, addr, half=2)
            top_hf  = _window_mean(head_fwd, conf, top, half=2)

            if addr_hf is not None and top_hf is not None:
                fwd_change = top_hf - addr_hf  # positive = head moved forward (toward ball)
                threshold  = 25.0  # pixels

                if abs(fwd_change) > threshold:
                    sev = min(1.0, (abs(fwd_change) - threshold) / 40.0)
                    direction = "forward (toward ball)" if fwd_change > 0 else "backward (away from ball)"
                    results.append(("back", abs(fwd_change), threshold, sev,
                                   f"Head drifted {direction} during backswing"))

        if not results:
            return None

        results.sort(key=lambda x: -x[3])
        view, measured, threshold, sev, detail = results[0]

        return FaultResult(
            name="head_movement",
            display_name="Excessive Head Movement",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="backswing",
            measured=round(measured, 1),
            threshold=round(threshold, 1),
            description=(
                f"{detail}. "
                + (f"Head moved {measured:.0f}px from its address position "
                   "(face-on view — lateral drift away from or toward the target)."
                   if view == "face" else
                   f"Head moved {measured:.0f}px forward toward the ball "
                   "from its address position (back-view depth change).")
            ),
            cause="Swaying body drags the head laterally. Or actively trying to 'keep your head down' "
                  "causes excessive head restriction and then a lurch to compensate.",
            ball_flight="Inconsistent contact, fat/thin, directional inconsistency",
            view=view,
        )

    # ------------------------------------------------------------------
    # Fault: Hip Shift Forward (toward target) — DTL
    # Measures the actual lateral movement of the hip midpoint toward
    # the target during the downswing using hip_slide_px from back metrics.
    # Distinct from early extension (which measures spine angle decrease).
    # ------------------------------------------------------------------
    def _hip_shift_forward(self) -> Optional[FaultResult]:
        if len(self.back) < 10:
            return None

        addr = self.bp["addr_idx"]
        top  = self.bp["top_idx"]
        imp  = self.bp["impact_idx"]
        conf = self.bc

        hip_slide = _smooth(_get(self.back, "hip_slide_px"), 3)

        addr_slide = _at(hip_slide, addr)
        if addr_slide is None:
            return None

        # Max target-ward slide in downswing (positive = toward target)
        ds_slide = _phase_max(hip_slide, conf, top, imp)
        if ds_slide is None:
            return None

        threshold = 45.0  # px — generous to avoid false positives

        if ds_slide < threshold:
            return None

        sev = min(1.0, (ds_slide - threshold) / 60.0)

        return FaultResult(
            name="hip_shift_forward",
            display_name="Hip Shift Toward Target (Lateral Slide)",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="downswing",
            measured=round(ds_slide, 1),
            threshold=threshold,
            description=(f"Hips slid {ds_slide:.0f}px toward the target (lead side) in the downswing "
                        f"(threshold: {threshold:.0f}px). Instead of rotating around a fixed axis, "
                        "the hips are pushing laterally toward the target — a slide, not a turn."),
            cause="Limited hip rotation mobility causes the hips to slide rather than turn. "
                  "Often paired with early extension. TPI calls this the Slide fault.",
            ball_flight="Pushes, blocks right, loss of distance, weak contact",
            view="back",
        )

    # ------------------------------------------------------------------
    # Fault 7: Low X-Factor (face-on)
    # X-factor = shoulder_rotation - hip_rotation at top of backswing.
    # Meister et al. 2011 (Stanford): professional mean = 56°, CV = 7.4%.
    # Amateurs with handicap >10 fell >2 SD below professional mean.
    # We flag < 25° as clearly deficient — that's ~2 SD below amateur mean.
    # Only checked at top of backswing where both signals are most reliable.
    # ------------------------------------------------------------------
    def _low_x_factor(self) -> Optional[FaultResult]:
        if len(self.face) < 10:
            return None

        top  = self.fp["top_idx"]
        conf = self.fc

        shld = _smooth(_get(self.face, "shoulder_rotation_deg"), 3)
        hip  = _smooth(_get(self.face, "hip_rotation_deg"), 3)

        # Average around the top (±2 frames) for stability
        w = max(2, (top - self.fp["addr_idx"]) // 5)
        top_shld = _window_mean(shld, conf, top, half=w)
        top_hip  = _window_mean(hip,  conf, top, half=w)

        if top_shld is None or top_hip is None:
            return None

        # X-factor = shoulder separation over hips
        x_factor = top_shld - top_hip

        # Require minimum shoulder turn to trust the measurement
        # (if shoulders barely moved, hip signal is also unreliable)
        if top_shld < 20.0:
            return None

        # Two sub-faults:
        # a) Near-zero X-factor: hips and shoulders turning together (no separation)
        # b) Negative X-factor: hips over-rotated past shoulders (rare but severe)
        threshold_low  = 25.0   # below this = low X-factor fault
        threshold_neg  = 5.0    # below 5° treated as zero/negative

        if x_factor >= threshold_low:
            return None

        if x_factor < threshold_neg:
            sev = min(1.0, (threshold_neg - x_factor) / 20.0 + 0.5)
            detail = (f"X-factor is {x_factor:.1f}° — hips have rotated as far or further "
                      "than the shoulders (near-zero or no separation). "
                      "At the top of the backswing the shoulders should have turned "
                      "significantly more than the hips, creating stretch")
            cause = "Over-active hips or restricted thoracic rotation. "  \
                    "Hips spinning out early with no shoulder resistance."
            ball_flight = "Weak push-fades, loss of distance, over-the-top tendency"
        else:
            sev = min(1.0, (threshold_low - x_factor) / threshold_low)
            detail = (f"Shoulders rotated {top_shld:.1f}° and hips rotated {top_hip:.1f}° "
                      f"at the top — only {x_factor:.1f}° of separation (threshold: {threshold_low:.0f}°, "
                      f"professional mean: 56°). The hips are following the shoulders "
                      "too closely instead of staying relatively still")
            cause = "Hips turning too much during backswing (sway/spin-out) or "  \
                    "shoulders not completing their turn. Both reduce elastic loading."
            ball_flight = "Loss of distance and power, inconsistent contact"

        return FaultResult(
            name="low_x_factor",
            display_name="Low X-Factor (Insufficient Hip-Shoulder Separation)",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="backswing (top)",
            measured=round(x_factor, 1),
            threshold=threshold_low,
            description=detail,
            cause=cause,
            ball_flight=ball_flight,
            view="face",
        )

    # ------------------------------------------------------------------
    # Fault 8: Insufficient shoulder turn at top (face-on)
    # Meister 2011: professional peak upper-torso rotation highly correlated
    # to clubhead speed (r=0.90). Pros average ~90° shoulder rotation.
    # We flag < 60° as clearly insufficient — generous threshold for amateurs
    # who have natural variation, but below 60° is universally a restriction.
    # Uses shoulder_rotation_deg which is computed from width-ratio + z-depth.
    # ------------------------------------------------------------------
    def _insufficient_shoulder_turn(self) -> Optional[FaultResult]:
        if len(self.face) < 10:
            return None

        addr = self.fp["addr_idx"]
        top  = self.fp["top_idx"]
        conf = self.fc

        shld = _smooth(_get(self.face, "shoulder_rotation_deg"), 3)

        # Peak shoulder rotation anywhere in the backswing window
        bs_peak = _phase_max(shld, conf, addr, min(len(shld), top+3))
        if bs_peak is None:
            return None

        threshold = 60.0   # degrees from address

        if bs_peak >= threshold:
            return None

        sev = min(1.0, (threshold - bs_peak) / threshold)

        return FaultResult(
            name="insufficient_shoulder_turn",
            display_name="Restricted Shoulder Turn",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="backswing",
            measured=round(bs_peak, 1),
            threshold=threshold,
            description=(f"Shoulders rotated only {bs_peak:.1f}° from address at the top of the "
                        f"backswing (threshold: {threshold:.0f}°, professional mean: ~90°). "
                        f"The trail shoulder is not moving far enough behind the golfer — "
                        f"the backswing is cutting short."),
            cause="Limited thoracic (upper back) mobility, or consciously restricting the turn "
                  "to maintain control. Often accompanies reverse spine angle.",
            ball_flight="Loss of distance, armsy swing, inconsistent direction",
            view="face",
        )

    # ------------------------------------------------------------------
    # Fault 9: Lead knee extension at impact (face-on)
    # The lead knee should straighten (extend) through impact as the hips
    # rotate and clear. TPI kinematic chain: ground force → hips → torso.
    # If lead_knee_angle_deg is LESS at impact than at address, the knee
    # is collapsing rather than extending — a clear fault.
    # Uses hip/knee/ankle — three of our most reliable landmarks.
    # ------------------------------------------------------------------
    def _lead_knee_extension(self) -> Optional[FaultResult]:
        if len(self.face) < 10:
            return None

        addr = self.fp["addr_idx"]
        imp  = self.fp["impact_idx"]
        conf = self.fc

        knee = _smooth(_get(self.face, "lead_knee_angle_deg"), 3)

        addr_knee = _window_mean(knee, conf, addr, half=2)
        # Average around impact for stability (impact detection is approximate)
        w = max(1, (imp - self.fp["top_idx"]) // 4)
        imp_knee  = _window_mean(knee, conf, imp, half=w)

        if addr_knee is None or imp_knee is None:
            return None

        # Require reasonable address knee angle (sanity check on detection)
        if addr_knee < 140 or addr_knee > 185:
            return None

        # At impact, knee should be AT LEAST as straight as address.
        # Good golfers straighten 5-15° beyond address.
        # Fault: knee is more bent at impact than at address (negative = collapsing)
        extension = imp_knee - addr_knee   # positive = straightened, negative = collapsed

        threshold = -8.0   # 8° MORE bent than address = clear collapse
                           # generous buffer — avoids flagging natural variation

        if extension > threshold:
            return None

        collapse = abs(extension)
        sev = min(1.0, (collapse - abs(threshold)) / 20.0)

        return FaultResult(
            name="lead_knee_collapse",
            display_name="Lead Knee Collapse at Impact",
            detected=True,
            severity=round(sev, 3),
            severity_label=_label(sev),
            phase="impact",
            measured=round(imp_knee, 1),
            threshold=round(addr_knee, 1),
            description=(f"Lead knee is {collapse:.1f}° more bent at impact ({imp_knee:.1f}°) "
                        f"than at address ({addr_knee:.1f}°). The lead knee is collapsing inward "
                        "and downward through impact instead of straightening and driving the "
                        "lead hip upward and around."),
            cause="Passive lower body — failing to drive the lead hip and leg through impact. "
                  "Often accompanies early extension or hanging back.",
            ball_flight="Loss of power, tendency to hit behind the ball, inconsistent low point",
            view="face",
        )


# ---------------------------------------------------------------------------
# Convenience wrapper — mirrors FaultReport interface for feedback.py
# ---------------------------------------------------------------------------

@dataclass
class FaultReport:
    faults: List[FaultResult] = field(default_factory=list)
    address_frame: int = 0
    top_frame: int = 0
    impact_frame: int = 0
    backswing_range: tuple = (0, 0)
    downswing_range: tuple = (0, 0)
    followthrough_range: tuple = (0, 0)
    impact_signals: Optional[dict] = None

    # FeedbackGenerator reads .faults[i].display_name / severity_label /
    # severity / phase / measured_value / elite_benchmark / description /
    # root_cause / ball_flight — map our fields to those names
    def summary(self) -> str:
        engine = FaultEngine.__new__(FaultEngine)
        engine.fp = {"addr_idx": self.address_frame, "top_idx": self.top_frame,
                     "impact_idx": self.impact_frame}
        engine.bp = engine.fp
        engine._results = self.faults
        lines = [
            "", "Swing Fault Report", "=" * 50,
            f"Phases: address={self.address_frame}  top={self.top_frame}  impact={self.impact_frame}",
            ""
        ]
        if not self.faults:
            lines.append("No significant faults detected.")
            return "\n".join(lines)
        lines.append(f"Faults: {len(self.faults)}\n")
        for i, r in enumerate(self.faults, 1):
            lines += [
                f"{i}. [{r.severity_label.upper()}] {r.display_name}  (severity {r.severity:.2f})",
                f"   Measured: {r.measured}  View: {r.view}",
                f"   {r.description}",
                f"   Cause: {r.cause}",
                f"   Ball flight: {r.ball_flight}",
                "",
            ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "faults": [r.to_dict() for r in self.faults],
            "fault_count": len(self.faults),
            "top_fault": self.faults[0].to_dict() if self.faults else None,
        }


def run_fault_detection(
    face_metrics: Optional[list] = None,
    back_metrics: Optional[list] = None,
    verbose: bool = True,
) -> FaultReport:
    """
    Top-level entry point. Replace FaultDetector().detect() with this.
    """
    engine = FaultEngine(face_metrics=face_metrics, back_metrics=back_metrics)
    faults = engine.detect()

    phases = engine.fp if engine.face else engine.bp

    if verbose:
        print(engine.summary())

    return FaultReport(
        faults=faults,
        address_frame=phases.get("addr_idx", 0),
        top_frame=phases.get("top_idx", 0),
        impact_frame=phases.get("impact_idx", 0),
        backswing_range=phases.get("backswing_range", (0, 0)),
        downswing_range=phases.get("downswing_range", (0, 0)),
        followthrough_range=phases.get("followthrough_range", (0, 0)),
        impact_signals=phases.get("_impact_signals"),
    )


# ---------------------------------------------------------------------------
# Club groups — used by main.py for CLI help text
# ---------------------------------------------------------------------------
CLUB_GROUPS = {
    "driver": "driver",
    "3w": "iron_long", "5w": "iron_long", "7w": "iron_long",
    "hybrid": "iron_long", "2i": "iron_long", "3i": "iron_long",
    "4i": "iron_mid", "5i": "iron_mid", "6i": "iron_mid", "7i": "iron_mid",
    "8i": "iron_short", "9i": "iron_short",
    "pw": "iron_short", "gw": "iron_short", "sw": "iron_short", "lw": "iron_short",
}