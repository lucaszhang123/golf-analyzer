"""
pose_filter.py — Golf-specific kinematic constraint filter.

Two passes, based on GolfMate (Ju et al. 2023):

  PASS 1 — GOLFMATE RULE-BASED CHECKS
    a) Inversion detector: left/right joint labels should never swap
    b) Ankle/foot stability: feet don't move during a golf swing
    c) Hand separation: both wrists are on the grip, always close together

  PASS 2 — BONE-LENGTH CONSISTENCY
    Each bone (shoulder→elbow, elbow→wrist, etc.) has a fixed length.
    Any frame where a bone deviates more than ±40% from its median
    length is flagged as a bad detection for the distal joint.

  PASS 3 — TEMPORAL INTERPOLATION
    Flagged joints are replaced by cubic spline (or linear) interpolation
    over surrounding good frames. Runs longer than max_bad_run are zeroed.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

try:
    from scipy import interpolate as scipy_interp
    _SCIPY_AVAILABLE = True
except ImportError:
    scipy_interp = None
    _SCIPY_AVAILABLE = False

from .landmarks import LM


# Only TRUE fixed-length bones that don't change with perspective or rotation.
# Excluded intentionally:
#   LEFT/RIGHT_SHOULDER-to-HIP  — apparent length changes with spine bend
#   LEFT_SHOULDER-to-RIGHT_SHOULDER — shrinks as golfer rotates away from camera
#   LEFT_HIP-to-RIGHT_HIP           — same perspective issue
# These exclusions prevent false positives on the spine and torso.
BONE_PAIRS: List[Tuple[LM, LM]] = [
    (LM.LEFT_SHOULDER,  LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW,     LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW,    LM.RIGHT_WRIST),
    (LM.LEFT_HIP,       LM.LEFT_KNEE),
    (LM.LEFT_KNEE,      LM.LEFT_ANKLE),
    (LM.RIGHT_HIP,      LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE,     LM.RIGHT_ANKLE),
]

MAX_BAD_RUN_SECONDS = 0.30


class _LandmarkProxy:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x, y, z, v):
        self.x = float(x); self.y = float(y)
        self.z = float(z); self.visibility = float(v)

class _LandmarkListProxy:
    def __init__(self, lms): self.landmark = lms

class _ResultProxy:
    def __init__(self, ll): self.pose_landmarks = ll


@dataclass
class FilterQuality:
    n_frames: int
    n_joints: int
    bad_frames_by_joint: Dict[int, int]
    interpolated_runs: int
    uncorrectable_runs: int

    def summary(self) -> str:
        total = sum(self.bad_frames_by_joint.values())
        lines = [
            "PoseFilter quality report:",
            f"  Frames      : {self.n_frames}",
            f"  Total flags : {total}",
            f"  Interp runs : {self.interpolated_runs}",
            f"  Uncorrectable (gap too long): {self.uncorrectable_runs}",
        ]
        top = sorted(self.bad_frames_by_joint.items(), key=lambda x: -x[1])[:5]
        if top:
            lines.append("  Most-corrected joints:")
            for lm_idx, count in top:
                try:    name = LM(lm_idx).name
                except: name = str(lm_idx)
                pct = 100 * count / max(self.n_frames, 1)
                lines.append(f"    {name:<22s}: {count} frames ({pct:.0f}%)")
        return "\n".join(lines)


class PoseFilter:
    """
    Parameters
    ----------
    image_w, image_h : frame dimensions in pixels
    fps              : video frame rate (scales max interpolation gap)
    min_visibility   : MediaPipe visibility below this = don't trust
    bone_tolerance   : ±fraction allowed deviation from median bone length
    verbose          : print quality summary after filtering
    """

    def __init__(
        self,
        image_w: int,
        image_h: int,
        fps: float = 30.0,
        min_visibility: float = 0.45,
        bone_tolerance: float = 0.60,   # loose — only catches obvious teleportation
        verbose: bool = False,
    ):
        self.image_w        = image_w
        self.image_h        = image_h
        self.fps            = fps
        self.min_visibility = min_visibility
        self.bone_tolerance = bone_tolerance
        self.verbose        = verbose
        self.max_bad_run    = max(4, int(round(fps * MAX_BAD_RUN_SECONDS)))
        self._joint_indices = [int(lm) for lm in LM]
        self._ji            = {lm_idx: ji for ji, lm_idx in enumerate(self._joint_indices)}

    def filter(self, raw_landmarks: list) -> list:
        clean, _ = self.filter_with_quality(raw_landmarks)
        return clean

    def filter_with_quality(self, raw_landmarks: list) -> Tuple[list, FilterQuality]:
        n_frames = len(raw_landmarks)
        if n_frames == 0:
            return [], FilterQuality(0, 0, {}, 0, 0)

        n_joints = len(self._joint_indices)
        coords   = np.zeros((n_frames, n_joints, 2), dtype=float)
        z_vals   = np.zeros((n_frames, n_joints),    dtype=float)
        vis      = np.zeros((n_frames, n_joints),    dtype=float)
        has_pose = np.zeros(n_frames, dtype=bool)

        for fi, res in enumerate(raw_landmarks):
            if res is None or res.pose_landmarks is None:
                continue
            has_pose[fi] = True
            lms = res.pose_landmarks.landmark
            for ji, lm_idx in enumerate(self._joint_indices):
                lm = lms[lm_idx]
                coords[fi, ji, 0] = lm.x * self.image_w
                coords[fi, ji, 1] = lm.y * self.image_h
                z_vals[fi, ji]    = lm.z
                vis[fi, ji]       = lm.visibility

        bad = np.zeros((n_frames, n_joints), dtype=bool)

        # Pass 1a: low MediaPipe confidence
        bad |= (vis < self.min_visibility) & has_pose[:, None]

        # Pass 1b: GolfMate inversion detector
        self._pass_inversion(coords, bad, has_pose)

        # Pass 1c: GolfMate ankle stability
        self._pass_ankle_stability(coords, bad, has_pose)

        # Pass 1d: hand separation (both wrists on the grip)
        self._pass_hand_separation(coords, bad, has_pose)

        # Pass 2: bone-length consistency
        self._pass_bone_length(coords, bad, has_pose)

        # Pass 3: interpolate
        n_interp, n_uncorr = self._pass_interpolate(
            coords, z_vals, vis, bad, has_pose
        )

        bad_counts = {
            self._joint_indices[ji]: int(bad[:, ji].sum())
            for ji in range(n_joints)
            if bad[:, ji].sum() > 0
        }

        quality = FilterQuality(
            n_frames=n_frames,
            n_joints=n_joints,
            bad_frames_by_joint=bad_counts,
            interpolated_runs=n_interp,
            uncorrectable_runs=n_uncorr,
        )

        if self.verbose:
            print(quality.summary())

        return self._rebuild(raw_landmarks, coords, z_vals, vis, bad, has_pose), quality

    # ------------------------------------------------------------------
    # Pass 1b: Inversion detector (GolfMate algorithm a)
    # "Throughout the entire swing, the right ankle x should never
    #  exceed the left ankle x" — extended to all limb pairs.
    # Uses a generous threshold so minor projection overlap doesn't trigger.
    # ------------------------------------------------------------------
    def _pass_inversion(self, coords, bad, has_pose):
        # Only flag when the crossing is substantial (>10% of frame width)
        # to avoid triggering on a DTL camera or slight overlap.
        thresh = self.image_w * 0.10

        pairs = [
            (LM.LEFT_ANKLE,    LM.RIGHT_ANKLE),
            (LM.LEFT_KNEE,     LM.RIGHT_KNEE),
            (LM.LEFT_HIP,      LM.RIGHT_HIP),
            (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
        ]
        for left_lm, right_lm in pairs:
            l_ji = self._ji[int(left_lm)]
            r_ji = self._ji[int(right_lm)]
            # right x should be > left x (right side of frame for face-on RH golfer)
            crossed = (
                has_pose &
                (coords[:, r_ji, 0] < coords[:, l_ji, 0] - thresh)
            )
            bad[crossed, l_ji] = True
            bad[crossed, r_ji] = True

    # ------------------------------------------------------------------
    # Pass 1c: Ankle/foot stability (GolfMate algorithm c)
    # "If ankle distance becomes less than half of expected, flag."
    # Extended: ankles shouldn't move more than 6% of frame height
    # from their median position across the whole swing.
    # ------------------------------------------------------------------
    def _pass_ankle_stability(self, coords, bad, has_pose):
        # 10% of frame height — looser than before because feet near the
        # frame boundary have more MediaPipe jitter.
        # Heels and foot_index are excluded — too small for reliable stability check.
        max_drift = self.image_h * 0.10

        for lm in [LM.LEFT_ANKLE, LM.RIGHT_ANKLE]:
            lm_ji = self._ji[int(lm)]
            good  = has_pose & ~bad[:, lm_ji]
            if good.sum() < 4:
                continue
            med_x = float(np.median(coords[good, lm_ji, 0]))
            med_y = float(np.median(coords[good, lm_ji, 1]))
            dist  = np.sqrt(
                (coords[:, lm_ji, 0] - med_x) ** 2 +
                (coords[:, lm_ji, 1] - med_y) ** 2
            )
            bad[has_pose & (dist > max_drift), lm_ji] = True

    # ------------------------------------------------------------------
    # Pass 1d: Hand separation
    # Both wrists are always on the grip — they can't be more than
    # ~18% of frame height apart. If they are, flag the one that moved
    # further from its previous position (the bad one).
    # ------------------------------------------------------------------
    def _pass_hand_separation(self, coords, bad, has_pose):
        max_sep = self.image_h * 0.18
        lw_ji   = self._ji[int(LM.LEFT_WRIST)]
        rw_ji   = self._ji[int(LM.RIGHT_WRIST)]

        sep = np.linalg.norm(
            coords[:, lw_ji, :] - coords[:, rw_ji, :], axis=1
        )
        too_far = has_pose & (sep > max_sep)

        for fi in np.where(too_far)[0]:
            if fi == 0:
                # Can't compare to previous — flag both
                bad[fi, lw_ji] = True
                bad[fi, rw_ji] = True
                continue
            # Flag whichever wrist moved more from the previous frame
            dl = np.linalg.norm(coords[fi, lw_ji] - coords[fi-1, lw_ji])
            dr = np.linalg.norm(coords[fi, rw_ji] - coords[fi-1, rw_ji])
            if dl > dr:
                bad[fi, lw_ji] = True
            else:
                bad[fi, rw_ji] = True

    # ------------------------------------------------------------------
    # Pass 2: Bone-length consistency
    # ------------------------------------------------------------------
    def _pass_bone_length(self, coords, bad, has_pose):
        for (lm_a, lm_b) in BONE_PAIRS:
            ja = self._ji.get(int(lm_a))
            jb = self._ji.get(int(lm_b))
            if ja is None or jb is None:
                continue

            good = has_pose & ~bad[:, ja] & ~bad[:, jb]
            if good.sum() < 5:
                continue

            lengths    = np.linalg.norm(
                coords[:, jb, :] - coords[:, ja, :], axis=1
            )
            median_len = float(np.median(lengths[good]))
            if median_len < 5:
                continue

            lo = median_len * (1.0 - self.bone_tolerance)
            hi = median_len * (1.0 + self.bone_tolerance)
            outlier = has_pose & ((lengths < lo) | (lengths > hi))
            bad[outlier, jb] = True

    # ------------------------------------------------------------------
    # Pass 3: Temporal interpolation
    # ------------------------------------------------------------------
    def _pass_interpolate(self, coords, z_vals, vis, bad, has_pose):
        n_interp = 0
        n_uncorr = 0

        for ji in range(coords.shape[1]):
            bad_ji  = bad[:, ji]
            good_ji = has_pose & ~bad_ji
            if good_ji.sum() < 4:
                vis[:, ji][bad_ji] = 0.0
                continue

            good_idx = np.where(good_ji)[0]
            bad_idx  = np.where(bad_ji & has_pose)[0]
            if len(bad_idx) == 0:
                continue

            # Group into contiguous runs
            runs = []
            rs = prev = bad_idx[0]
            for idx in bad_idx[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    runs.append((rs, prev))
                    rs = prev = idx
            runs.append((rs, prev))

            # Build interpolators
            use_spline = False
            if _SCIPY_AVAILABLE:
                try:
                    cs_x = scipy_interp.CubicSpline(
                        good_idx, coords[good_idx, ji, 0], extrapolate=False)
                    cs_y = scipy_interp.CubicSpline(
                        good_idx, coords[good_idx, ji, 1], extrapolate=False)
                    cs_z = scipy_interp.CubicSpline(
                        good_idx, z_vals[good_idx, ji], extrapolate=False)
                    use_spline = True
                except Exception:
                    pass

            for (rs, re) in runs:
                if re - rs + 1 > self.max_bad_run:
                    # Gap too long to interpolate reliably.
                    # Instead of zeroing (which causes the joint to vanish),
                    # hold the last known good position with low visibility.
                    # The renderer will draw it faintly rather than leaving a gap.
                    last_good = good_idx[good_idx < rs]
                    if len(last_good) > 0:
                        lg = last_good[-1]
                        coords[rs:re+1, ji, 0] = coords[lg, ji, 0]
                        coords[rs:re+1, ji, 1] = coords[lg, ji, 1]
                        z_vals[rs:re+1, ji]    = z_vals[lg, ji]
                    vis[rs:re+1, ji] = 0.12   # low but non-zero — draws faintly
                    n_uncorr += 1
                    continue

                ii = np.arange(rs, re + 1, dtype=float)
                if use_spline:
                    ix = cs_x(ii); iy = cs_y(ii); iz = cs_z(ii)
                else:
                    ix = np.interp(ii, good_idx, coords[good_idx, ji, 0])
                    iy = np.interp(ii, good_idx, coords[good_idx, ji, 1])
                    iz = np.interp(ii, good_idx, z_vals[good_idx, ji])

                valid = ~(np.isnan(ix) | np.isnan(iy))
                for off, fi in enumerate(range(rs, re + 1)):
                    if not valid[off]:
                        near = good_idx[np.argmin(np.abs(good_idx - fi))]
                        ix[off] = coords[near, ji, 0]
                        iy[off] = coords[near, ji, 1]
                        iz[off] = z_vals[near, ji]

                coords[rs:re+1, ji, 0] = ix
                coords[rs:re+1, ji, 1] = iy
                z_vals[rs:re+1, ji]    = iz
                vis[rs:re+1, ji]       = np.clip(vis[rs:re+1, ji], 0, 0.65)
                bad[rs:re+1, ji]       = False
                n_interp += 1

        return n_interp, n_uncorr

    # ------------------------------------------------------------------
    # Rebuild landmark objects
    # ------------------------------------------------------------------
    def _rebuild(self, raw_landmarks, coords, z_vals, vis, bad, has_pose):
        ji_map = {lm_idx: ji for ji, lm_idx in enumerate(self._joint_indices)}
        out    = []

        for fi, raw in enumerate(raw_landmarks):
            if not has_pose[fi] or raw is None or raw.pose_landmarks is None:
                out.append(raw)
                continue

            raw_lms = raw.pose_landmarks.landmark
            lm_list = []
            for lm_idx in range(33):
                raw_lm = raw_lms[lm_idx]
                if lm_idx in ji_map:
                    ji = ji_map[lm_idx]
                    lm_list.append(_LandmarkProxy(
                        x=coords[fi, ji, 0] / self.image_w,
                        y=coords[fi, ji, 1] / self.image_h,
                        z=float(z_vals[fi, ji]),
                        v=float(vis[fi, ji]),
                    ))
                else:
                    lm_list.append(_LandmarkProxy(
                        x=raw_lm.x, y=raw_lm.y,
                        z=raw_lm.z, v=raw_lm.visibility,
                    ))
            out.append(_ResultProxy(_LandmarkListProxy(lm_list)))

        return out


def filter_landmarks(
    raw_landmarks: list,
    image_w: int,
    image_h: int,
    fps: float = 30.0,
    verbose: bool = True,
    **kwargs,
) -> list:
    pf = PoseFilter(image_w=image_w, image_h=image_h, fps=fps,
                    verbose=verbose, **kwargs)
    return pf.filter(raw_landmarks)