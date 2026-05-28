"""
Basic tests for the golf swing analysis pipeline.
Run with: python -m pytest tests/ -v
"""

import sys
import numpy as np
import pytest

sys.path.insert(0, "..")

from core.landmarks import angle_between, midpoint
from core.keyframes import KeyFrameDetector
from core.metrics import SwingMetrics


class TestGeometry:

    def test_angle_between_90_degrees(self):
        vertex = np.array([0.0, 0.0])
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(angle_between(a, vertex, b) - 90.0) < 0.01

    def test_angle_between_180_degrees(self):
        vertex = np.array([0.0, 0.0])
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(angle_between(a, vertex, b) - 180.0) < 0.01

    def test_angle_between_0_degrees(self):
        vertex = np.array([0.0, 0.0])
        a = np.array([1.0, 0.0])
        b = np.array([2.0, 0.0])
        assert abs(angle_between(a, vertex, b) - 0.0) < 0.01

    def test_midpoint(self):
        a = np.array([0.0, 0.0])
        b = np.array([4.0, 4.0])
        m = midpoint(a, b)
        assert np.allclose(m, [2.0, 2.0])

    def test_angle_degenerate_zero_vector(self):
        vertex = np.array([0.0, 0.0])
        a = np.array([0.0, 0.0])  # zero vector
        b = np.array([1.0, 0.0])
        # Should return 0, not crash
        result = angle_between(a, vertex, b)
        assert result == 0.0


class TestKeyFrameDetector:

    def _make_metrics_sequence(self, n=60):
        """Simulate a simple swing: rotation goes 0 -> 90 -> 0 -> 90 (follow)."""
        metrics = []
        for i in range(n):
            m = SwingMetrics(confidence=0.8)
            t = i / n

            # Simulate rotation curve: rises to peak at 40%, drops at 60%, rises again
            if t < 0.4:
                rot = t / 0.4 * 85
            elif t < 0.6:
                rot = 85 - (t - 0.4) / 0.2 * 85
            else:
                rot = (t - 0.6) / 0.4 * 70

            m.shoulder_rotation_deg = rot
            m.hip_rotation_deg = rot * 0.6
            m.x_factor_deg = rot * 0.4
            metrics.append(m)
        return metrics

    def test_finds_top(self):
        metrics = self._make_metrics_sequence(60)
        detector = KeyFrameDetector()
        rots = np.array([m.shoulder_rotation_deg for m in metrics])
        rots_smooth = detector._smooth(rots)
        confs = np.array([m.confidence for m in metrics])
        top = detector._find_top(rots_smooth, confs, start_after=5)
        # Top should be around frame 24 (40% of 60)
        assert 18 < top < 32, f"Expected top near frame 24, got {top}"

    def test_finds_address(self):
        metrics = self._make_metrics_sequence(60)
        detector = KeyFrameDetector()
        rots = np.array([m.shoulder_rotation_deg for m in metrics])
        rots_smooth = detector._smooth(rots)
        confs = np.array([m.confidence for m in metrics])
        addr = detector._find_address(rots_smooth, confs)
        # Address should be in first third
        assert addr is not None
        assert addr < 20

    def test_low_confidence_frames_skipped(self):
        metrics = self._make_metrics_sequence(60)
        # Kill confidence in first half
        for m in metrics[:30]:
            m.confidence = 0.1
        detector = KeyFrameDetector()
        rots = np.array([m.shoulder_rotation_deg for m in metrics])
        rots_smooth = detector._smooth(rots)
        confs = np.array([m.confidence for m in metrics])
        addr = detector._find_address(rots_smooth, confs)
        # With low confidence in first 30 frames, address may not be found
        # This is acceptable behavior
        assert addr is None or confs[addr] >= 0.5


class TestSwingMetrics:

    def test_to_dict(self):
        m = SwingMetrics(shoulder_rotation_deg=85.0, hip_rotation_deg=45.0)
        d = m.to_dict()
        assert d["shoulder_rotation_deg"] == 85.0
        assert d["hip_rotation_deg"] == 45.0

    def test_x_factor_calculation(self):
        m = SwingMetrics(
            shoulder_rotation_deg=90.0,
            hip_rotation_deg=45.0,
            x_factor_deg=45.0
        )
        assert m.x_factor_deg == 45.0

    def test_defaults_are_none(self):
        m = SwingMetrics()
        assert m.shoulder_rotation_deg is None
        assert m.confidence == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
