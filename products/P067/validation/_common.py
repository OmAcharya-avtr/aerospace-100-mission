"""Shared configuration for the validation scripts. Not part of the package."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "scenarios.json"

# The reference configuration every quoted number uses unless stated otherwise.
SIGMA_RAD = 150e-6
BEAM_DIVERGENCE_RAD = 100e-6
DETECTOR_FOV_RAD = 200e-6
SLEW_RATE_RAD_S = 10e-3
SETTLE_TIME_S = 2e-3
SIGNAL_RATE_HZ = 2.0e4
BACKGROUND_RATE_HZ = 2.0e3
THRESHOLD_COUNTS = 8
RYTOV_VARIANCE = 0.2
DWELL_S = 1.4e-3
VERIFICATION_TIME_S = 0.5
EXTENT_SIGMA = 2.5

SEED = 20261016


def reference():
    """Build the reference uncertainty, geometry, detector and cell grid."""
    from acqsearch import (
        PointingUncertainty,
        PoissonDwellDetector,
        TerminalGeometry,
        build_cell_grid,
    )

    uncertainty = PointingUncertainty(SIGMA_RAD)
    geometry = TerminalGeometry(
        beam_divergence_rad=BEAM_DIVERGENCE_RAD,
        detector_fov_rad=DETECTOR_FOV_RAD,
        slew_rate_rad_s=SLEW_RATE_RAD_S,
        settle_time_s=SETTLE_TIME_S,
    )
    detector = PoissonDwellDetector(
        signal_rate_hz=SIGNAL_RATE_HZ,
        background_rate_hz=BACKGROUND_RATE_HZ,
        threshold_counts=THRESHOLD_COUNTS,
        rytov_variance=RYTOV_VARIANCE,
    )
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=EXTENT_SIGMA)
    return uncertainty, geometry, detector, grid


def rule(stream, text: str = "") -> None:
    """Write a section rule with an optional title."""
    stream.write("=" * 78 + "\n")
    if text:
        stream.write(text + "\n")
        stream.write("=" * 78 + "\n")
