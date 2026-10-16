"""Shared fixtures. The reference configuration is the one the README quotes."""

from __future__ import annotations

import numpy as np
import pytest

from acqsearch import (
    PointingUncertainty,
    PoissonDwellDetector,
    TerminalGeometry,
    build_cell_grid,
    cell_detection_probabilities,
    cell_visit_order,
    standard_paths,
)

REFERENCE = {
    "sigma_rad": 150e-6,
    "beam_divergence_rad": 100e-6,
    "detector_fov_rad": 200e-6,
    "slew_rate_rad_s": 10e-3,
    "settle_time_s": 2e-3,
    "signal_rate_hz": 2.0e4,
    "background_rate_hz": 2.0e3,
    "threshold_counts": 8,
    "rytov_variance": 0.2,
    "dwell_s": 1.4e-3,
    "verification_time_s": 0.5,
}


@pytest.fixture(scope="session")
def uncertainty() -> PointingUncertainty:
    """Reference pointing-uncertainty prior, 150 urad per axis."""
    return PointingUncertainty(REFERENCE["sigma_rad"])


@pytest.fixture(scope="session")
def geometry() -> TerminalGeometry:
    """Reference terminal geometry."""
    return TerminalGeometry(
        beam_divergence_rad=REFERENCE["beam_divergence_rad"],
        detector_fov_rad=REFERENCE["detector_fov_rad"],
        slew_rate_rad_s=REFERENCE["slew_rate_rad_s"],
        settle_time_s=REFERENCE["settle_time_s"],
    )


@pytest.fixture(scope="session")
def detector() -> PoissonDwellDetector:
    """Reference photon-counting dwell detector."""
    return PoissonDwellDetector(
        signal_rate_hz=REFERENCE["signal_rate_hz"],
        background_rate_hz=REFERENCE["background_rate_hz"],
        threshold_counts=REFERENCE["threshold_counts"],
        rytov_variance=REFERENCE["rytov_variance"],
    )


@pytest.fixture(scope="session")
def grid(uncertainty, geometry):
    """Reference cell grid, 89 cells out to 2.5 sigma."""
    return build_cell_grid(uncertainty, geometry, extent_sigma=2.5)


@pytest.fixture(scope="session")
def cell_q(grid, geometry, detector, uncertainty) -> np.ndarray:
    """Cell detection probabilities at the reference dwell time."""
    return cell_detection_probabilities(
        grid, geometry, detector, REFERENCE["dwell_s"], uncertainty
    )


@pytest.fixture(scope="session")
def spiral_order(grid) -> np.ndarray:
    """Cell visit order induced by the reference spiral path."""
    return cell_visit_order(standard_paths(grid)["spiral"], grid)
