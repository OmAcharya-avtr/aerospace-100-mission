"""WaveForge — adaptive-optics sizing, closed-loop simulation and predictive control.

A self-contained, research-grade toolkit for sizing an adaptive-optics system
against a turbulence profile: Kolmogorov phase screens, Zernike and zonal
wavefront representations, a Shack-Hartmann sensor model with photon and read
noise, a deformable mirror with influence functions and stroke limits, a
closed-loop integrator with configurable gain and latency, a cited residual
error budget, and a learned predictive controller benchmarked against both the
classical integrator and a pure-delay baseline.

This software is research-grade.  It is not flight-qualified, not certified,
and not approved for operational aerospace use.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .atmosphere import FrozenFlowAtmosphere, phase_screen, screen_psd, structure_function
from .control import (
    Integrator,
    noise_transfer,
    noise_variance_gain,
    rejection_transfer,
    stability_limit_gain,
)
from .datasets import SlopeDataset, make_slope_dataset
from .dm import DeformableMirror
from .errorbudget import (
    ErrorBudget,
    bandwidth_error,
    delay_error,
    fitting_error,
    ideal_filter_fitting_coefficient,
    noise_error,
    strehl_marechal,
    strehl_marechal_quadratic,
    variance_from_strehl,
)
from .loop import AOConfig, AOSystem, LoopResult, SlopePredictor
from .predictor import LinearSlopePredictor, PureDelayPredictor, build_lagged_dataset
from .pupil import PupilGrid, piston_removed, rms, strehl_from_field, variance
from .sensor import ShackHartmann, SlopeMeasurement
from .statistics import (
    NOLL_RESIDUAL_TABLE,
    fried_parameter_from_cn2,
    greenwood_frequency,
    greenwood_time_constant,
    noll_residual_asymptote,
    noll_residual_variance,
    phase_structure_function,
    total_phase_variance,
    zernike_variance,
)
from .zernike import (
    fit_zernike,
    nm_to_noll,
    noll_indices,
    noll_to_nm,
    radial_polynomial,
    zernike_basis,
    zernike_cartesian,
    zernike_gradient_basis,
    zernike_polar,
)

__all__ = [
    "AOConfig",
    "AOSystem",
    "DeformableMirror",
    "ErrorBudget",
    "FrozenFlowAtmosphere",
    "Integrator",
    "LinearSlopePredictor",
    "LoopResult",
    "NOLL_RESIDUAL_TABLE",
    "PupilGrid",
    "PureDelayPredictor",
    "ShackHartmann",
    "SlopeDataset",
    "SlopeMeasurement",
    "SlopePredictor",
    "__version__",
    "bandwidth_error",
    "build_lagged_dataset",
    "delay_error",
    "fit_zernike",
    "fitting_error",
    "fried_parameter_from_cn2",
    "greenwood_frequency",
    "greenwood_time_constant",
    "ideal_filter_fitting_coefficient",
    "make_slope_dataset",
    "nm_to_noll",
    "noise_error",
    "noise_transfer",
    "noise_variance_gain",
    "noll_indices",
    "noll_residual_asymptote",
    "noll_residual_variance",
    "noll_to_nm",
    "phase_screen",
    "phase_structure_function",
    "piston_removed",
    "radial_polynomial",
    "rejection_transfer",
    "rms",
    "screen_psd",
    "stability_limit_gain",
    "strehl_from_field",
    "strehl_marechal",
    "strehl_marechal_quadratic",
    "structure_function",
    "total_phase_variance",
    "variance",
    "variance_from_strehl",
    "zernike_basis",
    "zernike_cartesian",
    "zernike_gradient_basis",
    "zernike_polar",
    "zernike_variance",
]
