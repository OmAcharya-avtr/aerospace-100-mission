"""WaveForge -- adaptive-optics design, simulation and predictive control.

WaveForge sizes an adaptive-optics system for a free-space optical link:
Kolmogorov phase-screen atmosphere, Zernike and zonal wavefront
representations, a Shack-Hartmann sensor with photon and read noise, a
deformable mirror with finite actuator stroke, a closed-loop integrator with
configurable gain and latency, a decomposed residual-error budget, and a
learned predictive controller benchmarked against both a classical integrator
and a pure-delay baseline.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

Copyright (C) 2026 OPTIMA Organisation. Licensed under the GNU Affero General
Public License v3.0 or later.
"""

from __future__ import annotations

from .atmosphere import (
    FrozenFlow,
    PhaseScreen,
    coherence_time,
    discrete_structure_function,
    fried_parameter,
    greenwood_frequency,
    kolmogorov_psd,
    structure_function,
    theoretical_structure_function,
)
from .budget import (
    ErrorBudget,
    fitting_error,
    marechal_inverse,
    noise_error,
    noise_propagation_coefficient,
    strehl_exact,
    strehl_marechal,
    temporal_error,
    temporal_error_bandwidth,
)
from .control import (
    Integrator,
    closed_loop_poles,
    rejection_transfer_function,
    stability_gain_limit,
)
from .dm import DeformableMirror
from .loop import AOSystem, LoopResult, run_closed_loop
from .predictor import (
    EnsemblePredictor,
    LinearPredictor,
    PersistencePredictor,
    build_windows,
)
from .pupil import Pupil
from .sensor import ShackHartmann
from .zernike import (
    NOLL_RESIDUALS,
    ZernikeBasis,
    nm_to_noll,
    noll_mode_variance,
    noll_residual,
    noll_to_nm,
    orthonormality_matrix,
    zernike,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # pupil / modal
    "Pupil",
    "ZernikeBasis",
    "zernike",
    "noll_to_nm",
    "nm_to_noll",
    "noll_residual",
    "noll_mode_variance",
    "orthonormality_matrix",
    "NOLL_RESIDUALS",
    # atmosphere
    "PhaseScreen",
    "FrozenFlow",
    "kolmogorov_psd",
    "structure_function",
    "theoretical_structure_function",
    "discrete_structure_function",
    "fried_parameter",
    "greenwood_frequency",
    "coherence_time",
    # hardware
    "ShackHartmann",
    "DeformableMirror",
    # control
    "Integrator",
    "rejection_transfer_function",
    "stability_gain_limit",
    "closed_loop_poles",
    # loop
    "AOSystem",
    "LoopResult",
    "run_closed_loop",
    # budget
    "ErrorBudget",
    "strehl_exact",
    "strehl_marechal",
    "marechal_inverse",
    "fitting_error",
    "temporal_error",
    "temporal_error_bandwidth",
    "noise_propagation_coefficient",
    "noise_error",
    # prediction
    "PersistencePredictor",
    "LinearPredictor",
    "EnsemblePredictor",
    "build_windows",
]
