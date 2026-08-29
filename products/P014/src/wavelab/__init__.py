"""WaveLab: slope-to-phase wavefront reconstruction toolkit.

Zonal (Hudgin/Fried finite-difference geometry matrices) and modal
(Zernike-coefficient) regularized least-squares reconstruction, synthetic
Kolmogorov-screen slope data, and a learned slopes-to-Zernike ensemble
reconstructor with ensemble uncertainty, benchmarked against the regularized
modal least-squares baseline across photon flux and subaperture-dropout rate.

This software is research-grade. It is not flight-qualified, not certified,
and not approved for operational aerospace use (README "Safety statement").
"""

from __future__ import annotations

__version__ = "0.1.0"

from .geometry import PupilGrid, fried_matrix, hudgin_matrix, waffle_pattern
from .linalg import noise_propagation_coefficients, null_space, tikhonov_solve, tsvd_solve
from .ml import ZernikeSlopeEnsemble
from .modal import ModalReconstructor
from .noise import add_slope_noise, apply_dropout, slope_sigma
from .screens import kolmogorov_screen
from .zernike import (
    fit_zernike,
    noll_to_nm,
    nm_to_noll,
    unit_disc_grid,
    zernike,
    zernike_basis_matrix,
    zernike_gradient,
    zernike_noll,
    zernike_slope_matrix,
)
from .zonal import ZonalReconstructor

__all__ = [
    "__version__",
    "PupilGrid",
    "fried_matrix",
    "hudgin_matrix",
    "waffle_pattern",
    "noise_propagation_coefficients",
    "null_space",
    "tikhonov_solve",
    "tsvd_solve",
    "ZernikeSlopeEnsemble",
    "ModalReconstructor",
    "add_slope_noise",
    "apply_dropout",
    "slope_sigma",
    "kolmogorov_screen",
    "fit_zernike",
    "noll_to_nm",
    "nm_to_noll",
    "unit_disc_grid",
    "zernike",
    "zernike_basis_matrix",
    "zernike_gradient",
    "zernike_noll",
    "zernike_slope_matrix",
    "ZonalReconstructor",
]
