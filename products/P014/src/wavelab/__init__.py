"""WaveLab -- slope-to-phase wavefront reconstruction.

WaveLab takes Shack-Hartmann **slopes** and returns a **wavefront**: regularised
least-squares zonal reconstruction on both the Southwell and the Fried geometry,
modal (Zernike) least-squares reconstruction, an analytic noise-propagation
model, Kolmogorov synthetic data, and a learned reconstructor with ensemble
uncertainty benchmarked against the classical baselines.

Status: TESTING. Research-grade; not flight-qualified, not certified, and not
approved for operational aerospace use.
"""

from __future__ import annotations

from .benchmark import (
    OperatingPoint,
    evaluate_modal_baseline,
    make_operating_point,
    residual_rms,
    tune_modal_regularisation,
)
from .dataset import DEFAULT_NOLL_INDICES, SlopeDataset, generate_dataset, make_measurements
from .geometry import GeometryMatrices, SubapertureGeometry, build_geometry_matrices
from .ml import EnsembleZernikeReconstructor, build_features
from .noise import (
    SLOPE_NOISE_COEFF,
    add_slope_noise,
    photon_slope_noise,
    random_dropout_mask,
)
from .reconstruct import (
    ModalReconstructor,
    ZonalReconstructor,
    noise_propagation_coefficient,
    piston_remove,
    regularised_pinv,
)
from .turbulence import (
    KOLMOGOROV_PSD_COEFF,
    STRUCTURE_FUNCTION_COEFF,
    KolmogorovScreens,
    kolmogorov_psd,
    measure_structure_function,
    structure_function_theory,
)
from .zernike import (
    nm_to_noll,
    noll_to_nm,
    noll_to_osa,
    zernike_basis,
    zernike_gradient_basis,
    zernike_gradient_noll,
    zernike_noll,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_NOLL_INDICES",
    "KOLMOGOROV_PSD_COEFF",
    "STRUCTURE_FUNCTION_COEFF",
    "SLOPE_NOISE_COEFF",
    "EnsembleZernikeReconstructor",
    "GeometryMatrices",
    "KolmogorovScreens",
    "ModalReconstructor",
    "OperatingPoint",
    "SlopeDataset",
    "SubapertureGeometry",
    "ZonalReconstructor",
    "add_slope_noise",
    "build_features",
    "build_geometry_matrices",
    "evaluate_modal_baseline",
    "generate_dataset",
    "kolmogorov_psd",
    "make_measurements",
    "make_operating_point",
    "measure_structure_function",
    "nm_to_noll",
    "noise_propagation_coefficient",
    "noll_to_nm",
    "noll_to_osa",
    "photon_slope_noise",
    "piston_remove",
    "random_dropout_mask",
    "regularised_pinv",
    "residual_rms",
    "structure_function_theory",
    "tune_modal_regularisation",
    "zernike_basis",
    "zernike_gradient_basis",
    "zernike_gradient_noll",
    "zernike_noll",
]
