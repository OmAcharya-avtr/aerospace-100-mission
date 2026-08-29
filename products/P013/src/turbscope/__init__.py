"""TurbScope: path-averaged Cn^2 inference from scintillation and image-motion sensors.

TurbScope goes from *measurements* to turbulence strength.  It provides:

* forward models for the two commonest optical Cn^2 estimators -- irradiance
  scintillation (scintillometer) and differential image motion (DIMM);
* a seeded synthetic measurement generator driven by known Cn^2 profiles;
* closed-form inversion to a path-averaged Cn^2 with an uncertainty interval,
  including explicit handling of the multi-valued saturation regime;
* a learned multi-sensor regressor with calibrated prediction intervals,
  benchmarked against the closed-form inversions on the same held-out data.

Related work in this portfolio (cited, never imported): **P020 AtmoProfile**
computes turbulence integrals from a *known* Cn^2 profile; **P019 CnCast**
*predicts* a vertical Cn^2 profile from surface meteorology.  TurbScope solves the
inverse problem neither of them addresses: recovering path-averaged Cn^2 from
instrument readings.

Status: TESTING.  Research-grade; not flight-qualified, not certified, and not
approved for operational aerospace use.
"""

from __future__ import annotations

from .dataset import (
    FEATURE_NAMES,
    REGIME_NAMES,
    SyntheticDataset,
    features_from_measurement,
    generate_dataset,
)
from .dimm import (
    cn2_average_from_fried,
    dimm_coefficient,
    dimm_variance,
    fried_from_average,
    fried_parameter,
    r0_from_dimm_variance,
    seeing_fwhm_rad,
)
from .geometry import (
    PathGeometry,
    coherence_weight,
    scintillation_weight,
    wavenumber,
    weight_normalisation,
    weighted_path_average,
)
from .inversion import (
    Cn2Estimate,
    SaturationReport,
    invert_dimm,
    invert_scintillation,
    saturation_report,
    scintillation_branches,
)
from .measurements import Measurement, SensorSuite, simulate_measurement
from .model import BASELINES, Prediction, TurbScopeModel, train_default_model
from .scintillation import (
    RYTOV_COEFFICIENT,
    aperture_parameter_sq,
    gamma_gamma_parameters,
    rytov_variance,
    rytov_variance_from_average,
    saturation_peak,
    scintillation_index,
)

__version__ = "0.1.0"

__all__ = [
    "BASELINES",
    "Cn2Estimate",
    "FEATURE_NAMES",
    "Measurement",
    "PathGeometry",
    "Prediction",
    "REGIME_NAMES",
    "RYTOV_COEFFICIENT",
    "SaturationReport",
    "SensorSuite",
    "SyntheticDataset",
    "TurbScopeModel",
    "__version__",
    "aperture_parameter_sq",
    "cn2_average_from_fried",
    "coherence_weight",
    "dimm_coefficient",
    "dimm_variance",
    "features_from_measurement",
    "fried_from_average",
    "fried_parameter",
    "gamma_gamma_parameters",
    "generate_dataset",
    "invert_dimm",
    "invert_scintillation",
    "r0_from_dimm_variance",
    "rytov_variance",
    "rytov_variance_from_average",
    "saturation_peak",
    "saturation_report",
    "scintillation_branches",
    "scintillation_index",
    "scintillation_weight",
    "seeing_fwhm_rad",
    "simulate_measurement",
    "train_default_model",
    "wavenumber",
    "weight_normalisation",
    "weighted_path_average",
]
