"""TurbScope: path-averaged Cn^2 estimation from scintillometer and DIMM
measurements, with classical closed-form inversion and a learned multi-sensor
regressor with calibrated prediction intervals.

Research-grade software. Not flight-qualified, not certified, not approved
for operational aerospace use. See ``README.md`` and ``MODEL_CARD.md``.
"""

from __future__ import annotations

from .dimm import differential_variance, fried_parameter_from_cn2_path, invert_cn2_from_variance
from .inversion import (
    MultiSensorEstimate,
    PointEstimate,
    fuse_inverse_variance,
    invert_dimm_with_uncertainty,
    invert_scintillometer_weak_with_uncertainty,
    multi_sensor_closed_form_estimate,
)
from .model import PathCn2Prediction, TurbScopeModel, train_default_model
from .scintillometer import (
    MultiValuedInversion,
    invert_cn2_all_roots,
    invert_cn2_weak,
    rytov_variance,
    scintillation_index_full,
)

__version__ = "0.1.0"

__all__ = [
    "MultiSensorEstimate",
    "MultiValuedInversion",
    "PathCn2Prediction",
    "PointEstimate",
    "TurbScopeModel",
    "__version__",
    "differential_variance",
    "fried_parameter_from_cn2_path",
    "fuse_inverse_variance",
    "invert_cn2_all_roots",
    "invert_cn2_from_variance",
    "invert_cn2_weak",
    "invert_dimm_with_uncertainty",
    "invert_scintillometer_weak_with_uncertainty",
    "multi_sensor_closed_form_estimate",
    "rytov_variance",
    "scintillation_index_full",
    "train_default_model",
]
