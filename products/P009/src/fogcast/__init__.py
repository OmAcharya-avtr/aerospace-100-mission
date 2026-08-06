"""FogCast: fog/aerosol optical attenuation prediction for free-space optical links.

Baselines: Kim (SPIE 4214, 2001) and Kruse (1962) empirical visibility models.
ML: gradient-boosting regressor with 90 % prediction intervals, trained on a seeded
synthetic dataset (see DATASET_CARD.md). Research-grade; not certified for
operational flight use.
"""

from .baselines import (
    VISIBILITY_RANGE_KM,
    WAVELENGTH_RANGE_NM,
    kim_attenuation_db_km,
    kim_q,
    kruse_attenuation_db_km,
    kruse_q,
)
from .dataset import generate_dataset, split_indices
from .model import FogCastModel, predict

__version__ = "0.1.0"

__all__ = [
    "FogCastModel",
    "VISIBILITY_RANGE_KM",
    "WAVELENGTH_RANGE_NM",
    "__version__",
    "generate_dataset",
    "kim_attenuation_db_km",
    "kim_q",
    "kruse_attenuation_db_km",
    "kruse_q",
    "predict",
    "split_indices",
]
