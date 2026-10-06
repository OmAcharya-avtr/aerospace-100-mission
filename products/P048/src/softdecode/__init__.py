"""softdecode - soft-decision demapping over fading optical channels.

Exact and approximate log-likelihood ratios for OOK and M-ary PPM over
lognormal and gamma-gamma fading under a thermal-limited (signal-independent)
additive Gaussian noise model, the measured cost of the max-log approximation
and of LLR clipping, the mismatched-channel-state penalty in both directions,
and a learned LLR corrector benchmarked against the analytic alternatives.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.

Copyright 2026 OPTIMA Organisation. Licensed under the Apache License 2.0.
"""

from __future__ import annotations

from .channel import (
    GammaGammaFading,
    LognormalFading,
    amplitude_quadrature,
)
from .codes import EXTENDED_HAMMING_84, ExtendedHamming84
from .corrector import (
    FEATURE_NAMES,
    LlrCorrector,
    TunedParameters,
    build_features,
    gmi_of,
    tune_clip,
    tune_scale_and_clip,
)
from .csi import (
    MultiplicativeCsiError,
    StaleCsiError,
    csi_aware_llr_ook,
    db_to_log_gain,
    posterior_quadrature,
)
from .datasets import BASE_SEED, TRAIN_EBN0_DB, make_split, make_training_set
from .detection import DetectionModel, db_to_ratio, ratio_to_db
from .ldpc import LdpcCode, make_regular_ldpc, sum_product_decode
from .llr import (
    clip_llr,
    gray_labels,
    llr_ook_known_csi,
    llr_ook_marginal,
    llr_ook_maxlog,
    llr_ppm_known_csi,
    llr_ppm_marginal,
    llr_ppm_maxlog,
    rescale_llr,
)
from .metrics import (
    BitErrorRate,
    LlrError,
    bit_error_rate,
    generalised_mutual_information,
    llr_error,
)
from .simulate import DEMAPPERS, OokRealisation, decode_ber, demap_ook, simulate_ook

__version__ = "0.1.0"

__all__ = [
    "BASE_SEED",
    "DEMAPPERS",
    "EXTENDED_HAMMING_84",
    "FEATURE_NAMES",
    "TRAIN_EBN0_DB",
    "BitErrorRate",
    "DetectionModel",
    "ExtendedHamming84",
    "GammaGammaFading",
    "LdpcCode",
    "LlrCorrector",
    "LlrError",
    "LognormalFading",
    "MultiplicativeCsiError",
    "OokRealisation",
    "StaleCsiError",
    "TunedParameters",
    "__version__",
    "amplitude_quadrature",
    "bit_error_rate",
    "build_features",
    "clip_llr",
    "csi_aware_llr_ook",
    "db_to_log_gain",
    "db_to_ratio",
    "decode_ber",
    "demap_ook",
    "generalised_mutual_information",
    "gmi_of",
    "gray_labels",
    "llr_error",
    "llr_ook_known_csi",
    "llr_ook_marginal",
    "llr_ook_maxlog",
    "llr_ppm_known_csi",
    "llr_ppm_marginal",
    "llr_ppm_maxlog",
    "make_regular_ldpc",
    "make_split",
    "make_training_set",
    "posterior_quadrature",
    "ratio_to_db",
    "rescale_llr",
    "simulate_ook",
    "sum_product_decode",
    "tune_clip",
    "tune_scale_and_clip",
]
