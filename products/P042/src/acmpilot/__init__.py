"""acmpilot: adaptive coding and modulation on one optical link under feedback delay.

The package is five layers, each usable on its own:

``acmpilot.channel``
    Temporally correlated optical fading sample paths (lognormal and
    gamma-gamma) with an emergent fade duration.
``acmpilot.modulation`` / ``acmpilot.coding``
    Monte Carlo uncoded BER through a real modulator and detector, and exact
    Reed-Solomon bounded-distance accounting from channel BER to post-decoding
    BER.
``acmpilot.modcod``
    The MODCOD ladder and its **measured** switching thresholds, with a Monte
    Carlo uncertainty on each threshold.
``acmpilot.policy`` / ``acmpilot.simulate`` / ``acmpilot.accounting``
    Fixed-margin, hysteresis and clairvoyant-upper-bound selection under an
    explicit round-trip feedback delay, scored on goodput, outage and the two
    distinct kinds of mis-selection.
``acmpilot.predictor``
    An analytic AR(1) MMSE predictor and a learned quantile predictor over the
    feedback horizon, both emitting a confidence that gates aggressive rate
    choices.

Research-grade software. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from .accounting import Accounting, account
from .channel import (
    ChannelConfig,
    correlation_time_s,
    fade_statistics,
    gamma_gamma_irradiance,
    gamma_gamma_shapes,
    gauss_markov_path,
    irradiance_path,
    lognormal_irradiance,
    rice_level_crossing_rate_hz,
    snr_db_path,
)
from .coding import SHIPPED_CODES, ReedSolomonCode
from .modcod import (
    DEFAULT_TARGET_BER,
    Modcod,
    ModcodTable,
    measure_ber_curve,
    measure_thresholds,
    shipped_modcods,
)
from .modulation import (
    CONSTELLATION_NAMES,
    Constellation,
    constellation,
    measure_ber,
    qfunc,
    theoretical_ber,
)
from .policy import (
    CLAIRVOYANT_LABEL,
    ClairvoyantUpperBound,
    FixedMargin,
    Policy,
    ThresholdHysteresis,
    baseline_policies,
)
from .predictor import (
    QUANTILES,
    ChannelPredictor,
    GaussMarkovPredictor,
    PredictivePolicy,
    QuantilePredictor,
    analytic_calibration_report,
    calibration_report,
    make_lag_features,
)
from .simulate import (
    EpisodeResult,
    delayed_observation,
    run_episode,
    run_policy,
    sweep_delay,
)

__version__ = "0.1.0"

__all__ = [
    "CLAIRVOYANT_LABEL",
    "CONSTELLATION_NAMES",
    "DEFAULT_TARGET_BER",
    "QUANTILES",
    "SHIPPED_CODES",
    "Accounting",
    "ChannelConfig",
    "ChannelPredictor",
    "ClairvoyantUpperBound",
    "Constellation",
    "EpisodeResult",
    "FixedMargin",
    "GaussMarkovPredictor",
    "Modcod",
    "ModcodTable",
    "Policy",
    "PredictivePolicy",
    "QuantilePredictor",
    "ReedSolomonCode",
    "ThresholdHysteresis",
    "__version__",
    "account",
    "analytic_calibration_report",
    "baseline_policies",
    "calibration_report",
    "constellation",
    "correlation_time_s",
    "delayed_observation",
    "fade_statistics",
    "gamma_gamma_irradiance",
    "gamma_gamma_shapes",
    "gauss_markov_path",
    "irradiance_path",
    "lognormal_irradiance",
    "make_lag_features",
    "measure_ber",
    "measure_ber_curve",
    "measure_thresholds",
    "qfunc",
    "rice_level_crossing_rate_hz",
    "run_episode",
    "run_policy",
    "shipped_modcods",
    "snr_db_path",
    "sweep_delay",
    "theoretical_ber",
]
