"""jitterscope — platform jitter/vibration characterization for optical pointing.

Welch PSD estimation, band-limited RMS jitter budgets, Gaussian-beam
pointing-loss conversion, seeded synthetic vibration telemetry, and
telemetry anomaly detection (band z-score baseline + MLP autoencoder).

Research-grade software; not flight-qualified, not certified, and not
approved for operational aerospace use.
"""

from .detect import (
    BandZScoreBaseline,
    DetectionResult,
    FeatureExtractor,
    NominalModel,
    detect,
)
from .pointing import pointing_loss_avg, pointing_loss_avg_mc
from .psd import band_rms, cumulative_rms, psd
from .telemetry import generate_telemetry

__version__ = "0.1.0"

__all__ = [
    "psd",
    "band_rms",
    "cumulative_rms",
    "pointing_loss_avg",
    "pointing_loss_avg_mc",
    "generate_telemetry",
    "FeatureExtractor",
    "BandZScoreBaseline",
    "NominalModel",
    "DetectionResult",
    "detect",
    "__version__",
]
