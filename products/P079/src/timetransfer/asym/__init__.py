"""Delay-asymmetry correction: the analytic estimators first, then the learned one.

The ordering in this package is deliberate and is the mission's rule for AI
products: :mod:`timetransfer.asym.baselines` is implemented and scored before
:mod:`timetransfer.asym.learned` exists, and both are scored on the same
held-out split by :mod:`timetransfer.asym.metrics`. The result -- including a
baseline win -- is reported as measured in README.md, validation/VALIDATION.md
and MODEL_CARD.md.
"""

from __future__ import annotations

from .baselines import (
    AnalyticEstimator,
    MeanDifferenceEstimator,
    MedianFilterEstimator,
    SymmetricAssumption,
)
from .data import AsymmetryDataset, GeneratorConfig, build_dataset, generate_scenario
from .features import FEATURE_NAMES, MIN_WINDOW, centred_dwells_s, window_features
from .learned import LearnedAsymmetryCorrector, RidgeAsymmetryCorrector, TrainingConfig
from .metrics import Scores, ScoreTable, score_predictions

__all__ = [
    "FEATURE_NAMES",
    "MIN_WINDOW",
    "AnalyticEstimator",
    "AsymmetryDataset",
    "GeneratorConfig",
    "LearnedAsymmetryCorrector",
    "MeanDifferenceEstimator",
    "MedianFilterEstimator",
    "RidgeAsymmetryCorrector",
    "ScoreTable",
    "Scores",
    "SymmetricAssumption",
    "TrainingConfig",
    "build_dataset",
    "centred_dwells_s",
    "generate_scenario",
    "score_predictions",
    "window_features",
]
