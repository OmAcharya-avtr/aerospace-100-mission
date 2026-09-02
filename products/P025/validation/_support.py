"""Shared campaign builder for the validation scripts.

Adds ``src/`` to ``sys.path`` so the scripts run from a cold clone without an
install, and builds the one artefact set every benchmark script needs: the
signature bank, the training and held-out scenario sets, their simulated runs,
the fitted classifier and both threshold sets.

Every seed used anywhere in the campaign is listed here, in one place:

======================  =========================================
training scenarios      1000 .. 1000 + n_train - 1
held-out scenarios      5000 .. 5000 + n_test - 1
calibration runs        9000 .. 9000 + n_calib - 1 (all fault-free)
held-out fault-free     12000 .. 12000 + n_far - 1 (all fault-free)
classifier              ``random_state = 0``
======================  =========================================
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from fdiscope.classifier import FaultClassifier  # noqa: E402
from fdiscope.evaluate import (  # noqa: E402
    BenchmarkConfig,
    build_default_bank,
    calibrate_all_thresholds,
    default_scenario_sets,
    design_thresholds,
    harvest_training_rows,
    healthy_calibration_runs,
    run_scenarios,
)
from fdiscope.isolation import SignatureBank  # noqa: E402
from fdiscope.scenarios import Scenario  # noqa: E402
from fdiscope.simulate import LoopRun  # noqa: E402

TRAIN_SEED0 = 1000
TEST_SEED0 = 5000
CALIB_SEED0 = 9000
FAR_SEED0 = 12000
CLASSIFIER_SEED = 0
TARGET_RUN_FAR = 0.10


@dataclass(frozen=True)
class Campaign:
    """Everything the benchmark scripts share."""

    cfg: BenchmarkConfig
    bank: SignatureBank
    train: list[Scenario]
    test: list[Scenario]
    train_runs: list[LoopRun]
    test_runs: list[LoopRun]
    calib_runs: list[LoopRun]
    far_scenarios: list[Scenario]
    far_runs: list[LoopRun]
    classifier: FaultClassifier
    matched_thresholds: dict[str, float]
    design_thresholds: dict[str, float]
    features: np.ndarray
    labels: np.ndarray
    build_seconds: float


def build_campaign(
    n_train: int = 240,
    n_test: int = 240,
    n_calib: int = 150,
    n_far: int = 150,
    n_estimators: int = 150,
) -> Campaign:
    """Build the shared campaign.  Deterministic given the arguments."""
    t0 = time.perf_counter()
    cfg = BenchmarkConfig()
    bank = build_default_bank(cfg)
    train, test = default_scenario_sets(n_train, n_test, TRAIN_SEED0, TEST_SEED0)
    train_runs = run_scenarios(train, cfg)
    test_runs = run_scenarios(test, cfg)
    x, y = harvest_training_rows(train, train_runs, cfg)
    clf = FaultClassifier(n_estimators=n_estimators, random_state=CLASSIFIER_SEED).fit(x, y)
    _, calib_runs = healthy_calibration_runs(n_calib, CALIB_SEED0, cfg)
    matched = calibrate_all_thresholds(calib_runs, cfg, bank, clf, TARGET_RUN_FAR)
    far_scenarios, far_runs = healthy_calibration_runs(n_far, FAR_SEED0, cfg)
    return Campaign(
        cfg=cfg,
        bank=bank,
        train=train,
        test=test,
        train_runs=train_runs,
        test_runs=test_runs,
        calib_runs=calib_runs,
        far_scenarios=far_scenarios,
        far_runs=far_runs,
        classifier=clf,
        matched_thresholds=matched,
        design_thresholds=design_thresholds(cfg),
        features=x,
        labels=y,
        build_seconds=time.perf_counter() - t0,
    )


class Tee:
    """Write to stdout and to a file at the same time."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save(self) -> None:
        """Write the captured output beside the script that produced it."""
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        print(f"\n[raw output written to {self.path}]")
