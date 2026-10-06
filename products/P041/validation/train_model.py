"""Deterministic regeneration of the committed fade-exceedance model.

Writes ``fade_exceedance_model.joblib`` next to this script (a joblib archive, not
a model binary of a forbidden type) containing the fitted forest, the feature
names, the training configuration and the three baseline constants. Every seed is
fixed, so re-running this script reproduces the artefact exactly.

Run: ``PYTHONPATH=../src python3 train_model.py``
Runtime: about 45 s on one core.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib

from codedfade.fade import (
    exponential_exceedance,
    lognormal_standard_level,
    markov_mean_fade_duration,
)
from codedfade.predictor import (
    FEATURE_NAMES,
    FadeExceedancePredictor,
    build_dataset,
    empirical_exceedance_baseline,
)

CONFIG = {
    "scintillation_index": 0.6,
    "correlation_time_s": 2.0e-4,
    "sample_rate_hz": 1.0e6,
    "marginal": "lognormal",
    "kernel": "exp",
    "threshold_amplitude": 0.6,
    "target_samples": 14,
    "samples_per_path": 200_000,
    "train_seeds": list(range(24)),
    "test_seeds": list(range(100, 112)),
    "n_estimators": 160,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "forest_seed": 0,
}

ARTEFACT = "fade_exceedance_model.joblib"


def main() -> None:
    print("=" * 78)
    print("MODEL REGENERATION")
    print("=" * 78)
    for key, value in CONFIG.items():
        print(f"  {key:22s} {value}")
    print()
    t0 = time.perf_counter()
    train = build_dataset(
        CONFIG["train_seeds"],
        CONFIG["samples_per_path"],
        CONFIG["threshold_amplitude"],
        CONFIG["target_samples"],
        CONFIG["scintillation_index"],
        CONFIG["correlation_time_s"],
        CONFIG["sample_rate_hz"],
    )
    print(f"training events          {train.features.shape[0]} "
          f"({time.perf_counter() - t0:.1f} s)")
    print(f"exceedance frequency     {float(train.labels.mean()):.9f}")
    t0 = time.perf_counter()
    predictor = FadeExceedancePredictor(
        CONFIG["n_estimators"], CONFIG["max_depth"], CONFIG["forest_seed"]
    ).fit(train.features, train.labels)
    print(f"forest fitted in         {time.perf_counter() - t0:.1f} s")

    u = lognormal_standard_level(
        CONFIG["threshold_amplitude"], CONFIG["scintillation_index"]
    )
    mfd_s = markov_mean_fade_duration(
        u, CONFIG["correlation_time_s"], CONFIG["sample_rate_hz"]
    )
    analytic = float(
        exponential_exceedance(
            CONFIG["target_samples"] / CONFIG["sample_rate_hz"], mfd_s
        )
    )
    empirical = empirical_exceedance_baseline(train.labels)
    payload = {
        "config": CONFIG,
        "feature_names": list(FEATURE_NAMES),
        "model": predictor.model,
        "baseline_analytic_configured": analytic,
        "baseline_empirical_constant": empirical,
        "analytic_mfd_s": mfd_s,
        "standard_level_u": u,
        "training_events": int(train.features.shape[0]),
        "training_exceedance_frequency": float(train.labels.mean()),
        "feature_importances": predictor.feature_importances.tolist(),
    }
    out = Path(__file__).resolve().parent / ARTEFACT
    # lzma keeps the committed archive under a megabyte (zlib leaves it at 1.3 MB,
    # above the build guide's cap on committed data). The uncompressed forest is
    # 3.4 MB. Nothing is lost: joblib round-trips the payload exactly.
    joblib.dump(payload, out, compress=("lzma", 9))
    print()
    print(f"analytic baseline        {analytic:.9f}  (eq(16) with the exact eq(15) MFD)")
    print(f"empirical baseline       {empirical:.9f}  (measured training frequency)")
    print(f"analytic MFD             {mfd_s:.9e} s = "
          f"{mfd_s * CONFIG['sample_rate_hz']:.4f} samples")
    print()
    print("feature importances")
    for name, imp in zip(FEATURE_NAMES, predictor.feature_importances, strict=True):
        print(f"  {name:34s} {imp:.6f}")
    print()
    print(f"artefact written         {ARTEFACT} ({out.stat().st_size} bytes)")
    print("=" * 78)


if __name__ == "__main__":
    main()
