"""The analytic baseline against the learned model, in and out of distribution.

Writes ``screenshots/baseline_vs_learned.png``. The baseline is implemented
first and benchmarked on the same held-out data; whichever wins, that is the
published result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from conformalband.baseline import PhysicsRegressor
from conformalband.data import make_dataset
from conformalband.learned import LearnedRegressor
from conformalband.plotting import plot_model_comparison
from conformalband.shift import SHIFT_LEVELS, CovariateShift

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "screenshots" / "baseline_vs_learned.png"

REPLICATES = 30
N_FIT = 1500
N_TEST = 1000
SEED = 57041


def main() -> None:
    severities = np.array(SHIFT_LEVELS, dtype=float)
    totals = {"physics": np.zeros(severities.size), "learned": np.zeros(severities.size)}
    for index in range(REPLICATES):
        rng = np.random.default_rng(SEED + index)
        fit = make_dataset(N_FIT, rng=rng, severity=0.0)
        tests = [
            make_dataset(N_TEST, rng=rng, shift=CovariateShift(severity=float(s)))
            for s in severities
        ]
        models = {
            "physics": PhysicsRegressor().fit(fit.features, fit.energy),
            "learned": LearnedRegressor(random_state=SEED + index).fit(fit.features, fit.energy),
        }
        for name, model in models.items():
            for position, test in enumerate(tests):
                residual = test.energy - model.predict(test.features)
                totals[name][position] += float(np.mean(residual**2))
    rmse = {name: np.sqrt(total / REPLICATES) for name, total in totals.items()}

    header = f"{'severity':>8s} {'physics_Wh':>11s} {'learned_Wh':>11s} {'ratio':>8s}"
    print(header)
    print("-" * len(header))
    for position, severity in enumerate(severities):
        a = rmse["physics"][position]
        b = rmse["learned"][position]
        print(f"{severity:8.1f} {a:11.6f} {b:11.6f} {b / a:8.5f}")
    print()
    print(
        "ratio < 1 means the learned model has the lower root mean squared error. "
        f"{REPLICATES} replicates, n_fit={N_FIT}, n_test={N_TEST}, seed {SEED}."
    )
    path = plot_model_comparison(severities, rmse, OUTPUT)
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
