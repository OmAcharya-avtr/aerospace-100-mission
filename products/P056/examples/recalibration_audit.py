"""Where recalibration stops hurting: held-out Brier change against sample size.

Run: ``python examples/recalibration_audit.py``
Writes ``screenshots/recalibration_audit.png``.
"""

from __future__ import annotations

from _common import save

from calibaudit import get_spec, sample_size_sweep
from calibaudit.plotting import plot_sample_size_sweep

SIZES = (60, 100, 200, 400, 1000, 2500, 6000)
REPLICATES = 60
SEED = 56


def main() -> None:
    spec = get_spec("overconfident")
    sweep = sample_size_sweep(
        spec, n_samples_grid=SIZES, n_replicates=REPLICATES, seed=SEED
    )
    fig = plot_sample_size_sweep(
        sweep, title=f"{REPLICATES} replicates per point, 50/50 split"
    )
    save(fig, "recalibration_audit")
    print(sweep.table())
    print()
    for method in ("platt", "isotonic"):
        crossing = sweep.crossover(method)
        if crossing is None:
            print(f"{method}: never has a negative mean held-out Brier change on this grid")
        else:
            print(
                f"{method}: mean held-out Brier change turns negative from "
                f"n_total = {crossing} upward"
            )


if __name__ == "__main__":
    main()
