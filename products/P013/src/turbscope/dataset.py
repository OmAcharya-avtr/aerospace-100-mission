"""Feature-table construction for the learned multi-sensor model.

Builds ``(X, y)`` tables from :mod:`turbscope.synthetic` scenarios, one row
per (scenario, noise-realisation) pair, split **by scenario** so that
multiple noisy realisations of the same ground truth never appear on both
sides of a train/test boundary.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .synthetic import Scenario, generate_scenarios, synthesize_measurement

__all__ = [
    "FEATURE_NAMES",
    "N_REALISATIONS_DEFAULT",
    "build_table",
    "grouped_split",
]

FEATURE_NAMES: tuple[str, ...] = (
    "log10_sigma_i2_scint",
    "log10_var_long_dimm",
    "log10_var_trans_dimm",
    "log10_path_length_m",
)
"""Model input columns, in order. All four are quantities an operator has in
hand: two sensor readings, one derived combination, and the surveyed path
length -- never the target Cn2 itself."""

N_REALISATIONS_DEFAULT: int = 3
"""Default number of independent noisy measurement draws per scenario."""


def build_table(
    scenarios: list[Scenario], n_realisations: int = N_REALISATIONS_DEFAULT, seed: int = 99
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """Build a feature/target table with ``n_realisations`` noisy draws per scenario.

    Parameters
    ----------
    scenarios : list of Scenario
        Ground-truth cases (see :func:`turbscope.synthetic.generate_scenarios`).
    n_realisations : int
        Independent noisy measurement draws per scenario (>= 1).
    seed : int
        Seed for the noise generator (independent of the scenario-draw seed).

    Returns
    -------
    (X, y, groups) : tuple of ndarray
        ``X`` shape (n_rows, 4) in :data:`FEATURE_NAMES` order; ``y`` shape
        (n_rows,), ``log10(Cn2_path)``; ``groups`` shape (n_rows,), the index
        of the source scenario (for a grouped train/test split).
    """
    if not scenarios:
        raise ValueError("scenarios must be non-empty.")
    n_real = int(n_realisations)
    if n_real < 1:
        raise ValueError("n_realisations must be >= 1.")
    rng = np.random.default_rng(seed)
    rows = []
    targets = []
    groups = []
    for gi, sc in enumerate(scenarios):
        for _ in range(n_real):
            m = synthesize_measurement(sc, rng)
            rows.append(
                [
                    np.log10(m.sigma_i2_scint),
                    np.log10(m.var_long_dimm),
                    np.log10(m.var_trans_dimm),
                    np.log10(m.path_length_m),
                ]
            )
            targets.append(np.log10(sc.cn2_path))
            groups.append(gi)
    return (
        np.asarray(rows, dtype=float),
        np.asarray(targets, dtype=float),
        np.asarray(groups, dtype=np.intp),
    )


def grouped_split(
    n_scenarios: int, test_fraction: float = 0.25, seed: int = 4242
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Deterministic shuffle-split of scenario indices ``range(n_scenarios)``.

    Thin, explicitly-named wrapper so callers building fit/calibration/test
    splits read clearly at the call site; the mechanics are shared with
    :func:`turbscope.synthetic.split_indices`.
    """
    from .synthetic import split_indices

    return split_indices(n_scenarios, test_fraction, seed)


def generate_default_scenarios(n_scenarios: int, seed: int = 20260829) -> list[Scenario]:
    """Convenience re-export of :func:`turbscope.synthetic.generate_scenarios`."""
    return generate_scenarios(n_scenarios, seed)
