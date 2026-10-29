"""Scoring. One definition of every number this repository reports.

Three scoring ideas are kept apart on purpose, because conflating them is how a
useless estimator looks useful.

**Error against injected truth, split by subspace.** The total error of an
alignment estimate on a rank-deficient record is dominated by the component the
record cannot support, and every estimator -- including one that correctly
refuses -- carries that component in full. Reporting a single total error
therefore says almost nothing. :func:`subspace_split` separates the error into
the part inside the span of the floor-clearing degrees of freedom and the part
outside it, and every score in this repository is reported for both.

**Error against the Cramer-Rao bound.** For the unbiased batch solution the
standardised error along degree of freedom ``i``,

    (24)  z_i = ( theta_hat - theta )^T v_i / sigma_CRB,i

is asymptotically standard normal, so the measured distribution of ``z`` is a
check on the estimator and on the bound together. A biased estimator -- the MAP
solution of (18), any learned estimator -- can have ``|z| < 1`` on average,
which is not a defect and not a win: it is the bias-variance trade and it is
reported with the measured bias beside it.

**Interval coverage.** :func:`coverage` is the fraction of records whose truth
lies inside ``k`` reported standard deviations. An estimator whose point error
is small but whose coverage is far below the nominal value is overconfident,
and that is the measurement this product exists to make in the unobservable
subspace. Nominal one-sided coverage for a standard normal at ``k = 1`` is
0.682689 and at ``k = 2`` is 0.954500; those are the numbers the measured
coverage is compared against and they are printed with every coverage result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .observability import SensorObservability

__all__ = [
    "NOMINAL_COVERAGE",
    "ScoreTable",
    "coverage",
    "crb_z_scores",
    "score",
    "subspace_split",
]

#: Two-sided coverage of a standard normal at k standard deviations. Computed
#: from ``scipy.special.erf`` at import so no figure here is hand-written.
NOMINAL_COVERAGE: dict[int, float] = {}


def _fill_nominal() -> None:
    from scipy.special import erf

    for k in (1, 2, 3):
        NOMINAL_COVERAGE[k] = float(erf(k / np.sqrt(2.0)))


_fill_nominal()


def subspace_split(
    error: np.ndarray, sensor: SensorObservability
) -> tuple[np.ndarray, np.ndarray]:
    """Split an error vector into its observable and unobservable parts.

    Parameters
    ----------
    error:
        Shape ``(3,)`` in radians, usually ``theta_hat - theta_true``.
    sensor:
        The observability verdict whose floor-clearing subspace defines the
        split.

    Returns
    -------
    (observable, unobservable)
        Two ``(3,)`` vectors in radians summing to ``error``.
    """
    vector = np.asarray(error, dtype=float).reshape(3)
    observable = sensor.project_observable(vector)
    return observable, vector - observable


def crb_z_scores(error: np.ndarray, sensor: SensorObservability) -> np.ndarray:
    """Standardised error of equation (24) along each degree of freedom.

    Returns a ``(3,)`` array. Entries for degrees of freedom below the noise
    floor are ``nan``, not a large number: the bound there is infinite and
    dividing by it would report a perfect score for a quantity that was never
    estimated.
    """
    vector = np.asarray(error, dtype=float).reshape(3)
    out = np.full(3, np.nan)
    for dof in sensor.degrees_of_freedom:
        if dof.verdict == "unobservable" or not np.isfinite(dof.crb_sigma_rad):
            continue
        out[dof.index] = float(np.dot(vector, dof.direction)) / dof.crb_sigma_rad
    return out


def coverage(errors: np.ndarray, sigmas: np.ndarray, k: float = 2.0) -> float:
    """Fraction of ``|errors| <= k * sigmas``. Both arrays in the same unit.

    A zero or non-finite sigma is counted as a miss unless the error is also
    zero, because an interval of zero width covers nothing.
    """
    magnitude = np.abs(np.asarray(errors, dtype=float).reshape(-1))
    width = k * np.asarray(sigmas, dtype=float).reshape(-1)
    if magnitude.shape != width.shape:
        raise ValueError(
            f"coverage() got {magnitude.shape} errors and {width.shape} sigmas"
        )
    if magnitude.size == 0:
        raise ValueError("coverage() needs at least one record")
    inside = np.where(
        np.isfinite(width) & (width > 0.0),
        magnitude <= width,
        magnitude == 0.0,
    )
    return float(np.mean(inside))


@dataclass(frozen=True)
class ScoreTable:
    """Errors of one estimator over a population, split by subspace.

    All error figures are in **milliradians**; the conversion is applied once,
    here, so no caller repeats it. ``records`` is the number of records scored.
    """

    label: str
    records: int
    rmse_total_mrad: float
    rmse_observable_mrad: float
    rmse_unobservable_mrad: float
    median_total_mrad: float
    median_observable_mrad: float
    median_unobservable_mrad: float
    bias_mrad: np.ndarray
    worst_total_mrad: float

    def row(self) -> str:
        """One fixed-width table row, for the validation transcripts."""
        return (
            f"{self.label:<22s} {self.records:>5d} "
            f"{self.rmse_total_mrad:>12.6f} {self.rmse_observable_mrad:>12.6f} "
            f"{self.rmse_unobservable_mrad:>12.6f} {self.median_total_mrad:>12.6f} "
            f"{self.median_observable_mrad:>12.6f} {self.worst_total_mrad:>12.6f}"
        )

    @staticmethod
    def header() -> str:
        """Matching header for :meth:`row`."""
        return (
            f"{'estimator':<22s} {'n':>5s} {'RMSE tot':>12s} {'RMSE obs':>12s} "
            f"{'RMSE unobs':>12s} {'med tot':>12s} {'med obs':>12s} {'worst tot':>12s}"
        )


def score(
    label: str,
    errors: np.ndarray,
    sensors: list[SensorObservability],
) -> ScoreTable:
    """Score a population of error vectors against their observability verdicts.

    Parameters
    ----------
    label:
        Estimator name, used in the table.
    errors:
        Shape ``(m, 3)`` in radians.
    sensors:
        One :class:`SensorObservability` per row, giving that record's subspace
        split.

    Returns
    -------
    ScoreTable
        With every figure in milliradians. RMSE is the per-component root mean
        square, so it is directly comparable to a per-axis one-sigma; the
        median is of the vector magnitude, which is the number an engineer
        reads.
    """
    block = np.atleast_2d(np.asarray(errors, dtype=float))
    if block.shape[1] != 3:
        raise ValueError(f"score() needs errors of shape (m, 3), got {block.shape}")
    if block.shape[0] != len(sensors):
        raise ValueError(
            f"score() got {block.shape[0]} errors and {len(sensors)} observability verdicts"
        )
    if block.shape[0] == 0:
        raise ValueError("score() needs at least one record")
    observable = np.zeros_like(block)
    unobservable = np.zeros_like(block)
    for index, sensor in enumerate(sensors):
        observable[index], unobservable[index] = subspace_split(block[index], sensor)
    to_mrad = 1.0e3
    return ScoreTable(
        label=label,
        records=int(block.shape[0]),
        rmse_total_mrad=float(np.sqrt(np.mean(block**2))) * to_mrad,
        rmse_observable_mrad=float(np.sqrt(np.mean(observable**2))) * to_mrad,
        rmse_unobservable_mrad=float(np.sqrt(np.mean(unobservable**2))) * to_mrad,
        median_total_mrad=float(np.median(np.linalg.norm(block, axis=1))) * to_mrad,
        median_observable_mrad=float(np.median(np.linalg.norm(observable, axis=1))) * to_mrad,
        median_unobservable_mrad=float(np.median(np.linalg.norm(unobservable, axis=1)))
        * to_mrad,
        bias_mrad=block.mean(axis=0) * to_mrad,
        worst_total_mrad=float(np.max(np.linalg.norm(block, axis=1))) * to_mrad,
    )
