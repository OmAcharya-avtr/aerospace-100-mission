"""The staleness decision: how old can this element set be and still be usable.

This is what the product is for. Everything else -- the propagators, the
growth fits, the coverage measurement -- exists so that this function can
return a number with a measured interval attached instead of a rule of thumb.

The decision is posed as the operator poses it. "I need the position of this
object to better than ``tolerance_km`` for what I am about to do. The element
set I have is ``age_days`` old. Is it still good enough, and if not, when did
it stop being good enough?"

Two horizons are returned, and the difference between them is the whole
reason for measuring coverage:

``median_horizon_days``
    The age at which the *median* predicted drift reaches the tolerance. This
    is the optimistic answer and the one a point prediction gives. Half the
    time the drift is already worse than this says.

``conservative_horizon_days``
    The age at which the *upper end of the prediction interval* reaches the
    tolerance. This is the answer to use, and it is only as good as the
    measured coverage of that interval -- which is why
    :class:`StalenessVerdict` carries the coverage that was measured for the
    predictor that produced it, on held-out objects, rather than the nominal
    level. A conservative horizon from an interval measured to cover 72 per
    cent of the time is not a conservative horizon, and the verdict says so in
    words.

What this is not
----------------
It is not a covariance and it is not an input to a conjunction assessment.
:mod:`orbitdrift.guard` enforces that, and :meth:`StalenessVerdict.as_covariance`
raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import CovarianceRefused, InsufficientDataError
from .features import ObjectFeatures, design_matrix
from .guard import check_purpose, refuse_covariance


@dataclass(frozen=True)
class StalenessVerdict:
    """The answer to "is this element set still good enough".

    Attributes
    ----------
    tolerance_km
        The position tolerance the caller declared.
    age_days
        The age of the element set the caller holds, or ``nan`` if not given.
    median_horizon_days
        Age at which the median predicted drift reaches ``tolerance_km``;
        ``inf`` if it does not within the evaluated range.
    conservative_horizon_days
        Age at which the upper interval bound reaches ``tolerance_km``;
        ``inf`` if it does not within the evaluated range.
    predicted_drift_km
        Median predicted drift at ``age_days``, or ``nan``.
    predicted_interval_km
        Interval at ``age_days``, or ``(nan, nan)``.
    nominal_coverage
        The level the interval claims.
    measured_coverage
        The coverage measured for this predictor on held-out objects, or
        ``nan`` when the caller did not supply one. A verdict with ``nan``
        here is explicitly labelled as having no measured backing by
        :meth:`describe`.
    predictor_name
        Which predictor produced it.
    youngest_age_evaluated_days
        The first point of the age grid. A horizon equal to it does not mean
        "stale after that long": it means the predicted drift already exceeds
        the tolerance at the youngest age the grid evaluated, so the horizon is
        below the grid's resolution. :meth:`describe` says so rather than
        printing a number that looks like a measurement.
    """

    tolerance_km: float
    age_days: float
    median_horizon_days: float
    conservative_horizon_days: float
    predicted_drift_km: float
    predicted_interval_km: tuple[float, float]
    nominal_coverage: float
    measured_coverage: float
    predictor_name: str
    youngest_age_evaluated_days: float = float("nan")

    @property
    def still_good(self) -> bool | None:
        """Whether the held element set is inside tolerance on the conservative horizon.

        ``None`` when no age was supplied, rather than ``False``: a missing
        input is not a negative answer.
        """
        if not np.isfinite(self.age_days):
            return None
        return bool(self.age_days <= self.conservative_horizon_days)

    def as_covariance(self) -> None:
        """Always raises :class:`orbitdrift.errors.CovarianceRefused`."""
        refuse_covariance(
            "StalenessVerdict.as_covariance was called; a staleness horizon is "
            "not a state covariance"
        )

    def describe(self) -> str:
        """Plain-text verdict, including what the interval's coverage was measured to be."""
        lines = [
            f"predictor                : {self.predictor_name}",
            f"tolerance                : {self.tolerance_km:.3f} km",
            f"median horizon           : {self.median_horizon_days:.3f} days",
            f"conservative horizon     : {self.conservative_horizon_days:.3f} days "
            f"(upper end of a {self.nominal_coverage:.0%} interval)",
        ]
        floor = self.youngest_age_evaluated_days
        if np.isfinite(floor) and self.conservative_horizon_days <= floor:
            lines.append(
                f"                           the predicted drift already exceeds the "
                f"tolerance at {floor:.3f} days, the youngest age evaluated, so this "
                "horizon is below the grid resolution and is not a measurement"
            )
        if np.isfinite(self.measured_coverage):
            lines.append(
                f"interval coverage        : {self.measured_coverage:.4f} measured on "
                f"held-out objects against {self.nominal_coverage:.4f} nominal"
            )
            if self.measured_coverage < self.nominal_coverage - 0.02:
                lines.append(
                    "                           the interval UNDER-COVERS; the "
                    "conservative horizon above is optimistic by that much"
                )
        else:
            lines.append(
                "interval coverage        : not supplied. The conservative horizon "
                "above has no measured backing and must not be relied on."
            )
        if np.isfinite(self.age_days):
            lines.append(f"element set age          : {self.age_days:.3f} days")
            lines.append(
                f"predicted drift at age   : {self.predicted_drift_km:.3f} km "
                f"[{self.predicted_interval_km[0]:.3f}, "
                f"{self.predicted_interval_km[1]:.3f}]"
            )
            lines.append(f"verdict                  : {'WITHIN' if self.still_good else 'OUTSIDE'}")
        return "\n".join(lines)


def _first_crossing(ages: np.ndarray, values: np.ndarray, threshold: float) -> float:
    """Age at which a monotone-ish series first reaches ``threshold``, by interpolation."""
    above = np.flatnonzero(values >= threshold)
    if above.size == 0:
        return float("inf")
    index = int(above[0])
    if index == 0:
        return float(ages[0])
    y0, y1 = float(values[index - 1]), float(values[index])
    if y1 == y0:
        return float(ages[index])
    weight = (threshold - y0) / (y1 - y0)
    return float(ages[index - 1] + weight * (ages[index] - ages[index - 1]))


def staleness_verdict(
    predictor,
    features: ObjectFeatures,
    tolerance_km: float,
    *,
    age_days: float | None = None,
    purpose: str = "ephemeris staleness check",
    coverage: float = 0.90,
    measured_coverage: float = float("nan"),
    max_age_days: float = 14.0,
    grid_points: int = 281,
) -> StalenessVerdict:
    """Evaluate a predictor over a grid of ages and report both horizons.

    Parameters
    ----------
    predictor
        Anything with ``predict_median_km`` and ``predict_interval_km``.
    features
        Static features of the object.
    tolerance_km
        The position tolerance the decision needs.
    age_days
        The age of the element set in hand, if there is one.
    purpose
        Declared purpose, screened by :func:`orbitdrift.guard.check_purpose`.
        A purpose naming a conjunction or a covariance is refused.
    coverage
        Nominal interval level requested from the predictor.
    measured_coverage
        The coverage this predictor was measured to achieve on held-out
        objects. Pass it. A verdict without it is labelled as unbacked.
    max_age_days, grid_points
        The age grid. The default resolves to 1.2 hours over 14 days.

    Raises
    ------
    CovarianceRefused
        If ``purpose`` names a forbidden use.
    InsufficientDataError
        If ``tolerance_km`` is not positive and finite.
    """
    check_purpose(purpose)
    if not np.isfinite(tolerance_km) or tolerance_km <= 0.0:
        raise InsufficientDataError(f"tolerance_km must be positive and finite; got {tolerance_km}")
    ages = np.linspace(max_age_days / (grid_points - 1), max_age_days, grid_points - 1)
    static = np.tile(features.as_array(), (ages.size, 1))
    matrix, _ = design_matrix(static, ages)
    median = np.asarray(predictor.predict_median_km(matrix), dtype=float)
    lower, upper = predictor.predict_interval_km(matrix, coverage)
    upper = np.maximum(np.asarray(upper, dtype=float), np.asarray(lower, dtype=float))

    if age_days is not None and np.isfinite(age_days):
        at_age, _ = design_matrix(features.as_array()[None, :], np.array([float(age_days)]))
        drift = float(np.asarray(predictor.predict_median_km(at_age), dtype=float)[0])
        low_at, high_at = predictor.predict_interval_km(at_age, coverage)
        band = (float(np.asarray(low_at)[0]), float(np.asarray(high_at)[0]))
    else:
        drift = float("nan")
        band = (float("nan"), float("nan"))

    return StalenessVerdict(
        tolerance_km=float(tolerance_km),
        age_days=float(age_days) if age_days is not None else float("nan"),
        median_horizon_days=_first_crossing(ages, median, float(tolerance_km)),
        conservative_horizon_days=_first_crossing(ages, upper, float(tolerance_km)),
        predicted_drift_km=drift,
        predicted_interval_km=band,
        nominal_coverage=float(coverage),
        measured_coverage=float(measured_coverage),
        predictor_name=str(getattr(predictor, "name", type(predictor).__name__)),
        youngest_age_evaluated_days=float(ages[0]),
    )


__all__ = ["CovarianceRefused", "StalenessVerdict", "staleness_verdict"]
