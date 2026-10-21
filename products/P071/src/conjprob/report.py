"""The product's primary output: several Pc values and their disagreement.

:func:`assess_conjunction` is the one function most callers need. It returns a
:class:`ConjunctionReport` in which the collision probability is not a scalar.
It is four analytic values, an optional Monte Carlo reference with a confidence
interval, the spread between them, the dilution diagnosis, and the dimensionless
regime parameters that say which of the values are entitled to be believed.

The report deliberately has no single ``pc`` attribute. Code that wants one must
name the method it wants, because the choice of method is a choice of
assumptions and this package will not make it silently.

Units: metres, metres per second, metres squared in; seconds, metres and
dimensionless probabilities out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .analytic import (
    chan_recommended_order,
    pc_alfano,
    pc_chan,
    pc_foster,
    pc_small_body_approximation,
)
from .dilution import DilutionAssessment, assess_dilution
from .geometry import EncounterGeometry, ObjectState, encounter_geometry
from .montecarlo import MonteCarloResult, pc_monte_carlo
from .quadrature import pc_quadrature

ANALYTIC_METHODS: tuple[str, ...] = ("foster", "chan", "alfano", "small_body")
"""Analytic method keys present in every report, in report order."""

OPERATIONAL_DEFAULT = "foster"
"""The method this package treats as the operational default, named explicitly."""


@dataclass(frozen=True)
class Disagreement:
    """How far apart the methods are, which is the headline of the report.

    Attributes
    ----------
    values
        Method name to Pc, dimensionless, including ``quadrature``.
    reference_method
        Method used as the denominator of the relative spreads. ``quadrature``
        unless it was not computed.
    max_absolute_spread
        ``max(values) - min(values)``, dimensionless.
    max_relative_spread
        Largest ``|pc_method / pc_reference - 1|`` over the methods,
        dimensionless. ``inf`` if the reference is zero and another method is
        not.
    relative_errors
        Signed ``pc_method / pc_reference - 1`` per method, dimensionless.
    worst_method
        Method with the largest absolute relative error.
    order_of_magnitude_spread
        ``log10(max / min)`` over strictly positive values, dimensionless.
    """

    values: dict[str, float]
    reference_method: str
    max_absolute_spread: float
    max_relative_spread: float
    relative_errors: dict[str, float]
    worst_method: str
    order_of_magnitude_spread: float


def _disagreement(values: dict[str, float], reference_method: str) -> Disagreement:
    numbers = np.array(list(values.values()), dtype=float)
    reference = values[reference_method]
    relative: dict[str, float] = {}
    for name, value in values.items():
        if reference > 0.0:
            relative[name] = value / reference - 1.0
        elif value == 0.0:
            relative[name] = 0.0
        else:
            relative[name] = math.inf
    worst = max(relative, key=lambda key: abs(relative[key]))
    positive = numbers[numbers > 0.0]
    decades = (
        float(np.log10(positive.max() / positive.min())) if positive.size >= 2 else 0.0
    )
    return Disagreement(
        values=dict(values),
        reference_method=reference_method,
        max_absolute_spread=float(numbers.max() - numbers.min()),
        max_relative_spread=float(max(abs(v) for v in relative.values())),
        relative_errors=relative,
        worst_method=worst,
        order_of_magnitude_spread=decades,
    )


@dataclass(frozen=True)
class RegimeFlags:
    """Dimensionless parameters and the regime statements they support.

    Attributes
    ----------
    radius_over_sigma
        ``R / sqrt(sigma_x sigma_y)``, dimensionless. The small-hard-body
        approximation and Chan's equivalent-area substitution both degrade as
        this grows.
    aspect_ratio
        ``sigma_x / sigma_y >= 1``, dimensionless. Chan's equivalent-area step
        is exact only at one.
    mahalanobis
        Mahalanobis distance of the miss vector, dimensionless.
    crossing_time_s
        ``sqrt(sigma_x sigma_y) / |v_rel|``, seconds. The duration over which
        the rectilinear assumption has to hold.
    notes
        Human-readable regime statements, each generated from one of the
        measured boundaries in ``validation/validate_regimes_output.txt``.
    """

    radius_over_sigma: float
    aspect_ratio: float
    mahalanobis: float
    crossing_time_s: float
    notes: tuple[str, ...]


@dataclass(frozen=True)
class ConjunctionReport:
    """Full assessment of one close approach.

    There is no single ``pc`` attribute by design; see :mod:`conjprob.report`.
    """

    geometry: EncounterGeometry
    radius_m: float
    analytic: dict[str, float]
    quadrature_pc: float
    quadrature_abserr: float
    disagreement: Disagreement
    dilution: DilutionAssessment
    regime: RegimeFlags
    monte_carlo: MonteCarloResult | None = None
    monte_carlo_consistent: tuple[str, ...] = field(default_factory=tuple)
    monte_carlo_inconsistent: tuple[str, ...] = field(default_factory=tuple)

    def pc(self, method: str) -> float:
        """Pc from the named method, dimensionless.

        ``method`` is one of :data:`ANALYTIC_METHODS` or ``"quadrature"``.
        """
        if method == "quadrature":
            return self.quadrature_pc
        if method not in self.analytic:
            raise KeyError(
                f"unknown method {method!r}; have {sorted(self.analytic)} and 'quadrature'"
            )
        return self.analytic[method]

    def summary(self) -> str:
        """Multi-line plain-text report. Returns the text; never prints it."""
        geometry = self.geometry
        frame = geometry.frame
        lines = [
            "conjunction assessment",
            f"  time of closest approach : {geometry.tca_s:+.6f} s from epoch",
            f"  miss distance            : {geometry.miss_distance_m:.4f} m",
            f"  relative speed           : {geometry.relative_speed_mps:.4f} m/s",
            f"  hard-body radius         : {self.radius_m:.4f} m",
            f"  encounter-plane sigma    : {frame.sigma_x_m:.4f} m x {frame.sigma_y_m:.4f} m"
            f" (aspect {frame.aspect_ratio:.4f})",
            f"  mahalanobis distance     : {geometry.mahalanobis:.6f}",
            f"  R / sigma_eff            : {self.regime.radius_over_sigma:.6e}",
            f"  crossing time            : {self.regime.crossing_time_s:.6e} s",
            "",
            "collision probability by method",
        ]
        for name in ANALYTIC_METHODS:
            lines.append(
                f"  {name:<12s} : {self.analytic[name]:.6e}"
                f"   rel. to quadrature {self.disagreement.relative_errors[name]:+.3e}"
            )
        lines.append(
            f"  {'quadrature':<12s} : {self.quadrature_pc:.6e}"
            f"   abserr {self.quadrature_abserr:.3e}"
        )
        if self.monte_carlo is not None:
            mc = self.monte_carlo
            lines.append(
                f"  {'monte_carlo':<12s} : {mc.pc:.6e}"
                f"   {int(mc.confidence * 100)}% CI [{mc.ci_low:.6e}, {mc.ci_high:.6e}]"
                f"   hits {mc.hits}/{mc.n_samples}"
            )
            lines.append(
                f"    inside the interval  : {', '.join(self.monte_carlo_consistent) or 'none'}"
            )
            lines.append(
                f"    outside the interval : {', '.join(self.monte_carlo_inconsistent) or 'none'}"
            )
        lines += [
            "",
            "disagreement (the primary output)",
            f"  max absolute spread      : {self.disagreement.max_absolute_spread:.6e}",
            f"  max relative spread      : {self.disagreement.max_relative_spread:.6e}",
            f"  worst method             : {self.disagreement.worst_method}",
            f"  spread in decades        : {self.disagreement.order_of_magnitude_spread:.6f}",
            "",
            "dilution",
            f"  on dilution branch       : {self.dilution.on_dilution_branch}",
            f"  dilution ratio           : {self.dilution.dilution_ratio:.6f}",
            f"  dPc / d ln(scale)        : {self.dilution.log_scale_sensitivity:+.6e}",
            f"  attainable maximum Pc    : {self.dilution.pc_max_numerical:.6e}"
            f" at covariance scale {self.dilution.scale_at_max_numerical:.6f}",
            f"  dilution factor          : {self.dilution.dilution_factor:.6f}",
            "",
            "regime",
        ]
        lines += [f"  {note}" for note in self.regime.notes]
        lines += [
            "",
            "This is a method cross-validation report under the covariance you supplied.",
            "It is not an input to a manoeuvre decision and no output of this package is.",
        ]
        return "\n".join(lines)


def _regime_notes(
    radius_over_sigma: float, aspect_ratio: float, mahalanobis: float
) -> tuple[str, ...]:
    """Regime statements, each thresholded on a boundary measured in validation."""
    notes: list[str] = []
    if radius_over_sigma > 0.27:
        notes.append(
            "R / sigma_eff exceeds 0.27: the small-hard-body approximation is outside "
            "the 1 per cent boundary measured at aspect 1, Mahalanobis 1 "
            "(validation/validate_regimes_output.txt)."
        )
    else:
        notes.append(
            "R / sigma_eff is within the measured 1 per cent boundary of the "
            "small-hard-body approximation."
        )
    if aspect_ratio > 2.0:
        notes.append(
            "covariance aspect ratio exceeds 2: Chan's equivalent-area substitution is "
            "exact only for a circular scaled cross-section and its measured error "
            "grows with aspect ratio."
        )
    if mahalanobis < 1.4142135623730951:
        notes.append(
            "Mahalanobis distance is below sqrt(2): this Pc is on the dilution branch "
            "and a larger covariance would make it smaller."
        )
    if mahalanobis > 5.0:
        notes.append(
            "Mahalanobis distance exceeds 5: Pc is in the far tail where the series "
            "truncation of Chan's method dominates its error."
        )
    return tuple(notes)


def assess_conjunction(
    primary: ObjectState,
    secondary: ObjectState,
    radius_m: float,
    *,
    chan_order: int | None = None,
    alfano_nodes: int = 201,
    monte_carlo_samples: int | None = None,
    monte_carlo_seed: int = 0,
    monte_carlo_confidence: float = 0.95,
    warn_on_dilution: bool = True,
) -> ConjunctionReport:
    """Assess one close approach by every method in this package.

    Parameters
    ----------
    primary, secondary
        States with 3x3 position covariance at a common epoch. Positions in
        metres, velocities in metres per second, covariances in metres squared.
    radius_m
        Combined hard-body radius, metres.
    chan_order
        Series truncation order for :func:`conjprob.analytic.pc_chan`. ``None``
        uses :func:`conjprob.analytic.chan_recommended_order` for the measured
        Mahalanobis distance, so that the Chan error this report shows is the
        method's own equivalent-area error and not an artefact of a truncation
        order chosen here.
    alfano_nodes
        Simpson node count for :func:`conjprob.analytic.pc_alfano`.
    monte_carlo_samples
        If given, also run the 3-D Monte Carlo reference with this many
        samples. ``None`` skips it. A Monte Carlo estimate of a probability of
        order ``p`` needs of order ``100 / p`` samples for a ten per cent
        interval, so this is left to the caller rather than defaulted.
    monte_carlo_seed, monte_carlo_confidence
        Passed through to :func:`conjprob.montecarlo.pc_monte_carlo`.
    warn_on_dilution
        Whether the dilution diagnosis may raise :class:`DilutionWarning`.

    Returns
    -------
    ConjunctionReport

    Raises
    ------
    ValueError
        On a non-positive-definite projected covariance, a negative radius, or
        a zero relative velocity.
    """
    if not math.isfinite(radius_m) or radius_m < 0.0:
        raise ValueError(f"radius_m must be a non-negative finite number, got {radius_m!r}")
    geometry = encounter_geometry(primary, secondary)
    frame = geometry.frame
    if chan_order is None:
        chan_order = chan_recommended_order(geometry.mahalanobis)
    analytic = {
        "foster": pc_foster(
            frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, radius_m
        ),
        "chan": pc_chan(
            frame.sigma_x_m,
            frame.sigma_y_m,
            frame.x_bar_m,
            frame.y_bar_m,
            radius_m,
            order=chan_order,
        ),
        "alfano": pc_alfano(
            frame.sigma_x_m,
            frame.sigma_y_m,
            frame.x_bar_m,
            frame.y_bar_m,
            radius_m,
            nodes=alfano_nodes,
        ),
        "small_body": pc_small_body_approximation(
            frame.sigma_x_m, frame.sigma_y_m, frame.x_bar_m, frame.y_bar_m, radius_m
        ),
    }
    quadrature_pc, quadrature_abserr = pc_quadrature(
        geometry.covariance_2d_m2, geometry.miss_2d_m, radius_m
    )
    values = dict(analytic)
    values["quadrature"] = quadrature_pc
    disagreement = _disagreement(values, "quadrature")
    dilution = assess_dilution(
        frame.sigma_x_m,
        frame.sigma_y_m,
        frame.x_bar_m,
        frame.y_bar_m,
        radius_m,
        warn=warn_on_dilution,
    )
    regime = RegimeFlags(
        radius_over_sigma=float(radius_m / geometry.sigma_effective_m),
        aspect_ratio=frame.aspect_ratio,
        mahalanobis=geometry.mahalanobis,
        crossing_time_s=geometry.crossing_time_s,
        notes=_regime_notes(
            radius_m / geometry.sigma_effective_m, frame.aspect_ratio, geometry.mahalanobis
        ),
    )
    monte_carlo = None
    consistent: tuple[str, ...] = ()
    inconsistent: tuple[str, ...] = ()
    if monte_carlo_samples is not None:
        monte_carlo = pc_monte_carlo(
            geometry.combined_covariance_m2,
            geometry.miss_vector_m,
            geometry.relative_velocity_mps,
            radius_m,
            n_samples=monte_carlo_samples,
            seed=monte_carlo_seed,
            confidence=monte_carlo_confidence,
        )
        inside = [
            name
            for name, value in values.items()
            if monte_carlo.ci_low <= value <= monte_carlo.ci_high
        ]
        consistent = tuple(inside)
        inconsistent = tuple(name for name in values if name not in inside)
    return ConjunctionReport(
        geometry=geometry,
        radius_m=float(radius_m),
        analytic=analytic,
        quadrature_pc=quadrature_pc,
        quadrature_abserr=quadrature_abserr,
        disagreement=disagreement,
        dilution=dilution,
        regime=regime,
        monte_carlo=monte_carlo,
        monte_carlo_consistent=consistent,
        monte_carlo_inconsistent=inconsistent,
    )


def _as_array(values: dict[str, float]) -> NDArray[np.float64]:
    """Report values as an array in :data:`ANALYTIC_METHODS` order."""
    return np.array([values[name] for name in ANALYTIC_METHODS], dtype=float)
