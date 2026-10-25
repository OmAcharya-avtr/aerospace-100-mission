"""Per-block failure distributions, expressed as cumulative hazard functions.

Every distribution here exposes :meth:`FailureDistribution.cumulative_hazard`
as the primitive and derives reliability and unreliability from it. That choice
is not cosmetic. The cumulative hazard is additive over mission phases and over
series composition, it is the quantity a duty cycle and a stress factor
multiply, and it is the only one of the three that neither underflows nor
cancels for a block that is very likely to survive.

Equations, sources, units and validity
--------------------------------------

Exponential, constant hazard rate ``lambda`` with units 1/hour:

    ``h(t) = lambda``,  ``H(t) = lambda t``,  ``R(t) = exp(-lambda t)``

The standard constant-failure-rate model. Valid only where the hazard is
genuinely flat -- no infant mortality, no wear-out. This is the model behind a
quoted FIT rate, and the assumption it makes is usually left implicit. See
MIL-HDBK-217F (1991), section 3, for the parts-count form in which such rates
are published; no numerical rate in this repository is taken from it.

Weibull, shape ``beta`` (dimensionless) and scale ``eta`` (hours):

    ``H(t) = (t / eta)**beta``,  ``R(t) = exp(-(t / eta)**beta)``

Weibull, W. (1951), "A Statistical Distribution Function of Wide
Applicability", Journal of Applied Mechanics 18(3), 293-297. ``beta < 1`` is
decreasing hazard (infant mortality), ``beta = 1`` reduces exactly to the
exponential with ``lambda = 1 / eta``, ``beta > 1`` is wear-out. Fitting
``beta`` and ``eta`` from data is explicitly **not** in scope for this package;
use the ``reliability`` package, which does it properly.

Lognormal, log-scale median ``mu`` (log hours) and log standard deviation
``sigma`` (dimensionless):

    ``F(t) = Phi((ln t - mu) / sigma)``,  ``H(t) = -log(1 - F(t))``

The standard lognormal life model, used here for mechanisms and for
repair-free wear processes. Evaluated through
``scipy.stats.norm.logsf`` so that the hazard stays accurate in the upper tail
where ``1 - F`` cancels.

Fixed probability, a declared per-mission failure probability ``q``:

    ``H = -log1p(-q)``, independent of ``t``

For a one-shot item -- a pyrotechnic, a deployment, a switch -- whose figure of
merit is a probability per actuation and not a rate. Time has no meaning for
this distribution and a mission profile cannot age it; that is enforced rather
than silently ignored.

A note on what is not here
--------------------------

No distribution fitting, no censored-data likelihood, no accelerated-life
model, no repairable-system intensity. Those are the reason to use the
``reliability`` package instead, and the README says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from scipy.stats import norm

from .numerics import (
    MAX_HAZARD,
    hazard_from_unreliability,
    reliability_from_hazard,
    unreliability_from_hazard,
)

__all__ = [
    "Exponential",
    "FailureDistribution",
    "FixedProbability",
    "Lognormal",
    "Weibull",
    "distribution_from_mapping",
]


@dataclass(frozen=True)
class FailureDistribution:
    """Base class: a cumulative hazard function of time in hours.

    Subclasses implement :meth:`cumulative_hazard`. Everything else is derived
    from it, in the space that keeps the answer.
    """

    #: Whether a mission-phase duty cycle and stress factor may age this
    #: distribution. False for a one-shot item, whose figure of merit is a
    #: probability per actuation rather than a rate. A ``ClassVar`` and not a
    #: dataclass field, so that it never occupies a positional slot in a
    #: subclass constructor: ``Exponential(1e-5)`` must set the rate.
    ages_with_time: ClassVar[bool] = True

    def cumulative_hazard(self, t_hours: float) -> float:
        """Cumulative hazard ``H(t)``, dimensionless, for ``t`` in hours."""
        raise NotImplementedError

    def reliability(self, t_hours: float) -> float:
        """``R(t) = exp(-H(t))``, dimensionless, in ``[0, 1]``."""
        return reliability_from_hazard(self.cumulative_hazard(t_hours))

    def unreliability(self, t_hours: float) -> float:
        """``Q(t) = -expm1(-H(t))``, dimensionless, in ``[0, 1]``.

        Computed from the hazard, never as ``1 - reliability()``: the latter
        returns exactly zero for every block whose true unreliability is below
        ``eps``.
        """
        return unreliability_from_hazard(self.cumulative_hazard(t_hours))

    def inverse_cumulative_hazard(self, h: float) -> float:
        """Age in hours at which the cumulative hazard reaches ``h``.

        The inverse of :meth:`cumulative_hazard`, used by
        :mod:`reliablock.montecarlo` to sample a lifetime: drawing
        ``E ~ Exponential(1)`` and returning ``H^-1(E)`` gives a draw from this
        distribution exactly, because ``H(L)`` is ``Exponential(1)``-distributed
        for any continuous lifetime ``L``.
        """
        raise NotImplementedError

    def density(self, t_hours: float) -> float:
        """Failure probability density ``f(t)``, units 1/hour.

        Default implementation differentiates the cumulative hazard by a
        central difference and multiplies by ``R(t)``; subclasses with a closed
        form override it. Used by the standby convolution.
        """
        t = float(t_hours)
        if t < 0.0:
            raise ValueError(f"t_hours must be >= 0, got {t}")
        step = max(1e-6, 1e-6 * max(t, 1.0))
        lo = max(0.0, t - step)
        hi = t + step
        slope = (self.cumulative_hazard(hi) - self.cumulative_hazard(lo)) / (hi - lo)
        return slope * self.reliability(t)

    def describe(self) -> str:
        """One-line human-readable summary including parameter units."""
        raise NotImplementedError

    def to_mapping(self) -> dict[str, Any]:
        """Round-trippable declaration of this distribution."""
        raise NotImplementedError


@dataclass(frozen=True)
class Exponential(FailureDistribution):
    """Constant hazard rate ``rate_per_hour``; ``H(t) = lambda t``."""

    ages_with_time: ClassVar[bool] = True
    rate_per_hour: float = 0.0

    def __post_init__(self) -> None:
        rate = float(self.rate_per_hour)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError(f"rate_per_hour must be finite and >= 0, got {self.rate_per_hour!r}")
        object.__setattr__(self, "rate_per_hour", rate)

    def cumulative_hazard(self, t_hours: float) -> float:
        """``H(t) = lambda t``, dimensionless."""
        t = float(t_hours)
        if t < 0.0:
            raise ValueError(f"t_hours must be >= 0, got {t}")
        return self.rate_per_hour * t

    def inverse_cumulative_hazard(self, h: float) -> float:
        """``t = H / lambda``, hours; ``inf`` at zero rate."""
        hazard = float(h)
        if hazard < 0.0:
            raise ValueError(f"h must be >= 0, got {hazard}")
        if self.rate_per_hour == 0.0:
            return math.inf
        return hazard / self.rate_per_hour

    def density(self, t_hours: float) -> float:
        """``f(t) = lambda exp(-lambda t)``, units 1/hour."""
        return self.rate_per_hour * self.reliability(t_hours)

    def mttf_hours(self) -> float:
        """Mean time to failure ``1 / lambda``, hours; ``inf`` at zero rate."""
        return math.inf if self.rate_per_hour == 0.0 else 1.0 / self.rate_per_hour

    def describe(self) -> str:
        return f"exponential(rate={self.rate_per_hour:.6e} /h)"

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": "exponential", "rate_per_hour": self.rate_per_hour}


@dataclass(frozen=True)
class Weibull(FailureDistribution):
    """Weibull life model; ``H(t) = (t / eta)**beta``.

    ``shape`` is ``beta`` (dimensionless), ``scale_hours`` is ``eta`` (hours).
    ``beta = 1`` reduces exactly to :class:`Exponential` with
    ``lambda = 1 / eta`` -- asserted as a property test, to within rounding and
    not bit-exactly.
    """

    ages_with_time: ClassVar[bool] = True
    shape: float = 1.0
    scale_hours: float = 1.0

    def __post_init__(self) -> None:
        shape = float(self.shape)
        scale = float(self.scale_hours)
        if not math.isfinite(shape) or shape <= 0.0:
            raise ValueError(f"shape (Weibull beta) must be finite and > 0, got {self.shape!r}")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scale_hours (Weibull eta) must be finite and > 0, got {scale!r}")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "scale_hours", scale)

    def cumulative_hazard(self, t_hours: float) -> float:
        """``H(t) = (t / eta)**beta``, dimensionless."""
        t = float(t_hours)
        if t < 0.0:
            raise ValueError(f"t_hours must be >= 0, got {t}")
        if t == 0.0:
            return 0.0
        # Computed through logs so that a large beta with t/eta near 1 does not
        # overflow the intermediate power.
        exponent = self.shape * math.log(t / self.scale_hours)
        if exponent > math.log(MAX_HAZARD):
            return MAX_HAZARD
        return math.exp(exponent)

    def inverse_cumulative_hazard(self, h: float) -> float:
        """``t = eta H**(1/beta)``, hours."""
        hazard = float(h)
        if hazard < 0.0:
            raise ValueError(f"h must be >= 0, got {hazard}")
        return self.scale_hours * hazard ** (1.0 / self.shape)

    def density(self, t_hours: float) -> float:
        """``f(t) = (beta / eta) (t / eta)**(beta - 1) exp(-(t / eta)**beta)``, 1/hour."""
        t = float(t_hours)
        if t < 0.0:
            raise ValueError(f"t_hours must be >= 0, got {t}")
        if t == 0.0:
            return 0.0 if self.shape > 1.0 else (self.shape / self.scale_hours)
        rate = (self.shape / self.scale_hours) * (t / self.scale_hours) ** (self.shape - 1.0)
        return rate * self.reliability(t)

    def describe(self) -> str:
        return f"weibull(beta={self.shape:.6g}, eta={self.scale_hours:.6e} h)"

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": "weibull", "shape": self.shape, "scale_hours": self.scale_hours}


@dataclass(frozen=True)
class Lognormal(FailureDistribution):
    """Lognormal life model; ``F(t) = Phi((ln t - mu) / sigma)``.

    ``log_median`` is ``mu`` in log-hours, so the median life is ``exp(mu)``
    hours. ``log_sigma`` is dimensionless. The hazard is evaluated through
    ``scipy.stats.norm.logsf`` so that the upper tail stays accurate.
    """

    ages_with_time: ClassVar[bool] = True
    log_median: float = 0.0
    log_sigma: float = 1.0

    def __post_init__(self) -> None:
        mu = float(self.log_median)
        sigma = float(self.log_sigma)
        if not math.isfinite(mu):
            raise ValueError(f"log_median must be finite, got {self.log_median!r}")
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"log_sigma must be finite and > 0, got {self.log_sigma!r}")
        object.__setattr__(self, "log_median", mu)
        object.__setattr__(self, "log_sigma", sigma)

    def cumulative_hazard(self, t_hours: float) -> float:
        """``H(t) = -log(1 - F(t))``, dimensionless."""
        t = float(t_hours)
        if t < 0.0:
            raise ValueError(f"t_hours must be >= 0, got {t}")
        if t == 0.0:
            return 0.0
        z = (math.log(t) - self.log_median) / self.log_sigma
        log_survival = float(norm.logsf(z))
        return min(-log_survival, MAX_HAZARD)

    def inverse_cumulative_hazard(self, h: float) -> float:
        """``t = exp(mu + sigma * Phi^-1(1 - exp(-H)))``, hours.

        ``exp(-H)`` underflows to zero above ``H = 745``, which maps to an
        infinite age. The probability of drawing such an ``H`` from
        ``Exponential(1)`` is ``exp(-745)``, so the branch is unreachable in
        practice; it is handled rather than left to produce a nan.
        """
        hazard = float(h)
        if hazard < 0.0:
            raise ValueError(f"h must be >= 0, got {hazard}")
        survival = math.exp(-hazard) if hazard < MAX_HAZARD else 0.0
        if survival <= 0.0:
            return math.inf
        return math.exp(self.log_median + self.log_sigma * float(norm.isf(survival)))

    def describe(self) -> str:
        return f"lognormal(mu={self.log_median:.6g} log-h, sigma={self.log_sigma:.6g})"

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": "lognormal", "log_median": self.log_median, "log_sigma": self.log_sigma}


@dataclass(frozen=True)
class FixedProbability(FailureDistribution):
    """A declared per-mission failure probability, independent of time.

    For a one-shot item. ``ages_with_time`` is False, and
    :mod:`reliablock.mission` refuses to apply a duty cycle or a stress factor
    to it rather than quietly treating a probability as a rate.
    """

    ages_with_time: ClassVar[bool] = False
    probability: float = 0.0

    def __post_init__(self) -> None:
        q = float(self.probability)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"probability must lie in [0, 1], got {self.probability!r}")
        object.__setattr__(self, "probability", q)

    def cumulative_hazard(self, t_hours: float) -> float:
        """``H = -log1p(-q)``, dimensionless, independent of ``t``."""
        t = float(t_hours)
        if t < 0.0:
            raise ValueError(f"t_hours must be >= 0, got {t}")
        return hazard_from_unreliability(self.probability)

    def inverse_cumulative_hazard(self, h: float) -> float:
        """``0.0`` once ``h`` reaches this item's total hazard, else ``inf``.

        A one-shot item has no lifetime: it either fails when actuated, at
        age zero, or it never fails. The step is at ``H = -log1p(-q)``.
        """
        hazard = float(h)
        if hazard < 0.0:
            raise ValueError(f"h must be >= 0, got {hazard}")
        return 0.0 if hazard <= hazard_from_unreliability(self.probability) else math.inf

    def density(self, t_hours: float) -> float:
        """Zero: this distribution has no density in time.

        A one-shot item either works when actuated or does not. The standby
        convolution rejects a primary with no density rather than integrating
        zero and reporting a reliability that ignores the standby branch.
        """
        float(t_hours)
        return 0.0

    def describe(self) -> str:
        return f"fixed_probability(q={self.probability:.6e})"

    def to_mapping(self) -> dict[str, Any]:
        return {"kind": "fixed_probability", "probability": self.probability}


_KINDS: dict[str, type[FailureDistribution]] = {
    "exponential": Exponential,
    "weibull": Weibull,
    "lognormal": Lognormal,
    "fixed_probability": FixedProbability,
}


def distribution_from_mapping(mapping: dict[str, Any]) -> FailureDistribution:
    """Build a distribution from a declared mapping, rejecting unknown keys.

    The declaration is an artifact a reader has to be able to check, so an
    unknown key is an error and not a silently ignored field. A typo in
    ``rate_per_hour`` that fell through to a default would produce a
    reliability of 1.0 and no complaint.
    """
    if not isinstance(mapping, dict):
        raise TypeError(f"distribution declaration must be a mapping, got {type(mapping).__name__}")
    body = dict(mapping)
    kind = body.pop("kind", None)
    if kind is None:
        raise ValueError(
            "distribution declaration needs a 'kind' key; "
            f"one of {sorted(_KINDS)}, got keys {sorted(mapping)}"
        )
    if kind not in _KINDS:
        raise ValueError(f"unknown distribution kind {kind!r}; expected one of {sorted(_KINDS)}")
    cls = _KINDS[kind]
    # dataclasses.fields() excludes the ClassVar pseudo-field that
    # __dataclass_fields__ still carries.
    allowed = {f.name for f in fields(cls)}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(
            f"distribution kind {kind!r} does not accept {sorted(unknown)}; "
            f"accepted parameters are {sorted(allowed)}"
        )
    missing = allowed - set(body)
    if missing:
        raise ValueError(
            f"distribution kind {kind!r} requires {sorted(missing)}; "
            "no parameter of a failure distribution is defaulted, because a "
            "defaulted rate reports a reliability of 1.0 and no complaint"
        )
    return cls(**body)
