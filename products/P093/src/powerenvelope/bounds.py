"""Brackets derived from mathematics this product's own integrator does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot. A property test that shares an expression with the implementation
passes on the cases the expression gets wrong. This module therefore contains
no trapezoid.

Four brackets, each from a different argument:

1. :func:`riemann_bracket` -- the **left and right Riemann sums**. For any
   sampled series the trapezoid rule is their arithmetic mean, so the trapezoid
   lies between them. This is an algebraic identity about quadrature, not a
   restatement of the trapezoid formula.
2. :func:`extreme_value_bracket` -- the **mean value theorem for integrals**:
   an integral over a span lies between the span times the minimum and the span
   times the maximum of the integrand. It knows nothing about any quadrature
   rule at all.
3. :func:`duty_cycle_lower_bracket` -- a lower bound on the duty-cycle factor
   from the **positivity of the sleep-window power**: the whole-run energy is at
   least the active energy plus the measured minimum floor power times the sleep
   duration.
4. :func:`headline_lower_bracket` -- a lower bound on the headline factor built
   only from extreme values of the measured power and the measured latencies.
   It does not use the decomposition, the acceptance rate as a factor, or any
   integral.

Every slack here **follows the conditioning of its own expression** and has no
absolute floor. A slack floored at a constant makes a narrow bracket vacuous and
admits the very thing the bracket exists to catch; an earlier product of this
mission had both wrong forms falsified on the same structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Machine epsilon of the arithmetic every bracket here is evaluated in.
EPS = float(np.finfo(np.float64).eps)


def slack_from_conditioning(terms: np.ndarray) -> float:
    """Floating-point slack for a sum of ``terms``, following its conditioning.

    The error of a sequentially accumulated sum of ``n`` terms is bounded by
    about ``n * EPS * sum(|term|)``. That is the quantity returned: it scales
    with the magnitudes actually summed, so a bracket whose width is 1e-23 gets
    a slack of order 1e-23 and not of order 1e-15.

    Returns
    -------
    float
        Non-negative slack in the same units as the sum. Zero for an empty sum,
        which is exact.
    """
    values = np.asarray(terms, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(values.size) * EPS * float(np.sum(np.abs(values)))


@dataclass(frozen=True)
class Bracket:
    """A closed bracket with the slack its own conditioning implies.

    ``contains`` is the only way a caller should test membership: it widens the
    bracket by :attr:`slack` on both sides, so a value that fails has failed by
    more than floating-point arithmetic can explain.
    """

    low: float
    high: float
    slack: float
    source: str

    @property
    def width(self) -> float:
        """Bracket width before slack."""
        return self.high - self.low

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies in ``[low - slack, high + slack]``."""
        return self.low - self.slack <= value <= self.high + self.slack

    def excess(self, value: float) -> float:
        """How far ``value`` lies outside the slack-widened bracket, or 0.0."""
        if value < self.low - self.slack:
            return self.low - self.slack - value
        if value > self.high + self.slack:
            return value - self.high - self.slack
        return 0.0


def riemann_bracket(t_s: np.ndarray, power_mw: np.ndarray) -> Bracket:
    """Left and right Riemann sums of ``power_mw`` over ``t_s``, as a bracket.

    With ``dt_i = t_{i+1} - t_i``, the left sum is ``sum(p_i * dt_i)`` and the
    right sum is ``sum(p_{i+1} * dt_i)``. The trapezoid rule equals their mean
    exactly in exact arithmetic, so it lies between them; in floating-point it
    agrees to within :func:`slack_from_conditioning` of the summed terms.

    Raises
    ------
    ValueError
        On mismatched lengths, fewer than two samples, or decreasing timestamps.
    """
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(power_mw, dtype=np.float64)
    if t.size != p.size:
        raise ValueError(f"t_s has {t.size} samples and power_mw has {p.size}")
    if t.size < 2:
        raise ValueError("a Riemann bracket needs at least two samples")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("t_s must be non-decreasing")
    dt = np.diff(t)
    left = p[:-1] * dt
    right = p[1:] * dt
    low = min(float(left.sum()), float(right.sum()))
    high = max(float(left.sum()), float(right.sum()))
    return Bracket(
        low=low,
        high=high,
        slack=slack_from_conditioning(np.concatenate((left, right))),
        source="left and right Riemann sums; the trapezoid rule is their mean",
    )


def extreme_value_bracket(t_s: np.ndarray, power_mw: np.ndarray) -> Bracket:
    """Mean-value bracket ``[span*min(P), span*max(P)]`` for the integral.

    No quadrature rule is involved. The bracket is wide, and that is the point:
    a value outside it is not a quadrature disagreement but a mistake about
    which samples or which span were integrated.
    """
    t = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(power_mw, dtype=np.float64)
    if t.size != p.size:
        raise ValueError(f"t_s has {t.size} samples and power_mw has {p.size}")
    if t.size < 2:
        raise ValueError("a mean-value bracket needs at least two samples")
    span = float(t[-1] - t[0])
    low = span * float(p.min())
    high = span * float(p.max())
    return Bracket(
        low=low,
        high=high,
        slack=slack_from_conditioning(np.asarray([low, high])),
        source="mean value theorem for integrals over the sampled span",
    )


def duty_cycle_lower_bracket(
    active_energy_mj: float, floor_min_mw: float, sleep_duration_s: float
) -> Bracket:
    """Lower bound on ``E_total / E_active`` from the sleep window's own power.

    The premise is that during a sleep window the rail still draws at least
    ``floor_min_mw``. So

    ``E_total >= E_active + floor_min_mw * sleep_duration_s``

    and the duty-cycle factor is at least ``1 + that / E_active``. The upper end
    is left at infinity, expressed as :data:`numpy.inf`, because nothing here
    bounds it from above: a run could in principle spend any amount of energy
    while asleep.

    **The premise is an assumption about the board's state, and this product's
    own measured campaign falsifies it when ``floor_min_mw`` comes from the
    campaign's floor run.** The arithmetic is sound; the premise is not. The
    campaign's run order was not randomised and the board's foreign load fell
    throughout, so the floor cell was measured at the highest foreign load of
    the whole campaign and later cells' sleep windows draw less than the floor
    run's minimum ever did. Nine of nineteen pipeline cells violate the bound
    for that reason, and
    ``validation/validate_measured_headline.py`` publishes the violation with
    the per-cell table rather than weakening the bound. A caller who wants a
    bound that holds should pass each cell's OWN measured sleep-window minimum,
    which that script reports beside the campaign-floor form and labels as the
    weaker argument it is -- it reads its minimum from the same samples the
    integral it checks is taken over.

    Raises
    ------
    ValueError
        On a non-positive active energy, a negative floor power or a negative
        sleep duration.
    """
    if active_energy_mj <= 0.0:
        raise ValueError(f"active_energy_mj must be positive, got {active_energy_mj}")
    if floor_min_mw < 0.0:
        raise ValueError(f"floor_min_mw must be non-negative, got {floor_min_mw}")
    if sleep_duration_s < 0.0:
        raise ValueError(f"sleep_duration_s must be non-negative, got {sleep_duration_s}")
    low = 1.0 + floor_min_mw * sleep_duration_s / active_energy_mj
    return Bracket(
        low=low,
        high=float(np.inf),
        slack=slack_from_conditioning(np.asarray([1.0, low - 1.0])),
        source="positivity of the measured minimum floor power over the sleep duration",
    )


def headline_lower_bracket(
    run_duration_s: float,
    min_power_mw: float,
    max_active_power_mw: float,
    max_latency_s: float,
    n_accepted: int,
) -> Bracket:
    """Bracket on the headline factor from extreme values alone.

    The numerator ``E_total / N_accepted`` is at least
    ``min_power_mw * run_duration_s / N_accepted``, because the rail never drew
    less than the minimum sample. The denominator
    ``mean_active_power * mean_latency`` is at most
    ``max_active_power_mw * max_latency_s``, because a mean never exceeds a
    maximum. Dividing the two gives a lower bound on the factor that uses no
    integral, no acceptance rate as a multiplier and none of the decomposition.

    The upper end is ``numpy.inf``: the same argument run the other way needs a
    positive lower bound on the mean active power and on the mean latency, and
    the minimum sample of a duty-cycled run is a sleep-window sample, so that
    direction gives nothing useful and is not claimed.

    Raises
    ------
    ValueError
        On a non-positive duration, power, latency or accepted count.
    """
    for name, value in (
        ("run_duration_s", run_duration_s),
        ("min_power_mw", min_power_mw),
        ("max_active_power_mw", max_active_power_mw),
        ("max_latency_s", max_latency_s),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    if n_accepted <= 0:
        raise ValueError(f"n_accepted must be positive, got {n_accepted}")
    numerator_low = min_power_mw * run_duration_s / n_accepted
    denominator_high = max_active_power_mw * max_latency_s
    low = numerator_low / denominator_high
    return Bracket(
        low=low,
        high=float(np.inf),
        slack=slack_from_conditioning(np.asarray([numerator_low, denominator_high, low])),
        source="extreme values of the measured power and latency; no quadrature",
    )
