r"""An independently derived bracket on total loss, from counting alone.

WHY THIS EXISTS AND WHAT MAKES IT INDEPENDENT
---------------------------------------------

Two implementations agreeing cannot catch an error they share. This mission has
shipped that defect: a product's several hundred tests, property tests included,
all compared one closed form against expressions carrying the same cancellation,
and only a bound derived from different mathematics caught it.

The bracket here is derived from **conservation and capacity counting** over the
accounting window. It uses:

* the total volume offered by the source over the window;
* the buffer's free space at the break;
* the maximum volume the egress could possibly carry over the window, obtained
  by integrating the declared egress schedule in closed form.

It does **not** use: the per-phase decomposition, the clamped occupancy
recursion, the fluid model's phase-by-phase clamping argument, the trace
simulator's slot loop, or any per-slot quantity. The two sides of the bracket
are a pigeonhole count and a trivial upper count. Nothing in
:mod:`handoverloss.baselines`, :mod:`handoverloss.trace` or
:mod:`handoverloss.accounting` is called from this module, which is asserted by
``tests/test_bounds.py::test_bounds_module_imports_no_model``.

THE DERIVATION
--------------

Over the window ``[0, H]``, let ``A`` be the offered volume, ``D`` the delivered
volume, ``q_0`` and ``q_H`` the occupancy at the ends, and ``L`` the dropped
volume. Every offered bit is delivered, still buffered, or dropped:

.. math::

    A = D + (q_H - q_0) + L .

**Lower bound.** ``D`` cannot exceed the integral of the egress schedule,
``S := \int_0^H r(t)\,dt``, and ``q_H`` cannot exceed ``B``. Substituting both
upper bounds into the identity and solving for ``L``:

.. math::

    L \;\ge\; A - S - (B - q_0) .

Combined with ``L \ge 0`` this is the lower bound. It is a pigeonhole statement:
if more arrives than can possibly leave or be stored, the remainder is lost.

**Upper bound.** ``D \ge 0`` and ``q_H \ge 0``, so

.. math::

    L \;\le\; A + q_0 .

The ``q_0`` term is there because data present at the break can itself be
dropped in principle; this package's tail-drop rule never drops already-buffered
data, so the tighter ``L \le A`` also holds and is what :func:`loss_bracket`
returns as ``upper_tight``. Both are reported; the loose one is the one that
holds for any drop policy, including head-drop.

**The egress integral in closed form.** With the declared schedule of
:mod:`handoverloss.sequence` — zero for ``t < T_o``, linear to ``R_d`` over
``T_ramp``, then ``R_d``:

.. math::

    S = \tfrac12 R_d T_{ramp} + R_d \,\max(0,\; H - T_o - T_{ramp}) .

That is the only piece of arithmetic this module shares with the models, and it
shares it with the DECLARATION rather than with any solver: it is the area of a
trapezium under the declared rate curve.

CONDITIONING, AND HOW THE SLACK IS SET
--------------------------------------

The lower bound is a difference of three quantities of comparable size, so it is
subject to cancellation exactly where it is tightest. The slack allowed when
checking a model against the bracket must therefore follow the CONDITIONING of
the bound expression, not an absolute constant and not the operation count. For
``L_low = A - S - (B - q_0)`` evaluated in double precision, the forward error
is bounded by the unit roundoff times the sum of the magnitudes of the terms,
times a small factor for the number of operations:

.. math::

    \mathrm{slack} = 8\,\varepsilon \,\bigl(|A| + |S| + B + |q_0|\bigr)

with ``\varepsilon = 2^{-53}``. :func:`bracket_slack_bits` returns exactly that.
An absolute constant here would be wrong in both directions: vacuous for a
petabit-scale window and falsely tight for a megabit-scale one. The factor 8 is
the operation count of the expression with margin, and it multiplies the
conditioning term rather than standing alone.

UNITS
-----

Bits, bit/s, seconds.

SOURCES
-------

The arrival-curve and service-curve style of bounding a buffer's backlog and
loss from cumulative counts rather than from a queue recursion: Cruz, R. L., "A
calculus for network delay, Part I: Network elements in isolation", *IEEE
Transactions on Information Theory*, volume 37, number 1, pages 114-131, 1991;
and Le Boudec, J.-Y. and Thiran, P., *Network Calculus: A Theory of
Deterministic Queuing Systems for the Internet*, Springer Lecture Notes in
Computer Science 2050, 2001, chapter 1. The conservation identity itself is
elementary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Double-precision unit roundoff, 2**-53.
UNIT_ROUNDOFF = 2.0**-53


@dataclass(frozen=True)
class LossBracket:
    """A bracket on total loss over one accounting window. Bits.

    Attributes
    ----------
    lower_bits:
        Pigeonhole lower bound, ``max(0, A - S - (B - q0))``.
    upper_bits:
        Policy-independent upper bound, ``A + q0``.
    upper_tight_bits:
        Upper bound for a tail-drop policy, ``A``.
    offered_bits:
        ``A``, the offered volume over the window.
    max_service_bits:
        ``S``, the egress integral over the window.
    slack_bits:
        Conditioning-scaled slack for checking a model against this bracket; see
        :func:`bracket_slack_bits`.
    """

    lower_bits: float
    upper_bits: float
    upper_tight_bits: float
    offered_bits: float
    max_service_bits: float
    slack_bits: float

    def brackets(self, value: float, *, tight: bool = True) -> bool:
        """Whether ``value`` lies inside the bracket, allowing :attr:`slack_bits`.

        A floating-point membership test, so the boolean is not asserted as an
        invariant anywhere in this package; the tests assert the two numeric
        gaps against the slack instead.
        """
        ceiling = self.upper_tight_bits if tight else self.upper_bits
        return (
            value >= self.lower_bits - self.slack_bits
            and value <= ceiling + self.slack_bits
        )

    def gap_below_bits(self, value: float) -> float:
        """How far ``value`` falls below :attr:`lower_bits`. Negative means inside."""
        return self.lower_bits - value

    def gap_above_bits(self, value: float, *, tight: bool = True) -> float:
        """How far ``value`` rises above the upper bound. Negative means inside."""
        ceiling = self.upper_tight_bits if tight else self.upper_bits
        return value - ceiling


def bracket_slack_bits(
    offered_bits: float | np.ndarray,
    max_service_bits: float | np.ndarray,
    buffer_capacity_bits: float | np.ndarray,
    initial_occupancy_bits: float | np.ndarray,
) -> float | np.ndarray:
    """Conditioning-scaled slack for the lower bound, bits.

    ``8 * eps * (|A| + |S| + B + |q0|)``. See the module docstring for why the
    slack is proportional to the sum of the term magnitudes and not an absolute
    constant.
    """
    return (
        8.0
        * UNIT_ROUNDOFF
        * (
            np.abs(offered_bits)
            + np.abs(max_service_bits)
            + np.abs(buffer_capacity_bits)
            + np.abs(initial_occupancy_bits)
        )
    )


def max_service_bits(
    downlink_rate_bps: float | np.ndarray,
    outage_s: float | np.ndarray,
    ramp_s: float | np.ndarray,
    horizon_s: float | np.ndarray,
) -> float | np.ndarray:
    """Egress integral ``S`` over ``[0, H]`` from the declared schedule, bits.

    ``0.5 * R_d * T_ramp + R_d * max(0, H - T_o - T_ramp)``: the area of the
    trapezium under the declared rate curve. Raises on a window shorter than the
    outage, which would make the integral meaningless.
    """
    rate = np.asarray(downlink_rate_bps, dtype=float)
    outage = np.asarray(outage_s, dtype=float)
    ramp = np.asarray(ramp_s, dtype=float)
    horizon = np.asarray(horizon_s, dtype=float)
    if np.any(rate < 0.0) or np.any(outage < 0.0) or np.any(ramp < 0.0):
        raise ValueError(
            "downlink_rate_bps, outage_s and ramp_s must all be non-negative"
        )
    if np.any(horizon < outage):
        raise ValueError(
            "horizon_s must be at least outage_s; a window shorter than the outage has "
            "no egress integral to take"
        )
    within_ramp = np.minimum(np.maximum(horizon - outage, 0.0), ramp)
    # Area of the ramp actually inside the window, then the flat part.
    ramp_area = np.where(
        ramp > 0.0, 0.5 * rate * within_ramp * within_ramp / np.where(ramp > 0.0, ramp, 1.0), 0.0
    )
    flat_area = rate * np.maximum(horizon - outage - ramp, 0.0)
    result = ramp_area + flat_area
    return float(result) if result.ndim == 0 else result


def loss_bracket(
    offered_bits: float,
    downlink_rate_bps: float,
    outage_s: float,
    ramp_s: float,
    horizon_s: float,
    buffer_capacity_bits: float,
    initial_occupancy_bits: float,
) -> LossBracket:
    """Build the bracket for one accounting window.

    Parameters
    ----------
    offered_bits:
        ``A``, the volume the source offered over the window. Measured, not
        modelled: the caller supplies the trace's own arrival total, or
        ``R_s * H`` for the fluid case.
    downlink_rate_bps, outage_s, ramp_s, horizon_s:
        The declared egress schedule.
    buffer_capacity_bits, initial_occupancy_bits:
        The buffer at the break.

    Returns
    -------
    LossBracket

    Raises
    ------
    ValueError
        On a negative offered volume, a non-positive capacity, an occupancy
        outside ``[0, B]``, or a window shorter than the outage.
    """
    if offered_bits < 0.0:
        raise ValueError(f"offered_bits must be non-negative, got {offered_bits!r}")
    if buffer_capacity_bits <= 0.0:
        raise ValueError(
            f"buffer_capacity_bits must be strictly positive, got "
            f"{buffer_capacity_bits!r}"
        )
    if not 0.0 <= initial_occupancy_bits <= buffer_capacity_bits:
        raise ValueError(
            f"initial_occupancy_bits ({initial_occupancy_bits!r}) must lie in "
            f"[0, buffer_capacity_bits] ({buffer_capacity_bits!r})"
        )
    service = float(
        max_service_bits(downlink_rate_bps, outage_s, ramp_s, horizon_s)
    )
    headroom = buffer_capacity_bits - initial_occupancy_bits
    lower = max(0.0, offered_bits - service - headroom)
    return LossBracket(
        lower_bits=lower,
        upper_bits=offered_bits + initial_occupancy_bits,
        upper_tight_bits=offered_bits,
        offered_bits=float(offered_bits),
        max_service_bits=service,
        slack_bits=float(
            bracket_slack_bits(
                offered_bits, service, buffer_capacity_bits, initial_occupancy_bits
            )
        ),
    )
