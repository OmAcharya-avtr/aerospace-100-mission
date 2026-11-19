r"""Baseline 1 (acquisition-only) and baseline 2 (analytic fluid), both closed form.

The two differ in exactly one thing: how much of the declared handover sequence
they look at. Baseline 1 looks at the acquisition phase. Baseline 2 looks at all
three. Everything else — the buffer, the tail-drop rule, the arithmetic — is
identical, which is what makes the comparison between them a measurement of the
omitted mechanisms and not of two unrelated models.

BASELINE 1, THE THING BEING REFUTED, AT FULL STRENGTH
-----------------------------------------------------

The assumption under test is that a handover costs the data the buffer cannot
absorb during acquisition, because acquisition time is what a link budget and a
pointing analysis hand you. Implemented as

.. math::

    L_{\mathrm{acq\text{-}only}} = \max(0,\; R_s T_{acq} - (B - q_0))

This is the strongest honest form of that assumption and is deliberately not a
strawman:

* it credits the buffer with its full free space ``B - q_0``, rather than
  assuming the whole acquisition interval is lost;
* it is evaluated at whatever acquisition time the caller supplies, so a
  conservative engineer's upper quantile of the acquisition distribution is
  available and is reported beside the mean by
  ``validation/validate_learned_vs_baselines.py``;
* it is exactly the restriction of baseline 2 to its first phase. Section
  :func:`fluid_decomposition` proves the two agree on the acquisition term, so
  every difference between them is attributable to the two phases baseline 1
  omits and to nothing else.

An under-strength version of this baseline would inflate the apparent size of
the finding. That is a defect class this mission has shipped, so the agreement
with baseline 2 on the shared term is asserted by
``tests/test_baselines.py::test_acquisition_only_equals_fluid_acquisition_term``
and measured by ``validation/validate_baselines_against_truth.py``.

BASELINE 2, THE ANALYTIC FLUID MODEL, DERIVED
---------------------------------------------

Fluid (continuous-flow) buffer with constant source rate ``R_s``, capacity
``B``, occupancy ``q_0`` at the break, and the egress schedule declared in
:mod:`handoverloss.sequence`. Write ``q(t)`` for the occupancy. Tail drop means

.. math::

    \dot q = \min(R_s - r(t),\; 0 \text{ if } q = B \text{ and } R_s > r(t))

i.e. the occupancy is the unconstrained integral clamped at ``B``, and the
dropped volume is the excess that the clamp removes.

*Step 1 — the occupancy is non-decreasing from the break until the egress first
matches the source rate.* On ``[0, T_o)`` the egress is zero, so
``\dot q = R_s > 0``. On the ramp the egress is ``R_d (t - T_o)/T_{ramp}``,
which is below ``R_s`` for ``t - T_o < T_{ramp} R_s / R_d =: \tau^\*``, so
``\dot q > 0`` there too. For ``t - T_o > \tau^\*`` the egress exceeds ``R_s``
and the occupancy strictly decreases, and once it is below ``B`` it never
returns (the egress only grows, then stays at ``R_d > R_s``). So the single
monotone-up excursion ends at ``T_o + \tau^\*`` and all loss occurs at or before
that instant.

*Step 2 — a single monotone-up excursion may be clamped phase by phase.* For a
monotone-non-decreasing unconstrained integral, clamping at ``B`` continuously
and clamping only at a finite set of interior points give the same total removed
volume, because the clamp is active on a terminal sub-interval of the excursion
and the removed volume on that sub-interval is the difference between the
unconstrained integral and ``B``, which telescopes across any partition. This is
the step that lets the three phases be evaluated in sequence with no
integration.

*Step 3 — net accumulation over the ramp.* While the buffer is full the drained
excess is

.. math::

    \int_0^{\tau^\*}\!\!\left(R_s - \frac{R_d s}{T_{ramp}}\right) ds
      = R_s \tau^\* - \frac{R_d (\tau^\*)^2}{2 T_{ramp}}
      = \frac{R_s^2 T_{ramp}}{2 R_d}

with ``\tau^\* = T_{ramp} R_s / R_d``. The result is the ramp's contribution and
is the closed form the package calls the drain term's *drive*. When ``R_s >=
R_d`` the backlog never drains, ``\tau^\*`` exceeds ``T_{ramp}``, and the drive
becomes ``R_s T_{ramp} - R_d T_{ramp}/2 + (R_s - R_d)(H - T_o - T_{ramp})``: the
ramp integral to its end plus the constant-rate deficit over the rest of the
window. :func:`fluid_decomposition` handles both branches and says which it took.

*Step 4 — the decomposition.* With ``u`` the unconstrained level and ``q`` the
clamped one,

.. math::

    u_A = q_0 + R_s T_{acq}, \quad &L_{acq} = \max(0, u_A - B),
        \quad q_A = \min(u_A, B)\\
    u_R = q_A + R_s T_{rst}, \quad &L_{rst} = \max(0, u_R - B),
        \quad q_R = \min(u_R, B)\\
    u_D = q_R + \Delta_{ramp}, \quad &L_{drain} = \max(0, u_D - B)

and ``L_{total} = L_{acq} + L_{rst} + L_{drain}``, which is exact by step 2.

*Step 5 — evaluate in HEADROOM space, not occupancy space.* Step 4 as written
computes ``u_A - B = (q_0 + R_s T_{acq}) - B``, a difference of two nearly equal
large numbers whenever the buffer starts nearly full — which is the operationally
interesting case, because that is when a handover costs anything. Writing
``h = B - q`` for the free space, the same algebra is

.. math::

    L_{acq} = \max(0, R_s T_{acq} - h_0), \quad
        h_A = \max(0, h_0 - R_s T_{acq}) \\
    L_{rst} = \max(0, R_s T_{rst} - h_A), \quad
        h_R = \max(0, h_A - R_s T_{rst}) \\
    L_{drain} = \max(0, \Delta_{ramp} - h_R)

with no large-number cancellation anywhere: every subtraction is between the
free space and the volume offered against it. **This was not a stylistic
choice.** The occupancy form was implemented first and a Hypothesis property
test falsified
``test_acquisition_term_equals_baseline_one`` at ``R_s = 1.0e6``,
``B = q_0 = 15608060.0``, ``T_{acq} = 1.1691562848899153``: the occupancy form
returned ``1169156.284889914`` where baseline 1 returned
``1169156.2848899153``, a relative discrepancy of ``9.96e-16`` caused by
cancellation that baseline 1's own expression does not suffer. The
counterexample is pinned in
``tests/test_regressions.py::test_headroom_form_removes_the_cancellation_hypothesis_found``
with the literal inputs, because ``.hypothesis/`` is gitignored. In headroom
space the two agree to the last bit on that input, and the identity holds across
the whole declared strategy domain.

The reported :attr:`FluidDecomposition.peak_occupancy_bits` is still an
occupancy and still carries that cancellation. It is a diagnostic and no loss
term is derived from it; the note is here so a reader does not take it for a
quantity of the same precision as the three terms.

ASSUMPTIONS, STATED
-------------------

#. Constant source rate over the handover window. Burstiness is not modelled:
   the fluid model cannot produce loss from a burst arriving at a non-full
   buffer, and the discrete trace can. That difference is a reported number in
   ``validation/validate_baselines_against_truth.py``.
#. Constant drain rate ``R_d`` after the ramp and a LINEAR ramp before it.
#. Finite buffer, tail drop, no retransmission and no recovery of dropped data.
#. The declared restart latency is a pure dead time, with zero egress.
#. The accounting window is long enough to contain the whole excursion, which
   :class:`~handoverloss.sequence.HandoverSequence` enforces.

WHERE IT STOPS BEING VALID
--------------------------

* **Bursty sources.** With a coefficient of variation above about 0.3 per
  accounting slot the fluid model understates loss, because a burst can overflow
  a buffer the fluid model leaves with headroom. Measured on this repository's
  corpus and reported as the fluid model's own bias.
* **``R_s >= R_d``.** The second branch of step 3 applies and the answer then
  depends on the horizon, so it is no longer a property of the handover alone.
  :func:`fluid_decomposition` sets ``drains`` false in that case.
* **Non-linear egress ramps.** A concave ramp (true slow start is exponential in
  round trips, hence concave in the envelope only over its first phase) gives a
  larger drain term than the linear envelope used here; a convex one a smaller.
  The linear form is a declared choice and is the midpoint of those two.
* **Make-before-break.** ``T_acq = 0`` is accepted and gives ``L_acq = 0``, but
  a real make-before-break handover also removes the restart dead time, which
  this model does not do automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sequence import MECHANISMS, HandoverSequence


@dataclass(frozen=True)
class FluidDecomposition:
    """Closed-form per-mechanism loss from the declared sequence. Bits.

    Attributes
    ----------
    acquisition_bits, restart_bits, drain_bits:
        The three mechanism terms, bits, each non-negative.
    total_bits:
        Their sum. Equal to the sum in exact arithmetic; see
        :attr:`sum_residual_bits` for what floating point leaves behind.
    peak_occupancy_bits:
        Unconstrained occupancy at the end of the excursion, bits. Above ``B``
        when there was loss.
    drains:
        Whether ``R_s < R_d``, so the backlog drains after the ramp. False means
        the drain term depends on the accounting horizon and the result is not a
        property of the handover alone.
    ramp_drive_bits:
        The step-3 integral: net volume accumulated after the link came up,
        before the clamp, bits.
    """

    acquisition_bits: float
    restart_bits: float
    drain_bits: float
    total_bits: float
    peak_occupancy_bits: float
    drains: bool
    ramp_drive_bits: float

    @property
    def mechanisms(self) -> dict[str, float]:
        """The three terms keyed by mechanism name, in :data:`MECHANISMS` order."""
        return dict(
            zip(
                MECHANISMS,
                (self.acquisition_bits, self.restart_bits, self.drain_bits),
                strict=True,
            )
        )

    @property
    def sum_residual_bits(self) -> float:
        """``total_bits`` minus the sum of the three terms, bits.

        Zero in exact arithmetic. The value is the floating-point rounding of a
        three-term sum and is reported rather than asserted away.
        """
        return self.total_bits - (
            self.acquisition_bits + self.restart_bits + self.drain_bits
        )

    @property
    def dominant(self) -> str:
        """Name of the largest mechanism; ties resolve in :data:`MECHANISMS` order."""
        values = (self.acquisition_bits, self.restart_bits, self.drain_bits)
        return MECHANISMS[int(np.argmax(values))]


def acquisition_only_loss_bits(
    sequence: HandoverSequence, acquisition_s: float | None = None
) -> float:
    """Baseline 1: loss implied by the acquisition phase alone, bits.

    ``max(0, R_s * T_acq - (B - q0))``. See the module docstring for why this is
    the full-strength form of the assumption rather than a strawman.

    Parameters
    ----------
    sequence:
        The declared handover.
    acquisition_s:
        Acquisition time to evaluate at, seconds. Defaults to the sequence's
        own. Supply an upper quantile of the acquisition distribution to
        evaluate the conservative engineer's version of this baseline.

    Returns
    -------
    float
        Loss in bits, non-negative.
    """
    duration = sequence.acquisition_s if acquisition_s is None else float(acquisition_s)
    if duration < 0.0:
        raise ValueError(f"acquisition_s must be non-negative, got {duration!r}")
    return max(0.0, sequence.source_rate_bps * duration - sequence.headroom_bits)


def ramp_drive_bits(sequence: HandoverSequence) -> float:
    """Step-3 integral: volume accumulated after the link came up, bits.

    ``R_s**2 * T_ramp / (2 * R_d)`` when ``R_s < R_d``. Otherwise the ramp
    integral to its end plus the constant-rate deficit over the remainder of the
    accounting window, which is horizon-dependent by construction.
    """
    rate_s = sequence.source_rate_bps
    rate_d = sequence.downlink_rate_bps
    ramp = sequence.ramp_s
    if ramp <= 0.0:
        if rate_s <= rate_d:
            return 0.0
        return (rate_s - rate_d) * (sequence.horizon_s - sequence.outage_s)
    if rate_s < rate_d:
        return rate_s * rate_s * ramp / (2.0 * rate_d)
    # R_s >= R_d: the egress never catches the source. The ramp integral runs to
    # its own end and the deficit continues at the constant rate thereafter.
    tail_s = sequence.horizon_s - sequence.outage_s - ramp
    return rate_s * ramp - 0.5 * rate_d * ramp + (rate_s - rate_d) * max(tail_s, 0.0)


def fluid_decomposition(sequence: HandoverSequence) -> FluidDecomposition:
    """Baseline 2: the closed-form three-way decomposition. Bits.

    Implements step 4 of the module docstring. Each term is the volume the
    tail-drop clamp removes during one phase.

    Parameters
    ----------
    sequence:
        The declared handover, already validated by its own constructor.

    Returns
    -------
    FluidDecomposition
        With the three terms, their sum, the unconstrained peak and whether the
        backlog drains.
    """
    rate_s = sequence.source_rate_bps

    # Step 5: headroom space. Every subtraction below is between the free space
    # and the volume offered against it, so there is no large-number
    # cancellation. See the module docstring for the Hypothesis counterexample
    # that made this necessary.
    headroom = sequence.headroom_bits
    offered = rate_s * sequence.acquisition_s
    acquisition = max(0.0, offered - headroom)
    headroom = max(0.0, headroom - offered)

    offered = rate_s * sequence.restart_s
    restart = max(0.0, offered - headroom)
    headroom = max(0.0, headroom - offered)

    drive = ramp_drive_bits(sequence)
    drain = max(0.0, drive - headroom)
    # Diagnostic only; carries the occupancy-space cancellation noted above.
    peak = sequence.buffer_capacity_bits - headroom + drive

    return FluidDecomposition(
        acquisition_bits=acquisition,
        restart_bits=restart,
        drain_bits=drain,
        total_bits=acquisition + restart + drain,
        peak_occupancy_bits=peak,
        drains=sequence.source_rate_bps < sequence.downlink_rate_bps,
        ramp_drive_bits=drive,
    )


def fluid_decomposition_array(
    source_rate_bps: np.ndarray,
    downlink_rate_bps: np.ndarray,
    buffer_capacity_bits: np.ndarray,
    initial_occupancy_bits: np.ndarray,
    acquisition_s: np.ndarray,
    restart_s: np.ndarray,
    ramp_s: np.ndarray,
    horizon_s: np.ndarray,
) -> np.ndarray:
    """Vectorised :func:`fluid_decomposition` over many handovers.

    Returns an ``(n, 3)`` float64 array of per-mechanism loss in bits, with
    columns in :data:`MECHANISMS` order. The arithmetic is identical to the
    scalar function; ``tests/test_baselines.py`` asserts the two agree to within
    rounding on random inputs rather than asserting bit-exact equality, because
    the two evaluation orders are not guaranteed to round identically.

    Raises
    ------
    ValueError
        If the arrays do not broadcast to a common shape, or any bound declared
        by :class:`~handoverloss.sequence.HandoverSequence` is violated.
    """
    arrays = np.broadcast_arrays(
        np.asarray(source_rate_bps, dtype=float),
        np.asarray(downlink_rate_bps, dtype=float),
        np.asarray(buffer_capacity_bits, dtype=float),
        np.asarray(initial_occupancy_bits, dtype=float),
        np.asarray(acquisition_s, dtype=float),
        np.asarray(restart_s, dtype=float),
        np.asarray(ramp_s, dtype=float),
        np.asarray(horizon_s, dtype=float),
    )
    rate_s, rate_d, capacity, occupancy, acq, rst, ramp, horizon = (
        np.atleast_1d(a).astype(float) for a in arrays
    )
    if np.any(rate_s <= 0) or np.any(rate_d <= 0) or np.any(capacity <= 0):
        raise ValueError(
            "source_rate_bps, downlink_rate_bps and buffer_capacity_bits must all be "
            "strictly positive"
        )
    if np.any(occupancy < 0) or np.any(acq < 0) or np.any(rst < 0) or np.any(ramp < 0):
        raise ValueError(
            "initial_occupancy_bits, acquisition_s, restart_s and ramp_s must all be "
            "non-negative"
        )
    if np.any(occupancy > capacity):
        raise ValueError("initial_occupancy_bits must not exceed buffer_capacity_bits")
    if np.any(horizon < acq + rst + ramp):
        raise ValueError(
            "horizon_s must be at least acquisition_s + restart_s + ramp_s for every "
            "handover"
        )

    # Headroom space, as in the scalar function. See step 5 of the module
    # docstring.
    headroom = capacity - occupancy
    offered = rate_s * acq
    loss_acq = np.maximum(0.0, offered - headroom)
    headroom = np.maximum(0.0, headroom - offered)

    offered = rate_s * rst
    loss_rst = np.maximum(0.0, offered - headroom)
    headroom = np.maximum(0.0, headroom - offered)

    drains = rate_s < rate_d
    tail = np.maximum(horizon - acq - rst - ramp, 0.0)
    drive_draining = np.where(
        ramp > 0.0, rate_s * rate_s * ramp / (2.0 * rate_d), 0.0
    )
    drive_saturated = np.where(
        ramp > 0.0,
        rate_s * ramp - 0.5 * rate_d * ramp + (rate_s - rate_d) * tail,
        (rate_s - rate_d) * (horizon - acq - rst),
    )
    drive = np.where(drains, drive_draining, drive_saturated)
    loss_drain = np.maximum(0.0, drive - headroom)

    return np.stack([loss_acq, loss_rst, loss_drain], axis=-1)
