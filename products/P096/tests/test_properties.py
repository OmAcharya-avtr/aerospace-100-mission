"""Property-based tests for the algebraic identities the decomposition satisfies.

The decomposition sum is an algebraic identity in the fluid model, which is what
makes Hypothesis the right tool here: the identity must hold for every handover
in the declared domain, not for the ones a fixture happened to choose.

Every counterexample Hypothesis has found against this file is pinned as a
named regression test in ``tests/test_regressions.py`` with literal inputs in
the body, because ``.hypothesis/`` is gitignored and a falsifying example found
by one run is lost on a fresh clone.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from handoverloss.baselines import (
    acquisition_only_loss_bits,
    fluid_decomposition,
    ramp_drive_bits,
)
from handoverloss.bounds import loss_bracket
from handoverloss.sequence import HandoverSequence

#: Declared strategy domain. Rates 1e6 to 5e9 bit/s, buffers 1e6 to 1e12 bits,
#: durations 0 to 300 s. The horizon is always built to contain the excursion,
#: so no strategy can produce a sequence the constructor refuses.
RATES = st.floats(min_value=1.0e6, max_value=5.0e9, allow_nan=False, allow_infinity=False)
CAPACITIES = st.floats(
    min_value=1.0e6, max_value=1.0e12, allow_nan=False, allow_infinity=False
)
FRACTIONS = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
DURATIONS = st.floats(min_value=0.0, max_value=300.0, allow_nan=False)


@st.composite
def sequences(draw) -> HandoverSequence:
    """A HandoverSequence drawn from the declared domain, always constructible."""
    source = draw(RATES)
    downlink = draw(RATES)
    capacity = draw(CAPACITIES)
    occupancy = capacity * draw(FRACTIONS)
    acquisition = draw(DURATIONS)
    restart = draw(DURATIONS)
    ramp = draw(DURATIONS)
    extra = draw(st.floats(min_value=0.0, max_value=600.0, allow_nan=False))
    return HandoverSequence(
        source_rate_bps=source,
        downlink_rate_bps=downlink,
        buffer_capacity_bits=capacity,
        initial_occupancy_bits=min(occupancy, capacity),
        acquisition_s=acquisition,
        restart_s=restart,
        ramp_s=ramp,
        horizon_s=acquisition + restart + ramp + extra,
    )


SETTINGS = settings(
    max_examples=250,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


@given(sequences())
@SETTINGS
def test_decomposition_sums_to_its_total(sequence):
    # The algebraic identity: L_acq + L_rst + L_drain == L_total. Exact in exact
    # arithmetic; asserted to within the rounding of a three-term sum, which is
    # what double precision actually guarantees.
    decomposition = fluid_decomposition(sequence)
    parts = (
        decomposition.acquisition_bits
        + decomposition.restart_bits
        + decomposition.drain_bits
    )
    scale = (
        abs(decomposition.acquisition_bits)
        + abs(decomposition.restart_bits)
        + abs(decomposition.drain_bits)
    )
    assert abs(parts - decomposition.total_bits) <= 8.0 * (2.0**-53) * scale


@given(sequences())
@SETTINGS
def test_every_term_is_non_negative(sequence):
    decomposition = fluid_decomposition(sequence)
    assert decomposition.acquisition_bits >= 0.0
    assert decomposition.restart_bits >= 0.0
    assert decomposition.drain_bits >= 0.0
    assert decomposition.ramp_drive_bits >= 0.0


@given(sequences())
@SETTINGS
def test_total_never_exceeds_what_the_source_offered(sequence):
    # Nothing can be lost that was never offered, and the already-buffered data
    # is never dropped by a tail-drop rule.
    decomposition = fluid_decomposition(sequence)
    offered = sequence.source_rate_bps * sequence.horizon_s
    assert decomposition.total_bits <= offered * (1.0 + 1e-12) + 1.0


@given(sequences())
@SETTINGS
def test_independent_bracket_contains_the_closed_form(sequence):
    # The bound check, as a property: a bracket from conservation and capacity
    # counting alone contains the closed form's total, for every handover in the
    # declared domain. The slack follows the conditioning of the bound
    # expression, not an absolute constant.
    decomposition = fluid_decomposition(sequence)
    bracket = loss_bracket(
        offered_bits=sequence.source_rate_bps * sequence.horizon_s,
        downlink_rate_bps=sequence.downlink_rate_bps,
        outage_s=sequence.outage_s,
        ramp_s=sequence.ramp_s,
        horizon_s=sequence.horizon_s,
        buffer_capacity_bits=sequence.buffer_capacity_bits,
        initial_occupancy_bits=sequence.initial_occupancy_bits,
    )
    assert bracket.gap_below_bits(decomposition.total_bits) <= bracket.slack_bits
    assert bracket.gap_above_bits(decomposition.total_bits) <= bracket.slack_bits


@given(sequences())
@SETTINGS
def test_baseline_one_never_exceeds_baseline_two(sequence):
    # Omitting phases can only reduce the loss, so baseline 1 is a lower bound
    # on baseline 2 at the same acquisition time. The slack is the rounding of
    # the larger quantity.
    total = fluid_decomposition(sequence).total_bits
    only = acquisition_only_loss_bits(sequence)
    assert only <= total * (1.0 + 1e-12) + 1.0


@given(sequences())
@SETTINGS
def test_acquisition_term_equals_baseline_one(sequence):
    # The shared term. This is the identity that proves baseline 1 is the
    # full-strength restriction of baseline 2 rather than a weakened variant.
    decomposition = fluid_decomposition(sequence)
    only = acquisition_only_loss_bits(sequence)
    scale = max(abs(decomposition.acquisition_bits), abs(only), 1.0)
    assert abs(decomposition.acquisition_bits - only) <= 8.0 * (2.0**-53) * scale


@given(sequences(), st.floats(min_value=1.0, max_value=1.0e6, allow_nan=False))
@SETTINGS
def test_more_headroom_never_increases_the_total(sequence, extra_bits):
    # Monotonicity in the buffer: freeing space cannot make the loss larger.
    freed = HandoverSequence(
        source_rate_bps=sequence.source_rate_bps,
        downlink_rate_bps=sequence.downlink_rate_bps,
        buffer_capacity_bits=sequence.buffer_capacity_bits,
        initial_occupancy_bits=max(0.0, sequence.initial_occupancy_bits - extra_bits),
        acquisition_s=sequence.acquisition_s,
        restart_s=sequence.restart_s,
        ramp_s=sequence.ramp_s,
        horizon_s=sequence.horizon_s,
    )
    base = fluid_decomposition(sequence).total_bits
    loosened = fluid_decomposition(freed).total_bits
    assert loosened <= base * (1.0 + 1e-12) + 1.0


@given(sequences(), st.floats(min_value=0.0, max_value=200.0, allow_nan=False))
@SETTINGS
def test_a_longer_outage_never_reduces_the_total(sequence, extra_s):
    # Monotonicity in the outage: a longer acquisition cannot lose less.
    longer = HandoverSequence(
        source_rate_bps=sequence.source_rate_bps,
        downlink_rate_bps=sequence.downlink_rate_bps,
        buffer_capacity_bits=sequence.buffer_capacity_bits,
        initial_occupancy_bits=sequence.initial_occupancy_bits,
        acquisition_s=sequence.acquisition_s + extra_s,
        restart_s=sequence.restart_s,
        ramp_s=sequence.ramp_s,
        # Rebuilt from the PHASES, not by adding to the old horizon. Hypothesis
        # falsified the latter at restart_s=213.0, ramp_s=299.49081121423325,
        # extra_s=1.490811214233247, where the two summation orders differ by
        # one ulp and the constructor's own bound refuses the result. Pinned in
        # tests/test_regressions.py.
        horizon_s=(
            sequence.acquisition_s
            + extra_s
            + sequence.restart_s
            + sequence.ramp_s
            + max(0.0, sequence.horizon_s - sequence.acquisition_s
                  - sequence.restart_s - sequence.ramp_s)
        ),
    )
    base = fluid_decomposition(sequence).total_bits
    extended = fluid_decomposition(longer).total_bits
    assert extended >= base * (1.0 - 1e-12) - 1.0


@given(
    st.floats(min_value=1.0e6, max_value=1.0e9, allow_nan=False),
    st.floats(min_value=1.0e6, max_value=5.0e9, allow_nan=False),
    st.floats(min_value=0.0, max_value=300.0, allow_nan=False),
)
@SETTINGS
def test_ramp_drive_matches_its_closed_form_when_the_backlog_drains(
    source, downlink, ramp
):
    # Only meaningful when R_s < R_d, which is the branch with the closed form
    # R_s^2 T_ramp / (2 R_d). The other branch is horizon-dependent and is
    # covered by tests/test_baselines.py.
    if not source < downlink:
        return
    sequence = HandoverSequence(
        source_rate_bps=source,
        downlink_rate_bps=downlink,
        buffer_capacity_bits=1.0e12,
        initial_occupancy_bits=0.0,
        acquisition_s=1.0,
        restart_s=1.0,
        ramp_s=ramp,
        horizon_s=400.0 + ramp,
    )
    expected = source * source * ramp / (2.0 * downlink)
    measured = ramp_drive_bits(sequence)
    # Two orderings of the same product and quotient, so the agreement is to
    # within rounding scaled by the magnitude; no absolute constant.
    assert abs(measured - expected) <= 4.0 * (2.0**-53) * max(abs(expected), 1.0)
