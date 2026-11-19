"""The decomposition identity, its residual, the interval algebra, the edges."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passyield.decompose import (
    DEFAULT_NODES_PER_INTERVAL,
    MASK_CLAMP_TOLERANCE_DEG,
    DecompositionOptions,
    PassEvents,
    complement,
    decompose_pass,
    integrate_rate_mbit,
    nominal_rate_mbit_s,
    normalise_intervals,
    rate_at_offsets,
    simpson_error_scale,
)
from passyield.geometry import PassWindow

#: Relative tolerance on the decomposition identity. The identity is exact in
#: exact arithmetic; here the four terms are integrated on four different
#: quadrature partitions of the same window, so it closes to the quadrature
#: and not to the rounding. The figure is the measured 95th percentile over
#: the committed campaign rounded up by an order of magnitude, published by
#: validation/validate_decomposition.py.
IDENTITY_RELATIVE_TOLERANCE = 1.0e-5


# ---------------------------------------------------------------- interval algebra


def test_normalise_intervals_hand_calculation():
    """Hand-calculated: a 100 s window, 10 s acquisition, overlapping losses.

    Inputs: acquisition 10 s, re-sync ``(5, 20)`` so ``[5, 25]``, idle
    ``(20, 30)`` so ``[20, 50]``.

    Priority is acquisition, then re-sync, then idle. So:
      acquisition = [0, 10]
      re-sync     = [5, 25] minus [0, 10] = [10, 25]
      idle        = [20, 50] minus [0, 10] minus [10, 25] = [25, 50]
    Total charged = 10 + 15 + 25 = 50 s, and the active set is [50, 100],
    which is 50 s. The three lists are pairwise disjoint, which is the point:
    the raw inputs overlap by 15 s and charging that twice would overstate
    the shortfall.
    """
    acquisition, resync, idle, truncated = normalise_intervals(
        100.0, 10.0, ((5.0, 20.0),), ((20.0, 30.0),)
    )
    assert acquisition == [(0.0, 10.0)]
    assert resync == [(10.0, 25.0)]
    assert idle == [(25.0, 50.0)]
    assert truncated is False
    assert complement(100.0, acquisition + resync + idle) == [(50.0, 100.0)]


def test_normalise_intervals_clips_at_the_window_end():
    """An interval running past mask set is clipped, never extrapolated."""
    _acquisition, resync, _idle, _truncated = normalise_intervals(
        50.0, 5.0, ((40.0, 100.0),), ()
    )
    assert resync == [(40.0, 50.0)]


def test_normalise_intervals_flags_acquisition_longer_than_the_pass():
    """Edge case: acquisition exceeding the whole window.

    The interval is clipped to the window, the flag is set, and the active
    set is empty -- so the realised yield is exactly zero rather than a
    silent negative.
    """
    acquisition, resync, idle, truncated = normalise_intervals(
        50.0, 80.0, ((10.0, 5.0),), ((20.0, 5.0),)
    )
    assert acquisition == [(0.0, 50.0)]
    assert resync == []
    assert idle == []
    assert truncated is True
    assert complement(50.0, acquisition) == []


def test_normalise_intervals_rejects_a_non_positive_window():
    """Input validation: a window of zero length has no intervals."""
    with pytest.raises(ValueError, match="window_duration_s"):
        normalise_intervals(0.0, 1.0, (), ())


@given(
    acquisition=st.floats(min_value=0.0, max_value=120.0),
    first=st.tuples(
        st.floats(min_value=0.0, max_value=200.0), st.floats(min_value=0.0, max_value=60.0)
    ),
    second=st.tuples(
        st.floats(min_value=0.0, max_value=200.0), st.floats(min_value=0.0, max_value=60.0)
    ),
    third=st.tuples(
        st.floats(min_value=0.0, max_value=200.0), st.floats(min_value=0.0, max_value=60.0)
    ),
)
@settings(max_examples=250, deadline=None)
def test_normalised_intervals_partition_the_window(acquisition, first, second, third):
    """Property: the four pieces are disjoint and their lengths total the window.

    This is the algebraic identity the decomposition rests on. If the pieces
    overlapped, a second of outage inside an idle period would be charged
    twice and the shortfall would be overstated.
    """
    duration = 200.0
    pieces = normalise_intervals(duration, acquisition, (first, second), (third,))
    charged = pieces[0] + pieces[1] + pieces[2]
    active = complement(duration, charged)
    total = sum(stop - start for start, stop in charged + active)
    assert total == pytest.approx(duration, rel=1.0e-12, abs=1.0e-9)
    ordered = sorted(charged + active)
    for (_, stop), (start, _) in zip(ordered, ordered[1:], strict=False):
        assert start >= stop - 1.0e-12


def test_regression_zero_length_resync_interval_is_dropped():
    """Regression, pinned from a Hypothesis counterexample with literal inputs.

    Inputs: window 200 s, acquisition 0.0 s, re-sync ``((0.0, 0.0), (0.0,
    0.0))``, idle ``((0.0, 0.0),)``. A zero-length interval must contribute
    nothing and must not appear as a degenerate ``(x, x)`` piece, because a
    degenerate piece would make the partition test's adjacency check pass
    vacuously.
    """
    acquisition, resync, idle, truncated = normalise_intervals(
        200.0, 0.0, ((0.0, 0.0), (0.0, 0.0)), ((0.0, 0.0),)
    )
    assert acquisition == []
    assert resync == []
    assert idle == []
    assert truncated is False
    assert complement(200.0, []) == [(0.0, 200.0)]


# ---------------------------------------------------------------- quadrature


def test_simpson_error_scale_hand_calculation():
    """Hand-calculated: ``(b-a) h^4 / 180`` with ``b-a = 180`` s, 3 nodes.

    ``h = 180 / 2 = 90`` s, ``h^4 = 65610000``, so the factor is
    ``180 * 65610000 / 180 = 65610000`` exactly.
    """
    assert simpson_error_scale(180.0, 3) == pytest.approx(65610000.0, rel=1.0e-12, abs=0.0)


@pytest.mark.parametrize("nodes", [2, 4, 32, 1, 0, -1])
def test_simpson_error_scale_rejects_even_or_small_node_counts(nodes):
    """Composite Simpson needs an odd node count of at least 3."""
    with pytest.raises(ValueError, match="nodes"):
        simpson_error_scale(100.0, nodes)


def test_simpson_error_scale_rejects_a_negative_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="interval_s"):
        simpson_error_scale(-1.0, 33)


def test_integrate_rate_over_a_zero_length_interval_is_zero(satrec, station, long_window):
    """A zero-length interval integrates to zero without propagating."""
    volume, clamp = integrate_rate_mbit(satrec, station, long_window, 7.0, 7.0)
    assert volume == 0.0
    assert clamp == 0.0


def test_integrate_rate_rejects_a_reversed_interval(satrec, station, long_window):
    """Input validation: the stop must not precede the start."""
    with pytest.raises(ValueError, match="stop_s"):
        integrate_rate_mbit(satrec, station, long_window, 10.0, 5.0)


def test_integrate_rate_rejects_an_even_node_count(satrec, station, long_window):
    """Input validation on the quadrature."""
    with pytest.raises(ValueError, match="nodes"):
        integrate_rate_mbit(satrec, station, long_window, 0.0, 10.0, nodes=32)


def test_rate_at_offsets_reports_a_small_mask_clamp(satrec, station, long_window):
    """The endpoint clamp is below the declared tolerance on a real window."""
    offsets = np.linspace(0.0, long_window.duration_s, 65)
    rates, clamp = rate_at_offsets(satrec, station, long_window, offsets)
    assert clamp <= MASK_CLAMP_TOLERANCE_DEG
    assert float(rates.min()) > 0.0


def test_rate_at_offsets_refuses_an_interval_outside_the_window(satrec, station, long_window):
    """A window that is not a mask-to-mask interval raises, and says why."""
    offsets = np.asarray([-60.0, 0.0])
    with pytest.raises(ValueError, match="below the"):
        rate_at_offsets(satrec, station, long_window, offsets)


# ---------------------------------------------------------------- the identity


def test_decomposition_identity_closes_on_a_real_pass(satrec, station, long_window, events):
    """The four terms total the shortfall to within the stated residual.

    Not asserted bit-exactly: each term is a composite Simpson sum over its
    own partition of the window, and four sums over four partitions do not
    telescope. The residual is reported on the object and checked against a
    relative tolerance.
    """
    item = decompose_pass(satrec, station, long_window, events)
    assert item.decomposed_bits == pytest.approx(
        item.shortfall_bits, rel=IDENTITY_RELATIVE_TOLERANCE, abs=0.0
    )
    assert abs(item.residual_bits) <= IDENTITY_RELATIVE_TOLERANCE * abs(item.shortfall_bits)


def test_decomposition_residual_shrinks_under_grid_refinement(
    satrec, station, long_window, events
):
    """Refining the quadrature reduces the closing error of the identity.

    Asserted as a direction, not as an order: the rate profile is only
    piecewise smooth, because ``min(eta_max, ...)`` puts a kink at the
    saturation point, so the C4 Simpson error order does not hold there.
    validation/validate_decomposition.py measures and publishes what the
    observed reduction actually is.
    """
    coarse = decompose_pass(
        satrec, station, long_window, events,
        DecompositionOptions(nodes_per_interval=9, nodes_per_window=65),
    )
    fine = decompose_pass(
        satrec, station, long_window, events,
        DecompositionOptions(nodes_per_interval=129, nodes_per_window=1025),
    )
    assert abs(fine.residual_bits) < abs(coarse.residual_bits)


def test_realised_yield_is_bounded_by_the_capacity(satrec, station, long_window, events):
    """The record cannot account for more than the window could carry."""
    item = decompose_pass(satrec, station, long_window, events)
    assert 0.0 <= item.realised_bits <= item.capacity_bits
    assert 0.0 <= item.realised_fraction <= 1.0


def test_acquisition_longer_than_the_pass_yields_exactly_zero(satrec, station, long_window):
    """Edge case: documented behaviour, never a silent negative yield.

    The acquisition interval is clipped to the window, the flag is set, the
    realised yield is exactly 0.0 Mbit and the acquisition term carries the
    whole capacity.

    The acquisition term and the capacity are compared at the identity's
    stated tolerance and not bit-exactly, because the acquisition interval is
    integrated with ``nodes_per_interval`` nodes and the capacity with
    ``nodes_per_window``: the two cover the same interval with different
    quadratures, which is exactly the source of the reported residual.
    """
    events = PassEvents(acquisition_s=long_window.duration_s + 500.0)
    item = decompose_pass(satrec, station, long_window, events)
    assert item.acquisition_truncated is True
    assert item.realised_bits == 0.0
    assert item.acquisition_bits == pytest.approx(
        item.capacity_bits, rel=IDENTITY_RELATIVE_TOLERANCE, abs=0.0
    )
    assert item.resync_bits == 0.0
    assert item.idle_bits == 0.0
    assert item.shortfall_bits == pytest.approx(item.nominal_bits, rel=1.0e-14, abs=0.0)


def test_zero_duration_window_raises(satrec, station, long_window):
    """Edge case: a zero-duration pass is refused, with the reason."""
    degenerate = PassWindow(
        station=station.name,
        rise_jd=long_window.rise_jd,
        rise_fr=long_window.rise_fr,
        duration_s=0.0,
        max_elevation_deg=long_window.max_elevation_deg,
        min_slant_range_km=long_window.min_slant_range_km,
    )
    with pytest.raises(ValueError, match="zero or negative duration"):
        decompose_pass(satrec, station, degenerate, PassEvents(1.0))


def test_window_below_the_mask_raises(satrec, station, long_window):
    """Edge case: a pass that never rises above the mask is refused."""
    grazing = PassWindow(
        station=station.name,
        rise_jd=long_window.rise_jd,
        rise_fr=long_window.rise_fr,
        duration_s=long_window.duration_s,
        max_elevation_deg=station.mask_deg - 1.0,
        min_slant_range_km=long_window.min_slant_range_km,
    )
    with pytest.raises(ValueError, match="does not clear"):
        decompose_pass(satrec, station, grazing, PassEvents(1.0))


def test_realised_fraction_of_a_zero_capacity_window_raises():
    """A ratio with a zero denominator is refused rather than returned as a nan."""
    from passyield.decompose import Decomposition

    item = Decomposition(
        station="X", nominal_bits=1.0, capacity_bits=0.0, realised_bits=0.0,
        shortfall_bits=1.0, geometry_bits=1.0, acquisition_bits=0.0, resync_bits=0.0,
        idle_bits=0.0, residual_bits=0.0, window_duration_s=1.0, active_duration_s=1.0,
        acquisition_truncated=False, max_mask_clamp_deg=0.0,
    )
    with pytest.raises(ValueError, match="zero capacity"):
        _ = item.realised_fraction


@pytest.mark.parametrize(
    ("acquisition", "resync", "idle"),
    [
        (-1.0, (), ()),
        (float("nan"), (), ()),
        (1.0, ((-1.0, 5.0),), ()),
        (1.0, ((5.0, -1.0),), ()),
        (1.0, (), ((float("inf"), 5.0),)),
    ],
)
def test_pass_events_rejects_out_of_range_inputs(acquisition, resync, idle):
    """Input validation on the event timeline."""
    with pytest.raises(ValueError):
        PassEvents(acquisition, resync, idle)


def test_pass_events_totals(events):
    """The declared totals, before clipping, are sums of the declared pieces."""
    assert events.resync_total_s == pytest.approx(27.0, rel=0.0, abs=1.0e-12)
    assert events.idle_total_s == pytest.approx(25.0, rel=0.0, abs=1.0e-12)


def test_nominal_rate_refuses_a_reference_below_the_mask(satrec, station, long_window):
    """A schedule cannot book a rate at an elevation it never uses."""
    with pytest.raises(ValueError, match="below the station mask"):
        nominal_rate_mbit_s(satrec, station, long_window, station.mask_deg - 1.0)


def test_nominal_rate_is_positive_and_below_the_ceiling(satrec, station, long_window):
    """The booked rate is a real rate from the budget, not a stipulation."""
    from passyield.link import DEFAULT_BUDGET

    rate = nominal_rate_mbit_s(satrec, station, long_window, 30.0)
    assert 0.0 < rate <= DEFAULT_BUDGET.max_rate_mbit_s


def test_default_node_count_is_odd():
    """Composite Simpson needs an even number of sub-intervals."""
    assert DEFAULT_NODES_PER_INTERVAL % 2 == 1
