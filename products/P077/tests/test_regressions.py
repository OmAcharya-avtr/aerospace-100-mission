"""Counterexamples found by Hypothesis, pinned with literal inputs.

``.hypothesis/`` is not committed, so a falsifying example found by one run is
lost on a fresh clone. Every defect Hypothesis found in this package is
reproduced below with the exact inputs that falsified the property, named after
what went wrong, so that a regression is caught by ``pytest`` alone and does
not depend on the search finding it a second time.
"""

from __future__ import annotations

import pytest

from relfiledelivery.analytic import analytic_delivery
from relfiledelivery.link import (
    WINDOW_ADJACENCY_TOLERANCE,
    ContactSchedule,
    ContactWindow,
    LinkConditions,
    uniform_windows,
)
from relfiledelivery.modes import TransferMode
from relfiledelivery.payload import FileTransferSpec
from relfiledelivery.simulate import simulate_transfer


def test_regression_back_to_back_windows_are_adjacent_not_overlapping() -> None:
    """Found by Hypothesis: ``uniform_windows`` built a list its own consumer rejected.

    ``uniform_windows(11, d, d)`` with ``d = 114.06854261351316`` produces
    contiguous windows. The tenth window ends at ``first + 9 * d + d`` and the
    eleventh starts at ``first + 10 * d``; in binary floating point those two
    expressions differ by two parts in 1e16, and the strict comparison
    ``later.start_s < earlier.end_s`` in :class:`LinkConditions` then reported
    an overlap of 2.3e-13 seconds and refused the construction.

    The defect class is the one this mission has shipped twice: a boolean
    produced by a strict comparison of two rounded quantities. The fix compares
    with a relative tolerance, and the tolerance is named rather than inlined.
    """
    duration = 114.06854261351316
    windows = uniform_windows(11, duration, duration)
    # The two boundaries really do differ, so the test is not vacuous.
    assert windows[9].end_s != windows[10].start_s
    assert abs(windows[9].end_s - windows[10].start_s) < 1e-9
    link = LinkConditions(1000.0, 100.0, windows=windows)
    # Second defect in the same example: the summed window durations exceed the
    # differenced span by one part in 5e15, so the unclamped duty cycle was
    # 1.0000000000000002 and was reported as a probability.
    raw = sum(w.duration_s for w in windows) / (windows[-1].end_s - windows[0].start_s)
    assert raw > 1.0
    assert 0.0 < link.contact_duty_cycle <= 1.0
    assert link.contact_duty_cycle == pytest.approx(1.0, abs=1e-9)
    assert 0.0 < link.availability <= 1.0


def test_regression_a_genuine_overlap_is_still_rejected() -> None:
    """The tolerance must not swallow an overlap a reader would call an overlap."""
    with pytest.raises(ValueError, match="non-overlapping"):
        LinkConditions(
            1000.0,
            100.0,
            windows=(ContactWindow(0.0, 100.0), ContactWindow(99.0, 200.0)),
        )
    # One tolerance-width past the boundary is still accepted; ten thousand
    # tolerance-widths is not.
    scale = 100.0
    LinkConditions(
        1000.0,
        100.0,
        windows=(
            ContactWindow(0.0, scale),
            ContactWindow(scale - 0.5 * WINDOW_ADJACENCY_TOLERANCE * scale, 2 * scale),
        ),
    )
    with pytest.raises(ValueError, match="non-overlapping"):
        LinkConditions(
            1000.0,
            100.0,
            windows=(
                ContactWindow(0.0, scale),
                ContactWindow(scale - 1e4 * WINDOW_ADJACENCY_TOLERANCE * scale, 2 * scale),
            ),
        )


def test_regression_contiguous_windows_behave_like_continuous_contact() -> None:
    """A consequence of the fix above, measured rather than assumed.

    With contiguous windows a transfer should take the same time as it does
    under continuous contact, up to the single declared modelling choice that a
    unit is never split across a window boundary (M1), which can only *delay*
    it. The two must therefore agree to better than one unit time.
    """
    spec = FileTransferSpec(40 * 256, 256, 12)
    duration = 114.06854261351316
    contiguous = LinkConditions(
        5000.0, 500.0, 0.25, 0.0, windows=uniform_windows(11, duration, duration)
    )
    continuous = LinkConditions(5000.0, 500.0, 0.25, 0.0)
    windowed = simulate_transfer(spec, contiguous, TransferMode.DEFERRED_NAK, 1)
    plain = simulate_transfer(spec, continuous, TransferMode.DEFERRED_NAK, 1)
    assert windowed.delivered and plain.delivered
    unit_time = 8.0 * spec.full_wire_bytes / 5000.0
    assert windowed.delivery_time_s == pytest.approx(
        plain.delivery_time_s, abs=unit_time + 1e-9
    )


def test_regression_a_single_unit_file_has_one_round_and_no_nak() -> None:
    """A one-unit file is the smallest case and the easiest to get wrong.

    With ``n = 1`` the closed-form round count reduces to ``1 / (1 - p)`` and
    the delivery time to one unit time plus the light time at ``p = 0``. Kept
    as a regression because an off-by-one in the round loop would show here
    first.
    """
    spec = FileTransferSpec(1, 1024, 12)
    link = LinkConditions(1000.0, 100.0, 0.5, 0.0)
    assert spec.pdu_count == 1
    result = simulate_transfer(spec, link, TransferMode.DEFERRED_NAK, 1)
    assert result.rounds == 1
    assert result.nak_count == 0
    assert result.forward_transmissions == 1
    prediction = analytic_delivery(spec, link, TransferMode.DEFERRED_NAK)
    assert result.delivery_time_s == pytest.approx(
        prediction.expected_delivery_time_s, rel=1e-12
    )


def test_regression_zero_header_bytes_gives_unit_overhead_not_a_division_error() -> None:
    """``pdu_header_bytes = 0`` is admissible and must not be a special case."""
    spec = FileTransferSpec(1024, 256, 0)
    link = LinkConditions(1000.0, 100.0, 0.0, 0.0)
    assert spec.minimum_overhead_ratio == pytest.approx(1.0)
    result = simulate_transfer(spec, link, TransferMode.UNACKNOWLEDGED, 1)
    assert result.overhead_ratio == pytest.approx(1.0)


def test_regression_a_zero_second_request_is_placeable() -> None:
    """``ContactSchedule.place`` with a zero duration must not be rejected."""
    schedule = ContactSchedule(uniform_windows(2, 10.0, 100.0))
    placed = schedule.place(5.0, 0.0)
    assert placed is not None
    start, end, index = placed
    assert start == pytest.approx(end)
    assert index == 0
