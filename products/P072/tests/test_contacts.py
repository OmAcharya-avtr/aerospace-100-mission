"""Contact windows: rate profiles, the capacity integral, the inverse, extension."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from downlinkpick.contacts import (
    CAPACITY_TOLERANCE_BITS,
    ContactPlan,
    ContactWindow,
    constant_rate_profile,
    rate_profile_from_elevation,
)


def test_flat_capacity_is_rate_times_duration() -> None:
    # Hand calculation: 2e5 bit/s over 100 s = 2e7 bits.
    window = ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 2.0e5))
    assert window.capacity_bits() == pytest.approx(2.0e7, rel=1e-15)
    assert window.duration_s == pytest.approx(100.0, rel=1e-15)
    assert window.peak_rate_bps == pytest.approx(2.0e5, rel=1e-15)
    assert window.final_rate_bps == pytest.approx(2.0e5, rel=1e-15)


def test_triangular_capacity_by_hand() -> None:
    # Breakpoints (0, 0), (10, 100), (20, 0): two triangles of area 500 each.
    window = ContactWindow("w", 0.0, 20.0, ((0.0, 0.0), (10.0, 100.0), (20.0, 0.0)))
    assert window.capacity_bits() == pytest.approx(1000.0, rel=1e-15)


def test_rate_is_linear_between_breakpoints() -> None:
    window = ContactWindow("w", 0.0, 10.0, ((0.0, 0.0), (10.0, 100.0)))
    assert window.rate_bps(0.0) == pytest.approx(0.0)
    assert window.rate_bps(2.5) == pytest.approx(25.0, rel=1e-15)
    assert window.rate_bps(10.0) == pytest.approx(100.0, rel=1e-15)


def test_rate_is_zero_outside_the_window() -> None:
    window = ContactWindow("w", 5.0, 10.0, constant_rate_profile(5.0, 10.0, 7.0))
    assert window.rate_bps(4.999) == 0.0
    assert window.rate_bps(10.001) == 0.0


def test_rate_is_right_continuous_at_a_step() -> None:
    window = ContactWindow("w", 0.0, 20.0, ((0.0, 10.0), (10.0, 10.0), (10.0, 90.0), (20.0, 90.0)))
    assert window.rate_bps(9.5) == pytest.approx(10.0, rel=1e-15)
    assert window.rate_bps(10.0) == pytest.approx(90.0, rel=1e-15)


def test_capacity_with_a_step_is_integrated_per_segment() -> None:
    # Hand calculation: 10 bit/s for 10 s plus 90 bit/s for 10 s = 1000 bits.
    # This is the regression case for the defect that validate_capacity.py found:
    # reading the left segment's upper endpoint through rate_bps returns the
    # step's higher value and over-integrates by 800 bits, a 6 per cent error on
    # this window and 6.065e-02 relative on the irregular profile in that script.
    window = ContactWindow("w", 0.0, 20.0, ((0.0, 10.0), (10.0, 10.0), (10.0, 90.0), (20.0, 90.0)))
    assert window.capacity_bits() == pytest.approx(1000.0, rel=1e-15)


def test_capacity_agrees_with_adaptive_quadrature() -> None:
    window = ContactWindow(
        "w", 0.0, 480.0, rate_profile_from_elevation(0.0, 480.0, 1.2e8, floor_fraction=0.22)
    )
    closed = window.capacity_bits()
    integrated, _error = quad(window.rate_bps, 0.0, 480.0, limit=500)
    # Agreement to within rounding, not exact equality: the trapezoidal sum and
    # the Gauss-Kronrod rule accumulate different rounding over different terms.
    assert integrated == pytest.approx(closed, rel=1.0e-7)


def test_capacity_is_additive_over_a_split() -> None:
    window = ContactWindow(
        "w", 0.0, 480.0, rate_profile_from_elevation(0.0, 480.0, 1.2e8, points=13)
    )
    whole = window.delivered_bits(0.0, 480.0)
    parts = window.delivered_bits(0.0, 137.0) + window.delivered_bits(137.0, 480.0)
    assert parts == pytest.approx(whole, rel=1e-12)


def test_delivered_bits_clamps_to_the_window() -> None:
    window = ContactWindow("w", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 10.0))
    assert window.delivered_bits(0.0, 1000.0) == pytest.approx(1000.0, rel=1e-15)
    assert window.delivered_bits(300.0, 400.0) == 0.0
    assert window.delivered_bits(150.0, 150.0) == 0.0


def test_time_for_bits_inverts_delivered_bits() -> None:
    window = ContactWindow(
        "w", 0.0, 300.0, rate_profile_from_elevation(0.0, 300.0, 5.0e7, points=9)
    )
    for fraction in (0.1, 0.5, 0.95, 1.0):
        target = fraction * window.capacity_bits()
        seconds = window.time_for_bits(target)
        assert window.delivered_bits(0.0, seconds) == pytest.approx(target, rel=1e-9, abs=1.0)


def test_time_for_bits_zero_and_negative() -> None:
    window = ContactWindow("w", 0.0, 10.0, constant_rate_profile(0.0, 10.0, 100.0))
    assert window.time_for_bits(0.0) == 0.0
    with pytest.raises(ValueError, match="bits must be non-negative"):
        window.time_for_bits(-1.0)


def test_time_for_bits_beyond_capacity_raises() -> None:
    window = ContactWindow("w", 0.0, 10.0, constant_rate_profile(0.0, 10.0, 100.0))
    with pytest.raises(ValueError, match="cannot deliver"):
        window.time_for_bits(2000.0)
    # Within the one-bit tolerance it is accepted rather than rejected.
    assert window.time_for_bits(1000.0 + 0.5 * CAPACITY_TOLERANCE_BITS) <= 10.0


def test_extension_adds_the_held_final_rate() -> None:
    window = ContactWindow("w", 0.0, 100.0, ((0.0, 0.0), (50.0, 200.0), (100.0, 40.0)))
    extended = window.extended(10.0)
    # Hand calculation: the extension adds 40 bit/s for 10 s = 400 bits.
    assert extended.capacity_bits() - window.capacity_bits() == pytest.approx(400.0, rel=1e-12)
    assert extended.end_s == pytest.approx(110.0, rel=1e-15)
    assert window.extended(0.0) is window
    with pytest.raises(ValueError, match="extra_s must be non-negative"):
        window.extended(-1.0)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"start_s": 200.0, "end_s": 100.0}, "end_s must exceed start_s"),
        ({"start_s": -1.0, "end_s": 10.0}, "start_s must be non-negative"),
        ({"reliability": 1.5}, "reliability must lie in"),
        ({"reliability": -0.1}, "reliability must lie in"),
    ],
)
def test_invalid_window_fields_raise(kwargs: dict[str, float], fragment: str) -> None:
    base: dict[str, object] = {
        "name": "w",
        "start_s": 0.0,
        "end_s": 100.0,
        "rate_breakpoints": ((0.0, 1.0), (100.0, 1.0)),
    }
    base.update(kwargs)
    if "start_s" in kwargs or "end_s" in kwargs:
        base["rate_breakpoints"] = (
            (float(base["start_s"]), 1.0),  # type: ignore[arg-type]
            (float(base["end_s"]), 1.0),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match=fragment):
        ContactWindow(**base)  # type: ignore[arg-type]


def test_breakpoints_must_span_the_window() -> None:
    with pytest.raises(ValueError, match="must span exactly"):
        ContactWindow("w", 0.0, 100.0, ((0.0, 1.0), (90.0, 1.0)))


def test_breakpoints_must_be_non_decreasing_in_time() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        ContactWindow("w", 0.0, 100.0, ((0.0, 1.0), (50.0, 1.0), (40.0, 1.0), (100.0, 1.0)))


def test_breakpoints_must_be_non_negative_in_rate() -> None:
    with pytest.raises(ValueError, match="non-negative rates"):
        ContactWindow("w", 0.0, 100.0, ((0.0, 1.0), (50.0, -1.0), (100.0, 1.0)))


def test_at_least_two_breakpoints() -> None:
    with pytest.raises(ValueError, match="at least two points"):
        ContactWindow("w", 0.0, 100.0, ((0.0, 1.0),))


def test_empty_name_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ContactWindow(" ", 0.0, 10.0, ((0.0, 1.0), (10.0, 1.0)))


def test_plan_sorts_by_start_and_reports_capacities(tiny_plan: ContactPlan) -> None:
    assert tiny_plan.names == ("w1", "w2")
    assert tiny_plan.capacity_bits() == {"w1": pytest.approx(2.0e7), "w2": pytest.approx(2.0e7)}
    assert tiny_plan.total_capacity_bits() == pytest.approx(4.0e7, rel=1e-15)
    assert tiny_plan.total_contact_s() == pytest.approx(200.0, rel=1e-15)
    assert len(tiny_plan) == 2
    assert "w1" in tiny_plan
    assert "absent" not in tiny_plan


def test_plan_rejects_overlap() -> None:
    windows = [
        ContactWindow("a", 0.0, 100.0, constant_rate_profile(0.0, 100.0, 1.0)),
        ContactWindow("b", 50.0, 150.0, constant_rate_profile(50.0, 150.0, 1.0)),
    ]
    with pytest.raises(ValueError, match="overlap in time"):
        ContactPlan(windows)


def test_plan_allows_abutting_windows() -> None:
    windows = [
        ContactWindow("a", 0.0, 100.0, constant_rate_profile(0.0, 100.0, 1.0)),
        ContactWindow("b", 100.0, 150.0, constant_rate_profile(100.0, 150.0, 1.0)),
    ]
    assert ContactPlan(windows).names == ("a", "b")


def test_plan_rejects_duplicate_name_and_empty() -> None:
    window = ContactWindow("a", 0.0, 100.0, constant_rate_profile(0.0, 100.0, 1.0))
    with pytest.raises(ValueError, match="duplicate window name"):
        ContactPlan([window, window])
    with pytest.raises(ValueError, match="at least one ContactWindow"):
        ContactPlan([])
    with pytest.raises(TypeError, match="not ContactWindow"):
        ContactPlan(["nope"])  # type: ignore[list-item]


def test_windows_ending_by(tiny_plan: ContactPlan) -> None:
    assert tiny_plan.windows_ending_by(199.0) == ()
    assert tiny_plan.windows_ending_by(200.0) == ("w1",)
    assert tiny_plan.windows_ending_by(700.0) == ("w1", "w2")


def test_extended_window_and_replace(tiny_plan: ContactPlan) -> None:
    extended = tiny_plan.extended_window("w1", 50.0)
    assert extended["w1"].end_s == pytest.approx(250.0, rel=1e-15)
    # Hand calculation: 2e5 bit/s for 50 extra s = 1e7 extra bits.
    assert extended["w1"].capacity_bits() == pytest.approx(3.0e7, rel=1e-12)
    assert tiny_plan["w1"].capacity_bits() == pytest.approx(2.0e7, rel=1e-15)
    with pytest.raises(KeyError, match="no window named"):
        tiny_plan.extended_window("absent", 1.0)
    with pytest.raises(KeyError, match="no window named"):
        tiny_plan["absent"]
    replaced = tiny_plan.replace_window(
        ContactWindow("w1", 100.0, 200.0, constant_rate_profile(100.0, 200.0, 4.0e5))
    )
    assert replaced["w1"].capacity_bits() == pytest.approx(4.0e7, rel=1e-15)
    with pytest.raises(KeyError, match="no window named"):
        tiny_plan.replace_window(
            ContactWindow("zz", 0.0, 1.0, constant_rate_profile(0.0, 1.0, 1.0))
        )


def test_extending_into_the_next_window_raises(tiny_plan: ContactPlan) -> None:
    with pytest.raises(ValueError, match="overlap in time"):
        tiny_plan.extended_window("w1", 1000.0)


def test_elevation_profile_shape_and_validation() -> None:
    points = rate_profile_from_elevation(0.0, 100.0, 1.0e6, floor_fraction=0.2, points=5)
    assert len(points) == 5
    assert points[0] == (0.0, pytest.approx(2.0e5, rel=1e-15))
    assert points[-1] == (100.0, pytest.approx(2.0e5, rel=1e-15))
    assert max(rate for _t, rate in points) == pytest.approx(1.0e6, rel=1e-15)
    assert np.all(np.diff([t for t, _r in points]) > 0.0)
    with pytest.raises(ValueError, match="points must be >= 3"):
        rate_profile_from_elevation(0.0, 10.0, 1.0, points=2)
    with pytest.raises(ValueError, match="floor_fraction must lie in"):
        rate_profile_from_elevation(0.0, 10.0, 1.0, floor_fraction=2.0)
    with pytest.raises(ValueError, match="peak_rate_bps must be positive"):
        rate_profile_from_elevation(0.0, 10.0, 0.0)
    with pytest.raises(ValueError, match="end_s must exceed start_s"):
        rate_profile_from_elevation(10.0, 10.0, 1.0)


def test_constant_profile_validation() -> None:
    with pytest.raises(ValueError, match="rate_bps must be non-negative"):
        constant_rate_profile(0.0, 10.0, -1.0)
    with pytest.raises(ValueError, match="end_s must exceed start_s"):
        constant_rate_profile(10.0, 0.0, 1.0)
