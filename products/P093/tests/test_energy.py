"""The energy integral, the active-window derivation, and the edge case."""

from __future__ import annotations

import numpy as np
import pytest

from powerenvelope.energy import (
    GAP_TOLERANCE_S,
    SAMPLE_PERIOD_S,
    active_windows,
    cell_energy,
    clip_series,
    energy_over_windows,
    interstitial_samples,
    merge_intervals,
    sleep_energy,
    trapezoid_energy_mj,
    whole_run_energy,
)

from fixtures import synthetic_campaign


def test_known_answer_constant_power():
    # 1000 mW held for 2.0 s is 2000 mJ. Hand calculation.
    t = np.asarray([0.0, 1.0, 2.0])
    p = np.asarray([1000.0, 1000.0, 1000.0])
    assert trapezoid_energy_mj(t, p) == pytest.approx(2000.0, rel=1e-14, abs=0.0)


def test_known_answer_linear_ramp():
    # A linear ramp from 0 to 100 mW over 4 s: the trapezoid is exact for a
    # straight line, so the energy is 0.5 * 4 * 100 = 200 mJ.
    t = np.asarray([0.0, 2.0, 4.0])
    p = np.asarray([0.0, 50.0, 100.0])
    assert trapezoid_energy_mj(t, p) == pytest.approx(200.0, rel=1e-14, abs=0.0)


def test_known_answer_single_step():
    # Two samples 0.05 s apart at 100 and 300 mW: the trapezoid averages them,
    # 0.05 * 200 = 10 mJ.
    assert trapezoid_energy_mj(
        np.asarray([0.0, 0.05]), np.asarray([100.0, 300.0])
    ) == pytest.approx(10.0, rel=1e-14, abs=0.0)


@pytest.mark.parametrize(
    ("t", "p", "match"),
    [
        (np.asarray([0.0]), np.asarray([1.0]), "at least two samples"),
        (np.asarray([0.0, 1.0]), np.asarray([1.0]), "samples and power_mw has"),
        (np.asarray([1.0, 0.0]), np.asarray([1.0, 1.0]), "non-decreasing"),
        (np.asarray([0.0, np.nan]), np.asarray([1.0, 1.0]), "must be finite"),
    ],
)
def test_integral_rejects_bad_input(t, p, match):
    with pytest.raises(ValueError, match=match):
        trapezoid_energy_mj(t, p)


def test_integral_rejects_two_dimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        trapezoid_energy_mj(np.zeros((2, 2)), np.zeros((2, 2)))


def test_merge_absorbs_a_gap_at_the_tolerance():
    lo, hi = merge_intervals(
        np.asarray([0.0, 1.0 + GAP_TOLERANCE_S]), np.asarray([1.0, 2.0])
    )
    assert lo.size == 1
    assert hi[0] == pytest.approx(2.0, rel=0.0, abs=0.0)


def test_merge_splits_a_gap_above_the_tolerance():
    lo, _hi = merge_intervals(
        np.asarray([0.0, 1.0 + 2.0 * GAP_TOLERANCE_S]), np.asarray([1.0, 2.0])
    )
    assert lo.size == 2


def test_merge_sorts_unsorted_input():
    lo, hi = merge_intervals(np.asarray([5.0, 0.0]), np.asarray([6.0, 1.0]))
    assert lo.tolist() == [0.0, 5.0]
    assert hi.tolist() == [1.0, 6.0]


def test_merge_of_nothing_is_nothing():
    lo, hi = merge_intervals(np.asarray([]), np.asarray([]))
    assert lo.size == 0 and hi.size == 0


def test_merge_rejects_an_inverted_interval():
    with pytest.raises(ValueError, match="end >= start"):
        merge_intervals(np.asarray([1.0]), np.asarray([0.0]))


def test_merge_rejects_a_negative_tolerance():
    with pytest.raises(ValueError, match="non-negative"):
        merge_intervals(np.asarray([0.0]), np.asarray([1.0]), -1.0)


def test_clip_interpolates_both_endpoints():
    t = np.asarray([0.0, 1.0, 2.0])
    p = np.asarray([0.0, 100.0, 200.0])
    ct, cp = clip_series(t, p, 0.5, 1.5)
    assert ct.tolist() == [0.5, 1.0, 1.5]
    assert cp == pytest.approx([50.0, 100.0, 150.0], rel=1e-14, abs=0.0)


def test_clip_truncates_to_the_measured_span():
    t = np.asarray([10.0, 11.0])
    p = np.asarray([1.0, 2.0])
    ct, _cp = clip_series(t, p, 0.0, 100.0)
    assert ct.tolist() == [10.0, 11.0]


def test_clip_of_an_empty_intersection_is_empty():
    ct, cp = clip_series(np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0]), 5.0, 6.0)
    assert ct.size == 0 and cp.size == 0


def test_active_windows_are_one_per_cycle_at_half_duty(reference_cell):
    lo, hi = active_windows(reference_cell.require_decisions())
    assert lo.size == reference_cell.cycles
    assert np.allclose(hi - lo, 1.0, rtol=0.0, atol=1e-9)


def test_active_windows_merge_into_one_at_full_duty(campaign):
    cell = campaign.require("w64_d100")
    lo, _hi = active_windows(cell.require_decisions())
    assert lo.size == 1


def test_active_plus_sleep_equals_whole_run(reference_cell):
    whole, active = cell_energy(reference_cell)
    sleep = sleep_energy(reference_cell)
    assert active.energy_mj + sleep.energy_mj == pytest.approx(
        whole.energy_mj, rel=1e-12, abs=0.0
    )
    assert active.duration_s + sleep.duration_s == pytest.approx(
        whole.duration_s, rel=1e-12, abs=0.0
    )


def test_whole_run_energy_matches_the_closed_form(campaign):
    for label, duty in (("w64_d025", 0.25), ("w64_d050", 0.50), ("w96_d075", 0.75)):
        cell = campaign.require(label)
        expected = synthetic_campaign.total_energy_mj(duty, cell.cycles)
        assert whole_run_energy(cell.power).energy_mj == pytest.approx(
            expected, rel=synthetic_campaign.CLOSED_FORM_REL, abs=0.0
        )


def test_active_energy_matches_the_closed_form(campaign):
    for label, duty in (("w48_d010", 0.10), ("w64_d050", 0.50), ("w96_d075", 0.75)):
        cell = campaign.require(label)
        expected = synthetic_campaign.active_energy_mj(duty, cell.cycles)
        _whole, active = cell_energy(cell)
        assert active.energy_mj == pytest.approx(
            expected, rel=synthetic_campaign.CLOSED_FORM_REL, abs=0.0
        )


def test_at_full_duty_active_energy_is_the_whole_run(campaign):
    cell = campaign.require("w48_d100")
    whole, active = cell_energy(cell)
    assert active.energy_mj == pytest.approx(whole.energy_mj, rel=1e-12, abs=0.0)


def test_time_weighted_mean_is_energy_over_duration(reference_cell):
    _whole, active = cell_energy(reference_cell)
    assert active.mean_power_mw == pytest.approx(
        active.energy_mj / active.duration_s, rel=1e-14, abs=0.0
    )


def test_sample_mean_and_time_weighted_mean_differ_and_both_are_reported(reference_cell):
    _whole, active = cell_energy(reference_cell)
    # They differ here because the right edge of each active window lands on a
    # sample that already reads the floor. Both are reported; neither is hidden.
    assert active.sample_mean_power_mw < active.mean_power_mw


def test_interstitial_count_is_zero_on_contiguous_inferences(reference_cell):
    # The synthetic generator tiles each active phase exactly, so there is no
    # gap between consecutive inferences at all.
    result = interstitial_samples(reference_cell)
    assert result.n_samples == 0
    assert result.n_gaps == 0
    assert result.max_gap_s == pytest.approx(0.0, rel=0.0, abs=1e-12)


def test_interstitial_fires_when_a_gap_is_opened(tmp_path):
    campaign = synthetic_campaign.campaign(tmp_path / "gapped")
    cell = campaign.require("w64_d050")
    decisions = cell.require_decisions()
    # Anchor on the TRUE structure: drop the inferences covering the middle
    # 0.3 s of each cycle's active phase, which is six sampler periods and
    # therefore far more than the merge tolerance of one half period.
    phase = (decisions.t_start - decisions.t_start[0]) % 2.0
    keep = ~((phase > 0.30) & (phase < 0.60))
    lo, hi = merge_intervals(decisions.t_start[keep], decisions.t_end[keep])
    window = energy_over_windows(cell.power, lo, hi)
    assert lo.size > cell.cycles
    assert window.n_between_windows > 0


def test_energy_over_windows_rejects_a_column_that_is_not_power(reference_cell):
    with pytest.raises(ValueError, match="not a power array"):
        energy_over_windows(
            reference_cell.power, np.asarray([0.0]), np.asarray([1.0]), column="label"
        )


def test_whole_run_energy_rejects_a_one_sample_trace(reference_cell):
    class Stub:
        t = np.asarray([0.0])

    with pytest.raises(ValueError, match="fewer than two rail samples"):
        whole_run_energy(Stub())


def test_sample_period_constant_matches_the_campaign():
    assert SAMPLE_PERIOD_S == pytest.approx(0.05, rel=0.0, abs=0.0)
    assert GAP_TOLERANCE_S == pytest.approx(0.025, rel=0.0, abs=0.0)
