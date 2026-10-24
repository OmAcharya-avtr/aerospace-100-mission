"""Declared link conditions, outage traces and the contact schedule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from relfiledelivery.link import (
    ContactSchedule,
    ContactWindow,
    LinkConditions,
    OutageModel,
    OutageTrace,
    uniform_windows,
)


class TestContactWindow:
    def test_duration_and_containment(self) -> None:
        window = ContactWindow(100.0, 160.0)
        assert window.duration_s == pytest.approx(60.0)
        assert window.contains(100.0)
        assert window.contains(159.999)
        assert not window.contains(160.0)
        assert not window.contains(99.0)

    @pytest.mark.parametrize(("start", "end"), [(10.0, 10.0), (10.0, 5.0)])
    def test_non_positive_duration_is_rejected(self, start: float, end: float) -> None:
        with pytest.raises(ValueError, match="positive duration"):
            ContactWindow(start, end)

    def test_infinite_bound_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="finite seconds"):
            ContactWindow(0.0, math.inf)


class TestOutageModel:
    def test_no_outage_limit(self) -> None:
        model = OutageModel.none()
        assert model.is_outage_free
        assert model.duty_cycle == pytest.approx(1.0)

    def test_duty_cycle_is_the_stationary_up_probability(self) -> None:
        # Hand calculation: mean up 30 s, mean down 10 s, so the link is up
        # 30 / 40 = 0.75 of the time.
        model = OutageModel(mean_up_s=30.0, mean_down_s=10.0)
        assert model.duty_cycle == pytest.approx(0.75)

    def test_zero_mean_down_is_the_no_outage_limit(self) -> None:
        assert OutageModel(mean_up_s=5.0, mean_down_s=0.0).is_outage_free

    @pytest.mark.parametrize("mean_up", [0.0, -1.0])
    def test_non_positive_mean_up_is_rejected(self, mean_up: float) -> None:
        with pytest.raises(ValueError, match="mean_up_s must be positive"):
            OutageModel(mean_up_s=mean_up, mean_down_s=1.0)

    def test_negative_mean_down_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            OutageModel(mean_up_s=1.0, mean_down_s=-1.0)

    def test_infinite_mean_down_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="never completes"):
            OutageModel(mean_up_s=1.0, mean_down_s=math.inf)


class TestLinkConditions:
    def test_asymmetry_and_availability(self, outage_link: LinkConditions) -> None:
        # Hand calculation: 1000 / 100 = 10 asymmetry; outage duty
        # 12 / 18 = 2/3; four 40 s windows every 100 s spans 340 s of which
        # 160 s is in contact, so contact duty is 160 / 340.
        assert outage_link.asymmetry == pytest.approx(10.0)
        assert outage_link.outage.duty_cycle == pytest.approx(2.0 / 3.0)
        assert outage_link.contact_duty_cycle == pytest.approx(160.0 / 340.0)
        assert outage_link.availability == pytest.approx((2.0 / 3.0) * (160.0 / 340.0))

    def test_continuous_link_has_unit_contact_duty(self, clean_link: LinkConditions) -> None:
        assert clean_link.is_continuous
        assert clean_link.contact_duty_cycle == pytest.approx(1.0)
        assert clean_link.availability == pytest.approx(1.0)

    def test_without_outages_clears_both_outage_and_windows(
        self, outage_link: LinkConditions
    ) -> None:
        plain = outage_link.without_outages()
        assert plain.outage.is_outage_free
        assert plain.is_continuous
        assert plain.pdu_loss_probability == pytest.approx(outage_link.pdu_loss_probability)

    def test_loss_probability_of_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            LinkConditions(1000.0, 100.0, pdu_loss_probability=1.0)

    @pytest.mark.parametrize("rate", [0.0, -5.0, math.inf])
    def test_non_positive_or_infinite_rates_are_rejected(self, rate: float) -> None:
        with pytest.raises(ValueError, match="forward_rate_bps"):
            LinkConditions(rate, 100.0)
        with pytest.raises(ValueError, match="return_rate_bps"):
            LinkConditions(1000.0, rate)

    def test_negative_light_time_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="one_way_light_time_s"):
            LinkConditions(1000.0, 100.0, one_way_light_time_s=-0.1)

    def test_overlapping_windows_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-overlapping"):
            LinkConditions(
                1000.0,
                100.0,
                windows=(ContactWindow(0.0, 100.0), ContactWindow(50.0, 150.0)),
            )

    def test_outage_must_be_an_outage_model(self) -> None:
        with pytest.raises(TypeError, match="OutageModel"):
            LinkConditions(1000.0, 100.0, outage="none")  # type: ignore[arg-type]

    def test_nak_and_eof_times(self, clean_link: LinkConditions) -> None:
        # Hand calculation: a 64-byte NAK on a 100 bit/s return link is
        # 8 * 64 / 100 = 5.12 s; a 16-byte EOF on a 1000 bit/s forward link is
        # 8 * 16 / 1000 = 0.128 s.
        assert clean_link.nak_seconds(0) == pytest.approx(5.12)
        assert clean_link.eof_seconds == pytest.approx(0.128)

    def test_nak_grows_with_segments_when_declared(self) -> None:
        link = LinkConditions(1000.0, 100.0, nak_bytes=64, nak_bytes_per_segment=4.0)
        # 64 + 4 * 10 = 104 bytes, 8 * 104 / 100 = 8.32 s.
        assert link.nak_seconds(10) == pytest.approx(8.32)

    def test_negative_segment_count_is_rejected(self, clean_link: LinkConditions) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            clean_link.nak_seconds(-1)

    def test_describe_names_units_on_every_line(self, outage_link: LinkConditions) -> None:
        lines = outage_link.describe()
        assert any("bit/s" in line for line in lines)
        assert any("mean up" in line for line in lines)
        assert any("availability" in line for line in lines)


class TestUniformWindows:
    def test_count_duration_and_period(self) -> None:
        windows = uniform_windows(3, 50.0, 200.0, first_start_s=10.0)
        assert [w.start_s for w in windows] == [10.0, 210.0, 410.0]
        assert all(w.duration_s == pytest.approx(50.0) for w in windows)

    def test_period_shorter_than_duration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="windows would overlap"):
            uniform_windows(3, 100.0, 50.0)

    @pytest.mark.parametrize(("count", "duration"), [(0, 10.0), (3, 0.0)])
    def test_invalid_arguments_are_rejected(self, count: int, duration: float) -> None:
        with pytest.raises(ValueError):
            uniform_windows(count, duration, 100.0)


class TestOutageTrace:
    def test_no_outage_model_never_goes_down(self) -> None:
        trace = OutageTrace(OutageModel.none(), np.random.default_rng(0))
        assert not trace.is_down(0.0)
        assert not trace.intersects_outage(0.0, 1e6)
        assert trace.outage_seconds_in(0.0, 1e6) == pytest.approx(0.0)

    def test_equal_seeds_give_identical_traces(self) -> None:
        model = OutageModel(mean_up_s=3.0, mean_down_s=2.0)
        first = OutageTrace(model, np.random.default_rng(99))
        second = OutageTrace(model, np.random.default_rng(99))
        times = np.linspace(0.0, 200.0, 401)
        assert [first.is_down(t) for t in times] == [second.is_down(t) for t in times]

    def test_measured_duty_cycle_approaches_the_declared_one(self) -> None:
        model = OutageModel(mean_up_s=4.0, mean_down_s=1.0)
        trace = OutageTrace(model, np.random.default_rng(7))
        span = 40_000.0
        down = trace.outage_seconds_in(0.0, span)
        measured_up = 1.0 - down / span
        # Declared duty 0.8. A 40 000 s span contains roughly 8000 cycles, so
        # the sample standard error of the up fraction is of order 0.005.
        assert measured_up == pytest.approx(model.duty_cycle, abs=0.02)

    def test_intersecting_interval_is_lost_even_when_both_ends_are_up(self) -> None:
        model = OutageModel(mean_up_s=1.0, mean_down_s=1.0)
        trace = OutageTrace(model, np.random.default_rng(3))
        # Over a long span with a 50 per cent duty cycle, some interval of
        # length 10 s must intersect an outage.
        assert trace.intersects_outage(0.0, 10.0)

    def test_outage_seconds_in_is_zero_on_an_empty_interval(self) -> None:
        model = OutageModel(mean_up_s=1.0, mean_down_s=1.0)
        trace = OutageTrace(model, np.random.default_rng(3))
        assert trace.outage_seconds_in(5.0, 5.0) == pytest.approx(0.0)


class TestContactSchedule:
    def test_continuous_schedule_places_anything_immediately(self) -> None:
        schedule = ContactSchedule(())
        assert schedule.is_continuous
        assert schedule.horizon_s == math.inf
        start, end, index = schedule.place(7.0, 3.0)
        assert (start, end, index) == (7.0, 10.0, -1)

    def test_a_pdu_that_does_not_fit_starts_in_the_next_window(self) -> None:
        schedule = ContactSchedule(uniform_windows(2, 10.0, 100.0))
        # 6 s of transmission requested at t = 7 cannot finish by t = 10, so it
        # starts at the beginning of the second window at t = 100.
        start, end, index = schedule.place(7.0, 6.0)
        assert start == pytest.approx(100.0)
        assert end == pytest.approx(106.0)
        assert index == 1

    def test_place_returns_none_when_no_window_can_carry_it(self) -> None:
        schedule = ContactSchedule(uniform_windows(2, 10.0, 100.0))
        assert schedule.place(0.0, 20.0) is None

    def test_place_rejects_a_negative_duration(self) -> None:
        schedule = ContactSchedule(())
        with pytest.raises(ValueError, match="non-negative"):
            schedule.place(0.0, -1.0)

    def test_window_index_at(self) -> None:
        schedule = ContactSchedule(uniform_windows(2, 10.0, 100.0))
        assert schedule.window_index_at(5.0) == 0
        assert schedule.window_index_at(105.0) == 1
        with pytest.raises(ValueError, match="outside every declared"):
            schedule.window_index_at(50.0)

    def test_start_time_is_the_first_window_start(self) -> None:
        schedule = ContactSchedule(uniform_windows(2, 10.0, 100.0, first_start_s=33.0))
        assert schedule.start_time_s() == pytest.approx(33.0)
        assert ContactSchedule(()).start_time_s() == pytest.approx(0.0)
