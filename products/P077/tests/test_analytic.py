"""Baseline 1: the closed form, against hand calculations and exact identities.

Several tests here check an *independent* closed form rather than a number:
``E[R]`` for small ``n`` has an inclusion-exclusion expression that does not
share any code with the series in :func:`expected_rounds_detail`, and agreement
between the two is stronger evidence than a remembered constant.
"""

from __future__ import annotations

import math

import pytest

from relfiledelivery.analytic import (
    analytic_delivery,
    analytic_delivery_time_quantile,
    clean_pass_seconds,
    expected_rounds,
    expected_rounds_detail,
    expected_total_transmissions,
    expected_transmissions_per_pdu,
    extended_expected_delivery_time,
    require_closed_form_applicable,
    rounds_cdf,
    rounds_quantile,
    transmissions_quantile,
    turnaround_seconds,
)
from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows
from relfiledelivery.modes import TransferMode
from relfiledelivery.payload import FileTransferSpec


def _expected_rounds_by_inclusion_exclusion(n: int, p: float) -> float:
    """``E[max(G_1..G_n)]`` by inclusion-exclusion over minima.

    The minimum of ``k`` independent geometric variables with failure
    probability ``p`` is geometric with failure probability ``p**k``, so

        E[max] = sum_{k=1}^{n} (-1)^(k+1) C(n, k) / (1 - p**k).

    Independent of the series implementation, and exact in exact arithmetic.
    Used only for small ``n``, where the alternating binomial sum does not lose
    catastrophic precision.
    """
    return sum(
        (-1) ** (k + 1) * math.comb(n, k) / (1.0 - p**k) for k in range(1, n + 1)
    )


class TestTransmissionCounts:
    def test_hand_calculated_attempts_per_unit(self) -> None:
        # 1 / (1 - 0.5) = 2 exactly; 1 / (1 - 0.2) = 1.25 exactly.
        assert expected_transmissions_per_pdu(0.5) == pytest.approx(2.0)
        assert expected_transmissions_per_pdu(0.2) == pytest.approx(1.25)
        assert expected_transmissions_per_pdu(0.0) == pytest.approx(1.0)

    def test_total_is_n_times_per_unit(self) -> None:
        assert expected_total_transmissions(40, 0.2) == pytest.approx(50.0)

    @pytest.mark.parametrize("p", [-0.1, 1.0, 1.5])
    def test_out_of_range_probability_is_rejected(self, p: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            expected_transmissions_per_pdu(p)

    @pytest.mark.parametrize("n", [0, -3])
    def test_non_positive_count_is_rejected(self, n: int) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            expected_total_transmissions(n, 0.1)


class TestRounds:
    def test_cdf_is_the_declared_closed_form(self) -> None:
        # P(R <= 1) is the probability that no unit was lost on the first try,
        # which is (1 - p)**n. At n = 4, p = 0.5 that is 0.0625.
        assert rounds_cdf(4, 0.5, 1) == pytest.approx(0.0625)
        assert rounds_cdf(4, 0.5, 0) == pytest.approx(0.0)
        assert rounds_cdf(4, 0.0, 0) == pytest.approx(1.0)

    def test_cdf_rejects_a_negative_round_count(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            rounds_cdf(4, 0.5, -1)

    def test_single_unit_reduces_to_the_geometric_mean(self) -> None:
        # For n = 1 the max is the single geometric variable, so
        # E[R] = 1 / (1 - p).
        for p in (0.0, 0.1, 0.37, 0.8):
            assert expected_rounds(1, p) == pytest.approx(1.0 / (1.0 - p), rel=1e-12)

    def test_two_units_matches_the_hand_derivation(self) -> None:
        # E[max(G1, G2)] = E[G1] + E[G2] - E[min(G1, G2)] and the minimum of
        # two geometrics with failure probability p is geometric with failure
        # probability p**2, so E[R] = 2 / (1 - p) - 1 / (1 - p**2).
        # At p = 0.5 that is 4 - 4/3 = 8/3 = 2.6666666666666665.
        assert expected_rounds(2, 0.5) == pytest.approx(8.0 / 3.0, rel=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    @pytest.mark.parametrize("p", [0.05, 0.2, 0.5, 0.75])
    def test_series_matches_inclusion_exclusion(self, n: int, p: float) -> None:
        assert expected_rounds(n, p) == pytest.approx(
            _expected_rounds_by_inclusion_exclusion(n, p), rel=1e-10
        )

    def test_zero_loss_needs_exactly_one_round(self) -> None:
        detail = expected_rounds_detail(500, 0.0)
        assert detail.value == pytest.approx(1.0)
        assert detail.tail_bound == pytest.approx(0.0)

    def test_truncation_bound_is_reported_and_respected(self) -> None:
        detail = expected_rounds_detail(100, 0.3, tolerance=1e-10)
        assert detail.tail_bound <= 1e-10
        assert detail.terms > 1
        tighter = expected_rounds_detail(100, 0.3, tolerance=1e-15)
        assert tighter.terms >= detail.terms
        assert tighter.value == pytest.approx(detail.value, abs=1e-9)

    def test_non_positive_tolerance_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="tolerance must be positive"):
            expected_rounds_detail(10, 0.2, tolerance=0.0)

    def test_rounds_quantile_inverts_the_cdf(self) -> None:
        n, p = 200, 0.2
        for q in (0.1, 0.5, 0.9, 0.99):
            k = rounds_quantile(n, p, q)
            assert rounds_cdf(n, p, k) >= q
            assert k == 1 or rounds_cdf(n, p, k - 1) < q

    def test_rounds_quantile_is_one_without_loss(self) -> None:
        assert rounds_quantile(100, 0.0, 0.99) == 1

    @pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.2])
    def test_rounds_quantile_rejects_an_out_of_range_level(self, q: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            rounds_quantile(10, 0.2, q)


class TestTransmissionsQuantile:
    def test_lossless_quantile_is_exactly_the_unit_count(self) -> None:
        assert transmissions_quantile(37, 0.0, 0.95) == pytest.approx(37.0)

    def test_quantile_is_at_least_the_unit_count(self) -> None:
        assert transmissions_quantile(37, 0.3, 0.01) >= 37.0

    def test_quantile_is_monotone_in_the_level(self) -> None:
        values = [transmissions_quantile(80, 0.2, q) for q in (0.05, 0.5, 0.95)]
        assert values[0] <= values[1] <= values[2]

    @pytest.mark.parametrize("q", [0.0, 1.0])
    def test_out_of_range_level_is_rejected(self, q: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            transmissions_quantile(10, 0.2, q)


class TestApplicability:
    def test_outages_are_refused(self, tiny_spec: FileTransferSpec) -> None:
        link = LinkConditions(1000.0, 100.0, outage=OutageModel(10.0, 2.0))
        with pytest.raises(ValueError, match="assumes no outages"):
            require_closed_form_applicable(link)
        with pytest.raises(ValueError, match="assumes no outages"):
            analytic_delivery(tiny_spec, link, TransferMode.DEFERRED_NAK)

    def test_contact_windows_are_refused(self, tiny_spec: FileTransferSpec) -> None:
        link = LinkConditions(1000.0, 100.0, windows=uniform_windows(2, 10.0, 100.0))
        with pytest.raises(ValueError, match="continuous contact"):
            analytic_delivery(tiny_spec, link, TransferMode.DEFERRED_NAK)

    def test_variable_size_nak_is_refused(self, tiny_spec: FileTransferSpec) -> None:
        link = LinkConditions(1000.0, 100.0, nak_bytes_per_segment=4.0)
        with pytest.raises(ValueError, match="fixed-size NAK"):
            analytic_delivery(tiny_spec, link, TransferMode.DEFERRED_NAK)


class TestClosedFormDelivery:
    def test_clean_pass_seconds_is_hand_calculable(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        # 1070 wire bytes at 1000 bit/s is 8 * 1070 / 1000 = 8.56 s.
        assert clean_pass_seconds(tiny_spec, clean_link) == pytest.approx(8.56)

    def test_turnaround_is_eof_plus_nak_plus_two_light_times(
        self, clean_link: LinkConditions
    ) -> None:
        # 0.128 + 5.12 + 2 * 0.5 = 6.248 s.
        assert turnaround_seconds(clean_link) == pytest.approx(6.248)

    def test_lossless_deferred_nak_costs_one_pass_plus_light_time(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(tiny_spec, clean_link, TransferMode.DEFERRED_NAK)
        assert prediction.expected_rounds == pytest.approx(1.0)
        assert prediction.expected_transmissions == pytest.approx(10.0)
        assert prediction.expected_delivery_time_s == pytest.approx(8.56 + 0.5)
        assert prediction.expected_return_wire_bytes == pytest.approx(0.0)
        assert prediction.completion_probability == pytest.approx(1.0)

    def test_lossy_deferred_nak_matches_the_hand_assembled_expression(
        self, tiny_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(tiny_spec, lossy_link, TransferMode.DEFERRED_NAK)
        expected_r = _expected_rounds_by_inclusion_exclusion(10, 0.2)
        hand = 8.56 / 0.8 + (expected_r - 1.0) * 6.248 + 0.5
        assert prediction.expected_delivery_time_s == pytest.approx(hand, rel=1e-10)
        assert prediction.busy_seconds == pytest.approx(10.7)
        assert prediction.confirmation_time_s == pytest.approx(hand + 6.248, rel=1e-10)

    def test_immediate_nak_has_no_closed_form_delivery_time_only_a_bound(
        self, tiny_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(tiny_spec, lossy_link, TransferMode.IMMEDIATE_NAK)
        assert prediction.expected_delivery_time_s is None
        assert prediction.confirmation_time_s is None
        assert prediction.delivery_time_lower_bound_s == pytest.approx(10.7 + 0.5)
        # The transmission count is mode-independent in expectation.
        assert prediction.expected_transmissions == pytest.approx(12.5)

    def test_unacknowledged_completion_probability(
        self, tiny_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(tiny_spec, lossy_link, TransferMode.UNACKNOWLEDGED)
        # (1 - 0.2)**10 = 0.1073741824 exactly.
        assert prediction.completion_probability == pytest.approx(0.8**10, rel=1e-12)
        assert prediction.expected_transmissions == pytest.approx(10.0)
        assert prediction.expected_return_wire_bytes == pytest.approx(0.0)
        assert prediction.expected_delivery_time_s == pytest.approx(8.56 + 0.5)

    def test_overhead_ratio_exceeds_the_lossless_floor(
        self, tiny_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        prediction = analytic_delivery(tiny_spec, lossy_link, TransferMode.DEFERRED_NAK)
        assert prediction.expected_overhead_ratio > tiny_spec.minimum_overhead_ratio

    def test_describe_reports_the_tail_bound(
        self, tiny_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        lines = analytic_delivery(tiny_spec, lossy_link, TransferMode.DEFERRED_NAK).describe()
        assert any("series tail bound" in line for line in lines)


class TestSemiAnalyticQuantile:
    def test_quantile_is_monotone_in_the_level(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        values = [
            analytic_delivery_time_quantile(medium_spec, lossy_link, q)
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        ]
        assert values == sorted(values)

    def test_quantile_brackets_the_lossless_delivery_time(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        # With no loss the distribution is degenerate, so every quantile is the
        # single attainable value.
        for q in (0.05, 0.5, 0.95):
            assert analytic_delivery_time_quantile(tiny_spec, clean_link, q) == pytest.approx(
                8.56 + 0.5
            )


class TestExtension:
    def test_extension_reduces_to_the_closed_form_at_unit_availability(
        self, tiny_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        exact = analytic_delivery(
            tiny_spec, lossy_link, TransferMode.DEFERRED_NAK
        ).expected_delivery_time_s
        extended = extended_expected_delivery_time(
            tiny_spec, lossy_link, TransferMode.DEFERRED_NAK
        )
        assert extended == pytest.approx(exact, rel=1e-12)

    def test_extension_inflates_by_one_over_the_outage_duty_on_a_continuous_link(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        from relfiledelivery.link import OutageModel

        continuous = LinkConditions(
            1000.0, 100.0, 0.5, 0.2, outage=OutageModel(mean_up_s=30.0, mean_down_s=10.0)
        )
        plain = analytic_delivery(
            tiny_spec, continuous.without_outages(), TransferMode.DEFERRED_NAK
        ).expected_delivery_time_s
        extended = extended_expected_delivery_time(
            tiny_spec, continuous, TransferMode.DEFERRED_NAK
        )
        # Duty cycle 30 / 40 = 0.75, so the extension is plain / 0.75.
        assert extended == pytest.approx(plain / 0.75, rel=1e-12)
        assert extended > plain

    def test_extension_returns_the_contact_time_when_it_fits_in_the_first_window(
        self, tiny_spec: FileTransferSpec, outage_link: LinkConditions
    ) -> None:
        """A transfer that fits in one pass must not be inflated by the gaps."""
        plain = analytic_delivery(
            tiny_spec, outage_link.without_outages(), TransferMode.DEFERRED_NAK
        ).expected_delivery_time_s
        needed = plain / outage_link.outage.duty_cycle
        extended = extended_expected_delivery_time(
            tiny_spec, outage_link, TransferMode.DEFERRED_NAK
        )
        assert needed <= outage_link.windows[0].duration_s
        assert extended == pytest.approx(needed, rel=1e-12)
        # The availability-only extension would have been larger by the ratio of
        # the duty cycles; this is the straw-man figure the extension avoids.
        assert plain / outage_link.availability > extended

    def test_extension_spreads_over_windows_when_one_pass_is_not_enough(
        self, outage_link: LinkConditions
    ) -> None:
        big = FileTransferSpec(file_bytes=12_000, pdu_payload_bytes=100, pdu_header_bytes=12)
        plain = analytic_delivery(
            big, outage_link.without_outages(), TransferMode.DEFERRED_NAK
        ).expected_delivery_time_s
        needed = plain / outage_link.outage.duty_cycle
        extended = extended_expected_delivery_time(
            big, outage_link, TransferMode.DEFERRED_NAK
        )
        assert needed > outage_link.windows[0].duration_s
        # More wall clock than contact time, because the gaps now matter.
        assert extended > needed

    def test_extension_extrapolates_past_an_insufficient_contact_plan(
        self, outage_link: LinkConditions
    ) -> None:
        huge = FileTransferSpec(file_bytes=400_000, pdu_payload_bytes=100, pdu_header_bytes=12)
        span = outage_link.windows[-1].end_s - outage_link.windows[0].start_s
        extended = extended_expected_delivery_time(
            huge, outage_link, TransferMode.DEFERRED_NAK
        )
        assert extended > span

    def test_immediate_nak_uses_the_deferred_expression(
        self, tiny_spec: FileTransferSpec, outage_link: LinkConditions
    ) -> None:
        deferred = extended_expected_delivery_time(
            tiny_spec, outage_link, TransferMode.DEFERRED_NAK
        )
        immediate = extended_expected_delivery_time(
            tiny_spec, outage_link, TransferMode.IMMEDIATE_NAK
        )
        assert immediate == pytest.approx(deferred, rel=1e-12)
