"""The simulator: semantics, accounting, reproducibility and the lossless limit."""

from __future__ import annotations

import pytest

from relfiledelivery.analytic import analytic_delivery, clean_pass_seconds
from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows
from relfiledelivery.modes import ALL_MODES, TransferMode
from relfiledelivery.payload import FileTransferSpec
from relfiledelivery.simulate import (
    accounting_discrepancy,
    default_transmission_cap,
    expected_wire_bytes_from_attempts,
    independent_transmission_count,
    is_outage_free_continuous,
    simulate_transfer,
)


class TestLosslessLimit:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_lossless_transfer_is_one_pass_in_every_mode(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions, mode: TransferMode
    ) -> None:
        result = simulate_transfer(tiny_spec, clean_link, mode, 1)
        assert result.delivered
        assert result.forward_transmissions == 10
        assert result.retransmissions == 0
        assert result.nak_count == 0
        # 8.56 s of transmission plus 0.5 s of propagation.
        assert result.delivery_time_s == pytest.approx(8.56 + 0.5, rel=1e-12)
        assert result.idle_s == pytest.approx(0.5, rel=1e-9)

    @pytest.mark.parametrize("mode", [TransferMode.DEFERRED_NAK, TransferMode.IMMEDIATE_NAK])
    def test_lossless_delivery_matches_the_closed_form_exactly(
        self, medium_spec: FileTransferSpec, clean_link: LinkConditions, mode: TransferMode
    ) -> None:
        result = simulate_transfer(medium_spec, clean_link, mode, 5)
        reference = clean_pass_seconds(medium_spec, clean_link) + clean_link.one_way_light_time_s
        assert result.delivery_time_s == pytest.approx(reference, rel=1e-12)

    def test_confirmation_costs_one_more_turnaround(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        result = simulate_transfer(tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 1)
        assert result.confirmation_time_s == pytest.approx(
            result.delivery_time_s + 6.248, rel=1e-12
        )

    def test_unacknowledged_reports_no_confirmation_time(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            tiny_spec, clean_link, TransferMode.UNACKNOWLEDGED, 1
        )
        assert result.confirmation_time_s is None


class TestLossyBehaviour:
    def test_unacknowledged_can_fail_and_reports_no_delivery_time(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, lossy_link, TransferMode.UNACKNOWLEDGED, 11
        )
        assert not result.delivered
        assert result.delivery_time_s is None
        assert result.idle_s is None
        assert result.delivered_digest is None
        assert result.received_count < result.pdu_count

    @pytest.mark.parametrize("mode", [TransferMode.DEFERRED_NAK, TransferMode.IMMEDIATE_NAK])
    def test_retransmitting_modes_complete_on_a_lossy_continuous_link(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions, mode: TransferMode
    ) -> None:
        result = simulate_transfer(medium_spec, lossy_link, mode, 13)
        assert result.delivered
        assert result.forward_transmissions > result.pdu_count
        assert result.retransmissions == result.forward_transmissions - result.pdu_count

    def test_deferred_nak_sends_one_nak_per_round_boundary(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 17
        )
        assert result.nak_count == result.rounds - 1

    def test_immediate_nak_sends_more_naks_than_deferred(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        deferred = simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 19
        )
        immediate = simulate_transfer(
            medium_spec, lossy_link, TransferMode.IMMEDIATE_NAK, 19
        )
        assert immediate.nak_count > deferred.nak_count

    def test_overhead_ratio_exceeds_the_lossless_floor_when_units_are_lost(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 23
        )
        assert result.overhead_ratio > medium_spec.minimum_overhead_ratio


class TestAccounting:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_aggregate_count_equals_the_independent_per_unit_count(
        self,
        medium_spec: FileTransferSpec,
        lossy_link: LinkConditions,
        mode: TransferMode,
        seed: int,
    ) -> None:
        result = simulate_transfer(medium_spec, lossy_link, mode, seed)
        assert independent_transmission_count(result) == result.forward_transmissions
        assert accounting_discrepancy(result) == 0

    def test_forward_data_bytes_recompute_from_the_attempt_counts(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 29
        )
        control_bytes = (result.rounds - 1) * lossy_link.eof_bytes
        data_bytes = result.forward_wire_bytes - control_bytes
        assert expected_wire_bytes_from_attempts(medium_spec, result) == data_bytes

    def test_every_delivered_unit_was_attempted_at_least_once(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, lossy_link, TransferMode.IMMEDIATE_NAK, 31
        )
        assert all(count >= 1 for count in result.attempts)


class TestReproducibility:
    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_identical_seed_gives_identical_results(
        self, medium_spec: FileTransferSpec, outage_link: LinkConditions, mode: TransferMode
    ) -> None:
        first = simulate_transfer(medium_spec, outage_link, mode, 20261024)
        second = simulate_transfer(medium_spec, outage_link, mode, 20261024)
        assert first == second

    def test_different_seeds_give_different_transmission_counts(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        counts = {
            simulate_transfer(
                medium_spec, lossy_link, TransferMode.DEFERRED_NAK, seed
            ).forward_transmissions
            for seed in range(20)
        }
        assert len(counts) > 1

    def test_payload_seed_can_be_held_fixed_across_link_seeds(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        first = simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 1, payload_seed=99
        )
        second = simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 2, payload_seed=99
        )
        assert first.source_digest == second.source_digest
        assert first.forward_transmissions != second.forward_transmissions


class TestOutagesAndWindows:
    def test_an_outage_inflates_the_transmission_count(
        self, medium_spec: FileTransferSpec
    ) -> None:
        plain = LinkConditions(1000.0, 100.0, 0.5, 0.0)
        with_outage = LinkConditions(
            1000.0, 100.0, 0.5, 0.0, outage=OutageModel(mean_up_s=20.0, mean_down_s=20.0)
        )
        clean = simulate_transfer(
            medium_spec, plain, TransferMode.DEFERRED_NAK, 3
        )
        bursty = simulate_transfer(
            medium_spec, with_outage, TransferMode.DEFERRED_NAK, 3
        )
        assert bursty.forward_transmissions > clean.forward_transmissions
        assert bursty.outage_seconds > 0.0

    def test_delivery_is_attributed_to_a_contact_window(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        link = LinkConditions(
            1000.0, 100.0, 0.5, 0.0, windows=uniform_windows(4, 5.0, 50.0)
        )
        result = simulate_transfer(tiny_spec, link, TransferMode.DEFERRED_NAK, 1)
        assert result.delivered
        # 8.56 s of transmission cannot fit in a 5 s window, so the transfer
        # spans windows and completes in a later one.
        assert result.completion_contact_index is not None
        assert result.completion_contact_index > 0

    def test_continuous_contact_reports_window_index_minus_one(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        result = simulate_transfer(tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 1)
        assert result.completion_contact_index == -1

    def test_a_horizon_that_is_too_short_truncates_the_run(
        self, medium_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, clean_link, TransferMode.DEFERRED_NAK, 1, horizon_s=1.0
        )
        assert not result.delivered
        assert result.truncated
        assert "horizon" in result.truncation_reason

    def test_a_window_too_short_for_one_pdu_cannot_carry_the_file(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        # One full unit is 112 wire bytes, 0.896 s at 1000 bit/s, so a 0.5 s
        # window can never carry one.
        link = LinkConditions(1000.0, 100.0, 0.0, 0.0, windows=uniform_windows(3, 0.5, 10.0))
        result = simulate_transfer(tiny_spec, link, TransferMode.DEFERRED_NAK, 1)
        assert not result.delivered
        assert result.truncated
        assert "contact window" in result.truncation_reason

    def test_variable_size_nak_is_charged_on_the_return_link(
        self, medium_spec: FileTransferSpec
    ) -> None:
        fixed = LinkConditions(1000.0, 100.0, 0.5, 0.2)
        growing = LinkConditions(1000.0, 100.0, 0.5, 0.2, nak_bytes_per_segment=8.0)
        plain = simulate_transfer(medium_spec, fixed, TransferMode.DEFERRED_NAK, 7)
        grown = simulate_transfer(medium_spec, growing, TransferMode.DEFERRED_NAK, 7)
        assert grown.return_wire_bytes > plain.return_wire_bytes
        assert grown.delivery_time_s > plain.delivery_time_s


class TestGuards:
    def test_mode_must_be_a_transfer_mode(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        with pytest.raises(TypeError, match="TransferMode"):
            simulate_transfer(tiny_spec, clean_link, "deferred_nak", 1)  # type: ignore[arg-type]

    def test_a_zero_transmission_cap_is_rejected(
        self, tiny_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            simulate_transfer(
                tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 1,
                max_forward_transmissions=0,
            )

    def test_default_cap_scales_with_loss_and_outage_duty(
        self, medium_spec: FileTransferSpec
    ) -> None:
        plain = LinkConditions(1000.0, 100.0, 0.0, 0.0)
        bad = LinkConditions(
            1000.0, 100.0, 0.0, 0.2, outage=OutageModel(mean_up_s=1.0, mean_down_s=3.0)
        )
        assert default_transmission_cap(medium_spec, bad) > default_transmission_cap(
            medium_spec, plain
        )

    def test_regime_predicate_identifies_the_closed_form_regime(
        self, clean_link: LinkConditions, outage_link: LinkConditions
    ) -> None:
        assert is_outage_free_continuous(clean_link)
        assert not is_outage_free_continuous(outage_link)

    def test_describe_reports_the_truncation_reason(
        self, medium_spec: FileTransferSpec, clean_link: LinkConditions
    ) -> None:
        result = simulate_transfer(
            medium_spec, clean_link, TransferMode.DEFERRED_NAK, 1, horizon_s=1.0
        )
        assert any("truncated" in line for line in result.describe())


def test_closed_form_and_simulation_agree_on_transmission_count(
    medium_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    """The mode-independent expectation ``n / (1 - p)``, over 200 runs.

    This is the accounting check that does not depend on any timing model: the
    expected number of forward transmissions is ``n / (1 - p)`` whatever the
    retransmission schedule, so the simulated mean must agree with it to within
    sampling error.
    """
    reference = analytic_delivery(
        medium_spec, lossy_link, TransferMode.DEFERRED_NAK
    ).expected_transmissions
    counts = [
        simulate_transfer(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, seed
        ).forward_transmissions
        for seed in range(200)
    ]
    mean = sum(counts) / len(counts)
    # The per-unit attempt count has variance p / (1 - p)**2 = 0.3125 at
    # p = 0.2, so the total over 120 units has standard deviation about 6.1 and
    # the mean over 200 runs about 0.43. Three of those is 1.3.
    assert mean == pytest.approx(reference, abs=1.5)
