"""Baseline 2: Monte Carlo ensembles, quantiles, the tail and per-contact completion."""

from __future__ import annotations

import math

import numpy as np
import pytest

from relfiledelivery.analytic import analytic_delivery
from relfiledelivery.link import LinkConditions
from relfiledelivery.modes import ALL_MODES, TransferMode
from relfiledelivery.montecarlo import (
    DEFAULT_QUANTILES,
    completion_by_contact,
    run_ensemble,
    standard_error_of_mean,
    summarise,
)
from relfiledelivery.payload import FileTransferSpec
from relfiledelivery.simulate import simulate_transfer


class TestEnsemble:
    def test_quantiles_are_monotone_in_the_level(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        distribution = run_ensemble(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 200, 100
        )
        values = [distribution.quantiles[level] for level in sorted(DEFAULT_QUANTILES)]
        assert values == sorted(values)

    def test_tail_is_reported_separately_and_bounds_the_quantiles(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        distribution = run_ensemble(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 200, 100
        )
        assert distribution.tail["max_s"] >= distribution.quantiles[0.99]
        assert distribution.tail["max_over_median"] >= 1.0
        assert distribution.tail["incomplete_runs"] == pytest.approx(0.0)

    def test_every_run_passes_the_accounting_cross_check(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        distribution = run_ensemble(
            medium_spec, lossy_link, TransferMode.IMMEDIATE_NAK, 150, 500
        )
        assert np.all(distribution.accounting_discrepancies == 0)

    def test_ensemble_is_reproducible_from_one_integer(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        first = run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 60, 7)
        second = run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 60, 7)
        assert np.array_equal(first.delivery_times_s, second.delivery_times_s)
        assert first.quantiles == second.quantiles

    def test_extending_an_ensemble_does_not_change_the_runs_already_made(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        short = run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 40, 11)
        long = run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 80, 11)
        # The first forty seeds are the same runs, so the shorter sample is a
        # subsequence of the longer one once both are sorted.
        assert set(short.delivery_times_s.tolist()).issubset(set(long.delivery_times_s.tolist()))

    def test_simulated_mean_agrees_with_the_closed_form_within_sampling_error(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        """The headline no-outage check, at the sample size a test can afford."""
        distribution = run_ensemble(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 400, 3000
        )
        exact = analytic_delivery(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK
        ).expected_delivery_time_s
        error = standard_error_of_mean(distribution.delivery_times_s)
        assert distribution.mean_s == pytest.approx(exact, abs=4.0 * error)

    @pytest.mark.parametrize("runs", [0, -1])
    def test_non_positive_run_count_is_rejected(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions, runs: int
    ) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, runs, 1)

    def test_out_of_range_quantile_level_is_rejected(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            run_ensemble(
                medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 5, 1, quantiles=(0.0,)
            )


class TestUndeliveredEnsembles:
    def test_an_ensemble_with_no_completions_reports_no_distribution(
        self, medium_spec: FileTransferSpec
    ) -> None:
        link = LinkConditions(1000.0, 100.0, 0.5, 0.3)
        distribution = run_ensemble(
            medium_spec, link, TransferMode.UNACKNOWLEDGED, 30, 1
        )
        assert distribution.completed == 0
        assert distribution.completion_fraction == pytest.approx(0.0)
        assert math.isnan(distribution.median_s)
        assert distribution.tail["incomplete_runs"] == pytest.approx(30.0)
        assert any("no run completed" in line for line in distribution.describe())
        with pytest.raises(ValueError, match="no run completed"):
            distribution.quantile(0.5)

    def test_completion_fraction_is_over_all_runs_not_the_survivors(
        self, medium_spec: FileTransferSpec
    ) -> None:
        link = LinkConditions(1000.0, 100.0, 0.5, 0.02)
        distribution = run_ensemble(
            medium_spec, link, TransferMode.UNACKNOWLEDGED, 200, 1
        )
        assert 0.0 < distribution.completion_fraction < 1.0
        assert distribution.completed == distribution.delivery_times_s.size


class TestCompletionByContact:
    def test_cumulative_fractions_are_non_decreasing_and_bounded(
        self, tiny_spec: FileTransferSpec
    ) -> None:
        from relfiledelivery.link import OutageModel, uniform_windows

        link = LinkConditions(
            1000.0,
            100.0,
            0.25,
            0.05,
            outage=OutageModel(mean_up_s=6.0, mean_down_s=4.0),
            windows=uniform_windows(6, 6.0, 20.0),
        )
        distribution = run_ensemble(
            tiny_spec, link, TransferMode.DEFERRED_NAK, 200, 2
        )
        by_contact = distribution.completion_by_contact
        fractions = [by_contact[index] for index in sorted(by_contact)]
        assert fractions == sorted(fractions)
        assert fractions[-1] == pytest.approx(distribution.completion_fraction)
        assert fractions[-1] <= 1.0

    def test_continuous_contact_gives_a_single_minus_one_key(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions
    ) -> None:
        distribution = run_ensemble(
            medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 20, 1
        )
        assert set(distribution.completion_by_contact) == {-1}

    def test_no_completions_gives_an_empty_mapping(
        self, medium_spec: FileTransferSpec
    ) -> None:
        link = LinkConditions(1000.0, 100.0, 0.5, 0.3)
        results = [
            simulate_transfer(medium_spec, link, TransferMode.UNACKNOWLEDGED, seed)
            for seed in range(10)
        ]
        assert completion_by_contact(results, link) == {}
        assert completion_by_contact([], link) == {}


class TestSummariseAndStandardError:
    def test_summarise_rejects_an_empty_list(self, clean_link: LinkConditions) -> None:
        with pytest.raises(ValueError, match="empty list"):
            summarise([], clean_link, TransferMode.DEFERRED_NAK)

    def test_standard_error_needs_two_samples(self) -> None:
        with pytest.raises(ValueError, match="two samples"):
            standard_error_of_mean(np.array([1.0]))

    def test_standard_error_is_the_textbook_expression(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        # Sample standard deviation with ddof=1 is sqrt(5/3) = 1.2909944487;
        # divided by sqrt(4) that is 0.6454972244.
        assert standard_error_of_mean(values) == pytest.approx(
            math.sqrt(5.0 / 3.0) / 2.0, rel=1e-12
        )

    @pytest.mark.parametrize("mode", list(ALL_MODES))
    def test_describe_never_puts_the_mean_before_the_quantiles(
        self, medium_spec: FileTransferSpec, lossy_link: LinkConditions, mode: TransferMode
    ) -> None:
        distribution = run_ensemble(medium_spec, lossy_link, mode, 30, 1)
        lines = distribution.describe()
        if distribution.completed == 0:
            return
        quantile_at = next(i for i, line in enumerate(lines) if "q0.500" in line)
        mean_at = next(i for i, line in enumerate(lines) if line.startswith("mean"))
        assert quantile_at < mean_at
