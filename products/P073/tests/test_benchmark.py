"""Comparison harnesses."""

from __future__ import annotations

import numpy as np
import pytest

from contactsched.benchmark import (
    MethodResult,
    compare_methods,
    fairness_cost,
    gap_distribution,
    objective_of,
    optimality_gap,
    run_method,
    score_predictor,
)
from contactsched.exact import solve_exact
from contactsched.greedy import schedule_greedy
from contactsched.problem import ContactRequest, build_problem
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


class TestRunMethod:
    @pytest.mark.parametrize("method", ["greedy", "greedy-demand", "exact", "exact-fair"])
    def test_every_named_method_runs(self, toy_problem, method):
        result = run_method(toy_problem, method)
        assert isinstance(result, MethodResult)
        assert result.elapsed_s >= 0.0

    def test_unknown_method_raises(self, toy_problem):
        with pytest.raises(ValueError, match="unknown method"):
            run_method(toy_problem, "simulated-annealing")

    @pytest.mark.parametrize("method", ["greedy", "greedy-demand", "exact"])
    def test_no_method_returns_an_infeasible_schedule(self, instance_seed7, method):
        assert run_method(instance_seed7.problem, method).violations == ()

    def test_the_report_matches_the_assignment(self, toy_problem):
        result = run_method(toy_problem, "exact")
        assert result.report.method == result.assignment.method

    def test_exact_fair_enforces_minimums_when_it_can(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=3),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=2, max_contacts=2),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 400.0, 500.0),
            window("SC-1", 800.0, 900.0),
            window("SC-2", 0.0, 100.0),
            window("SC-2", 400.0, 500.0),
        )
        problem = build_problem((station(),), requests, windows, 100.0)
        result = run_method(problem, "exact-fair")
        assert result.report.starved == ()


class TestObjectiveAgreement:
    @pytest.mark.parametrize("method", ["greedy", "greedy-demand", "exact"])
    def test_reported_objective_matches_a_recomputation(self, instance_seed7, method):
        result = run_method(instance_seed7.problem, method)
        assert result.assignment.objective == pytest.approx(
            objective_of(instance_seed7.problem, result.assignment), rel=1e-12, abs=0.0
        )

    def test_agreement_holds_under_the_seconds_objective(self, instance_seed7):
        result = run_method(instance_seed7.problem, "exact", objective_mode="weighted_seconds")
        assert result.assignment.objective == pytest.approx(
            objective_of(instance_seed7.problem, result.assignment), rel=1e-12, abs=0.0
        )


class TestOptimalityGap:
    def test_is_zero_when_the_heuristic_matches(self, toy_problem):
        exact = solve_exact(toy_problem)
        assert optimality_gap(exact, exact) == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_is_one_when_the_heuristic_gets_nothing(self, toy_problem):
        from contactsched.assignment import STATUS_HEURISTIC, Assignment

        exact = solve_exact(toy_problem)
        empty = Assignment("none", STATUS_HEURISTIC, (), 0.0)
        assert optimality_gap(empty, exact) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    def test_is_zero_when_the_optimum_is_zero(self):
        from contactsched.assignment import STATUS_HEURISTIC, Assignment

        zero = Assignment("none", STATUS_HEURISTIC, (), 0.0)
        assert optimality_gap(zero, zero) == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_is_never_negative_on_generated_instances(self, small_instances):
        for instance in small_instances:
            gap = optimality_gap(
                schedule_greedy(instance.problem), solve_exact(instance.problem)
            )
            assert gap >= -1e-9, f"seed {instance.seed}"


class TestGapDistribution:
    @pytest.fixture(scope="class")
    @staticmethod
    def distribution(small_instances):
        return gap_distribution([i.problem for i in small_instances])

    def test_has_the_declared_keys(self, distribution):
        assert set(distribution) == {
            "gaps",
            "n",
            "mean",
            "median",
            "p90",
            "max",
            "wins",
            "win_fraction",
        }

    def test_returns_the_full_list_of_gaps(self, distribution, small_instances):
        assert len(distribution["gaps"]) == len(small_instances)

    def test_summary_statistics_agree_with_the_list(self, distribution):
        gaps = np.asarray(distribution["gaps"])
        assert distribution["mean"] == pytest.approx(gaps.mean(), rel=1e-12, abs=0.0)
        assert distribution["median"] == pytest.approx(np.median(gaps), rel=1e-12, abs=0.0)
        assert distribution["max"] == pytest.approx(gaps.max(), rel=1e-12, abs=0.0)

    def test_win_fraction_matches_the_win_count(self, distribution):
        assert distribution["win_fraction"] == pytest.approx(
            distribution["wins"] / distribution["n"], rel=1e-12, abs=0.0
        )

    def test_every_gap_is_within_zero_and_one(self, distribution):
        for gap in distribution["gaps"]:
            assert -1e-9 <= gap <= 1.0 + 1e-9

    def test_an_empty_family_gives_nan_summaries(self):
        result = gap_distribution([])
        assert result["gaps"] == []
        assert np.isnan(result["mean"])


class TestFairnessCost:
    @pytest.fixture
    def problem(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=1, max_contacts=3),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, min_contacts=2, max_contacts=2),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 400.0, 500.0),
            window("SC-1", 800.0, 900.0),
            window("SC-2", 0.0, 100.0),
            window("SC-2", 400.0, 500.0),
        )
        return build_problem((station(),), requests, windows, 100.0)

    def test_prices_the_constraint(self, problem):
        free = run_method(problem, "exact")
        fair = run_method(problem, "exact-fair")
        price = fairness_cost(free, fair, "weighted_contacts")
        # Free optimum 3.0, constrained 2.0, so the loss is 1.0 and one third.
        assert price["objective_unconstrained"] == pytest.approx(3.0, rel=0.0, abs=1e-12)
        assert price["objective_constrained"] == pytest.approx(2.0, rel=0.0, abs=1e-12)
        assert price["objective_loss"] == pytest.approx(1.0, rel=0.0, abs=1e-12)
        assert price["objective_loss_fraction"] == pytest.approx(
            1.0 / 3.0, rel=1e-12, abs=0.0
        )

    def test_reports_the_equality_gain(self, problem):
        price = fairness_cost(
            run_method(problem, "exact"), run_method(problem, "exact-fair"), "weighted_contacts"
        )
        assert price["max_min_gain"] > 0.0
        assert price["starved_before"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert price["starved_after"] == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_an_infeasible_constrained_solve_prices_nothing(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=3, max_contacts=3),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        price = fairness_cost(
            run_method(problem, "exact"), run_method(problem, "exact-fair"), "weighted_contacts"
        )
        assert np.isnan(price["objective_loss"])
        assert np.isnan(price["jain_gain"])
        assert price["seconds_short_before"] == pytest.approx(200.0, rel=0.0, abs=1e-12)

    def test_declared_keys_are_all_present(self, problem):
        price = fairness_cost(
            run_method(problem, "exact"), run_method(problem, "exact-fair"), "weighted_contacts"
        )
        assert set(price) == {
            "objective_unconstrained",
            "objective_constrained",
            "objective_loss",
            "objective_loss_fraction",
            "jain_gain",
            "max_min_gain",
            "starved_before",
            "starved_after",
            "seconds_short_before",
            "seconds_short_after",
        }


class TestCompareMethods:
    def test_returns_one_result_per_method(self, toy_problem):
        results = compare_methods(toy_problem)
        assert set(results) == {"greedy", "greedy-demand", "exact", "exact-fair"}

    def test_the_exact_method_is_never_beaten(self, instance_seed7):
        results = compare_methods(instance_seed7.problem)
        exact = results["exact"].assignment.objective
        assert results["greedy"].assignment.objective <= exact + 1e-9
        assert results["greedy-demand"].assignment.objective <= exact + 1e-9

    def test_a_subset_of_methods_can_be_requested(self, toy_problem):
        results = compare_methods(toy_problem, methods=("greedy",))
        assert set(results) == {"greedy"}


class TestScorePredictor:
    def test_row_has_the_declared_keys(self):
        row = score_predictor(
            "demo",
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([0.1, 0.9, 0.2, 0.8]),
            threshold=0.5,
            calibrated=True,
        )
        for key in ("predictor", "calibrated_claim", "roc_auc", "average_precision", "brier"):
            assert key in row
        assert row["predictor"] == "demo"
        assert row["calibrated_claim"] is True

    def test_a_perfect_predictor_scores_one(self):
        row = score_predictor(
            "perfect",
            np.array([0, 0, 1, 1]),
            np.array([0.0, 0.1, 0.9, 1.0]),
            np.array([0.0, 0.0, 1.0, 1.0]),
            threshold=0.5,
            calibrated=True,
        )
        assert row["roc_auc"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert row["brier"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert row["f1"] == pytest.approx(1.0, rel=0.0, abs=0.0)

    def test_rank_and_probability_arrays_can_differ(self):
        # The ranking baseline's case: the score orders correctly while the
        # pseudo-probability is badly calibrated.
        row = score_predictor(
            "ranking",
            np.array([0, 0, 1, 1]),
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([0.0, 0.5, 0.67, 0.75]),
            threshold=0.5,
            calibrated=False,
        )
        assert row["roc_auc"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert row["brier"] > 0.0
        assert row["calibrated_claim"] is False
