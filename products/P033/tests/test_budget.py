"""Budget declaration, feasibility, verdicts and the uncertainty annotation."""

from __future__ import annotations

import pytest

from edgeinfer.budget import (
    MARGINAL_SIGMA_THRESHOLD,
    Budget,
    CheckRow,
    Verdict,
    build_report,
)


class TestBudgetValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"latency_s": 0.0}, "latency_s"),
            ({"latency_s": -1.0}, "latency_s"),
            ({"peak_memory_bytes": 0}, "peak_memory_bytes"),
            ({"median_latency_s": 0.0}, "median_latency_s"),
            ({"power_w": -1.0}, "power_w"),
            ({"duty_cycle": 0.0}, "duty_cycle"),
            ({"duty_cycle": 1.5}, "duty_cycle"),
            ({"period_s": 0.0}, "period_s"),
            ({"worst_case_quantile": 0.5}, "worst_case_quantile"),
            ({"worst_case_quantile": 1.0}, "worst_case_quantile"),
            ({"name": ""}, "name"),
        ],
    )
    def test_invalid_declarations_are_rejected(self, kwargs: dict, match: str) -> None:
        base = {"name": "b", "latency_s": 1e-3, "peak_memory_bytes": 1024}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            Budget(**base)

    def test_duty_cycle_ceiling_is_the_product(self) -> None:
        # 0.25 of a 4 ms period is 1 ms.
        budget = Budget("b", 1e-3, 1024, duty_cycle=0.25, period_s=4e-3)
        assert budget.duty_cycle_latency_s == pytest.approx(1e-3)

    def test_effective_ceiling_is_the_tighter_of_the_two(self) -> None:
        tighter = Budget("b", 5e-3, 1024, duty_cycle=0.1, period_s=10e-3)
        assert tighter.effective_latency_s == pytest.approx(1e-3)
        looser = Budget("b", 5e-4, 1024, duty_cycle=0.5, period_s=10e-3)
        assert looser.effective_latency_s == pytest.approx(5e-4)

    def test_energy_per_inference_is_power_times_time(self) -> None:
        # E = P t: 5 W for 2 ms is 10 mJ.
        budget = Budget("b", 1e-2, 1024, power_w=5.0)
        assert budget.energy_per_inference_j(2e-3) == pytest.approx(1e-2)

    def test_energy_is_none_without_a_declared_power(self) -> None:
        assert Budget("b", 1e-2, 1024).energy_per_inference_j(1e-3) is None

    def test_power_row_is_labelled_not_measured_in_the_summary(self) -> None:
        text = "\n".join(Budget("b", 1e-3, 1024, power_w=7.0).summary_lines())
        assert "NOT MEASURED" in text


class TestFeasibility:
    def test_a_plain_budget_is_feasible(self) -> None:
        assert Budget("b", 1e-3, 1024).feasibility().feasible

    def test_median_above_worst_case_is_infeasible_by_construction(self) -> None:
        result = Budget("b", 1e-3, 1024, median_latency_s=2e-3).feasibility()
        assert not result
        assert any("median ceiling" in r for r in result.reasons)

    def test_duty_cycle_tighter_than_the_latency_ceiling_is_infeasible(self) -> None:
        # 10 % of a 10 ms period is 1 ms, below the declared 2 ms ceiling.
        result = Budget("b", 2e-3, 1024, duty_cycle=0.1, period_s=10e-3).feasibility()
        assert not result
        assert any("duty cycle" in r for r in result.reasons)

    def test_several_contradictions_are_all_reported(self) -> None:
        result = Budget(
            "b", 2e-3, 1024, median_latency_s=3e-3, duty_cycle=0.05, period_s=10e-3
        ).feasibility()
        assert len(result.reasons) >= 2

    def test_median_above_the_duty_cycle_ceiling_is_caught(self) -> None:
        result = Budget(
            "b", 4e-4, 1024, median_latency_s=3e-4, duty_cycle=0.02, period_s=10e-3
        ).feasibility()
        assert not result
        assert any("duty-cycle-implied" in r for r in result.reasons)

    def test_an_infeasible_budget_fails_the_report_outright(self) -> None:
        budget = Budget("b", 2e-3, 1024, duty_cycle=0.1, period_s=10e-3)
        report = build_report(
            budget,
            candidate="c",
            environment="test",
            worst_case_latency_s=1e-9,
            median_latency_s=1e-9,
            peak_memory_bytes=1.0,
        )
        assert report.overall is Verdict.FAIL
        assert "INFEASIBLE BY CONSTRUCTION" in "\n".join(report.summary_lines())


class TestVerdicts:
    def test_comfortably_inside_the_limit_is_a_plain_pass(self) -> None:
        row = CheckRow("q", "s", 1e-4, 1e-3, 1e-8, Verdict.PASS, "m")
        assert row.margin == pytest.approx(9e-4)
        assert row.margin_sigma is not None and row.margin_sigma > MARGINAL_SIGMA_THRESHOLD

    def test_utilisation_is_value_over_limit(self) -> None:
        row = CheckRow("q", "s", 5e-4, 1e-3, None, Verdict.PASS, "m")
        assert row.utilisation == pytest.approx(0.5)

    def test_a_pass_inside_the_uncertainty_is_marked_marginal(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024),
            candidate="c",
            environment="test",
            worst_case_latency_s=0.99e-3,
            worst_case_uncertainty_s=5e-5,  # margin 1e-5 is 0.2 sigma
            peak_memory_bytes=512.0,
        )
        tail = report.rows[0]
        assert tail.verdict is Verdict.MARGINAL_PASS
        assert report.overall is Verdict.MARGINAL_PASS

    def test_a_fail_inside_the_uncertainty_is_marked_marginal(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024),
            candidate="c",
            environment="test",
            worst_case_latency_s=1.01e-3,
            worst_case_uncertainty_s=5e-5,
            peak_memory_bytes=512.0,
        )
        assert report.rows[0].verdict is Verdict.MARGINAL_FAIL
        assert report.overall is Verdict.MARGINAL_FAIL

    def test_a_clear_overrun_fails(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024),
            candidate="c",
            environment="test",
            worst_case_latency_s=5e-3,
            worst_case_uncertainty_s=1e-7,
            peak_memory_bytes=512.0,
        )
        assert report.rows[0].verdict is Verdict.FAIL
        assert report.overall is Verdict.FAIL

    def test_worst_case_failure_overrides_a_median_pass(self) -> None:
        """The reason the two limits are separate at all."""
        budget = Budget("b", 1e-3, 1024, median_latency_s=5e-4)
        report = build_report(
            budget,
            candidate="c",
            environment="test",
            worst_case_latency_s=4e-3,  # tail blows the ceiling
            worst_case_uncertainty_s=1e-7,
            median_latency_s=1e-4,  # median is comfortable
            median_uncertainty_s=1e-8,
            peak_memory_bytes=512.0,
        )
        verdicts = {r.quantity: r.verdict for r in report.rows}
        assert verdicts["worst-case latency (p99)"] is Verdict.FAIL
        assert verdicts["median latency (p50)"] is Verdict.PASS
        assert report.overall is Verdict.FAIL

    def test_an_undeclared_limit_is_not_decided(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024),  # no median declared
            candidate="c",
            environment="test",
            worst_case_latency_s=1e-4,
            worst_case_uncertainty_s=1e-9,
            median_latency_s=5e-5,
            peak_memory_bytes=512.0,
        )
        median_row = next(r for r in report.rows if "median" in r.quantity)
        assert median_row.verdict is Verdict.NOT_DECLARED
        assert any("median" in u for u in report.undecided)

    def test_an_unmeasured_quantity_is_not_decided(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024, median_latency_s=5e-4),
            candidate="c",
            environment="test",
            worst_case_latency_s=1e-4,
            worst_case_uncertainty_s=1e-9,
            peak_memory_bytes=512.0,
        )
        median_row = next(r for r in report.rows if "median" in r.quantity)
        assert median_row.verdict is Verdict.NOT_MEASURED

    def test_power_is_always_not_measured_even_when_a_value_is_passed(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024, power_w=7.0),
            candidate="c",
            environment="test",
            worst_case_latency_s=1e-4,
            worst_case_uncertainty_s=1e-9,
            peak_memory_bytes=512.0,
            declared_power_w=3.0,
        )
        power_row = next(r for r in report.rows if "power" in r.quantity)
        assert power_row.verdict is Verdict.NOT_MEASURED
        assert "not measured by edgeinfer" in power_row.method

    def test_nothing_measured_gives_an_undecided_overall(self) -> None:
        report = build_report(
            Budget("b", 1e-3, 1024), candidate="c", environment="test"
        )
        assert report.overall is Verdict.NOT_MEASURED

    def test_verdict_is_pass_predicate_only_for_the_two_pass_values(self) -> None:
        assert Verdict.PASS.is_pass
        assert Verdict.MARGINAL_PASS.is_pass
        assert not Verdict.FAIL.is_pass
        assert not Verdict.MARGINAL_FAIL.is_pass
        assert not Verdict.NOT_MEASURED.is_pass


class TestReportText:
    def test_the_report_states_its_environment(self, loose_budget) -> None:
        report = build_report(
            loose_budget,
            candidate="c",
            environment="shared 1-core container",
            worst_case_latency_s=1e-4,
            worst_case_uncertainty_s=1e-9,
            peak_memory_bytes=512.0,
        )
        assert "shared 1-core container" in "\n".join(report.summary_lines())

    def test_every_row_carries_its_method(self, loose_budget) -> None:
        report = build_report(
            loose_budget,
            candidate="c",
            environment="e",
            worst_case_latency_s=1e-4,
            peak_memory_bytes=512.0,
            latency_method="perf_counter, n=200",
            memory_method="liveness analysis",
        )
        for row in report.rows:
            assert row.method.strip()

    def test_the_table_lists_units(self, loose_budget) -> None:
        report = build_report(
            loose_budget, candidate="c", environment="e", peak_memory_bytes=1.0
        )
        table = report.to_table()
        assert "[s]" in table
        assert "[B]" in table
        assert "[W]" in table
