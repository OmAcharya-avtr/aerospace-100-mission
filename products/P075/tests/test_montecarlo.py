"""Monte Carlo: scored against the closed forms, never the other way round.

Every structure is simulated and compared with its closed form. The comparison
is in units of the binomial standard error, which is the only honest way to say
whether an agreement was a hard check or a soft one.

The budgets here are small, because the suite has to finish quickly. The large
budgets, sized from a target interval half-width, live in
``validation/validate_montecarlo.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from reliablock.diagram import Block, Diagram, KOfN, Parallel, Series, SharedLoad, Standby
from reliablock.distributions import Exponential, FixedProbability, Weibull
from reliablock.montecarlo import (
    sample_lifetimes,
    simulate,
    trials_for_half_width,
    wilson_interval,
)
from reliablock.quantify import quantify

from conftest import abs_, exponential_block, rel

MISSION = 1000.0
TRIALS = 120_000
SEED = 20261025


class TestWilsonInterval:
    def test_hand_calculated(self) -> None:
        # k = 90, n = 100, z = 1.96: centre = (90 + 1.9208) / 103.8415
        lower, upper = wilson_interval(90, 100, 1.959963984540054)
        z2 = 1.959963984540054**2
        centre = (90.0 + 0.5 * z2) / (100.0 + z2)
        assert 0.5 * (lower + upper) == rel(centre, 1e-12)

    def test_stays_inside_the_unit_interval(self) -> None:
        for successes in (0, 1, 50, 99, 100):
            lower, upper = wilson_interval(successes, 100)
            assert 0.0 <= lower <= upper <= 1.0

    def test_does_not_collapse_at_zero_failures(self) -> None:
        # The normal-approximation interval has zero width here, which would
        # make a check on a redundant diagram vacuous.
        lower, upper = wilson_interval(1000, 1000)
        assert upper - lower > 0.0
        assert lower < 1.0

    def test_rejects_bad_counts(self) -> None:
        with pytest.raises(ValueError, match="trials must be > 0"):
            wilson_interval(0, 0)
        with pytest.raises(ValueError, match="successes must lie"):
            wilson_interval(11, 10)


class TestBudgetSizing:
    def test_sizing_is_derived_from_the_target_half_width(self) -> None:
        # n = z^2 Q(1-Q)/h^2. At Q = 0.02 and h = 4e-4 that is 470705.
        z = 1.959963984540054
        n = trials_for_half_width(0.02, 4.0e-4)
        assert n == math.ceil(z * z * 0.02 * 0.98 / (4.0e-4) ** 2)

    def test_the_achieved_half_width_is_close_to_the_target(self) -> None:
        target = 2.0e-3
        trials = trials_for_half_width(0.1, target)
        diagram = Diagram("one", Series(items=(exponential_block("x", 1.0536e-4),)))
        result = simulate(diagram, MISSION, trials, seed=SEED)
        assert result.half_width == rel(target, 0.1)

    def test_degenerate_sizing_is_refused_with_an_explanation(self) -> None:
        with pytest.raises(ValueError, match="strictly in"):
            trials_for_half_width(0.0, 1e-3)
        with pytest.raises(ValueError, match="half_width must be > 0"):
            trials_for_half_width(0.1, 0.0)


class TestAgreementWithClosedForms:
    @pytest.mark.parametrize(
        "fixture",
        [
            "series_pair",
            "parallel_pair",
            "two_of_three",
            "cold_standby",
            "warm_standby",
            "shared_load",
            "bridge_network",
        ],
    )
    def test_simulation_brackets_the_closed_form(
        self, fixture: str, request: pytest.FixtureRequest
    ) -> None:
        diagram = request.getfixturevalue(fixture)
        hours = 1.0 if fixture == "bridge_network" else MISSION
        closed = quantify(diagram, hours)
        result = simulate(diagram, hours, TRIALS, seed=SEED)
        assert result.brackets(closed.reliability), (
            f"{fixture}: closed {closed.reliability:.9f} outside "
            f"[{result.lower:.9f}, {result.upper:.9f}], "
            f"{result.standard_errors_from(closed.reliability):.2f} sigma"
        )

    @pytest.mark.parametrize(
        "fixture",
        ["series_pair", "parallel_pair", "two_of_three", "cold_standby", "warm_standby"],
    )
    def test_agreement_is_within_four_standard_errors(
        self, fixture: str, request: pytest.FixtureRequest
    ) -> None:
        diagram = request.getfixturevalue(fixture)
        closed = quantify(diagram, MISSION)
        result = simulate(diagram, MISSION, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0

    def test_two_cold_standbys_agree(self) -> None:
        diagram = Diagram(
            "two spares",
            Standby(
                primary=exponential_block("p", 3.0e-4),
                standbys=(exponential_block("s1", 3.0e-4), exponential_block("s2", 3.0e-4)),
                switch_reliability=0.95,
                label="sb",
            ),
        )
        closed = quantify(diagram, MISSION)
        result = simulate(diagram, MISSION, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0

    def test_a_weibull_standby_agrees(self) -> None:
        shared = Weibull(shape=1.8, scale_hours=4000.0)
        diagram = Diagram(
            "weibull standby",
            Standby(
                primary=Block(name="p", distribution=shared),
                standbys=(Block(name="s", distribution=Weibull(shape=1.8, scale_hours=4000.0)),),
                switch_reliability=0.99,
                label="sb",
            ),
        )
        closed = quantify(diagram, MISSION)
        result = simulate(diagram, MISSION, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0

    def test_a_common_cause_rewrite_agrees(self, beta_pair: Diagram) -> None:
        from reliablock.commoncause import apply_common_cause

        diagram = apply_common_cause(beta_pair)
        closed = quantify(diagram, MISSION)
        result = simulate(diagram, MISSION, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0

    def test_the_whole_smallsat_bus_agrees(self) -> None:
        from reliablock.cases import case, mission

        diagram = case("smallsat_bus")
        profile = mission("one_year")
        closed = quantify(diagram, profile)
        result = simulate(diagram, profile, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0

    def test_a_phased_mission_agrees(self) -> None:
        from reliablock.cases import case, mission

        diagram = case("smallsat_bus")
        profile = mission("leo_phased")
        closed = quantify(diagram, profile)
        result = simulate(diagram, profile, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0


class TestDeterminism:
    def test_the_same_seed_gives_the_same_answer(self, parallel_pair: Diagram) -> None:
        first = simulate(parallel_pair, MISSION, 20_000, seed=7)
        second = simulate(parallel_pair, MISSION, 20_000, seed=7)
        assert first.survivals == second.survivals

    def test_a_different_seed_gives_a_different_answer(self, parallel_pair: Diagram) -> None:
        first = simulate(parallel_pair, MISSION, 20_000, seed=7)
        second = simulate(parallel_pair, MISSION, 20_000, seed=8)
        assert first.survivals != second.survivals

    def test_batching_does_not_change_the_result_distribution(
        self, parallel_pair: Diagram
    ) -> None:
        # Batching changes the draw ORDER, so the count may differ; what must
        # not differ is agreement with the closed form.
        closed = quantify(parallel_pair, MISSION)
        for batch in (1000, 7777, 50_000):
            result = simulate(parallel_pair, MISSION, 50_000, seed=3, batch=batch)
            assert result.standard_errors_from(closed.reliability) < 4.0

    def test_the_seed_is_in_the_result(self, parallel_pair: Diagram) -> None:
        assert simulate(parallel_pair, MISSION, 1000, seed=42).seed == 42


class TestSampling:
    def test_exponential_lifetimes_have_the_right_mean(self) -> None:
        rng = np.random.default_rng(0)
        draws = sample_lifetimes(Exponential(1.0e-3), rng, 200_000)
        assert float(draws.mean()) == rel(1000.0, 0.02)

    def test_weibull_lifetimes_match_the_survival_function(self) -> None:
        rng = np.random.default_rng(1)
        d = Weibull(shape=2.0, scale_hours=1000.0)
        draws = sample_lifetimes(d, rng, 200_000)
        for t in (500.0, 1000.0, 1500.0):
            observed = float((draws > t).mean())
            assert observed == abs_(d.reliability(t), 0.005)

    def test_zero_rate_lifetimes_are_infinite(self) -> None:
        rng = np.random.default_rng(0)
        assert np.all(np.isinf(sample_lifetimes(Exponential(0.0), rng, 100)))

    def test_scaled_hazard_lifetimes_are_longer_by_the_factor(self) -> None:
        from reliablock.commoncause import ScaledHazard

        rng = np.random.default_rng(2)
        base = sample_lifetimes(Exponential(1.0e-3), np.random.default_rng(2), 100_000)
        scaled = sample_lifetimes(
            ScaledHazard(base=Exponential(1.0e-3), factor=0.25), rng, 100_000
        )
        assert float(scaled.mean()) == rel(4.0 * float(base.mean()), 0.02)

    def test_scaled_hazard_at_factor_zero_never_fails(self) -> None:
        from reliablock.commoncause import ScaledHazard

        rng = np.random.default_rng(0)
        draws = sample_lifetimes(ScaledHazard(base=Exponential(1.0), factor=0.0), rng, 100)
        assert np.all(np.isinf(draws))

    def test_fixed_probability_lifetimes_are_zero_or_infinite(self) -> None:
        rng = np.random.default_rng(0)
        draws = sample_lifetimes(FixedProbability(0.25), rng, 50_000)
        assert set(np.unique(np.isinf(draws)).tolist()) <= {True, False}
        assert float((draws == 0.0).mean()) == abs_(0.25, 0.01)

    def test_negative_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="size must be >= 0"):
            sample_lifetimes(Exponential(1e-3), np.random.default_rng(0), -1)


class TestResultReporting:
    def test_unreliability_is_the_complement(self, parallel_pair: Diagram) -> None:
        result = simulate(parallel_pair, MISSION, 10_000, seed=1)
        assert result.unreliability == abs_(1.0 - result.reliability, 0.0)

    def test_standard_errors_from_an_identical_value_is_zero(self) -> None:
        from reliablock.montecarlo import MonteCarloResult

        result = MonteCarloResult(
            trials=10, survivals=10, seed=0, reliability=1.0, lower=0.7, upper=1.0
        )
        assert result.standard_errors_from(1.0) == 0.0
        assert result.standard_errors_from(0.5) == math.inf

    def test_str_is_one_readable_line(self, parallel_pair: Diagram) -> None:
        text = str(simulate(parallel_pair, MISSION, 1000, seed=1))
        assert "MC n=1000" in text
        assert "\n" not in text

    def test_bad_arguments_are_rejected(self, parallel_pair: Diagram) -> None:
        with pytest.raises(ValueError, match="trials must be > 0"):
            simulate(parallel_pair, MISSION, 0)
        with pytest.raises(ValueError, match="batch must be > 0"):
            simulate(parallel_pair, MISSION, 10, batch=0)
        with pytest.raises(ValueError, match="confidence must lie"):
            simulate(parallel_pair, MISSION, 10, confidence=1.0)


class TestSimulationRefusals:
    def test_a_warm_multi_unit_standby_is_refused_as_in_the_quantifier(self) -> None:
        diagram = Diagram(
            "warm chain",
            Standby(
                primary=exponential_block("p", 1e-4),
                standbys=(exponential_block("s1", 1e-4), exponential_block("s2", 1e-4)),
                switch_reliability=0.99,
                standby_hazard_fraction=0.3,
                label="sb",
            ),
        )
        with pytest.raises(ValueError, match="same declarations"):
            simulate(diagram, MISSION, 100, seed=1)

    def test_a_nested_standby_branch_is_refused(self) -> None:
        diagram = Diagram(
            "nested",
            Standby(
                primary=Parallel(
                    items=(exponential_block("a", 1e-4), exponential_block("b", 1e-4))
                ),
                standbys=(exponential_block("s", 1e-4),),
                switch_reliability=0.99,
                label="sb",
            ),
        )
        with pytest.raises(ValueError, match="not\n?.*implemented|declare the group"):
            simulate(diagram, MISSION, 100, seed=1)

    def test_shared_load_with_unequal_ages_is_refused(self) -> None:
        from reliablock.mission import MissionPhase, MissionProfile

        diagram = Diagram(
            "pumps",
            SharedLoad(
                units=(exponential_block("u1", 1e-4), exponential_block("u2", 1e-4)),
                shared_rate_factor=0.5,
            ),
        )
        profile = MissionProfile(phases=(MissionPhase("p", 1000.0, duty_cycles={"u1": 0.5}),))
        with pytest.raises(ValueError, match="different effective ages"):
            simulate(diagram, profile, 100, seed=1)


class TestExtremeCases:
    def test_a_certainly_failing_diagram_gives_zero_survivals(self) -> None:
        doomed = Block(name="x", distribution=FixedProbability(1.0))
        diagram = Diagram("doomed", Series(items=(doomed,)))
        result = simulate(diagram, MISSION, 1000, seed=1)
        assert result.survivals == 0
        assert result.reliability == 0.0

    def test_a_never_failing_diagram_gives_all_survivals(self) -> None:
        diagram = Diagram("eternal", Series(items=(exponential_block("x", 0.0),)))
        result = simulate(diagram, MISSION, 1000, seed=1)
        assert result.survivals == 1000
        assert result.reliability == 1.0

    def test_a_k_of_n_group_with_k_equal_to_n_agrees(self) -> None:
        diagram = Diagram(
            "all four",
            KOfN(k=4, items=tuple(exponential_block(f"u{i}", 1.0e-4) for i in range(4))),
        )
        closed = quantify(diagram, MISSION)
        result = simulate(diagram, MISSION, TRIALS, seed=SEED)
        assert result.standard_errors_from(closed.reliability) < 4.0
