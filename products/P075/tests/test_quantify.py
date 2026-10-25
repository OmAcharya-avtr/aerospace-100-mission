"""Closed-form quantification: known answers, closed-form cross-checks, refusals.

Every structure here is checked against a reference derived independently of
the implementation: the binomial survival from ``scipy.stats`` for k-of-n, the
convolution integral from ``scipy.integrate.quad`` for standby, the Markov
ODE solution from ``scipy.integrate.solve_ivp`` for shared load, and exhaustive
state enumeration over the structure function for the Boolean structures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.stats import binom

from reliablock.bounds import bonferroni_series_bounds, esary_proschan_bounds
from reliablock.diagram import Block, Diagram, KOfN, Parallel, Series, SharedLoad, Standby
from reliablock.distributions import Exponential, Weibull
from reliablock.importance import boolean_system_unreliability, system_unreliability_at
from reliablock.quantify import (
    quantify,
    shared_load_reliability,
    standby_reliability_cold_identical,
    standby_reliability_general,
)

from conftest import abs_, blocks_at_unreliability, exponential_block, rel

MISSION = 1000.0


class TestSeries:
    def test_hand_calculated(self, series_pair: Diagram) -> None:
        # H = (2e-4 + 3e-4) * 1000 = 0.5, R = exp(-0.5) = 0.6065306597126334
        result = quantify(series_pair, MISSION)
        assert result.hazard == rel(0.5, 1e-14)
        assert result.reliability == rel(0.6065306597126334, 1e-14)

    def test_hazard_is_the_exact_sum_of_the_block_hazards(self, series_pair: Diagram) -> None:
        result = quantify(series_pair, MISSION)
        assert result.hazard == sum(p.hazard for p in result.parts)

    def test_bonferroni_bounds_bracket_the_series_unreliability(self) -> None:
        # The independently derived bound: a sum of positive terms either side.
        q = 1.0e-12
        names = [f"b{i}" for i in range(10)]
        diagram = Diagram("chain", Series(items=tuple(blocks_at_unreliability(names, q))))
        result = quantify(diagram, 1.0)
        bracket = bonferroni_series_bounds([q] * 10, result.unreliability)
        assert bracket.contains, str(bracket)

    def test_the_naive_expression_falls_outside_the_bound(self) -> None:
        # Not a tolerance question: the naive route is outside an
        # independently derived bracket, which is the whole precision claim.
        q = 1.0e-12
        n = 10
        naive = 1.0 - (1.0 - q) ** n
        bracket = bonferroni_series_bounds([q] * n, naive)
        assert not bracket.contains, str(bracket)

    def test_complement_residual_is_rounding_level(self, series_pair: Diagram) -> None:
        assert abs(quantify(series_pair, MISSION).complement_residual) < 1e-15


class TestParallel:
    def test_hand_calculated(self, parallel_pair: Diagram) -> None:
        # q = 1 - exp(-0.3) = 0.2591817793182821; Q = q^2
        q = 1.0 - math.exp(-0.3)
        result = quantify(parallel_pair, MISSION)
        assert result.unreliability == rel(q * q, 1e-14)
        assert result.reliability == rel(1.0 - q * q, 1e-14)

    def test_unreliability_is_the_exact_product(self, parallel_pair: Diagram) -> None:
        result = quantify(parallel_pair, MISSION)
        product = 1.0
        for part in result.parts:
            product *= part.unreliability
        assert result.unreliability == product

    def test_tiny_unreliability_survives(self) -> None:
        names = ["a", "b"]
        diagram = Diagram(
            "very redundant", Parallel(items=tuple(blocks_at_unreliability(names, 1.0e-9)))
        )
        result = quantify(diagram, 1.0)
        assert result.unreliability == rel(1.0e-18, 1e-9)
        # The reliability is exactly 1.0 in float64, so a package that carried
        # only the reliability would have reported Q = 0.
        assert result.reliability == 1.0

    def test_reliability_is_at_least_the_best_block(self, parallel_pair: Diagram) -> None:
        result = quantify(parallel_pair, MISSION)
        assert result.reliability >= max(p.reliability for p in result.parts)


class TestKOfN:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_identical_blocks_match_the_binomial_sum(self, k: int) -> None:
        rate = 3.0e-4
        p = math.exp(-rate * MISSION)
        diagram = Diagram(
            "k of n",
            KOfN(k=k, items=tuple(exponential_block(n, rate) for n in ("a", "b", "c"))),
        )
        result = quantify(diagram, MISSION)
        assert result.reliability == rel(float(binom.sf(k - 1, 3, p)), 1e-13)

    def test_k_equals_n_is_the_series_chain(self) -> None:
        rate = 3.0e-4
        names = ("a", "b", "c")
        items = tuple(exponential_block(n, rate) for n in names)
        k_of_n = quantify(Diagram("knn", KOfN(k=3, items=items)), MISSION)
        chain = quantify(Diagram("chain", Series(items=items)), MISSION)
        assert k_of_n.reliability == rel(chain.reliability, 1e-13)

    def test_k_equals_one_is_the_parallel_group(self) -> None:
        rate = 3.0e-4
        names = ("a", "b", "c")
        items = tuple(exponential_block(n, rate) for n in names)
        k_of_n = quantify(Diagram("knn", KOfN(k=1, items=items)), MISSION)
        group = quantify(Diagram("par", Parallel(items=items)), MISSION)
        assert k_of_n.reliability == rel(group.reliability, 1e-13)
        assert k_of_n.unreliability == rel(group.unreliability, 1e-13)

    def test_reliability_and_unreliability_are_computed_separately(self) -> None:
        # Both come from their own non-negative convolution, so R + Q is 1 only
        # to within rounding, which is what the residual reports.
        diagram = Diagram(
            "k of n",
            KOfN(k=2, items=tuple(blocks_at_unreliability(["a", "b", "c"], 1.0e-7))),
        )
        result = quantify(diagram, 1.0)
        assert abs(result.complement_residual) < 1e-15
        # Q = 3 q^2 (1 - q) + q^3 to leading order 3e-14
        assert result.unreliability == rel(3.0e-14, 1e-6)

    def test_non_identical_blocks_against_exhaustive_enumeration(self) -> None:
        diagram = Diagram(
            "mixed",
            KOfN(
                k=2,
                items=(
                    exponential_block("a", 1.0e-4),
                    exponential_block("b", 3.0e-4),
                    exponential_block("c", 7.0e-4),
                    exponential_block("d", 2.0e-4),
                ),
            ),
        )
        unreliabilities = {
            b.name: b.distribution.unreliability(MISSION) for b in diagram.blocks()
        }
        assert system_unreliability_at(diagram, unreliabilities) == rel(
            boolean_system_unreliability(diagram, unreliabilities), 1e-12
        )


class TestStandby:
    def test_cold_identical_hand_calculated(self) -> None:
        # lambda t = 0.3, p_s = 1: R = exp(-0.3)(1 + 0.3) = 0.9630774851584494
        assert standby_reliability_cold_identical(3.0e-4, MISSION, 1, 1.0) == rel(
            math.exp(-0.3) * 1.3, 1e-14
        )

    def test_cold_identical_with_an_imperfect_switch(self, cold_standby: Diagram) -> None:
        # R = exp(-0.3)(1 + 0.98 * 0.3)
        result = quantify(cold_standby, MISSION)
        assert result.reliability == rel(math.exp(-0.3) * (1.0 + 0.98 * 0.3), 1e-14)

    def test_erlang_form_matches_the_convolution_integral(self) -> None:
        rate = 3.0e-4
        primary = Exponential(rate)
        closed = standby_reliability_cold_identical(rate, MISSION, 1, 1.0)
        value, error = quad(
            lambda u: primary.density(u) * primary.reliability(MISSION - u), 0.0, MISSION
        )
        assert closed == abs_(primary.reliability(MISSION) + value, max(10 * error, 1e-14))

    def test_two_cold_standbys_match_the_erlang_sum(self) -> None:
        rate = 3.0e-4
        x = rate * MISSION
        expected = math.exp(-x) * (1.0 + x + x * x / 2.0)
        assert standby_reliability_cold_identical(rate, MISSION, 2, 1.0) == rel(expected, 1e-14)

    def test_general_route_reduces_to_the_erlang_form(self) -> None:
        rate = 3.0e-4
        primary = Exponential(rate)
        closed = standby_reliability_cold_identical(rate, MISSION, 1, 0.98)
        general, error = standby_reliability_general(primary, primary, MISSION, 0.98, 0.0)
        assert general == abs_(closed, max(10 * error, 1e-13))

    def test_warm_standby_hand_derived_closed_form(self, warm_standby: Diagram) -> None:
        # R = exp(-x)[1 + p_s (1 - exp(-f x)) / f] with x = lambda t, f the
        # hazard fraction. Derived in the quantify module docstring.
        rate, f, ps = 3.0e-4, 0.4, 0.97
        x = rate * MISSION
        expected = math.exp(-x) * (1.0 + ps * (1.0 - math.exp(-f * x)) / f)
        result = quantify(warm_standby, MISSION)
        assert result.reliability == rel(expected, 1e-9)

    def test_hot_standby_is_parallel_with_a_switch(self) -> None:
        rate = 3.0e-4
        primary = Exponential(rate)
        hot, _ = standby_reliability_general(primary, primary, MISSION, 1.0, 1.0)
        q = 1.0 - math.exp(-rate * MISSION)
        assert hot == rel(1.0 - q * q, 1e-8)

    def test_zero_switch_reliability_is_the_primary_alone(self) -> None:
        rate = 3.0e-4
        primary = Exponential(rate)
        value, error = standby_reliability_general(primary, primary, MISSION, 0.0, 0.0)
        assert value == primary.reliability(MISSION)
        assert error == 0.0

    def test_non_exponential_standby_goes_through_the_convolution(self) -> None:
        diagram = Diagram(
            "weibull standby",
            Standby(
                primary=Block(name="p", distribution=Weibull(shape=1.8, scale_hours=4000.0)),
                standbys=(
                    Block(name="s", distribution=Weibull(shape=1.8, scale_hours=4000.0)),
                ),
                switch_reliability=0.99,
                label="sb",
            ),
        )
        result = quantify(diagram, MISSION)
        primary = Weibull(shape=1.8, scale_hours=4000.0)
        reference, error = quad(
            lambda u: primary.density(u) * primary.reliability(MISSION - u), 0.0, MISSION
        )
        expected = primary.reliability(MISSION) + 0.99 * reference
        assert result.reliability == abs_(expected, max(10 * error, 1e-12))
        assert result.detail["quadrature_abs_error"] >= 0.0

    def test_multiple_warm_standbys_are_refused(self) -> None:
        diagram = Diagram(
            "warm chain",
            Standby(
                primary=exponential_block("p", 1e-5),
                standbys=(exponential_block("s1", 1e-5), exponential_block("s2", 2e-5)),
                switch_reliability=0.99,
                standby_hazard_fraction=0.3,
                label="sb",
            ),
        )
        with pytest.raises(ValueError, match="worse than refusing"):
            quantify(diagram, MISSION)

    def test_nested_standby_branch_is_refused(self) -> None:
        diagram = Diagram(
            "nested standby",
            Standby(
                primary=Parallel(
                    items=(exponential_block("a", 1e-5), exponential_block("b", 1e-5))
                ),
                standbys=(exponential_block("s", 1e-5),),
                switch_reliability=0.99,
                label="sb",
            ),
        )
        with pytest.raises(ValueError, match="not over nested structures"):
            quantify(diagram, MISSION)


class TestSharedLoad:
    def test_hand_derived_closed_form(self, shared_load: Diagram) -> None:
        l1, l2, s, t = 4.0e-4, 5.0e-4, 0.5, MISSION
        load = s * (l1 + l2)
        both = math.exp(-load * t)
        expected = both
        expected += s * l1 * (math.exp(-l2 * t) - both) / (load - l2)
        expected += s * l2 * (math.exp(-l1 * t) - both) / (load - l1)
        # The reference above is the plain difference-of-exponentials form,
        # which is well conditioned at these rates and is therefore a valid
        # independent check on the expm1 form the implementation uses.
        assert quantify(shared_load, t).reliability == rel(expected, 1e-13)

    def test_matches_the_markov_ode_solution(self, shared_load: Diagram) -> None:
        l1, l2, s = 4.0e-4, 5.0e-4, 0.5

        def rhs(_t: float, y: np.ndarray) -> list[float]:
            # y = [both alive, only unit 2 alive, only unit 1 alive]
            both, only2, only1 = y
            return [
                -s * (l1 + l2) * both,
                s * l1 * both - l2 * only2,
                s * l2 * both - l1 * only1,
            ]

        solution = solve_ivp(
            rhs, (0.0, MISSION), [1.0, 0.0, 0.0], method="DOP853", rtol=1e-12, atol=1e-14
        )
        reference = float(solution.y[:, -1].sum())
        assert quantify(shared_load, MISSION).reliability == rel(reference, 1e-9)

    def test_sharing_factor_one_is_a_parallel_pair(self) -> None:
        rate = 4.0e-4
        diagram = Diagram(
            "no benefit",
            SharedLoad(
                units=(exponential_block("a", rate), exponential_block("b", rate)),
                shared_rate_factor=1.0,
            ),
        )
        q = 1.0 - math.exp(-rate * MISSION)
        assert quantify(diagram, MISSION).reliability == rel(1.0 - q * q, 1e-12)

    def test_exactly_degenerate_denominator_gives_the_analytic_limit(self) -> None:
        # load = s(l1 + l2) equals l2 exactly when s = 0.5 and l1 = l2, so both
        # denominators vanish. The expm1 form is exact there.
        rate = 4.0e-4
        diagram = Diagram(
            "degenerate",
            SharedLoad(
                units=(exponential_block("a", rate), exponential_block("b", rate)),
                shared_rate_factor=0.5,
            ),
        )
        x = rate * MISSION
        assert quantify(diagram, MISSION).reliability == rel(math.exp(-x) * (1.0 + x), 1e-14)

    @pytest.mark.parametrize("epsilon", [1e-14, 1e-11, 1e-8, 1e-5, 1e-2])
    def test_near_degenerate_agrees_with_the_exact_limit_to_first_order(
        self, epsilon: float
    ) -> None:
        # The function is smooth through the degeneracy, so a relative
        # perturbation of epsilon in one rate must move the answer by O(epsilon)
        # and not by O(1) or by a jump. A switched-threshold implementation
        # fails this at the threshold; the expm1 form does not.
        rate = 4.0e-4
        at_zero = shared_load_reliability(rate, rate, 0.5, MISSION)
        perturbed = shared_load_reliability(rate, rate * (1.0 + epsilon), 0.5, MISSION)
        assert abs(perturbed - at_zero) < 10.0 * epsilon + 1e-15

    def test_zero_rates_never_fail(self) -> None:
        assert shared_load_reliability(0.0, 0.0, 0.5, MISSION) == 1.0


class TestNetwork:
    def test_bridge_pivotal_decomposition_matches_the_closed_form(
        self, bridge_network: Diagram
    ) -> None:
        from reliablock.bridge import bridge_reliability_equal

        assert quantify(bridge_network, 1.0).reliability == rel(
            bridge_reliability_equal(0.9), 1e-14
        )

    def test_bridge_unreliability_is_computed_in_q_space(self) -> None:
        from reliablock.bridge import bridge_diagram

        diagram = bridge_diagram(edge_unreliabilities={f"e{i}": 1e-6 for i in range(1, 6)},
                                 mission_hours=1.0)
        result = quantify(diagram, 1.0)
        # Leading order Q = 2 q^2 = 2e-12; recoverable only in Q space.
        assert result.unreliability == rel(2.0e-12, 1e-5)

    def test_esary_proschan_bounds_bracket_the_bridge(self, bridge_network: Diagram) -> None:
        result = quantify(bridge_network, 1.0)
        bracket = esary_proschan_bounds(bridge_network, 1.0, result.reliability)
        assert bracket.contains, str(bracket)
        # The bounds are not exact for the bridge, because its cut sets share
        # components. A zero-width bracket would mean the test proves nothing.
        assert bracket.width > 1e-4


class TestMissionAndRefusals:
    def test_bare_duration_is_a_single_phase(self, series_pair: Diagram) -> None:
        from reliablock.mission import MissionProfile

        assert quantify(series_pair, MISSION).reliability == rel(
            quantify(series_pair, MissionProfile.from_hours(MISSION)).reliability, 1e-15
        )

    def test_block_ages_override(self, series_pair: Diagram) -> None:
        result = quantify(series_pair, 0.0, block_ages={"a": 1000.0, "b": 0.0})
        assert result.reliability == rel(math.exp(-0.2), 1e-14)

    def test_missing_block_age_is_reported(self, series_pair: Diagram) -> None:
        with pytest.raises(KeyError, match="no effective age"):
            quantify(series_pair, 0.0, block_ages={"a": 1000.0})

    def test_shared_load_with_unequal_ages_is_refused(self, shared_load: Diagram) -> None:
        with pytest.raises(ValueError, match="one clock"):
            quantify(shared_load, 0.0, block_ages={"u1": 100.0, "u2": 200.0})

    def test_zero_duration_is_certain_success(self, series_pair: Diagram) -> None:
        result = quantify(series_pair, 0.0)
        assert result.reliability == 1.0
        assert result.unreliability == 0.0
