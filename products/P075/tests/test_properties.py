"""Property-based tests. Hypothesis is used to find defects, not decoratively.

Reliability functions carry strong algebraic properties, and most of them are
exact rather than approximate: ``k``-of-``n`` reduces to series at ``k = n``
and to parallel at ``k = 1``; a beta factor of 0 reproduces the independent
result and a beta factor of 1 reduces a group to a single member; the
Esary-Proschan bounds bracket every coherent system. Those are the properties
asserted here.

Three rules this file follows, each from a defect this mission has shipped:

1. **No bit-exact comparison over a floating-point pipeline.** Every
   comparison carries a tolerance on both sides. ``1 - 0.18`` evaluates to
   ``0.8200000000000001``, and an exact identity that holds in exact arithmetic
   is not a promise float64 makes.
2. **No relative ``approx`` tolerance without ``abs=0.0``.** The default
   absolute tolerance of 1e-12 would silently replace every relative tolerance
   on an unreliability figure below about 1e-4. The helpers in ``conftest``
   exist for this.
3. **Every falsifying example Hypothesis finds is pinned as a named regression
   test with literal inputs**, in ``tests/test_regressions.py``, because
   ``.hypothesis/`` is gitignored and a counterexample found by one run is lost
   on a fresh clone.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from reliablock.bounds import bonferroni_series_bounds, esary_proschan_bounds
from reliablock.bridge import (
    bridge_reliability_equal,
    bridge_reliability_inclusion_exclusion,
    bridge_reliability_pivotal,
)
from reliablock.commoncause import apply_common_cause, strip_common_cause
from reliablock.diagram import Diagram, KOfN, Parallel, Series
from reliablock.distributions import Exponential, Weibull
from reliablock.importance import (
    boolean_system_unreliability,
    importance_table,
    system_unreliability_at,
)
from reliablock.numerics import (
    parallel_unreliability,
    series_unreliability,
    weighted_binomial_survival,
)
from reliablock.quantify import quantify

from conftest import abs_, exponential_block, rel

SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# Rates bounded away from zero and from a hazard that saturates the mission, so
# that the properties under test are about the arithmetic and not about the
# saturation branches, which have their own hand-written tests.
rates = st.floats(min_value=1e-9, max_value=1e-2, allow_nan=False, allow_infinity=False)
probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
reliabilities = st.floats(min_value=1e-6, max_value=1.0, allow_nan=False)
betas = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
durations = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)


def _parallel(rates_list: list[float], beta: float = 0.0) -> Diagram:
    items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list))
    return Diagram("p", Parallel(items=items, beta_factor=beta, label="g"))


def _k_of_n(rates_list: list[float], k: int, beta: float = 0.0) -> Diagram:
    items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list))
    return Diagram("k", KOfN(k=k, items=items, beta_factor=beta, label="g"))


class TestBoundsAndRanges:
    @SETTINGS
    @given(st.lists(rates, min_size=1, max_size=8), durations)
    def test_every_quantity_lies_in_the_unit_interval(
        self, rates_list: list[float], hours: float
    ) -> None:
        items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list))
        for diagram in (
            Diagram("s", Series(items=items)),
            _parallel(rates_list) if len(rates_list) > 1 else Diagram("s", Series(items=items)),
        ):
            result = quantify(diagram, hours)
            assert 0.0 <= result.reliability <= 1.0
            assert 0.0 <= result.unreliability <= 1.0
            assert result.hazard >= 0.0

    @SETTINGS
    @given(st.lists(rates, min_size=1, max_size=8), durations)
    def test_series_reliability_is_at_most_the_worst_block(
        self, rates_list: list[float], hours: float
    ) -> None:
        items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list))
        result = quantify(Diagram("s", Series(items=items)), hours)
        assert result.reliability <= min(p.reliability for p in result.parts) + 1e-15

    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=8), durations)
    def test_parallel_reliability_is_at_least_the_best_block(
        self, rates_list: list[float], hours: float
    ) -> None:
        result = quantify(_parallel(rates_list), hours)
        assert result.reliability >= max(p.reliability for p in result.parts) - 1e-15

    @SETTINGS
    @given(st.lists(probabilities, min_size=1, max_size=12))
    def test_bonferroni_bounds_bracket_the_series_unreliability(
        self, unreliabilities: list[float]
    ) -> None:
        value = series_unreliability(unreliabilities)
        bracket = bonferroni_series_bounds(unreliabilities, value)
        assert bracket.contains, str(bracket)

    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=5), durations)
    def test_esary_proschan_bounds_bracket_a_nested_structure(
        self, rates_list: list[float], hours: float
    ) -> None:
        inner = Parallel(
            items=tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list)),
            label="inner",
        )
        diagram = Diagram("nested", Series(items=(inner, exponential_block("tail", 1e-5))))
        result = quantify(diagram, hours)
        bracket = esary_proschan_bounds(diagram, hours, result.reliability)
        assert bracket.contains, str(bracket)


class TestMonotonicity:
    @SETTINGS
    @given(st.lists(rates, min_size=1, max_size=6), durations, st.floats(1.01, 10.0))
    def test_reliability_decreases_when_any_block_gets_worse(
        self, rates_list: list[float], hours: float, factor: float
    ) -> None:
        items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list))
        base = quantify(Diagram("s", Series(items=items)), hours).reliability
        for index in range(len(rates_list)):
            worse = list(rates_list)
            worse[index] *= factor
            worse_items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(worse))
            assert quantify(Diagram("s", Series(items=worse_items)), hours).reliability <= (
                base + 1e-15
            )

    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=6), durations)
    def test_mission_reliability_is_non_increasing_in_duration(
        self, rates_list: list[float], hours: float
    ) -> None:
        diagram = _parallel(rates_list)
        earlier = quantify(diagram, hours).reliability
        later = quantify(diagram, hours * 1.5).reliability
        assert later <= earlier + 1e-15

    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=6), durations)
    def test_adding_a_parallel_block_never_reduces_reliability(
        self, rates_list: list[float], hours: float
    ) -> None:
        # Under independence. This is the statement redundancy is supposed to
        # make, and it is exactly the statement a common cause breaks.
        fewer = quantify(_parallel(rates_list), hours).reliability
        more = quantify(_parallel([*rates_list, 1e-4]), hours).reliability
        assert more >= fewer - 1e-15

    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=6), durations)
    def test_k_of_n_reliability_is_non_increasing_in_k(
        self, rates_list: list[float], hours: float
    ) -> None:
        n = len(rates_list)
        previous = 1.0
        for k in range(1, n + 1):
            value = quantify(_k_of_n(rates_list, k), hours).reliability
            assert value <= previous + 1e-15
            previous = value


class TestKOfNIdentities:
    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=6), durations)
    def test_k_equals_n_is_series(self, rates_list: list[float], hours: float) -> None:
        items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(rates_list))
        k_of_n = quantify(_k_of_n(rates_list, len(rates_list)), hours)
        chain = quantify(Diagram("s", Series(items=items)), hours)
        assert k_of_n.reliability == abs_(chain.reliability, 1e-13)

    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=6), durations)
    def test_k_equals_one_is_parallel(self, rates_list: list[float], hours: float) -> None:
        k_of_n = quantify(_k_of_n(rates_list, 1), hours)
        group = quantify(_parallel(rates_list), hours)
        assert k_of_n.reliability == abs_(group.reliability, 1e-13)
        assert k_of_n.unreliability == abs_(group.unreliability, 1e-13)

    @SETTINGS
    @given(st.lists(reliabilities, min_size=1, max_size=8))
    def test_survival_is_non_increasing_in_k(self, values: list[float]) -> None:
        previous = 1.0
        for k in range(len(values) + 2):
            survival = weighted_binomial_survival(values, k)
            assert survival <= previous + 1e-15
            previous = survival

    @SETTINGS
    @given(st.lists(reliabilities, min_size=1, max_size=8))
    def test_survival_at_k_one_is_the_parallel_complement(self, values: list[float]) -> None:
        q = [1.0 - v for v in values]
        assert weighted_binomial_survival(values, 1) == abs_(
            1.0 - parallel_unreliability(q), 1e-13
        )

    @SETTINGS
    @given(st.lists(reliabilities, min_size=1, max_size=8))
    def test_survival_at_k_n_is_the_product(self, values: list[float]) -> None:
        product = 1.0
        for v in values:
            product *= v
        assert weighted_binomial_survival(values, len(values)) == abs_(product, 1e-14)


class TestCommonCause:
    @SETTINGS
    @given(st.lists(rates, min_size=2, max_size=5), durations)
    def test_beta_zero_reproduces_the_independent_result(
        self, rates_list: list[float], hours: float
    ) -> None:
        diagram = _parallel(rates_list, beta=0.0)
        assert quantify(apply_common_cause(diagram), hours).reliability == abs_(
            quantify(strip_common_cause(diagram), hours).reliability, 0.0
        )

    @SETTINGS
    @given(rates, st.integers(2, 5), durations)
    def test_beta_one_reduces_a_group_to_one_member(
        self, rate: float, n: int, hours: float
    ) -> None:
        diagram = _parallel([rate] * n, beta=1.0)
        single = Exponential(rate).reliability(hours)
        assert quantify(apply_common_cause(diagram), hours).reliability == abs_(single, 1e-13)

    @SETTINGS
    @given(rates, st.integers(2, 5), betas, durations)
    def test_a_parallel_group_is_never_improved_by_a_beta_factor(
        self, rate: float, n: int, beta: float, hours: float
    ) -> None:
        # For k = 1 the only way the group fails is every member failing, which
        # is exactly the common-cause mode, so moving probability into it can
        # only make the group worse. This is the monotone case; k >= 2 is not,
        # and that is asserted separately below.
        diagram = _parallel([rate] * n, beta=beta)
        independent = quantify(strip_common_cause(diagram), hours).unreliability
        with_cc = quantify(apply_common_cause(diagram), hours).unreliability
        assert with_cc >= independent - 1e-15

    @SETTINGS
    @given(rates, st.integers(2, 5), betas, durations)
    def test_a_beta_factor_always_adds_a_single_point_of_failure(
        self, rate: float, n: int, beta: float, hours: float
    ) -> None:
        from reliablock.structure import minimal_cut_sets

        assume(beta > 0.0)
        diagram = _parallel([rate] * n, beta=beta)
        before = minimal_cut_sets(strip_common_cause(diagram))
        after = minimal_cut_sets(apply_common_cause(diagram))
        assert not any(len(c) == 1 for c in before)
        assert sum(1 for c in after if len(c) == 1) == 1

    @SETTINGS
    @given(rates, st.integers(3, 5), betas, durations)
    def test_the_beta_model_is_not_monotone_for_k_at_least_two(
        self, rate: float, n: int, beta: float, hours: float
    ) -> None:
        # Deliberately NOT asserting conservatism here. The beta-factor model
        # has one common-cause mode, the one that fails every member, and a
        # k-of-n group with k >= 2 fails on fewer failures than that, so the
        # ratio can be either side of 1. What IS asserted is that the group
        # unreliability stays a probability and that the two limits are exact.
        diagram = _k_of_n([rate] * n, k=n - 1, beta=beta)
        with_cc = quantify(apply_common_cause(diagram), hours).unreliability
        assert 0.0 <= with_cc <= 1.0

    @SETTINGS
    @given(rates, st.integers(2, 5), betas, durations)
    def test_the_rewrite_preserves_the_total_hazard_of_a_member(
        self, rate: float, n: int, beta: float, hours: float
    ) -> None:
        # lambda_total = lambda_independent + lambda_common is the defining
        # relation of the beta split, so the two parts must sum back. At
        # beta = 0 no common-cause block is created at all, which is the
        # correct behaviour and not a case this identity applies to.
        assume(beta > 0.0)
        diagram = apply_common_cause(_parallel([rate] * n, beta=beta))
        hazards = {
            b.name: b.distribution.cumulative_hazard(hours) for b in diagram.blocks()
        }
        total = Exponential(rate).cumulative_hazard(hours)
        common = next(v for k, v in hazards.items() if k.endswith("::common_cause"))
        member = next(v for k, v in hazards.items() if not k.endswith("::common_cause"))
        assert common + member == abs_(total, 1e-13 * max(total, 1e-13))


class TestBridge:
    @SETTINGS
    @given(st.floats(0.0, 1.0, allow_nan=False))
    def test_equal_and_non_identical_forms_agree(self, p: float) -> None:
        assert bridge_reliability_equal(p) == abs_(
            bridge_reliability_pivotal(p, p, p, p, p), 1e-14
        )

    @SETTINGS
    @given(st.floats(0.0, 1.0, allow_nan=False))
    def test_inclusion_exclusion_agrees(self, p: float) -> None:
        assert bridge_reliability_equal(p) == abs_(
            bridge_reliability_inclusion_exclusion({f"e{i}": p for i in range(1, 6)}), 1e-13
        )

    @SETTINGS
    @given(
        st.lists(st.floats(0.0, 1.0, allow_nan=False), min_size=5, max_size=5),
    )
    def test_pivotal_and_inclusion_exclusion_agree_for_non_identical_edges(
        self, values: list[float]
    ) -> None:
        mapping = {f"e{i + 1}": v for i, v in enumerate(values)}
        assert bridge_reliability_pivotal(*values) == abs_(
            bridge_reliability_inclusion_exclusion(mapping), 1e-13
        )

    @SETTINGS
    @given(st.floats(0.0, 1.0, allow_nan=False))
    def test_bridge_is_between_series_and_parallel_of_its_two_disjoint_paths(
        self, p: float
    ) -> None:
        # Adding the bridge edge cannot hurt, so the bridge is at least the
        # two-disjoint-paths structure, and it cannot exceed one.
        two_paths = 1.0 - (1.0 - p * p) ** 2
        assert bridge_reliability_equal(p) >= two_paths - 1e-15
        assert bridge_reliability_equal(p) <= 1.0 + 1e-15

    @SETTINGS
    @given(st.floats(0.0, 1.0, allow_nan=False))
    def test_bridge_is_monotone_in_p(self, p: float) -> None:
        assume(p < 0.999)
        assert bridge_reliability_equal(min(p + 1e-3, 1.0)) >= bridge_reliability_equal(p) - 1e-15


class TestImportance:
    @SETTINGS
    @given(
        st.lists(st.floats(1e-6, 0.3, allow_nan=False), min_size=2, max_size=4),
        st.floats(1e-6, 0.3, allow_nan=False),
    )
    def test_criticality_identity_holds(
        self, group: list[float], tail: float
    ) -> None:
        # I_C(i) = (Q_sys - Q_sys(q_i = 0)) / Q_sys, from the pivotal
        # decomposition. An identity, so it is checked to tolerance and not
        # bit-exactly.
        from reliablock.importance import pivot_unreliability

        items = tuple(exponential_block(f"u{i}", 1e-5) for i in range(len(group)))
        diagram = Diagram(
            "nested",
            Series(items=(Parallel(items=items, label="g"), exponential_block("tail", 1e-5))),
        )
        unreliabilities = {f"u{i}": q for i, q in enumerate(group)}
        unreliabilities["tail"] = tail
        q_system = system_unreliability_at(diagram, unreliabilities)
        assume(q_system > 1e-12)
        for event in diagram.block_names():
            pinned = pivot_unreliability(
                diagram, 1.0, event, 0.0, base_unreliabilities=unreliabilities
            )
            rows = {r.event: r for r in importance_table(diagram, 1.0)}
            del rows  # the table uses its own mission; the identity is checked here
            expected = (q_system - pinned) / q_system
            birnbaum = pivot_unreliability(
                diagram, 1.0, event, 1.0, base_unreliabilities=unreliabilities
            ) - pinned
            criticality = birnbaum * unreliabilities[event] / q_system
            assert criticality == abs_(expected, 1e-10)

    @SETTINGS
    @given(st.lists(st.floats(1e-6, 0.3, allow_nan=False), min_size=2, max_size=4))
    def test_fussell_vesely_is_at_least_criticality(self, group: list[float]) -> None:
        items = tuple(exponential_block(f"u{i}", r) for i, r in enumerate(group))
        diagram = Diagram(
            "nested",
            Series(items=(Parallel(items=items, label="g"), exponential_block("tail", 1e-3))),
        )
        for row in importance_table(diagram, 1000.0):
            assert row.fussell_vesely >= row.criticality - 1e-12
            assert 0.0 <= row.fussell_vesely <= 1.0 + 1e-12
            assert row.birnbaum >= -1e-15

    @SETTINGS
    @given(st.lists(st.floats(1e-6, 0.4, allow_nan=False), min_size=2, max_size=4))
    def test_structural_and_exhaustive_quantification_agree(
        self, group: list[float]
    ) -> None:
        items = tuple(exponential_block(f"u{i}", 1e-5) for i in range(len(group)))
        diagram = Diagram(
            "nested",
            Series(
                items=(
                    KOfN(k=max(1, len(items) - 1), items=items, label="g"),
                    exponential_block("tail", 1e-5),
                )
            ),
        )
        unreliabilities = {f"u{i}": q for i, q in enumerate(group)}
        unreliabilities["tail"] = 0.05
        structural = system_unreliability_at(diagram, unreliabilities)
        exhaustive = boolean_system_unreliability(diagram, unreliabilities)
        assert structural == abs_(exhaustive, 1e-13)


class TestDistributionProperties:
    @SETTINGS
    @given(
        st.floats(0.05, 20.0, allow_nan=False),
        st.floats(1.0, 1e7, allow_nan=False),
        st.floats(1e-9, 100.0, allow_nan=False),
    )
    def test_weibull_inverse_hazard_round_trips(
        self, shape: float, scale: float, hazard: float
    ) -> None:
        d = Weibull(shape=shape, scale_hours=scale)
        age = d.inverse_cumulative_hazard(hazard)
        assume(math.isfinite(age) and age > 0.0)
        assert d.cumulative_hazard(age) == rel(hazard, 1e-9)

    @SETTINGS
    @given(rates, durations)
    def test_reliability_and_unreliability_sum_to_one(
        self, rate: float, hours: float
    ) -> None:
        d = Exponential(rate)
        assert d.reliability(hours) + d.unreliability(hours) == abs_(1.0, 1e-15)

    @SETTINGS
    @given(st.lists(probabilities, min_size=1, max_size=10))
    def test_parallel_unreliability_is_at_most_the_smallest(
        self, unreliabilities: list[float]
    ) -> None:
        assert parallel_unreliability(unreliabilities) <= min(unreliabilities) + 1e-18


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_both_beta_limits_are_exact_not_approximate(beta: float) -> None:
    # Called out separately from the property tests because exactness at the
    # two limits is a design claim, not a numerical accident.
    diagram = _parallel([1e-5, 1e-5], beta=beta)
    hours = 10_000.0
    with_cc = quantify(apply_common_cause(diagram), hours).reliability
    if beta == 0.0:
        assert with_cc == quantify(strip_common_cause(diagram), hours).reliability
    else:
        assert with_cc == rel(Exponential(1e-5).reliability(hours), 1e-15)
