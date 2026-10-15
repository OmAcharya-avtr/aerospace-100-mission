"""Property-based tests over random declarations and random calibration data.

Nothing here asserts bit-exact equality over a floating-point pipeline. Set
equality is asserted only where both sides are sets of discrete states derived
from the same integer-labelled graph; every cost comparison carries a tolerance
on both sides.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is
lost on a fresh clone. The boundary cases the strategies above can draw --
an empty precondition, a single prediction, a confidence of exactly 0.0, a
zero latency budget -- are therefore pinned at the bottom of this file as named
regression tests with literal inputs, so they are checked on every run and not
only when Hypothesis happens to draw them.

Hypothesis found NO falsifying example for any property in this file during the
build session of 2026-10-15; the pinned tests below are boundary cases pinned
deliberately, not counterexamples. If a future run does falsify a property, the
example goes in this section with its literal inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from safemodetree import (
    DetectionAssumption,
    Literal,
    Precondition,
    ThresholdLogicClassifier,
    accuracy_ceiling,
    ambiguity_classes,
    analyse,
    brier_score,
    build_automaton,
    exhaustive_min_cost_to_safe,
    exhaustive_safe_reachable,
    make_observations,
    min_cost_to_safe,
    random_declaration,
    reachability_report,
    reliability_curve,
    safe_reachable_set,
    symptom_word,
)

_SLOW = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_fixpoint_reachability_equals_brute_force(seed: int) -> None:
    """The backward fixpoint and exhaustive path enumeration agree exactly.

    Both sides are sets of discrete automaton states over the same transition
    relation, so set equality is exact and no tolerance is involved.
    """
    automaton = build_automaton(random_declaration(seed, n_modes=4, n_faults=3, n_responses=6))
    assert safe_reachable_set(automaton) == exhaustive_safe_reachable(automaton)


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_dijkstra_cost_equals_brute_force_within_tolerance(seed: int) -> None:
    """Cheapest-path costs agree to within rounding, and the supports match."""
    automaton = build_automaton(random_declaration(seed, n_modes=4, n_faults=3, n_responses=6))
    fixpoint = min_cost_to_safe(automaton)
    brute = exhaustive_min_cost_to_safe(automaton)
    assert set(fixpoint) == set(brute)
    for state, (cost, _path, _hops) in fixpoint.items():
        assert cost == pytest.approx(brute[state], rel=1e-9, abs=1e-9)


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_witness_path_cost_matches_its_declared_responses(seed: int) -> None:
    spec = random_declaration(seed)
    report = reachability_report(spec)
    for row in report.rows:
        if not row.reachable:
            continue
        declared = sum(spec.response(rid).cost_min for rid in row.path)
        assert row.cost_min == pytest.approx(declared, rel=1e-9, abs=1e-9)


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_safe_states_are_always_reachable_from_themselves(seed: int) -> None:
    automaton = build_automaton(random_declaration(seed))
    reachable = safe_reachable_set(automaton)
    for state in automaton.safe_states:
        assert state in reachable


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_perfect_detection_can_only_add_transitions(seed: int) -> None:
    """Relaxing the detection assumption never removes a reachable state."""
    spec = random_declaration(seed)
    strict = build_automaton(spec, DetectionAssumption())
    relaxed = build_automaton(spec, DetectionAssumption(require_detection=False))
    assert len(relaxed.transitions) >= len(strict.transitions)
    assert safe_reachable_set(strict) <= safe_reachable_set(relaxed)


@given(
    seed=st.integers(min_value=0, max_value=4000),
    budget=st.floats(min_value=0.0, max_value=400.0, allow_nan=False),
)
@_SLOW
def test_a_larger_latency_budget_never_removes_a_transition(seed: int, budget: float) -> None:
    spec = random_declaration(seed)
    tight = build_automaton(spec, DetectionAssumption(latency_budget_s=budget))
    loose = build_automaton(spec, DetectionAssumption(latency_budget_s=budget + 10.0))
    assert set(tight.transitions) <= set(loose.transitions)


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_conservative_route_is_valid_for_every_class_member(seed: int) -> None:
    """A reported conservative route keeps every class member safe-reachable."""
    spec = random_declaration(seed, n_channels=2, n_faults=4)
    automaton = build_automaton(spec)
    costs = min_cost_to_safe(automaton)
    for item in ambiguity_classes(spec, automaton=automaton):
        if item.conservative_response is None:
            continue
        response = spec.response(item.conservative_response)
        assert response.is_fault_agnostic
        for fault_id in item.faults:
            after = automaton.states[0].__class__(
                response.to_mode, None if response.clears_fault else fault_id
            )
            assert automaton.is_safe(after) or after in costs
        assert math.isfinite(item.conservative_cost_min)


@given(seed=st.integers(min_value=0, max_value=4000))
@_SLOW
def test_finding_counts_are_non_negative_and_consistent(seed: int) -> None:
    report = analyse(random_declaration(seed))
    counts = report.finding_counts
    assert all(value >= 0 for value in counts.values())
    assert report.total_findings == sum(counts.values())
    assert report.ok == (report.total_findings == 0)


@given(
    seed=st.integers(min_value=0, max_value=500),
    margin=st.floats(min_value=0.0, max_value=6.0, allow_nan=False),
    extra=st.floats(min_value=0.0, max_value=6.0, allow_nan=False),
)
@_SLOW
def test_symptom_bits_are_monotone_in_the_margin(seed: int, margin: float, extra: float) -> None:
    """Raising the margin can only turn symptom bits off, never on.

    This is monotonicity of ``margin(value) > m`` in ``m`` for a FIXED value,
    which holds exactly in floating point. It is not a claim that the bit is
    invariant under a rescaling of the channel, which would be a strict
    threshold comparison on a recomputed quantity and is not asserted anywhere.
    """
    spec = random_declaration(seed, n_channels=3)
    observations = make_observations(spec, n_per_fault=4, noise_scale=2.0, seed=seed)
    for row in observations.features:
        tight = np.array(symptom_word(spec, row, margin))
        loose = np.array(symptom_word(spec, row, margin + extra))
        assert np.all(loose <= tight)


@given(seed=st.integers(min_value=0, max_value=500))
@_SLOW
def test_threshold_logic_confidence_is_a_probability(seed: int) -> None:
    spec = random_declaration(seed, n_channels=3)
    classifier = ThresholdLogicClassifier(spec)
    observations = make_observations(spec, n_per_fault=6, noise_scale=2.0, seed=seed + 1)
    confidence = classifier.confidence(observations.features)
    assert np.all(confidence > 0.0)
    assert np.all(confidence <= 1.0 + 1e-12)


@given(seed=st.integers(min_value=0, max_value=500))
@_SLOW
def test_accuracy_ceiling_is_a_fraction(seed: int) -> None:
    spec = random_declaration(seed, n_channels=2, n_faults=4)
    observations = make_observations(spec, n_per_fault=5, seed=seed)
    ceiling = accuracy_ceiling(spec, observations)
    assert 0.0 < ceiling <= 1.0 + 1e-12


@given(
    confidence=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    ),
    bits=st.lists(st.booleans(), min_size=1, max_size=200),
    n_bins=st.integers(min_value=1, max_value=20),
)
@settings(max_examples=120, deadline=None)
def test_ece_is_bounded_and_weights_sum_to_one(
    confidence: list[float], bits: list[bool], n_bins: int
) -> None:
    size = min(len(confidence), len(bits))
    result = reliability_curve(
        np.array(confidence[:size]), np.array(bits[:size]), n_bins=n_bins
    )
    assert 0.0 - 1e-12 <= result.ece <= 1.0 + 1e-12
    assert result.mce >= result.ece - 1e-12
    assert int(result.bin_counts.sum()) == size
    assert result.accuracy == pytest.approx(float(np.mean(bits[:size])), abs=1e-12)


@given(
    rows=st.lists(
        st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
        min_size=1,
        max_size=50,
    ),
    label_bits=st.lists(st.booleans(), min_size=1, max_size=50),
)
@settings(max_examples=120, deadline=None)
def test_brier_score_is_in_range(rows: list[float], label_bits: list[bool]) -> None:
    size = min(len(rows), len(label_bits))
    proba = np.array([[p, 1.0 - p] for p in rows[:size]])
    labels = np.array(["a" if bit else "b" for bit in label_bits[:size]])
    score = brier_score(proba, labels, ("a", "b"))
    assert -1e-12 <= score <= 2.0 + 1e-12


@given(text=st.text(alphabet="abcXYZ_", min_size=1, max_size=6))
@settings(max_examples=100, deadline=None)
def test_literal_round_trip(text: str) -> None:
    if not text[0].isalpha():
        return
    positive = Literal.parse(text)
    assert str(positive) == text
    negative = Literal.parse(f"!{text}")
    assert negative.condition == text
    assert not negative.positive
    assert str(Literal.parse(str(negative))) == str(negative)


@given(
    names=st.lists(st.sampled_from(["k0", "k1"]), min_size=0, max_size=4),
    signs=st.lists(st.booleans(), min_size=0, max_size=4),
)
@settings(max_examples=100, deadline=None)
def test_precondition_emptiness_matches_a_direct_check(
    names: list[str], signs: list[bool]
) -> None:
    """The linear-time conflict check agrees with an explicit truth-table search."""
    spec = random_declaration(11, n_channels=2)
    size = min(len(names), len(signs))
    literals = tuple(
        Literal(name, positive=sign) for name, sign in zip(names[:size], signs[:size], strict=True)
    )
    precondition = Precondition(literals)
    for mode in spec.modes:
        satisfiable = False
        for assignment in ((False, False), (False, True), (True, False), (True, True)):
            values = {"k0": assignment[0], "k1": assignment[1]}
            feasible = all(
                (not values[cid]) or spec.condition(cid).can_hold_in(mode.id)
                for cid in ("k0", "k1")
            )
            if not feasible:
                continue
            if all(values[lit.condition] == lit.positive for lit in literals):
                satisfiable = True
                break
        assert spec.precondition_can_hold(precondition, mode.id) == satisfiable


# -- pinned boundary cases, with literal inputs ------------------------------


def test_regression_empty_precondition_is_satisfiable_in_every_mode() -> None:
    """Boundary case: the empty conjunction.

    An empty list of literals is the constant True and must be satisfiable in
    every declared mode, including an absorbing one. Pinned with literal inputs
    because it is the degenerate end of the strategy above.
    """
    spec = random_declaration(11, n_channels=2)
    for mode in spec.modes:
        assert spec.precondition_can_hold(Precondition(()), mode.id)


def test_regression_reliability_curve_with_one_prediction() -> None:
    """Boundary case: a single prediction with n_bins = 1.

    One prediction in one bin: the bin gap is the whole calibration error and
    MCE equals ECE. Pinned because a single-element bin is where an empty-bin
    guard is easiest to get wrong.
    """
    result = reliability_curve(np.array([0.0]), np.array([False]), n_bins=1)
    assert result.ece == pytest.approx(0.0, abs=1e-12)
    assert result.mce == pytest.approx(0.0, abs=1e-12)
    assert int(result.bin_counts[0]) == 1


def test_regression_confidence_zero_is_accepted() -> None:
    """Boundary case: a confidence of exactly 0.0, the lower bound.

    The input check uses ``confidence < 0.0``, not ``<= 0.0``, so 0.0 is valid
    and must land in the first bin rather than being rejected.
    """
    result = reliability_curve(np.array([0.0, 1.0]), np.array([False, True]), n_bins=2)
    assert int(result.bin_counts[0]) == 1
    assert int(result.bin_counts[1]) == 1
    assert result.ece == pytest.approx(0.0, abs=1e-12)


def test_regression_zero_latency_budget_leaves_agnostic_responses() -> None:
    """Boundary case: a latency budget of exactly 0.0 s.

    A zero budget makes every fault with a positive declared latency
    non-actionable, so the only fault-specific transitions that survive are
    those of faults whose declared latency is exactly zero. The automaton must
    still be built rather than raise.
    """
    spec = random_declaration(3)
    automaton = build_automaton(spec, DetectionAssumption(latency_budget_s=0.0))
    for transition in automaton.transitions:
        if transition.response.fault is not None:
            assert spec.fault(transition.response.fault).detection_latency_s == 0.0
