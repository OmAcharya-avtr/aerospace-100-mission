"""Suite integrity: every instance well formed, box shared, horizon adequate."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.instances import (
    SEARCH_BOX,
    SUITE_ORDER,
    TIERS,
    Instance,
    instance,
    suite,
)
from falsifyloop.requirements import Always, Predicate, Signal, robustness, satisfies
from falsifyloop.systems import LoopInput, LoopParameters


def test_suite_has_at_least_six_instances() -> None:
    # The specification requires at least six seeded benchmark instances.
    assert len(suite()) >= 6
    assert len(suite()) == len(SUITE_ORDER)


def test_suite_order_is_decreasing_in_design_difficulty_target() -> None:
    targets = [instance(i).design_target_probability for i in SUITE_ORDER]
    assert targets == sorted(targets, reverse=True)


def test_every_tier_label_is_declared() -> None:
    for inst in suite():
        assert inst.tier in TIERS


def test_suite_spans_at_least_three_tiers() -> None:
    assert len({inst.tier for inst in suite()}) >= 3


def test_every_instance_shares_the_declared_box() -> None:
    for inst in suite():
        np.testing.assert_allclose(inst.box, SEARCH_BOX)
        assert inst.dimension == len(LoopInput.FIELDS)


def test_box_geometry_known_answers() -> None:
    inst = instance("overshoot-loose")
    # step_amplitude row is [1, 6], so the centre is 3.5 and the width 5.
    assert inst.centre()[0] == pytest.approx(3.5)
    assert inst.widths()[0] == pytest.approx(5.0)
    np.testing.assert_allclose(inst.clip(np.full(6, 1e6)), inst.box[:, 1])
    np.testing.assert_allclose(inst.clip(np.full(6, -1e6)), inst.box[:, 0])


def test_sampling_stays_inside_the_box() -> None:
    rng = np.random.default_rng(0)
    points = instance("rate-envelope").sample(rng, 64)
    assert points.shape == (64, 6)
    assert np.all(points >= SEARCH_BOX[:, 0])
    assert np.all(points <= SEARCH_BOX[:, 1])


def test_sampling_is_reproducible_from_a_seed() -> None:
    inst = instance("settling-band")
    first = inst.sample(np.random.default_rng(7), 8)
    second = inst.sample(np.random.default_rng(7), 8)
    np.testing.assert_array_equal(first, second)


def test_sample_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        instance("settling-band").sample(np.random.default_rng(0), 0)


def test_evaluate_agrees_with_robustness_of_the_simulated_trace() -> None:
    inst = instance("overshoot-tight")
    vector = inst.centre()
    trace = inst.simulate(vector)
    assert inst.evaluate(vector) == pytest.approx(
        robustness(inst.requirement, trace), rel=0.0, abs=0.0
    )


def test_evaluate_sign_agrees_with_the_boolean_semantics_on_every_instance() -> None:
    rng = np.random.default_rng(11)
    for inst in suite():
        for vector in inst.sample(rng, 4):
            trace = inst.simulate(vector)
            rho = robustness(inst.requirement, trace)
            assert (rho < 0.0) == (not satisfies(inst.requirement, trace))


def test_every_requirement_horizon_fits_inside_the_simulated_trace() -> None:
    for inst in suite():
        assert inst.requirement.horizon() <= inst.horizon + 1e-9


def test_every_requirement_only_reads_signals_the_simulator_produces() -> None:
    for inst in suite():
        trace = inst.simulate(inst.centre())
        assert inst.requirement.signals() <= set(trace.names)


def test_describe_mentions_tier_requirement_and_box() -> None:
    text = instance("nested-capture").describe()
    assert "nested-capture" in text
    assert "hard" in text
    assert "eventually" in text
    assert "step_amplitude" in text


def test_unknown_instance_raises_keyerror_listing_the_suite() -> None:
    with pytest.raises(KeyError, match="unknown instance"):
        instance("no-such-instance")


def test_instances_are_finite_and_mostly_satisfied_at_the_box_centre() -> None:
    # Not a requirement of the design, but a sanity condition: an instance whose
    # centre already violates is not measuring search at all.
    violating = [i.identifier for i in suite() if i.evaluate(i.centre()) < 0.0]
    assert violating == [], f"instances violated at the box centre: {violating}"


def test_instance_constructor_validation() -> None:
    good = Always(Predicate(Signal("theta"), "<=", 100.0), 0.0, 1.0)
    with pytest.raises(ValueError, match="non-empty string"):
        Instance("", good, "r", "easy", 0.5)
    with pytest.raises(ValueError, match="tier must be one of"):
        Instance("x", good, "r", "trivial", 0.5)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        Instance("x", good, "r", "easy", 0.0)
    with pytest.raises(ValueError, match="box must have shape"):
        Instance("x", good, "r", "easy", 0.5, box=np.zeros((3, 2)))
    bad_box = SEARCH_BOX.copy()
    bad_box[0, 1] = bad_box[0, 0]
    with pytest.raises(ValueError, match="upper bound at or below"):
        Instance("x", good, "r", "easy", 0.5, box=bad_box)
    nonfinite = SEARCH_BOX.copy()
    nonfinite[0, 1] = np.inf
    with pytest.raises(ValueError, match="non-finite bound"):
        Instance("x", good, "r", "easy", 0.5, box=nonfinite)


def test_instance_rejects_a_requirement_longer_than_its_horizon() -> None:
    too_long = Always(Predicate(Signal("theta"), "<=", 100.0), 0.0, 10.0)
    with pytest.raises(ValueError, match="clipped or empty window"):
        Instance("x", too_long, "r", "easy", 0.5)


def test_instance_carries_the_declared_loop_parameters() -> None:
    assert instance("overshoot-loose").parameters == LoopParameters()
