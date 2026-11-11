"""Hypothesis property tests for every lattice law the module claims.

``(Provenance, meet)`` is claimed to be a commutative, idempotent monoid with
identity ``LIVE`` and absorbing element ``SIMULATED``, and
``(Provenance, meet, join)`` a distributive lattice. Each of those words is a
law and each law is checked here. Reference for the axioms: Davey and
Priestley, *Introduction to Lattices and Order*, 2nd edition, chapter 2.

The strategy is ``sampled_from`` over three elements, so these tests are
exhaustive in all but name for the pairwise laws and dense for the triples.
They are written as property tests rather than loops because the laws are what
is being asserted, and because the lattice is the one place in this package
where an algebraic identity exists to be falsified.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from boardprobe.provenance import (
    ABSORBING,
    IDENTITY,
    ORDER,
    join,
    leq,
    meet,
    meet_all,
    rank,
)

provenances = st.sampled_from(ORDER)
SETTINGS = settings(max_examples=200, deadline=None)


@SETTINGS
@given(provenances, provenances)
def test_meet_is_commutative(left, right):
    assert meet(left, right) is meet(right, left)


@SETTINGS
@given(provenances, provenances, provenances)
def test_meet_is_associative(first, second, third):
    assert meet(meet(first, second), third) is meet(first, meet(second, third))


@SETTINGS
@given(provenances)
def test_meet_is_idempotent(value):
    assert meet(value, value) is value


@SETTINGS
@given(provenances)
def test_live_is_the_identity_of_meet(value):
    assert meet(IDENTITY, value) is value
    assert meet(value, IDENTITY) is value


@SETTINGS
@given(provenances)
def test_simulated_absorbs(value):
    assert meet(ABSORBING, value) is ABSORBING
    assert meet(value, ABSORBING) is ABSORBING


@SETTINGS
@given(provenances, provenances, provenances)
def test_meet_is_monotone_in_each_argument(weaker, stronger, other):
    """If ``a <= b`` then ``meet(a, c) <= meet(b, c)``. Monotonicity is why
    propagation can never strengthen a provenance."""
    if not leq(weaker, stronger):
        weaker, stronger = stronger, weaker
    assert leq(meet(weaker, other), meet(stronger, other))


@SETTINGS
@given(provenances, provenances)
def test_meet_is_the_greatest_lower_bound(left, right):
    result = meet(left, right)
    assert leq(result, left) and leq(result, right)
    for candidate in ORDER:
        if leq(candidate, left) and leq(candidate, right):
            assert leq(candidate, result)


@SETTINGS
@given(provenances, provenances)
def test_the_absorption_laws_relate_meet_and_join(left, right):
    assert meet(left, join(left, right)) is left
    assert join(left, meet(left, right)) is left


@SETTINGS
@given(provenances, provenances, provenances)
def test_the_lattice_is_distributive(first, second, third):
    assert meet(first, join(second, third)) is join(
        meet(first, second), meet(first, third)
    )
    assert join(first, meet(second, third)) is meet(
        join(first, second), join(first, third)
    )


@SETTINGS
@given(st.lists(provenances, min_size=1, max_size=12))
def test_meet_all_is_the_weakest_element_present(values):
    assert meet_all(values) is min(values, key=rank)


@SETTINGS
@given(st.lists(provenances, min_size=1, max_size=12))
def test_meet_all_is_order_independent(values):
    assert meet_all(values) is meet_all(list(reversed(values)))


@SETTINGS
@given(st.lists(provenances, min_size=1, max_size=8))
def test_meet_all_is_idempotent_under_duplication(values):
    assert meet_all(values) is meet_all(values + values)


@SETTINGS
@given(st.lists(provenances, min_size=1, max_size=8))
def test_adding_a_simulated_reading_always_lands_at_simulated(values):
    assert meet_all([*values, ABSORBING]) is ABSORBING


@SETTINGS
@given(st.lists(provenances, min_size=1, max_size=8))
def test_adding_a_live_reading_never_changes_anything(values):
    assert meet_all([*values, IDENTITY]) is meet_all(values)
