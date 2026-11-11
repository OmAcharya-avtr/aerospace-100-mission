"""Known-answer and edge-case tests for the provenance lattice.

The meet table is hand-written here, all nine cells, rather than computed from
the implementation's own rank encoding. That is the point: a test that derives
its expectation from the code under test checks that the code is
self-consistent and nothing else.
"""

from __future__ import annotations

import pytest

from boardprobe.provenance import (
    ABSORBING,
    IDENTITY,
    ORDER,
    Provenance,
    ProvenanceError,
    join,
    leq,
    meet,
    meet_all,
    rank,
)

LIVE = Provenance.LIVE
RECORDED = Provenance.RECORDED
SIMULATED = Provenance.SIMULATED

#: The meet table, written out by hand. Nine cells, no loops.
MEET_TABLE = {
    (LIVE, LIVE): LIVE,
    (LIVE, RECORDED): RECORDED,
    (LIVE, SIMULATED): SIMULATED,
    (RECORDED, LIVE): RECORDED,
    (RECORDED, RECORDED): RECORDED,
    (RECORDED, SIMULATED): SIMULATED,
    (SIMULATED, LIVE): SIMULATED,
    (SIMULATED, RECORDED): SIMULATED,
    (SIMULATED, SIMULATED): SIMULATED,
}

#: The join table, written out by hand.
JOIN_TABLE = {
    (LIVE, LIVE): LIVE,
    (LIVE, RECORDED): LIVE,
    (LIVE, SIMULATED): LIVE,
    (RECORDED, LIVE): LIVE,
    (RECORDED, RECORDED): RECORDED,
    (RECORDED, SIMULATED): RECORDED,
    (SIMULATED, LIVE): LIVE,
    (SIMULATED, RECORDED): RECORDED,
    (SIMULATED, SIMULATED): SIMULATED,
}


@pytest.mark.parametrize(("pair", "expected"), sorted(MEET_TABLE.items(), key=repr))
def test_meet_matches_the_hand_written_table(pair, expected):
    assert meet(*pair) is expected


@pytest.mark.parametrize(("pair", "expected"), sorted(JOIN_TABLE.items(), key=repr))
def test_join_matches_the_hand_written_table(pair, expected):
    assert join(*pair) is expected


def test_the_single_fact_the_product_exists_for():
    """LIVE and SIMULATED meet at SIMULATED. One live reading does not dilute."""
    assert meet(LIVE, SIMULATED) is SIMULATED
    assert meet_all([LIVE] * 99 + [SIMULATED]) is SIMULATED


def test_rank_is_the_declared_chain():
    assert (rank(SIMULATED), rank(RECORDED), rank(LIVE)) == (0, 1, 2)
    assert ORDER == (SIMULATED, RECORDED, LIVE)


def test_identity_and_absorbing_are_the_lattice_bounds():
    assert IDENTITY is LIVE
    assert ABSORBING is SIMULATED
    for value in ORDER:
        assert meet(IDENTITY, value) is value
        assert meet(ABSORBING, value) is ABSORBING


def test_leq_is_the_chain_order():
    assert leq(SIMULATED, RECORDED)
    assert leq(RECORDED, LIVE)
    assert leq(SIMULATED, LIVE)
    assert not leq(LIVE, SIMULATED)
    for value in ORDER:
        assert leq(value, value)


def test_meet_all_refuses_an_empty_collection():
    """The monoid identity is deliberately NOT returned for an empty fold."""
    with pytest.raises(ProvenanceError, match="empty collection"):
        meet_all([])


def test_meet_all_over_one_element_is_that_element():
    for value in ORDER:
        assert meet_all([value]) is value


@pytest.mark.parametrize("bad", ["live", 2, None, object()])
def test_a_non_member_is_refused_by_name(bad):
    with pytest.raises(ProvenanceError, match="boardprobe.Provenance member"):
        meet(LIVE, bad)
    with pytest.raises(ProvenanceError, match="boardprobe.Provenance member"):
        rank(bad)


def test_the_error_message_names_the_offending_index():
    with pytest.raises(ProvenanceError, match=r"values\[2\]"):
        meet_all([LIVE, RECORDED, "simulated"])


def test_str_is_the_lowercase_value():
    assert str(LIVE) == "live"
    assert str(SIMULATED) == "simulated"
