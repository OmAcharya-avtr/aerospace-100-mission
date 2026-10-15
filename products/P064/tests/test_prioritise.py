"""Known-answer, validation and edge-case tests for Baseline 2."""

from __future__ import annotations

import pytest

from regressselect.history import History, Revision
from regressselect.prioritise import CoFailurePrioritiser


def _tiny_history() -> History:
    # Module "m" changed in 4 revisions; test "t1" failed in 3 of them,
    # "t2" in none. Module "n" changed in 1 revision with no failure.
    revs = (
        Revision("r1", ("m",), {"m": 10}, ("t1",)),
        Revision("r2", ("m",), {"m": 10}, ("t1",)),
        Revision("r3", ("m",), {"m": 10}, ("t1",)),
        Revision("r4", ("m",), {"m": 10}, ()),
        Revision("r5", ("n",), {"n": 10}, ()),
    )
    return History(test_ids=("t1", "t2"), revisions=revs)


def test_conditional_rate_hand_calculated():
    # alpha = 0.5, a(m, t1) = 3, b(m) = 4: (3 + 0.5) / (4 + 1) = 0.7
    pri = CoFailurePrioritiser().fit(_tiny_history())
    assert pri.conditional_rate("m", "t1") == pytest.approx(0.7, abs=1e-12)
    # a(m, t2) = 0: (0 + 0.5) / 5 = 0.1
    assert pri.conditional_rate("m", "t2") == pytest.approx(0.1, abs=1e-12)


def test_unseen_module_falls_back_to_the_base_rate():
    # base rate of t1: (3 + 0.5) / (5 + 1) = 0.5833333...
    pri = CoFailurePrioritiser().fit(_tiny_history())
    assert pri.conditional_rate("never_seen", "t1") == pytest.approx(3.5 / 6.0, abs=1e-12)


def test_score_is_the_maximum_over_changed_modules():
    pri = CoFailurePrioritiser().fit(_tiny_history())
    assert pri.score("t1", ["m", "n"]) == pytest.approx(
        max(pri.conditional_rate("m", "t1"), pri.conditional_rate("n", "t1")), abs=1e-12
    )


def test_rank_orders_the_co_failing_test_first():
    pri = CoFailurePrioritiser().fit(_tiny_history())
    order = [rt.node_id for rt in pri.rank(["m"])]
    assert order == ["t1", "t2"]


def test_rank_ties_break_on_the_node_id():
    revs = (Revision("r1", ("m",), {"m": 1}, ()),)
    pri = CoFailurePrioritiser().fit(History(("zz", "aa", "mm"), revs))
    assert [rt.node_id for rt in pri.rank(["m"])] == ["aa", "mm", "zz"]


def test_rank_is_a_permutation_of_the_suite(history, prioritiser):
    order = [rt.node_id for rt in prioritiser.rank(["sampleproj.quat"], list(history.test_ids))]
    assert sorted(order) == sorted(history.test_ids)


def test_scores_lie_in_the_unit_interval(history, prioritiser):
    for rt in prioritiser.rank(["sampleproj.vecops"], list(history.test_ids)):
        assert 0.0 < rt.score < 1.0


def test_empty_change_set_gives_the_base_rate_order():
    pri = CoFailurePrioritiser().fit(_tiny_history())
    assert pri.score("t1", []) == pytest.approx(pri.score("t1", ["unseen"]), abs=1e-12)


def test_unfitted_prioritiser_is_an_error():
    with pytest.raises(ValueError, match="not fitted"):
        CoFailurePrioritiser().rank(["m"])


def test_empty_history_is_rejected():
    with pytest.raises(ValueError, match="empty history"):
        CoFailurePrioritiser().fit(History(("t",), ()))


def test_non_positive_alpha_is_rejected():
    with pytest.raises(ValueError, match="alpha must be positive"):
        CoFailurePrioritiser(alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be positive"):
        CoFailurePrioritiser(alpha=-1.0)
