"""The learned component, its uncertainty output, and the orderings it must beat."""

from __future__ import annotations

import numpy as np
import pytest

from cmdguard.dataset import build_corpus, generate_clean_case
from cmdguard.features import FEATURE_NAMES
from cmdguard.model import Verity
from cmdguard.ranker import (
    RiskRanker,
    involved_mask,
    irreversibility_ordering,
    precision_at_k,
    random_ordering,
    recall_at_k,
    resource_draw_ordering,
)
from cmdguard.trace import forward_execute


@pytest.fixture(scope="module")
def corpus() -> tuple:
    return build_corpus(range(8), include_clean=False)


@pytest.fixture(scope="module")
def fitted(corpus: tuple) -> RiskRanker:
    return RiskRanker(n_estimators=40, seed=7).fit(
        [(entry.case, entry.report) for entry in corpus]
    )


def test_score_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="called before fit"):
        RiskRanker().score(generate_clean_case(0))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError, match="n_estimators=0"):
        RiskRanker(n_estimators=0)
    with pytest.raises(ValueError, match="max_depth=0"):
        RiskRanker(max_depth=0)
    with pytest.raises(ValueError, match="names no such feature"):
        RiskRanker(drop_features=("not_a_feature",))
    with pytest.raises(ValueError, match="remove every feature"):
        RiskRanker(drop_features=FEATURE_NAMES)


def test_fit_on_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="training set is empty"):
        RiskRanker().fit([])


def test_fit_on_single_class_raises() -> None:
    cases = [generate_clean_case(seed) for seed in range(3)]
    with pytest.raises(ValueError, match="single class"):
        RiskRanker().fit([(case, forward_execute(case)) for case in cases])


def test_metrics_are_recorded(fitted: RiskRanker, corpus: tuple) -> None:
    metrics = fitted.metrics
    assert metrics is not None
    assert metrics.n_cases == len(corpus)
    assert metrics.n_rows > 0
    assert 0.0 < metrics.positive_fraction < 1.0
    assert metrics.n_positive == int(round(metrics.positive_fraction * metrics.n_rows))
    assert metrics.features_used == FEATURE_NAMES
    assert metrics.features_dropped == ()
    assert set(metrics.feature_importance) == set(FEATURE_NAMES)
    assert sum(metrics.feature_importance.values()) == pytest.approx(1.0, abs=1e-9)
    assert metrics.fit_seconds > 0.0


def test_score_shape_and_range(fitted: RiskRanker) -> None:
    case = generate_clean_case(900)
    scores = fitted.score(case)
    assert scores.risk.shape == (len(case.sequence),)
    assert scores.spread.shape == scores.risk.shape
    assert ((scores.risk >= 0.0) & (scores.risk <= 1.0)).all()
    assert (scores.spread >= 0.0).all()


def test_score_is_deterministic(fitted: RiskRanker) -> None:
    case = generate_clean_case(901)
    first = fitted.score(case)
    second = fitted.score(case)
    assert np.array_equal(first.risk, second.risk)
    assert np.array_equal(first.spread, second.spread)


def test_two_fits_with_the_same_seed_agree(corpus: tuple) -> None:
    pairs = [(entry.case, entry.report) for entry in corpus]
    left = RiskRanker(n_estimators=24, seed=3).fit(pairs)
    right = RiskRanker(n_estimators=24, seed=3).fit(pairs)
    case = generate_clean_case(902)
    assert np.array_equal(left.score(case).risk, right.score(case).risk)


def test_spread_is_an_ensemble_dispersion_not_an_interval(fitted: RiskRanker) -> None:
    """Asserted as a property, because the docs claim nothing stronger."""
    case = generate_clean_case(903)
    scores = fitted.score(case)
    assert (scores.spread <= 0.5 + 1e-12).all()
    certain = (scores.risk < 1e-12) | (scores.risk > 1.0 - 1e-12)
    assert (scores.spread[certain] <= 1e-9).all()


def test_order_is_descending_and_ties_break_by_step(fitted: RiskRanker) -> None:
    case = generate_clean_case(904)
    scores = fitted.score(case)
    order = scores.order()
    values = scores.risk[order]
    assert (np.diff(values) <= 1e-15).all()
    for index in range(len(order) - 1):
        if values[index] == values[index + 1]:
            assert order[index] < order[index + 1]


def test_top_k_validation(fitted: RiskRanker) -> None:
    scores = fitted.score(generate_clean_case(905))
    with pytest.raises(ValueError, match="k=0"):
        scores.top_k(0)
    assert len(scores.top_k(3)) == 3


def test_risk_scores_shape_mismatch_rejected() -> None:
    from cmdguard.ranker import RiskScores

    with pytest.raises(ValueError, match="differ"):
        RiskScores("c", np.zeros(3), np.zeros(2))


def test_involved_mask_marks_the_violating_and_traced_steps(cases: dict) -> None:
    case = cases["late_payload_enable"]
    report = forward_execute(case)
    mask = involved_mask(report, len(case.sequence))
    assert mask.sum() == 2
    assert mask[5] and mask[6]


def test_precision_and_recall_at_k_hand_calculated() -> None:
    labels = np.array([True, False, True, False, False])
    order = [0, 1, 2, 3, 4]
    assert precision_at_k(order, labels, 2) == pytest.approx(0.5)
    assert precision_at_k(order, labels, 4) == pytest.approx(0.5)
    assert recall_at_k(order, labels, 2) == pytest.approx(0.5)
    assert recall_at_k(order, labels, 3) == pytest.approx(1.0)


def test_precision_at_k_is_nan_when_undefined() -> None:
    labels = np.array([True, False])
    assert np.isnan(precision_at_k([0, 1], labels, 5))
    assert np.isnan(precision_at_k([0, 1], np.array([False, False]), 1))
    assert np.isnan(recall_at_k([0, 1], np.array([False, False]), 1))


@pytest.mark.parametrize("function", [precision_at_k, recall_at_k])
def test_k_must_be_positive(function) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        function([0], np.array([True]), 0)


def test_irreversibility_ordering_puts_irreversible_first() -> None:
    case = generate_clean_case(906)
    order = irreversibility_ordering(case)
    flags = [
        case.dictionary.commands[case.sequence.steps[i].command].irreversible
        for i in order
        if case.sequence.steps[i].command in case.dictionary
    ]
    assert flags == sorted(flags, reverse=True)


def test_resource_draw_ordering_is_descending_in_draw_fraction() -> None:
    from cmdguard.features import feature_matrix

    case = generate_clean_case(907)
    column = FEATURE_NAMES.index("draw_fraction_of_ceiling")
    values = feature_matrix(case)[resource_draw_ordering(case), column]
    assert (np.diff(values) <= 1e-15).all()


def test_random_ordering_is_a_permutation_and_seeded() -> None:
    case = generate_clean_case(908)
    first = random_ordering(case, 5)
    second = random_ordering(case, 5)
    assert sorted(first) == list(range(len(case.sequence)))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, random_ordering(case, 6))


def test_orderings_on_an_empty_sequence() -> None:
    from conftest import tiny_case

    case = tiny_case([])
    assert irreversibility_ordering(case).size == 0
    assert resource_draw_ordering(case).size == 0
    assert random_ordering(case, 1).size == 0


def test_dropping_features_changes_the_design_matrix(corpus: tuple) -> None:
    pairs = [(entry.case, entry.report) for entry in corpus]
    ablated = RiskRanker(n_estimators=24, seed=3, drop_features=("dt_prev_s",)).fit(pairs)
    assert ablated.feature_names == tuple(n for n in FEATURE_NAMES if n != "dt_prev_s")
    assert ablated.metrics is not None
    assert ablated.metrics.features_dropped == ("dt_prev_s",)
    assert "dt_prev_s" not in ablated.metrics.feature_importance


def test_score_on_empty_sequence_returns_empty(fitted: RiskRanker) -> None:
    from conftest import tiny_case

    scores = fitted.score(tiny_case([]))
    assert scores.risk.size == 0


def test_verity_import_is_used() -> None:
    """Guard against an unused import slipping past ruff in this file."""
    assert Verity.TRUE.value == "true"
