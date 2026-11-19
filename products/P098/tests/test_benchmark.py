"""The learned model against the baselines, on the committed corpus.

These are the product's regression tests on the published finding: the closure
comparison is exact on complete declarations and the learned model does not
beat it, on either split. The thresholds below are loose bands around the
measured values, not the measured values themselves, so that an incidental
change in scikit-learn's tie-breaking cannot fail the suite while a real
reversal of the finding would.
"""

from __future__ import annotations

import math

import pytest

from evidencestale.benchmark import THRESHOLD, confidence_table, run_benchmark


@pytest.fixture(scope="module")
def complete_benchmark(layered_corpus):
    """One benchmark run on the complete edge set, shared by this module."""
    return run_benchmark(layered_corpus, "complete")


@pytest.fixture(scope="module")
def partial_benchmark(layered_corpus):
    """One benchmark run on the declared subset, shared by this module."""
    return run_benchmark(layered_corpus, "partial")


def test_threshold_is_the_published_value() -> None:
    assert THRESHOLD == 0.5


def test_closure_comparison_is_exact_on_both_splits(complete_benchmark) -> None:
    for kind in ("artifact", "time"):
        summary = complete_benchmark.summary("baseline2_closure_hash", kind)
        assert summary["auc"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert summary["agreement"] == pytest.approx(1.0, rel=0.0, abs=0.0)
        assert summary["brier"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert summary["missed_staleness_rate"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert summary["false_alarm_rate"] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_the_learned_model_does_not_beat_the_exact_method(complete_benchmark) -> None:
    """The pre-registered expectation, asserted as the product's finding."""
    for kind in ("artifact", "time"):
        learned = complete_benchmark.summary("learned_forest", kind)
        exact = complete_benchmark.summary("baseline2_closure_hash", kind)
        assert learned["auc"] < exact["auc"]
        assert learned["agreement"] < exact["agreement"]
        assert learned["brier"] > exact["brier"]


def test_the_learned_model_does_beat_the_timestamp_baseline(complete_benchmark) -> None:
    for kind in ("artifact", "time"):
        learned = complete_benchmark.summary("learned_forest", kind)
        timestamp = complete_benchmark.summary("baseline1_timestamp", kind)
        assert learned["auc"] > timestamp["auc"]


def test_measured_learned_scores_are_in_the_published_bands(complete_benchmark) -> None:
    artifact = complete_benchmark.summary("learned_forest", "artifact")
    time = complete_benchmark.summary("learned_forest", "time")
    # Measured this session: 0.7152 held-out artifacts, 0.6977 held-out time.
    assert 0.60 < artifact["auc"] < 0.82
    assert 0.58 < time["auc"] < 0.80


def test_timestamp_baseline_is_near_or_below_chance(complete_benchmark) -> None:
    # Measured this session: 0.4510 held-out artifacts, 0.4460 held-out time.
    for kind in ("artifact", "time"):
        summary = complete_benchmark.summary("baseline1_timestamp", kind)
        assert summary["auc"] < 0.55


def test_partial_declarations_break_the_exactness(partial_benchmark) -> None:
    for kind in ("artifact", "time"):
        summary = partial_benchmark.summary("baseline2_closure_hash", kind)
        assert summary["auc"] < 1.0
        assert summary["missed_staleness_rate"] > 0.0
        # An undeclared edge hides a change; it cannot invent one.
        assert summary["false_alarm_rate"] == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_the_learned_model_does_not_beat_the_degraded_exact_method_either(
    partial_benchmark,
) -> None:
    """The only case in which the model could add anything, measured."""
    for kind in ("artifact", "time"):
        learned = partial_benchmark.summary("learned_forest", kind)
        degraded = partial_benchmark.summary("baseline2_closure_hash", kind)
        assert learned["auc"] < degraded["auc"]


def test_no_fold_is_degenerate_on_this_corpus(complete_benchmark) -> None:
    assert complete_benchmark.degenerate_folds == 0
    assert complete_benchmark.n_items == 480
    assert complete_benchmark.n_positive == 194


def test_eight_splits_are_scored_for_every_method(complete_benchmark) -> None:
    # Five grouped folds plus three time cutoffs.
    for method in complete_benchmark.methods():
        assert len(complete_benchmark.by(method, "artifact")) == 5
        assert len(complete_benchmark.by(method, "time")) == 3


def test_feature_importances_sum_to_one(complete_benchmark) -> None:
    total = sum(complete_benchmark.importances.values())
    assert total == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_fit_time_is_within_the_declared_budget(complete_benchmark) -> None:
    # A DIRECTION and a BOUND, never a ratio between two live timings: the
    # declared budget is 60 s of CPU for all fits in one benchmark run, and the
    # measured total this session was about 1.5 s on a shared two-core
    # container. Only the bound is asserted.
    assert complete_benchmark.fit_seconds > 0.0
    assert complete_benchmark.fit_seconds < 60.0


def test_confidence_table_bands_cover_every_item(layered_corpus) -> None:
    rows = confidence_table(layered_corpus)
    assert sum(row[2] for row in rows) == 480
    for low, high, count, agreement in rows:
        assert 0.5 <= low < high <= 1.0
        if count:
            assert 0.0 <= agreement <= 1.0
        else:
            assert math.isnan(agreement)


def test_leak_ablation_reaches_at_least_the_plain_model(layered_corpus) -> None:
    """Giving the model baseline 2's information does not make it worse.

    The ablation exists to show that the excluded feature is a leak rather than
    an oversight. It is not the product's model.
    """
    result = run_benchmark(layered_corpus, "complete", include_leak_ablation=True)
    for kind in ("artifact", "time"):
        plain = result.summary("learned_forest", kind)
        leak = result.summary("learned_leak_ablation", kind)
        assert leak["auc"] >= plain["auc"] - 1e-12
