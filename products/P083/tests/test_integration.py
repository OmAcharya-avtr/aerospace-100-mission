"""End-to-end: the whole pipeline on a population, and the benchmark budget."""

from __future__ import annotations

import time

import numpy as np
import pytest

from hilverdict.certificate import CertificateModel, certify_run
from hilverdict.features import FEATURE_NAMES, feature_matrix
from hilverdict.learned import MissClassifier
from hilverdict.plant import LoopModel
from hilverdict.resimulate import resimulate_run, simulate
from hilverdict.scoring import score_decision, threshold_at_recall
from hilverdict.synthetic import generate_population
from hilverdict.timeline import miss_events
from hilverdict.verdict import (
    NOT_TIMING_FAITHFUL,
    TIMING_FAITHFUL,
    UNDECIDABLE,
    UNDECIDED_RUN,
    run_verdict,
)

POPULATION = tuple(range(7000, 7009))


def _stratified_split(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alternate the positives and the negatives into two index arrays.

    A contiguous or strided split of this population puts every consequential
    miss on one side -- there are three of them in eighty-seven, and all three
    fall on odd indices -- so a fit would raise on a single-class training set.
    That is not a defect in the split, it is what a few per cent positive rate
    does to a small sample, and it is the reason the real study in
    ``validation/validate_learned_model.py`` splits by RUN over a population
    thirty times this size. Here the split is stratified so the integration
    path can be exercised at all.
    """
    positives = np.nonzero(labels)[0]
    negatives = np.nonzero(~labels)[0]
    train = np.concatenate([positives[0::2], negatives[0::2]])
    test = np.concatenate([positives[1::2], negatives[1::2]])
    return np.sort(train), np.sort(test)


@pytest.fixture(scope="module")
def population():
    return generate_population(POPULATION, steps=120)


@pytest.fixture(scope="module")
def labelled(population):
    features = []
    labels = []
    bounds = []
    for log in population:
        certificate = certify_run(log, 1.0)
        matrix, events = feature_matrix(log)
        if not events:
            continue
        results = resimulate_run(log, 1.0)
        features.append(matrix)
        labels.append(np.array([r.consequential for r in results], dtype=bool))
        bounds.append(np.array([c.bound for c in certificate.certificates]))
    return (
        np.vstack(features),
        np.concatenate(labels),
        np.concatenate(bounds),
    )


def test_every_run_in_the_population_gets_a_verdict(population):
    for log in population:
        verdict = run_verdict(log, 1.0)
        assert verdict.verdict in {
            TIMING_FAITHFUL, NOT_TIMING_FAITHFUL, UNDECIDED_RUN, UNDECIDABLE
        }


def test_no_declared_campaign_is_refused(population):
    for log in population:
        assert certify_run(log, 1.0).admissible is True


def test_the_certificate_dismisses_no_consequential_miss(labelled):
    _features, labels, bounds = labelled
    dismissed = bounds < 1.0
    assert int(np.count_nonzero(labels & dismissed)) == 0


def test_the_certificate_dismisses_most_misses(labelled):
    _features, _labels, bounds = labelled
    assert float(np.mean(bounds < 1.0)) > 0.5


def test_the_population_contains_both_classes(labelled):
    _features, labels, _bounds = labelled
    assert 0 < int(np.count_nonzero(labels)) < labels.shape[0]


def test_the_certificate_bound_dominates_the_resimulated_divergence(population):
    model = LoopModel()
    for log in population:
        certificate = certify_run(log, 1.0)
        results = resimulate_run(log, 1.0)
        for item, result in zip(certificate.certificates, results, strict=True):
            # The integrator floor is the only slack: a period split in two does
            # not integrate bit-identically to the same period integrated whole.
            assert result.peak_divergence <= item.bound + 1e-6
        assert model.metric.anisotropy > 1.0


def test_the_certificate_recall_is_one_on_the_population(labelled):
    _features, labels, bounds = labelled
    score = score_decision("certificate", bounds >= 1.0, labels, sound=True)
    assert score.counts.recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    assert score.missed_consequential == 0


def test_a_learned_model_fits_and_scores_on_the_population(labelled):
    features, labels, bounds = labelled
    train, test = _stratified_split(labels)
    classifier = MissClassifier().fit(features[train], labels[train])
    probability = classifier.predict_proba(features[test])
    assert probability.shape == (features[test].shape[0],)
    certificate = score_decision(
        "certificate", bounds[test] >= 1.0, labels[test], sound=True
    )
    assert certificate.counts.recall == pytest.approx(1.0, rel=0.0, abs=0.0)


def test_the_learned_threshold_transfers_without_a_guarantee(labelled):
    features, labels, _bounds = labelled
    train, test = _stratified_split(labels)
    classifier = MissClassifier().fit(features[train], labels[train])
    in_sample = classifier.predict_proba(features[train])
    threshold, recall, _flag = threshold_at_recall(in_sample, labels[train], 1.0)
    assert recall == pytest.approx(1.0, rel=0.0, abs=0.0)
    out_of_sample = classifier.predict_proba(features[test])
    score = score_decision("learned", out_of_sample >= threshold, labels[test])
    # The out-of-sample recall is reported, never asserted to be 1: that it may
    # be below 1 is the product's whole point.
    assert 0.0 <= score.counts.recall <= 1.0
    assert score.recall_interval[0] <= score.counts.recall <= score.recall_interval[1]


def test_the_residual_is_smaller_than_the_whole_miss_set(population):
    total = 0
    residual = 0
    for log in population:
        certificate = certify_run(log, 1.0)
        total += len(certificate.certificates)
        residual += len(certificate.undecided)
    assert 0 < residual < total


def test_resimulating_only_the_residual_costs_less(population):
    full = 0
    partial = 0
    for log in population:
        certificate = certify_run(log, 1.0)
        subset = tuple(item.event.index for item in certificate.undecided)
        full += sum(r.plant_calls for r in resimulate_run(log, 1.0))
        partial += sum(
            r.plant_calls for r in resimulate_run(log, 1.0, subset=subset)
        )
    assert partial < full


def test_the_aggregate_bound_dominates_the_all_misses_divergence(population):
    model = LoopModel()
    for log in population:
        certificate = certify_run(log, 1.0)
        schedule = {event.index: event.overrun_s for event in miss_events(log)}
        if not schedule:
            continue
        combined = simulate(model, log.command_rad, schedule)
        joint = float(np.max(model.metric.norm(combined.states - log.states)))
        assert joint <= certificate.aggregate_bound + 1e-6


def test_the_certificate_is_far_cheaper_than_the_reference(population):
    model = LoopModel()
    started = time.perf_counter()
    for log in population:
        CertificateModel.build(model, log.steps)
        certify_run(log, 1.0)
    certificate_s = time.perf_counter() - started
    started = time.perf_counter()
    for log in population:
        resimulate_run(log, 1.0)
    reference_s = time.perf_counter() - started
    # A deliberately loose budget: a tight wall-clock assertion on shared
    # hardware is a flaky test, and a flaky test is worse than no test. The
    # measured ratio during this build was about forty to one.
    assert certificate_s < reference_s


def test_the_whole_pipeline_runs_inside_a_loose_wall_clock_budget(population):
    started = time.perf_counter()
    for log in population:
        run_verdict(log, 1.0)
    elapsed = time.perf_counter() - started
    # About 100 times the measured cost on two cores, for the same reason.
    assert elapsed < 120.0


def test_the_feature_matrix_covers_every_miss_in_the_population(population):
    for log in population:
        matrix, events = feature_matrix(log)
        assert matrix.shape == (len(events), len(FEATURE_NAMES))
        assert len(events) == len(miss_events(log))


def test_the_verdict_agrees_with_its_own_parts(population):
    for log in population:
        verdict = run_verdict(log, 1.0)
        if verdict.certificate is None:
            continue
        assert verdict.misses == len(verdict.certificate.certificates)
        assert verdict.dismissed == len(verdict.certificate.dismissed)
        assert verdict.resimulated == len(verdict.resimulations)
