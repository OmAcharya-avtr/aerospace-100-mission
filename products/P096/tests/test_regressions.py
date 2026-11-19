"""Pinned counterexamples and pinned committed numbers.

Two kinds of test live here.

**Hypothesis counterexamples, pinned with literal inputs in the body.**
``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Each counterexample this product's property tests found is
reproduced here with the exact numbers, and each test says what the defect was.

**Regression and benchmark tests on the committed corpus.** Every number below
was produced by a script run in this session and is pinned with a tolerance, so
a change to the code that moves a published figure fails here rather than in a
reader's clone. No test in this file asserts a relationship between two live
wall-clock readings: the one timing assertion is a DIRECTION, and the measured
band is published in ``validation/VALIDATION.md`` rather than asserted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from handoverloss.accounting import attribute
from handoverloss.baselines import (
    acquisition_only_loss_bits,
    fluid_decomposition,
)
from handoverloss.corpus import CORPUS_MANIFEST, station_pair_split, time_split
from handoverloss.predictor import HandoverLossPredictor, score_baselines
from handoverloss.sequence import HandoverSequence
from handoverloss.trace import EventParameters, simulate

# --------------------------------------------------------------------------
# Pinned Hypothesis counterexamples
# --------------------------------------------------------------------------


def test_headroom_form_removes_the_cancellation_hypothesis_found():
    """Pinned counterexample: occupancy-space evaluation lost 9.96e-16 relative.

    Hypothesis falsified ``test_acquisition_term_equals_baseline_one`` on these
    literal inputs. ``fluid_decomposition`` evaluated ``(q0 + R_s*T_acq) - B``,
    a difference of two nearly equal large numbers, and returned
    ``1169156.284889914`` where baseline 1's ``R_s*T_acq - (B - q0)`` returned
    ``1169156.2848899153``. The fix was to evaluate the whole decomposition in
    HEADROOM space; see step 5 of the ``baselines`` module docstring. With that
    fix the two agree exactly on this input, and the occupancy-space value is
    recomputed here so the test shows the discrepancy it exists to prevent
    rather than merely asserting its absence.
    """
    sequence = HandoverSequence(
        source_rate_bps=1000000.0,
        downlink_rate_bps=1000000.0,
        buffer_capacity_bits=15608060.0,
        initial_occupancy_bits=15608060.0,
        acquisition_s=1.1691562848899153,
        restart_s=0.0,
        ramp_s=0.0,
        horizon_s=1.1691562848899153,
    )
    decomposition = fluid_decomposition(sequence)
    baseline_one = acquisition_only_loss_bits(sequence)
    # The headroom form and baseline 1 now use the same expression, so they
    # agree to the last bit on this input.
    assert decomposition.acquisition_bits == baseline_one
    # The occupancy form, recomputed here, still differs. This is the defect.
    occupancy_form = max(
        0.0,
        (sequence.initial_occupancy_bits
         + sequence.source_rate_bps * sequence.acquisition_s)
        - sequence.buffer_capacity_bits,
    )
    assert occupancy_form != baseline_one
    relative = abs(occupancy_form - baseline_one) / baseline_one
    assert 5.0e-16 < relative < 2.0e-15


def test_rebuilt_horizon_must_come_from_the_phases_not_the_old_horizon():
    """Pinned counterexample: adding to an existing horizon can round one ulp short.

    Hypothesis falsified ``test_a_longer_outage_never_reduces_the_total`` on
    these literal inputs. Rebuilding the window as ``horizon_s + extra_s``
    produced ``513.9816224284664`` while ``acquisition_s + extra_s + restart_s +
    ramp_s`` produced ``513.9816224284665``, so the constructor's bound refused
    the result. The constructor is right and the caller was wrong; this test
    pins both halves so the asymmetry is recorded rather than rediscovered.
    """
    acquisition = 0.0
    restart = 213.0
    ramp = 299.49081121423325
    horizon = 512.4908112142332
    extra = 1.490811214233247
    assert horizon + extra < acquisition + extra + restart + ramp
    with pytest.raises(ValueError, match="is shorter than"):
        HandoverSequence(
            source_rate_bps=1000000.0,
            downlink_rate_bps=1000000.0,
            buffer_capacity_bits=1000000.0,
            initial_occupancy_bits=0.0,
            acquisition_s=acquisition + extra,
            restart_s=restart,
            ramp_s=ramp,
            horizon_s=horizon + extra,
        )
    # Rebuilt from the phases, the same handover is accepted.
    rebuilt = HandoverSequence(
        source_rate_bps=1000000.0,
        downlink_rate_bps=1000000.0,
        buffer_capacity_bits=1000000.0,
        initial_occupancy_bits=0.0,
        acquisition_s=acquisition + extra,
        restart_s=restart,
        ramp_s=ramp,
        horizon_s=acquisition + extra + restart + ramp,
    )
    assert rebuilt.horizon_s == pytest.approx(
        acquisition + extra + restart + ramp, rel=0.0, abs=0.0
    )


# --------------------------------------------------------------------------
# Pinned committed numbers
# --------------------------------------------------------------------------


def test_corpus_manifest_is_the_one_the_published_numbers_came_from():
    declared = json.loads(Path(CORPUS_MANIFEST).read_text())
    assert declared["events"] == 1536
    assert declared["pairs"] == 16
    assert declared["epochs"] == 8
    assert declared["events_per_cell"] == 12
    assert declared["horizon_s"] == 700.0
    assert declared["slot_seconds"] == 0.25
    assert declared["slots_per_event"] == 2800
    assert declared["root_seed"] == 96_096_2026
    assert declared["sha256"] == (
        "3a0343a6c2c51a91eba0679856700f5495d8814c3bcdc94e6b710ba300c59706"
    )


def test_committed_dominance_counts(corpus):
    # Measured by data/generate_corpus.py this session and recorded in
    # DATASET_CARD.md. These are integer counts over a deterministic corpus, so
    # they are pinned exactly.
    assert corpus.dominance_counts() == {
        "acquisition": 762,
        "restart": 516,
        "drain": 258,
    }


def test_committed_baseline_scores_station_pairs(corpus):
    # Pinned from validation/validate_learned_vs_baselines.py, run this session.
    # Tolerances are on the bootstrap-free point estimates, which are
    # deterministic given the corpus; 1e-9 relative allows for summation order.
    scores = {s.predictor: s for s in score_baselines(corpus, station_pair_split(corpus),
                                                      resamples=100)}
    assert scores["baseline1_acquisition_only"].dominance_accuracy == pytest.approx(
        0.5364583333333334, rel=1e-9, abs=0.0
    )
    assert scores["baseline2_analytic_fluid"].dominance_accuracy == pytest.approx(
        0.6536458333333334, rel=1e-9, abs=0.0
    )


def test_committed_baseline_scores_time(corpus):
    scores = {s.predictor: s for s in score_baselines(corpus, time_split(corpus),
                                                      resamples=100)}
    assert scores["baseline1_acquisition_only"].dominance_accuracy == pytest.approx(
        0.5069444444444444, rel=1e-9, abs=0.0
    )
    assert scores["baseline2_analytic_fluid"].dominance_accuracy == pytest.approx(
        0.6822916666666666, rel=1e-9, abs=0.0
    )


def test_committed_worked_example_decomposition(worked_sequence):
    # The README's worked example. Hand arithmetic is in
    # tests/test_baselines.py::test_fluid_decomposition_known_answer.
    decomposition = fluid_decomposition(worked_sequence)
    assert decomposition.acquisition_bits == pytest.approx(5.0e9, rel=1e-14, abs=0.0)
    assert decomposition.restart_bits == pytest.approx(2.4e9, rel=1e-14, abs=0.0)
    assert decomposition.drain_bits == pytest.approx(1.2e9, rel=1e-14, abs=0.0)


def test_benchmark_simulation_throughput_direction_only():
    """A benchmark that asserts a DIRECTION, never a relationship between timings.

    This mission lost a whole product to checks comparing two live wall-clock
    readings on a shared two-core runner. So this test asserts only that the
    work completed, that every event produced a row, and that the elapsed time
    is strictly positive. The MEASURED band -- 128 events over 2800 slots in
    0.6 to 1.4 s on the two-core build machine under contention from four other
    agents -- is published in validation/VALIDATION.md and is deliberately NOT
    asserted here.
    """
    count = 128
    rng = np.random.default_rng(96_096)
    parameters = EventParameters(
        source_rate_bps=np.full(count, 2.0e8),
        downlink_rate_bps=np.full(count, 1.0e9),
        buffer_capacity_bits=np.full(count, 3.2e10),
        initial_occupancy_bits=np.full(count, 2.9e10),
        acquisition_s=np.linspace(5.0, 120.0, count),
        restart_s=np.full(count, 12.0),
        ramp_s=np.full(count, 60.0),
        horizon_s=700.0,
    )
    started = time.perf_counter()
    result = simulate(parameters, rng, slot_seconds=0.25)
    elapsed = time.perf_counter() - started
    assert elapsed > 0.0
    assert len(result) == count
    assert result.slot_count == 2800
    assert result.dropped_bits.shape == (count, 3)
    assert np.all(np.isfinite(result.dropped_bits))


def test_benchmark_training_completes_and_is_reproducible(corpus):
    """Regression on the learned component's cost and reproducibility.

    Direction-only on time, exact on the prediction, because the forest's seed
    is fixed. The measured training time is in validation/VALIDATION.md.
    """
    split = station_pair_split(corpus)
    started = time.perf_counter()
    model = HandoverLossPredictor(n_estimators=80).fit(
        corpus.features[split.train], corpus.targets[split.train]
    )
    elapsed = time.perf_counter() - started
    assert elapsed > 0.0
    first = model.predict(corpus.features[split.test]).loss_bits
    again = (
        HandoverLossPredictor(n_estimators=80)
        .fit(corpus.features[split.train], corpus.targets[split.train])
        .predict(corpus.features[split.test])
        .loss_bits
    )
    assert np.array_equal(first, again)


def test_committed_attribution_residual_magnitude(corpus):
    # Regression on the decomposition residual, which is the number this
    # product reports rather than assumes. Pinned as a BOUND, because the
    # residual depends on the summation order of the slot loop and a future
    # vectorisation would legitimately move it within that bound.
    index = np.arange(256)
    table = corpus.table
    parameters = EventParameters(
        source_rate_bps=table["source_rate_bps"][index],
        downlink_rate_bps=table["downlink_rate_bps"][index],
        buffer_capacity_bits=table["buffer_capacity_bits"][index],
        initial_occupancy_bits=table["initial_occupancy_bits"][index],
        acquisition_s=table["realised_acquisition_s"][index],
        restart_s=table["realised_restart_s"][index],
        ramp_s=table["declared_ramp_s"][index],
        horizon_s=700.0,
    )
    trace = simulate(parameters, np.random.default_rng(96_096))
    report = attribute(trace, resamples=200)
    assert abs(report.residual_relative) < 1.0e-12
    assert abs(report.residual_bits) <= report.residual_tolerance_bits
