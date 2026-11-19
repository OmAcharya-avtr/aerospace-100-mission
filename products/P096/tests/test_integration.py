"""Integration: the whole chain from a declared sequence to a published verdict.

One test per end-to-end path a reader of the README would actually follow, so a
break anywhere between the sequence declaration and the attribution report fails
here rather than in a reader's clone.
"""

from __future__ import annotations

import numpy as np

from handoverloss import (
    AcquisitionModel,
    EventParameters,
    HandoverSequence,
    acquisition_only_loss_bits,
    attribute,
    fluid_decomposition,
    load_corpus,
    loss_bracket,
    simulate,
    station_pair_split,
    time_split,
)
from handoverloss.predictor import HandoverLossPredictor, score_baselines, score_learned
from handoverloss.sequence import MECHANISMS


def test_end_to_end_single_handover_three_baselines_and_the_bracket():
    """Declare a handover, run all three baselines on it, check the bracket.

    This is the README's first-run path. Baseline 1 and baseline 2 are closed
    forms on the declaration; baseline 3 is the simulated trace; the bracket is
    independent of all three.
    """
    sequence = HandoverSequence(
        source_rate_bps=2.0e8,
        downlink_rate_bps=1.0e9,
        buffer_capacity_bits=3.2e10,
        initial_occupancy_bits=2.9e10,
        acquisition_s=40.0,
        restart_s=12.0,
        ramp_s=60.0,
        horizon_s=700.0,
    )
    baseline_one = acquisition_only_loss_bits(sequence)
    baseline_two = fluid_decomposition(sequence)

    # Baseline 3: 60 realisations of the same declared sequence, so the trace's
    # randomness is what separates it from the closed form.
    parameters = EventParameters.from_sequences([sequence] * 60)
    trace = simulate(parameters, np.random.default_rng(96_096), slot_seconds=0.25)
    report = attribute(trace, resamples=400)

    # Baseline 1 understates the total, because it omits two mechanisms.
    assert baseline_one < baseline_two.total_bits
    # The ground truth's own total lies within a factor of two of the closed
    # form's, which is the validation of baseline 2 stated as an integration
    # assertion. The factor is loose because burstiness is not in the closed
    # form; the measured discrepancy over the whole corpus is in VALIDATION.md.
    assert 0.5 < report.total_loss_bits / baseline_two.total_bits < 2.0
    # The decomposition's residual is inside its conditioning-scaled tolerance.
    assert abs(report.residual_bits) <= report.residual_tolerance_bits
    # And the independently derived bracket contains the ground truth.
    bracket = loss_bracket(
        offered_bits=float(trace.produced_bits.mean()),
        downlink_rate_bps=sequence.downlink_rate_bps,
        outage_s=sequence.outage_s,
        ramp_s=sequence.ramp_s,
        horizon_s=sequence.horizon_s,
        buffer_capacity_bits=sequence.buffer_capacity_bits,
        initial_occupancy_bits=sequence.initial_occupancy_bits,
    )
    assert bracket.brackets(report.total_loss_bits)
    # The report names a dominant mechanism and says whether the data decide it.
    assert report.dominant in MECHANISMS
    assert isinstance(report.dominance_is_decided, bool)


def test_end_to_end_corpus_both_splits_baselines_before_the_model():
    """Load the corpus, score both baselines, then the model, on both splits.

    The order is the one the mission's build guide requires and the one this
    package's own API enforces by having ``score_baselines`` take no model.
    """
    corpus = load_corpus()
    for split in (station_pair_split(corpus), time_split(corpus)):
        baselines = score_baselines(corpus, split, resamples=200)
        assert [s.predictor for s in baselines] == [
            "baseline1_acquisition_only",
            "baseline2_analytic_fluid",
        ]
        learned, prediction, model = score_learned(
            corpus,
            split,
            predictor=HandoverLossPredictor(n_estimators=80),
            resamples=200,
        )
        assert model.fitted
        assert learned.split == split.name
        assert learned.n_test == split.test.size
        # Every scored predictor saw the same test events.
        assert {s.n_test for s in (*baselines, learned)} == {split.test.size}
        # The learned component exposes an uncertainty output, not a point
        # estimate, on every test event and every mechanism.
        assert prediction.log_sd.shape == (split.test.size, 3)
        assert float(prediction.log_sd.max()) > 0.0
        # Baseline 2 is the strongest of the three on this corpus. That is the
        # product's finding and it is asserted here so a regression that
        # reversed it would fail rather than be published silently.
        assert (
            baselines[1].dominance_accuracy > baselines[0].dominance_accuracy
        ), "baseline 2 must beat baseline 1; see README Validation evidence"


def test_end_to_end_acquisition_model_drives_the_decomposition():
    """The acquisition distribution, the sequence and the decomposition, in one chain.

    An operator's actual question: given this terminal's acquisition statistics,
    what does the handover cost and which mechanism dominates at the mean and at
    an upper quantile.
    """
    model = AcquisitionModel(
        median_s=25.0, log_sigma=0.55, abort_probability=0.18, attempt_window_s=120.0
    )
    mean_s, _ = model.moments()
    upper_s = model.quantile(0.95, size=60_000)
    assert upper_s > mean_s

    totals = []
    for duration in (mean_s, upper_s):
        sequence = HandoverSequence(
            source_rate_bps=3.0e8,
            downlink_rate_bps=6.0e8,
            buffer_capacity_bits=2.0e10,
            initial_occupancy_bits=1.6e10,
            acquisition_s=duration,
            restart_s=20.0,
            ramp_s=90.0,
            horizon_s=duration + 20.0 + 90.0 + 300.0,
        )
        totals.append(fluid_decomposition(sequence).total_bits)
    # A longer acquisition cannot cost less.
    assert totals[1] >= totals[0]
