"""End-to-end: a written campaign through every stage to a scored comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deadlinerisk.accounting import account, miss_indicator
from deadlinerisk.baselines import choose_threshold, mean_plus_margin, threshold_stability
from deadlinerisk.bootstrap import counted_rate_interval, understatement_factor
from deadlinerisk.bounds import bonferroni_pair_bracket, cantelli_slack, cantelli_upper_bound
from deadlinerisk.predictor import ConstantRatePredictor, ContentionMissPredictor, score
from deadlinerisk.provenance import Origin
from deadlinerisk.speed_scaling import BOARD_CORES, predicted_miss_probability
from deadlinerisk.trace import load_campaign
from deadlinerisk.windows import extract_windows, grouped_splits, time_split

from fixtures import synthetic_trace


def test_write_load_account_baseline_bootstrap_bound_and_score(tmp_path: Path) -> None:
    """The whole pipeline on a freshly written SYNTHETIC campaign.

    Nothing here measures any board. The assertions are structural: that every
    stage consumes the previous stage's output, that the three baselines produce
    comparable probabilities on the same run, that the bound holds, and that
    both splits score all three predictors.
    """
    directory = synthetic_trace.write_synthetic_campaign(tmp_path / "campaign", seed=31337)
    campaign = load_campaign(
        directory,
        origin=Origin.SYNTHETIC,
        require=("gt_long", "mix_c0", "mix_c6", "mix_mem2", "mix_c2mem1"),
        directory_label="integration",
    )
    assert campaign.complete is True
    assert campaign.provenance.origin is Origin.SYNTHETIC

    jobs = campaign.jobs["gt_long"]
    totals = account(jobs)
    assert totals.releases == totals.executed + totals.dropped
    assert 0.0 <= totals.miss_rate <= 1.0

    margin = mean_plus_margin(jobs.response_s, jobs.deadline_s)
    assert 0.0 <= margin.probability <= 1.0

    fits = threshold_stability(jobs.response_s, jobs.deadline_s)
    choice = choose_threshold(fits)
    assert 0.0 <= choice.chosen.probability <= 1.0
    assert choice.chosen.n_exceedances >= 30

    interval = counted_rate_interval(jobs, resamples=300, seed=5)
    assert interval.low <= interval.rate <= interval.high

    headline = understatement_factor(jobs, resamples=300, seed=5)
    assert headline.counted_rate == pytest.approx(
        totals.miss_rate, rel=1e-12, abs=0.0
    )
    assert headline.margin_probability == pytest.approx(
        margin.probability, rel=1e-12, abs=0.0
    )

    bound = cantelli_upper_bound(margin.mean_s, margin.sd_s, jobs.deadline_s)
    assert margin.probability <= bound + cantelli_slack(
        margin.mean_s, margin.sd_s, jobs.deadline_s
    )
    assert bonferroni_pair_bracket(miss_indicator(jobs)).inside

    windows = extract_windows(campaign)
    assert len(windows) >= 20
    idle = campaign.jobs["mix_c0"].exec_s

    grouped = grouped_splits(windows)
    assert len(grouped) == len(set(windows.runs.tolist()))
    label, train, test = grouped[0]
    model = ContentionMissPredictor().fit(
        windows.features[train], windows.targets[train], windows.feature_names
    )
    learned = score(
        f"learned/{label}",
        windows.targets[test],
        model.predict(windows.features[test], windows.releases[test]),
    )
    constant = ConstantRatePredictor().fit(windows.features[train], windows.targets[train])
    null = score(
        f"constant/{label}",
        windows.targets[test],
        constant.predict(windows.features[test]).rate,
    )
    classical = score(
        f"speed-scaling/{label}",
        windows.targets[test],
        np.asarray(
            [
                predicted_miss_probability(
                    idle, campaign.deadline_s, BOARD_CORES, int(workers)
                )
                for workers in windows.features[test, 2]
            ]
        ),
    )
    for result in (learned, null, classical):
        assert result.n_test == int(test.size)
        assert result.mae >= 0.0
        assert np.isfinite(result.rmse)

    train_t, test_t = time_split(windows, 0.3)
    assert train_t.size + test_t.size == len(windows)
    time_model = ContentionMissPredictor().fit(
        windows.features[train_t], windows.targets[train_t]
    )
    time_score = score(
        "learned/time",
        windows.targets[test_t],
        time_model.predict(windows.features[test_t], windows.releases[test_t]),
    )
    assert time_score.n_test == int(test_t.size)


def test_a_partial_campaign_still_supports_accounting(tmp_path: Path) -> None:
    """A campaign read mid-flight: fewer runs, no completion marker, same arithmetic."""
    directory = synthetic_trace.write_synthetic_campaign(
        tmp_path / "partial",
        plan=(
            synthetic_trace.RunSpec("gt_long", 4, 0, 30.0),
            synthetic_trace.RunSpec("mix_c0", 0, 0, 20.0),
        ),
        complete=False,
    )
    campaign = load_campaign(
        directory, origin=Origin.SYNTHETIC, directory_label="partial"
    )
    assert campaign.complete is False
    assert set(campaign.jobs) == {"gt_long", "mix_c0"}
    for jobs in campaign.jobs.values():
        totals = account(jobs)
        assert totals.releases > 0
