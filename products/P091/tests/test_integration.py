"""End-to-end: a generated campaign through the loader to the rendered report."""

from __future__ import annotations

import math

import numpy as np
import pytest

from thermalbudget import schema
from thermalbudget.baselines import BURST_SKIP_S
from thermalbudget.provenance import Origin
from thermalbudget.report import analyse, render

from fixtures import synthetic_trace as st


@pytest.mark.parametrize("regime", ["no_throttle", "throttled"])
def test_the_whole_chain_runs_in_both_declared_regimes(tmp_path, regime):
    campaign = st.load_synthetic_campaign(tmp_path, regime=regime)
    analysis = analyse(campaign, n_resamples=1000)
    text = render(analysis)
    assert analysis.origin is Origin.SYNTHETIC
    assert analysis.headline.verdict in ("INDISTINGUISHABLE", "OVERSTATES", "UNDERSTATES")
    assert len(text.splitlines()) > 60
    # Build order: every classical result exists before the learned one is used.
    assert analysis.baseline1 and analysis.baseline2 and analysis.baseline3
    assert analysis.verdict is not None


def test_the_headline_point_estimates_are_the_means_the_baselines_averaged(tmp_path):
    campaign = st.load_synthetic_campaign(tmp_path, regime="throttled")
    analysis = analyse(campaign, n_resamples=500, run_learned=False)
    burst_means = [
        float(np.mean(group)) for group in analysis.baseline1.throughput_groups
    ]
    assert analysis.headline.burst_iters_per_s == pytest.approx(
        float(np.mean(burst_means)), rel=1e-12, abs=0.0
    )
    sustained_means = [
        float(np.mean(group)) for group in analysis.baseline3.throughput_groups
    ]
    assert analysis.headline.sustained_iters_per_s == pytest.approx(
        float(np.mean(sustained_means)), rel=1e-12, abs=0.0
    )


def test_baseline_1_drops_exactly_the_declared_skip(tmp_path):
    campaign = st.load_synthetic_campaign(tmp_path, regime="no_throttle")
    analysis = analyse(campaign, n_resamples=200, run_learned=False)
    for trace, group in zip(
        campaign.bursts(), analysis.baseline1.throughput_groups, strict=True
    ):
        expected = trace.throughput_iters_per_s[trace.t_rel > BURST_SKIP_S]
        assert group.size == int(np.sum(np.isfinite(expected)))


def test_the_report_is_reproducible_from_the_seeds(tmp_path):
    first = analyse(
        st.load_synthetic_campaign(tmp_path / "a", regime="throttled"),
        n_resamples=800,
        run_learned=False,
    )
    second = analyse(
        st.load_synthetic_campaign(tmp_path / "b", regime="throttled"),
        n_resamples=800,
        run_learned=False,
    )
    assert first.headline.overstatement.point == second.headline.overstatement.point
    assert first.headline.overstatement.ci_low == second.headline.overstatement.ci_low
    assert first.baseline2.tau_rise_mean_s == second.baseline2.tau_rise_mean_s


def test_a_campaign_with_a_missing_data_run_still_completes(tmp_path):
    # soak_0.50 carries three sentinel rows; the whole chain must complete and
    # the missing census must survive to the loaded trace.
    campaign = st.load_synthetic_campaign(tmp_path, regime="throttled")
    assert campaign["soak_0.50"].missing[schema.DIE_COLUMN] == 3
    analysis = analyse(campaign, n_resamples=500)
    assert math.isfinite(analysis.headline.overstatement.point)


def test_the_equilibrium_criterion_propagates_into_baseline_3(tmp_path):
    campaign = st.load_synthetic_campaign(tmp_path, regime="no_throttle")
    # An absurdly tight criterion excludes everything, and the chain refuses
    # rather than averaging unequilibrated runs.
    with pytest.raises(ValueError, match="at least 2 equilibrated runs"):
        analyse(campaign, threshold_c_per_s=1e-12, n_resamples=200, run_learned=False)


def test_the_rendered_report_quotes_no_absolute_path(tmp_path):
    campaign = st.load_synthetic_campaign(tmp_path, regime="no_throttle")
    text = render(analyse(campaign, n_resamples=300, run_learned=False))
    # The report prints the trace directory by NAME only. An absolute path in
    # committed output fails release-gate check 8 and publishes the build
    # machine's filesystem layout.
    assert str(tmp_path) not in text
    assert campaign.directory.name in text
