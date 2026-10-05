"""Probe-trace feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from latencynet.features import (
    DEPENDENCE_FEATURE_INDICES,
    FEATURE_NAMES,
    N_FEATURES,
    summarise_probe_trace,
)
from latencynet.pipeline import make_lognormal_pipeline, sample_stage_latencies


def test_feature_names_and_length_agree():
    assert len(FEATURE_NAMES) == N_FEATURES == 13
    assert len(set(FEATURE_NAMES)) == N_FEATURES
    assert all(0 <= i < N_FEATURES for i in DEPENDENCE_FEATURE_INDICES)


def test_summary_shapes(probe_trace):
    s = summarise_probe_trace(probe_trace)
    assert s.n_probe == 256
    assert s.n_stages == 3
    assert len(s.stage_mean_s) == len(s.stage_std_s) == len(s.stage_m4_s4) == 3
    assert len(s.features) == N_FEATURES
    assert s.covariance_matrix().shape == (3, 3)


def test_moment_estimates_match_numpy(probe_trace):
    s = summarise_probe_trace(probe_trace)
    assert np.allclose(s.stage_mean_s, probe_trace.mean(axis=0), rtol=1e-14)
    assert np.allclose(s.stage_std_s, probe_trace.std(axis=0, ddof=1), rtol=1e-14)
    assert np.allclose(
        s.covariance_matrix(), np.cov(probe_trace, rowvar=False, ddof=1), rtol=1e-13
    )


def test_known_answer_on_a_hand_built_trace():
    # Two stages, four passes, built so every feature is hand-checkable.
    #   stage 0: 1, 2, 3, 4   -> mean 2.5, sample sd sqrt(5/3) = 1.29099445
    #   stage 1: 2, 4, 6, 8   -> mean 5.0, sample sd sqrt(20/3) = 2.58198890
    # Perfectly correlated by construction, so the off-diagonal Pearson
    # correlation is exactly 1 and the dependence inflation is
    #   dep_sd / indep_sd = (1.29099445 + 2.58198890) / sqrt(5/3 + 20/3)
    #                     = 3.87298335 / 2.88675135 = 1.34164079.
    trace = np.tile(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]), (2, 1))
    s = summarise_probe_trace(trace)
    f = dict(zip(FEATURE_NAMES, s.features, strict=True))
    assert f["n_stages"] == 2.0
    assert np.exp(f["log_sum_mean"]) == pytest.approx(7.5, rel=1e-12)
    assert f["mean_offdiag_corr"] == pytest.approx(1.0, rel=1e-12)
    assert f["max_offdiag_corr"] == pytest.approx(1.0, rel=1e-12)
    assert f["dependence_inflation"] == pytest.approx(1.3416407864998738, rel=1e-10)
    # Perfect correlation means the dependence-aware sd is the sum of the
    # stage sds, and exceeds the independence sd.
    assert np.exp(f["log_dep_sd"]) > np.exp(f["log_indep_sd"])


def test_independent_trace_has_near_zero_correlation_features():
    spec = make_lognormal_pipeline((1e-4, 2e-4, 3e-4), (3e-5, 6e-5, 9e-5), latent_rho=0.0)
    s = summarise_probe_trace(sample_stage_latencies(spec, 4000, 13))
    f = dict(zip(FEATURE_NAMES, s.features, strict=True))
    # 4000 passes give a correlation standard error of 1/sqrt(4000) = 0.016.
    assert abs(f["mean_offdiag_corr"]) < 0.06
    assert f["dependence_inflation"] == pytest.approx(1.0, abs=0.05)


def test_correlated_trace_shows_inflation():
    spec = make_lognormal_pipeline((1e-4, 2e-4, 3e-4), (3e-5, 6e-5, 9e-5), latent_rho=0.8)
    s = summarise_probe_trace(sample_stage_latencies(spec, 4000, 14))
    f = dict(zip(FEATURE_NAMES, s.features, strict=True))
    assert f["mean_offdiag_corr"] > 0.5
    assert f["dependence_inflation"] > 1.4


def test_single_stage_correlation_features_are_zero():
    spec = make_lognormal_pipeline((1e-4,), (3e-5,))
    s = summarise_probe_trace(sample_stage_latencies(spec, 64, 15))
    f = dict(zip(FEATURE_NAMES, s.features, strict=True))
    assert f["mean_offdiag_corr"] == 0.0
    assert f["max_offdiag_corr"] == 0.0
    assert f["dependence_inflation"] == pytest.approx(1.0, rel=1e-12)


def test_features_are_finite_across_many_pipelines():
    for seed in range(12):
        rng = np.random.default_rng(seed)
        k = int(rng.integers(1, 7))
        means = np.exp(rng.uniform(np.log(2e-5), np.log(5e-4), size=k))
        spec = make_lognormal_pipeline(means, means * rng.uniform(0.05, 0.6, size=k))
        s = summarise_probe_trace(sample_stage_latencies(spec, 64, seed + 100))
        assert np.all(np.isfinite(s.features))


@pytest.mark.parametrize(
    ("trace", "match"),
    [
        (np.ones((4, 2)), "n_probe must be >= 8"),
        (np.ones(10), "must be 2-D"),
        (np.full((10, 2), np.nan), "finite"),
        (np.zeros((10, 2)), "strictly positive"),
    ],
)
def test_input_validation(trace, match):
    with pytest.raises(ValueError, match=match):
        summarise_probe_trace(trace)
