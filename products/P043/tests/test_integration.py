"""End-to-end: link budget to slot statistics to acquisition to rate correction.

Also the regression tests. The pinned numbers below come from the validation
scripts in ``validation/`` and exist so that a change to the code that moves a
published number fails here rather than quietly in a README.
"""

from __future__ import annotations

import numpy as np
import pytest

from photoncount.capacity import (
    erasure_channel_capacity,
    hard_decision_capacity,
    minimum_photons_per_bit,
    photons_per_bit,
    soft_decision_achievable_rate,
)
from photoncount.correction import composed_baseline, matched_baseline
from photoncount.deadtime import paralyzable_maximum
from photoncount.hal import AcquisitionRequest, BackendMode, SimulatedBackend
from photoncount.ops import RunJournal, RunRecord, backout_incomplete_runs, run_preflight
from photoncount.poisson import mean_counts_from_power
from photoncount.ppm import PPMConfig, bit_llrs, hard_decision, sample_counts
from photoncount.simulate import DetectorSpec
from photoncount.webb import WebbParameters, moments


def test_link_budget_to_slot_statistics_to_decoder_input():
    """A received power becomes slot counts, then symbol decisions, then bit LLRs."""
    # 1 fW at 1550 nm into a 1 us slot at 50 % detection efficiency.
    n_s = mean_counts_from_power(1e-15, 1550e-9, 1e-6, 0.5)
    assert n_s == pytest.approx(0.0039014403, rel=1e-6)
    # Not enough for a symbol: push the slot energy up by shortening the duty cycle.
    cfg = PPMConfig(16, n_s * 1000.0, 0.05)
    rng = np.random.default_rng(0)
    sent = rng.integers(0, 16, size=4000)
    counts = sample_counts(cfg, sent, rng)
    decided = hard_decision(counts, cfg, rng)
    ser = float(np.mean(decided != sent))
    assert 0.0 < ser < 1.0
    llr = bit_llrs(counts, cfg)
    assert llr.shape == (4000, 4)
    assert np.all(np.isfinite(llr))


def test_capacity_chain_is_internally_consistent():
    cfg = PPMConfig(64, 3.0, 0.1)
    hard = hard_decision_capacity(cfg)
    soft = soft_decision_achievable_rate(cfg, 30_000, np.random.default_rng(2))
    assert soft["bits_per_symbol"] >= hard["bits_per_symbol"] - 4 * soft["standard_error"]
    ppb_hard = photons_per_bit(cfg, hard["bits_per_symbol"])
    ppb_soft = photons_per_bit(cfg, soft["bits_per_symbol"])
    assert ppb_soft < ppb_hard
    assert ppb_soft > minimum_photons_per_bit(64)


def test_full_operational_sequence_with_recovery(tmp_path):
    """Preflight, journal, acquire, commit; then a crashed run is backed out."""
    spec = DetectorSpec(
        dead_time_s=1e-7,
        model="paralyzable",
        afterpulse_probability=0.05,
        afterpulse_mean_delay_s=5e-7,
    )
    backend = SimulatedBackend(spec, 5e5, np.random.default_rng(13))
    request = AcquisitionRequest(1e-3, 60, "integration")
    report = run_preflight(backend, request, 5e5, tmp_path)
    assert report.passed is True

    journal = RunJournal(request.label, tmp_path)
    journal.begin({"window_s": request.window_s, "n_windows": int(request.n_windows)})
    with backend:
        acq = backend.acquire(request, BackendMode.SIMULATION)
    record = RunRecord.from_acquisition(acq, request, backend, report, seed=13)
    assert journal.commit(record) == "integration.json"

    # A second run that never finishes, then recovery.
    RunJournal("crashed", tmp_path).begin({"window_s": 1e-3})
    backed_out = backout_incomplete_runs(tmp_path)
    assert [row["label"] for row in backed_out] == ["crashed"]
    assert (tmp_path / "integration.json").exists()
    assert (tmp_path / "crashed.aborted.json").exists()


def test_acquisition_feeds_the_closed_form_corrections(tmp_path):
    """The HAL's observed rate goes straight into the baselines and comes back sane."""
    tau = 1e-7
    spec = DetectorSpec(
        dead_time_s=tau,
        model="nonparalyzable",
        afterpulse_probability=0.06,
        afterpulse_mean_delay_s=6e-7,
    )
    backend = SimulatedBackend(spec, 2e6, np.random.default_rng(21))
    request = AcquisitionRequest(1e-3, 120, "corr")
    with backend:
        acq = backend.acquire(request)
    m = np.array([acq.observed_rate_hz])
    taus = np.array([tau])
    matched = float(matched_baseline(m, taus, np.array([0.0]))[0])
    composed = float(composed_baseline(m, taus, np.array([0.0]), np.array([0.06]))[0])
    # The composed form removes the afterpulse inflation, so it sits lower.
    assert composed < matched
    # Both are within a factor of 1.3 of the truth at n*tau = 0.2.
    assert 0.7 < matched / 2e6 < 1.3
    assert 0.7 < composed / 2e6 < 1.3


# --- regression: pinned published numbers -----------------------------------


def test_regression_pinned_symbol_error_probability():
    """16-PPM, n_s = 2, no background: exp(-2)*15/16."""
    assert PPMConfig(16, 2.0).signal_counts == 2.0
    from photoncount.ppm import symbol_error_probability

    assert symbol_error_probability(PPMConfig(16, 2.0))["error"] == pytest.approx(
        0.12687682803433, rel=1e-12
    )


def test_regression_pinned_erasure_capacity():
    assert erasure_channel_capacity(PPMConfig(16, 2.0))["bits_per_symbol"] == pytest.approx(
        3.45865886705355, rel=1e-12
    )


def test_regression_pinned_paralyzable_maximum():
    n_max, m_max = paralyzable_maximum(1e-6)
    assert n_max == pytest.approx(1e6, rel=1e-15)
    assert m_max == pytest.approx(367879.441171442, rel=1e-12)


def test_regression_pinned_webb_moments():
    mom = moments(WebbParameters(100.0, 2.0))
    assert (mom["mean"], mom["variance"], mom["third_central"]) == pytest.approx(
        (100.0, 200.0, 600.0)
    )
    assert mom["skewness"] == pytest.approx(0.212132034355964, rel=1e-12)
