"""Integration and benchmark/regression tests, plus CLI smoke test."""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from jitterscope import (
    BandZScoreBaseline,
    FeatureExtractor,
    NominalModel,
    band_rms,
    cumulative_rms,
    detect,
    generate_telemetry,
    pointing_loss_avg,
    psd,
)

FS = 1000.0


def test_end_to_end_generate_fit_detect():
    """Full pipeline: generate nominal -> fit both models -> detect
    injected faults on a fresh record -> convert jitter to pointing loss."""
    _, x_nom, _ = generate_telemetry(60, FS, seed=100)
    ext = FeatureExtractor(fs=FS)
    feats, _ = ext.transform(x_nom)

    faults = [
        {"kind": "new_tone", "t_start": 15.0, "freq_hz": 222.0, "rms": 1.2e-6},
        {"kind": "transient", "t_start": 25.0, "rate_hz": 1.0, "amp": 8e-6},
    ]
    t, x_fault, mask = generate_telemetry(40, FS, seed=101, faults=faults)

    for model in (BandZScoreBaseline().fit(feats), NominalModel(seed=1).fit(feats)):
        res = detect(x_fault, model=model, extractor=ext)
        assert res.n_anomalous > 0
        # majority of flags inside labeled fault windows
        centers = res.window_centers_s[res.flags]
        label = np.array([mask[int(c * FS)] or mask[min(len(mask) - 1, int(c * FS) + 400)]
                          for c in centers])
        assert label.mean() > 0.5

    # jitter budget -> pointing loss
    f, pxx = psd(x_nom, FS)
    _, cum = cumulative_rms((f, pxx))
    sigma_total = cum[-1]
    loss = pointing_loss_avg(sigma_total / np.sqrt(2), 10e-6)  # per-axis sigma
    assert 0.0 < loss <= 1.0
    rms_bands = band_rms((f, pxx), [(0, 10), (10, 100), (100, 500)])
    # bands partition total variance approximately
    assert np.sqrt(np.sum(rms_bands**2)) == pytest.approx(sigma_total, rel=0.05)


def test_cli_analyze_smoke(tmp_path: Path):
    """CLI produces PSD stats and an anomaly report from a CSV."""
    t, x, _ = generate_telemetry(
        20, FS, seed=5,
        faults=[{"kind": "new_tone", "t_start": 15.0, "freq_hz": 300.0, "rms": 3e-6}],
    )
    csv = tmp_path / "telemetry.csv"
    np.savetxt(csv, np.column_stack([t, x]), delimiter=",", header="t,x")
    src = Path(__file__).resolve().parents[1] / "src"
    out = subprocess.run(
        [sys.executable, "-m", "jitterscope", "analyze", "--input", str(csv), "--fs", "1000"],
        capture_output=True, text=True, env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    assert "Band-limited RMS" in out.stdout
    assert "Anomaly report" in out.stdout
    assert "flagged" in out.stdout


def test_benchmark_runtime_budget():
    """Regression guard: full analyze pipeline on 60 s @ 1 kHz nominal
    + fit MLP + score 40 s record must run in < 30 s (2-CPU budget;
    typical ~3 s)."""
    start = time.perf_counter()
    _, x_nom, _ = generate_telemetry(60, FS, seed=200)
    ext = FeatureExtractor(fs=FS)
    feats, _ = ext.transform(x_nom)
    mdl = NominalModel(seed=0).fit(feats)
    _, x_test, _ = generate_telemetry(40, FS, seed=201)
    detect(x_test, model=mdl, extractor=ext)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0, f"pipeline took {elapsed:.1f} s (budget 30 s)"


def test_psd_throughput_benchmark():
    """PSD of 10^6 samples must complete in < 2 s (typically ~50 ms)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-6, 1_000_000)
    start = time.perf_counter()
    psd(x, FS)
    assert time.perf_counter() - start < 2.0
