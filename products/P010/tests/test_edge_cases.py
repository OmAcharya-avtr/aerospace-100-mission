"""Edge cases: extreme SNR, log-domain stability, tiny/huge inputs, CLI."""

import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from berbench import analytic_ber, log10_qfunc, log_qfunc, mc_ber, qfunc

ROOT = Path(__file__).resolve().parents[1]


class TestHighSNR:
    def test_very_high_snr_no_blowup(self):
        # 60 dB: Q(sqrt(2e6)) underflows cleanly to 0.0 — no NaN, no warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for mod in ("ook", "bpsk", "ppm"):
                ber = analytic_ber(mod, np.array([40.0, 60.0, 100.0])).ber
                assert np.all(np.isfinite(ber))
                assert np.all(ber >= 0.0)
                assert np.all(ber < 1e-30)

    def test_ppm_exact_no_cancellation_at_high_snr(self):
        # The naive 1 - prod form loses all precision near BER ~ 1e-17;
        # the expm1/log_ndtr form must stay positive and monotone.
        ber = analytic_ber("ppm", np.array([12.0, 14.0, 16.0]), M=4).ber
        assert np.all(ber > 0.0)
        assert np.all(np.diff(ber) < 0.0)

    def test_log_domain_q_stability(self):
        # Q underflows at x ~ 38.6; ln Q stays finite far beyond.
        x = 1414.2135  # sqrt(2 * 1e6), i.e. BPSK at 60 dB
        assert qfunc(x) == 0.0
        lq = log_qfunc(x)
        assert np.isfinite(lq)
        # asymptotic ln Q(x) ~ -x^2/2 - ln(x sqrt(2 pi))
        expected = -(x**2) / 2 - np.log(x * np.sqrt(2 * np.pi))
        assert lq == pytest.approx(expected, rel=1e-6)
        assert log10_qfunc(x) == pytest.approx(lq / np.log(10.0), rel=1e-12)

    def test_lognormal_high_snr_finite(self):
        ber = analytic_ber("ook", 50.0, channel="lognormal", sigma_i2=0.3).ber
        assert np.isfinite(ber[0]) and ber[0] >= 0.0

    def test_mc_high_snr_zero_errors_valid_ci(self):
        res = mc_ber("bpsk", 20.0, n=10_000, seed=0)
        assert res.n_errors[0] == 0
        assert res.ber[0] == 0.0
        assert res.ci_low[0] == 0.0 and 0.0 < res.ci_high[0] < 1e-3


class TestLowSNR:
    def test_deep_negative_snr_approaches_half(self):
        for mod in ("ook", "bpsk"):
            ber = analytic_ber(mod, -40.0).ber[0]
            assert 0.45 < ber <= 0.5

    def test_ppm_low_snr_bounded(self):
        # Pb <= M/(2(M-1)) * 1 for any SNR; must never exceed it.
        for m in (2, 4, 16):
            ber = analytic_ber("ppm", -30.0, M=m).ber[0]
            assert ber <= m / (2 * (m - 1)) + 1e-12
            assert ber > 0.3


class TestScalarVsArray:
    def test_scalar_input_gives_length_one_arrays(self):
        res = analytic_ber("bpsk", 5.0)
        assert res.ber.shape == (1,)
        res2 = analytic_ber("bpsk", [1.0, 2.0, 3.0])
        assert res2.ber.shape == (3,)


class TestCLI:
    def test_sweep_table(self, tmp_path):
        out = subprocess.run(
            [sys.executable, "-m", "berbench", "sweep", "--mod", "ook", "bpsk", "ppm",
             "--snr", "0:8:4", "--channel", "awgn"],
            capture_output=True, text=True, cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        assert out.returncode == 0, out.stderr
        assert "BER analytic" in out.stdout
        # 3 mods x 3 SNR points = 9 data rows
        assert sum(1 for ln in out.stdout.splitlines()
                   if ln.startswith(("ook", "bpsk", "ppm"))) == 9

    def test_sweep_png_and_mc(self, tmp_path):
        png = tmp_path / "sweep.png"
        out = subprocess.run(
            [sys.executable, "-m", "berbench", "sweep", "--mod", "bpsk",
             "--snr", "0:4:2", "--mc", "--n", "20000", "--seed", "7",
             "--png", str(png)],
            capture_output=True, text=True, cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin",
                 "MPLBACKEND": "Agg"},
        )
        assert out.returncode == 0, out.stderr
        assert png.exists() and png.stat().st_size > 5000
        assert "Wilson" in out.stdout

    def test_bad_snr_spec_fails_cleanly(self):
        out = subprocess.run(
            [sys.executable, "-m", "berbench", "sweep", "--mod", "ook",
             "--snr", "banana"],
            capture_output=True, text=True, cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        )
        assert out.returncode != 0
        assert "cannot parse" in out.stderr
