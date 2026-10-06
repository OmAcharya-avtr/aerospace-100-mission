"""Fade statistics and the analytic level-crossing baseline.

Exercises REQ-15, REQ-16, REQ-17, REQ-18 of docs/REQUIREMENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special

from codedfade.channel import ChannelConfig, correlated_gaussian
from codedfade.fade import (
    exponential_exceedance,
    fade_runs,
    fade_statistics,
    lognormal_standard_level,
    markov_crossing_rate,
    markov_mean_fade_duration,
    required_interleaver_depth,
    rice_crossing_rate_gauss_kernel,
    rice_mean_fade_duration_gauss_kernel,
)


class TestFadeRuns:
    def test_known_answer(self) -> None:
        """below = 0 1 1 0 0 1 0 1 1 1 -> starts 1,5,7 lengths 2,1,3."""
        below = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
        starts, lengths = fade_runs(below)
        assert starts.tolist() == [1, 5, 7]
        assert lengths.tolist() == [2, 1, 3]

    def test_all_below_is_one_run(self) -> None:
        starts, lengths = fade_runs(np.ones(5, dtype=bool))
        assert starts.tolist() == [0]
        assert lengths.tolist() == [5]

    def test_none_below_is_no_runs(self) -> None:
        starts, lengths = fade_runs(np.zeros(5, dtype=bool))
        assert starts.size == 0 and lengths.size == 0

    def test_empty_input(self) -> None:
        starts, lengths = fade_runs(np.zeros(0, dtype=bool))
        assert starts.size == 0 and lengths.size == 0

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            fade_runs(np.zeros((2, 2), dtype=bool))


class TestFadeStatisticsDefinitions:
    def test_hand_computed_record(self) -> None:
        """a = [1, 0.5, 0.5, 1, 1, 0.5, 1], threshold 0.6, fs = 1000 Hz.

        below        = F T T F F T F
        down-crossings at index 1 and 5 -> 2 crossings over 7/1000 s
                     -> LCR = 2 / 0.007 = 285.714... s^-1
        runs: start 1 length 2 (complete), start 5 length 1 (complete)
        MFD = mean(2, 1) / 1000 = 1.5e-3 s
        outage = 3/7 = 0.428571...
        """
        a = np.array([1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 1.0])
        st = fade_statistics(a, 0.6, 1000.0)
        assert st.down_crossings == 2
        assert st.level_crossing_rate_hz == pytest.approx(2.0 / (7 / 1000.0))
        assert st.complete_fades == 2
        assert st.censored_fades == 0
        assert st.mean_fade_duration_s == pytest.approx(1.5e-3)
        assert st.median_fade_duration_s == pytest.approx(1.5e-3)
        assert st.max_fade_duration_s == pytest.approx(2e-3)
        assert st.outage_fraction == pytest.approx(3 / 7)

    def test_censored_runs_are_excluded(self) -> None:
        """A run touching either end of the record must not enter the MFD."""
        a = np.array([0.5, 0.5, 1.0, 0.5, 1.0, 0.5, 0.5])
        st = fade_statistics(a, 0.6, 1000.0)
        assert st.censored_fades == 2
        assert st.complete_fades == 1
        assert st.mean_fade_duration_s == pytest.approx(1e-3)

    def test_single_sample_fade_has_non_zero_duration(self) -> None:
        a = np.array([1.0, 0.5, 1.0])
        st = fade_statistics(a, 0.6, 2000.0)
        assert st.mean_fade_duration_s == pytest.approx(1 / 2000.0)

    def test_no_fades_gives_nan_durations(self) -> None:
        st = fade_statistics(np.ones(10), 0.5, 1000.0)
        assert st.complete_fades == 0
        assert np.isnan(st.mean_fade_duration_s)
        assert np.isnan(st.median_fade_duration_s)
        assert np.isnan(st.max_fade_duration_s)
        assert st.outage_fraction == 0.0

    @pytest.mark.parametrize(
        "a,thr,fs,msg",
        [
            (np.ones(1), 0.5, 1e3, "at least 2 samples"),
            (np.ones(5), 0.0, 1e3, "threshold must be > 0"),
            (np.ones(5), 0.5, 0.0, "sample_rate_hz must be > 0"),
        ],
    )
    def test_rejects_bad_inputs(
        self, a: np.ndarray, thr: float, fs: float, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            fade_statistics(a, thr, fs)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            fade_statistics(np.ones((2, 3)), 0.5, 1e3)

    def test_renewal_identity_holds_on_a_long_record(self, amplitude) -> None:
        """outage / LCR must equal the mean of ALL runs, so the sample MFD over
        complete runs only differs from it by the censoring correction."""
        st = fade_statistics(amplitude, 0.6, 1e6)
        renewal = st.outage_fraction / st.level_crossing_rate_hz
        assert renewal == pytest.approx(st.mean_fade_duration_s, rel=0.02)


class TestStandardLevel:
    def test_unit_amplitude_threshold_maps_to_half_sigma(self) -> None:
        """At a_th = 1, I_th = 1, so u = (0 + s^2/2)/s = s/2."""
        si = 0.6
        s = np.sqrt(np.log1p(si))
        assert lognormal_standard_level(1.0, si) == pytest.approx(s / 2)

    def test_monotone_in_threshold(self) -> None:
        values = [lognormal_standard_level(t, 0.6) for t in (0.3, 0.6, 0.9, 1.2)]
        assert values == sorted(values)

    @pytest.mark.parametrize(
        "thr,si,msg",
        [(0.0, 0.6, "threshold_amplitude"), (0.6, 0.0, "scintillation_index")],
    )
    def test_rejects_bad_inputs(self, thr: float, si: float, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            lognormal_standard_level(thr, si)


class TestAnalyticCrossingRates:
    def test_rice_formula_known_answer(self) -> None:
        """N(u) = sqrt(2)/(2 pi tau) exp(-u^2/2); at u = 0, tau = 1e-3 s:
        sqrt(2)/(2 pi 1e-3) = 225.0790... s^-1."""
        assert rice_crossing_rate_gauss_kernel(0.0, 1e-3) == pytest.approx(
            np.sqrt(2) / (2 * np.pi * 1e-3)
        )

    def test_rice_mfd_is_phi_over_rate(self) -> None:
        u, tau = -1.0, 1e-3
        expected = special.ndtr(u) / rice_crossing_rate_gauss_kernel(u, tau)
        assert rice_mean_fade_duration_gauss_kernel(u, tau) == pytest.approx(expected)

    def test_markov_rate_matches_a_monte_carlo_of_the_bivariate_normal(self) -> None:
        """Equation (14) against a direct draw of (g[n-1], g[n])."""
        u, tau, fs = -1.1474416982823366, 2e-4, 1e6
        rho = float(np.exp(-1.0 / (fs * tau)))
        rng = np.random.default_rng(11)
        n = 2_000_000
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        mc = float(np.mean((z1 >= u) & (z2 < u))) * fs
        assert markov_crossing_rate(u, tau, fs) == pytest.approx(mc, rel=0.02)

    def test_markov_rate_matches_a_generated_path(self) -> None:
        """Equation (14) against the AR(1) path the channel model actually uses."""
        u, lc = -1.1474416982823366, 200.0
        g = correlated_gaussian(2_000_000, lc, np.random.default_rng(12), "exp")
        below = g < u
        sample = float(np.mean(below[1:] & ~below[:-1]))
        assert markov_crossing_rate(u, lc, 1.0) == pytest.approx(sample, rel=0.03)

    def test_markov_rate_grows_with_sample_rate(self) -> None:
        """The OU process has no finite continuous-time crossing rate, so the
        discrete rate must keep rising as fs rises. This is a property of the
        kernel, asserted so that nobody later 'fixes' it."""
        u, tau = -1.0, 1e-3
        rates = [markov_crossing_rate(u, tau, fs) for fs in (1e4, 1e5, 1e6)]
        assert rates[0] < rates[1] < rates[2]

    def test_markov_mfd_shrinks_with_sample_rate(self) -> None:
        u, tau = -1.0, 1e-3
        mfds = [markov_mean_fade_duration(u, tau, fs) for fs in (1e4, 1e5, 1e6)]
        assert mfds[0] > mfds[1] > mfds[2]

    @pytest.mark.parametrize(
        "fn,args,msg",
        [
            (rice_crossing_rate_gauss_kernel, (0.0, 0.0), "correlation_time_s"),
            (markov_crossing_rate, (0.0, 0.0, 1e6), "correlation_time_s"),
            (markov_crossing_rate, (0.0, 1e-3, 0.0), "sample_rate_hz"),
        ],
    )
    def test_rejects_bad_inputs(self, fn, args, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            fn(*args)

    def test_very_deep_level_gives_infinite_mfd_rather_than_dividing_by_zero(
        self,
    ) -> None:
        assert np.isinf(rice_mean_fade_duration_gauss_kernel(-60.0, 1e-3))
        assert np.isinf(markov_mean_fade_duration(-60.0, 1e-3, 1e6))


class TestExceedanceAndDepth:
    def test_exponential_exceedance_at_the_mean(self) -> None:
        assert exponential_exceedance(1.0, 1.0) == pytest.approx(np.exp(-1.0))

    def test_exceedance_is_one_at_zero(self) -> None:
        assert exponential_exceedance(0.0, 2.0) == pytest.approx(1.0)

    def test_exceedance_accepts_arrays(self) -> None:
        out = exponential_exceedance(np.array([0.0, 1.0, 2.0]), 1.0)
        assert out.shape == (3,)
        assert out[0] > out[1] > out[2]

    @pytest.mark.parametrize(
        "t,mfd,msg", [(1.0, 0.0, "mean_fade_duration_s"), (-1.0, 1.0, "duration_s")]
    )
    def test_exceedance_rejects_bad_inputs(self, t, mfd, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            exponential_exceedance(t, mfd)

    def test_required_depth_known_answer(self) -> None:
        """MFD = 14 us at Rs = 1 Mbaud is 14 symbols; margin 3 gives 42."""
        assert required_interleaver_depth(14e-6, 1e6) == 14
        assert required_interleaver_depth(14e-6, 1e6, margin=3.0) == 42

    def test_required_depth_is_at_least_one(self) -> None:
        assert required_interleaver_depth(1e-12, 1e6) == 1

    @pytest.mark.parametrize(
        "mfd,rs,margin,msg",
        [
            (0.0, 1e6, 1.0, "mean_fade_duration_s"),
            (1e-5, 0.0, 1.0, "symbol_rate_hz"),
            (1e-5, 1e6, 0.0, "margin"),
        ],
    )
    def test_required_depth_rejects_bad_inputs(
        self, mfd: float, rs: float, margin: float, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            required_interleaver_depth(mfd, rs, margin)


class TestSampleAgainstAnalytic:
    def test_mean_fade_duration_agrees_with_the_exact_discrete_result(self) -> None:
        """Sample MFD on a 2e6-sample lognormal Gauss-Markov path against (15)."""
        cfg = ChannelConfig(0.6, 2e-4, 1e6, "lognormal", "exp", seed=21)
        from codedfade.channel import generate_amplitude

        a = generate_amplitude(cfg, 2_000_000)
        st = fade_statistics(a, 0.6, cfg.sample_rate_hz)
        u = lognormal_standard_level(0.6, cfg.scintillation_index)
        analytic = markov_mean_fade_duration(u, cfg.correlation_time_s, cfg.sample_rate_hz)
        assert st.mean_fade_duration_s == pytest.approx(analytic, rel=0.06)

    def test_level_crossing_rate_agrees_with_the_exact_discrete_result(self) -> None:
        cfg = ChannelConfig(0.6, 2e-4, 1e6, "lognormal", "exp", seed=21)
        from codedfade.channel import generate_amplitude

        a = generate_amplitude(cfg, 2_000_000)
        st = fade_statistics(a, 0.6, cfg.sample_rate_hz)
        u = lognormal_standard_level(0.6, cfg.scintillation_index)
        analytic = markov_crossing_rate(u, cfg.correlation_time_s, cfg.sample_rate_hz)
        assert st.level_crossing_rate_hz == pytest.approx(analytic, rel=0.05)
