"""Unit / known-answer / property tests for linkswitch.rf."""


import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linkswitch.rf import (
    RFParams,
    rain_markov_transition_probs,
    rain_specific_attenuation_db_per_km,
    rf_path_attenuation_db,
    simulate_rain_indicator,
)


class TestMarkovTransitionProbs:
    def test_known_answer(self):
        # p_rain=0.1, mean_event_steps=5
        # p_rain_to_clear = 1/5 = 0.2
        # p_clear_to_rain = 0.1 * 0.2 / 0.9 = 0.022222...
        p_c2r, p_r2c = rain_markov_transition_probs(0.1, 5.0)
        assert p_r2c == pytest.approx(0.2, rel=1e-12)
        assert p_c2r == pytest.approx(0.1 * 0.2 / 0.9, rel=1e-12)

    def test_stationary_distribution_matches_p_rain(self):
        # For a 2-state chain, pi_rain = p_c2r / (p_c2r + p_r2c); verify this
        # reproduces the requested p_rain for several parameter combinations.
        for p_rain, mean_len in [(0.05, 10.0), (0.2, 3.0), (0.5, 2.0), (0.01, 50.0)]:
            p_c2r, p_r2c = rain_markov_transition_probs(p_rain, mean_len)
            pi_rain = p_c2r / (p_c2r + p_r2c)
            assert pi_rain == pytest.approx(p_rain, rel=1e-9)

    def test_p_rain_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            rain_markov_transition_probs(0.0, 5.0)
        with pytest.raises(ValueError):
            rain_markov_transition_probs(1.0, 5.0)
        with pytest.raises(ValueError):
            rain_markov_transition_probs(-0.1, 5.0)

    def test_mean_event_steps_below_one_rejected(self):
        with pytest.raises(ValueError):
            rain_markov_transition_probs(0.1, 0.5)

    def test_infeasible_combo_rejected(self):
        # p_rain close to 1 with a very short mean event forces
        # p_clear_to_rain > 1, which is not a valid probability.
        with pytest.raises(ValueError):
            rain_markov_transition_probs(0.999, 1.0)


class TestSimulateRainIndicator:
    def test_length_and_dtype(self):
        rng = np.random.default_rng(0)
        out = simulate_rain_indicator(rng, 200, p_rain=0.1, mean_event_steps=5.0)
        assert out.shape == (200,)
        assert out.dtype == bool

    def test_long_run_fraction_matches_p_rain(self):
        rng = np.random.default_rng(1)
        p_rain = 0.08
        out = simulate_rain_indicator(rng, 200_000, p_rain=p_rain, mean_event_steps=10.0)
        assert out.mean() == pytest.approx(p_rain, abs=0.01)

    def test_mean_event_length_matches_config(self):
        # Measure mean run-length of True segments and compare to the
        # requested mean_event_steps (geometric holding time).
        rng = np.random.default_rng(2)
        mean_event_steps = 8.0
        out = simulate_rain_indicator(rng, 400_000, p_rain=0.1, mean_event_steps=mean_event_steps)
        # find run lengths of True
        diffs = np.diff(np.concatenate(([0], out.astype(int), [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        run_lengths = ends - starts
        assert run_lengths.mean() == pytest.approx(mean_event_steps, rel=0.15)

    def test_seeded_reproducibility(self):
        a = simulate_rain_indicator(np.random.default_rng(9), 100, 0.1, 5.0)
        b = simulate_rain_indicator(np.random.default_rng(9), 100, 0.1, 5.0)
        np.testing.assert_array_equal(a, b)

    def test_n_steps_must_be_positive(self):
        with pytest.raises(ValueError):
            simulate_rain_indicator(np.random.default_rng(0), 0, 0.1, 5.0)


class TestSpecificAttenuation:
    def test_known_answer(self):
        # gamma_R = k * R^alpha; k=0.07, alpha=1.1, R=10 mm/hr
        # 0.07 * 10^1.1 = 0.07 * 12.589254117941675 = 0.8812477882559173
        got = rain_specific_attenuation_db_per_km(10.0, k=0.07, alpha=1.1)
        assert got == pytest.approx(0.8812477882559173, rel=1e-10)

    def test_zero_rain_rate_zero_attenuation(self):
        assert rain_specific_attenuation_db_per_km(0.0, k=0.07, alpha=1.1) == pytest.approx(0.0)

    def test_vectorised(self):
        r = np.array([0.0, 5.0, 10.0, 50.0])
        out = rain_specific_attenuation_db_per_km(r, k=0.07, alpha=1.1)
        assert out.shape == (4,)
        assert np.all(np.diff(out) > 0.0)  # monotone increasing in rain rate

    def test_negative_rate_rejected(self):
        with pytest.raises(ValueError):
            rain_specific_attenuation_db_per_km(-1.0, k=0.07, alpha=1.1)

    def test_nonpositive_k_rejected(self):
        with pytest.raises(ValueError):
            rain_specific_attenuation_db_per_km(10.0, k=0.0, alpha=1.1)

    def test_nonpositive_alpha_rejected(self):
        with pytest.raises(ValueError):
            rain_specific_attenuation_db_per_km(10.0, k=0.07, alpha=-1.0)

    @given(r=st.floats(0.0, 300.0), k=st.floats(0.001, 1.0), alpha=st.floats(0.3, 2.0))
    @settings(max_examples=50)
    def test_property_nonnegative(self, r, k, alpha):
        assert rain_specific_attenuation_db_per_km(r, k, alpha) >= 0.0


class TestPathAttenuation:
    def test_known_answer(self):
        # gamma_R(10) = 0.8812477882559173 dB/km (from above)
        # L_eff = 5 / (1 + 5/20) = 5 / 1.25 = 4.0 km
        # A = 0.8812477882559173 * 4.0 = 3.5249911530236693
        got = rf_path_attenuation_db(10.0, k=0.07, alpha=1.1, path_length_km=5.0,
                                     reduction_length_km=20.0)
        assert got == pytest.approx(3.5249911530236693, rel=1e-9)

    def test_zero_rain_zero_attenuation(self):
        assert rf_path_attenuation_db(0.0, 0.07, 1.1, 5.0, 20.0) == pytest.approx(0.0)

    def test_longer_path_gives_more_attenuation(self):
        short = rf_path_attenuation_db(10.0, 0.07, 1.1, 2.0, 20.0)
        long_ = rf_path_attenuation_db(10.0, 0.07, 1.1, 20.0, 20.0)
        assert long_ > short

    def test_invalid_path_length_rejected(self):
        with pytest.raises(ValueError):
            rf_path_attenuation_db(10.0, 0.07, 1.1, 0.0, 20.0)

    def test_invalid_reduction_length_rejected(self):
        with pytest.raises(ValueError):
            rf_path_attenuation_db(10.0, 0.07, 1.1, 5.0, -5.0)


class TestRFParams:
    def test_defaults_construct(self):
        p = RFParams()
        assert p.margin_db == pytest.approx(p.snr_clear_db - p.snr_min_db)

    def test_snr_min_exceeding_clear_rejected(self):
        with pytest.raises(ValueError):
            RFParams(snr_clear_db=10.0, snr_min_db=20.0)

    def test_invalid_rate_rejected(self):
        with pytest.raises(ValueError):
            RFParams(rate_mbps=-1.0)

    def test_invalid_p_rain_rejected(self):
        with pytest.raises(ValueError):
            RFParams(p_rain=0.0)

    def test_frozen_dataclass_is_immutable(self):
        p = RFParams()
        with pytest.raises(Exception):
            p.p_rain = 0.5

    def test_nan_snr_rejected(self):
        with pytest.raises(ValueError):
            RFParams(snr_clear_db=float("nan"))
