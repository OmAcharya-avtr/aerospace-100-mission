"""Tests for linkswitch.features: rolling features and imminent-outage labels."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from linkswitch.features import FEATURE_NAMES, label_imminent_outage, rolling_features


class TestRollingFeatures:
    def test_shape(self):
        irr = np.exp(np.random.default_rng(0).normal(size=50))
        feats = rolling_features(irr, window=5)
        assert feats.shape == (50, len(FEATURE_NAMES))

    def test_log_i_column_is_log(self):
        irr = np.array([1.0, 2.0, 0.5, 4.0])
        feats = rolling_features(irr, window=2)
        np.testing.assert_allclose(feats[:, 0], np.log(irr))

    def test_constant_series_zero_std_zero_slope(self):
        irr = np.full(20, 2.0)
        feats = rolling_features(irr, window=4)
        assert np.allclose(feats[:, 2], 0.0)  # roll_std
        assert np.allclose(feats[4:, 4], 0.0)  # slope after window fills in

    def test_known_answer_roll_mean_window_3(self):
        # log I = [0, ln2, ln4, ln8] -> ln values [0, 0.6931, 1.3863, 2.0794]
        irr = np.array([1.0, 2.0, 4.0, 8.0])
        feats = rolling_features(irr, window=3)
        log_i = np.log(irr)
        # t=0: mean of [log_i[0]] = log_i[0]
        assert feats[0, 1] == pytest.approx(log_i[0])
        # t=1: mean of [log_i[0], log_i[1]]
        assert feats[1, 1] == pytest.approx(log_i[:2].mean())
        # t=2: mean of [log_i[0], log_i[1], log_i[2]]
        assert feats[2, 1] == pytest.approx(log_i[:3].mean())
        # t=3: mean of [log_i[1], log_i[2], log_i[3]] (window=3, trailing)
        assert feats[3, 1] == pytest.approx(log_i[1:4].mean())

    def test_known_answer_roll_min(self):
        irr = np.array([1.0, 4.0, 0.5, 8.0])
        feats = rolling_features(irr, window=2)
        log_i = np.log(irr)
        assert feats[0, 3] == pytest.approx(log_i[0])
        assert feats[1, 3] == pytest.approx(min(log_i[0], log_i[1]))
        assert feats[2, 3] == pytest.approx(min(log_i[1], log_i[2]))
        assert feats[3, 3] == pytest.approx(min(log_i[2], log_i[3]))

    def test_slope_known_answer(self):
        # window=2: slope[t] = (log_i[t] - log_i[t-2]) / 2 for t >= 2
        irr = np.array([1.0, 2.0, 4.0, 16.0, 64.0])
        feats = rolling_features(irr, window=2)
        log_i = np.log(irr)
        assert feats[2, 4] == pytest.approx((log_i[2] - log_i[0]) / 2.0)
        assert feats[3, 4] == pytest.approx((log_i[3] - log_i[1]) / 2.0)
        assert feats[4, 4] == pytest.approx((log_i[4] - log_i[2]) / 2.0)

    def test_no_lookahead(self):
        # Changing a future sample must not change any earlier feature row.
        rng = np.random.default_rng(0)
        irr = np.exp(rng.normal(size=30))
        feats_a = rolling_features(irr, window=5)
        irr2 = irr.copy()
        irr2[20:] *= 100.0  # perturb only the tail
        feats_b = rolling_features(irr2, window=5)
        np.testing.assert_allclose(feats_a[:15], feats_b[:15])

    def test_nonpositive_irradiance_rejected(self):
        with pytest.raises(ValueError):
            rolling_features(np.array([1.0, -1.0, 2.0]), window=2)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            rolling_features(np.array([]), window=2)

    def test_invalid_window_rejected(self):
        with pytest.raises(ValueError):
            rolling_features(np.array([1.0, 2.0]), window=0)

    @given(st.lists(st.floats(0.01, 100.0), min_size=3, max_size=40))
    @settings(max_examples=30)
    def test_property_no_nan(self, values):
        feats = rolling_features(np.array(values), window=3)
        assert np.all(np.isfinite(feats))


class TestLabelImminentOutage:
    def test_known_answer_simple(self):
        # tau=1.0, horizon=2. irradiance = [2, 2, 0.5, 2, 2]
        # t=0: next 2 = [2, 0.5] -> has outage -> True
        # t=1: next 2 = [0.5, 2] -> has outage -> True
        # t=2: next 2 = [2, 2]   -> no outage  -> False
        # t=3: next 1 (clipped)  = [2]  -> False
        # t=4: next 0 (clipped)  = []   -> False
        irr = np.array([2.0, 2.0, 0.5, 2.0, 2.0])
        label = label_imminent_outage(irr, tau_phys=1.0, horizon=2)
        np.testing.assert_array_equal(label, [True, True, False, False, False])

    def test_horizon_1_equals_next_step_outage(self):
        irr = np.array([2.0, 0.5, 2.0, 0.5])
        label = label_imminent_outage(irr, tau_phys=1.0, horizon=1)
        # label[t] = irr[t+1] < 1.0 (last one clipped to False):
        # t=0: irr[1]=0.5<1.0 -> True; t=1: irr[2]=2.0 -> False;
        # t=2: irr[3]=0.5<1.0 -> True; t=3: no future -> False
        np.testing.assert_array_equal(label, [True, False, True, False])

    def test_no_outage_ever_all_false(self):
        irr = np.full(10, 5.0)
        label = label_imminent_outage(irr, tau_phys=1.0, horizon=3)
        assert not label.any()

    def test_always_outage_all_true_except_tail(self):
        irr = np.full(10, 0.1)
        label = label_imminent_outage(irr, tau_phys=1.0, horizon=3)
        assert label[:7].all()  # steps with a full horizon ahead

    def test_invalid_horizon_rejected(self):
        with pytest.raises(ValueError):
            label_imminent_outage(np.array([1.0, 2.0]), tau_phys=1.0, horizon=0)

    def test_invalid_tau_rejected(self):
        with pytest.raises(ValueError):
            label_imminent_outage(np.array([1.0, 2.0]), tau_phys=-1.0, horizon=1)

    def test_larger_horizon_never_decreases_true_count(self):
        rng = np.random.default_rng(0)
        irr = np.exp(rng.normal(size=200))
        n_true_small = label_imminent_outage(irr, tau_phys=1.0, horizon=2).sum()
        n_true_large = label_imminent_outage(irr, tau_phys=1.0, horizon=10).sum()
        assert n_true_large >= n_true_small
