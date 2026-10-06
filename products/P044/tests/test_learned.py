"""Tests for the combiner weightings, the penalty metric and the learned model."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aperturediv.datasets import build_features, make_combiner_dataset, split_dataset
from aperturediv.learned import (
    BASELINES,
    LearnedCombiner,
    egc_weights,
    fit_shrinkage_exponent,
    mean_ber,
    mrc_estimated_weights,
    mrc_true_weights,
    penalty_db,
    score_weights,
    shrinkage_weights,
)

_POSITIVE = hnp.arrays(
    dtype=float,
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
    elements=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)


@pytest.fixture(scope="module")
def trained():
    """One small trained model, shared across the tests that need it."""
    data = make_combiner_dataset(
        30_000, n_apertures=3, si=0.9, aperture_spacing_m=0.15,
        correlation_scale_m=0.10, sigma_e_range=(0.0, 2.0), seed=44044,
    )
    train, val, test = split_dataset(data, n_train=18_000, n_validation=6_000)
    model = LearnedCombiner(max_iter=60, random_state=44044).fit(train)
    return model, train, val, test


class TestWeightConstructors:
    def test_mrc_true_is_the_amplitude_direction(self):
        irr = np.array([[4.0, 1.0]])
        w = mrc_true_weights(irr)
        assert np.allclose(w, np.array([[2.0, 1.0]]) / np.sqrt(5.0), rtol=1e-14)

    def test_unit_norm(self):
        irr = np.array([[4.0, 1.0, 0.25]])
        for w in (mrc_true_weights(irr), mrc_estimated_weights(irr), egc_weights(1, 3),
                  shrinkage_weights(irr, 0.4)):
            assert np.linalg.norm(w, axis=1) == pytest.approx(1.0, rel=1e-14)

    def test_egc_known_answer(self):
        w = egc_weights(2, 4)
        assert w.shape == (2, 4)
        assert np.allclose(w, 0.5)

    def test_shrinkage_endpoints(self):
        irr = np.array([[4.0, 1.0, 0.25]])
        assert np.allclose(shrinkage_weights(irr, 1.0), mrc_estimated_weights(irr), rtol=1e-14)
        assert np.allclose(shrinkage_weights(irr, 0.0), egc_weights(1, 3), rtol=1e-14)

    def test_shrinkage_known_answer(self):
        # p = 0.5, Ihat = (16, 1): w ∝ Ihat^0.25 = (2, 1), normalised (2,1)/sqrt5.
        w = shrinkage_weights(np.array([[16.0, 1.0]]), 0.5)
        assert np.allclose(w, np.array([[2.0, 1.0]]) / np.sqrt(5.0), rtol=1e-14)

    def test_shrinkage_rejects_nonpositive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            shrinkage_weights(np.array([[0.0, 1.0]]), 0.5)

    def test_shrinkage_rejects_non_finite_exponent(self):
        with pytest.raises(ValueError, match="exponent"):
            shrinkage_weights(np.array([[1.0]]), float("nan"))

    @pytest.mark.parametrize("n,ell", [(0, 2), (2, 0), (-1, 2)])
    def test_egc_rejects_bad_sizes(self, n, ell):
        with pytest.raises(ValueError):
            egc_weights(n, ell)

    def test_baselines_listed(self):
        assert set(BASELINES) == {"mrc_true", "mrc_estimated", "egc", "shrinkage"}


class TestPenalty:
    def test_zero_for_the_true_mrc_direction(self):
        irr = np.array([[4.0, 1.0, 0.25], [1.0, 1.0, 1.0]])
        pen = penalty_db(mrc_true_weights(irr), np.sqrt(irr))
        assert np.allclose(pen, 0.0, atol=1e-13)

    def test_known_answer_equal_gain_on_a_dead_branch(self):
        # h = (1, 0), w = (1,1)/sqrt2: (w.h)^2 = 0.5, ||h||^2 = 1,
        # penalty = 10 log10(1/0.5) = 3.0103 dB.
        pen = penalty_db(np.array([[1.0, 1.0]]) / np.sqrt(2.0), np.array([[1.0, 0.0]]))
        assert float(pen[0]) == pytest.approx(10.0 * np.log10(2.0), rel=1e-12)

    def test_scale_invariant_in_the_weights(self):
        h = np.array([[2.0, 1.0]])
        w = np.array([[1.0, 1.0]])
        assert float(penalty_db(w, h)[0]) == pytest.approx(float(penalty_db(5.0 * w, h)[0]),
                                                           rel=1e-14)

    @given(_POSITIVE)
    def test_never_negative(self, irr):
        h = np.sqrt(irr)
        w = np.ones_like(h)
        assert np.all(penalty_db(w, h) >= -1e-9)

    @given(_POSITIVE)
    def test_mrc_true_is_the_minimiser(self, irr):
        h = np.sqrt(irr)
        best = penalty_db(mrc_true_weights(irr), h)
        other = penalty_db(np.ones_like(h), h)
        assert np.all(best <= other + 1e-9)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            penalty_db(np.ones((2, 3)), np.ones((2, 2)))

    def test_rejects_orthogonal_weights(self):
        with pytest.raises(ValueError, match="positive projection"):
            penalty_db(np.array([[1.0, -1.0]]), np.array([[1.0, 1.0]]))


class TestMeanBer:
    def test_matches_the_awgn_formula_without_fading(self):
        from aperturediv.ber import bpsk_ber_awgn

        h = np.ones((1, 1))
        got = mean_ber(h, h, 10.0)
        assert got == pytest.approx(float(bpsk_ber_awgn(10.0)), rel=1e-12)

    def test_mrc_true_gives_the_lowest_ber(self):
        irr = np.array([[4.0, 1.0], [0.5, 2.0], [1.0, 1.0]])
        h = np.sqrt(irr)
        best = mean_ber(mrc_true_weights(irr), h, 8.0)
        for w in (egc_weights(3, 2), shrinkage_weights(irr, 0.3)):
            assert best <= mean_ber(w, h, 8.0) + 1e-18

    def test_decreasing_in_ebn0(self):
        irr = np.array([[1.0, 1.0], [2.0, 0.5]])
        h = np.sqrt(irr)
        vals = [mean_ber(egc_weights(2, 2), h, e) for e in (0.0, 5.0, 10.0)]
        assert all(y < x for x, y in zip(vals, vals[1:], strict=False))


class TestScoreWeights:
    def test_fields(self):
        irr = np.array([[4.0, 1.0], [1.0, 1.0]])
        sc = score_weights("egc", egc_weights(2, 2), np.sqrt(irr), 10.0)
        assert sc.name == "egc"
        assert sc.n_rows == 2
        assert sc.mean_penalty_db >= 0.0
        assert sc.max_penalty_db >= sc.p90_penalty_db >= sc.median_penalty_db >= 0.0
        assert 0.0 < sc.mean_ber < 1.0


class TestShrinkageFit:
    def test_returns_a_value_in_the_grid_range(self, trained):
        _, _, val, _ = trained
        p = fit_shrinkage_exponent(val)
        assert 0.0 <= p <= 1.0

    def test_zero_error_prefers_full_mrc(self):
        d = make_combiner_dataset(8000, n_apertures=3, sigma_e_range=(0.0, 0.0), seed=21)
        assert fit_shrinkage_exponent(d) == pytest.approx(1.0, abs=1e-12)

    def test_large_error_prefers_heavy_shrinkage(self):
        d = make_combiner_dataset(8000, n_apertures=3, sigma_e_range=(3.0, 3.0), seed=22)
        assert fit_shrinkage_exponent(d) < 0.4

    def test_custom_grid_respected(self, trained):
        _, _, val, _ = trained
        p = fit_shrinkage_exponent(val, grid=np.array([0.25, 0.75]))
        assert p in (0.25, 0.75)


class TestLearnedCombiner:
    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            LearnedCombiner().predict_weights(np.zeros((2, 6)))

    def test_quantiles_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            LearnedCombiner().predict_penalty_quantiles(np.zeros((2, 6)))

    def test_weights_unit_norm_and_non_negative(self, trained):
        model, _, _, test = trained
        w, _ = model.combine(test.irradiance_estimated, test.sigma_e)
        assert w.shape == (len(test), test.n_apertures)
        assert np.all(w >= 0.0)
        assert np.allclose(np.linalg.norm(w, axis=1), 1.0, rtol=1e-12)

    def test_penalty_non_negative_on_held_out_rows(self, trained):
        model, _, _, test = trained
        w, _ = model.combine(test.irradiance_estimated, test.sigma_e)
        assert penalty_db(w, test.amplitude_true).min() >= -1e-9

    def test_cannot_beat_mrc_with_the_truth(self, trained):
        model, _, _, test = trained
        w, _ = model.combine(test.irradiance_estimated, test.sigma_e)
        learned = penalty_db(w, test.amplitude_true).mean()
        true_mrc = penalty_db(mrc_true_weights(test.irradiance_true),
                              test.amplitude_true).mean()
        assert learned >= true_mrc - 1e-9

    def test_beats_naive_mrc_on_the_estimate_when_pooled(self, trained):
        model, _, _, test = trained
        w, _ = model.combine(test.irradiance_estimated, test.sigma_e)
        learned = penalty_db(w, test.amplitude_true).mean()
        naive = penalty_db(mrc_estimated_weights(test.irradiance_estimated),
                           test.amplitude_true).mean()
        assert learned < naive

    def test_quantile_output_shape_ordering_and_floor(self, trained):
        model, _, _, test = trained
        _, q = model.combine(test.irradiance_estimated, test.sigma_e)
        assert q.shape == (len(test), len(model.quantiles))
        assert np.all(q >= 0.0)
        assert np.mean(q[:, 0] <= q[:, 2]) > 0.98

    def test_quantile_coverage_roughly_calibrated(self, trained):
        model, _, _, test = trained
        w, q = model.combine(test.irradiance_estimated, test.sigma_e)
        realised = penalty_db(w, test.amplitude_true)
        for j, level in enumerate(model.quantiles):
            cov = float(np.mean(realised <= q[:, j]))
            assert abs(cov - level) < 0.12

    def test_deterministic_fit(self):
        data = make_combiner_dataset(6000, n_apertures=2, seed=23)
        tr, _, te = split_dataset(data, n_train=4000, n_validation=1000)
        a = LearnedCombiner(max_iter=40, random_state=5).fit(tr)
        b = LearnedCombiner(max_iter=40, random_state=5).fit(tr)
        wa, _ = a.combine(te.irradiance_estimated, te.sigma_e)
        wb, _ = b.combine(te.irradiance_estimated, te.sigma_e)
        assert np.array_equal(wa, wb)

    def test_n_apertures_recorded(self, trained):
        model, train, _, _ = trained
        assert model.n_apertures == train.n_apertures

    def test_predict_rejects_one_dimensional_features(self, trained):
        model, _, _, _ = trained
        with pytest.raises(ValueError, match="n_features"):
            model.predict_weights(np.zeros(6))

    def test_combine_accepts_a_single_row(self, trained):
        model, _, _, _ = trained
        est = np.array([[1.0, 0.5, 2.0]])
        w, q = model.combine(est, np.array([0.5]))
        assert w.shape == (1, 3)
        assert q.shape == (1, 3)

    def test_features_built_from_the_estimate_only(self, trained):
        model, _, _, test = trained
        direct = model.predict_weights(
            build_features(test.irradiance_estimated, test.sigma_e)
        )
        via_combine, _ = model.combine(test.irradiance_estimated, test.sigma_e)
        assert np.array_equal(direct, via_combine)
