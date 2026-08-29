"""Tests for waveforge.predictor."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.predictor import (
    LinearSlopePredictor,
    PureDelayPredictor,
    build_lagged_dataset,
)


@pytest.fixture(scope="module")
def toy_sequences():
    """Three deterministic, strongly predictable multivariate series.

    Each channel is a sinusoid with a channel-dependent frequency and a
    sequence-dependent phase, so a linear model on four lags can forecast it
    almost exactly.  This exercises the machinery without an AO simulation.
    """
    t = np.arange(200)
    out = []
    for phase in (0.0, 1.1, 2.3):
        cols = [np.sin(0.05 * (k + 1) * t + phase + 0.3 * k) for k in range(6)]
        out.append(np.stack(cols, axis=1))
    return out


class TestLaggedDataset:
    def test_shapes(self):
        seq = np.arange(60, dtype=float).reshape(20, 3)
        x, y = build_lagged_dataset([seq], n_history=4, horizon=2)
        assert x.shape == (15, 12)
        assert y.shape == (15, 3)

    def test_content_is_ordered_oldest_first(self):
        seq = np.arange(30, dtype=float).reshape(10, 3)
        x, y = build_lagged_dataset([seq], n_history=2, horizon=1)
        assert np.allclose(x[0], np.concatenate([seq[0], seq[1]]))
        assert np.allclose(y[0], seq[2])

    def test_horizon_offset(self):
        seq = np.arange(30, dtype=float).reshape(10, 3)
        _, y = build_lagged_dataset([seq], n_history=1, horizon=3)
        assert np.allclose(y[0], seq[3])

    def test_multiple_sequences_concatenate(self):
        seq = np.zeros((12, 2))
        x, _ = build_lagged_dataset([seq, seq], n_history=3, horizon=1)
        assert x.shape[0] == 2 * (12 - 3)

    def test_no_sample_straddles_two_sequences(self):
        a = np.zeros((8, 1))
        b = np.ones((8, 1))
        x, y = build_lagged_dataset([a, b], n_history=2, horizon=1)
        # every row is either all zeros or all ones
        assert np.all((x.sum(axis=1) == 0.0) | (x.sum(axis=1) == 2.0))
        assert np.all((y == 0.0) | (y == 1.0))

    def test_rejects_short_sequence(self):
        with pytest.raises(ValueError, match="too short"):
            build_lagged_dataset([np.zeros((3, 2))], n_history=3, horizon=2)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="at least one sequence"):
            build_lagged_dataset([], 2, 1)

    def test_rejects_1d_sequence(self):
        with pytest.raises(ValueError, match="2-D"):
            build_lagged_dataset([np.zeros(10)], 2, 1)

    def test_rejects_mixed_widths(self):
        with pytest.raises(ValueError, match="same number of slopes"):
            build_lagged_dataset([np.zeros((10, 2)), np.zeros((10, 3))], 2, 1)

    @pytest.mark.parametrize(("n_history", "horizon"), [(0, 1), (1, 0), (1.5, 1), (1, 2.5)])
    def test_rejects_bad_parameters(self, n_history, horizon):
        with pytest.raises(ValueError):
            build_lagged_dataset([np.zeros((20, 2))], n_history, horizon)


class TestPureDelay:
    def test_returns_last_row(self):
        history = np.arange(6, dtype=float).reshape(3, 2)
        mean, sigma = PureDelayPredictor().predict(history)
        assert np.allclose(mean, [4.0, 5.0])
        assert sigma is None

    def test_does_not_alias_the_input(self):
        history = np.zeros((2, 3))
        mean, _ = PureDelayPredictor().predict(history)
        mean[0] = 99.0
        assert history[-1, 0] == 0.0

    def test_horizon_is_none(self):
        assert PureDelayPredictor().horizon is None

    def test_rejects_bad_history(self):
        with pytest.raises(ValueError, match="history"):
            PureDelayPredictor().predict(np.zeros(4))

    def test_rejects_bad_n_history(self):
        with pytest.raises(ValueError, match="n_history"):
            PureDelayPredictor(n_history=0)


class TestLinearSlopePredictorValidation:
    @pytest.mark.parametrize("n_history", [0, -1, 2.5])
    def test_bad_n_history(self, n_history):
        with pytest.raises(ValueError, match="n_history"):
            LinearSlopePredictor(n_history=n_history)

    @pytest.mark.parametrize("horizon", [0, -1, 1.5])
    def test_bad_horizon(self, horizon):
        with pytest.raises(ValueError, match="horizon"):
            LinearSlopePredictor(horizon=horizon)

    def test_bad_model(self):
        with pytest.raises(ValueError, match="model"):
            LinearSlopePredictor(model="transformer")

    def test_bad_members(self):
        with pytest.raises(ValueError, match="n_members"):
            LinearSlopePredictor(n_members=0)

    def test_bad_validation_fraction(self):
        with pytest.raises(ValueError, match="validation_fraction"):
            LinearSlopePredictor(validation_fraction=1.0)

    @pytest.mark.parametrize("alpha", ["best", 0.0, -1.0])
    def test_bad_alpha(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            LinearSlopePredictor(alpha=alpha)

    def test_predict_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            LinearSlopePredictor().predict(np.zeros((4, 3)))

    def test_predict_batch_before_fit(self):
        with pytest.raises(RuntimeError, match="not fitted"):
            LinearSlopePredictor().predict_batch(np.zeros((2, 12)))

    def test_fit_needs_enough_samples(self):
        model = LinearSlopePredictor(n_history=2, horizon=1)
        with pytest.raises(ValueError, match="at least 4 training samples"):
            model.fit([np.zeros((5, 2))])


class TestLinearSlopePredictorBehaviour:
    @pytest.fixture(scope="class")
    @staticmethod
    def fitted(toy_sequences):
        return LinearSlopePredictor(n_history=4, horizon=2, n_members=4, random_state=1).fit(
            toy_sequences
        )

    def test_is_fitted(self, fitted):
        assert fitted.is_fitted
        assert fitted.n_slopes == 6
        assert fitted.n_parameters > 0

    def test_alpha_chosen_from_grid(self, fitted):
        assert fitted.chosen_alpha in LinearSlopePredictor().alpha_grid

    def test_predict_shapes(self, fitted, toy_sequences):
        mean, sigma = fitted.predict(toy_sequences[0][:4])
        assert mean.shape == (6,)
        assert sigma.shape == (6,)
        assert np.all(sigma >= 0.0)

    def test_beats_pure_delay_on_a_predictable_signal(self, fitted, toy_sequences):
        x, y = build_lagged_dataset([toy_sequences[1]], 4, 2)
        mean, _ = fitted.predict_batch(x)
        delay = x[:, -6:]
        assert np.mean((mean - y) ** 2) < np.mean((delay - y) ** 2)

    def test_predict_matches_predict_batch(self, fitted, toy_sequences):
        x, _ = build_lagged_dataset([toy_sequences[0]], 4, 2)
        single, _ = fitted.predict(toy_sequences[0][:4])
        batch, _ = fitted.predict_batch(x[:1])
        assert np.allclose(single, batch[0])

    def test_uncertainty_is_positive(self, fitted, toy_sequences):
        _, sigma = fitted.predict(toy_sequences[2][10:14])
        assert np.all(sigma > 0.0)

    def test_single_member_has_no_epistemic_spread(self, toy_sequences):
        model = LinearSlopePredictor(
            n_history=3, horizon=1, n_members=1, alpha=1.0, random_state=0
        ).fit(toy_sequences)
        _, sigma = model.predict(toy_sequences[0][:3])
        assert np.all(sigma >= 0.0)

    def test_explicit_alpha_is_used(self, toy_sequences):
        model = LinearSlopePredictor(n_history=2, horizon=1, alpha=7.0, n_members=2).fit(
            toy_sequences
        )
        assert model.chosen_alpha == pytest.approx(7.0)

    def test_stronger_alpha_shrinks_predictions(self, toy_sequences):
        weak = LinearSlopePredictor(n_history=2, horizon=1, alpha=1e-3, n_members=1).fit(
            toy_sequences
        )
        strong = LinearSlopePredictor(n_history=2, horizon=1, alpha=1e5, n_members=1).fit(
            toy_sequences
        )
        history = toy_sequences[0][:2]
        assert np.linalg.norm(strong.predict(history)[0]) < np.linalg.norm(
            weak.predict(history)[0]
        )

    def test_reproducible(self, toy_sequences):
        a = LinearSlopePredictor(n_history=3, horizon=1, n_members=3, random_state=2).fit(
            toy_sequences
        )
        b = LinearSlopePredictor(n_history=3, horizon=1, n_members=3, random_state=2).fit(
            toy_sequences
        )
        history = toy_sequences[0][:3]
        assert np.allclose(a.predict(history)[0], b.predict(history)[0])

    def test_wrong_history_shape(self, fitted):
        with pytest.raises(ValueError, match="history must have shape"):
            fitted.predict(np.zeros((3, 6)))
        with pytest.raises(ValueError, match="history must have shape"):
            fitted.predict(np.zeros((4, 5)))

    def test_wrong_batch_shape(self, fitted):
        with pytest.raises(ValueError, match="x must have shape"):
            fitted.predict_batch(np.zeros((2, 7)))

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
    def test_mlp_model_fits_and_predicts(self, toy_sequences):
        model = LinearSlopePredictor(
            n_history=3,
            horizon=1,
            model="mlp",
            n_members=2,
            hidden_layer_sizes=(16,),
            max_iter=60,
            alpha=1e-4,
            random_state=0,
        ).fit(toy_sequences)
        mean, sigma = model.predict(toy_sequences[0][:3])
        assert mean.shape == (6,)
        assert np.all(np.isfinite(sigma))
        assert np.isnan(model.chosen_alpha)
        assert model.n_parameters > 0
