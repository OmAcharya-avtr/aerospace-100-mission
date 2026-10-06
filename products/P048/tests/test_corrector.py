"""Learned-corrector tests. Kept small so the suite stays inside the budget."""

from __future__ import annotations

import numpy as np
import pytest

from softdecode.corrector import (
    FEATURE_NAMES,
    LlrCorrector,
    TunedParameters,
    build_features,
    gmi_of,
    tune_clip,
    tune_scale_and_clip,
)
from softdecode.datasets import BASE_SEED, make_split, make_training_set
from softdecode.simulate import decode_ber, demap_ook


def test_feature_matrix_shape_and_content(code, lognormal, csi_error):
    r = make_split(code, 8.0, lognormal, csi_error, 8, "train")
    plugin = demap_ook(r, "plugin", lognormal, csi_error)
    x = build_features(r, plugin)
    assert x.shape == (r.channel_bits, len(FEATURE_NAMES))
    assert np.allclose(x[:, 0], r.y.ravel() / r.sigma)
    assert np.allclose(x[:, 2], np.log(r.h_hat.ravel()))
    assert np.allclose(x[:, 4], plugin.ravel())
    assert np.all(x[:, 5] == r.amplitude / r.sigma)


def test_feature_length_validation(code, lognormal, csi_error):
    r = make_split(code, 8.0, lognormal, csi_error, 4, "train")
    with pytest.raises(ValueError, match="plugin_llr"):
        build_features(r, np.zeros(3))


def test_splits_are_disjoint(code, lognormal, csi_error):
    train = make_split(code, 8.0, lognormal, csi_error, 8, "train")
    tune = make_split(code, 8.0, lognormal, csi_error, 8, "tune")
    report = make_split(code, 8.0, lognormal, csi_error, 8, "report")
    assert not np.array_equal(train.y, tune.y)
    assert not np.array_equal(train.y, report.y)
    assert not np.array_equal(tune.y, report.y)


def test_unknown_split_raises(code, lognormal, csi_error):
    with pytest.raises(ValueError, match="split must be"):
        make_split(code, 8.0, lognormal, csi_error, 4, "holdout")


def test_training_set_spans_requested_points(code, lognormal, csi_error):
    ts = make_training_set(code, lognormal, csi_error, blocks_per_point=4, ebn0_db=(4.0, 8.0))
    assert ts.ebn0_db == (4.0, 8.0)
    assert ts.n_samples == 2 * 4 * code.length
    assert ts.seed == BASE_SEED
    assert set(np.unique(ts.features[:, 5]).tolist()).__len__() == 2


def test_corrector_requires_fit_before_predict():
    corrector = LlrCorrector()
    assert not corrector.fitted
    with pytest.raises(RuntimeError, match="before fit"):
        corrector.predict(np.zeros((2, len(FEATURE_NAMES))))


def test_corrector_input_validation():
    corrector = LlrCorrector(n_estimators=3, max_depth=3, min_samples_leaf=2)
    with pytest.raises(ValueError, match="features must be"):
        corrector.fit(np.zeros((4, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="length mismatch"):
        corrector.fit(np.zeros((4, len(FEATURE_NAMES))), np.zeros(3))
    with pytest.raises(ValueError, match="0/1"):
        corrector.fit(np.zeros((4, len(FEATURE_NAMES))), np.full(4, 2))


def test_corrector_learns_and_beats_the_plugin(code, lognormal, csi_error):
    ts = make_training_set(code, lognormal, csi_error, blocks_per_point=60, ebn0_db=(6.0, 8.0))
    corrector = LlrCorrector(n_estimators=12, max_depth=10, min_samples_leaf=100).fit(
        ts.features, ts.bits
    )
    assert corrector.fitted
    report = make_split(code, 8.0, lognormal, csi_error, 120, "report")
    plugin = demap_ook(report, "plugin", lognormal, csi_error)
    features = build_features(report, plugin)
    learned = corrector.predict(features).reshape(report.codeword.shape)
    assert gmi_of(learned, report) > gmi_of(plugin, report)
    assert decode_ber(code, report, learned).rate < decode_ber(code, report, plugin).rate


def test_corrector_uncertainty_output(code, lognormal, csi_error):
    ts = make_training_set(code, lognormal, csi_error, blocks_per_point=40, ebn0_db=(8.0,))
    corrector = LlrCorrector(n_estimators=8, max_depth=8, min_samples_leaf=100).fit(
        ts.features, ts.bits
    )
    report = make_split(code, 8.0, lognormal, csi_error, 40, "report")
    features = build_features(report, demap_ook(report, "plugin", lognormal, csi_error))
    mean, sigma = corrector.predict(features, with_uncertainty=True)
    assert mean.shape == sigma.shape == (report.channel_bits,)
    assert np.all(sigma >= 0.0)
    assert np.any(sigma > 0.0)
    assert np.all(np.abs(mean) <= corrector.clip + 1e-9)


def test_save_and_load_round_trip(tmp_path, code, lognormal, csi_error):
    ts = make_training_set(code, lognormal, csi_error, blocks_per_point=20, ebn0_db=(8.0,))
    corrector = LlrCorrector(n_estimators=4, max_depth=6, min_samples_leaf=100).fit(
        ts.features, ts.bits
    )
    path = tmp_path / "corrector.joblib"
    corrector.save(str(path))
    other = LlrCorrector.load(str(path))
    assert other.fitted
    assert other.clip == corrector.clip
    x = ts.features[:50]
    assert np.allclose(other.predict(x), corrector.predict(x))


def test_tuning_returns_parameters_from_the_tuning_split(code, lognormal, csi_error):
    tune = make_split(code, 8.0, lognormal, csi_error, 100, "tune")
    tuned = tune_scale_and_clip(
        code, tune, lognormal, csi_error, scales=np.array([0.2, 1.0]),
        clips=np.array([2.0, 20.0])
    )
    assert set(tuned) == {"plugin_scaled", "plugin_clipped", "plugin_scaled_clipped"}
    for key, value in tuned.items():
        assert isinstance(value, TunedParameters)
        assert 0.0 <= value.tuning_ber <= 1.0, key
    assert tuned["plugin_clipped"].scale == 1.0
    assert np.isinf(tuned["plugin_scaled"].clip)
    assert "scale" in str(tuned["plugin_scaled"])


def test_tune_clip_sets_the_corrector_clip(code, lognormal, csi_error):
    ts = make_training_set(code, lognormal, csi_error, blocks_per_point=20, ebn0_db=(8.0,))
    corrector = LlrCorrector(n_estimators=6, max_depth=8, min_samples_leaf=100).fit(
        ts.features, ts.bits
    )
    tune = make_split(code, 8.0, lognormal, csi_error, 60, "tune")
    features = build_features(tune, demap_ook(tune, "plugin", lognormal, csi_error))
    tuned = tune_clip(code, tune, corrector, features, clips=np.array([3.0, 30.0]))
    assert corrector.clip == tuned.clip
    assert tuned.clip in (3.0, 30.0)
