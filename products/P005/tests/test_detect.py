"""Detector tests: reproducibility, correctness, confidence, validation."""

import numpy as np
import pytest

from jitterscope import (
    BandZScoreBaseline,
    FeatureExtractor,
    NominalModel,
    detect,
    generate_telemetry,
)

FS = 1000.0


@pytest.fixture(scope="module")
def nominal_features():
    _, x, _ = generate_telemetry(60, FS, seed=42)
    ext = FeatureExtractor(fs=FS)
    feats, _ = ext.transform(x)
    return ext, feats


@pytest.fixture(scope="module")
def faulty_record():
    t, x, mask = generate_telemetry(
        40, FS, seed=43,
        faults=[{"kind": "new_tone", "t_start": 20.0, "freq_hz": 137.0, "rms": 1e-6}],
    )
    return t, x, mask


def test_mlp_reproducibility(nominal_features, faulty_record):
    """Same seed + data -> identical scores and threshold across fits."""
    ext, feats = nominal_features
    _, x, _ = faulty_record
    m1 = NominalModel(seed=0).fit(feats)
    m2 = NominalModel(seed=0).fit(feats)
    assert m1.threshold_ == m2.threshold_
    f, _ = ext.transform(x)
    np.testing.assert_array_equal(m1.score(f), m2.score(f))


def test_baseline_flags_fault_region_only(nominal_features, faulty_record):
    ext, feats = nominal_features
    _, x, _ = faulty_record
    base = BandZScoreBaseline().fit(feats)
    res = detect(x, model=base, extractor=ext)
    assert res.n_anomalous > 0
    # every flagged window center is inside the faulty half (t >= 20 s,
    # minus half a window of edge slack)
    assert np.all(res.window_centers_s[res.flags] >= 20.0 - ext.window_s)


def test_mlp_flags_fault_region(nominal_features, faulty_record):
    ext, feats = nominal_features
    _, x, mask = faulty_record
    mdl = NominalModel(seed=0).fit(feats)
    res = detect(x, model=mdl, extractor=ext)
    assert res.n_anomalous > 10
    frac_in_fault = np.mean(res.window_centers_s[res.flags] >= 20.0 - ext.window_s)
    assert frac_in_fault > 0.9


def test_low_false_alarms_on_unseen_nominal(nominal_features):
    """<= 5 % of windows flagged on an unseen nominal record (q=0.995)."""
    ext, feats = nominal_features
    _, x, _ = generate_telemetry(40, FS, seed=7)
    for model in (NominalModel(seed=0).fit(feats), BandZScoreBaseline().fit(feats)):
        res = detect(x, model=model, extractor=ext)
        assert res.n_anomalous <= 0.05 * res.scores.size


def test_confidence_output_range_and_ordering(nominal_features, faulty_record):
    """Confidence is in [0,1] and higher for flagged than unflagged windows."""
    ext, feats = nominal_features
    _, x, _ = faulty_record
    mdl = NominalModel(seed=0).fit(feats)
    res = detect(x, model=mdl, extractor=ext)
    assert np.all((res.confidence >= 0) & (res.confidence <= 1))
    if res.flags.any() and (~res.flags).any():
        assert res.confidence[res.flags].min() >= res.confidence[~res.flags].mean()


def test_explicit_threshold_overrides_fitted(nominal_features, faulty_record):
    ext, feats = nominal_features
    _, x, _ = faulty_record
    base = BandZScoreBaseline().fit(feats)
    res = detect(x, threshold=1e9, model=base, extractor=ext)
    assert res.n_anomalous == 0
    assert res.threshold == 1e9


class TestInputValidation:
    def test_score_before_fit_raises(self):
        with pytest.raises(ValueError, match="fit"):
            NominalModel().score(np.zeros((5, 24)))
        with pytest.raises(ValueError, match="fit"):
            BandZScoreBaseline().score(np.zeros((5, 24)))

    def test_too_few_windows_raises(self):
        with pytest.raises(ValueError, match=">= 20"):
            NominalModel().fit(np.zeros((5, 24)))

    def test_bad_quantile_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            NominalModel(quantile=1.5)
        with pytest.raises(ValueError, match="quantile"):
            BandZScoreBaseline(quantile=0.2)

    def test_nan_telemetry_raises(self, nominal_features):
        ext, feats = nominal_features
        base = BandZScoreBaseline().fit(feats)
        x = np.ones(5000)
        x[100] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            detect(x, model=base, extractor=ext)

    def test_nan_features_raise(self):
        feats = np.zeros((30, 24))
        feats[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            BandZScoreBaseline().fit(feats)

    def test_bad_extractor_params_raise(self):
        with pytest.raises(ValueError, match="overlap"):
            FeatureExtractor(fs=FS, overlap=1.0)
        with pytest.raises(ValueError, match="f_min"):
            FeatureExtractor(fs=FS, f_min=0.0)

    def test_negative_threshold_raises(self, nominal_features):
        ext, feats = nominal_features
        base = BandZScoreBaseline().fit(feats)
        _, x, _ = generate_telemetry(10, FS, seed=1)
        with pytest.raises(ValueError, match="threshold"):
            detect(x, threshold=-1.0, model=base, extractor=ext)
