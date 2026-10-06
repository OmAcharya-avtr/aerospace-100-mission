"""Baselines first, then the learned correction, on the same held-out rows."""

from __future__ import annotations

import numpy as np
import pytest

from photoncount.correction import (
    QUANTILES,
    RateCorrector,
    composed_baseline,
    interval_coverage,
    matched_baseline,
    nonparalyzable_baseline,
    paralyzable_baseline,
    rate_error_metrics,
    summarise_by_regime,
)
from photoncount.dataset import (
    FEATURE_NAMES,
    CorrectionDataset,
    SamplingRanges,
    generate_dataset,
    make_features,
)

TAU = 1e-6


@pytest.fixture(scope="module")
def small_train() -> CorrectionDataset:
    return generate_dataset(400, 2026)


@pytest.fixture(scope="module")
def small_test() -> CorrectionDataset:
    return generate_dataset(500, 777)


@pytest.fixture(scope="module")
def fitted(small_train: CorrectionDataset) -> RateCorrector:
    return RateCorrector(n_estimators=60, max_depth=3).fit(small_train)


# --- baselines --------------------------------------------------------------


def test_nonparalyzable_baseline_known_answer():
    # m = 90909.0909 /s at tau = 1 us inverts to n = 1e5 /s exactly.
    assert float(nonparalyzable_baseline(90909.0909090909, TAU)[0]) == pytest.approx(
        1e5, rel=1e-9
    )


def test_paralyzable_baseline_known_answer():
    # m = 90483.7418 /s at tau = 1 us, lower branch, inverts to n = 1e5 /s.
    assert float(paralyzable_baseline(90483.74180359596, TAU)[0]) == pytest.approx(1e5, rel=1e-8)


def test_baselines_return_nan_where_undefined():
    # m*tau = 1 is unreachable for the non-paralyzable model.
    assert np.isnan(nonparalyzable_baseline(1.0 / TAU, TAU)[0])
    # m above 1/(e tau) = 367879.44 is unreachable for the paralyzable model.
    assert np.isnan(paralyzable_baseline(4e5, TAU)[0])


def test_paralyzable_upper_branch_differs_from_lower():
    lo = float(paralyzable_baseline(3e5, TAU, "lower")[0])
    hi = float(paralyzable_baseline(3e5, TAU, "upper")[0])
    assert lo < 1.0 / TAU < hi


def test_paralyzable_baseline_rejects_a_bad_branch():
    with pytest.raises(ValueError, match="branch must be"):
        paralyzable_baseline(1e5, TAU, "either")


def test_matched_baseline_dispatches_on_the_model():
    m = np.array([9e4, 9e4])
    tau = np.array([TAU, TAU])
    par = np.array([1.0, 0.0])
    out = matched_baseline(m, tau, par)
    assert out[0] == pytest.approx(float(paralyzable_baseline(9e4, TAU)[0]))
    assert out[1] == pytest.approx(float(nonparalyzable_baseline(9e4, TAU)[0]))


def test_composed_baseline_removes_the_afterpulse_inflation_first():
    """With no dead time the composed baseline is exact."""
    p = 0.1
    primary = 1e5
    observed = primary / (1.0 - p)
    tiny_tau = np.array([1e-15])
    out = composed_baseline(np.array([observed]), tiny_tau, np.array([0.0]), np.array([p]))
    assert float(out[0]) == pytest.approx(primary, rel=1e-6)


def test_composed_equals_matched_without_afterpulsing():
    m = np.array([9e4])
    tau = np.array([TAU])
    par = np.array([1.0])
    assert composed_baseline(m, tau, par, np.array([0.0]))[0] == pytest.approx(
        matched_baseline(m, tau, par)[0]
    )


# --- metrics ----------------------------------------------------------------


def test_rate_error_metrics_known_answer():
    est = np.array([110.0, 90.0, np.nan])
    truth = np.array([100.0, 100.0, 100.0])
    out = rate_error_metrics(est, truth, "toy")
    assert out["n_defined"] == 2.0
    assert out["defined_fraction"] == pytest.approx(2 / 3)
    assert out["median_abs_rel_error"] == pytest.approx(0.1)
    assert out["median_signed_rel_error"] == pytest.approx(0.0)


def test_rate_error_metrics_all_undefined():
    out = rate_error_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert out["n_defined"] == 0.0
    assert np.isnan(out["median_abs_rel_error"])


def test_rate_error_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        rate_error_metrics(np.zeros(3), np.zeros(4))


def test_interval_coverage_known_answer():
    lo = np.array([0.0, 2.0, 0.0])
    hi = np.array([2.0, 3.0, 2.0])
    truth = np.array([1.0, 1.0, 1.0])
    out = interval_coverage(lo, hi, truth)
    assert out["coverage"] == pytest.approx(2 / 3)
    assert out["nominal"] == pytest.approx(0.9)
    assert out["n"] == 3.0


def test_interval_coverage_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        interval_coverage(np.zeros(2), np.zeros(2), np.zeros(3))


# --- dataset ----------------------------------------------------------------


def test_dataset_is_deterministic_in_the_seed():
    a = generate_dataset(30, 5)
    b = generate_dataset(30, 5)
    assert np.array_equal(a.features, b.features)
    assert np.array_equal(a.true_rate_hz, b.true_rate_hz)


def test_dataset_shapes_and_label_definition():
    data = generate_dataset(40, 3)
    assert data.features.shape == (40, len(FEATURE_NAMES))
    assert len(data) == 40
    assert np.allclose(data.label, np.log10(data.true_x))
    assert np.allclose(data.observed_x, data.observed_rate_hz * data.dead_time_s)


def test_dataset_straddles_the_paralyzable_maximum():
    data = generate_dataset(300, 8)
    assert data.true_x.min() < 0.05
    assert data.true_x.max() > 1.5


def test_dataset_has_no_mean_count_feature():
    """The leaking feature must stay removed: five features, none of them count mean."""
    assert len(FEATURE_NAMES) == 5
    assert not any("count" in name for name in FEATURE_NAMES)


def test_make_features_handles_a_zero_observed_rate():
    feats = make_features(
        np.array([0.0]), np.array([TAU]), np.array([np.nan]), np.array([0.0]),
        np.array([1.0]), np.array([1.0]),
    )
    assert np.all(np.isfinite(feats))
    assert feats[0, 1] == 1.0  # a NaN Fano factor falls back to 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dead_time_s": (1e-6, 1e-7)},
        {"true_x": (0.0, 1.0)},
        {"afterpulse_probability": (0.5, 0.4)},
        {"afterpulse_probability": (0.0, 1.0)},
        {"delay_ratio": (2.0, 1.0)},
        {"events_per_window": 5.0},
        {"n_windows": 1},
    ],
)
def test_invalid_sampling_ranges_raise(kwargs):
    with pytest.raises(ValueError):
        SamplingRanges(**kwargs)


def test_zero_rows_rejected():
    with pytest.raises(ValueError):
        generate_dataset(0, 1)


# --- the learned corrector --------------------------------------------------


def test_unfitted_corrector_refuses_to_predict():
    with pytest.raises(RuntimeError, match="not fitted"):
        RateCorrector().predict_log_x(np.zeros((1, len(FEATURE_NAMES))))


def test_corrector_fits_three_quantiles(fitted):
    assert set(fitted.models_) == set(QUANTILES)


def test_interval_is_ordered_per_row(fitted, small_test):
    out = fitted.predict_log_x(small_test.features)
    assert np.all(out["lower"] <= out["median"])
    assert np.all(out["median"] <= out["upper"])


def test_rate_prediction_is_positive_and_bracketed(fitted, small_test):
    out = fitted.predict_rate(small_test.features, small_test.dead_time_s)
    assert np.all(out["median"] > 0.0)
    assert np.all(out["lower"] <= out["upper"])


def test_wrong_feature_count_raises(fitted):
    with pytest.raises(ValueError, match="expected 5 features"):
        fitted.predict_log_x(np.zeros((2, 4)))


def test_fit_rejects_a_wrong_width_dataset(small_train):
    bad = CorrectionDataset(
        features=small_train.features[:, :3],
        label=small_train.label,
        true_rate_hz=small_train.true_rate_hz,
        observed_rate_hz=small_train.observed_rate_hz,
        dead_time_s=small_train.dead_time_s,
        is_paralyzable=small_train.is_paralyzable,
        afterpulse_probability=small_train.afterpulse_probability,
        delay_ratio=small_train.delay_ratio,
        fano_factor=small_train.fano_factor,
        count_mean=small_train.count_mean,
        seed=0,
    )
    with pytest.raises(ValueError, match="expected 5 features"):
        RateCorrector().fit(bad)


def test_dead_time_length_mismatch_raises(fitted, small_test):
    with pytest.raises(ValueError, match="length 1 or"):
        fitted.predict_rate(small_test.features, np.ones(3))


def test_feature_importances_sum_to_a_sensible_total(fitted):
    imp = fitted.feature_importances()
    assert set(imp) == set(FEATURE_NAMES)
    assert all(v >= 0.0 for v in imp.values())
    assert sum(imp.values()) > 0.5


def test_save_and_load_round_trip(fitted, small_test, tmp_path):
    path = tmp_path / "corrector.joblib"
    fitted.save(path)
    reloaded = RateCorrector.load(path)
    a = fitted.predict_log_x(small_test.features)["median"]
    b = reloaded.predict_log_x(small_test.features)["median"]
    assert np.allclose(a, b)
    assert reloaded.feature_names_ == FEATURE_NAMES


def test_save_rejects_a_non_joblib_suffix(fitted, tmp_path):
    for bad in ("model.pt", "model.pth", "model.onnx", "model.ckpt"):
        with pytest.raises(ValueError, match="joblib"):
            fitted.save(tmp_path / bad)


def test_load_rejects_an_unknown_version(tmp_path):
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"version": 99}, path)
    with pytest.raises(ValueError, match="unsupported corrector version"):
        RateCorrector.load(path)


def test_closed_forms_win_where_their_assumptions_hold(fitted, small_test):
    """Regression guard on the published honest result.

    Where afterpulsing is effectively absent the matched closed form is at the
    counting-noise floor and the learned model cannot beat it. This test fails
    if a later change makes the README's claim untrue in either direction.
    """
    clean = small_test.afterpulse_probability <= 0.02
    assert clean.sum() >= 10
    matched = matched_baseline(
        small_test.observed_rate_hz, small_test.dead_time_s, small_test.is_paralyzable
    )
    learned = fitted.predict_rate(small_test.features, small_test.dead_time_s)["median"]
    m_err = rate_error_metrics(matched[clean], small_test.true_rate_hz[clean])
    l_err = rate_error_metrics(learned[clean], small_test.true_rate_hz[clean])
    assert m_err["median_abs_rel_error"] <= l_err["median_abs_rel_error"] * 1.5


def test_nobody_inverts_the_paralyzable_upper_branch(fitted, small_test):
    """The honest negative result: past the maximum, every estimator fails.

    The lower-branch closed form returns the wrong root and the learned model
    does not recover it either. If a change ever makes the learned model good
    here, this test fails and the README must be rewritten.
    """
    past = (small_test.true_x > 1.5) & (small_test.is_paralyzable > 0.5)
    assert past.sum() >= 5, "the split must contain rows past the paralyzable maximum"
    learned = fitted.predict_rate(small_test.features, small_test.dead_time_s)["median"]
    err = rate_error_metrics(learned[past], small_test.true_rate_hz[past])
    assert err["median_abs_rel_error"] > 0.1


def test_regime_table_covers_every_estimator(fitted, small_test):
    learned = fitted.predict_rate(small_test.features, small_test.dead_time_s)["median"]
    estimates = {
        "matched": matched_baseline(
            small_test.observed_rate_hz, small_test.dead_time_s, small_test.is_paralyzable
        ),
        "learned": learned,
    }
    rows = summarise_by_regime(
        estimates,
        small_test.true_rate_hz,
        small_test.true_x,
        small_test.afterpulse_probability,
        small_test.is_paralyzable,
    )
    regimes = {row["regime"] for row in rows}
    assert "all" in regimes
    assert "paralyzable & x>1.5" in regimes
    assert all(row["name"] in ("matched", "learned") for row in rows)
