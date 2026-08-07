"""Tests for the learned model, its baselines, and its prediction intervals.

The model used here is the small session-scoped fixture (240 scenarios x 16
altitudes), not the production configuration; tolerances are set accordingly.
Production numbers are in ``validation/benchmark_results.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from cncast.dataset import (
    default_altitude_grid,
    generate_scenarios,
    met_features,
    profile_cn2,
)
from cncast.model import (
    ClimatologyBaseline,
    CnCastModel,
    Hv57Baseline,
    SlcBaseline,
    interval_coverage,
    train_default_model,
)

# ---------------------------------------------------------------- baseline predictors


def test_hv57_baseline_matches_the_analytic_profile() -> None:
    """Hv57Baseline is the analytic HV 5/7 curve, read off column 7 (log10 h)."""
    from cncast.baselines import hv57

    h = np.geomspace(5.0, 20_000.0, 20)
    x = met_features(15.0, 5.0, 50.0, 12.0, 180, h)
    assert np.allclose(10.0 ** Hv57Baseline().predict_log10_cn2(x), hv57(h), rtol=1e-12)


def test_slc_baseline_switches_on_hour_of_day() -> None:
    """SLC-Day is used 07:00-17:00, SLC-Night otherwise; hour is recovered from
    the (sin, cos) encoding, so this also tests that inversion."""
    from cncast.baselines import slc_day, slc_night

    h = np.array([50.0])
    day = 10.0 ** SlcBaseline().predict_log10_cn2(met_features(15.0, 5.0, 50.0, 12.0, 180, h))
    night = 10.0 ** SlcBaseline().predict_log10_cn2(met_features(15.0, 5.0, 50.0, 2.0, 180, h))
    assert day[0] == pytest.approx(float(slc_day(h)[0]), rel=1e-9)
    assert night[0] == pytest.approx(float(slc_night(h)[0]), rel=1e-9)


def test_climatology_baseline_requires_fitting() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        ClimatologyBaseline().predict_log10_cn2(np.zeros((2, 8)))


# ---------------------------------------------------------------- benchmark vs baseline


def test_learned_model_beats_hv57_on_held_out_data(small_model) -> None:
    """The mandated benchmark: learned model vs HV 5/7 on the same held-out rows.

    Note what this does and does not show.  The data is generated from an H-V
    family whose parameters depend on the very features the model is given, so a
    model that recovers the generator MUST beat a single fixed climatological
    curve.  The test guards against regression, not against the atmosphere.
    """
    model, art = small_model
    x_te, y_te = art["x_test"], art["y_test"]
    ml = float(np.sqrt(np.mean((model.predict_log10_cn2(x_te) - y_te) ** 2)))
    hv = float(np.sqrt(np.mean((Hv57Baseline().predict_log10_cn2(x_te) - y_te) ** 2)))
    assert ml < hv
    assert ml < 0.35  # dex; production configuration reaches 0.21


def test_learned_model_beats_the_training_climatology(small_model) -> None:
    """The stronger comparator: the mean training profile, which uses no weather.

    If this failed, the model would be learning nothing beyond the average
    shape, and the meteorological inputs would be decorative.
    """
    model, art = small_model
    x_tr, y_tr = art["x_train"], art["y_train"]
    x_te, y_te = art["x_test"], art["y_test"]
    clim = ClimatologyBaseline().fit(x_tr, y_tr)
    ml = float(np.sqrt(np.mean((model.predict_log10_cn2(x_te) - y_te) ** 2)))
    cl = float(np.sqrt(np.mean((clim.predict_log10_cn2(x_te) - y_te) ** 2)))
    assert ml < cl


# ---------------------------------------------------------------- intervals


def test_interval_coverage_is_near_nominal_on_held_out_data(small_model) -> None:
    """Empirical coverage of the calibrated 90 % interval on held-out scenarios.

    Conformal calibration (Romano et al. 2019) targets 90 %; with 60 held-out
    scenarios the sampling scatter is a few per cent, so the band is 0.83-0.96.
    The production run measures 0.899 (validation/benchmark_results.md §3).
    """
    model, art = small_model
    lo, _, hi = model._three(art["x_test"])
    coverage, width = interval_coverage(art["y_test"], lo, hi)
    assert 0.83 <= coverage <= 0.96
    assert 0.2 < width < 1.5  # dex; a wider interval would be useless


def test_calibration_widens_the_raw_quantile_interval(small_model) -> None:
    """The raw quantile models under-cover; the conformal step must widen them."""
    model, _ = small_model
    assert model.conformal_delta_dex > 0.0


def test_uncalibrated_model_undercovers(small_model) -> None:
    """Documented failure mode: without calibration the interval is too narrow."""
    _, art = small_model
    raw = CnCastModel(random_state=7).fit(art["x_train"], art["y_train"])
    lo, _, hi = raw._three(art["x_test"])
    raw_cov, _ = interval_coverage(art["y_test"], lo, hi)
    assert raw_cov < 0.90


def test_interval_brackets_the_median_everywhere(small_model) -> None:
    model, _ = small_model
    pred = model.predict(20.0, 4.0, 55.0, 13.0, 200, default_altitude_grid(24))
    assert np.all(pred.cn2_lower <= pred.cn2)
    assert np.all(pred.cn2 <= pred.cn2_upper)
    assert np.all(pred.interval_width_dex >= 0.0)


def test_coverage_helper_known_answer() -> None:
    """3 of 4 points inside [0, 1] -> coverage 0.75, mean width 1.0."""
    cov, width = interval_coverage([0.5, 0.2, 0.9, 1.5], [0, 0, 0, 0], [1, 1, 1, 1])
    assert cov == pytest.approx(0.75)
    assert width == pytest.approx(1.0)


# ---------------------------------------------------------------- prediction behaviour


def test_prediction_is_physically_plausible(small_model) -> None:
    """Predicted Cn^2 stays inside the range spanned by the published models and
    falls by orders of magnitude from the surface to 20 km."""
    model, _ = small_model
    grid = default_altitude_grid(24)
    pred = model.predict(25.0, 3.0, 40.0, 12.0, 200, grid)
    assert np.all(pred.cn2 > 1e-21) and np.all(pred.cn2 < 1e-12)
    assert pred.cn2[0] / pred.cn2[-1] > 100.0


def test_prediction_decreases_through_the_lower_troposphere(small_model) -> None:
    """Cn^2 decreases with altitude above the boundary layer, on the predictions.

    Checked between fixed altitude pairs rather than pointwise, because the tree
    ensemble is piecewise constant in log-altitude and can be locally flat.
    """
    model, _ = small_model
    for hour in (2.0, 8.0, 13.0, 19.0):
        pred = model.predict(18.0, 6.0, 60.0, hour, 150, np.array([50.0, 500.0, 5000.0, 20000.0]))
        assert np.all(np.diff(pred.cn2) < 0.0)


def test_daytime_surface_prediction_exceeds_night(small_model) -> None:
    """The learned model reproduces the diurnal cycle the generator encodes."""
    model, _ = small_model
    h = np.array([10.0])
    noon = model.predict(25.0, 5.0, 50.0, 12.0, 200, h).cn2[0]
    midnight = model.predict(25.0, 5.0, 50.0, 0.0, 200, h).cn2[0]
    assert noon > 2.0 * midnight


def test_extrapolation_flag(small_model) -> None:
    """Queries outside the training domain must be flagged, not silently served."""
    model, _ = small_model
    grid = default_altitude_grid(8)
    assert not model.predict(20.0, 5.0, 50.0, 12.0, 180, grid).extrapolating
    assert model.predict(55.0, 5.0, 50.0, 12.0, 180, grid).extrapolating  # T above training max
    assert model.predict(20.0, 25.0, 50.0, 12.0, 180, grid).extrapolating  # wind above max
    assert model.predict(20.0, 5.0, 2.0, 12.0, 180, grid).extrapolating  # RH below min
    assert model.predict(20.0, 5.0, 50.0, 12.0, 180, np.array([1.0])).extrapolating  # h below min


def test_predict_scenario_ignores_latent_state(small_model) -> None:
    """predict_scenario must use only the five observable met fields."""
    model, _ = small_model
    sc = generate_scenarios(1, seed=31)[0]
    grid = default_altitude_grid(12)
    a = model.predict_scenario(sc, grid)
    b = model.predict(
        sc.surface_temp_c,
        sc.surface_wind_m_s,
        sc.relative_humidity_pct,
        sc.hour_of_day,
        sc.day_of_year,
        grid,
    )
    assert np.array_equal(a.cn2, b.cn2)


def test_prediction_error_against_truth_is_bounded(small_model) -> None:
    """Sanity: median prediction is within 1 dex of truth for a typical scenario."""
    model, art = small_model
    sc = art["test_scenarios"][0]
    grid = default_altitude_grid(24)
    err = np.abs(np.log10(model.predict_scenario(sc, grid).cn2) - np.log10(profile_cn2(sc, grid)))
    assert float(np.max(err)) < 1.0


# ---------------------------------------------------------------- reproducibility


def test_training_is_reproducible() -> None:
    """Same seeds -> identical predictions and identical conformal delta."""
    m1, a1 = train_default_model(n_scenarios=60, n_altitudes=8)
    m2, a2 = train_default_model(n_scenarios=60, n_altitudes=8)
    p1 = m1.predict_log10_cn2(a1["x_test"])
    p2 = m2.predict_log10_cn2(a2["x_test"])
    assert np.array_equal(a1["x_test"], a2["x_test"])
    assert np.array_equal(p1, p2)
    assert m1.conformal_delta_dex == m2.conformal_delta_dex


def test_different_data_seed_changes_the_model() -> None:
    m1, a1 = train_default_model(n_scenarios=60, n_altitudes=8, data_seed=1)
    m2, _ = train_default_model(n_scenarios=60, n_altitudes=8, data_seed=2)
    assert not np.array_equal(
        m1.predict_log10_cn2(a1["x_test"]), m2.predict_log10_cn2(a1["x_test"])
    )


# ---------------------------------------------------------------- input validation


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        CnCastModel().predict(20.0, 5.0, 50.0, 12.0, 180, np.array([100.0]))


def test_calibrate_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        CnCastModel().calibrate(np.zeros((30, 8)), np.zeros(30))


def test_bad_coverage_rejected() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="coverage"):
            CnCastModel(coverage=bad)


def test_fit_rejects_wrong_feature_count() -> None:
    with pytest.raises(ValueError, match="8 feature columns"):
        CnCastModel().fit(np.zeros((10, 5)), np.zeros(10))


def test_fit_rejects_non_finite_targets() -> None:
    with pytest.raises(ValueError, match="finite"):
        CnCastModel().fit(np.zeros((10, 8)), np.full(10, np.nan))


def test_calibrate_rejects_tiny_calibration_sets(small_model) -> None:
    model, art = small_model
    with pytest.raises(ValueError, match="20 calibration"):
        model.calibrate(art["x_test"][:5], art["y_test"][:5])


def test_predict_rejects_invalid_weather(small_model) -> None:
    model, _ = small_model
    with pytest.raises(ValueError, match="relative_humidity_pct"):
        model.predict(20.0, 5.0, -5.0, 12.0, 180, np.array([100.0]))


def test_interval_coverage_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        interval_coverage([1.0, 2.0], [0.0], [3.0])


def test_interval_coverage_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        interval_coverage([], [], [])
