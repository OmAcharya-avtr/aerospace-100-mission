"""Population predictors: the analytic forms, the learned one, and conformal wrapping."""

from __future__ import annotations

import numpy as np
import pytest

from orbitdrift.errors import InsufficientDataError
from orbitdrift.features import ALL_FEATURE_NAMES, FEATURE_NAMES
from orbitdrift.predictors import (
    ANALYTIC_FORMS,
    AnalyticGrowthPredictor,
    ConformalPredictor,
    LearnedQuantilePredictor,
    RuleOfThumbPredictor,
    interval_crossing_fraction,
)

_PERIGEE = FEATURE_NAMES.index("perigee_altitude_km")


def _synthetic(n_objects: int = 40, n_ages: int = 14, seed: int = 3):
    """A population whose drift is an exact power law in age times a feature factor.

    Built so the analytic form is the truth: ``drift = 0.1 * (alt/700)^-1.5 *
    t^1.6`` with a multiplicative log-normal scatter. A predictor that cannot
    recover this cannot recover anything.
    """
    rng = np.random.default_rng(seed)
    altitude = rng.uniform(350.0, 1500.0, n_objects)
    ages = np.linspace(0.25, 7.0, n_ages)
    rows = []
    drift = []
    groups = []
    for index in range(n_objects):
        static = np.zeros(len(FEATURE_NAMES))
        static[_PERIGEE] = altitude[index]
        static[FEATURE_NAMES.index("apogee_altitude_km")] = altitude[index] + 20.0
        static[FEATURE_NAMES.index("eccentricity")] = 0.002
        static[FEATURE_NAMES.index("inclination_deg")] = rng.uniform(10.0, 98.0)
        static[FEATURE_NAMES.index("bstar")] = 1e-4
        static[FEATURE_NAMES.index("log10_abs_bstar")] = -4.0
        static[FEATURE_NAMES.index("bstar_sign")] = 1.0
        static[FEATURE_NAMES.index("period_min")] = 95.0
        static[FEATURE_NAMES.index("log10_perigee_density")] = -12.0
        static[FEATURE_NAMES.index("log10_drag_parameter")] = -15.0
        scatter = np.exp(rng.normal(0.0, 0.15))
        for age in ages:
            rows.append(np.concatenate([static, [age]]))
            drift.append(0.1 * (altitude[index] / 700.0) ** -1.5 * age**1.6 * scatter)
            groups.append(index)
    return np.array(rows), np.array(drift), np.array(groups)


def test_rule_of_thumb_ignores_its_training_data_entirely() -> None:
    x, y, _ = _synthetic()
    first = RuleOfThumbPredictor().fit(x, y)
    second = RuleOfThumbPredictor().fit(x, y * 1000.0)
    assert np.allclose(first.predict_median_km(x), second.predict_median_km(x))
    assert np.allclose(first.predict_median_km(x), x[:, -1])


def test_rule_of_thumb_band_is_fixed_and_ignores_the_requested_coverage() -> None:
    x, y, _ = _synthetic(n_objects=4)
    predictor = RuleOfThumbPredictor(rate_km_per_day=2.0, band_factor=3.0).fit(x, y)
    low, high = predictor.predict_interval_km(x, 0.50)
    wide_low, wide_high = predictor.predict_interval_km(x, 0.99)
    assert np.allclose(low, wide_low)
    assert np.allclose(high, wide_high)
    assert np.allclose(high / low, 9.0)


@pytest.mark.parametrize("form", ANALYTIC_FORMS)
def test_every_analytic_form_fits_and_predicts_positive_finite_values(form: str) -> None:
    x, y, _ = _synthetic()
    predictor = AnalyticGrowthPredictor(form=form).fit(x, y)
    median = predictor.predict_median_km(x)
    assert np.all(np.isfinite(median))
    assert np.all(median > 0.0)
    low, high = predictor.predict_interval_km(x, 0.90)
    assert np.all(low < median)
    assert np.all(median < high)


def test_free_power_form_recovers_the_generating_exponent() -> None:
    """The truth is ``t^1.6``, so the fitted global exponent must come back at 1.6."""
    x, y, _ = _synthetic(n_objects=60, n_ages=20, seed=5)
    predictor = AnalyticGrowthPredictor(form="power").fit(x, y)
    assert predictor.effective_exponent == pytest.approx(1.6, rel=0.0, abs=0.02)


def test_power_altitude_form_reduces_to_the_power_form_when_the_exponent_is_flat() -> None:
    """With an altitude-independent truth, the altitude slope must be near zero."""
    x, y, _ = _synthetic(n_objects=60, n_ages=20, seed=7)
    predictor = AnalyticGrowthPredictor(form="power_altitude").fit(x, y)
    low = predictor.exponent_at_altitude(400.0)
    high = predictor.exponent_at_altitude(1400.0)
    assert low == pytest.approx(1.6, rel=0.0, abs=0.08)
    assert high == pytest.approx(1.6, rel=0.0, abs=0.08)
    assert abs(low - high) < 0.1


def test_fixed_exponent_forms_report_their_fixed_exponent() -> None:
    x, y, _ = _synthetic()
    assert AnalyticGrowthPredictor(form="linear").fit(x, y).effective_exponent == 1.0
    assert AnalyticGrowthPredictor(form="quadratic").fit(x, y).effective_exponent == 2.0


def test_drag_exponential_form_finds_a_timescale_inside_its_bounds() -> None:
    x, y, _ = _synthetic()
    predictor = AnalyticGrowthPredictor(form="drag_exponential").fit(x, y)
    low, high = predictor.timescale_bounds_days
    assert low <= predictor.timescale_days_ <= high
    assert np.isnan(predictor.effective_exponent)


def test_unknown_analytic_form_is_refused() -> None:
    with pytest.raises(ValueError, match="form must be one of"):
        AnalyticGrowthPredictor(form="cubic")


def test_analytic_predictor_refuses_too_few_rows() -> None:
    x, y, _ = _synthetic(n_objects=1, n_ages=3)
    with pytest.raises(InsufficientDataError, match="too few to fit"):
        AnalyticGrowthPredictor().fit(x, y)


def test_analytic_predictor_refuses_to_predict_before_fitting() -> None:
    x, _, _ = _synthetic(n_objects=2)
    with pytest.raises(InsufficientDataError, match="not fitted"):
        AnalyticGrowthPredictor().predict_median_km(x)


def test_analytic_predictor_survives_a_feature_with_no_spread() -> None:
    """A constant column must not divide by zero; its coefficient is unidentified.

    Every synthetic population here has ``bstar_sign`` identically 1.0, which
    is exactly such a column, so this is not a contrived case.
    """
    x, y, _ = _synthetic()
    assert float(np.ptp(x[:, FEATURE_NAMES.index("bstar_sign")])) == 0.0
    predictor = AnalyticGrowthPredictor().fit(x, y)
    assert np.all(np.isfinite(predictor.coefficients_))


def test_design_matrix_with_the_wrong_number_of_columns_is_refused() -> None:
    x, y, _ = _synthetic(n_objects=4)
    with pytest.raises(ValueError, match="must have 11 columns"):
        AnalyticGrowthPredictor().fit(x[:, :5], y)
    assert len(ALL_FEATURE_NAMES) == 11


def test_learned_predictor_fits_and_its_quantiles_bracket_its_median() -> None:
    x, y, _ = _synthetic(n_objects=60, n_ages=16, seed=11)
    predictor = LearnedQuantilePredictor(n_estimators=60).fit(x, y)
    median = predictor.predict_median_km(x)
    low, high = predictor.predict_interval_km(x, 0.90)
    assert np.all(np.isfinite(median))
    assert interval_crossing_fraction(low, high) < 0.02


def test_learned_predictor_refuses_a_coverage_it_was_not_fitted_for() -> None:
    """Rescaling a quantile pair to another level would be a fabrication."""
    x, y, _ = _synthetic(n_objects=50, n_ages=12)
    predictor = LearnedQuantilePredictor(n_estimators=40).fit(x, y)
    with pytest.raises(ValueError, match="was fitted for coverage"):
        predictor.predict_interval_km(x, 0.50)


def test_learned_predictor_refuses_too_few_rows() -> None:
    x, y, _ = _synthetic(n_objects=2, n_ages=4)
    with pytest.raises(InsufficientDataError, match="too few for a 300-tree"):
        LearnedQuantilePredictor().fit(x, y)


def test_learned_predictor_refuses_to_predict_before_fitting() -> None:
    x, _, _ = _synthetic(n_objects=2)
    with pytest.raises(InsufficientDataError, match="not fitted"):
        LearnedQuantilePredictor().predict_median_km(x)


def test_learned_predictor_is_deterministic_for_a_fixed_seed() -> None:
    x, y, _ = _synthetic(n_objects=50, n_ages=12, seed=13)
    first = LearnedQuantilePredictor(n_estimators=50, random_state=99).fit(x, y)
    second = LearnedQuantilePredictor(n_estimators=50, random_state=99).fit(x, y)
    assert np.allclose(first.predict_median_km(x), second.predict_median_km(x))


def test_conformal_interval_is_symmetric_in_the_log_and_covers_near_nominal() -> None:
    """Split conformal must deliver about its nominal coverage on exchangeable data.

    The synthetic population has one scatter factor per object, so the rows
    are grouped and exchangeability holds at the object level. The split here
    is by group, and the measured coverage on the training population must
    land near 0.90. This is a sanity check on the implementation, not a
    published coverage figure: the published ones are on held-out objects of
    the real divergence dataset.
    """
    x, y, groups = _synthetic(n_objects=120, n_ages=14, seed=17)
    predictor = ConformalPredictor(AnalyticGrowthPredictor(form="power"), coverage=0.90)
    predictor.fit(x, y, groups=groups)
    low, high = predictor.predict_interval_km(x)
    median = predictor.predict_median_km(x)
    assert np.allclose(high / median, median / low, rtol=1e-12, atol=0.0)
    covered = float(np.mean((y >= low) & (y <= high)))
    assert 0.84 < covered < 0.97
    assert predictor.n_calibration_ > 0


def test_conformal_group_split_leaves_no_object_in_both_halves() -> None:
    """A row-wise split would leak an object's own physics into its calibration.

    Checked by fitting with and without ``groups`` and requiring the
    half-widths to differ: if the group split were being ignored, they would
    be the same.
    """
    x, y, groups = _synthetic(n_objects=80, n_ages=14, seed=19)
    grouped = ConformalPredictor(AnalyticGrowthPredictor(form="power"), random_state=5)
    grouped.fit(x, y, groups=groups)
    by_row = ConformalPredictor(AnalyticGrowthPredictor(form="power"), random_state=5)
    by_row.fit(x, y)
    assert grouped.log_half_width_ != by_row.log_half_width_


def test_conformal_refuses_a_coverage_other_than_the_one_it_calibrated() -> None:
    x, y, groups = _synthetic(n_objects=60)
    predictor = ConformalPredictor(AnalyticGrowthPredictor(form="power"), coverage=0.90)
    predictor.fit(x, y, groups=groups)
    with pytest.raises(ValueError, match="calibrated for coverage"):
        predictor.predict_interval_km(x, 0.80)


def test_conformal_refuses_an_out_of_range_calibration_fraction() -> None:
    for fraction in (0.0, 1.0, -0.2, 1.4):
        with pytest.raises(ValueError, match="calibration_fraction must be in"):
            ConformalPredictor(AnalyticGrowthPredictor(), calibration_fraction=fraction)


def test_conformal_widens_as_the_requested_coverage_rises() -> None:
    x, y, groups = _synthetic(n_objects=120, n_ages=14, seed=23)
    widths = []
    for coverage in (0.50, 0.80, 0.95):
        predictor = ConformalPredictor(
            AnalyticGrowthPredictor(form="power"), coverage=coverage, random_state=5
        )
        predictor.fit(x, y, groups=groups)
        widths.append(predictor.log_half_width_)
    assert widths[0] < widths[1] < widths[2]


def test_conformal_names_itself_after_its_base() -> None:
    predictor = ConformalPredictor(AnalyticGrowthPredictor(form="linear"))
    assert predictor.name == "conformal[analytic_linear]"


def test_interval_crossing_fraction_counts_crossings_and_handles_empty_input() -> None:
    assert interval_crossing_fraction(np.array([1.0, 5.0]), np.array([2.0, 3.0])) == 0.5
    assert interval_crossing_fraction(np.array([]), np.array([])) == 0.0


def test_leverage_separates_real_rows_from_an_inconsistent_feature_vector(
    committed_table,
) -> None:
    """The collinearity diagnostic, measured on the real committed population.

    The ten features are strongly collinear: perigee altitude, apogee
    altitude, period, the logarithm of the perigee density and the logarithm
    of the drag parameter are all functions of the same orbit. A feature
    vector that is internally inconsistent -- a 1300 km perigee with an apogee
    25 km above it and a period of 105.6 minutes, where a 1300 by 1325 km
    orbit has a period of 112 -- lies off the manifold the model was fitted
    on, and the model's answer there is an extrapolation of a collinear fit
    rather than a prediction. That is not hypothetical: such a vector drew a
    predicted divergence of 2.5e8 km from this model against the 17.7 km
    measured for a real object at that altitude.

    The synthetic population used by the other tests in this module cannot
    exercise this, because several of its columns are constants and a
    pseudo-inverse assigns a zero-variance direction no weight. The check
    therefore uses the committed table, whose features came from real element
    sets and carry the real collinearity.
    """
    from orbitdrift.dataset import split_by_object

    train, test = split_by_object(committed_table, test_fraction=0.3)
    design = committed_table.design()
    drift = committed_table.column("drift_km")
    predictor = AnalyticGrowthPredictor(form="power_altitude").fit(design[train], drift[train])
    assert predictor.training_leverage_max_ > 1.0
    assert float(np.max(predictor.leverage(design[train]))) == pytest.approx(
        predictor.training_leverage_max_, rel=1e-9, abs=0.0
    )
    # Held-out rows are on the manifold: their leverage is of the same order.
    assert float(np.max(predictor.leverage(design[test]))) < (
        2.0 * predictor.training_leverage_max_
    )

    inconsistent = design[train][0].copy()
    inconsistent[_PERIGEE] = 1300.0
    inconsistent[FEATURE_NAMES.index("apogee_altitude_km")] = 1325.0
    inconsistent[FEATURE_NAMES.index("period_min")] = 105.6
    inconsistent[FEATURE_NAMES.index("log10_perigee_density")] = -15.0
    inconsistent[FEATURE_NAMES.index("log10_drag_parameter")] = -18.76
    assert float(predictor.leverage(inconsistent[None, :])[0]) > (
        3.0 * predictor.training_leverage_max_
    )


def test_leverage_refuses_before_fitting() -> None:
    x, _, _ = _synthetic(n_objects=2)
    with pytest.raises(InsufficientDataError, match="not fitted"):
        AnalyticGrowthPredictor().leverage(x)
