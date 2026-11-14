"""The learned predictor, the licence check that guards it, and its baselines."""

from __future__ import annotations

import time

import numpy as np
import pytest

from powerenvelope.baselines import floor_power
from powerenvelope.predictor import (
    EXCLUDED_DESCRIPTOR_CLASSES,
    LICENSED_DESCRIPTORS,
    ActivePowerPredictor,
    AnalyticActivePowerModel,
    ConstantPowerPredictor,
    DescriptorLeakError,
    LinearActivePowerModel,
    assert_descriptors_licensed,
    score,
    target_quantisation_sd,
)
from powerenvelope.windows import FEATURE_NAMES, extract_windows, grouped_splits

# -------------------------------------------------- the licence check, by name


def test_licensed_set_is_the_window_descriptor_set():
    assert LICENSED_DESCRIPTORS == FEATURE_NAMES


def test_licence_check_passes_the_licensed_set():
    assert_descriptors_licensed(LICENSED_DESCRIPTORS)


@pytest.mark.parametrize(
    ("name", "expected_class"),
    [
        ("vdd_in_mw", "the pipeline's own measured power"),
        ("mean_active_power_mw", "the pipeline's own measured power"),
        ("mean_latency_s", "the pipeline's own measured latency"),
        ("cpu_busy_fraction", "the pipeline's own CPU occupancy"),
        ("active_cpu_s", "the pipeline's own CPU occupancy"),
        ("inferences", "the pipeline's realised inference count"),
        ("n_inf_per_cycle", "the pipeline's realised inference count"),
        ("acceptance_rate", "the pipeline's realised acceptance rate"),
        ("n_candidates", "the pipeline's realised acceptance rate"),
    ],
)
def test_licence_check_fires_on_each_excluded_class(name, expected_class):
    with pytest.raises(DescriptorLeakError) as error:
        assert_descriptors_licensed((*LICENSED_DESCRIPTORS, name))
    assert expected_class in str(error.value)
    assert name in str(error.value)


def test_licence_check_fires_on_an_unanticipated_name():
    # The membership test is exact, so a quantity nobody listed is still
    # rejected. That is what makes the check a check rather than a blacklist.
    with pytest.raises(DescriptorLeakError, match="not in the licensed set"):
        assert_descriptors_licensed(("moon_phase",))


def test_licence_check_rejects_a_bare_string():
    with pytest.raises(TypeError, match="sequence of descriptor name"):
        assert_descriptors_licensed("duty")


def test_every_excluded_class_from_the_specification_is_named():
    assert set(EXCLUDED_DESCRIPTOR_CLASSES) == {
        "the pipeline's own measured power",
        "the pipeline's own measured latency",
        "the pipeline's own CPU occupancy",
        "the pipeline's realised inference count",
        "the pipeline's realised acceptance rate",
    }


def test_fit_refuses_an_unlicensed_column(campaign):
    windows = extract_windows(campaign)
    names = (*FEATURE_NAMES[:-1], "vdd_in_mw")
    with pytest.raises(DescriptorLeakError, match="vdd_in_mw"):
        ActivePowerPredictor().fit(windows.features, windows.targets, names)


def test_fit_refuses_a_name_count_that_does_not_match(campaign):
    windows = extract_windows(campaign)
    with pytest.raises(ValueError, match="descriptor name"):
        ActivePowerPredictor().fit(
            windows.features, windows.targets, FEATURE_NAMES[:-1]
        )


# -------------------------------------------------------------- the forest


def test_fit_and_predict_shapes(campaign):
    windows = extract_windows(campaign)
    model = ActivePowerPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    prediction = model.predict(windows.features)
    assert prediction.power_mw.shape == windows.targets.shape
    assert prediction.sigma_mw.shape == windows.targets.shape
    assert np.all(prediction.sigma_mw >= 0.0)
    assert np.all(prediction.sigma_target_mw == 0.0)


def test_uncertainty_grows_when_the_target_noise_is_supplied(campaign):
    windows = extract_windows(campaign)
    model = ActivePowerPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    bare = model.predict(windows.features)
    with_noise = model.predict(windows.features, quantisation_step_mw=40.704, n_samples=160)
    assert np.all(with_noise.sigma_mw >= bare.sigma_mw)
    assert np.any(with_noise.sigma_mw > bare.sigma_mw)


def test_forest_cannot_predict_outside_its_training_range(campaign):
    # A property of the estimator, not of its API: every leaf is a mean of
    # training targets, so no prediction can leave their range.
    windows = extract_windows(campaign)
    model = ActivePowerPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    prediction = model.predict(windows.features * 10.0)
    assert float(prediction.power_mw.min()) >= float(windows.targets.min()) - 1e-9
    assert float(prediction.power_mw.max()) <= float(windows.targets.max()) + 1e-9


def test_importances_are_named_and_sum_to_one(campaign):
    windows = extract_windows(campaign)
    model = ActivePowerPredictor().fit(
        windows.features, windows.targets, windows.feature_names
    )
    importances = model.importances()
    assert set(importances) == set(FEATURE_NAMES)
    assert sum(importances.values()) == pytest.approx(1.0, rel=1e-9, abs=0.0)


def test_predict_before_fit_raises(campaign):
    with pytest.raises(RuntimeError, match="called before fit"):
        ActivePowerPredictor().predict(np.zeros((2, len(FEATURE_NAMES))))


def test_importances_before_fit_raises():
    with pytest.raises(RuntimeError, match="called before fit"):
        ActivePowerPredictor().importances()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"n_estimators": 1}, "n_estimators must be"), ({"max_depth": 0}, "max_depth must be")],
)
def test_constructor_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ActivePowerPredictor(**kwargs)


def test_fit_rejects_a_non_positive_target(campaign):
    windows = extract_windows(campaign)
    targets = windows.targets.copy()
    targets[0] = 0.0
    with pytest.raises(ValueError, match="positive rail powers"):
        ActivePowerPredictor().fit(windows.features, targets, windows.feature_names)


def test_fit_rejects_a_one_row_training_set(campaign):
    windows = extract_windows(campaign)
    with pytest.raises(ValueError, match="at least 2 training windows"):
        ActivePowerPredictor().fit(
            windows.features[:1], windows.targets[:1], windows.feature_names
        )


def test_fit_rejects_mismatched_shapes(campaign):
    windows = extract_windows(campaign)
    with pytest.raises(ValueError, match="does not match"):
        ActivePowerPredictor().fit(
            windows.features, windows.targets[:-1], windows.feature_names
        )


def test_fit_rejects_a_one_dimensional_matrix():
    with pytest.raises(ValueError, match="two-dimensional"):
        ActivePowerPredictor().fit(np.zeros(5), np.ones(5), ("duty",))


def test_fit_is_inside_the_compute_budget(campaign):
    # The build environment has two cores and a three-minute budget for any
    # fit. This records the measured time rather than assuming it.
    windows = extract_windows(campaign)
    started = time.perf_counter()
    ActivePowerPredictor().fit(windows.features, windows.targets, windows.feature_names)
    assert time.perf_counter() - started < 20.0


# ------------------------------------------------------------- the baselines


def test_constant_predictor_is_the_training_mean(campaign):
    windows = extract_windows(campaign)
    model = ConstantPowerPredictor().fit(windows.features, windows.targets)
    prediction = model.predict(windows.features)
    assert np.all(prediction.power_mw == pytest.approx(
        float(windows.targets.mean()), rel=1e-12, abs=0.0
    ))


def test_constant_predictor_refuses_an_empty_target(campaign):
    with pytest.raises(ValueError, match="empty target vector"):
        ConstantPowerPredictor().fit(np.zeros((0, 3)), np.zeros(0))


def test_constant_predictor_before_fit_raises():
    with pytest.raises(RuntimeError, match="called before fit"):
        ConstantPowerPredictor().predict(np.zeros((1, 1)))


def test_analytic_model_uses_the_floor_and_the_declared_duty(campaign):
    windows = extract_windows(campaign)
    floor = floor_power(campaign)
    model = AnalyticActivePowerModel().fit(campaign, windows.cells, floor_mw=floor.mean_mw)
    assert model.floor_mw_ == pytest.approx(floor.mean_mw, rel=0.0, abs=0.0)
    assert model.excess_mw_ is not None and model.excess_mw_ > 0.0
    assert len(model.cells_) == 19
    prediction = model.predict(windows.features)
    assert np.all(prediction.power_mw == prediction.power_mw[0])
    assert np.all(prediction.sigma_mw > 0.0)


def test_analytic_model_prediction_is_the_right_order_of_magnitude(campaign):
    windows = extract_windows(campaign)
    model = AnalyticActivePowerModel().fit(campaign, windows.cells)
    prediction = model.predict(windows.features[:1])
    assert float(prediction.power_mw[0]) == pytest.approx(
        float(windows.targets.mean()), rel=0.10, abs=0.0
    )


def test_analytic_model_refuses_when_no_training_cell_has_duty(campaign):
    with pytest.raises(ValueError, match="no training cell with a positive duty"):
        AnalyticActivePowerModel().fit(campaign, ("floor",))


def test_analytic_model_before_fit_raises():
    with pytest.raises(RuntimeError, match="called before fit"):
        AnalyticActivePowerModel().predict(np.zeros((1, 1)))


def test_analytic_model_never_sees_a_test_cell(campaign):
    windows = extract_windows(campaign)
    for _label, train, test in grouped_splits(windows, "duty"):
        model = AnalyticActivePowerModel().fit(campaign, windows.cells[train])
        assert not set(model.cells_) & set(windows.cells[test])


# ------------------------------------------------------------------ scoring


def test_quantisation_sd_is_the_uniform_quantiser_formula():
    # A uniform quantiser of width q has sd q / sqrt(12). For q = 40.704 mW and
    # one sample: 40.704 / 3.4641016 = 11.7502 mW. Hand calculation.
    value = float(target_quantisation_sd(40.704, 1.0))
    assert value == pytest.approx(11.75017, rel=1e-5, abs=0.0)


def test_quantisation_sd_falls_as_the_root_of_the_sample_count():
    one = float(target_quantisation_sd(40.0, 1.0))
    hundred = float(target_quantisation_sd(40.0, 100.0))
    assert hundred == pytest.approx(one / 10.0, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("step", "samples", "match"),
    [(0.0, 1.0, "step_mw must be positive"), (1.0, 0.0, "n_samples must be positive")],
)
def test_quantisation_sd_validates(step, samples, match):
    with pytest.raises(ValueError, match=match):
        target_quantisation_sd(step, samples)


def test_score_of_a_perfect_prediction():
    truth = np.asarray([1.0, 2.0, 3.0])
    result = score("perfect", truth, truth)
    assert result.mae_mw == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.rmse_mw == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.bias_mw == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert result.coverage != result.coverage  # nan: no sigma was reported


def test_score_known_answer():
    # truth 10, 20; prediction 12, 17. residuals +2, -3.
    # MAE = 2.5, RMSE = sqrt((4 + 9) / 2) = 2.5495098, bias = -0.5.
    result = score("known", np.asarray([10.0, 20.0]), np.asarray([12.0, 17.0]))
    assert result.mae_mw == pytest.approx(2.5, rel=1e-12, abs=0.0)
    assert result.rmse_mw == pytest.approx(2.5495097567963922, rel=1e-12, abs=0.0)
    assert result.bias_mw == pytest.approx(-0.5, rel=1e-12, abs=0.0)


def test_score_coverage_counts_residuals_inside_one_sigma():
    result = score(
        "cover",
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([11.0, 25.0, 30.5]),
        sigma=np.asarray([2.0, 2.0, 2.0]),
    )
    assert result.coverage == pytest.approx(2.0 / 3.0, rel=1e-12, abs=0.0)


def test_score_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        score("bad", np.zeros(3), np.zeros(2))


def test_score_rejects_an_empty_split():
    with pytest.raises(ValueError, match="empty split"):
        score("bad", np.zeros(0), np.zeros(0))


# --------------------------------------- the parametric classical alternative


def test_linear_model_fits_three_parameters_on_training_cells(campaign):
    windows = extract_windows(campaign)
    model = LinearActivePowerModel().fit(campaign, windows.cells)
    assert model.coefficients_ is not None
    assert model.coefficients_.shape == (3,)
    assert model.residual_sd_mw_ is not None and model.residual_sd_mw_ >= 0.0
    assert len(model.cells_) == 19


def test_linear_model_recovers_an_exactly_linear_target(campaign):
    # Fitted on a target that IS a + b*duty + c*dim, the coefficients must come
    # back. This is a known-answer test: a = 1000, b = 200, c = 3.
    targets = {
        cell.label: 1000.0 + 200.0 * cell.duty + 3.0 * cell.frame_dim
        for cell in campaign.pipeline_cells()
    }
    model = LinearActivePowerModel().fit(campaign, tuple(targets), targets=targets)
    assert model.coefficients_ == pytest.approx([1000.0, 200.0, 3.0], rel=1e-9, abs=1e-6)
    assert model.residual_sd_mw_ == pytest.approx(0.0, rel=0.0, abs=1e-6)
    assert model.predict_cell(campaign, "w64_d050") == pytest.approx(
        1000.0 + 200.0 * 0.5 + 3.0 * 64, rel=1e-9, abs=0.0
    )


def test_linear_model_predict_reads_the_named_columns(campaign):
    windows = extract_windows(campaign)
    model = LinearActivePowerModel().fit(campaign, windows.cells)
    prediction = model.predict(windows.features, windows.feature_names)
    assert prediction.power_mw.shape == (len(windows),)
    assert np.all(prediction.sigma_mw >= 0.0)
    # One window of w64_d050 must agree with the cell-level prediction.
    index = int(np.nonzero(windows.cells == "w64_d050")[0][0])
    assert prediction.power_mw[index] == pytest.approx(
        model.predict_cell(campaign, "w64_d050"), rel=1e-12, abs=0.0
    )


def test_linear_model_reads_only_licensed_descriptors():
    assert set(LinearActivePowerModel.DESCRIPTORS) <= set(LICENSED_DESCRIPTORS)


def test_linear_model_refuses_too_few_training_cells(campaign):
    with pytest.raises(ValueError, match="at least four are needed"):
        LinearActivePowerModel().fit(campaign, ("w64_d050", "w64_d075", "w64_d100"))


def test_linear_model_refuses_a_matrix_without_the_named_columns(campaign):
    windows = extract_windows(campaign)
    model = LinearActivePowerModel().fit(campaign, windows.cells)
    with pytest.raises(ValueError, match="duty.*frame_dim"):
        model.predict(windows.features, ("a",) * len(windows.feature_names))


def test_linear_model_before_fit_raises(campaign):
    with pytest.raises(RuntimeError, match="called before fit"):
        LinearActivePowerModel().predict(np.zeros((1, 10)))
    with pytest.raises(RuntimeError, match="called before fit"):
        LinearActivePowerModel().predict_cell(campaign, "w64_d050")


def test_linear_model_never_sees_a_test_cell(campaign):
    windows = extract_windows(campaign)
    for _label, train, test in grouped_splits(windows, "frame_dim"):
        model = LinearActivePowerModel().fit(campaign, windows.cells[train])
        assert not set(model.cells_) & set(windows.cells[test])
