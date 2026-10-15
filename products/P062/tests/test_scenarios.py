"""Design, truth-parameter and dataset tests."""

from __future__ import annotations

import numpy as np
import pytest

from twincalib.model import ThermalTwinModel
from twincalib.scenarios import (
    DEFAULT_SIGMA_K,
    DESIGN_NAMES,
    LINEAR_THETA_TRUE,
    THETA_TRUE,
    TRUTH_CONDUCTANCE_SLOPE,
    ExperimentDesign,
    build_inputs,
    calibration_model,
    design,
    designs,
    linear_design,
    linear_model,
    make_dataset,
    truth_model,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        name="probe",
        description="probe",
        duration=1200.0,
        dt=30.0,
        sample_stride=1,
        channels=(0, 1),
        orbit_period=5700.0,
        eclipse_fraction=0.35,
        sun_constant=None,
        duty_period=600.0,
        duty_fraction=0.5,
        dissipation_high=35.0,
        dissipation_low=12.0,
        initial_state=(285.0, 275.0),
    )
    kwargs.update(overrides)
    return kwargs


def test_five_designs_are_shipped_with_distinct_names():
    assert DESIGN_NAMES == (
        "baseline-eclipse",
        "full-sun",
        "lumped-nodes",
        "coarse-sampled",
        "weak-excitation",
    )
    assert set(designs()) == set(DESIGN_NAMES)


def test_designs_returns_a_copy():
    d = designs()
    d.pop("full-sun")
    assert "full-sun" in designs()


def test_unknown_design_names_the_valid_ones():
    with pytest.raises(KeyError, match="baseline-eclipse"):
        design("no-such-design")


def test_truth_parameters_are_the_declared_values():
    assert THETA_TRUE == (800.0, 4000.0, 2.5, 1.205e-8, 55.0, 18.0)
    assert LINEAR_THETA_TRUE == (800.0, 2.5, 40.0)
    assert DEFAULT_SIGMA_K == pytest.approx(0.3)
    assert TRUTH_CONDUCTANCE_SLOPE == pytest.approx(1.5)


def test_design_counts_are_consistent():
    for name in DESIGN_NAMES:
        d = design(name)
        inputs = build_inputs(d)
        assert inputs.times.size == d.n_integration_points
        assert d.n_samples == len(range(0, d.n_integration_points, d.sample_stride))
        assert d.n_observations == d.n_samples * len(d.channels)


def test_theta_override_only_moves_the_named_parameter():
    base = design("baseline-eclipse").theta_true
    lumped = design("lumped-nodes").theta_true
    names = ThermalTwinModel.parameter_names
    for i, name in enumerate(names):
        if name == "k12":
            assert lumped[i] == pytest.approx(40.0)
        else:
            assert lumped[i] == pytest.approx(base[i])


def test_design_validation():
    with pytest.raises(ValueError, match="0 < dt < duration"):
        ExperimentDesign(**_base_kwargs(dt=0.0))
    with pytest.raises(ValueError, match="sample_stride"):
        ExperimentDesign(**_base_kwargs(sample_stride=0))
    with pytest.raises(ValueError, match="at least one channel"):
        ExperimentDesign(**_base_kwargs(channels=()))
    with pytest.raises(ValueError, match=r"subset of \(0, 1\)"):
        ExperimentDesign(**_base_kwargs(channels=(0, 7)))
    with pytest.raises(ValueError, match="must not repeat"):
        ExperimentDesign(**_base_kwargs(channels=(0, 0)))
    with pytest.raises(ValueError, match="eclipse_fraction"):
        ExperimentDesign(**_base_kwargs(eclipse_fraction=1.0))
    with pytest.raises(ValueError, match="sun_constant"):
        ExperimentDesign(**_base_kwargs(sun_constant=2.0))
    with pytest.raises(ValueError, match="duty_fraction"):
        ExperimentDesign(**_base_kwargs(duty_fraction=0.0))
    with pytest.raises(ValueError, match="sigma_k"):
        ExperimentDesign(**_base_kwargs(sigma_k=-0.1))
    with pytest.raises(ValueError, match="theta_override"):
        ExperimentDesign(**_base_kwargs(theta_override={"nope": 1.0}))


def test_build_inputs_phase_validation_and_effect():
    d = design("baseline-eclipse")
    with pytest.raises(ValueError, match="phase"):
        build_inputs(d, phase=1.0)
    a = build_inputs(d, phase=0.0)
    b = build_inputs(d, phase=0.37)
    assert np.any(a.dissipation != b.dissipation)
    assert np.any(a.sun != b.sun)


def test_full_sun_design_has_constant_illumination():
    inputs = build_inputs(design("full-sun"))
    assert np.max(inputs.sun) == pytest.approx(np.min(inputs.sun), abs=0.0)
    assert float(inputs.sun[0]) == pytest.approx(1.0)


def test_weak_excitation_design_has_constant_dissipation():
    inputs = build_inputs(design("weak-excitation"))
    assert np.ptp(inputs.dissipation) == pytest.approx(0.0, abs=0.0)
    assert np.ptp(inputs.sun) > 0.5


def test_eclipse_fraction_is_realised_in_the_profile():
    d = design("baseline-eclipse")
    inputs = build_inputs(d)
    # The 12000 s record is 2.105 orbits of 5700 s, so the truncated last
    # orbit biases the sampled dark fraction below the declared 0.35; the
    # measured value is 0.329. The tolerance covers that truncation, not a
    # sampling accident.
    dark = float(np.mean(inputs.sun < 0.5))
    assert dark == pytest.approx(d.eclipse_fraction, abs=0.03)


def test_dataset_noise_has_the_declared_scale():
    d = design("baseline-eclipse")
    data = make_dataset(d, seed=11)
    noise = data.y - data.y_clean
    assert noise.std(ddof=1) == pytest.approx(d.sigma_k, rel=0.08)
    assert data.n_observations == d.n_observations
    assert np.all(data.sigma == d.sigma_k)


def test_dataset_is_reproducible_from_its_seed():
    d = design("baseline-eclipse")
    a = make_dataset(d, seed=5)
    b = make_dataset(d, seed=5)
    c = make_dataset(d, seed=6)
    assert np.max(np.abs(a.y - b.y)) == pytest.approx(0.0, abs=0.0)
    assert np.max(np.abs(a.y - c.y)) > 0.0


def test_dataset_channel_view_and_bounds():
    d = design("baseline-eclipse")
    data = make_dataset(d, seed=3)
    first = data.channel_view(0)
    assert first.shape == (d.n_samples,)
    assert np.max(np.abs(first - data.y[: d.n_samples])) == pytest.approx(0.0, abs=0.0)
    with pytest.raises(IndexError):
        data.channel_view(5)


def test_exact_model_data_has_no_discrepancy():
    d = design("baseline-eclipse")
    model = calibration_model()
    data = make_dataset(d, seed=2, model=model)
    clean = model.observe(
        d.theta_true, data.inputs, d.initial_state, d.channels, d.sample_stride
    )
    assert np.max(np.abs(data.y_clean - clean)) == pytest.approx(0.0, abs=1e-12)


def test_truth_model_data_has_a_discrepancy_above_the_noise():
    d = design("baseline-eclipse")
    data = make_dataset(d, seed=2, model=truth_model())
    clean = calibration_model().observe(
        d.theta_true, data.inputs, d.initial_state, d.channels, d.sample_stride
    )
    rms = float(np.sqrt(np.mean((data.y_clean - clean) ** 2)))
    assert rms > d.sigma_k


def test_lumped_nodes_discrepancy_is_below_the_noise():
    # This is what makes lumped-nodes the control design for the absorption
    # measurement: there is almost nothing real for a residual model to learn.
    d = design("lumped-nodes")
    data = make_dataset(d, seed=2, model=truth_model())
    clean = calibration_model().observe(
        d.theta_true, data.inputs, d.initial_state, d.channels, d.sample_stride
    )
    rms = float(np.sqrt(np.mean((data.y_clean - clean) ** 2)))
    assert rms < 0.5 * d.sigma_k


def test_linear_helpers():
    assert linear_model().t_sink == pytest.approx(280.0)
    inputs = linear_design(31, 10.0)
    assert inputs.times.size == 31
    assert inputs.dt == pytest.approx(10.0)
    with pytest.raises(ValueError, match="n_points"):
        linear_design(1)


def test_channel_labels():
    assert design("baseline-eclipse").channel_labels == ("T1", "T2")
    assert design("lumped-nodes").channel_labels == ("T1",)
