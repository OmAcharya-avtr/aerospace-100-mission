"""Baseline 2: refusal, the identifiable combinations, and the equivalence family."""

from __future__ import annotations

import numpy as np
import pytest

from twindiverge.attribution import CONFIDENCE_LEVEL, attribute
from twindiverge.identifiability import analyse
from twindiverge.plant import NOMINAL, PARAMETER_NAMES, perturb, simulate
from twindiverge.telemetry import design_matrix, measure, residual_vector


def _case(telemetry, offset, seed=11, twin=None):
    observed, _ = measure(telemetry, perturb(NOMINAL, offset), seed=seed)
    report = analyse(telemetry, NOMINAL)
    reference = twin if twin is not None else design_matrix(telemetry, NOMINAL)[1]
    residual = residual_vector(telemetry, observed, reference)
    return attribute(telemetry, report, residual, twin=reference)


def test_the_identifiable_directions_recover_a_small_perturbation(short_set):
    """Equation (13) is a LINEAR estimator, so this is tested in the linear
    regime: a 0.1 per cent move in J, where the second-order term of the model
    is of order 1e-6 relative and the statistical intervals dominate it. The
    same test at a 6 per cent move fails on the two best-determined directions
    -- their intervals are narrower than the linearisation error -- and that is
    a property of the estimator, reported in README.md under Limitations rather
    than hidden by a looser tolerance here."""
    offset = np.zeros(5)
    offset[0] = 1.0e-3
    result = _case(short_set, offset)
    for direction in result.directions:
        truth = float(direction.weights @ offset)
        assert abs(direction.estimate - truth) < direction.half_width, direction.label


def test_a_refused_parameter_gets_no_number(short_set):
    offset = np.zeros(5)
    offset[2] = 0.05
    result = _case(short_set, offset)
    assert result.refused == PARAMETER_NAMES
    assert np.all(np.isnan(result.parameter_estimate))
    assert np.all(np.isnan(result.parameter_half_width))


def test_a_full_rank_set_gives_every_parameter_a_number(torque_set):
    offset = np.zeros(5)
    offset[1] = 0.04
    result = _case(torque_set, offset)
    assert result.refused == ()
    assert np.all(np.isfinite(result.parameter_estimate))
    assert np.all(np.isfinite(result.parameter_half_width))


def test_detection_fires_on_a_real_change_and_not_on_noise(short_set):
    offset = np.zeros(5)
    offset[0] = 0.05
    assert _case(short_set, offset).detected
    quiet = _case(short_set, np.zeros(5))
    assert not quiet.detected
    assert quiet.residual_chi2 < quiet.detection_threshold


def test_a_pure_scale_perturbation_is_not_detected_on_a_rate_only_set(short_set):
    """Equation (6): the noiseless telemetry is identical to the twin's."""
    result = _case(short_set, np.full(5, 0.12))
    assert not result.detected


def test_a_pure_scale_perturbation_is_detected_once_torque_is_recorded(torque_set):
    """Equation (7): the scale direction's torque sensitivity is the torque itself."""
    assert _case(torque_set, np.full(5, 0.12)).detected


def test_the_equivalence_family_produces_identical_noiseless_telemetry(short_set):
    """The sharp statement. Every member of the family is a different physical
    story and they cannot be told apart. Exact in exact arithmetic."""
    offset = np.zeros(5)
    offset[3] = 0.09
    result = _case(short_set, offset)
    family = result.equivalence_family()
    assert family.shape[0] == 2
    first = simulate(perturb(NOMINAL, family[0]), short_set.campaign).rate_rad_s
    second = simulate(perturb(NOMINAL, family[1]), short_set.campaign).rate_rad_s
    assert float(np.abs(first - second).max() / np.abs(first).max()) < 1e-14


def test_the_equivalence_family_zeroes_the_chosen_parameter_in_its_alternative(short_set):
    offset = np.zeros(5)
    offset[0] = 0.07
    result = _case(short_set, offset)
    family = result.equivalence_family("J")
    assert float(family[1][0]) == pytest.approx(0.0, rel=0.0, abs=1e-15)
    assert float(family[0][0]) != pytest.approx(0.0, rel=0.0, abs=1e-15)


def test_a_full_rank_set_has_no_equivalence_family(torque_set):
    offset = np.zeros(5)
    offset[0] = 0.07
    assert _case(torque_set, offset).equivalence_family().shape == (1, 5)


def test_the_equivalence_family_rejects_an_unknown_parameter(short_set):
    with pytest.raises(ValueError, match="unknown parameter"):
        _case(short_set, np.zeros(5)).equivalence_family("mass")


@pytest.mark.parametrize("index", range(5))
def test_the_constrained_route_recovers_the_right_parameter(index: int, full_set):
    """Equation (15) under its own assumption. The amplitude is checked too, so
    this is not only an argmin test."""
    offset = np.zeros(5)
    offset[index] = 0.08
    result = _case(full_set, offset)
    assert result.constrained_choice == PARAMETER_NAMES[index]
    # The amplitude is the LINEAR estimate of a nonlinear model at an 8 per cent
    # perturbation, so a few per cent of second-order error is expected and is
    # what the tolerance admits; the measured worst case over the five
    # parameters is 3.81 per cent.
    assert float(result.constrained_estimate[index]) == pytest.approx(
        0.08, rel=0.06, abs=0.0
    )


def test_the_constrained_weights_are_a_normalised_likelihood_ratio(full_set):
    offset = np.zeros(5)
    offset[1] = 0.05
    result = _case(full_set, offset)
    assert float(result.constrained_weight.sum()) == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert float(result.constrained_weight.max()) == result.constrained_confidence


def test_the_sensitivity_shares_lie_in_the_unit_interval(full_set):
    offset = np.zeros(5)
    offset[4] = 0.10
    result = _case(full_set, offset)
    assert np.all(result.sensitivity_share >= 0.0)
    assert np.all(result.sensitivity_share <= 1.0 + 1e-12)


def test_the_two_single_parameter_routes_can_disagree(coast_set):
    """On the coast set the torque constant is never excited, so equation (14)
    gives it a share of exactly zero and equation (15) gives it the null
    chi-square, which is the largest. Neither can choose it, and that agreement
    is asserted; the test exists to show the routes are separately computed."""
    offset = np.zeros(5)
    offset[2] = 0.06
    result = _case(coast_set, offset)
    index = PARAMETER_NAMES.index("k_t")
    assert float(result.sensitivity_share[index]) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert float(result.constrained_chi2[index]) == pytest.approx(
        result.residual_chi2, rel=1e-12, abs=0.0
    )
    assert result.sensitivity_choice != "k_t"
    assert result.constrained_choice != "k_t"


def test_the_direction_significance_flag_follows_its_z_score(full_set):
    """The flag is a thresholded boolean, so the Z SCORE is asserted and the
    flag is only required to agree with it away from the critical value."""
    from scipy import stats

    offset = np.zeros(5)
    offset[0] = 0.06
    critical = float(stats.norm.ppf(0.5 + CONFIDENCE_LEVEL / 2.0))
    for direction in _case(full_set, offset).directions:
        if abs(abs(direction.z_score) - critical) > 1e-6:
            assert direction.significant == (abs(direction.z_score) > critical)


def test_the_fit_chi_square_is_the_residual_outside_the_identifiable_subspace(full_set):
    offset = np.zeros(5)
    offset[0] = 0.06
    result = _case(full_set, offset)
    projected = sum(direction.z_score**2 for direction in result.directions)
    assert result.fit_chi2 == pytest.approx(
        result.residual_chi2 - projected, rel=1e-9, abs=1e-9
    )


def test_attribute_validates_its_arguments(short_set, full_set):
    report = analyse(short_set, NOMINAL)
    with pytest.raises(ValueError, match="identifiability report is for"):
        attribute(full_set, report, np.zeros(full_set.rows))
    with pytest.raises(ValueError, match="confidence_level"):
        attribute(short_set, report, np.zeros(short_set.rows), confidence_level=0.0)
    with pytest.raises(ValueError, match="detection_level"):
        attribute(short_set, report, np.zeros(short_set.rows), detection_level=1.0)
    with pytest.raises(ValueError, match="rows"):
        attribute(short_set, report, np.zeros(3))


def test_the_combination_answer_names_only_significant_directions(full_set):
    offset = np.zeros(5)
    offset[0] = 0.06
    result = _case(full_set, offset)
    assert len(result.combination_answer) == sum(
        1 for direction in result.directions if direction.significant
    )
