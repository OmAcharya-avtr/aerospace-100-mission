"""Baseline 2: the fitted exponential models and the matrix projection.

The important test here is the one on identifiability: the pair ``(rho_0, L)``
is weakly determined and the figure of merit is the FUNCTION the fit implies at
the separations that exist. The tests check both and report the difference,
rather than asserting a tight tolerance on either parameter.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.correlation import (
    EIGENVALUE_FLOOR,
    LENGTH_BOUNDS,
    NUGGET_BOUNDS,
    ExponentialModel,
    fit_exponential,
    fit_per_regime,
    matrix_from_pairs,
)


def test_a_noiseless_exponential_is_recovered_exactly():
    """The fit of equation (23) on data it can represent perfectly.

    Both parameters are recovered to the optimiser's tolerance, which is what
    least squares does on a model with zero residual. This is the known-answer
    case: ``rho_0 = 0.87``, ``L = 450`` km, hand-set.
    """
    separation = np.linspace(50.0, 3000.0, 60)
    target = 0.87 * np.exp(-separation / 450.0)
    model = fit_exponential(separation, target)
    assert model.nugget == pytest.approx(0.87, rel=1e-6, abs=0.0)
    assert model.length_km == pytest.approx(450.0, rel=1e-6, abs=0.0)
    assert model.residual_rms < 1e-9
    assert model.at_bound == ()


def test_the_fit_of_a_constant_target_pushes_the_length_to_its_bound():
    """A field with no decay is not an exponential and the flag says so."""
    separation = np.linspace(50.0, 3000.0, 60)
    model = fit_exponential(separation, np.full(60, 0.5))
    assert "length_km" in model.at_bound
    assert model.length_km == pytest.approx(
        LENGTH_BOUNDS[1], rel=1e-6, abs=0.0
    )


def test_the_fit_of_an_uncorrelated_target_gives_a_function_near_zero():
    """An uncorrelated field is fitted by a function that is zero where sampled.

    The parameters do NOT go to a bound here, and the test says so rather than
    asserting they do: the optimiser reaches ``rho_0 = 0.004`` with a length of
    11 km, which is a degenerate interior solution whose implied correlation is
    below 1e-5 at every sampled separation. The pair ``(rho_0, L)`` is not
    separately identified -- equation (25) -- and this is that fact in its most
    extreme form. What CAN be asserted is the function, which is what this
    product quotes.
    """
    separation = np.linspace(50.0, 3000.0, 60)
    model = fit_exponential(separation, np.zeros(60))
    assert float(np.max(np.abs(model.predict(separation)))) < 1e-4
    assert model.residual_rms < 1e-4
    assert NUGGET_BOUNDS[0] <= model.nugget <= NUGGET_BOUNDS[1]


def test_the_fitted_function_is_monotone_decreasing_in_separation():
    """Equation (23) is an exponential decay, by construction."""
    model = ExponentialModel(0.9, 400.0, 0.0, 10, None, ())
    values = model.predict(np.linspace(0.0, 5000.0, 200))
    assert np.all(np.diff(values) < 0.0)


def test_the_fitted_function_at_zero_separation_is_the_nugget():
    """``rho(0) = rho_0 exp(0) = rho_0``, exactly up to ``exp(0) == 1.0``."""
    model = ExponentialModel(0.9, 400.0, 0.0, 10, None, ())
    assert float(model.predict(0.0)) == pytest.approx(0.9, rel=1e-15, abs=0.0)


def test_the_fitted_function_at_the_e_folding_length_falls_by_one_e():
    """``rho(L) = rho_0 / e``, a hand-checkable identity."""
    model = ExponentialModel(0.9, 400.0, 0.0, 10, None, ())
    assert float(model.predict(model.e_folding_km())) == pytest.approx(
        0.9 / np.e, rel=1e-14, abs=0.0
    )


def test_the_fitted_function_rejects_a_negative_separation():
    """Input validation."""
    model = ExponentialModel(0.9, 400.0, 0.0, 10, None, ())
    with pytest.raises(ValueError, match="non-negative"):
        model.predict(-1.0)


def test_equation_25_the_parameters_are_weakly_identified_and_the_function_is_not():
    """The identifiability claim of equation (25), measured.

    A noiseless exponential is perturbed by noise of a realistic size and the
    fit is repeated. The two PARAMETERS move by a large relative amount while
    the FUNCTION they imply at the sampled separations barely moves, which is
    why this product quotes the function error as its primary figure. Both
    numbers are measured here.
    """
    rng = np.random.default_rng(2525)
    separation = np.concatenate(
        [np.linspace(90.0, 300.0, 20), np.linspace(1000.0, 7000.0, 40)]
    )
    truth = 0.95 * np.exp(-separation / 420.0)
    parameter_spread = []
    function_spread = []
    for trial in range(12):
        noisy = truth + 0.03 * np.random.default_rng(trial).standard_normal(
            separation.size
        )
        model = fit_exponential(separation, noisy)
        parameter_spread.append(abs(model.length_km - 420.0) / 420.0)
        function_spread.append(
            float(np.max(np.abs(model.predict(separation) - truth)))
        )
    _ = rng
    assert max(parameter_spread) > 0.05
    assert max(function_spread) < 0.06
    assert max(function_spread) < max(parameter_spread)


def test_the_pooled_fit_recovers_the_isotropic_generators_length(isotropic_record):
    """Equation (23) against the ground truth of the CORRECTLY SPECIFIED record.

    On the isotropic control the generator's correlation IS an exponential in
    great-circle separation, so the fitted length must land near the
    generator's. The regimes differ, so the pooled fit must land between the
    extremes of the four declared lengths rather than on any one of them, and
    that is what is asserted.
    """
    from sitediverse.climatology import estimate_climatology
    from sitediverse.features import build_pair_table
    from sitediverse.field import REGIME_COUNT
    from sitediverse.geometry import upper_triangle_pairs
    from sitediverse.splits import make_split

    climatology = estimate_climatology(isotropic_record.cloud)
    split = make_split(isotropic_record.epochs, isotropic_record.site_count)
    table = build_pair_table(
        isotropic_record.cloud,
        isotropic_record.regime,
        climatology,
        split,
        REGIME_COUNT,
        upper_triangle_pairs(isotropic_record.site_count),
        isotropic_record.sites,
    )
    train = table.mask(0)
    model = fit_exponential(table.separation_km()[train], table.target[train])
    declared = np.array(isotropic_record.spec.along_km)
    assert 0.5 * float(declared.min()) < model.length_km < 2.0 * float(declared.max())


def test_the_per_regime_fit_returns_one_model_per_regime():
    """Equation (24)'s structure."""
    rng = np.random.default_rng(24)
    separation = np.tile(np.linspace(90.0, 4000.0, 40), 4)
    regime = np.repeat(np.arange(4), 40)
    lengths = np.array([300.0, 500.0, 700.0, 900.0])[regime]
    target = 0.9 * np.exp(-separation / lengths) + 0.002 * rng.standard_normal(160)
    fitted = fit_per_regime(separation, target, regime, 4)
    assert len(fitted.models) == 4
    for index, model in enumerate(fitted.models):
        assert model.regime == index
        assert model.sample_count == 40


def test_the_per_regime_fit_recovers_four_different_declared_lengths():
    """A known-answer case with four hand-set lengths and small noise."""
    rng = np.random.default_rng(242)
    separation = np.tile(np.linspace(90.0, 4000.0, 60), 4)
    regime = np.repeat(np.arange(4), 60)
    declared = np.array([300.0, 500.0, 700.0, 900.0])
    target = 0.9 * np.exp(-separation / declared[regime])
    target = target + 0.001 * rng.standard_normal(240)
    fitted = fit_per_regime(separation, target, regime, 4)
    for index, model in enumerate(fitted.models):
        assert model.length_km == pytest.approx(
            declared[index], rel=0.08, abs=0.0
        )


def test_the_per_regime_prediction_uses_each_rows_own_regime():
    """Which is the only thing equation (24) adds over equation (23)."""
    separation = np.tile(np.linspace(90.0, 4000.0, 30), 4)
    regime = np.repeat(np.arange(4), 30)
    declared = np.array([300.0, 500.0, 700.0, 900.0])
    target = 0.9 * np.exp(-separation / declared[regime])
    fitted = fit_per_regime(separation, target, regime, 4)
    predicted = fitted.predict(separation, regime)
    assert float(np.max(np.abs(predicted - target))) < 1e-6
    wrong = fitted.predict(separation, np.zeros_like(regime))
    assert float(np.max(np.abs(wrong - target))) > 0.05


def test_the_per_regime_residual_is_a_weighted_root_mean_square():
    """An accessor identity, checked against its own definition."""
    separation = np.tile(np.linspace(90.0, 4000.0, 30), 4)
    regime = np.repeat(np.arange(4), 30)
    rng = np.random.default_rng(25)
    target = 0.9 * np.exp(-separation / 500.0) + 0.01 * rng.standard_normal(120)
    fitted = fit_per_regime(separation, target, regime, 4)
    weights = np.array([m.sample_count for m in fitted.models], dtype=float)
    squares = np.array([m.residual_rms**2 for m in fitted.models])
    expected = float(np.sqrt(np.sum(weights * squares) / np.sum(weights)))
    assert fitted.residual_rms == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_the_per_regime_prediction_rejects_a_regime_out_of_range():
    """Input validation with the range in the message."""
    separation = np.tile(np.linspace(90.0, 4000.0, 10), 4)
    regime = np.repeat(np.arange(4), 10)
    target = 0.9 * np.exp(-separation / 500.0)
    fitted = fit_per_regime(separation, target, regime, 4)
    with pytest.raises(ValueError, match="regime labels must lie in"):
        fitted.predict(np.array([100.0]), np.array([9]))


def test_the_per_regime_fit_rejects_a_regime_with_too_few_points():
    """Two parameters need at least two points, and the message says which regime."""
    separation = np.linspace(90.0, 4000.0, 10)
    regime = np.zeros(10, dtype=int)
    target = 0.9 * np.exp(-separation / 500.0)
    with pytest.raises(ValueError, match="regime 1 has 0 training points"):
        fit_per_regime(separation, target, regime, 4)


def test_fit_exponential_rejects_mismatched_inputs():
    """Input validation."""
    with pytest.raises(ValueError, match="same shape"):
        fit_exponential(np.zeros(5), np.zeros(6))


def test_fit_exponential_rejects_a_single_point():
    """Two parameters need two points."""
    with pytest.raises(ValueError, match="at least 2 points"):
        fit_exponential(np.array([100.0]), np.array([0.5]))


def test_fit_exponential_rejects_a_negative_separation():
    """Input validation."""
    with pytest.raises(ValueError, match="non-negative"):
        fit_exponential(np.array([-1.0, 100.0]), np.array([0.5, 0.4]))


def test_fit_exponential_rejects_a_two_dimensional_input():
    """Input validation."""
    with pytest.raises(ValueError, match="one-dimensional"):
        fit_exponential(np.zeros((2, 2)), np.zeros((2, 2)))


def test_a_positive_definite_matrix_is_returned_unprojected():
    """Equation (26) must not move a matrix that needs no projection."""
    matrix = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.3], [0.2, 0.3, 1.0]])
    result = matrix_from_pairs(matrix)
    assert not result.projected
    assert result.frobenius_shift == 0.0
    assert result.least_eigenvalue_before > EIGENVALUE_FLOOR
    assert float(np.max(np.abs(result.matrix - matrix))) < 1e-15


def test_an_indefinite_matrix_is_projected_and_the_projection_is_reported():
    """Equation (26) on a matrix no Gaussian field has.

    The reported least eigenvalue is NEGATIVE, which is the statement about the
    pair model that the flag exists to carry, and the projected matrix is
    positive definite and has a unit diagonal.
    """
    matrix = np.array([[1.0, 0.95, -0.95], [0.95, 1.0, 0.95], [-0.95, 0.95, 1.0]])
    result = matrix_from_pairs(matrix)
    assert result.projected
    assert result.least_eigenvalue_before < 0.0
    assert result.frobenius_shift > 0.0
    assert float(np.min(np.linalg.eigvalsh(result.matrix))) > 0.0
    assert float(np.max(np.abs(np.diag(result.matrix) - 1.0))) < 1e-14


def test_the_projection_is_a_valid_correlation_matrix_for_a_learned_prediction():
    """The case equation (26) exists for: a tree ensemble has no PSD guarantee."""
    rng = np.random.default_rng(26)
    raw = rng.uniform(-0.9, 0.95, (8, 8))
    raw = 0.5 * (raw + raw.T)
    result = matrix_from_pairs(raw)
    assert float(np.min(np.linalg.eigvalsh(result.matrix))) > 0.0
    assert float(np.max(np.abs(result.matrix))) <= 1.0 + 1e-12


def test_the_projection_symmetrises_an_asymmetric_input():
    """A pair model evaluated twice can give two slightly different entries."""
    matrix = np.array([[1.0, 0.4], [0.5, 1.0]])
    result = matrix_from_pairs(matrix)
    assert result.matrix[0, 1] == pytest.approx(
        result.matrix[1, 0], rel=1e-15, abs=0.0
    )
    assert result.matrix[0, 1] == pytest.approx(0.45, rel=1e-13, abs=0.0)


def test_matrix_from_pairs_rejects_a_non_square_input():
    """Input validation."""
    with pytest.raises(ValueError, match="must be square"):
        matrix_from_pairs(np.zeros((2, 3)))


def test_matrix_from_pairs_rejects_a_non_positive_floor():
    """Input validation."""
    with pytest.raises(ValueError, match="floor must be strictly positive"):
        matrix_from_pairs(np.eye(2), floor=0.0)


@given(
    st.floats(0.05, 1.0, allow_nan=False, allow_infinity=False),
    st.floats(50.0, 5000.0, allow_nan=False, allow_infinity=False),
)
def test_any_declared_exponential_is_recovered_from_its_own_samples(nugget, length):
    """A property over the whole admissible parameter range."""
    separation = np.linspace(10.0, 4.0 * length, 80)
    target = nugget * np.exp(-separation / length)
    model = fit_exponential(separation, target)
    assert float(
        np.max(np.abs(model.predict(separation) - target))
    ) == pytest.approx(0.0, rel=0.0, abs=1e-6)
