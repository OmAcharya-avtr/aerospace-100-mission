"""Every strategy: contract, reproducibility, budget, early stop, validation."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.instances import instance
from falsifyloop.search import (
    BASELINE,
    STRATEGIES,
    SearchResult,
    _reflect,
    analytic_random_curve,
    cross_entropy,
    latin_hypercube,
    simulated_annealing,
    strategy,
    surrogate_guided,
    uniform_random,
)

EASY = instance("overshoot-loose")
HARD = instance("rate-envelope")
ALL_NAMES = tuple(STRATEGIES)


def test_the_baseline_is_uniform_random_and_is_registered_first() -> None:
    assert BASELINE == "uniform-random"
    assert ALL_NAMES[0] == BASELINE


def test_registry_has_the_five_required_strategies() -> None:
    assert set(ALL_NAMES) == {
        "uniform-random",
        "latin-hypercube",
        "simulated-annealing",
        "cross-entropy",
        "surrogate-guided",
    }


@pytest.mark.parametrize("name", ALL_NAMES)
def test_strategy_respects_its_budget(name) -> None:
    result = STRATEGIES[name](HARD, 25, 3)
    assert isinstance(result, SearchResult)
    assert result.simulations <= 25
    assert result.budget == 25
    assert result.history.shape == (result.simulations,)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_strategy_is_reproducible_from_its_seed(name) -> None:
    first = STRATEGIES[name](HARD, 20, 5)
    second = STRATEGIES[name](HARD, 20, 5)
    np.testing.assert_array_equal(first.history, second.history)
    assert first.first_violation == second.first_violation
    np.testing.assert_array_equal(first.best_vector, second.best_vector)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_different_seeds_give_different_runs(name) -> None:
    first = STRATEGIES[name](HARD, 20, 1)
    second = STRATEGIES[name](HARD, 20, 2)
    assert not np.array_equal(first.history, second.history)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_strategy_stops_at_the_first_violation(name) -> None:
    result = STRATEGIES[name](EASY, 60, 4)
    if result.found:
        assert result.first_violation == result.simulations
        assert result.history[-1] < 0.0
        assert np.all(result.history[:-1] >= 0.0)
    else:
        assert result.simulations == result.budget
        assert np.all(result.history >= 0.0)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_strategy_names_itself_and_its_instance(name) -> None:
    result = STRATEGIES[name](EASY, 10, 0)
    assert result.instance_id == EASY.identifier
    assert result.strategy == name
    assert result.seed == 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_best_vector_lies_inside_the_declared_box(name) -> None:
    result = STRATEGIES[name](HARD, 25, 9)
    assert np.all(result.best_vector >= HARD.box[:, 0] - 1e-12)
    assert np.all(result.best_vector <= HARD.box[:, 1] + 1e-12)


@pytest.mark.parametrize("name", ALL_NAMES)
def test_best_robustness_is_the_minimum_of_the_history(name) -> None:
    result = STRATEGIES[name](HARD, 25, 9)
    assert result.best_robustness == pytest.approx(float(result.history.min()))
    assert result.found == (result.first_violation is not None)


def test_uniform_random_on_an_easy_instance_finds_a_violation() -> None:
    # Design target p = 0.31, so 60 draws miss with probability 0.69**60 < 1e-9.
    assert uniform_random(EASY, 60, 0).found


def test_analytic_random_curve_known_answer() -> None:
    # 1 - (1 - 0.5)**n for n = 1, 2, 3 is 0.5, 0.75, 0.875.
    curve = analytic_random_curve(0.5, 3)
    np.testing.assert_allclose(curve, [0.5, 0.75, 0.875])
    assert curve.shape == (3,)


def test_analytic_random_curve_is_monotone_and_bounded() -> None:
    curve = analytic_random_curve(0.01, 500)
    assert np.all(np.diff(curve) > 0.0)
    assert curve[0] == pytest.approx(0.01)
    assert curve[-1] < 1.0


@pytest.mark.parametrize(("p", "budget"), [(0.0, 10), (1.0, 10), (0.5, 0)])
def test_analytic_random_curve_validation(p, budget) -> None:
    with pytest.raises(ValueError):
        analytic_random_curve(p, budget)


def test_reflect_known_answers() -> None:
    lo = np.array([0.0, 0.0])
    hi = np.array([1.0, 1.0])
    # 1.25 reflects off the upper face to 0.75; -0.25 reflects to 0.25.
    np.testing.assert_allclose(_reflect(np.array([1.25, -0.25]), lo, hi), [0.75, 0.25])
    # A point already inside is unchanged.
    np.testing.assert_allclose(_reflect(np.array([0.3, 0.7]), lo, hi), [0.3, 0.7])
    # Far outside still lands inside.
    folded = _reflect(np.array([7.3, -11.4]), lo, hi)
    assert np.all(folded >= 0.0) and np.all(folded <= 1.0)


def test_latin_hypercube_evaluates_its_design_in_order() -> None:
    from scipy.stats import qmc

    result = latin_hypercube(HARD, 12, 3)
    sampler = qmc.LatinHypercube(d=HARD.dimension, seed=3)
    design = qmc.scale(sampler.random(n=12), HARD.box[:, 0], HARD.box[:, 1])
    expected = [HARD.evaluate(row) for row in design[: result.simulations]]
    np.testing.assert_allclose(result.history, expected)


def test_simulated_annealing_validation() -> None:
    with pytest.raises(ValueError, match="step_fraction"):
        simulated_annealing(EASY, 10, 0, step_fraction=0.0)
    with pytest.raises(ValueError, match="initial_temperature"):
        simulated_annealing(EASY, 10, 0, initial_temperature=-1.0)
    with pytest.raises(ValueError, match="cooling"):
        simulated_annealing(EASY, 10, 0, cooling=1.5)


def test_cross_entropy_validation() -> None:
    with pytest.raises(ValueError, match="population"):
        cross_entropy(EASY, 10, 0, population=3)
    with pytest.raises(ValueError, match="elite_fraction"):
        cross_entropy(EASY, 10, 0, elite_fraction=0.0)
    with pytest.raises(ValueError, match="smoothing"):
        cross_entropy(EASY, 10, 0, smoothing=1.5)
    with pytest.raises(ValueError, match="min_sigma_fraction"):
        cross_entropy(EASY, 10, 0, min_sigma_fraction=1.0)


def test_surrogate_guided_validation() -> None:
    with pytest.raises(ValueError, match="n_initial"):
        surrogate_guided(EASY, 10, 0, n_initial=1)
    with pytest.raises(ValueError, match="n_candidates"):
        surrogate_guided(EASY, 10, 0, n_candidates=0)
    with pytest.raises(ValueError, match="refit_every"):
        surrogate_guided(EASY, 10, 0, refit_every=0)


def test_surrogate_guided_warm_start_is_the_latin_hypercube_prefix() -> None:
    # The first n_initial simulations of the surrogate strategy are a Latin
    # hypercube design, so on a budget that short the two strategies agree
    # exactly. This is the accounting the README reports: the learned part only
    # gets budget - n_initial simulations.
    warm = surrogate_guided(HARD, 8, 21, n_initial=8)
    lhs = latin_hypercube(HARD, 8, 21)
    np.testing.assert_allclose(warm.history, lhs.history)


def test_surrogate_guided_uses_the_surrogate_after_the_warm_start() -> None:
    warm_only = surrogate_guided(HARD, 6, 31, n_initial=6)
    guided = surrogate_guided(HARD, 20, 31, n_initial=6)
    np.testing.assert_allclose(guided.history[:6], warm_only.history)
    assert guided.simulations > 6


@pytest.mark.parametrize("name", ALL_NAMES)
def test_budget_of_one_is_legal(name) -> None:
    result = STRATEGIES[name](EASY, 1, 0)
    assert result.simulations == 1


@pytest.mark.parametrize("name", ALL_NAMES)
def test_zero_budget_is_rejected(name) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        STRATEGIES[name](EASY, 0, 0)


def test_strategy_lookup_rejects_an_unknown_name() -> None:
    with pytest.raises(KeyError, match="unknown strategy"):
        strategy("gradient-descent")
    assert strategy(BASELINE) is uniform_random
