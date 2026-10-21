"""Perturbation, Jaccard and the stability curve."""

from __future__ import annotations

import numpy as np
import pytest

from downlinkpick.exact import exact_select
from downlinkpick.greedy import greedy_select
from downlinkpick.selection import SelectionProblem
from downlinkpick.stability import jaccard, perturb_values, stability_curve


def test_zero_sigma_is_the_identity() -> None:
    values = {"a": 1.0, "b": 2.5}
    rng = np.random.default_rng(0)
    assert perturb_values(values, 0.0, rng) == values


def test_perturbation_is_multiplicative_and_positive() -> None:
    rng = np.random.default_rng(1)
    values = {f"p{i}": float(i + 1) for i in range(50)}
    perturbed = perturb_values(values, 0.3, rng)
    assert set(perturbed) == set(values)
    assert all(value > 0.0 for value in perturbed.values())
    ratios = np.array([perturbed[k] / values[k] for k in values])
    # The log of the ratio is the drawn normal, so its mean is near zero.
    assert abs(float(np.mean(np.log(ratios)))) < 0.2


def test_perturbation_is_reproducible_from_a_seed() -> None:
    values = {"a": 1.0, "b": 2.0}
    first = perturb_values(values, 0.1, np.random.default_rng(7))
    second = perturb_values(values, 0.1, np.random.default_rng(7))
    assert first == second


def test_negative_sigma_rejected() -> None:
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        perturb_values({"a": 1.0}, -0.1, np.random.default_rng(0))


def test_jaccard_identities() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({"a"}), frozenset({"a"})) == 1.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1.0 / 3.0)
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_is_symmetric() -> None:
    left = frozenset({"a", "b", "c"})
    right = frozenset({"b", "c", "d", "e"})
    assert jaccard(left, right) == jaccard(right, left)


def test_curve_at_zero_sigma_is_perfectly_stable(
    instance_problem: SelectionProblem,
) -> None:
    curve = stability_curve(
        instance_problem, (0.0,), selector=exact_select, n_trials=3, seed=11
    )
    assert len(curve) == 1
    point = curve[0]
    assert point.mean_jaccard == 1.0
    assert point.min_jaccard == 1.0
    assert point.p05_jaccard == 1.0
    assert point.mean_value_retained == pytest.approx(1.0, rel=1e-12)
    assert point.mean_flips == 0.0
    assert point.n_trials == 3


def test_curve_is_reproducible(instance_problem: SelectionProblem) -> None:
    first = stability_curve(
        instance_problem, (0.1,), selector=greedy_select, n_trials=5, seed=3
    )
    second = stability_curve(
        instance_problem, (0.1,), selector=greedy_select, n_trials=5, seed=3
    )
    assert first == second


def test_curve_degrades_with_noise(instance_problem: SelectionProblem) -> None:
    curve = stability_curve(
        instance_problem, (0.0, 0.5), selector=exact_select, n_trials=8, seed=5
    )
    assert curve[1].mean_jaccard < curve[0].mean_jaccard
    assert curve[1].mean_flips > curve[0].mean_flips
    # The exact solver's unperturbed selection is the optimum, so a perturbed
    # selection cannot score above it under the unperturbed values.
    assert curve[1].mean_value_retained <= 1.0 + 1e-12


def test_curve_quantiles_are_ordered(instance_problem: SelectionProblem) -> None:
    curve = stability_curve(
        instance_problem, (0.2,), selector=exact_select, n_trials=12, seed=9
    )
    point = curve[0]
    assert point.min_jaccard <= point.p05_jaccard <= point.mean_jaccard
    assert point.p05_value_retained <= point.mean_value_retained + 1e-12


def test_curve_validation(instance_problem: SelectionProblem) -> None:
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        stability_curve(instance_problem, (0.1,), selector=exact_select, n_trials=0)
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        stability_curve(instance_problem, (-0.1,), selector=exact_select, n_trials=1)


def test_curve_order_follows_the_sigmas(instance_problem: SelectionProblem) -> None:
    sigmas = (0.2, 0.0, 0.1)
    curve = stability_curve(
        instance_problem, sigmas, selector=greedy_select, n_trials=2, seed=1
    )
    assert tuple(point.sigma for point in curve) == sigmas
