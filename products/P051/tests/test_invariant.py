"""The robust invariant set: known answer, exact verification, failure modes."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import (
    BaselineController,
    EmptyInvariantSet,
    Plant,
    RecursionDidNotConverge,
    box,
    closed_loop_matrix,
    robust_invariant_set,
    verify_robust_invariance,
)


def _one_d_plant(wmax: float, xmax: float, umax: float):
    return Plant(
        A=np.array([[1.2]]),
        B=np.array([[1.0]]),
        disturbance=box([wmax]),
        state_constraints=box([xmax]),
        input_constraints=box([umax]),
    )


def test_one_dimensional_invariant_set_known_answer():
    # Hand calculation, the whole thing. Plant x' = 1.2 x + u + w with the
    # baseline u = -0.7 x, so the closed loop is x' = 0.5 x + w, lambda = 0.5.
    #
    # Omega_0 = {|x| <= b0} with b0 = min(xmax, umax / 0.7), because the input
    # row |-0.7 x| <= umax is |x| <= umax / 0.7.
    # One step of the recursion:
    #   Pre(Omega_i) = {x : 0.5 x + w in Omega_i for all |w| <= wmax}
    #                = {x : |0.5 x| <= b_i - wmax} = {|x| <= 2 (b_i - wmax)},
    # so b_{i+1} = min(b_i, (b_i - wmax) / 0.5).
    # The recursion is a fixed point at b_i exactly when
    #   (b_i - wmax) / 0.5 >= b_i  <=>  b_i (1 - 0.5) >= wmax
    #                              <=>  b_i >= wmax / (1 - 0.5) = 2 wmax,
    # and otherwise b decreases without bound and the set empties. So in one
    # dimension the answer is exactly
    #   S = Omega_0 if b0 >= 2 wmax, and empty otherwise,
    # which is the standard maximal-robust-invariant-set result specialised to a
    # scalar contraction. NOTE this is the MAXIMAL set, not the minimal robust
    # invariant set wmax / (1 - lambda) of Rakovic et al. (2005); the two are
    # different objects and the switching condition needs the maximal one.
    #
    # Case 1: wmax = 0.05, xmax = 1.0, umax = 5.0.
    #   b0 = min(1.0, 5.0 / 0.7 = 7.142857) = 1.0;  2 wmax = 0.1 <= 1.0,
    #   so S = {|x| <= 1.0} and the recursion converges at iteration 1.
    baseline = BaselineController(gain=np.array([[0.7]]), input_set=box([5.0]))
    plant = _one_d_plant(0.05, 1.0, 5.0)
    assert closed_loop_matrix(plant, baseline.gain) == pytest.approx(np.array([[0.5]]))
    result = robust_invariant_set(plant, baseline)
    assert result.converged
    assert result.iterations == 1
    assert result.polytope.support(np.array([1.0])) == pytest.approx(1.0, abs=1e-9)
    assert result.polytope.support(np.array([-1.0])) == pytest.approx(1.0, abs=1e-9)


def test_one_dimensional_set_truncated_by_the_input_constraint_known_answer():
    # Case 2 of the hand calculation above: wmax = 0.05, xmax = 1.0, umax = 0.1.
    #   b0 = min(1.0, 0.1 / 0.7) = 0.14285714285714288;  2 wmax = 0.1 <= b0,
    # so S = {|x| <= 0.1 / 0.7} exactly, again converging at iteration 1.
    plant = _one_d_plant(0.05, 1.0, 0.1)
    baseline = BaselineController(gain=np.array([[0.7]]), input_set=box([0.1]))
    result = robust_invariant_set(plant, baseline)
    assert result.converged
    assert result.polytope.support(np.array([1.0])) == pytest.approx(0.1 / 0.7, abs=1e-9)


def test_one_dimensional_set_empties_when_b0_is_below_twice_wmax():
    # Case 3: wmax = 0.05, xmax = 0.08, umax = 5.0 gives b0 = 0.08 < 2 wmax = 0.1,
    # so the recursion must empty the set rather than return a non-certificate.
    plant = _one_d_plant(0.05, 0.08, 5.0)
    baseline = BaselineController(gain=np.array([[0.7]]), input_set=box([5.0]))
    with pytest.raises(EmptyInvariantSet):
        robust_invariant_set(plant, baseline, max_iterations=40)


def test_one_dimensional_boundary_case_is_exactly_invariant():
    # Case 4: b0 = 2 wmax exactly. wmax = 0.05, xmax = 0.1, umax = 5.0.
    # The recursion is a fixed point with equality, so S = {|x| <= 0.1}.
    plant = _one_d_plant(0.05, 0.1, 5.0)
    baseline = BaselineController(gain=np.array([[0.7]]), input_set=box([5.0]))
    result = robust_invariant_set(plant, baseline)
    assert result.polytope.support(np.array([1.0])) == pytest.approx(0.1, abs=1e-9)
    report = verify_robust_invariance(plant, baseline, result.polytope)
    assert report["robustly_invariant"]
    assert abs(report["invariance_margin"]) < 1e-12


def test_reference_plant_set_is_verified_exactly(plant, controllers, invariant_set):
    report = verify_robust_invariance(plant, controllers[0], invariant_set)
    assert report["subset_of_X"]
    assert report["baseline_input_admissible"]
    assert report["robustly_invariant"]
    assert report["invariance_margin"] >= -1e-9
    assert report["X_margin"] >= -1e-9


def test_recursion_is_monotone_decreasing(invariant_result, plant, controllers):
    # Omega_{i+1} is a subset of Omega_i, so the final set is a subset of Omega_0.
    assert invariant_result.initial_set is not None
    assert invariant_result.initial_set.contains_polytope(
        invariant_result.polytope, tol=1e-9
    )
    assert invariant_result.polytope.area_2d() <= invariant_result.initial_set.area_2d()


def test_invariant_set_is_a_strict_subset_of_the_constraint_set(plant, invariant_set):
    assert plant.state_constraints.contains_polytope(invariant_set, tol=1e-9)
    assert not invariant_set.contains_polytope(plant.state_constraints, tol=1e-9)


def test_halfspace_count_grows_by_two_per_iteration(invariant_result):
    counts = np.array(invariant_result.halfspaces_per_iteration)
    assert counts[0] == 4
    assert np.all(np.diff(counts) == 2)


def test_larger_disturbance_gives_a_smaller_set(plant, controllers):
    small = robust_invariant_set(plant, controllers[0])
    bigger = robust_invariant_set(
        plant.with_disturbance(box(plant.disturbance.half_widths * 2.0)), controllers[0]
    )
    assert small.polytope.contains_polytope(bigger.polytope, tol=1e-8)
    assert bigger.polytope.area_2d() < small.polytope.area_2d()


def test_a_too_conservative_baseline_empties_the_set(plant):
    # A large LQR input weight gives a low-authority baseline that cannot hold
    # the declared rate constraint; the recursion must say so rather than return
    # a set that is not invariant. Measured here: it empties at iteration 52.
    from simplexguard import dlqr_gain

    weak = BaselineController(
        gain=dlqr_gain(plant, np.ones(2), np.array([400.0])),
        input_set=plant.input_constraints,
    )
    with pytest.raises(EmptyInvariantSet, match="emptied at iteration"):
        robust_invariant_set(plant, weak)


def test_a_disturbance_larger_than_the_constraints_empties_the_set(plant, controllers):
    huge = plant.with_disturbance(box([1.0, 1.0]))
    with pytest.raises(EmptyInvariantSet):
        robust_invariant_set(huge, controllers[0])


def test_recursion_cap_raises_rather_than_returning_a_non_certificate(plant, controllers):
    with pytest.raises(RecursionDidNotConverge, match="is NOT returned"):
        robust_invariant_set(plant, controllers[0], max_iterations=3)


def test_invariant_set_rejects_bad_arguments(plant, controllers):
    with pytest.raises(ValueError, match="max_iterations"):
        robust_invariant_set(plant, controllers[0], max_iterations=0)
    with pytest.raises(ValueError, match="tol must be finite"):
        robust_invariant_set(plant, controllers[0], tol=-1.0)
    wrong = BaselineController(gain=np.ones((1, 1)), input_set=plant.input_constraints)
    with pytest.raises(ValueError, match="baseline gain has shape"):
        robust_invariant_set(plant, wrong)


def test_describe_reports_convergence_and_area(invariant_result):
    text = invariant_result.describe()
    assert "converged                 True" in text
    assert "baseline spectral radius" in text
    assert "area of S" in text


def test_unreduced_recursion_reaches_the_same_set():
    # With redundancy removal off, the row count of the accumulated intersection
    # doubles every iteration, so this is checked on the one-dimensional plant
    # that converges in a single iteration. Case 2 of the known-answer
    # arithmetic above: S = {|x| <= 0.1 / 0.7}.
    plant = _one_d_plant(0.05, 1.0, 0.1)
    baseline = BaselineController(gain=np.array([[0.7]]), input_set=box([0.1]))
    reduced = robust_invariant_set(plant, baseline, reduce_every_iteration=True)
    unreduced = robust_invariant_set(plant, baseline, reduce_every_iteration=False)
    assert unreduced.polytope.n_halfspaces >= reduced.polytope.n_halfspaces
    assert unreduced.polytope.contains_polytope(reduced.polytope, tol=1e-9)
    assert reduced.polytope.contains_polytope(unreduced.polytope, tol=1e-9)


def test_verify_detects_a_set_that_is_not_invariant(plant, controllers):
    # The declared constraint set X is itself NOT robustly invariant for the
    # baseline: the recursion had to shrink it 24 times to reach S. The
    # verifier must say so, independently of the recursion that found S.
    report = verify_robust_invariance(plant, controllers[0], plant.state_constraints)
    assert report["subset_of_X"]
    assert not report["robustly_invariant"]
    assert report["invariance_margin"] < 0.0


def test_scaling_an_invariant_set_keeps_invariance_but_leaves_the_constraints(
    plant, controllers, invariant_set
):
    # The three properties are independent, and this is the case that shows it.
    # For a linear closed loop, t S with t > 1 is still robustly invariant:
    #   h_{tS}(A_b' c) + h_W(c) = t h_S(A_b' c) + h_W(c)
    #                          <= t (d - h_W(c)) + h_W(c)
    #                           = t d - (t - 1) h_W(c)  <=  t d,
    # so inflating cannot break property (iii). It does break property (i):
    # the inflated set is no longer inside X, and the verifier reports exactly
    # that split rather than a single pass/fail.
    inflated = invariant_set.scaled(1.2)
    report = verify_robust_invariance(plant, controllers[0], inflated)
    assert report["robustly_invariant"]
    assert not report["subset_of_X"]
    assert report["X_margin"] < 0.0
