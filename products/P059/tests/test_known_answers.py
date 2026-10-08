"""Known-answer tests with the hand arithmetic shown in full.

Every expected number below was derived with pen and paper before the code was
run.  **Every case here is the MAXIMAL robust invariant set**, the set of
states from which the constraint can be held forever against any admissible
disturbance sequence.  It is not the minimal robust positively invariant set
`F_inf = sum_i A^i W` of Rakovic, Kerrigan, Kouramas and Mayne (2005); the
first test below asserts that the two differ on the same data, because
confusing them is the only real correctness question in this package and a
sibling product got it wrong first time round.
"""

from __future__ import annotations

import numpy as np
import pytest

from invariantset import (
    CONVERGED,
    EMPTY,
    Polytope,
    get_system,
    maximal_robust_invariant_set,
    verify_robust_invariance,
)


class TestScalarMaximalSetEqualsX:
    """1-D: lambda = 0.8, |w| <= 0.1, |x| <= 1.

    Hand arithmetic.  Omega_0 = [-1, 1].  The one-step set is

        Pre(Omega_0) = {x : |0.8 x| <= 1 - 0.1} = [-1.125, 1.125]

    because 0.9 / 0.8 = 1.125.  Omega_1 = Omega_0 & Pre(Omega_0) = [-1, 1] =
    Omega_0, so the recursion terminates at k = 1 with S_inf = X.

    Equivalently, in one dimension S_inf = X exactly when
    b >= w / (1 - |lambda|) = 0.1 / 0.2 = 0.5, and 1.0 >= 0.5.

    The minimal robust positively invariant set for the same data is
    [-0.5, 0.5], half-width w / (1 - |lambda|) = 0.5 (Rakovic et al. 2005).
    **0.5 is not the answer to this question.**  The maximal set is [-1, 1],
    width 2, and the test asserts both facts so the distinction is recorded in
    executable form.
    """

    def test_converges_at_k_1_to_x(self):
        s = get_system("scalar_invariant_equals_x")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        assert res.termination == CONVERGED
        assert res.iterations == 1
        assert res.polytope.support([1.0]) == pytest.approx(1.0, abs=1e-12)
        assert res.polytope.support([-1.0]) == pytest.approx(1.0, abs=1e-12)
        assert res.polytope.volume() == pytest.approx(2.0, abs=1e-12)

    def test_the_maximal_set_is_not_the_minimal_rpi_set(self):
        s = get_system("scalar_invariant_equals_x")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        minimal_rpi_half_width = 0.1 / (1.0 - 0.8)  # = 0.5, Rakovic et al. 2005
        assert minimal_rpi_half_width == pytest.approx(0.5, abs=1e-15)
        assert res.polytope.volume() == pytest.approx(2.0, abs=1e-12)
        assert res.polytope.volume() != pytest.approx(2.0 * minimal_rpi_half_width, abs=1e-6)

    def test_slack_margin_is_exactly_minus_w(self):
        # Facet x <= 1: h_S(0.8) + h_W(1) - 1 = 0.8 + 0.1 - 1 = -0.1.
        s = get_system("scalar_invariant_equals_x")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        assert invariant
        assert margin == pytest.approx(-0.1, abs=1e-12)


class TestScalarThresholdCase:
    """1-D boundary: lambda = 0.5, |w| <= 0.25, |x| <= 0.5.

    Hand arithmetic.  w / (1 - |lambda|) = 0.25 / 0.5 = 0.5 = b exactly, so X
    is robustly invariant with zero margin:

        Pre bound = (0.5 - 0.25) / 0.5 = 0.5 = b

    Expect convergence at k = 1 with margin exactly 0.  A maximal set has by
    definition spent all its margin, so zero is the correct answer here, not a
    sign of trouble.
    """

    def test_zero_margin_at_the_threshold(self):
        A = np.array([[0.5]])
        X = Polytope.from_box([0.0], [0.5])
        W = Polytope.from_box([0.0], [0.25])
        res = maximal_robust_invariant_set(A, X, W)
        assert res.termination == CONVERGED
        assert res.iterations == 1
        assert res.polytope.volume() == pytest.approx(1.0, abs=1e-12)
        invariant, margin = verify_robust_invariance(A, res.polytope, W)
        assert invariant
        assert margin == pytest.approx(0.0, abs=1e-15)

    def test_just_below_the_threshold_is_empty(self):
        # b = 0.49 < 0.5 = w / (1 - lambda), so S_inf must be empty.
        A = np.array([[0.5]])
        X = Polytope.from_box([0.0], [0.49])
        W = Polytope.from_box([0.0], [0.25])
        res = maximal_robust_invariant_set(A, X, W, max_iter=200)
        assert res.termination == EMPTY
        assert res.polytope is None


class TestScalarEmpty:
    """1-D: lambda = 0.9, |w| <= 0.2, |x| <= 1.  S_inf is EMPTY.

    Hand arithmetic.  With Omega_k = [-r_k, r_k], the recursion is

        r_{k+1} = min(r_k, (r_k - w) / lambda) = (r_k - 0.2) / 0.9

    whenever r_k is below the fixed point r* = w / (1 - lambda) = 0.2 / 0.1 =
    2.0.  Writing s_k = r_k - r* gives s_{k+1} = s_k / lambda, so with
    s_0 = 1 - 2 = -1,

        r_k = 2 - (10/9)^k

    Evaluating (10/9)^k:
        k=1  1.111111   r_1 = 0.888889
        k=2  1.234568   r_2 = 0.765432
        k=3  1.371742   r_3 = 0.628258
        k=4  1.524158   r_4 = 0.475842
        k=5  1.693508   r_5 = 0.306492
        k=6  1.881676   r_6 = 0.118324
        k=7  2.090751   r_7 = -0.090751   -> infeasible

    So the recursion detects emptiness at k = 7, and the last non-empty
    iterate has half-width 0.118324, i.e. length 0.236647.
    """

    def test_empty_detected_at_iteration_7(self):
        s = get_system("scalar_empty")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50)
        assert res.termination == EMPTY
        assert res.is_empty
        assert res.iterations == 7
        assert res.polytope is None

    def test_closed_form_half_widths_per_iteration(self):
        s = get_system("scalar_empty")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50, track_geometry=True)
        # history[k-1] holds Omega_k; the last record is the empty one.
        for rec in res.history[:-1]:
            expected_half_width = 2.0 - (10.0 / 9.0) ** rec.k
            assert rec.volume == pytest.approx(2.0 * expected_half_width, rel=1e-9)
        assert res.history[-2].volume == pytest.approx(0.236647153682, rel=1e-9)

    def test_report_says_empty(self):
        s = get_system("scalar_empty")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50)
        assert "EMPTY" in res.report()


class TestNilpotent2D:
    """2-D: A = [[0, 1], [0, 0]], X = box 1, W = box 0.1.

    A maps (x1, x2) to (x2, 0).  Hand trace of recursion (8):

    Omega_0 = {|x1| <= 1, |x2| <= 1}, rows (e1, 1), (-e1, 1), (e2, 1), (-e2, 1).

    Pre(Omega_0): row c, right-hand side e, contributes (c^T A) x <= e - h_W(c).
        c = e1 : c^T A = (0, 1), h_W(e1) = 0.1  ->  x2 <= 0.9
        c = -e1: c^T A = (0, -1), h_W(-e1) = 0.1 -> -x2 <= 0.9
        c = e2 : c^T A = (0, 0), h_W(e2) = 0.1  ->  0 <= 0.9, trivially true
        c = -e2: c^T A = (0, 0)                  ->  0 <= 0.9, trivially true

    So Omega_1 = {|x1| <= 1, |x2| <= 0.9}, four facets, area 2 * 1.8 = 3.6.

    Omega_2: rows of Omega_1 are (e1, 1), (-e1, 1), (e2, 0.9), (-e2, 0.9).
        c = e1 : x2 <= 1 - 0.1 = 0.9    (already implied)
        c = e2 : 0 <= 0.9 - 0.1 = 0.8   (trivially true)
    so Omega_2 = Omega_1 and the recursion terminates at k = 2.

    This fixture also exercises the degenerate-row path, because A is singular
    and two rows of Pre have a zero normal.
    """

    def test_known_set_and_iteration_count(self):
        s = get_system("nilpotent_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        assert res.termination == CONVERGED
        assert res.iterations == 2
        S = res.polytope
        assert S.n_halfspaces == 4
        assert S.support([1.0, 0.0]) == pytest.approx(1.0, abs=1e-12)
        assert S.support([0.0, 1.0]) == pytest.approx(0.9, abs=1e-12)
        assert S.support([0.0, -1.0]) == pytest.approx(0.9, abs=1e-12)
        assert S.volume() == pytest.approx(3.6, abs=1e-9)
        assert S.vertices().shape == (4, 2)

    def test_margin_is_zero_because_the_set_is_maximal(self):
        # Facet x2 <= 0.9: h_S(A^T e2) + h_W(e2) - 0.9 = 0 + 0.1 - 0.9 = -0.8.
        # Facet x1 <= 1  : h_S(A^T e1) + h_W(e1) - 1 = 0.9 + 0.1 - 1 = 0.
        s = get_system("nilpotent_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        assert invariant
        assert margin == pytest.approx(0.0, abs=1e-15)


class TestDecoupled2D:
    """2-D: A = 0.5 I, X = box 1, W = box 0.1.

    A diagonal A makes recursion (8) separate into two independent 1-D
    recursions, each with threshold w / (1 - lambda) = 0.1 / 0.5 = 0.2.  Since
    0.2 <= 1, both axes satisfy the threshold and S_inf = X = [-1, 1]^2:
    four facets, four vertices, area 4.0, reached at k = 1.

    Margin on facet x1 <= 1: h_X(0.5 e1) + h_W(e1) - 1 = 0.5 + 0.1 - 1 = -0.4.
    """

    def test_known_set(self):
        s = get_system("decoupled_2d")
        res = maximal_robust_invariant_set(s.A, s.X, s.W)
        assert res.termination == CONVERGED
        assert res.iterations == 1
        assert res.polytope.n_halfspaces == 4
        assert res.polytope.volume() == pytest.approx(4.0, abs=1e-12)
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        assert invariant
        assert margin == pytest.approx(-0.4, abs=1e-12)


class TestAttitudeLoopRegression:
    """The illustrative attitude loop, as a regression on measured numbers.

    This one is not hand-computable; the numbers come from
    `validation/validate_known_answers.py` run in this container and are
    pinned so that a change in the algorithm shows up as a test failure.
    The closed-loop gain is hand-derived: K = [8.0, 3.8] places the poles of
    A_plant - B K at 0.9 +/- 0.1j, characteristic polynomial
    z^2 - 1.8 z + 0.82, which the first test checks.
    """

    def test_closed_loop_poles_are_where_the_gain_puts_them(self):
        s = get_system("attitude_loop")
        assert np.trace(s.A) == pytest.approx(1.8, abs=1e-12)
        assert np.linalg.det(s.A) == pytest.approx(0.82, abs=1e-12)
        eig = np.sort_complex(np.linalg.eigvals(s.A))
        assert eig[0] == pytest.approx(0.9 - 0.1j, abs=1e-9)
        assert eig[1] == pytest.approx(0.9 + 0.1j, abs=1e-9)

    def test_disturbance_box_from_the_declared_acceleration(self):
        # a_w = 0.12 rad/s^2 for one sample of dt = 0.05 s enters the state as
        # [dt^2/2 * a_w, dt * a_w] = [1.5e-4 rad, 6.0e-3 rad/s].
        s = get_system("attitude_loop")
        assert s.W.support([1.0, 0.0]) == pytest.approx(1.5e-4, rel=1e-12)
        assert s.W.support([0.0, 1.0]) == pytest.approx(6.0e-3, rel=1e-12)

    def test_measured_outcome(self):
        s = get_system("attitude_loop")
        res = maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=50, track_geometry=True)
        assert res.termination == CONVERGED
        assert res.iterations == 5
        assert res.polytope.n_halfspaces == 16
        assert res.polytope.vertices().shape[0] == 16
        assert res.polytope.volume() == pytest.approx(0.538347284806, rel=1e-9)
        invariant, margin = verify_robust_invariance(s.A, res.polytope, s.W)
        assert invariant
        assert abs(margin) <= 1e-12

    def test_s_inf_is_a_proper_subset_of_x(self):
        # X itself is not robustly invariant here, which is why the recursion
        # has anything to do.
        s = get_system("attitude_loop")
        invariant, margin = verify_robust_invariance(s.A, s.X.remove_redundant(), s.W)
        assert not invariant
        assert margin > 0.0
