"""Input-validation tests: every public entry point must reject bad input."""

from __future__ import annotations

import numpy as np
import pytest

from invariantset import (
    Polytope,
    get_system,
    maximal_robust_invariant_set,
    tolerance_sweep,
)
from invariantset.plotting import plot_polytope, polygon_vertices


class TestPolytopeConstruction:
    def test_row_count_mismatch(self):
        with pytest.raises(ValueError, match="rows but b has"):
            Polytope(np.eye(2), [1.0])

    def test_zero_halfspaces(self):
        with pytest.raises(ValueError, match="at least one halfspace"):
            Polytope(np.zeros((0, 2)), np.zeros(0))

    def test_non_finite_a(self):
        with pytest.raises(ValueError, match="A contains a non-finite"):
            Polytope(np.array([[np.inf, 0.0]]), [1.0])

    def test_non_finite_b(self):
        with pytest.raises(ValueError, match="b contains a non-finite"):
            Polytope(np.array([[1.0, 0.0]]), [np.nan])

    def test_three_dimensional_a(self):
        with pytest.raises(ValueError, match="A must be 2-D"):
            Polytope(np.zeros((2, 2, 2)), [1.0, 1.0])


class TestBoxConstruction:
    def test_negative_half_width(self):
        with pytest.raises(ValueError, match="non-negative"):
            Polytope.from_box([0.0], [-1.0])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="half_widths shape"):
            Polytope.from_box([0.0, 0.0], [1.0])

    def test_empty_box(self):
        with pytest.raises(ValueError, match="at least one dimension"):
            Polytope.from_box([], [])

    def test_non_finite_box(self):
        with pytest.raises(ValueError, match="must be finite"):
            Polytope.from_box([0.0], [np.inf])

    def test_inverted_bounds(self):
        with pytest.raises(ValueError, match="upper bound"):
            Polytope.from_bounds([1.0], [0.0])

    def test_bounds_shape_mismatch(self):
        with pytest.raises(ValueError, match="lower shape"):
            Polytope.from_bounds([0.0, 0.0], [1.0])

    @pytest.mark.parametrize("dim", [0, -1, 2.5])
    def test_bad_unit_box_dim(self, dim):
        with pytest.raises(ValueError, match="positive integer"):
            Polytope.unit_box(dim)

    def test_bad_unit_box_radius(self):
        with pytest.raises(ValueError, match="finite and non-negative"):
            Polytope.unit_box(2, -1.0)


class TestMethodArguments:
    def test_support_wrong_length(self):
        with pytest.raises(ValueError, match="expected 2"):
            Polytope.unit_box(2).support([1.0])

    def test_support_non_finite_direction(self):
        with pytest.raises(ValueError, match="non-finite"):
            Polytope.unit_box(2).support([np.nan, 0.0])

    def test_support_many_wrong_width(self):
        with pytest.raises(ValueError, match="columns"):
            Polytope.unit_box(2).support_many(np.ones((3, 3)))

    def test_contains_wrong_length(self):
        with pytest.raises(ValueError, match="expected 2"):
            Polytope.unit_box(2).contains([1.0, 2.0, 3.0])

    def test_contains_negative_tol(self):
        with pytest.raises(ValueError, match="non-negative"):
            Polytope.unit_box(2).contains([0.0, 0.0], tol=-1e-9)

    def test_is_empty_negative_tol(self):
        with pytest.raises(ValueError, match="non-negative"):
            Polytope.unit_box(2).is_empty(tol=-1.0)

    @pytest.mark.parametrize("tol", [-1.0, np.nan, np.inf])
    def test_remove_redundant_bad_tol(self, tol):
        with pytest.raises(ValueError, match="finite and non-negative"):
            Polytope.unit_box(2).remove_redundant(tol=tol)


class TestRecursionArguments:
    def test_non_square_a(self):
        s = get_system("decoupled_2d")
        with pytest.raises(ValueError, match="square"):
            maximal_robust_invariant_set(np.zeros((2, 3)), s.X, s.W)

    def test_a_dim_mismatch(self):
        s = get_system("decoupled_2d")
        with pytest.raises(ValueError, match="has dim"):
            maximal_robust_invariant_set(np.eye(3), s.X, s.W)

    def test_w_dim_mismatch(self):
        s = get_system("decoupled_2d")
        with pytest.raises(ValueError, match="W has dim"):
            maximal_robust_invariant_set(s.A, s.X, Polytope.unit_box(3))

    @pytest.mark.parametrize("max_iter", [0, -5, 2.5])
    def test_bad_max_iter(self, max_iter):
        s = get_system("decoupled_2d")
        with pytest.raises(ValueError, match="positive integer"):
            maximal_robust_invariant_set(s.A, s.X, s.W, max_iter=max_iter)

    @pytest.mark.parametrize("name", ["convergence_tol", "redundancy_tol"])
    def test_negative_tolerances(self, name):
        s = get_system("decoupled_2d")
        with pytest.raises(ValueError, match=name):
            maximal_robust_invariant_set(s.A, s.X, s.W, **{name: -1e-9})

    def test_empty_tolerance_list(self):
        s = get_system("decoupled_2d")
        with pytest.raises(ValueError, match="non-empty"):
            tolerance_sweep(s.A, s.X, s.W, [])


class TestSystemLookup:
    def test_unknown_system_lists_the_valid_names(self):
        with pytest.raises(KeyError) as excinfo:
            get_system("not_a_system")
        assert "attitude_loop" in str(excinfo.value)

    def test_every_listed_system_resolves(self):
        from invariantset import system_names

        for name in system_names():
            s = get_system(name)
            assert s.A.shape == (s.dim, s.dim)
            assert s.W.dim == s.dim
            assert len(s.state_units) == s.dim


class TestPlottingArguments:
    def test_polygon_vertices_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            polygon_vertices(Polytope.unit_box(3))

    def test_plot_polytope_rejects_non_2d(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError, match="2-D"):
                plot_polytope(ax, Polytope.unit_box(3))
        finally:
            plt.close(fig)
