"""Plotting tests: Agg only, no window, figures actually written."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

from invariantset import Polytope, get_system, maximal_robust_invariant_set
from invariantset.plotting import plot_polytope, polygon_vertices


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_polygon_vertices_are_ordered_counter_clockwise():
    V = polygon_vertices(Polytope.unit_box(2))
    assert V.shape == (4, 2)
    centre = V.mean(axis=0)
    angles = np.arctan2(V[:, 1] - centre[1], V[:, 0] - centre[0])
    assert np.all(np.diff(angles) > 0)


def test_plot_polytope_draws_a_closed_outline(tmp_path):
    import matplotlib.pyplot as plt

    s = get_system("attitude_loop")
    res = maximal_robust_invariant_set(s.A, s.X, s.W)
    fig, ax = plt.subplots()
    try:
        lines = plot_polytope(ax, res.polytope, color="k")
        assert len(lines) == 1
        xdata = lines[0].get_xdata()
        assert len(xdata) == res.polytope.n_halfspaces + 1
        assert xdata[0] == pytest.approx(xdata[-1])
        out = tmp_path / "plot.png"
        fig.savefig(out, dpi=60)
        assert out.stat().st_size > 1000
    finally:
        plt.close(fig)


def test_plot_polytope_can_fill():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        patches = plot_polytope(ax, Polytope.unit_box(2), fill=True, alpha=0.3)
        assert len(patches) == 1
    finally:
        plt.close(fig)
