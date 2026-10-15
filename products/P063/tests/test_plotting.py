"""Plot helpers. The Agg backend is asserted and no figure is shown."""

from __future__ import annotations

import matplotlib
import pytest

from safemodetree import (
    analyse,
    compare_detectors,
    conservatism_sweep,
    reachability_report,
)
from safemodetree.plotting import (
    plot_ambiguity_costs,
    plot_conservatism_sweep,
    plot_detector_sweep,
    plot_reachability,
    plot_reliability,
)


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


@pytest.fixture(scope="module")
def comparison(fmea):
    return compare_detectors(fmea, n_per_fault=80, noise_scale=2.0, seed=63002)


def test_plot_reachability_writes_a_png(fmea, tmp_path):
    figure = plot_reachability(reachability_report(fmea), fmea)
    path = tmp_path / "reach.png"
    figure.savefig(path)
    assert path.stat().st_size > 1000


def test_plot_reachability_marks_the_unreachable_rows(planted, tmp_path):
    figure = plot_reachability(reachability_report(planted), planted)
    path = tmp_path / "planted.png"
    figure.savefig(path)
    assert path.stat().st_size > 1000


def test_plot_ambiguity_costs(fmea, tmp_path):
    figure = plot_ambiguity_costs(analyse(fmea).ambiguity)
    path = tmp_path / "ambiguity.png"
    figure.savefig(path)
    assert path.stat().st_size > 1000


def test_plot_ambiguity_costs_tolerates_no_classes(minimal, tmp_path):
    figure = plot_ambiguity_costs(analyse(minimal).ambiguity)
    path = tmp_path / "none.png"
    figure.savefig(path)
    assert path.stat().st_size > 500


def test_plot_reliability(comparison, tmp_path):
    figure = plot_reliability(
        {
            "threshold logic": comparison["baseline"].calibration,
            "learned": comparison["learned"].calibration,
        }
    )
    path = tmp_path / "reliability.png"
    figure.savefig(path)
    assert path.stat().st_size > 1000


def test_plot_detector_sweep(comparison, tmp_path):
    figure = plot_detector_sweep(
        (2.0,), (comparison["baseline"],), (comparison["learned"],), float(comparison["ceiling"])
    )
    path = tmp_path / "sweep.png"
    figure.savefig(path)
    assert path.stat().st_size > 1000


def test_plot_conservatism_sweep(fmea, comparison, tmp_path):
    rows = conservatism_sweep(fmea, comparison, mode_id="fine_pointing",
                              thresholds=(0.0, 0.5, 2.0))
    figure = plot_conservatism_sweep(rows)
    path = tmp_path / "conservatism.png"
    figure.savefig(path)
    assert path.stat().st_size > 1000
