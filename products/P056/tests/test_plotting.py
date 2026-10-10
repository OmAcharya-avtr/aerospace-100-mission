"""Plotting smoke tests. Agg backend, no display, figures closed afterwards."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from calibaudit import (  # noqa: E402
    binned_decomposition,
    bootstrap_reliability,
    calibration_map,
    ece_bias_curve,
    get_spec,
    reliability_curve,
    sample_size_sweep,
)
from calibaudit.plotting import (  # noqa: E402
    plot_decomposition_bars,
    plot_ece_bias_curve,
    plot_reliability,
    plot_sample_size_sweep,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_backend_is_agg():
    assert matplotlib.get_backend().lower() == "agg"


def test_plot_reliability_without_band(overconfident_small):
    s = overconfident_small
    fig = plot_reliability(reliability_curve(s.forecasts, s.outcomes, n_bins=8))
    assert len(fig.axes) == 2


def test_plot_reliability_with_band_and_truth(overconfident_small):
    import numpy as np

    s = overconfident_small
    curve = bootstrap_reliability(
        s.forecasts, s.outcomes, n_bins=8, n_bootstrap=40, seed=1
    )
    grid = np.linspace(0.01, 0.99, 50)
    fig = plot_reliability(
        curve, truth_x=grid, truth_y=calibration_map(get_spec("overconfident"), grid)
    )
    assert len(fig.axes) == 2
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("band" in label for label in labels)


def test_plot_ece_bias_curve(tmp_path):
    curve = ece_bias_curve(
        get_spec("calibrated"),
        n_bins_grid=[5, 10, 20],
        n_samples_grid=[200, 800],
        n_replicates=8,
        seed=1,
    )
    fig = plot_ece_bias_curve(curve)
    assert len(fig.axes) == 2
    out = tmp_path / "bias.png"
    fig.savefig(out)
    assert out.stat().st_size > 5000


def test_plot_ece_bias_curve_survives_an_unfittable_grid():
    curve = ece_bias_curve(
        get_spec("calibrated"),
        n_bins_grid=[10],
        n_samples_grid=[400],
        n_replicates=4,
        seed=1,
    )
    fig = plot_ece_bias_curve(curve)
    assert len(fig.axes) == 2


def test_plot_decomposition_bars(overconfident_small, calibrated_small):
    decs = {
        "overconfident": binned_decomposition(
            overconfident_small.forecasts, overconfident_small.outcomes, n_bins=8
        ),
        "calibrated": binned_decomposition(
            calibrated_small.forecasts, calibrated_small.outcomes, n_bins=8
        ),
    }
    fig = plot_decomposition_bars(decs)
    assert len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == list(decs)


def test_plot_sample_size_sweep(tmp_path):
    sweep = sample_size_sweep(
        get_spec("overconfident"), n_samples_grid=[100, 400], n_replicates=5, seed=1
    )
    fig = plot_sample_size_sweep(sweep)
    assert len(fig.axes) == 2
    out = tmp_path / "sweep.png"
    fig.savefig(out)
    assert out.stat().st_size > 5000
