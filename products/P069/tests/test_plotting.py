"""The figures. Only that they are produced, are non-trivial and use Agg."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

from scenariocov import (
    GreedyMaxMinStrategy,
    UniformRandomStrategy,
    build_report,
    coverage_benchmark,
    grid_occupancy,
)
from scenariocov.plotting import (
    plot_coverage_curves,
    plot_occupancy_slices,
    plot_report_summary,
)


def test_backend_is_agg() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_report_summary_figure(tmp_path: Path, rendezvous_space, rendezvous_campaign) -> None:
    report = build_report(rendezvous_space, rendezvous_campaign)
    out = plot_report_summary(report, tmp_path / "summary.png")
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_occupancy_figure(tmp_path: Path, rendezvous_space, rendezvous_campaign) -> None:
    occ = grid_occupancy(rendezvous_space, rendezvous_space.encode_many(rendezvous_campaign))
    out = plot_occupancy_slices(occ, tmp_path / "occ.png")
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_occupancy_figure_rejects_a_bad_axis_pair(
    tmp_path: Path, rendezvous_space, rendezvous_campaign
) -> None:
    occ = grid_occupancy(rendezvous_space, rendezvous_space.encode_many(rendezvous_campaign))
    with pytest.raises(ValueError, match="two distinct valid grid axes"):
        plot_occupancy_slices(occ, tmp_path / "bad.png", axis_pair=(0, 0))


def test_coverage_curve_figure(
    tmp_path: Path, rendezvous_space, rendezvous_campaign, candidate_pool
) -> None:
    result = coverage_benchmark(
        rendezvous_space,
        rendezvous_campaign,
        candidate_pool,
        [UniformRandomStrategy(), GreedyMaxMinStrategy()],
        n_start=20,
        n_steps=6,
        pool_size=40,
        seeds=(0, 1, 2),
    )
    out = plot_coverage_curves(result, tmp_path / "curves.png")
    assert out.is_file()
    assert out.stat().st_size > 10_000
