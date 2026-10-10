"""Plotting tests. The figures are evidence, so the functions that make them are
tested for the thing that can silently break: writing nothing, or writing an
empty file."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from telemdrift.detectors import alarm_ratio_trace, make_detector  # noqa: E402
from telemdrift.plotting import (  # noqa: E402
    plot_detector_traces,
    plot_operating_point_spread,
    plot_stream_panel,
    plot_tradeoff,
    plot_transient_response,
)
from telemdrift.streams import ChangeSpec, change_stream, stationary  # noqa: E402


def _nonempty(path):
    assert path.exists()
    assert path.stat().st_size > 5_000, f"{path} is only {path.stat().st_size} bytes"


def test_plot_tradeoff_writes_a_figure(tmp_path):
    curves = {
        "A": [(100.0, 20.0, 2.0), (500.0, 30.0, 3.0), (2000.0, 45.0, 4.0)],
        "B": [(120.0, 10.0, 1.0), (480.0, 14.0, 1.5), (1900.0, 22.0, 2.0)],
    }
    _nonempty(plot_tradeoff(curves, tmp_path / "t.png", "title", target_arl0=500.0))


def test_plot_tradeoff_tolerates_an_empty_curve(tmp_path):
    """A detector whose sweep produced no usable point must not break the figure."""
    curves = {"A": [(100.0, 20.0, 2.0), (500.0, 30.0, 3.0)], "B": []}
    _nonempty(plot_tradeoff(curves, tmp_path / "t2.png", "title"))


def test_plot_tradeoff_without_a_target_line(tmp_path):
    curves = {"A": [(100.0, 20.0, 2.0), (500.0, 30.0, 3.0)]}
    _nonempty(plot_tradeoff(curves, tmp_path / "t3.png", "title", target_arl0=None))


def test_plot_operating_point_spread_writes_a_figure(tmp_path):
    _nonempty(plot_operating_point_spread(
        ["A", "B", "C"], [100.0, 900.0, 8000.0], [5.0, 40.0, 1200.0],
        ["h=4", "L=3", "c=0.21"], tmp_path / "s.png", 500.0,
    ))


def test_plot_stream_panel_writes_a_figure(tmp_path):
    streams = {}
    for i, spec in enumerate((ChangeSpec("mean_step", 1.0),
                              ChangeSpec("variance_step", 2.0))):
        streams[spec.kind] = change_stream(100, 200, spec, 70 + i)
    _nonempty(plot_stream_panel(streams, tmp_path / "p.png"))


def test_plot_stream_panel_with_a_single_panel(tmp_path):
    streams = {"only": change_stream(50, 50, ChangeSpec("mean_step", 1.0), 71)}
    _nonempty(plot_stream_panel(streams, tmp_path / "p1.png"))


def test_plot_detector_traces_writes_a_figure(tmp_path):
    stream, idx = change_stream(200, 300, ChangeSpec("mean_step", 1.5), 72)
    ratios, alarms = {}, {}
    for key in ("cusum", "ewma"):
        series, fired = alarm_ratio_trace(make_detector(key), stream)
        ratios[key] = series
        alarms[key] = int(fired[0]) if fired.size else -1
    _nonempty(plot_detector_traces(stream, idx, ratios, tmp_path / "d.png", "title",
                                   alarms))


def test_plot_detector_traces_without_alarm_markers(tmp_path):
    stream, idx = change_stream(100, 100, ChangeSpec("mean_step", 1.0), 73)
    ratios = {"cusum": np.zeros(stream.size)}
    _nonempty(plot_detector_traces(stream, idx, ratios, tmp_path / "d2.png", "title"))


def test_plot_transient_response_writes_a_figure(tmp_path):
    _nonempty(plot_transient_response(
        ["A", "B", "C"], [1.0, 0.5, 0.0],
        [(0.98, 1.0), (0.44, 0.56), (0.0, 0.02)], tmp_path / "x.png", "title",
    ))


def test_plot_transient_response_at_both_degenerate_rates(tmp_path):
    """0/n and n/n are the rates this experiment actually produces."""
    _nonempty(plot_transient_response(
        ["none", "all"], [0.0, 1.0], [(0.0, 0.015), (0.985, 1.0)],
        tmp_path / "y.png", "title",
    ))


def test_figures_are_created_under_a_missing_directory(tmp_path):
    target = tmp_path / "a" / "b" / "c.png"
    curves = {"A": [(100.0, 20.0, 2.0), (500.0, 30.0, 3.0)]}
    _nonempty(plot_tradeoff(curves, target, "title"))


def test_plot_stream_panel_rejects_mismatched_panel_counts(tmp_path):
    """zip(strict=True) inside the plotting code must not silently truncate."""
    with pytest.raises((ValueError, TypeError)):
        plot_stream_panel({"a": (stationary(10, 1),)}, tmp_path / "bad.png")
