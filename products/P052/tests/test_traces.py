"""Trace construction and its validation."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.traces import Trace


def _times(n: int = 5, dt: float = 0.1) -> np.ndarray:
    return np.arange(n) * dt


def test_trace_known_answer_properties() -> None:
    # Hand calculation: 5 samples at dt = 0.1 s span t = 0.0 .. 0.4 s, so the
    # duration is 0.4 s exactly and dt is 0.1 s.
    trace = Trace(_times(), {"y": np.arange(5.0)})
    assert trace.length == 5
    assert trace.dt == pytest.approx(0.1)
    assert trace.duration == pytest.approx(0.4)
    assert trace.names == ("y",)
    assert "y" in trace
    assert "q" not in trace


def test_signal_returns_the_stored_array() -> None:
    values = np.array([1.0, 2.0, 3.0])
    trace = Trace(_times(3), {"y": values})
    np.testing.assert_allclose(trace.signal("y"), values)


def test_multiple_signals_are_sorted_in_names() -> None:
    trace = Trace(_times(3), {"z": np.zeros(3), "a": np.ones(3)})
    assert trace.names == ("a", "z")


def test_missing_signal_raises_keyerror_naming_what_exists() -> None:
    trace = Trace(_times(3), {"y": np.zeros(3)})
    with pytest.raises(KeyError, match="available signals"):
        trace.signal("theta")


def test_repr_mentions_length_and_dt() -> None:
    text = repr(Trace(_times(4), {"y": np.zeros(4)}))
    assert "T=4" in text
    assert "dt=0.1" in text


@pytest.mark.parametrize(
    ("times", "signals", "match"),
    [
        (np.zeros((2, 2)), {"y": np.zeros(4)}, "one-dimensional"),
        (np.array([0.0]), {"y": np.zeros(1)}, "at least 2 samples"),
        (np.array([0.0, np.nan]), {"y": np.zeros(2)}, "non-finite"),
        (np.array([0.0, 0.1, 0.05]), {"y": np.zeros(3)}, "strictly increasing"),
        (np.array([0.0, 0.1, 0.4]), {"y": np.zeros(3)}, "uniformly spaced"),
        (np.arange(3) * 0.1, {"y": np.zeros(4)}, "samples but times has"),
        (np.arange(3) * 0.1, {"y": np.zeros((3, 2))}, "one-dimensional"),
        (np.arange(3) * 0.1, {"y": np.array([0.0, np.inf, 1.0])}, "non-finite"),
        (np.arange(3) * 0.1, {}, "at least one signal"),
        (np.arange(3) * 0.1, {"": np.zeros(3)}, "non-empty strings"),
    ],
)
def test_trace_rejects_bad_input(times, signals, match) -> None:
    with pytest.raises(ValueError, match=match):
        Trace(times, signals)


def test_trace_rejects_non_mapping_signals() -> None:
    with pytest.raises(TypeError, match="mapping"):
        Trace(_times(3), [np.zeros(3)])  # type: ignore[arg-type]


def test_uniformity_tolerance_accepts_floating_point_arange() -> None:
    # np.arange with a non-representable step does not produce exactly equal
    # diffs; the constructor must accept that rather than demanding bit equality.
    times = np.arange(0.0, 2.0005, 0.005)
    trace = Trace(times, {"y": np.zeros(times.size)})
    assert trace.length == times.size
