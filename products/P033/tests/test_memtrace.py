"""Peak-memory measurement against known allocation patterns."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.memtrace import (
    measure_peak_python_bytes,
    resident_set_bytes,
    resident_set_high_water_bytes,
    track_peak_python_bytes,
)

#: tracemalloc counts the allocation request, so a float64 array of n elements
#: shows up as 8 n bytes plus a small NumPy object header. The header is a few
#: hundred bytes at most, so an assertion with this tolerance is checking the
#: array payload rather than the interpreter's bookkeeping.
HEADER_SLACK_BYTES = 1024


class TestKnownAllocationPattern:
    def test_a_single_array_is_measured_to_its_payload(self) -> None:
        """8 MB: 1_000_000 float64 elements at 8 B each."""
        n = 1_000_000
        expected = n * 8

        def allocate() -> np.ndarray:
            return np.zeros(n, dtype=np.float64)

        _result, peak = measure_peak_python_bytes(allocate)
        assert expected <= peak.peak_above_baseline_bytes <= expected + HEADER_SLACK_BYTES

    def test_two_live_arrays_peak_at_their_sum(self) -> None:
        """A (1e6, 2e6) pair held simultaneously peaks at 8 MB + 16 MB."""
        expected = 1_000_000 * 8 + 2_000_000 * 8

        def allocate() -> int:
            a = np.zeros(1_000_000, dtype=np.float64)
            b = np.zeros(2_000_000, dtype=np.float64)
            return int(a[0] + b[0])

        _result, peak = measure_peak_python_bytes(allocate)
        assert expected <= peak.peak_above_baseline_bytes <= expected + HEADER_SLACK_BYTES

    def test_a_released_array_does_not_enter_the_peak_twice(self) -> None:
        """Two 8 MB arrays allocated in sequence, the first released before the
        second is created, peak at 8 MB and not 16 MB."""
        expected = 1_000_000 * 8

        def allocate() -> int:
            a = np.zeros(1_000_000, dtype=np.float64)
            value = int(a[0])
            del a
            b = np.zeros(1_000_000, dtype=np.float64)
            return value + int(b[0])

        _result, peak = measure_peak_python_bytes(allocate)
        assert expected <= peak.peak_above_baseline_bytes < 1.5 * expected

    def test_dtype_width_changes_the_measured_peak_proportionally(self) -> None:
        n = 500_000

        def allocate(dtype) -> np.ndarray:
            return np.zeros(n, dtype=dtype)

        _a, peak32 = measure_peak_python_bytes(allocate, np.float32)
        _b, peak64 = measure_peak_python_bytes(allocate, np.float64)
        ratio = peak64.peak_above_baseline_bytes / peak32.peak_above_baseline_bytes
        assert ratio == pytest.approx(2.0, rel=0.01)

    def test_the_returned_value_is_passed_through(self) -> None:
        result, _peak = measure_peak_python_bytes(lambda: 42)
        assert result == 42

    def test_the_method_string_states_what_tracemalloc_does_not_see(self) -> None:
        _r, peak = measure_peak_python_bytes(lambda: None)
        assert "not a C++ runtime's own arena" in peak.method

    def test_summary_lines_carry_units(self) -> None:
        _r, peak = measure_peak_python_bytes(lambda: np.zeros(1000))
        assert any("B" in line for line in peak.summary_lines())


class TestContextManager:
    def test_nesting_is_safe(self) -> None:
        with track_peak_python_bytes() as outer:
            with track_peak_python_bytes() as inner:
                _buffer = np.zeros(200_000, dtype=np.float64)
            assert inner[0].peak_above_baseline_bytes >= 200_000 * 8
        assert outer[0].peak_bytes >= inner[0].peak_bytes - HEADER_SLACK_BYTES

    def test_tracing_is_stopped_again_afterwards(self) -> None:
        import tracemalloc

        was_tracing = tracemalloc.is_tracing()
        with track_peak_python_bytes():
            pass
        assert tracemalloc.is_tracing() == was_tracing

    def test_an_exception_inside_the_region_still_records_a_peak(self) -> None:
        holder: list = []
        with pytest.raises(RuntimeError):
            with track_peak_python_bytes() as holder:
                np.zeros(100_000, dtype=np.float64)
                raise RuntimeError("boom")
        assert holder and holder[0].peak_bytes > 0


class TestResidentSet:
    def test_rss_is_positive_on_linux_or_none_elsewhere(self) -> None:
        value = resident_set_bytes()
        assert value is None or value > 0

    def test_high_water_is_at_least_the_current_rss(self) -> None:
        current = resident_set_bytes()
        high = resident_set_high_water_bytes()
        if current is None or high is None:
            pytest.skip("no /proc on this platform")
        assert high >= current
