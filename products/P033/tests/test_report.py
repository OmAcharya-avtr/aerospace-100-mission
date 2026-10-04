"""Result files: the environment label, the empty device column, the JSON."""

from __future__ import annotations

import json

import pytest

from edgeinfer.environment import JETSON_ORIN_NANO_COLUMN, environment_record
from edgeinfer.report import (
    CLOUD_ENVIRONMENT_NOTE,
    NOT_MEASURED,
    ResultRow,
    crosscheck_payload,
    results_table,
    write_crosscheck_json,
    write_results_file,
)


def _rows() -> tuple[ResultRow, ...]:
    return (
        ResultRow("p99 latency", "s", 3.4e-5, 2.0e-6, "perf_counter, n=200, 10 warm-up"),
        ResultRow("p50 latency", "s", 8.0e-6, 3.0e-8, "perf_counter, n=200, 10 warm-up"),
        ResultRow("analytic peak memory", "B", 69408.0, None, "liveness analysis"),
        ResultRow("average power", "W", None, None, "not measured by edgeinfer"),
    )


class TestResultsTable:
    def test_the_device_column_is_empty_in_every_row(self) -> None:
        table = results_table(_rows(), "Jetson Orin Nano (measured on device)")
        for line in table.splitlines():
            if line.startswith("| p") or line.startswith("| analytic"):
                assert line.rstrip().endswith(f"{NOT_MEASURED} |")

    def test_the_device_column_header_is_present(self) -> None:
        table = results_table(_rows(), "Jetson Orin Nano (measured on device)")
        assert "Jetson Orin Nano (measured on device)" in table

    def test_the_host_column_says_it_is_a_shared_single_core_container(self) -> None:
        table = results_table(_rows(), "Target")
        assert "shared 1-core container" in table

    def test_every_method_is_listed(self) -> None:
        table = results_table(_rows(), "Target")
        for row in _rows():
            assert row.method in table

    def test_an_unmeasured_host_value_also_reads_not_measured(self) -> None:
        table = results_table(_rows(), "Target")
        power_line = next(line for line in table.splitlines() if "average power" in line)
        assert power_line.count(NOT_MEASURED) == 2

    def test_a_row_without_a_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no method"):
            ResultRow("q", "s", 1.0, None, "   ")

    def test_an_empty_table_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            results_table((), "Target")

    def test_an_empty_device_header_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty header"):
            results_table(_rows(), "  ")

    def test_the_jetson_column_constant_is_empty(self) -> None:
        """The constant exists so that nothing can quietly put a number in it."""
        assert JETSON_ORIN_NANO_COLUMN == ""


class TestWriteResultsFile:
    def test_the_file_states_its_environment_in_its_own_text(self, tmp_path) -> None:
        path = write_results_file(tmp_path / "r.md", "Test results", _rows())
        text = path.read_text()
        assert "shared" in text
        assert "1 CPU core" in text or "CPU core(s)" in text
        assert CLOUD_ENVIRONMENT_NOTE in text

    def test_the_file_contains_the_empty_device_column(self, tmp_path) -> None:
        path = write_results_file(tmp_path / "r.md", "Test results", _rows())
        assert "Jetson Orin Nano (measured on device)" in path.read_text()
        assert NOT_MEASURED in path.read_text()

    def test_extra_sections_are_appended(self, tmp_path) -> None:
        path = write_results_file(
            tmp_path / "r.md",
            "Test results",
            _rows(),
            extra_sections=(("What is missing", "Measured timing from the device."),),
        )
        text = path.read_text()
        assert "## What is missing" in text
        assert "Measured timing from the device." in text

    def test_parent_directories_are_created(self, tmp_path) -> None:
        path = write_results_file(tmp_path / "a" / "b" / "r.md", "T", _rows())
        assert path.is_file()

    def test_a_supplied_environment_is_used_verbatim(self, tmp_path) -> None:
        record = environment_record(shared_host=False, note="DEDICATED HOST CLAIM")
        path = write_results_file(
            tmp_path / "r.md", "T", _rows(), environment=record
        )
        assert "DEDICATED HOST CLAIM" in path.read_text()


class TestCrosscheckJson:
    def _payload(self) -> dict:
        return crosscheck_payload(
            stages=(
                {"name": "a", "mean_s": 1e-4, "std_s": 2e-5, "dist": "lognormal"},
                {"name": "b", "mean_s": 2e-4, "std_s": 1e-5, "dist": "lognormal"},
            ),
            n_samples=400,
            seed=20260401,
            mean_s=3.1e-4,
            p50_s=3.0e-4,
            p99_s=4.5e-4,
        )

    def test_the_required_schema_keys_are_at_the_top_level(self) -> None:
        payload = self._payload()
        assert set(payload) >= {"stages", "n_samples", "seed", "measured"}
        assert set(payload["measured"]) == {"mean_s", "p50_s", "p99_s"}
        for stage in payload["stages"]:
            assert set(stage) == {"name", "mean_s", "std_s", "dist"}

    def test_the_written_json_round_trips(self, tmp_path) -> None:
        path = write_crosscheck_json(tmp_path / "c.json", self._payload())
        loaded = json.loads(path.read_text())
        assert loaded["n_samples"] == 400
        assert loaded["seed"] == 20260401
        assert loaded["measured"]["p99_s"] == pytest.approx(4.5e-4)

    def test_the_environment_is_attached_under_a_prefixed_key(self, tmp_path) -> None:
        path = write_crosscheck_json(tmp_path / "c.json", self._payload())
        loaded = json.loads(path.read_text())
        assert "_environment" in loaded
        assert loaded["_environment_note"] == CLOUD_ENVIRONMENT_NOTE
        assert loaded["_environment"]["shared_host"] is True

    def test_extra_keys_do_not_disturb_the_required_ones(self, tmp_path) -> None:
        path = write_crosscheck_json(
            tmp_path / "c.json", self._payload(), extra={"injected_mean_s": 3e-4}
        )
        loaded = json.loads(path.read_text())
        assert loaded["injected_mean_s"] == pytest.approx(3e-4)
        assert loaded["n_samples"] == 400

    def test_no_stages_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            crosscheck_payload((), 10, 1, 1.0, 1.0, 1.0)

    def test_a_non_positive_statistic_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="mean_s must be > 0"):
            crosscheck_payload(({"name": "a"},), 10, 1, 0.0, 1.0, 1.0)

    def test_an_inverted_quantile_order_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            crosscheck_payload(({"name": "a"},), 10, 1, 1.0, 2.0, 1.0)

    def test_zero_samples_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            crosscheck_payload(({"name": "a"},), 0, 1, 1.0, 1.0, 1.0)
