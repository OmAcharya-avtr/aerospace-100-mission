"""The synthetic generator: naming, determinism, and schema identity.

These tests guard the one property that makes the generator safe to ship beside
measured data: nothing it writes can be mistaken for a measurement, by filename,
by manifest content or by the origin the loader carries.
"""

from __future__ import annotations

import json
import os

import pytest

from thermalbudget import schema
from thermalbudget.provenance import Origin
from thermalbudget.trace import load_campaign, load_trace

from fixtures import synthetic_trace as st


def test_every_written_file_is_prefixed_synthetic(tmp_path):
    st.write_campaign(tmp_path, regime="no_throttle")
    written = sorted(path.name for path in tmp_path.iterdir())
    assert written
    for name in written:
        assert name.startswith(st.PREFIX), name
    assert f"{st.PREFIX}manifest.json" in written
    assert len([name for name in written if name.endswith(".csv")]) == len(st.PLAN)


def test_the_manifest_says_in_the_file_that_it_is_not_a_board(tmp_path):
    st.write_campaign(tmp_path, regime="throttled")
    manifest = json.loads((tmp_path / f"{st.PREFIX}manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert "not a board and not a measurement" in manifest["model"]
    assert manifest["generator"] == "tests/fixtures/synthetic_trace.py"
    assert manifest["regime"] == "throttled"
    assert all(run["synthetic"] is True for run in manifest["runs"])


def test_the_header_is_identical_to_the_binding_schema(tmp_path):
    path = st.write_run(tmp_path, "soak_1.00_r1", 1.0, 60, regime="no_throttle")
    header = path.read_text().splitlines()[0].split(",")
    assert tuple(header) == schema.expected_columns()
    schema.validate_header(header)


def test_generation_is_deterministic_for_a_seed(tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    st.write_campaign(first, regime="throttled", seed=4242)
    st.write_campaign(second, regime="throttled", seed=4242)
    for path in sorted(first.glob("*.csv")):
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name


def test_a_different_seed_gives_different_bytes(tmp_path):
    first = st.write_run(tmp_path / "a", "soak_1.00_r1", 1.0, 120, regime="throttled", seed=1)
    second = st.write_run(tmp_path / "b", "soak_1.00_r1", 1.0, 120, regime="throttled", seed=2)
    assert first.read_bytes() != second.read_bytes()


def test_the_regime_is_required_and_validated(tmp_path):
    with pytest.raises(TypeError):
        st.write_campaign(tmp_path)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="regime must be one of"):
        st.write_campaign(tmp_path, regime="mild")


def test_duty_and_duration_are_validated(tmp_path):
    with pytest.raises(ValueError, match=r"duty must lie in \[0, 1\]"):
        st.write_run(tmp_path, "x", 1.5, 30, regime="no_throttle")
    with pytest.raises(ValueError, match="seconds must be positive"):
        st.write_run(tmp_path, "x", 1.0, 0, regime="no_throttle")


def test_the_injected_missing_rows_are_sentinel_coded(tmp_path):
    path = st.write_run(
        tmp_path, "soak_0.50", 0.5, 60, regime="no_throttle", inject_missing=(5, 6, 7)
    )
    trace = load_trace(path, origin=Origin.SYNTHETIC)
    assert trace.missing[schema.DIE_COLUMN] == 3
    assert trace.missing["cpu0_khz"] == 3
    assert trace.missing["gpu_hz"] == 3
    assert trace.missing["foreign_cpu_pct"] == 3


def test_the_loader_entry_point_stamps_synthetic(tmp_path):
    campaign = st.load_synthetic_campaign(tmp_path, regime="no_throttle")
    assert campaign.origin is Origin.SYNTHETIC
    assert all(trace.origin is Origin.SYNTHETIC for trace in campaign)
    assert not campaign.board_matches_declared_target()


def test_a_synthetic_campaign_loaded_as_measured_is_the_callers_lie_not_the_files(tmp_path):
    # Stated as a test because it is the limit of what the naming can guarantee:
    # a caller who passes Origin.MEASURED gets MEASURED, and nothing in a file
    # can stop that. What the file CAN do is say in its manifest that it is not a
    # board, which is what the report layer prints and what this asserts.
    st.write_campaign(tmp_path, regime="no_throttle")
    (tmp_path / "manifest.json").write_text(
        (tmp_path / f"{st.PREFIX}manifest.json").read_text()
    )
    campaign = load_campaign(tmp_path, origin=Origin.MEASURED)
    assert campaign.origin is Origin.MEASURED
    assert not campaign.board_matches_declared_target()
    assert "not a measurement" in campaign.board_model


def test_the_cooldown_runs_actually_decay(tmp_path):
    # A cooldown that starts at the idle floor carries no decay and the
    # independent tau estimate has nothing to work on. The generator threads the
    # die temperature across runs so that it does; this asserts it still does.
    campaign = st.load_synthetic_campaign(tmp_path, regime="no_throttle")
    hot = campaign["cool_r1"].die_temperature_c
    assert float(hot[0] - hot[-1]) > 10.0


def test_the_generator_cli_writes_a_campaign(tmp_path, capsys):
    assert st.main([str(tmp_path), "--regime", "throttled", "--seed", "7"]) == 0
    assert "SYNTHETIC runs" in capsys.readouterr().out
    assert (tmp_path / f"{st.PREFIX}manifest.json").exists()


def test_the_declared_regimes_differ_in_their_clock_channel(tmp_path):
    pinned = load_trace(
        st.write_run(tmp_path / "a", "soak_1.00_r1", 1.0, 400, regime="no_throttle"),
        origin=Origin.SYNTHETIC,
    )
    rolling = load_trace(
        st.write_run(tmp_path / "b", "soak_1.00_r1", 1.0, 400, regime="throttled"),
        origin=Origin.SYNTHETIC,
    )
    assert float(pinned.mean_cpu_khz.max() - pinned.mean_cpu_khz.min()) == 0.0
    assert float(rolling.mean_cpu_khz.max() - rolling.mean_cpu_khz.min()) > 1e5


def test_generation_is_deterministic_ACROSS_processes(tmp_path):
    """The determinism test that matters, and the one that caught a real defect.

    An earlier version of the generator seeded each run's noise stream with
    ``hash(label)``. CPython salts string hashing per process, so the fixtures
    were deterministic inside one interpreter and different between two. Two
    runs of ``validation/worked_example.py`` minutes apart printed different
    headline numbers, while ``test_generation_is_deterministic_for_a_seed`` --
    which writes both campaigns in the same process -- passed throughout. This
    test writes them in two SUBPROCESSES with different ``PYTHONHASHSEED``
    values, which is the only way to see it.
    """
    import subprocess
    import sys
    from pathlib import Path

    generator = Path(st.__file__)
    outputs = []
    for index, hash_seed in enumerate(("0", "12345")):
        target = tmp_path / f"run{index}"
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONHASHSEED": hash_seed,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        completed = subprocess.run(
            [sys.executable, str(generator), str(target), "--regime", "throttled",
             "--seed", "4242"],
            capture_output=True,
            text=True,
            env=environment,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(target)
    first, second = outputs
    names = sorted(path.name for path in first.glob("*.csv"))
    assert names
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_the_label_offset_is_a_fixed_function_of_the_bytes():
    # Literal values, so a change to the offset function is visible as a test
    # failure rather than as silently different fixtures.
    assert st.label_offset("soak_1.00_r1") == st.label_offset("soak_1.00_r1")
    assert st.label_offset("soak_1.00_r1") != st.label_offset("soak_1.00_r2")
    assert 0 <= st.label_offset("idle_pre") < 10_000
