"""The committed synthetic generator: determinism, schema and faithfulness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deadlinerisk.accounting import account
from deadlinerisk.provenance import Origin
from deadlinerisk.schema import JOB_COLUMNS, MONITOR_COLUMNS

from fixtures import synthetic_trace


def test_generated_bytes_are_identical_at_the_same_seed(tmp_path: Path) -> None:
    first = synthetic_trace.write_synthetic_campaign(tmp_path / "a", seed=4242)
    second = synthetic_trace.write_synthetic_campaign(tmp_path / "b", seed=4242)
    names = sorted(path.name for path in first.iterdir())
    assert names == sorted(path.name for path in second.iterdir())
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_a_different_seed_changes_the_traces(tmp_path: Path) -> None:
    first = synthetic_trace.write_synthetic_campaign(tmp_path / "a", seed=1)
    second = synthetic_trace.write_synthetic_campaign(tmp_path / "b", seed=2)
    assert (first / "jobs_gt_long.csv").read_bytes() != (
        second / "jobs_gt_long.csv"
    ).read_bytes()


def test_written_headers_are_the_declared_schema(synthetic_directory: Path) -> None:
    jobs_header = (synthetic_directory / "jobs_gt_long.csv").read_text().splitlines()[0]
    monitor_header = (synthetic_directory / "mon_gt_long.csv").read_text().splitlines()[0]
    assert jobs_header == ",".join(JOB_COLUMNS)
    assert monitor_header == ",".join(MONITOR_COLUMNS)


def test_manifest_says_it_is_a_fixture(campaign) -> None:
    assert "SYNTHETIC FIXTURE" in campaign.manifest["campaign"]
    assert campaign.manifest["env"]["model"].startswith("synthetic fixture")
    assert campaign.provenance.origin is Origin.SYNTHETIC


def test_completion_marker_is_written_only_when_asked(tmp_path: Path) -> None:
    complete = synthetic_trace.write_synthetic_campaign(tmp_path / "c", complete=True)
    partial = synthetic_trace.write_synthetic_campaign(
        tmp_path / "p", complete=False, plan=(synthetic_trace.RunSpec("mix_c0", 0, 0, 5.0),)
    )
    assert (complete / "CAMPAIGN_COMPLETE").exists()
    assert not (partial / "CAMPAIGN_COMPLETE").exists()


def test_manifest_run_totals_agree_with_the_reconstruction(campaign) -> None:
    for run, jobs in campaign.jobs.items():
        totals = account(jobs)
        meta = campaign.run_meta(run)
        # Integer identities: the generator records what the reconstruction
        # recomputes, so a disagreement is a defect and not a tolerance.
        assert meta["releases"] == totals.releases, run
        assert meta["misses"] == totals.misses, run


def test_contention_factor_is_monotone_in_both_arguments() -> None:
    base = synthetic_trace.contention_factor(0, 0)
    assert base == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert synthetic_trace.contention_factor(1, 0) > base
    assert synthetic_trace.contention_factor(0, 1) > synthetic_trace.contention_factor(1, 0)
    for cpu in range(6):
        assert synthetic_trace.contention_factor(cpu + 1, 0) > (
            synthetic_trace.contention_factor(cpu, 0)
        )


def test_contention_factor_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="worker counts"):
        synthetic_trace.contention_factor(-1, 0)


def test_known_answer_simulate_loop_drops_a_release() -> None:
    # HAND CALCULATION. T = D = 10 ms. Execution times [25 ms, 4 ms].
    # Job 0: release 0, start 0, end 25 ms, response 25 ms > 10 ms so missed.
    #   next_index = floor(25/10) + 1 = 2 + 1 = 3, dropped_before = 3-0-1 = 2
    #   (releases at 10 ms and 20 ms never ran).
    # Job 3: release 30 ms, start max(30, 25) = 30 ms, end 34 ms,
    #   response 4 ms, not missed, next_index = floor(34/10)+1 = 4,
    #   dropped_before = 4-3-1 = 0.
    # So releases = (1+2) + (1+0) = 4 and misses = (1+0) + (2+0) = 3.
    columns = synthetic_trace.simulate_loop(
        np.asarray([25e-3, 4e-3]), period_s=10e-3, deadline_s=10e-3
    )
    assert columns["job_index"].tolist() == [0, 3]
    assert columns["dropped_before"].tolist() == [2, 0]
    assert columns["missed"].tolist() == [1, 0]
    assert columns["response_s"][0] == pytest.approx(25e-3, rel=1e-12, abs=0.0)
    assert columns["response_s"][1] == pytest.approx(4e-3, rel=1e-12, abs=0.0)
    assert int((1 + columns["dropped_before"]).sum()) == 4
    assert int((columns["missed"] + columns["dropped_before"]).sum()) == 3


def test_known_answer_simulate_loop_with_no_overrun() -> None:
    # HAND CALCULATION. T = D = 10 ms, exec = [4 ms, 4 ms, 4 ms].
    # Every job starts at its release and finishes 4 ms later, so no release is
    # dropped and nothing is missed: indices 0, 1, 2 and releases = misses = 0.
    columns = synthetic_trace.simulate_loop(
        np.asarray([4e-3, 4e-3, 4e-3]), period_s=10e-3, deadline_s=10e-3
    )
    assert columns["job_index"].tolist() == [0, 1, 2]
    assert columns["dropped_before"].tolist() == [0, 0, 0]
    assert columns["missed"].tolist() == [0, 0, 0]


def test_execution_times_input_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_jobs"):
        synthetic_trace.execution_times(0, 0, 0, rng)
    with pytest.raises(ValueError, match="tail_probability"):
        synthetic_trace.execution_times(10, 0, 0, rng, tail_probability=1.5)
    with pytest.raises(ValueError, match="tail_alpha"):
        synthetic_trace.execution_times(10, 0, 0, rng, tail_alpha=0.0)


def test_simulate_loop_rejects_a_non_positive_period() -> None:
    with pytest.raises(ValueError, match="must both be > 0"):
        synthetic_trace.simulate_loop(np.asarray([1e-3]), period_s=0.0)


def test_heavier_tail_gives_a_higher_miss_rate(tmp_path: Path) -> None:
    # The generator's tail knob must move the miss rate in the direction the
    # negative controls rely on. Both runs use the same seed, so the difference
    # is the knob and not the draw.
    light = synthetic_trace.load_synthetic_campaign(
        tmp_path / "light",
        plan=(synthetic_trace.RunSpec("gt_long", 4, 0, 40.0, tail_probability=0.0),),
        seed=99,
    )
    heavy = synthetic_trace.load_synthetic_campaign(
        tmp_path / "heavy",
        plan=(
            synthetic_trace.RunSpec(
                "gt_long", 4, 0, 40.0, tail_probability=0.05, tail_alpha=1.2
            ),
        ),
        seed=99,
    )
    assert account(heavy.jobs["gt_long"]).miss_rate > account(
        light.jobs["gt_long"]
    ).miss_rate
