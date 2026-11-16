"""Simulation mode: the committed pipeline, executed here, unmodified.

The load-bearing test in this file is
``test_the_timed_source_is_byte_identical_to_the_board_copy``. SPEC §1's whole
claim is that the same file runs in both modes, so the file the package times
is hashed at the moment it is timed.
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

from hiltimingdiff import sources
from hiltimingdiff.simulate import (
    DEFAULT_FRAMES,
    DEFAULT_PASSES,
    calibrate_ns_per_op,
    measure_simulation,
)

STAGES = ("ingest", "background", "threshold", "centroid", "gate", "track")


def test_default_pass_count_is_twelve():
    assert DEFAULT_PASSES == 12


def test_default_frame_count_matches_the_board_stack():
    assert DEFAULT_FRAMES == 1500


def test_the_timed_source_is_byte_identical_to_the_board_copy():
    # SPEC §1 rests on this. The module the simulation times is the file at
    # data/pat_pipeline.py, and that file's md5 is the one the coordinator
    # verified against the copy that executed on the Jetson.
    module = sources.pipeline_module()
    from pathlib import Path

    on_disk = Path(module.__file__).resolve()
    assert on_disk == sources.source_path("pat_pipeline.py").resolve()
    assert sources.md5_of_file(on_disk) == "80ba9015b54fa8e95eae96b2c93eb9c2"


def test_the_timing_instrument_is_byte_identical_to_the_board_copy():
    module = sources.executor_module()
    from pathlib import Path

    on_disk = Path(module.__file__).resolve()
    assert on_disk == sources.source_path("staged_exec.py").resolve()
    assert sources.md5_of_file(on_disk) == "0f654894ea7dc3c6ec5d3880e9150459"


def test_simulation_covers_every_cell_of_the_campaign(simulation, campaign):
    assert set(simulation.table["cell"].unique()) == set(campaign.runs["cell"])


def test_simulation_covers_the_six_stages(simulation):
    assert set(simulation.table["stage"].unique()) == set(STAGES)
    assert simulation.stages == STAGES


def test_simulation_row_count_is_cells_times_passes_times_stages(simulation):
    assert len(simulation.table) == 9 * simulation.passes * 6


@pytest.mark.parametrize("stage", STAGES)
def test_simulated_stage_median_is_positive(simulation, stage):
    assert simulation.stage_medians_ns()[stage] > 0.0


@pytest.mark.parametrize("stage", STAGES)
def test_simulated_declared_ops_are_positive(simulation, stage):
    assert simulation.declared_ops_medians()[stage] > 0.0


def test_simulation_records_this_machine(simulation):
    assert simulation.machine == platform.machine()


def test_simulation_records_this_interpreter(simulation):
    assert simulation.python == sys.version.split()[0]


def test_simulation_records_this_numpy(simulation):
    assert simulation.numpy == np.__version__


def test_simulation_measures_its_own_bracket_cost(simulation):
    assert simulation.bracket_ns > 0.0


def test_simulation_records_the_clock_resolution(simulation):
    assert simulation.clock_resolution_ns > 0.0


def test_simulation_elapsed_time_is_positive(simulation):
    assert simulation.elapsed_s > 0.0


def test_simulation_counts_timed_frames(simulation):
    assert simulation.n_timed_frames == 9 * simulation.passes * simulation.frames


def test_simulation_census_has_the_five_gate_branches(simulation):
    assert set(simulation.census) == {
        "accept",
        "no_candidate",
        "low_score",
        "area_out",
        "border",
    }


def test_simulation_fires_every_gate_branch_on_this_host_too(simulation):
    # Not a SPEC §9 requirement, which is about the board. Recorded because a
    # branch that fires on the board and not here would mean the two modes are
    # exercising different paths of the same file.
    for branch, count in simulation.census.items():
        assert count >= 1, branch


@pytest.mark.parametrize("stage", STAGES)
def test_per_pass_summary_columns_are_internally_consistent(simulation, stage):
    rows = simulation.table.loc[simulation.table["stage"] == stage]
    for row in rows.itertuples():
        assert row.min_ns <= row.median_ns <= row.max_ns
        assert row.p05_ns <= row.median_ns <= row.p95_ns
        assert row.sum_ns == pytest.approx(row.mean_ns * row.n_frames, rel=1e-9, abs=0.0)


def test_simulation_k_mean_is_positive(simulation):
    assert float(simulation.table["k_mean"].min()) > 0.0


def test_simulation_k_mean_grows_with_frame_width(simulation):
    by_cell = simulation.table.groupby("cell")["k_mean"].mean()
    assert by_cell["w96_sparse"] > by_cell["w48_sparse"]


def test_ns_per_op_is_a_positive_finite_nanosecond_cost(ns_per_op):
    assert np.isfinite(ns_per_op)
    assert ns_per_op > 0.0


def test_ns_per_op_is_a_single_global_total_matching_scalar(simulation, ns_per_op):
    total_ns = float(simulation.table["sum_ns"].sum())
    total_ops = float(
        (simulation.table["declared_ops"] * simulation.table["n_frames"]).sum()
    )
    assert ns_per_op == pytest.approx(total_ns / total_ops, rel=1e-12, abs=0.0)


def test_ns_per_op_makes_the_whole_pipeline_total_come_out_right(simulation, ns_per_op):
    total_ns = float(simulation.table["sum_ns"].sum())
    total_ops = float(
        (simulation.table["declared_ops"] * simulation.table["n_frames"]).sum()
    )
    assert total_ops * ns_per_op == pytest.approx(total_ns, rel=1e-9, abs=0.0)


def test_simulation_uses_the_board_frame_stack_by_default(simulation, campaign):
    assert simulation.frames == campaign.meta["n_frames_per_stack"] == 1500
    assert simulation.stack_matches_campaign is True
    assert simulation.notes == ()


def test_short_stack_is_flagged_as_a_mismatch(short_stack_simulation):
    assert short_stack_simulation.stack_matches_campaign is False
    assert len(short_stack_simulation.notes) == 1
    assert "FRAME-STACK MISMATCH" in short_stack_simulation.notes[0]


def test_short_stack_note_names_both_stack_lengths(short_stack_simulation):
    note = short_stack_simulation.notes[0]
    assert "150" in note
    assert "1500" in note


def test_measure_simulation_refuses_a_zero_pass_count(campaign):
    with pytest.raises(ValueError, match="passes must be at least 1"):
        measure_simulation(campaign, passes=0, frames=10)


def test_measure_simulation_refuses_a_zero_frame_count(campaign):
    with pytest.raises(ValueError, match="frames must be at least 1"):
        measure_simulation(campaign, passes=1, frames=0)


def test_measure_simulation_refuses_an_unknown_cell(campaign):
    with pytest.raises(ValueError, match="no cells selected"):
        measure_simulation(campaign, passes=1, frames=10, cells=("w7_imaginary",))


def test_measure_simulation_uses_the_campaign_synthesis_seed(campaign):
    # Both modes must time the same pixels. The frame stack is regenerated from
    # runs.csv's own synth_seed, so a run with the same seed is deterministic.
    first = measure_simulation(campaign, passes=1, frames=40, cells=("w48_sparse",))
    second = measure_simulation(campaign, passes=1, frames=40, cells=("w48_sparse",))
    assert first.table["k_mean"].tolist() == second.table["k_mean"].tolist()


def test_calibrate_ns_per_op_refuses_a_table_with_no_declared_operations(simulation):
    import copy

    broken = copy.copy(simulation)
    table = simulation.table.copy()
    table["declared_ops"] = 0.0
    object.__setattr__(broken, "table", table)
    with pytest.raises(ValueError, match="non-positive"):
        calibrate_ns_per_op(broken)
