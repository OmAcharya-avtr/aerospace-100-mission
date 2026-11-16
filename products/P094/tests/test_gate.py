"""Gate branch reachability, SPEC §9, asserted on the committed campaign.

P093 shipped a gate two of whose five reject reasons never fired in 268425
inferences, and that is recorded in this mission as a weak result. SPEC §9
requires a test asserting that every branch fires here. It does, in every cell,
and that is asserted both in aggregate and per cell.
"""

from __future__ import annotations

import pytest

from hiltimingdiff.gate import gate_census
from hiltimingdiff.traces import GATE_BRANCHES, REQUIRED_CELLS

#: The reachability requirement of SPEC §9, as a literal.
REQUIRED_BRANCHES = ("accept", "no_candidate", "low_score", "area_out", "border")


def test_required_branches_match_the_trace_constant():
    assert REQUIRED_BRANCHES == GATE_BRANCHES


def test_census_covers_the_five_branches(census):
    assert census.branches == REQUIRED_BRANCHES


def test_census_covers_the_nine_cells(census):
    assert set(census.cells) == set(REQUIRED_CELLS)


def test_spec_section_9_every_branch_fires_at_least_once(census):
    # THE SPEC §9 REQUIREMENT. Asserted on the committed campaign.
    assert census.all_branches_fired is True
    assert census.silent_branches == ()


@pytest.mark.parametrize("branch", REQUIRED_BRANCHES)
def test_branch_fired_at_least_once_across_the_campaign(census, branch):
    assert census.totals[branch] >= 1


@pytest.mark.parametrize("branch", REQUIRED_BRANCHES)
def test_branch_fired_in_every_cell(census, branch):
    # Stronger than SPEC §9 asks, and it holds here: unlike P093, every branch
    # fires in every one of the nine cells.
    assert census.cells_where_silent(branch) == ()


def test_all_five_branches_fire_in_every_cell(census):
    assert census.branches_firing_in_every_cell == REQUIRED_BRANCHES


def test_cells_where_silent_refuses_an_unknown_branch(census):
    with pytest.raises(KeyError, match="is not one of"):
        census.cells_where_silent("maybe")


def test_total_frames_is_eight_hundred_ten_thousand(census):
    # Nine cells x 90000 timed frames.
    assert census.n_frames_total == 810000


@pytest.mark.parametrize("cell", REQUIRED_CELLS)
def test_every_cell_timed_ninety_thousand_frames(census, cell):
    row = census.per_cell.set_index("cell").loc[cell]
    assert int(row["n_frames"]) == 90000


def test_acceptance_rate_is_accept_over_frames(census):
    assert census.acceptance_rate == pytest.approx(
        census.totals["accept"] / census.n_frames_total, rel=1e-12, abs=0.0
    )


def test_acceptance_rate_lies_strictly_inside_the_unit_interval(census):
    assert 0.0 < census.acceptance_rate < 1.0


def test_rejection_counts_exclude_accept(census):
    assert "accept" not in census.rejection_counts
    assert len(census.rejection_counts) == 4


def test_candidate_level_counts_exceed_the_frame_count(census):
    # low_score is counted PER CANDIDATE, so its total may exceed the number of
    # frames. This is why the five counters are not a partition of the frames
    # and are not reported as one.
    assert census.totals["low_score"] > census.n_frames_total


def test_accept_plus_no_candidate_does_not_exceed_the_frame_count(census):
    # These two ARE per-frame and mutually exclusive.
    assert census.totals["accept"] + census.totals["no_candidate"] <= census.n_frames_total


def test_frame_appends_a_totals_row(census):
    frame = census.frame()
    assert list(frame["cell"])[-1] == "TOTAL"
    assert len(frame) == 10


def test_totals_row_sums_the_per_cell_counts(census):
    frame = census.frame().set_index("cell")
    for branch in REQUIRED_BRANCHES:
        assert int(frame.loc["TOTAL", branch]) == int(
            frame.drop(index="TOTAL")[branch].sum()
        )


def test_census_refuses_a_runs_table_missing_a_branch_column(campaign):
    class Shim:
        pass

    shim = Shim()
    shim.runs = campaign.runs.drop(columns=["border"])
    shim.label = campaign.label
    with pytest.raises(ValueError, match="no column for gate branch"):
        gate_census(shim)


def test_dense_cells_accept_more_often_than_their_sparse_counterparts(census):
    frame = census.per_cell.set_index("cell")
    for width in (48, 64, 96):
        dense = int(frame.loc[f"w{width}_dense", "accept"])
        sparse = int(frame.loc[f"w{width}_sparse", "accept"])
        assert dense > sparse


def test_the_three_reference_repeats_accept_within_a_factor_of_two(census):
    frame = census.per_cell.set_index("cell")
    values = [
        int(frame.loc[name, "accept"])
        for name in ("w64_sparse", "w64_sparse_rep1", "w64_sparse_rep2", "w64_sparse_rep3")
    ]
    assert max(values) / min(values) < 2.0
