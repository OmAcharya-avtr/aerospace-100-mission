"""SPEC §9: every one of the five gate branches fires at least once.

P093 shipped a gate two of whose five reject reasons never fired in 268425
inferences, and that is recorded in this mission as a weak result: an acceptance
figure from a gate whose rejection paths were never exercised is a figure about
a narrower gate than the one in the source. SPEC §9 therefore requires a test
asserting reachability here, and this script measures it per cell as well as in
aggregate.

The counting rule is stated because it is not the obvious one:
``staged_exec.run_timed`` counts ``low_score``, ``area_out`` and ``border`` PER
CANDIDATE and ``accept`` and ``no_candidate`` PER FRAME. The five counters are
therefore not a partition of the frames and are not reported as one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import gate_census, load_campaign, measure_simulation  # noqa: E402
from hiltimingdiff.sources import reject_reasons  # noqa: E402
from hiltimingdiff.traces import GATE_BRANCHES  # noqa: E402

#: The five branches SPEC §9 requires, as literals independent of the package.
REQUIRED_BRANCHES = ("accept", "no_candidate", "low_score", "area_out", "border")

#: Pinned campaign totals.
PINNED_TOTALS = {
    "accept": 174420,
    "no_candidate": 8460,
    "low_score": 4567860,
    "area_out": 158520,
    "border": 33840,
}
PINNED_FRAMES = 810000


def main() -> int:
    tee = _bootstrap.Tee("validate_gate_reachability_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- gate branch reachability, SPEC §9")

    campaign = load_campaign()
    census = gate_census(campaign)

    tee.rule("the branch set, taken from the board source and checked against the trace")
    tee(f"  reject reasons declared in data/pat_pipeline.py : {', '.join(reject_reasons())}")
    tee(f"  branches recorded in data/traces/runs.csv       : {', '.join(GATE_BRANCHES)}")
    tee("")
    tee.check(
        "the trace records accept plus every reject reason the source declares",
        set(reject_reasons()) | {"accept"} == set(GATE_BRANCHES),
    )
    tee.check(
        "the branch set is the five pinned names",
        GATE_BRANCHES == REQUIRED_BRANCHES,
    )

    tee.rule("per-cell counts over the whole campaign")
    tee(census.frame().to_string(index=False))

    tee.rule("THE SPEC §9 REQUIREMENT")
    tee(f"  every branch fired at least once : {census.all_branches_fired}")
    tee(f"  silent branches                  : "
        f"{', '.join(census.silent_branches) if census.silent_branches else 'none'}")
    tee(
        f"  branches firing in EVERY cell    : "
        f"{', '.join(census.branches_firing_in_every_cell)}"
    )
    tee("")
    tee.check(
        "SPEC §9: every one of the five gate branches fires at least once",
        census.all_branches_fired,
    )
    for branch in REQUIRED_BRANCHES:
        tee.check(
            f"branch {branch} fired at least once",
            census.totals[branch] >= 1,
            f"{census.totals[branch]} times",
        )
    for branch in REQUIRED_BRANCHES:
        silent_cells = census.cells_where_silent(branch)
        tee.check(
            f"branch {branch} fired in every one of the nine cells",
            silent_cells == (),
            "stronger than SPEC §9 asks"
            if not silent_cells
            else f"silent in {', '.join(silent_cells)}",
        )
    tee.finding(
        "SPEC §9 IS SATISFIED and the stronger property holds too: all five branches "
        "fire in all nine cells, over 810000 timed frames. P093's gate left two of "
        "five reject reasons unexercised in 268425 inferences; this gate's thresholds "
        "(GATE_MIN_SCORE 1.35, GATE_MIN_AREA 2, GATE_MAX_AREA 160, GATE_BORDER 2) were "
        "set so every branch is reachable, and the reachability is measured rather "
        "than assumed."
    )

    tee.rule("the counts, and what they are and are not")
    tee(f"  total timed frames              : {census.n_frames_total}")
    tee(f"  per-frame acceptance rate       : {census.acceptance_rate:.9f}")
    tee(
        f"  accept + no_candidate           : "
        f"{census.totals['accept'] + census.totals['no_candidate']} "
        f"of {census.n_frames_total} frames (both are PER-FRAME and exclusive)"
    )
    tee(
        f"  low_score + area_out + border   : "
        f"{sum(census.totals[name] for name in ('low_score', 'area_out', 'border'))} "
        "(PER-CANDIDATE, so this is not a frame count)"
    )
    tee("")
    for branch in sorted(PINNED_TOTALS):
        tee.check(
            f"{branch} total matches the pinned value",
            census.totals[branch] == PINNED_TOTALS[branch],
            f"pinned {PINNED_TOTALS[branch]}, measured {census.totals[branch]}",
        )
    tee.check(
        "the frame total matches the pinned value",
        census.n_frames_total == PINNED_FRAMES,
    )
    tee.check(
        "accept and no_candidate together do not exceed the frame count",
        census.totals["accept"] + census.totals["no_candidate"] <= census.n_frames_total,
    )
    tee.check(
        "low_score exceeds the frame count, confirming it is a per-candidate counter",
        census.totals["low_score"] > census.n_frames_total,
        f"{census.totals['low_score']} against {census.n_frames_total} frames",
    )
    tee.finding(
        f"The per-frame acceptance rate is {census.acceptance_rate:.6f}, i.e. "
        f"{census.totals['accept']} of {census.n_frames_total} frames produced an "
        "accepted track measurement. The great majority of candidate-level rejections "
        f"are low_score ({census.totals['low_score']}), which is a property of this "
        "synthetic frame population's noise floor rather than of the gate: the "
        "threshold is a peak-to-median ratio of 1.35 and most thresholded pixel groups "
        "in a noise-dominated frame do not reach it."
    )

    tee.rule("simulation mode fires every branch too")
    simulation = measure_simulation(campaign, passes=1)
    tee("  " + ", ".join(f"{name}={count}" for name, count in sorted(simulation.census.items())))
    tee("")
    tee.check(
        "every gate branch fires in simulation mode on this host as well",
        all(count >= 1 for count in simulation.census.values()),
        "a branch firing on the board and not here would mean the two modes exercise "
        "different paths of the same file, which would invalidate the comparison",
    )

    tee.rule("negative controls")
    # Anchored on the TRUE census. A census with one branch zeroed must be
    # detected as silent; the real census must not be.
    import pandas as pd

    from hiltimingdiff.gate import GateCensus

    zeroed = census.per_cell.copy()
    zeroed["border"] = 0
    broken = GateCensus(
        branches=census.branches,
        per_cell=zeroed,
        totals={**census.totals, "border": 0},
        n_frames_total=census.n_frames_total,
        cells=census.cells,
    )
    tee.control(
        "a census with border zeroed is detected as having a silent branch",
        not broken.all_branches_fired and broken.silent_branches == ("border",),
        must_fire=True,
        detail=f"silent branches {broken.silent_branches}",
    )
    tee.control(
        "the real census is detected as having a silent branch",
        not census.all_branches_fired,
        must_fire=False,
        detail=f"silent branches {census.silent_branches}",
    )
    single_cell = GateCensus(
        branches=census.branches,
        per_cell=pd.concat(
            [
                census.per_cell.iloc[:1].assign(border=0),
                census.per_cell.iloc[1:],
            ],
            ignore_index=True,
        ),
        totals=census.totals,
        n_frames_total=census.n_frames_total,
        cells=census.cells,
    )
    tee.control(
        "a branch silent in ONE cell is detected per cell even when the total is nonzero",
        single_cell.cells_where_silent("border") == (census.cells[0],),
        must_fire=True,
        detail=f"silent in {single_cell.cells_where_silent('border')} while the "
        "campaign total is unchanged -- which is the P093 failure mode, a branch that "
        "fires somewhere and not everywhere",
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
