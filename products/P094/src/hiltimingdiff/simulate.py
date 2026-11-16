"""Simulation mode: the committed pipeline, timed here, by the committed instrument.

This is the half of SPEC §1 that is measured in this container rather than read
from a trace. ``data/pat_pipeline.py`` is imported, not re-implemented, and
``data/staged_exec.run_timed`` is the function that brackets its six stages,
exactly as it did on the board. Both are md5-verified before the first frame is
timed (``sources.verify_source_identity``), so a simulation-mode number is a
number about the same instructions the hardware-mode number is about.

The frame stacks are regenerated from the campaign's own per-cell
``synth_seed`` in ``runs.csv``, through the committed
``staged_exec.make_frames``. The board and the host therefore time the same
pixels, which removes frame content as an explanation of a per-stage
difference.

What differs from the board run, stated rather than implied
-----------------------------------------------------------

* **Pass count.** The board ran 60 passes of 1500 frames per cell, 734 s in
  total. The default here is 12 passes, which is 18000 timed frames per cell
  and runs in well under the mission's three-minute compute budget on two
  cores. The pass count is a parameter and is recorded in the result. It is the
  one dimension that was measured to be safe to shorten: at 1, 2, 4 and 12
  passes over all nine cells, baseline 2's held-out transfer error measured
  0.1601, 0.1554, 0.1622 and 0.1599.

* **Frame-stack length, which is NOT safe to shorten.** ``frames`` defaults to
  the campaign's own ``n_frames_per_stack`` and a shorter stack is recorded on
  the result as a mismatch. Measured on 2026-11-16: at the board's 1500-frame
  stack baseline 2's held-out transfer error is 0.1554 (two passes, nine
  cells); at a 300-frame stack it is 0.5386 and at 150 frames 0.5742, because
  the ``gate`` and ``track`` hardware-to-simulation ratios fall from 3.58 and
  4.78 to 2.49 and 1.86. The exponential running background has not converged
  over a short stack, so the candidate population and therefore the gate's
  branch mix differ, and the two stages whose cost depends on that mix stop
  being comparable between modes. A short-stack figure is a figure about a
  different frame stack, and SPEC §7's secondary expectation lands on the
  OTHER SIDE of its 0.5 threshold there. Both are reported in
  validation/VALIDATION.md; the declared configuration is the board's own
  stack length, because the comparison is only paired if both modes time the
  same stack.
* **Machine.** x86_64 here, aarch64 there. That is the comparison, not a
  caveat.
* **Wall-clock variability.** A simulation-mode figure is measured afresh on
  every run and moves between runs on the same host. The figures that depend on
  it are labelled; the divergence spread, the rank statistic, the permutation
  and both arg-extrema do not depend on it at all.

Units: nanoseconds, from ``time.perf_counter_ns``, with no conversion applied.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .sources import declared_ops, executor_module, stage_names, verify_source_identity
from .traces import Campaign

#: Passes per cell in simulation mode. 12 x 1500 = 18000 timed frames per cell,
#: nine cells, inside the three-minute compute budget on two cores. The board's
#: 60 passes are not matched here: a stable per-cell median is what the
#: comparison needs and 12 passes deliver one, while 60 would cost five times
#: the wall clock for no change in the reported statistic.
DEFAULT_PASSES = 12

#: Frames per pass. Matches the board's ``n_frames_per_stack``, and must: see
#: the module docstring. ``measure_simulation`` defaults to the loaded
#: campaign's own value and flags a mismatch on the result.
DEFAULT_FRAMES = 1500


@dataclass(frozen=True)
class SimulationResult:
    """Simulation-mode per-pass summaries, measured on this host, now."""

    table: pd.DataFrame
    stages: tuple[str, ...]
    passes: int
    frames: int
    bracket_ns: float
    clock_resolution_ns: float
    census: dict[str, int]
    machine: str
    python: str
    numpy: str
    elapsed_s: float
    campaign_stack_frames: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def stack_matches_campaign(self) -> bool:
        """Whether this measurement used the board's own frame-stack length.

        ``False`` means the per-stage figures are about a different frame
        stack. It is not a failure and it is not silent: the flag is carried on
        the result, printed by the CLI and reported in the validation output,
        because baseline 2's held-out transfer error crosses SPEC §7's 0.5
        threshold between a 300-frame and a 1500-frame stack.
        """
        return self.frames == self.campaign_stack_frames

    @property
    def n_timed_frames(self) -> int:
        """Total frames timed across every cell and pass."""
        return int(self.table["n_frames"].sum() // len(self.stages))

    def stage_medians_ns(self) -> dict[str, float]:
        """Median of the per-pass medians, per stage, in nanoseconds."""
        grouped = self.table.groupby("stage")["median_ns"].median()
        return {stage: float(grouped[stage]) for stage in self.stages}

    def declared_ops_medians(self) -> dict[str, float]:
        """Median declared operation count per stage, dimensionless."""
        grouped = self.table.groupby("stage")["declared_ops"].median()
        return {stage: float(grouped[stage]) for stage in self.stages}


def _cell_plan(campaign: Campaign) -> list[dict[str, object]]:
    """Per-cell width, target count and synthesis seed, from ``runs.csv``."""
    runs = campaign.runs
    plan: list[dict[str, object]] = []
    for row in runs.itertuples():
        plan.append(
            {
                "cell": str(row.cell),
                "w": int(row.w),
                "density": str(row.density),
                "n_targets": int(row.n_targets),
                "synth_seed": int(row.synth_seed),
                "order_index": int(row.order_index),
            }
        )
    return plan


def measure_simulation(
    campaign: Campaign,
    *,
    passes: int = DEFAULT_PASSES,
    frames: int | None = None,
    cells: tuple[str, ...] | None = None,
    progress: bool = False,
) -> SimulationResult:
    """Time the committed pipeline in simulation mode on this host.

    Parameters
    ----------
    campaign:
        The loaded hardware campaign, read only for the per-cell widths, target
        counts and synthesis seeds, so that both modes time the same pixels.
    passes:
        Passes per cell. Each pass is one call of the committed
        ``staged_exec.run_timed`` over the whole frame stack.
    frames:
        Frames per stack. ``None``, the default, takes the campaign's own
        ``n_frames_per_stack``, which is the only value for which the two modes
        time the same stack. A shorter stack is permitted and is recorded as
        :attr:`SimulationResult.stack_matches_campaign` being ``False``.
    cells:
        Cell names to measure; defaults to every cell of the campaign.
    progress:
        Whether to write one line per cell to stderr. Library code prints
        nothing by default; the CLI and the examples pass ``True``.

    Returns
    -------
    SimulationResult
        Per-pass, per-stage summaries with the same column meanings as
        ``stages_pass.csv``, plus the declared operation count per row.
    """
    campaign_stack = int(campaign.meta["n_frames_per_stack"])
    if frames is None:
        frames = campaign_stack
    if passes < 1:
        raise ValueError(f"passes must be at least 1, got {passes}")
    if frames < 1:
        raise ValueError(f"frames must be at least 1, got {frames}")
    verify_source_identity()
    executor = executor_module()
    stages = stage_names()

    plan = [row for row in _cell_plan(campaign) if cells is None or row["cell"] in cells]
    if not plan:
        raise ValueError(
            "no cells selected for simulation mode; asked for "
            f"{cells!r} and the campaign has "
            f"{', '.join(sorted(str(r['cell']) for r in _cell_plan(campaign)))}"
        )

    bracket = float(executor.bracket_overhead_ns(4000))
    resolution = float(time.get_clock_info("perf_counter").resolution * 1e9)
    census_total = {"accept": 0, "no_candidate": 0, "low_score": 0, "area_out": 0, "border": 0}
    rows: list[dict[str, object]] = []
    started = time.monotonic()
    for entry in plan:
        width = int(entry["w"])
        stack = executor.make_frames(
            width, int(frames), int(entry["n_targets"]), int(entry["synth_seed"])
        )
        for pass_index in range(int(passes)):
            per_frame, census, k_per_frame = executor.run_timed(stack)
            for key in census_total:
                census_total[key] += int(census[key])
            k_mean = float(np.mean(k_per_frame))
            counts = declared_ops(width, k_mean)
            for column, stage in enumerate(stages):
                series = per_frame[:, column].astype(np.float64)
                rows.append(
                    {
                        "cell": entry["cell"],
                        "w": width,
                        "density": entry["density"],
                        "pass_idx": pass_index,
                        "stage": stage,
                        "n_frames": int(series.size),
                        "mean_ns": float(series.mean()),
                        "median_ns": float(np.median(series)),
                        "p05_ns": float(np.percentile(series, 5)),
                        "p95_ns": float(np.percentile(series, 95)),
                        "min_ns": float(series.min()),
                        "max_ns": float(series.max()),
                        "std_ns": float(series.std(ddof=1)) if series.size > 1 else 0.0,
                        "sum_ns": float(series.sum()),
                        "k_mean": k_mean,
                        "declared_ops": float(counts[stage]),
                    }
                )
        if progress:
            print(
                f"simulation mode: {entry['cell']} w={width} passes={passes}",
                file=sys.stderr,
            )
        del stack
    elapsed = time.monotonic() - started
    table = pd.DataFrame(rows)
    notes: list[str] = []
    if int(frames) != campaign_stack:
        notes.append(
            f"FRAME-STACK MISMATCH: measured over {frames} frames per pass "
            f"against the campaign's {campaign_stack}. The per-stage figures "
            "are about a different frame stack; baseline 2's held-out transfer "
            "error crosses SPEC §7's 0.5 threshold between a 300-frame and a "
            "1500-frame stack."
        )
    return SimulationResult(
        table=table,
        stages=stages,
        passes=int(passes),
        frames=int(frames),
        bracket_ns=bracket,
        clock_resolution_ns=resolution,
        census=census_total,
        machine=platform.machine(),
        python=sys.version.split()[0],
        numpy=np.__version__,
        elapsed_s=float(elapsed),
        campaign_stack_frames=campaign_stack,
        notes=tuple(notes),
    )


def calibrate_ns_per_op(result: SimulationResult) -> float:
    """The single global ``ns_per_op`` scalar of SPEC §3, calibrated once.

    One scalar, fitted on the simulation host and on nothing else:

    ``ns_per_op = sum over (stage, cell, pass) of measured ns
                  / sum over the same of declared operations``

    A total-matching scalar rather than a mean of per-stage ratios, because a
    schedulability analysis that reads an operation count off the source and
    scales it has exactly one number to scale it by, and the honest calibration
    of that one number is the one that makes the whole-pipeline total come out
    right. A per-stage scalar would be six numbers and would not be the object
    SPEC §4 baseline 1 names.

    Returns
    -------
    float
        Nanoseconds per declared operation, on this host, now.
    """
    total_ns = float(result.table["sum_ns"].sum())
    total_ops = float(
        (result.table["declared_ops"] * result.table["n_frames"]).sum()
    )
    if total_ops <= 0.0:
        raise ValueError("total declared operation count is non-positive")
    return total_ns / total_ops
