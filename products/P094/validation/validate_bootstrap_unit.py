"""The resampling unit is the PASS, and this measures what the wrong unit costs.

SPEC §4 declares the pass as baseline 3's resampling unit and says why:
adjacent frames share cache and governor state and are serially correlated, so
resampling frames would understate the interval. That is a claim about the data
and this script measures it rather than repeating it.

Three measurements
------------------
1. The declared unit, asserted on every interval the package returns.
2. The serial correlation of adjacent frames WITHIN a pass, from the committed
   subsampled per-frame table, with the subsampling stated. A lag-1
   autocorrelation near zero would mean SPEC §4's premise is wrong.
3. A frame-unit interval beside the declared pass-unit interval, so the
   understatement is a number and not an argument.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from hiltimingdiff import baseline_3, load_campaign, pass_units  # noqa: E402
from hiltimingdiff.bootstrap import (  # noqa: E402
    RESAMPLING_UNIT,
    bootstrap_statistic,
)

#: The cell the per-frame analysis uses. The reference cell of the campaign.
REFERENCE_CELL = "w64_sparse"

#: Pinned partition of the six stages by the DIRECTION in which the frame unit
#: moves the interval width on the reference cell. SPEC §4 predicts every stage
#: in the narrower set; two are not, and the partition is pinned as literals so
#: this script fails if a stage ever moves between the sets.
PINNED_NARROWER = frozenset({"background", "track"})
PINNED_WIDER = frozenset({"ingest", "threshold", "centroid", "gate"})


def main() -> int:
    tee = _bootstrap.Tee("validate_bootstrap_unit_output.txt")
    _bootstrap.header(tee, "P094 HilTimingDiff -- the declared resampling unit, SPEC §4")

    campaign = load_campaign()
    units = pass_units(campaign)

    tee.rule("1. the declared unit, asserted on the returned objects")
    third = baseline_3(units, resamples=_bootstrap.RESAMPLES_USED)
    tee(f"  package constant RESAMPLING_UNIT : {RESAMPLING_UNIT!r}")
    tee(f"  baseline 3 unit                  : {third.unit!r}")
    tee(f"  units                            : {third.n_units}")
    tee("")
    tee.check("the package's declared unit is the pass", RESAMPLING_UNIT == "pass")
    tee.check("baseline 3 records the pass as its unit", third.unit == "pass")
    tee.check(
        "every per-stage interval records the pass as its unit",
        all(third.interval[stage].unit == "pass" for stage in third.stages),
    )
    tee.check(
        "the unit count is 540 passes, not 810000 frames",
        third.n_units == 540,
    )

    tee.rule("2. is SPEC §4's premise true: are adjacent frames serially correlated")
    tee(
        f"  Measured on the committed per-frame table, which is SUBSAMPLED EVERY "
        f"{campaign.sample_every} FRAMES"
    )
    tee(
        "  (meta.json: raw_sample_every). A lag-1 autocorrelation on a 3-subsampled "
        "series is the"
    )
    tee(
        "  correlation at a true lag of 3 frames, so it UNDERSTATES the adjacent-frame "
        "correlation."
    )
    tee("")
    frames = campaign.stages_subsampled
    subset = frames.loc[frames["cell"] == REFERENCE_CELL]
    tee(f"  cell {REFERENCE_CELL}, {len(subset)} subsampled rows over 60 passes")
    tee("")
    tee("  stage        lag-1 autocorrelation (within-pass, 3-subsampled)")
    correlations: dict[str, float] = {}
    for stage in units.stages:
        column = f"ns_{stage}"
        values: list[float] = []
        for _, group in subset.groupby("pass_idx"):
            series = group[column].to_numpy(dtype=np.float64)
            if series.size < 3 or np.ptp(series) == 0.0:
                continue
            centred = series - series.mean()
            denominator = float(np.dot(centred, centred))
            if denominator == 0.0:
                continue
            values.append(float(np.dot(centred[:-1], centred[1:])) / denominator)
        correlations[stage] = float(np.mean(values)) if values else float("nan")
        tee(f"  {stage:<11s} {correlations[stage]:>+10.6f}   over {len(values)} passes")
    tee("")
    positive = [name for name, value in correlations.items() if value > 0.0]
    tee.check(
        "at least one stage shows a positive within-pass serial correlation",
        bool(positive),
        f"positive for {', '.join(positive) if positive else 'no stage'}",
    )
    tee.finding(
        "SPEC §4's premise is only PARTLY supported by this measurement, and the "
        "measurement is published as it is. Lag-1 autocorrelation within a pass, on "
        f"the 3-subsampled series, is positive for {len(positive)} of 6 stages "
        f"({', '.join(positive)}) and the largest is "
        f"{max(correlations.values()):+.6f} "
        f"({max(correlations, key=correlations.get)}). These are small numbers. Two "
        "reasons they understate the real serial dependence, both structural: the "
        "committed series is subsampled every third frame, so a lag-1 correlation here "
        "is a true lag of 3; and the quantity the bootstrap resamples is a PASS MEDIAN "
        "over 1500 frames, whose dependence on the governor and thermal state of that "
        "pass is not visible in within-pass frame-to-frame correlation at all. The "
        "pass unit is therefore defended by the between-pass structure measured in "
        "step 3 rather than by these autocorrelations."
    )

    tee.rule("3. what the wrong unit costs, measured")
    tee(
        "  The frame-unit interval below resamples the SUBSAMPLED per-frame rows of a "
        "cell as if"
    )
    tee(
        "  they were independent draws. The pass-unit interval resamples that cell's "
        "60 pass"
    )
    tee("  medians. Both estimate the same median; only the interval differs.")
    tee("")
    tee(
        "  stage        pass-unit width    frame-unit width   ratio   pass n   frame n"
    )
    pass_rows = campaign.stages_pass
    cell_passes = pass_rows.loc[pass_rows["cell"] == REFERENCE_CELL]
    ratios: list[float] = []
    for stage in units.stages:
        pass_values = cell_passes.loc[cell_passes["stage"] == stage, "median_ns"].to_numpy(
            dtype=np.float64
        )
        frame_values = subset[f"ns_{stage}"].to_numpy(dtype=np.float64)
        pass_interval = bootstrap_statistic(
            pass_values,
            lambda sample: float(np.median(sample)),
            resamples=_bootstrap.RESAMPLES_USED,
        )
        frame_interval = bootstrap_statistic(
            frame_values,
            lambda sample: float(np.median(sample)),
            resamples=_bootstrap.RESAMPLES_USED,
            unit="frame (NOT the declared unit)",
        )
        ratio = (
            frame_interval.width / pass_interval.width
            if pass_interval.width > 0.0
            else float("nan")
        )
        ratios.append(ratio)
        tee(
            f"  {stage:<11s} {pass_interval.width:>15.3f} "
            f"{frame_interval.width:>19.3f} {ratio:>7.3f} "
            f"{pass_interval.units:>8d} {frame_interval.units:>9d}"
        )
    finite = [value for value in ratios if np.isfinite(value)]
    tee("")
    tee(
        f"  frame-unit width as a fraction of pass-unit width: min "
        f"{min(finite):.4f}, median {float(np.median(finite)):.4f}, max {max(finite):.4f}"
    )
    tee("")
    narrower = [
        stage
        for stage, value in zip(units.stages, ratios, strict=True)
        if np.isfinite(value) and value < 1.0
    ]
    wider = [
        stage
        for stage, value in zip(units.stages, ratios, strict=True)
        if np.isfinite(value) and value > 1.0
    ]
    tee(f"  narrower under the frame unit : {', '.join(narrower)}")
    tee(f"  WIDER under the frame unit    : {', '.join(wider)}")
    tee("")
    tee.falsified(
        "SPEC §4's stated reason -- resampling frames would UNDERSTATE the interval",
        f"measured on {REFERENCE_CELL}, the frame-unit interval is narrower on "
        f"{len(narrower)} of 6 stages ({', '.join(narrower)}) and WIDER on "
        f"{len(wider)} ({', '.join(wider)}), by up to a factor of {max(finite):.2f}. "
        "The direction is stage-dependent, so the stated reason does not hold as a "
        "general claim about this campaign",
        set(narrower) == PINNED_NARROWER and set(wider) == PINNED_WIDER,
        f"pinned narrower {sorted(PINNED_NARROWER)}, pinned wider {sorted(PINNED_WIDER)}; "
        "this script exits 0 while the falsification is exactly this partition and "
        "non-zero the moment a stage moves between the two sets",
    )
    tee.finding(
        "SPEC §4's REASON IS FALSIFIED WHILE ITS DECISION STANDS, and both halves are "
        "published. The specification says a frame-unit bootstrap would understate the "
        f"interval. Measured on {REFERENCE_CELL}, the frame-unit width is "
        f"{min(finite):.3f} to {max(finite):.3f} times the pass-unit width, median "
        f"{float(np.median(finite)):.3f}: narrower on {', '.join(narrower)} and WIDER "
        f"on {', '.join(wider)}. The mechanism for the wider cases is that the "
        "per-frame distribution of threshold and centroid is heavily right-tailed, "
        "because both stages cost in proportion to a candidate count that varies "
        "frame to frame, so a median over 30000 raw frames is less stable than a "
        "median over 60 pass medians. THE DECISION TO USE THE PASS IS STILL RIGHT, for "
        "a reason the specification does not give: the pass is the unit over which the "
        "governor and thermal state are approximately constant, and the quantity a "
        "schedulability budget consumes is a typical PASS's per-stage time, not a "
        "typical frame's. A frame-unit interval answers a different question. The "
        "specification's justification is wrong about this campaign and its choice of "
        "unit is not."
    )

    tee.rule("negative controls")
    # Anchored on TRUE values. An artificially block-correlated series must show
    # a frame-unit interval far narrower than its pass-unit one; an i.i.d.
    # series must not.
    generator = np.random.default_rng(4040)
    block_means = generator.normal(100.0, 10.0, size=60)
    blocked = np.repeat(block_means, 500) + generator.normal(0.0, 0.1, size=30000)
    blocked_frame = bootstrap_statistic(
        blocked, lambda sample: float(np.median(sample)), resamples=400
    )
    blocked_pass = bootstrap_statistic(
        block_means, lambda sample: float(np.median(sample)), resamples=400
    )
    tee.control(
        "a strongly block-correlated series gives a much narrower frame-unit interval",
        blocked_frame.width < 0.5 * blocked_pass.width,
        must_fire=True,
        detail=f"frame-unit width {blocked_frame.width:.4f} against pass-unit "
        f"{blocked_pass.width:.4f}",
    )
    independent = generator.normal(100.0, 10.0, size=30000)
    independent_frame = bootstrap_statistic(
        independent, lambda sample: float(np.median(sample)), resamples=400
    )
    independent_pass = bootstrap_statistic(
        independent[:60], lambda sample: float(np.median(sample)), resamples=400
    )
    tee.control(
        "an i.i.d. series gives a frame-unit interval WIDER than its 60-sample interval",
        independent_frame.width > independent_pass.width,
        must_fire=False,
        detail=f"frame-unit width {independent_frame.width:.4f} against 60-sample "
        f"{independent_pass.width:.4f}; more independent draws legitimately narrow an "
        "interval, which is exactly why claiming them when they are not independent is "
        "the defect",
    )

    path = tee.save()
    print(f"\nwritten: {path}")
    return 1 if tee.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
