"""The loader's contract: schema, validation, and the measurement-model identities.

INPUT ORIGIN: the committed SYNTHETIC generator for the identity checks, plus
the MEASURED campaign for the same checks when its traces are on disk. Every
line below says which it is reporting. The synthetic figures measure no board.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import tempfile  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

import numpy as np  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from deadlinerisk.accounting import account, miss_indicator  # noqa: E402
from deadlinerisk.schema import (  # noqa: E402
    JOB_COLUMNS,
    MONITOR_COLUMNS,
    SchemaError,
    validate_job_header,
)
from deadlinerisk.trace import TraceDataAbsentError, load_job_trace  # noqa: E402


def _identity_section(out, campaign, label: str) -> None:
    out.rule(f"identities on the {label} campaign")
    out(campaign.provenance.banner())
    out(f"  period T  : {campaign.period_s * 1e3:.3f} ms")
    out(f"  deadline D: {campaign.deadline_s * 1e3:.3f} ms")
    out("")
    out(
        f"  {'run':<12}{'rows':>8}{'drops':>8}{'releases':>10}{'misses':>8}"
        f"{'rate':>13}{'exec_rate':>13}"
    )
    for run in sorted(campaign.jobs):
        jobs = campaign.jobs[run]
        totals = account(jobs)
        out(
            f"  {run:<12}{totals.executed:>8}{totals.dropped:>8}"
            f"{totals.releases:>10}{totals.misses:>8}"
            f"{totals.miss_rate:>13.6e}{totals.executed_miss_rate:>13.6e}"
        )
    out("")
    for run in sorted(campaign.jobs):
        jobs = campaign.jobs[run]
        totals = account(jobs)
        indicator = miss_indicator(jobs)
        out.check(
            f"{label}/{run}: releases = sum(1 + dropped_before)",
            totals.releases == int((1 + jobs.dropped_before).sum()),
            f"{totals.releases}",
        )
        out.check(
            f"{label}/{run}: misses = sum(missed + dropped_before)",
            totals.misses == int((jobs.missed + jobs.dropped_before).sum()),
            f"{totals.misses}",
        )
        out.check(
            f"{label}/{run}: indicator length equals the release count",
            indicator.size == totals.releases,
            f"{indicator.size} vs {totals.releases}",
        )
        out.check(
            f"{label}/{run}: indicator sum equals the miss count",
            int(indicator.sum()) == totals.misses,
            f"{int(indicator.sum())} vs {totals.misses}",
        )
        implied = jobs.end_s - jobs.release_s
        worst = float(np.max(np.abs(implied - jobs.response_s))) if len(jobs) else 0.0
        out.check(
            f"{label}/{run}: response_s agrees with end_s - release_s",
            worst <= 2e-6,
            f"worst |difference| {worst:.3e} s against the CSV's own 1 us rounding "
            "of end_s",
        )
        # The miss flag is a strict threshold comparison, so rows whose response
        # sits within the CSV's rounding of the deadline are excluded rather
        # than asserted: the boolean of a rounded quantity is the one thing
        # floating point cannot promise.
        resolved = np.abs(jobs.response_s - jobs.deadline_s) > 2e-6
        recomputed = (jobs.response_s[resolved] > jobs.deadline_s).astype(np.int64)
        out.check(
            f"{label}/{run}: missed flag agrees with response > D away from the "
            "rounding of D",
            bool(np.array_equal(recomputed, jobs.missed[resolved])),
            f"{int(resolved.size - resolved.sum())} row(s) excluded as unresolved",
        )


def main() -> int:
    out = _bootstrap.Tee("validate_loader_contract_output.txt")
    out.rule("LOADER CONTRACT -- schema, validation and the measurement model")
    out("INPUT ORIGIN: SYNTHETIC generator, plus the MEASURED campaign when present.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)
    out("")
    out(_bootstrap.MEASUREMENT_MODEL)

    out.rule("1  declared schema")
    out(f"  jobs columns    : {list(JOB_COLUMNS)}")
    out(f"  monitor columns : {len(MONITOR_COLUMNS)} columns, "
        f"{MONITOR_COLUMNS[0]} .. {MONITOR_COLUMNS[-1]}")
    out.check("the jobs schema has eight columns", len(JOB_COLUMNS) == 8, "8")
    # Eight aggregate columns, six per-core clocks, and four derived columns.
    out.check(
        "the monitor schema has eighteen columns",
        len(MONITOR_COLUMNS) == 18,
        f"{len(MONITOR_COLUMNS)}",
    )

    out.rule("2  rejections -- every one exercised, with the message it produced")
    probes: list[tuple[str, str, str]] = [
        ("wrong header", "a,b,c\n1,2,3\n", "SchemaError"),
        (
            "reordered header",
            ",".join([JOB_COLUMNS[1], JOB_COLUMNS[0], *JOB_COLUMNS[2:]]) + "\n",
            "SchemaError",
        ),
        ("empty file", "", "TraceDataAbsentError"),
        (
            "short row",
            ",".join(JOB_COLUMNS) + "\n0,0.0,0.0,0.004\n",
            "ValueError",
        ),
        (
            "non-increasing job_index",
            ",".join(JOB_COLUMNS)
            + "\n0,0.0,0.0,0.004,0.004,0.004,0,0\n0,0.010,0.0,0.014,0.004,0.004,0,0\n",
            "ValueError",
        ),
        (
            "missed flag of 2",
            ",".join(JOB_COLUMNS) + "\n0,0.0,0.0,0.004,0.004,0.004,2,0\n",
            "ValueError",
        ),
        (
            "negative dropped_before",
            ",".join(JOB_COLUMNS) + "\n0,0.0,0.0,0.004,0.004,0.004,0,-1\n",
            "ValueError",
        ),
    ]
    scratch = _Path(tempfile.mkdtemp(prefix="deadlinerisk-probe-"))
    for name, body, expected in probes:
        path = scratch / "jobs_probe.csv"
        path.write_text(body)
        raised = "nothing"
        message = ""
        try:
            load_job_trace(path, "probe", 0.010, 0.010)
        except (SchemaError, TraceDataAbsentError, ValueError) as error:
            raised = type(error).__name__
            message = str(error).splitlines()[0][:110]
        out.check(f"{name} raises {expected}", raised == expected, f"got {raised}")
        if message:
            out(f"        message: {message}")
    # The error text must quote a RELATIVE path, because an absolute path in
    # committed output is a release-gate failure.
    out.check(
        "the probe directory is outside the repository, so no probe path is reported",
        not str(scratch).startswith(str(ROOT)),
        "probe files are written to a temporary directory and their names are not printed",
    )
    for bad_period, bad_deadline, what in (
        (0.0, 0.010, "zero period"),
        (0.010, 0.020, "deadline beyond the period"),
    ):
        path = scratch / "jobs_probe.csv"
        path.write_text(",".join(JOB_COLUMNS) + "\n0,0.0,0.0,0.004,0.004,0.004,0,0\n")
        try:
            load_job_trace(path, "probe", bad_period, bad_deadline)
            raised = False
        except ValueError:
            raised = True
        out.check(f"{what} is rejected", raised)
    try:
        validate_job_header(list(JOB_COLUMNS))
        header_ok = True
    except SchemaError:
        header_ok = False
    out.check("the declared header is accepted", header_ok)

    out.rule("3  SYNTHETIC campaign identities")
    synthetic = _bootstrap.synthetic_campaign()
    _identity_section(out, synthetic, "SYNTHETIC")

    out.rule("4  MEASURED campaign identities")
    try:
        measured = _bootstrap.require_measured_campaign()
    except TraceDataAbsentError as error:
        out("  MEASURED CAMPAIGN NOT AVAILABLE. The refusal text follows verbatim:")
        for line in str(error).splitlines():
            out(f"    {line}")
        out.unmeasurable_check(
            "measurement-model identities on the MEASURED campaign",
            "the measured traces were not on disk, or the campaign had not written "
            "CAMPAIGN_COMPLETE, when this script ran. This is not a pass: nothing "
            "was checked and nothing was substituted.",
        )
    else:
        _identity_section(out, measured, "MEASURED")

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
