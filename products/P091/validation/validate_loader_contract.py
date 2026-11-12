"""The loader contract: schema, the missing sentinel, the inferred scale, origin.

Input: SYNTHETIC fixtures and hand-built rows, plus the five real rows quoted in
``docs/SPEC.md``. Every number here is a property of the CODE, not a measurement
of the board; the five real rows are real data but five rows are a shape
contract, not a thermal measurement.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

import numpy as np  # noqa: E402

from thermalbudget import schema  # noqa: E402
from thermalbudget.provenance import (  # noqa: E402
    NotMeasuredError,
    Origin,
    meet,
    require_measured,
)
from thermalbudget.trace import TraceDataAbsentError, load_campaign, load_trace  # noqa: E402

#: The five real rows of the completed cold-burst run, as quoted in docs/SPEC.md.
REAL_HEADER = ",".join(schema.expected_columns())
REAL_ROWS = (
    "1794457189.964,1.001,0,19447.34,0.7512,0.38,20.10,45875,46000,45812,45125,44531,"
    "45875,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457191.0,2.051,32760,31200.88,1.1942,0.40,20.10,46343,46400,46312,45625,45031,"
    "46343,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457195.2,5.191,100120,25168.28,0.9620,0.52,20.10,46906,47000,46812,46125,45531,"
    "46906,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457204.6,14.601,340210,29038.24,1.1222,0.80,20.00,47781,47900,47712,47025,46431,"
    "47781,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
    "1794457210.9,20.879,510880,27503.54,1.0825,0.95,19.80,48687,48800,48712,48025,47431,"
    "48687,1728000,1728000,1728000,1728000,1728000,1728000,306000000",
)


def main() -> int:
    out = _bootstrap.Tee("validate_loader_contract_output.txt")
    out.rule("LOADER CONTRACT -- schema, sentinel, inferred scale, carried origin")
    out("INPUT ORIGIN: SYNTHETIC fixtures and hand-built rows, plus the five real rows")
    out("quoted verbatim in docs/SPEC.md. Five rows are a SHAPE contract, not a thermal")
    out("measurement, and nothing here is a figure about the board.")
    out("")
    for line in _bootstrap.environment_lines():
        out(line)

    scratch = Path(tempfile.mkdtemp(prefix="thermalbudget-loader-"))

    out.rule("1  the binding header")
    columns = schema.expected_columns()
    out.check(
        "the schema header is 20 columns wide",
        len(columns) == 20,
        f"{len(columns)} columns: 7 fixed + 6 zones + 6 cores + gpu_hz",
    )
    out.check(
        "the die-temperature channel is temp_tj-thermal",
        schema.DIE_COLUMN == "temp_tj-thermal",
        schema.DIE_COLUMN,
    )
    out.check(
        "the three unreadable zones are absent from the header",
        all(f"temp_{zone}" not in columns for zone in schema.UNREADABLE_ZONES),
        "cv0/cv1/cv2-thermal return no bytes on the measured board (declared limit 4)",
    )
    fired = False
    try:
        schema.validate_header((*columns, "temp_cv0-thermal"))
    except schema.TraceSchemaError as error:
        fired = "cv0-thermal" in str(error)
    out.control("a present cv0-thermal column is rejected by name", fired, must_fire=True)

    out.rule("2  the five real rows load and convert")
    real = scratch / "burst_1.00_a.csv"
    real.write_text(REAL_HEADER + "\n" + "\n".join(REAL_ROWS) + "\n")
    trace = load_trace(real, origin=Origin.MEASURED)
    out.check("five rows loaded", trace.n_rows == 5, f"n_rows={trace.n_rows}")
    out.check(
        "the temperature scale was inferred as milli-degrees",
        trace.temperature_divisor == 1000.0,
        f"divisor={trace.temperature_divisor:g}",
    )
    out.check(
        "45875 milli-degC converts to 45.875 degC",
        abs(float(trace.die_temperature_c[0]) - 45.875) < 1e-12,
        f"first die sample {float(trace.die_temperature_c[0]):.6f} degC",
    )
    out.check(
        "all six cores are at cpuinfo_max_freq throughout these five rows",
        float(np.nanmin(trace.mean_cpu_khz)) == 1728000.0,
        "1728000 kHz on every core of all FIVE of these rows. The claim for the whole "
        "run is a separate measurement and is in validate_measured_campaign_output.txt",
    )
    out(
        "    NOTE: the first row carries iters_cum=0 with iters_per_s=19447.34, because "
        "the six workers flush their counters into shared memory every 0.25 s and the "
        "first one-second sample is taken while they are still starting. That is why "
        "baseline 1 drops the first 2 s of every run."
    )

    out.rule("3  the missing sentinel is not a value")
    rows = list(REAL_ROWS)
    rows[2] = rows[2].replace(",46906,1728000", ",-1,1728000")
    sentinel_path = scratch / "soak_1.00_r1.csv"
    sentinel_path.write_text(REAL_HEADER + "\n" + "\n".join(rows) + "\n")
    with_sentinel = load_trace(sentinel_path, origin=Origin.MEASURED)
    out.check(
        "a -1 die reading becomes nan",
        math.isnan(float(with_sentinel.die_temperature_c[2])),
        "not -0.001 degC, which is what dividing the sentinel by 1000 would give",
    )
    out.check(
        "the missing count is carried on the trace",
        with_sentinel.missing[schema.DIE_COLUMN] == 1,
        f"missing[{schema.DIE_COLUMN}]={with_sentinel.missing[schema.DIE_COLUMN]}",
    )
    honest = float(with_sentinel.finite(schema.DIE_COLUMN).mean())
    naive_values = np.array(
        [float(row.split(",")[12]) / 1000.0 for row in rows], dtype=float
    )
    naive = float(naive_values.mean())
    out(
        f"    the mean over finite samples is {honest:.6f} degC; treating the sentinel as "
        f"a value gives {naive:.6f} degC, an error of {honest - naive:+.6f} degC on five "
        "rows. On a 120-sample window one sentinel would shift the mean by about "
        f"{(honest + 0.001) / 120:.4f} degC and would bias a fitted slope."
    )
    out.control(
        "treating the sentinel as a value changes the mean",
        abs(honest - naive) > 1.0,
        must_fire=True,
        detail=f"difference {honest - naive:+.4f} degC",
    )

    out.rule("4  the temperature scale is inferred, and refuses when it cannot be")
    out.check(
        "degree-scale samples are recognised",
        schema.infer_temperature_divisor([45.875, 46.343, 48.687]) == 1.0,
        "a file already converted to degrees loads correctly",
    )
    for name, samples, pattern in (
        ("all samples missing", [-1.0, -1.0], "no finite non-sentinel"),
        ("a scale in neither band", [500.0, 505.0], "neither the degree band"),
    ):
        fired = False
        try:
            schema.infer_temperature_divisor(samples)
        except schema.TraceSchemaError as error:
            fired = pattern in str(error)
        out.control(f"{name} refuses rather than guessing", fired, must_fire=True)

    out.rule("5  damaged files are refused, each by what is wrong with it")
    cases = (
        ("empty file", "", "no header row"),
        ("header with no data rows", REAL_HEADER + "\n", "no data rows"),
        (
            "ragged row",
            REAL_HEADER + "\n" + REAL_ROWS[0] + "\n" + REAL_ROWS[1].rsplit(",", 1)[0] + "\n",
            "field count",
        ),
        (
            "negative value in a non-sentinel column",
            REAL_HEADER + "\n" + REAL_ROWS[0].replace(",19447.34,", ",-19447.34,") + "\n",
            "iters_per_s",
        ),
    )
    for name, content, pattern in cases:
        path = scratch / f"damaged_{name.replace(' ', '_')}.csv"
        path.write_text(content)
        fired = False
        try:
            load_trace(path, origin=Origin.MEASURED)
        except schema.TraceSchemaError as error:
            fired = pattern in str(error)
        out.control(f"{name} is refused naming {pattern!r}", fired, must_fire=True)

    out.rule("6  an absent or empty input set refuses; it never returns a verdict")
    fired = False
    try:
        load_campaign(scratch / "nowhere", origin=Origin.MEASURED)
    except TraceDataAbsentError as error:
        fired = "does not exist" in str(error)
    out.control("an absent directory refuses", fired, must_fire=True)
    empty = scratch / "empty_dir"
    empty.mkdir()
    fired = False
    try:
        load_campaign(empty, origin=Origin.MEASURED)
    except TraceDataAbsentError as error:
        fired = "empty input set" in str(error)
    out.control("an empty directory refuses", fired, must_fire=True)
    partial = scratch / "partial"
    partial.mkdir()
    (partial / "burst_1.00_a.csv").write_text(REAL_HEADER + "\n" + "\n".join(REAL_ROWS) + "\n")
    fired = False
    try:
        load_campaign(partial, origin=Origin.MEASURED, require=["soak_1.00_r1"])
    except TraceDataAbsentError as error:
        fired = "soak_1.00_r1" in str(error)
    out.control("a required run absent is refused BY NAME", fired, must_fire=True)

    out.rule("7  the origin lattice, exhaustively over its four assignments")
    assignments = [
        (Origin.MEASURED, Origin.MEASURED, Origin.MEASURED),
        (Origin.MEASURED, Origin.SYNTHETIC, Origin.SYNTHETIC),
        (Origin.SYNTHETIC, Origin.MEASURED, Origin.SYNTHETIC),
        (Origin.SYNTHETIC, Origin.SYNTHETIC, Origin.SYNTHETIC),
    ]
    agree = all(meet(left, right) is expected for left, right, expected in assignments)
    out.check(
        "meet is the minimum over all 4 assignments of the 2-element lattice",
        agree,
        "MEASURED is the identity, SYNTHETIC is absorbing",
    )
    out.check(
        "the meet of an empty input set is the identity, which is why "
        "require_measured takes a count",
        meet() is Origin.MEASURED,
        "an empty set would otherwise be labelled a measurement",
    )
    fired = False
    try:
        require_measured(meet(), "a figure over no inputs", n_inputs=0)
    except NotMeasuredError as error:
        fired = "empty input set" in str(error)
    out.control("a figure over zero inputs is refused", fired, must_fire=True)
    fired = False
    try:
        require_measured(Origin.SYNTHETIC, "the headline", n_inputs=3)
    except NotMeasuredError as error:
        fired = "synthetic input" in str(error)
    out.control("a synthetic figure cannot be labelled measured", fired, must_fire=True)
    out.control(
        "a measured figure over real inputs is NOT refused",
        require_measured(Origin.MEASURED, "x", n_inputs=3) is not Origin.MEASURED,
        must_fire=False,
        detail="require_measured returns the origin unchanged",
    )

    out.rule("8  a synthetic campaign never matches the declared board")
    synthetic = _bootstrap.synthetic_campaign("no_throttle")
    out.check(
        "every synthetic trace carries Origin.SYNTHETIC",
        all(trace.origin is Origin.SYNTHETIC for trace in synthetic),
        f"{len(synthetic)} runs loaded",
    )
    out.control(
        "the synthetic manifest does not match the declared board",
        not synthetic.board_matches_declared_target(),
        must_fire=True,
        detail=f"manifest model: {synthetic.board_model[:60]}...",
    )
    out.check(
        "the synthetic header is identical to the binding schema",
        tuple(synthetic["soak_1.00_r1"].columns) == schema.expected_columns(),
        "schema-identical by construction, so the measured path is exercised",
    )

    out("")
    out(f"raw output written to {out.save()}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
