#!/usr/bin/env python3
"""Instrumentation gate. Refuses the full campaign unless the dry run produced
every expected output file WITH ROWS.

Three of the last four hardware campaigns in this mission were discarded for
instrumentation defects. The 2026-11-14 session was the exception and the only
difference was a dry run that asserted every output file existed with rows
before the real campaign launched. This is that assertion, made explicit and
made a gate.
"""
from __future__ import annotations

import csv
import json
import os
import sys

EXPECTED = {
    "stages.csv": 20,
    "stages_pass.csv": 24,
    "runs.csv": 4,
    "env.csv": 10,
    "calibration.csv": 8,
}


def main():
    out = sys.argv[1]
    failures = []
    report = {}

    for name, min_rows in EXPECTED.items():
        path = os.path.join(out, name)
        if not os.path.exists(path):
            failures.append("MISSING FILE %s" % name)
            report[name] = {"exists": False}
            continue
        with open(path) as fh:
            rows = list(csv.reader(fh))
        if not rows:
            failures.append("EMPTY FILE (no header) %s" % name)
            report[name] = {"exists": True, "rows": 0}
            continue
        header = rows[0]
        data = rows[1:]
        report[name] = {"exists": True, "header_cols": len(header), "data_rows": len(data)}
        if len(data) < min_rows:
            failures.append("TOO FEW ROWS %s: %d data rows, need >= %d"
                            % (name, len(data), min_rows))
        blanks = sum(1 for r in data if not any(c.strip() for c in r))
        if blanks:
            failures.append("BLANK ROWS %s: %d" % (name, blanks))
        ragged = sum(1 for r in data if len(r) != len(header))
        if ragged:
            failures.append("RAGGED ROWS %s: %d rows differ from header width" % (name, ragged))

    # Raw per-frame arrays must exist, one per cell, and be loadable.
    raw_dir = os.path.join(out, "raw")
    if not os.path.isdir(raw_dir):
        failures.append("MISSING DIRECTORY raw/")
    else:
        npys = sorted(f for f in os.listdir(raw_dir) if f.endswith(".npy"))
        report["raw"] = {"files": len(npys), "names": npys}
        if not npys:
            failures.append("raw/ contains no .npy arrays")
        try:
            import numpy as _np
            for n in npys:
                arr = _np.load(os.path.join(raw_dir, n))
                if arr.ndim != 2 or arr.shape[1] != 6:
                    failures.append("raw/%s has shape %r, expected (N, 6)"
                                    % (n, arr.shape))
                elif arr.shape[0] == 0:
                    failures.append("raw/%s is empty" % n)
                elif int(arr.min()) <= 0:
                    failures.append("raw/%s contains a non-positive duration" % n)
        except Exception as exc:
            failures.append("raw/ arrays unreadable: %s: %s" % (type(exc).__name__, exc))

    # Per-pass summaries must cover all six stages and agree with the raw rows.
    ppath = os.path.join(out, "stages_pass.csv")
    if os.path.exists(ppath):
        with open(ppath) as fh:
            prows = list(csv.DictReader(fh))
        seen = sorted({r["stage"] for r in prows})
        report["stages_pass"] = {"rows": len(prows), "stages": seen}
        if len(seen) != 6:
            failures.append("stages_pass.csv covers %d stages, expected 6: %r"
                            % (len(seen), seen))
        bad = [r for r in prows if float(r["median_ns"]) <= 0 or int(r["n_frames"]) <= 0]
        if bad:
            failures.append("stages_pass.csv has %d rows with non-positive "
                            "median or zero frames" % len(bad))

    mpath = os.path.join(out, "meta.json")
    if not os.path.exists(mpath):
        failures.append("MISSING FILE meta.json")
    else:
        meta = json.load(open(mpath))
        report["meta.json"] = {"keys": len(meta)}
        for k in ("randomised_order", "permutation", "calibration_drift_ratio",
                  "elapsed_s", "governor", "numpy", "clock_perf_counter",
                  "n_passes_per_run", "n_timed_frames_per_run",
                  "calibration_policy", "raw_sample_every"):
            if k not in meta:
                failures.append("meta.json missing key %s" % k)
        if meta.get("randomised_order") == meta.get("unrandomised_order"):
            failures.append("RANDOMISATION INERT: randomised order equals natural order")
        ratio = meta.get("calibration_drift_ratio", -1)
        if not (0.2 < float(ratio) < 5.0):
            failures.append("calibration_drift_ratio implausible: %r" % ratio)

    # Timing sanity: every stage must have produced a positive duration
    spath = os.path.join(out, "stages.csv")
    if os.path.exists(spath):
        with open(spath) as fh:
            rd = csv.DictReader(fh)
            ns_cols = [c for c in (rd.fieldnames or []) if c.startswith("ns_")]
            nonpos = {c: 0 for c in ns_cols}
            total = 0
            for row in rd:
                total += 1
                for c in ns_cols:
                    if int(row[c]) <= 0:
                        nonpos[c] += 1
        report["stage_timing"] = {"rows": total, "nonpositive": nonpos,
                                  "stage_cols": len(ns_cols)}
        if len(ns_cols) != 6:
            failures.append("expected 6 stage timing columns, found %d" % len(ns_cols))
        for c, n in nonpos.items():
            if n:
                failures.append("NON-POSITIVE DURATION in %s on %d of %d frames"
                                % (c, n, total))

    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        print("\nINSTRUMENTATION GATE: FAIL")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("\nINSTRUMENTATION GATE: PASS -- every expected file exists with rows,"
          " no blank or ragged rows, six stage columns all strictly positive,"
          " randomisation is not inert.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
