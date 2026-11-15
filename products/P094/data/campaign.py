#!/usr/bin/env python3
"""P094 HilTimingDiff — hardware-mode timing campaign on the Jetson Orin Nano.

Design constraints that this campaign honours explicitly, each because a
previous session in this mission was damaged by its absence:

  * THE FLOOR IS MEASURED AT THE START AND AT THE END. P093 measured its floor
    once, first, under the highest foreign load of the session; that single
    leading measurement falsified a bound and accounted for the entire error
    scatter of one baseline. Here a fixed calibration workload runs before the
    grid and again after it, and the two are reported as a drift bracket.

  * THE GRID ORDER IS RANDOMISED under a recorded seed. P093's cells ran in
    ascending frame dimension, which confounded dimension with elapsed time
    and with a falling foreign load.

  * THE CLOCK IS NOT PINNED AND THAT IS MEASURED, NOT ASSUMED AWAY. The board
    runs the schedutil governor and passwordless sudo is unavailable, so
    nvpmodel and the governor cannot be changed. Per-core scaling_cur_freq and
    every thermal zone are therefore sampled before, during and after each
    run, and the frequency trace is a first-class output rather than a note.

  * THE INSTRUMENT IS MEASURED BEFORE THE SUBJECT. The empty bracket cost and
    the clock resolution are recorded per run, so that a stage time can be
    compared against the cost of observing it.

Usage:
  campaign.py --dry-run --out DIR     60-second instrumentation check
  campaign.py --out DIR               full campaign
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pat_pipeline import STAGES  # noqa: E402
from staged_exec import bracket_overhead_ns, make_frames, run_timed  # noqa: E402

CPUS = (0, 1, 2, 3, 4, 5)
CAMPAIGN_SEED = 20261115


def read_int(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except Exception:
        return -1


def sample_env():
    """One environment sample: per-core frequency, thermals, load."""
    row = {"t_mono": time.monotonic(), "t_wall": time.time()}
    for c in CPUS:
        row["freq_cpu%d" % c] = read_int(
            "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % c
        )
    zones = {}
    base = "/sys/devices/virtual/thermal"
    try:
        for name in sorted(os.listdir(base)):
            if not name.startswith("thermal_zone"):
                continue
            zpath = os.path.join(base, name)
            try:
                with open(os.path.join(zpath, "type")) as fh:
                    ztype = fh.read().strip()
            except Exception:
                continue
            t = read_int(os.path.join(zpath, "temp"))
            if t > 0:
                zones[ztype] = t
    except Exception:
        pass
    for k in ("cpu-thermal", "gpu-thermal", "soc0-thermal", "tj-thermal"):
        row["temp_" + k.replace("-", "_")] = zones.get(k, -1)
    try:
        with open("/proc/loadavg") as fh:
            parts = fh.read().split()
        row["load1"] = float(parts[0])
        row["load5"] = float(parts[1])
    except Exception:
        row["load1"] = -1.0
        row["load5"] = -1.0
    return row


ENV_FIELDS = (
    ["run_id", "point", "t_mono", "t_wall"]
    + ["freq_cpu%d" % c for c in CPUS]
    + ["temp_cpu_thermal", "temp_gpu_thermal", "temp_soc0_thermal", "temp_tj_thermal"]
    + ["load1", "load5"]
)


def calibration_workload(reps: int, size: int = 160):
    """Fixed, data-independent workload used as the timing drift reference.

    Deliberately NOT one of the pipeline stages: a drift reference that shares
    code with the subject cannot distinguish board drift from subject
    variation. Fixed size, fixed seed, identical every invocation.
    """
    rng = np.random.default_rng(777)
    a = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float64)
    b = rng.normal(0.0, 1.0, size=(size, size)).astype(np.float64)
    out = np.zeros(reps, dtype=np.int64)
    for i in range(reps):
        t0 = time.perf_counter_ns()
        c = a @ b
        s = float(c[0, 0]) + float(c[-1, -1])
        t1 = time.perf_counter_ns()
        out[i] = t1 - t0
        del c, s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.dry_run:
        widths = (48, 64)
        densities = (("sparse", 2), ("dense", 10))
        n_frames = 12
        n_passes = 2
        n_repeats = 1
        cal_reps = 40
        sample_every = 1
        label = "DRY"
    else:
        widths = (48, 64, 96)
        densities = (("sparse", 2), ("dense", 12))
        n_frames = 1500
        n_passes = 60
        n_repeats = 3
        cal_reps = 2000
        sample_every = 3
        label = "FULL"

    # ---- build the cell list, then randomise its order under a recorded seed
    cells = []
    for w in widths:
        for dname, ntg in densities:
            cells.append({"cell": "w%d_%s" % (w, dname), "w": w,
                          "density": dname, "n_targets": ntg, "rep": 0})
    ref_w = 64
    ref_d = "sparse"
    ref_n = dict(densities)[ref_d]
    for r in range(1, n_repeats + 1):
        cells.append({"cell": "w%d_%s_rep%d" % (ref_w, ref_d, r), "w": ref_w,
                      "density": ref_d, "n_targets": ref_n, "rep": r})

    order_rng = np.random.default_rng(CAMPAIGN_SEED)
    perm = order_rng.permutation(len(cells))
    cells_randomised = [cells[int(i)] for i in perm]

    meta = {
        "label": label,
        "campaign_seed": CAMPAIGN_SEED,
        "randomised_order": [c["cell"] for c in cells_randomised],
        "unrandomised_order": [c["cell"] for c in cells],
        "permutation": [int(i) for i in perm],
        "n_frames_per_stack": n_frames,
        "n_passes_per_run": n_passes,
        "n_timed_frames_per_run": n_frames * n_passes,
        "raw_sample_every": sample_every,
        "cal_reps": cal_reps,
        "calibration_policy": (
            "A fixed 160x160 float64 matmul, data-independent and sharing no code "
            "with the pipeline, timed in a block of cal_reps repetitions BEFORE EVERY "
            "GRID CELL and once more after the last. Ten blocks rather than the two "
            "the 2026-11-14 carry-over requires as a minimum, so board drift can be "
            "fitted over the whole campaign instead of only bracketed."
        ),
        "stages": list(STAGES),
        "host": platform.node(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "clock_perf_counter": str(time.get_clock_info("perf_counter")),
        "clock_monotonic": str(time.get_clock_info("monotonic")),
        "governor": open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip()
        if os.path.exists("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") else "unknown",
        "nvpmodel": "unknown",
        "sudo_passwordless": False,
        "t_campaign_start_wall": time.time(),
        "t_campaign_start_mono": time.monotonic(),
    }
    try:
        meta["nvpmodel"] = subprocess.run(
            ["nvpmodel", "-q"], capture_output=True, text=True, timeout=20
        ).stdout.strip().replace("\n", " | ")
    except Exception as exc:
        meta["nvpmodel"] = "unavailable: %s" % type(exc).__name__

    stages_path = os.path.join(args.out, "stages.csv")
    pass_path = os.path.join(args.out, "stages_pass.csv")
    runs_path = os.path.join(args.out, "runs.csv")
    env_path = os.path.join(args.out, "env.csv")
    cal_path = os.path.join(args.out, "calibration.csv")
    raw_dir = os.path.join(args.out, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    sfh = open(stages_path, "w", newline="")
    swr = csv.writer(sfh)
    swr.writerow(["run_id", "cell", "w", "density", "pass_idx", "frame_idx",
                  "k_candidates"] + ["ns_" + s for s in STAGES])

    # Per-pass summaries are computed over EVERY timed frame, not over the
    # subsample written to stages.csv. The pass is the bootstrap unit for
    # baseline 3: adjacent frames share cache and governor state and are
    # serially correlated, so resampling frames would understate the interval,
    # while a pass is a far closer approximation to an independent draw.
    pfh = open(pass_path, "w", newline="")
    pwr = csv.writer(pfh)
    pwr.writerow(["run_id", "cell", "w", "density", "pass_idx", "stage",
                  "n_frames", "mean_ns", "median_ns", "p05_ns", "p95_ns",
                  "min_ns", "max_ns", "std_ns", "sum_ns", "k_mean"])

    rfh = open(runs_path, "w", newline="")
    rwr = csv.writer(rfh)
    rwr.writerow(["run_id", "cell", "w", "density", "n_targets", "rep", "order_index",
                  "n_frames", "t0_mono", "t1_mono", "bracket_ns", "clock_res_ns",
                  "accept", "no_candidate", "low_score", "area_out", "border",
                  "synth_seed"])

    efh = open(env_path, "w", newline="")
    ewr = csv.writer(efh)
    ewr.writerow(ENV_FIELDS)

    cfh = open(cal_path, "w", newline="")
    cwr = csv.writer(cfh)
    cwr.writerow(["which", "rep_idx", "ns", "t_mono"])

    def log_env(run_id, point):
        row = sample_env()
        ewr.writerow([run_id, point] + [row.get(f, -1) for f in ENV_FIELDS[2:]])
        efh.flush()
        return row

    def run_calibration(which):
        log_env(which, "before")
        t_m0 = time.monotonic()
        ns = calibration_workload(cal_reps)
        for i, v in enumerate(ns):
            cwr.writerow([which, i, int(v), t_m0])
        cfh.flush()
        log_env(which, "after")
        print("[%s] calibration %s median %.0f ns over %d reps"
              % (label, which, float(np.median(ns)), len(ns)), flush=True)
        return ns

    # ======== FLOOR AT THE START ============================================
    # floor_00 is emitted inside the grid loop before the first cell; this
    # explicit leading block is the start of the drift series.
    cal_start = run_calibration("floor_start")

    # ======== RANDOMISED GRID ===============================================
    for idx, cell in enumerate(cells_randomised):
        run_id = cell["cell"]
        w = cell["w"]
        synth_seed = 4000 + 13 * idx + w
        frames = make_frames(w, n_frames, cell["n_targets"], synth_seed)

        # Calibration runs BEFORE every cell, so drift is fitted not bracketed.
        run_calibration("floor_%02d" % idx)

        log_env(run_id, "before")
        bracket = bracket_overhead_ns(400 if args.dry_run else 4000)
        res = time.get_clock_info("perf_counter").resolution * 1e9

        census_tot = {"accept": 0, "no_candidate": 0, "low_score": 0,
                      "area_out": 0, "border": 0}
        all_pf = []
        t0 = time.monotonic()
        for p in range(n_passes):
            per_frame, census, kpf = run_timed(frames)
            for key in census_tot:
                census_tot[key] += census[key]
            all_pf.append(per_frame.astype(np.int32))

            # per-pass summaries over ALL frames of the pass
            for si, sname in enumerate(STAGES):
                col = per_frame[:, si].astype(np.float64)
                pwr.writerow([run_id, cell["cell"], w, cell["density"], p, sname,
                              col.size,
                              "%.3f" % col.mean(), "%.1f" % np.median(col),
                              "%.1f" % np.percentile(col, 5),
                              "%.1f" % np.percentile(col, 95),
                              int(col.min()), int(col.max()),
                              "%.3f" % col.std(ddof=1), int(col.sum()),
                              "%.4f" % float(kpf.mean())])

            # subsampled raw rows
            for i in range(0, per_frame.shape[0], sample_every):
                swr.writerow([run_id, cell["cell"], w, cell["density"], p, i,
                              int(kpf[i])] + [int(v) for v in per_frame[i]])

            if p % max(1, n_passes // 6) == 0:
                log_env(run_id, "pass%d" % p)
        t1 = time.monotonic()
        pfh.flush()
        sfh.flush()
        log_env(run_id, "after")

        stacked = np.concatenate(all_pf, axis=0)
        np.save(os.path.join(raw_dir, "%s.npy" % run_id), stacked[::sample_every])

        rwr.writerow([run_id, cell["cell"], w, cell["density"], cell["n_targets"],
                      cell["rep"], idx, int(stacked.shape[0]), t0, t1,
                      "%.3f" % bracket, "%.3f" % res,
                      census_tot["accept"], census_tot["no_candidate"],
                      census_tot["low_score"], census_tot["area_out"],
                      census_tot["border"], synth_seed])
        rfh.flush()

        print("[%s] %2d/%2d %-18s passes=%d frames=%d median_total=%.1f us "
              "accept=%d k_mean=%.2f elapsed=%.1fs"
              % (label, idx + 1, len(cells_randomised), run_id, n_passes,
                 int(stacked.shape[0]),
                 float(np.median(stacked.sum(axis=1))) / 1000.0,
                 census_tot["accept"], float(kpf.mean()), t1 - t0), flush=True)
        del frames, all_pf, stacked

    # ======== FLOOR AT THE END ==============================================
    cal_end = run_calibration("floor_end")

    meta["t_campaign_end_wall"] = time.time()
    meta["t_campaign_end_mono"] = time.monotonic()
    meta["elapsed_s"] = meta["t_campaign_end_mono"] - meta["t_campaign_start_mono"]
    meta["calibration_start_median_ns"] = float(np.median(cal_start))
    meta["calibration_end_median_ns"] = float(np.median(cal_end))
    meta["calibration_drift_ratio"] = (
        float(np.median(cal_end)) / float(np.median(cal_start))
        if float(np.median(cal_start)) > 0 else -1.0
    )

    for fh in (sfh, pfh, rfh, efh, cfh):
        fh.close()

    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    print("[%s] CAMPAIGN_COMPLETE elapsed %.1f s drift_ratio %.6f"
          % (label, meta["elapsed_s"], meta["calibration_drift_ratio"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
