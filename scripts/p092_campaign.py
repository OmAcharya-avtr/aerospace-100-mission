#!/usr/bin/env python3
"""P092 DeadlineRisk — on-board measurement campaign (Jetson Orin Nano).

Runs a periodic hard-real-time loop with an implicit deadline under a set of
co-resident contention mixes, and records, per released job, when it was
released, when it actually started, when it finished, and whether it met its
deadline. A separate monitor process records contention descriptors that an
operator could observe WITHOUT running the task: foreign CPU occupancy, our
own occupancy, per-core clocks, die temperature and load average.

Stdlib only, on purpose: the measurement loop must not depend on numpy's
import or allocation behaviour, and scipy/sklearn are broken on this board.

Writes, into --out:
  manifest.json                  campaign metadata, written LAST but after the
                                 per-run files, so a reader can tell which runs
                                 completed even if the campaign is interrupted
  jobs_<run>.csv                 one row per RELEASED-AND-EXECUTED job
  mon_<run>.csv                  one row per monitor sample (1 Hz)
  RUN_COMPLETE_<run>             zero-byte marker, written after both CSVs
  CAMPAIGN_COMPLETE              zero-byte marker, written last of all

Deadline accounting, stated once because every number depends on it:
  * Period T, implicit deadline D = T.
  * Job i is released at t0 + i*T. Its response time is end - release.
  * A job with response > D is a MISS.
  * If a job finishes after one or more later releases have already passed,
    those releases are never executed. They are DROPPED, and a dropped
    release is also a miss. Each executed row carries dropped_before, so the
    total release count and the total miss count are both reconstructible:
        releases = sum(1 + dropped_before)
        misses   = sum(missed + dropped_before)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PERIOD_S = 0.010
DEADLINE_S = PERIOD_S          # implicit deadline
TARGET_FRAC = 0.45             # calibrate nominal exec to this fraction of D
CAL_W0 = 20000                 # starting payload size for calibration
CAL_JOBS = 120                 # jobs used per calibration measurement
PROBE_S = 15.0                 # per-level probe duration when choosing gt level
GT_TARGET_RATE = 0.02          # miss rate we would like the long run to sit at
GT_RATE_LO, GT_RATE_HI = 0.0005, 0.25
GT_DURATION_S = 600.0
MIX_DURATION_S = 150.0
SETTLE_S = 8.0
TJ_ZONE = 8                    # thermal_zone8 == tj-thermal on this board
NCORE = 6


# ---------------------------------------------------------------- payload ----
def payload(w: int, x: int) -> tuple[float, int]:
    """Fixed work, no allocation, no data-dependent branch."""
    acc = 0.0
    for _ in range(w):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        acc += (x & 1023) * 1e-3
    return acc, x


# --------------------------------------------------------------- stressors ---
CPU_STRESS = (
    "x=1\n"
    "while True:\n"
    "    x = (x*1103515245 + 12345) & 0x7FFFFFFF\n"
)
MEM_STRESS = (
    "a=bytearray(64*1024*1024)\n"
    "b=bytearray(64*1024*1024)\n"
    "while True:\n"
    "    a[:]=b\n"
    "    b[:]=a\n"
)


def start_stressors(n_cpu: int, n_mem: int) -> list[subprocess.Popen]:
    procs = []
    for _ in range(n_cpu):
        procs.append(subprocess.Popen([sys.executable, "-c", CPU_STRESS],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    for _ in range(n_mem):
        procs.append(subprocess.Popen([sys.executable, "-c", MEM_STRESS],
                                      stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    return procs


def stop_stressors(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()


# ----------------------------------------------------------------- monitor ---
def read_text(path: str) -> str:
    """Read a sysfs file defensively.

    thermal_zone2/3/4 on this board raise TypeError from pathlib.read_text
    ('can't concat NoneType to bytes') while a shell cat prints nothing and
    exits 0.  Read bytes and catch everything.
    """
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace").strip()
    except Exception:
        return ""


def cpu_jiffies() -> tuple[int, int]:
    """(total, idle) jiffies from the aggregate /proc/stat cpu line."""
    line = read_text("/proc/stat").splitlines()[0]
    f = [int(v) for v in line.split()[1:]]
    idle = f[3] + (f[4] if len(f) > 4 else 0)
    return sum(f), idle


def proc_jiffies(pid: int) -> int:
    s = read_text(f"/proc/{pid}/stat")
    if not s:
        return 0
    try:
        rest = s[s.rindex(")") + 2:].split()
        return int(rest[11]) + int(rest[12])      # utime + stime
    except Exception:
        return 0


def clocks_khz() -> list[int]:
    out = []
    for c in range(NCORE):
        v = read_text(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
        out.append(int(v) if v.isdigit() else -1)
    return out


def monitor(out_csv: Path, task_pid: int, stress_pids: list[int],
            duration: float) -> None:
    hz = os.sysconf("SC_CLK_TCK")
    rows = ["t_s,total_busy_j,task_busy_j,stress_busy_j,foreign_busy_j,"
            "foreign_pct,task_pct,stress_pct,"
            + ",".join(f"clk{c}_khz" for c in range(NCORE))
            + ",clk_mean_khz,clk_min_khz,tj_mdegC,load1\n"]
    t_tot, t_idle = cpu_jiffies()
    t_task = proc_jiffies(task_pid)
    t_str = sum(proc_jiffies(p) for p in stress_pids)
    t0 = time.perf_counter()
    nxt = t0 + 1.0
    while time.perf_counter() - t0 < duration:
        slp = nxt - time.perf_counter()
        if slp > 0:
            time.sleep(slp)
        nxt += 1.0
        tot, idle = cpu_jiffies()
        task = proc_jiffies(task_pid)
        stre = sum(proc_jiffies(p) for p in stress_pids)
        d_tot = tot - t_tot
        d_busy = (tot - idle) - (t_tot - t_idle)
        d_task = task - t_task
        d_str = stre - t_str
        t_tot, t_idle, t_task, t_str = tot, idle, task, stre
        d_for = d_busy - d_task - d_str
        cap = d_tot if d_tot > 0 else 1
        clk = clocks_khz()
        good = [c for c in clk if c > 0]
        tj = read_text(f"/sys/class/thermal/thermal_zone{TJ_ZONE}/temp")
        load1 = (read_text("/proc/loadavg").split() or ["-1"])[0]
        rows.append(
            f"{time.perf_counter()-t0:.3f},{d_busy},{d_task},{d_str},{d_for},"
            f"{100.0*d_for/cap:.4f},{100.0*d_task/cap:.4f},"
            f"{100.0*d_str/cap:.4f},"
            + ",".join(str(c) for c in clk)
            + f",{(sum(good)/len(good)) if good else -1:.1f},"
            f"{min(good) if good else -1},"
            f"{tj if tj else -1},{load1}\n")
        _ = hz
    out_csv.write_text("".join(rows))


# -------------------------------------------------------------- the RT loop ---
def rt_loop(w: int, duration: float, record: bool):
    """Periodic loop. Returns (rows, releases, misses, execs)."""
    rows = []
    x = 123456789
    payload(w, x)                      # warm the interpreter's caches once
    t0 = time.perf_counter()
    i = 0
    releases = 0
    misses = 0
    execs = []
    while True:
        rel = t0 + i * PERIOD_S
        now = time.perf_counter()
        if now < rel:
            time.sleep(rel - now)
        t_start = time.perf_counter()
        _, x = payload(w, x)
        t_end = time.perf_counter()
        resp = t_end - rel
        ex = t_end - t_start
        missed = 1 if resp > DEADLINE_S else 0
        j = int((t_end - t0) // PERIOD_S) + 1
        dropped = j - i - 1
        if dropped < 0:
            dropped = 0
        releases += 1 + dropped
        misses += missed + dropped
        execs.append(ex)
        if record:
            rows.append(f"{i},{rel-t0:.6f},{t_start-rel:.9f},{t_end-t0:.6f},"
                        f"{ex:.9f},{resp:.9f},{missed},{dropped}\n")
        i = j
        if t_end - t0 >= duration:
            break
    return rows, releases, misses, execs


def median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ----------------------------------------------------------------- campaign ---
def calibrate(log) -> int:
    w = CAL_W0
    history = []
    for attempt in range(6):
        _, _, _, execs = rt_loop(w, CAL_JOBS * PERIOD_S, record=False)
        med = median(execs)
        history.append({"attempt": attempt, "w": w, "median_exec_s": med,
                        "frac_of_deadline": med / DEADLINE_S})
        log(f"CAL w={w} median_exec={med*1e3:.4f} ms "
            f"frac={med/DEADLINE_S:.4f}")
        target = TARGET_FRAC * DEADLINE_S
        if abs(med - target) / target < 0.06:
            break
        scale = target / med
        scale = max(0.2, min(5.0, scale))
        w = max(200, int(w * scale))
    return w, history


def run_one(name: str, w: int, n_cpu: int, n_mem: int, duration: float,
            out: Path, log) -> dict:
    log(f"RUN {name} start n_cpu={n_cpu} n_mem={n_mem} dur={duration:.0f}s")
    procs = start_stressors(n_cpu, n_mem)
    time.sleep(SETTLE_S)
    mon_csv = out / f"mon_{name}.csv"
    mon = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--monitor",
         "--mon-out", str(mon_csv), "--task-pid", str(os.getpid()),
         "--stress-pids", ",".join(str(p.pid) for p in procs) or "0",
         "--mon-duration", str(duration + 2.0)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_wall0 = time.time()
    rows, releases, misses, execs = rt_loop(w, duration, record=True)
    t_wall1 = time.time()
    try:
        mon.wait(timeout=20)
    except Exception:
        mon.kill()
    stop_stressors(procs)
    hdr = ("job_index,release_s,start_delay_s,end_s,exec_s,response_s,"
           "missed,dropped_before\n")
    (out / f"jobs_{name}.csv").write_text(hdr + "".join(rows))
    (out / f"RUN_COMPLETE_{name}").write_text("")
    meta = {"run": name, "n_cpu_stress": n_cpu, "n_mem_stress": n_mem,
            "duration_s": duration, "payload_w": w,
            "executed_jobs": len(rows), "releases": releases,
            "misses": misses,
            "miss_rate": (misses / releases) if releases else -1,
            "median_exec_s": median(execs) if execs else -1,
            "wall_start_unix": t_wall0, "wall_end_unix": t_wall1}
    log(f"RUN {name} done releases={releases} misses={misses} "
        f"rate={meta['miss_rate']:.6f} median_exec="
        f"{meta['median_exec_s']*1e3:.4f} ms")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--mon-out")
    ap.add_argument("--task-pid", type=int)
    ap.add_argument("--stress-pids", default="")
    ap.add_argument("--mon-duration", type=float, default=10.0)
    a = ap.parse_args()

    if a.monitor:
        pids = [int(p) for p in a.stress_pids.split(",") if p and p != "0"]
        monitor(Path(a.mon_out), a.task_pid, pids, a.mon_duration)
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = out / "campaign.log"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n"
        with open(logf, "a") as fh:
            fh.write(line)

    log("CAMPAIGN start")
    env = {
        "model": read_text("/proc/device-tree/model").replace("\x00", ""),
        "uname": " ".join(os.uname()),       # structseq has no _asdict
        "nproc": os.cpu_count(),
        "nv_tegra_release": read_text("/etc/nv_tegra_release"),
        "python": sys.version.split()[0],
        "clk_max_khz": read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
        "governor": read_text(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "loadavg_at_start": read_text("/proc/loadavg"),
        "tj_zone": TJ_ZONE,
        "sudo_n_true": subprocess.run(["sudo", "-n", "true"],
                                      capture_output=True).returncode,
    }
    log(f"ENV {json.dumps(env)}")

    w, cal_hist = calibrate(log)
    log(f"CALIBRATED w={w}")

    probes = []
    for lvl in range(0, NCORE + 1):
        procs = start_stressors(lvl, 0)
        time.sleep(4.0)
        _, rel, mis, ex = rt_loop(w, PROBE_S, record=False)
        stop_stressors(procs)
        rate = mis / rel if rel else -1
        probes.append({"n_cpu_stress": lvl, "releases": rel, "misses": mis,
                       "miss_rate": rate, "median_exec_s": median(ex)})
        log(f"PROBE n_cpu={lvl} releases={rel} misses={mis} rate={rate:.6f}")

    usable = [p for p in probes if GT_RATE_LO <= p["miss_rate"] <= GT_RATE_HI]
    if usable:
        gt_level = min(usable,
                       key=lambda p: abs(p["miss_rate"] - GT_TARGET_RATE)
                       )["n_cpu_stress"]
        gt_reason = ("probe miss rate closest to the declared target "
                     f"{GT_TARGET_RATE} inside [{GT_RATE_LO}, {GT_RATE_HI}]")
    else:
        gt_level = 4
        gt_reason = ("NO probe landed inside the declared usable band; fell "
                     "back to the pre-declared default of 4 CPU stressors")
    log(f"GT_LEVEL {gt_level} ({gt_reason})")

    plan = [("gt_long", gt_level, 0, GT_DURATION_S)]
    for lvl in range(0, NCORE + 1):
        plan.append((f"mix_c{lvl}", lvl, 0, MIX_DURATION_S))
    plan.append(("mix_mem2", 0, 2, MIX_DURATION_S))
    plan.append(("mix_c2mem1", 2, 1, MIX_DURATION_S))

    runs = []
    for name, nc, nm, dur in plan:
        try:
            runs.append(run_one(name, w, nc, nm, dur, out, log))
        except Exception as exc:                       # noqa: BLE001
            log(f"RUN {name} FAILED {type(exc).__name__}: {exc}")
            runs.append({"run": name, "failed": f"{type(exc).__name__}: {exc}"})
        (out / "manifest.json").write_text(json.dumps(
            {"campaign": "P092 DeadlineRisk", "env": env,
             "period_s": PERIOD_S, "deadline_s": DEADLINE_S,
             "payload_w": w, "calibration": cal_hist,
             "target_frac_of_deadline": TARGET_FRAC,
             "probes": probes, "gt_level": gt_level, "gt_reason": gt_reason,
             "gt_rate_band": [GT_RATE_LO, GT_RATE_HI],
             "gt_target_rate": GT_TARGET_RATE,
             "runs": runs, "complete": False}, indent=2) + "\n")

    (out / "manifest.json").write_text(json.dumps(
        {"campaign": "P092 DeadlineRisk", "env": env,
         "period_s": PERIOD_S, "deadline_s": DEADLINE_S,
         "payload_w": w, "calibration": cal_hist,
         "target_frac_of_deadline": TARGET_FRAC,
         "probes": probes, "gt_level": gt_level, "gt_reason": gt_reason,
         "gt_rate_band": [GT_RATE_LO, GT_RATE_HI],
         "gt_target_rate": GT_TARGET_RATE,
         "runs": runs, "complete": True}, indent=2) + "\n")
    (out / "CAMPAIGN_COMPLETE").write_text("")
    log("CAMPAIGN complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
