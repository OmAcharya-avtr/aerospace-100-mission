#!/usr/bin/env python3
"""P093 PowerEnvelope - on-board measurement campaign.

Stdlib only, on purpose. The measurement loop must not depend on numpy's
import, threading or allocation behaviour, and the rail sampler must not be
blocked by anything the pipeline does.

Three roles, one file:

  --role campaign   orchestrator: calibration, then one run per cell
  --role sampler    INA3221 rail sampler, writes a CSV, nothing else
  --role monitor    1 Hz environment monitor, writes a CSV, nothing else

Every role requires --out. A role that cannot write its output exits non-zero
and says why on stderr. The orchestrator checks, after every single run, that
each expected file exists and has more than one data row, and ABORTS THE WHOLE
CAMPAIGN if not: on 2026-11-13 a campaign for another product ran 36 minutes
and wrote no monitor file at all, because a required argument was omitted and
the subprocess's stderr went to DEVNULL.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

MONO = time.CLOCK_MONOTONIC
HWMON = "/sys/class/hwmon/hwmon1/"
TJ = "/sys/class/thermal/thermal_zone8/temp"
NCORE = 6

# ----------------------------------------------------------------- pipeline

MASK = (1 << 31) - 1


def make_frame(n, idx):
    """Deterministic synthetic frame, n*n bytes, from a per-frame LCG seed.

    Deterministic in the frame index alone, so the decision sequence of any run
    is exactly reproducible off-board from the committed CSV. The power is not.
    """
    s = (idx * 2654435761 + 12345) & MASK
    out = bytearray(n * n)
    for i in range(n * n):
        s = (1103515245 * s + 12345) & MASK
        out[i] = (s >> 16) & 0xFF
    # one bright blob whose position and size depend on the frame index
    bw = 4 + (idx % 7)
    bx = 3 + (idx * 7) % max(1, n - bw - 6)
    by = 3 + (idx * 13) % max(1, n - bw - 6)
    for y in range(by, by + bw):
        row = y * n
        for x in range(bx, bx + bw):
            out[row + x] = 250
    return out


def downsample(buf, n):
    m = n // 2
    out = [0] * (m * m)
    for y in range(m):
        r0 = (2 * y) * n
        r1 = r0 + n
        o = y * m
        for x in range(m):
            c = 2 * x
            out[o + x] = (buf[r0 + c] + buf[r0 + c + 1] + buf[r1 + c] + buf[r1 + c + 1]) >> 2
    return out, m


def gradient(img, m):
    g = [0] * (m * m)
    for y in range(1, m - 1):
        o = y * m
        up = o - m
        dn = o + m
        for x in range(1, m - 1):
            dx = img[o + x + 1] - img[o + x - 1]
            dy = img[dn + x] - img[up + x]
            g[o + x] = (dx if dx >= 0 else -dx) + (dy if dy >= 0 else -dy)
    return g


def components(g, m, thr):
    """Two-pass 4-connected labelling of {g >= thr}, dict union-find."""
    lab = [0] * (m * m)
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nxt = 1
    for y in range(m):
        o = y * m
        for x in range(m):
            if g[o + x] < thr:
                continue
            left = lab[o + x - 1] if x > 0 else 0
            upp = lab[o - m + x] if y > 0 else 0
            if left and upp:
                lab[o + x] = left
                union(left, upp)
            elif left:
                lab[o + x] = left
            elif upp:
                lab[o + x] = upp
            else:
                lab[o + x] = nxt
                parent[nxt] = nxt
                nxt += 1
    stats = {}
    for y in range(m):
        o = y * m
        for x in range(m):
            L = lab[o + x]
            if not L:
                continue
            r = find(L)
            st = stats.get(r)
            if st is None:
                stats[r] = [1, g[o + x], x, x, y, y]
            else:
                st[0] += 1
                st[1] += g[o + x]
                if x < st[2]:
                    st[2] = x
                if x > st[3]:
                    st[3] = x
                if y < st[4]:
                    st[4] = y
                if y > st[5]:
                    st[5] = y
    return stats


GRAD_THR = 90
AREA_REF = 6.0
AREA_MIN = 2
AREA_MAX = 60


def infer(n, idx):
    """One inference. Returns (score, n_candidates, area, touches_border)."""
    buf = make_frame(n, idx)
    img, m = downsample(buf, n)
    g = gradient(img, m)
    st = components(g, m, GRAD_THR)
    if not st:
        return (0.0, 0, 0, False)
    best = None
    for area, gsum, x0, x1, y0, y1 in st.values():
        mean_g = gsum / area
        sc = mean_g * min(1.0, area / AREA_REF)
        if best is None or sc > best[0]:
            best = (sc, area, x0, x1, y0, y1)
    sc, area, x0, x1, y0, y1 = best
    border = (x0 == 0) or (y0 == 0) or (x1 == m - 1) or (y1 == m - 1)
    return (sc, len(st), area, border)


def gate(score, area, border, tau):
    if score <= 0.0:
        return 0, "no_candidate"
    if border:
        return 0, "border"
    if area < AREA_MIN or area > AREA_MAX:
        return 0, "area_out"
    if score < tau:
        return 0, "low_score"
    return 1, "accept"


# ------------------------------------------------------------------ sampler

def role_sampler(out, duration, period):
    try:
        fh = {k: open(HWMON + k) for k in (
            "in1_input", "curr1_input", "in2_input", "curr2_input",
            "in3_input", "curr3_input")}
    except OSError as exc:
        sys.stderr.write("sampler: cannot open INA3221: %s\n" % exc)
        return 3
    rows = []
    t_end = time.clock_gettime(MONO) + duration
    nxt = time.clock_gettime(MONO)
    while True:
        now = time.clock_gettime(MONO)
        if now >= t_end:
            break
        vals = []
        for k in ("in1_input", "curr1_input", "in2_input", "curr2_input",
                  "in3_input", "curr3_input"):
            h = fh[k]
            h.seek(0)
            vals.append(int(h.read()))
        rows.append((now, vals))
        nxt += period
        sl = nxt - time.clock_gettime(MONO)
        if sl > 0:
            time.sleep(sl)
        else:
            nxt = time.clock_gettime(MONO)
    ru = resource.getrusage(resource.RUSAGE_SELF)
    with open(out, "w") as f:
        f.write("t_mono,vdd_in_mv,vdd_in_ma,cpu_gpu_cv_mv,cpu_gpu_cv_ma,"
                "soc_mv,soc_ma,vdd_in_mw\n")
        for t, v in rows:
            f.write("%.6f,%d,%d,%d,%d,%d,%d,%.3f\n" % (
                t, v[0], v[1], v[2], v[3], v[4], v[5], v[0] * v[1] / 1000.0))
    with open(out + ".rusage", "w") as f:
        f.write(json.dumps({"utime": ru.ru_utime, "stime": ru.ru_stime,
                            "rows": len(rows), "period_s": period}) + "\n")
    if len(rows) < 2:
        sys.stderr.write("sampler: only %d rows\n" % len(rows))
        return 4
    return 0


# ------------------------------------------------------------------ monitor

def read_stat():
    with open("/proc/stat") as f:
        line = f.readline().split()
    v = [int(x) for x in line[1:]]
    idle = v[3] + v[4]
    return sum(v), idle


def core_clocks():
    out = []
    for c in range(NCORE):
        p = "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq" % c
        try:
            with open(p) as f:
                out.append(int(f.read().strip()))
        except OSError:
            out.append(-1)
    return out


def role_monitor(out, duration):
    rows = []
    prev = read_stat()
    t_end = time.clock_gettime(MONO) + duration
    while time.clock_gettime(MONO) < t_end:
        time.sleep(1.0)
        cur = read_stat()
        dt_total = cur[0] - prev[0]
        dt_idle = cur[1] - prev[1]
        busy = (dt_total - dt_idle) / dt_total if dt_total else 0.0
        prev = cur
        try:
            with open(TJ) as f:
                tj = int(f.read().strip()) / 1000.0
        except OSError:
            tj = -1.0
        cl = core_clocks()
        la = os.getloadavg()[0]
        rows.append((time.clock_gettime(MONO), busy, tj, la, cl))
    with open(out, "w") as f:
        f.write("t_mono,cpu_busy_fraction,tj_c,load1," +
                ",".join("clk%d_khz" % i for i in range(NCORE)) + "\n")
        for t, busy, tj, la, cl in rows:
            f.write("%.6f,%.6f,%.3f,%.2f,%s\n" % (
                t, busy, tj, la, ",".join(str(c) for c in cl)))
    if len(rows) < 2:
        sys.stderr.write("monitor: only %d rows\n" % len(rows))
        return 4
    return 0


# -------------------------------------------------------------- orchestrator

CYCLE_S = 2.0
CYCLES = 38           # 76 s per run
SETTLE_S = 4.0
SAMPLE_PERIOD = 0.05  # 20 Hz

# Declared BEFORE the campaign runs: tau is the 65th percentile of the
# calibration probe's score distribution for that workload, so the strict
# acceptance rate is ~0.35 by construction; the permissive control uses the
# 5th percentile. The probe scores and both tau values are written to
# manifest.json, so the choice is auditable and was not made after the fact.
TAU_STRICT_Q = 0.65
TAU_PERMISSIVE_Q = 0.05
PROBE_SECONDS = 8.0
WARMUP_S = 60.0


def spawn(role, out, duration, log):
    cmd = [sys.executable, os.path.abspath(__file__), "--role", role,
           "--out", out, "--duration", "%.3f" % duration]
    if role == "sampler":
        cmd += ["--period", "%.4f" % SAMPLE_PERIOD]
    err = open(log, "w")
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err), err


def check_file(path, min_rows, logf):
    if not os.path.exists(path):
        logf("FATAL missing file %s" % path)
        return False
    with open(path) as f:
        nrows = sum(1 for _ in f) - 1
    if nrows < min_rows:
        logf("FATAL %s has %d data rows, need >= %d" % (path, nrows, min_rows))
        return False
    logf("  ok %-34s %6d rows" % (os.path.basename(path), nrows))
    return True


def run_cell(outdir, cell, n, duty, tau, cycles, logf, pipeline=True):
    dur = cycles * CYCLE_S
    pw = os.path.join(outdir, "power_%s.csv" % cell)
    mo = os.path.join(outdir, "monitor_%s.csv" % cell)
    de = os.path.join(outdir, "decisions_%s.csv" % cell)
    ps, pe = spawn("sampler", pw, dur + 1.0, pw + ".stderr")
    pm, me = spawn("monitor", mo, dur + 1.0, mo + ".stderr")
    time.sleep(0.5)
    t0 = time.clock_gettime(MONO)
    epoch0 = time.time()
    rows = []
    idx = 0
    active_total = 0.0
    if pipeline:
        for c in range(cycles):
            c_start = t0 + c * CYCLE_S
            a_end = c_start + duty * CYCLE_S
            while time.clock_gettime(MONO) < a_end:
                ts = time.clock_gettime(MONO)
                sc, ncand, area, border = infer(n, idx)
                acc, why = gate(sc, area, border, tau)
                te = time.clock_gettime(MONO)
                rows.append((ts, te, c, idx, sc, ncand, area,
                             1 if border else 0, acc, why))
                active_total += te - ts
                idx += 1
            nxt = c_start + CYCLE_S
            sl = nxt - time.clock_gettime(MONO)
            if sl > 0:
                time.sleep(sl)
    else:
        time.sleep(dur)
    t1 = time.clock_gettime(MONO)
    ps.wait(timeout=30)
    pm.wait(timeout=30)
    pe.close()
    me.close()
    with open(de, "w") as f:
        f.write("t_start,t_end,cycle,frame_idx,score,n_candidates,area,"
                "touches_border,accepted,reject_reason\n")
        for r in rows:
            f.write("%.6f,%.6f,%d,%d,%.6f,%d,%d,%d,%d,%s\n" % r)
    logf("run %-22s n=%-3s duty=%.2f tau=%.4f inferences=%d accepted=%d "
         "active_cpu_s=%.3f wall=%.1fs sampler_rc=%s monitor_rc=%s"
         % (cell, n, duty, tau, len(rows), sum(r[8] for r in rows),
            active_total, t1 - t0, ps.returncode, pm.returncode))
    ok = check_file(pw, 10, logf) and check_file(mo, 3, logf)
    if pipeline:
        ok = check_file(de, 5, logf) and ok
    if ps.returncode != 0 or pm.returncode != 0:
        logf("FATAL subprocess returncode sampler=%s monitor=%s; stderr follows"
             % (ps.returncode, pm.returncode))
        for p in (pw + ".stderr", mo + ".stderr"):
            try:
                with open(p) as f:
                    logf("  %s: %s" % (os.path.basename(p), f.read().strip()))
            except OSError:
                pass
        ok = False
    return ok, {
        "cell": cell, "n": n, "duty": duty, "tau": tau, "cycles": cycles,
        "t0_mono": t0, "t1_mono": t1, "epoch_at_t0": epoch0,
        "inferences": len(rows), "accepted": sum(r[8] for r in rows),
        "active_cpu_s": active_total, "wall_s": t1 - t0,
        "sampler_rc": ps.returncode, "monitor_rc": pm.returncode,
        "pipeline": pipeline,
    }


def calibrate(n, logf):
    """Sustained probe AFTER warm-up. Never a cold probe: a cold probe on this
    board is taken while schedutil is still ramping the clock, which on
    2026-11-13 mis-sized another product's payload by a factor of 2.3."""
    scores = []
    lat = []
    t_end = time.clock_gettime(MONO) + PROBE_SECONDS
    idx = 0
    while time.clock_gettime(MONO) < t_end:
        ts = time.clock_gettime(MONO)
        sc, ncand, area, border = infer(n, idx)
        te = time.clock_gettime(MONO)
        scores.append(sc)
        lat.append(te - ts)
        idx += 1
    ss = sorted(scores)
    sl = sorted(lat)

    def q(v, p):
        if not v:
            return 0.0
        k = min(len(v) - 1, max(0, int(round(p * (len(v) - 1)))))
        return v[k]

    rec = {
        "n": n, "probe_inferences": idx,
        "latency_median_s": q(sl, 0.5), "latency_p95_s": q(sl, 0.95),
        "latency_min_s": sl[0], "latency_max_s": sl[-1],
        "tau_strict": q(ss, TAU_STRICT_Q),
        "tau_permissive": q(ss, TAU_PERMISSIVE_Q),
        "score_median": q(ss, 0.5),
        "score_quantiles": {str(p): q(ss, p)
                            for p in (0.05, 0.25, 0.5, 0.65, 0.75, 0.95)},
    }
    logf("calibration n=%d inferences=%d median_latency=%.6fs tau_strict=%.6f "
         "tau_permissive=%.6f" % (n, idx, rec["latency_median_s"],
                                  rec["tau_strict"], rec["tau_permissive"]))
    return rec


def role_campaign(outdir, dry):
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "campaign.log")
    lf = open(logpath, "a", buffering=1)

    def logf(msg):
        line = "[%s] %s" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg)
        lf.write(line + "\n")
        print(line, flush=True)

    man = {
        "product": "P093 PowerEnvelope",
        "role": "campaign",
        "dry_run": bool(dry),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cycle_s": CYCLE_S, "cycles_per_run": CYCLES,
        "sample_period_s": SAMPLE_PERIOD,
        "grad_thr": GRAD_THR, "area_ref": AREA_REF,
        "area_min": AREA_MIN, "area_max": AREA_MAX,
        "tau_rule": "tau_strict = quantile(probe_scores, %.2f); "
                    "tau_permissive = quantile(probe_scores, %.2f); declared "
                    "before the campaign ran" % (TAU_STRICT_Q, TAU_PERMISSIVE_Q),
        "ina3221": {},
        "env": {},
        "calibration": [],
        "runs": [],
        "aborted": False,
    }
    for k in ("update_interval", "samples"):
        try:
            with open(HWMON + k) as f:
                man["ina3221"][k] = f.read().strip()
        except OSError:
            man["ina3221"][k] = None
    for k in ("in1_label", "in2_label", "in3_label"):
        try:
            with open(HWMON + k) as f:
                man["ina3221"][k] = f.read().strip()
        except OSError:
            pass
    try:
        man["env"]["uname"] = " ".join(os.uname())
        man["env"]["python"] = sys.version.split()[0]
        man["env"]["ncpu_affinity"] = len(os.sched_getaffinity(0))
        man["env"]["loadavg_at_start"] = os.getloadavg()
        with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
            man["env"]["cpuinfo_max_freq_khz"] = int(f.read().strip())
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            man["env"]["governor"] = f.read().strip()
    except OSError:
        pass

    if dry:
        logf("DRY RUN: instrumentation only. 1 short pipeline cell and 1 "
             "no-pipeline cell, then every file checked.")
        cal = calibrate(64, logf)
        man["calibration"].append(cal)
        ok1, r1 = run_cell(outdir, "dry_pipeline", 64, 0.5,
                           cal["tau_strict"], 10, logf, True)
        man["runs"].append(r1)
        ok2, r2 = run_cell(outdir, "dry_floor", 64, 0.0, 0.0, 8, logf, False)
        man["runs"].append(r2)
        man["dry_run_all_files_ok"] = bool(ok1 and ok2)
        man["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(os.path.join(outdir, "manifest_dry.json"), "w") as f:
            json.dump(man, f, indent=1)
        if not (ok1 and ok2):
            logf("DRY RUN FAILED. The full campaign is NOT launched.")
            return 5
        logf("DRY RUN PASSED. Every expected file exists with rows.")
        return 0

    logf("WARM-UP %.0f s at full occupancy before any calibration, so the "
         "probe is not taken while the governor is ramping." % WARMUP_S)
    t_end = time.clock_gettime(MONO) + WARMUP_S
    i = 0
    while time.clock_gettime(MONO) < t_end:
        infer(64, i)
        i += 1
    logf("warm-up done, %d inferences" % i)

    workloads = [48, 64, 96]
    duties = [0.10, 0.25, 0.50, 0.75, 1.00]
    taus = {}
    for n in workloads:
        cal = calibrate(n, logf)
        man["calibration"].append(cal)
        taus[n] = cal

    plan = []
    plan.append(("floor", 64, 0.0, 0.0, False))
    for n in workloads:
        for d in duties:
            plan.append(("w%d_d%03d" % (n, int(round(d * 100))), n, d,
                         taus[n]["tau_strict"], True))
    for rep in (2, 3, 4):
        plan.append(("w64_d050_rep%d" % rep, 64, 0.50,
                     taus[64]["tau_strict"], True))
    plan.append(("w64_d050_permissive", 64, 0.50,
                 taus[64]["tau_permissive"], True))

    man["plan"] = [p[0] for p in plan]
    logf("plan: %d runs x %.0f s = %.1f min of measurement"
         % (len(plan), CYCLES * CYCLE_S, len(plan) * CYCLES * CYCLE_S / 60.0))
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)

    for cell, n, d, tau, pipe in plan:
        ok, rec = run_cell(outdir, cell, n, d, tau, CYCLES, logf, pipe)
        man["runs"].append(rec)
        with open(os.path.join(outdir, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)
        if not ok:
            man["aborted"] = True
            man["abort_cell"] = cell
            with open(os.path.join(outdir, "manifest.json"), "w") as f:
                json.dump(man, f, indent=1)
            logf("CAMPAIGN ABORTED at cell %s. A campaign that completes is "
                 "not a campaign that measured." % cell)
            return 6
        time.sleep(SETTLE_S)

    man["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    man["env"]["loadavg_at_end"] = os.getloadavg()
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    with open(os.path.join(outdir, "CAMPAIGN_COMPLETE"), "w") as f:
        f.write(man["finished_utc"] + "\n")
    logf("CAMPAIGN COMPLETE, %d runs" % len(man["runs"]))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True,
                    choices=("campaign", "sampler", "monitor"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--period", type=float, default=SAMPLE_PERIOD)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.role == "sampler":
        return role_sampler(a.out, a.duration, a.period)
    if a.role == "monitor":
        return role_monitor(a.out, a.duration)
    return role_campaign(a.out, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
