"""P091 ThermalBudget hardware measurement campaign (runs ON the Jetson).

Writes one CSV per run into ~/p091_data/ plus a manifest.  Every figure here is
measured on the board; nothing is simulated.  The payload is a fixed-work
GNC-representative kernel: one EKF propagate+update on a 12-state / 6-measurement
system plus one 96x96 float64 GEMM per iteration, deterministic, no RNG in the
loop.  Throughput is counted iterations per wall second across all workers.
"""
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

OUT = pathlib.Path.home() / "p091_data"
NSTATE, NMEAS, GEMM = 12, 6, 96
WINDOW = 0.050  # duty-cycle control window, seconds
SAMPLE = 1.0    # sysfs sample interval, seconds

ZONES = {}  # populated at start: name -> path


def discover_zones():
    z = {}
    base = pathlib.Path("/sys/devices/virtual/thermal")
    for p in sorted(base.glob("thermal_zone*")):
        try:
            t = _read_bytes(p / "type").decode().strip()
            v = _read_bytes(p / "temp").decode().strip()
        except Exception:
            # MEASURED ON THIS BOARD: thermal_zone2/3/4 (cv0/cv1/cv2-thermal)
            # make pathlib.read_text raise TypeError: can't concat NoneType to
            # bytes, because the driver returns no bytes at all rather than an
            # empty string.  A shell `cat` of the same file prints nothing and
            # exits 0, which is why a shell probe does not see this.
            continue
        if v == "":
            continue
        z[t] = str(p / "temp")
    return z


def _read_bytes(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data is None:
        raise TypeError(f"{path} returned no bytes")
    return data


def read_int(path):
    try:
        return int(_read_bytes(path).decode().strip())
    except Exception:
        return -1


def kernel_setup():
    rng = np.random.default_rng(20261112)
    F = np.eye(NSTATE) + 0.01 * rng.standard_normal((NSTATE, NSTATE))
    H = rng.standard_normal((NMEAS, NSTATE))
    Q = np.eye(NSTATE) * 1e-4
    R = np.eye(NMEAS) * 1e-2
    A = rng.standard_normal((GEMM, GEMM))
    B = rng.standard_normal((GEMM, GEMM))
    return F, H, Q, R, A, B


def worker(duty, counter, stop, busy_ns):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    F, H, Q, R, A, B = kernel_setup()
    x = np.zeros(NSTATE)
    P = np.eye(NSTATE)
    z = np.ones(NMEAS) * 0.1
    I = np.eye(NSTATE)
    local = 0
    busy = 0
    flush_at = time.monotonic() + 0.25
    while not stop.is_set():
        w_start = time.monotonic()
        w_end = w_start + duty * WINDOW
        while time.monotonic() < w_end:
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (I - K @ H) @ P
            _ = A @ B
            local += 1
        busy += time.monotonic() - w_start
        rest = (w_start + WINDOW) - time.monotonic()
        if rest > 0:
            time.sleep(rest)
        now = time.monotonic()
        if now >= flush_at:
            with counter.get_lock():
                counter.value += local
            with busy_ns.get_lock():
                busy_ns.value += int(busy * 1e9)
            local = 0
            busy = 0.0
            flush_at = now + 0.25
    with counter.get_lock():
        counter.value += local
    with busy_ns.get_lock():
        busy_ns.value += int(busy * 1e9)


def foreign_cpu():
    """Total CPU% of processes that are not this campaign, as a contamination probe."""
    mine = {os.getpid()}
    tot = 0.0
    try:
        import subprocess
        out = subprocess.run(["ps", "-eo", "pid,pcpu,comm"], capture_output=True,
                             text=True, timeout=5).stdout.splitlines()[1:]
        for line in out:
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, pcpu, comm = parts
            if comm.startswith("python3") or comm.startswith("campaign"):
                continue
            if int(pid) in mine:
                continue
            tot += float(pcpu)
    except Exception:
        return -1.0
    return round(tot, 2)


def run(label, duty, seconds, nworkers):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{label}.csv"
    counter = mp.Value("q", 0)
    busy_ns = mp.Value("q", 0)
    stop = mp.Event()
    procs = []
    zone_names = sorted(ZONES)
    hdr = (["t_wall", "t_rel", "iters_cum", "iters_per_s", "achieved_duty",
            "loadavg1", "foreign_cpu_pct"]
           + [f"temp_{n}" for n in zone_names]
           + [f"cpu{c}_khz" for c in range(nworkers)]
           + ["gpu_hz"])
    fh = open(path, "w", buffering=1)
    fh.write(",".join(hdr) + "\n")

    if duty > 0:
        for _ in range(nworkers):
            p = mp.Process(target=worker, args=(duty, counter, stop, busy_ns))
            p.start()
            procs.append(p)

    t0 = time.monotonic()
    prev_c, prev_b, prev_t = 0, 0, t0
    deadline = t0 + seconds
    while time.monotonic() < deadline:
        time.sleep(SAMPLE)
        now = time.monotonic()
        with counter.get_lock():
            c = counter.value
        with busy_ns.get_lock():
            b = busy_ns.value
        dt = now - prev_t
        row = [f"{time.time():.3f}", f"{now - t0:.3f}", str(c),
               f"{(c - prev_c) / dt:.2f}",
               f"{(b - prev_b) / 1e9 / dt / max(nworkers, 1):.4f}" if duty > 0 else "0.0",
               open("/proc/loadavg").read().split()[0],
               f"{foreign_cpu():.2f}"]
        row += [str(read_int(ZONES[n])) for n in zone_names]
        row += [str(read_int(f"/sys/devices/system/cpu/cpu{c_}/cpufreq/scaling_cur_freq"))
                for c_ in range(nworkers)]
        row.append(str(read_int("/sys/class/devfreq/17000000.gpu/cur_freq")))
        fh.write(",".join(row) + "\n")
        prev_c, prev_b, prev_t = c, b, now

    stop.set()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    fh.close()
    return str(path)


def main():
    global ZONES
    ZONES = discover_zones()
    nworkers = os.cpu_count() or 6
    manifest = {
        "started_wall": time.time(),
        "model": pathlib.Path("/proc/device-tree/model").read_bytes().decode(errors="replace").strip("\x00"),
        "uname": {k: getattr(os.uname(), k) for k in ("sysname", "nodename", "release", "version", "machine")},
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "nworkers": nworkers,
        "cpuinfo_max_freq_khz": read_int("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"),
        "governor": open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read().strip(),
        "zones_readable": sorted(ZONES),
        "zones_present_but_unreadable": ["cv0-thermal", "cv1-thermal", "cv2-thermal"],
        "payload": f"EKF {NSTATE}-state/{NMEAS}-meas propagate+update plus one {GEMM}x{GEMM} float64 GEMM per iteration",
        "window_s": WINDOW,
        "sample_s": SAMPLE,
        "runs": [],
    }
    # (label, duty, seconds)  cool = duty 0 cooldown trace, doubles as the RC decay data
    plan = [
        ("idle_pre",        0.0,  120),
        ("burst_1.00_a",    1.0,   20),
        ("cool_a",          0.0,  240),
        ("burst_1.00_b",    1.0,   20),
        ("cool_b",          0.0,  240),
        ("soak_1.00_r1",    1.0,  720),
        ("cool_r1",         0.0,  300),
        ("soak_0.50",       0.5,  480),
        ("cool_d50",        0.0,  240),
        ("soak_0.75",       0.75, 480),
        ("cool_d75",        0.0,  240),
        ("soak_0.25",       0.25, 420),
        ("cool_d25",        0.0,  180),
        ("soak_1.00_r2",    1.0,  480),
        ("cool_r2",         0.0,  300),
        ("soak_1.00_r3",    1.0,  480),
        ("cool_r3",         0.0,  180),
    ]
    for label, duty, secs in plan:
        t = time.time()
        p = run(label, duty, secs, nworkers)
        manifest["runs"].append({"label": label, "duty": duty, "seconds": secs,
                                 "csv": p, "started_wall": t,
                                 "ended_wall": time.time()})
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    manifest["finished_wall"] = time.time()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (OUT / "CAMPAIGN_COMPLETE").write_text(str(time.time()))


if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(_v) != "1":
            print(f"WARNING {_v}={os.environ.get(_v)!r}, expected '1'", flush=True)
    main()
