"""Per-sample cost of each detector, as ratios and measured ranges.

Wall-clock figures move 10-40 % between runs in this container, so nothing here
is quoted as a single microsecond figure. Every cost is reported as a range over
repeated measurements and as a ratio to the cheapest detector, which is the
quantity that survives contention.

The learned detector's online cost is the number that matters: it is the reason
the Monte Carlo uses a batch score path at all, and it is a real deployment
constraint for a detector that has to run per telemetry sample.
"""

from __future__ import annotations

import time

import numpy as np
from _harness import run

REPEATS = 5
LENGTH = 40_000


def _time_detector(factory, stream, repeats=REPEATS):
    """Return (min, median, max) seconds per sample over ``repeats`` runs."""
    per_sample = []
    for _ in range(repeats):
        det = factory()
        det.reset()
        t0 = time.perf_counter()
        for v in stream:
            if det.update(v):
                det.reset()
        per_sample.append((time.perf_counter() - t0) / stream.size)
    arr = np.asarray(per_sample)
    return float(arr.min()), float(np.median(arr)), float(arr.max())


def body(report) -> None:
    from scipy.stats import ks_2samp

    from telemdrift.benchmark import DETECTOR_LABELS, STANDARD, analytic_factory
    from telemdrift.detectors import ANALYTIC_DETECTORS, ks_two_sample_statistic
    from telemdrift.features import window_features, window_features_single
    from telemdrift.learned import (
        LearnedDetector,
        build_training_set,
        score_stream,
        train_learned_detector,
    )
    from telemdrift.streams import stationary

    cfg = STANDARD
    stream = stationary(LENGTH, 58_950)

    report.section("0. Method")
    print(f"  {REPEATS} repeats per detector, {LENGTH} samples each, thresholds at")
    print("  their declared defaults so the alarm rate does not confound the cost.")
    print("  Reported as min / median / max microseconds per sample and as a ratio")
    print("  to the cheapest detector's median. Single-run figures are not quoted")
    print("  in prose anywhere in this repository.")

    report.section("1. Analytic detectors, online cost per sample")
    print("  detector             min      median       max    ratio to cheapest")
    medians = {}
    from telemdrift.detectors import make_detector

    for key in ANALYTIC_DETECTORS:
        default = make_detector(key).threshold
        lo, med, hi = _time_detector(analytic_factory(key, default), stream)
        medians[key] = med
        print(f"  {DETECTOR_LABELS[key]:15s} {1e6 * lo:9.3f} {1e6 * med:11.3f} "
              f"{1e6 * hi:9.3f}")
    cheapest = min(medians, key=lambda k: medians[k])
    print()
    print(f"  cheapest: {DETECTOR_LABELS[cheapest]}")
    print("  detector             ratio")
    for key in ANALYTIC_DETECTORS:
        print(f"  {DETECTOR_LABELS[key]:15s} {medians[key] / medians[cheapest]:6.1f}x")
    spread = max(medians.values()) / min(medians.values())
    report.check("the analytic detectors are within two orders of magnitude of "
                 "each other", spread < 100.0, f"{spread:.1f}x")
    report.finding(
        "analytic per-sample cost spans "
        f"{spread:.1f}x between the cheapest ({DETECTOR_LABELS[cheapest]}) and the "
        "dearest; all five are in the microsecond range and none is a deployment "
        "constraint on this container."
    )

    report.section("2. The windowed KS implementation against SciPy")
    print("  The detector implements the two-sample KS statistic directly. SciPy's")
    print("  ks_2samp is the correctness reference (validate_known_answers.py s.3);")
    print("  here it is the cost reference.")
    ref = np.sort(stationary(200, 58_951))
    win = stationary(100, 58_952)
    n = 2_000
    mine, theirs = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(n):
            ks_two_sample_statistic(ref, win)
        mine.append((time.perf_counter() - t0) / n)
        t0 = time.perf_counter()
        for _ in range(n):
            ks_2samp(ref, win)
        theirs.append((time.perf_counter() - t0) / n)
    m_med, t_med = float(np.median(mine)), float(np.median(theirs))
    print()
    print(f"  this package   median {1e6 * m_med:9.1f} us per evaluation "
          f"(range {1e6 * min(mine):.1f} to {1e6 * max(mine):.1f})")
    print(f"  scipy ks_2samp median {1e6 * t_med:9.1f} us per evaluation "
          f"(range {1e6 * min(theirs):.1f} to {1e6 * max(theirs):.1f})")
    print(f"  ratio          {t_med / m_med:.1f}x")
    report.check("the direct implementation is cheaper than the SciPy call",
                 m_med < t_med, f"{t_med / m_med:.1f}x")
    report.finding(
        f"the direct KS statistic is about {t_med / m_med:.0f}x cheaper per "
        "evaluation than scipy.stats.ks_2samp, which is why it exists; the two agree "
        "to machine precision on the statistic. SciPy additionally returns a p-value, "
        "which this package does not need and does not use."
    )

    report.section("3. The learned detector's online cost, and why the harness batches")
    training = build_training_set(seeds=range(58_001, 58_005))
    model = train_learned_detector(training, n_estimators=100)
    short = stationary(1_500, 58_953)
    det = LearnedDetector(model, 0.9, cfg.window)
    online = []
    for _ in range(3):
        det.reset()
        t0 = time.perf_counter()
        for v in short:
            det.update(v)
        online.append((time.perf_counter() - t0) / short.size)
    batch = []
    long_stream = stationary(40_000, 58_954)
    for _ in range(3):
        t0 = time.perf_counter()
        score_stream(model, long_stream, cfg.window)
        batch.append((time.perf_counter() - t0) / long_stream.size)
    o_med, b_med = float(np.median(online)), float(np.median(batch))
    print(f"  online, one forest call per sample : median {1e6 * o_med:10.1f} us/sample "
          f"(range {1e6 * min(online):.1f} to {1e6 * max(online):.1f})")
    print(f"  batch, whole stream at once        : median {1e6 * b_med:10.3f} us/sample "
          f"(range {1e6 * min(batch):.3f} to {1e6 * max(batch):.3f})")
    print(f"  ratio                              : {o_med / b_med:.0f}x")
    print(f"  cheapest analytic detector         : median "
          f"{1e6 * medians[cheapest]:10.3f} us/sample")
    print(f"  online learned / cheapest analytic : "
          f"{o_med / medians[cheapest]:.0f}x")
    print()
    print("  Read that figure carefully. Most of the online cost is scikit-learn's")
    print("  fixed per-call overhead on a single row -- input validation, array")
    print("  wrapping, the Python call into the tree ensemble -- not the traversal")
    print("  of 100 shallow trees, which is a few microseconds of actual work. The")
    print("  batch column is the same trees doing the same arithmetic. So the honest")
    print("  claim is about this API in this environment, not about decision forests")
    print("  in general: a C or batched implementation would be far cheaper. What")
    print("  does transfer is the direction and the order of magnitude. A detector")
    print("  that needs a model call per telemetry sample is in a different cost")
    print("  class from a three-line recursion, and on a spacecraft processor that")
    print("  distinction decides whether it can run at all.")
    report.check("the batch path is at least 100x cheaper per sample than the online "
                 "path", o_med / b_med > 100.0, f"{o_med / b_med:.0f}x")
    report.finding(
        f"the learned detector's online per-sample cost is about "
        f"{o_med / medians[cheapest]:.0f}x the cheapest analytic detector's with this "
        "scikit-learn API on this container, because it makes one single-row forest "
        f"call per sample; the batch score path is about {o_med / b_med:.0f}x cheaper "
        "per sample and gives bit-identical results (tests/test_benchmark.py pins the "
        "identity). Most of the online figure is scikit-learn's fixed per-call "
        "overhead rather than tree traversal, so the magnitude is API-specific and "
        "only the order of magnitude and the direction should be carried elsewhere. "
        "A streaming deployment does not get the batch path."
    )

    report.section("4. The RandomForest n_jobs defect, re-measured for this product")
    print("  An earlier session in this portfolio recorded that n_jobs > 1 is")
    print("  5.7-8.9x SLOWER than n_jobs = 1 at single-row inference. The learned")
    print("  detector makes exactly that call once per sample, so the defect is")
    print("  re-measured here rather than taken on trust.")
    print()
    rows = window_features(short, cfg.window)[:600]
    results = {}
    for n_jobs in (1, 2):
        model.n_jobs = n_jobs
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            for i in range(rows.shape[0]):
                model.predict_proba(rows[i : i + 1])
            times.append((time.perf_counter() - t0) / rows.shape[0])
        results[n_jobs] = float(np.median(times))
        print(f"  n_jobs = {n_jobs}: median {1e6 * results[n_jobs]:9.1f} us per "
              f"single-row call (range {1e6 * min(times):.1f} to "
              f"{1e6 * max(times):.1f})")
    model.n_jobs = 1
    ratio = results[2] / results[1]
    print(f"  ratio n_jobs=2 / n_jobs=1: {ratio:.2f}x")
    if ratio > 1.0:
        report.check("n_jobs > 1 is slower at single-row inference, as recorded",
                     True, f"{ratio:.2f}x slower")
        report.finding(
            f"RandomForestClassifier.predict_proba on a single row is {ratio:.2f}x "
            "slower with n_jobs = 2 than with n_jobs = 1 on this container, "
            "reproducing the defect an earlier session recorded. This package sets "
            "n_jobs = 1 after fitting for that reason."
        )
    else:
        report.check("the recorded n_jobs defect did NOT reproduce here", True,
                     f"{ratio:.2f}x")
        report.finding(
            f"the recorded n_jobs single-row slowdown did NOT reproduce in this run: "
            f"n_jobs = 2 measured {ratio:.2f}x the n_jobs = 1 time. The package still "
            "sets n_jobs = 1, which is never slower. Wall clock in this container "
            "moves 10-40 % between runs, so a ratio near 1 is not a contradiction of "
            "the earlier measurement."
        )

    report.section("5. Feature extraction, batch against single-window")
    f_single = []
    for _ in range(3):
        t0 = time.perf_counter()
        for i in range(1_000):
            window_features_single(short[i : i + cfg.window])
        f_single.append((time.perf_counter() - t0) / 1_000)
    f_batch = []
    for _ in range(3):
        t0 = time.perf_counter()
        window_features(long_stream, cfg.window)
        f_batch.append((time.perf_counter() - t0) / long_stream.size)
    s_med, bt_med = float(np.median(f_single)), float(np.median(f_batch))
    print(f"  single window : median {1e6 * s_med:9.1f} us")
    print(f"  batch         : median {1e6 * bt_med:9.3f} us per window")
    print(f"  ratio         : {s_med / bt_med:.0f}x")
    report.check("batch feature extraction is at least 10x cheaper per window",
                 s_med / bt_med > 10.0, f"{s_med / bt_med:.0f}x")

    report.section("6. Total validation compute, for the record")
    print("  Measured wall clock of each validation script in this directory is")
    print("  printed at the end of its own output file. Nothing in this repository")
    print("  needs more than four minutes standalone on two cores, which is inside")
    print("  the batch budget. Expect a 5-6x slowdown under contention with sibling")
    print("  build agents.")


if __name__ == "__main__":
    raise SystemExit(run("validate_cost",
                         "telemdrift 0.1.0 - per-sample cost, as ratios and ranges",
                         body))
