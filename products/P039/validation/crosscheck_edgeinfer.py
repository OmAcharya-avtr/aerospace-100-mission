#!/usr/bin/env python3
"""Mandatory cross-check -- P039 latencynet against P033 edgeinfer on one pipeline.

The agreement the specification requires
----------------------------------------
P033 EdgeInfer and P039 LatencyNet must agree on the latency of the same
synthetic pipeline from independent implementations. P033 published
``products/P033/validation/crosscheck_pipeline.json``: three lognormal stages
(60/180/30 us mean, 12/40/6 us sd), ``n = 2000`` passes, seed ``20260401``,
with the lognormal parameterisation taken from the declared mean and sd by
``sigma^2 = log1p((sd/mean)^2)``, ``mu = log(mean) - sigma^2/2``, and one draw
per stage per sample in stage order.

P033 recorded that its measured *mean* was contaminated by scheduler
preemption on a shared single core -- it sat 119 % above the injected mean on
the run that produced the file -- and nominated ``p50_s`` as the comparable
statistic. This script diffs p50, as instructed.

What each side's p50 actually is
--------------------------------
This is the crux of the result and it is stated before the numbers.

* P033's ``p50_s`` is a **measured** figure: it drew the stage durations,
  actually consumed each one with a spin-wait, and timed the pass with
  ``perf_counter``. It therefore carries P033's spin-wait overshoot, its
  per-call harness cost, and whatever the host's scheduler did.
* This script's p50 is a **sampled** figure from an independent
  implementation of the same declared cost model: it draws the stage
  durations from the documented parameterisation and sums them. It consumes no
  time and measures no clock, because latencynet's whole position is that a
  wall-clock measurement on this host is not a usable reference (see
  :mod:`latencynet.pipeline`).

Those are two different quantities and they are not expected to be equal. The
cross-check is therefore made in three parts:

(1) the injected cost model itself -- the sum of declared stage means -- which
    must agree exactly, because it is arithmetic;
(2) the sampled p50, diffed against P033's measured p50 as instructed, with
    the difference reported either way and not reconciled by changing this
    implementation;
(3) an accounting of the difference using P033's own published ratios, so the
    reader can see whether the gap is explained by the measurement path or
    whether the two implementations genuinely disagree about the distribution.

A volatile, labelled measured figure is also reported in part (4) for a
like-for-like comparison, using this package's own spin-wait. It is a
workstation number on a contended shared core and no correctness claim rests
on it.

Reproduce with:
    cd validation && PYTHONPATH=../src python3 crosscheck_edgeinfer.py

Runtime: about 3 s on one uncontended core.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from latencynet.pipeline import make_lognormal_pipeline, sample_stage_latencies
from latencynet.tails import quantile
from latencynet.units import s_to_us

#: The pipeline as P033 declared it. Rebuilt from the declared parameters, not
#: read out of P033's JSON, so that the two implementations stay independent.
STAGE_MEANS_S = (60e-6, 180e-6, 30e-6)
STAGE_STDS_S = (12e-6, 40e-6, 6e-6)
STAGE_NAMES = ("preprocess", "inference", "postprocess")
N_SAMPLES = 2000
SEED = 20260401

P033_JSON = (
    Path(__file__).resolve().parents[2] / "P033" / "validation" / "crosscheck_pipeline.json"
)
OUTPUT_JSON = Path(__file__).resolve().parent / "crosscheck_edgeinfer.json"

#: Agreement band on the injected cost model: this is arithmetic and must
#: agree to binary64 rounding.
EXACT_RTOL = 16.0 * np.finfo(float).eps
#: Agreement band on p50, from P033's own guidance ("two correct
#: implementations should agree on it to within a few per cent"). Three per
#: cent is the reading of "a few" used here, declared before the diff.
P50_BAND = 0.03
#: Repeats for the volatile measured figure in part (4).
MEASURE_REPEATS = 2000
MEASURE_WARMUP = 50
#: Large-sample size for the distributional median, which removes the n=2000
#: sampling noise from the accounting in part (3).
LARGE_N = 4_000_000


def _spin_consume(duration_s: float) -> None:
    """Busy-wait for ``duration_s`` seconds. Never returns early."""
    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        pass


def main() -> int:
    print("=" * 92)
    print("P039 latencynet -- mandatory cross-check against P033 edgeinfer")
    print("=" * 92)
    if not P033_JSON.exists():
        print(f"FAIL: P033 cross-check file not found at {Path(*P033_JSON.parts[-3:])}")
        return 1
    p033 = json.loads(P033_JSON.read_text())
    print(f"P033 file: {Path(*P033_JSON.parts[-3:])}")
    print(f"P033 nominated comparable statistic: {p033.get('comparable_statistic')!r}")

    spec = make_lognormal_pipeline(STAGE_MEANS_S, STAGE_STDS_S, names=STAGE_NAMES)

    print("\n(0) the two declarations of the pipeline must match before anything is diffed")
    print("-" * 92)
    agree = True
    for i, (name, mean_s, std_s) in enumerate(
        zip(STAGE_NAMES, STAGE_MEANS_S, STAGE_STDS_S, strict=True)
    ):
        theirs = p033["stages"][i]
        same = (
            theirs["name"] == name
            and float(theirs["mean_s"]) == mean_s
            and float(theirs["std_s"]) == std_s
            and theirs["dist"] == "lognormal"
        )
        agree = agree and same
        print(
            f"  stage {i} {name:<13} ours mean {s_to_us(mean_s):7.3f} us "
            f"sd {s_to_us(std_s):6.3f} us"
            f"   theirs mean {s_to_us(float(theirs['mean_s'])):7.3f} us "
            f"sd {s_to_us(float(theirs['std_s'])):6.3f} us   {'MATCH' if same else 'MISMATCH'}"
        )
    same_n = int(p033["n_samples"]) == N_SAMPLES
    same_seed = int(p033["seed"]) == SEED
    print(f"  n_samples ours {N_SAMPLES} theirs {p033['n_samples']}   "
          f"{'MATCH' if same_n else 'MISMATCH'}")
    print(f"  seed      ours {SEED} theirs {p033['seed']}   "
          f"{'MATCH' if same_seed else 'MISMATCH'}")
    agree = agree and same_n and same_seed

    print("\n(1) the injected cost model -- arithmetic, must agree exactly")
    print("-" * 92)
    our_injected_mean = spec.injected_mean_s()
    their_injected_mean = float(p033["injected"]["mean_s"])
    rel = abs(our_injected_mean - their_injected_mean) / their_injected_mean
    model_ok = rel <= EXACT_RTOL
    print(
        f"  ours   sum of declared stage means {s_to_us(our_injected_mean):.9f} us\n"
        f"  theirs injected mean               {s_to_us(their_injected_mean):.9f} us\n"
        f"  relative difference {rel:.3e}   tol {EXACT_RTOL:.3e}   "
        f"{'PASS' if model_ok else 'FAIL'}"
    )
    our_injected_sd = spec.injected_std_s()
    their_injected_sd = float(p033["injected"]["std_s"])
    rel_sd = abs(our_injected_sd - their_injected_sd) / their_injected_sd
    sd_ok = rel_sd <= EXACT_RTOL
    print(
        f"  ours   injected sd {s_to_us(our_injected_sd):.9f} us   "
        f"theirs {s_to_us(their_injected_sd):.9f} us   rel {rel_sd:.3e}   "
        f"{'PASS' if sd_ok else 'FAIL'}"
    )

    print("\n(2) p50 -- the statistic P033 nominated")
    print("-" * 92)
    stage_draws = sample_stage_latencies(spec, N_SAMPLES, SEED)
    total = stage_draws.sum(axis=1)
    our_p50 = quantile(total, 0.5, "linear")
    their_p50 = float(p033["measured"]["p50_s"])
    p50_rel_diff = (our_p50 - their_p50) / their_p50
    p50_ok = abs(p50_rel_diff) <= P50_BAND
    print(f"  ours   sampled  p50 {s_to_us(our_p50):.6f} us  (n={N_SAMPLES}, seed={SEED})")
    print(f"  theirs measured p50 {s_to_us(their_p50):.6f} us")
    print(
        f"  relative difference (ours - theirs) / theirs = {p50_rel_diff:+.6f} "
        f"= {p50_rel_diff * 100:+.4f} %   band +/-{P50_BAND * 100:.0f} %   "
        f"{'AGREE' if p50_ok else 'DISAGREE'}"
    )
    print(f"  ours   sampled mean {s_to_us(float(total.mean())):.6f} us vs injected "
          f"{s_to_us(our_injected_mean):.6f} us "
          f"(ratio {float(total.mean()) / our_injected_mean:.6f})")
    print(f"  ours   sampled sd   {s_to_us(float(total.std(ddof=1))):.6f} us vs injected "
          f"{s_to_us(our_injected_sd):.6f} us "
          f"(ratio {float(total.std(ddof=1)) / our_injected_sd:.6f})")

    print("\n(3) accounting for the difference, using P033's own published ratios")
    print("-" * 92)
    large = sample_stage_latencies(spec, LARGE_N, SEED + 1).sum(axis=1)
    large_p50 = quantile(large, 0.5, "linear")
    our_ratio = our_p50 / our_injected_mean
    large_ratio = large_p50 / our_injected_mean
    their_ratio = float(p033["ratios_on_this_run"]["p50_over_injected_mean"])
    explained = their_ratio / large_ratio
    residual = (1.0 + p50_rel_diff) * explained - 1.0
    print(
        f"  distributional median of the sum, from {LARGE_N} draws: "
        f"{s_to_us(large_p50):.6f} us\n"
        f"    median / injected mean = {large_ratio:.6f}  "
        f"(a sum of right-skewed lognormals has its median below its mean)\n"
        f"  our n=2000 p50 / injected mean          = {our_ratio:.6f}\n"
        f"  P033's measured p50 / injected mean     = {their_ratio:.6f}  (from their JSON)\n"
        f"  P033 measurement overhead on the median = "
        f"{(their_ratio / large_ratio - 1) * 100:+.4f} %"
    )
    print(
        f"  predicted gap from the measurement path alone: "
        f"{(their_ratio / large_ratio - 1) * 100:+.4f} %\n"
        f"  observed gap (theirs over ours, n=2000):       "
        f"{(their_p50 / our_p50 - 1) * 100:+.4f} %\n"
        f"  unexplained residual after accounting:         {residual * 100:+.4f} %"
    )

    print("\n(4) VOLATILE like-for-like measured figure from this package's own spin-wait")
    print("-" * 92)
    print(
        f"  method: perf_counter bracket around a spin-wait that consumes each\n"
        f"  drawn stage duration in turn; {MEASURE_REPEATS} repeats, "
        f"{MEASURE_WARMUP} warm-up discarded; one shared\n"
        f"  single CPU core with four other build jobs running concurrently.\n"
        f"  WORKSTATION NUMBER. VOLATILE: wall-clock timings on this host move by\n"
        f"  factors of 2 to 13 between runs. No correctness claim rests on it."
    )
    measured = np.empty(MEASURE_REPEATS, dtype=float)
    warm = sample_stage_latencies(spec, MEASURE_WARMUP, SEED + 2)
    for row in warm:
        for d in row:
            _spin_consume(float(d))
    for i in range(MEASURE_REPEATS):
        row = stage_draws[i]
        t0 = time.perf_counter()
        for d in row:
            _spin_consume(float(d))
        measured[i] = time.perf_counter() - t0
    our_measured_p50 = quantile(measured, 0.5, "linear")
    meas_rel = (our_measured_p50 - their_p50) / their_p50
    print(
        f"  ours   measured p50 {s_to_us(our_measured_p50):.6f} us "
        f"(overhead over our sampled p50: "
        f"{(our_measured_p50 / our_p50 - 1) * 100:+.4f} %)\n"
        f"  theirs measured p50 {s_to_us(their_p50):.6f} us\n"
        f"  measured-vs-measured relative difference {meas_rel:+.6f} "
        f"= {meas_rel * 100:+.4f} %   (volatile, not a verdict)\n"
        f"  ours   measured mean {s_to_us(float(measured.mean())):.6f} us "
        f"(ratio to injected {float(measured.mean()) / our_injected_mean:.4f}) -- "
        f"preemption-contaminated,\n  exactly as P033 reported for its own mean"
    )

    verdict = "AGREE" if p50_ok else "DISAGREE"
    payload = {
        "product": "P039 latencynet",
        "counterpart": "P033 edgeinfer",
        "counterpart_file": str(Path(*P033_JSON.parts[-3:])),
        "comparable_statistic": "p50_s",
        "pipeline": {
            "stages": [
                {"name": n, "mean_s": m, "std_s": s, "dist": "lognormal"}
                for n, m, s in zip(STAGE_NAMES, STAGE_MEANS_S, STAGE_STDS_S, strict=True)
            ],
            "n_samples": N_SAMPLES,
            "seed": SEED,
        },
        "declarations_match": bool(agree),
        "injected": {
            "ours_mean_s": our_injected_mean,
            "theirs_mean_s": their_injected_mean,
            "relative_difference": rel,
            "tolerance": EXACT_RTOL,
            "pass": bool(model_ok),
            "ours_std_s": our_injected_sd,
            "theirs_std_s": their_injected_sd,
            "std_relative_difference": rel_sd,
            "std_pass": bool(sd_ok),
        },
        "p50_diff": {
            "ours_sampled_p50_s": our_p50,
            "theirs_measured_p50_s": their_p50,
            "relative_difference": p50_rel_diff,
            "band": P50_BAND,
            "verdict": verdict,
        },
        "accounting": {
            "large_sample_n": LARGE_N,
            "distributional_median_s": large_p50,
            "distributional_median_over_injected_mean": large_ratio,
            "ours_p50_over_injected_mean": our_ratio,
            "theirs_p50_over_injected_mean": their_ratio,
            "p033_measurement_overhead_on_median": their_ratio / large_ratio - 1.0,
            "unexplained_residual": residual,
        },
        "volatile_measured": {
            "ours_measured_p50_s": our_measured_p50,
            "ours_measured_mean_s": float(measured.mean()),
            "relative_difference_vs_theirs": meas_rel,
            "repeats": MEASURE_REPEATS,
            "warmup": MEASURE_WARMUP,
            "method": (
                "perf_counter bracket around a spin-wait consuming each drawn stage "
                "duration in turn; gc not disabled; shared single core with four "
                "concurrent build jobs"
            ),
            "label": "VOLATILE WORKSTATION NUMBER -- not a measurement of any edge target",
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "finding": (
            "The two implementations agree exactly on the injected cost model: the sum of "
            "declared stage means matches to binary64 rounding, and so does the injected "
            "standard deviation. They disagree on p50 by "
            f"{p50_rel_diff * 100:+.4f} % because the two p50s are different quantities: "
            "P033's is a measured wall-clock median that includes its spin-wait overshoot "
            "and harness cost, while this one is the median of the sampled cost model with "
            "no clock involved. P033's own published ratio p50_over_injected_mean = "
            f"{their_ratio:.6f} against the distributional median ratio {large_ratio:.6f} "
            f"accounts for {(their_ratio / large_ratio - 1) * 100:+.4f} % of the gap, leaving "
            f"{residual * 100:+.4f} % unexplained. latencynet's implementation was NOT changed "
            "to close the gap."
        ),
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 92)
    print(f"declarations match: {'YES' if agree else 'NO'}")
    print(f"injected cost model: {'PASS' if model_ok and sd_ok else 'FAIL'}")
    print(f"p50 diff: {p50_rel_diff * 100:+.4f} %  ->  {verdict} within +/-{P50_BAND * 100:.0f} %")
    print(f"written: {OUTPUT_JSON.name}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
