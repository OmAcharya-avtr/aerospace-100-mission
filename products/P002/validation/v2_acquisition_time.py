"""V2 - Monte Carlo acquisition time vs the uniform-coverage analytic approximation.

Claim under test
----------------
The uniform-coverage approximation

    E[T | r <= r_max] = (pi / (s v)) * E[r^2 | r <= r_max]

(``scan.expected_acquisition_time_spiral``; derivation in the module
docstring) predicts the mean spiral acquisition time within a stated
tolerance. This is an APPROXIMATION: it treats the spiral as sweeping area
at a constant rate s*v from radius 0 outward, ignoring (a) the discrete
inner turns where the "annulus" model is poor, (b) the fact that the beam
covers a disc of radius R_beam rather than a line of width s, and (c) the
finite dwell time.

Method
------
Draw 20 000 targets from the 2-D Gaussian, run the dwell-by-dwell
acquisition simulation with p_dwell = 1 and with p_dwell = 0.9, and compare
the conditional mean detection time against the analytic expression. Both
are conditioned on targets inside the design containment radius.

Run: python validation/v2_acquisition_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackbench.scan import (  # noqa: E402
    GaussianUncertainty,
    expected_acquisition_time_spiral,
    simulate_acquisition,
    spiral_scan,
)

SIGMA = 3.0e-4
BEAM = 2.0e-5
OVERLAP = 0.25
DWELL = 1.0e-3
CONTAINMENT = 0.995
N_TRIALS = 20_000


def effective_p_pass(p_dwell: float, step_fraction: float = 0.5) -> float:
    """Per-crossing detection probability from the per-dwell probability.

    A crossing of the beam footprint contains about
    n = 2 * R_beam / (step_fraction * R_beam) = 2 / step_fraction dwells,
    each an independent Bernoulli(p_dwell) trial, so the crossing is missed
    only if all of them fail.
    """
    n = 2.0 / step_fraction
    return 1.0 - (1.0 - p_dwell) ** n


def run_case(p_dwell: float, seed: int) -> dict:
    """Monte Carlo mean acquisition time for one detection probability."""
    u = GaussianUncertainty(SIGMA)
    pattern = spiral_scan(u, BEAM, overlap=OVERLAP, containment=CONTAINMENT,
                          dwell_time=DWELL)
    rng = np.random.default_rng(seed)
    targets = u.sample(N_TRIALS, rng)
    r = np.linalg.norm(targets, axis=1)
    inside = r <= pattern.max_radius
    times = []
    for tgt in targets[inside]:
        t = simulate_acquisition(pattern, tgt, p_dwell=p_dwell, rng=rng, max_passes=20)
        if t is not None:
            times.append(t)
    times = np.asarray(times)
    naive = expected_acquisition_time_spiral(
        u, BEAM, OVERLAP, pattern.scan_speed, containment=CONTAINMENT, p_pass=p_dwell
    )
    p_cross = effective_p_pass(p_dwell)
    analytic = expected_acquisition_time_spiral(
        u, BEAM, OVERLAP, pattern.scan_speed, containment=CONTAINMENT, p_pass=p_cross
    )
    return {
        "p_dwell": p_dwell,
        "p_cross": p_cross,
        "naive": naive,
        "n_inside": int(inside.sum()),
        "n_detected": times.size,
        "mc_mean": float(np.mean(times)),
        "mc_sem": float(np.std(times, ddof=1) / np.sqrt(times.size)),
        "mc_median": float(np.median(times)),
        "analytic": analytic,
        "scan_speed": pattern.scan_speed,
        "full_scan_time": pattern.scan_time,
    }


def main() -> int:
    """Run the acquisition-time validation and print a table."""
    print("V2 - MC acquisition time vs uniform-coverage analytic approximation")
    print(f"sigma = {SIGMA:.3e} rad, R_beam = {BEAM:.3e} rad, overlap = {OVERLAP}, "
          f"dwell = {DWELL * 1e3:.1f} ms")
    print(f"trials drawn = {N_TRIALS}, containment = {CONTAINMENT}")
    print()
    rows = [run_case(1.0, 101), run_case(0.9, 202)]
    print(f"{'p_dwell':>8} {'p_cross':>9} {'n_det':>7} {'MC mean [s]':>12} "
          f"{'+/- SEM':>9} {'naive [s]':>11} {'naive dev':>10} "
          f"{'corrected':>11} {'corr dev':>9}")
    print("-" * 96)
    devs = []
    naive_devs = []
    for r in rows:
        dev = (r["mc_mean"] - r["analytic"]) / r["analytic"]
        ndev = (r["mc_mean"] - r["naive"]) / r["naive"]
        devs.append(abs(dev))
        naive_devs.append(ndev)
        print(f"{r['p_dwell']:8.2f} {r['p_cross']:9.5f} {r['n_detected']:7d} "
              f"{r['mc_mean']:12.4f} {r['mc_sem']:9.4f} {r['naive']:11.4f} "
              f"{ndev:+10.2%} {r['analytic']:11.4f} {dev:+9.2%}")
    print()
    print("REPORTED FINDING: passing the PER-DWELL probability to the analytic model")
    print("as if it were a per-crossing probability overestimates the acquisition time")
    print(f"by {abs(naive_devs[1]):.1%} at p_dwell = 0.9. The simulated beam covers the")
    print("target for ~2/step_fraction = 4 consecutive dwells, so the per-crossing")
    print("detection probability is 1 - (1-p_dwell)^4 = "
          f"{rows[1]['p_cross']:.5f}, not 0.9. With that")
    print("correction the analytic model agrees with Monte Carlo to "
          f"{abs(devs[1]):.2%}.")
    print("The scan.py docstring now states this explicitly.")
    print()
    print(f"single-pass scan time  = {rows[0]['full_scan_time']:.4f} s")
    print(f"mean along-track speed = {rows[0]['scan_speed']:.6f} rad/s")
    print()
    print("Interpretation: the analytic model assumes area is swept uniformly from")
    print("radius 0 at rate s*v. The simulated spiral covers a disc of radius R_beam")
    print("around every dwell point, i.e. an effective swath wider than s whenever")
    print("overlap > 0, so it acquires SOONER than the uniform-coverage model")
    print("predicts. The deviation is therefore expected to be negative and of order")
    print("the overlap factor.")
    print()
    tol = 0.05
    ok = max(devs) < tol
    print(f"PASS criterion: |relative deviation| < {tol:.0%} for both cases "
          "(corrected p_pass)")
    print(f"max |deviation| = {max(devs):.3%}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
