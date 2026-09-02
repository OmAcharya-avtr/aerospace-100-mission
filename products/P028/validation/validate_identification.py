"""Validation 3: the classical matchers, and both of their error rates.

The identification rate is the number star-identification papers report. The
false-identification rate is the number they mostly do not, and it is the one
that decides whether a star tracker is safe to believe: an attitude that is
reported and wrong will be used. Every table here reports both, with the
Wilson interval on each, so a measured zero is presented as an upper bound
rather than as a zero.

What is checked:

* 3a exactness with no centroid noise;
* 3b both rates against centroid noise, triangle rule versus Pyramid rule;
* 3c the attitude error of the identifications that were correct;
* 3d both rates against the catalogue magnitude limit;
* 3e the tolerance-width trade: too narrow loses true matches, too wide
  admits false ones;
* 3f the regime that breaks the Pyramid rule's low false-identification rate
  -- a tolerance sized for far more noise than the frame actually has.

Comparison with the published behaviour of the Pyramid algorithm (Mortari,
Samaan, Bruccoleri & Junkins, *Navigation* 51(3), 171-183, 2004): the
published claim is qualitative here, that the fourth-star check makes false
identifications very rare and that identification stays robust as centroid
noise grows. Sections 3b and 3f test that claim on this implementation and
this synthetic sky. They neither reproduce nor refute the paper's own numbers,
which were measured on a different catalogue, a different camera and a
different noise model.

Run: ``python validation/validate_identification.py``
"""

from __future__ import annotations

import time

import numpy as np

from _common import (
    CAMERA_FOV_DEG,
    CAMERA_PIXELS,
    RATE_HEADER,
    SEED,
    banner,
    finish,
    rate_row,
    report,
    verdict,
)

from skymatch.benchmark import run_trials, wilson_interval
from skymatch.camera import CameraModel
from skymatch.catalogue import generate_catalogue
from skymatch.pairtable import PairTable
from skymatch.scene import SceneConfig
from skymatch.triangle import separation_tolerance

SIGMAS = (1.0, 5.0, 10.0, 20.0, 40.0, 60.0)
MAG_LIMITS = (5.0, 5.5, 6.0, 6.5)
TRIALS = 110


def _point_table(point, indent: str = "") -> None:
    print(indent + RATE_HEADER)
    for name, result in point.methods.items():
        row = rate_row(name, result.n_correct, result.n_false, result.n_none, result.n_trials)
        print(indent + row)


def main() -> int:
    passed: list[bool] = []
    cam = CameraModel(fov_deg=CAMERA_FOV_DEG, pixels=CAMERA_PIXELS)
    cat = generate_catalogue(6.0, seed=SEED)
    table = PairTable(cat, cam.max_separation_rad)
    print(f"reference catalogue: magnitude limit 6.0, {cat.n_stars} stars, "
          f"{table.n_pairs} pairs, {table.nbytes / 1e6:.1f} MB")
    print(f"reference camera: {cam.fov_deg:.1f} deg, {cam.pixels} px, "
          f"{cam.arcsec_per_pixel:.2f} arcsec/pixel, {TRIALS} trials per row")
    print()

    banner("VALIDATION 3a: noise-free exactness")
    cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=0.0, magnitude_sigma=0.0)
    point = run_trials(cat, table, cam, cfg, TRIALS, SEED, label="sigma = 0")
    _point_table(point)
    tri, pyr = point.methods["triangle"], point.methods["pyramid"]
    passed.append(
        verdict("triangle identification rate, no noise", tri.identification_rate, 1.0, mode=">=")
    )
    passed.append(
        verdict("pyramid identification rate, no noise", pyr.identification_rate, 1.0, mode=">=")
    )
    passed.append(verdict("pyramid false identifications, no noise", float(pyr.n_false), 0.0))
    report("median attitude error, noise free [arcsec]", pyr.median_attitude_error_arcsec)
    print("    with no centroid noise the attitude error is the projection round-off only")

    print()
    banner("VALIDATION 3b: both error rates against centroid noise")
    print("    magnitude limit 6.0, no false stars, tolerance matched to the true sigma")
    noise_points = {}
    for sigma in SIGMAS:
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=sigma)
        t0 = time.perf_counter()
        point = run_trials(cat, table, cam, cfg, TRIALS, SEED + 1, label=f"sigma = {sigma}")
        noise_points[sigma] = point
        tol_as = np.degrees(separation_tolerance(sigma)) * 3600.0
        print(f"\n  sigma = {sigma:g} arcsec ({sigma / cam.arcsec_per_pixel:.3f} px), "
              f"tolerance {tol_as:.1f} arcsec, {point.mean_candidates:.1f} candidates/frame, "
              f"{point.mean_seconds_per_frame * 1000:.0f} ms/frame, "
              f"{time.perf_counter() - t0:.0f} s")
        _point_table(point, indent="  ")

    print()
    print("  summary: identification rate and false-identification rate against sigma")
    print(f"  {'sigma':>8} {'tri ident':>10} {'tri false':>10} {'pyr ident':>10} {'pyr false':>10} "
          f"{'pyr false 95% upper':>20}")
    for sigma in SIGMAS:
        p = noise_points[sigma]
        t, y = p.methods["triangle"], p.methods["pyramid"]
        print(f"  {sigma:8g} {t.identification_rate:10.4f} {t.false_identification_rate:10.4f} "
              f"{y.identification_rate:10.4f} {y.false_identification_rate:10.4f} "
              f"{y.false_identification_ci[1]:20.4f}")

    pyr_rates = [noise_points[s].methods["pyramid"].identification_rate for s in SIGMAS]
    tri_rates = [noise_points[s].methods["triangle"].identification_rate for s in SIGMAS]
    passed.append(
        verdict("worst pyramid identification rate over the sweep", min(pyr_rates), 0.97, mode=">=")
    )
    total_false = sum(noise_points[s].methods["pyramid"].n_false for s in SIGMAS)
    n_total = TRIALS * len(SIGMAS)
    lo, hi = wilson_interval(total_false, n_total)
    print(f"\n  pooled pyramid false identifications over the whole sweep: "
          f"{total_false} / {n_total}, 95% interval [{lo:.5f}, {hi:.5f}]")
    print("  A measured zero is not a zero. With this many trials the honest statement is")
    print(f"  'below {hi:.3f} at 95% confidence', not 'never'.")
    passed.append(verdict("pooled pyramid false-identification rate", total_false / n_total, 0.01))
    print("\n  The triangle rule loses its identification rate as the tolerance widens with")
    print("  sigma, because a unique catalogue triangle stops being unique. The Pyramid")
    print("  rule does not, because the fourth star restores the discrimination the wider")
    print("  window gave away. That is the qualitative behaviour the 2004 paper reports,")
    print("  reproduced here on a synthetic sky:")
    print(f"    triangle rule: {tri_rates[0]:.3f} at 1 arcsec -> {tri_rates[-1]:.3f} at 60 arcsec")
    print(f"    pyramid rule:  {pyr_rates[0]:.3f} at 1 arcsec -> {pyr_rates[-1]:.3f} at 60 arcsec")

    print()
    banner("VALIDATION 3c: attitude error of the correct identifications")
    print("    the identification is a correspondence; the attitude that follows from it is")
    print("    a Wahba solve over the matched stars and its error scales with sigma")
    print(f"  {'sigma [as]':>10} {'median err [as]':>16} {'p95 err [as]':>13} {'median/sigma':>13} "
          f"{'matched stars':>14}")
    ratios = []
    for sigma in SIGMAS:
        p = noise_points[sigma].methods["pyramid"]
        med = p.median_attitude_error_arcsec
        ratios.append(med / sigma)
        print(f"  {sigma:10g} {med:16.3f} {p.p95_attitude_error_arcsec:13.3f} {med / sigma:13.4f} "
              f"{noise_points[sigma].mean_spots:14.2f}")
    spread = float(np.max(ratios) / np.min(ratios))
    print("    The ratio is roughly constant: the estimator is linear in the noise to first")
    print("    order, and the number of matched stars barely changes across the sweep.")
    passed.append(verdict("ratio of the largest to smallest median-error/sigma", spread, 1.6))

    print()
    banner("VALIDATION 3d: both error rates against the catalogue magnitude limit")
    print("    sigma = 5 arcsec, 2 false stars, tolerance matched")
    print(f"  {'mag':>6} {'stars':>7} {'pairs':>9} {'in field':>9} {'ceiling':>8} "
          f"{'tri ident':>10} {'tri false':>10} {'pyr ident':>10} {'pyr false':>10} {'ms':>6}")
    for limit in MAG_LIMITS:
        mag_cat = generate_catalogue(limit, seed=SEED)
        mag_table = PairTable(mag_cat, cam.max_separation_rad)
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=2)
        p = run_trials(mag_cat, mag_table, cam, cfg, TRIALS, SEED + 2, with_attitude=False)
        t, y = p.methods["triangle"], p.methods["pyramid"]
        print(f"  {limit:6.2f} {mag_cat.n_stars:7d} {mag_table.n_pairs:9d} "
              f"{p.mean_true_spots:9.2f} "
              f"{p.ceiling:8.3f} {t.identification_rate:10.4f} {t.false_identification_rate:10.4f} "
              f"{y.identification_rate:10.4f} {y.false_identification_rate:10.4f} "
              f"{p.mean_seconds_per_frame * 1000:6.0f}")
        if limit == 6.0:
            passed.append(
                verdict("pyramid identification rate at mag 6.0 with 2 false stars",
                        y.identification_rate, 0.90, mode=">=")
            )
    print("    'ceiling' is the fraction of frames whose candidate list contained the truth")
    print("    at all: the upper bound on any decision rule. At magnitude limit 5.0 it is")
    print("    limited by frames with too few stars in the field (validation 2g).")

    print()
    banner("VALIDATION 3e: how wide should the tolerance be?")
    print("    tau = k sqrt(2) sigma at a true sigma of 10 arcsec; k = 3 is the default")
    print("    NOTE: the first version of this check asserted that k = 1 would lose true")
    print("    matches. It does not, and the reason is combinatorial rather than statistical:")
    print("    a single edge lands inside a 1-sigma gate about 68% of the time and all three")
    print("    edges of one triangle about 32%, but 25 triples are scanned, so the truth")
    print("    survives somewhere in the candidate list with probability 1 - (1 - 0.32)^25,")
    print("    which rounds to 1. The gate had to go far narrower before the ceiling moved;")
    print("    k = 0.25 is where it does. See validation/VALIDATION.md section 7.")
    print(f"  {'k':>5} {'tau [as]':>9} {'cands/frame':>12} {'ceiling':>8} {'tri ident':>10} "
          f"{'tri false':>10} {'pyr ident':>10} {'pyr false':>10}")
    for k in (0.25, 0.5, 1.0, 2.0, 3.0, 6.0):
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=10.0)
        p = run_trials(
            cat, table, cam, cfg, 80, SEED + 3,
            tolerance_sigma_arcsec=10.0 * k / 3.0, with_attitude=False,
        )
        t, y = p.methods["triangle"], p.methods["pyramid"]
        tau = np.degrees(separation_tolerance(10.0 * k / 3.0)) * 3600.0
        print(f"  {k:5.2f} {tau:9.2f} {p.mean_candidates:12.1f} {p.ceiling:8.3f} "
              f"{t.identification_rate:10.4f} {t.false_identification_rate:10.4f} "
              f"{y.identification_rate:10.4f} {y.false_identification_rate:10.4f}")
        if k == 0.25:
            passed.append(
                verdict("ceiling at k = 0.25 (a quarter-sigma gate loses the truth)",
                        p.ceiling, 0.6)
            )
        if k == 3.0:
            passed.append(
                verdict("ceiling at the default k = 3", p.ceiling, 0.999, mode=">=")
            )
    print("    Narrow gates drop the truth out of the candidate list entirely (the ceiling")
    print("    falls); wide gates keep it but bury it, which costs the triangle rule its")
    print("    uniqueness and costs every rule compute.")

    print()
    banner("VALIDATION 3f: the regime that breaks the Pyramid rule")
    print("    a tolerance sized for 60 arcsec of centroid noise on a frame that actually")
    print("    has 5 arcsec: 12x too wide. This is a mis-sized gate, not a hard sky.")
    for n_false in (0, 4):
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=n_false)
        p = run_trials(
            cat, table, cam, cfg, 120, SEED + 4,
            tolerance_sigma_arcsec=60.0, with_attitude=False,
        )
        print(f"\n  {n_false} false star(s), ceiling {p.ceiling:.3f}, "
              f"{p.mean_candidates:.1f} candidates/frame, "
              f"{p.mean_seconds_per_frame * 1000:.0f} ms/frame")
        _point_table(p, indent="  ")
        if n_false == 4:
            y = p.methods["pyramid"]
            report("pyramid false-identification rate, mis-sized gate + 4 false stars",
                   y.false_identification_rate)
            passed.append(
                verdict("this regime produces a NON-ZERO pyramid false-ID rate",
                        y.false_identification_rate, 0.05, mode=">=")
            )
    print("\n  The Pyramid rule's low false-identification rate is a property of a correctly")
    print("  sized gate, not of the algorithm alone. Widen the gate by 12x and add a few")
    print("  false detections and the fourth star stops discriminating: several catalogue")
    print("  stars fit inside the window, the 'unique confirmation' test passes on the")
    print("  wrong triangle, and the rule reports a wrong attitude with no warning.")

    return finish(passed)


if __name__ == "__main__":
    raise SystemExit(main())
