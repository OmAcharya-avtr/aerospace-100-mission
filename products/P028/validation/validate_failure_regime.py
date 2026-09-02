"""Validation 4: the false-star failure regime, quantified.

A star tracker's detected-spot list is not a list of stars. Hot pixels, cosmic
rays, debris, unmasked planets, sunlit particles and internal reflections all
produce spots that look exactly like stars to a centroider. The frame is then
sorted by brightness and truncated, so a bright false detection does not add
to the spot list -- it *displaces a real star*, and the algorithm never sees
the star it needed.

This section measures where that stops being survivable. It reports three
things at each false-star count:

* the **ceiling**, the fraction of frames whose candidate list contained the
  truth at all. A decision rule cannot beat this. It separates "the search
  failed" from "the decision failed", which is the difference between a
  problem a better ranker could fix and one it could not;
* the identification rate of each rule;
* the false-identification rate of each rule, with its Wilson interval.

The headline result is that the two classical rules fail in opposite
directions, and that both fail completely once false detections outnumber the
real stars in the spot list.

Run: ``python validation/validate_failure_regime.py``
"""

from __future__ import annotations

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

from skymatch.benchmark import run_trials
from skymatch.camera import CameraModel
from skymatch.catalogue import generate_catalogue
from skymatch.pairtable import PairTable
from skymatch.scene import SceneConfig, simulate_scene

FALSE_COUNTS = (0, 2, 4, 6, 8, 10, 12, 16, 20, 30)
TRIALS = 200


def main() -> int:
    passed: list[bool] = []
    cam = CameraModel(fov_deg=CAMERA_FOV_DEG, pixels=CAMERA_PIXELS)
    cat = generate_catalogue(6.0, seed=SEED)
    table = PairTable(cat, cam.max_separation_rad)
    print(f"magnitude limit 6.0, {cat.n_stars} stars; camera {cam.fov_deg:.1f} deg / "
          f"{cam.pixels} px; sigma = 5 arcsec; 10 brightest spots used; "
          f"{TRIALS} trials per row")
    print(f"mean real stars on the detector: "
          f"{cat.expected_in_solid_angle(cam.solid_angle_sr):.2f}")
    print()

    banner("VALIDATION 4a: what the false stars do to the spot list")
    print("    false detections are drawn uniform on the detector with magnitudes uniform")
    print("    on [2.0, 6.0], then the whole list is sorted by brightness and truncated to")
    print("    10 spots -- so a bright false detection removes a real star from the list")
    print(f"{'n false':>9} {'real spots kept':>17} {'false spots kept':>18} {'false fraction':>16}")
    for n_false in FALSE_COUNTS:
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=n_false)
        rng = np.random.default_rng(SEED + 10)
        real, fake = [], []
        for _ in range(400):
            scene = simulate_scene(cat, cfg, rng)
            real.append(scene.n_true_stars)
            fake.append(scene.n_false_stars)
        print(f"{n_false:9d} {np.mean(real):17.3f} {np.mean(fake):18.3f} "
              f"{np.mean(fake) / (np.mean(real) + np.mean(fake)):16.3f}")

    print()
    banner("VALIDATION 4b: both error rates against false-star count")
    points = {}
    for n_false in FALSE_COUNTS:
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=n_false)
        point = run_trials(cat, table, cam, cfg, TRIALS, SEED + 11, with_attitude=False)
        points[n_false] = point
        print(f"\n  {n_false} false star(s): ceiling {point.ceiling:.4f}, "
              f"{point.mean_candidates:.1f} candidates/frame, "
              f"{point.mean_seconds_per_frame * 1000:.0f} ms/frame")
        print("  " + RATE_HEADER)
        for name, result in point.methods.items():
            print("  " + rate_row(name, result.n_correct, result.n_false, result.n_none,
                                  result.n_trials))

    print()
    print("  summary")
    print(f"  {'n false':>8} {'ceiling':>8} {'tri ident':>10} {'tri false':>10} "
          f"{'pyr ident':>10} {'pyr false':>10} {'pyr ident/ceiling':>18}")
    for n_false in FALSE_COUNTS:
        p = points[n_false]
        t, y = p.methods["triangle"], p.methods["pyramid"]
        share = y.identification_rate / p.ceiling if p.ceiling > 0 else float("nan")
        print(f"  {n_false:8d} {p.ceiling:8.4f} {t.identification_rate:10.4f} "
              f"{t.false_identification_rate:10.4f} {y.identification_rate:10.4f} "
              f"{y.false_identification_rate:10.4f} {share:18.4f}")

    print()
    print("  The two rules fail in opposite directions. The triangle rule keeps identifying")
    print("  and starts being wrong; the Pyramid rule keeps being right and stops")
    print("  identifying. The last column is the fraction of the achievable frames the")
    print("  Pyramid rule actually takes, and it is where a better decision rule has room.")

    worst_tri = max(points[n].methods["triangle"].false_identification_rate for n in FALSE_COUNTS)
    passed.append(
        verdict("worst triangle-rule false-identification rate over the sweep",
                worst_tri, 0.10, mode=">=")
    )
    worst_pyr = max(points[n].methods["pyramid"].false_identification_rate for n in FALSE_COUNTS)
    passed.append(
        verdict("worst pyramid-rule false-identification rate over the sweep", worst_pyr, 0.02)
    )
    passed.append(
        verdict("pyramid identification rate at 20 false stars (both methods fail here)",
                points[20].methods["pyramid"].identification_rate, 0.10)
    )
    passed.append(
        verdict("search ceiling at 20 false stars", points[20].ceiling, 0.20)
    )

    print()
    banner("VALIDATION 4c: is the collapse a search failure or a decision failure?")
    print("    if the ceiling is near zero, no decision rule -- classical, learned, or")
    print("    otherwise -- can recover the frame, because the correct catalogue triangle")
    print("    was never proposed")
    print(f"{'n false':>9} {'ceiling':>9} {'best possible ident':>21} {'pyramid shortfall':>19}")
    for n_false in FALSE_COUNTS:
        p = points[n_false]
        y = p.methods["pyramid"]
        print(f"{n_false:9d} {p.ceiling:9.4f} {p.ceiling:21.4f} "
              f"{p.ceiling - y.identification_rate:19.4f}")
    report("ceiling at 30 false stars", points[30].ceiling)
    print("    Below a ceiling of about 0.1 the frame is unrecoverable in principle: the")
    print("    brightest ten spots no longer contain three real stars often enough for any")
    print("    triangle to exist. This is the documented failure regime, and it is a")
    print("    property of the frame, not of the matcher.")
    passed.append(verdict("ceiling at 30 false stars", points[30].ceiling, 0.05))

    print()
    banner("VALIDATION 4d: how many spots should the matcher use?")
    print("    10 false stars, sigma = 5 arcsec, varying how many spots the matcher uses")
    print("    The first version of this section was written expecting that more spots would")
    print("    not help, because the false detections are brighter on average and the scan is")
    print("    capped at 25 triples. The measurement says otherwise, and the measurement is")
    print("    what is reported. See validation/VALIDATION.md section 7.")
    print(f"{'max spots':>10} {'real kept':>10} {'triples':>9} {'ceiling':>8} {'pyr ident':>10} "
          f"{'pyr false':>10} {'ms/frame':>9}")
    for max_stars in (6, 8, 10, 14, 20):
        cfg = SceneConfig(
            camera=cam, centroid_sigma_arcsec=5.0, n_false_stars=10, max_stars=max_stars
        )
        p = run_trials(cat, table, cam, cfg, 150, SEED + 12, with_attitude=False)
        y = p.methods["pyramid"]
        n_triples = min(25, max_stars * (max_stars - 1) * (max_stars - 2) // 6)
        print(f"{max_stars:10d} {p.mean_true_spots:10.2f} {n_triples:9d} "
              f"{p.ceiling:8.4f} {y.identification_rate:10.4f} "
              f"{y.false_identification_rate:10.4f} {p.mean_seconds_per_frame * 1000:9.0f}")
    print("    Using more spots does rescue this regime, and by a lot: at 20 spots the")
    print("    ceiling recovers to near 1 and the Pyramid rule identifies most frames. The")
    print("    reason is truncation, not search: with 10 spots the ten brightest are mostly")
    print("    the false detections and only ~3.5 real stars survive, while with 20 spots")
    print("    ~11 real stars do. The 25-triple cap does not block this, because the Pyramid")
    print("    scan order reaches high spot indices early. The costs are a longer spot list")
    print("    to centroid and, at low false-star counts, more compute for no gain -- so the")
    print("    package default stays at 10 and this table is the reason to raise it.")

    return finish(passed)


if __name__ == "__main__":
    raise SystemExit(main())
