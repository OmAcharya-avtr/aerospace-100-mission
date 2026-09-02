"""Validation 2: the synthetic catalogue and the pair table.

The catalogue is generated, not measured, so "validating" it means two things
and not a third. It means checking that the generator reproduces the model it
claims to implement (Eq. C1, Eq. C2, isotropy), and that the derived
structures agree with their closed forms (Eq. C3, Eq. P1). It does **not**
mean checking the model against the real sky, which it is not fitted to and
does not claim to reproduce. ``DATASET_CARD.md`` says so in more detail.

The quantities that matter downstream and are measured here:

* star count against Eq. C1, over five magnitude limits;
* the magnitude distribution against Eq. C2 by Kolmogorov-Smirnov;
* isotropy: the dipole moment of the star positions against its sampling
  distribution;
* close-pair removal against Eq. C3;
* pair-table size against Eq. P1, and its build time and memory, which is the
  actual cost of a magnitude limit;
* pair-table correctness against brute force on a small catalogue;
* stars in the field and the fraction of pointings with fewer than four --
  the failure mode that no identification algorithm can do anything about.

Run: ``python validation/validate_catalogue.py``
"""

from __future__ import annotations

import time

import numpy as np

from _common import CAMERA_FOV_DEG, CAMERA_PIXELS, SEED, banner, finish, report, verdict

from skymatch.camera import CameraModel
from skymatch.catalogue import (
    DEFAULT_SLOPE,
    expected_close_pairs,
    generate_catalogue,
    predicted_count,
    remove_close_pairs,
)
from skymatch.geometry import angular_separation, random_rotation
from skymatch.pairtable import PairTable, expected_pair_count
from skymatch.scene import SceneConfig, simulate_scene

MAG_LIMITS = (5.0, 5.5, 6.0, 6.5, 7.0)


def main() -> int:
    passed: list[bool] = []
    cam = CameraModel(fov_deg=CAMERA_FOV_DEG, pixels=CAMERA_PIXELS)

    banner("VALIDATION 2a: star count against Eq. C1")
    print("    N(<m) = 4800 * 10^(0.52 (m - 6)), minus the count below m = -1.5")
    print(f"{'mag limit':>10} {'stars':>8} {'Eq. C1':>10} {'ratio':>8} {'growth/mag':>11}")
    worst = 0.0
    previous = None
    for limit in MAG_LIMITS:
        cat = generate_catalogue(limit, seed=SEED)
        pred = predicted_count(limit)
        ratio = cat.n_stars / pred
        worst = max(worst, abs(ratio - 1.0))
        growth = "" if previous is None else f"{cat.n_stars / previous:11.3f}"
        print(f"{limit:10.2f} {cat.n_stars:8d} {pred:10.1f} {ratio:8.5f} {growth:>11}")
        previous = cat.n_stars
    print(f"    expected growth per 0.5 mag = 10^(0.52 * 0.5) = {10 ** (DEFAULT_SLOPE * 0.5):.4f}")
    passed.append(verdict("worst |count / Eq. C1 - 1| (rounding only)", worst, 1e-3))

    print()
    banner("VALIDATION 2b: magnitude distribution against Eq. C2")
    cat = generate_catalogue(6.5, seed=SEED)
    m = np.sort(cat.magnitude)
    lo = 10.0 ** (DEFAULT_SLOPE * -1.5)
    hi = 10.0 ** (DEFAULT_SLOPE * 6.5)
    cdf = (10.0 ** (DEFAULT_SLOPE * m) - lo) / (hi - lo)
    empirical = np.arange(1, m.size + 1) / m.size
    ks = float(np.max(np.abs(cdf - empirical)))
    critical = 1.36 / np.sqrt(m.size)
    report("Kolmogorov-Smirnov statistic", ks)
    report("5% critical value 1.36/sqrt(N)", critical)
    passed.append(verdict("KS statistic vs the 5% critical value", ks, critical))
    for q in (0.1, 0.5, 0.9):
        target = np.log10(lo + q * (hi - lo)) / DEFAULT_SLOPE
        print(f"    quantile {q:.1f}: sampled {np.quantile(m, q):7.4f}, Eq. C2 {target:7.4f}")

    print()
    banner("VALIDATION 2c: isotropy of the sky positions")
    print("    The dipole |sum v| / N of N isotropic directions is a random variable of")
    print("    order 1/sqrt(N). Rather than quote a closed form, the check is against an")
    print("    empirical null built from 200 independent isotropic samples of the same size.")
    cat = generate_catalogue(6.0, seed=SEED)
    dipole = float(np.linalg.norm(cat.vectors.sum(axis=0)) / cat.n_stars)
    rng = np.random.default_rng(SEED + 1)
    null = []
    for _ in range(200):
        v = rng.normal(size=(cat.n_stars, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        null.append(float(np.linalg.norm(v.sum(axis=0)) / cat.n_stars))
    null_arr = np.array(null)
    percentile = float((null_arr < dipole).mean())
    report("catalogue dipole |sum v| / N", dipole)
    report("null 95th percentile", float(np.quantile(null_arr, 0.95)))
    report("percentile of the catalogue in the null", percentile)
    p95 = float(np.quantile(null_arr, 0.95))
    passed.append(verdict("dipole below the null 95th percentile", dipole, p95))
    counts, _ = np.histogram(np.sin(cat.dec), bins=10, range=(-1.0, 1.0))
    chi2 = float(np.sum((counts - cat.n_stars / 10.0) ** 2 / (cat.n_stars / 10.0)))
    report("chi-square of sin(dec) over 10 equal-area bands (9 dof)", chi2)
    passed.append(verdict("chi-square vs the 1% critical value 21.67", chi2, 21.67))

    print()
    banner("VALIDATION 2d: close-pair removal against Eq. C3")
    print("    E[pairs closer than theta] = C(N,2) (1 - cos theta) / 2")
    print(f"{'mag limit':>10} {'sep [deg]':>10} {'removed':>10} {'pairs':>8} {'Eq. C3':>10}")
    for limit, sep_deg in ((6.0, 0.02), (6.5, 0.05), (7.0, 0.10)):
        cat = generate_catalogue(limit, seed=SEED)
        prepared, removed = remove_close_pairs(cat, np.radians(sep_deg))
        pairs = removed // 2
        expected = expected_close_pairs(cat.n_stars, np.radians(sep_deg))
        print(f"{limit:10.2f} {sep_deg:10.3f} {removed:10d} {pairs:8d} {expected:10.3f}")
        assert prepared.n_stars == cat.n_stars - removed
    print("    counts this small are Poisson; the check is that a prepared catalogue has")
    print("    NO pair below the threshold, which is exact")
    prepared, _ = remove_close_pairs(generate_catalogue(7.0, seed=SEED), np.radians(0.10))
    table = PairTable(prepared, np.radians(0.5))
    passed.append(
        verdict("smallest separation surviving preparation [deg]",
                float(np.degrees(table.separations.min())), 0.10, mode=">=")
    )

    print()
    banner("VALIDATION 2e: pair-table size, cost, and Eq. P1")
    print(f"    max pair separation {np.degrees(cam.max_separation_rad):.4f} deg "
          f"(twice the {np.degrees(cam.half_diagonal_rad):.4f} deg half-diagonal)")
    print(f"{'mag limit':>10} {'stars':>7} {'pairs':>10} {'Eq. P1':>10} {'ratio':>8} "
          f"{'MB':>7} {'build s':>8}")
    worst_ratio = 0.0
    for limit in MAG_LIMITS:
        cat = generate_catalogue(limit, seed=SEED)
        t0 = time.perf_counter()
        table = PairTable(cat, cam.max_separation_rad)
        build = time.perf_counter() - t0
        pred = expected_pair_count(cat.n_stars, cam.max_separation_rad)
        ratio = table.n_pairs / pred
        worst_ratio = max(worst_ratio, abs(ratio - 1.0))
        print(f"{limit:10.2f} {cat.n_stars:7d} {table.n_pairs:10d} {pred:10.0f} {ratio:8.5f} "
              f"{table.nbytes / 1e6:7.2f} {build:8.3f}")
    print("    the count is a random variable; the deviation is a few times 1/sqrt(P)")
    passed.append(verdict("worst |pairs / Eq. P1 - 1| over 5 magnitude limits", worst_ratio, 0.02))

    print()
    banner("VALIDATION 2f: pair-table queries against brute force")
    small = generate_catalogue(4.5, seed=SEED)
    table = PairTable(small, cam.max_separation_rad)
    full = np.arccos(np.clip(small.vectors @ small.vectors.T, -1.0, 1.0))
    iu = np.triu_indices(small.n_stars, k=1)
    brute = full[iu]
    brute_pairs = int(np.sum(brute <= cam.max_separation_rad))
    print(f"    catalogue of {small.n_stars} stars, {brute_pairs} pairs by brute force")
    delta = abs(table.n_pairs - brute_pairs)
    passed.append(verdict("|table pairs - brute force pairs|", delta, 0.5))

    rng = np.random.default_rng(SEED + 2)
    worst_lookup = 0.0
    for _ in range(200):
        a, b = rng.integers(0, small.n_stars, size=2)
        if a == b:
            continue
        got = float(table.separation_lookup(np.array([a]), np.array([b]))[0])
        truth = float(angular_separation(small.vectors[a : a + 1], small.vectors[b : b + 1])[0])
        if truth <= cam.max_separation_rad:
            worst_lookup = max(worst_lookup, abs(got - truth))
        else:
            worst_lookup = max(worst_lookup, 0.0 if np.isnan(got) else 1.0)
    passed.append(verdict("max |separation_lookup - truth| [rad]", worst_lookup, 1e-14))

    worst_range = 0
    for _ in range(60):
        centre = float(rng.uniform(0.02, cam.max_separation_rad - 0.02))
        width = 2e-3
        a, b = table.ordered_range(centre - width, centre + width)
        got = {(int(x), int(y)) for x, y in zip(a, b, strict=True) if x < y}
        want = {
            (int(iu[0][t]), int(iu[1][t]))
            for t in np.flatnonzero(np.abs(brute - centre) <= width)
        }
        worst_range = max(worst_range, len(got ^ want))
    passed.append(verdict("worst set difference, ordered_range vs brute force", worst_range, 0.5))

    worst_nb = 0
    for _ in range(60):
        star = int(rng.integers(0, small.n_stars))
        centre = float(rng.uniform(0.02, cam.max_separation_rad - 0.02))
        width = 5e-3
        _, nb = table.neighbours_range(np.array([star]), centre - width, centre + width)
        want = {
            int(o)
            for o in np.flatnonzero(np.abs(full[star] - centre) <= width)
            if o != star and full[star, o] <= cam.max_separation_rad
        }
        worst_nb = max(worst_nb, len(set(nb.tolist()) ^ want))
    passed.append(verdict("worst set difference, neighbours_range vs brute force", worst_nb, 0.5))

    print()
    banner("VALIDATION 2g: stars in the field, and the frames nothing can solve")
    print(f"    {cam.fov_deg:.1f} deg field, solid angle {cam.solid_angle_sqdeg:.2f} sq.deg; "
          "600 uniform random pointings per magnitude limit")
    print(f"{'mag limit':>10} {'mean in field':>14} {'predicted':>10} {'P(<4 stars)':>12} "
          f"{'P(<4 spots)':>12}")
    worst_pred = 0.0
    for limit in MAG_LIMITS:
        cat = generate_catalogue(limit, seed=SEED)
        cfg = SceneConfig(camera=cam, centroid_sigma_arcsec=5.0)
        rng = np.random.default_rng(SEED + 3)
        in_field, spots = [], []
        for _ in range(600):
            scene = simulate_scene(cat, cfg, rng, attitude=random_rotation(rng))
            in_field.append(scene.n_in_field)
            spots.append(scene.n_spots)
        pred = cat.expected_in_solid_angle(cam.solid_angle_sr)
        mean = float(np.mean(in_field))
        worst_pred = max(worst_pred, abs(mean / pred - 1.0))
        print(f"{limit:10.2f} {mean:14.3f} {pred:10.3f} "
              f"{float(np.mean(np.array(in_field) < 4)):12.4f} "
              f"{float(np.mean(np.array(spots) < 4)):12.4f}")
    print("    a frame with fewer than four spots cannot produce a pyramid, whatever the")
    print("    decision rule; this is the floor every identification rate below sits on")
    passed.append(verdict("worst |mean in field / density x solid angle - 1|", worst_pred, 0.05))

    return finish(passed)


if __name__ == "__main__":
    raise SystemExit(main())
