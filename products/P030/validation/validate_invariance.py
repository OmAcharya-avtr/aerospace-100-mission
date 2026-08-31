"""Rotational invariance of the keep-out verdict, and window-search consistency.

A keep-out verdict is a statement about relative geometry. If the boresight and
every cone axis are rotated by the same element of SO(3), nothing physical has
changed and the verdict, the margins and the allowed solid angle must all be
unchanged. These checks measure by how much they are not, in double precision.

Checks
------
1. Per-cone margins under 20 000 random rotations drawn from the Haar measure
   on SO(3): worst absolute change [rad].
2. Discrete violation verdicts under the same rotations, restricted to
   boresights at least 1e-7 rad clear of every boundary.
3. Verdicts with no clearance restriction, to measure how often a boresight
   sitting essentially on a boundary flips.
4. Allowed solid angle of a three-cone set under 200 random rotations.
5. Window search: refined boundaries have zero margin, and the windows agree
   with a direct margin test at 1 s resolution over one orbit.

Run from products/P030/:  python validation/validate_invariance.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from keepout import (  # noqa: E402
    ExclusionCone,
    KeepOutSet,
    OrbitPointingProblem,
    allowed_solid_angle,
    random_rotations,
    unit,
)


def random_set(rng, n_cones: int) -> KeepOutSet:
    cones = []
    for i in range(n_cones):
        axis = unit(rng.normal(size=3))
        r = rng.uniform(np.radians(1.0), np.radians(100.0))
        cones.append(ExclusionCone(axis, r, f"c{i}"))
    return KeepOutSet(tuple(cones))


def check_margins(n_trials: int = 20_000, seed: int = 11) -> float:
    print(f"1. Margin invariance under {n_trials} random rotations from SO(3)")
    rng = np.random.default_rng(seed)
    rots = random_rotations(n_trials, seed=seed + 1)
    worst = 0.0
    worst_detail = None
    for k in range(n_trials):
        ks = random_set(rng, 3)
        b = unit(rng.normal(size=3))
        r = rots[k]
        before = ks.margins(b)
        after = ks.rotated(r).margins(r @ b)
        diff = float(np.max(np.abs(before - after)))
        if diff > worst:
            worst = diff
            worst_detail = (np.degrees(before), np.degrees(after))
    print(f"   worst |margin change| = {worst:.3e} rad = "
          f"{np.degrees(worst) * 3600e3:.3e} milliarcsec")
    print(f"   at margins (deg) before = {np.array2string(worst_detail[0], precision=9)}")
    print(f"                     after = {np.array2string(worst_detail[1], precision=9)}")
    print(f"   tolerance 1e-12 rad   {'PASS' if worst < 1e-12 else 'FAILED'}\n")
    return worst


def check_verdicts(n_trials: int = 20_000, seed: int = 23) -> tuple[int, int, int]:
    print(f"2 & 3. Verdict invariance under {n_trials} random rotations")
    rng = np.random.default_rng(seed)
    rots = random_rotations(n_trials, seed=seed + 1)
    flips_all = 0
    flips_clear = 0
    n_clear = 0
    smallest_flip_margin = np.inf
    for k in range(n_trials):
        ks = random_set(rng, 3)
        b = unit(rng.normal(size=3))
        r = rots[k]
        clearance = float(np.min(np.abs(ks.margins(b))))
        same = set(ks.violations(b)) == set(ks.rotated(r).violations(r @ b))
        if not same:
            flips_all += 1
            smallest_flip_margin = min(smallest_flip_margin, clearance)
        if clearance > 1e-7:
            n_clear += 1
            if not same:
                flips_clear += 1
    print(f"   boresights at least 1e-7 rad clear of every boundary: {n_clear} of {n_trials}")
    print(f"   verdict changes among those                         : {flips_clear}")
    print(f"   verdict changes over all {n_trials} trials              : {flips_all}")
    if flips_all:
        print(f"   smallest clearance at which a verdict changed       : "
              f"{smallest_flip_margin:.3e} rad")
    print(f"   tolerance: 0 changes among cleared boresights   "
          f"{'PASS' if flips_clear == 0 else 'FAILED'}\n")
    return flips_clear, flips_all, n_clear


def check_solid_angle(n_trials: int = 200, seed: int = 31) -> float:
    print(f"4. Allowed solid angle invariance under {n_trials} random rotations")
    ks = KeepOutSet(
        (
            ExclusionCone([1.0, 0.0, 0.0], np.radians(45.0), "sun"),
            ExclusionCone([-0.2, -0.9, 0.3], np.radians(77.0), "earth"),
            ExclusionCone([0.1, 0.3, 0.95], np.radians(15.0), "moon"),
        )
    )
    base = allowed_solid_angle(ks).solid_angle
    rots = random_rotations(n_trials, seed=seed)
    values = np.array([allowed_solid_angle(ks.rotated(r)).solid_angle for r in rots])
    worst = float(np.max(np.abs(values - base)))
    print(f"   unrotated     = {base:.15f} sr")
    print(f"   rotated mean  = {values.mean():.15f} sr")
    print(f"   rotated range = [{values.min():.15f}, {values.max():.15f}] sr")
    print(f"   worst |change| = {worst:.3e} sr   tolerance 1e-9   "
          f"{'PASS' if worst < 1e-9 else 'FAILED'}\n")
    return worst


def check_windows() -> tuple[float, bool]:
    print("5. Window search: refined boundaries and agreement with a dense margin test")
    problem = OrbitPointingProblem(
        epoch_jd=2461119.5,  # 2026-03-20 00:00 UTC
        altitude_m=550e3,
        inclination=np.radians(97.6),
        raan=0.4,
        sun_exclusion=np.radians(45.0),
        earth_exclusion=np.radians(10.0),
        moon_exclusion=np.radians(15.0),
    )
    target = unit([0.2, -0.9, 0.3])
    period = problem.period
    t_scan = np.arange(0.0, 2.0 * period, 20.0)
    windows = problem.windows(t_scan, target, refine=True)
    print(f"   orbital period = {period:.3f} s ({period / 60.0:.3f} min)")
    print(f"   scan 0 to {2 * period:.1f} s at 20 s, {len(t_scan)} samples")
    print(f"   {len(windows)} windows found")
    worst_boundary = 0.0
    for w in windows:
        for tb, kind in ((w.start, "start"), (w.end, "end")):
            if 1e-6 < tb < t_scan[-1] - 1e-6:
                m = abs(problem.margin(tb, target))
                worst_boundary = max(worst_boundary, m)
                print(f"     {kind:>5} = {tb:12.6f} s   |margin| = {m:.3e} rad")
            else:
                print(f"     {kind:>5} = {tb:12.6f} s   (scan edge, not refined)")
        print(f"     duration = {w.duration:.3f} s")
    print(f"   worst |margin| at a refined boundary = {worst_boundary:.3e} rad"
          f"   tolerance 1e-7   {'PASS' if worst_boundary < 1e-7 else 'FAILED'}")

    # Only inside the coarse scan's own span: samples past t_scan[-1] were
    # never examined by the window search, so comparing them is meaningless.
    t_dense = np.arange(0.0, t_scan[-1] + 0.5, 1.0)
    direct = problem.margin_series(t_dense, target) >= 0.0
    from_windows = np.zeros_like(t_dense, dtype=bool)
    for w in windows:
        from_windows |= (t_dense >= w.start) & (t_dense <= w.end)
    n_disagree = int(np.count_nonzero(direct != from_windows))
    print(f"   dense 1 s check over {len(t_dense)} samples: {n_disagree} disagreements"
          f"   {'PASS' if n_disagree == 0 else 'FAILED'}\n")
    return worst_boundary, n_disagree == 0


def main() -> None:
    print("=" * 88)
    print("KeepOut validation: rotational invariance and window-search consistency")
    print("=" * 88)
    print()
    w1 = check_margins()
    flips_clear, _, _ = check_verdicts()
    w4 = check_solid_angle()
    wb, dense_ok = check_windows()
    passed = w1 < 1e-12 and flips_clear == 0 and w4 < 1e-9 and wb < 1e-7 and dense_ok
    print("=" * 88)
    print(f"OVERALL: {'PASS' if passed else 'FAILED'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
