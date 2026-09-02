"""Reaction-wheel torque and momentum envelopes.

PART A  the zonotope facet formula against linear programming
PART B  hand-checked capabilities for the orthogonal triad and the pyramid
PART C  the minimum-norm allocation is a strict lower bound on the envelope
PART D  pyramid isotropy: how spherical the envelope is, against elevation
PART E  a failed wheel: what the array can still do

Run: ``python validation/validate_wheel_envelope.py``
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slewforge.wheels import WheelArray, orthogonal_wheels, pyramid_wheels  # noqa: E402


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-equal-area directions on the unit sphere (Gonzalez 2010)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)


def main() -> int:
    failures = 0

    # ---------------------------------------------------------------- PART A
    rule("PART A -- zonotope facet formula against linear programming")
    print("cap(u) = max{t >= 0 : t u = A c, |c_i| <= bound} is a linear programme.")
    print("Because {A c : |c| <= bound} is a zonotope, every facet is spanned by a")
    print("pair of generators, so cap(u) = min_f bound h_f / |u . n_f| with")
    print("n_f = a_i x a_j normalised and h_f = sum_k |a_k . n_f|. The two routes")
    print("share no code: one is HiGHS simplex, the other is six dot products.")
    print()
    rng = np.random.default_rng(20260902)
    arrays = {
        "orthogonal triad": orthogonal_wheels(1.0, 2.0),
        "pyramid 54.7356 deg": pyramid_wheels(0.15, 12.0),
        "pyramid 40 deg": pyramid_wheels(0.15, 12.0, elevation=math.radians(40.0)),
        "5 random axes": WheelArray(rng.normal(size=(5, 3)), 0.2, 8.0, name="random5"),
        "6 random axes": WheelArray(rng.normal(size=(6, 3)), 0.2, 8.0, name="random6"),
    }
    print(f"{'array':<24}{'n dirs':>8}{'worst |zono - LP|':>22}{'zono [us]':>12}{'LP [us]':>12}")
    worst_all = 0.0
    for name, arr in arrays.items():
        dirs = fibonacci_sphere(300)
        t0 = time.perf_counter()
        zono = np.array([arr.max_torque_along(d) for d in dirs])
        t_z = (time.perf_counter() - t0) / len(dirs) * 1e6
        t0 = time.perf_counter()
        lp = np.array([arr.max_torque_along_lp(d) for d in dirs])
        t_l = (time.perf_counter() - t0) / len(dirs) * 1e6
        worst = float(np.max(np.abs(zono - lp)))
        worst_all = max(worst_all, worst)
        print(f"{name:<24}{len(dirs):>8}{worst:>22.6e}{t_z:>12.2f}{t_l:>12.2f}")
    print(f"\nworst over all arrays : {worst_all:.6e} N*m   tolerance 1e-12")
    failures += worst_all > 1e-12

    # ---------------------------------------------------------------- PART B
    rule("PART B -- hand-checked capabilities")
    triad = orthogonal_wheels(1.0, 2.0)
    print("Orthogonal triad, A = I, per-wheel torque 1 N*m. The reachable set is")
    print("the cube [-1, 1]^3, so along +x the capability is 1 and along (1,1,1)")
    print("it is sqrt(3) (the cube's corner).")
    cases = [
        ("+x", np.array([1.0, 0.0, 0.0]), 1.0),
        ("(1,1,0)", np.array([1.0, 1.0, 0.0]), math.sqrt(2.0)),
        ("(1,1,1)", np.ones(3), math.sqrt(3.0)),
    ]
    for label, u, hand in cases:
        got = triad.max_torque_along(u)
        print(f"  {label:<10} library {got!r}  hand {hand!r}  |diff| {abs(got - hand):.3e}")
        failures += abs(got - hand) > 1e-12

    print()
    el = math.atan(math.sqrt(2.0))
    print(f"Pyramid at arctan(sqrt 2) = {math.degrees(el):.10f} deg, torque 0.15 N*m.")
    print("Along +z all four wheels contribute cos(el) each, so the capability is")
    print("4 * 0.15 * cos(el):")
    pyr = pyramid_wheels(0.15, 12.0)
    hand_z = 4.0 * 0.15 * math.cos(el)
    got_z = pyr.max_torque_along([0.0, 0.0, 1.0])
    print(f"  hand    {hand_z!r}")
    print(f"  library {got_z!r}   |diff| {abs(got_z - hand_z):.3e}   tolerance 1e-12")
    failures += abs(got_z - hand_z) > 1e-12
    print("Along +x only the two wheels with an x component contribute, each")
    print("sin(el) cos(0) or sin(el) cos(pi), so the capability is 2 * 0.15 * sin(el):")
    hand_x = 2.0 * 0.15 * math.sin(el)
    got_x = pyr.max_torque_along([1.0, 0.0, 0.0])
    print(f"  hand    {hand_x!r}")
    print(f"  library {got_x!r}   |diff| {abs(got_x - hand_x):.3e}   tolerance 1e-12")
    failures += abs(got_x - hand_x) > 1e-12

    # ---------------------------------------------------------------- PART C
    rule("PART C -- the minimum-norm allocation is a lower bound on the envelope")
    print("The planner sizes against the minimum-norm allocation, because that is")
    print("what a controller using a plain pseudo-inverse delivers. The exact")
    print("envelope is larger, and the ratio below is the headroom a bounded")
    print("allocator (AllocLab, P023) would recover. It is reported, not assumed.")
    print()
    print(f"{'array':<24}{'min ratio':>12}{'mean ratio':>12}{'max ratio':>12}{'never > 1':>12}")
    dirs = fibonacci_sphere(2000)
    for name, arr in arrays.items():
        exact = np.array([arr.max_torque_along(d) for d in dirs])
        pinv = np.array([arr.pseudo_inverse_capability(d) for d in dirs])
        ratio = pinv / exact
        ok = bool(np.all(ratio <= 1.0 + 1e-12))
        print(
            f"{name:<24}{float(np.min(ratio)):>12.6f}{float(np.mean(ratio)):>12.6f}"
            f"{float(np.max(ratio)):>12.6f}{str(ok):>12}"
        )
        failures += not ok

    # ---------------------------------------------------------------- PART D
    rule("PART D -- pyramid isotropy against elevation")
    print("The four-wheel pyramid's momentum envelope is closest to a sphere at")
    print("elevation arctan(sqrt 2) = 54.7356 deg (Wie 2008, Sec. 7.3). Measured")
    print("as the ratio of the smallest to the largest directional capability over")
    print("2000 near-equal-area directions:")
    print()
    print(f"{'elevation [deg]':>18}{'min/max capability':>22}{'min [N*m]':>14}{'max [N*m]':>14}")
    best = (None, -1.0)
    for el_deg in (30.0, 40.0, 50.0, 54.7356103172, 60.0, 70.0, 80.0):
        arr = pyramid_wheels(0.15, 12.0, elevation=math.radians(el_deg))
        caps = np.array([arr.max_torque_along(d) for d in dirs])
        r = float(np.min(caps) / np.max(caps))
        if r > best[1]:
            best = (el_deg, r)
        print(f"{el_deg:>18.6f}{r:>22.6f}{float(np.min(caps)):>14.6f}{float(np.max(caps)):>14.6f}")
    print(f"\nmost isotropic of those sampled: {best[0]} deg, ratio {best[1]:.6f}")
    print("(The optimum over this grid is the textbook angle; a finer sweep would")
    print("be needed to claim more than that.)")
    failures += abs(best[0] - 54.7356103172) > 1e-9

    # ---------------------------------------------------------------- PART E
    rule("PART E -- a failed wheel")
    print("Removing one wheel from the pyramid leaves a rank-3 array with a much")
    print("smaller envelope. The capability along each body axis, before and after:")
    print()
    full = pyramid_wheels(0.15, 12.0)
    reduced = WheelArray(full.axes[1:], 0.15, 12.0, name="pyramid-1")
    print(f"rank full {full.rank}, rank reduced {reduced.rank}")
    print(f"{'direction':<12}{'full [N*m]':>14}{'3 wheels [N*m]':>18}{'ratio':>10}")
    for label, u in (("+x", [1, 0, 0]), ("+y", [0, 1, 0]), ("+z", [0, 0, 1]),
                     ("-x", [-1, 0, 0]), ("(1,1,1)", [1, 1, 1])):
        a = full.max_torque_along(u)
        b = reduced.max_torque_along(u)
        print(f"{label:<12}{a:>14.6f}{b:>18.6f}{b / a:>10.4f}")
    caps_full = np.array([full.max_torque_along(d) for d in dirs])
    caps_red = np.array([reduced.max_torque_along(d) for d in dirs])
    print(f"\nworst-direction capability: full {float(np.min(caps_full)):.6f} N*m, "
          f"3 wheels {float(np.min(caps_red)):.6f} N*m "
          f"({float(np.min(caps_red) / np.min(caps_full)) * 100:.2f} %)")
    print("The array is still rank 3, so no direction is unreachable, but the loss")
    print("is strongly direction-dependent: exactly half along x, y and z but none")
    print("along +y, where the surviving wheels already did all the work. That is")
    print("why the planner sizes each segment against the direction that segment")
    print("actually needs rather than against a single scalar torque budget.")
    failures += reduced.rank != 3 or float(np.min(caps_red)) <= 0.0

    rule("SUMMARY")
    print(f"failed checks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
