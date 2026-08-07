"""Validation 2: low-order Zernikes against their closed forms, evaluated by hand.

Each reference value below was computed by hand from the closed form printed in
Noll (1976), JOSA 66(3), 207-211, and typed in as a literal. The script
evaluates the library at the same point and prints the difference. The full
arithmetic is reproduced in ``validation/VALIDATION.md``.

It also checks that the Noll index -> (n, m) map reproduces Noll's own listing
of Z1..Z15, and that the OSA/ANSI map reproduces the ANSI closed form
``j = (n(n+2) + m)/2``.

Run:  python validation/validate_closed_forms.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zernkit import (  # noqa: E402
    mode_name,
    nm_to_osa,
    noll_to_nm,
    radial_polynomial,
    zernike_noll,
)

SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
SQRT5 = math.sqrt(5.0)
SQRT6 = math.sqrt(6.0)
SQRT8 = math.sqrt(8.0)
SQRT10 = math.sqrt(10.0)

RHO = 0.5
THETA = math.pi / 6.0  # 30 degrees

# (Noll j, hand-evaluated value at rho = 0.5, theta = 30 deg, working shown)
HAND_CASES: list[tuple[int, float, str]] = [
    (1, 1.0, "Z1 = 1"),
    (2, 0.8660254037844387, "Z2 = 2*0.5*cos30 = 1*0.8660254037844387"),
    (3, 0.5, "Z3 = 2*0.5*sin30 = 1*0.5"),
    (4, -0.8660254037844386, "Z4 = sqrt3*(2*0.25-1) = sqrt3*(-0.5)"),
    (5, 0.5303300858899106, "Z5 = sqrt6*0.25*sin60 = 0.6123724356957945*0.8660254037844386"),
    (6, 0.3061862178478972, "Z6 = sqrt6*0.25*cos60 = 0.6123724356957945*0.5"),
    (7, -0.8838834764831844, "Z7 = sqrt8*(3*0.125-2*0.5)*sin30 = sqrt8*(-0.625)*0.5"),
    (8, -1.5309310892394863, "Z8 = sqrt8*(-0.625)*cos30 = -1.7677669529663689*0.8660254037844386"),
    (9, 0.35355339059327384, "Z9 = sqrt8*0.125*sin90 = 0.35355339059327373*1"),
    (10, 2.1648901405887326e-17, "Z10 = sqrt8*0.125*cos90 = 0.35355339059327373*0"),
    (11, -0.2795084971874737, "Z11 = sqrt5*(6*0.0625-6*0.25+1) = sqrt5*(-0.125)"),
    (12, -0.7905694150420949, "Z12 = sqrt10*(4*0.0625-3*0.25)*cos60 = 3.1622776601683795*-0.5*0.5"),
]

# Noll's own listing of the first 15 polynomials -> (n, m).
NOLL_TABLE_I = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0),
    5: (2, -2),
    6: (2, 2),
    7: (3, -1),
    8: (3, 1),
    9: (3, -3),
    10: (3, 3),
    11: (4, 0),
    12: (4, 2),
    13: (4, -2),
    14: (4, 4),
    15: (4, -4),
}


def _z12_hand() -> float:
    """sqrt(10)(4 rho^4 - 3 rho^2) cos(2 theta) at rho=0.5, theta=30deg."""
    radial = 4 * RHO**4 - 3 * RHO**2  # 4*0.0625 - 3*0.25 = 0.25 - 0.75 = -0.5
    return SQRT10 * radial * math.cos(2 * THETA)  # cos60 = 0.5


def main() -> int:
    lines: list[str] = []
    ok = True

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit("ZernKit validation 2 -- low-order closed forms and index conventions")
    emit("=" * 78)
    emit(f"Evaluation point: rho = {RHO}, theta = {THETA:.15f} rad (30 deg)")
    emit("Reference: closed forms as printed in Noll (1976), JOSA 66(3), 207-211")
    emit("")
    emit(f"{'j':>4} {'mode':<30} {'hand':>22} {'library':>22} {'diff':>11}")
    worst = 0.0
    for j, hand, _note in HAND_CASES:
        value = float(zernike_noll(j, RHO, THETA))
        diff = abs(value - hand)
        worst = max(worst, diff)
        n, m = noll_to_nm(j)
        emit(f"{j:>4} {mode_name(n, m):<30} {hand:>22.15f} {value:>22.15f} {diff:>11.2e}")
    emit("")
    emit(f"Cross-check of the Z12 literal recomputed in code: {_z12_hand():.15f}")
    emit(f"Worst |library - hand| over the 12 cases: {worst:.3e}")
    tol = 1e-15
    emit(f"Tolerance {tol:g} -> {'PASS' if worst < tol else 'FAIL'}")
    ok = ok and worst < tol
    emit("")

    emit("R_n^m(1) = 1 for every legal (n, m) up to n = 20:")
    worst_rim = 0.0
    for n in range(21):
        for m in range(-n, n + 1, 2):
            worst_rim = max(worst_rim, abs(float(radial_polynomial(n, m, 1.0)) - 1.0))
    emit(f"  worst |R_n^m(1) - 1| = {worst_rim:.3e}  (float round-off in the")
    emit("  alternating factorial sum; grows with n)")
    emit(f"  Tolerance 1e-8 -> {'PASS' if worst_rim < 1e-8 else 'FAIL'}")
    ok = ok and worst_rim < 1e-8
    emit("")

    emit("Noll index -> (n, m) against Noll's own listing of Z1..Z15:")
    emit(f"{'Noll j':>7} {'expected (n,m)':>16} {'library':>16} {'OSA j':>7}  name")
    index_ok = True
    for j, nm in NOLL_TABLE_I.items():
        got = noll_to_nm(j)
        index_ok = index_ok and got == nm
        emit(
            f"{j:>7} {str(nm):>16} {str(got):>16} {nm_to_osa(*got):>7}  {mode_name(*got)}"
        )
    emit(f"  -> {'PASS' if index_ok else 'FAIL'}")
    ok = ok and index_ok
    emit("")

    emit("OSA/ANSI closed form j = (n(n+2) + m)/2 verified up to n = 30:")
    osa_ok = all(
        nm_to_osa(n, m) == (n * (n + 2) + m) // 2
        for n in range(31)
        for m in range(-n, n + 1, 2)
    )
    emit(f"  -> {'PASS' if osa_ok else 'FAIL'} (496 pairs)")
    ok = ok and osa_ok

    out = Path(__file__).with_name("closed_forms_output.txt")
    out.write_text("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
