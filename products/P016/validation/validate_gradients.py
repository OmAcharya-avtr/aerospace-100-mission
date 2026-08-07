"""Validation 3: analytic Zernike gradients against high-accuracy finite differences.

Reference: Richardson-extrapolated central differences. The plain central
difference ``D(h) = (f(x+h) - f(x-h)) / (2h)`` has an error series in even
powers of ``h``, so two Richardson steps

    D1 = (4 D(h/2)  - D(h))  / 3        -> O(h^4)
    D2 = (4 D(h/4)  - D(h/2)) / 3       -> O(h^4)
    D3 = (16 D2 - D1) / 15              -> O(h^6)

give a reference accurate to ~1e-12 relative for the smooth polynomials here,
without the round-off blow-up of a single very small ``h``.

Also reports the closed-form hand checks:
  Noll j=2: Z = 2x           -> dZ/dx = 2,           dZ/dy = 0
  Noll j=4: Z = sqrt3(2rho^2 - 1) -> dZ/dx = 4 sqrt3 x, dZ/dy = 4 sqrt3 y
  Noll j=6: Z = sqrt6(x^2 - y^2)  -> dZ/dx = 2 sqrt6 x, dZ/dy = -2 sqrt6 y
  Noll j=7: Z = sqrt8(3rho^2 - 2) y -> at the origin dZ/dy = -2 sqrt8

Run:  python validation/validate_gradients.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zernkit import (  # noqa: E402
    mode_name,
    nm_to_noll,
    zernike_cartesian,
    zernike_gradient,
)

MAX_N = 10
SEED = 20260807
N_POINTS = 400


def _central(f, x, y, h, axis):  # type: ignore[no-untyped-def]
    if axis == 0:
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    return (f(x, y + h) - f(x, y - h)) / (2 * h)


def richardson_gradient(n: int, m: int, x, y, h: float = 1e-2):  # type: ignore[no-untyped-def]
    """Richardson-extrapolated central difference, O(h^6)."""

    def f(xx, yy):  # type: ignore[no-untyped-def]
        return zernike_cartesian(n, m, xx, yy)

    out = []
    for axis in (0, 1):
        d0 = _central(f, x, y, h, axis)
        d1 = _central(f, x, y, h / 2, axis)
        d2 = _central(f, x, y, h / 4, axis)
        r1 = (4 * d1 - d0) / 3
        r2 = (4 * d2 - d1) / 3
        out.append((16 * r2 - r1) / 15)
    return out[0], out[1]


def main() -> int:
    lines: list[str] = []
    ok = True

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    rng = np.random.default_rng(SEED)
    r = 0.9 * np.sqrt(rng.random(N_POINTS))
    t = 2 * np.pi * rng.random(N_POINTS)
    x, y = r * np.cos(t), r * np.sin(t)

    emit("ZernKit validation 3 -- analytic gradients vs high-accuracy finite differences")
    emit("=" * 82)
    emit(f"Sample points: {N_POINTS} uniform on the disc rho <= 0.9, seed {SEED}")
    emit("Reference: Richardson-extrapolated central differences, base h = 1e-2, O(h^6)")
    emit("")
    emit(f"{'Noll j':>7} {'(n, m)':>10} {'max|dZ/dx err|':>16} {'max|dZ/dy err|':>16}  mode")
    worst = 0.0
    for n in range(MAX_N + 1):
        for m in range(-n, n + 1, 2):
            gx, gy = zernike_gradient(n, m, x, y)
            fx, fy = richardson_gradient(n, m, x, y)
            scale = max(1.0, float(np.max(np.abs(gx))), float(np.max(np.abs(gy))))
            ex = float(np.max(np.abs(gx - fx))) / scale
            ey = float(np.max(np.abs(gy - fy))) / scale
            worst = max(worst, ex, ey)
            if n <= 4:
                emit(
                    f"{nm_to_noll(n, m):>7} {str((n, m)):>10} {ex:>16.3e} {ey:>16.3e}"
                    f"  {mode_name(n, m)}"
                )
    emit("  ... (orders 5..10 evaluated, not printed individually)")
    emit("")
    emit(f"Worst scaled deviation over all {(MAX_N + 1) * (MAX_N + 2) // 2} modes: {worst:.3e}")
    tol = 1e-9
    emit(f"Tolerance {tol:g} -> {'PASS' if worst < tol else 'FAIL'}")
    ok = ok and worst < tol
    emit("")

    emit("Closed-form hand checks:")
    checks: list[tuple[str, float, float]] = []
    gx, gy = zernike_gradient(1, 1, 0.37, -0.21)
    checks.append(("j=2  dZ/dx (exact 2)", float(gx), 2.0))
    checks.append(("j=2  dZ/dy (exact 0)", float(gy), 0.0))
    gx, gy = zernike_gradient(2, 0, 0.37, -0.21)
    checks.append(("j=4  dZ/dx (exact 4*sqrt3*0.37)", float(gx), 4 * math.sqrt(3) * 0.37))
    checks.append(("j=4  dZ/dy (exact 4*sqrt3*-0.21)", float(gy), 4 * math.sqrt(3) * -0.21))
    gx, gy = zernike_gradient(2, 2, 0.37, -0.21)
    checks.append(("j=6  dZ/dx (exact 2*sqrt6*0.37)", float(gx), 2 * math.sqrt(6) * 0.37))
    checks.append(("j=6  dZ/dy (exact -2*sqrt6*-0.21)", float(gy), -2 * math.sqrt(6) * -0.21))
    gx, gy = zernike_gradient(3, -1, 0.0, 0.0)
    checks.append(("j=7  dZ/dx at origin (exact 0)", float(gx), 0.0))
    checks.append(("j=7  dZ/dy at origin (exact -2*sqrt8)", float(gy), -2 * math.sqrt(8)))
    emit(f"{'check':<38} {'library':>20} {'exact':>20} {'diff':>10}")
    worst_hand = 0.0
    for label, got, exact in checks:
        worst_hand = max(worst_hand, abs(got - exact))
        emit(f"{label:<38} {got:>20.15f} {exact:>20.15f} {abs(got - exact):>10.2e}")
    emit(f"Worst |library - exact|: {worst_hand:.3e}")
    emit(f"Tolerance 1e-15 -> {'PASS' if worst_hand < 1e-15 else 'FAIL'}")
    ok = ok and worst_hand < 1e-15
    emit("")

    emit("No non-finite gradients at the pupil centre (the 1/rho factor cancels):")
    bad = [
        (n, m)
        for n in range(MAX_N + 1)
        for m in range(-n, n + 1, 2)
        if not all(np.isfinite(v) for v in zernike_gradient(n, m, 0.0, 0.0))
    ]
    emit(f"  modes with nan/inf at rho = 0: {len(bad)} -> {'PASS' if not bad else 'FAIL'}")
    ok = ok and not bad

    out = Path(__file__).with_name("gradients_output.txt")
    out.write_text("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
