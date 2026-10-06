"""Known-answer and survey checks on the S-curve and the detector gain K_d.

Run from this directory with ``PYTHONPATH=../src``.  Runtime about 3 s.

Three hand-computable families are checked, then the whole detector-by-pulse
gain table is printed including the cases where the gain collapses to zero or
the two estimators disagree.  Derivations are in ``docs/TIMING_MODEL.md``
section 2 and repeated in ``tests/test_scurve.py``.
"""

from __future__ import annotations

import math

import numpy as np

from slotsync.pulses import (
    half_sine,
    nyquist_raised_cosine,
    raised_cosine_time,
    rectangular,
    triangular,
)
from slotsync.scurve import default_offsets, scurve
from slotsync.ted import TedConfig

FINE = np.linspace(-0.02, 0.02, 41)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def check_triangle_early_late() -> float:
    banner("1. early-late gate on a triangular pulse: K_d = 2 exactly")
    print("   p(t) = 1 - |t|, e(eps) = p(eps - d) - p(eps + d) = 2 eps for |eps| < d")
    print(f"   {'delta':>6} {'K_d (fit)':>14} {'K_d (central)':>14} {'abs error':>12}")
    worst = 0.0
    for delta in (0.10, 0.25, 0.40, 0.50):
        curve = scurve(TedConfig("early-late", "antipodal", delta, "dd"), triangular(1.0), FINE)
        error = max(abs(curve.gain - 2.0), abs(curve.gain_central_difference - 2.0))
        worst = max(worst, error)
        print(
            f"   {delta:6.2f} {curve.gain:14.10f} {curve.gain_central_difference:14.10f} "
            f"{error:12.3e}"
        )
    print(f"   worst absolute error over 4 gate spacings: {worst:.3e}  (exact expected)")
    return worst


def check_triangle_mueller_muller() -> float:
    banner("2. Mueller-Mueller on a triangular pulse: K_d = E[a^2]")
    print("   S(eps) = E[a^2] (h(eps - 1) - h(eps + 1)) = E[a^2] eps for 0 < eps < 1")
    worst = 0.0
    for alphabet, expected in (("antipodal", 1.0), ("ook", 0.25)):
        curve = scurve(TedConfig("mueller-muller", alphabet), triangular(1.0), FINE)
        error = abs(curve.gain - expected)
        worst = max(worst, error)
        print(
            f"   {alphabet:<10} expected {expected:7.4f}  measured {curve.gain:14.10f}  "
            f"bias {curve.bias:+.3e}  error {error:.3e}"
        )
    print("   the unipolar case loses a factor of four of gain and gains no lock-point bias")
    print(f"   worst absolute error: {worst:.3e}  (exact expected)")
    return worst


def check_nyquist_mueller_muller() -> float:
    banner("3. Mueller-Mueller on a Nyquist raised cosine: K_d = 2 cos(pi a) / (1 - 4 a^2)")
    print("   derived from K_d = -2 h'(1) with sinc'(1) = -1; a = 1/2 is the limit pi/2")
    print(f"   {'rolloff':>8} {'predicted':>14} {'measured':>14} {'rel error':>12}")
    worst = 0.0
    for rolloff in (0.0, 0.25, 0.5, 0.75, 1.0):
        if abs(1.0 - 4.0 * rolloff**2) < 1e-12:
            predicted = math.pi / 2.0
        else:
            predicted = 2.0 * math.cos(math.pi * rolloff) / (1.0 - 4.0 * rolloff**2)
        curve = scurve(
            TedConfig("mueller-muller", "antipodal"),
            nyquist_raised_cosine(rolloff, 4.0),
            FINE,
            max_exact_symbols=16,
        )
        measured = curve.gain_central_difference
        relative = abs(measured - predicted) / abs(predicted)
        worst = max(worst, relative)
        print(f"   {rolloff:8.2f} {predicted:14.9f} {measured:14.9f} {relative:12.3e}")
    print(f"   worst relative error over 5 rolloffs: {worst:.3e}  (gate 1e-05)")
    return worst


def survey() -> None:
    banner("4. gain survey: every detector on every pulse shape")
    print("   K_d(fit) is a least-squares slope over |eps| <= 0.05; K_d(cd) a central difference.")
    print("   A large disagreement between them means the S-curve is not differentiable at the")
    print("   origin and the gain is not usable, which is reported rather than hidden.")
    pulses = [
        rectangular(1.0),
        triangular(1.0),
        raised_cosine_time(1.0),
        half_sine(1.0),
        nyquist_raised_cosine(0.5, 4.0),
    ]
    configs = [
        TedConfig("early-late", "antipodal", 0.25, "dd"),
        TedConfig("early-late", "antipodal", 0.25, "square"),
        TedConfig("gardner", "antipodal"),
        TedConfig("mueller-muller", "antipodal"),
    ]
    header = (
        f"   {'pulse':<12} {'detector':<26} {'K_d(fit)':>12} {'K_d(cd)':>12} "
        f"{'cd/fit':>8} {'linear':>8} {'peak':>7} {'reversal':>9} {'self-noise':>11}"
    )
    print(header)
    for pulse in pulses:
        for config in configs:
            curve = scurve(config, pulse, default_offsets(0.75, 301), max_exact_symbols=16)
            ratio = (
                curve.gain_central_difference / curve.gain if curve.gain != 0.0 else float("nan")
            )
            reversal = curve.reversal_offset
            reversal_text = "none" if reversal is None else f"{reversal:9.4f}"
            print(
                f"   {pulse.name:<12} {config.label:<26} {curve.gain:12.6f} "
                f"{curve.gain_central_difference:12.6f} {ratio:8.3f} "
                f"{curve.linear_halfwidth:8.3f} {curve.peak_offset:7.3f} "
                f"{reversal_text:>9} {curve.self_noise_at_origin:11.6f}"
            )
    print()
    print("   notable rows, all measured not assumed:")
    print("   - early-late on rect: K_d = 0. A rectangle is flat inside the pulse, so the")
    print("     difference of two gates inside it is identically zero and no loop exists.")
    print("   - mueller-muller on rect and on half-sine: K_d = 0. The detector is driven by")
    print("     h(+-1); a rectangle is flat there and a half-sine is zero there.")
    print("   - mueller-muller on rc-time: cd/fit far from 1. h'(+-1) = 0 for that shape, so")
    print("     the S-curve is third order at the origin and the gain is ill-conditioned.")
    print("   - gardner on rect: cd/fit far from 1 for the same reason in reverse - the pulse")
    print("     is discontinuous, so the S-curve is a step at the origin.")


def main() -> int:
    print("slotsync validation: S-curve and detector gain")
    print("exact ensemble averages by enumerating every data pattern in the pulse's span")
    worst = [
        check_triangle_early_late(),
        check_triangle_mueller_muller(),
        check_nyquist_mueller_muller(),
    ]
    survey()
    banner("summary")
    print(f"   known answer 1 (triangle, early-late)       worst abs error {worst[0]:.3e}")
    print(f"   known answer 2 (triangle, Mueller-Mueller)  worst abs error {worst[1]:.3e}")
    print(f"   known answer 3 (Nyquist, Mueller-Mueller)   worst rel error {worst[2]:.3e}")
    passed = worst[0] < 1e-10 and worst[1] < 1e-10 and worst[2] < 1e-5
    print(f"   all three known-answer families within tolerance: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
