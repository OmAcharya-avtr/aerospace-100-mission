"""Loop-design checks: noise bandwidth, pole placement, inverse mapping, approximation.

Run from this directory with ``PYTHONPATH=../src``.  Runtime about 5 s.

Four independent checks, none of which involves a Monte Carlo:

1. the closed-form loop noise bandwidth against numerical quadrature of
   ``|H(j 2 pi f)|^2``;
2. the discrete loop's poles against ``exp(s_i T)`` of the analogue prototype;
3. the coefficient mapping round-tripped through its closed-form inverse;
4. the classical small-bandwidth jitter expression against the exact discrete-time
   Lyapunov solution, which is where the approximation's validity range comes from.
"""

from __future__ import annotations

import math

import numpy as np

from slotsync.loop import (
    LoopDesign,
    jitter_variance_closed_form,
    jitter_variance_exact,
    noise_bandwidth_closed_form,
    noise_bandwidth_numeric,
)

DAMPINGS = (0.3, 0.5, 1.0 / math.sqrt(2.0), 1.0, 2.0, 3.0)
THETAS = (0.005, 0.05, 0.5)
BANDWIDTHS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def check_noise_bandwidth() -> float:
    banner("1. loop noise bandwidth: closed form against quadrature")
    print("   B_n = theta (1 + 4 zeta^2) / (8 zeta) = (theta / 2)(zeta + 1 / (4 zeta))")
    print("   quadrature of int_0^inf |H(j 2 pi f)|^2 df with an analytic tail correction")
    print(f"   {'zeta':>7} {'theta':>7} {'closed form':>16} {'quadrature':>16} {'rel error':>12}")
    worst = 0.0
    for zeta in DAMPINGS:
        for theta in THETAS:
            closed = noise_bandwidth_closed_form(theta, zeta)
            numeric = noise_bandwidth_numeric(theta, zeta)
            relative = abs(closed - numeric) / numeric
            worst = max(worst, relative)
            print(f"   {zeta:7.4f} {theta:7.3f} {closed:16.10f} {numeric:16.10f} {relative:12.3e}")
    print(f"   worst relative error over 18 points: {worst:.3e}  (gate 1e-06)")
    print("   this check caught a spurious factor of 2 pi in the first draft of the closed form")
    return worst


def check_pole_placement() -> float:
    banner("2. discrete poles against exp(s T) of the analogue prototype")
    print("   the coefficient mapping is exact pole placement, so the identity must hold")
    print(
        f"   {'zeta':>7} {'B_n':>8} {'theta':>10} {'alpha':>12} "
        f"{'beta':>12} {'max pole err':>14}"
    )
    worst = 0.0
    for zeta in (0.5, 1.0 / math.sqrt(2.0), 1.0, 2.0):
        for bandwidth in BANDWIDTHS:
            design = LoopDesign.from_bandwidth(bandwidth, zeta, 1.75)
            theta = design.natural_frequency
            analogue = np.roots([1.0, 2.0 * zeta * theta, theta**2])
            expected = np.sort_complex(np.exp(analogue))
            actual = np.sort_complex(design.closed_loop_poles)
            error = float(np.max(np.abs(actual - expected)))
            worst = max(worst, error)
            alpha, beta = design.normalised_coefficients
            print(
                f"   {zeta:7.4f} {bandwidth:8.4f} {theta:10.6f} {alpha:12.6f} "
                f"{beta:12.3e} {error:14.3e}"
            )
    print(f"   worst pole error over 32 designs: {worst:.3e}  (gate 1e-06)")
    return worst


def check_inverse_mapping() -> float:
    banner("3. coefficient mapping round trip through its closed-form inverse")
    print(f"   {'zeta in':>9} {'B_n in':>9} {'zeta out':>12} {'B_n out':>12} {'worst rel':>12}")
    worst = 0.0
    for zeta in (0.3, 0.5, 1.0 / math.sqrt(2.0), 1.0, 2.0):
        for bandwidth in (0.001, 0.01, 0.05):
            design = LoopDesign.from_bandwidth(bandwidth, zeta, 2.5)
            back = LoopDesign.from_coefficients(
                design.k_proportional, design.k_integral, design.detector_gain
            )
            relative = max(
                abs(back.damping - zeta) / zeta,
                abs(back.noise_bandwidth - bandwidth) / bandwidth,
            )
            worst = max(worst, relative)
            print(
                f"   {zeta:9.6f} {bandwidth:9.4f} {back.damping:12.8f} "
                f"{back.noise_bandwidth:12.8f} {relative:12.3e}"
            )
    print(f"   worst relative round-trip error over 15 designs: {worst:.3e}  (gate 1e-08)")
    return worst


def check_small_bandwidth_approximation() -> None:
    banner("4. classical jitter expression against the exact discrete loop")
    print("   closed form  sigma^2 = 2 B_n sigma_n^2 / K_d^2   (white noise, analogue prototype)")
    print("   exact        stationary solution of P = A P A' + B B' sigma_n^2")
    print("   the ratio is the cost of the small-bandwidth approximation, with no Monte Carlo")
    print(f"   {'zeta':>7} {'B_n':>8} {'closed form':>14} {'exact':>14} {'exact/closed':>14}")
    for zeta in (0.5, 1.0 / math.sqrt(2.0), 1.0):
        for bandwidth in BANDWIDTHS:
            design = LoopDesign.from_bandwidth(bandwidth, zeta, 1.5)
            closed = jitter_variance_closed_form(bandwidth, 1.5, 0.3)
            exact = jitter_variance_exact(design, 0.3)
            print(
                f"   {zeta:7.4f} {bandwidth:8.4f} {closed:14.6e} {exact:14.6e} "
                f"{exact / closed:14.6f}"
            )
    print()
    print("   the approximation is optimistic by less than 0.2 % below B_n = 0.001,")
    print("   by about 1 % at B_n = 0.01 and by 3 to 7 % at B_n = 0.05 to 0.1.")
    print("   nothing here involves the detector: this is purely the loop discretisation.")


def main() -> int:
    print("slotsync validation: second-order loop design")
    bandwidth_error = check_noise_bandwidth()
    pole_error = check_pole_placement()
    inverse_error = check_inverse_mapping()
    check_small_bandwidth_approximation()
    banner("summary")
    print(f"   noise bandwidth vs quadrature   worst rel error {bandwidth_error:.3e}  gate 1e-06")
    print(f"   pole placement identity         worst abs error {pole_error:.3e}  gate 1e-06")
    print(f"   coefficient round trip          worst rel error {inverse_error:.3e}  gate 1e-08")
    passed = bandwidth_error < 1e-6 and pole_error < 1e-6 and inverse_error < 1e-8
    print(f"   all three gated checks within tolerance: {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
