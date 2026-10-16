"""Detection model: known answers, the scintillation average and the table error.

    python validation/validate_detection_model.py > validation/validate_detection_model_output.txt

References checked against
--------------------------
Gagliardi, R. M., & Karp, S. (1995). *Optical Communications*, 2nd ed., Wiley,
    Ch. 5: threshold detection on a Poisson photon count.
Marcum, J. I. (1960). "A statistical theory of target detection by pulsed
    radar", *IRE Transactions on Information Theory* 6(2), 59-267: the
    detection / false-alarm pair of a threshold test.
Siegman, A. E. (1986). *Lasers*, University Science Books, Ch. 17: the
    exp(-2 rho^2 / w^2) far-field Gaussian-beam irradiance profile.
Andrews, L. C., & Phillips, R. L. (2005). *Laser Beam Propagation through
    Random Media*, 2nd ed., SPIE Press, Ch. 9: the lognormal irradiance PDF in
    the weak-fluctuation regime, used here with E[I] = 1 and log-irradiance
    variance equal to the Rytov variance.
Abramowitz, M., & Stegun, I. A. (1964). *Handbook of Mathematical Functions*,
    eq. 25.4.46: Gauss-Hermite quadrature.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from _common import DWELL_S, RYTOV_VARIANCE, rule

from acqsearch import PoissonDwellDetector, poisson_upper_tail


def main() -> int:
    """Write the detection-model evidence."""
    out = sys.stdout
    failures = 0
    rule(out, "acqsearch detection model")

    out.write("\n1. Poisson upper tail against closed forms (Abramowitz & Stegun 26.4)\n")
    cases = [
        (1, 1.0, 1.0 - math.exp(-1.0), "P(N>=1|1) = 1 - e^-1"),
        (2, 1.0, 1.0 - 2.0 * math.exp(-1.0), "P(N>=2|1) = 1 - 2 e^-1"),
        (3, 2.0, 1.0 - 5.0 * math.exp(-2.0), "P(N>=3|2) = 1 - 5 e^-2"),
        (1, 0.0, 0.0, "P(N>=1|0) = 0"),
        (4, 3.0, 1.0 - math.exp(-3.0) * (1.0 + 3.0 + 4.5 + 4.5), "P(N>=4|3)"),
    ]
    out.write(f"{'case':<28}{'computed':>20}{'closed form':>20}{'abs error':>12}\n")
    for k, mean, expected, label in cases:
        got = float(poisson_upper_tail(k, mean))
        error = abs(got - expected)
        failures += error > 1e-15
        out.write(f"{label:<28}{got:>20.15f}{expected:>20.15f}{error:>12.2e}\n")

    out.write("\n2. Gaussian-beam shape factor (Siegman 1986 Ch. 17)\n")
    plain = PoissonDwellDetector(1.0e4, 0.0, 1, 0.0)
    w, tau = 5.0e-5, 1.0e-3
    out.write(f"{'rho/w':>8}{'computed P_d':>20}{'1 - exp(-lam tau)':>22}{'abs error':>12}\n")
    for ratio in (0.0, 0.5, 1.0, 1.5, 2.0):
        rho = ratio * w
        got = float(plain.detection_probability(tau, rho, w))
        expected = 1.0 - math.exp(-1.0e4 * math.exp(-2.0 * ratio**2) * tau)
        failures += abs(got - expected) > 1e-14
        out.write(f"{ratio:>8.2f}{got:>20.15f}{expected:>22.15f}{abs(got - expected):>12.2e}\n")

    out.write("\n3. Lognormal irradiance quadrature (Andrews & Phillips 2005 Ch. 9)\n")
    out.write(f"{'rytov':>8}{'nodes':>7}{'sum W':>20}{'E[I]':>20}{'E[I] error':>13}\n")
    for rytov in (0.0, 0.05, 0.2, 0.6, 1.0):
        detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, rytov)
        values, weights = detector.irradiance_nodes()
        mean = float((weights * values).sum())
        failures += abs(float(weights.sum()) - 1.0) > 1e-12
        failures += abs(mean - 1.0) > 1e-6
        out.write(
            f"{rytov:>8.2f}{values.size:>7d}{float(weights.sum()):>20.15f}"
            f"{mean:>20.15f}{abs(mean - 1.0):>13.2e}\n"
        )

    out.write("\n4. Quadrature convergence of P_d in the number of Hermite nodes\n")
    reference = PoissonDwellDetector(2.0e4, 2.0e3, 8, RYTOV_VARIANCE, quadrature_nodes=96)
    offsets = np.linspace(0.0, 1.2e-4, 25)
    exact = np.asarray(reference.detection_probability(DWELL_S, offsets, 5.0e-5))
    out.write(f"{'nodes':>8}{'max abs error vs 96 nodes':>30}\n")
    for nodes in (8, 16, 24, 48):
        detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, RYTOV_VARIANCE, quadrature_nodes=nodes)
        got = np.asarray(detector.detection_probability(DWELL_S, offsets, 5.0e-5))
        out.write(f"{nodes:>8d}{float(np.max(np.abs(got - exact))):>30.3e}\n")

    out.write("\n5. Interpolation error of the 1025-point P_d table, out to five half-widths\n")
    detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, RYTOV_VARIANCE)
    table_x, table_y = detector.detection_probability_table(DWELL_S, 5.0e-5, 2.5e-4, 1025)
    probe = np.linspace(0.0, 2.5e-4, 977)
    direct = np.asarray(detector.detection_probability(DWELL_S, probe, 5.0e-5))
    error = np.abs(np.interp(probe, table_x, table_y) - direct)
    worst = int(np.argmax(error))
    out.write(f"max absolute error      {float(error.max()):.6e}\n")
    out.write(f"at offset               {probe[worst]:.6e} rad ({probe[worst] / 5.0e-5:.3f} w)\n")
    out.write(f"mean absolute error     {float(error.mean()):.6e}\n")
    failures += float(error.max()) > 5e-5

    out.write("\n6. False alarm and detection at the reference configuration\n")
    out.write(f"dwell                   {DWELL_S:.6e} s\n")
    out.write(f"P_fa                    {detector.false_alarm_probability(DWELL_S):.9f}\n")
    for ratio in (0.0, 0.5, 1.0):
        rho = ratio * 5.0e-5
        value = float(detector.detection_probability(DWELL_S, rho, 5.0e-5))
        out.write(f"P_d at rho = {ratio:.1f} w      {value:.9f}\n")

    rule(out)
    out.write(f"checks failed: {failures}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
