"""Validation 2: the AWGN known-answer limit of the fading LLRs.

Claim under test
----------------
As the scintillation index goes to zero the fading becomes deterministic at
``h = 1`` and every exact fading LLR in this package must reduce to the
textbook linear LLR of :func:`softdecode.llr.llr_ook_known_csi`,

    L(y) = ( a**2 - 2 a y ) / ( 2 sigma**2 ),

which is the log-likelihood ratio of two equal-variance Gaussian hypotheses
and is affine in ``y`` (Proakis & Salehi, *Digital Communications*, 5th ed.,
chapter 4).

A single tolerance would prove little, because any tolerance can be met by
choosing a small enough scintillation index. The stronger statement checked
here is that the deviation is **first order** in the scintillation index: the
ratio ``max|L_exact - L_linear| / sigma_I**2`` is constant over five decades.
That pins the limit rather than merely passing it.

The same is checked for gamma-gamma fading, where the zero-scintillation limit
is ``alpha, beta -> infinity``, and for M-ary PPM.

Runtime: under 5 s.
"""

from __future__ import annotations

import numpy as np

from softdecode.channel import GammaGammaFading, LognormalFading
from softdecode.detection import DetectionModel
from softdecode.llr import (
    llr_ook_known_csi,
    llr_ook_marginal,
    llr_ppm_known_csi,
    llr_ppm_marginal,
)

NODES = 200
Y = np.linspace(-3.0, 9.0, 61)
TOLERANCE_COEFFICIENT = 300.0


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


def lognormal_limit() -> float:
    banner("1. Lognormal fading: sigma_I**2 -> 0")
    det = DetectionModel()
    amplitude = det.ook_amplitude(10.0)
    linear = llr_ook_known_csi(Y, amplitude, 1.0, det.sigma)
    print(f"  a = {amplitude:.6f}, sigma = 1, y grid {Y.min():.1f} .. {Y.max():.1f} "
          f"({Y.size} points), {NODES} Gauss-Hermite nodes")
    print(f"  reference: L(y) = (a**2 - 2 a y) / 2, slope {-amplitude:.6f}")
    print(f"  {'sigma_I**2':>12} {'max |L - ref|':>16} {'rms |L - ref|':>16} "
          f"{'max err / sigma_I**2':>21}")
    coefficients = []
    for sigma_i2 in (1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        h, w = LognormalFading(sigma_i2).quadrature(NODES)
        exact = llr_ook_marginal(Y, amplitude, h, w, det.sigma)
        err = np.abs(exact - linear)
        coefficients.append(float(err.max() / sigma_i2))
        print(f"  {sigma_i2:12.1e} {err.max():16.6e} "
              f"{np.sqrt(np.mean(err**2)):16.6e} {err.max() / sigma_i2:21.6f}")
    spread = max(coefficients) / min(coefficients)
    print(f"  coefficient spread over five decades: {spread:.6f} "
          f"(1.0 would be exact first-order behaviour)")
    print(f"  conclusion: the deviation is first order in sigma_I**2 with "
          f"coefficient {np.mean(coefficients):.3f} on this grid")
    print("  The coefficient is a property of the grid and of a/sigma, not of the")
    print("  package: it grows with the largest |y| and with the amplitude, because")
    print("  the leading term is the curvature of L in h times Var[h]/2. The")
    print("  test-suite check tests/test_llr.py::test_awgn_limit_known_answer uses")
    print("  a = 2, y in [-3, 9], 25 points; the coefficient there is measured below.")
    narrow = np.linspace(-3.0, 9.0, 25)
    narrow_linear = llr_ook_known_csi(narrow, 2.0, 1.0, 1.0)
    narrow_coeffs = []
    for sigma_i2 in (1e-8, 1e-7, 1e-6):
        h, w = LognormalFading(sigma_i2).quadrature(60)
        err = float(np.max(np.abs(llr_ook_marginal(narrow, 2.0, h, w, 1.0) - narrow_linear)))
        narrow_coeffs.append(err / sigma_i2)
    print(f"  test-grid coefficient {max(narrow_coeffs):.3f}; the test tolerance of "
          f"{TOLERANCE_COEFFICIENT:.0f} * sigma_I**2")
    print(f"  is therefore slack by a factor of "
          f"{TOLERANCE_COEFFICIENT / max(narrow_coeffs):.2f}")
    return spread


def gammagamma_limit() -> None:
    banner("2. Gamma-gamma fading: alpha = beta = A -> infinity")
    det = DetectionModel()
    amplitude = det.ook_amplitude(10.0)
    linear = llr_ook_known_csi(Y, amplitude, 1.0, det.sigma)
    print(f"  {'A':>10} {'sigma_I**2':>13} {'nodes':>7} {'max |L - ref|':>16} "
          f"{'max err / sigma_I**2':>21}")
    print("  (the density is evaluated in log space via kve, so these do not overflow)")
    for a_param in (1e1, 1e2, 1e3, 1e4):
        model = GammaGammaFading(a_param, a_param)
        h, w = model.quadrature()
        exact = llr_ook_marginal(Y, amplitude, h, w, det.sigma)
        err = float(np.max(np.abs(exact - linear)))
        si = model.scintillation_index
        print(f"  {a_param:10.0e} {si:13.6e} {h.size:7d} {err:16.6e} {err / si:21.6f}")


def ppm_limit() -> None:
    banner("3. 4-PPM: the same limit for the bit LLRs")
    det = DetectionModel()
    amplitude = det.ppm_amplitude(10.0, 4)
    rng = np.random.default_rng(20261006)
    y = rng.standard_normal((400, 4)) + amplitude * np.eye(4)[rng.integers(0, 4, 400)]
    reference = llr_ppm_known_csi(y, amplitude, np.ones(400), det.sigma)
    print(f"  a = {amplitude:.6f}, 400 random 4-PPM symbols at Eb/N0 = 10 dB")
    print(f"  {'sigma_I**2':>12} {'max |L - ref|':>16} {'max err / sigma_I**2':>21}")
    for sigma_i2 in (1e-9, 1e-8, 1e-7, 1e-6):
        h, w = LognormalFading(sigma_i2).quadrature(NODES)
        exact = llr_ppm_marginal(y, amplitude, h, w, det.sigma)
        err = float(np.max(np.abs(exact - reference)))
        print(f"  {sigma_i2:12.1e} {err:16.6e} {err / sigma_i2:21.6f}")


def degenerate_quadrature() -> None:
    banner("4. A one-node quadrature is the known-CSI formula exactly")
    det = DetectionModel()
    amplitude = det.ook_amplitude(7.0)
    for h_value in (0.25, 1.0, 2.5):
        exact = llr_ook_marginal(
            Y, amplitude, np.array([h_value]), np.array([1.0]), det.sigma
        )
        known = llr_ook_known_csi(Y, amplitude, h_value, det.sigma)
        print(f"  h = {h_value:4.2f}: max |difference| {np.max(np.abs(exact - known)):.6e}")


def main() -> int:
    print("softdecode validation 2: the AWGN limit as a known answer")
    print("raw output committed as validation/awgn_limit_output.txt")
    spread = lognormal_limit()
    gammagamma_limit()
    ppm_limit()
    degenerate_quadrature()
    banner("Verdict")
    if spread < 1.05:
        print("  PASS: the lognormal exact LLR is first order in sigma_I**2 and")
        print("  reduces to the textbook linear LLR as sigma_I**2 -> 0.")
    else:
        print(f"  FAIL: coefficient spread {spread:.6f} exceeds 1.05; the deviation")
        print("  is not first order in sigma_I**2 over the decades checked.")
    print("\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
