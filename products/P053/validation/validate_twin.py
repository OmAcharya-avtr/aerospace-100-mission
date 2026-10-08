"""Validate the declared twin and its residual generator.

Checks
------
1. The steady-state Riccati solution satisfies its own equation.
2. The in-control normalised residual is N(0, 1) and white -- the innovations
   property (Kailath 1968). This is the check that caught the missing ``A``
   factor in the predictor gain during the build.
3. The parameter-step residual mean matches the closed form, equation (10).
4. The residual statistics of all three declared scenarios, which are the
   numbers quoted in the README and in DATASET_CARD.md.
"""

from __future__ import annotations

import numpy as np
from _bootstrap import add_src_to_path

add_src_to_path()

from twininvalidate import (  # noqa: E402
    SCENARIOS,
    AssetChange,
    StreamSpec,
    in_control_streams,
    reference_twin,
    simulate_residuals,
)

N_RUNS = 400
N_SAMPLES = 4000


def main() -> int:
    twin = reference_twin()
    filt = twin.steady_state()
    print("=== declared reference twin ===")
    print(f"dt                    {twin.dt:.4f} s ({1.0 / twin.dt:.1f} Hz)")
    print(f"A                     {twin.A.ravel().tolist()}")
    print(f"B                     {twin.B.ravel().tolist()}")
    print(f"Q                     {twin.Q.ravel().tolist()}")
    print(f"R                     {twin.R[0, 0]:.6e} rad^2")
    print(f"predictor gain K      {filt.K.ravel().tolist()}")
    print(f"innovation variance S {filt.S:.9e} rad^2")
    print(f"sqrt(S)               {np.sqrt(filt.S):.9e} rad")
    print(f"spectral radius A-KC  {filt.spectral_radius():.9f}   (requirement < 1)")
    print(f"Riccati residual      {filt.riccati_residual():.3e}  (tolerance 1e-15)")
    print(f"CHECK riccati         {'PASS' if filt.riccati_residual() < 1e-15 else 'FAIL'}")
    print(f"CHECK stability       {'PASS' if filt.spectral_radius() < 1.0 else 'FAIL'}")

    print()
    print("=== innovations property: in-control residual is i.i.d. N(0,1) ===")
    z = in_control_streams(n_runs=N_RUNS, n_samples=N_SAMPLES, seed=53001)
    n = z.size
    se = 1.0 / np.sqrt(n)
    mean = float(z.mean())
    std = float(z.std())
    lag1 = float(np.mean(z[:, :-1] * z[:, 1:]))
    lag2 = float(np.mean(z[:, :-2] * z[:, 2:]))
    lag5 = float(np.mean(z[:, :-5] * z[:, 5:]))
    print(f"samples               {n}")
    print(f"standard error        {se:.6f}  (= 1/sqrt(n), the tolerance unit below)")
    print(f"mean                  {mean:+.6f}   ({mean / se:+.2f} se)   tolerance 4 se")
    print(f"std                   {std:.6f}    ({(std - 1) / se:+.2f} se)   tolerance 4 se")
    print(f"lag-1 autocorrelation {lag1:+.6f}   ({lag1 / se:+.2f} se)   tolerance 4 se")
    print(f"lag-2 autocorrelation {lag2:+.6f}   ({lag2 / se:+.2f} se)   tolerance 4 se")
    print(f"lag-5 autocorrelation {lag5:+.6f}   ({lag5 / se:+.2f} se)   tolerance 4 se")
    checks = [abs(mean) < 4 * se, abs(std - 1.0) < 4 * se, abs(lag1) < 4 * se,
              abs(lag2) < 4 * se, abs(lag5) < 4 * se]
    print(f"CHECK whiteness       {'PASS' if all(checks) else 'FAIL'}")
    print("NOTE  the first version of this package used the filter gain P C^T/S")
    print("NOTE  in the predictor recursion instead of A P C^T/S. The measured")
    print("NOTE  lag-1 autocorrelation was then +0.0105, i.e. +13 se, and this")
    print("NOTE  check failed. See validation/VALIDATION.md, error 1.")

    print()
    print("=== parameter-step mean shift against the closed form, equation (10) ===")
    m = np.eye(twin.n_states) - (twin.A - filt.K @ twin.C)
    for delta in (-0.004, -0.01, -0.03):
        x_ss = np.linalg.solve(m, (twin.B * delta).ravel())
        predicted = float((twin.C @ x_ss)[0] / np.sqrt(filt.S))
        zz = simulate_residuals(
            StreamSpec(
                change=AssetChange("parameter_step", onset=0, magnitude=delta),
                n_runs=200,
                n_samples=2000,
                seed=53100,
            )
        )
        measured = float(zz[:, 500:].mean())
        rel = abs(measured / predicted - 1.0)
        print(
            f"delta={delta:+.3f}  predicted {predicted:+.6f}  measured {measured:+.6f}  "
            f"relative error {rel * 100:.2f} %  tolerance 3 %  "
            f"{'PASS' if rel < 0.03 else 'FAIL'}"
        )

    print()
    print("=== residual statistics of the three declared scenarios ===")
    print(f"{'scenario':<17}{'mean':>10}{'variance':>11}{'lag-1':>9}{'excess kurt':>13}")
    for name, change in SCENARIOS.items():
        zz = simulate_residuals(
            StreamSpec(change=change, n_runs=200, n_samples=2000, seed=53150)
        )
        post = zz[:, 600:]
        dev = post - post.mean()
        kurt = float((dev**4).mean() / post.var() ** 2 - 3.0)
        print(
            f"{name:<17}{post.mean():>10.4f}{post.var():>11.4f}"
            f"{float(np.mean(dev[:, :-1] * dev[:, 1:]) / post.var()):>9.4f}{kurt:>13.4f}"
        )
    print("NOTE  the noise_variance scenario multiplies the asset PROCESS noise by 2")
    print("NOTE  but raises the RESIDUAL variance only to the value above, because")
    print("NOTE  the measurement noise R dominates the innovation variance S.")
    print("NOTE  That number is what sets the variance-CUSUM oracle's reference value.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
