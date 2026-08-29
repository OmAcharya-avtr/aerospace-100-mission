"""Validation 2: Monte-Carlo round-trip recovery of a known Cn2 and interval coverage.

Run:  python validation/validate_roundtrip.py > validation/validate_roundtrip_output.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope import (  # noqa: E402
    PathGeometry,
    SensorSuite,
    invert_dimm,
    invert_scintillation,
    simulate_measurement,
    weighted_path_average,
)
from turbscope.scintillation import rytov_variance  # noqa: E402

N_TRIALS = 4000
SEED = 20260829
TOLERANCE_BIAS = 0.02  # |mean relative error| must stay under 2 %


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _profile(path: PathGeometry, level: float, kind: str) -> np.ndarray:
    z = path.uniform_grid(401)
    u = z / path.length_m
    if kind == "uniform":
        return z, np.full_like(z, level)
    if kind == "ramp":
        return z, level * (0.4 + 1.2 * u)
    if kind == "hot-patch":
        return z, level * (1.0 + 3.0 * np.exp(-0.5 * ((u - 0.25) / 0.06) ** 2))
    raise ValueError(kind)


def section_1_noise_free_round_trip() -> None:
    rule("1. Noise-free round trip: does the inversion return the weighted average exactly?")
    print("The estimator is defined to return <Cn2>_W, not the plain path average.")
    print(f"{'profile':>12} {'<Cn2>_Wsc':>14} {'recovered':>14} {'rel err':>12} "
          f"{'plain avg':>14} {'plain rel err':>14}")
    path = PathGeometry(1000.0, 1550e-9)
    for kind in ("uniform", "ramp", "hot-patch"):
        z, cn2 = _profile(path, 1e-15, kind)
        beta = rytov_variance(z, cn2, path)
        from turbscope import scintillation_index

        est = invert_scintillation(float(scintillation_index(beta, 0.0)), path)
        target = weighted_path_average(z, cn2, kind="scintillation")
        plain = float(np.trapezoid(cn2, z) / path.length_m)
        print(f"{kind:>12} {target:>14.6e} {est.cn2:>14.6e} {est.cn2 / target - 1:>12.2e} "
              f"{plain:>14.6e} {est.cn2 / plain - 1:>14.2e}")
    print("\nThe last column is not an error of the estimator: it is the weighting bias")
    print("of the sensor, which is what a single-sensor inversion cannot remove.")


def section_2_monte_carlo_weak() -> None:
    rule(f"2. Monte-Carlo round trip in the weak regime ({N_TRIALS} trials)")
    rng = np.random.default_rng(SEED)
    path = PathGeometry(1000.0, 1550e-9)
    suite = SensorSuite(
        receiver_diameter_m=0.10,
        n_irradiance_samples=1000,
        n_dimm_frames=500,
        dimm_noise_arcsec=0.05,
    )
    z, cn2 = _profile(path, 2e-15, "hot-patch")
    beta_true = rytov_variance(z, cn2, path)
    target_sc = weighted_path_average(z, cn2, kind="scintillation")
    target_co = weighted_path_average(z, cn2, kind="coherence")
    print(f"true beta_0^2 = {beta_true:.5f}  (weak-regime limit 0.3)")
    print(f"true <Cn2>_Wsc = {target_sc:.6e} m^-2/3")
    print(f"true <Cn2>_Wco = {target_co:.6e} m^-2/3")
    print(f"kernel-mismatch ratio <Cn2>_Wsc/<Cn2>_Wco = {target_sc / target_co:.4f}\n")

    t0 = time.perf_counter()
    err_sc, err_co, cov_sc, cov_co = [], [], [], []
    for _ in range(N_TRIALS):
        meas = simulate_measurement(z, cn2, path, suite, rng)
        est = invert_scintillation(
            meas.sigma_i2_point, path, n_samples=suite.n_irradiance_samples, coverage=0.90
        )
        err_sc.append(est.cn2 / target_sc - 1.0)
        cov_sc.append(est.cn2_lower <= target_sc <= est.cn2_upper)
        dim = invert_dimm(
            meas.sigma_l2_rad2,
            path,
            subaperture_m=suite.dimm_subaperture_m,
            baseline_m=suite.dimm_baseline_m,
            n_frames=suite.n_dimm_frames,
            noise_variance_rad2=suite.dimm_noise_variance_rad2,
            coverage=0.90,
        )
        err_co.append(dim.cn2 / target_co - 1.0)
        cov_co.append(dim.cn2_lower <= target_co <= dim.cn2_upper)
    elapsed = time.perf_counter() - t0
    print(f"Monte Carlo wall time: {elapsed:.1f} s on 2 CPU cores\n")

    for name, err, cov, target in (
        ("scintillometer -> <Cn2>_Wsc", err_sc, cov_sc, "scintillation kernel"),
        ("DIMM           -> <Cn2>_Wco", err_co, cov_co, "coherence kernel"),
    ):
        e = np.asarray(err)
        c = float(np.mean(cov))
        print(f"{name}   (target = {target})")
        print(f"    mean relative error   {e.mean():+.5f}")
        print(f"    RMS relative error    {np.sqrt(np.mean(e**2)):.5f}")
        print(f"    median |rel err|      {np.median(np.abs(e)):.5f}")
        print(f"    p95 |rel err|         {np.percentile(np.abs(e), 95):.5f}")
        print(f"    90 % interval coverage (nominal 0.900): {c:.4f}")
        verdict = "PASS" if abs(e.mean()) < TOLERANCE_BIAS else "FAILED"
        print(f"    bias tolerance |mean| < {TOLERANCE_BIAS}: {verdict}")
        verdict = "PASS" if abs(c - 0.90) <= 0.02 else "FAILED"
        print(f"    coverage tolerance 0.90 +/- 0.02:       {verdict}\n")


def section_3_dimm_noise_floor_failure() -> None:
    rule("3. DIMM failure mode: turbulence below the centroid-noise floor")
    rng = np.random.default_rng(SEED + 1)
    path = PathGeometry(1000.0, 1550e-9)
    print(f"{'Cn2 (m^-2/3)':>14} {'SNR':>7} {'sigma_l^2':>12} {'noise':>12} "
          f"{'invalid':>9} {'med |rel err|':>14} {'p95 |rel err|':>14}")
    for level in (1e-19, 3e-19, 1e-18, 3e-18, 1e-17, 3e-17, 1e-16, 1e-15):
        suite = SensorSuite(n_dimm_frames=500, dimm_noise_arcsec=0.05)
        z, cn2 = _profile(path, level, "uniform")
        target = weighted_path_average(z, cn2, kind="coherence")
        invalid, errs, true_var = 0, [], None
        for _ in range(300):
            meas = simulate_measurement(z, cn2, path, suite, rng)
            true_var = meas.true_sigma_l2_rad2
            est = invert_dimm(
                meas.sigma_l2_rad2,
                path,
                subaperture_m=0.06,
                baseline_m=0.20,
                n_frames=500,
                noise_variance_rad2=suite.dimm_noise_variance_rad2,
            )
            if not est.valid:
                invalid += 1
            else:
                errs.append(est.cn2 / target - 1.0)
        med = np.median(np.abs(errs)) if errs else float("nan")
        p95 = np.percentile(np.abs(errs), 95) if errs else float("nan")
        snr = true_var / suite.dimm_noise_variance_rad2
        print(f"{level:>14.1e} {snr:>7.2f} {true_var:>12.3e} "
              f"{suite.dimm_noise_variance_rad2:>12.3e} {invalid / 300:>9.3f} "
              f"{med:>14.4f} {p95:>14.4f}")
    print("\nThe 0.05 arcsec centroid-noise floor is 5.876e-14 rad^2 on this path.  Above")
    print("SNR ~ 6 the inversion is accurate to ~4.7 % (median); at SNR 0.62 the median")
    print("error is 11.8 % and the p95 is 34 %; at SNR 0.19 the median is 27 %; and below")
    print("SNR ~ 0.06 more than 18 % of readings fall at or under the floor and return")
    print("valid=False rather than a number, rising to 46 % at SNR 0.01.  This is a real")
    print("instrument limit, reproduced rather than engineered away.")


def main() -> int:
    print("TurbScope validation 2 - round-trip recovery and interval coverage")
    print(f"seed {SEED}, {N_TRIALS} Monte-Carlo trials")
    section_1_noise_free_round_trip()
    section_2_monte_carlo_weak()
    section_3_dimm_noise_floor_failure()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
