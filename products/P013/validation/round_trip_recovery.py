"""Validation script 1: round-trip recovery of a known Cn2 in the
weak-fluctuation regime.

Run from the product root:

    python validation/round_trip_recovery.py | tee validation/round_trip_recovery_output.txt

Every number in ``VALIDATION.md`` S1 comes from this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope.constants import WEAK_REGIME_MAX_SIGMA_R2  # noqa: E402
from turbscope.dimm import (  # noqa: E402
    differential_variance,
    invert_cn2_from_variance,
)
from turbscope.inversion import multi_sensor_closed_form_estimate  # noqa: E402
from turbscope.scintillometer import (  # noqa: E402
    invert_cn2_weak,
    rytov_variance,
    scintillation_index_full,
)
from turbscope.synthetic import (  # noqa: E402
    APERTURE_DIAM_M,
    DIMM_NOISE_STD,
    DIMM_WAVELENGTH_M,
    SCINT_NOISE_STD,
    SCINT_WAVELENGTH_M,
    SEPARATION_M,
    WAVE_TYPE,
    Scenario,
    cn2_from_target_rytov,
    generate_scenarios,
    synthesize_measurement,
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("TurbScope round-trip recovery -- all values computed in this run")
    print(f"numpy {np.__version__}")

    # ---------------------------------------------------------------- 1.1
    section(
        "1.1 Noiseless closed-form round trip, scintillometer only, vs sigma_R^2 "
        "(quantifies where the weak-theory approximation itself costs accuracy, "
        "independent of any sensor noise)"
    )
    length = 500.0
    cols = f"{'target sigma_R^2':>18}{'Cn2_true [m^-2/3]':>22}{'Cn2_recovered':>18}"
    print(cols + f"{'rel. error':>14}")
    targets = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, WEAK_REGIME_MAX_SIGMA_R2, 0.5, 1.0]
    for target in targets:
        cn2_true = cn2_from_target_rytov(target, length)
        sigma_i2 = float(scintillation_index_full(target))
        cn2_rec = invert_cn2_weak(sigma_i2, length, SCINT_WAVELENGTH_M, WAVE_TYPE)
        rel_err = (cn2_rec - cn2_true) / cn2_true
        flag = (
            " <= WEAK_REGIME_MAX" if target <= WEAK_REGIME_MAX_SIGMA_R2 else "  (beyond threshold)"
        )
        print(f"{target:18.4f}{cn2_true:22.6e}{cn2_rec:18.6e}{rel_err:14.4%}{flag}")

    # ---------------------------------------------------------------- 1.2
    section("1.2 DIMM noiseless closed-form round trip (exact by construction, both channels)")
    for target in [1e-3, 0.05, 0.2]:
        cn2_true = cn2_from_target_rytov(target, length)
        for comp in ("longitudinal", "transverse"):
            var = float(
                differential_variance(
                    cn2_true, length, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M, comp
                )
            )
            cn2_rec = invert_cn2_from_variance(
                var, length, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M, comp
            )
            rel_err = (cn2_rec - cn2_true) / cn2_true
            print(f"  target sigma_R^2={target:g}, {comp:<13}: rel. error = {rel_err:.3e}")

    # ---------------------------------------------------------------- 2
    section(
        "2. Round-trip recovery WITH realistic measurement noise "
        f"(scintillometer {SCINT_NOISE_STD:.0%}, DIMM {DIMM_NOISE_STD:.0%} relative 1-sigma), "
        "single canonical weak-regime scenario, N=1000 independent noisy draws"
    )
    cn2_true = cn2_from_target_rytov(0.02, 300.0)  # a clearly weak-regime case
    scenario = Scenario(cn2_path=cn2_true, path_length_m=300.0, rytov_variance_true=0.02)
    print(f"true Cn2_path = {cn2_true:.6e} m^-2/3, L = 300 m, target sigma_R^2 = 0.02")

    rng = np.random.default_rng(20260829)
    scint_errs, dimm_errs, fused_errs = [], [], []
    for _ in range(1000):
        m = synthesize_measurement(scenario, rng)
        cn2_scint = invert_cn2_weak(
            m.sigma_i2_scint, m.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE
        )
        scint_errs.append((cn2_scint - cn2_true) / cn2_true)

        cn2_dimm = invert_cn2_from_variance(
            m.var_long_dimm, m.path_length_m, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M,
            "longitudinal",
        )
        dimm_errs.append((cn2_dimm - cn2_true) / cn2_true)

        result = multi_sensor_closed_form_estimate(
            m.sigma_i2_scint, m.var_long_dimm, m.var_trans_dimm, m.path_length_m,
            SCINT_WAVELENGTH_M, WAVE_TYPE, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M,
            SCINT_NOISE_STD, DIMM_NOISE_STD,
        )
        fused_errs.append((result.fused.cn2_path - cn2_true) / cn2_true)

    for name, errs in [
        ("scintillometer weak inversion", scint_errs),
        ("DIMM longitudinal inversion", dimm_errs),
        ("multi-sensor fused (inverse-variance)", fused_errs),
    ]:
        a = np.asarray(errs)
        print(
            f"{name:<40} mean rel. error {np.mean(a):+.4%}   std {np.std(a):.4%}   "
            f"RMSE {np.sqrt(np.mean(a**2)):.4%}   median |err| {np.median(np.abs(a)):.4%}"
        )

    # ---------------------------------------------------------------- 3
    section(
        "3. Round-trip recovery across MANY independent weak-regime scenarios "
        "(one noisy draw each) -- the headline aggregate number"
    )
    scenarios = generate_scenarios(3000, seed=555)
    weak_scenarios = [
        s
        for s in scenarios
        if float(rytov_variance(s.cn2_path, s.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE))
        <= WEAK_REGIME_MAX_SIGMA_R2
    ]
    print(
        f"{len(weak_scenarios)} of {len(scenarios)} drawn scenarios are weak-regime "
        f"(sigma_R^2 <= {WEAK_REGIME_MAX_SIGMA_R2})"
    )
    rng = np.random.default_rng(2026)
    fused_rel_errs = []
    scint_rel_errs = []
    for sc in weak_scenarios:
        m = synthesize_measurement(sc, rng)
        result = multi_sensor_closed_form_estimate(
            m.sigma_i2_scint, m.var_long_dimm, m.var_trans_dimm, m.path_length_m,
            SCINT_WAVELENGTH_M, WAVE_TYPE, DIMM_WAVELENGTH_M, APERTURE_DIAM_M, SEPARATION_M,
            SCINT_NOISE_STD, DIMM_NOISE_STD,
        )
        fused_rel_errs.append(abs(result.fused.cn2_path - sc.cn2_path) / sc.cn2_path)
        scint_only = invert_cn2_weak(
            m.sigma_i2_scint, m.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE
        )
        scint_rel_errs.append(abs(scint_only - sc.cn2_path) / sc.cn2_path)

    fused_arr = np.asarray(fused_rel_errs)
    scint_arr = np.asarray(scint_rel_errs)
    print(
        f"multi-sensor fused    : median |rel err| {np.median(fused_arr):.4%}, "
        f"mean {np.mean(fused_arr):.4%}, p90 {np.percentile(fused_arr, 90):.4%}"
    )
    print(
        f"scintillometer alone  : median |rel err| {np.median(scint_arr):.4%}, "
        f"mean {np.mean(scint_arr):.4%}, p90 {np.percentile(scint_arr, 90):.4%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
