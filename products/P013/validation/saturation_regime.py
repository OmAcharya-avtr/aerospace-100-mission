"""Validation script 3: demonstration and quantification of the scintillation
saturation failure mode.

Run from the product root:

    python validation/saturation_regime.py | tee validation/saturation_regime_output.txt

Every number in ``VALIDATION.md`` S2 comes from this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from turbscope.scintillometer import (  # noqa: E402
    SATURATION_ASYMPTOTE,
    invert_cn2_all_roots,
    invert_cn2_weak,
    rytov_variance,
    saturation_peak,
    scintillation_index_full,
)
from turbscope.synthetic import (  # noqa: E402
    SCINT_WAVELENGTH_M,
    WAVE_TYPE,
    cn2_from_target_rytov,
    generate_scenarios,
    synthesize_measurement,
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("TurbScope saturation-regime validation -- all values computed in this run")
    print(f"numpy {np.__version__}")

    # ---------------------------------------------------------------- 1
    section("1. Shape of the saturation curve (heuristic bridging model; see module docstring)")
    x_peak, val_peak = saturation_peak()
    asymptote = float(scintillation_index_full(1000.0))
    print(f"asymptote as sigma_R^2 -> infinity : {asymptote:.6f}")
    print(f"  (design target {SATURATION_ASYMPTOTE})")
    print(f"local maximum ('focusing' overshoot) at sigma_R^2 = {x_peak:.4f}")
    print(f"  sigma_I^2 at the peak             : {val_peak:.6f}")
    overshoot_pct = val_peak / asymptote - 1.0
    print(f"overshoot above asymptote          : {val_peak - asymptote:+.6f} ({overshoot_pct:.2%})")
    print("\nCurve sample:")
    print(f"{'sigma_R^2':>12}{'sigma_I^2 (full)':>20}{'sigma_I^2 (weak)':>20}")
    for x in [0.01, 0.1, 0.3, 1.0, x_peak, 3.0, 10.0, 50.0, 200.0]:
        print(f"{x:12.4f}{float(scintillation_index_full(x)):20.6f}{x:20.6f}")

    # ---------------------------------------------------------------- 2
    section("2. Existence and width of the multi-valued inversion band")
    # Multi-valued exactly where a horizontal line crosses the curve more than
    # once: between the asymptote and the peak value, for sigma_I^2 values the
    # curve attains on both the rising branch (before the peak) and the
    # falling/re-approaching branch (after the peak).
    lo_band, hi_band = asymptote, val_peak
    print(f"multi-valued sigma_I^2 band (approx.): [{lo_band:.6f}, {hi_band:.6f}]")
    print(f"band width                            : {hi_band - lo_band:.6f}")

    n_test = 25
    test_targets = np.linspace(lo_band + 1e-4, hi_band - 1e-4, n_test)
    n_multivalued = 0
    for t in test_targets:
        result = invert_cn2_all_roots(float(t), 1000.0, SCINT_WAVELENGTH_M, WAVE_TYPE)
        n_multivalued += int(result.is_multivalued)
    print(
        f"of {n_test} probe measurements spanning the band, {n_multivalued} "
        f"({n_multivalued / n_test:.0%}) were found genuinely multi-valued by the root finder"
    )

    # ---------------------------------------------------------------- 3
    section("3. Concrete worked example: two Cn2 candidates for one measurement")
    length = 1000.0
    target = 0.5 * (lo_band + hi_band)
    result = invert_cn2_all_roots(target, length, SCINT_WAVELENGTH_M, WAVE_TYPE)
    print(f"measured sigma_I^2 = {target:.6f} at L = {length:g} m")
    for i, (rv, cn2) in enumerate(zip(result.rytov_roots, result.cn2_roots, strict=True)):
        print(f"  root {i}: sigma_R^2 = {rv:.4f}  ->  Cn2_path = {cn2:.6e} m^-2/3")
    if len(result.cn2_roots) >= 2:
        ratio = result.cn2_roots[1] / result.cn2_roots[0]
        print(f"ratio of the two candidate Cn2 values: {ratio:.2f}x")
        print("No information in sigma_I^2 alone distinguishes which root is physically")
        print("correct -- would need an independent sensor (e.g. DIMM) or a priori")
        print("knowledge of the regime.")

    # ---------------------------------------------------------------- 4
    section(
        "4. Quantified failure of the weak-regime baseline when the TRUE regime "
        "is saturated (noiseless, isolates the model-form error from sensor noise)"
    )
    print(f"{'true sigma_R^2':>16}{'true Cn2':>16}{'baseline Cn2':>16}{'rel. error':>14}")
    for target in [0.5, 1.0, x_peak, 3.0, 10.0, 50.0]:
        cn2_true = cn2_from_target_rytov(target, length)
        sigma_i2 = float(scintillation_index_full(target))
        cn2_baseline = invert_cn2_weak(sigma_i2, length, SCINT_WAVELENGTH_M, WAVE_TYPE)
        rel_err = (cn2_baseline - cn2_true) / cn2_true
        print(f"{target:16.3f}{cn2_true:16.4e}{cn2_baseline:16.4e}{rel_err:14.2%}")

    # ---------------------------------------------------------------- 5
    section(
        "5. Aggregate baseline failure across many independently drawn saturated "
        "scenarios (with realistic sensor noise) -- the headline number"
    )
    scenarios = generate_scenarios(3000, seed=777)
    saturated = [
        s
        for s in scenarios
        if float(rytov_variance(s.cn2_path, s.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE)) > 1.0
    ]
    print(f"{len(saturated)} of {len(scenarios)} drawn scenarios have true sigma_R^2 > 1.0")
    rng = np.random.default_rng(31415)
    rel_errs = []
    for sc in saturated:
        m = synthesize_measurement(sc, rng)
        cn2_base = invert_cn2_weak(m.sigma_i2_scint, m.path_length_m, SCINT_WAVELENGTH_M, WAVE_TYPE)
        rel_errs.append(abs(cn2_base - sc.cn2_path) / sc.cn2_path)
    arr = np.asarray(rel_errs)
    print(
        f"scintillometer weak baseline in the saturated regime: median |rel err| "
        f"{np.median(arr):.1%}, mean {np.mean(arr):.1%}, p90 {np.percentile(arr, 90):.1%}"
    )
    print(
        "Compare to the weak-regime headline number in "
        "validation/round_trip_recovery_output.txt S3 -- the same baseline formula, "
        "applied outside its validity range, degrades by roughly an order of magnitude."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
