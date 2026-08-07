"""Validation script 1: published baselines and derived seeing quantities.

Run from the product root:

    python validation/validate_baselines.py | tee validation/validate_baselines_output.txt

Every number in ``VALIDATION.md`` §1-§3 comes from this script.  Nothing is
asserted here that is not computed; where a computed value disagrees with the
published one the disagreement is printed, not hidden.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cncast.baselines import (  # noqa: E402
    bufton_wind,
    hufnagel_valley,
    hv57,
    rms_high_altitude_wind,
    slc_day,
    slc_night,
)
from cncast.seeing import (  # noqa: E402
    fried_parameter,
    greenwood_frequency,
    isoplanatic_angle,
    seeing_fwhm_arcsec,
    turbulence_moment,
)

LAM = 500e-9
FINE = np.linspace(0.0, 20_000.0, 200_001)  # 10 cm steps


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    print("CnCast baseline validation — all values computed in this run")
    print(f"numpy {np.__version__}")

    # ---------------------------------------------------------------- 1.1
    section("1.1  HV 5/7 Cn^2 at reference altitudes (m^-2/3)")
    print(f"{'h [m]':>9} {'ground term':>13} {'tropo term':>13} {'high term':>13} {'total':>13}")
    for h in [0.0, 10.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 15000.0, 20000.0]:
        ha = np.array([h])
        ground = 1.7e-14 * np.exp(-h / 100.0)
        tropo = 2.7e-16 * np.exp(-h / 1500.0)
        high = 0.00594 * (21.0 / 27.0) ** 2 * (1e-5 * h) ** 10 * np.exp(-h / 1000.0)
        total = float(hv57(ha)[0])
        print(f"{h:9.0f} {ground:13.4e} {tropo:13.4e} {high:13.4e} {total:13.4e}")
        assert abs(total - (ground + tropo + high)) <= 1e-24, "term decomposition mismatch"

    # ---------------------------------------------------------------- 1.2
    section("1.2  HV 5/7 defining property: r0 ~ 5 cm and theta0 ~ 7 urad @ 500 nm, zenith")
    cn2 = hv57(FINE)
    r0 = fried_parameter(FINE, cn2, LAM, 0.0)
    th0 = isoplanatic_angle(FINE, cn2, LAM, 0.0)
    print(f"mu_0   = int Cn^2 dh          = {turbulence_moment(FINE, cn2, 0.0):.6e} m^(1/3)")
    print(f"mu_5/3 = int Cn^2 h^(5/3) dh  = {turbulence_moment(FINE, cn2, 5 / 3):.6e} m^(2)")
    print(f"r0     = {r0 * 100:.4f} cm      (published nickname value: 5 cm, "
          f" err {abs(r0 * 100 - 5) / 5:.2%})")
    print(f"theta0 = {th0 * 1e6:.4f} urad   (published nickname value: 7 urad,"
          f" err {abs(th0 * 1e6 - 7) / 7:.2%})")
    print(f"seeing FWHM = {seeing_fwhm_arcsec(r0, LAM):.4f} arcsec")

    # grid-convergence of the integrals
    print("\ngrid convergence of r0 (relative to the 200001-point value):")
    for n in [201, 2001, 20001, 200001]:
        g = np.linspace(0.0, 20000.0, n)
        r = fried_parameter(g, hv57(g), LAM, 0.0)
        print(f"  n = {n:7d}  step = {20000 / (n - 1):8.3f} m  r0 = {r * 100:.5f} cm"
              f"  rel.diff = {abs(r - r0) / r0:.2e}")

    # ---------------------------------------------------------------- 1.3
    section("1.3  SLC-Day / SLC-Night: branch values and branch-boundary continuity")
    print("SLC-Day and SLC-Night Cn^2 at reference altitudes (m above the AMOS site):")
    print(f"{'h [m]':>9} {'SLC-Day':>13} {'SLC-Night':>13} {'day/night':>10}")
    for h in [1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 19000.0]:
        d = float(slc_day(np.array([h]))[0])
        n = float(slc_night(np.array([h]))[0])
        ratio = d / n if n > 0 else float("nan")
        print(f"{h:9.0f} {d:13.4e} {n:13.4e} {ratio:10.2f}")

    print("\npublished-branch spot checks (hand-evaluable closed forms):")
    checks = [
        ("SLC-Day  h=100 m  -> 3.13e-13/h^1.05", float(slc_day(np.array([100.0]))[0]),
         3.13e-13 / 100.0**1.05),
        ("SLC-Day  h=2000 m -> 8.87e-7/h^3", float(slc_day(np.array([2000.0]))[0]),
         8.87e-7 / 2000.0**3),
        ("SLC-Day  h=10000 m-> 2.0e-16/h^0.5", float(slc_day(np.array([10000.0]))[0]),
         2.0e-16 / 10000.0**0.5),
        ("SLC-Night h=50 m  -> 2.87e-12/h^2", float(slc_night(np.array([50.0]))[0]),
         2.87e-12 / 50.0**2),
        ("SLC-Night h=500 m -> 2.5e-16 (constant)",
         float(slc_night(np.array([500.0]))[0]), 2.5e-16),
        ("SLC-Night h=3000 m-> 8.87e-7/h^3", float(slc_night(np.array([3000.0]))[0]),
         8.87e-7 / 3000.0**3),
    ]
    for label, got, want in checks:
        print(f"  {label:42s} got {got:.6e}  expect {want:.6e}"
              f"  rel.err {abs(got - want) / want:.1e}")

    print("\nbranch-boundary discontinuity of the PUBLISHED fits (not an implementation bug):")
    eps = 1e-6
    for name, fn, edges in [
        ("SLC-Day", slc_day, [18.5, 240.0, 880.0, 7220.0]),
        ("SLC-Night", slc_night, [18.5, 110.0, 1500.0, 7200.0]),
    ]:
        for e in edges:
            below = float(fn(np.array([e - eps]))[0])
            above = float(fn(np.array([e + eps]))[0])
            print(f"  {name:10s} h = {e:8.1f} m: below {below:.4e}  above {above:.4e}"
                  f"  jump {above / below - 1:+7.1%}")

    print("\nSLC ceilings (the fits are defined as identically zero above them):")
    for name, fn, ceil in [("SLC-Day", slc_day, 20500.0), ("SLC-Night", slc_night, 20000.0)]:
        print(f"  {name:10s} Cn^2({ceil - 1:.0f}) = {float(fn(np.array([ceil - 1]))[0]):.4e}"
              f"   Cn^2({ceil + 1:.0f}) = {float(fn(np.array([ceil + 1]))[0]):.4e}")

    # ---------------------------------------------------------------- 1.4
    section("1.4  Integrated seeing quantities for each published baseline @ 500 nm, zenith")
    print(f"{'model':>12} {'r0 [cm]':>10} {'theta0 [urad]':>14} {'f_G [Hz]':>10}"
          f" {'FWHM arcsec':>12}")
    wind5 = bufton_wind(FINE, 5.0)
    for name, prof in [("HV 5/7", hv57(FINE)), ("SLC-Day", slc_day(FINE)),
                       ("SLC-Night", slc_night(FINE))]:
        r = fried_parameter(FINE, prof, LAM, 0.0)
        t = isoplanatic_angle(FINE, prof, LAM, 0.0)
        f = greenwood_frequency(FINE, prof, wind5, LAM, 0.0)
        print(f"{name:>12} {r * 100:10.3f} {t * 1e6:14.3f} {f:10.2f} "
              f"{seeing_fwhm_arcsec(r, LAM):10.3f}")

    print("\nGreenwood frequency of HV 5/7 vs ground wind speed (Bufton profile):")
    for wg in [0.0, 5.0, 10.0, 21.0]:
        f = greenwood_frequency(FINE, hv57(FINE), bufton_wind(FINE, wg), LAM, 0.0)
        print(f"  ground wind {wg:5.1f} m/s -> rms high-altitude wind "
              f"{rms_high_altitude_wind(wg):6.2f} m/s, f_G = {f:7.2f} Hz")

    # ---------------------------------------------------------------- 2
    section("2  Analytic cross-checks with closed forms")

    print("2.1 constant-Cn^2 slab, Cn^2 = 1.0e-15 m^-2/3 from 0 to 10 km, lambda = 1.55 um")
    h = np.linspace(0.0, 10_000.0, 100_001)
    c = np.full_like(h, 1e-15)
    lam = 1.55e-6
    mu0 = turbulence_moment(h, c, 0.0)
    k = 2 * np.pi / lam
    r0_closed = (0.423 * k**2 * mu0) ** (-3 / 5)
    r0_code = fried_parameter(h, c, lam, 0.0)
    print(f"  mu_0 (closed form 1e-15*1e4)   = {1e-15 * 1e4:.6e}   code {mu0:.6e}")
    print(f"  r0 closed form                 = {r0_closed * 100:.6f} cm")
    print(f"  r0 from cncast.seeing          = {r0_code * 100:.6f} cm"
          f"   rel.err {abs(r0_code - r0_closed) / r0_closed:.2e}")
    mu53 = turbulence_moment(h, c, 5 / 3)
    mu53_closed = 1e-15 * 10_000.0 ** (8 / 3) / (8 / 3)
    print(f"  mu_5/3 closed form Cn^2 H^(8/3)/(8/3) = {mu53_closed:.6e}   code {mu53:.6e}"
          f"   rel.err {abs(mu53 - mu53_closed) / mu53_closed:.2e}")

    print("\n2.2 wavelength scaling r0 ~ lambda^(6/5) on HV 5/7")
    r_500 = fried_parameter(FINE, hv57(FINE), 500e-9, 0.0)
    for lam_nm in [850.0, 1550.0]:
        r = fried_parameter(FINE, hv57(FINE), lam_nm * 1e-9, 0.0)
        pred = r_500 * (lam_nm / 500.0) ** (6 / 5)
        print(f"  lambda {lam_nm:6.0f} nm: r0 = {r * 100:7.3f} cm, "
              f"scaling prediction {pred * 100:7.3f} cm, rel.err {abs(r - pred) / pred:.2e}")

    print("\n2.3 zenith scaling: r0 ~ cos(z)^(3/5), theta0 ~ cos(z)^(8/5)")
    for z in [30.0, 60.0]:
        r = fried_parameter(FINE, hv57(FINE), LAM, z)
        t = isoplanatic_angle(FINE, hv57(FINE), LAM, z)
        rp = r_500 * np.cos(np.radians(z)) ** (3 / 5)
        tp = isoplanatic_angle(FINE, hv57(FINE), LAM, 0.0) * np.cos(np.radians(z)) ** (8 / 5)
        print(f"  zeta {z:4.0f} deg: r0 {r * 100:7.4f} cm (pred {rp * 100:7.4f}, "
              f"err {abs(r - rp) / rp:.1e});  theta0 {t * 1e6:7.4f} urad "
              f"(pred {tp * 1e6:7.4f}, err {abs(t - tp) / tp:.1e})")

    # ---------------------------------------------------------------- 3
    section("3  Physical-plausibility properties of the baselines")
    print("  Claim under test: Cn^2 falls with altitude through the free troposphere,")
    print("  EXCEPT for the jet-stream bump that the H-V high-altitude term exists to")
    print("  produce.  A naive 'monotone everywhere above the boundary layer' test is")
    print("  therefore WRONG for these models; the ranges below are the honest claim.")
    h_free = np.geomspace(300.0, 19_000.0, 4000)
    for name, prof in [("HV 5/7", hv57(h_free)), ("SLC-Day", slc_day(h_free)),
                       ("SLC-Night", slc_night(h_free))]:
        d = np.diff(prof)
        first_rise = int(np.argmax(d > 0.0)) if np.any(d > 0.0) else -1
        where = f"{h_free[first_rise]:.0f} m" if first_rise >= 0 else "nowhere"
        print(f"  {name:10s} decreasing on 300 m .. {where} "
              f"(first altitude where the model rises again); "
              f"Cn^2(300 m)/Cn^2(19 km) = {prof[0] / prof[-1]:.1f}")
    h_hv = np.geomspace(300.0, 8_000.0, 200_000)
    c_hv = hv57(h_hv)
    i_min = int(np.argmin(c_hv))
    h_up = np.geomspace(6_000.0, 20_000.0, 200_000)
    c_up = hv57(h_up)
    i_bump = int(np.argmax(c_up))
    print(f"  HV 5/7 pre-bump minimum on 300 m..8 km at h = {h_hv[i_min]:.0f} m "
          f"(Cn^2 = {c_hv[i_min]:.3e}); jet-stream bump peak at h = {h_up[i_bump]:.0f} m "
          f"(Cn^2 = {c_up[i_bump]:.3e})")
    d_mono = np.diff(hv57(np.geomspace(300.0, 5000.0, 4000)))
    print(f"  HV 5/7 strictly decreasing on 300 m .. 5 km: {bool(np.all(d_mono < 0.0))}")
    hv_hi = hufnagel_valley(np.array([9000.0, 10000.0, 11000.0]), 40.0, 1.7e-14)
    hv_lo = hufnagel_valley(np.array([9000.0, 10000.0, 11000.0]), 10.0, 1.7e-14)
    print(f"  HV tropopause bump grows with pseudowind: v=40 {hv_hi[1]:.3e} > "
          f"v=10 {hv_lo[1]:.3e}  ratio {hv_hi[1] / hv_lo[1]:.2f} "
          f"(expected (40/10)^2 = 16 for the high-altitude term alone)")
    print(f"  HV 5/7 is NOT monotone overall: Cn^2(9 km) = "
          f"{float(hv57(np.array([9000.0]))[0]):.3e} vs Cn^2(11 km) = "
          f"{float(hv57(np.array([11000.0]))[0]):.3e} — the jet-stream bump peaks near 10 km")
    print("\nOK — validation script completed without assertion failure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
