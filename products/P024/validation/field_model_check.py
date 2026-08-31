"""V1 - Does the tilted centred dipole reproduce published field magnitudes?

Reference data
--------------
Twelve total-intensity values from the **British Geological Survey IGRF-14 web
service**, ``https://geomag.bgs.ac.uk/web_service/GMModels/igrf/14/``, queried
at date 2025-01-01 for the geodetic latitude, east longitude and altitude
listed in each row.  They are the full IGRF-14 model (degree 13), against
which this package's degree-1 truncation is being measured.

Two caveats are stated rather than buried: the BGS service takes *geodetic*
latitude while this package uses geocentric spherical latitude (a difference
of up to about 0.19 deg, small compared with the errors reported below), and a
single web-service query is not an independent implementation of IGRF-14 - it
is the reference model itself.

Pass criteria, fixed before the run
-----------------------------------
A1  median absolute relative error over the 12 reference points <= 15 %
A2  maximum absolute relative error over the 12 reference points <= 25 %
A3  geomagnetic north pole within 0.05 deg of the WDC Kyoto published
    IGRF-14 2025 value (80.8 N, 72.8 W)
A4  |B| at the geomagnetic pole is exactly 2 B0 and at the geomagnetic
    equator exactly B0, on the reference sphere (rtol 1e-9)
"""

from __future__ import annotations

import numpy as np

from _support import Tee  # noqa: E402  (path setup happens on import)

from detumblesim.constants import B0_NT, NT_TO_T  # noqa: E402
from detumblesim.magfield import (  # noqa: E402
    DIPOLE_G_NT,
    dipole_field_ecef,
    dipole_tilt_deg,
    field_magnitude_nt,
    geomagnetic_north_pole_deg,
    spherical_position_ecef,
)

# (latitude_deg_north, longitude_deg_east, altitude_km, IGRF-14 |B| in nT)
BGS_IGRF14_2025 = [
    (0.0, 0.0, 0.0, 31835.0),
    (0.0, 0.0, 500.0, 24172.0),
    (0.0, 180.0, 500.0, 26925.0),
    (45.0, 0.0, 500.0, 37356.0),
    (-45.0, 0.0, 500.0, 21033.0),
    (80.0, 0.0, 500.0, 45087.0),
    (-80.0, 0.0, 500.0, 37254.0),
    (0.0, 90.0, 400.0, 34445.0),
    (30.0, 270.0, 400.0, 38097.0),
    (60.0, 120.0, 800.0, 41591.0),
    (-30.0, 315.0, 400.0, 19625.0),
    (90.0, 0.0, 0.0, 56879.0),
]

MEDIAN_TOL_PCT = 15.0
MAX_TOL_PCT = 25.0
POLE_TOL_DEG = 0.05
PUBLISHED_POLE = (80.8, -72.8)


def main() -> None:
    with Tee(__file__) as out:
        out("V1  Tilted centred-dipole field model vs IGRF-14")
        out("=" * 78)
        out("")
        out("Model coefficients (IGRF-14 main field, epoch 2025.0, degree 1 only):")
        out(f"  g(1,0) = {DIPOLE_G_NT[2]:>10.1f} nT")
        out(f"  g(1,1) = {DIPOLE_G_NT[0]:>10.1f} nT")
        out(f"  h(1,1) = {DIPOLE_G_NT[1]:>10.1f} nT")
        out(f"  B0     = {B0_NT:>10.4f} nT   (derived: sqrt(g10^2+g11^2+h11^2))")
        out("")

        out("A3  Geomagnetic (dipole) north pole")
        lat, lon = geomagnetic_north_pole_deg()
        out(f"  derived from the coefficients : {lat:.4f} N, {lon:.4f} E")
        out(f"  WDC Kyoto published IGRF-14 2025: {PUBLISHED_POLE[0]:.1f} N, "
            f"{PUBLISHED_POLE[1]:.1f} E")
        d_lat = abs(lat - PUBLISHED_POLE[0])
        d_lon = abs(lon - PUBLISHED_POLE[1])
        out(f"  difference                    : {d_lat:.4f} deg lat, {d_lon:.4f} deg lon")
        out(f"  dipole tilt from rotation axis: {dipole_tilt_deg():.4f} deg")
        a3 = d_lat <= POLE_TOL_DEG and d_lon <= POLE_TOL_DEG
        out(f"  A3 (tolerance {POLE_TOL_DEG} deg): {'PASS' if a3 else 'FAIL'}")
        out("")

        out("A4  Analytic self-consistency on the reference sphere")
        pole_pos = spherical_position_ecef(lat, lon, 0.0) * (1.0 + 1e-13)
        pole_b = float(np.linalg.norm(dipole_field_ecef(pole_pos))) / NT_TO_T
        axis = -DIPOLE_G_NT / np.linalg.norm(DIPOLE_G_NT)
        perp = np.cross(axis, [0.0, 0.0, 1.0])
        perp = perp / np.linalg.norm(perp)
        eq_pos = perp * float(np.linalg.norm(pole_pos))
        eq_b = float(np.linalg.norm(dipole_field_ecef(eq_pos))) / NT_TO_T
        out(f"  |B| at geomagnetic pole    = {pole_b:12.6f} nT   (expected 2*B0 = "
            f"{2 * B0_NT:.6f})")
        out(f"  |B| at geomagnetic equator = {eq_b:12.6f} nT   (expected   B0 = "
            f"{B0_NT:.6f})")
        rel_pole = abs(pole_b - 2 * B0_NT) / (2 * B0_NT)
        rel_eq = abs(eq_b - B0_NT) / B0_NT
        out(f"  relative errors            = {rel_pole:.3e}, {rel_eq:.3e}")
        a4 = rel_pole < 1e-9 and rel_eq < 1e-9
        out(f"  A4 (rtol 1e-9): {'PASS' if a4 else 'FAIL'}")
        out("")

        out("A1/A2  Dipole truncation error against IGRF-14 (BGS web service)")
        out("")
        out(f"{'lat[N]':>7} {'lon[E]':>7} {'alt[km]':>8} {'IGRF-14[nT]':>12} "
            f"{'dipole[nT]':>12} {'error[nT]':>11} {'rel[%]':>9}")
        errs = []
        for lat_d, lon_d, alt, ref in BGS_IGRF14_2025:
            model = field_magnitude_nt(lat_d, lon_d, alt)
            rel = 100.0 * (model - ref) / ref
            errs.append(abs(rel))
            out(f"{lat_d:7.1f} {lon_d:7.1f} {alt:8.0f} {ref:12.1f} {model:12.1f} "
                f"{model - ref:11.1f} {rel:+9.2f}")
        errs = np.array(errs)
        out("")
        out(f"  n points                    = {errs.size}")
        out(f"  median |relative error|     = {np.median(errs):8.3f} %")
        out(f"  mean   |relative error|     = {errs.mean():8.3f} %")
        out(f"  RMS    relative error       = {np.sqrt(np.mean(errs**2)):8.3f} %")
        out(f"  max    |relative error|     = {errs.max():8.3f} %  at "
            f"lat {BGS_IGRF14_2025[int(np.argmax(errs))][0]:.0f} N, "
            f"lon {BGS_IGRF14_2025[int(np.argmax(errs))][1]:.0f} E, "
            f"{BGS_IGRF14_2025[int(np.argmax(errs))][2]:.0f} km")
        a1 = float(np.median(errs)) <= MEDIAN_TOL_PCT
        a2 = float(errs.max()) <= MAX_TOL_PCT
        out("")
        out(f"  A1 median <= {MEDIAN_TOL_PCT:.0f} %  : "
            f"{'PASS' if a1 else 'FAIL'}  (measured {np.median(errs):.3f} %)")
        out(f"  A2 max    <= {MAX_TOL_PCT:.0f} %  : "
            f"{'PASS' if a2 else 'FAIL'}  (measured {errs.max():.3f} %)")
        out("")
        worst = sorted(
            zip(BGS_IGRF14_2025, errs, strict=True), key=lambda r: -r[1]
        )[:3]
        out("  Worst three points, all in or near the South Atlantic Anomaly and the")
        out("  Indian Ocean low, where the non-dipole field is largest:")
        for (la, lo, al, ref), e in worst:
            out(f"    lat {la:+6.1f} N  lon {lo:6.1f} E  {al:5.0f} km  "
                f"IGRF-14 {ref:7.0f} nT  |error| {e:6.2f} %")
        out("")
        out(f"OVERALL V1: A1 {'PASS' if a1 else 'FAIL'}, A2 "
            f"{'PASS' if a2 else 'FAIL'}, A3 {'PASS' if a3 else 'FAIL'}, "
            f"A4 {'PASS' if a4 else 'FAIL'}")
        if not a2:
            out("")
            out("A2 FAILED and is reported as failed.  No coefficient was adjusted and")
            out("no tolerance was widened.  The degree-1 truncation is simply not")
            out("accurate to 25 % everywhere; the consequences for every detumble")
            out("number in this package are set out in README 'Limitations'.")


if __name__ == "__main__":
    main()
