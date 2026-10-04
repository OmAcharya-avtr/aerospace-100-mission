"""Reproduce the SGP4 verification vector shipped with the ``sgp4`` package.

Reference
---------
The ``sgp4`` distribution installs two files from the official SGP4
verification suite of Vallado, Crawford, Hujsak & Kelso 2006, "Revisiting
Spacetrack Report #3", AIAA 2006-6753:

* ``SGP4-VER.TLE``   -- the verification element sets with their requested
  start/stop/step in minutes from epoch, appended after column 69 of line 2.
* ``tcppver.out``     -- the reference TEME position and velocity produced by
  the authors' C++ implementation for each of those cases.

This script reads both from the installed ``sgp4`` package directory (so the
reference is the one the user's own install ships, not a copy pasted into this
repository), propagates satellite 00005 -- the TEME example, a Molniya-class
orbit with eccentricity 0.186 -- and reports the position and velocity
residuals against ``tcppver.out``.

Tolerance
---------
``tcppver.out`` prints positions to 1e-8 km and velocities to 1e-9 km/s.  The
expectation is therefore agreement at the print precision of the reference
file, not a loose engineering tolerance: the Python ``sgp4`` package wraps the
same C++ source, so any disagreement beyond rounding would indicate a
different gravity model, a different mode flag, or a unit error in this
repository's wrapper.  The declared pass tolerance is 1e-6 km on position and
1e-7 km/s on velocity, which is two decades above the print precision and
still far tighter than any physical accuracy claim.

This checks that :mod:`constellink.constellation` drives ``sgp4`` correctly.
It is NOT a validation of SGP4 itself against truth orbits, and it says
nothing about the accuracy of SGP4 far from epoch.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import sgp4
from sgp4.api import Satrec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from constellink.constellation import TLE, Satellite  # noqa: E402

TARGET_SATNUM = "00005"
POS_TOL_KM = 1e-6
VEL_TOL_KM_S = 1e-7


def sgp4_data_dir() -> str:
    """Directory of the installed ``sgp4`` package."""
    return os.path.dirname(os.path.abspath(sgp4.__file__))


def read_ver_tle(path: str, satnum: str) -> tuple[str, str, float, float, float]:
    """Return ``(line1, line2, start_min, stop_min, step_min)`` for ``satnum``.

    Lines in ``SGP4-VER.TLE`` carry the requested propagation span appended
    after column 69 of line 2.
    """
    with open(path, encoding="ascii") as fh:
        lines = [ln.rstrip("\n") for ln in fh]
    for i, ln in enumerate(lines):
        if ln.startswith("1 " + satnum):
            line1 = ln[:69]
            raw2 = lines[i + 1]
            line2 = raw2[:69]
            tail = raw2[69:].split()
            start, stop, step = (float(tail[0]), float(tail[1]), float(tail[2]))
            return line1, line2, start, stop, step
    raise LookupError(f"satellite {satnum} not found in {path}")


def read_tcppver(path: str, satnum: str) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Return ``[(minutes_from_epoch, r_km, v_km_s), ...]`` for ``satnum``."""
    want = str(int(satnum))
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    inside = False
    with open(path, encoding="ascii") as fh:
        for raw in fh:
            ln = raw.rstrip("\n")
            if ln.strip().endswith("xx"):
                inside = ln.split()[0] == want
                continue
            if not inside or not ln.strip():
                continue
            parts = ln.split()
            if len(parts) < 7:
                continue
            try:
                vals = [float(p) for p in parts[:7]]
            except ValueError:
                continue
            rows.append((vals[0], np.array(vals[1:4]), np.array(vals[4:7])))
    if not rows:
        raise LookupError(f"no reference rows for satellite {satnum} in {path}")
    return rows


def main() -> int:
    data_dir = sgp4_data_dir()
    tle_path = os.path.join(data_dir, "SGP4-VER.TLE")
    ref_path = os.path.join(data_dir, "tcppver.out")
    print("SGP4 verification-vector check")
    print("=" * 78)
    print(f"sgp4 package version   : {getattr(sgp4, '__version__', 'unknown')}")
    print(f"element set            : {tle_path}")
    print(f"reference output       : {ref_path}")
    print("reference provenance   : Vallado, Crawford, Hujsak & Kelso 2006,")
    print("                         'Revisiting Spacetrack Report #3', AIAA 2006-6753")
    print(f"satellite              : {TARGET_SATNUM} (TEME example, ecc ~0.186)")
    print(f"pass tolerance         : {POS_TOL_KM:g} km position, "
          f"{VEL_TOL_KM_S:g} km/s velocity")
    print("")

    line1, line2, start_min, stop_min, step_min = read_ver_tle(tle_path, TARGET_SATNUM)
    print(f"line 1 : {line1}")
    print(f"line 2 : {line2}")
    print(f"requested span from epoch: {start_min:g} .. {stop_min:g} min, "
          f"step {step_min:g} min")
    print("")

    tle = TLE(name=f"SGP4-VER {TARGET_SATNUM}", line1=line1, line2=line2)
    sat = Satellite.from_tle(tle)
    satrec: Satrec = sat.satrec
    rows = read_tcppver(ref_path, TARGET_SATNUM)
    # The first reference row at t = 0 is the epoch state printed without the
    # trailing orbital-element columns; it is included in the comparison.
    print(f"reference rows read      : {len(rows)}")
    print("")

    head = (f"{'t [min]':>12}{'|dr| [km]':>16}{'|dv| [km/s]':>16}"
            f"{'|r| [km]':>14}{'pass':>6}")
    print(head)
    print("-" * len(head))
    dr_all, dv_all = [], []
    all_pass = True
    for minutes, r_ref, v_ref in rows:
        err, r, v = satrec.sgp4_tsince(minutes)
        if err != 0:
            print(f"{minutes:>12.2f}   SGP4 error code {err} -- propagation refused")
            all_pass = False
            continue
        dr = float(np.linalg.norm(np.array(r) - r_ref))
        dv = float(np.linalg.norm(np.array(v) - v_ref))
        dr_all.append(dr)
        dv_all.append(dv)
        ok = dr <= POS_TOL_KM and dv <= VEL_TOL_KM_S
        all_pass &= ok
        print(f"{minutes:>12.2f}{dr:>16.3e}{dv:>16.3e}"
              f"{float(np.linalg.norm(r_ref)):>14.3f}{'PASS' if ok else 'FAIL':>6}")
    print("-" * len(head))
    print(f"max |dr| = {max(dr_all):.3e} km   (tolerance {POS_TOL_KM:g} km)")
    print(f"max |dv| = {max(dv_all):.3e} km/s (tolerance {VEL_TOL_KM_S:g} km/s)")
    print("")

    # Cross-check that the package wrapper's own datetime path agrees with the
    # tsince path used above, which is what constellink actually calls.
    from datetime import timedelta

    epoch = sat.epoch
    t_probe = epoch + timedelta(minutes=float(rows[-1][0]))
    r_wrap, v_wrap = sat.propagate([t_probe], check_epoch=False)
    err, r_ts, v_ts = satrec.sgp4_tsince(float(rows[-1][0]))
    d_wrap = float(np.linalg.norm(r_wrap[0] - np.array(r_ts)))
    print("wrapper cross-check (constellink Satellite.propagate vs sgp4_tsince)")
    print(f"  element epoch reconstructed as : {epoch.isoformat()}")
    print(f"  probe time                     : {t_probe.isoformat()}")
    print(f"  |dr| between the two paths     : {d_wrap:.3e} km")
    wrapper_tol_km = 1.0
    wrapper_ok = d_wrap <= wrapper_tol_km
    print(f"  tolerance                      : {wrapper_tol_km:g} km "
          f"(epoch reconstruction is limited by the TLE's own epoch precision "
          f"of ~1e-8 day = ~0.9 ms, and this orbit moves at ~7 km/s)")
    print(f"  result                         : {'PASS' if wrapper_ok else 'FAIL'}")
    print("")
    overall = all_pass and wrapper_ok
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
