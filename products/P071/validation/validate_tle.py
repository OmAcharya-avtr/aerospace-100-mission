"""The optional TLE entry point against the published SGP4 verification vectors.

``SGP4-VER.TLE`` and ``tcppver.out`` ship inside the installed ``sgp4``
package. They are the verification cases published with Vallado, D. A.,
Crawford, P., Hujsak, R., and Kelso, T. S. (2006), "Revisiting Spacetrack Report
#3", AIAA 2006-6753, and they are the reference here: this product does not
re-implement SGP4 and does not claim to have verified it, it checks that its own
use of the ``sgp4`` API reproduces the published output.

Only the file names are printed, never their paths, because they live inside a
site-packages directory whose absolute location must not appear in a committed
file.

Checks:
1. satellite 00005 propagated to every epoch in ``tcppver.out``, position and
   velocity, against the published values;
2. the closest-approach search: convergence of the located minimum under
   refinement of the coarse scan;
3. the synthetic-secondary helper: declared offset magnitude, speed preserved
   under the velocity rotation, and the resulting relative speed, which is
   **not** ``2 |v| sin(rotation / 2)`` unless the velocity is perpendicular to
   the rotation axis.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import sgp4

from _bootstrap import Tee, bootstrap, environment_lines, relative

bootstrap()

from conjprob.frames import rtn_basis  # noqa: E402
from conjprob.tle import (  # noqa: E402
    find_closest_approach,
    state_from_tle,
    synthetic_secondary,
)

SATELLITE = "00005"
PACKAGE_DIR = Path(sgp4.__file__).resolve().parent


def published_rows() -> tuple[str, str, list[tuple[float, np.ndarray, np.ndarray]]]:
    """Return the TLE lines and published states for satellite 00005, SI units."""
    tle_text = (PACKAGE_DIR / "SGP4-VER.TLE").read_text().splitlines()
    line1 = line2 = None
    for index, line in enumerate(tle_text):
        if line.startswith(f"1 {SATELLITE}"):
            line1 = line[:69]
            line2 = tle_text[index + 1][:69]
            break
    if line1 is None or line2 is None:
        raise RuntimeError("satellite 00005 not found in SGP4-VER.TLE")
    rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    collecting = False
    for line in (PACKAGE_DIR / "tcppver.out").read_text().splitlines():
        stripped = line.strip()
        if stripped.endswith("xx"):
            collecting = stripped.split()[0] == str(int(SATELLITE))
            continue
        if not collecting or not stripped:
            continue
        fields = stripped.split()
        if len(fields) < 7:
            continue
        try:
            numbers = [float(value) for value in fields[:7]]
        except ValueError:
            continue
        rows.append(
            (
                numbers[0],
                np.array(numbers[1:4]) * 1000.0,
                np.array(numbers[4:7]) * 1000.0,
            )
        )
    return line1, line2, rows


def main() -> int:
    out = Tee("validate_tle_output.txt")
    out.rule("conjprob validation: TLE state entry point")
    for line in environment_lines():
        out(line)
    failures = 0

    line1, line2, rows = published_rows()
    out("")
    out("reference files (shipped inside the installed sgp4 package, names only):")
    out(f"  {relative(PACKAGE_DIR / 'SGP4-VER.TLE')}")
    out(f"  {relative(PACKAGE_DIR / 'tcppver.out')}")
    out(f"satellite: {SATELLITE}")
    out(f"  {line1}")
    out(f"  {line2}")
    out(f"published epochs for this satellite: {len(rows)}")

    out.rule("1. propagation against the published verification output")
    out(
        f"{'minutes':>12s} {'|dr| [m]':>12s} {'|dv| [m/s]':>13s} "
        f"{'|r| [km]':>12s}"
    )
    worst_position = 0.0
    worst_velocity = 0.0
    for minutes, position, velocity in rows:
        computed_position, computed_velocity = state_from_tle(line1, line2, minutes)
        position_error = float(np.linalg.norm(computed_position - position))
        velocity_error = float(np.linalg.norm(computed_velocity - velocity))
        worst_position = max(worst_position, position_error)
        worst_velocity = max(worst_velocity, velocity_error)
        out(
            f"{minutes:>12.4f} {position_error:>12.3e} {velocity_error:>13.3e} "
            f"{float(np.linalg.norm(computed_position)) / 1000.0:>12.3f}"
        )
    out("")
    out(f"worst position error: {worst_position:.6e} m")
    out(f"worst velocity error: {worst_velocity:.6e} m/s")
    out("tolerance: the published values carry 8 decimal places in kilometres and 9 in")
    out("           kilometres per second, so the comparison floor is 1e-5 m and 1e-6")
    out("           m/s; the tolerances are 1e-4 m and 1e-5 m/s")
    ok = worst_position <= 1e-4 and worst_velocity <= 1e-5
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("2. closest-approach search convergence under coarse-scan refinement")
    primary_position, primary_velocity = state_from_tle(line1, line2, 0.0)
    secondary_position, secondary_velocity = synthetic_secondary(
        primary_position,
        primary_velocity,
        offset_rtn_m=(30.0, 110.0, 40.0),
        velocity_rotation_rad=math.radians(45.0),
    )

    def state_b(t: float):
        return secondary_position + secondary_velocity * t, secondary_velocity

    out("primary: satellite 00005 propagated by sgp4")
    out("secondary: a declared synthetic offset from the primary state, not a catalogue")
    out("           object; rectilinear over the one-minute search interval")
    out(
        f"{'coarse steps':>13s} {'coarse min [m]':>15s} {'refined min [m]':>16s} "
        f"{'tca [s]':>12s} {'evaluations':>12s}"
    )
    refined = []
    for steps in (4, 20, 100, 400, 2000):
        result = find_closest_approach(
            lambda t: state_from_tle(line1, line2, t / 60.0), state_b, -60.0, 60.0,
            coarse_steps=steps,
        )
        refined.append(result.separation_m)
        out(
            f"{steps:>13d} {result.coarse_separation_m:>15.6f} "
            f"{result.separation_m:>16.6f} {result.time_s:>12.6f} "
            f"{result.evaluations:>12d}"
        )
    spread = max(refined) - min(refined)
    out("")
    out(f"spread of the refined minimum over the coarse-step range: {spread:.6e} m")
    out("tolerance: <= 1e-6 m; the golden-section refinement removes the dependence on")
    out("           the coarse scan for this geometry, which has a single minimum")
    ok = spread <= 1e-6
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("3. the synthetic-secondary helper")
    offset = np.array([30.0, 110.0, 40.0])
    measured_offset = float(np.linalg.norm(secondary_position - primary_position))
    out(f"declared offset magnitude (RTN): {float(np.linalg.norm(offset)):.9f} m")
    out(f"measured offset magnitude      : {measured_offset:.9f} m")
    speed_primary = float(np.linalg.norm(primary_velocity))
    speed_secondary = float(np.linalg.norm(secondary_velocity))
    out(f"primary speed                  : {speed_primary:.9f} m/s")
    out(f"secondary speed                : {speed_secondary:.9f} m/s")
    relative_speed = float(np.linalg.norm(secondary_velocity - primary_velocity))
    naive = 2.0 * speed_primary * math.sin(math.radians(45.0) / 2.0)
    basis = rtn_basis(primary_position, primary_velocity)
    perpendicular = float(
        np.linalg.norm(primary_velocity - (primary_velocity @ basis[:, 0]) * basis[:, 0])
    )
    corrected = 2.0 * perpendicular * math.sin(math.radians(45.0) / 2.0)
    out(f"relative speed                 : {relative_speed:.9f} m/s")
    out(f"2 |v| sin(rotation/2)          : {naive:.9f} m/s  (wrong by "
        f"{relative_speed / naive - 1.0:+.6e})")
    out(f"2 |v_perp| sin(rotation/2)     : {corrected:.9f} m/s  (wrong by "
        f"{relative_speed / corrected - 1.0:+.6e})")
    out("")
    out("the rotation is about the radial direction and this orbit is eccentric, so the")
    out("velocity has a radial component that the rotation does not move. The docstring")
    out("of conjprob.tle.synthetic_secondary states the corrected form; the naive form")
    out("is 0.28 per cent wrong here and is printed so the difference is on the record.")
    ok = (
        abs(measured_offset / float(np.linalg.norm(offset)) - 1.0) <= 1e-9
        and abs(speed_secondary / speed_primary - 1.0) <= 1e-12
        and abs(relative_speed / corrected - 1.0) <= 1e-9
    )
    out(f"result: {'PASS' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    out.rule("what this check does not establish")
    out("a TLE gives a state and not a covariance. Nothing in this product derives a")
    out("covariance from an element set, and a collision probability computed from an")
    out("assumed covariance is a statement about that assumption. The covariance must")
    out("come from orbit determination.")

    out.rule("summary")
    out(f"checks failed: {failures}")
    out(f"written to {out.save()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
