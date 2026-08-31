"""V4 - Quantifying the controllability gap along the instantaneous field.

A magnetorquer produces ``L = m x B``, which is identically perpendicular to
``B``.  The rate component along ``B`` receives no torque at that instant, so
detumbling only works because ``B_hat`` moves as the spacecraft orbits and the
tilted dipole rotates with the Earth.  This script measures how much of that
motion each orbit geometry actually provides.

Metrics
-------
D1  eigenvalues of ``(<|B|^2> I - <B B^T>) / <|B|^2>`` (the "geometry
    factors"), which sum to exactly 2 for any field history and equal
    ``(2/3, 2/3, 2/3)`` only for a perfectly isotropic one.  The smallest
    eigenvalue divided by 2/3 is the damping deficit of the worst axis.
D2  the instantaneous uncontrollable fraction ``|omega . B_hat| / |omega|``,
    time-averaged along the orbit for a rate along the weakest direction.
D3  a direct simulation: identical spacecraft, identical initial rate,
    identical gain, on a near-equatorial and a sun-synchronous orbit, with the
    residual rate resolved along the weakest inertial direction.
D4  the instantaneous rank of the achievable-torque map, which is always 2.
"""

from __future__ import annotations

import time

import numpy as np

from _support import Tee  # noqa: E402

from detumblesim.analytic import geometry_factors, orbit_field_moments  # noqa: E402
from detumblesim.controllability import (  # noqa: E402
    controllability_report,
    instantaneous_projector,
    residual_rate_along,
    uncontrollable_fraction,
)
from detumblesim.orbit import CircularOrbit  # noqa: E402
from detumblesim.policies import FixedGainPolicy  # noqa: E402
from detumblesim.simulate import DetumbleConfig, field_history_eci, simulate_detumble  # noqa: E402
from detumblesim.spacecraft import Magnetorquer, inertia_from_diagonal  # noqa: E402

INCLINATIONS = [0.0, 5.0, 20.0, 45.0, 63.4, 80.0, 97.4]
N_SAMPLES = 8000
SPAN_ORBITS = 10.0
ISOTROPIC = 2.0 / 3.0


def main() -> None:
    t0 = time.perf_counter()
    with Tee(__file__) as out:
        out("V4  The magnetorquer controllability gap")
        out("=" * 78)
        out("")
        out("D4  Instantaneous rank of the achievable-torque map")
        for b in ([0.0, 0.0, 3e-5], [1e-5, -2e-5, 3e-5], [4e-5, 0.0, 0.0]):
            p = instantaneous_projector(b)
            out(f"    B = {np.array2string(np.array(b), precision=6):32s}  "
                f"rank(I - B_hat B_hat^T) = {np.linalg.matrix_rank(p, tol=1e-10)}  "
                f"trace = {np.trace(p):.12f}")
        out("    The achievable torque set is always a plane, never all of R^3.")
        out("")

        out(f"D1  Orbit-averaged geometry factors, 500 km, {SPAN_ORBITS:.0f}-orbit span, "
            f"{N_SAMPLES} samples")
        out("")
        out(f"  {'incl[deg]':>9} {'B_rms[uT]':>10} {'lam_min':>9} {'lam_mid':>9} "
            f"{'lam_max':>9} {'sum':>7} {'aniso':>8} {'deficit':>8} {'unctrl':>7}")
        rows = []
        for inc in INCLINATIONS:
            orbit = CircularOrbit(altitude_km=500.0, inclination_deg=inc)
            rep = controllability_report(
                orbit, N_SAMPLES, SPAN_ORBITS * orbit.period_s
            )
            lam = rep.weighted_eigenvalues
            deficit = float(lam[0] / ISOTROPIC)
            rows.append((inc, rep, deficit))
            out(f"  {inc:9.1f} {rep.rms_field_t * 1e6:10.3f} {lam[0]:9.5f} "
                f"{lam[1]:9.5f} {lam[2]:9.5f} {lam.sum():7.4f} "
                f"{rep.anisotropy:8.3f} {deficit:8.4f} "
                f"{rep.mean_uncontrollable_fraction:7.4f}")
        out("")
        out("  'deficit' is lam_min / (2/3): the smallest modal damping rate as a")
        out("  fraction of the perfectly isotropic value.  'unctrl' is D2, the mean")
        out("  of |omega . B_hat| / |omega| for a rate along the weakest direction.")
        out("")
        eq_rep = rows[0][1]
        ss_rep = rows[-1][1]
        out(f"  Equatorial (i = 0 deg)  : lam_min = {eq_rep.weighted_eigenvalues[0]:.5f}, "
            f"{ISOTROPIC / eq_rep.weighted_eigenvalues[0]:.2f}x slower than isotropic")
        out(f"  Sun-sync   (i = 97.4 deg): lam_min = {ss_rep.weighted_eigenvalues[0]:.5f}, "
            f"{ISOTROPIC / ss_rep.weighted_eigenvalues[0]:.2f}x slower than isotropic")
        out(f"  Ratio of the two worst-axis time constants = "
            f"{ss_rep.weighted_eigenvalues[0] / eq_rep.weighted_eigenvalues[0]:.2f}")
        out(f"  Weakest inertial direction, equatorial = "
            f"{np.array2string(eq_rep.weakest_direction_eci, precision=5)}")
        out(f"  Weakest inertial direction, sun-sync   = "
            f"{np.array2string(ss_rep.weakest_direction_eci, precision=5)}")
        out("  For the equatorial orbit the weak axis is the inertial z axis, i.e.")
        out("  the direction the tilted dipole stays closest to along the track.")
        out("")

        out("D2  Instantaneous uncontrollable fraction along the track")
        for inc in (0.0, 97.4):
            orbit = CircularOrbit(altitude_km=500.0, inclination_deg=inc)
            rep = controllability_report(orbit, N_SAMPLES, SPAN_ORBITS * orbit.period_s)
            t = np.linspace(0.0, SPAN_ORBITS * orbit.period_s, 2000, endpoint=False)
            b = field_history_eci(orbit, t)
            frac = np.array(
                [uncontrollable_fraction(rep.weakest_direction_eci, bi) for bi in b]
            )
            out(f"    inclination {inc:5.1f} deg: mean {frac.mean():.4f}, "
                f"min {frac.min():.4f}, max {frac.max():.4f}, "
                f"fraction of time above 0.9 = {np.mean(frac > 0.9):.4f}")
        out("")

        out("D3  Direct simulation, identical vehicle and gain, two orbits")
        inertia = inertia_from_diagonal(0.05, 0.05, 0.05)
        gain = 1.5e5
        w0 = np.radians([5.0, 5.0, 5.0])
        for inc in (0.0, 97.4):
            orbit = CircularOrbit(altitude_km=500.0, inclination_deg=inc)
            rep = controllability_report(orbit, 4000, SPAN_ORBITS * orbit.period_s)
            cfg = DetumbleConfig(
                inertia=inertia,
                orbit=orbit,
                magnetorquer=Magnetorquer.isotropic(0.2),
                omega0_rad_s=w0,
                duration_s=40000.0,
                control_dt_s=2.0,
                substeps=2,
                target_rate_rad_s=np.radians(1.0),
            )
            r = simulate_detumble(cfg, FixedGainPolicy(gain))
            weak = residual_rate_along(
                r.omega_rad_s, rep.weakest_direction_eci, r.quat
            )
            strong = residual_rate_along(
                r.omega_rad_s, _well_damped_axis(rep), r.quat
            )
            t_det = (
                f"{r.detumble_time_s:.1f} s" if r.detumbled else "not reached in 40000 s"
            )
            out(f"    inclination {inc:5.1f} deg")
            out(f"      detumble time to 1 deg/s     = {t_det}")
            out(f"      |omega| at 40000 s           = "
                f"{np.degrees(r.rate_norm_rad_s[-1]):.5f} deg/s")
            out(f"      rate about the WEAK axis     = "
                f"{np.degrees(abs(weak[-1])):.5f} deg/s")
            out(f"      rate in the WELL-DAMPED plane = "
                f"{np.degrees(abs(strong[-1])):.5f} deg/s")
            out(f"      weak-axis share of the final rate = "
                f"{abs(weak[-1]) / r.rate_norm_rad_s[-1]:.4f}")
            out(f"      saturated control steps      = "
                f"{100 * r.saturated_fraction:.2f} %")
        out("")

        out("D5  Cross-check: geometry_factors() and controllability_report() agree")
        orbit = CircularOrbit(altitude_km=500.0, inclination_deg=97.4)
        mom = orbit_field_moments(orbit, N_SAMPLES, SPAN_ORBITS * orbit.period_s)
        rep = controllability_report(orbit, N_SAMPLES, SPAN_ORBITS * orbit.period_s)
        diff = float(np.max(np.abs(geometry_factors(mom) - rep.weighted_eigenvalues)))
        out(f"    max |difference| between the two code paths = {diff:.3e}")
        out(f"    D5: {'PASS' if diff < 1e-12 else 'FAIL'} (tolerance 1e-12)")
        out("")
        out("OVERALL V4: the gap is real, always rank-2 instantaneously, and costs a")
        out(f"            factor {ISOTROPIC / eq_rep.weighted_eigenvalues[0]:.1f} in "
            "worst-axis damping rate on an equatorial orbit")
        out(f"            against {ISOTROPIC / ss_rep.weighted_eigenvalues[0]:.2f} on a "
            "sun-synchronous one.")
        out(f"wall time {time.perf_counter() - t0:.1f} s")


def _well_damped_axis(rep):
    """A unit direction inside the well-damped plane, for contrast with the weak one.

    Any vector orthogonal to the weakest direction lies in the plane spanned by
    the two better-damped modes; this picks one deterministically.
    """
    w = rep.weakest_direction_eci
    ref = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    v = np.cross(w, ref)
    return v / np.linalg.norm(v)


if __name__ == "__main__":
    main()
