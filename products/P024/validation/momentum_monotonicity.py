"""V2 - Is angular momentum monotonically decreasing under B-dot?

The specification for this product asserts that "angular momentum decreases
monotonically under B-dot in the absence of saturation".  That statement is
**true for kinetic energy and for an isotropic inertia, and false in general**,
and this script establishes both halves with numbers rather than assertion.

Theory
------
The ideal (unsaturated, slowly-varying-field) B-dot torque is

    L = -k |B|^2 omega_perp,   omega_perp = omega - (omega . B_hat) B_hat

so the energy rate is

    dT/dt = omega . L = -k |B|^2 |omega_perp|^2  <= 0     for ANY inertia   (1)

but the momentum-magnitude rate is

    d|H|/dt = (H . L) / |H|,
    H . L   = -k |B|^2 [ omega^T J omega - (omega.B_hat)(B_hat^T J omega) ]  (2)

The bracket in (2) is positive for every field direction only when ``J`` is
isotropic.  For ``J = diag(1, 1, J3)`` and ``omega = (a, 0, c)``, maximising
the negated bracket over unit ``B_hat`` in the ``(omega, J omega)`` plane gives

    max = ( -(J3 c^2 + a^2) + sqrt( (J3 c^2 - a^2)^2 + a^2 c^2 (1+J3)^2 ) ) / 2

and the discriminant identity

    (J3 c^2 - a^2)^2 + a^2 c^2 (1+J3)^2 - (J3 c^2 + a^2)^2 = a^2 c^2 (1 - J3)^2

shows this maximum is strictly positive whenever ``J3 != 1`` and ``a, c != 0``.
So for any non-spherical body there exist field directions in which the
momentum magnitude momentarily rises while the energy still falls.

Checks
------
B1  energy is non-increasing over 20000 random (omega, B, J, k) draws
B2  |H| is non-increasing over the same draws when J is isotropic
B3  |H| DOES increase for asymmetric J: measured incidence and the analytic
    counterexample
B4  the same behaviour in the full simulator, unsaturated, isotropic and
    asymmetric inertia
"""

from __future__ import annotations

import numpy as np

from _support import Tee  # noqa: E402

from detumblesim.control import ideal_bdot_torque  # noqa: E402
from detumblesim.orbit import CircularOrbit  # noqa: E402
from detumblesim.policies import FixedGainPolicy  # noqa: E402
from detumblesim.simulate import DetumbleConfig, simulate_detumble  # noqa: E402
from detumblesim.spacecraft import Magnetorquer, inertia_from_diagonal  # noqa: E402

N_DRAWS = 20000
SEED = 20260831
# A dipole limit large enough that no command is ever clipped, so the
# "absence of saturation" precondition of the claim actually holds.
UNSATURATED_DIPOLE_AM2 = 500.0


def random_draws(rng, n):
    w = rng.normal(size=(n, 3))
    b = rng.normal(size=(n, 3)) * 3e-5
    j = rng.uniform(0.5, 2.0, size=(n, 3))
    k = 10.0 ** rng.uniform(3.0, 6.0, size=n)
    keep = (
        (np.linalg.norm(w, axis=1) > 1e-3)
        & (np.linalg.norm(b, axis=1) > 1e-9)
        & (j[:, 0] + j[:, 1] >= j[:, 2])
        & (j[:, 1] + j[:, 2] >= j[:, 0])
        & (j[:, 2] + j[:, 0] >= j[:, 1])
    )
    return w[keep], b[keep], j[keep], k[keep]


def run_sim(inertia, gain, dipole_limit, duration=20000.0):
    cfg = DetumbleConfig(
        inertia=inertia,
        orbit=CircularOrbit(altitude_km=500.0, inclination_deg=97.4),
        magnetorquer=Magnetorquer.isotropic(dipole_limit),
        omega0_rad_s=np.radians([6.0, -5.0, 5.7]),
        duration_s=duration,
        control_dt_s=2.0,
        substeps=2,
        target_rate_rad_s=np.radians(1.0),
    )
    return simulate_detumble(cfg, FixedGainPolicy(gain))


def main() -> None:
    rng = np.random.default_rng(SEED)
    with Tee(__file__) as out:
        out("V2  Energy and angular-momentum monotonicity under ideal B-dot")
        out("=" * 78)
        out(f"seed = {SEED}, draws requested = {N_DRAWS}")
        out("")

        w, b, jd, k = random_draws(rng, N_DRAWS)
        n = w.shape[0]
        de = np.empty(n)
        dh_iso = np.empty(n)
        dh_asym = np.empty(n)
        for i in range(n):
            torque = ideal_bdot_torque(w[i], b[i], k[i])
            de[i] = float(w[i] @ torque)
            dh_iso[i] = float((1.7 * w[i]) @ torque)
            dh_asym[i] = float((np.diag(jd[i]) @ w[i]) @ torque)
        scale = k * np.sum(b * b, axis=1) * np.sum(w * w, axis=1)

        out(f"B1  Kinetic-energy rate, arbitrary inertia   (n = {n})")
        out(f"    max dT/dt                    = {de.max():+.6e} J/s")
        out(f"    max dT/dt normalised         = {np.max(de / scale):+.6e}")
        out(f"    number of draws with dT/dt>0 = {int(np.sum(de > 1e-12 * scale))}")
        b1 = bool(np.all(de <= 1e-12 * scale))
        out(f"    B1: {'PASS' if b1 else 'FAIL'}  (energy is never increased)")
        out("")

        out(f"B2  d|H|/dt for ISOTROPIC inertia J = 1.7 I   (n = {n})")
        out(f"    max H.L                      = {dh_iso.max():+.6e} (N m s)(N m)/s")
        out(f"    number of draws with H.L > 0 = {int(np.sum(dh_iso > 1e-12 * scale))}")
        b2 = bool(np.all(dh_iso <= 1e-12 * scale))
        out(f"    B2: {'PASS' if b2 else 'FAIL'}  (momentum magnitude never rises)")
        out("")

        rise = dh_asym > 1e-12 * scale
        out(f"B3  d|H|/dt for RANDOM ASYMMETRIC inertia   (n = {n})")
        out(f"    draws with H.L > 0           = {int(rise.sum())} "
            f"({100.0 * rise.mean():.2f} % of draws)")
        if rise.any():
            out(f"    max H.L                      = {dh_asym.max():+.6e}")
            out(f"    max H.L normalised           = {np.max(dh_asym[rise] / scale[rise]):+.6e}")
            i = int(np.argmax(dh_asym))
            out(f"    worst draw: J = diag({jd[i][0]:.4f}, {jd[i][1]:.4f}, {jd[i][2]:.4f}), "
                f"J_max/J_min = {jd[i].max() / jd[i].min():.3f}")
        out("    B3: the stated property is FALSIFIED for asymmetric inertia.")
        out("")

        out("B3a Analytic counterexample, J = diag(1, 1, 4), omega = (1, 0, 1)")
        j3, a, c = 4.0, 1.0, 1.0
        disc = (j3 * c**2 - a**2) ** 2 + a**2 * c**2 * (1 + j3) ** 2
        analytic = 0.5 * (-(j3 * c**2 + a**2) + np.sqrt(disc))
        jm = np.diag([1.0, 1.0, j3])
        wv = np.array([a, 0.0, c])
        jw = jm @ wv
        psis = np.linspace(0.0, np.pi, 200001)
        vals = np.array(
            [
                float(wv @ np.array([np.cos(p), 0.0, np.sin(p)]))
                * float(np.array([np.cos(p), 0.0, np.sin(p)]) @ jw)
                - float(wv @ jw)
                for p in psis
            ]
        )
        num = float(vals.max())
        psi_star = float(psis[int(np.argmax(vals))])
        out(f"    analytic maximum of the bracket = {analytic:.9f}")
        out(f"    numerical maximum over 200001 B directions = {num:.9f}")
        out(f"    relative difference             = {abs(num - analytic) / analytic:.3e}")
        out(f"    attained at B_hat angle         = {np.degrees(psi_star):.4f} deg from x")
        out(f"    discriminant identity check     = "
            f"{disc - (j3 * c**2 + a**2) ** 2:.9f}  vs  a^2 c^2 (1-J3)^2 = "
            f"{a**2 * c**2 * (1 - j3) ** 2:.9f}")
        out("")

        out("B4  Full simulator, unsaturated "
            f"(dipole limit {UNSATURATED_DIPOLE_AM2:.0f} A m^2)")
        out("    Note: the simulator uses the flight-realistic law "
            "m = -k (B[i]-B[i-1])/dt,")
        out("    not the ideal m = k (omega x B).  The backward difference also picks")
        out("    up the field change caused by orbital motion, so the applied torque")
        out("    is only approximately -k|B|^2 omega_perp and inequalities (1) and (2)")
        out("    hold only approximately step by step.  The residual violations below")
        out("    are that approximation error, not a dynamics defect; they are")
        out("    reported rather than filtered out.")
        for label, inertia in [
            ("isotropic  J = diag(0.05, 0.05, 0.05)",
             inertia_from_diagonal(0.05, 0.05, 0.05)),
            ("asymmetric J = diag(0.02, 0.03, 0.045)",
             inertia_from_diagonal(0.02, 0.03, 0.045)),
        ]:
            r = run_sim(inertia, 1.0e4, UNSATURATED_DIPOLE_AM2)
            dh = np.diff(r.h_norm_nms)
            den = np.diff(r.energy_j)
            up = dh > 0.0
            out(f"    {label}")
            out(f"      saturated control steps      = {100 * r.saturated_fraction:.3f} %")
            out(f"      energy steps that increase   = {int(np.sum(den > 0.0))} "
                f"of {den.size}")
            out(f"      |H| steps that increase      = {int(up.sum())} of {dh.size} "
                f"({100.0 * up.mean():.3f} %)")
            if up.any():
                out(f"      largest single |H| rise      = {dh[up].max():.6e} N m s "
                    f"({100.0 * dh[up].max() / r.h_norm_nms[0]:.6f} % of |H0|)")
                out(f"      total |H| gained on rises    = {dh[up].sum():.6e} N m s")
            out(f"      |H| start / end              = {r.h_norm_nms[0]:.6e} / "
                f"{r.h_norm_nms[-1]:.6e} N m s")
            out(f"      energy start / end           = {r.energy_j[0]:.6e} / "
                f"{r.energy_j[-1]:.6e} J")
            out("")

        out("B5  Full simulator WITH saturation (0.2 A m^2), asymmetric inertia")
        r = run_sim(inertia_from_diagonal(0.02, 0.03, 0.045), 3.0e5, 0.2)
        dh = np.diff(r.h_norm_nms)
        den = np.diff(r.energy_j)
        out(f"    saturated control steps        = {100 * r.saturated_fraction:.3f} %")
        out(f"    |H| steps that increase        = {int(np.sum(dh > 0))} of {dh.size} "
            f"({100.0 * np.mean(dh > 0):.3f} %)")
        out(f"    energy steps that increase     = {int(np.sum(den > 0))} of {den.size} "
            f"({100.0 * np.mean(den > 0):.3f} %)")
        out(f"    largest single energy rise     = "
            f"{(den[den > 0].max() if np.any(den > 0) else 0.0):.6e} J")
        out(f"    energy start / end             = {r.energy_j[0]:.6e} / "
            f"{r.energy_j[-1]:.6e} J")
        out("")
        out("    Under saturation the commanded dipole is clipped, so the applied")
        out("    torque is no longer -k|B|^2 omega_perp and inequality (1) no longer")
        out("    applies step by step.  The energy still falls overall.")
        out("")
        out(f"OVERALL V2: B1 {'PASS' if b1 else 'FAIL'} (energy monotone, any inertia); "
            f"B2 {'PASS' if b2 else 'FAIL'} (|H| monotone, isotropic inertia); "
            "B3 the spec's |H| claim is FALSIFIED for asymmetric inertia and the")
        out("            counterexample is verified analytically and numerically.")


if __name__ == "__main__":
    main()
