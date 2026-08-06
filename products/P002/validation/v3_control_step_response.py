"""V3 - PID/LQR step-response metrics vs analytic second-order expectations.

Claim under test
----------------
With derivative-on-measurement PD control of the plant J s^2 + b s, the
closed loop is EXACTLY the canonical second-order system

    theta(s)/r(s) = Kp / (J s^2 + (Kd + b) s + Kp)
    wn = sqrt(Kp / J)      zeta = (Kd + b) / (2 sqrt(Kp J))

so the simulated response must reproduce the textbook metrics
(Ogata 2010, "Modern Control Engineering", 5th ed., ch. 5):

    Mp = exp(-pi zeta / sqrt(1 - zeta^2))
    tp = pi / (wn sqrt(1 - zeta^2))
    y(t) = 1 - e^{-zeta wn t} [cos(wd t) + zeta/sqrt(1-zeta^2) sin(wd t)]

The 10-90 % rise time is taken from the analytic y(t) numerically (no
approximation formula is used).

For LQR, weights from eq. (10) of control.py must place the closed-loop
poles at |p| = wn with zeta = sqrt(2)/2 (Butterworth pattern) for the
undamped plant.

Run: python validation/v3_control_step_response.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trackforge.control import (  # noqa: E402
    LQRController,
    PIDController,
    bandwidth_estimate,
    disturbance_rejection_rms,
    lqr_weights_from_bandwidth,
    pid_gains_from_bandwidth,
    step_response,
)
from trackforge.dynamics import (  # noqa: E402
    GimbalAxis,
    JitterPSD,
    synthesize_jitter,
)

J = 0.05
B = 0.02
TORQUE_MAX = 2.0
RATE_MAX = 1.0
DT = 1.0e-4
SETPOINT = 1.0e-4


def axis() -> GimbalAxis:
    """Reference axis J = 0.05 kg m^2, b = 0.02 N m s/rad."""
    return GimbalAxis(J, B, TORQUE_MAX, RATE_MAX)


def analytic_step(zeta: float, wn: float, t: np.ndarray) -> np.ndarray:
    """Unit step of the canonical second-order system."""
    wd = wn * math.sqrt(1.0 - zeta**2)
    return 1.0 - np.exp(-zeta * wn * t) * (
        np.cos(wd * t) + zeta / math.sqrt(1.0 - zeta**2) * np.sin(wd * t)
    )


def main() -> int:
    """Run the control validation and print tables."""
    print("V3 - PD/PID/LQR step response vs analytic second-order expectations")
    print(f"plant: J = {J} kg m^2, b = {B} N m s/rad, tau_max = {TORQUE_MAX} N m")
    print(f"integration: RK4, dt = {DT:.1e} s, step = {SETPOINT:.1e} rad")
    print()

    print("A) PD control (Ki = 0) - hand-checkable exact second-order case")
    hdr = (f"{'f_n [Hz]':>9} {'zeta_des':>9} {'zeta_eff':>9} {'Mp sim':>9} "
           f"{'Mp ana':>9} {'dMp':>8} {'tp sim':>9} {'tp ana':>9} {'dtp':>8} "
           f"{'tr sim':>9} {'tr ana':>9} {'dtr':>8}")
    print(hdr)
    print("-" * len(hdr))
    worst = 0.0
    worst_mp_abs = 0.0
    for f_n, zeta_des in ((2.0, 0.5), (5.0, 0.707), (5.0, 0.90), (10.0, 0.707)):
        wn_des = 2.0 * math.pi * f_n
        kp, _, kd = pid_gains_from_bandwidth(J, wn_des, zeta_des)
        # simulate long enough to contain the peak: tp <= pi/(wn sqrt(1-z^2))
        duration = 12.0 / (zeta_des * wn_des)
        _, _, m = step_response(axis(), PIDController(kp, 0.0, kd, TORQUE_MAX),
                                SETPOINT, DT, duration)
        wn = math.sqrt(kp / J)
        zeta = (kd + B) / (2.0 * math.sqrt(kp * J))
        t = np.linspace(0.0, duration, 400001)
        y = analytic_step(zeta, wn, t)
        tr_a = t[int(np.argmax(y >= 0.9))] - t[int(np.argmax(y >= 0.1))]
        if zeta < 1.0:
            mp_a = math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta**2))
            tp_a = math.pi / (wn * math.sqrt(1.0 - zeta**2))
        else:
            mp_a, tp_a = 0.0, float("nan")
        d_mp = (m.overshoot - mp_a) / mp_a if mp_a > 0 else abs(m.overshoot)
        d_tp = (m.peak_time - tp_a) / tp_a if tp_a == tp_a else float("nan")
        d_tr = (m.rise_time - tr_a) / tr_a
        worst_mp_abs = max(worst_mp_abs, abs(m.overshoot - mp_a))
        # relative deviations are only meaningful where the analytic metric is
        # not numerically marginal; Mp < 1 % is judged on the absolute error
        candidates = [d_tp, d_tr] + ([d_mp] if mp_a > 0.01 else [])
        for d in candidates:
            if d == d:  # not NaN
                worst = max(worst, abs(d))
        print(f"{f_n:9.1f} {zeta_des:9.3f} {zeta:9.5f} {m.overshoot:9.5f} "
              f"{mp_a:9.5f} {d_mp:+8.2%} {m.peak_time:9.5f} {tp_a:9.5f} "
              f"{d_tp:+8.2%} {m.rise_time:9.5f} {tr_a:9.5f} {d_tr:+8.2%}")
    print()

    print("B) pointwise trajectory error, PD, f_n = 5 Hz, zeta_des = 0.707")
    kp, ki, kd = pid_gains_from_bandwidth(J, 2 * math.pi * 5.0, 0.707)
    t, y, _ = step_response(axis(), PIDController(kp, 0.0, kd, TORQUE_MAX),
                            SETPOINT, DT, 1.0)
    wn = math.sqrt(kp / J)
    zeta = (kd + B) / (2.0 * math.sqrt(kp * J))
    y_ref = SETPOINT * analytic_step(zeta, wn, t)
    max_abs = float(np.max(np.abs(y - y_ref)))
    print(f"   max |sim - analytic| = {max_abs:.4e} rad "
          f"({max_abs / SETPOINT:.4%} of the step)")
    print(f"   RMS |sim - analytic| = {float(np.sqrt(np.mean((y - y_ref) ** 2))):.4e} rad")
    print()

    print("C) LQR pole placement from eq. (10), undamped plant (b = 0)")
    print(f"{'f_n [Hz]':>9} {'|p| sim':>12} {'|p| des':>12} {'d|p|':>9} "
          f"{'zeta sim':>10} {'zeta ana':>9}")
    print("-" * 66)
    lqr_worst = 0.0
    for f_n in (2.0, 5.0, 10.0, 20.0):
        wn_des = 2.0 * math.pi * f_n
        q, qr, r = lqr_weights_from_bandwidth(J, wn_des)
        lqr = LQRController(GimbalAxis(J, 0.0, TORQUE_MAX, RATE_MAX),
                            q_angle=q, q_rate=qr, r_torque=r)
        p0 = lqr.closed_loop_poles[0]
        zeta_sim = -p0.real / abs(p0)
        d = (abs(p0) - wn_des) / wn_des
        lqr_worst = max(lqr_worst, abs(d), abs(zeta_sim - math.sqrt(0.5)))
        print(f"{f_n:9.1f} {abs(p0):12.5f} {wn_des:12.5f} {d:+9.2e} "
              f"{zeta_sim:10.6f} {math.sqrt(0.5):9.6f}")
    print()

    print("D) controller comparison on the reference plant (from actual runs)")
    q, qr, r = lqr_weights_from_bandwidth(J, 2 * math.pi * 5.0)
    dist = synthesize_jitter(JitterPSD(1e-12, 3.0, 2.0), 10000, 1.0 / 2e-4,
                             np.random.default_rng(7))
    ol_rms = float(np.std(dist))
    print(f"   open-loop disturbance RMS = {ol_rms:.4e} rad "
          "(PSD S0 = 1e-12 rad^2/Hz, fc = 3 Hz, order 2)")
    hdr = (f"{'controller':>12} {'tr [s]':>9} {'Mp':>9} {'ts [s]':>9} "
           f"{'RMS [rad]':>12} {'reject':>8} {'BW [Hz]':>9}")
    print(hdr)
    print("-" * len(hdr))
    controllers = {
        "PD": lambda ax: PIDController(kp, 0.0, kd, ax.torque_max),
        "PID": lambda ax: PIDController(kp, ki, kd, ax.torque_max),
        "LQR": lambda ax: LQRController(ax, q_angle=q, q_rate=qr, r_torque=r),
    }
    for name, fac in controllers.items():
        _, _, m = step_response(axis(), fac(axis()), SETPOINT, 2e-4, 2.0)
        rms = disturbance_rejection_rms(axis(), fac(axis()), dist, 2e-4)
        bw = bandwidth_estimate(axis(), fac(axis()), 2e-4)
        print(f"{name:>12} {m.rise_time:9.5f} {m.overshoot:9.5f} "
              f"{m.settling_time:9.5f} {rms:12.4e} {ol_rms / rms:8.2f} {bw:9.4f}")
    print()

    tol_a, tol_b, tol_c, tol_d = 0.03, 0.01, 1e-6, 0.005
    ok = (worst < tol_a and max_abs / SETPOINT < tol_b and lqr_worst < tol_c
          and worst_mp_abs < tol_d)
    print(f"PASS criteria: (A) worst metric deviation < {tol_a:.0%} -> {worst:.3%}")
    print(f"               (A') worst |Mp_sim - Mp_analytic| < {tol_d} -> "
          f"{worst_mp_abs:.5f}")
    print(f"               (B) trajectory error < {tol_b:.0%} of step -> "
          f"{max_abs / SETPOINT:.4%}")
    print(f"               (C) LQR pole/damping error < {tol_c:.0e} -> {lqr_worst:.2e}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
