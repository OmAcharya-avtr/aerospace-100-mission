"""Validate the twin models: integrator order, closed forms, equilibrium.

Run from the repository root: ``python validation/validate_model.py``
Writes ``validation/validate_model_output.txt``.

References for every reference quantity used here:

* Butcher, J. C., *Numerical Methods for Ordinary Differential Equations*, 3rd
  ed., Wiley, 2016 -- the classical RK4 scheme is fourth order, so halving the
  step must divide the global error by about 16.
* Incropera, F. P. and DeWitt, D. P., *Fundamentals of Heat and Mass Transfer*,
  Chapter 5 -- the lumped-capacitance solution the single-node model is checked
  against in closed form.
* Gilmore, D. G. (ed.), *Spacecraft Thermal Control Handbook, Volume I*, 2nd
  ed., AIAA, 2002, Chapter 1 -- the two-node network and the grey-body
  radiation term.
* Fixsen, D. J. (2009), *Astrophysical Journal* 707, 916 -- the 2.72548 K sink.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.constants as sc

from _bootstrap import Tee, add_src_to_path

ROOT = add_src_to_path()

from twincalib import (  # noqa: E402
    STEFAN_BOLTZMANN,
    THETA_TRUE,
    Inputs,
    LinearNodeModel,
    ThermalTwinModel,
    build_inputs,
    calibration_model,
    design,
    linear_model,
    truth_model,
)
from twincalib.scenarios import LINEAR_THETA_TRUE  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    out = Tee(ROOT / "validation" / "validate_model_output.txt")
    out("validate_model.py -- twin models, integrator and closed forms")
    out(f"numpy {np.__version__}, scipy {sc.__name__} loaded for its constants")

    out.rule("1. Physical constants against scipy.constants")
    rel = abs(STEFAN_BOLTZMANN / sc.Stefan_Boltzmann - 1.0)
    out(f"twincalib STEFAN_BOLTZMANN   = {STEFAN_BOLTZMANN:.15e} W/m^2/K^4")
    out(f"scipy.constants              = {sc.Stefan_Boltzmann:.15e} W/m^2/K^4")
    out(f"relative difference          = {rel:.3e}   tolerance 1e-12")
    out(f"verdict: {'PASS' if rel < 1e-12 else 'FAIL'}")

    out.rule("2. Single-node integrator against the closed-form solution")
    out("C dT/dt = q - k (T - T_sink); Incropera and DeWitt Chapter 5.")
    out("C = 800 J/K, k = 2.5 W/K, q = 40 W, T_sink = 280 K, T0 = 300 K,")
    out("so tau = C/k = 320.000 s and the steady state is 280 + q/k = 296.000 K.")
    model = linear_model()
    steady = model.t_sink + LINEAR_THETA_TRUE[2] / LINEAR_THETA_TRUE[1]
    out(f"steady state from the closed form = {steady:.6f} K")
    out("")
    out(f"  {'dt [s]':>8}{'worst |numeric - exact| [K]':>30}{'ratio to previous':>20}")
    previous = None
    orders = []
    for dt in (80.0, 40.0, 20.0, 10.0, 5.0):
        times = np.arange(0.0, 3200.0 + dt, dt)
        inputs = Inputs(times, np.zeros(times.size), np.zeros(times.size))
        numeric = model.simulate(LINEAR_THETA_TRUE, inputs, [300.0])[:, 0]
        exact = model.analytic_solution(LINEAR_THETA_TRUE, times, 300.0)
        worst = float(np.max(np.abs(numeric - exact)))
        ratio = "" if previous is None else f"{previous / worst:18.3f}"
        if previous is not None:
            orders.append(previous / worst)
        out(f"  {dt:>8.1f}{worst:>30.6e}{ratio:>20}")
        previous = worst
    out("")
    out(
        "a fourth-order scheme gives a ratio near 16 per halving "
        "(Butcher 2016, section 23)"
    )
    ok = all(10.0 < r < 24.0 for r in orders)
    out(f"measured ratios: {', '.join(f'{r:.3f}' for r in orders)}")
    out(f"verdict: {'PASS' if ok else 'FAIL'}  criterion every ratio in (10, 24)")

    out.rule("3. Single-node closed form against a hand calculation")
    out("At t = tau = 320 s: T = 296 + (300 - 296) exp(-1) = 296 + 4 * 0.36787944")
    hand = 296.0 + 4.0 * float(np.exp(-1.0))
    value = float(
        model.analytic_solution(LINEAR_THETA_TRUE, np.array([0.0, 320.0]), 300.0)[1]
    )
    out(f"hand value   = {hand:.8f} K")
    out(f"closed form  = {value:.8f} K")
    out(f"difference   = {abs(value - hand):.3e} K   tolerance 1e-9")
    out(f"verdict: {'PASS' if abs(value - hand) < 1e-9 else 'FAIL'}")

    out.rule("4. Two-node model relaxes to its algebraic equilibrium")
    out("h_rad (T2^4 - T_space^4) = P_dis + q_sun s + q_par and T1 = T2 + P_dis/k12")
    twin = calibration_model()
    out(f"  {'P [W]':>7}{'s':>5}{'T1 algebraic':>15}{'T2 algebraic':>15}"
        f"{'T1 integrated':>16}{'T2 integrated':>16}{'worst diff [K]':>16}")
    worst_all = 0.0
    for p_dis, sun in ((12.0, 0.0), (25.0, 0.65), (35.0, 1.0)):
        t1_a, t2_a = twin.steady_state(THETA_TRUE, p_dis, sun)
        times = np.arange(0.0, 400001.0, 100.0)
        inputs = Inputs(
            times, np.full(times.size, p_dis), np.full(times.size, sun)
        )
        final = twin.simulate(THETA_TRUE, inputs, [250.0, 240.0])[-1]
        diff = max(abs(final[0] - t1_a), abs(final[1] - t2_a))
        worst_all = max(worst_all, diff)
        out(
            f"  {p_dis:>7.1f}{sun:>5.2f}{t1_a:>15.6f}{t2_a:>15.6f}"
            f"{final[0]:>16.6f}{final[1]:>16.6f}{diff:>16.3e}"
        )
    out("")
    out(f"worst discrepancy = {worst_all:.3e} K   tolerance 1e-3 K after 400000 s")
    out(f"verdict: {'PASS' if worst_all < 1e-3 else 'FAIL'}")

    out.rule("5a. Two-node integrator alone, on an input the grid represents exactly")
    out("The design's 30 s step against the same scheme at 30/8 = 3.75 s, with the")
    out("SAME zero-order-hold input profile resampled onto the fine grid, so that")
    out("this number is the integrator's error and nothing else.")
    d = design("baseline-eclipse")
    reference_stride = 8
    fine_dt = d.dt / reference_stride
    coarse_inputs = build_inputs(d)
    fine_times = np.arange(0.0, d.duration + fine_dt, fine_dt)
    held_index = np.minimum(
        (fine_times // d.dt).astype(int), coarse_inputs.times.size - 1
    )
    fine_held = Inputs(
        fine_times,
        coarse_inputs.dissipation[held_index],
        coarse_inputs.sun[held_index],
    )
    fine = twin.simulate(THETA_TRUE, fine_held, d.initial_state)[::reference_stride]
    coarse = twin.simulate(THETA_TRUE, coarse_inputs, d.initial_state)
    worst = float(np.max(np.abs(fine - coarse)))
    out(f"worst |T(dt=30 s) - T(dt=3.75 s)| = {worst:.6e} K")
    out(f"thermometer noise for comparison  = {d.sigma_k:.3f} K")
    out(f"ratio to the noise                = {worst / d.sigma_k:.3e}")
    out(
        f"verdict: {'PASS' if worst < 0.01 * d.sigma_k else 'FAIL'}  "
        "criterion below 1 per cent of the measurement noise"
    )

    out.rule("5b. The input grid itself, which is a larger error than the integrator")
    out("This check FAILED on its first run and the failure is the finding, kept")
    out("here rather than removed. Refining the grid also refines the zero-order")
    out("hold on the exogenous inputs, and the eclipse edge at")
    out("0.65 * 5700 s = 3705 s is NOT a multiple of 30 s, so a 30 s grid places")
    out("that edge up to 15 s away from where a 3.75 s grid places it. The")
    out("resulting temperature difference is far larger than the integrator's own")
    out("error measured in 5a.")
    fine_true = Inputs(
        fine_times,
        np.where(
            ((fine_times / d.duty_period) % 1.0) < d.duty_fraction,
            d.dissipation_high,
            d.dissipation_low,
        ),
        np.where(
            ((fine_times / d.orbit_period) % 1.0) < (1.0 - d.eclipse_fraction),
            1.0,
            0.0,
        ),
    )
    fine_shifted = twin.simulate(THETA_TRUE, fine_true, d.initial_state)[
        ::reference_stride
    ]
    quantisation = float(np.max(np.abs(fine_shifted - coarse)))
    edge = (1.0 - d.eclipse_fraction) * d.orbit_period
    out("")
    out(f"eclipse edge                       = {edge:.3f} s")
    out(f"edge / dt                          = {edge / d.dt:.4f}  (not an integer)")
    out(f"worst |T(30 s grid) - T(3.75 s grid, true edges)| = {quantisation:.6f} K")
    out(f"ratio to the integrator error in 5a = {quantisation / worst:.1f}")
    out(f"ratio to the thermometer noise      = {quantisation / d.sigma_k:.3f}")
    out("")
    out("CONSEQUENCE, recorded in README Limitations: the zero-order-hold input")
    out("grid is part of the model, not an implementation detail. Every number in")
    out("this repository is internally consistent because the data generator and")
    out("the fitter use the same grid, so the quantisation cancels exactly. A real")
    out("calibration against real telemetry would not have that cancellation and")
    out("would carry this as a model error of order 0.6 K on this design.")
    out("verdict: REPORTED, not bounded")

    out.rule("6. Linearised time constants against the sampling interval")
    fast, slow = twin.time_constants(THETA_TRUE, 280.0)
    series = (THETA_TRUE[0] * THETA_TRUE[1] / (THETA_TRUE[0] + THETA_TRUE[1])) / (
        THETA_TRUE[2]
    )
    out(f"fast mode from the 2x2 eigenvalues      = {fast:.3f} s")
    out(f"series-capacitance estimate C_eq/k12    = {series:.3f} s")
    out(f"slow mode from the 2x2 eigenvalues      = {slow:.3f} s")
    out(f"radiative estimate (C1+C2)/(4 h T2^3)   = "
        f"{(THETA_TRUE[0] + THETA_TRUE[1]) / (4.0 * THETA_TRUE[3] * 280.0**3):.3f} s")
    out(f"design sampling interval                = {design('baseline-eclipse').dt:g} s")
    out(f"dt / tau_fast                           = {30.0 / fast:.4f}")
    out("the step is a ninth of the fastest time constant, which is why the")
    out("fixed-step scheme in check 5 agrees with itself to 1e-5 K")

    out.rule("7. Stability limit is reported, not silently wrong")
    bad = (800.0, 4000.0, 400.0, 1.205e-8, 55.0, 18.0)
    try:
        twin.simulate(bad, build_inputs(d), d.initial_state)
        out("verdict: FAIL  the divergent case did not raise")
    except ValueError as error:
        out(f"k12 = 400 W/K at dt = 30 s raised ValueError: {error}")
        out("verdict: PASS")

    out.rule("8. Declared model discrepancy, measured")
    out("truth model: k12_eff = k12 (1 + 1.5 (T1 - 290 K)/100 K)")
    out("The calibration model has no such term and cannot reproduce it.")
    out(f"  {'design':<18}{'RMS [K]':>10}{'max |diff| [K]':>16}{'RMS/sigma':>11}")
    for name in ("baseline-eclipse", "full-sun", "lumped-nodes", "weak-excitation"):
        dd = design(name)
        inputs = build_inputs(dd)
        clean = calibration_model().observe(
            dd.theta_true, inputs, dd.initial_state, dd.channels, dd.sample_stride
        )
        dirty = truth_model().observe(
            dd.theta_true, inputs, dd.initial_state, dd.channels, dd.sample_stride
        )
        diff = dirty - clean
        rms = float(np.sqrt(np.mean(diff**2)))
        out(
            f"  {name:<18}{rms:>10.4f}{float(np.max(np.abs(diff))):>16.4f}"
            f"{rms / dd.sigma_k:>11.3f}"
        )
    out("")
    out("lumped-nodes is the control design: its discrepancy is well below the")
    out("noise, so a residual model has almost nothing real to find there.")

    out.rule("9. Model validity range on the shipped designs")
    out("The declared validity range of the two-node model is 200-350 K.")
    out(f"  {'design':<18}{'min T [K]':>11}{'max T [K]':>11}{'inside 200-350':>16}")
    inside_all = True
    for name in ("baseline-eclipse", "full-sun", "lumped-nodes", "weak-excitation"):
        dd = design(name)
        states = truth_model().simulate(
            dd.theta_true, build_inputs(dd), dd.initial_state
        )
        lo, hi = float(states.min()), float(states.max())
        inside = 200.0 <= lo and hi <= 350.0
        inside_all = inside_all and inside
        out(f"  {name:<18}{lo:>11.3f}{hi:>11.3f}{str(inside):>16}")
    out("")
    out(f"verdict: {'PASS' if inside_all else 'FAIL'}")

    out.rule("10. Negative controls on the input validation")
    checks = [
        ("times with 1 entry", lambda: Inputs(np.array([0.0]), np.zeros(1), np.zeros(1))),
        (
            "non-monotone times",
            lambda: Inputs(np.array([0.0, 2.0, 1.0]), np.zeros(3), np.zeros(3)),
        ),
        (
            "non-uniform times",
            lambda: Inputs(np.array([0.0, 1.0, 3.0]), np.zeros(3), np.zeros(3)),
        ),
        (
            "sun outside [0, 1]",
            lambda: Inputs(np.arange(3.0), np.zeros(3), np.full(3, 2.0)),
        ),
        ("negative parameter", lambda: twin.check_parameters([-1.0] * 6)),
        ("wrong parameter count", lambda: twin.check_parameters([1.0, 2.0])),
        ("negative sink temperature", lambda: LinearNodeModel(t_sink=-5.0)),
        ("non-finite discrepancy slope", lambda: ThermalTwinModel(conductance_slope=np.inf)),
        (
            "total heat input not positive",
            lambda: twin.steady_state(
                (800.0, 4000.0, 2.5, 1.205e-8, 1e-12, 1e-12), -10.0, 0.0
            ),
        ),
    ]
    failures = 0
    for label, call in checks:
        try:
            call()
            out(f"  {label:<32} NOT REJECTED  <- defect")
            failures += 1
        except (ValueError, TypeError) as error:
            out(f"  {label:<32} rejected: {str(error)[:60]}")
    out("")
    out(f"verdict: {'PASS' if failures == 0 else 'FAIL'}  {len(checks)} negative controls")

    out("")
    out(f"elapsed {time.perf_counter() - started:.1f} s")
    out.save()


if __name__ == "__main__":
    main()
