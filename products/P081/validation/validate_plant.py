"""The declared plant against an independent closed form, and its two identities.

What is checked, against what
-----------------------------
1. **The integrator against the exact solution of equation (18)**, derived in
   :mod:`twindiverge.reference` and used by nothing else in the package. This is
   the independent reference for every trajectory in this repository: it is
   closed-form algebra, not a second numerical integration, so it cannot share
   the integrator's error.
2. **The convergence order**, by Richardson extrapolation against that closed
   form. The declared plant is too smooth to measure it on -- its truncation
   error at the declared step is already below rounding -- so a declared stiff
   parameter set is used, named in the output as a numerical exercise and not a
   claim about any wheel.
3. **Identity A**, equation (6): the five rate sensitivities sum to zero, at
   every sample, for every campaign and at perturbed parameter points as well
   as the nominal one. This is the algebraic fact the whole product rests on.
4. **Identity B**, equation (7): the five torque sensitivities sum to the torque
   signal itself.
5. **The forward sensitivities against a complex-step derivative**, computed by
   an independently written complex-arithmetic integrator in this file. The
   right-hand side of equation (1) is a polynomial in the rate and linear in the
   parameters, so it is analytic and the complex-step derivative
   ``Im(omega(theta_j * (1 + i*h))) / h`` is exact to machine precision with no
   subtractive cancellation at all -- unlike a finite difference, whose error
   floor is ``eps * |omega| / h``. Both are reported: the complex step is the
   check, and the central difference is reported beside it with its own measured
   floor so that the difference between the two routes is visible.
6. **The zero-order-hold handling**, which is where this build found a defect.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that running it ADDS no __pycache__ directory and no
# .pyc file, so an interpreter that caches its own imports here makes that
# check report itself. It is set in each entry script, before _bootstrap is
# imported, and not inside _bootstrap, because CPython writes a module's cache
# BEFORE executing that module: _bootstrap cannot set this on its own behalf.
# The imports below therefore follow a statement and carry an E402 waiver.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import Tee, bootstrap, environment_lines  # noqa: E402

ROOT = bootstrap()

from twindiverge.plant import (  # noqa: E402
    CAMPAIGNS,
    NOMINAL,
    OMEGA_MAX,
    OMEGA_MIN,
    PARAMETER_NAMES,
    PARAMETER_UNITS,
    Campaign,
    WheelParameters,
    perturb,
    simulate,
    simulate_with_sensitivities,
)
from twindiverge.reference import exact_campaign, exact_rate  # noqa: E402
from twindiverge.report import format_table, number  # noqa: E402


def complex_step_sensitivity(
    parameters: WheelParameters, campaign: Campaign, step: float = 1.0e-20
) -> np.ndarray:
    """Sensitivities by complex-step differentiation, shape ``(samples, 5)``.

    An independent implementation. The Runge-Kutta recurrence is written out
    again here in complex arithmetic rather than reusing
    ``twindiverge.plant``, because a check that calls the code under test is
    not a check. The right-hand side of equation (1) is a polynomial in the
    rate and linear in the parameters, hence analytic, so

        d(omega)/d(log theta_j) = Im(omega(theta_j * (1 + i*h))) / h

    is exact up to O(h^2) in the imaginary part and carries NO subtractive
    cancellation: the real and imaginary parts never meet. With h = 1e-20 the
    truncation term is 1e-40 and the result is machine-accurate.
    """
    base = parameters.as_array()
    columns = []
    for index in range(5):
        values = base.astype(complex)
        values[index] = values[index] * (1.0 + 1.0j * step)
        inertia, torque_constant, coulomb, viscous, drag = values
        period = campaign.sample_period_s
        rate = complex(campaign.initial_rate_rad_s)
        rates = np.empty(campaign.samples, dtype=complex)

        def derivative(
            value: complex,
            current: float,
            inertia: complex = inertia,
            torque_constant: complex = torque_constant,
            coulomb: complex = coulomb,
            viscous: complex = viscous,
            drag: complex = drag,
        ) -> complex:
            # The five coefficients are bound as defaults rather than captured,
            # so that the closure cannot see a later iteration's values.
            return (
                torque_constant * current
                - coulomb
                - viscous * value
                - drag * value * value
            ) / inertia

        for sample in range(campaign.samples):
            rates[sample] = rate
            if sample == campaign.samples - 1:
                break
            held = campaign._current_at(sample * period)
            first = derivative(rate, held)
            second = derivative(rate + 0.5 * period * first, held)
            third = derivative(rate + 0.5 * period * second, held)
            fourth = derivative(rate + period * third, held)
            rate = rate + period * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
        columns.append(np.imag(rates) / step)
    return np.column_stack(columns)


#: The declared stiff set used only to measure the convergence order. A viscous
#: coefficient of 1.3e-2 N m s / rad and a torque constant of 1.3025 N m / A are
#: both implausible by three orders of magnitude; this is arithmetic, not a
#: wheel.
ORDER_PARAMETERS = WheelParameters(1.30e-2, 1.3025, 2.50e-3, 1.30e-2, 2.00e-8)
ORDER_INITIAL_RATE = 150.0
ORDER_DURATION = 2.0
ORDER_STEPS = (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125)

#: Perturbed parameter points at which the identities are re-checked. Declared,
#: not random: each moves a different subset by a different amount.
PROBE_OFFSETS = (
    np.zeros(5),
    np.array([0.2, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, -0.15, 0.0, 0.0, 0.0]),
    np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
    np.array([-0.12, 0.09, -0.07, 0.21, -0.18]),
)


def main() -> int:
    out = Tee("validate_plant_output.txt")
    out.rule("twindiverge -- the declared plant, against mathematics it does not use")
    for line in environment_lines():
        out(line)

    out.rule("0. the declared plant")
    out(
        format_table(
            ["parameter", "nominal", "unit"],
            [
                [name, number(value), unit]
                for name, value, unit in zip(
                    PARAMETER_NAMES, NOMINAL.as_array(), PARAMETER_UNITS, strict=True
                )
            ],
        )
    )
    out("")
    out("Values are DECLARED, not measured, and are not taken from any source.")
    out(f"Declared validity window : {OMEGA_MIN} to {OMEGA_MAX} rad/s, one sign only.")
    out("")
    rows = []
    for name, campaign in CAMPAIGNS.items():
        trajectory = simulate(NOMINAL, campaign)
        rows.append(
            [
                name,
                f"{campaign.duration_s:.1f}",
                str(campaign.samples),
                f"{campaign.drive_duration_s:.1f}",
                f"{float(trajectory.rate_rad_s.min()):.6f}",
                f"{float(trajectory.rate_rad_s.max()):.6f}",
                f"{trajectory.rate_excursion_rad_s:.6f}",
            ]
        )
    out(
        format_table(
            ["campaign", "length [s]", "samples", "driven [s]",
             "min rate", "max rate", "excursion"],
            rows,
        )
    )

    out.rule("1. the integrator against the exact closed form, equations (20) and (21)")
    out("twindiverge.reference solves equation (18) in closed form. Nothing in the")
    out("package uses it. Exact in exact arithmetic; the figures below are rounding.")
    out("")
    rows = []
    worst = 0.0
    for name, campaign in CAMPAIGNS.items():
        integrated = simulate(NOMINAL, campaign).rate_rad_s
        closed_form = exact_campaign(NOMINAL, campaign)
        relative = float(np.abs((integrated - closed_form) / closed_form).max())
        worst = max(worst, relative)
        rows.append([
            name, f"{float(np.abs(integrated - closed_form).max()):.6e}",
            f"{relative:.6e}",
        ])
    out(format_table(["campaign", "worst absolute [rad/s]", "worst relative"], rows))
    out("")
    out.check(
        "the integrator matches the closed form of equation (18) on every campaign",
        worst < 1.0e-13,
        f"worst relative difference {worst:.6e} over {len(CAMPAIGNS)} campaigns",
    )

    out.rule("2. convergence order, by Richardson extrapolation")
    out("The declared plant's truncation error at its declared 0.1 s step is already")
    out("below rounding, so the order is measured on the declared stiff set below.")
    out("")
    out(
        format_table(
            ["parameter", "value", "unit"],
            [
                [name, number(value), unit]
                for name, value, unit in zip(
                    PARAMETER_NAMES, ORDER_PARAMETERS.as_array(), PARAMETER_UNITS,
                    strict=True,
                )
            ],
        )
    )
    out("")
    truth = float(
        exact_rate(ORDER_PARAMETERS, 1.0, ORDER_INITIAL_RATE, np.array([ORDER_DURATION]))[0]
    )
    errors = []
    for step in ORDER_STEPS:
        samples = int(round(ORDER_DURATION / step)) + 1
        campaign = Campaign(
            f"order{step}", step, samples, 1.0, ORDER_DURATION,
            initial_rate_rad_s=ORDER_INITIAL_RATE,
        )
        errors.append(
            abs(float(simulate(ORDER_PARAMETERS, campaign).rate_rad_s[-1]) - truth)
        )
    rows = []
    orders = []
    for index, step in enumerate(ORDER_STEPS):
        order = (
            math.log2(errors[index - 1] / errors[index]) if index else float("nan")
        )
        if index:
            orders.append(order)
        rows.append([f"{step:g}", f"{errors[index]:.6e}", number(order, 4)])
    out(f"exact terminal rate at t = {ORDER_DURATION} s : {truth:.12f} rad/s")
    out("")
    out(format_table(["step [s]", "absolute error [rad/s]", "measured order"], rows))
    out("")
    out.check(
        "the integrator converges at fourth order",
        3.8 < min(orders) and max(orders) < 4.6,
        f"measured orders from {min(orders):.4f} to {max(orders):.4f}",
    )

    out.rule("3. identity A, equation (6): the rate channel cannot see the torque scale")
    out("sum over j of d(omega)/d(log theta_j) = 0, at every sample. Proved on paper")
    out("in twindiverge.plant; the rank computation of baseline 1 is the independent")
    out("numerical route and is checked in validate_identifiability.py.")
    out("")
    rows = []
    worst_a = 0.0
    for name, campaign in CAMPAIGNS.items():
        for offset in PROBE_OFFSETS:
            trajectory = simulate_with_sensitivities(perturb(NOMINAL, offset), campaign)
            sensitivity = trajectory.rate_sensitivity
            assert sensitivity is not None
            scale = float(np.abs(sensitivity).max())
            relative = float(np.abs(sensitivity.sum(axis=1)).max()) / scale
            worst_a = max(worst_a, relative)
            rows.append([
                name,
                np.array2string(offset, precision=2),
                f"{scale:.6e}",
                f"{relative:.6e}",
            ])
    out(format_table(
        ["campaign", "log-parameter offset", "largest sensitivity", "relative residual"],
        rows,
    ))
    out("")
    out.check(
        "identity A holds at every sample, every campaign and every probe point",
        worst_a < 1.0e-11,
        f"worst relative residual {worst_a:.6e} over {len(rows)} combinations",
    )

    out.rule("4. identity B, equation (7): the torque channel can")
    out("sum over j of d(tau)/d(log theta_j) = tau(t), exactly.")
    out("")
    rows = []
    worst_b = 0.0
    for name, campaign in CAMPAIGNS.items():
        for offset in PROBE_OFFSETS:
            trajectory = simulate_with_sensitivities(perturb(NOMINAL, offset), campaign)
            sensitivity = trajectory.torque_sensitivity
            assert sensitivity is not None
            scale = float(np.abs(trajectory.torque_nm).max())
            relative = float(
                np.abs(sensitivity.sum(axis=1) - trajectory.torque_nm).max()
            ) / scale
            worst_b = max(worst_b, relative)
            rows.append([
                name, np.array2string(offset, precision=2),
                f"{scale:.6e}", f"{relative:.6e}",
            ])
    out(format_table(
        ["campaign", "log-parameter offset", "largest torque [N m]", "relative residual"],
        rows,
    ))
    out("")
    out.check(
        "identity B holds at every sample, every campaign and every probe point",
        worst_b < 1.0e-10,
        f"worst relative residual {worst_b:.6e} over {len(rows)} combinations",
    )

    out.rule("5. the forward sensitivities against an independent complex step")
    out("Equation (3) integrated alongside the state, against a complex-step")
    out("derivative computed by a separate complex-arithmetic Runge-Kutta recurrence")
    out("written out again in this file. Equation (1) is a polynomial in the rate and")
    out("linear in the parameters, hence analytic, so the complex step is exact to")
    out("machine precision with no subtractive cancellation. A central difference is")
    out("reported beside it with its own cancellation floor, eps * |omega| / h, which")
    out("is the reason the complex step and not the difference is the check.")
    out("")
    difference_step = 1.0e-5
    rows = []
    worst_c = 0.0
    worst_margin = 0.0
    for name, campaign in CAMPAIGNS.items():
        analytic = simulate_with_sensitivities(NOMINAL, campaign).rate_sensitivity
        assert analytic is not None
        complex_step = complex_step_sensitivity(NOMINAL, campaign)
        peak_rate = float(np.abs(simulate(NOMINAL, campaign).rate_rad_s).max())
        for index, parameter in enumerate(PARAMETER_NAMES):
            scale = float(np.abs(analytic[:, index]).max())
            offset = np.zeros(5)
            offset[index] = difference_step
            forward = simulate(perturb(NOMINAL, offset), campaign).rate_rad_s
            backward = simulate(perturb(NOMINAL, -offset), campaign).rate_rad_s
            central = (forward - backward) / (2.0 * difference_step)
            if scale == 0.0:
                exact_zero = (
                    float(np.abs(complex_step[:, index]).max()) == 0.0
                    and float(np.abs(central).max()) == 0.0
                )
                rows.append([
                    name, parameter, "0.000000e+00", "0.000000e+00", "0.000000e+00",
                    "unexcited",
                ])
                out.check(
                    f"{name}/{parameter}: an unexcited parameter is zero on all "
                    "three routes",
                    exact_zero,
                    "complex step and central difference both exactly zero"
                    if exact_zero
                    else "a route returned a non-zero sensitivity",
                )
                continue
            relative = float(np.abs(complex_step[:, index] - analytic[:, index]).max()) / scale
            central_relative = float(np.abs(central - analytic[:, index]).max()) / scale
            floor = np.finfo(float).eps * peak_rate / (difference_step * scale)
            worst_c = max(worst_c, relative)
            worst_margin = max(worst_margin, central_relative / floor)
            rows.append([
                name, parameter, f"{scale:.6e}", f"{relative:.6e}",
                f"{central_relative:.6e}", f"{floor:.6e}",
            ])
    out(format_table(
        ["campaign", "parameter", "largest sensitivity", "complex step",
         "central difference", "difference floor"],
        rows,
    ))
    out("")
    out("'complex step' and 'central difference' are each the worst relative")
    out("disagreement with equation (3) over the record; 'difference floor' is the")
    out("central difference's own cancellation floor, eps * max|omega| / (h * |s_j|).")
    out("")
    out.check(
        "the forward sensitivities agree with the independent complex step",
        worst_c < 1.0e-12,
        f"worst relative difference {worst_c:.6e} over {len(rows)} columns",
    )
    out.check(
        "the central difference agrees with equation (3) to within its own floor",
        worst_margin < 20.0,
        f"worst central-difference disagreement is {worst_margin:.3f} times its "
        "cancellation floor",
    )

    out.rule("6. the zero-order hold, where this build found a defect")
    out("The current is sampled ONCE PER STEP at the step's left endpoint and used")
    out("for all four Runge-Kutta stages. The first version of the integrator")
    out("evaluated it per stage, so the fourth stage of the step ENDING at the")
    out("current switch saw the new current. Measured then: global error")
    out("4.102564e-02 rad/s at the end of a 20 s driven segment at a 0.1 s step, and")
    out("a convergence order of 1.0000 over steps from 0.4 s to 0.025 s. Measured")
    out("now, at the sample immediately after the switch of the `full` campaign:")
    out("")
    campaign = CAMPAIGNS["full"]
    switch = int(round(campaign.drive_duration_s / campaign.sample_period_s))
    integrated = simulate(NOMINAL, campaign).rate_rad_s
    closed_form = exact_campaign(NOMINAL, campaign)
    absolute = abs(float(integrated[switch] - closed_form[switch]))
    relative = absolute / abs(float(closed_form[switch]))
    out(f"  switch sample index      : {switch} (t = {campaign.drive_duration_s} s)")
    out(f"  integrated rate          : {float(integrated[switch]):.12f} rad/s")
    out(f"  closed-form rate         : {float(closed_form[switch]):.12f} rad/s")
    out(f"  absolute difference      : {absolute:.6e} rad/s")
    out(f"  relative difference      : {relative:.6e}")
    out(f"  the defect's own figure  : 4.102564e-02 rad/s, a factor of "
        f"{4.102564e-02 / max(absolute, 1e-300):.3e} larger")
    out("")
    out.check(
        "the sample at the current switch matches the closed form",
        relative < 1.0e-14,
        f"relative difference {relative:.6e}",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
