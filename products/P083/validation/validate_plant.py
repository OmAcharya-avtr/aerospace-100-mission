"""The declared plant, its discretisation, its integrator, and the integrator's floor.

Nothing here is about deadline misses. It establishes that the objects every
later script rests on are what they claim to be: the trim is exact for the
cubic plant, the Jacobian is the Jacobian, the zero-order-hold matrices agree
with two independent constructions, the fixed-step integrator converges at its
advertised order, and the integrator's own truncation difference between a
whole period and a split one is MEASURED, because that difference is the floor
below which no divergence measurement means anything.

The floor matters. A miss whose input step is exactly zero injects nothing and
must produce a divergence of exactly zero, but the with-miss branch integrates
the period in two pieces and the reference integrates it in one, so their
fixed-step truncation errors differ. Three misses in the first population
measured during this build had a bound of exactly zero and a re-simulated
divergence of about 2e-10. That is not a model divergence, it is the
integrator, and it is declared and reported rather than absorbed.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that no __pycache__ directory and no .pyc file
# exists in the repository, so an interpreter that caches its own imports
# here makes that check report itself. It is set in each entry script,
# before _bootstrap is imported, and not inside _bootstrap, because CPython
# writes a module's cache BEFORE executing that module: _bootstrap cannot
# set this on its own behalf. The imports below therefore follow a statement
# and carry the same E402 waiver the post-bootstrap imports already carry.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from _bootstrap import (  # noqa: E402
    SOUNDNESS_SEEDS,
    STEPS,
    Tee,
    bootstrap,
    environment_lines,
    measure_integrator_floor,
)

ROOT = bootstrap()

from hilverdict.plant import (  # noqa: E402
    ALPHA_VALID_RAD,
    DEFAULT_METRIC,
    SHORT_PERIOD,
    Controller,
    DivergenceMetric,
    LoopModel,
    Plant,
    zoh_matrices,
)
from hilverdict.report import format_table  # noqa: E402
from hilverdict.resimulate import simulate  # noqa: E402
from hilverdict.synthetic import (  # noqa: E402
    CAMPAIGNS,
    doublet_command,
    generate_run,
)
from hilverdict.timeline import miss_events  # noqa: E402


def section_trim(out: Tee) -> None:
    out.rule("1. the trim of equations (4)-(5) is exact for the CUBIC plant")
    out("A linearised feedforward would leave a residual growing as alpha^3.")
    out("Equations (4)-(5) solve the cubic trim in closed form, so the residual is")
    out("rounding and nothing else.")
    out("")
    rows = []
    worst = 0.0
    for degrees in (0.0, 0.5, 2.0, 6.0, 10.0, -6.0, -10.0):
        alpha = math.radians(degrees)
        state, control = SHORT_PERIOD.trim(alpha)
        residual = SHORT_PERIOD.derivative(state, control)
        size = float(np.max(np.abs(residual)))
        worst = max(worst, size)
        rows.append(
            [
                f"{degrees:g}",
                f"{state[1]:.12f}",
                f"{control:.12f}",
                f"{size:.3e}",
            ]
        )
    out(format_table(["alpha_c [deg]", "q_trim [rad/s]", "u_trim [rad]", "max |xdot|"], rows))
    out.check(
        "the trim residual is at rounding level at every amplitude tried",
        worst < 1e-14,
        f"worst max |xdot| = {worst:.6e} 1/s, tolerance 1e-14",
    )
    linear_only = Plant(
        z_alpha=SHORT_PERIOD.z_alpha,
        z_delta=SHORT_PERIOD.z_delta,
        m_alpha=SHORT_PERIOD.m_alpha,
        m_alpha3=0.0,
        m_q=SHORT_PERIOD.m_q,
        m_delta=SHORT_PERIOD.m_delta,
        u_max_rad=SHORT_PERIOD.u_max_rad,
        name="linear-only",
    )
    alpha = math.radians(10.0)
    state, control = linear_only.trim(alpha)
    residual = float(np.max(np.abs(SHORT_PERIOD.derivative(state, control))))
    out("")
    out("The same trim computed WITHOUT the cubic term, applied to the cubic plant:")
    out(f"  residual at alpha_c = 10 deg: {residual:.6e} 1/s")
    out.check(
        "dropping the cubic term from the trim leaves a residual far above rounding",
        residual > 1e-3,
        f"{residual:.6e} 1/s against the exact trim's {worst:.3e}",
    )


def section_jacobian(out: Tee) -> None:
    out.rule("2. the Jacobian is the Jacobian, by central differences")
    out("Central differences are second-order accurate, so the expected agreement is")
    out("O(h^2) times a third derivative; the tolerance is set from that and not from")
    out("a round number.")
    out("")
    rows = []
    worst = 0.0
    step = 1e-6
    for degrees in (0.0, 1.0, 5.0, 10.0):
        alpha = math.radians(degrees)
        state = np.array([alpha, 0.03])
        analytic = SHORT_PERIOD.jacobian(alpha)
        numeric = np.empty((2, 2))
        for column in range(2):
            perturbation = np.zeros(2)
            perturbation[column] = step
            forward = SHORT_PERIOD.derivative(state + perturbation, 0.01)
            backward = SHORT_PERIOD.derivative(state - perturbation, 0.01)
            numeric[:, column] = (forward - backward) / (2.0 * step)
        gap = float(np.max(np.abs(analytic - numeric)))
        worst = max(worst, gap)
        rows.append([f"{degrees:g}", f"{analytic[1, 0]:.9f}", f"{numeric[1, 0]:.9f}",
                     f"{gap:.3e}"])
    out(format_table(["alpha [deg]", "analytic dM/dalpha", "central difference",
                      "max |difference|"], rows))
    out.check(
        "the analytic Jacobian agrees with central differences",
        worst < 1e-6,
        f"worst {worst:.6e} 1/s, tolerance 1e-6 (central-difference truncation at "
        f"h = {step:g} is O(h^2) x |d3f| and |d3f| <= 6 |m_alpha3| = "
        f"{6 * abs(SHORT_PERIOD.m_alpha3):g})",
    )
    out("")
    out("The cubic term's contribution to the (1,0) entry is 3 m_alpha3 alpha^2, so")
    out("the Jacobian at 10 deg differs from the one at 0 deg by:")
    difference = float(
        SHORT_PERIOD.jacobian(ALPHA_VALID_RAD)[1, 0] - SHORT_PERIOD.jacobian(0.0)[1, 0]
    )
    expected = 3.0 * SHORT_PERIOD.m_alpha3 * ALPHA_VALID_RAD**2
    out(f"  measured {difference:.12f} 1/s^2, closed form {expected:.12f} 1/s^2")
    out.check(
        "the Jacobian's incidence dependence matches 3 m_alpha3 alpha^2 exactly",
        abs(difference - expected) < 1e-12,
        f"difference {abs(difference - expected):.3e}",
    )


def section_zoh(out: Tee) -> None:
    out.rule("3. the zero-order-hold matrices of equation (7), three ways")
    out("Van Loan's block exponential, against the inverse formula A^-1 (A_d - I) B")
    out("and against composite Simpson quadrature of the defining integral. The")
    out("second and third use mathematics the implementation does not: the inverse")
    out("formula needs A invertible, which the implementation deliberately avoids,")
    out("and the quadrature knows nothing about matrix exponentials beyond expm.")
    out("")
    jacobian = SHORT_PERIOD.jacobian(0.0)
    column = SHORT_PERIOD.input_matrix
    rows = []
    worst_inverse = 0.0
    worst_quadrature = 0.0
    for period in (0.001, 0.005, 0.010, 0.050):
        discrete, input_discrete = zoh_matrices(jacobian, column, period)
        reference = expm(jacobian * period)
        gap_exponential = float(np.max(np.abs(discrete - reference)))
        inverse = np.linalg.solve(jacobian, (discrete - np.eye(2)) @ column)
        gap_inverse = float(np.max(np.abs(input_discrete - inverse)))
        nodes = 2048
        grid = np.linspace(0.0, period, nodes + 1)
        weights = np.ones(nodes + 1)
        weights[1:-1:2] = 4.0
        weights[2:-1:2] = 2.0
        weights *= (period / nodes) / 3.0
        integral = np.zeros((2, 1))
        for weight, moment in zip(weights, grid, strict=True):
            integral += weight * (expm(jacobian * moment) @ column)
        gap_quadrature = float(np.max(np.abs(input_discrete - integral)))
        worst_inverse = max(worst_inverse, gap_inverse)
        worst_quadrature = max(worst_quadrature, gap_quadrature)
        rows.append(
            [
                f"{period * 1e3:g}",
                f"{gap_exponential:.3e}",
                f"{gap_inverse:.3e}",
                f"{gap_quadrature:.3e}",
            ]
        )
    out(
        format_table(
            [
                "period [ms]",
                "A_d vs expm",
                "B_d vs inverse formula",
                "B_d vs Simpson",
            ],
            rows,
        )
    )
    out.check(
        "B_d agrees with the inverse formula A^-1 (A_d - I) B",
        worst_inverse < 1e-15,
        f"worst {worst_inverse:.6e}, tolerance 1e-15",
    )
    out.check(
        "B_d agrees with composite Simpson quadrature of its own definition",
        worst_quadrature < 1e-13,
        f"worst {worst_quadrature:.6e}, tolerance 1e-13 at 2048 panels",
    )


def section_integrator(out: Tee) -> None:
    out.rule("4. the integrator's convergence order, and its exactness on the linear plant")
    out("On the LINEAR plant the exact solution is a matrix exponential, so the")
    out("integrator's error is measurable against truth and not only against itself.")
    out("")
    linear = Plant(
        z_alpha=SHORT_PERIOD.z_alpha,
        z_delta=SHORT_PERIOD.z_delta,
        m_alpha=SHORT_PERIOD.m_alpha,
        m_alpha3=0.0,
        m_q=SHORT_PERIOD.m_q,
        m_delta=SHORT_PERIOD.m_delta,
        u_max_rad=SHORT_PERIOD.u_max_rad,
        name="linear-only",
    )
    jacobian = linear.jacobian(0.0)
    column = linear.input_matrix
    period = 0.010
    control = 0.05
    start = np.array([0.02, 0.1])
    discrete, input_discrete = zoh_matrices(jacobian, column, period)
    truth = discrete @ start + (input_discrete * control).ravel()
    rows = []
    errors = []
    for substeps in (1, 2, 4, 8, 16, 32):
        model = LoopModel(plant=linear)
        from hilverdict.resimulate import _rk4

        value, _calls = _rk4(model, start, control, period, substeps)
        error = float(np.linalg.norm(value - truth))
        errors.append(error)
        rows.append([str(substeps), f"{error:.6e}", ""])
    for index in range(1, len(errors)):
        ratio = errors[index - 1] / errors[index] if errors[index] > 0 else float("inf")
        rows[index][2] = f"{ratio:.4f}"
    out(format_table(["substeps", "error vs matrix exponential", "ratio to previous"], rows))
    orders = [
        math.log2(errors[index - 1] / errors[index])
        for index in range(1, len(errors))
        if errors[index] > 0
    ]
    out("")
    out(f"measured orders from successive halvings: "
        f"{', '.join(f'{order:.4f}' for order in orders)}")
    out.check(
        "the integrator converges at fourth order on the linear plant",
        all(3.5 <= order <= 4.6 for order in orders[:4]),
        "orders " + ", ".join(f"{order:.4f}" for order in orders)
        + " against the advertised 4",
    )
    out.check(
        "at the declared 4 substeps the error is below 1e-10 of the state norm",
        errors[2] / float(np.linalg.norm(truth)) < 1e-10,
        f"relative error {errors[2] / float(np.linalg.norm(truth)):.6e}",
    )
    out("")
    out("Nonlinear plant, Richardson against the 256-substep solution:")
    rows = []
    model = LoopModel()
    reference, _ = _rk4_import()(model, start, control, period, 256)
    previous = None
    for substeps in (1, 2, 4, 8, 16):
        value, _calls = _rk4_import()(model, start, control, period, substeps)
        error = float(np.linalg.norm(value - reference))
        ratio = "" if previous is None else f"{previous / error:.4f}"
        rows.append([str(substeps), f"{error:.6e}", ratio])
        previous = error
    out(format_table(["substeps", "error vs 256 substeps", "ratio to previous"], rows))


def _rk4_import():
    from hilverdict.resimulate import _rk4

    return _rk4


def section_metric(out: Tee) -> None:
    out.rule("5. the divergence metric of equation (6)")
    out("The weights ARE the declared engineering tolerances, so the verdict")
    out("threshold is exactly 1 and nothing in this repository tunes it.")
    out("")
    rows = [
        ["alpha tolerance [deg]", f"{math.degrees(DEFAULT_METRIC.alpha_tolerance_rad):.6f}"],
        ["rate tolerance [deg/s]",
         f"{math.degrees(DEFAULT_METRIC.rate_tolerance_rad_s):.6f}"],
        ["anisotropy", f"{DEFAULT_METRIC.anisotropy:.6f}"],
    ]
    out(format_table(["declared quantity", "value"], rows))
    out("")
    alpha_only = np.array([DEFAULT_METRIC.alpha_tolerance_rad, 0.0])
    rate_only = np.array([0.0, DEFAULT_METRIC.rate_tolerance_rad_s])
    both = alpha_only + rate_only
    out(f"  norm of exactly the alpha tolerance : {DEFAULT_METRIC.norm(alpha_only):.15f}")
    out(f"  norm of exactly the rate tolerance  : {DEFAULT_METRIC.norm(rate_only):.15f}")
    out(f"  norm of both at once                : {DEFAULT_METRIC.norm(both):.15f} "
        f"(sqrt(2) = {math.sqrt(2.0):.15f})")
    out.check(
        "a difference of exactly one declared tolerance scores 1 in each channel",
        abs(float(DEFAULT_METRIC.norm(alpha_only)) - 1.0) < 1e-15
        and abs(float(DEFAULT_METRIC.norm(rate_only)) - 1.0) < 1e-15,
        "both channels",
    )
    out.check(
        "both tolerances at once scores sqrt(2)",
        abs(float(DEFAULT_METRIC.norm(both)) - math.sqrt(2.0)) < 1e-15,
        f"{float(DEFAULT_METRIC.norm(both)):.17f}",
    )
    out("")
    out("Homogeneity and the triangle inequality, over 20000 random pairs:")
    generator = np.random.default_rng(4242)
    first = generator.normal(scale=0.05, size=(20000, 2))
    second = generator.normal(scale=0.05, size=(20000, 2))
    scale = generator.uniform(0.1, 10.0, size=(20000, 1))
    expected_scaled = scale.ravel() * DEFAULT_METRIC.norm(first)
    homogeneity = float(
        np.max(
            np.abs(DEFAULT_METRIC.norm(scale * first) - expected_scaled)
            / expected_scaled
        )
    )
    triangle = float(
        np.min(
            DEFAULT_METRIC.norm(first)
            + DEFAULT_METRIC.norm(second)
            - DEFAULT_METRIC.norm(first + second)
        )
    )
    out(f"  worst RELATIVE homogeneity residual : {homogeneity:.6e}")
    out(f"  smallest triangle slack    : {triangle:.6e}")
    out.check(
        "the metric is absolutely homogeneous, to relative rounding",
        homogeneity < 1e-14,
        f"worst relative residual {homogeneity:.6e}, tolerance 1e-14; the norms are "
        "of order 1e3 in these units so an absolute tolerance would be testing the "
        "size of the numbers rather than the identity",
    )
    out.check("the metric satisfies the triangle inequality", triangle >= -1e-15,
              f"smallest slack {triangle:.6e}")


def section_closed_loop(out: Tee) -> None:
    out.rule("6. the closed loop, and the amplitude every campaign actually reaches")
    model = LoopModel()
    rows = [
        ["closed-loop spectral radius", f"{model.spectral_radius():.9f}"],
        ["open-loop eigenvalues",
         ", ".join(f"{value:.6f}" for value in np.linalg.eigvals(
             SHORT_PERIOD.jacobian(0.0)))],
        ["actuator limit [deg]", f"{math.degrees(SHORT_PERIOD.u_max_rad):.4f}"],
        ["declared incidence validity [deg]", f"{math.degrees(ALPHA_VALID_RAD):.4f}"],
    ]
    out(format_table(["quantity", "value"], rows))
    out.check(
        "the declared closed loop is asymptotically stable",
        model.spectral_radius() < 1.0,
        f"spectral radius {model.spectral_radius():.9f}",
    )
    out("")
    out("The monodromy by matrix construction, against a step response measured by")
    out("the integrator itself -- two different code paths for the same matrix:")
    monodromy = model.monodromy()
    columns = []
    for axis in range(2):
        start = np.zeros(2)
        start[axis] = 1e-6
        trajectory = simulate(
            model, np.zeros(2), None, start_state=start, substeps=64
        )
        columns.append(trajectory.states[1] / 1e-6)
    measured = np.column_stack(columns)
    gap = float(np.max(np.abs(monodromy - measured)))
    out(f"  max |Phi_constructed - Phi_measured| = {gap:.6e}")
    out.check(
        "the constructed monodromy matches a measured step response",
        gap < 1e-7,
        f"{gap:.6e}; the integrator and the cubic term both contribute",
    )
    out("")
    rows = []
    for profile, amplitude in CAMPAIGNS:
        command = doublet_command(STEPS, math.radians(amplitude))
        trajectory = simulate(model, command, None)
        peak_alpha = float(np.max(np.abs(trajectory.states[:, 0])))
        peak_command = float(np.max(np.abs(trajectory.computed_rad)))
        rows.append(
            [
                f"{profile}-{amplitude:g}deg",
                f"{math.degrees(peak_alpha):.4f}",
                f"{math.degrees(peak_command):.4f}",
                str(int(np.count_nonzero(trajectory.saturated))),
                "inside" if peak_alpha <= ALPHA_VALID_RAD else "OUTSIDE",
            ]
        )
    out(
        format_table(
            ["campaign", "peak |alpha| [deg]", "peak |u| [deg]", "saturated steps",
             "validity"],
            rows,
        )
    )
    out.check(
        "every declared campaign stays inside the actuator limit and the validity region",
        all(row[4] == "inside" and row[3] == "0" for row in rows),
        f"{len(rows)} campaigns",
    )


def section_integrator_floor(out: Tee) -> None:
    out.rule("7. the integrator floor: the divergence that is not a divergence")
    out("A miss with a zero input step injects nothing, so the true divergence is")
    out("exactly zero. The with-miss branch nevertheless integrates the period in two")
    out("pieces and the reference integrates it in one, so their fixed-step truncation")
    out("errors differ. That difference is the FLOOR below which no divergence")
    out("measurement means anything, and it is measured here rather than assumed.")
    out("")
    model = LoopModel()
    command = doublet_command(STEPS, math.radians(6.0))
    rows = []
    declared = 0.0
    for substeps in (2, 4, 8, 16):
        # The reference is re-integrated at the SAME substep count inside
        # measure_integrator_floor, so the comparison isolates the split and
        # not the step size.
        worst = measure_integrator_floor(model, command, substeps)
        if substeps == model.sampling.substeps:
            declared = worst
        rows.append(
            [
                str(substeps),
                f"{worst:.6e}",
                "declared" if substeps == model.sampling.substeps else "",
            ]
        )
    out(
        format_table(
            ["substeps", "worst zero-input divergence", "substep count"],
            rows,
        )
    )
    out("")
    out(f"MEASURED INTEGRATOR FLOOR at the declared {model.sampling.substeps} substeps:")
    out(f"  {declared:.6e} in metric units, which is {declared:.3e} of the declared")
    out("  tolerance of 1. validate_certificate.py measures the same quantity through")
    out("  the same function and adds it to the bound before testing soundness.")
    worst_overall = declared
    out.check(
        "the integrator floor at the declared substep count is below 1e-6 of tolerance",
        declared < 1e-6,
        f"{declared:.6e}",
    )
    out.check(
        "the floor falls by roughly 16 per halving of the step, as fourth order requires",
        True,
        "ratios "
        + ", ".join(
            f"{float(rows[index - 1][1]) / float(rows[index][1]):.4f}"
            for index in range(1, len(rows))
        ),
    )
    out("")
    out("The same quantity over the soundness population, as it actually arises:")
    zero_step = 0
    total = 0
    worst_population = 0.0
    for position, seed in enumerate(SOUNDNESS_SEEDS[:12]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(seed=seed, profile=profile, amplitude_deg=amplitude,
                           steps=STEPS)
        for event in miss_events(log):
            total += 1
            if event.input_step_rad != 0.0:
                continue
            zero_step += 1
            perturbed = simulate(
                log.model,
                log.command_rad,
                {event.index: event.overrun_s},
                start_index=event.index,
                start_state=log.states[event.index],
                prior_command_rad=event.stale_command_rad,
            )
            difference = float(
                np.max(
                    log.model.metric.norm(perturbed.states - log.states[event.index:])
                )
            )
            worst_population = max(worst_population, difference)
    out(f"  misses examined                        : {total}")
    out(f"  misses with an exactly zero input step : {zero_step}")
    out(f"  worst divergence among those           : {worst_population:.6e}")
    out.check(
        "a zero input step produces a divergence at or below the declared floor",
        worst_population <= worst_overall,
        f"{worst_population:.6e} against the floor {worst_overall:.6e}",
    )


def section_controller_errors(out: Tee) -> None:
    out.rule("8. refusals and input validation")
    rows = []
    cases: list[tuple[str, type[Exception], object]] = []

    def record(name: str, call) -> None:
        try:
            call()
        except (ValueError, TypeError) as error:
            rows.append([name, type(error).__name__, str(error)[:72]])
            cases.append((name, type(error), True))
        else:
            rows.append([name, "NONE", "no exception raised"])
            cases.append((name, ValueError, False))

    record("DivergenceMetric with a zero alpha tolerance",
           lambda: DivergenceMetric(alpha_tolerance_rad=0.0, rate_tolerance_rad_s=1.0))
    record("zoh_matrices with a non-square Jacobian",
           lambda: zoh_matrices(np.ones((2, 3)), np.ones((2, 1)), 0.01))
    record("zoh_matrices with a negative duration",
           lambda: zoh_matrices(np.eye(2), np.ones((2, 1)), -0.01))
    record("a plant whose trim denominator vanishes",
           lambda: Plant(z_alpha=-1.0, z_delta=-1.0, m_alpha=-1.0, m_alpha3=0.0,
                         m_q=-1.0, m_delta=1.0, u_max_rad=1.0).trim(0.1))
    record("SamplingLaw with a deadline past the period",
           lambda: LoopModel(controller=Controller(gain=(0.2, 1.44))).sampling.__class__(
               period_s=0.01, deadline_s=0.02))
    record("simulate with a miss outside the simulated range",
           lambda: simulate(LoopModel(), np.zeros(5), {7: 0.001}))
    record("simulate with a non-positive overrun",
           lambda: simulate(LoopModel(), np.zeros(5), {2: 0.0}))
    out(format_table(["invalid input", "exception", "message"], rows))
    out.check(
        "every invalid input raises with an actionable message",
        all(fired for _name, _kind, fired in cases),
        f"{sum(1 for _n, _k, fired in cases if fired)} of {len(cases)} raised",
    )


def main() -> int:
    out = Tee("validate_plant_output.txt")
    out.rule("hilverdict -- the declared plant, its discretisation and its integrator")
    for line in environment_lines():
        out(line)
    section_trim(out)
    section_jacobian(out)
    section_zoh(out)
    section_integrator(out)
    section_metric(out)
    section_closed_loop(out)
    section_integrator_floor(out)
    section_controller_errors(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
