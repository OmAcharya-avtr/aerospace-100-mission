"""The certificate against mathematics it does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and the second catches what the
first cannot. Four independent comparisons are made here, each against a route
the implementation does not take:

1. the horizon factor of equation (12), a discrete monodromy power, against the
   continuous Gronwall factor of equation (16), which uses the sub-additivity
   of the logarithmic norm and no monodromy at all;
2. the hold injection bound of equation (10) against the exact vector integral
   of the same quantity, computed through Van Loan's block exponential, which
   involves no norm inequality;
3. the whole certificate against the **exact** discrete response of the LINEAR
   plant, propagated by matrix multiplication with no norms anywhere;
4. equation (9) against adaptive quadrature from ``scipy.integrate``.

Section 5 then removes one factor at a time from equation (13) and shows that
each truncated bound IS violated on the same data. A bound whose factors could
all be deleted without consequence would not be evidence of anything.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402

from _bootstrap import (  # noqa: E402
    SOUNDNESS_SEEDS,
    STEPS,
    TOLERANCE,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from hilverdict.certificate import (  # noqa: E402
    CertificateModel,
    divergence_bound,
    exponential_integral,
    gronwall_horizon_factor,
    horizon_gains,
)
from hilverdict.plant import SHORT_PERIOD, LoopModel, Plant, zoh_matrices  # noqa: E402
from hilverdict.report import format_table  # noqa: E402
from hilverdict.resimulate import resimulate_run, simulate  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, generate_run  # noqa: E402
from hilverdict.timeline import miss_events  # noqa: E402

LINEAR_PLANT = Plant(
    z_alpha=SHORT_PERIOD.z_alpha,
    z_delta=SHORT_PERIOD.z_delta,
    m_alpha=SHORT_PERIOD.m_alpha,
    m_alpha3=0.0,
    m_q=SHORT_PERIOD.m_q,
    m_delta=SHORT_PERIOD.m_delta,
    u_max_rad=SHORT_PERIOD.u_max_rad,
    name="declared-short-period-linear-only",
)


def section_gronwall(out: Tee) -> None:
    out.rule("1. the horizon factor against a Gronwall factor from different mathematics")
    out("Equation (12) takes powers of the discrete closed-loop map. Equation (16)")
    out("never forms that map: it bounds the closed loop as the open loop plus a")
    out("state-dependent input and uses the sub-additivity of the logarithmic norm.")
    out("The second must dominate the first. It is much looser, and the horizon at")
    out("which it becomes vacuous is reported, because a check that is vacuous")
    out("everywhere is not a check.")
    out("")
    model = LoopModel()
    gains = horizon_gains(model.scaled_monodromy(), STEPS)
    rows = []
    violations = 0
    first_vacuous = None
    for horizon in (0, 1, 2, 3, 5, 8, 12, 20, 40, 100, STEPS):
        measured = float(np.max(gains[: horizon + 1]))
        bound = gronwall_horizon_factor(model, horizon)
        if measured > bound:
            violations += 1
        if bound > 1e6 and first_vacuous is None:
            first_vacuous = horizon
        rows.append(
            [
                str(horizon),
                f"{measured:.6f}",
                f"{bound:.6e}",
                f"{measured / bound:.6e}",
            ]
        )
    out(
        format_table(
            ["horizon N", "gamma_sup, eq (12)", "Gronwall, eq (16)", "ratio"], rows
        )
    )
    out.check(
        "the discrete horizon factor never exceeds the Gronwall factor",
        violations == 0,
        f"{violations} of {len(rows)} horizons",
    )
    out("")
    out(f"The Gronwall factor first exceeds 1e6 at a horizon of {first_vacuous} sample")
    out("instants, which is 'vacuous' by any reading. It is a real check only over")
    out("the first few periods, and that is exactly where the certificate's peak")
    out(f"amplification lives: gamma_m peaks at m = {int(np.argmax(gains))}.")
    peak_m = int(np.argmax(gains))
    out(f"gamma_m peaks at m = {peak_m}, where the Gronwall factor already stands at")
    out(f"{gronwall_horizon_factor(model, peak_m):.6e}. So this check constrains")
    out("equation (12) only over the first period or two and is a weak check beyond")
    out("that. It is reported at that strength rather than as a general verification:")
    out("the sub-additivity route throws away the cancellation that makes the closed")
    out("loop contract, and in a metric with an anisotropy of "
        f"{model.metric.anisotropy:.0f} it throws away a great deal.")
    out.check(
        "the Gronwall check is non-vacuous over the first period",
        gronwall_horizon_factor(model, 1) < 1e3,
        f"Gronwall at N = 1 is {gronwall_horizon_factor(model, 1):.6f} against "
        f"gamma_sup(1) = {float(np.max(gains[:2])):.6f}",
    )


def section_hold_injection(out: Tee) -> None:
    out.rule("2. the hold injection bound against the exact vector integral")
    out("Equation (10) bounds || integral_0^tau exp(A(tau-s)) B du(s) ds ||_S by")
    out("|| S B ||_2 phi(mu_S, tau) du_max. For a CONSTANT input error the integral")
    out("is exact and computable by Van Loan's block exponential, so the inequality")
    out("can be checked rather than believed. Van Loan's method involves no norm")
    out("inequality of any kind.")
    out("")
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    jacobian = model.plant.jacobian(model.operating_alpha_rad)
    column = model.plant.input_matrix
    rows = []
    violations = 0
    tightest = 0.0
    for overrun_ms in (0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 50.0):
        overrun = overrun_ms * 1e-3
        _discrete, integral = zoh_matrices(jacobian, column, overrun)
        exact = float(model.metric.norm(integral.ravel()))
        bound = constants.input_gain * exponential_integral(constants.mu_s, overrun)
        if exact > bound * (1.0 + 1e-12):
            violations += 1
        tightest = max(tightest, exact / bound) if bound > 0 else tightest
        rows.append(
            [
                f"{overrun_ms:g}",
                f"{exact:.9f}",
                f"{bound:.9f}",
                f"{exact / bound:.9f}" if bound > 0 else "-",
            ]
        )
    out(
        format_table(
            ["tau [ms]", "exact || S Int B ||", "bound, eq (10)", "ratio"], rows
        )
    )
    out.check(
        "equation (10) is never violated by the exact integral",
        violations == 0,
        f"{violations} of {len(rows)} overruns",
    )
    out.check(
        "equation (10) is tight as tau goes to zero, so it is not vacuous",
        tightest > 0.9,
        f"tightest ratio {tightest:.9f}",
    )


def section_linear_exact(out: Tee) -> None:
    out.rule("3. the whole certificate against the EXACT linear response")
    out("On the linear plant the divergence a miss causes has a closed form that uses")
    out("no norm and no inequality: the injected error is a vector integral, and it")
    out("propagates by repeated multiplication by the closed-loop matrix. Taking the")
    out("weighted norm of that sequence gives the answer the certificate must bound.")
    out("")
    model = LoopModel(plant=LINEAR_PLANT)
    constants = CertificateModel.build(model, STEPS)
    period = model.sampling.period_s
    jacobian = LINEAR_PLANT.jacobian(0.0)
    column = LINEAR_PLANT.input_matrix
    monodromy = model.monodromy()
    violations = 0
    total = 0
    worst = 0.0
    simulated_gap = 0.0
    rows = []
    for position, seed in enumerate(SOUNDNESS_SEEDS[:12]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
            model=model,
        )
        results = {r.event.index: r for r in resimulate_run(log, TOLERANCE)}
        for event in miss_events(log):
            if event.hold_periods > 0:
                # The closed form below is written for a hold inside one period;
                # a multi-period hold injects a different error in each period
                # and its exact form is the sum, which is not the independent
                # route this section is about. Those are covered by section 1
                # of validate_certificate.py.
                continue
            stale = event.stale_command_rad
            fresh = float(np.asarray(log.computed_rad)[event.index])
            step = stale - fresh
            _discrete, integral = zoh_matrices(
                jacobian, column, event.remainder_s
            )
            # exp(A (T - tau)) applied to the hold integral, then the closed loop.
            tail, _unused = zoh_matrices(jacobian, column, period - event.remainder_s)
            error = tail @ (integral.ravel() * step)
            peak = float(model.metric.norm(error))
            current = error
            for _ in range(event.steps_remaining):
                current = monodromy @ current
                peak = max(peak, float(model.metric.norm(current)))
            bound = divergence_bound(
                constants, event.overrun_s, event.input_step_rad, event.steps_remaining
            )
            total += 1
            if peak > bound * (1.0 + 1e-9):
                violations += 1
            if bound > 0.0:
                worst = max(worst, peak / bound)
            result = results.get(event.index)
            if result is not None:
                simulated_gap = max(
                    simulated_gap, abs(result.peak_divergence - peak)
                )
    rows.append(["misses with a sub-period hold", str(total)])
    rows.append(["violations of eq (13) by the exact linear response", str(violations)])
    rows.append(["worst ratio exact/bound", f"{worst:.9f}"])
    rows.append(
        ["worst |simulated - closed form| on the linear plant", f"{simulated_gap:.6e}"]
    )
    out(format_table(["quantity", "value"], rows))
    out.check(
        "the exact linear response never exceeds the certificate",
        violations == 0,
        f"{violations} of {total} misses, worst ratio {worst:.9f}",
    )
    out.check(
        "the integrator reproduces the closed-form linear response",
        simulated_gap < 1e-6,
        f"worst difference {simulated_gap:.6e} in metric units; the integrator is "
        "fourth-order at 4 substeps and the comparison is against an exact "
        "matrix-exponential propagation",
    )


def section_quadrature(out: Tee) -> None:
    out.rule("4. equation (9) against adaptive quadrature")
    out("phi(mu, t) is the integral of exp(mu s) from 0 to t. scipy.integrate.quad")
    out("evaluates that integral adaptively and knows nothing about expm1.")
    out("")
    rows = []
    worst = 0.0
    for mu in (-98.5777, -1.0, 0.0, 1.0, 18.4885, 98.5777):
        for duration in (1e-4, 1e-3, 1e-2, 5e-2):
            closed = exponential_integral(mu, duration)
            numeric, _error = quad(lambda s, rate=mu: math.exp(rate * s), 0.0, duration)
            relative = abs(closed - numeric) / max(abs(numeric), 1e-300)
            worst = max(worst, relative)
            if duration == 1e-2:
                rows.append(
                    [f"{mu:g}", f"{duration:g}", f"{closed:.15e}", f"{numeric:.15e}",
                     f"{relative:.3e}"]
                )
    out(
        format_table(
            ["mu [1/s]", "t [s]", "phi closed form", "quad", "relative difference"],
            rows,
        )
    )
    out.check(
        "equation (9) agrees with adaptive quadrature",
        worst < 1e-12,
        f"worst relative difference {worst:.6e} over 24 cases, tolerance 1e-12",
    )


def section_necessity(out: Tee) -> None:
    out.rule("5. every factor in equation (13) is necessary on this data")
    out("A bound whose factors could be deleted without consequence would not be")
    out("evidence of anything. Each truncated bound below is compared against the")
    out("same nonlinear re-simulations, and each one IS violated. These are negative")
    out("controls on the derivation itself, anchored on the re-simulated truth.")
    out("")
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    bounds: list[float] = []
    divergences: list[float] = []
    horizons: list[int] = []
    overruns: list[float] = []
    steps: list[float] = []
    for position, seed in enumerate(SOUNDNESS_SEEDS[:18]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
            model=model,
        )
        results = resimulate_run(log, TOLERANCE)
        from hilverdict.certificate import certify_run

        certificate = certify_run(log, TOLERANCE)
        for item, result in zip(certificate.certificates, results, strict=True):
            bounds.append(item.bound)
            divergences.append(result.peak_divergence)
            horizons.append(item.event.steps_remaining)
            overruns.append(item.event.overrun_s)
            steps.append(item.event.input_step_rad)
    bound_array = np.array(bounds)
    divergence_array = np.array(divergences)
    usable = bound_array > 0.0
    variants = {
        "full equation (13)": bound_array,
        "without c_free, eq (11)": bound_array / constants.free_factor,
        "without gamma_sup, eq (12)": np.array(
            [
                divergence_bound(constants, tau, du, 0)
                for tau, du in zip(overruns, steps, strict=True)
            ]
        ),
        "without c_free AND without gamma_sup": bound_array
        / (constants.free_factor * constants.gamma_sup(STEPS)),
    }
    rows = []
    for name, candidate in variants.items():
        mask = usable & (candidate > 0.0)
        ratio = divergence_array[mask] / candidate[mask]
        rows.append(
            [
                name,
                str(int(np.count_nonzero(mask))),
                f"{int(np.count_nonzero(ratio > 1.0))}",
                f"{float(np.max(ratio)):.6f}",
            ]
        )
    out(format_table(["bound variant", "misses", "violations", "worst ratio"], rows))
    out("")
    full_violations = int(rows[0][2])
    out.control(
        "the full bound is not violated",
        fired=full_violations > 0,
        must_fire=False,
        detail=f"{full_violations} violations of {rows[0][1]} misses",
    )
    for row in rows[1:]:
        out.control(
            f"the bound {row[0]} is violated",
            fired=int(row[2]) > 0,
            must_fire=True,
            detail=f"{row[2]} violations, worst ratio {row[3]}",
        )
    out("")
    out("THE ONE FACTOR THIS POPULATION DOES NOT FALSIFY: phi(mu_S, tau) against tau.")
    out("Replacing equation (9) by tau -- which would make the bound invalid in")
    out("general, since mu_S > 0 means exp(mu_S s) > 1 over the whole hold -- is NOT")
    out("violated anywhere in this population. That is reported as measured and is")
    out("NOT offered as a control, because a control that cannot fire is a failed")
    out("control and not a passing product. What it means is that the exponential in")
    out("equation (9) is the certificate's largest single source of conservatism on")
    out("this plant, and section 6 measures how large.")
    substituted = np.array(
        [
            constants.prefactor(horizon) * tau * du
            for tau, du, horizon in zip(overruns, steps, horizons, strict=True)
        ]
    )
    mask = usable & (substituted > 0.0)
    ratio = divergence_array[mask] / substituted[mask]
    out("")
    out(f"  misses compared              : {int(np.count_nonzero(mask))}")
    out(f"  violations of the tau form   : {int(np.count_nonzero(ratio > 1.0))}")
    out(f"  worst ratio divergence/bound : {float(np.max(ratio)):.6f}")
    out.check(
        "the phi-versus-tau substitution is measured, and its result is reported "
        "as found",
        True,
        f"{int(np.count_nonzero(ratio > 1.0))} violations of "
        f"{int(np.count_nonzero(mask))}; not a control, see the paragraph above",
    )


def section_long_holds(out: Tee) -> None:
    out.rule("6. where the conservatism comes from: long holds")
    out("Equation (9) grows like exp(mu_S tau) while the measured divergence of this")
    out("plant grows roughly linearly in tau, because mu_S is large here as an")
    out("artifact of the anisotropic metric and not because the dynamics expand. The")
    out("consequence is a bound that is within a small factor of the truth for a")
    out("sub-period overrun and")
    out("enormously conservative for a hold of several periods. That is measured")
    out("here, against re-simulation, at the three largest input steps in a declared")
    out("run.")
    out("")
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    period = model.sampling.period_s
    log = generate_run(seed=3003, profile="loaded", amplitude_deg=6.0, steps=STEPS,
                       model=model)
    commands = np.asarray(log.computed_rad, dtype=float)
    candidates = sorted(
        ((abs(commands[k] - commands[k - 1]), k) for k in range(1, 150)), reverse=True
    )[:3]
    rows = []
    for _step, index in candidates:
        for multiple in (0.5, 1.5, 3.0, 5.0, 8.0):
            overrun = multiple * period
            hold_periods = int(math.floor(overrun / period + 1e-12))
            stale = float(commands[index - 1])
            largest = max(
                abs(stale - float(commands[min(index + offset, STEPS - 1)]))
                for offset in range(hold_periods + 1)
            )
            perturbed = simulate(
                model,
                log.command_rad,
                {index: overrun},
                start_index=index,
                start_state=log.states[index],
                prior_command_rad=stale,
            )
            divergence = float(
                np.max(model.metric.norm(perturbed.states - log.states[index:]))
            )
            remaining = max(STEPS - (index + hold_periods + 1), 0)
            bound = divergence_bound(constants, overrun, largest, remaining)
            rows.append(
                [
                    str(index),
                    f"{multiple:g}",
                    f"{math.degrees(largest):.4f}",
                    f"{divergence:.4f}",
                    f"{bound:.4f}",
                    f"{divergence / bound:.6f}",
                ]
            )
    out(
        format_table(
            ["iter", "tau/T", "du [deg]", "re-simulated divergence", "bound, eq (13)",
             "ratio"],
            rows,
        )
    )
    out("")
    ratios = [float(row[5]) for row in rows]
    short = [float(row[5]) for row in rows if float(row[1]) < 1.0]
    long_holds = [float(row[5]) for row in rows if float(row[1]) >= 5.0]
    out(f"  tightest ratio at a sub-period hold : {max(short):.6f}")
    out(f"  loosest ratio at a hold of 5 periods or more : {min(long_holds):.6f}")
    out("")
    out("So at a hold of eight periods the bound overstates the divergence by about a")
    out(f"factor of {1.0 / min(ratios):.0f}. A miss that long is not dismissed by the")
    out("certificate and is sent to re-simulation, which is the correct outcome and")
    out("costs one integration; but a user wanting the certificate to settle long")
    out("holds will not get it, and that is a limitation, not a tuning knob.")
    out.check(
        "the bound is far tighter at sub-period holds than at multi-period holds",
        max(short) > 5.0 * min(long_holds),
        f"tightest sub-period ratio {max(short):.6f} against loosest multi-period "
        f"ratio {min(long_holds):.6f}",
    )


def main() -> int:
    out = Tee("validate_independent_bound_output.txt")
    out.rule("hilverdict -- the certificate against mathematics it does not use")
    for line in environment_lines():
        out(line)
    section_gronwall(out)
    section_hold_injection(out)
    section_linear_exact(out)
    section_quadrature(out)
    section_necessity(out)
    section_long_holds(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
