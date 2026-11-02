"""Baseline 1 on its own terms, and then against the nonlinear re-simulation.

This is the product's central evidence. It establishes, in order:

1. the closed forms of equations (8), (9), (12), (13) and (14) against hand
   calculations and against the inequalities they are derived from;
2. that the certificate is **sound** against the nonlinear re-simulation over a
   population of runs the learned comparison never sees, with the number of
   violations and the worst and median ratio of true divergence to bound
   reported as measured;
3. that it is **useful**: what fraction of misses it disposes of, and that not
   one consequential miss is among them;
4. what drives its looseness, by sweeping the declared metric's anisotropy;
5. what it costs, against what re-simulation costs.

The certificate is a theorem about the declared LINEARISATION. Whether the
real plant's divergence respects it is a question about the linearisation, and
section 2 answers that question by measurement. The integrator floor measured
by ``_bootstrap.measure_integrator_floor`` is added to the bound before the
comparison, because a fixed-step integration of a period split in two does not
agree to the last bit with one integration of the whole period, and three
misses in the first population measured during this build had a bound of
exactly zero and a re-simulated divergence of about 2e-10 for that reason
alone.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3 for why this is set in every entry
# script rather than inside _bootstrap: CPython writes a module's bytecode
# cache BEFORE executing that module, so _bootstrap cannot set it on its own
# behalf. The imports below therefore follow a statement and carry an E402
# waiver.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from _bootstrap import (  # noqa: E402
    SOUNDNESS_SEEDS,
    STEPS,
    TOLERANCE,
    Tee,
    bootstrap,
    environment_lines,
    measure_integrator_floor,
)

ROOT = bootstrap()

from hilverdict.certificate import (  # noqa: E402
    CertificateModel,
    certified_overrun_s,
    certify_run,
    divergence_bound,
    exponential_integral,
    horizon_gains,
    log_norm,
)
from hilverdict.plant import DivergenceMetric, LoopModel  # noqa: E402
from hilverdict.report import format_table  # noqa: E402
from hilverdict.resimulate import resimulate_run  # noqa: E402
from hilverdict.synthetic import CAMPAIGNS, doublet_command, generate_run  # noqa: E402

TAU_BANDS: tuple[tuple[str, float, float], ...] = (
    ("tau < 0.1 T", 0.0, 0.1),
    ("0.1 T <= tau < 0.5 T", 0.1, 0.5),
    ("0.5 T <= tau < T", 0.5, 1.0),
    ("T <= tau < 2 T", 1.0, 2.0),
    ("tau >= 2 T", 2.0, float("inf")),
)


def section_closed_forms(out: Tee) -> None:
    out.rule("1. the closed forms, against hand calculations")
    out("The logarithmic norm of equation (8) on a matrix whose symmetric part is")
    out("diagonal by hand: [[-2, 5], [-5, -4]] has symmetric part diag(-2, -4), so")
    out("mu_2 = -2 exactly.")
    hand = np.array([[-2.0, 5.0], [-5.0, -4.0]])
    measured = log_norm(hand)
    out(f"  measured {measured:.17f}, hand calculation -2")
    out.check(
        "mu_2 of a matrix with a diagonal symmetric part is its largest diagonal entry",
        abs(measured + 2.0) < 1e-15,
        f"|difference| = {abs(measured + 2.0):.3e}",
    )
    out("")
    out("A symmetric matrix is its own symmetric part, so mu_2 is its largest")
    out("eigenvalue: [[1, 2], [2, 1]] has eigenvalues 3 and -1.")
    symmetric = np.array([[1.0, 2.0], [2.0, 1.0]])
    out(f"  measured {log_norm(symmetric):.17f}, hand calculation 3")
    out.check(
        "mu_2 of a symmetric matrix is its largest eigenvalue",
        abs(log_norm(symmetric) - 3.0) < 1e-15,
        f"|difference| = {abs(log_norm(symmetric) - 3.0):.3e}",
    )
    out("")
    out("Equation (9) at mu = 0 is exactly t, and the removable singularity is")
    out("handled rather than divided through:")
    rows = []
    for mu in (0.0, 1e-300, 1e-12, 1.0, -1.0, 98.5777):
        value = exponential_integral(mu, 0.01)
        closed = 0.01 if mu == 0.0 else (math.expm1(mu * 0.01) / mu)
        rows.append([f"{mu:g}", f"{value:.17g}", f"{closed:.17g}"])
    out(format_table(["mu [1/s]", "phi(mu, 0.01)", "closed form"], rows))
    out.check(
        "phi(0, t) is exactly t",
        exponential_integral(0.0, 0.01) == 0.01,
        f"{exponential_integral(0.0, 0.01):.17g}",
    )
    out.check(
        "phi is continuous at mu = 0: phi(1e-300, t) agrees with t to rounding",
        abs(exponential_integral(1e-300, 0.01) - 0.01) < 1e-18,
        f"{exponential_integral(1e-300, 0.01):.17g}",
    )
    out("")
    out("Equation (12) at m = 0 is the identity's 2-norm, which is exactly 1:")
    model = LoopModel()
    gains = horizon_gains(model.scaled_monodromy(), 12)
    out("  " + ", ".join(f"{value:.6f}" for value in gains))
    out.check(
        "gamma_0 is exactly 1",
        gains[0] == 1.0,
        f"{gains[0]:.17g}",
    )
    out.check(
        "gamma_1 is the 2-norm of the scaled monodromy itself",
        abs(gains[1] - float(np.linalg.norm(model.scaled_monodromy(), 2))) < 1e-15,
        f"{gains[1]:.12f}",
    )


def section_dahlquist(out: Tee) -> None:
    out.rule("2. Dahlquist's inequality (8), verified numerically")
    out("|| exp(M t) ||_2 <= exp(mu_2(M) t) is the inequality every step of the")
    out("certificate rests on. It is checked over the declared plant's Jacobian at")
    out("several incidences and over 200 random matrices, on a time grid spanning")
    out("four decades. A violation here would invalidate equations (10) and (11).")
    out("")
    generator = np.random.default_rng(90210)
    matrices = [LoopModel().plant.jacobian(math.radians(d)) for d in (0.0, 5.0, 10.0)]
    matrices += [LoopModel().scaled_jacobian()]
    matrices += [generator.normal(scale=3.0, size=(2, 2)) for _ in range(60)]
    matrices += [generator.normal(scale=3.0, size=(4, 4)) for _ in range(60)]
    grid = np.concatenate([np.logspace(-5, -1, 25), [0.0]])
    violations = 0
    total = 0
    worst_slack = float("inf")
    tightest = 0.0
    for matrix in matrices:
        mu = log_norm(matrix)
        for moment in grid:
            total += 1
            left = float(np.linalg.norm(expm(matrix * moment), 2))
            right = math.exp(mu * moment)
            # Both sides are of order 1; a relative slack of 64 eps covers the
            # matrix exponential's own rounding and nothing more.
            if left > right * (1.0 + 64.0 * np.finfo(float).eps):
                violations += 1
            worst_slack = min(worst_slack, right - left)
            tightest = max(tightest, left / right)
    out(f"  matrices                  : {len(matrices)}")
    out(f"  (matrix, time) pairs      : {total}")
    out(f"  violations                : {violations}")
    out(f"  smallest slack right-left : {worst_slack:.6e}")
    out(f"  tightest ratio left/right : {tightest:.17f}")
    out.check(
        "the logarithmic-norm bound is never violated",
        violations == 0,
        f"{violations} of {total} pairs, slack tolerance 64 eps relative",
    )
    out.check(
        "the bound is an equality at t = 0, so it is not vacuous",
        abs(tightest - 1.0) < 1e-12,
        f"tightest ratio {tightest:.17f}",
    )


def section_monotonicity(out: Tee) -> None:
    out.rule("3. the bound is monotone in tau and in du, and equation (14) inverts it")
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    rows = [
        ["mu_S [1/s]", f"{constants.mu_s:.9f}"],
        ["|| S B ||_2 [1/(s rad)]", f"{constants.input_gain:.9f}"],
        ["c_free, eq (11)", f"{constants.free_factor:.9f}"],
        ["gamma_sup at the full horizon, eq (12)",
         f"{constants.gamma_sup(STEPS):.9f}"],
        ["m at which gamma_m peaks",
         str(int(np.argmax(constants.gains)))],
        ["metric anisotropy", f"{model.metric.anisotropy:.6f}"],
    ]
    out(format_table(["model constant", "value"], rows))
    out("")
    overruns = np.linspace(0.0, 0.05, 501)
    bounds = np.array(
        [divergence_bound(constants, tau, math.radians(1.0), STEPS) for tau in overruns]
    )
    out.check(
        "the bound is strictly increasing in tau over 500 steps to 50 ms",
        bool(np.all(np.diff(bounds) > 0.0)),
        f"smallest increment {float(np.min(np.diff(bounds))):.6e}",
    )
    steps = np.linspace(0.0, math.radians(20.0), 401)
    scaled = np.array(
        [divergence_bound(constants, 0.004, du, STEPS) for du in steps]
    )
    proportional = float(
        np.max(np.abs(scaled[1:] / steps[1:] - scaled[1] / steps[1]))
    )
    out.check(
        "the bound is exactly proportional to du",
        proportional < 1e-12,
        f"largest departure from proportionality {proportional:.6e}",
    )
    out("")
    out("Equation (14) round trip: the bound evaluated AT tau* must equal the")
    out("tolerance, and a hair past tau* must exceed it. Both are checked with a")
    out("relative tolerance; the boolean verdict is NOT asserted at the threshold,")
    out("because a strict comparison on a rounded quantity is the one thing floating")
    out("point cannot promise.")
    out("")
    rows = []
    worst_round_trip = 0.0
    exceeds = 0
    cases = 0
    for degrees in (0.01, 0.05, 0.2, 1.0, 5.0, 20.0):
        for horizon in (5, 40, STEPS):
            step = math.radians(degrees)
            threshold = certified_overrun_s(constants, step, horizon, TOLERANCE)
            at = divergence_bound(constants, threshold, step, horizon)
            past = divergence_bound(constants, threshold * 1.001, step, horizon)
            worst_round_trip = max(worst_round_trip, abs(at / TOLERANCE - 1.0))
            cases += 1
            if past > TOLERANCE:
                exceeds += 1
            if horizon == STEPS:
                rows.append(
                    [
                        f"{degrees:g}",
                        str(horizon),
                        f"{threshold * 1e3:.6f}",
                        f"{at:.15f}",
                        f"{past:.9f}",
                    ]
                )
    out(
        format_table(
            ["du [deg]", "horizon", "tau* [ms]", "bound at tau*", "bound at 1.001 tau*"],
            rows,
        )
    )
    out.check(
        "the bound at tau* equals the tolerance to relative rounding",
        worst_round_trip < 1e-12,
        f"worst relative departure {worst_round_trip:.6e} over {cases} cases",
    )
    out.check(
        "the bound just past tau* exceeds the tolerance in every case",
        exceeds == cases,
        f"{exceeds} of {cases}",
    )


def _population(model: LoopModel, seeds: tuple[int, ...]):
    """Certify and re-simulate every miss in a population; return flat arrays."""
    bounds: list[float] = []
    divergences: list[float] = []
    consequential: list[bool] = []
    overruns: list[float] = []
    refused = 0
    for position, seed in enumerate(seeds):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
            model=model,
        )
        certificate = certify_run(log, TOLERANCE)
        if certificate.refusals:
            refused += 1
            continue
        results = resimulate_run(log, TOLERANCE)
        for item, result in zip(certificate.certificates, results, strict=True):
            bounds.append(item.bound)
            divergences.append(result.peak_divergence)
            consequential.append(result.consequential)
            overruns.append(item.event.overrun_s / model.sampling.period_s)
    return (
        np.array(bounds),
        np.array(divergences),
        np.array(consequential, dtype=bool),
        np.array(overruns),
        refused,
    )


def section_soundness(out: Tee) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out.rule("4. soundness against the nonlinear re-simulation")
    model = LoopModel()
    floor = measure_integrator_floor(model, doublet_command(STEPS, math.radians(6.0)))
    out("integrator floor, measured by the same function validate_plant.py uses:")
    out(f"  {floor:.6e} in metric units, added to every bound before comparison")
    out("")
    bounds, divergences, consequential, overruns, refused = _population(
        model, SOUNDNESS_SEEDS
    )
    allowed = bounds + floor
    violations = int(np.count_nonzero(divergences > allowed))
    ratio = divergences / allowed
    out(f"  runs in the population    : {len(SOUNDNESS_SEEDS)} ({refused} refused)")
    out(f"  misses certified AND re-simulated : {bounds.shape[0]}")
    out(f"  consequential misses      : {int(np.count_nonzero(consequential))}")
    out(f"  VIOLATIONS (divergence above bound + floor) : {violations}")
    out(f"  worst ratio divergence/(bound+floor)        : {float(np.max(ratio)):.6f}")
    out(f"  median ratio                                : {float(np.median(ratio)):.6f}")
    out(f"  10th / 90th percentile ratio                : "
        f"{float(np.percentile(ratio, 10)):.6f} / "
        f"{float(np.percentile(ratio, 90)):.6f}")
    out.check(
        "the certificate is never violated by the nonlinear re-simulation",
        violations == 0,
        f"{violations} of {bounds.shape[0]} misses",
    )
    out.check(
        "the bound is not vacuous: its worst ratio is above 0.5",
        float(np.max(ratio)) > 0.5,
        f"worst ratio {float(np.max(ratio)):.6f}, so the bound is within a factor of "
        f"{1.0 / float(np.max(ratio)):.4f} of the truth at its tightest",
    )
    out("")
    out("By overrun band, so the looseness can be read where it matters:")
    rows = []
    for name, low, high in TAU_BANDS:
        mask = (overruns >= low) & (overruns < high)
        if not np.any(mask):
            rows.append([name, "0", "-", "-", "-", "-"])
            continue
        band = ratio[mask]
        rows.append(
            [
                name,
                str(int(np.count_nonzero(mask))),
                f"{float(np.median(band)):.6f}",
                f"{float(np.max(band)):.6f}",
                f"{int(np.count_nonzero(band > 1.0))}",
                f"{float(np.mean(consequential[mask])):.6f}",
            ]
        )
    out(
        format_table(
            ["band", "misses", "median ratio", "worst ratio", "violations",
             "consequential rate"],
            rows,
        )
    )
    out("")
    out("The inflation factor a user would need for zero violations on this")
    out("population is the worst ratio above; it is BELOW 1, so no inflation is")
    out("needed and the bound holds as derived. That is reported rather than")
    out("claimed: it is a measurement over a finite population of a declared")
    out("generator, and it is not a proof about the nonlinear plant.")
    return bounds, divergences, consequential


def section_usefulness(
    out: Tee, bounds: np.ndarray, divergences: np.ndarray, consequential: np.ndarray
) -> None:
    out.rule("5. usefulness: what the certificate disposes of, and what it must not")
    dismissed = bounds < TOLERANCE
    out(f"  misses                              : {bounds.shape[0]}")
    out(f"  dismissed in closed form            : {int(np.count_nonzero(dismissed))}")
    out(f"  dismissal rate                      : {float(np.mean(dismissed)):.6f}")
    out(f"  left undecided                      : {int(np.count_nonzero(~dismissed))}")
    out(f"  consequential misses                : "
        f"{int(np.count_nonzero(consequential))}")
    out(f"  CONSEQUENTIAL MISSES AMONG THE DISMISSED : "
        f"{int(np.count_nonzero(consequential & dismissed))}")
    out(f"  consequential rate in the undecided set  : "
        f"{float(np.mean(consequential[~dismissed])):.6f}")
    out.check(
        "no consequential miss is dismissed, which is what soundness means in use",
        int(np.count_nonzero(consequential & dismissed)) == 0,
        f"{int(np.count_nonzero(consequential & dismissed))} of "
        f"{int(np.count_nonzero(consequential))} consequential misses",
    )
    out.check(
        "the certificate disposes of the large majority of misses",
        float(np.mean(dismissed)) > 0.5,
        f"dismissal rate {float(np.mean(dismissed)):.6f}",
    )
    out("")
    out("The largest re-simulated divergence among the DISMISSED misses, against the")
    out("tolerance, is the margin the proof actually had in this population:")
    out(f"  largest divergence among dismissed : "
        f"{float(np.max(divergences[dismissed])):.6f}")
    out(f"  tolerance                          : {TOLERANCE:.6f}")


def section_anisotropy(out: Tee) -> None:
    out.rule("6. what drives the looseness: the declared metric's anisotropy")
    out("The logarithmic norm of equation (8) is metric-dependent, and so is the")
    out("non-normality of the closed-loop map in equation (12). A metric that weighs")
    out("incidence two hundred times more heavily than pitch rate makes both worse.")
    out("This is a real cost of declaring a tight incidence tolerance and it is")
    out("measured rather than mentioned.")
    out("")
    rows = []
    seeds = SOUNDNESS_SEEDS[:12]
    for rate_degrees in (0.05, 0.2, 0.5, 1.0, 2.0):
        metric = DivergenceMetric(
            alpha_tolerance_rad=math.radians(0.01),
            rate_tolerance_rad_s=math.radians(rate_degrees),
        )
        model = LoopModel(metric=metric)
        constants = CertificateModel.build(model, STEPS)
        floor = measure_integrator_floor(
            model, doublet_command(STEPS, math.radians(6.0))
        )
        bounds, divergences, consequential, _overruns, _refused = _population(
            model, seeds
        )
        ratio = divergences / (bounds + floor)
        rows.append(
            [
                f"{metric.anisotropy:.1f}",
                f"{constants.mu_s:.4f}",
                f"{constants.gamma_sup(STEPS):.4f}",
                f"{constants.free_factor:.4f}",
                f"{float(np.median(ratio)):.6f}",
                f"{float(np.max(ratio)):.6f}",
                f"{float(np.mean(bounds < TOLERANCE)):.6f}",
                f"{float(np.mean(consequential)):.6f}",
            ]
        )
    out(
        format_table(
            [
                "anisotropy",
                "mu_S [1/s]",
                "gamma_sup",
                "c_free",
                "median ratio",
                "worst ratio",
                "dismissal rate",
                "consequential rate",
            ],
            rows,
        )
    )
    out("")
    medians = [float(row[4]) for row in rows]
    out.check(
        "the bound gets measurably looser as the metric gets more anisotropic",
        medians[0] > medians[-1],
        f"median ratio {medians[0]:.6f} at anisotropy {rows[0][0]} against "
        f"{medians[-1]:.6f} at {rows[-1][0]}",
    )
    out.check(
        "the bound is sound at every anisotropy swept",
        all(float(row[5]) <= 1.0 for row in rows),
        "worst ratios " + ", ".join(row[5] for row in rows),
    )


def section_cost(out: Tee) -> None:
    out.rule("7. what each baseline costs")
    out("Plant evaluations are the unit: one right-hand-side evaluation of equations")
    out("(1)-(2). Wall-clock seconds are reported too and are UNSTABLE on shared")
    out("hardware; the evaluation counts are not.")
    out("")
    model = LoopModel()
    rows = []
    for position, seed in enumerate(SOUNDNESS_SEEDS[:9]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(seed=seed, profile=profile, amplitude_deg=amplitude,
                           steps=STEPS, model=model)
        started = time.perf_counter()
        certificate = certify_run(log, TOLERANCE)
        certificate_s = time.perf_counter() - started
        started = time.perf_counter()
        everything = resimulate_run(log, TOLERANCE)
        full_s = time.perf_counter() - started
        undecided = tuple(c.event.index for c in certificate.undecided)
        started = time.perf_counter()
        residual = resimulate_run(log, TOLERANCE, subset=undecided)
        residual_s = time.perf_counter() - started
        rows.append(
            [
                log.label,
                str(len(certificate.certificates)),
                str(len(undecided)),
                f"{certificate_s * 1e3:.2f}",
                f"{full_s * 1e3:.2f}",
                f"{residual_s * 1e3:.2f}",
                str(sum(result.plant_calls for result in everything)),
                str(sum(result.plant_calls for result in residual)),
            ]
        )
    out(
        format_table(
            [
                "run",
                "misses",
                "undecided",
                "certify [ms]",
                "resim all [ms]",
                "resim residual [ms]",
                "plant calls all",
                "plant calls residual",
            ],
            rows,
        )
    )
    full_calls = sum(int(row[6]) for row in rows)
    residual_calls = sum(int(row[7]) for row in rows)
    out("")
    out(f"  plant evaluations, every miss re-simulated : {full_calls}")
    out(f"  plant evaluations, residual only           : {residual_calls}")
    saving = 1.0 - residual_calls / full_calls if full_calls else float("nan")
    out(f"  evaluations saved by baseline 1             : {saving:.6f}")
    out.check(
        "the certificate saves most of the re-simulation cost",
        saving > 0.5,
        f"{saving:.6f} of plant evaluations over {len(rows)} runs",
    )
    out("")
    out("The certificate's own cost is two matrix exponentials and one 2x2 matrix")
    out("product per sample instant, once per run, plus one logarithm per miss. It")
    out("does not grow with the number of misses in the way re-simulation does.")


def main() -> int:
    out = Tee("validate_certificate_output.txt")
    out.rule("hilverdict -- baseline 1, the analytic certificate")
    for line in environment_lines():
        out(line)
    section_closed_forms(out)
    section_dahlquist(out)
    section_monotonicity(out)
    bounds, divergences, consequential = section_soundness(out)
    section_usefulness(out, bounds, divergences, consequential)
    section_anisotropy(out)
    section_cost(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
