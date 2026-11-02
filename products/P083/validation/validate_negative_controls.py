"""Negative controls: every check in this product, shown capable of failing.

A control that cannot fire is a failed control and not a passing product. Each
control below is anchored on the TRUTH -- the re-simulated divergence, the
injected clock behaviour, the declared refusal condition, or a shuffled label
set -- and never on an output of the thing being tested. Both directions are
asserted: a control declared must-fire that does not fire is a failure, and so
is one declared must-not-fire that does.

The anchoring matters. Injecting a violation by nudging a bound is not a
control, because a bound with margin absorbs a small nudge; one product of
this mission shipped two controls that were silently too weak for exactly that
reason. Every must-fire control here is built by corrupting an INPUT the
derivation depends on, by a factor large enough that the derivation cannot
survive it, and the corrupted result is then compared against the same
re-simulated truth the honest result is compared against.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    ALLOWANCE_PPM,
    SOUNDNESS_SEEDS,
    STEPS,
    TARGET_RECALL,
    TOLERANCE,
    Tee,
    bootstrap,
    environment_lines,
    measure_integrator_floor,
)

ROOT = bootstrap()

from hilverdict.certificate import (  # noqa: E402
    CertificateModel,
    certify_run,
    divergence_bound,
    require_admissible,
)
from hilverdict.clock import (  # noqa: E402
    CLOCK_CLEAN_VERDICT,
    CLOCK_DRIFTING,
    estimate_clock,
)
from hilverdict.features import FEATURE_NAMES, feature_matrix  # noqa: E402
from hilverdict.learned import DEFAULT_SPEC, MissClassifier  # noqa: E402
from hilverdict.plant import SHORT_PERIOD, DivergenceMetric, LoopModel, Plant  # noqa: E402
from hilverdict.report import format_table  # noqa: E402
from hilverdict.resimulate import resimulate_run, simulate  # noqa: E402
from hilverdict.scoring import score_decision, threshold_at_recall  # noqa: E402
from hilverdict.synthetic import (  # noqa: E402
    CAMPAIGNS,
    ClockSpec,
    doublet_command,
    generate_run,
)
from hilverdict.timeline import RunLog, miss_events, utilisation  # noqa: E402
from hilverdict.verdict import TIMING_FAITHFUL, UNDECIDABLE, run_verdict  # noqa: E402


def _population(model: LoopModel, count: int = 18):
    bounds: list[float] = []
    divergences: list[float] = []
    overruns: list[float] = []
    steps: list[float] = []
    horizons: list[int] = []
    for position, seed in enumerate(SOUNDNESS_SEEDS[:count]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(
            seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
            model=model,
        )
        certificate = certify_run(log, TOLERANCE)
        results = resimulate_run(log, TOLERANCE)
        for item, result in zip(certificate.certificates, results, strict=True):
            bounds.append(item.bound)
            divergences.append(result.peak_divergence)
            overruns.append(item.event.overrun_s)
            steps.append(item.event.input_step_rad)
            horizons.append(item.event.steps_remaining)
    return (
        np.array(bounds),
        np.array(divergences),
        np.array(overruns),
        np.array(steps),
        np.array(horizons),
    )


def section_bound_inputs(out: Tee) -> None:
    out.rule("1. the certificate, with each of its inputs corrupted")
    out("Every row is the SAME re-simulated divergences compared against a bound")
    out("computed from a corrupted input. The honest bound must not be violated; each")
    out("corrupted bound must be, or the input it corrupts was not load-bearing.")
    out("")
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    floor = measure_integrator_floor(model, doublet_command(STEPS, math.radians(6.0)))
    bounds, divergences, overruns, steps, horizons = _population(model)
    variants: list[tuple[str, np.ndarray, bool]] = [
        ("the honest bound", bounds, False),
        (
            "the input step understated by a factor of 10",
            np.array(
                [
                    divergence_bound(constants, tau, du / 10.0, horizon)
                    for tau, du, horizon in zip(overruns, steps, horizons, strict=True)
                ]
            ),
            True,
        ),
        (
            "the overrun understated by a factor of 10",
            np.array(
                [
                    divergence_bound(constants, tau / 10.0, du, horizon)
                    for tau, du, horizon in zip(overruns, steps, horizons, strict=True)
                ]
            ),
            True,
        ),
        (
            "the input gain || S B || understated by a factor of 4",
            bounds / 4.0,
            True,
        ),
    ]
    rows = []
    for name, candidate, _must in variants:
        usable = candidate > 0.0
        ratio = divergences[usable] / (candidate[usable] + floor)
        rows.append(
            [
                name,
                str(int(np.count_nonzero(usable))),
                str(int(np.count_nonzero(ratio > 1.0))),
                f"{float(np.max(ratio)):.6f}",
            ]
        )
    out(format_table(["bound variant", "misses", "violations", "worst ratio"], rows))
    out("")
    for (name, _candidate, must_fire), row in zip(variants, rows, strict=True):
        out.control(
            f"the bound with {name} is violated",
            fired=int(row[2]) > 0,
            must_fire=must_fire,
            detail=f"{row[2]} violations of {row[1]} misses, worst ratio {row[3]}",
        )
    out("")
    out("A corrupted LINEARISATION, rather than a corrupted scalar: the certificate")
    out("built at an operating point 10 deg away from the one the run sits at.")
    away = LoopModel(operating_alpha_rad=math.radians(10.0))
    away_constants = CertificateModel.build(away, STEPS)
    away_bounds = np.array(
        [
            divergence_bound(away_constants, tau, du, horizon)
            for tau, du, horizon in zip(overruns, steps, horizons, strict=True)
        ]
    )
    usable = away_bounds > 0.0
    ratio = divergences[usable] / (away_bounds[usable] + floor)
    out(f"  violations : {int(np.count_nonzero(ratio > 1.0))} of "
        f"{int(np.count_nonzero(usable))}")
    out(f"  worst ratio: {float(np.max(ratio)):.6f}")
    out(f"  mu_S moves from {constants.mu_s:.6f} to {away_constants.mu_s:.6f} 1/s and")
    out(f"  gamma_sup from {constants.gamma_sup(STEPS):.6f} to "
        f"{away_constants.gamma_sup(STEPS):.6f}")
    out("")
    out("Reported as measured, not as a control: moving the operating point to the")
    out("destabilising end of the declared cubic makes the linearisation LESS stable")
    out("and the bound LARGER, so it is not violated. A mis-declared operating point")
    out("is therefore dangerous in the other direction -- declaring a high-incidence")
    out("run at the origin -- and that is the direction the refusal in")
    out("hilverdict.certificate guards.")
    out.check(
        "the operating-point sensitivity of the bound is measured and reported",
        True,
        f"worst ratio {float(np.max(ratio)):.6f} at a 10 deg operating-point error",
    )


def section_refusals(out: Tee) -> None:
    out.rule("2. the three declared refusals, each shown to fire and not to over-fire")
    model = LoopModel()
    clean = generate_run(seed=SOUNDNESS_SEEDS[0], profile="loaded", amplitude_deg=6.0,
                         steps=STEPS, model=model)
    clean_certificate = certify_run(clean, TOLERANCE)
    out.control(
        "an admissible run is refused",
        fired=bool(clean_certificate.refusals),
        must_fire=False,
        detail=f"refusals {list(clean_certificate.refusals)}",
    )

    saturating = generate_run(seed=SOUNDNESS_SEEDS[0], profile="loaded",
                              amplitude_deg=14.0, steps=STEPS, model=model)
    saturating_certificate = certify_run(saturating, TOLERANCE)
    out(f"  a 14 deg manoeuvre saturates {saturating.saturated_steps} of "
        f"{saturating.steps} iterations")
    out.control(
        "a run that saturates the actuator is refused",
        fired=any("saturat" in reason for reason in saturating_certificate.refusals),
        must_fire=True,
        detail=f"{len(saturating_certificate.refusals)} refusal(s)",
    )

    # A rig with more actuator authority, on the LINEAR-ONLY variant of the
    # declared plant, reaches an incidence the declared cubic term was never
    # fitted for. The linear variant is used deliberately: on the cubic plant the
    # incidence response saturates just below the validity limit -- 42 deg of
    # command gives 10.0059 deg of incidence and 60 deg gives only 8.5393 -- so a
    # control built on it would clear the limit by six thousandths of a degree,
    # which is not a control. On the linear variant the response scales, so
    # 40 deg of command gives 14 deg of incidence and the refusal is exercised
    # with a wide margin.
    strong = Plant(
        z_alpha=SHORT_PERIOD.z_alpha,
        z_delta=SHORT_PERIOD.z_delta,
        m_alpha=SHORT_PERIOD.m_alpha,
        m_alpha3=0.0,
        m_q=SHORT_PERIOD.m_q,
        m_delta=SHORT_PERIOD.m_delta,
        u_max_rad=math.radians(400.0),
        name="declared-short-period-linear-high-authority",
    )
    strong_model = LoopModel(plant=strong)
    big = generate_run(seed=SOUNDNESS_SEEDS[0], profile="loaded", amplitude_deg=40.0,
                       steps=STEPS, model=strong_model)
    big_certificate = certify_run(big, TOLERANCE)
    out(f"  a 40 deg manoeuvre on the linear high-authority variant reaches "
        f"{math.degrees(big.peak_alpha_rad):.4f} deg of incidence")
    out.control(
        "a run beyond the declared incidence validity limit is refused",
        fired=any("validity" in reason for reason in big_certificate.refusals),
        must_fire=True,
        detail=f"{len(big_certificate.refusals)} refusal(s)",
    )

    overloaded = RunLog(
        model=clean.model,
        command_rad=clean.command_rad,
        durations_s=np.asarray(clean.durations_s, dtype=float) * 1.6,
        states=clean.states,
        computed_rad=clean.computed_rad,
        saturated=clean.saturated,
        host_time_s=clean.host_time_s,
        device_time_s=clean.device_time_s,
        label="overloaded",
    )
    overloaded_certificate = certify_run(overloaded, TOLERANCE)
    out(f"  scaling the durations by 1.6 gives a mean utilisation of "
        f"{utilisation(overloaded):.6f}")
    out.control(
        "a run whose mean utilisation reaches 1 is refused",
        fired=any("utilisation" in reason for reason in overloaded_certificate.refusals),
        must_fire=True,
        detail=f"utilisation {utilisation(overloaded):.6f}",
    )
    out("")
    raised = False
    try:
        require_admissible(saturating_certificate)
    except ValueError:
        raised = True
    out.control(
        "require_admissible raises on a refused run",
        fired=raised,
        must_fire=True,
        detail="ValueError from require_admissible",
    )
    raised = False
    try:
        require_admissible(clean_certificate)
    except ValueError:
        raised = True
    out.control(
        "require_admissible raises on an admissible run",
        fired=raised,
        must_fire=False,
        detail="no exception expected",
    )
    out("")
    rows = [
        ["admissible run", clean.label, str(len(clean_certificate.refusals)),
         run_verdict(clean, TOLERANCE).verdict],
        ["saturating run", saturating.label,
         str(len(saturating_certificate.refusals)),
         run_verdict(saturating, TOLERANCE).verdict],
        ["beyond validity", big.label, str(len(big_certificate.refusals)),
         run_verdict(big, TOLERANCE).verdict],
        ["utilisation >= 1", overloaded.label,
         str(len(overloaded_certificate.refusals)),
         run_verdict(overloaded, TOLERANCE).verdict],
    ]
    out(format_table(["case", "run", "refusals", "run verdict"], rows))
    out.check(
        "a refused run is reported UNDECIDABLE and never as timing-faithful",
        all(row[3] == UNDECIDABLE for row in rows[1:])
        and rows[0][3] != UNDECIDABLE,
        f"verdicts {[row[3] for row in rows]}",
    )


def section_clock_controls(out: Tee) -> None:
    out.rule("3. the clock checks, against injected clock behaviour")
    base = dict(jitter_s=2.0e-7, tick_s=1.0e-7, name="control")
    clean_spec = ClockSpec(drift_ppm=0.0, reset_fractions=(), **base)
    small_spec = ClockSpec(drift_ppm=ALLOWANCE_PPM * 0.5, reset_fractions=(), **base)
    big_spec = ClockSpec(drift_ppm=ALLOWANCE_PPM * 20.0, reset_fractions=(), **base)
    reset_spec = ClockSpec(drift_ppm=0.0, reset_fractions=(0.45,), **base)
    rows = []
    estimates = {}
    for name, spec in (
        ("no drift, no reset", clean_spec),
        ("drift inside the allowance", small_spec),
        ("drift far beyond the allowance", big_spec),
        ("a reset at 45 per cent of the run", reset_spec),
    ):
        log = generate_run(seed=SOUNDNESS_SEEDS[1], profile="loaded",
                           amplitude_deg=6.0, steps=STEPS, clock=spec)
        estimate = estimate_clock(log.host_time_s, log.device_time_s, ALLOWANCE_PPM)
        estimates[name] = (estimate, log)
        rows.append(
            [
                name,
                f"{spec.drift_ppm:g}",
                f"{estimate.drift_ppm:.6f}",
                str(list(estimate.resets)),
                estimate.verdict,
                run_verdict(log, TOLERANCE).verdict,
            ]
        )
    out(
        format_table(
            ["injected", "drift [ppm]", "estimated [ppm]", "resets", "clock verdict",
             "run verdict"],
            rows,
        )
    )
    out("")
    out.control(
        "a clean clock is reported as drifting",
        fired=estimates["no drift, no reset"][0].verdict != CLOCK_CLEAN_VERDICT,
        must_fire=False,
        detail=estimates["no drift, no reset"][0].verdict,
    )
    out.control(
        "a drift inside the declared allowance is reported as drifting",
        fired=estimates["drift inside the allowance"][0].verdict != CLOCK_CLEAN_VERDICT,
        must_fire=False,
        detail=estimates["drift inside the allowance"][0].verdict,
    )
    out.control(
        "a drift far beyond the allowance is reported as drifting",
        fired=estimates["drift far beyond the allowance"][0].verdict == CLOCK_DRIFTING,
        must_fire=True,
        detail=estimates["drift far beyond the allowance"][0].verdict,
    )
    out.control(
        "an injected reset makes the run undecidable",
        fired=run_verdict(estimates["a reset at 45 per cent of the run"][1], TOLERANCE)
        .verdict
        == UNDECIDABLE,
        must_fire=True,
        detail="run verdict "
        + run_verdict(
            estimates["a reset at 45 per cent of the run"][1], TOLERANCE
        ).verdict,
    )


def section_resimulation_controls(out: Tee) -> None:
    out.rule("4. the re-simulation reference, against cases with a known answer")
    model = LoopModel()
    log = generate_run(seed=SOUNDNESS_SEEDS[2], profile="loaded", amplitude_deg=6.0,
                       steps=STEPS, model=model)
    floor = measure_integrator_floor(model, log.command_rad)
    empty = simulate(model, log.command_rad, None)
    difference = float(np.max(model.metric.norm(empty.states - log.states)))
    out(f"  re-simulating with an EMPTY miss set reproduces the reference to "
        f"{difference:.3e}")
    out.control(
        "an empty miss set produces a non-zero divergence",
        fired=difference > 0.0,
        must_fire=False,
        detail=f"largest divergence {difference:.3e}",
    )
    out("")
    zero_step = [
        event
        for event in miss_events(log)
        if event.input_step_rad == 0.0
    ]
    worst = 0.0
    for event in zero_step:
        perturbed = simulate(
            model,
            log.command_rad,
            {event.index: event.overrun_s},
            start_index=event.index,
            start_state=log.states[event.index],
            prior_command_rad=event.stale_command_rad,
        )
        worst = max(
            worst,
            float(
                np.max(model.metric.norm(perturbed.states - log.states[event.index:]))
            ),
        )
    out(f"  misses with an exactly zero input step : {len(zero_step)}")
    out(f"  worst divergence among them            : {worst:.3e}")
    out(f"  measured integrator floor              : {floor:.3e}")
    out.control(
        "a zero input step produces a divergence above the integrator floor",
        fired=worst > floor,
        must_fire=False,
        detail=f"{worst:.3e} against a floor of {floor:.3e}",
    )
    out("")
    clean_timing = RunLog(
        model=log.model,
        command_rad=log.command_rad,
        durations_s=np.full(log.steps, 0.4 * log.model.sampling.period_s),
        states=log.states,
        computed_rad=log.computed_rad,
        saturated=log.saturated,
        host_time_s=log.host_time_s,
        device_time_s=log.device_time_s,
        label="no-overrun",
    )
    verdict = run_verdict(clean_timing, TOLERANCE)
    out(f"  a run with no overrun at all : {verdict.verdict}, "
        f"{verdict.misses} misses")
    out.control(
        "a run with no overrun is reported as anything but timing-faithful",
        fired=verdict.verdict != TIMING_FAITHFUL,
        must_fire=False,
        detail=f"verdict {verdict.verdict}",
    )


def section_learned_controls(out: Tee) -> None:
    out.rule("5. the learned pipeline, against shuffled labels and blinded features")
    out("A model whose labels have been shuffled must lose its skill, and a model")
    out("blinded to the two features the certificate uses must get measurably worse.")
    out("Both are scored at the SAME flag rate so the comparison is of recall and")
    out("not of operating point.")
    out("")
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for position, seed in enumerate(SOUNDNESS_SEEDS[:27]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        log = generate_run(seed=seed, profile=profile, amplitude_deg=amplitude,
                           steps=STEPS)
        matrix, events = feature_matrix(log)
        if not events:
            continue
        results = resimulate_run(log, TOLERANCE)
        labels = np.array([r.consequential for r in results], dtype=bool)
        if position < 18:
            train_features.append(matrix)
            train_labels.append(labels)
        else:
            test_features.append(matrix)
            test_labels.append(labels)
    train_matrix = np.vstack(train_features)
    train_label = np.concatenate(train_labels)
    test_matrix = np.vstack(test_features)
    test_label = np.concatenate(test_labels)

    honest = MissClassifier(spec=DEFAULT_SPEC).fit(train_matrix, train_label)
    generator = np.random.default_rng(515151)
    shuffled_labels = train_label.copy()
    generator.shuffle(shuffled_labels)
    shuffled = MissClassifier(spec=DEFAULT_SPEC).fit(train_matrix, shuffled_labels)
    blind_columns = [
        index
        for index, name in enumerate(FEATURE_NAMES)
        if name in {"overrun_s", "overrun_periods", "input_step_rad", "hold_periods"}
    ]
    blind_train = train_matrix.copy()
    blind_test = test_matrix.copy()
    blind_train[:, blind_columns] = 0.0
    blind_test[:, blind_columns] = 0.0
    blinded = MissClassifier(spec=DEFAULT_SPEC).fit(blind_train, train_label)

    out("The statistic is the SMALLEST flag rate at which a model still catches")
    out("every consequential miss on the held-out runs. A model that has lost")
    out("information has to flag more to catch the same set. Comparing at a fixed")
    out("flag rate would not work: at a flag rate of 0.25 all three models reach a")
    out("recall of 1 on this population and the comparison is saturated, which is")
    out("how an earlier version of this control came to be too weak to fire.")
    out("")
    rate = 0.25
    rows = []
    required = {}
    for name, model, features in (
        ("honest model", honest, test_matrix),
        ("labels shuffled in training", shuffled, test_matrix),
        ("blinded to tau and du", blinded, blind_test),
    ):
        probability = model.predict_proba(features)
        saturated = float(np.quantile(probability, 1.0 - rate))
        score = score_decision(name, probability >= saturated, test_label)
        _threshold, _recall, needed = threshold_at_recall(
            probability, test_label, 1.0
        )
        required[name] = needed
        rows.append(
            [
                name,
                str(score.counts.total),
                str(score.counts.positives),
                f"{score.counts.recall:.6f}",
                f"{needed:.6f}",
            ]
        )
    out(
        format_table(
            ["model", "misses", "consequential", "recall at a 0.25 flag rate",
             "flag rate needed for recall 1"],
            rows,
        )
    )
    out("")
    out.control(
        "shuffling the training labels makes the model need a far larger flag rate",
        fired=required["labels shuffled in training"]
        > required["honest model"] + 0.2,
        must_fire=True,
        detail=f"{required['labels shuffled in training']:.6f} against the honest "
        f"model's {required['honest model']:.6f}",
    )
    out.control(
        "blinding the model to tau and du makes it need a larger flag rate",
        fired=required["blinded to tau and du"] > required["honest model"] * 1.05,
        must_fire=True,
        detail=f"{required['blinded to tau and du']:.6f} against the honest model's "
        f"{required['honest model']:.6f}",
    )
    out("")
    raised = False
    try:
        MissClassifier(spec=DEFAULT_SPEC).fit(
            train_matrix, np.zeros(train_matrix.shape[0], dtype=bool)
        )
    except ValueError:
        raised = True
    out.control(
        "fitting on a single-class label set raises",
        fired=raised,
        must_fire=True,
        detail="ValueError from MissClassifier.fit",
    )
    raised = False
    try:
        threshold_at_recall(np.zeros(5), np.zeros(5, dtype=bool), 1.5)
    except ValueError:
        raised = True
    out.control(
        "asking for a recall above 1 raises",
        fired=raised,
        must_fire=True,
        detail=f"target_recall {1.5} is outside (0, 1]",
    )
    _ = TARGET_RECALL


def section_metric_controls(out: Tee) -> None:
    out.rule("6. the metric, mis-declared")
    out("The bound is a statement in the units of the declared metric. A bound")
    out("computed in one metric and compared against a divergence measured in a")
    out("tighter one is wrong, and must be caught by the same comparison.")
    out("")
    truth_model = LoopModel()
    floor = measure_integrator_floor(
        truth_model, doublet_command(STEPS, math.radians(6.0))
    )
    loose = LoopModel(
        metric=DivergenceMetric(
            alpha_tolerance_rad=math.radians(0.10),
            rate_tolerance_rad_s=math.radians(2.0),
        )
    )
    loose_constants = CertificateModel.build(loose, STEPS)
    bounds, divergences, overruns, steps, horizons = _population(truth_model, 6)
    mismatched = np.array(
        [
            divergence_bound(loose_constants, tau, du, horizon)
            for tau, du, horizon in zip(overruns, steps, horizons, strict=True)
        ]
    )
    usable = mismatched > 0.0
    ratio = divergences[usable] / (mismatched[usable] + floor)
    honest_ratio = divergences[bounds > 0.0] / (bounds[bounds > 0.0] + floor)
    out(f"  honest bound     : {int(np.count_nonzero(honest_ratio > 1.0))} violations, "
        f"worst ratio {float(np.max(honest_ratio)):.6f}")
    out(f"  mis-declared (0.10 deg instead of 0.01 deg) : "
        f"{int(np.count_nonzero(ratio > 1.0))} violations, worst ratio "
        f"{float(np.max(ratio)):.6f}")
    out.control(
        "a bound computed in a 10x looser metric is violated by the tighter metric's "
        "divergences",
        fired=int(np.count_nonzero(ratio > 1.0)) > 0,
        must_fire=True,
        detail=f"{int(np.count_nonzero(ratio > 1.0))} violations, worst ratio "
        f"{float(np.max(ratio)):.6f}",
    )
    out.control(
        "the honest bound in its own metric is violated",
        fired=int(np.count_nonzero(honest_ratio > 1.0)) > 0,
        must_fire=False,
        detail=f"{int(np.count_nonzero(honest_ratio > 1.0))} violations",
    )


def main() -> int:
    out = Tee("validate_negative_controls_output.txt")
    out.rule("hilverdict -- negative controls, each asserted to fire or not to fire")
    for line in environment_lines():
        out(line)
    section_bound_inputs(out)
    section_refusals(out)
    section_clock_controls(out)
    section_resimulation_controls(out)
    section_learned_controls(out)
    section_metric_controls(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
