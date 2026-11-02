"""Baseline 2: that the reference is a reference, and what it costs.

Baseline 2 is the thing every other number in this product is scored against,
so the questions asked of it here are not about accuracy against some further
truth -- there is none -- but about whether it is self-consistent, whether it
is reproducible, whether the shortcut that makes it affordable is lossless,
and how much it costs.

Five things are established:

1. **Determinism.** The same inputs give bit-identical states, and a different
   seed gives a different trajectory, so the first claim is not vacuous.
2. **Prefix reuse is lossless.** The per-miss study re-simulates only the tail
   of a run, starting from the reference state at the miss. That it agrees with
   re-simulating the whole run from iteration zero is a correctness condition
   and is measured, not assumed.
3. **Step-size convergence of the VERDICT.** The peak divergence is a function
   of the integrator's step; the verdict is a threshold comparison on it, so
   the number of verdict flips between step sizes is what matters and is
   counted.
4. **The optional early exit is lossless on this population.** Its criterion
   rests on the same linearisation as baseline 1, so it is off by default and
   its agreement with the full-horizon reference is measured.
5. **Superposition.** The per-miss analysis is one miss at a time. Re-simulating
   a run with ALL its misses at once, and comparing against the aggregate bound
   of equation (15) and against the sum of the individual divergences, is the
   only statement this product makes about interactions.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    SOUNDNESS_SEEDS,
    STEPS,
    TOLERANCE,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from hilverdict.certificate import CertificateModel, certify_run  # noqa: E402
from hilverdict.plant import LoopModel  # noqa: E402
from hilverdict.report import format_table  # noqa: E402
from hilverdict.resimulate import (  # noqa: E402
    peak_divergence,
    resimulate_run,
    simulate,
)
from hilverdict.synthetic import CAMPAIGNS, generate_run  # noqa: E402
from hilverdict.timeline import miss_events  # noqa: E402


def _logs(count: int) -> list:
    logs = []
    for position, seed in enumerate(SOUNDNESS_SEEDS[:count]):
        profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
        logs.append(
            generate_run(
                seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS
            )
        )
    return logs


def section_determinism(out: Tee) -> None:
    out.rule("1. determinism, and a control that it is not vacuous")
    model = LoopModel()
    log = _logs(1)[0]
    first = simulate(model, log.command_rad, {40: 0.013})
    second = simulate(model, log.command_rad, {40: 0.013})
    identical = bool(np.array_equal(first.states, second.states))
    out(f"  two identical calls, states bit-identical : {identical}")
    out.check(
        "the simulator is bit-deterministic for identical inputs",
        identical,
        "np.array_equal on the full state array",
    )
    profile, amplitude = CAMPAIGNS[0]
    other = generate_run(
        seed=999001, profile=profile, amplitude_deg=amplitude, steps=STEPS
    )
    difference = float(np.max(np.abs(other.states - log.states)))
    out.control(
        "a different seed gives a different timing trace",
        fired=float(np.max(np.abs(other.durations_s - log.durations_s))) > 0.0,
        must_fire=True,
        detail=f"largest duration difference "
        f"{float(np.max(np.abs(other.durations_s - log.durations_s))):.6e} s",
    )
    out("")
    out("The reference TRAJECTORY does not depend on the seed, only the timing trace")
    out("does: the manoeuvre is a declared command schedule and the plant is")
    out(f"deterministic. Largest state difference between two seeds: {difference:.3e}.")
    out.check(
        "two runs of the same campaign share their reference trajectory exactly",
        difference == 0.0,
        f"{difference:.3e}; the seed drives the execution times only",
    )


def section_prefix(out: Tee) -> None:
    out.rule("2. prefix reuse is lossless")
    out("The per-miss study starts the with-miss branch at the miss, from the")
    out("reference state there. The affordable thing and the obviously-correct thing")
    out("must agree.")
    out("")
    model = LoopModel()
    worst = 0.0
    compared = 0
    for log in _logs(6):
        for event in miss_events(log):
            whole = simulate(model, log.command_rad, {event.index: event.overrun_s})
            tail = simulate(
                model,
                log.command_rad,
                {event.index: event.overrun_s},
                start_index=event.index,
                start_state=log.states[event.index],
                prior_command_rad=event.stale_command_rad,
            )
            gap = float(np.max(np.abs(whole.states[event.index:] - tail.states)))
            worst = max(worst, gap)
            compared += 1
    out(f"  misses compared                    : {compared}")
    out(f"  worst |whole-run - tail| state gap : {worst:.6e}")
    out.check(
        "re-simulating only the tail is bit-identical to re-simulating the whole run",
        worst == 0.0,
        f"worst gap {worst:.6e}; the prefix is the reference trajectory and the "
        "integrator is deterministic, so the agreement is exact and not merely "
        "within tolerance",
    )


def section_substeps(out: Tee) -> None:
    out.rule("3. step-size convergence of the peak divergence, and of the VERDICT")
    out("The verdict is a threshold comparison on a number the integrator produces,")
    out("so the question is not how the number moves but whether the comparison")
    out("flips. Flips are counted, and any miss whose peak sits within the measured")
    out("step-size spread of the tolerance is named.")
    out("")
    model = LoopModel()
    logs = _logs(3)
    reference_peaks: dict[tuple[int, int], float] = {}
    rows = []
    for substeps in (2, 4, 8, 16):
        peaks = {}
        for position, log in enumerate(logs):
            for event in miss_events(log):
                value, _where, _calls, _steps = peak_divergence(
                    model,
                    log.command_rad,
                    log.states,
                    event,
                    substeps=substeps,
                )
                peaks[(position, event.index)] = value
        if substeps == 16:
            reference_peaks = peaks
        rows.append([str(substeps), peaks])
    table = []
    flips_total = 0
    for substeps, peaks in rows:
        if not reference_peaks:
            continue
        shifts = np.array(
            [abs(peaks[key] - reference_peaks[key]) for key in peaks]
        )
        relative = np.array(
            [
                abs(peaks[key] - reference_peaks[key]) / max(reference_peaks[key], 1e-12)
                for key in peaks
            ]
        )
        flips = int(
            np.count_nonzero(
                [
                    (peaks[key] >= TOLERANCE) != (reference_peaks[key] >= TOLERANCE)
                    for key in peaks
                ]
            )
        )
        flips_total += flips
        table.append(
            [
                substeps,
                str(len(peaks)),
                f"{float(np.max(shifts)):.6e}",
                f"{float(np.max(relative)):.6e}",
                str(flips),
            ]
        )
    out(
        format_table(
            ["substeps", "misses", "worst absolute shift", "worst relative shift",
             "verdict flips vs 16 substeps"],
            table,
        )
    )
    out.check(
        "no verdict flips between the declared 4 substeps and 16",
        all(int(row[4]) == 0 for row in table),
        f"{flips_total} flips in total across the step sizes compared",
    )
    out("")
    out("Per miss, the step-size shift between 4 and 16 substeps against that miss's")
    out("own distance from the tolerance. A miss whose shift exceeds its margin is one")
    out("whose verdict the integrator, not the plant, decides.")
    declared = {}
    for position, log in enumerate(logs):
        for event in miss_events(log):
            value, _where, _calls, _steps = peak_divergence(
                model, log.command_rad, log.states, event, substeps=4
            )
            declared[(position, event.index)] = value
    at_risk = []
    worst_quotient = 0.0
    for key, peak16 in reference_peaks.items():
        margin = abs(peak16 - TOLERANCE)
        shift = abs(declared[key] - peak16)
        quotient = shift / margin if margin > 0 else float("inf")
        worst_quotient = max(worst_quotient, quotient)
        if quotient >= 1.0:
            at_risk.append((key, peak16, shift, margin))
    out(f"  misses examined                         : {len(reference_peaks)}")
    out(f"  worst shift-to-margin quotient          : {worst_quotient:.6e}")
    out(f"  misses whose quotient reaches 1         : {len(at_risk)}")
    out(f"  closest any peak comes to the tolerance : "
        f"{min(abs(v - TOLERANCE) for v in reference_peaks.values()):.6e}")
    out.check(
        "no miss's verdict is decided by the integrator's step size",
        not at_risk,
        f"worst shift-to-margin quotient {worst_quotient:.6e} over "
        f"{len(reference_peaks)} misses",
    )


def section_early_exit(out: Tee) -> None:
    out.rule("4. the optional early exit, measured rather than trusted")
    out("Its stopping criterion rests on the same linearisation as baseline 1, so")
    out("using it by default would make the reference depend on the thing the")
    out("reference exists to check. It is off by default and its agreement with the")
    out("full-horizon reference is measured here.")
    out("")
    model = LoopModel()
    constants = CertificateModel.build(model, STEPS)
    worst = 0.0
    flips = 0
    compared = 0
    full_calls = 0
    exit_calls = 0
    for log in _logs(12):
        for event in miss_events(log):
            full, _w1, calls_full, _s1 = peak_divergence(
                model, log.command_rad, log.states, event
            )
            early, _w2, calls_early, _s2 = peak_divergence(
                model,
                log.command_rad,
                log.states,
                event,
                constants=constants,
                early_exit=True,
            )
            worst = max(worst, abs(full - early))
            flips += int((full >= TOLERANCE) != (early >= TOLERANCE))
            compared += 1
            full_calls += calls_full
            exit_calls += calls_early
    out(f"  misses compared                  : {compared}")
    out(f"  worst |full - early| peak        : {worst:.6e}")
    out(f"  verdict flips                    : {flips}")
    out(f"  plant evaluations, full horizon  : {full_calls}")
    out(f"  plant evaluations, early exit    : {exit_calls}")
    saving = 1.0 - exit_calls / full_calls if full_calls else float("nan")
    out(f"  evaluations saved                : {saving:.6f}")
    out.check(
        "the early exit reproduces the full-horizon peak exactly on this population",
        worst == 0.0 and flips == 0,
        f"worst difference {worst:.6e}, {flips} verdict flips over {compared} misses",
    )
    out.check(
        "the early exit actually saves work, so the previous check is not vacuous",
        saving > 0.1,
        f"{saving:.6f} of plant evaluations",
    )


def section_superposition(out: Tee) -> None:
    out.rule("5. superposition: the only statement here about miss interactions")
    out("The per-miss analysis is one miss at a time. Re-simulating a run with ALL")
    out("its misses at once gives the quantity the per-miss answers do not cover, and")
    out("it is compared against two things: the aggregate bound of equation (15),")
    out("which must dominate it, and the SUM of the individual divergences, whose")
    out("departure from it is the nonlinear interaction residual.")
    out("")
    model = LoopModel()
    rows = []
    violations = 0
    for log in _logs(18):
        certificate = certify_run(log, TOLERANCE)
        if certificate.refusals:
            continue
        events = miss_events(log)
        schedule = {event.index: event.overrun_s for event in events}
        if not schedule:
            continue
        combined = simulate(model, log.command_rad, schedule)
        joint = float(np.max(model.metric.norm(combined.states - log.states)))
        individual = resimulate_run(log, TOLERANCE)
        total = float(sum(result.peak_divergence for result in individual))
        largest = float(max(result.peak_divergence for result in individual))
        aggregate = certificate.aggregate_bound
        if joint > aggregate:
            violations += 1
        rows.append(
            [
                log.label,
                str(len(events)),
                f"{joint:.6f}",
                f"{largest:.6f}",
                f"{total:.6f}",
                f"{aggregate:.6f}",
                f"{joint / aggregate:.6f}",
            ]
        )
    out(
        format_table(
            [
                "run",
                "misses",
                "all-misses divergence",
                "largest single",
                "sum of singles",
                "aggregate bound eq (15)",
                "ratio to bound",
            ],
            rows,
        )
    )
    out.check(
        "the aggregate bound of equation (15) dominates the all-misses divergence",
        violations == 0,
        f"{violations} of {len(rows)} runs",
    )
    out("")
    out("The all-misses divergence is not the sum of the singles and is not claimed")
    out("to be: misses close together interact through the closed loop, and a run")
    out("whose largest single miss is already the dominant term has a joint figure")
    out("close to it. What the table shows is that the superposition BOUND holds,")
    out("which is the claim equation (15) makes.")
    joint_values = np.array([float(row[2]) for row in rows])
    sums = np.array([float(row[4]) for row in rows])
    largest_values = np.array([float(row[3]) for row in rows])
    out(f"  worst joint / sum-of-singles   : {float(np.max(joint_values / sums)):.6f}")
    out(f"  worst joint / largest single   : "
        f"{float(np.max(joint_values / largest_values)):.6f}")
    out.check(
        "the all-misses divergence exceeds the largest single miss on at least one run",
        bool(np.any(joint_values > largest_values * 1.001)),
        f"{int(np.count_nonzero(joint_values > largest_values * 1.001))} of "
        f"{len(rows)} runs, which is why the per-miss answer is not the whole answer",
    )


def section_cost(out: Tee) -> None:
    out.rule("6. what the reference costs")
    model = LoopModel()
    rows = []
    for log in _logs(6):
        started = time.perf_counter()
        results = resimulate_run(log, TOLERANCE)
        elapsed = time.perf_counter() - started
        calls = sum(result.plant_calls for result in results)
        rows.append(
            [
                log.label,
                str(len(results)),
                str(calls),
                f"{elapsed * 1e3:.2f}",
                f"{elapsed * 1e6 / max(calls, 1):.4f}",
            ]
        )
    out(
        format_table(
            ["run", "misses", "plant evaluations", "wall [ms]",
             "microseconds per evaluation"],
            rows,
        )
    )
    out("")
    out("Wall-clock figures are UNSTABLE on shared hardware and are reported for")
    out("information. The evaluation counts are exact: a miss costs")
    out(f"4 x {model.sampling.substeps} x steps_remaining right-hand-side evaluations,")
    out("plus two more segments for the split period.")
    total_calls = sum(int(row[2]) for row in rows)
    out.check(
        "the whole cost section stays inside a few seconds on two cores",
        sum(float(row[3]) for row in rows) < 10000.0,
        f"{sum(float(row[3]) for row in rows):.1f} ms for {total_calls} evaluations",
    )


def main() -> int:
    out = Tee("validate_resimulation_output.txt")
    out.rule("hilverdict -- baseline 2, the re-simulation reference")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared tolerance {TOLERANCE}, declared substeps "
        f"{LoopModel().sampling.substeps}, run length {STEPS} iterations at "
        f"{LoopModel().sampling.period_s * 1e3:.1f} ms "
        f"({STEPS * LoopModel().sampling.period_s:.1f} s)")
    out(f"metric anisotropy {LoopModel().metric.anisotropy:.0f}, incidence tolerance "
        f"{math.degrees(LoopModel().metric.alpha_tolerance_rad):.4f} deg")
    section_determinism(out)
    section_prefix(out)
    section_substeps(out)
    section_early_exit(out)
    section_superposition(out)
    section_cost(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
