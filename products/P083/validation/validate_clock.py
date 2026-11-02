"""Host-to-device clock drift and reset detection, against synthetic truth.

Every number in a timing verdict is a duration and every duration was measured
by a clock, so this is checked against a known answer rather than assumed.
Five things are measured:

1. **Drift recovery** over a sweep of known rates from -20000 to +20000 parts
   per million crossed with four timestamp-jitter levels, scored on absolute
   error in ppm against the injected rate.
2. **Reset detection** against injected resets of known index, scored on recall
   and precision, with the false-positive rate on reset-free series reported
   separately.
3. **The detection floor**: the smallest offset jump the detector finds. A
   reset smaller than that is NOT detected, and the floor is measured rather
   than claimed to be zero.
4. **Robustness**: least squares against Theil-Sen on series with corrupted
   timestamps, which is the case the two estimators are chosen to fail
   differently on.
5. **The consequence for the verdict**: the number of iterations that appear to
   have missed their deadline, before and after the drift correction. This is
   the reason the clock check is part of the timing verdict rather than a side
   report.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. See
# validation/validate_cli.py section 3; CPython writes a module's bytecode
# cache BEFORE executing it, so _bootstrap cannot set this on its own behalf.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    ALLOWANCE_PPM,
    DRIFT_SWEEP_PPM,
    JITTER_SWEEP_S,
    SOUNDNESS_SEEDS,
    STEPS,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from hilverdict.clock import (  # noqa: E402
    CLOCK_CLEAN_VERDICT,
    CLOCK_DRIFTING,
    CLOCK_RESET,
    detect_resets,
    drift_corrected_log,
    estimate_clock,
    least_squares_rate,
    theil_sen_rate,
)
from hilverdict.report import format_table  # noqa: E402
from hilverdict.synthetic import (  # noqa: E402
    CAMPAIGNS,
    ClockSpec,
    generate_clock,
    generate_run,
)
from hilverdict.timeline import miss_events  # noqa: E402

PERIOD_S = 0.010


def _host(samples: int) -> np.ndarray:
    return np.arange(samples, dtype=float) * PERIOD_S


def section_drift(out: Tee) -> None:
    out.rule("1. drift recovery against synthetic drift of KNOWN rate")
    out("The injected rate is the truth. The reported error is |estimated - injected|")
    out("in parts per million, and the standard error the estimator reports is")
    out("printed beside it so the two can be compared.")
    out("")
    rows = []
    worst_error = 0.0
    inside = 0
    total = 0
    for jitter in JITTER_SWEEP_S:
        for drift in DRIFT_SWEEP_PPM:
            spec = ClockSpec(
                drift_ppm=drift,
                jitter_s=jitter,
                tick_s=1.0e-7,
                reset_fractions=(),
                name="sweep",
            )
            generator = np.random.default_rng(int(abs(drift) * 7) + 11)
            host = _host(STEPS + 1)
            device, _resets = generate_clock(spec, host, generator)
            estimate = estimate_clock(host, device, ALLOWANCE_PPM)
            error = abs(estimate.drift_ppm - drift)
            worst_error = max(worst_error, error)
            total += 1
            # Four standard errors is the declared criterion, fixed before the
            # measurement, and it is a criterion on the ESTIMATOR's own reported
            # error rather than a round number in ppm.
            if error <= 4.0 * estimate.drift_ppm_standard_error:
                inside += 1
            rows.append(
                [
                    f"{jitter:g}",
                    f"{drift:g}",
                    f"{estimate.drift_ppm:.6f}",
                    f"{error:.6f}",
                    f"{estimate.drift_ppm_standard_error:.6f}",
                    f"{error / max(estimate.drift_ppm_standard_error, 1e-300):.3f}",
                    estimate.verdict,
                ]
            )
    out(
        format_table(
            [
                "jitter [s]",
                "injected [ppm]",
                "estimated [ppm]",
                "|error| [ppm]",
                "se [ppm]",
                "error/se",
                "verdict",
            ],
            rows,
        )
    )
    out(f"  worst absolute error over the sweep : {worst_error:.6f} ppm")
    out(f"  estimates inside 4 standard errors  : {inside} of {total}")
    out.check(
        "every drift in the sweep is recovered inside four of its own standard errors",
        inside == total,
        f"{inside} of {total}",
    )
    out.check(
        "the worst absolute recovery error over the sweep is below 10 ppm",
        worst_error < 10.0,
        f"{worst_error:.6f} ppm against injected rates up to 20000 ppm",
    )
    out("")
    out("Verdict agreement with the declared allowance:")
    agree = 0
    for row in rows:
        expected = (
            CLOCK_DRIFTING if abs(float(row[1])) > ALLOWANCE_PPM else CLOCK_CLEAN_VERDICT
        )
        agree += int(row[6] == expected)
    out(f"  verdicts matching the declared allowance of {ALLOWANCE_PPM:g} ppm: "
        f"{agree} of {len(rows)}")
    out.check(
        "the clock verdict matches the declared allowance on every sweep point",
        agree == len(rows),
        f"{agree} of {len(rows)}",
    )


def section_resets(out: Tee) -> None:
    out.rule("2. reset detection against injected resets of known index")
    out("Recall and precision are scored against the indices the generator used,")
    out("which it returns for this purpose rather than letting the detector grade")
    out("its own homework. A detection is counted correct when it lands on the")
    out("injected index exactly: the reset takes effect at one sample and there is")
    out("no ambiguity to allow for.")
    out("")
    rows = []
    found = 0
    injected = 0
    detections = 0
    for jitter in JITTER_SWEEP_S:
        for fractions in ((), (0.4,), (0.25, 0.7), (0.15, 0.45, 0.8)):
            spec = ClockSpec(
                drift_ppm=250.0,
                jitter_s=jitter,
                tick_s=1.0e-7,
                reset_fractions=fractions,
                name="sweep",
            )
            generator = np.random.default_rng(len(fractions) * 101 + 5)
            host = _host(STEPS + 1)
            device, truth = generate_clock(spec, host, generator)
            detected = detect_resets(host, device)
            hits = len(set(truth) & set(detected))
            found += hits
            injected += len(truth)
            detections += len(detected)
            rows.append(
                [
                    f"{jitter:g}",
                    str(len(truth)),
                    str(list(truth)),
                    str(list(detected)),
                    str(hits),
                    str(len(detected) - hits),
                ]
            )
    out(
        format_table(
            ["jitter [s]", "injected", "truth", "detected", "correct", "spurious"],
            rows,
        )
    )
    recall = found / injected if injected else float("nan")
    precision = found / detections if detections else float("nan")
    out(f"  resets injected : {injected}")
    out(f"  resets detected : {detections}")
    out(f"  correct         : {found}")
    out(f"  recall          : {recall:.6f}")
    out(f"  precision       : {precision:.6f}")
    out.check(
        "every injected reset in the sweep is detected at the exact index",
        found == injected,
        f"recall {recall:.6f} over {injected} injected resets",
    )
    out.check(
        "no spurious reset is reported anywhere in the sweep",
        detections == found,
        f"precision {precision:.6f}, {detections - found} spurious detections",
    )
    out("")
    out("The reset-free rows above are the false-positive control: a run with no")
    out("reset must produce no detection, and a reset-free series is included at")
    out("every jitter level.")
    clean_rows = [row for row in rows if row[1] == "0"]
    out.control(
        "a reset-free series produces a detection",
        fired=any(row[3] != "[]" for row in clean_rows),
        must_fire=False,
        detail=f"{len(clean_rows)} reset-free series, detections "
        f"{[row[3] for row in clean_rows]}",
    )
    out.control(
        "a series with three injected resets produces three detections",
        fired=all(
            int(row[4]) == 3 for row in rows if row[1] == "3"
        ),
        must_fire=True,
        detail=f"{sum(1 for row in rows if row[1] == '3')} such series",
    )


def section_floor(out: Tee) -> None:
    out.rule("3. the detection floor: the smallest reset the detector finds")
    out("The threshold is the larger of eight robust scales of the first differences")
    out("and a declared absolute floor of 1e-4 s, so a reset whose offset jump is")
    out("smaller than that floor is NOT detected. That is a real limitation and the")
    out("floor is measured here rather than asserted to be zero.")
    out("")
    host = _host(STEPS + 1)
    generator = np.random.default_rng(777)
    rows = []
    smallest_detected = None
    largest_missed = 0.0
    for jump in (1e-7, 1e-6, 1e-5, 5e-5, 9e-5, 1.1e-4, 1e-3, 1e-2, 1e-1):
        device = host.copy()
        device[100:] -= jump
        device += 2.0e-7 * generator.standard_normal(device.shape[0])
        device = np.round(device / 1.0e-7) * 1.0e-7
        detected = detect_resets(host, device)
        hit = 100 in detected
        if hit and smallest_detected is None:
            smallest_detected = jump
        if not hit:
            largest_missed = max(largest_missed, jump)
        rows.append(
            [f"{jump:g}", "yes" if hit else "no", str(list(detected))]
        )
    out(format_table(["offset jump [s]", "detected", "indices reported"], rows))
    out("")
    out(f"  smallest jump detected : {smallest_detected:g} s")
    out(f"  largest jump missed    : {largest_missed:g} s")
    out(f"  as a fraction of a {PERIOD_S * 1e3:g} ms period: "
        f"{largest_missed / PERIOD_S:.6f} missed, "
        f"{smallest_detected / PERIOD_S:.6f} detected")
    out.check(
        "the detection floor is between the declared 1e-4 s and one period",
        1e-4 <= smallest_detected <= PERIOD_S,
        f"smallest detected {smallest_detected:g} s",
    )
    out.control(
        "a reset smaller than the declared floor is missed",
        fired=largest_missed > 0.0,
        must_fire=True,
        detail=f"largest missed {largest_missed:g} s; this is the limitation, "
        "reported rather than hidden",
    )
    out("")
    out("What a missed reset costs: an offset jump below the floor shifts every")
    out("later timestamp by that amount, which is at most 1e-4 s, or 1 per cent of")
    out("a period. A duration straddling it is wrong by the same amount. So the")
    out("undetected cases are exactly the ones whose consequence for a duration is")
    out("bounded by the floor, and the report says so.")


def section_robustness(out: Tee) -> None:
    out.rule("4. least squares against Theil-Sen under corrupted timestamps")
    out("The two estimators are provided because they fail differently. Least")
    out("squares is efficient under the declared Gaussian jitter and is pulled by a")
    out("single bad timestamp; Theil-Sen has a breakdown point of 29 per cent. Both")
    out("are reported in every segment table, and this is the measurement behind")
    out("that choice.")
    out("")
    host = _host(STEPS + 1)
    generator = np.random.default_rng(31337)
    truth_ppm = 300.0
    clean = (1.0 + truth_ppm * 1e-6) * host + 2.0e-7 * generator.standard_normal(
        host.shape[0]
    )
    rows = []
    for corrupted in (0, 1, 2, 5, 10, 20, 40):
        device = clean.copy()
        if corrupted:
            which = generator.choice(host.shape[0], size=corrupted, replace=False)
            device[which] += 0.05
        squares, _error = least_squares_rate(host, device)
        robust = theil_sen_rate(host, device)
        rows.append(
            [
                str(corrupted),
                f"{(squares - 1.0) * 1e6:.4f}",
                f"{abs((squares - 1.0) * 1e6 - truth_ppm):.4f}",
                f"{(robust - 1.0) * 1e6:.4f}",
                f"{abs((robust - 1.0) * 1e6 - truth_ppm):.4f}",
            ]
        )
    out(
        format_table(
            [
                "corrupted samples",
                "least squares [ppm]",
                "|error|",
                "theil-sen [ppm]",
                "|error|",
            ],
            rows,
        )
    )
    out("")
    heavy = rows[-1]
    out.check(
        "Theil-Sen beats least squares when a fifth of the timestamps are corrupted",
        float(heavy[4]) < float(heavy[2]),
        f"Theil-Sen error {heavy[4]} ppm against least squares {heavy[2]} ppm at "
        f"{heavy[0]} corrupted samples of {STEPS + 1}",
    )
    out.control(
        "least squares is measurably degraded by corrupted timestamps",
        fired=float(rows[-1][2]) > 10.0 * float(rows[0][2]) + 1.0,
        must_fire=True,
        detail=f"error grows from {rows[0][2]} ppm clean to {rows[-1][2]} ppm at "
        f"{rows[-1][0]} corrupted samples",
    )
    out.control(
        "Theil-Sen is degraded as badly as least squares",
        fired=float(rows[-1][4]) > float(rows[-1][2]),
        must_fire=False,
        detail=f"Theil-Sen {rows[-1][4]} ppm against least squares {rows[-1][2]} ppm",
    )


def section_consequence(out: Tee) -> None:
    out.rule("5. the consequence for the verdict: which iterations appear to miss")
    out("The recorded durations are measured on the device clock, so a clock running")
    out("fast by rho reports every interval as (1 + rho) times its true length and")
    out("the set of iterations that appear to have overrun their deadline changes.")
    out("This is why the clock check runs before anything else in")
    out("hilverdict.verdict, and the size of the effect is measured here.")
    out("")
    rows = []
    for drift in DRIFT_SWEEP_PPM:
        raw_total = 0
        corrected_total = 0
        truth_total = 0
        gained = 0
        lost = 0
        for position, seed in enumerate(SOUNDNESS_SEEDS[:12]):
            profile, amplitude = CAMPAIGNS[position % len(CAMPAIGNS)]
            spec = ClockSpec(
                drift_ppm=drift,
                jitter_s=2.0e-7,
                tick_s=1.0e-7,
                reset_fractions=(),
                name="consequence",
            )
            drifted = generate_run(
                seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
                clock=spec,
            )
            truth = generate_run(
                seed=seed, profile=profile, amplitude_deg=amplitude, steps=STEPS,
            )
            estimate = estimate_clock(
                drifted.host_time_s, drifted.device_time_s, ALLOWANCE_PPM
            )
            corrected = drift_corrected_log(drifted, estimate.drift_ppm)
            raw_indices = {event.index for event in miss_events(drifted)}
            corrected_indices = {event.index for event in miss_events(corrected)}
            truth_indices = {event.index for event in miss_events(truth)}
            raw_total += len(raw_indices)
            corrected_total += len(corrected_indices)
            truth_total += len(truth_indices)
            gained += len(raw_indices - truth_indices)
            lost += len(truth_indices - raw_indices)
        rows.append(
            [
                f"{drift:g}",
                str(truth_total),
                str(raw_total),
                str(corrected_total),
                str(gained),
                str(lost),
                str(abs(corrected_total - truth_total)),
            ]
        )
    out(
        format_table(
            [
                "drift [ppm]",
                "true misses",
                "apparent misses",
                "after correction",
                "spurious",
                "missed",
                "residual after correction",
            ],
            rows,
        )
    )
    out("")
    worst_uncorrected = max(int(row[4]) + int(row[5]) for row in rows)
    worst_corrected = max(int(row[6]) for row in rows)
    out(f"  worst miscount from uncorrected drift : {worst_uncorrected} misses")
    out(f"  worst residual miscount after correction : {worst_corrected} misses")
    out.check(
        "the drift correction removes most of the miscount",
        worst_corrected < worst_uncorrected,
        f"{worst_corrected} against {worst_uncorrected} over "
        f"{len(SOUNDNESS_SEEDS[:12])} runs per drift level",
    )
    out.control(
        "uncorrected drift changes the set of apparent misses",
        fired=worst_uncorrected > 0,
        must_fire=True,
        detail=f"{worst_uncorrected} misses gained or lost at the worst drift level",
    )
    out("")
    out("A reset is NOT correctable in this way, which is why a detected reset makes")
    out("the run undecidable:")
    spec = ClockSpec(
        drift_ppm=0.0, jitter_s=2.0e-7, tick_s=1.0e-7, reset_fractions=(0.4,),
        name="reset",
    )
    reset_run = generate_run(seed=SOUNDNESS_SEEDS[0], profile="loaded",
                             amplitude_deg=6.0, steps=STEPS, clock=spec)
    estimate = estimate_clock(
        reset_run.host_time_s, reset_run.device_time_s, ALLOWANCE_PPM
    )
    out(f"  verdict : {estimate.verdict}")
    out(f"  resets  : {list(estimate.resets)}")
    out.check(
        "a run with an injected reset is reported as CLOCK_RESET",
        estimate.verdict == CLOCK_RESET,
        f"verdict {estimate.verdict}",
    )


def main() -> int:
    out = Tee("validate_clock_output.txt")
    out.rule("hilverdict -- host-to-device clock drift and reset detection")
    for line in environment_lines():
        out(line)
    out("")
    out(f"declared drift allowance {ALLOWANCE_PPM:g} ppm, run length {STEPS} "
        f"iterations at {PERIOD_S * 1e3:g} ms")
    section_drift(out)
    section_resets(out)
    section_floor(out)
    section_robustness(out)
    section_consequence(out)
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
