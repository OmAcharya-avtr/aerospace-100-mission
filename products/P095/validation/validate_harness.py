"""The benchmark harness, and the honest status of every number it produces.

EVERY TIMING AND RESOURCE FIGURE IN THIS SCRIPT'S OUTPUT IS A PROPERTY OF THE
BUILD HOST AND NOT OF AN NVIDIA JETSON ORIN NANO. The host's specification is
printed in the header of every output file in this directory, and the
ARCHITECTURE line matters: the declared target is aarch64 and this is x86_64,
so even the instruction counts a working perf adapter would report on the
target are not comparable with anything here.

What the script does measure, and what the README quotes from it:

- the label the harness earns for each combination of clock backend and
  environment backend, which is the propagation rule applied to a real
  artifact rather than to an enum;
- the harness's own cost, broken into the clock read, the sealed reading and
  the in-bracket total, and the perturbation of a declared reference payload;
- the clock resolution available at this host's uptime, so the reader can see
  that the reported durations are far above the floor instead of being told so.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from _bootstrap import (  # noqa: E402
    BENCHMARK_REPEATS,
    OVERHEAD_REPEATS,
    REFERENCE_WORK,
    REPLAY_TRACE,
    SEED_CAMPAIGN,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORIN_NANO,
    HarnessError,
    LiveSource,
    RecordedSource,
    SimulatedSource,
    Trace,
    benchmark,
    characterise_overhead,
    format_table,
    reference_payload,
    render_benchmark,
    render_overhead,
    timing_resolution_s,
)


def main() -> int:
    out = Tee("validate_harness_output.txt")
    out.rule("boardprobe -- the benchmark harness, on the build host and not on a board")
    for line in environment_lines():
        out(line)
    out("")
    out("NOT ONE FIGURE BELOW IS A HARDWARE MEASUREMENT OF THE DECLARED TARGET.")
    out("The NVIDIA Jetson Orin Nano was not reachable from this environment. The")
    out("figures are container numbers, labelled as such, with the container's")
    out("specification in the header above. The target is aarch64; this is x86_64.")

    clock = LiveSource(ORIN_NANO)
    payload = reference_payload(REFERENCE_WORK)
    trace = Trace.from_file(ROOT / REPLAY_TRACE)

    out.rule("1. the label the harness earns, per backend combination")
    out("The propagation rule applied to a real artifact. The clock decides the")
    out("timing label; the environment drags the overall label down to the meet.")
    out("")
    rows = []
    for clock_name, clock_source in (
        ("live", clock),
        ("simulated", SimulatedSource(ORIN_NANO, seed=SEED_CAMPAIGN)),
    ):
        for environment_name, environment, channels in (
            ("none", None, None),
            ("simulated", SimulatedSource(ORIN_NANO, seed=SEED_CAMPAIGN + 1), None),
            ("recorded", RecordedSource(ORIN_NANO, trace), trace.channels),
        ):
            result = benchmark(
                payload, f"reference_payload({REFERENCE_WORK})", clock_source,
                repeats=24, warmup=3, environment=environment,
                environment_channels=channels,
            )
            rows.append([
                clock_name,
                environment_name,
                result.timing_certificate.label.value,
                result.label.value,
                "yes" if result.overall_certificate.is_hardware_measurement else "NO",
                str(len(result.refusals)),
            ])
    out(format_table(
        ["clock backend", "environment backend", "timing label", "overall label",
         "hardware measurement", "refusals"],
        rows,
    ))
    out("")
    out.check(
        "no combination available on this host yields a hardware measurement",
        all(row[4] == "NO" for row in rows),
        f"{len(rows)} combinations",
    )
    out.check(
        "a live clock with a simulated environment is SYNTHETIC overall",
        any(
            row[:2] == ["live", "simulated"]
            and row[2] == "MEASURED"
            and row[3] == "SYNTHETIC"
            for row in rows
        ),
        "timing MEASURED, overall SYNTHETIC",
    )
    out.check(
        "a live clock with a recorded environment is REPLAYED overall",
        any(
            row[:2] == ["live", "recorded"] and row[3] == "REPLAYED" for row in rows
        ),
        "the meet of live and recorded",
    )
    out.check(
        "a simulated clock can never yield a MEASURED timing label",
        all(row[2] != "MEASURED" for row in rows if row[0] == "simulated"),
        "3 combinations",
    )

    out.rule("2. the board-absent environment refuses rather than degrading")
    refused = ""
    try:
        benchmark(
            payload, "absent", clock, repeats=4,
            environment=LiveSource(ORIN_NANO),
        )
    except HarnessError as error:
        refused = str(error)
    out(f"  {refused[:200]}")
    out.check(
        "a live environment backend on a board-less host raises, pointing at "
        "the recovery procedure",
        "could serve none of" in refused and "RECOVERY.md" in refused,
        "HarnessError with the recovery pointer",
    )

    out.rule("3. a timing-only benchmark, rendered as the CLI renders it")
    result = benchmark(
        payload, f"reference_payload({REFERENCE_WORK})", clock,
        repeats=BENCHMARK_REPEATS, warmup=5,
    )
    out(render_benchmark(result))
    out("")
    durations = np.asarray(result.durations_s)
    out(f"  repeats                       : {result.repeats}")
    out(f"  mean duration        [us]     : {durations.mean() * 1e6:.3f}")
    out(f"  sd                   [us]     : {durations.std(ddof=1) * 1e6:.3f}")
    out(f"  median               [us]     : {np.median(durations) * 1e6:.3f}")
    out(f"  clock resolution at uptime [s]: {result.timing_resolution_s:.3e}")
    ratio = float(np.median(durations)) / result.timing_resolution_s
    out(f"  median duration / resolution  : {ratio:.3e}")
    out.check(
        "the reported durations are many orders of magnitude above the clock's "
        "resolution at this uptime",
        ratio > 1e6,
        f"{ratio:.3e} times the floor",
    )
    out.check(
        "every per-repeat duration is non-negative",
        bool((durations >= 0.0).all()),
        f"{durations.size} repeats",
    )

    out.rule("4. the harness's own cost, and the perturbation it causes")
    report = characterise_overhead(clock, repeats=OVERHEAD_REPEATS)
    out(render_overhead(report))
    out("")
    out("The perturbation ratio is the reference payload's per-repeat cost with")
    out("sampling divided by its cost without. It is NOISY: on a shared two-core")
    out("host single trials of it have come back BELOW 1.0, which is not a")
    out("negative overhead but a statement that the perturbation is at the level")
    out("of the measurement noise. validation/validate_independent_bound_output.txt")
    out("reports it over several trials for that reason, and this single figure")
    out("should be read with that in mind.")
    out("")
    out.check(
        "the in-bracket overhead is positive and smaller than the reference "
        "payload's own cost",
        0.0 < report.in_bracket_overhead_s < report.reference_without_sampling_s,
        f"{report.in_bracket_overhead_s * 1e6:.3f} us against "
        f"{report.reference_without_sampling_s * 1e6:.3f} us",
    )
    out.check(
        "the overhead characterisation is labelled MEASURED OFF-TARGET, never "
        "a bare MEASURED",
        report.headline.startswith("MEASURED OFF-TARGET"),
        report.headline[:60],
    )

    out.rule("5. the clock resolution, as a function of uptime")
    out("A clock channel's value is float seconds, so a duration is a difference")
    out("of two floats of magnitude equal to the host's uptime, and the floor on")
    out("what the harness can resolve is one ulp at that magnitude.")
    out("")
    out(format_table(
        ["uptime [s]", "resolution [s]"],
        [[f"{value:.0e}", f"{timing_resolution_s(value):.3e}"]
         for value in (1e3, 1e5, 1e6, 1e7, 1e8, 1e9)],
    ))
    out("")
    out(f"  this host's uptime magnitude : {result.timing_resolution_s:.3e} s per ulp")
    out.check(
        "the resolution grows monotonically with uptime",
        all(
            timing_resolution_s(a) <= timing_resolution_s(b)
            for a, b in zip((1e3, 1e5, 1e6, 1e7, 1e8), (1e5, 1e6, 1e7, 1e8, 1e9),
                            strict=True)
        ),
        "6 magnitudes",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
