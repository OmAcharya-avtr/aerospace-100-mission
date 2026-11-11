"""The three backends: what each can serve, what it stamps, and what it refuses.

The live backend's reading path cannot be exercised against the declared
target from here, so what this script measures about it is the refusal: on
every board channel, with the message a user on a board-less host will
actually see. The simulated backend's determinism is measured by replaying a
sequence twice and comparing bit for bit, and the recorded backend's fidelity
by comparing a replay against the committed document it replays.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import os  # noqa: E402

from _bootstrap import (  # noqa: E402
    IDLE_TRACE,
    REPLAY_TRACE,
    SEED_CAMPAIGN,
    Tee,
    bootstrap,
    environment_lines,
    relative,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORIN_NANO,
    Access,
    DeviceUnreachableError,
    LiveSource,
    Provenance,
    RecordedSource,
    SimulatedSource,
    Trace,
    format_table,
    meet,
    read_available,
    render_channels,
    sealed_backends,
)
from boardprobe.sysio import PERF_EVENT_ATTR_SIZE  # noqa: E402

BOARD_CHANNELS = tuple(
    channel.name
    for channel in ORIN_NANO.channels
    if channel.access is not Access.CLOCK_SYSCALL
)


def main() -> int:
    out = Tee("validate_backends_output.txt")
    out.rule("boardprobe -- the three backends, and the declared channel table")
    for line in environment_lines():
        out(line)

    out.rule("1. the declared channel table")
    out(render_channels(ORIN_NANO))
    out("")
    out(f"board model                   : {ORIN_NANO.model}")
    out(f"channels declared             : {len(ORIN_NANO.channels)}")
    out(f"channels verified on hardware : {ORIN_NANO.verified_count}")
    out("")
    out("EVERY LOCATOR, RAW UNIT AND SCALE IN THAT TABLE IS DECLARED FROM")
    out("DOCUMENTATION AND IS UNVERIFIED ON HARDWARE. No NVIDIA Jetson Orin Nano")
    out("was reachable from the build environment; not one of those files was")
    out("opened and not one of those counters was read. The verified column reads")
    out("NO on every row, and the check below asserts that it does, so that the")
    out("day a reading is actually taken the flag has to be flipped deliberately.")
    out("")
    out.check(
        "no channel in the shipped table claims to be verified on hardware",
        ORIN_NANO.verified_count == 0,
        f"{ORIN_NANO.verified_count} of {len(ORIN_NANO.channels)}",
    )
    out.check(
        "every channel cites a reference and a relative locator",
        all(
            channel.reference and not channel.locator.startswith("/")
            for channel in ORIN_NANO.channels
        ),
        f"{len(ORIN_NANO.channels)} channels",
    )
    out.check(
        "all four source classes the specification names are declared",
        len({channel.kind for channel in ORIN_NANO.channels}) == 4,
        ", ".join(sorted({str(c.kind) for c in ORIN_NANO.channels})),
    )
    out("")
    out("the reference behind every channel")
    out(format_table(
        ["channel", "reference"],
        [[c.name, c.reference[:120]] for c in ORIN_NANO.channels],
    ))

    out.rule("2. the simulated backend is bit-reproducible at one seed")
    out("Exact float equality is correct HERE and nowhere else in this repository:")
    out("the claim is bit-reproducibility of a pseudo-random stream, not the value")
    out("of a physical quantity.")
    out("")
    first = {r.channel: r.value for r in SimulatedSource(ORIN_NANO, SEED_CAMPAIGN).read_all()}
    second = {r.channel: r.value for r in SimulatedSource(ORIN_NANO, SEED_CAMPAIGN).read_all()}
    identical = sum(1 for name in first if first[name] == second[name])
    out(format_table(
        ["channel", "run 1", "run 2", "identical"],
        [
            [name, f"{first[name]:.12g}", f"{second[name]:.12g}",
             "yes" if first[name] == second[name] else "NO"]
            for name in ORIN_NANO.names
        ],
    ))
    out("")
    out.check(
        f"all {len(first)} channels reproduce bit for bit at seed {SEED_CAMPAIGN}",
        identical == len(first),
        f"{identical} of {len(first)}",
    )
    subset = SimulatedSource(ORIN_NANO, SEED_CAMPAIGN).read_all(
        ["die_temp_gpu", "die_temp_cpu"]
    )
    out.check(
        "reading a subset gives the same values as reading every channel",
        all(reading.value == first[reading.channel] for reading in subset),
        "one generator per channel, so a value cannot depend on what else was read",
    )
    out.check(
        "every simulated reading is stamped simulated",
        all(
            reading.provenance is Provenance.SIMULATED
            for reading in SimulatedSource(ORIN_NANO, SEED_CAMPAIGN).read_all()
        ),
        f"{len(ORIN_NANO.channels)} channels",
    )

    out.rule("3. the recorded backend replays the document, and meets its provenance")
    rows = []
    for name, path in (("idle", IDLE_TRACE), ("replay path", REPLAY_TRACE)):
        trace = Trace.from_file(ROOT / path)
        source = RecordedSource(ORIN_NANO, trace)
        expected = meet(Provenance.RECORDED, trace.content_provenance)
        readings = source.read_all(trace.channels)
        matched = 0
        for reading in readings:
            sample = trace.samples_for(reading.channel)[0]
            if reading.value == sample.value and reading.unit == sample.unit:
                matched += 1
        rows.append([
            name,
            relative(ROOT / path),
            str(trace.sample_count),
            str(len(trace.channels)),
            trace.content_provenance.value,
            source.provenance.value,
            f"{matched}/{len(readings)}",
        ])
        out.check(
            f"the {name} trace replays as "
            f"{expected.value} = meet(recorded, {trace.content_provenance.value})",
            source.provenance is expected,
            source.provenance.value,
        )
        out.check(
            f"every first sample of the {name} trace replays with its own value and unit",
            matched == len(readings),
            f"{matched} of {len(readings)}",
        )
    out("")
    out(format_table(
        ["trace", "file", "samples", "channels", "content", "replay stamp",
         "first samples matched"],
        rows,
    ))
    out("")
    out("The idle fixture declares SIMULATED content, so replaying it yields")
    out("SIMULATED readings: meet(recorded, simulated) = simulated. That is the")
    out("lattice doing real work rather than decorating a log line. Both fixtures'")
    out("origin fields say, in the file, that their numbers were synthesised in a")
    out("container and are not measurements of anything.")

    out.rule("4. the live backend refuses every board channel, cleanly")
    out("Measured against the real filesystem root of the build host. The declared")
    out("target was not reachable, which is why this is the testable half of the")
    out("live path.")
    out("")
    live = LiveSource(ORIN_NANO)
    readings, refusals = read_available(live, BOARD_CHANNELS)
    out(format_table(
        ["channel", "first line of the refusal"],
        [[name, reason[:104]] for name, reason in refusals],
    ))
    out("")
    out(f"board channels probed : {len(BOARD_CHANNELS)}")
    out(f"served                : {len(readings)}")
    out(f"refused               : {len(refusals)}")
    out.check(
        "every board channel is refused on a host without the board",
        len(refusals) == len(BOARD_CHANNELS) and not readings,
        f"{len(refusals)} of {len(BOARD_CHANNELS)}",
    )
    actionable = sum(
        1
        for _, reason in refusals
        if any(
            hint in reason
            for hint in ("not reachable", "not present", "paranoid", "driver is bound")
        )
    )
    out.check(
        "every refusal says what a reader should check",
        actionable == len(refusals),
        f"{actionable} of {len(refusals)}",
    )
    # The needles are assembled from os.sep rather than written as literals,
    # because this file is committed and a committed file containing the
    # literal pattern is exactly what a repository-wide absolute-path scanner
    # is looking for. Three consecutive sessions of this mission introduced
    # that defect in the act of writing about it.
    needles = tuple(f"{os.sep}{name}{os.sep}" for name in ("home", "Users", "root"))
    absolute = [
        name
        for name, reason in refusals
        if any(needle in reason for needle in needles)
    ]
    out.check(
        "no refusal message quotes an absolute host path",
        not absolute,
        f"{len(absolute)} offenders; needles named, not printed: "
        + ", ".join(needle.strip(os.sep) for needle in needles),
    )

    out.rule("5. the live backend serves the monotonic clock, and that is not the board")
    clock_first = live.read("monotonic_clock")
    clock_second = live.read("monotonic_clock")
    out(f"  provenance : {clock_first.provenance.value}")
    out(f"  unit       : {clock_first.unit}")
    out(f"  monotonic  : {clock_second.value >= clock_first.value}")
    out(f"  detail     : {clock_first.detail}")
    out.check(
        "the monotonic clock is served and stamped live",
        clock_first.provenance is Provenance.LIVE and clock_first.unit == "s",
        f"{clock_first.value:.6f} s",
    )
    out.check(
        "two consecutive clock readings do not go backwards",
        clock_second.value >= clock_first.value,
        f"{(clock_second.value - clock_first.value) * 1e6:.3f} us apart",
    )
    out("")
    out("A live reading of this host's clock is a live reading of a real clock and")
    out("is NOT a reading of a Jetson Orin Nano. That distinction is the platform")
    out("attestation's job, not the lattice's, and it is why every MEASURED")
    out("certificate in this repository reads MEASURED OFF-TARGET.")

    out.rule("6. the perf_event_open adapter, and its clean failure here")
    out("Written from perf_event_open(2) and linux/perf_event.h. UNVERIFIED on the")
    out("declared target: the syscall number on aarch64 is 241 and has never been")
    out("executed; 298 on x86_64 is the one that was.")
    out("")
    out(f"  sizeof(struct perf_event_attr) as declared : {PERF_EVENT_ATTR_SIZE} bytes")
    error_text = ""
    try:
        live.read("perf_cpu_cycles")
    except DeviceUnreachableError as error:
        error_text = str(error)
    out(f"  the failure on this host                   : {error_text[:150]}")
    out.check(
        "the perf adapter fails with the C errno and what to check",
        "errno" in error_text and "paranoid" in error_text,
        "errno and the paranoid hint both present",
    )

    out.rule("7. the three backends are sealed")
    out(f"  sealed classes : {[cls.__name__ for cls in sealed_backends()]}")
    refusals_counted = 0
    for base in sealed_backends():
        try:
            type("Sneaky", (base,), {})
        except TypeError:
            refusals_counted += 1
    out.check(
        "no sealed backend can be subclassed",
        refusals_counted == len(sealed_backends()),
        f"{refusals_counted} of {len(sealed_backends())} refused at class creation",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
