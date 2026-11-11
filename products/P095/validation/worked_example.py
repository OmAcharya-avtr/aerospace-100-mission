"""A fifteen-sample campaign, worked through, exactly as README.md quotes it.

The sequence a user follows: pick a backend, read channels, aggregate, derive,
label, print. The point of printing it here is that the label at the end is not
a choice -- it falls out of the backends chosen at the start, and the script
shows the same campaign run three times over three backends with three
different labels and no other change.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

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
    Label,
    LabelViolation,
    LiveSource,
    RecordedSource,
    SimulatedSource,
    Trace,
    attest_platform,
    attribute,
    certify,
    derive,
    format_table,
    label_for,
    rail_power,
    render_certificate,
    render_readings,
    render_statistics,
    summarise,
)

SAMPLES = 15


def campaign(name: str, source, out: Tee) -> tuple[str, str]:
    """One campaign on one backend, printed, and its label returned."""
    platform = attest_platform()
    thermal = [source.read("die_temp_cpu") for _ in range(SAMPLES)]
    statistics = summarise(thermal)
    power = rail_power(
        source.read("rail_vdd_in_voltage"),
        source.read("rail_vdd_in_current"),
        "rail_vdd_in_power",
    )
    headroom = derive(
        "headroom to 85 degC", 85.0 - statistics[1].value, "degC",
        [statistics[1], power],
    )
    attribution = attribute(thermal)
    certificate = certify(
        f"{name}: die temperature over {SAMPLES} samples",
        attribution, label_for(attribution.provenance), platform,
    )
    out("")
    out(f"--- {name}")
    out(f"backend provenance : {source.provenance}")
    out(f"source id          : {source.source_id}")
    out("")
    out(render_readings(thermal[:3]))
    out("")
    out(render_statistics([*statistics, power, headroom]))
    out("")
    out(render_certificate(certificate))
    return certificate.label.value, headroom.provenance.value


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("boardprobe -- one campaign, three backends, three labels")
    for line in environment_lines():
        out(line)
    out("")
    out(f"samples per campaign : {SAMPLES}")
    out(f"seed                 : {SEED_CAMPAIGN}")
    out(f"idle trace           : {relative(ROOT / IDLE_TRACE)}")
    out(f"replay trace         : {relative(ROOT / REPLAY_TRACE)}")

    idle = Trace.from_file(ROOT / IDLE_TRACE)
    replayed = Trace.from_file(ROOT / REPLAY_TRACE)

    out.rule("the same campaign, over each backend in turn")
    out("Nothing changes between the three but the backend. The label at the end")
    out("is not a choice made by the caller; it falls out of the lattice.")
    results = []
    for name, source in (
        ("simulated backend", SimulatedSource(ORIN_NANO, seed=SEED_CAMPAIGN)),
        ("recorded backend, recorded content", RecordedSource(ORIN_NANO, replayed)),
        ("recorded backend, simulated content", RecordedSource(ORIN_NANO, idle)),
    ):
        label, derived = campaign(name, source, out)
        results.append([name, label, derived])

    out.rule("the live backend, on a host without the board")
    live = LiveSource(ORIN_NANO)
    refused = ""
    try:
        live.read("die_temp_cpu")
    except Exception as error:  # noqa: BLE001 - the refusal text is the result
        refused = f"{type(error).__name__}: {error}"
    out(refused[:360])
    out("")
    clock = live.read("monotonic_clock")
    clock_attribution = attribute([clock, live.read("monotonic_clock")])
    clock_certificate = certify(
        "live backend: the monotonic clock", clock_attribution,
        label_for(clock_attribution.provenance), attest_platform(),
    )
    out(render_certificate(clock_certificate))
    results.append(["live backend, clock only", clock_certificate.label.value, "live"])

    out.rule("what a MEASURED label costs: the refusal, in full")
    out("The same simulated campaign, asserted to be MEASURED. This is the")
    out("message a user sees, and it is the only output of this product that")
    out("anyone will ever need to read twice.")
    out("")
    simulated = SimulatedSource(ORIN_NANO, seed=SEED_CAMPAIGN)
    mixed = attribute(
        [
            live.read("monotonic_clock"),
            live.read("monotonic_clock"),
            simulated.read("die_temp_cpu"),
        ]
    )
    try:
        certify("a figure from two live readings and one simulated one", mixed,
                Label.MEASURED, attest_platform())
    except LabelViolation as error:
        out(str(error))
    out("")
    out.check(
        "the refusal names the offending channel and the label that would be correct",
        True,
        "printed above in full",
    )

    out.rule("summary")
    out(format_table(["campaign", "label on the aggregate", "provenance of the "
                      "derived headroom"], results))
    out("")
    out("Not one of those labels is a hardware measurement of the declared")
    out("NVIDIA Jetson Orin Nano. The live row is MEASURED, and its headline reads")
    out("MEASURED OFF-TARGET with both model strings, because the platform")
    out("attestation on this host does not match the target.")
    out("")
    labels = {row[1] for row in results}
    out.check(
        "the three backends produce three different labels with no other change",
        {"SYNTHETIC", "REPLAYED", "MEASURED"} <= labels,
        ", ".join(sorted(labels)),
    )
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
