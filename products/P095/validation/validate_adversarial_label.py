"""The adversarial check: every route from a simulated reading to a measured label.

The specification for this product asks for "a validation check asserting that
no simulated reading can reach a result labelled as measured". A check that
tries one route and reports success is worth nothing, so this one tries every
route the API exposes, names each, and reports what happened to it.

The routes are organised by what they attack:

  1-4   the labelling itself: direct, aggregated, diluted, second-order
  5-7   replay: a replayed trace, a document that declares live content, a
        document round-tripped back into a reading
  8-10  the value object: a mutated field, a malicious subclass, a
        ``dataclasses.replace`` relabelling
 11-13  the backend: an overriding subclass, a subclass of a sealed backend, a
        rebound class attribute
 14-15  the lattice and the certificate: the empty-fold identity, direct
        construction of a certificate

Two of these were FOUND to succeed during this build and the design was
changed, not the check. Both are recorded below in their own sections, with
what the successful attack was, and both are now refused. Routes that remain
open are reported as open, with the reason, because a check that reported
fifteen of fifteen refused while something was open would be the worst
artifact in this repository.
"""

from __future__ import annotations

# Bytecode must not be written anywhere inside this tree. validation/
# validate_cli.py asserts that running it ADDS no __pycache__ directory and no
# .pyc file, so an interpreter that caches its own imports here makes that
# check report itself. It is set in each entry script, before _bootstrap is
# imported, and not inside _bootstrap, because CPython writes a module's cache
# BEFORE executing that module: _bootstrap cannot set this on its own behalf.
# The imports below therefore follow a statement and carry the same E402
# waiver the post-bootstrap imports already carry.
import sys  # noqa: E402

sys.dont_write_bytecode = True

import dataclasses  # noqa: E402
import json  # noqa: E402

from _bootstrap import (  # noqa: E402
    IDLE_TRACE,
    REPLAY_TRACE,
    SEED_CAMPAIGN,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORIN_NANO,
    Certificate,
    Label,
    LiveSource,
    Provenance,
    Reading,
    RecordedSource,
    SimulatedSource,
    Source,
    Trace,
    attest_platform,
    attribute,
    certify,
    derive,
    format_table,
    from_mapping,
    label_for,
    meet_all,
    mint,
    summarise,
)

PLATFORM = attest_platform()


class RouteSucceeded(AssertionError):
    """Raised when an attack reaches a measured label. A defect, not a result."""


def attempt(out: Tee, number: int, name: str, action) -> tuple[int, str, str]:
    """Run one attack. Report whether it was refused and by what.

    Returns ``(number, verdict, detail)``. ``action`` must either raise -- in
    which case the route is REFUSED and the exception type and first line are
    recorded -- or return, in which case the route SUCCEEDED and that is a
    defect.
    """
    try:
        result = action()
    except Exception as error:  # noqa: BLE001 - every refusal shape is a result
        detail = f"{type(error).__name__}: {str(error).splitlines()[0][:120]}"
        out.check(f"route {number:>2}: {name} -- REFUSED", True, detail)
        return number, "REFUSED", detail
    detail = f"returned {result!r}"[:140]
    out.check(f"route {number:>2}: {name} -- SUCCEEDED, which is a DEFECT", False, detail)
    return number, "SUCCEEDED", detail


def main() -> int:  # noqa: C901, PLR0915 - one function per route, read top to bottom
    out = Tee("validate_adversarial_label_output.txt")
    out.rule("boardprobe -- every route from a simulated reading to a measured label")
    for line in environment_lines():
        out(line)
    out("")
    out("The question: can a reading that is not live reach a figure labelled")
    out("MEASURED, by any route the public API exposes? Each route below either")
    out("raises -- which is a refusal and a pass -- or returns, which is a defect.")
    out("")
    out("TWO ROUTES SUCCEEDED when this check was first written and the DESIGN was")
    out("changed in response. Both are listed in their own section at the end with")
    out("what the successful attack was.")

    simulated = SimulatedSource(ORIN_NANO, seed=SEED_CAMPAIGN)
    clock = LiveSource(ORIN_NANO)
    replayed = Trace.from_file(ROOT / REPLAY_TRACE)
    idle = Trace.from_file(ROOT / IDLE_TRACE)

    simulated_readings = [simulated.read("die_temp_cpu") for _ in range(8)]
    live_readings = [clock.read("monotonic_clock") for _ in range(8)]
    results: list[tuple[int, str, str]] = []
    out("")
    out(f"idle fixture    : {idle.sample_count} samples, content "
        f"{idle.content_provenance.value}; replaying it stamps "
        f"{RecordedSource(ORIN_NANO, idle).provenance.value}")
    out(f"replay fixture  : {replayed.sample_count} samples, content "
        f"{replayed.content_provenance.value}; replaying it stamps "
        f"{RecordedSource(ORIN_NANO, replayed).provenance.value}")

    out.rule("1-4. the labelling itself")

    results.append(attempt(
        out, 1, "label a wholly simulated reading set MEASURED, directly",
        lambda: certify(
            "direct", attribute(simulated_readings), Label.MEASURED, PLATFORM
        ).headline,
    ))

    results.append(attempt(
        out, 2, "aggregate first, then label the statistic MEASURED",
        lambda: certify(
            "aggregated", summarise(simulated_readings)[1].attribution, Label.MEASURED,
            PLATFORM,
        ).headline,
    ))

    results.append(attempt(
        out, 3, "dilute: 999 live readings and one simulated one",
        lambda: certify(
            "diluted",
            attribute([*(clock.read("monotonic_clock") for _ in range(999)),
                       simulated.read("die_temp_cpu")]),
            Label.MEASURED, PLATFORM,
        ).headline,
    ))

    results.append(attempt(
        out, 4, "derive a second-order figure from a live and a simulated statistic",
        lambda: certify(
            "second order",
            derive(
                "headroom", 1.0, "degC",
                [summarise(live_readings)[1], summarise(simulated_readings)[1]],
            ).attribution,
            Label.MEASURED, PLATFORM,
        ).headline,
    ))

    out.rule("5-7. replay")

    results.append(attempt(
        out, 5, "replay a trace and label the replay MEASURED",
        lambda: certify(
            "replay",
            attribute(RecordedSource(ORIN_NANO, replayed).read_all(replayed.channels)),
            Label.MEASURED, PLATFORM,
        ).headline,
    ))

    def live_claiming_document() -> str:
        payload = json.loads((ROOT / REPLAY_TRACE).read_text())
        payload["content_provenance"] = "live"
        return Trace.from_mapping(payload).content_provenance.value

    results.append(attempt(
        out, 6, "edit a trace document to declare live content, then replay it",
        live_claiming_document,
    ))

    def round_trip_a_reading() -> str:
        document = simulated_readings[0].to_mapping()
        document["provenance"] = "live"
        rebuilt = from_mapping(document, floor=Provenance.SIMULATED)
        return certify(
            "round trip", attribute([rebuilt]), Label.MEASURED, PLATFORM
        ).headline

    results.append(attempt(
        out, 7, "serialise a simulated reading, edit it to live, deserialise it",
        round_trip_a_reading,
    ))

    out.rule("8-10. the value object")

    def mutate_a_field() -> str:
        reading = simulated.read("die_temp_cpu")
        object.__setattr__(reading, "provenance", Provenance.LIVE)
        return certify(
            "mutated", attribute([reading]), Label.MEASURED, PLATFORM
        ).headline

    results.append(attempt(
        out, 8, "rewrite provenance with object.__setattr__, past the frozen guard",
        mutate_a_field,
    ))

    def malicious_subclass() -> str:
        class Liar(Reading):
            def __post_init__(self, mint_token) -> None:  # noqa: ARG002
                object.__setattr__(self, "value", float(self.value))
                object.__setattr__(self, "seal", "forged")

            @property
            def intact(self) -> bool:
                return True

        liar = Liar(
            None, channel="die_temp_cpu", value=simulated_readings[0].value,
            unit="degC", timestamp_ns=1, provenance=Provenance.LIVE,
            source_id="liar",
        )
        return certify(
            "subclassed", attribute([liar]), Label.MEASURED, PLATFORM
        ).headline

    results.append(attempt(
        out, 9, "a Reading subclass overriding the seal check and the token",
        malicious_subclass,
    ))

    def replace_relabel() -> str:
        laundered = dataclasses.replace(
            simulated_readings[0], provenance=Provenance.LIVE
        )
        return certify(
            "replaced", attribute([laundered]), Label.MEASURED, PLATFORM
        ).headline

    results.append(attempt(
        out, 10, "dataclasses.replace(reading, provenance=LIVE) -- FOUND TO WORK",
        replace_relabel,
    ))

    out.rule("11-13. the backend")

    def overriding_backend() -> str:
        namespace = {
            "backend_provenance": Provenance.SIMULATED,
            "_raw": lambda self, channel: 1.0,
            "read": lambda self, name: mint(
                channel=name, value=1.0, unit="degC", timestamp_ns=1,
                provenance=Provenance.LIVE, source_id="overriding",
            ),
        }
        backend = type("Overriding", (Source,), namespace)(ORIN_NANO, "overriding")
        return certify(
            "overriding", attribute([backend.read("die_temp_cpu")]), Label.MEASURED,
            PLATFORM,
        ).headline

    results.append(attempt(
        out, 11, "a Source subclass overriding read() to stamp LIVE", overriding_backend,
    ))

    def sealed_subclass() -> str:
        class TraceAsLive(LiveSource):
            def _raw(self, channel):
                return 41_000.0

        return TraceAsLive(ORIN_NANO).read("die_temp_cpu").provenance.value

    results.append(attempt(
        out, 12, "subclass LiveSource and serve it trace values", sealed_subclass,
    ))

    def rebind_class_attribute() -> str:
        original = SimulatedSource.backend_provenance
        try:
            SimulatedSource.backend_provenance = Provenance.LIVE
            source = SimulatedSource(ORIN_NANO, seed=1)
            return certify(
                "rebound", attribute([source.read("die_temp_cpu")]), Label.MEASURED,
                PLATFORM,
            ).headline
        finally:
            SimulatedSource.backend_provenance = original

    results.append(attempt(
        out, 13, "rebind SimulatedSource.backend_provenance to LIVE",
        rebind_class_attribute,
    ))

    out.rule("14-15. the lattice and the certificate")

    def empty_fold() -> str:
        provenance = meet_all([])
        return certify(
            "empty", attribute([]), label_for(provenance), PLATFORM
        ).headline

    results.append(attempt(
        out, 14, "exploit the lattice identity: meet of an empty set is LIVE",
        empty_fold,
    ))

    def forge_a_certificate() -> str:
        return Certificate(
            object(), "forged", Label.MEASURED, attribute(simulated_readings), PLATFORM
        ).headline

    results.append(attempt(
        out, 15, "construct a Certificate directly, bypassing certify()",
        forge_a_certificate,
    ))

    out.rule("the positive control: the choke point does grant a label")
    out("A check that only ever observes refusals cannot distinguish a working")
    out("invariant from a function that raises unconditionally. These three must")
    out("SUCCEED, and a failure here is as much a defect as a successful attack.")
    out("")
    granted = []
    for provenance, readings in (
        (Provenance.SIMULATED, simulated_readings),
        (Provenance.RECORDED,
         RecordedSource(ORIN_NANO, replayed).read_all(replayed.channels)),
        (Provenance.LIVE, live_readings),
    ):
        attribution = attribute(readings)
        certificate = certify(
            f"positive control, {provenance.value}", attribution,
            label_for(provenance), PLATFORM,
        )
        granted.append([
            provenance.value, certificate.label.value,
            "yes" if certificate.is_hardware_measurement else "NO",
            certificate.headline[:46],
        ])
        out.check(
            f"the {provenance.value} label is GRANTED by the choke point",
            certificate.label is label_for(provenance)
            and certificate.provenance is provenance,
            certificate.label.value,
        )
    out("")
    out(format_table(
        ["provenance", "label granted", "hardware measurement", "headline (truncated)"],
        granted,
    ))
    out("")
    out("The MEASURED row is granted on a LIVE clock reading, and its headline")
    out("says MEASURED OFF-TARGET, because the platform attestation does not match")
    out("the declared NVIDIA Jetson Orin Nano. A bare MEASURED is unreachable on")
    out("this host by construction and not by discipline.")

    out.rule("the two routes that SUCCEEDED, and what was changed")
    out("Route 10, dataclasses.replace. Found on 2026-11-11. The exact attack:")
    out("")
    out("    laundered = dataclasses.replace(simulated_reading, provenance=LIVE)")
    out("    certify('figure', attribute([laundered]), Label.MEASURED, platform)")
    out("")
    out("`dataclasses.replace` is public API on every dataclass. It re-ran the")
    out("Reading's __post_init__, which RECOMPUTED THE SEAL, so the laundered")
    out("reading was validly sealed, carried the simulated reading's own channel")
    out("and value, and claimed LIVE. The choke point granted MEASURED. One line,")
    out("no private state touched. The seal did not help, because the seal was")
    out("recomputed by the very mechanism being abused.")
    out("")
    out("THE FIX WAS TO THE DESIGN. A Reading's first field is now a")
    out("dataclasses.InitVar holding a module-private minting token, so")
    out("`dataclasses.replace` and `copy.replace` both raise and the single")
    out("minting entry point is boardprobe.readings.mint(), whose only in-package")
    out("callers are Source.read() and from_mapping(). The route is pinned as a")
    out("regression test in tests/test_readings.py.")
    out("")
    out("Route 13, rebinding the class attribute. Found the same day. Provenance")
    out("was read from `type(self).backend_provenance`, which is an ordinary class")
    out("attribute and can be assigned to. THE FIX: the three backends' declared")
    out("provenances are captured in a module-private registry when")
    out("boardprobe.sources finishes importing, and the property compares the")
    out("class attribute against the registry on every read.")

    out.rule("what is NOT closed, named rather than omitted")
    out("Three things, and none of them is a route from an EXISTING simulated")
    out("reading to a measured label, which is what this check is about.")
    out("")
    out("  a. boardprobe.readings.mint() lets its caller choose a provenance. A")
    out("     caller who passes LIVE for a generated number has made a false")
    out("     statement at the source. No type system can prevent that; a")
    out("     hardware abstraction layer is a program that believes the kernel.")
    out("     What changed is that the false statement is now made in ONE named")
    out("     function, is attributable to a source_id that travels with the")
    out("     reading, and cannot be made by accident or by a refactor.")
    out("  b. A PlatformAttestation can be constructed by hand with")
    out("     matches_target=True. That is a false claim about WHICH machine, not")
    out("     about provenance, and every certificate records the attestation's")
    out("     own `source` string so the claim is legible in the output.")
    out("  c. Code running in this process can reach into module-private state")
    out("     (boardprobe.readings._KEY, boardprobe.sources._SEALED and")
    out("     _DECLARED) and defeat all of the above. This is true of every")
    out("     Python program and is stated rather than papered over.")

    refused = sum(1 for _, verdict, _ in results if verdict == "REFUSED")
    succeeded = [number for number, verdict, _ in results if verdict == "SUCCEEDED"]
    out.rule("summary")
    out(format_table(
        ["route", "verdict", "refused by"],
        [[str(number), verdict, detail[:84]] for number, verdict, detail in results],
    ))
    out("")
    out(f"routes attempted : {len(results)}")
    out(f"routes refused   : {refused}")
    out(f"routes succeeded : {len(succeeded)} {succeeded if succeeded else ''}")
    out("positive controls: 3 labels granted (synthetic, replayed, measured)")
    out("routes found to succeed during the build and then closed by a design")
    out("change: 2 (route 10, dataclasses.replace; route 13, class-attribute")
    out("rebinding)")
    out.check(
        "no route the public API exposes gets a non-live reading a MEASURED label",
        not succeeded,
        f"{refused} of {len(results)} refused",
    )
    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
