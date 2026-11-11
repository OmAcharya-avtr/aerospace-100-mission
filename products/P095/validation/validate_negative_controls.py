"""Negative controls, anchored on true values, each verified to fire or stay silent.

A control that does not fire is a failure of the control, not a success of the
product: it means the check it guards would not have caught the violation it
exists to catch. A product of this mission shipped two of eleven controls
silently too weak, and the certificate was right while the control was wrong,
which reads as evidence and is the worse of the two failures.

Every control here is anchored on a value known independently of the code under
test:

- the provenance a backend DECLARES, which is a class constant and not an
  output;
- a raw sysfs integer written by this script, so the correctly scaled value is
  known by hand before anything reads it: 48123 millidegrees Celsius is
  48.123 degrees Celsius, and 4870 millivolts is 4.870 volts;
- a declared physical plausibility window, stated below with its reasoning and
  NOT taken from a datasheet, because no datasheet was read on hardware and
  quoting one would be a citation this build cannot support.

Controls that must NOT fire are asserted in that direction too, because a
control that fires on correct input is equally wrong and would quietly
invalidate every must-fire result beside it.
"""

from __future__ import annotations

import sys  # noqa: E402

sys.dont_write_bytecode = True

import dataclasses  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

from _bootstrap import (  # noqa: E402
    REPLAY_TRACE,
    SEED_CONTROL,
    Tee,
    bootstrap,
    environment_lines,
)

ROOT = bootstrap()

from boardprobe import (  # noqa: E402
    ORIN_NANO,
    Channel,
    DeviceUnreachableError,
    HarnessError,
    Label,
    LabelViolation,
    LiveSource,
    Provenance,
    ProvenanceError,
    RecordedSource,
    SimulatedSource,
    TamperError,
    Trace,
    TraceError,
    TraceSample,
    attest_platform,
    attribute,
    benchmark,
    certify,
    format_table,
    rail_power,
    reference_payload,
)
from boardprobe.channels import BoardDeclaration  # noqa: E402

PLATFORM = attest_platform()

#: Declared plausibility windows. THESE ARE ENGINEERING WINDOWS DECLARED HERE,
#: not figures quoted from any datasheet: no NVIDIA document was read against
#: hardware for this build and a datasheet citation would be unsupported. The
#: reasoning is the whole justification and is written down so it can be
#: argued with.
#:
#: die temperature: a silicon die reading outside -40 to 125 degrees Celsius is
#: a scaling, wiring or driver error rather than a temperature. The window is
#: deliberately wider than any operating range a board would be used in, so
#: that it catches the factor-of-1000 class of error and nothing subtler.
#:
#: rail power: 0 to 60 watts spans every published Jetson Orin Nano power mode
#: with large margin. Again the target is the factor-of-1000 error, not a
#: thermal-design judgement.
WINDOWS = {
    "die temperature [degC]": (-40.0, 125.0),
    "rail power [W]": (0.0, 60.0),
}


def outside(window: str, value: float) -> bool:
    """``True`` when ``value`` lies outside the declared window for ``window``."""
    low, high = WINDOWS[window]
    return not (low <= value <= high)


def write_node(root: Path, locator: str, text: str) -> None:
    """Write a fake sysfs node under ``root`` so a live read has something to find."""
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:  # noqa: C901, PLR0915 - one block per control, read top to bottom
    out = Tee("validate_negative_controls_output.txt")
    out.rule("boardprobe -- negative controls, anchored on true values")
    for line in environment_lines():
        out(line)
    out("")
    out(format_table(
        ["declared plausibility window", "low", "high"],
        [[name, f"{low:g}", f"{high:g}"] for name, (low, high) in WINDOWS.items()],
    ))
    out("")
    out("THOSE WINDOWS ARE DECLARED HERE AND ARE NOT QUOTED FROM ANY DATASHEET.")
    out("No NVIDIA document was read against hardware for this build. They are")
    out("deliberately wide: they catch the factor-of-1000 scaling class of error")
    out("and nothing subtler, which is exactly what they are for.")

    simulated = SimulatedSource(ORIN_NANO, seed=SEED_CONTROL)
    clock = LiveSource(ORIN_NANO)
    replayed = Trace.from_file(ROOT / REPLAY_TRACE)
    recorded = RecordedSource(ORIN_NANO, replayed)

    simulated_readings = [simulated.read("die_temp_cpu") for _ in range(6)]
    recorded_readings = list(recorded.read_all(replayed.channels))
    live_readings = [clock.read("monotonic_clock") for _ in range(6)]

    out.rule("A. must FIRE: the label refusals")

    def label_fires(readings, label) -> bool:
        try:
            certify("control", attribute(readings), label, PLATFORM)
        except LabelViolation:
            return True
        return False

    out.control(
        "a wholly simulated reading set labelled MEASURED",
        label_fires(simulated_readings, Label.MEASURED), True,
        "true provenance is simulated, declared by SimulatedSource",
    )
    out.control(
        "a replayed reading set labelled MEASURED",
        label_fires(recorded_readings, Label.MEASURED), True,
        "true provenance is recorded, declared by RecordedSource",
    )
    out.control(
        "five live readings and one simulated one labelled MEASURED",
        label_fires([*live_readings[:5], simulated.read("die_temp_cpu")],
                    Label.MEASURED), True,
        "the meet of the true provenances is simulated",
    )
    out.control(
        "a simulated reading set labelled REPLAYED",
        label_fires(simulated_readings, Label.REPLAYED), True,
        "simulated admits SYNTHETIC only",
    )
    out.control(
        "a live reading set labelled SYNTHETIC",
        label_fires(live_readings, Label.SYNTHETIC), True,
        "the label must follow the provenance in BOTH directions",
    )

    out.rule("B. must FIRE: the integrity refusals")

    def tamper_fires() -> bool:
        reading = simulated.read("die_temp_cpu")
        object.__setattr__(reading, "provenance", Provenance.LIVE)
        try:
            attribute([reading])
        except TamperError:
            return True
        return False

    def replace_fires() -> bool:
        try:
            dataclasses.replace(simulated_readings[0], provenance=Provenance.LIVE)
        except TypeError:
            return True
        return False

    def empty_fires() -> bool:
        try:
            attribute([])
        except ProvenanceError:
            return True
        return False

    def live_document_fires() -> bool:
        payload = json.loads((ROOT / REPLAY_TRACE).read_text())
        payload["content_provenance"] = "live"
        try:
            Trace.from_mapping(payload)
        except TraceError:
            return True
        return False

    out.control(
        "provenance rewritten past the frozen guard with object.__setattr__",
        tamper_fires(), True, "the seal is recomputed and compared",
    )
    out.control(
        "dataclasses.replace(reading, provenance=LIVE)",
        replace_fires(), True,
        "the minting token; this route SUCCEEDED before the design was changed",
    )
    out.control(
        "an aggregate over an empty reading set",
        empty_fires(), True, "the lattice identity is live and is refused",
    )
    out.control(
        "a trace document edited to declare live content",
        live_document_fires(), True, "a replay is not a live reading",
    )

    out.rule("C. must FIRE: the scaling controls, anchored on a known raw integer")
    out("This script writes the raw integers itself, so the correct value is known")
    out("by hand before anything reads it. 48123 millidegrees Celsius is 48.123")
    out("degrees Celsius; 4870 millivolts is 4.870 volts; 4860 milliamperes is")
    out("4.860 amperes, giving 23.6682 watts.")
    out("")
    scratch = Path(tempfile.mkdtemp(prefix="boardprobe-controls-"))
    thermal_locator = ORIN_NANO.channel("die_temp_cpu").locator
    write_node(scratch, thermal_locator, "48123\n")
    write_node(
        scratch,
        "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/in1_input",
        "4870\n",
    )
    write_node(
        scratch,
        "sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon3/curr1_input",
        "4860\n",
    )

    correct = LiveSource(ORIN_NANO, root=scratch).read("die_temp_cpu")
    out(f"  correctly scaled die temperature : {correct.value:.6f} {correct.unit}")

    # A board table with the thermal scale corrupted to 1.0: the millidegree
    # convention dropped. The true value is 48.123 degC, known from the raw
    # integer this script wrote, so the corrupted reading is 48123 degC and
    # the declared window must reject it.
    corrupted_table = BoardDeclaration(
        model=ORIN_NANO.model,
        notes="a deliberately corrupted table for a negative control",
        channels=tuple(
            dataclasses.replace(channel, scale=1.0)
            if channel.name == "die_temp_cpu"
            else channel
            for channel in ORIN_NANO.channels
        ),
    )
    corrupted = LiveSource(corrupted_table, root=scratch).read("die_temp_cpu")
    out(f"  with the millidegree scale dropped: {corrupted.value:.6f} "
        f"{corrupted.unit}")
    out(f"  ratio to the true value           : "
        f"{corrupted.value / correct.value:.1f}")
    out.control(
        "the millidegree scale dropped from the thermal channel",
        outside("die temperature [degC]", corrupted.value), True,
        f"{corrupted.value:.1f} degC against a window of "
        f"{WINDOWS['die temperature [degC]']}",
    )

    voltage = LiveSource(ORIN_NANO, root=scratch).read("rail_vdd_in_voltage")
    current = LiveSource(ORIN_NANO, root=scratch).read("rail_vdd_in_current")
    true_power = rail_power(voltage, current, "rail_vdd_in_power")
    out("")
    out(f"  correctly scaled rail power      : {true_power.value:.6f} "
        f"{true_power.unit}")

    millivolt_table = BoardDeclaration(
        model=ORIN_NANO.model,
        notes="a deliberately corrupted table for a negative control",
        channels=tuple(
            Channel(
                name=channel.name, kind=channel.kind, unit="V",
                access=channel.access, locator=channel.locator, scale=1.0,
                raw_unit="mV", reference=channel.reference,
            )
            if channel.name == "rail_vdd_in_voltage"
            else channel
            for channel in ORIN_NANO.channels
        ),
    )
    bad_voltage = LiveSource(millivolt_table, root=scratch).read("rail_vdd_in_voltage")
    bad_power = rail_power(bad_voltage, current, "rail_vdd_in_power")
    out(f"  with millivolts labelled as volts: {bad_power.value:.6f} "
        f"{bad_power.unit}")
    out(f"  ratio to the true value          : "
        f"{bad_power.value / true_power.value:.1f}")
    out.control(
        "millivolts labelled as volts in the rail-power derivation",
        outside("rail power [W]", bad_power.value), True,
        f"{bad_power.value:.1f} W against a window of {WINDOWS['rail power [W]']}",
    )

    def unit_mismatch_fires() -> bool:
        trace = Trace(
            board_model=ORIN_NANO.model,
            origin="a deliberately corrupted trace for a negative control",
            content_provenance=Provenance.RECORDED,
            channels=("die_temp_cpu",),
            samples=(TraceSample("die_temp_cpu", 48.123, "K", 0),),
        )
        try:
            RecordedSource(ORIN_NANO, trace).read("die_temp_cpu")
        except DeviceUnreachableError:
            return True
        return False

    out.control(
        "a trace recording kelvin where the board table declares degrees Celsius",
        unit_mismatch_fires(), True, "a unit mismatch is a wrong number",
    )

    def backwards_clock_fires() -> bool:
        trace = Trace(
            board_model=ORIN_NANO.model,
            origin="a deliberately decreasing clock for a negative control",
            content_provenance=Provenance.RECORDED,
            channels=("monotonic_clock",),
            samples=tuple(
                TraceSample("monotonic_clock", 10.0 - index, "s", index)
                for index in range(8)
            ),
        )
        try:
            benchmark(
                reference_payload(10), "backwards", RecordedSource(ORIN_NANO, trace),
                repeats=3, warmup=0,
            )
        except HarnessError:
            return True
        return False

    out.control(
        "a clock channel whose values decrease during a benchmark",
        backwards_clock_fires(), True,
        "a monotonic clock that decreases means the channel is not monotonic",
    )

    out.rule("D. must NOT fire: the same checks on correct input")
    out("A control that fires on correct input is as wrong as one that fails to")
    out("fire on corrupt input, and would quietly invalidate every result above.")
    out("")
    out.control(
        "a wholly live reading set labelled MEASURED",
        label_fires(live_readings, Label.MEASURED), False,
        "live admits MEASURED; the headline says OFF-TARGET, which is a separate fact",
    )
    out.control(
        "a replayed reading set labelled REPLAYED",
        label_fires(recorded_readings, Label.REPLAYED), False,
        "recorded admits REPLAYED",
    )
    out.control(
        "a simulated reading set labelled SYNTHETIC",
        label_fires(simulated_readings, Label.SYNTHETIC), False,
        "simulated admits SYNTHETIC",
    )
    out.control(
        "an untouched reading passed to attribute()",
        (lambda: not _survives(simulated.read("die_temp_cpu")))(), False,
        "the seal agrees with the fields",
    )
    out.control(
        "the correctly scaled die temperature against the declared window",
        outside("die temperature [degC]", correct.value), False,
        f"{correct.value:.3f} degC, inside {WINDOWS['die temperature [degC]']}",
    )
    out.control(
        "the correctly scaled rail power against the declared window",
        outside("rail power [W]", true_power.value), False,
        f"{true_power.value:.4f} W, inside {WINDOWS['rail power [W]']}",
    )

    def increasing_clock_fires() -> bool:
        trace = Trace(
            board_model=ORIN_NANO.model,
            origin="an increasing clock, the correct-input half of the control",
            content_provenance=Provenance.RECORDED,
            channels=("monotonic_clock",),
            samples=tuple(
                TraceSample("monotonic_clock", 10.0 + index, "s", index)
                for index in range(8)
            ),
        )
        try:
            benchmark(
                reference_payload(10), "forwards", RecordedSource(ORIN_NANO, trace),
                repeats=3, warmup=0,
            )
        except HarnessError:
            return True
        return False

    out.control(
        "a clock channel whose values increase during a benchmark",
        increasing_clock_fires(), False, "a monotonic clock that increases is fine",
    )
    out.control(
        "a trace document declaring recorded content",
        (lambda: _trace_refused(ROOT / REPLAY_TRACE))(), False,
        "recorded and simulated are the two provenances a document may declare",
    )

    out.rule("summary")
    out(f"controls declared              : {out.controls_expected}")
    out(f"controls that fired            : {out.controls_fired}")
    out(f"must-fire, fired               : {out.must_fire_fired} of {out.must_fire}")
    out(f"must-NOT-fire, stayed silent   : {out.must_not_fire_silent} of "
        f"{out.must_not_fire}")
    out("")
    out("No control's criterion was weakened and no seed was changed to make a")
    out("control behave. The declared windows are stated at the top of this output")
    out("and are wider than any operating range, so a control that fires against")
    out("one of them is firing on a factor-of-1000 error and not on a borderline")
    out("reading.")
    out.check(
        "every control behaved as declared",
        out.must_fire_fired == out.must_fire
        and out.must_not_fire_silent == out.must_not_fire,
        f"{out.must_fire_fired}/{out.must_fire} fired, "
        f"{out.must_not_fire_silent}/{out.must_not_fire} silent",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


def _survives(reading) -> bool:
    """``True`` when ``attribute`` accepts ``reading``."""
    try:
        attribute([reading])
    except TamperError:
        return False
    return True


def _trace_refused(path: Path) -> bool:
    """``True`` when loading the document at ``path`` raises."""
    try:
        Trace.from_file(path)
    except TraceError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
