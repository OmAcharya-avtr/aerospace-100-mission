"""The worked example quoted in README.md, run so that the output is the real one.

Two small dictionaries are written out by hand rather than generated, so that
every change in them is visible in the source of this script. Version 1.0 has
four telemetry parameters and one command; version 1.1 keeps three of
those mnemonics and changes each of them in a way that keeps the name,
renames the fourth, adds a fifth, and swaps the positions of the command's two
arguments while keeping both argument names.

Nothing in this example is from a real spacecraft.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff import (  # noqa: E402
    Command,
    CommandArgument,
    Corpus,
    CorpusSource,
    Dictionary,
    Parameter,
    assess_impact,
    diff_dictionaries,
    render_diff,
    render_impact,
)

OLD = Dictionary(
    version="1.0",
    parameters=(
        Parameter(
            name="BATT_T_1",
            path="TM/PWR/BATT_T_1",
            description="battery one temperature",
            data_type="int",
            bit_offset=0,
            bit_width=12,
            scale=0.0625,
            unit="K",
            low_limit=250.0,
            high_limit=330.0,
            packet="PWR_HK",
        ),
        Parameter(
            name="BUS_V",
            path="TM/PWR/BUS_V",
            description="main bus voltage",
            data_type="uint",
            bit_offset=12,
            bit_width=16,
            scale=0.001,
            unit="V",
            packet="PWR_HK",
        ),
        Parameter(
            name="AOCS_MODE",
            path="TM/AOCS/AOCS_MODE",
            description="attitude control mode",
            data_type="enum",
            bit_offset=28,
            bit_width=4,
            enum=((0, "IDLE"), (1, "SAFE_MODE"), (2, "NOMINAL_MODE")),
            packet="AOCS_HK",
        ),
        Parameter(
            name="PAYL_CNT",
            path="TM/PAYL/PAYL_CNT",
            description="payload frame counter",
            data_type="uint",
            bit_offset=32,
            bit_width=32,
            unit="count",
            packet="PAYLOAD_HK",
        ),
    ),
    commands=(
        Command(
            name="SET_HEATER",
            path="TC/PWR/SET_HEATER",
            description="set a heater duty cycle",
            arguments=(
                CommandArgument(name="HEATER_ID", position=0, bit_width=8),
                CommandArgument(name="DUTY", position=1, bit_width=8, unit="percent"),
            ),
        ),
    ),
)

NEW = Dictionary(
    version="1.1",
    parameters=(
        # Same name, same bits, different calibration and unit: silent.
        OLD.parameter_map["BATT_T_1"].with_fields(
            scale=0.0625, offset=-273.15, unit="degC", low_limit=-23.0, high_limit=57.0
        ),
        # Same name, two bits further along: silent.
        OLD.parameter_map["BUS_V"].with_fields(bit_offset=14),
        # Same name, two codes swapped: silent.
        OLD.parameter_map["AOCS_MODE"].with_fields(
            enum=((0, "IDLE"), (1, "NOMINAL_MODE"), (2, "SAFE_MODE"))
        ),
        # Renamed, structurally identical: loud, with a suggested replacement.
        OLD.parameter_map["PAYL_CNT"].with_fields(
            name="PAYLOAD_FRAME_COUNT", path="TM/PAYL/PAYLOAD_FRAME_COUNT"
        ),
        # New parameter: advisory.
        Parameter(
            name="BATT_I_1",
            path="TM/PWR/BATT_I_1",
            description="battery one current",
            data_type="int",
            bit_offset=64,
            bit_width=14,
            scale=0.0005,
            unit="A",
            packet="PWR_HK",
        ),
    ),
    commands=(
        # Argument positions swapped, names unchanged: silent, and the most
        # dangerous entry in this example. SET_HEATER(HEATER_ID=3, DUTY=50)
        # still parses and now encodes 50 into the heater identifier.
        Command(
            name="SET_HEATER",
            path="TC/PWR/SET_HEATER",
            description="set a heater duty cycle",
            arguments=(
                CommandArgument(name="DUTY", position=0, bit_width=8, unit="percent"),
                CommandArgument(name="HEATER_ID", position=1, bit_width=8),
            ),
        ),
    ),
)

PROCEDURE = """\
* battery survival check, written against dictionary 1.0
CHECK BATT_T_1 > 253.0
CHECK BUS_V > 26.0
WAIT UNTIL AOCS_MODE == "SAFE_MODE"
SEND SET_HEATER(HEATER_ID=3, DUTY=50)
"""

SCRIPT = """\
# trend analysis, written against dictionary 1.0
from client import tm
battery = tm["BATT_T_1"]
counter = tm["PAYL_CNT"]
for index in range(4):
    panel = tm["PANEL_T_%d" % index]
"""


def main() -> int:
    out = Tee("worked_example_output.txt")
    out.rule("tmtcdiff -- worked example")
    for line in environment_lines():
        out(line)
    out("")
    out(f"old dictionary : version {OLD.version}, {len(OLD.parameters)} parameters, "
        f"{len(OLD.commands)} command")
    out(f"new dictionary : version {NEW.version}, {len(NEW.parameters)} parameters, "
        f"{len(NEW.commands)} command")

    result = diff_dictionaries(OLD, NEW)
    out.rule("the semantic diff")
    out(render_diff(result))

    corpus = Corpus(
        (
            CorpusSource(label="procedures/battery.proc", kind="procedure", text=PROCEDURE),
            CorpusSource(label="analysis/trend.py", kind="script", text=SCRIPT),
        )
    )
    report = assess_impact(result, corpus, OLD, NEW)
    out.rule("the impact report")
    out(render_impact(report))

    out.rule("the three numbers a reader should take from this example")
    out(f"silent changes       : {len(result.silent)}")
    out(f"loud changes         : {len(result.loud)}")
    out(f"silently changed refs: {len(report.silently_changed)}")
    out(f"coverage of the dictionary by this corpus: "
        f"{report.verdict.coverage_fraction:.4f}")
    out(f"assessment fully determinable: {report.verdict.determinable}")

    out.check(
        "the calibration change on BATT_T_1 is reported as silent",
        any(
            change.name == "BATT_T_1" and change.is_silent for change in result.changes
        ),
    )
    out.check(
        "the swapped command argument positions are reported as silent",
        any(
            change.change_class.value == "argument_reordered" and change.is_silent
            for change in result.changes
        ),
    )
    out.check(
        "the renamed counter is reported with its replacement name",
        result.rename_map.get("PAYL_CNT") == "PAYLOAD_FRAME_COUNT",
        str(result.rename_map),
    )
    out.check(
        "the run-time-constructed reference is reported as undecidable, not guessed",
        len(report.undecidable) == 1,
        f"{len(report.undecidable)} undecidable findings",
    )
    out.check(
        "the assessment is reported as not fully determinable",
        not report.verdict.determinable,
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
