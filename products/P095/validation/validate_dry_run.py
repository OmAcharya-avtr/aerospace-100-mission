"""Dry-run mode, and the evidence that the dry run was dry.

A dry run that asserts it touched no board is worth as much as the mechanism
behind the assertion, which is why the assertion here is a measurement. Every
device access in this package goes through ``boardprobe.sysio``, which appends
to a ledger; the dry run runs inside a fresh ledger window and this script
reports the ledger's contents.

The script also runs the dry run TWICE from the same tree and reports whether
the second run agrees with the first. A check whose own execution can create
the condition it tests is not a check, and this mission has shipped that
defect five times; running it twice and comparing is the cheap way to find out.
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
)

ROOT = bootstrap()

from boardprobe import ORIN_NANO, Access, Trace, dry_run, format_table  # noqa: E402

BOARD_CHANNELS = tuple(
    channel.name
    for channel in ORIN_NANO.channels
    if channel.access is not Access.CLOCK_SYSCALL
)


def main() -> int:
    out = Tee("validate_dry_run_output.txt")
    out.rule("boardprobe -- the dry run, and the evidence that it stayed dry")
    for line in environment_lines():
        out(line)

    idle = Trace.from_file(ROOT / IDLE_TRACE)
    replay = Trace.from_file(ROOT / REPLAY_TRACE)

    out.rule("1. the walk, with the repository root watched")
    first = dry_run(ORIN_NANO, idle, seed=SEED_CAMPAIGN, watch=ROOT)
    out(format_table(
        ["#", "code path exercised"],
        [[str(index + 1), name] for index, name in enumerate(first.paths)],
    ))
    out("")
    out(f"code paths exercised                        : {first.path_count}")
    out(f"device accesses against the real root       : {len(first.device_accesses)}")
    out(f"attempts against a root asserted not to exist: "
        f"{len(first.absent_root_accesses)}")
    out(f"files under the watched tree, before        : {len(first.files_before)}")
    out(f"files under the watched tree, after         : {len(first.files_after)}")
    out(f"files added                                 : "
        f"{list(first.files_added) or 'none'}")
    out(f"watched directory                           : {first.watched}")
    out(f"absent root used for the refusal leg        : {first.absent_root}")
    out("")
    out.check(
        "the dry run makes no device access against the real filesystem root",
        first.device_accesses == (),
        f"{len(first.device_accesses)} accesses",
    )
    out.check(
        "the dry run adds no file to the watched tree",
        first.files_added == (),
        f"{len(first.files_added)} added",
    )
    out.check(
        "the dry run reports itself side-effect free",
        first.side_effect_free,
    )
    out.check(
        "every code path name is distinct, so the count is a count and not a "
        "repetition",
        len(set(first.paths)) == first.path_count,
        f"{first.path_count} paths",
    )

    out.rule("2. the live refusal leg, counted separately and not hidden")
    out("The one part of the walk that calls into the device funnel is the live")
    out("backend's refusal path. It runs inside a NESTED ledger window against a")
    out("root the dry run first asserts does not exist, and its attempts are")
    out("reported in their own count. Folding them into the main ledger would")
    out("make the main count non-zero for accesses that provably cannot reach a")
    out("device; omitting them would be a self-reporting scanner. Both numbers")
    out("are printed.")
    out("")
    out(format_table(
        ["kind", "locator attempted (relative)"],
        [[kind, target[:92]] for kind, target in first.absent_root_accesses],
    ))
    out("")
    out.check(
        "the refusal leg attempts exactly one access per board channel",
        len(first.absent_root_accesses) == len(BOARD_CHANNELS),
        f"{len(first.absent_root_accesses)} of {len(BOARD_CHANNELS)}",
    )
    out.check(
        "every attempted locator is relative",
        all(not target.startswith("/") for _, target in first.absent_root_accesses),
        f"{len(first.absent_root_accesses)} locators",
    )

    out.rule("3. the labels the walk was granted and refused")
    out(format_table(
        ["granted"], [[entry] for entry in first.labels_granted]
    ))
    out("")
    out(format_table(
        ["refused"], [[entry] for entry in first.labels_refused]
    ))
    out("")
    out.check(
        "the walk exercises the choke point in both directions",
        bool(first.labels_granted) and len(first.labels_refused) >= 3,
        f"{len(first.labels_granted)} granted, {len(first.labels_refused)} refused",
    )

    out.rule("4. the same walk over the recorded fixture")
    replayed = dry_run(ORIN_NANO, replay, seed=SEED_CAMPAIGN, watch=ROOT)
    out(f"  labels granted : {'; '.join(replayed.labels_granted)}")
    out(f"  side-effect free : {'yes' if replayed.side_effect_free else 'NO'}")
    out.check(
        "replaying the recorded fixture in the walk yields a REPLAYED label",
        any("REPLAYED" in entry for entry in replayed.labels_granted),
        "; ".join(replayed.labels_granted),
    )

    out.rule("5. run twice from the same tree, and compare")
    out("A check whose own execution can create the condition it tests is not a")
    out("check. This mission has shipped that defect five times, so the walk is")
    out("run again and the two runs are compared.")
    out("")
    second = dry_run(ORIN_NANO, idle, seed=SEED_CAMPAIGN, watch=ROOT)
    out(format_table(
        ["quantity", "run 1", "run 2", "agree"],
        [
            ["code paths", str(first.path_count), str(second.path_count),
             "yes" if first.paths == second.paths else "NO"],
            ["device accesses", str(len(first.device_accesses)),
             str(len(second.device_accesses)),
             "yes" if first.device_accesses == second.device_accesses else "NO"],
            ["absent-root attempts", str(len(first.absent_root_accesses)),
             str(len(second.absent_root_accesses)),
             "yes" if first.absent_root_accesses == second.absent_root_accesses
             else "NO"],
            ["files added", str(len(first.files_added)), str(len(second.files_added)),
             "yes" if first.files_added == second.files_added else "NO"],
            ["files in tree", str(len(first.files_after)), str(len(second.files_after)),
             "yes" if len(first.files_after) == len(second.files_after) else "NO"],
        ],
    ))
    out("")
    out.check(
        "the second run agrees with the first on every reported quantity",
        first.paths == second.paths
        and first.device_accesses == second.device_accesses
        and first.absent_root_accesses == second.absent_root_accesses
        and first.files_added == second.files_added == ()
        and len(first.files_after) == len(second.files_after),
        "paths, accesses, attempts, files added and file count all equal",
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
