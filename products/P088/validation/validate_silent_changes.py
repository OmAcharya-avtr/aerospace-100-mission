"""The silent tier: measured separately, reported first, and ranked above loud breakage.

This is the script for the claim the product rests on. Four things are
measured.

1. **Ranking.** In every generated pair carrying both tiers, every silent
   change precedes every loud change in the ranked diff, and the rendered
   report puts the silent section first. Checked on the ordering, not asserted
   from the sort key.

2. **Silence itself.** The tier assignment claims that a silent change leaves
   an existing by-name reference resolving. That is testable: for every
   injected silent change the old name must still resolve in the new version,
   and for every loud change whose mechanism is resolution failure the old
   reference must stop resolving. Both directions are checked.

3. **The one tier assignment this tool cannot verify.** ``argument_added`` is
   classified loud on the assumption that a command system rejects a call
   missing an argument. The command still resolves and the argument list still
   parses, so nothing in this package confirms that assumption. It is reported
   here as an assumption, with the measurement that shows why.

4. **Silent-tier detection in the presence of the loud tier**, which is the
   regime a user is in. Per-class recall and precision for the silent classes
   with every loud class also injected into the same pair.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff.changes import ChangeClass, Severity  # noqa: E402
from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.metrics import aggregate, per_class_metrics  # noqa: E402
from tmtcdiff.report import format_table, render_diff  # noqa: E402
from tmtcdiff.synth import (  # noqa: E402
    INJECTABLE,
    INJECTABLE_SILENT,
    baseline_dictionary,
    change_pair,
    inject_changes,
)

SIZES: tuple[int, ...] = (40, 160)
SEEDS: tuple[int, ...] = tuple(6300 + 13 * index for index in range(10))

#: Loud classes whose mechanism is that the old reference stops resolving.
#: ``path_moved`` is checked on the path rather than on the name, because that
#: is the reference form it breaks. ``argument_removed`` and ``argument_added``
#: are deliberately absent: the command itself still resolves and the breakage
#: is at the argument level, which a resolution check does not model. See the
#: module docstring for what that means for ``argument_added``.
RESOLUTION_BREAKING: tuple[ChangeClass, ...] = (
    ChangeClass.REMOVED,
    ChangeClass.RENAMED,
    ChangeClass.COMMAND_REMOVED,
    ChangeClass.COMMAND_RENAMED,
    ChangeClass.PATH_MOVED,
)


def main() -> int:
    out = Tee("validate_silent_changes_output.txt")
    out.rule("tmtcdiff -- the silent tier, measured separately and reported first")
    for line in environment_lines():
        out(line)
    out("")
    out(f"silent classes : {len(INJECTABLE_SILENT)}")
    out(f"sizes          : {', '.join(str(size) for size in SIZES)}")
    out(f"seeds          : {len(SEEDS)} per size")

    config = DiffConfig()

    # --- 1. the silent tier is detected, in the presence of the loud tier --
    out.rule("silent-tier detection with every loud class also present in the pair")
    truth: list[tuple[ChangeClass, str]] = []
    detected: list[tuple[ChangeClass, str]] = []
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, config)
            truth.extend(pair.truth_keys)
            # Keys are deduplicated per pair, the same convention
            # validate_change_classes.py uses. A class detected on one item
            # through two fields counts once: argument_reordered always moves
            # two arguments, so counting the raw changes would charge it a
            # false positive for every true detection and report a precision
            # of 0.5000 that is an artefact of the counting rule rather than
            # a property of the detector.
            detected.extend({(c.change_class, c.name) for c in result.changes})
    metrics = per_class_metrics(truth, detected)
    rows = []
    for change_class in INJECTABLE_SILENT:
        entry = metrics.get(change_class)
        if entry is None:
            continue
        rows.append(
            [
                change_class.value,
                entry.injected,
                entry.detections,
                entry.true_positives,
                entry.false_positives,
                entry.false_negatives,
                f"{entry.precision:.4f}",
                f"{entry.recall:.4f}",
            ]
        )
    out(
        format_table(
            ["silent class", "injected", "detected", "TP", "FP", "FN", "precision", "recall"],
            rows,
        )
    )
    silent_total = aggregate(
        [metrics[c] for c in INJECTABLE_SILENT if c in metrics]
    )
    loud_total = aggregate(
        [metrics[c] for c in INJECTABLE if c.severity is Severity.LOUD and c in metrics]
    )
    out("")
    out(
        format_table
        (
            ["tier", "injected", "TP", "FP", "FN", "precision", "recall"],
            [
                [
                    "silent",
                    silent_total.injected,
                    silent_total.true_positives,
                    silent_total.false_positives,
                    silent_total.false_negatives,
                    f"{silent_total.precision:.4f}",
                    f"{silent_total.recall:.4f}",
                ],
                [
                    "loud",
                    loud_total.injected,
                    loud_total.true_positives,
                    loud_total.false_positives,
                    loud_total.false_negatives,
                    f"{loud_total.precision:.4f}",
                    f"{loud_total.recall:.4f}",
                ],
            ],
        )
    )
    out("")
    out("Where the silent tier measures above the loud tier in this regime, the reason")
    out("is structural rather than flattering: a silent change is read off two records")
    out("that share a key, while a loud change in the presence of unrelated additions")
    out("can be absorbed into an inferred rename. The two tiers are printed side by")
    out("side above so a reader can see which way each metric actually went.")
    out.check(
        "silent-tier recall is 1.0000 with the loud tier present",
        silent_total.recall == 1.0,
        f"recall {silent_total.recall:.4f}, {silent_total.false_negatives} missed",
    )

    # --- 2. ranking ------------------------------------------------------
    out.rule("ranking -- every silent change precedes every loud change")
    violations = 0
    render_violations = 0
    pairs_with_both = 0
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, config)
            ranks = [change.severity.rank for change in result.changes]
            if ranks != sorted(ranks):
                violations += 1
            if result.silent and result.loud:
                pairs_with_both += 1
                last_silent = max(
                    index
                    for index, change in enumerate(result.changes)
                    if change.severity is Severity.SILENT
                )
                first_loud = min(
                    index
                    for index, change in enumerate(result.changes)
                    if change.severity is Severity.LOUD
                )
                if last_silent > first_loud:
                    violations += 1
                text = render_diff(result)
                if text.index("-- silent (") > text.index("-- loud ("):
                    render_violations += 1
    out(f"pairs checked              : {len(SIZES) * len(SEEDS)}")
    out(f"pairs carrying both tiers  : {pairs_with_both}")
    out(f"ordering violations        : {violations}")
    out(f"rendered-order violations  : {render_violations}")
    out.check("the ranked diff puts every silent change before every loud change", violations == 0)
    out.check("the rendered report puts the silent section first", render_violations == 0)

    # --- 3. silence itself ------------------------------------------------
    out.rule("silence is testable -- does the old reference still resolve?")
    out("A silent change must leave an existing by-name reference resolving; a loud")
    out("change whose mechanism is resolution failure must stop it resolving. Both")
    out("directions are measured below over isolated injections.")
    out("")
    rows = []
    silent_resolving = 0
    silent_total_count = 0
    loud_breaking = 0
    loud_total_count = 0
    for change_class in INJECTABLE:
        if change_class.severity is Severity.ADVISORY:
            continue
        resolves = 0
        checked = 0
        for size in SIZES:
            for seed in SEEDS:
                old = baseline_dictionary(size, seed)
                pair = inject_changes(old, (change_class,), seed + 1, per_class=1)
                for injected in pair.truth:
                    checked += 1
                    if change_class is ChangeClass.PATH_MOVED:
                        item = pair.old.parameter_map.get(
                            injected.name
                        ) or pair.old.command_map.get(injected.name)
                        target = item.path if item is not None else injected.name
                    elif change_class is ChangeClass.ARGUMENT_REMOVED:
                        target = injected.name
                    else:
                        target = injected.name
                    if pair.new.resolve(target) is not None:
                        resolves += 1
        expectation = "resolves" if change_class.is_silent else "stops resolving"
        if change_class.is_silent:
            silent_resolving += resolves
            silent_total_count += checked
            verdict = "as expected" if resolves == checked else "UNEXPECTED"
        elif change_class in RESOLUTION_BREAKING:
            loud_breaking += checked - resolves
            loud_total_count += checked
            verdict = "as expected" if resolves == 0 else "UNEXPECTED"
        else:
            verdict = "command still resolves; breakage is at the argument level"
        rows.append(
            [
                change_class.value,
                change_class.severity.value,
                expectation,
                f"{resolves}/{checked}",
                verdict,
            ]
        )
    out(
        format_table(
            ["class", "tier", "expected", "old reference still resolves", "verdict"], rows
        )
    )
    out.check(
        "every silent change leaves the old by-name reference resolving",
        silent_resolving == silent_total_count,
        f"{silent_resolving} of {silent_total_count}",
    )
    out.check(
        "every resolution-breaking loud change stops the old reference resolving",
        loud_breaking == loud_total_count,
        f"{loud_breaking} of {loud_total_count}",
    )

    # --- 4. the tier assignment this tool cannot verify -------------------
    out.rule("the one tier assignment this package cannot verify")
    out("argument_added is classified LOUD on the assumption that a command system")
    out("rejects a call that omits an argument. Measured below: after the injection the")
    out("command still resolves and every argument an existing call passes is still")
    out("present, so nothing in this package confirms the call fails. If the uplink")
    out("system defaults a missing argument instead of rejecting the call, this class")
    out("is SILENT and is ranked one tier too low by this taxonomy.")
    out("")
    still_resolving = 0
    args_still_present = 0
    checked = 0
    for size in SIZES:
        for seed in SEEDS:
            old = baseline_dictionary(size, seed)
            pair = inject_changes(old, (ChangeClass.ARGUMENT_ADDED,), seed + 1, per_class=1)
            for injected in pair.truth:
                checked += 1
                if pair.new.resolve(injected.name) is not None:
                    still_resolving += 1
                old_command = pair.old.command_map[injected.name]
                new_command = pair.new.command_map[injected.name]
                if set(old_command.argument_map) <= set(new_command.argument_map):
                    args_still_present += 1
    out(f"injections checked                              : {checked}")
    out(f"command still resolves after the injection     : {still_resolving}")
    out(f"every pre-existing argument still present      : {args_still_present}")
    out.check(
        "argument_added leaves the command resolving with its old arguments intact",
        still_resolving == checked and args_still_present == checked,
        "so its LOUD tier is an assumption about the command system, not a measurement",
    )

    # --- 5. the enum-label removal case ----------------------------------
    out.rule("why a removed enumeration label is silent rather than loud")
    old = baseline_dictionary(40, SEEDS[0])
    target = next(p for p in old.parameters if p.enum)
    trimmed = target.with_fields(enum=target.enum[:-1])
    removed_label = target.enum[-1][1]
    from tmtcdiff.model import Dictionary

    new = Dictionary(
        version="1.1",
        parameters=tuple(trimmed if p.name == target.name else p for p in old.parameters),
        commands=old.commands,
    )
    result = diff_dictionaries(old, new, config)
    change = next(c for c in result.changes if c.change_class is ChangeClass.REENUMERATED)
    out(f"parameter            : {target.name}")
    out(f"label removed        : {removed_label!r}")
    out(f"reported class       : {change.change_class.value}")
    out(f"reported tier        : {change.severity.value}")
    out(f"detail               : {change.detail}")
    out(f"parameter resolves   : {new.resolve(target.name) is not None}")
    out(f"label still on this parameter : {removed_label in trimmed.label_map}")
    out(f"label used by another parameter: "
        f"{any(removed_label in p.label_map for p in new.parameters if p.name != target.name)}")
    out("")
    out("The last line is a second finding rather than an aside: the same label is in")
    out("use on another parameter, so a bare label reference in a procedure cannot be")
    out("attributed to one parameter. That ambiguity is counted in")
    out("CoverageVerdict.ambiguous_enum_labels on every impact assessment.")
    out("")
    out("A procedure comparing the parameter against the removed label keeps running")
    out("and its branch never fires. Nothing raises. That is the definition of silent")
    out("used throughout this package.")
    out.check(
        "a removed enumeration label is classified silent, not loud",
        change.severity is Severity.SILENT and new.resolve(target.name) is not None,
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
