"""Impact classification against known injections, and the coverage verdict.

Part 1 measures the classifier. For each change class, a pair is generated
with that class injected alone, and a one-file corpus is built containing
exactly three references: one to the injected item in the form that class
affects, one to an item no injection touched, and one assembled by a run-time
expression. The expected impact class of each is known from the injection, so
the result is a confusion table rather than an impression.

Part 2 measures the coverage verdict -- the first-class output of this library
that says what the assessment could not determine. Three cases are
constructed: full coverage with nothing undetermined, partial coverage with
the changed item outside the corpus, and a corpus with an unreadable source.

Part 3 is the claim the product specification requires this package not to
make. A corpus that references only unchanged items produces an empty
actionable finding list while a silent change sits outside it. The rendered
interpretation of that report is checked to say the assessment is incomplete,
and every rendered output in this script is scanned for the word "safe".
"""

from __future__ import annotations

import re

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff.changes import ChangeClass  # noqa: E402
from tmtcdiff.corpus import Corpus, CorpusSource, extract_references  # noqa: E402
from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.impact import ImpactClass, assess_impact  # noqa: E402
from tmtcdiff.report import format_table, render_impact  # noqa: E402
from tmtcdiff.synth import baseline_dictionary, inject_changes, synthetic_corpus  # noqa: E402

SEEDS: tuple[int, ...] = tuple(8500 + 19 * index for index in range(8))
SIZE = 60

#: class -> (how the corpus references the injected item, expected impact class)
EXPECTATIONS: dict[ChangeClass, tuple[str, ImpactClass]] = {
    ChangeClass.RESCALED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.UNIT_CHANGED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.REENUMERATED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.BIT_SHIFTED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.RETYPED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.ENDIANNESS_CHANGED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.PACKET_MOVED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.LIMITS_CHANGED: ("name", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.REMOVED: ("name", ImpactClass.BROKEN),
    ChangeClass.RENAMED: ("name", ImpactClass.RENAMED_AWAY),
    ChangeClass.PATH_MOVED: ("path", ImpactClass.BROKEN),
    ChangeClass.DESCRIPTION_CHANGED: ("name", ImpactClass.UNAFFECTED),
    ChangeClass.COMMAND_REMOVED: ("command", ImpactClass.BROKEN),
    ChangeClass.COMMAND_RENAMED: ("command", ImpactClass.RENAMED_AWAY),
    ChangeClass.ARGUMENT_REORDERED: ("argument", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.ARGUMENT_RETYPED: ("argument", ImpactClass.SILENTLY_CHANGED),
    ChangeClass.ARGUMENT_RESCALED: ("argument", ImpactClass.SILENTLY_CHANGED),
}


def _reference_line(form: str, old, name: str) -> str:
    if form == "name":
        return f'series = tm["{name}"]'
    if form == "path":
        item = old.parameter_map.get(name) or old.command_map.get(name)
        return f'series = tm["{item.path}"]'
    if form == "command":
        command = old.command_map[name]
        arguments = ", ".join(f"{argument.name}=1" for argument in command.arguments)
        return f'send_command("{name}", {arguments})'
    command = old.command_map[name]
    first = command.arguments[0].name
    return f'send_command("{name}", {first}=1)'


def main() -> int:
    out = Tee("validate_impact_report_output.txt")
    out.rule("tmtcdiff -- impact classification and the coverage verdict")
    for line in environment_lines():
        out(line)
    out("")
    out(f"dictionary size : {SIZE} parameters")
    out(f"seeds           : {len(SEEDS)} per class")
    out(f"classes probed  : {len(EXPECTATIONS)}")

    config = DiffConfig()
    rendered_outputs: list[str] = []

    # --- part 1: the classifier ------------------------------------------
    out.rule("part 1 -- impact classification against known injections")
    confusion: dict[tuple[ImpactClass, ImpactClass], int] = {}
    rows = []
    for change_class, (form, expected) in EXPECTATIONS.items():
        correct = 0
        seen: dict[ImpactClass, int] = {}
        for seed in SEEDS:
            old = baseline_dictionary(SIZE, seed)
            pair = inject_changes(old, (change_class,), seed + 1)
            injected = pair.truth[0]
            untouched = next(
                parameter.name
                for parameter in pair.old.parameters
                if parameter.name != injected.name
                and parameter.name in pair.new.parameter_map
                and not diff_dictionaries(pair.old, pair.new, config).changes_for(parameter.name)
            )
            text = "\n".join(
                [
                    "# generated probe corpus",
                    _reference_line(form, pair.old, injected.name),
                    f'control = tm["{untouched}"]',
                    'dynamic = tm["PREFIX_%02d" % index]',
                ]
            )
            corpus = Corpus((CorpusSource(label="probe/one.py", kind="script", text=text),))
            result = diff_dictionaries(pair.old, pair.new, config)
            report = assess_impact(result, corpus, pair.old, pair.new)
            references = extract_references(corpus)
            target_reference = references[0]
            finding = next(
                f for f in report.findings if f.reference == target_reference
            )
            seen[finding.impact_class] = seen.get(finding.impact_class, 0) + 1
            confusion[(expected, finding.impact_class)] = (
                confusion.get((expected, finding.impact_class), 0) + 1
            )
            if finding.impact_class is expected:
                correct += 1
            # the control and the dynamic reference, every time
            control_finding = next(
                f for f in report.findings if f.reference.target == untouched
            )
            dynamic_finding = next(
                f for f in report.findings if not f.reference.decidable
            )
            confusion[(ImpactClass.UNAFFECTED, control_finding.impact_class)] = (
                confusion.get((ImpactClass.UNAFFECTED, control_finding.impact_class), 0) + 1
            )
            confusion[(ImpactClass.UNDECIDABLE, dynamic_finding.impact_class)] = (
                confusion.get((ImpactClass.UNDECIDABLE, dynamic_finding.impact_class), 0) + 1
            )
        rows.append(
            [
                change_class.value,
                form,
                expected.value,
                f"{correct}/{len(SEEDS)}",
                ", ".join(
                    f"{k.value}:{v}" for k, v in sorted(seen.items(), key=lambda i: i[0].rank)
                ),
            ]
        )
    out(
        format_table(
            ["injected class", "reference form", "expected impact", "correct", "observed"], rows
        )
    )
    out("")
    out("confusion table over every probed reference, expected against observed:")
    classes = sorted(ImpactClass, key=lambda c: c.rank)
    out(
        format_table(
            ["expected \\ observed", *[c.value for c in classes]],
            [
                [
                    expected.value,
                    *[str(confusion.get((expected, observed), 0)) for observed in classes],
                ]
                for expected in classes
            ],
        )
    )
    off_diagonal = sum(
        count for (expected, observed), count in confusion.items() if expected is not observed
    )
    total_probed = sum(confusion.values())
    out("")
    out(f"references probed : {total_probed}")
    out(f"misclassified     : {off_diagonal}")
    out.check(
        "every probed reference lands in its expected impact class",
        off_diagonal == 0,
        f"{off_diagonal} of {total_probed} misclassified",
    )

    # --- rename suggestion correctness -----------------------------------
    out.rule("rename suggestions -- is the suggested new name the injected one?")
    suggested_correct = 0
    suggested_total = 0
    for seed in SEEDS:
        old = baseline_dictionary(SIZE, seed)
        pair = inject_changes(old, (ChangeClass.RENAMED,), seed + 1)
        injected = pair.truth[0]
        text = f'series = tm["{injected.name}"]'
        corpus = Corpus((CorpusSource(label="probe/rename.py", kind="script", text=text),))
        result = diff_dictionaries(pair.old, pair.new, config)
        report = assess_impact(result, corpus, pair.old, pair.new)
        finding = report.findings[0]
        suggested_total += 1
        if finding.suggestion == injected.new_name:
            suggested_correct += 1
    out(f"rename references probed : {suggested_total}")
    out(f"suggestion correct       : {suggested_correct}")
    out.check(
        "the suggested replacement name is the injected rename target",
        suggested_correct == suggested_total,
        f"{suggested_correct} of {suggested_total}",
    )

    # --- part 2: the coverage verdict ------------------------------------
    out.rule("part 2 -- the coverage verdict, a computed output")
    rows = []

    # case A: full coverage, nothing undetermined.
    old = baseline_dictionary(SIZE, SEEDS[0])
    pair = inject_changes(old, (ChangeClass.RESCALED,), SEEDS[0] + 1)
    every_name = "\n".join(f'x = tm["{name}"]' for name in pair.old.item_names)
    full = Corpus((CorpusSource(label="full/all.py", kind="script", text=every_name),))
    result = diff_dictionaries(pair.old, pair.new, DiffConfig(numeric_rel_tol=0.0))
    report_full = assess_impact(result, full, pair.old, pair.new)
    rendered_outputs.append(render_impact(report_full))
    rows.append(
        [
            "A full coverage",
            report_full.verdict.references_total,
            report_full.verdict.references_undecidable,
            f"{report_full.verdict.coverage_fraction:.4f}",
            report_full.verdict.items_unreferenced,
            report_full.verdict.determinable,
            len(report_full.verdict.reasons),
        ]
    )

    # case B: partial coverage, the changed item outside the corpus.
    unchanged_names = [
        name for name in pair.old.item_names if name != pair.truth[0].name
    ][:10]
    partial = Corpus(
        (
            CorpusSource(
                label="partial/some.py",
                kind="script",
                text="\n".join(f'x = tm["{name}"]' for name in unchanged_names),
            ),
        )
    )
    report_partial = assess_impact(result, partial, pair.old, pair.new)
    rendered_outputs.append(render_impact(report_partial))
    rows.append(
        [
            "B changed item outside corpus",
            report_partial.verdict.references_total,
            report_partial.verdict.references_undecidable,
            f"{report_partial.verdict.coverage_fraction:.4f}",
            report_partial.verdict.items_unreferenced,
            report_partial.verdict.determinable,
            len(report_partial.verdict.reasons),
        ]
    )

    # case C: an unreadable source and run-time references.
    generated = synthetic_corpus(
        pair.old, SEEDS[1], names=[pair.truth[0].name], include_unreadable=True
    )
    report_generated = assess_impact(result, generated, pair.old, pair.new)
    rendered_outputs.append(render_impact(report_generated))
    rows.append(
        [
            "C unreadable source present",
            report_generated.verdict.references_total,
            report_generated.verdict.references_undecidable,
            f"{report_generated.verdict.coverage_fraction:.4f}",
            report_generated.verdict.items_unreferenced,
            report_generated.verdict.determinable,
            len(report_generated.verdict.reasons),
        ]
    )

    # case E: full coverage and no change at all.
    no_change = diff_dictionaries(pair.old, pair.old, DiffConfig(numeric_rel_tol=0.0))
    report_no_change = assess_impact(no_change, full, pair.old, pair.old)
    rendered_outputs.append(render_impact(report_no_change))

    # case D: an empty corpus.
    report_empty = assess_impact(result, Corpus(), pair.old, pair.new)
    rendered_outputs.append(render_impact(report_empty))
    rows.append(
        [
            "D empty corpus",
            report_empty.verdict.references_total,
            report_empty.verdict.references_undecidable,
            f"{report_empty.verdict.coverage_fraction:.4f}",
            report_empty.verdict.items_unreferenced,
            report_empty.verdict.determinable,
            len(report_empty.verdict.reasons),
        ]
    )
    rows.append(
        [
            "E full coverage, no change",
            report_no_change.verdict.references_total,
            report_no_change.verdict.references_undecidable,
            f"{report_no_change.verdict.coverage_fraction:.4f}",
            report_no_change.verdict.items_unreferenced,
            report_no_change.verdict.determinable,
            len(report_no_change.verdict.reasons),
        ]
    )
    out(
        format_table(
            [
                "case",
                "references",
                "undecidable",
                "coverage",
                "items unreferenced",
                "determinable",
                "reasons",
            ],
            rows,
        )
    )
    out("")
    out("case A reasons:")
    for reason in report_full.verdict.reasons:
        out(f"  - {reason}")
    out("case B reasons:")
    for reason in report_partial.verdict.reasons:
        out(f"  - {reason}")
    out("case C reasons:")
    for reason in report_generated.verdict.reasons:
        out(f"  - {reason}")
    out("case D reasons:")
    for reason in report_empty.verdict.reasons:
        out(f"  - {reason}")

    out.check(
        "full coverage with a zero numeric tolerance is reported as fully determinable",
        report_full.verdict.determinable,
        f"reasons: {report_full.verdict.reasons}",
    )
    out.check(
        "an independent count of referenced items agrees with the verdict",
        report_full.verdict.items_referenced == len(pair.old),
        f"{report_full.verdict.items_referenced} against {len(pair.old)}",
    )
    out.check(
        "partial coverage is reported as not determinable",
        not report_partial.verdict.determinable,
    )
    out.check(
        "an empty corpus is reported as not determinable",
        not report_empty.verdict.determinable,
    )
    out.check(
        "an unreadable source appears in the verdict",
        bool(report_generated.verdict.unreadable_sources),
        str(report_generated.verdict.unreadable_sources),
    )

    # --- part 3: the statement this package must not make ----------------
    out.rule("part 3 -- an empty finding list is not a safety statement")
    out("Case B has a silent change in the dictionary and a corpus that does not")
    out("reference the changed item. The actionable finding list is empty. The")
    out("rendered interpretation must say the assessment is incomplete.")
    out("")
    out(f"case B actionable findings : {len(report_partial.actionable)}")
    out(f"case B silently changed    : {len(report_partial.silently_changed)}")
    out("case B interpretation:")
    for line in report_partial.interpretation.split(". "):
        out(f"  {line.strip()}")
    out.check(
        "case B has no actionable finding at all",
        not report_partial.actionable,
        f"{len(report_partial.actionable)} findings",
    )
    out.check(
        "the interpretation of an empty report under partial coverage says it is incomplete",
        "NOT complete" in report_partial.interpretation,
    )
    out.check(
        "the interpretation of an empty report under partial coverage names the reasons",
        all(reason in report_partial.interpretation for reason in report_partial.verdict.reasons),
    )
    out("")
    out("case E interpretation, full coverage and nothing changed, which is the only")
    out("case in this script where the assessment is complete and empty:")
    for line in report_no_change.interpretation.split(". "):
        out(f"  {line.strip()}")
    out.check(
        "case E is reported as fully determinable",
        report_no_change.verdict.determinable,
        f"reasons: {report_no_change.verdict.reasons}",
    )
    out.check(
        "even a complete, empty assessment is scoped to this corpus and these two versions",
        "this corpus and these two dictionary" in report_no_change.interpretation,
    )
    out("")
    out("case A interpretation, where a reference IS affected:")
    for line in report_full.interpretation.split(". "):
        out(f"  {line.strip()}")

    # Word boundaries, not substrings: the generator mints enumeration labels
    # such as SAFE_MODE, and a substring scan would report those as safety
    # claims. The scan is for the word, in any case, standing on its own.
    forbidden = ("safe", "safely", "certified", "flight-qualified", "production-ready")
    offenders = [
        (word, index)
        for index, text in enumerate(rendered_outputs)
        for word in forbidden
        if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
    ]
    out("")
    out(f"rendered reports scanned : {len(rendered_outputs)}")
    out(f"forbidden words searched : {', '.join(forbidden)}")
    out(f"occurrences found        : {len(offenders)}")
    out.check(
        "no rendered impact report contains a safety or certification word",
        not offenders,
        str(offenders[:4]),
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
