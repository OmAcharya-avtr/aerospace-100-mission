"""The negative control: substantial textual reordering, empty semantic diff.

The control required of this product is a constructed pair with **no**
semantic change but substantial textual reordering, verified to produce an
empty semantic diff. It is implemented by re-serialising one dictionary with

- the parameter and command entries in a shuffled order,
- the keys inside every entry in a shuffled order,
- every enumeration's entries in a shuffled order,
- every command's argument declarations in a shuffled order while each
  argument keeps its explicit position,
- a different indent, and
- a different accepted spelling for most keys (``mnemonic`` for ``name``,
  ``bits`` for ``bit_width``, ``factor`` for ``scale``, and so on).

"Substantial" is a number here, not an adjective: the count and fraction of
text lines that differ are measured and printed, and the same pair is run
through :mod:`difflib`, which is named in this package's alternatives table,
so the contrast between a text diff and a semantic diff is measured rather
than asserted.

A control that cannot fail is not evidence. Three further controls are run and
each is required to FIRE: reordering plus one injected change of each silent
class must still report exactly that change, so the reordering cannot mask a
real change. Both directions are reported.
"""

from __future__ import annotations

import difflib

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.parse import dumps, loads  # noqa: E402
from tmtcdiff.report import format_table  # noqa: E402
from tmtcdiff.synth import (  # noqa: E402
    INJECTABLE_SILENT,
    baseline_dictionary,
    inject_changes,
    reordering_control,
)

SIZES: tuple[int, ...] = (20, 60, 150, 300)
SEEDS: tuple[int, ...] = tuple(7400 + 17 * index for index in range(6))


def _text_difference(left: str, right: str) -> tuple[int, int]:
    """``(differing lines, total lines)`` from a zero-context unified diff."""
    differing = sum(
        1
        for line in difflib.unified_diff(
            left.splitlines(), right.splitlines(), n=0, lineterm=""
        )
        if line[:1] in "+-" and not line.startswith(("---", "+++"))
    )
    return (differing, len(left.splitlines()) + len(right.splitlines()))


def main() -> int:
    out = Tee("validate_negative_control_output.txt")
    out.rule("tmtcdiff -- negative control: textual reordering with no semantic change")
    for line in environment_lines():
        out(line)
    out("")
    out(f"sizes : {', '.join(str(size) for size in SIZES)}")
    out(f"seeds : {len(SEEDS)} per size")

    config = DiffConfig()

    # --- 1. the control ---------------------------------------------------
    out.rule("the control -- reordered and respelled, semantically identical")
    rows = []
    total_changes = 0
    identity_failures = 0
    for size in SIZES:
        for seed in SEEDS:
            dictionary = baseline_dictionary(size, seed)
            canonical, reordered = reordering_control(dictionary, seed + 1)
            rebuilt = loads(reordered)
            result = diff_dictionaries(dictionary, rebuilt, config)
            differing, total_lines = _text_difference(canonical, reordered)
            total_changes += len(result.changes)
            if rebuilt.parameters != dictionary.parameters or (
                rebuilt.commands != dictionary.commands
            ):
                identity_failures += 1
            rows.append(
                [
                    size,
                    seed,
                    len(canonical.splitlines()),
                    differing,
                    f"{differing / max(1, len(canonical.splitlines())):.4f}",
                    len(result.changes),
                    len(result.silent),
                    len(result.loud),
                    len(result.advisory),
                ]
            )
    out(
        format_table(
            [
                "size",
                "seed",
                "text lines",
                "lines differing",
                "fraction",
                "changes",
                "silent",
                "loud",
                "advisory",
            ],
            rows,
        )
    )
    out("")
    out(f"pairs run                       : {len(rows)}")
    out(f"semantic changes reported, total: {total_changes}")
    out(f"round-trip identity failures    : {identity_failures}")
    out.check(
        "the reordering control produces an empty semantic diff on every pair",
        total_changes == 0,
        f"{total_changes} changes reported across {len(rows)} pairs",
    )
    out.check(
        "the reordered document re-reads to a field-identical dictionary",
        identity_failures == 0,
        f"{identity_failures} of {len(rows)} pairs differed",
    )

    # --- 2. the contrast with a text diff ---------------------------------
    out.rule("what a text diff reports on the same pair -- difflib, measured")
    out("difflib is named in this package's alternatives table. It is the right tool")
    out("for comparing text and it is doing exactly its job here; the point of the")
    out("table is the size of the gap between a text diff and a semantic one on a")
    out("change set that contains no semantic change at all.")
    out("")
    rows = []
    for size in SIZES:
        dictionary = baseline_dictionary(size, SEEDS[0])
        canonical, reordered = reordering_control(dictionary, SEEDS[0] + 1)
        differing, _ = _text_difference(canonical, reordered)
        opcodes = difflib.SequenceMatcher(
            None, canonical.splitlines(), reordered.splitlines()
        ).get_opcodes()
        changed_blocks = sum(1 for tag, *_ in opcodes if tag != "equal")
        # Line-level, not character-level: a character-level ratio over a
        # three-hundred-parameter document is quadratic in the text length and
        # does not finish inside this script's budget.
        ratio = difflib.SequenceMatcher(
            None, canonical.splitlines(), reordered.splitlines()
        ).ratio()
        result = diff_dictionaries(dictionary, loads(reordered), config)
        rows.append(
            [
                size,
                len(canonical.splitlines()),
                differing,
                changed_blocks,
                f"{ratio:.4f}",
                len(result.changes),
            ]
        )
    out(
        format_table(
            [
                "size",
                "text lines",
                "difflib lines differing",
                "difflib changed blocks",
                "difflib line ratio",
                "tmtcdiff semantic changes",
            ],
            rows,
        )
    )
    out.check(
        "a text diff reports a large difference where the semantic diff reports none",
        all(int(row[2]) > 0 and int(row[5]) == 0 for row in rows),
    )

    # --- 3. the control must not be able to pass vacuously ----------------
    out.rule("controls that must FIRE -- reordering must not mask a real change")
    out("Each row reorders the document AND injects one real change of one silent")
    out("class. The diff must report exactly that change and nothing else. A control")
    out("that cannot fail is not evidence, so both directions are asserted.")
    out("")
    rows = []
    fired = 0
    spurious = 0
    for change_class in INJECTABLE_SILENT:
        detected_count = 0
        extra_count = 0
        for size in (60, 150):
            for seed in SEEDS[:3]:
                old = baseline_dictionary(size, seed)
                pair = inject_changes(old, (change_class,), seed + 1)
                _, reordered = reordering_control(pair.new, seed + 2)
                rebuilt = loads(reordered)
                result = diff_dictionaries(pair.old, rebuilt, config)
                keys = {(c.change_class, c.name) for c in result.changes}
                if pair.truth_keys <= keys:
                    detected_count += 1
                extra_count += len(keys - pair.truth_keys)
        fired += detected_count
        spurious += extra_count
        rows.append(
            [
                change_class.value,
                f"{detected_count}/6",
                extra_count,
                "fired" if detected_count == 6 else "DID NOT FIRE",
            ]
        )
    out(format_table(["injected silent class", "detected", "extra changes", "verdict"], rows))
    out("")
    out(f"controls fired : {fired} of {len(INJECTABLE_SILENT) * 6}")
    out(f"extra changes reported across all controls : {spurious}")
    out.check(
        "every silent class is still detected through the reordering",
        fired == len(INJECTABLE_SILENT) * 6,
        f"{fired} of {len(INJECTABLE_SILENT) * 6}",
    )
    out.check(
        "the reordering introduces no change of its own alongside a real one",
        spurious == 0,
        f"{spurious} extra changes",
    )

    # --- 4. the two reordering mechanisms, separately ---------------------
    out.rule("reordering and respelling, separated")
    out("The control combines two mechanisms. Each is run alone, so a failure could")
    out("be attributed.")
    out("")
    rows = []
    for aliases in (False, True):
        for size in SIZES:
            dictionary = baseline_dictionary(size, SEEDS[0])
            canonical = dumps(dictionary)
            _, reordered = reordering_control(dictionary, SEEDS[0] + 3, aliases=aliases)
            differing, _ = _text_difference(canonical, reordered)
            result = diff_dictionaries(dictionary, loads(reordered), config)
            rows.append(
                [
                    "reorder + respell" if aliases else "reorder only",
                    size,
                    differing,
                    len(result.changes),
                ]
            )
    out(format_table(["mechanism", "size", "lines differing", "semantic changes"], rows))
    out.check(
        "both mechanisms independently produce an empty semantic diff",
        all(int(row[3]) == 0 for row in rows),
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
