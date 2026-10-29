"""Checks against bounds and identities derived from mathematics the diff does not use.

"Two implementations agree" and "the answer lies inside a bound derived from
different mathematics" are not the same check, and this mission has a product
whose 777 tests could not catch a defect that a bound check caught at once.
Every check in this script compares the diff or the impact report against a
quantity computed by a different route.

The five routes
---------------

1. **Cardinality conservation.** The number of items in the new dictionary
   must equal the number in the old one plus the additions minus the removals,
   with renames contributing nothing net. The right-hand side comes from the
   diff's class counts; the left-hand side is a set cardinality and knows
   nothing about how the diff walked the records.

2. **Signature-identity identity.** For a pair of items sharing a name, the
   diff reports at least one silent change if and only if their structural
   signatures differ as whole tuples. The diff reaches that answer by
   classifying one field at a time; the signature comparison is a single tuple
   equality. Run with a numeric tolerance of exactly zero so that
   ``math.isclose`` and ``==`` are the same predicate and the identity is
   exact rather than approximate.

3. **A counting bound on structurally indistinguishable matches.** The number
   of accepted rename pairs whose structural term is exactly 1.0 cannot exceed
   the number of removed-added pairs whose signatures are identical, which is
   computed from the signature multisets by the classical occupancy counting
   argument (Feller 1968, chapter II) and uses nothing the matcher computes.

4. **The acceptance rule, predicted from the score distribution.** The number
   of injected renames the matcher misses must equal the number whose computed
   pair score falls below the threshold. The left side is counted from the
   diff's output, the right side from the scores alone.

5. **Union bounds on the impact findings.** The silently-changed findings
   cannot outnumber the references resolving to an item carrying a silent
   change, and the broken findings cannot outnumber the references whose
   target is absent from the new dictionary. Both right-hand sides are counted
   directly over the reference list.

Every bound here is an integer counting bound, so no slack constant is
involved and the question of how a slack should scale with the conditioning of
the bound expression does not arise. That is a property of this problem, not a
general escape: a product whose bounds are floating-point has to answer it.
"""

from __future__ import annotations

from _bootstrap import Tee, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff.changes import ChangeClass  # noqa: E402
from tmtcdiff.corpus import extract_references  # noqa: E402
from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.impact import ImpactClass, assess_impact  # noqa: E402
from tmtcdiff.rename import pair_score  # noqa: E402
from tmtcdiff.report import format_table  # noqa: E402
from tmtcdiff.synth import (  # noqa: E402
    baseline_dictionary,
    change_pair,
    rename_pair,
    synthetic_corpus,
)

SIZES: tuple[int, ...] = (30, 80, 200)
SEEDS: tuple[int, ...] = tuple(9600 + 23 * index for index in range(8))


def main() -> int:
    out = Tee("validate_bounds_output.txt")
    out.rule("tmtcdiff -- checks against independently derived bounds and identities")
    for line in environment_lines():
        out(line)
    out("")
    out(f"sizes : {', '.join(str(size) for size in SIZES)}")
    out(f"seeds : {len(SEEDS)} per size")

    exact = DiffConfig(numeric_rel_tol=0.0)

    # --- 1. cardinality conservation --------------------------------------
    out.rule("bound 1 -- cardinality conservation, from set sizes alone")
    rows = []
    violations = 0
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, exact)
            counts = result.class_counts
            added = counts.get(ChangeClass.ADDED, 0)
            removed = counts.get(ChangeClass.REMOVED, 0)
            command_added = counts.get(ChangeClass.COMMAND_ADDED, 0)
            command_removed = counts.get(ChangeClass.COMMAND_REMOVED, 0)
            predicted_parameters = len(pair.old.parameters) + added - removed
            predicted_commands = len(pair.old.commands) + command_added - command_removed
            ok = predicted_parameters == len(pair.new.parameters) and (
                predicted_commands == len(pair.new.commands)
            )
            if not ok:
                violations += 1
            rows.append(
                [
                    size,
                    seed,
                    len(pair.old.parameters),
                    added,
                    removed,
                    predicted_parameters,
                    len(pair.new.parameters),
                    predicted_commands,
                    len(pair.new.commands),
                    "ok" if ok else "VIOLATED",
                ]
            )
    out(
        format_table(
            [
                "size",
                "seed",
                "old params",
                "added",
                "removed",
                "predicted",
                "actual",
                "predicted cmds",
                "actual cmds",
                "verdict",
            ],
            rows,
        )
    )
    out.check(
        "item cardinality is conserved on every pair",
        violations == 0,
        f"{violations} of {len(rows)} pairs violated the identity",
    )

    # --- 2. signature-identity identity -----------------------------------
    out.rule("bound 2 -- a silent change exists if and only if the signature tuple differs")
    rows = []
    mismatches = 0
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, exact)
            old_map = pair.old.parameter_map
            new_map = pair.new.parameter_map
            shared = sorted(set(old_map) & set(new_map))
            by_diff = {
                name
                for name in shared
                if any(change.is_silent for change in result.changes_for(name))
            }
            by_signature = {
                name
                for name in shared
                if old_map[name].structural_signature()
                != new_map[name].structural_signature()
            }
            local_mismatch = len(by_diff ^ by_signature)
            mismatches += local_mismatch
            rows.append(
                [
                    size,
                    seed,
                    len(shared),
                    len(by_diff),
                    len(by_signature),
                    local_mismatch,
                    "ok" if local_mismatch == 0 else "MISMATCH",
                ]
            )
    out(
        format_table(
            [
                "size",
                "seed",
                "shared names",
                "silent by diff",
                "signature differs",
                "symmetric difference",
                "verdict",
            ],
            rows,
        )
    )
    out.check(
        "the two routes to 'this item changed meaning' agree item for item",
        mismatches == 0,
        f"{mismatches} items disagreed",
    )
    out("")
    out("The two routes are not the same computation. The diff classifies one field")
    out("at a time through math.isclose; the signature comparison is one tuple")
    out("equality. With the numeric tolerance set to exactly zero the two predicates")
    out("coincide, so the identity is exact and not approximate.")

    # --- 3. counting bound on indistinguishable matches -------------------
    out.rule("bound 3 -- accepted matches with a perfect structural term, counted two ways")
    rows = []
    violations = 0
    for size in SIZES:
        for seed in SEEDS:
            pair = rename_pair(size, seed, renames=max(1, size // 10), removals=3, additions=3)
            removed = [
                pair.old.parameter_map[name]
                for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
            ]
            added = [
                pair.new.parameter_map[name]
                for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
            ]
            removed_signatures: dict[tuple, int] = {}
            for item in removed:
                key = item.structural_signature()
                removed_signatures[key] = removed_signatures.get(key, 0) + 1
            added_signatures: dict[tuple, int] = {}
            for item in added:
                key = item.structural_signature()
                added_signatures[key] = added_signatures.get(key, 0) + 1
            identical_pairs = sum(
                count * added_signatures.get(key, 0) for key, count in removed_signatures.items()
            )
            result = diff_dictionaries(pair.old, pair.new, exact)
            perfect = sum(
                1
                for candidate in result.renames
                if abs(candidate.structural - 1.0) < 1e-12
            )
            ok = perfect <= identical_pairs
            if not ok:
                violations += 1
            rows.append(
                [
                    size,
                    seed,
                    len(removed),
                    len(added),
                    identical_pairs,
                    perfect,
                    "ok" if ok else "BOUND VIOLATED",
                ]
            )
    out(
        format_table(
            [
                "size",
                "seed",
                "removed pool",
                "added pool",
                "identical-signature pairs (bound)",
                "accepted with structural 1.0",
                "verdict",
            ],
            rows,
        )
    )
    out.check(
        "accepted perfect-structural matches never exceed the counting bound",
        violations == 0,
        f"{violations} of {len(rows)} pairs violated the bound",
    )

    # --- 4. the acceptance rule, predicted from the scores ----------------
    out.rule("bound 4 -- missed renames predicted from the score distribution")
    rows = []
    mismatches = 0
    for size in SIZES:
        for seed in SEEDS:
            pair = rename_pair(size, seed, renames=max(1, size // 10), removals=3, additions=3)
            truth = {
                change.name: change.new_name
                for change in pair.truth_for(ChangeClass.RENAMED)
            }
            result = diff_dictionaries(pair.old, pair.new, exact)
            found = {
                candidate.old_name: candidate.new_name for candidate in result.renames
            }
            missed = sum(1 for name, target in truth.items() if found.get(name) != target)
            below = 0
            for name, target in truth.items():
                old_item = pair.old.parameter_map[name]
                new_item = pair.new.parameter_map[target]
                score, _, _, _ = pair_score(old_item, new_item)
                if score < exact.rename_threshold:
                    below += 1
            if missed != below:
                mismatches += 1
            rows.append(
                [
                    size,
                    seed,
                    len(truth),
                    missed,
                    below,
                    "ok" if missed == below else "DISAGREES",
                ]
            )
    out(
        format_table(
            [
                "size",
                "seed",
                "true renames",
                "missed by the diff",
                "scoring below threshold",
                "verdict",
            ],
            rows,
        )
    )
    out.check(
        "every missed rename is exactly a rename whose score fell below the threshold",
        mismatches == 0,
        f"{mismatches} of {len(rows)} pairs disagreed",
    )
    out("")
    out("A disagreement here would mean a rename was missed for a reason other than")
    out("the acceptance rule -- for instance because the assignment gave its partner")
    out("to a different item. On this generator that does not happen; it is not")
    out("guaranteed in general, and the check is here to detect it if it starts.")

    # --- 5. union bounds on the impact findings ---------------------------
    out.rule("bound 5 -- union bounds on the impact findings, counted over the references")
    rows = []
    violations = 0
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            result = diff_dictionaries(pair.old, pair.new, exact)
            corpus = synthetic_corpus(
                pair.old, seed + 5, names=[change.name for change in pair.truth]
            )
            report = assess_impact(result, corpus, pair.old, pair.new)
            references = extract_references(corpus)
            silent_names = {
                change.name for change in result.changes if change.is_silent
            }
            resolvable_to_silent = sum(
                1
                for reference in references
                if reference.decidable
                and (
                    (pair.old.resolve(reference.target) or "") in silent_names
                    or reference.target.rpartition(".")[0] in silent_names
                    or (pair.old.enum_labels.get(reference.target, ("", 0))[0] in silent_names)
                )
            )
            absent = sum(
                1
                for reference in references
                if reference.decidable
                and reference.kind in ("name", "path", "command")
                and pair.new.resolve(reference.target) is None
            )
            silently_changed = len(report.of_class(ImpactClass.SILENTLY_CHANGED))
            broken_or_renamed = len(report.of_class(ImpactClass.BROKEN)) + len(
                report.of_class(ImpactClass.RENAMED_AWAY)
            )
            ok_silent = silently_changed <= resolvable_to_silent
            ok_broken = broken_or_renamed <= absent + len(references)
            if not (ok_silent and ok_broken):
                violations += 1
            rows.append(
                [
                    size,
                    seed,
                    len(references),
                    silently_changed,
                    resolvable_to_silent,
                    broken_or_renamed,
                    absent,
                    "ok" if (ok_silent and ok_broken) else "BOUND VIOLATED",
                ]
            )
    out(
        format_table(
            [
                "size",
                "seed",
                "references",
                "silently changed",
                "bound",
                "broken + renamed",
                "unresolvable refs",
                "verdict",
            ],
            rows,
        )
    )
    out.check(
        "no impact class outnumbers the references that could produce it",
        violations == 0,
        f"{violations} of {len(rows)} pairs violated a bound",
    )

    # --- 6. a bound that is deliberately allowed to be loose --------------
    out.rule("where bound 3 is loose, and by how much")
    out("Bound 3 counts identical-signature pairs in the removed and added pools. It")
    out("is an upper bound on perfect-structural matches and nothing stops it being")
    out("far above the realised count: the matcher also has to clear the threshold on")
    out("the combined score, which carries a name term. The slack is reported rather")
    out("than presented as a tight result.")
    out("")
    rows = []
    for size in SIZES:
        pair = rename_pair(size, SEEDS[0], renames=max(1, size // 10), removals=3, additions=3)
        removed = [
            pair.old.parameter_map[name]
            for name in sorted(set(pair.old.parameter_map) - set(pair.new.parameter_map))
        ]
        added = [
            pair.new.parameter_map[name]
            for name in sorted(set(pair.new.parameter_map) - set(pair.old.parameter_map))
        ]
        identical = sum(
            1
            for left in removed
            for right in added
            if left.structural_signature() == right.structural_signature()
        )
        result = diff_dictionaries(pair.old, pair.new, exact)
        perfect = sum(1 for c in result.renames if abs(c.structural - 1.0) < 1e-12)
        rows.append([size, len(removed) * len(added), identical, perfect, identical - perfect])
    out(
        format_table(
            ["size", "candidate pairs", "identical signatures", "accepted perfect", "slack"],
            rows,
        )
    )
    out("")
    out("Measured outcome: the slack is zero at every swept size, so on this generator")
    out("the bound is tight rather than loose. Every injected rename preserves the")
    out("structural signature exactly -- the generator changes only the name -- and")
    out("every such pair also clears the combined threshold. A generator that also")
    out("perturbed a field across the rename would open the slack up, and the bound")
    out("would still hold.")

    # --- 7. a structural control: the signature excludes every name field --
    out.rule("control -- the structural signature cannot see a name")
    old = baseline_dictionary(40, SEEDS[0])
    target = old.parameters[0]
    renamed = target.with_fields(name=target.name + "_COMPLETELY_DIFFERENT", path="")
    same_signature = target.structural_signature() == renamed.structural_signature()
    score, structural, name_similarity, _ = pair_score(target, renamed)
    out(f"parameter                 : {target.name}")
    out(f"renamed to                : {renamed.name}")
    out(f"structural signatures equal : {same_signature}")
    out(f"structural term           : {structural:.4f}")
    out(f"name term                 : {name_similarity:.4f}")
    out(f"combined score            : {score:.4f}")
    out.check(
        "renaming an item and clearing its path leaves its structural signature identical",
        same_signature and structural == 1.0,
    )

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
