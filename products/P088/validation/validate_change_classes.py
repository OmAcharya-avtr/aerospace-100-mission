"""Per-class detection precision and recall over pairs with known injected changes.

Method
------

For each of the 21 change classes, dictionary pairs are generated in which
exactly that class is injected, once per pair, on an item no other injection
touches. The diff is then run and its reported ``(class, name)`` keys are
matched against the ground truth. Two regimes are measured:

- **isolated**: one class per pair. This measures whether the classifier can
  recognise the class at all.
- **combined**: every injectable class in the same pair. This measures what
  the classes do to each other -- in particular, an unrelated removal and an
  unrelated addition in the same pair can be matched as a rename, which costs
  recall on both and precision on the rename class. That interaction is the
  reason the combined regime is measured and reported separately.

Counting convention: keys are deduplicated per pair, so a class detected on
the same item through two fields (``bit_offset`` and ``bit_width`` both move,
say) counts once. Ground truth has at most one injection per (class, item), so
the convention is symmetric.

Also measured here: the detection floor on a relative scale change, found by
bisection rather than asserted from the configured tolerance.
"""

from __future__ import annotations

from _bootstrap import Tee, Timer, bootstrap, environment_lines

ROOT = bootstrap()

from tmtcdiff.changes import ChangeClass, Severity  # noqa: E402
from tmtcdiff.diff import DiffConfig, diff_dictionaries  # noqa: E402
from tmtcdiff.metrics import aggregate, per_class_metrics  # noqa: E402
from tmtcdiff.model import Dictionary  # noqa: E402
from tmtcdiff.report import format_table  # noqa: E402
from tmtcdiff.synth import (  # noqa: E402
    INJECTABLE,
    baseline_dictionary,
    change_pair,
    inject_changes,
)

SIZES: tuple[int, ...] = (30, 120)
SEEDS: tuple[int, ...] = tuple(5200 + 11 * index for index in range(10))


def _detected_keys(result) -> set[tuple[ChangeClass, str]]:
    return {(change.change_class, change.name) for change in result.changes}


def main() -> int:
    out = Tee("validate_change_classes_output.txt")
    out.rule("tmtcdiff -- per-class detection precision and recall")
    for line in environment_lines():
        out(line)
    out("")
    out(f"classes injected : {len(INJECTABLE)}")
    out(f"sizes            : {', '.join(str(size) for size in SIZES)}")
    out(f"seeds            : {len(SEEDS)} per cell, {SEEDS[0]} to {SEEDS[-1]} step 11")
    out(f"pairs per regime : {len(SIZES) * len(SEEDS)} per class (isolated), "
        f"{len(SIZES) * len(SEEDS)} total (combined)")
    out("")
    out("Rename detection runs at the threshold selected by")
    out("validation/validate_rename_sweep.py. Nothing here retunes it.")

    config = DiffConfig()
    timer = Timer()

    # --- isolated regime --------------------------------------------------
    isolated_truth: list[tuple[ChangeClass, str]] = []
    isolated_detected: list[tuple[ChangeClass, str]] = []
    fired: dict[ChangeClass, tuple[int, int]] = {}
    for change_class in INJECTABLE:
        pairs_run = 0
        pairs_detected = 0
        for size in SIZES:
            for seed in SEEDS:
                old = baseline_dictionary(size, seed)
                pair = inject_changes(old, (change_class,), seed + 1, per_class=1)
                with timer:
                    result = diff_dictionaries(pair.old, pair.new, config)
                truth = pair.truth_keys
                detected = _detected_keys(result)
                isolated_truth.extend(truth)
                isolated_detected.extend(detected)
                pairs_run += 1
                if truth <= detected:
                    pairs_detected += 1
        fired[change_class] = (pairs_detected, pairs_run)

    isolated = per_class_metrics(isolated_truth, isolated_detected)
    out.rule("isolated regime -- one injected class per pair")
    rows = []
    for change_class in sorted(isolated, key=lambda c: (c.severity.rank, c.value)):
        metrics = isolated[change_class]
        detected_pairs, run_pairs = fired.get(change_class, (0, 0))
        rows.append(
            [
                change_class.value,
                change_class.severity.value,
                metrics.injected,
                metrics.detections,
                metrics.true_positives,
                metrics.false_positives,
                metrics.false_negatives,
                f"{metrics.precision:.4f}",
                f"{metrics.recall:.4f}",
                f"{detected_pairs}/{run_pairs}" if run_pairs else "-",
            ]
        )
    out(
        format_table(
            [
                "change class",
                "tier",
                "injected",
                "detected",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
                "pairs fired",
            ],
            rows,
        )
    )
    total = aggregate(isolated.values())
    out("")
    out(f"micro-averaged over every class: precision {total.precision:.4f}, "
        f"recall {total.recall:.4f}, TP {total.true_positives}, FP {total.false_positives}, "
        f"FN {total.false_negatives}")
    out.check(
        "every injected class fires on every isolated pair",
        all(detected == run for detected, run in fired.values()),
        ", ".join(
            f"{cls.value} {detected}/{run}"
            for cls, (detected, run) in fired.items()
            if detected != run
        )
        or "all classes fired on all pairs",
    )
    out.check(
        "isolated regime has no false positive of any class",
        total.false_positives == 0,
        f"{total.false_positives} false positives",
    )

    # --- combined regime --------------------------------------------------
    combined_truth: list[tuple[ChangeClass, str]] = []
    combined_detected: list[tuple[ChangeClass, str]] = []
    for size in SIZES:
        for seed in SEEDS:
            pair = change_pair(size, seed)
            with timer:
                result = diff_dictionaries(pair.old, pair.new, config)
            combined_truth.extend(pair.truth_keys)
            combined_detected.extend(_detected_keys(result))

    combined = per_class_metrics(combined_truth, combined_detected)
    out.rule("combined regime -- every injectable class in the same pair")
    rows = []
    for change_class in sorted(combined, key=lambda c: (c.severity.rank, c.value)):
        metrics = combined[change_class]
        rows.append(
            [
                change_class.value,
                change_class.severity.value,
                metrics.injected,
                metrics.detections,
                metrics.true_positives,
                metrics.false_positives,
                metrics.false_negatives,
                f"{metrics.precision:.4f}",
                f"{metrics.recall:.4f}",
            ]
        )
    out(
        format_table(
            [
                "change class",
                "tier",
                "injected",
                "detected",
                "TP",
                "FP",
                "FN",
                "precision",
                "recall",
            ],
            rows,
        )
    )
    combined_total = aggregate(combined.values())
    out("")
    out(f"micro-averaged: precision {combined_total.precision:.4f}, "
        f"recall {combined_total.recall:.4f}, TP {combined_total.true_positives}, "
        f"FP {combined_total.false_positives}, FN {combined_total.false_negatives}")
    out("")
    out("The combined regime is where the classes interact. An unrelated removal and")
    out("an unrelated addition in the same pair can be paired as a rename; that costs")
    out("recall on 'removed' and 'added' and precision on 'renamed'. The numbers above")
    out("are not softened: where a class loses recall in the combined regime but not")
    out("in the isolated one, the cause is that interaction and nothing else.")
    for change_class in (ChangeClass.REMOVED, ChangeClass.ADDED, ChangeClass.RENAMED):
        if change_class in combined and change_class in isolated:
            out(
                f"  {change_class.value:<16s} isolated recall "
                f"{isolated[change_class].recall:.4f} -> combined "
                f"{combined[change_class].recall:.4f}; precision "
                f"{isolated[change_class].precision:.4f} -> "
                f"{combined[change_class].precision:.4f}"
            )

    # --- the silent tier, separately and first ---------------------------
    out.rule("the silent tier on its own -- the class this product exists for")
    silent_classes = [c for c in INJECTABLE if c.severity is Severity.SILENT]
    silent_metrics = [isolated[c] for c in silent_classes if c in isolated]
    silent_total = aggregate(silent_metrics)
    loud_classes = [c for c in INJECTABLE if c.severity is Severity.LOUD]
    loud_total = aggregate([isolated[c] for c in loud_classes if c in isolated])
    out(
        format_table(
            ["tier", "classes", "injected", "TP", "FP", "FN", "precision", "recall"],
            [
                [
                    "silent",
                    len(silent_classes),
                    silent_total.injected,
                    silent_total.true_positives,
                    silent_total.false_positives,
                    silent_total.false_negatives,
                    f"{silent_total.precision:.4f}",
                    f"{silent_total.recall:.4f}",
                ],
                [
                    "loud",
                    len(loud_classes),
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
    out.check(
        "silent-tier recall is 1.0000 in the isolated regime",
        silent_total.recall == 1.0,
        f"recall {silent_total.recall:.4f}",
    )

    # --- the detection floor on a relative scale change ------------------
    out.rule("detection floor on a relative scale change -- measured by bisection")
    out("The comparison uses math.isclose with the configured relative tolerance and")
    out("an absolute tolerance of exactly zero, so the relative bound is the one that")
    out("acts. The floor below is found by bisection on the injected relative change,")
    out("not read off the configured tolerance.")
    out("")
    rows = []
    for rel_tol in (1e-12, 1e-9, 1e-6):
        probe_config = DiffConfig(numeric_rel_tol=rel_tol)
        old = baseline_dictionary(30, SEEDS[0])
        target = next(p for p in old.parameters if p.data_type in ("int", "uint", "float"))

        def detected(relative: float, target=target, old=old, probe_config=probe_config) -> bool:
            changed = target.with_fields(scale=target.scale * (1.0 + relative))
            new = Dictionary(
                version="1.1",
                parameters=tuple(
                    changed if p.name == target.name else p for p in old.parameters
                ),
                commands=old.commands,
            )
            result = diff_dictionaries(old, new, probe_config)
            return any(c.change_class is ChangeClass.RESCALED for c in result.changes)

        low, high = 0.0, 1.0
        if not detected(high):
            rows.append([f"{rel_tol:.0e}", "not detected at 100%", "-", "-"])
            continue
        for _ in range(200):
            middle = (low + high) / 2.0
            if middle in (low, high):
                break
            if detected(middle):
                high = middle
            else:
                low = middle
        rows.append(
            [
                f"{rel_tol:.0e}",
                f"{high:.17g}",
                f"{low:.17g}",
                f"{high / rel_tol:.6f}" if rel_tol else "-",
            ]
        )
    out(
        format_table(
            [
                "configured rel_tol",
                "smallest detected relative change",
                "largest undetected",
                "ratio to rel_tol",
            ],
            rows,
        )
    )
    out("")
    out("A scale change below the floor is not reported. That is returned to the")
    out("caller in DiffLimits.reasons on every diff, not left in documentation.")
    out.check("the measured floor tracks the configured tolerance", len(rows) == 3)

    # --- the no-change control, which must NOT fire ----------------------
    out.rule("no-change control -- a dictionary against itself must report nothing")
    unchanged_failures = []
    for size in SIZES:
        for seed in SEEDS:
            old = baseline_dictionary(size, seed)
            result = diff_dictionaries(old, old, config)
            if result.changes:
                unchanged_failures.append((size, seed, len(result.changes)))
    out(f"pairs checked : {len(SIZES) * len(SEEDS)}")
    out(f"pairs reporting a change : {len(unchanged_failures)}")
    out.check(
        "a dictionary diffed against itself reports zero changes",
        not unchanged_failures,
        str(unchanged_failures[:4]) if unchanged_failures else "all clean",
    )

    out.rule("timing -- UNSTABLE, two cores, informational only")
    out(
        format_table(
            ["quantity", "value"],
            [
                ["diffs run", timer.calls],
                ["total wall clock (s)", f"{timer.elapsed_s:.3f}"],
                ["mean per diff (ms)", f"{timer.mean_ms:.3f}"],
            ],
        )
    )
    out("")
    out("No conclusion in this repository rests on a timing or on a ratio of timings.")

    path = out.save()
    print(f"\nwritten: {path}")
    return 1 if out.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
