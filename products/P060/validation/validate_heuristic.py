"""Measure the assertion heuristic against hand labels, and report it badly.

This is the only check in this package that can be wrong about a correct
input, so the numbers here are the ones a reader should look at before
trusting finding TA006. The corpus is 23 hand-labelled functions in
fixtures/heuristic/; labels.json records the reason for each label.

Exits 0. The measured error rates are an output, not a gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import index_test_files  # noqa: E402

CORPUS = ROOT / "fixtures" / "heuristic"
FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    if not ok:
        FAILURES.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<50s} got {got!r}   expected {expected!r}")


def main() -> int:
    labels = json.loads((CORPUS / "labels.json").read_text(encoding="utf-8"))["labels"]
    index, notes = index_test_files(CORPUS)
    verdicts = {
        name: verdict for (stem, name), verdict in index.items()
        if stem and name.startswith("test_")
    }

    print("1. corpus and labels")
    check("labelled functions", len(labels), 23)
    check("inspected test functions", len(verdicts), 23)
    check("label set equals corpus set", sorted(labels) == sorted(verdicts), True)
    truly_empty = [n for n, lab in labels.items() if not lab["asserts_something"]]
    check("hand-labelled as verifying nothing", len(truly_empty), 8)
    check("hand-labelled as verifying something", 23 - len(truly_empty), 15)

    print()
    print("2. per-function verdicts (positive class = flagged as having no assertion)")
    print(f"   {'function':<44s} {'label':<14s} {'verdict':<14s} cell  rules")
    cells = {"tp": [], "fp": [], "fn": [], "tn": []}
    for name in sorted(labels):
        empty = not labels[name]["asserts_something"]
        flagged = not verdicts[name].has_assertion
        cell = "tp" if (flagged and empty) else "fp" if flagged else "fn" if empty else "tn"
        cells[cell].append(name)
        print(f"   {name:<44s} {'empty' if empty else 'asserts':<14s} "
              f"{'flagged' if flagged else 'not flagged':<14s} {cell:<5s} "
              f"{','.join(verdicts[name].rules_fired)}")

    tp, fp = len(cells["tp"]), len(cells["fp"])
    fn, tn = len(cells["fn"]), len(cells["tn"])
    total = tp + fp + fn + tn

    print()
    print("3. confusion matrix")
    print("                       truly empty   truly asserts")
    print(f"   flagged empty       {tp:>11d}   {fp:>13d}")
    print(f"   not flagged         {fn:>11d}   {tn:>13d}")
    check("true positives", tp, 5)
    check("false positives", fp, 3)
    check("false negatives", fn, 3)
    check("true negatives", tn, 12)
    check("total", total, 23)

    print()
    print("4. rates, stated the unflattering way")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fp_rate = fp / (fp + tn)
    fn_rate = fn / (fn + tp)
    print(f"   precision                     {precision:.6f}  "
          f"({tp} of {tp + fp} flags were right)")
    print(f"   recall                        {recall:.6f}  "
          f"({tp} of {tp + fn} empty tests were caught)")
    print(f"   false-positive rate           {fp_rate:.6f}  "
          f"({fp} of {fp + tn} asserting tests were wrongly flagged)")
    print(f"   false-negative rate           {fn_rate:.6f}  "
          f"({fn} of {fn + tp} empty tests were missed)")
    print(f"   accuracy                      {(tp + tn) / total:.6f}")
    print("   Read that as: on this corpus, more than a third of the empty tests")
    print("   are missed and three flags in eight are wrong. TA006 is a prompt to look,")
    print("   not a result.")

    print()
    print("5. the error modes, named")
    print("   false positives, all three of the same kind -- the assertion is one")
    print("   call away and the heuristic does not follow calls:")
    for name in sorted(cells["fp"]):
        print(f"     {name:<44s} {labels[name]['why']}")
    print("   false negatives, all three of the same kind -- the assert statement")
    print("   exists and is syntactically non-constant, but is vacuous:")
    for name in sorted(cells["fn"]):
        print(f"     {name:<44s} {labels[name]['why']}")

    print()
    print("6. what would move these numbers")
    print("   following one level of call into a module-level helper would remove")
    print("   two of the three false positives and would introduce a new error mode")
    print("   (a helper that only logs). Catching the three false negatives needs")
    print("   data flow, not syntax. Neither is implemented, and the heuristic is")
    print("   labelled as a heuristic everywhere it appears.")
    for note in notes:
        print(f"   note: {note}")

    print()
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
