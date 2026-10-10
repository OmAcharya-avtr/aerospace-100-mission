"""Measure the assertion heuristic against the hand-labelled corpus.

Writes ../screenshots/heuristic_confusion.png. The positive class is "the
heuristic says this test has no assertion".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import index_test_files  # noqa: E402
from traceaudit.plotting import plot_confusion  # noqa: E402

CORPUS = ROOT / "fixtures" / "heuristic"


def main() -> int:
    labels = json.loads((CORPUS / "labels.json").read_text(encoding="utf-8"))["labels"]
    index, notes = index_test_files(CORPUS)
    verdicts = {
        name: verdict
        for (stem, name), verdict in index.items()
        if stem and name.startswith("test_")
    }
    missing = sorted(set(labels) - set(verdicts))
    extra = sorted(set(verdicts) - set(labels))
    if missing or extra:
        print(f"label/corpus mismatch: missing {missing}, unlabelled {extra}")
        return 1

    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    rows = []
    for name in sorted(labels):
        truly_empty = not labels[name]["asserts_something"]
        flagged = not verdicts[name].has_assertion
        if flagged and truly_empty:
            cell = "tp"
        elif flagged:
            cell = "fp"
        elif truly_empty:
            cell = "fn"
        else:
            cell = "tn"
        counts[cell] += 1
        rows.append((name, truly_empty, flagged, cell, verdicts[name].rules_fired))

    print(f"{'test function':<42s} {'empty?':<7s} {'flagged?':<9s} cell  rules")
    for name, truly_empty, flagged, cell, rules in rows:
        print(f"{name:<42s} {str(truly_empty):<7s} {str(flagged):<9s} {cell:<5s} "
              f"{','.join(rules)}")

    total = sum(counts.values())
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    fp_rate = fp / (fp + tn) if fp + tn else float("nan")
    fn_rate = fn / (fn + tp) if fn + tp else float("nan")
    print()
    print(f"corpus size                     : {total}")
    print(f"truly empty / truly asserting   : {tp + fn} / {fp + tn}")
    print(f"tp {tp}  fp {fp}  fn {fn}  tn {tn}")
    print(f"precision                       : {precision:.4f}")
    print(f"recall                          : {recall:.4f}")
    print(f"false-positive rate             : {fp_rate:.4f}"
          "   (truly asserting tests wrongly flagged)")
    print(f"false-negative rate             : {fn_rate:.4f}"
          "   (truly empty tests missed)")
    print(f"accuracy                        : {(tp + tn) / total:.4f}")
    for note in notes:
        print(f"note: {note}")

    subtitle = (f"n = {total}   precision {precision:.3f}   recall {recall:.3f}   "
                f"FP rate {fp_rate:.3f}   FN rate {fn_rate:.3f}")
    out = plot_confusion(
        counts,
        ROOT / "screenshots" / "heuristic_confusion.png",
        title="assertion heuristic vs hand labels: it is a heuristic",
        subtitle=subtitle,
    )
    print()
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
