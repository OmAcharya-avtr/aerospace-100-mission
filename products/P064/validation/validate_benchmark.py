"""Score all three strategies on the committed fixture history.

This is the script the headline numbers in README.md, VALIDATION.md and
MODEL_CARD.md come from. It reports APFD and safety loss for the two baselines
and the learned selector, including the equal-cost comparison in which the
dependency baseline wins, and it does not soften that result.

    python validation/validate_benchmark.py > validation/validate_benchmark_output.txt
"""

from __future__ import annotations

import json
import sys
import time

from _common import HISTORY_JSON, SAMPLE, TRAIN_FRACTION

from regressselect.benchmark import dependency_order, evaluate_strategies
from regressselect.history import load_history
from regressselect.importgraph import build_import_graph
from regressselect.metrics import apfd_single_fault
from regressselect.selector import LearnedSelector

BUDGETS = (0.10, 0.25, 0.4167, 0.50, 0.75, 1.00)


def main() -> int:
    out = sys.stdout
    started = time.monotonic()
    history = load_history(HISTORY_JSON)
    graph = build_import_graph(SAMPLE)
    train, test = history.split_chronological(TRAIN_FRACTION)
    nodes = list(history.test_ids)
    selector = LearnedSelector().fit(train, graph)
    scores = evaluate_strategies(train, test, graph, budgets=BUDGETS, selector=selector)
    regressions = [r for r in test.revisions if r.is_regression]

    params = json.loads(
        (HISTORY_JSON.parent / "generation_parameters.json").read_text(encoding="utf-8")
    )

    out.write("Three-strategy benchmark on the committed fixture history\n")
    out.write("=" * 86 + "\n")
    out.write(f"history                {HISTORY_JSON.name}, seed {history.seed}\n")
    out.write(
        f"revisions              {len(history)} total; "
        f"{len(train)} train, {len(test)} test\n"
    )
    out.write(f"split                  chronological at {TRAIN_FRACTION:.2f}\n")
    out.write(f"test nodes             {len(nodes)}\n")
    out.write(f"evaluation revisions   {len(regressions)} of {len(test)} have a real failure\n")
    out.write(
        f"declared noise         out-of-closure coupling "
        f"{params['out_of_closure_coupling_rate']}, flake "
        f"{params['flake_rate_per_test_per_revision']} per test per revision\n"
    )
    out.write(
        f"training rows          {selector.n_train_rows} "
        f"({selector.n_train_positives} positives, "
        f"{selector.n_train_positives / selector.n_train_rows:.4f})\n\n"
    )

    out.write("APFD over the full order, averaged over the evaluation revisions\n")
    out.write("-" * 86 + "\n")
    lexical = sorted(nodes)
    lex = sum(apfd_single_fault(lexical, r.failed_tests) for r in regressions) / len(regressions)
    out.write(f"{'lexical control (no information)':<40}{lex:>12.6f}\n")
    for name in ("dependency", "history", "learned"):
        out.write(f"{name:<40}{scores[name].apfd_full_order:>12.6f}\n")

    out.write("\nsafety loss and cost against budget\n")
    out.write(
        f"{'strategy':<12}{'budget':>9}{'cost':>9}{'safety loss':>13}"
        f"{'caught':>9}{'revisions':>11}\n"
    )
    out.write("-" * 86 + "\n")
    for name in ("dependency", "history", "learned"):
        for point in scores[name].points:
            out.write(
                f"{name:<12}{point.budget:>9.4f}{point.tests_run_fraction:>9.4f}"
                f"{point.safety_loss:>13.6f}{point.regressions_caught:>9.4f}"
                f"{point.revisions_scored:>11d}\n"
            )

    dep_cost = scores["dependency"].points[0].tests_run_fraction
    matched = evaluate_strategies(train, test, graph, budgets=(dep_cost,), selector=selector)
    out.write(
        f"\nequal-cost comparison at the dependency baseline's measured cost "
        f"{dep_cost:.4f}\n"
    )
    out.write(f"{'strategy':<12}{'cost':>9}{'safety loss':>13}{'APFD':>12}  verdict\n")
    out.write("-" * 86 + "\n")
    best = min(
        ("dependency", "history", "learned"),
        key=lambda n: matched[n].points[0].safety_loss,
    )
    for name in ("dependency", "history", "learned"):
        point = matched[name].points[0]
        out.write(
            f"{name:<12}{point.tests_run_fraction:>9.4f}{point.safety_loss:>13.6f}"
            f"{matched[name].apfd_full_order:>12.6f}"
            f"{'  <- lowest safety loss' if name == best else ''}\n"
        )

    out.write("\nper-revision dependency-baseline cost over the whole history\n")
    out.write("-" * 86 + "\n")
    costs = []
    for rev in history.revisions:
        _, selected = dependency_order(graph, rev, nodes)
        costs.append(len(selected) / len(nodes))
    whole = sum(1 for c in costs if c > 0.999)
    out.write(
        f"mean {sum(costs) / len(costs):.4f}  min {min(costs):.4f}  max {max(costs):.4f}  "
        f"revisions selecting the whole suite {whole}/{len(costs)} "
        f"({whole / len(costs):.4f})\n"
    )

    out.write("\nlearned selector diagnostics\n")
    out.write("-" * 86 + "\n")
    out.write(f"Brier score, training slice   {selector.brier_score(train):.6f}\n")
    out.write(f"Brier score, held-out slice   {selector.brier_score(test):.6f}\n")
    out.write("feature importances, descending\n")
    for name, value in sorted(selector.feature_importances().items(), key=lambda kv: -kv[1]):
        out.write(f"  {name:<28}{value:.4f}\n")

    out.write("\nFINDINGS, stated as measured\n")
    out.write("-" * 86 + "\n")
    dep_apfd = scores["dependency"].apfd_full_order
    learn_apfd = scores["learned"].apfd_full_order
    hist_apfd = scores["history"].apfd_full_order
    dep_loss = matched["dependency"].points[0].safety_loss
    learn_loss = matched["learned"].points[0].safety_loss
    imp = selector.feature_importances()
    graph_weight = imp["distance_hops"] + imp["inverse_distance"] + imp["reachable"]
    out.write(
        f"1. The dependency baseline has the highest APFD ({dep_apfd:.6f}) and, at matched\n"
        f"   cost, the lowest safety loss ({dep_loss:.6f} against {learn_loss:.6f} for the\n"
        f"   learned selector). Baseline 1 wins this comparison outright.\n"
        f"2. The learned selector beats the history baseline on APFD ({learn_apfd:.6f}\n"
        f"   against {hist_apfd:.6f}) and on safety loss at every budget, so the model is\n"
        f"   not a no-op; it is just not better than the safe baseline.\n"
        f"3. The dependency baseline's safety loss is "
        f"{scores['dependency'].points[0].safety_loss:.6f}, not zero.\n"
        f"   Safe by construction means safe with respect to the declared import graph;\n"
        f"   the fixture history declares two channels the graph cannot see.\n"
        f"4. The learned selector's Brier score roughly doubles from the training slice\n"
        f"   to the held-out slice, so its probabilities are over-confident out of sample.\n"
        f"5. Import distance and reachability carry almost no weight in the fitted model\n"
        f"   (combined importance {graph_weight:.4f}); it keys on churn and co-failure.\n"
        f"   That is the mechanical reason it is less safe than the baseline that uses\n"
        f"   nothing but the dependency structure.\n"
    )

    checks = [
        ("dependency APFD > lexical control", dep_apfd > lex),
        ("history APFD > lexical control", hist_apfd > lex),
        ("learned APFD > lexical control", learn_apfd > lex),
        ("learned APFD > history APFD", learn_apfd > hist_apfd),
        (
            "safety loss is 0 at full budget for both ranked strategies",
            abs(scores["history"].points[-1].safety_loss) <= 1e-12
            and abs(scores["learned"].points[-1].safety_loss) <= 1e-12,
        ),
        (
            "safety loss is non-increasing in the budget",
            all(
                b <= a + 1e-12
                for name in ("history", "learned")
                for a, b in zip(
                    [p.safety_loss for p in scores[name].points],
                    [p.safety_loss for p in scores[name].points][1:],
                    strict=False,
                )
            ),
        ),
    ]
    out.write("\nchecks\n")
    out.write("-" * 86 + "\n")
    failures = 0
    for label, ok in checks:
        failures += 0 if ok else 1
        out.write(f"{label:<60}{'PASS' if ok else 'FAIL'}\n")
    out.write(f"\nwall clock {time.monotonic() - started:.1f} s\n")
    out.write(f"checks run {len(checks)}, failures {failures}\n")
    out.write("RESULT: " + ("PASS\n" if failures == 0 else f"FAIL ({failures})\n"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
