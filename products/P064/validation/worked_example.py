"""The worked example the README quotes, run end to end.

One declared change -- 31 lines in sampleproj/quat.py -- put through all three
strategies, with the selection, the cost and the probabilities the learned
selector attaches.

    python validation/worked_example.py > validation/worked_example_output.txt
"""

from __future__ import annotations

import sys

from _common import HISTORY_JSON, SAMPLE, TRAIN_FRACTION

from regressselect.benchmark import dependency_order
from regressselect.history import Revision, load_history
from regressselect.importgraph import build_import_graph
from regressselect.metrics import selection_cost
from regressselect.prioritise import CoFailurePrioritiser
from regressselect.selector import LearnedSelector

BUDGET = 0.25


def main() -> int:
    out = sys.stdout
    graph = build_import_graph(SAMPLE)
    history = load_history(HISTORY_JSON)
    train, _ = history.split_chronological(TRAIN_FRACTION)
    nodes = list(history.test_ids)

    change = Revision(
        revision="working-tree",
        changed_modules=("sampleproj.quat",),
        churn={"sampleproj.quat": 31},
    )

    out.write("Worked example: 31 lines changed in sampleproj/quat.py\n")
    out.write("=" * 78 + "\n")
    out.write(f"suite size          {len(nodes)} test nodes\n")
    out.write(f"training history    {len(train)} revisions\n")
    out.write(f"budget              {BUDGET:.2f} of the suite\n\n")

    _, selected = dependency_order(graph, change, nodes)
    out.write("Baseline 1, import-graph reachability (safe, budget does not apply)\n")
    out.write(
        f"  selects {len(selected)}/{len(nodes)} nodes, cost "
        f"{selection_cost(selected, nodes):.4f}\n"
    )
    for node in sorted(selected):
        out.write(f"    {node}\n")

    prioritiser = CoFailurePrioritiser().fit(train)
    ranked_history = prioritiser.rank(change.changed_modules, nodes)
    k = max(1, round(BUDGET * len(nodes)))
    out.write(f"\nBaseline 2, co-failure history: top {k} of {len(nodes)}\n")
    for item in ranked_history[:k]:
        out.write(f"    {item.score:.6f}  {item.node_id}\n")

    selector = LearnedSelector().fit(train, graph)
    ranked_learned = selector.rank(change, nodes)
    out.write(f"\nLearned selector: top {k} of {len(nodes)}, score is P(this test fails)\n")
    for item in ranked_learned[:k]:
        out.write(f"    {item.score:.6f}  {item.node_id}\n")

    dep_modules = {n.split("::")[0] for n in selected}
    hist_modules = {item.node_id.split("::")[0] for item in ranked_history[:k]}
    learn_modules = {item.node_id.split("::")[0] for item in ranked_learned[:k]}
    out.write("\ntest files touched by each selection\n")
    out.write(f"  Baseline 1   {sorted(dep_modules)}\n")
    out.write(f"  Baseline 2   {sorted(hist_modules)}\n")
    out.write(f"  learned      {sorted(learn_modules)}\n")
    agree = dep_modules == hist_modules == learn_modules
    out.write(
        "\nBaseline 1 picks the two test files that import quat, directly or through\n"
        "kinematics, and nothing else.\n"
    )
    if agree:
        out.write(
            "On this change all three strategies land on the same two test files, so the\n"
            f"only difference is cost: Baseline 1 runs {len(selected)} nodes and the two\n"
            f"ranked strategies run {k} each at the {BUDGET:.2f} budget. The ranked\n"
            "strategies drop two of the eight reachable nodes to get there, and whether\n"
            "that is a good trade is the safety-loss question; the measured answer over\n"
            "the held-out revisions is in validate_benchmark_output.txt, and there the\n"
            "dependency baseline has the lower loss.\n"
        )
    else:
        out.write(
            "The ranked strategies pick a different set of files, so the comparison on\n"
            "this change is about direction as well as cost; see\n"
            "validate_benchmark_output.txt.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
