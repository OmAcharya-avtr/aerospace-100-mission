"""Regenerate every committed fixture in this directory, deterministically.

Run from the product root:

    python fixtures/regenerate_fixtures.py

Writes:

  fixtures/sample_project/junit.xml     a clean full-suite run of the sample project
  fixtures/faults/fault_matrix.json     measured fault -> detecting-nodes matrix
  fixtures/history/history.json         synthetic revision history
  fixtures/history/generation_parameters.json  the declared rates behind it

The history is SYNTHETIC. It is generated from the sample project's real
import graph plus a declared fault-proneness, flakiness and out-of-closure
coupling model, with a fixed seed. It is not a mined history of a real
repository, and every number computed from it inherits that limitation. The
generation parameters are written out beside it so that any claim made about
the history can be checked against the model that produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from regressselect.history import History, Revision, save_history  # noqa: E402
from regressselect.importgraph import build_import_graph  # noqa: E402
from regressselect.inventory import collect_tests, module_of_node  # noqa: E402
from regressselect.mutation import load_faults, run_mutation_evaluation, run_suite  # noqa: E402

SEED = 20261015
N_REVISIONS = 320
TRAIN_FRACTION = 0.70

# Declared generation model. Every rate below is a probability per revision or
# per (revision, test) pair, dimensionless, and is written to
# generation_parameters.json so the history is auditable.
CHANGE_WEIGHT = {
    "sampleproj.vecops": 0.22,
    "sampleproj.quat": 0.18,
    "sampleproj.kinematics": 0.15,
    "sampleproj.stats": 0.12,
    "sampleproj.filters": 0.12,
    "sampleproj.integrate": 0.10,
    "sampleproj.orbit": 0.08,
    "sampleproj": 0.03,
}
FAULT_PRONENESS = {
    "sampleproj.vecops": 0.30,
    "sampleproj.quat": 0.40,
    "sampleproj.kinematics": 0.45,
    "sampleproj.stats": 0.25,
    "sampleproj.filters": 0.35,
    "sampleproj.integrate": 0.20,
    "sampleproj.orbit": 0.15,
    "sampleproj": 0.10,
}
CHURN_MIN, CHURN_MAX = 2, 80
DISTANCE_DECAY = 0.90
COUPLING_RATE = 0.010
FLAKE_RATE = 0.003
N_CHANGED_CHOICES = (1, 1, 1, 2, 2, 3)


def build_history(sample_root: Path) -> History:
    graph = build_import_graph(sample_root)
    inventory = collect_tests(sample_root)
    rng = np.random.default_rng(SEED)
    nodes = list(inventory.node_ids)
    sensitivity = {n: float(rng.uniform(0.35, 0.95)) for n in nodes}
    modules = sorted(CHANGE_WEIGHT)
    weights = np.array([CHANGE_WEIGHT[m] for m in modules], dtype=float)
    weights /= weights.sum()

    # Precompute, per test node, the hop distance to each source module.
    distance: dict[tuple[str, str], int | None] = {}
    for node in nodes:
        test_mod = module_of_node(node)
        for mod in modules:
            distance[(node, mod)] = graph.distance(test_mod, mod)

    revisions: list[Revision] = []
    for i in range(N_REVISIONS):
        k = int(rng.choice(N_CHANGED_CHOICES))
        changed = list(rng.choice(modules, size=k, replace=False, p=weights))
        churn = {m: int(rng.integers(CHURN_MIN, CHURN_MAX + 1)) for m in changed}
        p_clean = 1.0
        for m in changed:
            severity = 0.30 + 0.70 * (churn[m] - CHURN_MIN) / (CHURN_MAX - CHURN_MIN)
            p_clean *= 1.0 - FAULT_PRONENESS[m] * severity
        failing: set[str] = set()
        if rng.random() < 1.0 - p_clean:
            fp = np.array([FAULT_PRONENESS[m] for m in changed], dtype=float)
            broken = str(rng.choice(changed, p=fp / fp.sum()))
            for node in nodes:
                d = distance[(node, broken)]
                p = (
                    sensitivity[node] * DISTANCE_DECAY ** max(d - 1, 0)
                    if d is not None
                    else COUPLING_RATE
                )
                if rng.random() < p:
                    failing.add(node)
            if not failing:
                reachable = [n for n in nodes if distance[(n, broken)] is not None]
                pick = max(reachable, key=lambda n: sensitivity[n]) if reachable else nodes[0]
                failing.add(pick)
        for node in nodes:
            if rng.random() < FLAKE_RATE:
                failing.add(node)
        revisions.append(
            Revision(
                revision=f"r{i + 1:04d}",
                changed_modules=tuple(changed),
                churn=churn,
                failed_tests=tuple(sorted(failing)),
            )
        )
    return History(
        test_ids=tuple(nodes),
        revisions=tuple(revisions),
        project_root="fixtures/sample_project",
        seed=SEED,
    )


def main() -> int:
    sample_root = HERE / "sample_project"

    clean = run_suite(sample_root, junit_path=sample_root / "junit.xml")
    if clean.failing_nodes or not clean.cross_check():
        raise SystemExit(
            f"sample project is not green: tests={clean.tests} failures={clean.failures} "
            f"errors={clean.errors} cross_check={clean.cross_check()}"
        )

    faults = load_faults(HERE / "faults" / "faults.json")
    matrix = run_mutation_evaluation(sample_root, faults)
    (HERE / "faults" / "fault_matrix.json").write_text(matrix.to_json(), encoding="utf-8")

    history = build_history(sample_root)
    save_history(history, HERE / "history" / "history.json")
    params = {
        "schema": "regressselect-historyparams/1",
        "seed": SEED,
        "n_revisions": N_REVISIONS,
        "train_fraction": TRAIN_FRACTION,
        "change_weight": CHANGE_WEIGHT,
        "fault_proneness": FAULT_PRONENESS,
        "churn_range_lines": [CHURN_MIN, CHURN_MAX],
        "distance_decay": DISTANCE_DECAY,
        "out_of_closure_coupling_rate": COUPLING_RATE,
        "flake_rate_per_test_per_revision": FLAKE_RATE,
        "n_changed_modules_choices": list(N_CHANGED_CHOICES),
        "note": (
            "Synthetic. Failures are drawn from the sample project's real import graph "
            "plus the declared rates above. out_of_closure_coupling_rate and "
            "flake_rate_per_test_per_revision are the two channels that make the "
            "dependency-based baseline measurably unsafe; both are declared, not hidden."
        ),
    }
    (HERE / "history" / "generation_parameters.json").write_text(
        json.dumps(params, indent=1) + "\n", encoding="utf-8"
    )

    sys.stdout.write(
        f"sample project junit: tests={clean.tests} failures={clean.failures} "
        f"errors={clean.errors} skipped={clean.skipped} "
        f"testcase_elements={clean.testcase_count}\n"
        f"fault matrix: {len(matrix.fault_ids)} faults, "
        f"{len(matrix.detected_fault_ids)} detected by the full suite "
        f"({matrix.full_suite_detection():.4f})\n"
        f"history: {len(history)} revisions, {history.regression_count} with at least "
        f"one failing test, {len(history.test_ids)} test nodes\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
