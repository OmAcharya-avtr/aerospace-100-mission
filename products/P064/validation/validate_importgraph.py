"""Check Baseline 1 against a hand-written reachability table.

The bundled sample project's import structure is small enough to write out by
hand, so this is a known-answer check and not a self-consistency check. The
table below was written from the source files, not from the graph.

    python validation/validate_importgraph.py > validation/validate_importgraph_output.txt
"""

from __future__ import annotations

import sys

from _common import SAMPLE

from regressselect.importgraph import build_import_graph

# Hand-written: for each changed source module, the test modules that import
# it directly or transitively.
EXPECTED_SELECTION = {
    "sampleproj.vecops": {"tests.test_vecops", "tests.test_quat", "tests.test_kinematics"},
    "sampleproj.stats": {"tests.test_stats", "tests.test_filters"},
    "sampleproj.integrate": {"tests.test_integrate"},
    "sampleproj.orbit": {"tests.test_orbit"},
    "sampleproj.quat": {"tests.test_quat", "tests.test_kinematics"},
    "sampleproj.filters": {"tests.test_filters"},
    "sampleproj.kinematics": {"tests.test_kinematics"},
    "sampleproj": {
        "tests.test_filters",
        "tests.test_integrate",
        "tests.test_kinematics",
        "tests.test_orbit",
        "tests.test_quat",
        "tests.test_stats",
        "tests.test_vecops",
    },
}

# Hand-written hop distances, (test module, source module) -> hops or None.
EXPECTED_DISTANCE = {
    ("tests.test_vecops", "sampleproj.vecops"): 1,
    ("tests.test_quat", "sampleproj.quat"): 1,
    ("tests.test_quat", "sampleproj.vecops"): 2,
    ("tests.test_kinematics", "sampleproj.kinematics"): 1,
    ("tests.test_kinematics", "sampleproj.vecops"): 1,
    ("tests.test_kinematics", "sampleproj.quat"): 2,
    ("tests.test_filters", "sampleproj.filters"): 1,
    ("tests.test_filters", "sampleproj.stats"): 2,
    ("tests.test_filters", "sampleproj.vecops"): None,
    ("tests.test_orbit", "sampleproj.quat"): None,
    ("tests.test_stats", "sampleproj.integrate"): None,
}


def main() -> int:
    out = sys.stdout
    graph = build_import_graph(SAMPLE)
    failures = 0

    out.write("Baseline 1: import-graph reachability, known-answer check\n")
    out.write("=" * 78 + "\n")
    out.write(f"project root      {SAMPLE.name}\n")
    out.write(f"modules parsed    {len(graph.modules)}\n")
    out.write(f"test modules      {len(graph.test_modules)}\n\n")

    out.write("selection per changed module (hand-written table vs computed)\n")
    out.write(f"{'changed module':<24}{'selected':>9}{'of':>4}{'fraction':>10}  verdict\n")
    out.write("-" * 78 + "\n")
    n_tests = len(graph.test_modules)
    for module, expected in EXPECTED_SELECTION.items():
        got = graph.select_tests([module])
        ok = got == expected
        failures += 0 if ok else 1
        out.write(
            f"{module:<24}{len(got):>9}{n_tests:>4}{len(got) / n_tests:>10.4f}  "
            f"{'PASS' if ok else 'FAIL'}\n"
        )
        if not ok:
            out.write(f"    expected {sorted(expected)}\n    got      {sorted(got)}\n")

    out.write("\nhop distances (hand-written table vs computed)\n")
    out.write(f"{'test module':<24}{'source module':<24}{'expected':>9}{'got':>6}  verdict\n")
    out.write("-" * 78 + "\n")
    for (test_mod, src_mod), expected_d in EXPECTED_DISTANCE.items():
        got_d = graph.distance(test_mod, src_mod)
        ok = got_d == expected_d
        failures += 0 if ok else 1
        out.write(
            f"{test_mod:<24}{src_mod:<24}{str(expected_d):>9}{str(got_d):>6}  "
            f"{'PASS' if ok else 'FAIL'}\n"
        )

    mean_fraction = sum(
        len(graph.select_tests([m])) / n_tests for m in EXPECTED_SELECTION
    ) / len(EXPECTED_SELECTION)
    out.write(
        f"\nmean selection fraction over the {len(EXPECTED_SELECTION)} single-module changes "
        f"above: {mean_fraction:.4f}\n"
        "A change to sampleproj/__init__.py selects 7/7 test modules, which is the\n"
        "behaviour a dependency-based selector shows on any module that everything\n"
        "imports. The sample project has seven largely independent leaf modules and\n"
        "no shared utility hub, so its mean is well below 1; a repository with a\n"
        "common helper module does not have that property.\n"
    )

    out.write(f"\nchecks run {len(EXPECTED_SELECTION) + len(EXPECTED_DISTANCE)}, ")
    out.write(f"failures {failures}\n")
    out.write("RESULT: " + ("PASS\n" if failures == 0 else f"FAIL ({failures})\n"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
