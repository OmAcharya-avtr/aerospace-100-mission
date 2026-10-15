"""Apply Baseline 1 to regressselect's own source tree, with and without
pytest's implicit conftest dependency.

The bundled sample project has seven independent leaf modules and no
``conftest.py``, so the dependency baseline selects well under the whole suite
on it. That is a property of the fixture. This script measures the same
baseline on this repository, which has a ``tests/conftest.py`` that imports
five of the twelve source modules -- the structure a real suite has -- and
reports the selection fraction twice: from the import text alone, and with the
conftest edges pytest actually imposes.

The gap between the two columns is a soundness gap in any selector built from
imports alone, this one included.

    python validation/validate_self_selection.py \\
        > validation/validate_self_selection_output.txt
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from _common import ROOT

from regressselect.importgraph import build_import_graph


def main() -> int:
    out = sys.stdout
    # The package lives under src/ and the suite under tests/, so neither is a
    # subdirectory of the other. Stage a tree in which both sit at the root,
    # which is the layout module names are resolved against.
    staging = Path(tempfile.mkdtemp(prefix="regressselect_self_"))
    try:
        shutil.copytree(ROOT / "src" / "regressselect", staging / "regressselect")
        shutil.copytree(
            ROOT / "tests",
            staging / "tests",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        plain = build_import_graph(staging)
        coupled = plain.with_conftest_coupling()
        n_tests = len(plain.test_modules)
        source_modules = sorted(
            m
            for m in plain.modules
            if m not in plain.test_modules and m.rsplit(".", 1)[-1] != "conftest"
        )

        out.write("Baseline 1 applied to regressselect's own source tree\n")
        out.write("=" * 84 + "\n")
        out.write(f"source modules              {len(source_modules)}\n")
        out.write(f"selectable test modules     {n_tests}\n")
        out.write("conftest imports:\n")
        for module in sorted(plain.imports_of("tests.conftest")):
            out.write(f"  {module}\n")
        out.write("\n")
        out.write(
            f"{'changed module':<30}{'imports only':>14}{'with conftest':>15}"
            f"{'increase':>11}\n"
        )
        out.write("-" * 84 + "\n")
        plain_fractions: list[float] = []
        coupled_fractions: list[float] = []
        for module in source_modules:
            a = len(plain.select_tests([module])) / n_tests
            b = len(coupled.select_tests([module])) / n_tests
            plain_fractions.append(a)
            coupled_fractions.append(b)
            out.write(f"{module:<30}{a:>14.4f}{b:>15.4f}{b - a:>11.4f}\n")

        plain_whole = sum(1 for f in plain_fractions if f > 0.999)
        coupled_whole = sum(1 for f in coupled_fractions if f > 0.999)
        out.write(
            f"\n{'mean':<30}"
            f"{sum(plain_fractions) / len(plain_fractions):>14.4f}"
            f"{sum(coupled_fractions) / len(coupled_fractions):>15.4f}\n"
            f"{'max':<30}{max(plain_fractions):>14.4f}{max(coupled_fractions):>15.4f}\n"
            f"{'modules selecting everything':<30}"
            f"{plain_whole:>14d}{coupled_whole:>15d}\n"
        )

        features_idx = source_modules.index("regressselect.features")
        features_gap = coupled_fractions[features_idx] - plain_fractions[features_idx]
        out.write(
            "\nWhat this measures\n"
            "------------------\n"
            "From the import text alone, a single-module change on this repository\n"
            f"selects a mean {sum(plain_fractions) / len(plain_fractions):.4f} of the suite "
            "and no single module selects all of it.\n"
            "That does NOT support the common claim that a dependency-based selector is\n"
            "near-useless because it selects almost everything.\n"
            "\n"
            "With pytest's conftest dependency included -- which is a real dependency, not\n"
            f"a modelling choice -- the mean rises to "
            f"{sum(coupled_fractions) / len(coupled_fractions):.4f} and {coupled_whole} of "
            f"the {len(source_modules)} source modules\n"
            "select every test module, because conftest imports them. That is where the\n"
            "claim comes from, and it holds only once the fixture coupling is counted.\n"
            "\n"
            "The consequence for this product is the one worth stating: a selector built\n"
            "from imports alone is not safe for a pytest suite. For regressselect.features\n"
            f"the import-only selection misses {features_gap:.4f} of the suite -- tests "
            "that depend\n"
            "on it only through a fixture.\n"
        )

        checks = [
            ("every source module selects at least one test module", min(plain_fractions) > 0.0),
            (
                "conftest coupling never shrinks a selection",
                all(
                    b >= a - 1e-12
                    for a, b in zip(plain_fractions, coupled_fractions, strict=True)
                ),
            ),
            (
                "conftest coupling grows at least one selection to the whole suite",
                coupled_whole >= 1,
            ),
            (
                "a module imported by conftest reaches the whole suite once coupled",
                coupled_fractions[features_idx] > 0.999,
            ),
        ]
        out.write("\nchecks\n")
        out.write("-" * 84 + "\n")
        failures = 0
        for label, ok in checks:
            failures += 0 if ok else 1
            out.write(f"{label:<70}{'PASS' if ok else 'FAIL'}\n")
        out.write(f"\nchecks run {len(checks)}, failures {failures}\n")
        out.write("RESULT: " + ("PASS\n" if failures == 0 else f"FAIL ({failures})\n"))
        return 0 if failures == 0 else 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
