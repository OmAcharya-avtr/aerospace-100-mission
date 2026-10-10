"""Coverage arithmetic, hand-checked, with every denominator written out.

Also measures the one property that matters for a coverage number: that the
three definitions form a chain, on randomly generated inputs.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traceaudit import (  # noqa: E402
    Outcome,
    TestCase,
    TestReport,
    build_matrix,
    parse_junit_xml,
    parse_requirements_file,
    parse_requirements_text,
)

SAMPLE = ROOT / "fixtures" / "sample_project"
SEED = 60011
FAILURES: list[str] = []


def check(label: str, got, expected) -> None:
    ok = got == expected
    if not ok:
        FAILURES.append(label)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<52s} got {got!r}   expected {expected!r}")


def main() -> int:
    print("1. the sample project, hand arithmetic")
    print("   denominator = 8 unique declared ids (REQ-007 is declared twice and")
    print("   counts once). Numerators read off the fixture by hand:")
    print("     nominal  001 002 003 004 006 007 008        -> 7")
    print("     executed 001 002 006 007 008                -> 5")
    print("     passing  001 002 006 007                    -> 4")
    document = parse_requirements_file(SAMPLE / "docs" / "REQUIREMENTS.md", relative_to=SAMPLE)
    report = parse_junit_xml(SAMPLE / "junit.xml", relative_to=SAMPLE)
    matrix = build_matrix(document, report)
    figures = {f.label: f for f in matrix.coverage()}
    check("nominal numerator/denominator",
          (figures["nominal coverage"].numerator, figures["nominal coverage"].denominator),
          (7, 8))
    check("executed numerator/denominator",
          (figures["executed coverage"].numerator, figures["executed coverage"].denominator),
          (5, 8))
    check("passing numerator/denominator",
          (figures["passing coverage"].numerator, figures["passing coverage"].denominator),
          (4, 8))
    for label in ("nominal coverage", "executed coverage", "passing coverage"):
        print(f"       {figures[label].render()}")
    check("nominal percent", round(figures["nominal coverage"].percent, 4), 87.5)
    check("executed percent", round(figures["executed coverage"].percent, 4), 62.5)
    check("passing percent", round(figures["passing coverage"].percent, 4), 50.0)

    print()
    print("2. the headline overstatement this product exists to refuse")
    nominal = figures["nominal coverage"].percent
    passing = figures["passing coverage"].percent
    print(f"   quoting nominal coverage alone: {nominal:.4f} %")
    print(f"   quoting passing coverage      : {passing:.4f} %")
    print(f"   difference                    : {nominal - passing:.4f} percentage points")
    check("the overstatement is 37.5 points on this fixture",
          round(nominal - passing, 4), 37.5)

    print()
    print("3. a zero denominator is reported as undefined, not as zero or as 100")
    empty = build_matrix(parse_requirements_text("", source="s.md"),
                         TestReport(cases=(), source="j.xml", kind="junit-xml"))
    check("denominator", empty.coverage()[0].denominator, 0)
    check("percent", empty.coverage()[0].percent, None)
    check("rendered", "undefined" in empty.coverage()[0].render(), True)

    print()
    print(f"4. the coverage chain on random inputs, seed {SEED}")
    print("   nominal >= executed >= passing must hold for every input")
    rng = random.Random(SEED)
    outcomes = [Outcome.PASSED, Outcome.FAILED, Outcome.ERRORED,
                Outcome.SKIPPED, Outcome.XFAILED]
    violations = 0
    strict_gaps = 0
    trials = 2000
    for _ in range(trials):
        n_reqs = rng.randint(1, 10)
        ids = rng.sample(range(1, 40), n_reqs)
        doc = parse_requirements_text(
            "".join(f"## REQ-{i:03d} R\n" for i in ids), source="s.md")
        cases = []
        for k in range(rng.randint(0, 8)):
            claims = tuple(f"REQ-{i:03d}" for i in rng.sample(range(1, 40),
                                                              rng.randint(0, 3)))
            cases.append(TestCase(node_id=f"t::{k}", classname="t", name=f"t{k}",
                                  outcome=rng.choice(outcomes), claims=claims))
        m = build_matrix(doc, TestReport(cases=tuple(cases), source="j", kind="junit-xml"))
        f = {x.label: x.numerator for x in m.coverage()}
        if not (f["nominal coverage"] >= f["executed coverage"]
                >= f["passing coverage"]):
            violations += 1
        if f["nominal coverage"] > f["passing coverage"]:
            strict_gaps += 1
    check(f"violations over {trials} random inputs", violations, 0)
    print(f"       inputs where nominal strictly exceeded passing: {strict_gaps}"
          f" of {trials}")

    print()
    print(f"FAILED CHECKS: {len(FAILURES)}")
    for name in FAILURES:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
