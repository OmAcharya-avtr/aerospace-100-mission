"""Every check, raised deliberately, and the finding counts on every case.

For each of BR001 to BR010 the script builds the smallest declaration that
raises it, confirms that the check fires, and confirms that a minimally
altered declaration does not. A check that cannot be made silent is a check
that reports nothing.
"""

from __future__ import annotations

from _support import CASES, Emitter

from budgetroll import (
    Budget,
    Correlation,
    Item,
    MarginPolicy,
    Node,
    load,
    run_checks,
    severity_counts,
)
from budgetroll.checks import CODES

POLICY = MarginPolicy(
    name="v",
    description="validation policy",
    source="declared in validation/validate_checks.py",
    levels={"mature": 0.05, "new": 0.20},
)


def leaf(ident, value=1.0, uncertainty=0.1, maturity="mature", **kwargs):
    return Item(
        id=ident, name=ident, value=value, uncertainty=uncertainty, maturity=maturity, **kwargs
    )


def budget(root, correlations=(), k_sigma=1.0) -> Budget:
    return Budget(
        id="v",
        name="V",
        unit="kg",
        quantity="mass",
        policy=POLICY,
        root=root,
        correlations=correlations,
        k_sigma=k_sigma,
    )


def codes(b: Budget) -> list[str]:
    return [f.code for f in run_checks(b)]


def pair(rule="rss", **node_kwargs):
    return Node(
        id="root", name="root", combination=rule, children=(leaf("a"), leaf("b")), **node_kwargs
    )


TRIGGERS = {
    "BR001": (
        budget(pair("rss"), (Correlation("a", "b", 0.5, "same supplier"),)),
        budget(pair("rss")),
    ),
    "BR002": (
        budget(pair("linear")),
        budget(pair("rss")),
    ),
    "BR003": (
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", value=10.0), leaf("b", value=10.0)), allocation=5.0)),
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a"), leaf("b")), allocation=500.0)),
    ),
    "BR004": (
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", uncertainty=3.0), leaf("b", uncertainty=4.0)),
                    allocation=3.0)),
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a"), leaf("b")), allocation=500.0)),
    ),
    "BR005": (
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", contingency=0.4), leaf("b")))),
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", contingency=0.05), leaf("b")))),
    ),
    "BR006": (
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", uncertainty=0.0, maturity="new"), leaf("b")))),
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", uncertainty=0.0, maturity="mature"), leaf("b")))),
    ),
    "BR007": (
        budget(Node(id="root", name="root", combination="linear",
                    children=(Node(id="n", name="n", combination="rss", children=(leaf("a"),)),
                              leaf("b")))),
        budget(Node(id="root", name="root", combination="linear",
                    children=(Node(id="n", name="n", combination="linear",
                                   children=(leaf("a"),)), leaf("b")))),
    ),
    "BR008": (
        budget(Node(id="root", name="root", combination="linear",
                    children=(leaf("a"), leaf("b"), leaf("c"))),
               (Correlation("a", "b", 0.9, "r"), Correlation("a", "c", 0.9, "r"))),
        budget(Node(id="root", name="root", combination="linear",
                    children=(leaf("a"), leaf("b"), leaf("c"))),
               (Correlation("a", "b", 0.5, "r"),)),
    ),
    "BR009": (
        budget(pair("correlated")),
        budget(pair("correlated"), (Correlation("a", "b", 0.5, "r"),)),
    ),
    "BR010": (
        budget(Node(id="root", name="root", combination="rss",
                    children=(leaf("a", common_driver="v"), leaf("b", common_driver="v")))),
        budget(Node(id="root", name="root", combination="linear",
                    children=(leaf("a", common_driver="v"), leaf("b", common_driver="v"))),
               (Correlation("a", "b", 0.5, "r"),)),
    ),
}

CASE_FILES = [
    "clean_mass.yaml",
    "planted_findings.yaml",
    "cubesat_power.yaml",
    "cubesat_link.yaml",
]


def main() -> int:
    """Raise and silence each check, then count the findings on each case."""
    out = Emitter(__file__)
    out("budgetroll validation: the ten checks")
    out("=" * 72)
    out("Educational-grade (validation level 1). Not flight-qualified, not certified,")
    out("not approved for operational aerospace use. A clean run means a declaration")
    out("is arithmetically consistent with its own stated assumptions; it is not a")
    out("statement that any estimate is good or any margin adequate.")
    out()

    out("1. Each check raised and then silenced")
    for code in sorted(CODES):
        raising, silent = TRIGGERS[code]
        out(f"   {code}  {CODES[code]}")
        out.check(f"{code} fires on the triggering declaration", code in codes(raising))
        out.check(f"{code} is silent on the corrected declaration", code not in codes(silent))
    out()

    out("2. Findings on every shipped case")
    header = f"   {'case':<26s}{'error':>7s}{'warning':>9s}{'info':>6s}{'blocking':>10s}"
    out(header)
    out("   " + "-" * (len(header) - 3))
    expected = {
        "clean_mass.yaml": {"error": 0, "warning": 0, "info": 0},
        "planted_findings.yaml": {"error": 6, "warning": 4, "info": 2},
        "cubesat_power.yaml": {"error": 4, "warning": 2, "info": 3},
        "cubesat_link.yaml": {"error": 0, "warning": 0, "info": 2},
    }
    for name in CASE_FILES:
        b = load(CASES / name)
        findings = run_checks(b)
        counts = severity_counts(findings)
        blocking = "yes" if counts["error"] else "no"
        out(
            f"   {name:<26s}{counts['error']:>7d}{counts['warning']:>9d}"
            f"{counts['info']:>6d}{blocking:>10s}"
        )
        out.check(f"{name} counts are as recorded", counts == expected[name], str(counts))
    out()

    out("3. Every code has a description and the set is exactly BR001 to BR010")
    out.check(
        "ten codes, BR001 to BR010",
        sorted(CODES) == [f"BR{n:03d}" for n in range(1, 11)],
        str(sorted(CODES)),
    )
    out.check(
        "every code carries a non-empty description",
        all(text.strip() for text in CODES.values()),
    )
    return out.finish()


if __name__ == "__main__":
    raise SystemExit(main())
