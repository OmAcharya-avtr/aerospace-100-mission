"""Check the APFD implementation against hand-worked cases and the fixture history.

APFD is implemented exactly as published in

    G. Rothermel, R. H. Untch, C. Chu and M. J. Harrold, "Prioritizing Test
    Cases For Regression Testing", IEEE Transactions on Software Engineering,
    27(10), 929-948, October 2001. doi:10.1109/32.962562

        APFD = 1 - (TF_1 + ... + TF_m) / (n * m) + 1 / (2 * n)

Every reference value below is evaluated by hand from that formula and written
into the table, then compared with the implementation to 1e-12.

    python validation/validate_apfd.py > validation/validate_apfd_output.txt
"""

from __future__ import annotations

import sys

from _common import HISTORY_JSON, TRAIN_FRACTION

from regressselect.history import load_history
from regressselect.metrics import apfd, apfd_single_fault
from regressselect.prioritise import CoFailurePrioritiser

TOL = 1e-12

# (label, order length n, detection positions 1-based, hand-worked APFD)
CASES = (
    ("n=5 m=2, TF=(1,3)", 5, (1, 3), 1 - (1 + 3) / (5 * 2) + 1 / (2 * 5)),
    ("n=5 m=2, TF=(5,5)", 5, (5, 5), 1 - (5 + 5) / (5 * 2) + 1 / (2 * 5)),
    ("n=4 m=3, TF=(1,1,1)", 4, (1, 1, 1), 1 - 3 / (4 * 3) + 1 / 8),
    ("n=4 m=3, TF=(4,4,4)", 4, (4, 4, 4), 1 - 12 / (4 * 3) + 1 / 8),
    ("n=1 m=1, TF=(1,)", 1, (1,), 1 - 1 / 1 + 1 / 2),
    ("n=10 m=1, TF=(1,)", 10, (1,), 1 - 1 / 10 + 1 / 20),
    ("n=10 m=1, TF=(10,)", 10, (10,), 1 - 10 / 10 + 1 / 20),
    ("n=24 m=4, TF=(1,2,3,4)", 24, (1, 2, 3, 4), 1 - 10 / (24 * 4) + 1 / 48),
)


def main() -> int:
    out = sys.stdout
    failures = 0
    out.write("APFD known-answer check (Rothermel, Untch, Chu and Harrold 2001)\n")
    out.write("=" * 82 + "\n")
    out.write("formula: APFD = 1 - (sum TF_i) / (n * m) + 1 / (2 n)\n\n")
    out.write(f"{'case':<26}{'hand-worked':>17}{'computed':>17}{'abs diff':>12}  verdict\n")
    out.write("-" * 82 + "\n")
    for label, n, positions, expected in CASES:
        order = [f"t{i}" for i in range(1, n + 1)]
        detecting = {f"f{j}": [order[p - 1]] for j, p in enumerate(positions)}
        got = apfd(order, detecting)
        diff = abs(got - expected)
        ok = diff <= TOL
        failures += 0 if ok else 1
        out.write(
            f"{label:<26}{expected:>17.12f}{got:>17.12f}{diff:>12.3e}  "
            f"{'PASS' if ok else 'FAIL'}\n"
        )

    out.write("\nundefined cases are rejected rather than substituted\n")
    out.write("-" * 82 + "\n")
    for label, args in (
        ("empty order", ([], {"f": ["a"]})),
        ("empty fault set", (["a"], {})),
        ("fault detected by no test", (["a"], {"f": ["zz"]})),
        ("duplicate test ids", (["a", "a"], {"f": ["a"]})),
    ):
        try:
            apfd(*args)
            out.write(f"{label:<40}FAIL (no exception)\n")
            failures += 1
        except ValueError as exc:
            out.write(f"{label:<40}PASS ({str(exc).splitlines()[0][:34]})\n")

    out.write("\nAPFD on the committed fixture history\n")
    out.write("-" * 82 + "\n")
    history = load_history(HISTORY_JSON)
    train, test = history.split_chronological(TRAIN_FRACTION)
    nodes = list(history.test_ids)
    prioritiser = CoFailurePrioritiser().fit(train)
    regressions = [r for r in test.revisions if r.is_regression]
    lexical = sorted(nodes)
    lex_apfd = sum(apfd_single_fault(lexical, r.failed_tests) for r in regressions) / len(
        regressions
    )
    hist_apfd = sum(
        apfd_single_fault([rt.node_id for rt in prioritiser.rank(r.changed_modules, nodes)],
                          r.failed_tests)
        for r in regressions
    ) / len(regressions)
    out.write(f"history file           {HISTORY_JSON.name} (seed {history.seed})\n")
    out.write(
        f"revisions              {len(history)} total, "
        f"{len(train)} train, {len(test)} test\n"
    )
    out.write(f"test nodes             {len(nodes)}\n")
    out.write(f"evaluation revisions   {len(regressions)} with at least one failing test\n")
    out.write(f"lexical control APFD   {lex_apfd:.6f}  (no information; ordering by node id)\n")
    out.write(f"Baseline 2 APFD        {hist_apfd:.6f}\n")
    beats = hist_apfd > lex_apfd
    out.write(
        f"Baseline 2 beats the no-information control: {'PASS' if beats else 'FAIL'} "
        f"(margin {hist_apfd - lex_apfd:+.6f})\n"
    )
    failures += 0 if beats else 1

    total = len(CASES) + 4 + 1
    out.write(f"\nchecks run {total}, failures {failures}\n")
    out.write("RESULT: " + ("PASS\n" if failures == 0 else f"FAIL ({failures})\n"))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
