"""The deliverable: how much a per-command static lint misses, measured.

The population is a seeded family of injected defects: one clean generated
sequence per seed, and one defective sequence per seed per defect family. Nine
families, eight of them reachable only by a sequence-aware check and one -- a
mistyped mnemonic -- that a per-command lint finds every time, so the headline
rate is a rate over a mixed population rather than one selected to embarrass
the baseline.

Two lint strengths are measured. ``per_command`` is strictly one command at a
time against the declared initial state. ``with_timing`` adds a pass over the
time column, which a shell script can do, and which reproduces the forward
walk's timing findings exactly. Reporting the stronger one is what makes the
measurement a measurement.

The identity used to decide whether the lint found a violation is ``(step,
kind)``, which is generous to the baseline: a lint that reports the right kind
at the right step for the wrong reason is counted as having found it.

The false-positive rate is reported beside the miss rate because on a long
sequence it is the number that decides whether anybody keeps running the tool.
"""

from __future__ import annotations

import _bootstrap
from _bootstrap import Tee, header

ROOT = _bootstrap.bootstrap()

from cmdguard.benchmark import lint_miss_rate  # noqa: E402
from cmdguard.dataset import build_corpus  # noqa: E402
from cmdguard.defects import FAMILIES  # noqa: E402
from cmdguard.lint import LintStrength  # noqa: E402
from cmdguard.trace import ORDERING_KINDS  # noqa: E402

N_SEEDS = 60


def main() -> int:
    tee = Tee("validate_lint_missrate_output.txt")
    header(tee, "cmdguard -- static-lint miss rate on a seeded family of injected defects")

    tee.rule("1. the population")
    corpus = build_corpus(range(N_SEEDS), include_clean=True)
    defective = [entry for entry in corpus if entry.record is not None]
    applied = [entry for entry in defective if entry.record is not None and entry.record.applied]
    steps = sum(len(entry.case.sequence) for entry in corpus)
    with_violations = [entry for entry in corpus if not entry.report.clean]
    tee(f"  generator seeds                 : {N_SEEDS} (0 .. {N_SEEDS - 1})")
    tee(f"  defect families                 : {len(FAMILIES)}")
    tee(f"  clean cases                     : {len(corpus) - len(defective)}")
    tee(f"  defective cases                 : {len(defective)}")
    tee(f"  of which the injection applied  : {len(applied)}")
    tee(f"  of which a no-op                : {len(defective) - len(applied)}")
    tee(f"  cases with at least one violation: {len(with_violations)}")
    tee(f"  total command occurrences       : {steps}")
    tee("")
    tee("  violation kinds only a sequence-aware check can reach:")
    for kind in ORDERING_KINDS:
        tee(f"    {kind.value}")

    headline: dict[str, tuple[int, int, float, float, float]] = {}

    for strength in LintStrength:
        tee.rule(f"2. miss rate, lint strength = {strength.value}")
        result = lint_miss_rate(corpus, strength)
        total = result.total
        tee(f"  forward-execution violations (ground truth) : {total.n_truth}")
        tee(f"  of those missed by the lint                 : {total.n_missed}")
        tee(f"  MISS RATE                                   : {total.miss_rate:.6f}")
        tee(f"  ordering-only violations                    : {total.n_truth_ordering}")
        tee(f"  of those missed by the lint                 : {total.n_missed_ordering}")
        tee(f"  ORDERING-ONLY MISS RATE                     : "
            f"{total.ordering_miss_rate:.6f}")
        tee(f"  lint findings emitted                       : {total.n_lint_findings}")
        tee(f"  findings the forward walk does not confirm  : "
            f"{total.n_lint_false_positive}")
        tee(f"  FALSE-POSITIVE RATE                         : "
            f"{total.false_positive_rate:.6f}")
        tee(f"  measurement wall clock (unstable)           : {result.seconds:.3f} s")
        headline[strength.value] = (
            total.n_truth,
            total.n_missed,
            total.miss_rate,
            total.ordering_miss_rate,
            total.false_positive_rate,
        )

        tee("")
        tee(f"  {'family':<20s} {'truth':>7s} {'missed':>7s} {'miss':>9s} "
            f"{'ord.truth':>10s} {'ord.miss':>9s} {'fp':>9s} {'no-ops':>7s}")
        for family in sorted(result.families):
            record = result.families[family]
            ordering_rate = (
                f"{record.ordering_miss_rate:.6f}"
                if record.n_truth_ordering
                else "       --"
            )
            miss_rate = f"{record.miss_rate:.6f}" if record.n_truth else "      --"
            fp_rate = (
                f"{record.false_positive_rate:.6f}" if record.n_lint_findings else "      --"
            )
            tee(f"  {family:<20s} {record.n_truth:>7d} {record.n_missed:>7d} "
                f"{miss_rate:>9s} {record.n_truth_ordering:>10d} {ordering_rate:>9s} "
                f"{fp_rate:>9s} {record.n_noop_injections:>7d}")

        tee("")
        tee("  missed violations by kind:")
        for kind, count in total.missed_by_kind.items():
            share = count / total.n_missed if total.n_missed else float("nan")
            tee(f"    {kind:<26s} {count:>6d}  {share:.6f} of all misses")

        tee.check(
            f"{strength.value}: the ratios are the counts divided",
            abs(total.miss_rate - total.n_missed / total.n_truth) < 1e-12,
        )
        tee.check(
            f"{strength.value}: missed-by-kind sums to the miss count",
            sum(total.missed_by_kind.values()) == total.n_missed,
        )
        tee.check(
            f"{strength.value}: the control family is found every time",
            result.families["undeclared_typo"].missed_by_kind.get("undeclared_command", 0) == 0,
            "a mistyped mnemonic is exactly what a per-command lint is for",
        )

    tee.rule("3. the two strengths against each other")
    weak_truth, weak_missed, weak_rate, weak_ord, weak_fp = headline["per_command"]
    strong_truth, strong_missed, strong_rate, strong_ord, strong_fp = headline["with_timing"]
    tee(f"  ground truth is the same population       : {weak_truth} == {strong_truth}")
    tee(f"  per_command  missed {weak_missed:>5d}  rate {weak_rate:.6f}  "
        f"ordering {weak_ord:.6f}  fp {weak_fp:.6f}")
    tee(f"  with_timing  missed {strong_missed:>5d}  rate {strong_rate:.6f}  "
        f"ordering {strong_ord:.6f}  fp {strong_fp:.6f}")
    tee(f"  violations the timing pass recovers       : {weak_missed - strong_missed}")
    tee.check("both strengths face the same ground truth", weak_truth == strong_truth)
    tee.check("adding a check cannot increase the miss count", strong_missed <= weak_missed,
              f"{strong_missed} <= {weak_missed}")

    tee.rule("4. what the numbers mean, stated plainly")
    tee("  Against the stronger baseline, a per-command lint with a timing pass misses")
    tee(f"  {strong_rate:.4f} of the violations a forward walk finds on this population,")
    tee(f"  and {strong_ord:.4f} of the violations that only a sequence-aware check can")
    tee("  reach. It also emits more findings than there are violations: its")
    tee(f"  false-positive rate against the forward walk is {strong_fp:.4f}, because it")
    tee("  evaluates every occurrence of every command against the declared initial")
    tee("  state and so re-reports a condition that an earlier command already fixed.")
    tee("")
    tee("  Both numbers are statements about this generator and this defect")
    tee("  population. They are not estimates of how often a real mission's own")
    tee("  checking script misses a real defect, and nothing here measures that.")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
