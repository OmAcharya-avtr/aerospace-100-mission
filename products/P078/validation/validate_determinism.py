"""Determinism under re-run, for every component that could drift.

Four things are checked, because four things could move between two runs of the
same command: the forward walk's finding set, the generator's sequences, the
injection edits, and the learned model's scores. Each is re-run and compared
against the first run by a digest of its full output rather than by a summary
statistic, so a difference in any field of any finding shows up.

The timing table at the end is wall clock on a shared two-core container. It is
declared unstable, it lives in a table, and no conclusion anywhere in this
repository is derived from it.
"""

from __future__ import annotations

import hashlib

import numpy as np

import _bootstrap
from _bootstrap import Tee, Timer, header

ROOT = _bootstrap.bootstrap()

from cmdguard.benchmark import lint_miss_rate, timing_table  # noqa: E402
from cmdguard.dataset import build_corpus, generate_clean_case  # noqa: E402
from cmdguard.defects import FAMILIES, inject  # noqa: E402
from cmdguard.lint import LintStrength, static_lint  # noqa: E402
from cmdguard.loader import load_case  # noqa: E402
from cmdguard.model import EffectPolicy  # noqa: E402
from cmdguard.ranker import RiskRanker  # noqa: E402
from cmdguard.trace import forward_execute  # noqa: E402

CASES = ROOT / "examples" / "cases"
HAND = (
    "nominal_science_pass",
    "late_payload_enable",
    "irreversible_before_verify",
    "irreversible_after_verify",
    "resource_and_exclusion",
)
REPEATS = 20
N_SEEDS = 30


def digest(text: str) -> str:
    """Short SHA-256 digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def main() -> int:
    tee = Tee("validate_determinism_output.txt")
    header(tee, "cmdguard -- determinism under re-run")

    tee.rule(f"1. the forward walk, {REPEATS} repeats per case")
    tee(f"  {'case':<30s} {'steps':>6s} {'violations':>11s} {'fingerprint':>34s}")
    all_stable = True
    for name in HAND:
        case = load_case(CASES / f"{name}.yaml")
        prints = {forward_execute(case).fingerprint() for _ in range(REPEATS)}
        report = forward_execute(case)
        tee(f"  {name:<30s} {len(case.sequence):>6d} {report.n_violations:>11d} "
            f"{sorted(prints)[0]:>34s}")
        if len(prints) != 1:
            all_stable = False
    tee.check("every hand case has one fingerprint over the repeats", all_stable)

    tee("")
    tee("  generated cases, both effect policies:")
    unstable = 0
    for seed in range(N_SEEDS):
        case = generate_clean_case(seed)
        for policy in EffectPolicy:
            prints = {
                forward_execute(case, policy).fingerprint() for _ in range(5)
            }
            if len(prints) != 1:
                unstable += 1
    tee(f"    cases x policies checked : {N_SEEDS * len(EffectPolicy)}")
    tee(f"    unstable                 : {unstable}")
    tee.check("every generated case has one fingerprint under both policies", unstable == 0)

    tee.rule("2. the generator")
    mismatches = 0
    for seed in range(N_SEEDS):
        first = generate_clean_case(seed)
        second = generate_clean_case(seed)
        left = [(s.t_s, s.command) for s in first.sequence]
        right = [(s.t_s, s.command) for s in second.sequence]
        if left != right:
            mismatches += 1
    tee(f"  seeds re-generated : {N_SEEDS}")
    tee(f"  sequences differing: {mismatches}")
    tee.check("the generator is reproducible from its seed", mismatches == 0)

    tee.rule("3. the injections")
    mismatches = 0
    checked = 0
    for seed in range(12):
        clean = generate_clean_case(seed)
        for family in FAMILIES:
            first, record_a = inject(clean, family, seed)
            second, record_b = inject(clean, family, seed)
            checked += 1
            if record_a != record_b:
                mismatches += 1
            elif forward_execute(first).fingerprint() != forward_execute(second).fingerprint():
                mismatches += 1
    tee(f"  injections re-run : {checked}")
    tee(f"  differing         : {mismatches}")
    tee.check("every injection is reproducible from its seed", mismatches == 0)

    tee.rule("4. the lint and the miss-rate measurement")
    corpus = build_corpus(range(12), include_clean=True)
    for strength in LintStrength:
        digests = set()
        for _ in range(3):
            lines = []
            for entry in corpus:
                report = static_lint(entry.case, strength)
                lines.append(f"{report.case_name}:{sorted(report.identities())}")
            digests.add(digest("\n".join(lines)))
        tee(f"  lint {strength.value:<12s} digests over 3 runs: {len(digests)} "
            f"({sorted(digests)[0]})")
        tee.check(f"the {strength.value} lint is deterministic", len(digests) == 1)
    rates = {
        f"{lint_miss_rate(corpus, LintStrength.WITH_TIMING).total.miss_rate:.17g}"
        for _ in range(3)
    }
    tee(f"  miss rate over 3 runs: {sorted(rates)}")
    tee.check("the miss rate is bit-identical across runs", len(rates) == 1)

    tee.rule("5. the learned ranker")
    train = build_corpus(range(20), include_clean=False)
    pairs = [(entry.case, entry.report) for entry in train]
    probe = generate_clean_case(500_000)
    first = RiskRanker(n_estimators=60, seed=99).fit(pairs).score(probe)
    second = RiskRanker(n_estimators=60, seed=99).fit(pairs).score(probe)
    same_risk = bool(np.array_equal(first.risk, second.risk))
    same_spread = bool(np.array_equal(first.spread, second.spread))
    tee(f"  steps scored           : {first.risk.size}")
    tee(f"  risk arrays identical  : {same_risk}")
    tee(f"  spread arrays identical: {same_spread}")
    tee(f"  max absolute risk difference: "
        f"{float(np.max(np.abs(first.risk - second.risk))):.3e}")
    tee.check("two fits with the same seed give identical risks", same_risk)
    tee.check("two fits with the same seed give identical spreads", same_spread)
    different = RiskRanker(n_estimators=60, seed=100).fit(pairs).score(probe)
    tee(f"  a different seed moves the risks: "
        f"{not np.array_equal(first.risk, different.risk)}")
    tee.check("a different seed is a different model",
              not np.array_equal(first.risk, different.risk))

    tee.rule("6. cost, in a table explicitly declared unstable")
    tee("  Wall clock on a shared two-core container, mean over 5 repeats. These")
    tee("  numbers move between runs and between machines. Nothing in this repository")
    tee("  states a ratio derived from them in prose.")
    tee("")
    cases = [load_case(CASES / f"{name}.yaml") for name in HAND]
    cases += [generate_clean_case(seed) for seed in (0, 1)]
    from cmdguard.dataset import GeneratorConfig

    cases.append(generate_clean_case(7, GeneratorConfig(n_steps=240), name="clean-240-steps"))
    tee(f"  {'case':<30s} {'steps':>6s} {'forward ms':>12s} {'lint ms':>10s}")
    for name, steps, forward_ms, lint_ms in timing_table(cases, repeats=5):
        tee(f"  {name:<30s} {steps:>6d} {forward_ms:>12.4f} {lint_ms:>10.4f}")
    tee("")
    fit = Timer()
    with fit:
        RiskRanker(n_estimators=200).fit(pairs)
    tee(f"  {'component':<30s} {'value':>12s}")
    tee(f"  {'ranker fit, 200 trees':<30s} {fit.elapsed_s:>12.3f} s")
    tee(f"  {'training rows':<30s} "
        f"{sum(len(entry.case.sequence) for entry in train):>12d}")
    tee(f"  {'corpus build, 20 seeds':<30s} {len(train):>12d} cases")
    tee("")
    tee("  The whole measurement set in this repository runs in under a minute on two")
    tee("  cores, so the compute budget is not a limitation of this product. The")
    tee("  limitation is the generator, not the arithmetic.")

    code = tee.summarise()
    tee(f"transcript written to {tee.save()}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
