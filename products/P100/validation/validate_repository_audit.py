"""Run the audit over this repository and record what it found.

This is the honest test. The product is pointed at the mission repository that
contains it and asked the question it exists to answer: how many committed
numbers claimed reproducible actually move when only the random seed changes.

Whatever it finds is recorded, including findings against products already
published, and including finding nothing -- in which case the escaped-rate
upper bound on what could have escaped is the result.

No other product is modified. Every execution happens in a pristine copy in a
temporary directory; this script never writes outside ``products/P100/`` and a
temporary directory.

Two exclusions are declared rather than quietly applied, and both are printed
with the result: ``products/P094/`` because another build agent is writing it
concurrently, so any number derived from it is not reproducible by a reader;
and ``products/P100/`` because a tool scanning its own tree while it is being
written is the class it exists to detect, and it is audited deliberately and
separately in ``validate_self_audit.py``.

This script is the MEASUREMENT. It does not pin the finding set: the set is
pinned as literals in ``validate_repository_findings.py``, which re-reads the
evidence file this script writes and fails if the set has changed. Splitting
them is deliberate -- the measurement takes tens of minutes and the pinned
check takes under a second, so the pinned check can be re-run as often as the
build guide requires.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import time  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from seedaudit.audit import (  # noqa: E402
    DEFAULT_EXCLUSIONS,
    DEFAULT_ROOTS,
    EXCLUSION_REASONS,
    audit_repository,
)
from seedaudit.report import AUDIT_MODEL, looks_like_a_duration  # noqa: E402

#: Wall-clock limit for one execution of one audited script, seconds. Measured
#: on a 14-script sample of this corpus: mean 5.21 s, worst 23.80 s. 75 s is
#: about three times the worst observed, and a script that exceeds it is
#: reported as a timeout rather than silently dropped.
TIMEOUT = 20.0 if _bootstrap.FAST else 75.0

#: Concurrent candidates. Kept at 2 on a 2-core machine shared with another
#: build agent: oversubscribing would change the timing of the very scripts
#: being measured.
WORKERS = 2

#: Global wall-clock budget, seconds. Partial coverage is reported as partial
#: coverage; it is never implied to be complete. The first full run of this
#: audit reached a 3000 s budget after 180 of 229 candidates, so the budget was
#: raised rather than the shortfall being left in the coverage line.
DEADLINE = 120.0 if _bootstrap.FAST else 6000.0

#: Candidate cap for the fast path only. ``None`` on the measured path.
LIMIT = 6 if _bootstrap.FAST else None

#: Where the machine-readable evidence is written, relative to the root.
EVIDENCE = "validation/repository_findings.json"


def main() -> int:
    """Run the audit, print the report and write the evidence file."""
    out = _bootstrap.Tee("validate_repository_audit_output.txt")
    out("Repository audit -- P100 SeedAudit")
    out("")
    out(AUDIT_MODEL)
    out("")
    out(f"fast path (SEEDAUDIT_FAST=1): {_bootstrap.FAST}")
    out(f"per-run timeout, seconds: {TIMEOUT!r}")
    out(f"workers: {WORKERS}")
    out(f"global deadline, seconds: {DEADLINE!r}")
    out(f"candidate limit: {LIMIT}")

    repository = _bootstrap.repository_root()
    if repository is None:
        out.unavailable(
            "repository audit",
            "this product is not inside a mission repository, so the corpus "
            "does not exist here; a missing corpus is not a code defect",
        )
        return out.finish()

    out("")
    out(f"repository root directory name: {repository.name}")
    out(f"roots scanned: {', '.join(DEFAULT_ROOTS)}")
    for item in DEFAULT_EXCLUSIONS:
        out(f"excluded: {item} -- {EXCLUSION_REASONS[item]}")

    started = time.monotonic()
    audit = audit_repository(
        repository,
        DEFAULT_ROOTS,
        DEFAULT_EXCLUSIONS,
        timeout=TIMEOUT,
        workers=WORKERS,
        deadline_seconds=DEADLINE,
        limit=LIMIT,
    )
    elapsed = time.monotonic() - started

    out.rule("1. HEADLINE")
    out(f"committed numbers claimed reproducible that MOVE under a seed change: {audit.headline}")
    out("  (net of the instrument's own footprint; see section 3b)")
    out(f"scripts contributing at least one: {len(audit.scripts_with_seed_movement)}")
    out("")
    out(
        "committed numbers that move with the seed held at its DECLARED value "
        f"(separate, worse class): {audit.declared_seed_headline}"
    )
    out(
        "scripts contributing at least one of those: "
        f"{len(audit.scripts_with_declared_seed_movement)}"
    )

    out.rule("2. COVERAGE, stated as a number")
    counts = audit.status_counts()
    out(f"candidates found: {len(audit.candidates)}")
    out(f"candidates attempted: {len(audit.results)}")
    out(f"compared: {counts.get('compared', 0)}")
    out(f"no_output (claim UNEVALUABLE, not clean): {counts.get('no_output', 0)}")
    out(f"run_failed: {counts.get('run_failed', 0)}")
    out(f"timeout: {counts.get('timeout', 0)}")
    out(f"global deadline reached: {audit.deadline_reached}")
    out(f"committed numeric tokens across all candidates: {audit.claim_tokens_total}")
    out(f"numeric tokens the perturbed comparison evaluated: {audit.claim_tokens_compared}")
    out(f"repeats per claim: {audit.repeats_per_claim}")
    out(
        "escaped-rate upper bound at that repeat count on the REPEAT axis, "
        f"alpha=0.05: {audit.escaped_bound!r}"
    )
    out(
        "note: the seed axis is sampled with one perturbed draw, which for a "
        "seed-sensitive continuous output has detection probability near 1; "
        "the repeat axis is sampled once per arrangement, so its bound is weak "
        "and is stated rather than hidden."
    )
    out(f"audit wall-clock seconds: {elapsed:.1f}")

    out.rule("3. SEED-ATTRIBUTABLE MOVED NUMBERS -- the headline, itemised")
    seed_items: list[dict[str, object]] = []
    if audit.headline == 0:
        out("none found.")
        out(
            "a clean result is NOT a zero: with one perturbed draw on the seed "
            "axis and one run per arrangement on the repeat axis, the "
            f"escaped-rate upper bound on the repeat axis is {audit.escaped_bound!r}"
        )
    for result in audit.results:
        for moved in result.seed_attributable:
            seed_items.append(
                {
                    "script": result.candidate.label,
                    "output": f"{result.candidate.source_dir}/{result.candidate.output}",
                    "line": moved.line,
                    "index": moved.index,
                    "before": moved.before,
                    "after": moved.after,
                    "absolute_change": moved.absolute_change,
                    "relative_change": moved.relative_change,
                    "skeleton": moved.skeleton.strip()[:160],
                }
            )
            out(
                f"{result.candidate.label}  line {moved.line} token {moved.index}: "
                f"{moved.before} -> {moved.after}  (abs {moved.absolute_change:.6g}, "
                f"rel {moved.relative_change:.6g})"
            )
            out(f"    context: {moved.skeleton.strip()[:140]}")

    out.rule("3b. THE INSTRUMENT'S OWN FOOTPRINT, excluded from the headline")
    out("a script that prints its own seed reports a changed number in the")
    out("perturbed run for one reason: the audit changed the seed. The control")
    out("cannot absorb it, because in the control the seed is unchanged. Counting")
    out("it as a finding would be the audit reporting its own perturbation back to")
    out("itself, which is the self-polluting-check class one level up.")
    out("")
    footprint_items: list[dict[str, object]] = []
    for result in audit.results:
        for moved in result.instrument_footprint:
            footprint_items.append(
                {
                    "script": result.candidate.label,
                    "line": moved.line,
                    "index": moved.index,
                    "before": moved.before,
                    "after": moved.after,
                    "skeleton": moved.skeleton.strip()[:160],
                }
            )
            out(
                f"{result.candidate.label}  line {moved.line} token {moved.index}: "
                f"{moved.before} -> {moved.after}  (declared seed plus the perturbation)"
            )
    out("")
    out(f"instrument-footprint numbers excluded from the headline: "
        f"{audit.instrument_footprint_count}")
    out(f"scripts that print a seed the audit rewrote: "
        f"{len(audit.scripts_with_instrument_footprint)}")

    out.rule("4. NUMBERS THAT MOVE AT THE DECLARED SEED -- separate, worse class")
    declared_items: list[dict[str, object]] = []
    declared_durations = 0
    if audit.declared_seed_headline == 0:
        out("none found.")
    for result in audit.results:
        for moved in result.declared_seed_moved:
            duration = looks_like_a_duration(moved.skeleton)
            declared_durations += int(duration)
            declared_items.append(
                {
                    "script": result.candidate.label,
                    "line": moved.line,
                    "index": moved.index,
                    "before": moved.before,
                    "after": moved.after,
                    "skeleton": moved.skeleton.strip()[:160],
                    "duration_marker": duration,
                }
            )
            out(
                f"{result.candidate.label}  line {moved.line} token {moved.index}: "
                f"{moved.before} -> {moved.after}"
                f"{'  [duration marker]' if duration else ''}"
            )
            out(f"    context: {moved.skeleton.strip()[:140]}")
    out("")
    out(f"of {audit.declared_seed_headline} numbers that move at the declared seed, "
        f"{declared_durations} sit on a line carrying a wall-clock duration marker and "
        f"{audit.declared_seed_headline - declared_durations} do not.")
    out("the duration marker GROUPS these findings and never removes one. A wall-clock")
    out("reading written into a file the repository treats as reproducible evidence is")
    out("still a number that is not reproducible.")

    out.rule("5. CANDIDATES WHOSE CLAIM COULD NOT BE EVALUATED")
    unevaluable: dict[str, list[str]] = {"no_output": [], "run_failed": [], "timeout": []}
    for result in audit.results:
        if result.status in unevaluable:
            unevaluable[result.status].append(result.candidate.label)
    for status, labels in unevaluable.items():
        out(f"{status}: {len(labels)}")
        for label in sorted(labels):
            out(f"  {label}")

    out.rule("6. STATIC SELF-POLLUTION SCREEN over the same corpus")
    by_subkind: dict[str, int] = {}
    for findings in audit.static_findings.values():
        for finding in findings:
            by_subkind[finding.subkind] = by_subkind.get(finding.subkind, 0) + 1
    out(f"python files parsed: {audit.files_scanned}")
    out(f"files with at least one finding: {len(audit.static_findings)}")
    out(f"findings in total: {audit.static_finding_count}")
    out(f"files that would not parse: {len(audit.parse_failures)}")
    for subkind in sorted(by_subkind):
        out(f"  {subkind}: {by_subkind[subkind]}")
    out("")
    out("files with the most findings, with one finding each shown in full:")
    ranked = sorted(audit.static_findings.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for path, findings in ranked[:20]:
        out(f"  {path}: {len(findings)} findings")
        out(f"    {findings[0].describe()}")
    out("")
    out(
        "the screen is conservative by construction: a path expression it "
        "cannot resolve becomes a wildcard and a wildcard is under everything, "
        "so it over-flags rather than under-flags. Its false-positive rate on "
        "the deterministic negative controls of the injected corpus is "
        "measured in validate_injected_corpus.py and published there."
    )

    evidence = {
        "root_name": audit.root_name,
        "roots": list(audit.roots),
        "exclusions": list(audit.exclusions),
        "candidates_found": len(audit.candidates),
        "candidates_attempted": len(audit.results),
        "status_counts": counts,
        "deadline_reached": audit.deadline_reached,
        "claim_tokens_total": audit.claim_tokens_total,
        "claim_tokens_compared": audit.claim_tokens_compared,
        "repeats_per_claim": audit.repeats_per_claim,
        "escaped_bound": audit.escaped_bound,
        "headline_seed_attributable": audit.headline,
        "headline_declared_seed": audit.declared_seed_headline,
        "scripts_with_seed_movement": list(audit.scripts_with_seed_movement),
        "scripts_with_declared_seed_movement": list(
            audit.scripts_with_declared_seed_movement
        ),
        "seed_attributable": seed_items,
        "declared_seed_moved": declared_items,
        "declared_seed_duration_marked": declared_durations,
        "instrument_footprint": footprint_items,
        "instrument_footprint_count": audit.instrument_footprint_count,
        "scripts_with_instrument_footprint": list(audit.scripts_with_instrument_footprint),
        "unevaluable": {k: sorted(v) for k, v in unevaluable.items()},
        "static_files_parsed": audit.files_scanned,
        "static_files_with_findings": len(audit.static_findings),
        "static_findings_total": audit.static_finding_count,
        "static_by_subkind": by_subkind,
        "static_parse_failures": len(audit.parse_failures),
        "static_top_files": [
            {"path": path, "findings": len(findings)} for path, findings in ranked[:40]
        ],
    }
    target = ROOT / EVIDENCE
    if target.exists():
        target.unlink()
    target.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out("")
    out(f"machine-readable evidence written to {_bootstrap.relative(target)}")

    out.rule("7. WHAT THIS RUN DOES NOT CLAIM")
    out(
        "a candidate reported as compared with no moved numbers is evidence "
        "that this ONE perturbed seed draw did not move it. It is not evidence "
        "that no seed does, and it is not evidence about intermittency at a "
        "fixed seed: that is the repeat axis and this run samples it once per "
        f"arrangement, bound {audit.escaped_bound!r}."
    )
    out(
        "the committed output artifact is deleted inside the pristine copy "
        "before each run, so a script that does not rewrite it is reported as "
        "no_output rather than compared against itself. That deletion applies "
        "identically to the control run, so it cannot reach the headline."
    )
    out.check(
        "the audit completed with at least one candidate compared",
        counts.get("compared", 0) > 0,
        f"compared={counts.get('compared', 0)}",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
