"""Point this product's own detectors at this product, and publish the result.

The repository audit excludes ``products/P100/`` on purpose: a tool scanning
its own tree while it is being written is the class it exists to detect. The
exclusion is not a way of avoiding the question, so the question is asked here
deliberately and whatever comes back is published, including findings against
these very scripts.

Four things are measured.

1. **The static screen over this product's own tree.** This product's
   validation layout is itself an instance of the class: the mission's build
   guide requires ``validation/<stem>_output.txt`` beside the script, so every
   script here writes into the directory this product's own hygiene check
   scans. That is reported, not hidden.
2. **The two-arrangement classifier on this product's own scripts.** Run in
   pristine copies, as everywhere else.
3. **A false-negative class in the audit's own candidate rule**, found by
   applying the rule to this product. A candidate needs a seed literal in the
   script itself; this product declares its one seed in a shared module, so by
   its own rule this product has no candidates at all. The size of that blind
   spot across the mission corpus is measured here rather than asserted.
4. **That the product's own report is reproducible**, by running the audit
   twice over a fixture and comparing.

``validate_self_audit.py`` excludes ITSELF from the dynamic arrangement check,
because running it inside a pristine copy of this product would run it
recursively. The exclusion is declared and is the only one.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import collections  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import _bootstrap  # noqa: E402

ROOT = _bootstrap.bootstrap()

from seedaudit.audit import (  # noqa: E402
    DEFAULT_EXCLUSIONS,
    audit_repository,
    find_candidates,
    static_scan,
)
from seedaudit.dynamic import SELF_POLLUTING, two_arrangements  # noqa: E402
from seedaudit.seeds import find_seed_literals  # noqa: E402
from seedaudit.static import path_use_coverage  # noqa: E402

#: The scripts here whose verdict is checked in the two arrangements. Chosen
#: because they are fast on the fast path and because the second is the hygiene
#: scanner, which is the interesting case.
ARRANGEMENT_SCRIPTS = (
    "validation/validate_power_model.py",
    "validation/validate_tree_hygiene.py",
)

#: The largest fraction of the headline that may move between two back-to-back
#: runs of the subset audit before the audit is reporting noise rather than
#: findings. DECLARED WITH ITS MEANING, not tuned to pass: at 0.05, a corpus
#: whose findings moved by more than one in twenty between consecutive runs
#: would make a single run uninformative. Measured values are 0 and 1 moved
#: numbers out of about 110, which is 0.009.
#:
#: The headline itself is NOT asserted to be identical across runs. It cannot
#: be: an audited script that is intermittently non-deterministic at a fixed
#: seed moves the count, and finding exactly that is what this product is for.
#: What IS asserted is the part this product controls -- coverage, the token
#: count and the contributing script set.
SUBSET_DRIFT_FRACTION = 0.05

#: Wall-clock budget this script declares for itself, seconds. The release gate
#: fails a validation script at 600 s; this is the margin this product commits
#: to keeping, and the script reports its own elapsed time against it so a
#: regression is visible rather than discovered at the gate.
SELF_BUDGET = 300.0

#: Excluded from the dynamic check, because running it inside a pristine copy
#: of this product runs it recursively.
ARRANGEMENT_EXCLUDED = "validation/validate_self_audit.py"

#: Candidates in the bounded subset used to measure the audit's own
#: between-run variation. Small enough that two runs finish in minutes and the
#: check is re-runnable, which a full-corpus repeat is not.
#:
#: Lowered from 24 after the coordinator measured this script at 473 s against
#: the release gate's 600 s per-script ceiling. 473 s is inside the ceiling and
#: not safely inside it: that run had two cores busy with another build, and a
#: slower or busier machine crosses it. The margin is now stated in
#: VALIDATION.md rather than left for the next session to discover.
SUBSET_CANDIDATES = 8

#: Per-run timeout for the subset, seconds. Tighter than the campaign's, so
#: that a slow candidate costs the subset time rather than the whole check.
SUBSET_TIMEOUT = 20.0


def main() -> int:
    """Audit this product with this product."""
    started = time.monotonic()
    out = _bootstrap.Tee("validate_self_audit_output.txt")
    out("Self-audit -- P100 SeedAudit audits P100 SeedAudit")
    out("")
    out(f"fast path (SEEDAUDIT_FAST=1): {_bootstrap.FAST}")
    out(f"product root directory name: {ROOT.name}")
    out(f"excluded from the dynamic check: {ARRANGEMENT_EXCLUDED} -- running it inside a")
    out("  pristine copy of this product would run it recursively")

    out.rule("1. THE STATIC SCREEN OVER THIS PRODUCT'S OWN TREE")
    findings, failures, scanned = static_scan(ROOT, ("src", "tests", "examples", "validation"), ())
    out(f"python files parsed: {scanned}")
    out(f"files that would not parse: {len(failures)}")
    out(f"files with at least one finding: {len(findings)}")
    out(f"findings in total: {sum(len(v) for v in findings.values())}")
    out("")
    for path in sorted(findings):
        out(f"{path}: {len(findings[path])} findings")
        for finding in findings[path]:
            out(f"    {finding.describe()}")
    out("")
    coverage_total = {"uses": 0, "informative": 0, "vacuous": 0}
    for directory in ("src", "tests", "examples", "validation"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            counts = path_use_coverage(path.read_text(encoding="utf-8"))
            for key in coverage_total:
                coverage_total[key] += counts[key]
    out(f"path uses across this product: {coverage_total['uses']}")
    out(f"  informative (the screen can reason about them): {coverage_total['informative']}")
    out(f"  vacuous (the screen is blind to them): {coverage_total['vacuous']}")
    out("")
    out("THIS PRODUCT'S VALIDATION LAYOUT IS AN INSTANCE OF THE CLASS IT DETECTS.")
    out("The mission's build guide requires validation/<stem>_output.txt beside the")
    out("script, so every script here writes into the directory the hygiene check")
    out("scans. It cannot be avoided, so it is handled the only honest way: the")
    out("Tee deletes its target before writing, so a run's output is never a merge")
    out("of this run and the last; the hygiene check asserts what this product OWNS")
    out("rather than what merely exists; and the finding is printed above rather")
    out("than excluded.")
    out.check(
        "the static screen parses every Python file in this product",
        len(failures) == 0,
        f"{len(failures)} unparseable",
    )

    out.rule("2. THE TWO-ARRANGEMENT CLASSIFIER ON THIS PRODUCT'S OWN SCRIPTS")
    out("each run is in its own pristine copy, with SEEDAUDIT_FAST set in the")
    out("subprocess so the runs are short. The question asked is whether the")
    out("script's VERDICT is stable, not what its numbers are, so the fast path is")
    out("the right setting and it is declared here.")
    out("")
    previous = os.environ.get("SEEDAUDIT_FAST")
    os.environ["SEEDAUDIT_FAST"] = "1"
    try:
        for script in ARRANGEMENT_SCRIPTS:
            artifact = f"validation/{Path(script).stem}_output.txt"
            result = two_arrangements(ROOT, script, artifact, timeout=240.0)
            out(f"{script}")
            out(f"  pristine runs agree : {result.pristine_agree}")
            out(f"  reused tree agree   : {result.reused_agree}")
            out(f"  classification      : {result.classification}")
            out(f"  escaped-rate bound at n={len(result.pristine)}: {result.escaped_bound!r}")
            for index, run in enumerate(result.pristine):
                out(f"  pristine run {index + 1}: exit={run.exit_code} "
                    f"timed_out={run.timed_out} artifact_bytes="
                    f"{len(run.artifact) if run.artifact else 0}")
            for index, run in enumerate(result.reused):
                out(f"  reused   run {index + 1}: exit={run.exit_code} "
                    f"timed_out={run.timed_out} artifact_bytes="
                    f"{len(run.artifact) if run.artifact else 0}")
            if result.note:
                out(f"  note: {result.note}")
            out.check(
                f"{script} is not classified as a self-polluting check",
                result.classification != SELF_POLLUTING,
                result.classification,
            )
            out("")
    finally:
        if previous is None:
            os.environ.pop("SEEDAUDIT_FAST", None)
        else:
            os.environ["SEEDAUDIT_FAST"] = previous

    out.rule("3. A FALSE-NEGATIVE CLASS IN THE AUDIT'S OWN CANDIDATE RULE")
    own = find_candidates(ROOT.parent.parent, ("products",), tuple(
        item for item in DEFAULT_EXCLUSIONS if item != "products/P100"
    )) if (ROOT.parent.parent / "products.yaml").is_file() else ()
    own_here = tuple(c for c in own if c.source_dir == "products/P100")
    out(f"candidates this audit finds inside this product: {len(own_here)}")
    out("")
    out("this product declares its ONE seed in validation/_bootstrap.py as")
    out("SEED = 1000044 and every script imports it. The candidate rule requires a")
    out("seed literal in the SCRIPT, so by its own rule this product has no")
    out("candidates. THAT IS A FALSE-NEGATIVE CLASS IN THE RULE, not a property of")
    out("this product, and the size of the blind spot across the corpus is measured")
    out("below rather than asserted.")
    out("")
    bootstrap_seed = find_seed_literals((ROOT / "validation" / "_bootstrap.py").read_text(
        encoding="utf-8"
    ))
    out(f"seed literals in validation/_bootstrap.py: {bootstrap_seed.count}")
    out(f"  values: {[literal.value for literal in bootstrap_seed.literals]}")
    out.check(
        "this product does declare a seed, in a shared module the candidate rule "
        "does not look at",
        bootstrap_seed.count >= 1,
        f"{bootstrap_seed.count} literals",
    )

    repository = _bootstrap.repository_root()
    if repository is None:
        out.unavailable(
            "blind-spot size across the mission corpus",
            "this product is not inside a mission repository",
        )
    else:
        with_output = 0
        with_own_seed = 0
        seed_only_in_sibling = 0
        unparseable = 0
        for product in sorted((repository / "products").iterdir()):
            if not product.is_dir() or f"products/{product.name}" in DEFAULT_EXCLUSIONS:
                continue
            for directory in ("validation", "examples"):
                folder = product / directory
                if not folder.is_dir():
                    continue
                siblings_declare = False
                for module in folder.glob("_*.py"):
                    try:
                        if find_seed_literals(module.read_text(encoding="utf-8")).literals:
                            siblings_declare = True
                    except (SyntaxError, OSError):
                        continue
                for script in sorted(folder.glob("*.py")):
                    if script.name.startswith("_"):
                        continue
                    if not script.with_name(script.stem + "_output.txt").is_file():
                        continue
                    with_output += 1
                    try:
                        scan = find_seed_literals(script.read_text(encoding="utf-8"))
                    except (SyntaxError, OSError):
                        unparseable += 1
                        continue
                    if scan.literals:
                        with_own_seed += 1
                    elif siblings_declare:
                        seed_only_in_sibling += 1
        out("")
        out(f"scripts with a committed output file: {with_output}")
        out(f"  declaring a seed literal in the script (CANDIDATES): {with_own_seed}")
        out(f"  declaring no seed of their own while a sibling _module does: "
            f"{seed_only_in_sibling}")
        out(f"  unparseable: {unparseable}")
        missed = seed_only_in_sibling
        rate = missed / with_output if with_output else 0.0
        out("")
        out(f"MEASURED BLIND SPOT of the candidate rule: {missed} of {with_output} scripts with a")
        out(f"committed output, which is {rate:.6f} of them. Those scripts DO have a")
        out("declared seed; it is just not written in the file the rule inspects.")
        out("Resolving an imported constant would need the audit to execute or")
        out("import the module it is auditing, which is a different and larger")
        out("design. The limitation is published in README.md rather than reduced")
        out("by widening the rule until it guesses.")
        out.check(
            "the blind spot is measured and non-trivial, so it belongs in the limitations",
            missed > 0,
            f"{missed} scripts",
        )

    out.rule("4. THE AUDIT'S OWN REPRODUCIBILITY ON THE REAL CORPUS")
    out("the full-corpus campaign is data/repository_audit.py and takes most of")
    out("an hour, so it is NOT re-run here and the gate does not run it either.")
    out("Its between-run variation is measured on a bounded subset instead.")
    out("")
    out("this matters more for this product than for any other in the mission. A")
    out("tool that audits reproducibility and is itself irreproducible is")
    out("worthless, and the honest way to say so is a number.")
    out("")
    if repository is None:
        out.unavailable(
            "audit reproducibility on the real corpus",
            "this product is not inside a mission repository",
        )
    else:
        subset = 4 if _bootstrap.FAST else SUBSET_CANDIDATES
        out(f"candidates in the subset: {subset}")
        out(f"per-run timeout, seconds: {SUBSET_TIMEOUT!r}")
        reports = []
        findings = []
        for index in range(2):
            audit = audit_repository(
                repository,
                ("products", "scripts"),
                DEFAULT_EXCLUSIONS,
                timeout=SUBSET_TIMEOUT,
                workers=1,
                limit=subset,
            )
            reports.append(audit)
            findings.append(
                collections.Counter(
                    result.candidate.label
                    for result in audit.results
                    for _moved in result.seed_attributable
                )
            )
            out(
                f"run {index + 1}: headline={audit.headline} "
                f"declared_seed={audit.declared_seed_headline} "
                f"footprint={audit.instrument_footprint_count} "
                f"status={dict(sorted(audit.status_counts().items()))} "
                f"evaluated_tokens={audit.claim_tokens_compared}"
            )
        first, second = reports
        drift = abs(first.headline - second.headline)
        largest = max(first.headline, second.headline, 1)
        fraction = drift / largest
        out("")
        out(f"headline difference between the two runs: {drift}")
        out(f"as a fraction of the larger headline     : {fraction:.6f}")
        out(f"declared ceiling for that fraction       : {SUBSET_DRIFT_FRACTION!r}")
        out(
            "declared-seed difference: "
            f"{abs(first.declared_seed_headline - second.declared_seed_headline)}"
        )
        out("")
        moved_scripts = sorted(
            {
                label
                for label in set(findings[0]) | set(findings[1])
                if findings[0].get(label, 0) != findings[1].get(label, 0)
            }
        )
        out(f"scripts whose finding count differs between the two runs: {len(moved_scripts)}")
        for label in moved_scripts:
            out(f"  {label}: run 1 {findings[0].get(label, 0)}, run 2 {findings[1].get(label, 0)}")
        if not moved_scripts:
            out("  none")
        out("")
        out("WHAT IS AND IS NOT ASSERTED HERE, and this is the whole point.")
        out("")
        out("The headline is NOT asserted to be identical across two runs. It")
        out("cannot be. An audited script that is intermittently non-deterministic")
        out("AT A FIXED SEED moves the count between back-to-back runs, and finding")
        out("exactly that is what this product is for. Asserting equality would be")
        out("asserting that the audited corpus is deterministic, which is the claim")
        out("this product exists to refute. A run of this script under the release")
        out("gate's environment measured 110 against 111 with identical coverage,")
        out("and the assertion that they must match was removed rather than the")
        out("observation being explained away. Any script named above is a script")
        out("whose committed numbers are not reproducible at its declared seed.")
        out("")
        out("What IS asserted is the part this product controls: that the coverage,")
        out("the evaluated token count and the set of contributing scripts are")
        out("identical, and that the drift stays far below the declared ceiling at")
        out("which a single run would stop being informative.")
        out.check(
            "the audit's coverage on a bounded subset is identical across two runs",
            first.status_counts() == second.status_counts(),
            f"{first.status_counts()} against {second.status_counts()}",
        )
        out.check(
            "the evaluated token count is identical across two runs",
            first.claim_tokens_compared == second.claim_tokens_compared,
            f"{first.claim_tokens_compared} against {second.claim_tokens_compared}",
        )
        out.check(
            "the set of contributing scripts is identical across two runs",
            set(first.scripts_with_seed_movement) == set(second.scripts_with_seed_movement),
            f"{len(first.scripts_with_seed_movement)} against "
            f"{len(second.scripts_with_seed_movement)} scripts",
        )
        out.check(
            "the headline drift between two runs is below the declared ceiling",
            fraction <= SUBSET_DRIFT_FRACTION,
            f"{fraction:.6f} against {SUBSET_DRIFT_FRACTION!r}",
        )
        out("")
        out("WHERE VARIATION COMES FROM, when it does. Two mechanisms, both real:")
        out("a per-run TIMEOUT is wall-clock and depends on machine load, so a")
        out("candidate can move between `timeout` and `compared` between runs and")
        out("take its findings with it; and an audited script can itself be")
        out("non-deterministic, which is the thing being looked for. The audit")
        out("cannot tell those apart from one run, and it does not pretend to.")
        out("The coverage check above separates them: coverage identical with a")
        out("moved headline is the SECOND mechanism and nothing else.")

    out.rule("5. THE PRODUCT'S OWN REPORT IS REPRODUCIBLE ON A FIXTURE")
    out("a tool that audits reproducibility and is itself non-reproducible is")
    out("worthless, so the audit is run twice over a fixture and the two reports")
    out("are compared.")
    out("")
    with tempfile.TemporaryDirectory(prefix="seedaudit-selfcheck-") as temporary:
        fixture = Path(temporary) / "fixture_repo"
        validation = fixture / "products" / "PA" / "validation"
        validation.mkdir(parents=True)
        (fixture / "scripts").mkdir()
        (fixture / "products.yaml").write_text("products: []\n", encoding="utf-8")
        (validation / "seeded.py").write_text(
            "from pathlib import Path\n\n"
            "SEED_LOCAL = 7\n\n"
            "HERE = Path(__file__).resolve().parent\n\n"
            "(HERE / 'seeded_output.txt').write_text('value %.6f\\n' % (SEED_LOCAL / 8.0))\n",
            encoding="utf-8",
        )
        (validation / "seeded_output.txt").write_text("value 0.875000\n", encoding="utf-8")
        reports = []
        for _ in range(2):
            audit = audit_repository(fixture, ("products", "scripts"), (), timeout=60.0, workers=1)
            reports.append(
                (
                    audit.headline,
                    audit.declared_seed_headline,
                    audit.scripts_with_seed_movement,
                    tuple(sorted(audit.status_counts().items())),
                    audit.claim_tokens_total,
                    audit.claim_tokens_compared,
                    audit.static_finding_count,
                )
            )
        out(f"report 1: {reports[0]}")
        out(f"report 2: {reports[1]}")
        out.check(
            "two runs of the audit over one fixture produce identical reports",
            reports[0] == reports[1],
        )
        out.check(
            "the audit found the fixture's one seed-dependent number both times",
            reports[0][0] == 1 and reports[1][0] == 1,
            f"headline {reports[0][0]} and {reports[1][0]}",
        )

    out.rule("6. THIS SCRIPT'S OWN WALL-CLOCK MARGIN AGAINST THE RELEASE GATE")
    elapsed = time.monotonic() - started
    out("the release gate fails a validation script at 600 s. An earlier version")
    out("of this script measured 473 s on a machine with both cores busy, which")
    out("is inside the ceiling and not safely inside it. The subset was reduced")
    out("and the margin is reported here so a regression is visible.")
    out("")
    out(f"declared budget for this script, seconds: {SELF_BUDGET!r}")
    out("release gate ceiling, seconds           : 600.0")
    out(f"measured elapsed, seconds               : {elapsed:.1f}")
    out(f"margin against the gate ceiling, seconds: {600.0 - elapsed:.1f}")
    out.check(
        "this script finishes inside the budget it declares for itself",
        elapsed <= SELF_BUDGET,
        f"{elapsed:.1f} s against {SELF_BUDGET!r} s",
    )
    return out.finish()


if __name__ == "__main__":
    sys.exit(main())
