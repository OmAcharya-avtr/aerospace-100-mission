"""The corpus-wide audit campaign. NOT a validation script: a data generator.

This program executes every candidate script in a repository twice, each run in
its own pristine copy, and writes the itemised evidence the fast validation
check then verifies. It takes most of an hour on two cores.

**Why it lives in `data/` and not in `validation/`.**

A 70-minute corpus-wide execution campaign is a data-generation campaign, not a
validation re-check. The mission's release gate runs every script in
`validation/` with a 600 s per-script ceiling, and this program is seven times
over it. More importantly, the gate runs those scripts **in sorted order**, so
an audit living in `validation/` would run immediately before
`validate_repository_findings.py` and **rewrite the evidence that script
pins** — from a tree that has changed and under a machine load that is not the
one the pins were taken under. A script that silently overwrites the file
another script pins is precisely the self-polluting-check class this product
exists to name, and it arrived in this product's own validation directory.

So the campaign is here, its output is committed as frozen evidence at
`validation/repository_findings.json`, and `validation/` holds only the fast
check that verifies that evidence. Committing a campaign's output and
verifying it is not a weaker claim than re-running it; it is the same claim
with its provenance stated.

**It cannot clobber the evidence by accident.** It writes to the directory
given by ``--out``, which defaults to a fresh temporary directory. Refreshing
the committed evidence requires the explicit ``--refresh-committed`` flag, and
the program refuses to write into `validation/` without it.

Usage::

    python3 data/repository_audit.py                      # to a temp directory
    python3 data/repository_audit.py --out /some/dir
    python3 data/repository_audit.py --refresh-committed   # overwrite evidence
    python3 data/repository_audit.py --limit 8             # a smoke run

Nothing is ever written to the repository being audited. Every execution
happens in a copy in a temporary directory.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from seedaudit.audit import (  # noqa: E402
    DEFAULT_EXCLUSIONS,
    DEFAULT_ROOTS,
    EXCLUSION_REASONS,
    audit_repository,
)
from seedaudit.report import AUDIT_MODEL, looks_like_a_duration  # noqa: E402

#: Wall-clock limit for one execution of one audited script, seconds. Measured
#: on a 14-script sample of this corpus: mean 5.21 s, worst 23.80 s. 75 s is
#: about three times the worst observed. A script that exceeds it is reported
#: as a timeout rather than silently dropped -- and the timeout SET is a
#: property of the machine as well as of the repository, which is why the
#: validation check reports it as a bound and does not pin it.
TIMEOUT = 75.0

#: Concurrent candidates. Kept at 2 on a 2-core machine: oversubscribing would
#: change the timing of the very scripts being measured.
WORKERS = 2

#: Global wall-clock budget, seconds.
DEADLINE = 6000.0

#: Where the committed evidence lives, relative to the product root.
COMMITTED_EVIDENCE = "validation/repository_findings.json"

#: The campaign's own transcript, written beside whatever evidence it produces.
TRANSCRIPT = "repository_audit_output.txt"


def relative(path: Path) -> str:
    """Render a path relative to the product root, never absolutely."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


class Transcript:
    """Collect lines, print them, and write them beside the evidence."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / TRANSCRIPT
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        for line in str(text).splitlines() or [""]:
            self.lines.append(line)
            print(line, flush=True)

    def rule(self, title: str) -> None:
        """Print a section rule."""
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def write(self) -> None:
        """Write the transcript, replacing any previous one."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()
        self.path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def repository_root() -> Path | None:
    """The mission repository root, if this product sits inside one."""
    for parent in [ROOT, *ROOT.parents]:
        if (parent / "products.yaml").is_file() and (parent / "products").is_dir():
            return parent
    return None


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python3 data/repository_audit.py",
        description=(
            "Run the corpus-wide audit campaign and write its evidence. "
            "Defaults to a temporary directory; refreshing the committed "
            "evidence requires --refresh-committed."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write the evidence and transcript into "
        "(default: a fresh temporary directory)",
    )
    parser.add_argument(
        "--refresh-committed",
        action="store_true",
        help="overwrite the committed evidence at " + COMMITTED_EVIDENCE,
    )
    parser.add_argument("--timeout", type=float, default=TIMEOUT)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--deadline", type=float, default=DEADLINE)
    parser.add_argument("--limit", type=int, default=None, help="smoke run only")
    return parser


def resolve_destination(args: argparse.Namespace) -> tuple[Path, Path]:
    """Decide where the transcript and the evidence go, refusing a silent clobber.

    Returns:
        ``(transcript_directory, evidence_path)``.

    Raises:
        SystemExit: if ``--out`` points into ``validation/`` without
            ``--refresh-committed``. A campaign that silently overwrites the
            file another script pins is the defect this product exists to name,
            so it is refused rather than warned about.
    """
    committed = (ROOT / COMMITTED_EVIDENCE).resolve()
    if args.refresh_committed:
        # The evidence replaces the committed file; the transcript stays beside
        # this program, which is where the committed transcript lives.
        return Path(__file__).resolve().parent, committed
    if args.out is None:
        # mkdtemp, not TemporaryDirectory: a campaign that deleted its own
        # output on the way out would be useless. The path is printed.
        directory = Path(tempfile.mkdtemp(prefix="seedaudit-campaign-"))
        return directory, directory / committed.name
    directory = Path(args.out).resolve()
    if directory == committed.parent:
        raise SystemExit(
            f"refusing to write into {relative(committed.parent)} without "
            "--refresh-committed: validate_repository_findings.py pins the "
            "evidence in that directory, and overwriting it from an ordinary "
            "run is the self-polluting-check class this product detects."
        )
    directory.mkdir(parents=True, exist_ok=True)
    return directory, directory / committed.name


def main(argv: list[str] | None = None) -> int:
    """Run the campaign and write its evidence."""
    args = build_parser().parse_args(argv)
    directory, evidence_path = resolve_destination(args)
    out = Transcript(directory)

    out("Repository audit campaign -- P100 SeedAudit")
    out("")
    out("THIS IS A DATA-GENERATION CAMPAIGN, NOT A VALIDATION SCRIPT. It takes")
    out("most of an hour and lives in data/ for that reason. The fast check that")
    out("verifies its output is validation/validate_repository_findings.py.")
    out("")
    out(AUDIT_MODEL)
    out("")
    out(f"evidence will be written to  : {relative(evidence_path)}")
    out(f"transcript will be written to: {relative(out.path)}")
    out(f"refresh-committed: {args.refresh_committed}")
    out(f"per-run timeout, seconds: {args.timeout!r}")
    out(f"workers: {args.workers}")
    out(f"global deadline, seconds: {args.deadline!r}")
    out(f"candidate limit: {args.limit}")

    repository = repository_root()
    if repository is None:
        out("")
        out("UNAVAILABLE: this product is not inside a mission repository, so the")
        out("corpus does not exist here. A missing corpus is not a code defect.")
        out.write()
        return 0

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
        timeout=args.timeout,
        workers=args.workers,
        deadline_seconds=args.deadline,
        limit=args.limit,
    )
    elapsed = time.monotonic() - started
    counts = audit.status_counts()

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
    out("")
    out("THE HEADLINE IS A LOWER BOUND, NOT A TOTAL. It counts moved numbers")
    out(f"among the {counts.get('compared', 0)} claims that completed inside the")
    out(f"{args.timeout!r} s per-run limit on this machine under this load. A")
    out("claim that timed out contributes nothing, so a machine with more")
    out("headroom would find at least as many and never fewer.")

    out.rule("2. COVERAGE, stated as a number")
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
    out(f"audit wall-clock seconds: {elapsed:.1f}")
    out(f"cpu count reported by the operating system: {os.cpu_count()}")
    out("")
    out("THE timeout AND no_output COUNTS ARE PROPERTIES OF THIS MACHINE AND THIS")
    out("LOAD AS WELL AS OF THE REPOSITORY. A re-run on a busier machine moves")
    out("candidates between `timeout` and `compared`. They are reported here and")
    out("are deliberately NOT pinned as exact sets by the validation check.")

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
    out("the static screen executes nothing. It is therefore the one part of this")
    out("campaign that is INDEPENDENT OF MACHINE LOAD, and it is the part the")
    out("validation check pins as exact literals.")
    out("")
    out("files with the most findings, with one finding each shown in full:")
    ranked = sorted(audit.static_findings.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for path, findings in ranked[:20]:
        out(f"  {path}: {len(findings)} findings")
        out(f"    {findings[0].describe()}")

    evidence = {
        "root_name": audit.root_name,
        "roots": list(audit.roots),
        "exclusions": list(audit.exclusions),
        "per_run_timeout_seconds": args.timeout,
        "workers": args.workers,
        "cpu_count": os.cpu_count(),
        "wall_clock_seconds": round(elapsed, 1),
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
        "declared_seed_duration_marked": declared_durations,
        "instrument_footprint_count": audit.instrument_footprint_count,
        "scripts_with_instrument_footprint": list(audit.scripts_with_instrument_footprint),
        "scripts_with_seed_movement": list(audit.scripts_with_seed_movement),
        "scripts_with_declared_seed_movement": list(audit.scripts_with_declared_seed_movement),
        "seed_attributable": seed_items,
        "declared_seed_moved": declared_items,
        "instrument_footprint": footprint_items,
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
    if evidence_path.exists():
        evidence_path.unlink()
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    out("")
    out(f"evidence written to {relative(evidence_path)}")
    out(f"transcript written to {relative(out.path)}")
    out.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
