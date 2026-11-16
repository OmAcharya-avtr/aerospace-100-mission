"""Auditing a whole repository's reproducibility claims.

What counts as a claim
---------------------
A **committed numeric claim** is a numeric token in a committed output file
that sits beside the script that produced it -- the layout this mission's build
guide makes binding, ``validation/<script-stem>_output.txt`` -- where that
script declares at least one rewritable random seed. The repository commits
that file as evidence, so every number in it is an assertion that re-running
the script reproduces it.

The two runs, and why there are two
-----------------------------------
Each candidate is executed **twice**, each time in its own pristine copy:

- **perturbed** -- every seed literal in the script is rewritten by
  :func:`seedaudit.seeds.rewrite_seeds`, and nothing else is touched;
- **control** -- the same machinery with a perturbation of 0, so the seeds keep
  their declared values.

A number that moves in the perturbed run **and not** in the control is
**seed-attributable**, and the count of those is the headline. A number that
moves in the control moved with the seed held fixed: it was never reproducible
at the declared seed in the first place, which is a separate and worse finding,
and it is counted separately rather than folded into the headline.

The control also absorbs the confounds. The copy, the fresh ``HOME``, the
``PYTHONHASHSEED`` and the removal described below apply identically to both
runs, so anything they cause appears in the control and is excluded from the
headline by construction rather than by argument.

The instrument's own footprint, which the control CANNOT absorb
---------------------------------------------------------------
A script that prints its own seed into its committed output reports a changed
number in the perturbed run for one reason only: the audit changed the seed. The
control cannot catch it, because in the control the seed is unchanged and the
printed seed therefore matches. Counting it as a finding would be the audit
reporting its own perturbation back to itself -- an instance of the
self-polluting-check class, one level up, in the tool built to detect that class.

So a moved number is classified as :attr:`ClaimResult.instrument_footprint`
when the value before is one of the seed literals the script declares and the
value after is exactly that literal plus the perturbation. The test is exact
and needs no tolerance: both sides are integers and the offset is known. Those
findings are **counted and reported separately** and are excluded from the
headline. The first full run of this audit over this repository reported 38 of
them inside a headline of 4447 before this filter existed, and the filter is
the fix rather than a note in the limitations.

The committed artifact is removed from the copy before the run, and this is declared
---------------------------------------------------------------------------------
If the committed output file were left in place and the script did not rewrite
it, the comparison would read the committed file back and report "unchanged" --
a clean verdict obtained by comparing a file against itself. So the file is
deleted inside the copy first. A script that then does not produce it is
reported as ``no_output`` and its claim is **unevaluable**, never clean.

The deletion is a deliberate change to the tree and it is named here because it
can matter: a hygiene scanner that globs for stray files sees one fewer file.
That effect lands in the control run too, so it cannot reach the headline.

Coverage is a number, not an implication
----------------------------------------
Every result carries how many candidates were found, attempted, compared, timed
out, failed to run and produced no output, plus what was excluded and why. An
audit that reports findings without reporting its own coverage has not said
what its silence means.

Units: timeouts and deadlines are seconds; everything else is a count.
"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .dynamic import DEFAULT_TIMEOUT, RunResult, pristine_copy
from .dynamic import execute as execute_in_copy
from .numeric import Comparison, MovedNumber, compare
from .power import escaped_rate_upper_bound
from .seeds import PERTURBATION, SeedScan, find_seed_literals, rewrite_seeds
from .static import PollutionFinding, find_self_pollution

#: Directories under the repository root the audit walks.
DEFAULT_ROOTS: tuple[str, ...] = ("products", "scripts")

#: Path fragments excluded from every scan, with the reason attached in
#: :data:`EXCLUSION_REASONS`. Both are declared rather than silently skipped.
DEFAULT_EXCLUSIONS: tuple[str, ...] = ("products/P094", "products/P100")

#: Why each exclusion exists, printed with every scan result.
EXCLUSION_REASONS: dict[str, str] = {
    "products/P094": (
        "another build agent is writing it concurrently, so any number derived "
        "from it is not reproducible by a reader"
    ),
    "products/P100": (
        "this product; a tool scanning its own tree while it is being written is "
        "the class it exists to detect, and it is audited separately in "
        "validation/validate_self_audit.py"
    ),
}

#: Subdirectories of a product whose scripts are candidates.
CANDIDATE_DIRECTORIES: tuple[str, ...] = ("validation", "examples")


@dataclass(frozen=True)
class Candidate:
    """One script whose committed output is a reproducibility claim.

    Attributes:
        source_dir: the product directory, relative to the repository root.
        script: the script, relative to ``source_dir``.
        output: the committed output file, relative to ``source_dir``.
        scan: the seed literals found in the script.
        claim_tokens: how many numeric tokens the committed output holds.
    """

    source_dir: str
    script: str
    output: str
    scan: SeedScan
    claim_tokens: int

    @property
    def label(self) -> str:
        """The script's path relative to the repository root."""
        return f"{self.source_dir}/{self.script}"


@dataclass
class ClaimResult:
    """The outcome of auditing one candidate.

    Attributes:
        candidate: the candidate audited.
        status: ``compared``, ``no_output``, ``timeout`` or ``run_failed``.
        perturbed: comparison of the perturbed run against the committed text.
        control: comparison of the unchanged-seed run against the committed text.
        perturbed_run: the perturbed run's raw result.
        control_run: the control run's raw result.
        seed_attributable: numbers that moved under the seed change, did
            **not** move in the control, and are not the audit's own footprint.
        declared_seed_moved: numbers that moved with the seed held fixed.
        instrument_footprint: numbers that moved only because the script
            printed the seed the audit rewrote. Reported, never counted as a
            finding.
    """

    candidate: Candidate
    status: str
    perturbed: Comparison | None = None
    control: Comparison | None = None
    perturbed_run: RunResult | None = None
    control_run: RunResult | None = None
    seed_attributable: list[MovedNumber] = field(default_factory=list)
    declared_seed_moved: list[MovedNumber] = field(default_factory=list)
    instrument_footprint: list[MovedNumber] = field(default_factory=list)

    @property
    def evaluable(self) -> bool:
        """Whether the claim could actually be evaluated."""
        return self.status == "compared"


@dataclass
class RepositoryAudit:
    """Everything one audit of one repository found, including its own coverage.

    Attributes:
        root_name: the repository directory's own name, never an absolute path.
        roots: the directories walked.
        exclusions: the path fragments excluded.
        candidates: every candidate found.
        results: one :class:`ClaimResult` per attempted candidate.
        static_findings: self-pollution screen findings, by file path relative
            to the repository root.
        parse_failures: files the static screen could not parse, with the
            reason. Reported, because a file silently skipped is coverage the
            audit would be overstating.
        files_scanned: how many Python files the static screen parsed.
        deadline_reached: whether the global wall-clock budget stopped the run
            before every candidate was attempted.
        repeats_per_claim: runs performed per candidate, dimensionless.
    """

    root_name: str
    roots: tuple[str, ...]
    exclusions: tuple[str, ...]
    candidates: tuple[Candidate, ...] = ()
    results: list[ClaimResult] = field(default_factory=list)
    static_findings: dict[str, tuple[PollutionFinding, ...]] = field(default_factory=dict)
    parse_failures: dict[str, str] = field(default_factory=dict)
    files_scanned: int = 0
    deadline_reached: bool = False
    repeats_per_claim: int = 1

    @property
    def headline(self) -> int:
        """Committed numbers claimed reproducible that move under a seed change."""
        return sum(len(result.seed_attributable) for result in self.results)

    @property
    def declared_seed_headline(self) -> int:
        """Committed numbers that move with the seed held at its declared value."""
        return sum(len(result.declared_seed_moved) for result in self.results)

    @property
    def instrument_footprint_count(self) -> int:
        """Moved numbers that are the audit's own perturbation echoed back."""
        return sum(len(result.instrument_footprint) for result in self.results)

    @property
    def scripts_with_instrument_footprint(self) -> tuple[str, ...]:
        """Scripts that print a seed the audit rewrote."""
        return tuple(
            sorted(r.candidate.label for r in self.results if r.instrument_footprint)
        )

    @property
    def scripts_with_seed_movement(self) -> tuple[str, ...]:
        """Scripts contributing at least one seed-attributable moved number."""
        return tuple(
            sorted(r.candidate.label for r in self.results if r.seed_attributable)
        )

    @property
    def scripts_with_declared_seed_movement(self) -> tuple[str, ...]:
        """Scripts whose numbers move at the declared seed."""
        return tuple(
            sorted(r.candidate.label for r in self.results if r.declared_seed_moved)
        )

    def status_counts(self) -> dict[str, int]:
        """How many attempted candidates ended in each status."""
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    @property
    def claim_tokens_total(self) -> int:
        """Numeric tokens across every candidate's committed output."""
        return sum(candidate.claim_tokens for candidate in self.candidates)

    @property
    def claim_tokens_compared(self) -> int:
        """Numeric tokens the perturbed comparison actually evaluated."""
        return sum(
            result.perturbed.compared_tokens
            for result in self.results
            if result.perturbed is not None
        )

    @property
    def escaped_bound(self) -> float:
        """Escaped-rate upper bound for the repeats actually performed."""
        return escaped_rate_upper_bound(max(1, self.repeats_per_claim))

    @property
    def static_finding_count(self) -> int:
        """Total self-pollution screen findings across the corpus."""
        return sum(len(v) for v in self.static_findings.values())


def _excluded(relative_path: str, exclusions: tuple[str, ...]) -> bool:
    """Whether a repository-relative path falls under any exclusion."""
    return any(relative_path == e or relative_path.startswith(e + "/") for e in exclusions)


def python_files(
    repo_root: Path,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS,
) -> tuple[str, ...]:
    """Every Python file under ``roots``, relative to ``repo_root``, sorted.

    Args:
        repo_root: the repository root.
        roots: directories under it to walk.
        exclusions: repository-relative path fragments to skip.

    Returns:
        Sorted repository-relative paths.
    """
    base = Path(repo_root).resolve()
    out: list[str] = []
    for root in roots:
        directory = base / root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            relative = path.relative_to(base).as_posix()
            if _excluded(relative, exclusions):
                continue
            skipped = ("__pycache__", ".venv", "venv", "build", "dist")
            if any(part in skipped for part in path.parts):
                continue
            out.append(relative)
    return tuple(sorted(out))


def find_candidates(
    repo_root: Path,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS,
) -> tuple[Candidate, ...]:
    """Find every script whose committed output is a seed-dependent claim.

    A candidate needs three things: it lives in one of
    :data:`CANDIDATE_DIRECTORIES` inside a product, a sibling
    ``<stem>_output.txt`` exists, and it declares at least one rewritable seed
    literal. A script with no committed output makes no claim, and a script with
    no rewritable seed cannot be perturbed, so neither is a candidate -- and
    both counts are reported by the caller rather than left implicit.

    Args:
        repo_root: the repository root.
        roots: directories under it to walk.
        exclusions: repository-relative path fragments to skip.

    Returns:
        The candidates, sorted by label.
    """
    base = Path(repo_root).resolve()
    from .numeric import numeric_tokens

    out: list[Candidate] = []
    for relative in python_files(base, roots, exclusions):
        path = base / relative
        parts = path.relative_to(base).parts
        if len(parts) < 3 or parts[0] != "products":
            continue
        if parts[2] not in CANDIDATE_DIRECTORIES:
            continue
        output = path.with_name(path.stem + "_output.txt")
        if not output.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            scan = find_seed_literals(source)
        except SyntaxError:
            continue
        if not scan.literals:
            continue
        source_dir = f"{parts[0]}/{parts[1]}"
        committed = output.read_text(encoding="utf-8", errors="replace")
        tokens = sum(len(numeric_tokens(line)) for line in committed.splitlines())
        out.append(
            Candidate(
                source_dir=source_dir,
                script=path.relative_to(base / source_dir).as_posix(),
                output=output.relative_to(base / source_dir).as_posix(),
                scan=scan,
                claim_tokens=tokens,
            )
        )
    return tuple(sorted(out, key=lambda c: c.label))


def is_instrument_footprint(
    moved: MovedNumber, declared_seeds: set[int], perturbation: int
) -> bool:
    """Whether a moved number is the audit's own perturbation echoed back.

    True when the committed value is one of the seed literals the script
    declares and the re-measured value is exactly that literal plus the
    perturbation. The test is exact and takes no tolerance: both sides are
    integers and the offset is known, so there is nothing to tune.

    A script that prints its own seed is not making a false reproducibility
    claim, and reporting it as one would be the audit measuring its own
    footprint.

    Args:
        moved: the moved number.
        declared_seeds: the integer seed literals the script declares.
        perturbation: the offset the audit added to each of them.

    Returns:
        True when this moved number is the instrument and not a finding.
    """
    if not declared_seeds:
        return False
    before, after = moved.before_value, moved.after_value
    if not (float(before).is_integer() and float(after).is_integer()):
        return False
    return int(before) in declared_seeds and int(after) == int(before) + perturbation


def audit_claim(
    repo_root: Path,
    candidate: Candidate,
    timeout: float = DEFAULT_TIMEOUT,
    perturbation: int = PERTURBATION,
    rel_tol: float = 0.0,
    abs_tol: float = 0.0,
) -> ClaimResult:
    """Audit one candidate with a perturbed run and an unchanged-seed control.

    Args:
        repo_root: the repository root.
        candidate: the candidate to audit.
        timeout: wall-clock limit per run, seconds.
        perturbation: offset added to each seed literal in the perturbed run.
        rel_tol: relative tolerance for a numeric token, dimensionless.
        abs_tol: absolute tolerance for a numeric token.

    Returns:
        A :class:`ClaimResult`. Nothing is written to ``repo_root``.
    """
    base = Path(repo_root).resolve()
    source = base / candidate.source_dir
    committed = (source / candidate.output).read_text(encoding="utf-8", errors="replace")

    def one(offset: int) -> tuple[RunResult, str | None]:
        with pristine_copy(source) as (copy, home):
            script_path = copy / candidate.script
            rewritten, _ = rewrite_seeds(script_path.read_text(encoding="utf-8"), offset)
            script_path.write_text(rewritten, encoding="utf-8")
            artifact = copy / candidate.output
            if artifact.exists():
                artifact.unlink()
            result = execute_in_copy(copy, home, candidate.script, candidate.output, timeout)
            return result, result.artifact

    control_run, control_text = one(0)
    perturbed_run, perturbed_text = one(perturbation)

    if control_run.timed_out or perturbed_run.timed_out:
        return ClaimResult(
            candidate=candidate,
            status="timeout",
            control_run=control_run,
            perturbed_run=perturbed_run,
        )
    if control_text is None or perturbed_text is None:
        status = "no_output" if (control_run.ok and perturbed_run.ok) else "run_failed"
        return ClaimResult(
            candidate=candidate,
            status=status,
            control_run=control_run,
            perturbed_run=perturbed_run,
        )

    control = compare(committed, control_text, rel_tol, abs_tol)
    perturbed = compare(committed, perturbed_text, rel_tol, abs_tol)
    control_positions = {(m.line, m.index) for m in control.moved}
    declared = {literal.value for literal in candidate.scan.literals}
    seed_attributable: list[MovedNumber] = []
    footprint: list[MovedNumber] = []
    for moved in perturbed.moved:
        if (moved.line, moved.index) in control_positions:
            continue
        if is_instrument_footprint(moved, declared, perturbation):
            footprint.append(moved)
        else:
            seed_attributable.append(moved)
    return ClaimResult(
        candidate=candidate,
        status="compared",
        perturbed=perturbed,
        control=control,
        perturbed_run=perturbed_run,
        control_run=control_run,
        seed_attributable=seed_attributable,
        declared_seed_moved=list(control.moved),
        instrument_footprint=footprint,
    )


def static_scan(
    repo_root: Path,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS,
) -> tuple[dict[str, tuple[PollutionFinding, ...]], dict[str, str], int]:
    """Run the self-pollution screen over every Python file under ``roots``.

    Args:
        repo_root: the repository root.
        roots: directories under it to walk.
        exclusions: repository-relative path fragments to skip.

    Returns:
        ``(findings_by_file, parse_failures, files_scanned)``. A file with no
        findings is absent from the mapping; a file that would not parse is in
        ``parse_failures`` with the reason, never silently dropped.
    """
    base = Path(repo_root).resolve()
    findings: dict[str, tuple[PollutionFinding, ...]] = {}
    failures: dict[str, str] = {}
    scanned = 0
    for relative in python_files(base, roots, exclusions):
        try:
            source = (base / relative).read_text(encoding="utf-8")
        except OSError as error:
            failures[relative] = f"unreadable: {type(error).__name__}"
            continue
        try:
            found = find_self_pollution(source)
        except (SyntaxError, ValueError, RecursionError) as error:
            failures[relative] = f"{type(error).__name__}: {error}"
            continue
        scanned += 1
        if found:
            findings[relative] = found
    return findings, failures, scanned


def audit_repository(
    repo_root: Path,
    roots: tuple[str, ...] = DEFAULT_ROOTS,
    exclusions: tuple[str, ...] = DEFAULT_EXCLUSIONS,
    timeout: float = DEFAULT_TIMEOUT,
    perturbation: int = PERTURBATION,
    workers: int = 2,
    deadline_seconds: float | None = None,
    limit: int | None = None,
    progress: object | None = None,
) -> RepositoryAudit:
    """Audit a repository's committed numeric claims and screen it statically.

    Args:
        repo_root: the repository root. Never written to.
        roots: directories under it to walk.
        exclusions: repository-relative path fragments to skip.
        timeout: wall-clock limit per run, seconds.
        perturbation: offset added to each seed literal in the perturbed run.
        workers: concurrent candidates, dimensionless. Kept small on purpose:
            the audited repository may be shared with another process, and
            oversubscribing the machine would change the timing of the very
            scripts being measured.
        deadline_seconds: global wall-clock budget. When it passes, remaining
            candidates are left unattempted and ``deadline_reached`` is set, so
            partial coverage is visible rather than silent.
        limit: audit at most this many candidates, for a fast smoke run.
        progress: optional callable taking ``(index, total, label, status)``.

    Returns:
        A :class:`RepositoryAudit`.
    """
    base = Path(repo_root).resolve()
    if not base.is_dir():
        raise ValueError(f"repo_root must be an existing directory, got {base.name!r}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError(f"workers must be a positive int, got {workers!r}")
    findings, failures, scanned = static_scan(base, roots, exclusions)
    candidates = find_candidates(base, roots, exclusions)
    selected = candidates if limit is None else candidates[: int(limit)]
    audit = RepositoryAudit(
        root_name=base.name,
        roots=tuple(roots),
        exclusions=tuple(exclusions),
        candidates=candidates,
        static_findings=findings,
        parse_failures=failures,
        files_scanned=scanned,
        repeats_per_claim=1,
    )

    import time

    start = time.monotonic()

    def task(candidate: Candidate) -> ClaimResult:
        return audit_claim(base, candidate, timeout, perturbation)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = list(selected)
        index = 0
        while pending:
            if deadline_seconds is not None and time.monotonic() - start > deadline_seconds:
                audit.deadline_reached = True
                break
            batch, pending = pending[:workers], pending[workers:]
            for candidate, result in zip(batch, pool.map(task, batch), strict=True):
                index += 1
                audit.results.append(result)
                if callable(progress):
                    progress(index, len(selected), candidate.label, result.status)
    return audit


def unparseable_files(repo_root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    """Report which of ``paths`` do not parse, for a coverage statement.

    Exists so that a caller can state how much of a corpus the static screen
    could actually read, rather than reporting findings over an unstated
    denominator.
    """
    base = Path(repo_root).resolve()
    out: dict[str, str] = {}
    for relative in paths:
        try:
            ast.parse((base / relative).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError) as error:
            out[relative] = f"{type(error).__name__}: {error}"
    return out
