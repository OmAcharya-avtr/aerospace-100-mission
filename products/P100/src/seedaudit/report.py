"""Declared constants and text rendering, in one place.

Every seed, confidence level and repeat budget this product uses is declared
here and nowhere else, so that no two reported numbers can rest on different
ones. A tool that audits reproducibility and scatters its own constants across
its modules has already lost the argument.
"""

from __future__ import annotations

from .audit import EXCLUSION_REASONS, RepositoryAudit
from .power import DEFAULT_ALPHA, MISSION_RATE, PowerRow, power_table

#: The one declared seed for every stochastic step in this product's own tests,
#: examples and validation scripts. Analysis of a measurement is never allowed
#: to be stochastic in a way the reader cannot reproduce.
SEED = 1000044

#: Confidence level of every reported interval.
CONFIDENCE = 0.95

#: Bootstrap resamples for every reported interval.
BOOTSTRAP_RESAMPLES = 2000

#: The repeat budgets the published power table is rendered at.
REPORTED_REPEATS: tuple[int, ...] = (2, 3, 5, 10, 20, 44, 100, 131, 200, 500, 1000)

#: Line-local markers that a printed quantity is a wall-clock reading or is
#: derived from one, and is therefore a property of the machine rather than of
#: the computation. Used ONLY to GROUP findings in a report so a reader can see
#: what they are; never to filter one out. A wall-clock reading written into a
#: file the repository treats as reproducible evidence is still a number that
#: is not reproducible, and grouping it is not dismissing it.
#:
#: The list was widened once, after a full audit of this mission's repository
#: found 349 numbers moving at the declared seed of which the first, narrow
#: list recognised only 21. Inspecting the rest showed the misses were "ms",
#: "wall clock" against the listed "wall time", bare unit suffixes such as
#: "<#> s," inside a sentence, and energy in watt-hours, which is a power
#: integrated over a measured duration.
#:
#: THIS IS A KEYWORD TEST ON ONE LINE, NOT A CLAIM ABOUT WHAT A NUMBER MEANS.
#: A table row whose unit lives in a column header several lines above carries
#: no marker and is counted as unmarked. The unmarked count is therefore an
#: UPPER bound on the non-timing findings, never a lower one, and the report
#: says so wherever it prints the split.
DURATION_MARKERS: tuple[str, ...] = (
    "elapsed",
    "wall time",
    "wall clock",
    "wall-clock",
    "walltime",
    "runtime",
    "run time",
    "seconds",
    "training time",
    "fit:",
    "fit ",
    "duration",
    "on this machine",
    "cpu time",
    "timing",
    " s,",
    " s ",
    " s]",
    " ms",
    " us",
    " ns",
    " s\n",
    "predict ",
    "solve ",
    "w*h",
    "wh]",
    "per second",
    "throughput",
)

#: Time units that can terminate a line, checked separately because a marker
#: test with a trailing separator cannot see a unit with nothing after it.
TRAILING_TIME_UNITS: tuple[str, ...] = (" s", " ms", " us", " ns", " h", " min", " hz")

#: The statement that goes at the head of every output, so that a reader of a
#: single file knows what the numbers do and do not claim.
AUDIT_MODEL = (
    "A committed numeric claim is a numeric token in a committed output file "
    "beside the script that produced it, where that script declares a "
    "rewritable random seed. Each candidate is run twice in two separate "
    "pristine copies: PERTURBED, with every seed literal rewritten, and "
    "CONTROL, with the seeds left alone. A number that moves in the perturbed "
    "run and not in the control is seed-attributable and is the headline. A "
    "number that moves in the control was never reproducible at its declared "
    "seed and is counted separately. Detection power is "
    "1-(1-p)^n against a committed reference and 1-p^n-(1-p)^n for "
    "run-and-diff with no reference; a clean result reports the escaped-rate "
    "upper bound 1-alpha^(1/n), never zero."
)


def looks_like_a_duration(skeleton: str) -> bool:
    """Whether a report line looks like it prints a wall-clock duration.

    A declared keyword test over :data:`DURATION_MARKERS`, case-insensitive,
    applied to ONE line. It groups findings in a report and never removes one:
    a duration written into a committed evidence file is still a number the
    repository is claiming is reproducible, and it is not.

    A line whose unit lives in a column header elsewhere in the file carries no
    marker and is reported as unmarked, so the unmarked count is an UPPER bound
    on the non-timing findings.

    Args:
        skeleton: the report line with its numbers replaced.

    Returns:
        True when the line carries a duration marker.
    """
    lowered = str(skeleton).lower().rstrip()
    if any(marker in lowered for marker in DURATION_MARKERS):
        return True
    # A unit at the very end of the line, which no "marker in line" test with a
    # trailing separator can see: "... , <#> s" has nothing after the unit.
    return lowered.endswith(TRAILING_TIME_UNITS)


def format_power_table(
    rate: float = MISSION_RATE,
    repeats: tuple[int, ...] = REPORTED_REPEATS,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    """Render the closed-form detection-power table as fixed-width text.

    Args:
        rate: the per-run manifestation rate the detection columns are
            evaluated at, dimensionless.
        repeats: the repeat budgets, dimensionless.
        alpha: one-sided significance level, dimensionless.

    Returns:
        The table, including a header naming the rate.
    """
    rows: tuple[PowerRow, ...] = power_table(repeats, rate, alpha)
    lines = [
        f"detection power at p = {rate!r}, alpha = {alpha!r}",
        f"{'n':>6}  {'P_ref':>18}  {'P_pair':>18}  {'p_min(n)':>18}  {'p_max(n) clean':>18}",
    ]
    for row in rows:
        lines.append(
            f"{row.repeats:>6}  {row.reference:>18.12f}  {row.pairwise:>18.12f}"
            f"  {row.smallest_detectable:>18.12f}  {row.escaped_upper_bound:>18.12f}"
        )
    return "\n".join(lines)


def format_exclusions(exclusions: tuple[str, ...]) -> str:
    """Render the declared exclusions with the reason for each."""
    if not exclusions:
        return "exclusions: none"
    lines = ["exclusions, declared with the reason for each:"]
    for item in exclusions:
        reason = EXCLUSION_REASONS.get(item, "no reason recorded, which is itself a defect")
        lines.append(f"  {item}: {reason}")
    return "\n".join(lines)


def format_audit(audit: RepositoryAudit) -> str:
    """Render a repository audit: headline, coverage, then findings.

    Coverage comes before the findings on purpose. A finding count without a
    denominator is the thing this product exists to object to.
    """
    counts = audit.status_counts()
    lines = [
        f"repository: {audit.root_name}",
        f"roots scanned: {', '.join(audit.roots)}",
        format_exclusions(audit.exclusions),
        "",
        "HEADLINE",
        f"  committed numbers claimed reproducible that MOVE under a seed change: {audit.headline}",
        f"  scripts contributing at least one: {len(audit.scripts_with_seed_movement)}",
        f"  committed numbers that move at the DECLARED seed (separate, worse): "
        f"{audit.declared_seed_headline}",
        f"  scripts contributing at least one of those: "
        f"{len(audit.scripts_with_declared_seed_movement)}",
        "",
        "COVERAGE",
        f"  candidates found: {len(audit.candidates)}",
        f"  candidates attempted: {len(audit.results)}",
        f"  compared: {counts.get('compared', 0)}",
        f"  no_output (claim unevaluable, NOT clean): {counts.get('no_output', 0)}",
        f"  run_failed: {counts.get('run_failed', 0)}",
        f"  timeout: {counts.get('timeout', 0)}",
        f"  global deadline reached: {audit.deadline_reached}",
        f"  committed numeric tokens across all candidates: {audit.claim_tokens_total}",
        f"  numeric tokens the perturbed comparison evaluated: {audit.claim_tokens_compared}",
        f"  repeats per claim: {audit.repeats_per_claim}",
        f"  escaped-rate upper bound at that repeat count, alpha=0.05: "
        f"{audit.escaped_bound!r}",
        "",
        "STATIC SELF-POLLUTION SCREEN",
        f"  python files parsed: {audit.files_scanned}",
        f"  files with at least one finding: {len(audit.static_findings)}",
        f"  findings in total: {audit.static_finding_count}",
        f"  files that would not parse: {len(audit.parse_failures)}",
    ]
    return "\n".join(lines)
