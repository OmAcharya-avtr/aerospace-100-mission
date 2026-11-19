"""The headline staleness report over an assurance case.

The headline output is the STALE FRACTION: the share of evidence items in the
assurance case whose subject's dependency closure has changed since the
evidence was produced. A goal-weighted figure is reported beside it -- the
share of GSN leaf goals with at least one stale supporting solution -- because
an audit asks about claims, not about files, and one stale item can invalidate
a claim that several fresh items also support.
"""

from __future__ import annotations

from dataclasses import dataclass

from .baselines import (
    StalenessVerdicts,
    baseline_closure_hash,
    baseline_timestamp,
    baseline_timestamp_depth1,
    ground_truth,
)
from .bootstrap import BootstrapInterval, bootstrap_stale_fraction
from .corpus import Corpus
from .metrics import ErrorDirections, error_directions


@dataclass(frozen=True)
class StalenessReport:
    """Every method's verdicts over one corpus and one graph view."""

    corpus_name: str
    declarations: str
    evaluation_time: int
    truth: StalenessVerdicts
    timestamp: StalenessVerdicts
    timestamp_depth1: StalenessVerdicts
    closure_hash: StalenessVerdicts
    cluster_interval: BootstrapInterval
    item_interval: BootstrapInterval
    goal_stale_fraction: float
    goal_count: int

    def errors(self, method: str) -> ErrorDirections:
        """Error directions of one method against the truth."""
        table = {
            "baseline1_timestamp": self.timestamp,
            "baseline1b_timestamp_depth1": self.timestamp_depth1,
            "baseline2_closure_hash": self.closure_hash,
        }
        if method not in table:
            raise ValueError(
                f"method={method!r} is not scored in this report; pass one of "
                f"{sorted(table)}"
            )
        return error_directions(table[method].as_array(), self.truth.as_array())


def build_report(
    corpus: Corpus,
    declarations: str = "complete",
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 980098,
) -> StalenessReport:
    """Run all three baselines over ``corpus`` under the named graph view.

    ``declarations`` is ``"complete"`` (the true edge set) or ``"partial"``
    (the declared subset). The ground truth is always computed over the TRUE
    graph, whichever view the other methods see.
    """
    view = corpus.graph(declarations)
    items = corpus.evidence
    truth = ground_truth(items, corpus.history, corpus.true_graph, corpus.evaluation_time)
    timestamp = baseline_timestamp(items, corpus.history, view, corpus.evaluation_time)
    depth1 = baseline_timestamp_depth1(
        items, corpus.history, view, corpus.evaluation_time
    )
    closure = baseline_closure_hash(
        items, corpus.history, view, corpus.evaluation_time
    )
    labels = truth.as_array()
    subjects = corpus.subjects()
    cluster = bootstrap_stale_fraction(
        labels,
        subjects,
        unit="cluster",
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    item = bootstrap_stale_fraction(
        labels,
        subjects,
        unit="item",
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    solution_stale = {
        entry.solution: truth.verdicts[entry.identifier] for entry in items
    }
    children: dict[str, list[str]] = {}
    for link_type, source, target in corpus.case.links:
        if link_type == "SupportedBy":
            children.setdefault(source, []).append(target)
    leaf_goals = corpus.case.leaf_goals()
    stale_goals = sum(
        1
        for goal in leaf_goals
        if any(solution_stale.get(child, False) for child in children.get(goal, []))
    )
    goal_fraction = stale_goals / len(leaf_goals) if leaf_goals else 0.0
    return StalenessReport(
        corpus_name=corpus.name,
        declarations=declarations,
        evaluation_time=corpus.evaluation_time,
        truth=truth,
        timestamp=timestamp,
        timestamp_depth1=depth1,
        closure_hash=closure,
        cluster_interval=cluster,
        item_interval=item,
        goal_stale_fraction=goal_fraction,
        goal_count=len(leaf_goals),
    )


def render_report(report: StalenessReport) -> str:
    """The report as plain text, for the command-line interface."""
    lines: list[str] = []
    lines.append(f"corpus                       : {report.corpus_name}")
    lines.append(f"dependency declarations      : {report.declarations}")
    lines.append(f"evaluation revision          : {report.evaluation_time}")
    lines.append(f"evidence items               : {report.truth.total}")
    lines.append("")
    lines.append("HEADLINE -- stale fraction of the assurance case")
    lines.append(
        f"  closure comparison (baseline 2) : {report.closure_hash.stale_fraction:.6f}"
        f"  ({report.closure_hash.stale_count} of {report.closure_hash.total})"
    )
    lines.append(
        f"  ground truth (baseline 3)       : {report.truth.stale_fraction:.6f}"
        f"  ({report.truth.stale_count} of {report.truth.total})"
    )
    cluster = report.cluster_interval
    item = report.item_interval
    lines.append(
        f"  bootstrap, unit={cluster.unit:<8}        : "
        f"[{cluster.low:.6f}, {cluster.high:.6f}] "
        f"at {cluster.confidence:.2f}, {cluster.n_units} units, "
        f"{cluster.resamples} resamples"
    )
    lines.append(
        f"  bootstrap, unit={item.unit:<8}        : "
        f"[{item.low:.6f}, {item.high:.6f}] "
        f"at {item.confidence:.2f}, {item.n_units} units, "
        f"{item.resamples} resamples"
    )
    lines.append(
        f"  goal-weighted stale fraction    : {report.goal_stale_fraction:.6f}"
        f"  (over {report.goal_count} GSN leaf goals)"
    )
    lines.append("")
    lines.append("BASELINE 1 -- timestamp comparison, the thing being refuted")
    for method in ("baseline1_timestamp", "baseline1b_timestamp_depth1"):
        errors = report.errors(method)
        lines.append(f"  {method}")
        lines.append(
            f"    missed staleness : {errors.false_negative} of "
            f"{errors.actually_stale} truly stale "
            f"(rate {errors.missed_staleness_rate:.6f})"
        )
        lines.append(
            f"    false alarms     : {errors.false_positive} of "
            f"{errors.actually_current} truly current "
            f"(rate {errors.false_alarm_rate:.6f})"
        )
        lines.append(f"    agreement        : {errors.agreement:.6f}")
    lines.append("")
    lines.append("BASELINE 2 -- content-hash comparison over the dependency closure")
    errors = report.errors("baseline2_closure_hash")
    lines.append(
        f"  missed staleness : {errors.false_negative} of {errors.actually_stale}"
    )
    lines.append(
        f"  false alarms     : {errors.false_positive} of {errors.actually_current}"
    )
    lines.append(f"  agreement        : {errors.agreement:.6f}")
    lines.append("")
    lines.append(
        "This software is research-grade. It is not flight-qualified, not certified "
        "and not approved for operational aerospace use."
    )
    return "\n".join(lines)
