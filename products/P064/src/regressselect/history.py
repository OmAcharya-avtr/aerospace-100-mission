"""Committed fixture history: revisions, their changed modules and failures.

A history is the only input that makes a history-based or learned selector
possible, and it is the input that is hardest to get honestly. This module
defines the on-disk form, loads it, and splits it *chronologically* -- never
at random -- because a random split lets a model see a revision's neighbours
and inflates every number that follows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Revision", "History", "load_history", "save_history"]

SCHEMA = "regressselect-history/1"


@dataclass(frozen=True)
class Revision:
    """One revision: what changed, by how much, and which tests then failed.

    ``churn`` is lines added plus lines deleted per changed module (unit:
    lines). ``failed_tests`` are pytest node ids that failed *at this
    revision*; an empty tuple means the revision was green.
    """

    revision: str
    changed_modules: tuple[str, ...]
    churn: dict[str, int] = field(default_factory=dict)
    failed_tests: tuple[str, ...] = ()

    @property
    def is_regression(self) -> bool:
        """True when at least one test failed at this revision."""
        return len(self.failed_tests) > 0

    def total_churn(self) -> int:
        """Sum of churn over the changed modules (lines)."""
        return sum(self.churn.values())

    def max_churn(self) -> int:
        """Largest per-module churn at this revision (lines)."""
        return max(self.churn.values()) if self.churn else 0


@dataclass(frozen=True)
class History:
    """An ordered sequence of revisions over a fixed test inventory."""

    test_ids: tuple[str, ...]
    revisions: tuple[Revision, ...]
    project_root: str = ""
    seed: int | None = None

    def __len__(self) -> int:
        return len(self.revisions)

    @property
    def regression_count(self) -> int:
        """Number of revisions with at least one failing test."""
        return sum(1 for r in self.revisions if r.is_regression)

    def split_chronological(self, train_fraction: float) -> tuple[History, History]:
        """Split into (earlier, later) at ``train_fraction`` of the revisions.

        ``train_fraction`` is a dimensionless fraction in (0, 1). The split is
        by position in the recorded order, which is the revision order; this is
        the only split that does not leak future outcomes into training.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        cut = int(round(len(self.revisions) * train_fraction))
        if cut < 1 or cut >= len(self.revisions):
            raise ValueError(
                f"train_fraction {train_fraction} leaves an empty side for "
                f"{len(self.revisions)} revisions"
            )
        return (
            History(self.test_ids, self.revisions[:cut], self.project_root, self.seed),
            History(self.test_ids, self.revisions[cut:], self.project_root, self.seed),
        )


def load_history(path: str | Path) -> History:
    """Load a history from the committed JSON fixture form."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = raw.get("schema")
    if schema != SCHEMA:
        raise ValueError(f"unsupported history schema {schema!r}, expected {SCHEMA!r}")
    revisions = tuple(
        Revision(
            revision=r["revision"],
            changed_modules=tuple(r["changed_modules"]),
            churn=dict(r.get("churn", {})),
            failed_tests=tuple(r.get("failed_tests", ())),
        )
        for r in raw["revisions"]
    )
    return History(
        test_ids=tuple(raw["test_ids"]),
        revisions=revisions,
        project_root=raw.get("project_root", ""),
        seed=raw.get("seed"),
    )


def save_history(history: History, path: str | Path) -> None:
    """Write a history to the committed JSON fixture form (stable key order)."""
    payload = {
        "schema": SCHEMA,
        "project_root": history.project_root,
        "seed": history.seed,
        "test_ids": list(history.test_ids),
        "revisions": [
            {
                "revision": r.revision,
                "changed_modules": list(r.changed_modules),
                "churn": dict(sorted(r.churn.items())),
                "failed_tests": list(r.failed_tests),
            }
            for r in history.revisions
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8")
