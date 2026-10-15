"""Baseline 2: history-based prioritisation by failure co-occurrence.

The estimator is a smoothed conditional failure rate. For a changed module
``M`` and a test node ``t``, over the training revisions,

    P(t fails | M changed) ~ (a(M, t) + alpha) / (b(M) + 2 * alpha)

where ``a(M, t)`` counts revisions in which ``M`` was among the changed
modules and ``t`` failed, ``b(M)`` counts revisions in which ``M`` was among
the changed modules, and ``alpha`` is a Laplace pseudo-count (default 0.5,
the Jeffreys prior for a Bernoulli rate). Units: dimensionless probability.
Validity range: any history with at least one revision; the estimate is only
informative for modules that appear in the training revisions, and for a
module never seen the score falls back to the test's own base failure rate.

The score of a test for a given change is the maximum over the changed
modules, which treats the change as "any one of these modules may have broken
something" rather than assuming independence between them. Independence would
let a change to five unrelated modules multiply five small probabilities into
a smaller one, which is the wrong direction for a safety-relevant ranking.

This baseline is a *prioritiser*: it returns a total order over the whole
suite. Converted to a selection by truncating the order at a budget, it is
not safe and the measured safety loss is reported in
``validation/validate_benchmark_output.txt``.

Reference for prioritisation and for the APFD measure applied to it:
G. Rothermel, R. H. Untch, C. Chu and M. J. Harrold, "Prioritizing Test Cases
For Regression Testing", IEEE Transactions on Software Engineering, 27(10),
929-948, October 2001.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .history import History

__all__ = ["CoFailurePrioritiser", "RankedTest"]


@dataclass(frozen=True)
class RankedTest:
    """One test node with its score, in rank order.

    ``score`` is dimensionless. For :class:`CoFailurePrioritiser` it is a
    smoothed conditional failure probability; for the learned selector it is a
    model probability. The two are not on the same calibration scale and are
    never compared numerically, only by the order they induce.
    """

    node_id: str
    score: float


@dataclass
class CoFailurePrioritiser:
    """Smoothed conditional-failure-rate prioritiser over a commit history."""

    alpha: float = 0.5
    _test_ids: tuple[str, ...] = field(default_factory=tuple, repr=False)
    _module_revisions: dict[str, int] = field(default_factory=dict, repr=False)
    _cofail: dict[tuple[str, str], int] = field(default_factory=dict, repr=False)
    _base_fail: dict[str, int] = field(default_factory=dict, repr=False)
    _n_revisions: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive; 0 gives undefined rates for unseen modules")

    def fit(self, history: History) -> CoFailurePrioritiser:
        """Accumulate co-failure counts from ``history``. Returns ``self``."""
        if len(history) == 0:
            raise ValueError("cannot fit a prioritiser on an empty history")
        self._test_ids = tuple(history.test_ids)
        self._module_revisions = {}
        self._cofail = {}
        self._base_fail = dict.fromkeys(self._test_ids, 0)
        self._n_revisions = len(history)
        for rev in history.revisions:
            failed = set(rev.failed_tests)
            for t in failed:
                if t in self._base_fail:
                    self._base_fail[t] += 1
            for mod in rev.changed_modules:
                self._module_revisions[mod] = self._module_revisions.get(mod, 0) + 1
                for t in failed:
                    key = (mod, t)
                    self._cofail[key] = self._cofail.get(key, 0) + 1
        return self

    def _base_rate(self, test: str) -> float:
        a = self._base_fail.get(test, 0)
        return (a + self.alpha) / (self._n_revisions + 2.0 * self.alpha)

    def conditional_rate(self, module: str, test: str) -> float:
        """Smoothed P(test fails | module changed), dimensionless."""
        b = self._module_revisions.get(module)
        if not b:
            return self._base_rate(test)
        a = self._cofail.get((module, test), 0)
        return (a + self.alpha) / (b + 2.0 * self.alpha)

    def score(self, test: str, changed_modules: Iterable[str]) -> float:
        """Score of one test for one change (dimensionless probability)."""
        mods = list(changed_modules)
        if not mods:
            return self._base_rate(test)
        return max(self.conditional_rate(m, test) for m in mods)

    def rank(
        self, changed_modules: Iterable[str], tests: Sequence[str] | None = None
    ) -> list[RankedTest]:
        """Total order over ``tests``, highest score first.

        Ties break on the node id, ascending, so the order is deterministic
        and independent of dict iteration order.
        """
        if self._n_revisions == 0:
            raise ValueError("prioritiser is not fitted; call fit(history) first")
        universe = tuple(tests) if tests is not None else self._test_ids
        mods = list(changed_modules)
        scored = [(self.score(t, mods), t) for t in universe]
        scored.sort(key=lambda st: (-st[0], st[1]))
        return [RankedTest(node_id=t, score=s) for s, t in scored]
