"""Campaign definition, execution, replay and case pools.

A :class:`FaultCase` is the unit of a campaign: one injection, one seed, one
run length.  It serialises to JSON and its ``case_id`` is a content hash, so a
case found interesting in one session can be replayed byte-for-byte in another
(``faultinject replay``).

A campaign here injects **one fault per case**.  Multiple simultaneous faults
are representable (``InjectionWrapper`` accepts a list) but the coverage metric,
the case pool and the prioritiser all assume a single fault, and that
restriction is stated in the README limitations rather than worked around.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .coverage import CoverageCell, CoverageTracker, cell_of
from .faults import Injection
from .harness import Trace, nominal_trace, run_case
from .severity import SeverityReport, score
from .target import N_STEPS
from .taxonomy import (
    DURATION_FRAC,
    START_FRAC,
    FaultKind,
    Scale,
    kinds,
    spec,
)


def _canonical(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FaultCase:
    """One reproducible injected case."""

    injection: Injection
    seed: int
    n_steps: int = N_STEPS

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}")

    @property
    def case_id(self) -> str:
        """16 hex characters of SHA-256 over the canonical case description.

        The hash covers the fault kind, channel, parameters (as ``repr`` of the
        binary64 value, so no precision is lost), window, seed and run length --
        everything ``run_case`` depends on.  Two cases with the same id produce
        the same trace.
        """
        payload = {
            "kind": self.injection.kind.value,
            "channel": self.injection.channel,
            "params": {k: repr(v) for k, v in sorted(self.injection.params.items())},
            "start_step": self.injection.start_step,
            "duration_steps": self.injection.duration_steps,
            "seed": self.seed,
            "n_steps": self.n_steps,
        }
        return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:16]

    def cell(self) -> CoverageCell:
        """Coverage cell this case lands in."""
        return cell_of(self.injection, self.n_steps)

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form, including the content hash."""
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "n_steps": self.n_steps,
            "injection": self.injection.to_dict(),
        }

    def to_json(self) -> str:
        """Canonical JSON string."""
        return _canonical(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> FaultCase:
        """Inverse of :meth:`to_dict`.

        If the mapping carries a ``case_id`` it is verified against the
        recomputed hash and a mismatch raises ``ValueError`` -- a silently
        edited case file is a reproducibility defect, not a convenience.
        """
        case = cls(
            injection=Injection.from_dict(d["injection"]),  # type: ignore[arg-type]
            seed=int(d["seed"]),  # type: ignore[arg-type]
            n_steps=int(d.get("n_steps", N_STEPS)),  # type: ignore[arg-type]
        )
        claimed = d.get("case_id")
        if claimed is not None and str(claimed) != case.case_id:
            raise ValueError(
                f"case_id mismatch: file says {claimed}, content hashes to {case.case_id}"
            )
        return case

    @classmethod
    def from_json(cls, text: str) -> FaultCase:
        """Parse a canonical JSON case."""
        return cls.from_dict(json.loads(text))


@dataclass
class CaseResult:
    """Outcome of executing one case."""

    case: FaultCase
    severity: SeverityReport
    monitor: dict[str, object]
    event_counts: dict[str, int]
    aborted_step: int | None
    updates: int
    skipped: int
    trace: Trace | None = field(default=None, repr=False)

    @property
    def severe(self) -> bool:
        return self.severity.severe

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable summary (the trace itself is not included)."""
        return {
            "case": self.case.to_dict(),
            "severity": self.severity.to_dict(),
            "monitor": self.monitor,
            "event_counts": self.event_counts,
            "aborted_step": self.aborted_step,
            "updates": self.updates,
            "skipped": self.skipped,
        }


def execute_case(case: FaultCase, keep_trace: bool = False, target=None) -> CaseResult:
    """Run one case, score it against the nominal run of the same seed.

    Parameters
    ----------
    case
        The case to run.
    keep_trace
        Keep the full faulted trace on the result (memory: ~6 floats per step).
    target
        Optional target instance; a fresh ``GncController`` is used otherwise.
        The nominal reference trace always uses the default controller, so pass
        a non-default target only when comparing targets deliberately.
    """
    faulted = run_case([case.injection], case.seed, case.n_steps, target=target)
    nominal = nominal_trace(case.seed, case.n_steps)
    rep = score(faulted, nominal)
    counts: dict[str, int] = {}
    for e in faulted.events:
        counts[e.event] = counts.get(e.event, 0) + 1
    return CaseResult(
        case=case,
        severity=rep,
        monitor=dict(faulted.monitor),
        event_counts=dict(sorted(counts.items())),
        aborted_step=faulted.aborted_step,
        updates=faulted.updates,
        skipped=faulted.skipped,
        trace=faulted if keep_trace else None,
    )


def replay_case(case: FaultCase) -> tuple[Trace, Trace]:
    """Run ``case`` twice and return both traces, for bit-identity checking."""
    a = run_case([case.injection], case.seed, case.n_steps)
    b = run_case([case.injection], case.seed, case.n_steps)
    return a, b


def _admissible_integers(pspec, bin_index: int) -> tuple[int, ...]:
    """Integers inside ``pspec`` that bin to ``bin_index``."""
    lo = math.ceil(pspec.lo - 1e-12)
    hi = math.floor(pspec.hi + 1e-12)
    return tuple(v for v in range(lo, hi + 1) if pspec.bin_of(float(v)) == bin_index)


def _admissible_steps(pspec, bin_index: int, n_steps: int, lowest: int) -> tuple[int, ...]:
    """Step counts whose fraction of ``n_steps`` bins to ``bin_index``."""
    return tuple(
        k
        for k in range(lowest, n_steps + 1)
        if pspec.bin_of(k / n_steps) == bin_index
    )


def _draw_continuous(pspec, bin_index: int, rng: np.random.Generator | None) -> float:
    """A value strictly inside bin ``bin_index`` of a continuous parameter."""
    edges = pspec.edges()
    lo, hi = edges[bin_index], edges[bin_index + 1]
    if rng is None:
        return pspec.representative(bin_index)
    u = 0.02 + 0.96 * float(rng.random())
    if pspec.scale is Scale.LOG:
        value = 10.0 ** (math.log10(lo) + (math.log10(hi) - math.log10(lo)) * u)
    else:
        value = lo + (hi - lo) * u
    return min(max(value, lo), hi)


def _pick(options: Sequence[int], rng: np.random.Generator | None, cell_label: str) -> int:
    if not options:
        raise ValueError(f"cell {cell_label} contains no admissible integer value")
    if rng is None:
        return options[len(options) // 2]
    return int(options[int(rng.integers(0, len(options)))])


def case_in_cell(
    cell: CoverageCell,
    seed: int,
    n_steps: int = N_STEPS,
    rng: np.random.Generator | None = None,
) -> FaultCase:
    """Build a case that lands in ``cell``.

    Integer-valued quantities (the start step, the duration and any integer
    parameter such as ``delay_steps``) are drawn from the set of integers that
    actually bin to the requested bin, rather than from the continuous bin and
    then rounded -- rounding after the fact moves cases into neighbouring cells
    and would corrupt the coverage accounting.  Continuous parameters are drawn
    log-uniformly or uniformly inside the bin according to the parameter scale.

    The resulting case is checked against :func:`~faultinject.coverage.cell_of`
    before being returned, so a binning bug cannot produce a case that claims a
    cell it does not occupy.
    """
    sp = spec(cell.kind)
    lbl = cell.label()
    start_step = _pick(
        _admissible_steps(START_FRAC, cell.start_bin, n_steps - 1, 0), rng, lbl
    )
    duration = _pick(
        _admissible_steps(DURATION_FRAC, cell.duration_bin, n_steps, 1), rng, lbl
    )
    params: dict[str, float] = {}
    for pspec, bin_index in zip(sp.params, cell.param_bins, strict=True):
        if pspec.integer:
            params[pspec.name] = float(_pick(_admissible_integers(pspec, bin_index), rng, lbl))
        else:
            params[pspec.name] = _draw_continuous(pspec, bin_index, rng)

    inj = Injection.create(cell.kind, cell.channel, params, start_step, duration)
    case = FaultCase(inj, int(seed), int(n_steps))
    got = case.cell()
    if got != cell:
        raise AssertionError(
            f"case_in_cell produced {got.label()} for requested {lbl}"
        )
    return case


def build_pool(
    pool_seed: int,
    replicates: int = 2,
    subset: Sequence[FaultKind] | None = None,
    n_steps: int = N_STEPS,
) -> tuple[FaultCase, ...]:
    """Deterministic case pool covering every cell ``replicates`` times.

    The pool is the search space for the campaign benchmark.  It is finite and
    enumerable *on purpose*: the ground-truth severity of every case can then be
    computed once, so three search strategies can be compared on the same
    problem instance without re-running the target.  A real campaign faces a
    space too large for that; see the README limitations.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")
    tracker = CoverageTracker(tuple(subset) if subset is not None else kinds())
    rng = np.random.default_rng([int(pool_seed), 9])
    cases: list[FaultCase] = []
    for rep in range(replicates):
        for cell in tracker.cells():
            seed = int(rng.integers(0, 2**31 - 1))
            cases.append(case_in_cell(cell, seed, n_steps, rng))
        del rep
    return tuple(cases)


def evaluate_pool(pool: Sequence[FaultCase]) -> tuple[float, ...]:
    """Severity of every case in ``pool``, in order.

    Cost: one faulted run plus (cached) one nominal run per case. On the 1-core
    build container a 496-case pool takes about 1 s.
    """
    return tuple(execute_case(c).severity.severity for c in pool)


@dataclass
class CampaignResult:
    """Outcome of running a list of cases in order."""

    results: list[CaseResult]
    tracker: CoverageTracker

    @property
    def severities(self) -> list[float]:
        return [r.severity.severity for r in self.results]

    @property
    def n_severe(self) -> int:
        return sum(1 for r in self.results if r.severe)

    def coverage_curve(self) -> list[float]:
        """Coverage fraction after each executed case."""
        tracker = CoverageTracker(self.tracker.subset)
        out = []
        for r in self.results:
            tracker.add(r.case.injection, r.case.n_steps)
            out.append(tracker.fraction)
        return out


def run_campaign(
    cases: Sequence[FaultCase],
    subset: Sequence[FaultKind] | None = None,
    keep_traces: bool = False,
) -> CampaignResult:
    """Execute ``cases`` in order and accumulate coverage."""
    tracker = CoverageTracker(tuple(subset) if subset is not None else kinds())
    results = []
    for c in cases:
        results.append(execute_case(c, keep_trace=keep_traces))
        tracker.add(c.injection, c.n_steps)
    return CampaignResult(results=results, tracker=tracker)
