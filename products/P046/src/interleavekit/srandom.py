"""S-random (spread) interleaver.

Construction
------------
The S-random interleaver is built by a greedy randomised search.  Source indices
are assigned to transmitted positions ``i = 0, 1, ..., N-1`` in order.  For each
position a candidate is drawn uniformly from the source indices not yet used and
accepted if, for every already-assigned position ``j`` with ``i - S < j < i``,

    |pi[i] - pi[j]| >= S

that is, no two source symbols within ``S`` transmitted positions of each other
are within ``S`` of each other in the source.  The search enumerates the legal
candidates at each position, draws one uniformly, and **backtracks** when a
position has none left: the previous position's value is withdrawn and another
drawn there.  A whole attempt is restarted at most ``max_attempts`` times.

Forward-only greedy placement, which is how the construction is usually
described, stalls on the last few positions: measured in this build at length
1024 and spread 16 it reached position 1023 of 1024 and gave up.  Backtracking
costs little and removes that failure mode; ``validation/validate_srandom.py``
reports the spreads actually reached.

The resulting permutation satisfies ``s_parameter(pi) >= spread`` by
construction, which :func:`interleavekit.metrics.s_parameter` verifies
independently and ``tests/test_srandom.py`` asserts.

Feasibility
-----------
The search is not guaranteed to succeed.  Requesting a large ``spread`` for a
short ``length`` leads to repeated dead ends.  The rule of thumb in the
turbo-code literature is that ``spread`` up to roughly ``sqrt(length / 2)`` is
reliably achievable; beyond that the greedy search stalls.  This package does not
assert that bound -- it measures it.
``validation/validate_srandom.py`` reports, for a range of lengths, the largest
spread this implementation actually reached, next to ``sqrt(length / 2)``, so the
reader can see where the rule of thumb holds and where it does not.

On failure the constructor raises ``ValueError`` naming the length, the requested
spread, how far the best attempt got, and the ``sqrt(length / 2)`` figure, so the
caller can pick a workable spread without guessing.

Determinism
-----------
The permutation is a deterministic function of ``(length, spread, seed)``.  The
generator is ``numpy.random.default_rng(seed)``.  Rebuilding with the same three
values gives the identical permutation; ``tests/test_srandom.py`` asserts this.

References
----------
Semirandom permutations for turbo codes are discussed in Dolinar, S. and
D. Divsalar, "Weight Distributions for Turbo Codes Using Random and Nonrandom
Permutations", *Interplanetary Network Progress Report* 42-122, 15 August 1995,
pp. 56-65 (article read on 2026-10-06).  **That article does not state the
S-random selection rule** -- it compares random, nonrandom and semirandom
permutations -- so it is cited here as context for semirandom permutations and
not as the source of the rule.  The rule implemented above is written out in full
in this docstring rather than attributed, because no primary source for it was
verified during this build.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .base import Interleaver, InterleaverCost, check_length

__all__ = ["SRandomInterleaver"]


class SRandomInterleaver(Interleaver):
    """Greedy randomised interleaver with a guaranteed S-parameter.

    Parameters
    ----------
    length:
        Block length ``N``, in symbols.  Must be >= 1.
    spread:
        Requested spread ``S``, in symbols.  Must be >= 1.  ``spread = 1`` imposes
        no constraint and yields a uniform random permutation.
    seed:
        Seed for ``numpy.random.default_rng``.  The permutation is a
        deterministic function of ``(length, spread, seed)``.
    max_attempts:
        Number of complete restarts allowed before giving up.  Must be >= 1.
    max_draws_per_position:
        Retained for API stability and validated, but the backtracking search in
        :meth:`_build` does not draw repeatedly at one position -- it enumerates
        the legal candidates there and backtracks when there are none.  Must
        be >= 1.

    Raises
    ------
    TypeError
        If a parameter is not an integer.
    ValueError
        If ``length < 1``, ``spread < 1``, ``spread > length``,
        ``max_attempts < 1``, ``max_draws_per_position < 1``, or if the search
        fails within the node budget.  The failure message names the length, the
        spread, how far the best attempt got and the ``sqrt(length/2)``
        practical ceiling.

    Examples
    --------
    >>> from interleavekit.metrics import s_parameter
    >>> il = SRandomInterleaver(length=256, spread=8, seed=0)
    >>> s_parameter(il.permutation()) >= 8
    True
    >>> a = SRandomInterleaver(length=64, spread=4, seed=7).permutation()
    >>> b = SRandomInterleaver(length=64, spread=4, seed=7).permutation()
    >>> bool((a == b).all())
    True
    """

    def __init__(
        self,
        length: int,
        spread: int,
        seed: int = 0,
        max_attempts: int = 20,
        max_draws_per_position: int = 400,
    ) -> None:
        self._length = check_length(length, "length")
        self._spread = check_length(spread, "spread")
        if self._spread > self._length:
            raise ValueError(
                f"spread {self._spread} cannot exceed length {self._length}; the constraint "
                "would have to hold between a symbol and itself"
            )
        self._max_attempts = check_length(max_attempts, "max_attempts")
        self._max_draws = check_length(max_draws_per_position, "max_draws_per_position")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
        self._seed = int(seed)
        self._attempts_used = 0
        self._pi = self._build()

    @property
    def length(self) -> int:
        """Block length ``N``, in symbols."""
        return self._length

    @property
    def spread(self) -> int:
        """Requested and achieved spread ``S``, in symbols."""
        return self._spread

    @property
    def seed(self) -> int:
        """Seed used for the search."""
        return self._seed

    @property
    def attempts_used(self) -> int:
        """Number of restarts consumed before the search succeeded."""
        return self._attempts_used

    def permutation(self) -> NDArray[np.int64]:
        """The permutation found by the search."""
        return self._pi.copy()

    def cost(self) -> InterleaverCost:
        """Full-block buffering cost, in symbols.

        Returns
        -------
        InterleaverCost
            One-way latency and memory ``length``; pair latency and memory
            ``2 * length``.

        Notes
        -----
        An S-random permutation has no algebraic form, so both ends must also
        store the permutation itself: ``length`` index words on top of the
        ``length`` symbol buffer.  That index table is **not** counted in
        :attr:`InterleaverCost.one_way_memory_symbols`, which counts symbols; the
        README says so under Limitations, and it is the one cost where the block
        and helical constructions are strictly cheaper, since their permutations
        are computed from two integers.
        """
        n = self._length
        return InterleaverCost(
            one_way_latency_symbols=n,
            pair_latency_symbols=2 * n,
            one_way_memory_symbols=n,
            pair_memory_symbols=2 * n,
            model="full-block buffering (plus a stored index table)",
        )

    def _build(self) -> NDArray[np.int64]:
        """Randomised backjumping search for a permutation meeting the condition.

        Positions are filled in order.  At each position the legal candidates are
        those not yet used, not already tried at this position, and at least
        ``spread`` away in source index from every source index already placed
        within ``spread`` positions; one is drawn uniformly from them.

        When a position has no legal candidate the search does not give up and does
        not step back one position: it **backjumps** ``min(spread, 3)`` positions,
        clears what was tried in between and records the value at the jump target
        as tried there.  Dead ends in this problem are created by the whole recent
        window, not by the single last placement, so single-step backtracking
        re-enters the same dead end.  Measured on this implementation at length 256
        and spread 10, single-step backtracking failed inside the node budget while
        a three-position backjump succeeded.

        An attempt that exhausts its node budget of ``20 * length`` nodes restarts
        from a fresh seed, up to ``max_attempts`` times.  The search is therefore
        bounded, not exhaustive: it is not guaranteed to find the largest feasible
        spread, and ``validation/srandom_output.txt`` reports the spreads it
        actually reaches against the ``sqrt(length / 2)`` rule of thumb.
        """
        n, s = self._length, self._spread
        node_budget = max(256, 20 * n)
        backjump = max(1, min(s, 3))
        best_reached = 0
        for attempt in range(1, self._max_attempts + 1):
            # SeedSequence over (seed, attempt) keeps attempt 2 of seed 11 distinct
            # from attempt 1 of seed 12; a plain seed + attempt would collide.
            rng = np.random.default_rng([self._seed, attempt])
            pi = np.full(n, -1, dtype=np.int64)
            used = np.zeros(n, dtype=bool)
            tried = np.zeros((n, n), dtype=bool)
            i = 0
            nodes = 0
            while 0 <= i < n and nodes < node_budget:
                nodes += 1
                free = np.flatnonzero(~used & ~tried[i])
                if free.size and s > 1 and i > 0:
                    window = pi[max(0, i - s + 1) : i]
                    if window.size:
                        keep = (np.abs(free[:, None] - window[None, :]) >= s).all(axis=1)
                        free = free[keep]
                if free.size:
                    pick = int(free[rng.integers(free.size)])
                    pi[i] = pick
                    used[pick] = True
                    i += 1
                    continue
                if i > best_reached:
                    best_reached = i
                target = i - min(backjump, i)
                if target < 0:
                    break
                tried[i, :] = False
                for k in range(i - 1, target - 1, -1):
                    value = int(pi[k])
                    if value >= 0:
                        used[value] = False
                        pi[k] = -1
                    if k == target:
                        tried[k, :] = False
                        if value >= 0:
                            tried[k, value] = True
                    else:
                        tried[k, :] = False
                i = target
            if i == n:
                self._attempts_used = attempt
                return pi
        raise ValueError(
            f"S-random search failed for length={n}, spread={s}: {self._max_attempts} "
            f"backjumping attempts of up to {node_budget} nodes each got no further than "
            f"position {best_reached} of {n}. Reduce spread -- the spreads this "
            f"implementation actually reaches are tabulated for a range of lengths in "
            f"validation/srandom_output.txt, and sqrt(length/2) = {np.sqrt(n / 2):.1f} is the "
            f"usual rule-of-thumb ceiling -- or raise max_attempts."
        )

    def __repr__(self) -> str:
        return (
            f"SRandomInterleaver(length={self._length}, spread={self._spread}, "
            f"seed={self._seed})"
        )
