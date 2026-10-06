"""Spread, dispersion and burst-dispersion metrics for interleavers.

All metrics are exact and deterministic; none of them samples.

Definitions used here
---------------------
Let ``pi`` be a permutation of ``0 .. N-1`` with ``y[i] = x[pi[i]]``.

**Spread of a pair.**  ``spread(i, j) = |i - j| + |pi[i] - pi[j]|``, the L1
distance between the points ``(i, pi[i])`` and ``(j, pi[j])``.

**Minimum spread.**  ``min over i != j of spread(i, j)``.  A large minimum spread
means no two symbols that were close in the source are close after interleaving
*and* vice versa.

**S-parameter.**  The largest ``S`` such that ``|i - j| < S`` implies
``|pi[i] - pi[j]| >= S``.  This is the quantity the S-random construction is
built to achieve.  It is bounded by the minimum spread:
``minimum_spread >= s_parameter + 1`` always, proved in
:func:`s_parameter` and property-tested in ``tests/test_properties.py``.

**Dispersion.**  The number of distinct displacement pairs
``(j - i, pi[j] - pi[i])`` over all ``i < j``, optionally divided by the maximum
possible ``N(N-1)/2``.  A permutation whose displacement pairs are all distinct
has normalised dispersion 1; a permutation with much structure repeats
displacements and scores lower.

**Burst dispersion.**  This is the metric a fading-link designer actually needs.
A channel burst is contiguous in the *transmitted* stream.  Given a burst of
``burst_length`` consecutive transmitted positions, the errored source symbols
are those whose transmitted position falls inside the window.  The figure of
merit is the **longest run of consecutive source indices** among them, maximised
over every window position -- that is, the worst case.  A result of 1 means every
errored source symbol is isolated, which is the condition under which a
symbol-level code correcting one symbol per codeword can clean the burst up.
:func:`burst_dispersion` returns that worst-case run length and
:func:`max_burst_fully_dispersed` returns the largest burst length for which it
is still 1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "as_permutation",
    "is_bijection",
    "minimum_spread",
    "s_parameter",
    "dispersion",
    "burst_dispersion",
    "burst_dispersion_by_window_scan",
    "transmitted_span_profile",
    "burst_dispersion_profile",
    "max_burst_fully_dispersed",
    "longest_consecutive_run",
]

#: Pair-count ceiling for :func:`dispersion`, which is inherently O(N^2).
#: 2048 symbols is 2 096 128 pairs, about 34 MB of int64 displacement data,
#: which fits the 2-core / 7.8 GiB build budget with room to spare.
MAX_DISPERSION_LENGTH = 2048


def as_permutation(pi: NDArray | list[int], name: str = "pi") -> NDArray[np.int64]:
    """Validate and normalise a permutation array.

    Parameters
    ----------
    pi:
        Candidate permutation of ``0 .. N-1``.
    name:
        Parameter name used in error messages.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)``, dtype ``int64``.

    Raises
    ------
    ValueError
        If ``pi`` is not one-dimensional, is empty, or is not a bijection on
        ``0 .. N-1``.  The message says which of the three failed.
    """
    arr = np.asarray(pi)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got {arr.ndim} dimensions")
    if arr.size == 0:
        raise ValueError(f"{name} must have at least 1 element, got an empty array")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype, got {arr.dtype}")
    arr = arr.astype(np.int64, copy=False)
    if not is_bijection(arr):
        raise ValueError(
            f"{name} is not a permutation of 0..{arr.size - 1}: "
            f"it has {np.unique(arr).size} distinct values in "
            f"[{int(arr.min())}, {int(arr.max())}]"
        )
    return arr


def is_bijection(pi: NDArray | list[int]) -> bool:
    """Whether ``pi`` is a bijection on ``0 .. len(pi)-1``.

    Parameters
    ----------
    pi:
        Candidate index array.

    Returns
    -------
    bool
        ``True`` if every index in ``0 .. len(pi)-1`` appears exactly once.
    """
    arr = np.asarray(pi)
    if arr.ndim != 1 or arr.size == 0:
        return False
    if not np.issubdtype(arr.dtype, np.integer):
        return False
    seen = np.zeros(arr.size, dtype=bool)
    in_range = (arr >= 0) & (arr < arr.size)
    if not bool(in_range.all()):
        return False
    seen[arr] = True
    return bool(seen.all())


def minimum_spread(pi: NDArray | list[int]) -> int:
    """Minimum over distinct index pairs of ``|i-j| + |pi[i]-pi[j]|``.

    Parameters
    ----------
    pi:
        Permutation of ``0 .. N-1``.

    Returns
    -------
    int
        The minimum spread, in index units.  ``0`` for ``N == 1``, where no pair
        of distinct positions exists.

    Raises
    ------
    ValueError
        If ``pi`` is not a permutation.

    Notes
    -----
    Exact, and ``O(N * S)`` rather than ``O(N^2)``: any pair separated by
    ``d >= best`` has spread at least ``d + 1 > best``, so the scan over
    separations stops as soon as ``d`` reaches the best spread found.  Since the
    minimum spread of a permutation of length ``N`` is itself ``O(sqrt(N))`` in
    practice, the loop is short.
    """
    arr = as_permutation(pi)
    n = arr.size
    if n < 2:
        return 0
    best = np.iinfo(np.int64).max
    for d in range(1, n):
        if d >= best:
            break
        gap = int(np.abs(arr[d:] - arr[:-d]).min())
        best = min(best, d + gap)
    return int(best)


def s_parameter(pi: NDArray | list[int]) -> int:
    """Largest ``S`` with ``|i-j| < S`` implying ``|pi[i]-pi[j]| >= S``.

    Parameters
    ----------
    pi:
        Permutation of ``0 .. N-1``.

    Returns
    -------
    int
        The S-parameter.  ``1`` is always attainable (``|i-j| < 1`` is vacuous),
        so the result is at least 1 for ``N >= 1``.

    Raises
    ------
    ValueError
        If ``pi`` is not a permutation.

    Notes
    -----
    The bound ``minimum_spread(pi) >= s_parameter(pi) + 1`` holds for every
    permutation.  Proof: let ``S = s_parameter(pi)`` and take any ``i != j``.  If
    ``|i-j| >= S`` then the spread is at least ``S + 1`` because
    ``|pi[i]-pi[j]| >= 1``.  If ``|i-j| < S`` then ``|pi[i]-pi[j]| >= S`` by the
    definition of ``S``, so the spread is at least ``S + 1`` again.
    """
    arr = as_permutation(pi)
    n = arr.size
    s = 1
    while s < n:
        candidate = s + 1
        ok = True
        for d in range(1, candidate):
            if d >= n:
                break
            if int(np.abs(arr[d:] - arr[:-d]).min()) < candidate:
                ok = False
                break
        if not ok:
            break
        s = candidate
    return int(s)


def dispersion(
    pi: NDArray | list[int], normalise: bool = True
) -> float:
    """Count of distinct displacement pairs, optionally normalised.

    Parameters
    ----------
    pi:
        Permutation of ``0 .. N-1``.
    normalise:
        If ``True`` (default) divide by ``N(N-1)/2``, giving a value in
        ``(0, 1]``.  If ``False`` return the raw count as a float.

    Returns
    -------
    float
        Normalised dispersion in ``(0, 1]``, or the raw distinct-pair count.
        ``0.0`` for ``N == 1``.

    Raises
    ------
    ValueError
        If ``pi`` is not a permutation, or if ``N > 2048``.  The metric is
        inherently ``O(N^2)`` and the ceiling keeps it inside the documented
        compute budget; the error message names the ceiling.
    """
    arr = as_permutation(pi)
    n = arr.size
    if n < 2:
        return 0.0
    if n > MAX_DISPERSION_LENGTH:
        raise ValueError(
            f"dispersion is O(N^2) and is capped at N = {MAX_DISPERSION_LENGTH}; got N = {n}. "
            "Measure dispersion on a representative shorter interleaver, or use "
            "minimum_spread, which is O(N*sqrt(N))."
        )
    i, j = np.triu_indices(n, k=1)
    di = (j - i).astype(np.int64)
    dp = (arr[j] - arr[i]).astype(np.int64)
    # Pack the two displacements into one collision-free int64 key: dp + n lies in
    # [1, 2n-1] < 2n+1, so di * (2n+1) + (dp + n) is injective on (di, dp).
    key = di * (2 * n + 1) + (dp + n)
    count = int(np.unique(key).size)
    if not normalise:
        return float(count)
    return count / (n * (n - 1) / 2)


def longest_consecutive_run(indices: NDArray | list[int]) -> int:
    """Longest run of consecutive integers in a set of indices.

    Parameters
    ----------
    indices:
        Integer indices, in any order, possibly with repeats.

    Returns
    -------
    int
        Length of the longest run of consecutive integers present.  ``0`` for an
        empty input.

    Examples
    --------
    >>> longest_consecutive_run([5, 1, 2, 9, 3])
    3
    >>> longest_consecutive_run([])
    0
    """
    arr = np.asarray(indices, dtype=np.int64).ravel()
    if arr.size == 0:
        return 0
    u = np.unique(arr)
    if u.size == 1:
        return 1
    breaks = np.flatnonzero(np.diff(u) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [u.size - 1]])
    return int((ends - starts + 1).max())


def _runs_in_window(
    pos: NDArray[np.int64], lo: int, hi: int
) -> list[NDArray[np.int64]]:
    """Maximal runs of consecutive source indices whose positions lie in ``[lo, hi)``.

    Returns
    -------
    list of numpy.ndarray
        Each element holds the transmitted positions of one maximal run of
        consecutive source indices, in source order.
    """
    inside = np.flatnonzero((pos >= lo) & (pos < hi))
    if inside.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(inside) != 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [inside.size - 1]])
    return [pos[inside[a : b + 1]] for a, b in zip(starts, ends, strict=True)]


def transmitted_span_profile(
    position_of_input: NDArray | list[int],
    max_run: int,
    window_range: tuple[int, int] | None = None,
) -> NDArray[np.int64]:
    """Narrowest transmitted window that can hold ``R`` consecutive source symbols.

    For each run length ``R``, this is

        ``m(R) = min over j of ( max(pos[j..j+R-1]) - min(pos[j..j+R-1]) )``

    over every run of ``R`` consecutive source indices that lies wholly inside
    ``window_range``.  ``m`` is non-decreasing in ``R``, because a run of ``R+1``
    contains a run of ``R``.

    This is the quantity every burst metric in this module reduces to: a burst of
    ``L`` transmitted symbols can damage ``R`` consecutive source symbols exactly
    when ``m(R) <= L - 1``.

    Parameters
    ----------
    position_of_input:
        Element ``j`` is the transmitted position of source symbol ``j``.
    max_run:
        Largest run length to profile, >= 1.
    window_range:
        Half-open range of transmitted positions to restrict to; defaults to the
        full extent of ``position_of_input``.

    Returns
    -------
    numpy.ndarray
        Shape ``(max_run,)``, dtype ``int64``.  Element ``R-1`` is ``m(R)``.
        ``m(1) = 0`` always.  Where no run of length ``R`` exists inside the
        window, the entry and every later one are
        ``numpy.iinfo(numpy.int64).max``.

    Raises
    ------
    ValueError
        If ``max_run < 1`` or ``position_of_input`` is empty.
    """
    pos = np.asarray(position_of_input, dtype=np.int64).ravel()
    if pos.size == 0:
        raise ValueError("position_of_input must have at least 1 element, got an empty array")
    max_run = int(max_run)
    if max_run < 1:
        raise ValueError(f"max_run must be >= 1, got {max_run}")
    if window_range is None:
        lo, hi = int(pos.min()), int(pos.max()) + 1
    else:
        lo, hi = int(window_range[0]), int(window_range[1])

    sentinel = np.iinfo(np.int64).max
    profile = np.full(max_run, sentinel, dtype=np.int64)
    groups = _runs_in_window(pos, lo, hi)
    if not groups:
        return profile
    profile[0] = 0
    for group in groups:
        run_max = group.copy()
        run_min = group.copy()
        for r in range(2, max_run + 1):
            if run_max.size < 2:
                break
            run_max = np.maximum(run_max[:-1], group[r - 1 :])
            run_min = np.minimum(run_min[:-1], group[r - 1 :])
            width = int((run_max - run_min).min())
            if width < profile[r - 1]:
                profile[r - 1] = width
    return profile


def _window_bounds(
    pos: NDArray[np.int64], window_range: tuple[int, int] | None
) -> tuple[int, int]:
    if window_range is None:
        return int(pos.min()), int(pos.max()) + 1
    return int(window_range[0]), int(window_range[1])


def _validated_positions(position_of_input: NDArray | list[int]) -> NDArray[np.int64]:
    pos = np.asarray(position_of_input, dtype=np.int64).ravel()
    if pos.size == 0:
        raise ValueError("position_of_input must have at least 1 element, got an empty array")
    if np.unique(pos).size != pos.size:
        raise ValueError(
            "position_of_input has repeated transmitted positions, so it does not describe "
            "an interleaver: two source symbols cannot occupy the same transmitted slot"
        )
    return pos


def _check_burst_length(burst_length: int) -> int:
    if isinstance(burst_length, bool) or not isinstance(burst_length, (int, np.integer)):
        raise TypeError(f"burst_length must be an integer, got {type(burst_length).__name__}")
    burst_length = int(burst_length)
    if burst_length < 1:
        raise ValueError(f"burst_length must be >= 1 symbol, got {burst_length}")
    return burst_length


def burst_dispersion(
    position_of_input: NDArray | list[int],
    burst_length: int,
    window_range: tuple[int, int] | None = None,
) -> int:
    """Worst-case surviving run length after de-interleaving a channel burst.

    A channel burst occupies ``burst_length`` consecutive transmitted positions.
    The source symbols it damages are those whose transmitted position falls in
    the window.  This function returns the longest run of *consecutive source
    indices* among them, maximised over every window position -- the worst case a
    designer has to survive.

    Parameters
    ----------
    position_of_input:
        Element ``j`` is the transmitted position of source symbol ``j``.  For a
        permutation interleaver this is
        :meth:`interleavekit.base.Interleaver.position_of_input`; for the
        convolutional interleaver it is
        :meth:`interleavekit.convolutional.ConvolutionalInterleaver.transmitted_position`
        evaluated on ``0 .. n-1``.
    burst_length:
        Burst length in transmitted symbols.  Must be >= 1.
    window_range:
        Half-open range ``(lo, hi)`` of transmitted positions the burst may
        occupy, as ``lo <= t`` and ``t + burst_length <= hi``.  Defaults to the
        full extent of ``position_of_input``.  Pass the convolutional
        interleaver's
        :meth:`~interleavekit.convolutional.ConvolutionalInterleaver.steady_state_range`
        here, so that bursts are not measured across start-up fill symbols.

    Returns
    -------
    int
        Worst-case longest surviving run of consecutive source indices, in
        symbols.  ``1`` means the burst is fully dispersed: every errored source
        symbol is isolated.  ``0`` only if the window contains no source symbol
        at all.

    Raises
    ------
    ValueError
        If ``burst_length < 1``, if ``position_of_input`` is empty or has
        repeated positions, or if ``window_range`` admits no window of the
        requested length.  The last message names the widest burst the range can
        hold.

    Notes
    -----
    Computed from :func:`transmitted_span_profile` rather than by scanning
    windows: ``R`` consecutive source symbols can be damaged by a burst of ``L``
    exactly when ``m(R) <= L - 1``, so the answer is the largest such ``R``.  The
    definitional window scan is kept as :func:`burst_dispersion_by_window_scan`
    and the two are checked against each other in
    ``validation/validate_burst_metric_equivalence.py``.

    Examples
    --------
    A 4x4 block interleaver fully disperses any burst of 4 but not of 5:

    >>> from interleavekit import BlockInterleaver
    >>> pos = BlockInterleaver(depth=4, span=4).position_of_input()
    >>> burst_dispersion(pos, 4)
    1
    >>> burst_dispersion(pos, 5)
    2
    """
    pos = _validated_positions(position_of_input)
    burst_length = _check_burst_length(burst_length)
    lo, hi = _window_bounds(pos, window_range)
    width = hi - lo
    if burst_length > width:
        raise ValueError(
            f"burst_length {burst_length} exceeds the {max(width, 0)} transmitted positions "
            f"available in window_range [{lo}, {hi}); feed more symbols or shorten the burst"
        )
    cap = min(burst_length, pos.size)
    profile = transmitted_span_profile(pos, cap, (lo, hi))
    ok = np.flatnonzero(profile <= burst_length - 1)
    if ok.size == 0:
        return 0
    return int(ok[-1]) + 1


def burst_dispersion_by_window_scan(
    position_of_input: NDArray | list[int],
    burst_length: int,
    window_range: tuple[int, int] | None = None,
) -> int:
    """Definitional reference implementation of :func:`burst_dispersion`.

    Scans every admissible burst window and takes the longest run of consecutive
    source indices among the symbols it damages.  ``O(N * burst_length)``, so it
    is slow; it exists so that the fast path in :func:`burst_dispersion` can be
    checked against the definition rather than trusted.  The two agree on every
    case in ``validation/validate_burst_metric_equivalence.py``.

    Parameters and return value are identical to :func:`burst_dispersion`.
    """
    pos = _validated_positions(position_of_input)
    burst_length = _check_burst_length(burst_length)
    lo, hi = _window_bounds(pos, window_range)
    width = hi - lo
    if burst_length > width:
        raise ValueError(
            f"burst_length {burst_length} exceeds the {max(width, 0)} transmitted positions "
            f"available in window_range [{lo}, {hi}); feed more symbols or shorten the burst"
        )
    source_at = np.full(width, -1, dtype=np.int64)
    inside = (pos >= lo) & (pos < hi)
    source_at[pos[inside] - lo] = np.flatnonzero(inside)

    worst = 0
    for start in range(0, width - burst_length + 1):
        hit = source_at[start : start + burst_length]
        hit = hit[hit >= 0]
        run = longest_consecutive_run(hit)
        if run > worst:
            worst = run
            if worst == burst_length:
                break
    return int(worst)


def burst_dispersion_profile(
    position_of_input: NDArray | list[int],
    burst_lengths: NDArray | list[int],
    window_range: tuple[int, int] | None = None,
) -> NDArray[np.int64]:
    """:func:`burst_dispersion` evaluated at several burst lengths.

    Parameters
    ----------
    position_of_input:
        As for :func:`burst_dispersion`.
    burst_lengths:
        Burst lengths in transmitted symbols, each >= 1.
    window_range:
        As for :func:`burst_dispersion`.

    Returns
    -------
    numpy.ndarray
        Worst-case surviving run for each burst length, dtype ``int64``.
    """
    lengths = np.asarray(burst_lengths, dtype=np.int64).ravel()
    if lengths.size == 0:
        return np.zeros(0, dtype=np.int64)
    pos = _validated_positions(position_of_input)
    lo, hi = _window_bounds(pos, window_range)
    biggest = int(lengths.max())
    if biggest > hi - lo:
        raise ValueError(
            f"burst_length {biggest} exceeds the {max(hi - lo, 0)} transmitted positions "
            f"available in window_range [{lo}, {hi}); feed more symbols or shorten the burst"
        )
    profile = transmitted_span_profile(pos, min(biggest, pos.size), (lo, hi))
    out = np.zeros(lengths.size, dtype=np.int64)
    for k, bl in enumerate(lengths.tolist()):
        bl = _check_burst_length(bl)
        ok = np.flatnonzero(profile[: min(bl, profile.size)] <= bl - 1)
        out[k] = 0 if ok.size == 0 else int(ok[-1]) + 1
    return out


def max_burst_fully_dispersed(
    position_of_input: NDArray | list[int],
    window_range: tuple[int, int] | None = None,
    max_search: int | None = None,
) -> int:
    """Largest burst length whose errors are all isolated after de-interleaving.

    Parameters
    ----------
    position_of_input:
        As for :func:`burst_dispersion`.
    window_range:
        As for :func:`burst_dispersion`.
    max_search:
        Upper limit on the answer.  Defaults to the width of the window range.

    Returns
    -------
    int
        Largest ``L`` with ``burst_dispersion(..., L) == 1``.

    Notes
    -----
    This is exactly ``m(2)``, the narrowest transmitted window that can hold two
    *adjacent* source symbols, clipped to the search limit:
    ``burst_dispersion(L) == 1`` holds precisely while ``L - 1 < m(2)``.  So the
    answer is the minimum transmitted separation between source symbols ``j`` and
    ``j+1``, which makes this an ``O(N)`` quantity and not a search.  Where no two
    adjacent source symbols both fall inside the window, every burst the window
    can hold is fully dispersed and the window width is returned.
    """
    pos = _validated_positions(position_of_input)
    lo, hi = _window_bounds(pos, window_range)
    width = max(hi - lo, 0)
    limit = width if max_search is None else min(int(max_search), width)
    if limit < 1:
        return 0
    profile = transmitted_span_profile(pos, 2, (lo, hi))
    if profile[0] > 0:  # sentinel: the window holds no source symbol at all
        return 0
    m2 = int(profile[1])
    if m2 == np.iinfo(np.int64).max:
        return limit
    return max(0, min(m2, limit))
