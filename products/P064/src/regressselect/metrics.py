"""Selection and prioritisation metrics: APFD, safety loss, selection cost.

APFD (Average Percentage of Faults Detected) is used exactly as defined in

    G. Rothermel, R. H. Untch, C. Chu and M. J. Harrold, "Prioritizing Test
    Cases For Regression Testing", IEEE Transactions on Software Engineering,
    27(10), 929-948, October 2001. doi:10.1109/32.962562

For a test order of ``n`` tests and a fault set of ``m`` faults, with ``TF_i``
the 1-based position in the order of the first test that detects fault ``i``:

    APFD = 1 - (TF_1 + ... + TF_m) / (n * m) + 1 / (2 * n)

Units: dimensionless fraction in (0, 1). Validity range: ``n >= 1``,
``m >= 1``, and every fault must be detected by at least one test in the
order. A fault detected by no test in the order has no ``TF_i``, the formula
is undefined for it, and :func:`apfd` raises rather than substituting a value;
the companion number for undetected faults is the safety loss below.

Safety loss has no single canonical source. It is defined here as the fraction
of genuinely failing tests that a selection omits,

    safety_loss(S, F) = |F \\ S| / |F|,   F non-empty

which is the complement of the recall of the selection with respect to the
tests that actually failed. The "safe" terminology follows Rothermel and
Harrold, "Analyzing Regression Test Selection Techniques", IEEE Transactions
on Software Engineering, 22(8), August 1996: a selection technique is safe
when it omits no test that could reveal a fault. Safety loss measures how far
from safe an unsafe technique actually is on a given history, which is the
number this product exists to report.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "apfd",
    "apfd_single_fault",
    "first_detection_positions",
    "safety_loss",
    "selection_cost",
    "detection_curve",
]


def first_detection_positions(
    order: Sequence[str], detecting: Mapping[str, Iterable[str]]
) -> dict[str, int]:
    """1-based position in ``order`` of the first test detecting each fault.

    ``detecting[f]`` is the set of test identifiers that detect fault ``f``.
    Faults with no detecting test present in ``order`` are omitted from the
    result, so the caller can see which ones were missed.
    """
    index = {t: i + 1 for i, t in enumerate(order)}
    if len(index) != len(order):
        raise ValueError("order contains duplicate test identifiers")
    out: dict[str, int] = {}
    for fault, tests in detecting.items():
        hits = [index[t] for t in tests if t in index]
        if hits:
            out[fault] = min(hits)
    return out


def apfd(order: Sequence[str], detecting: Mapping[str, Iterable[str]]) -> float:
    """APFD of ``order`` against the fault/test detection matrix ``detecting``.

    Returns a dimensionless fraction. Raises :class:`ValueError` when the
    order is empty, the fault set is empty, or any fault is detected by no
    test in the order -- in the last case the metric is undefined and the
    honest report is the safety loss, not a patched APFD.
    """
    n = len(order)
    if n == 0:
        raise ValueError("APFD is undefined for an empty test order")
    m = len(detecting)
    if m == 0:
        raise ValueError("APFD is undefined for an empty fault set")
    positions = first_detection_positions(order, detecting)
    missed = sorted(set(detecting) - set(positions))
    if missed:
        raise ValueError(
            "APFD is undefined: these faults are detected by no test in the order: "
            f"{missed}. Report safety_loss for this case instead."
        )
    return 1.0 - sum(positions.values()) / (n * m) + 1.0 / (2.0 * n)


def apfd_single_fault(order: Sequence[str], failing: Iterable[str]) -> float:
    """APFD for one fault revealed by any of ``failing``.

    This is the ``m = 1`` case of :func:`apfd` with the detection set
    ``failing``; it reduces to ``1 - TF/n + 1/(2n)``. Used for a per-revision
    history where the unit of regression is one revision.
    """
    return apfd(order, {"fault": list(failing)})


def safety_loss(selected: Iterable[str], failing: Iterable[str]) -> float:
    """Fraction of genuinely failing tests that ``selected`` omits.

    Dimensionless fraction in [0, 1]. Raises :class:`ValueError` when
    ``failing`` is empty, because the quantity is 0/0 there and averaging a
    substituted zero over revisions with no failures is exactly how a selector
    comes to look safer than it is.
    """
    fail = set(failing)
    if not fail:
        raise ValueError("safety_loss is undefined when no test actually failed")
    sel = set(selected)
    return len(fail - sel) / len(fail)


def selection_cost(selected: Iterable[str], universe: Iterable[str]) -> float:
    """Fraction of the whole suite that ``selected`` runs (dimensionless)."""
    uni = set(universe)
    if not uni:
        raise ValueError("selection_cost is undefined for an empty suite")
    sel = set(selected) & uni
    return len(sel) / len(uni)


def detection_curve(
    order: Sequence[str], detecting: Mapping[str, Iterable[str]]
) -> list[tuple[int, float]]:
    """Faults-detected fraction after each prefix length of ``order``.

    Returns ``[(k, fraction_detected_in_first_k), ...]`` for
    ``k = 0 .. len(order)``. Unit: (tests run, dimensionless fraction).
    """
    m = len(detecting)
    if m == 0:
        raise ValueError("detection_curve is undefined for an empty fault set")
    remaining = {f: set(ts) for f, ts in detecting.items()}
    found = 0
    curve = [(0, 0.0)]
    for k, test in enumerate(order, start=1):
        for fault in [f for f, ts in remaining.items() if test in ts]:
            del remaining[fault]
            found += 1
        curve.append((k, found / m))
    return curve
