"""The covariance prohibition, and the code that enforces it.

Why this module exists
----------------------
This package measures how fast two propagators disagree. It has no tracking
data, no measurement residuals and no filter, so it cannot produce a
covariance: a covariance is a statement about the distribution of the error of
*one* estimate given the observations that produced it, and nothing here
observes anything. A divergence spread between two models is not that, however
much it looks like a one-sigma number on a plot.

The specific misuse this guards against is feeding an output of this package
into a conjunction assessment as the state covariance. A probability of
collision is a ratio of two small numbers and is dominated by the covariance
it is given; supplying a model-divergence spread in that slot would produce a
number with a plausible magnitude and no meaning, and the failure would be
silent. **P071 ConjProb** of this mission is the product that computes
collision probability, and its covariance must come from orbit determination
-- a filter with residuals, or a published covariance in a conjunction data
message -- and never from this package.

How it is enforced
------------------
Three ways, because a sentence in a README is not enforcement:

1. :meth:`orbitdrift.report.DriftReport.as_covariance` always raises.
2. :func:`refuse_covariance` raises on any request naming a covariance or a
   conjunction use, and :func:`check_purpose` screens a free-text purpose
   string against :data:`FORBIDDEN_PURPOSE_TERMS`.
3. :func:`audit_mapping_for_covariance_keys` rejects any serialised output
   that has grown a covariance-shaped key, so a future change that adds one
   fails a test rather than shipping.

``tests/test_guard.py`` checks that each of the three fires, and -- because an
assertion that something raises is worthless if it would raise on everything
-- that each one *passes* a legitimate request. ``validation/validate_guard.py``
records the same evidence with counts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .errors import CovarianceRefused

CROSS_REFERENCE = (
    "A conjunction assessment needs a covariance from orbit determination. "
    "See P071 ConjProb of this mission for the collision-probability "
    "computation, and give it a covariance produced by a filter with "
    "measurement residuals or one published in a conjunction data message. "
    "orbitdrift measures model divergence between two propagators and has no "
    "measurements, so nothing it returns is a covariance."
)

FORBIDDEN_PURPOSE_TERMS: tuple[str, ...] = (
    "covariance",
    "conjunction",
    "collision",
    "probability of collision",
    "pc",
    "cdm",
    "close approach",
    "miss distance",
    "screening",
    "maneuver screening",
    "manoeuvre screening",
    "encounter",
)
"""Terms in a declared purpose that make the request a refusal.

``pc`` is matched as a whole word only. The list is deliberately broad: a
false refusal costs the caller one keyword argument, and a false acceptance
costs a conjunction analysis its meaning.
"""

COVARIANCE_KEY_PATTERN = re.compile(
    r"(covar|sigma_matrix|cov_matrix|\bcov\b|p_matrix|state_covariance)", re.IGNORECASE
)
"""Keys that may not appear in anything this package serialises."""

_WORD = re.compile(r"[a-z0-9]+")


def refuse_covariance(detail: str = "") -> None:
    """Always raise :class:`orbitdrift.errors.CovarianceRefused`.

    This function has no success path. It exists so that every refusal in the
    package carries the same message and the same cross-reference, and so that
    a reader grepping for the prohibition finds one place.
    """
    prefix = f"{detail.rstrip('.')}. " if detail else ""
    raise CovarianceRefused(f"{prefix}{CROSS_REFERENCE}")


def check_purpose(purpose: str) -> str:
    """Return ``purpose`` unchanged, or raise if it names a forbidden use.

    Matching is on whole lower-cased words, plus a substring search for the
    multi-word terms. ``"stale ephemeris check"`` passes; ``"covariance for
    conjunction screening"`` does not.

    Raises
    ------
    CovarianceRefused
        If any term of :data:`FORBIDDEN_PURPOSE_TERMS` appears.
    """
    if not isinstance(purpose, str):
        raise TypeError(f"purpose must be a string; got {type(purpose).__name__}")
    lowered = purpose.lower()
    words = set(_WORD.findall(lowered))
    for term in FORBIDDEN_PURPOSE_TERMS:
        hit = term in words if " " not in term else term in lowered
        if hit:
            refuse_covariance(
                f"The declared purpose {purpose!r} names {term!r}, which this "
                "package refuses to serve"
            )
    return purpose


def audit_mapping_for_covariance_keys(mapping: Mapping[str, Any], *, where: str) -> None:
    """Raise if any key of ``mapping``, at any depth, looks like a covariance.

    Raises
    ------
    CovarianceRefused
        Naming the offending key and ``where`` it was found.
    """
    stack: list[tuple[str, Mapping[str, Any]]] = [(where, mapping)]
    while stack:
        path, current = stack.pop()
        for key, value in current.items():
            if COVARIANCE_KEY_PATTERN.search(str(key)):
                refuse_covariance(
                    f"{path}.{key} is a covariance-shaped key and this package "
                    "must not serialise one"
                )
            if isinstance(value, Mapping):
                stack.append((f"{path}.{key}", value))
