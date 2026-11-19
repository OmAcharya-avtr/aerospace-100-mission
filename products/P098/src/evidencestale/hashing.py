"""Content hashing and closure snapshots.

SHA-256 as specified in FIPS PUB 180-4, *Secure Hash Standard*, National
Institute of Standards and Technology, August 2015. The digest is used as a
content identity, which is the same use a content-addressed version-control
system makes of it; the construction of a digest over a set of component
digests follows the hash-tree idea of R. C. Merkle, "A Digital Signature Based
on a Conventional Encryption Function", *Advances in Cryptology -- CRYPTO '87*,
LNCS 293, pages 369--378, Springer, 1988.

WHAT THE EXACTNESS OF THE CLOSURE COMPARISON IS CONDITIONAL ON, in the part
that lives in this module: two distinct contents must not share a digest. For
``n`` distinct contents the probability that some pair collides is bounded
above by ``n(n-1)/2 * 2**-256`` under the standard random-oracle treatment of
the digest. :func:`collision_probability_bound` returns that number so the
condition is published rather than asserted, and
``validation/validate_baselines.py`` prints it for the corpus actually used.
For the corpus sizes here the value is representable in binary64 -- of order
1e-73 for a few thousand distinct contents, far above the smallest normal
double -- so it is reported as a float and :func:`collision_bound_exponent`
reports its base-10 logarithm beside it for readability. This is a bound on an
idealised model of SHA-256, not a proof about SHA-256, and it says nothing
about an adversary choosing the contents.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping

#: Digest size of SHA-256 in bits (FIPS 180-4).
DIGEST_BITS = 256


def content_hash(content: str) -> str:
    """SHA-256 hexadecimal digest of ``content`` encoded as UTF-8.

    Parameters
    ----------
    content:
        Artifact content. Dimensionless text.

    Returns
    -------
    str
        64 lowercase hexadecimal characters.
    """
    if not isinstance(content, str):
        raise TypeError(
            f"content_hash() takes a str, got {type(content).__name__}. Decode bytes "
            f"to str with an explicit encoding before hashing, so the digest is a "
            f"function of the text and not of the caller's default encoding."
        )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def snapshot_hashes(contents: Mapping[str, str]) -> dict[str, str]:
    """Digest every artifact content in ``contents``.

    Returns a new mapping ``artifact -> digest``. This is the record an
    evidence item carries forward: the identity of every artifact in its
    subject's closure at the moment the evidence was produced.
    """
    return {name: content_hash(value) for name, value in contents.items()}


def closure_digest(hashes: Mapping[str, str]) -> str:
    """One digest over a whole closure snapshot, order-independent.

    The component digests are sorted by artifact identifier and concatenated
    with a separator that cannot occur in a hexadecimal digest or in an
    identifier that passed validation, then hashed. Sorting is what makes the
    result independent of iteration order, which matters because the snapshot
    arrives as a mapping.
    """
    parts = [f"{name}\x00{digest}" for name, digest in sorted(hashes.items())]
    return hashlib.sha256("\x01".join(parts).encode("utf-8")).hexdigest()


def collision_probability_bound(distinct_contents: int, digest_bits: int = DIGEST_BITS) -> float:
    """Upper bound on the probability that any two of ``n`` contents collide.

    ``n(n-1)/2 * 2**-digest_bits``, the union bound over pairs under the
    random-oracle model of the digest. Returns ``0.0`` only when there are
    fewer than two contents; for the corpus sizes here the value is a small but
    representable double, so a reported ``0.0`` always means "fewer than two
    contents" and never "too small to represent".
    """
    if isinstance(distinct_contents, bool) or not isinstance(distinct_contents, int):
        raise TypeError("distinct_contents must be an int")
    if distinct_contents < 0:
        raise ValueError(
            f"distinct_contents must be non-negative, got {distinct_contents}"
        )
    pairs = distinct_contents * (distinct_contents - 1) // 2
    return float(pairs) * 2.0 ** (-digest_bits)


def collision_bound_exponent(distinct_contents: int, digest_bits: int = DIGEST_BITS) -> float:
    """``log10`` of :func:`collision_probability_bound`.

    Reported beside the float because a base-10 exponent is the readable form
    of a number of order 1e-73, and because it stays finite for counts far
    beyond anything this product handles, where the float form would eventually
    underflow.
    """
    import math

    pairs = distinct_contents * (distinct_contents - 1) // 2
    if pairs == 0:
        return float("-inf")
    return math.log10(pairs) - digest_bits * math.log10(2.0)


def distinct_contents(values: Iterable[str]) -> int:
    """Number of distinct content strings in ``values``."""
    return len(set(values))
