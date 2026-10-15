"""Vector primitives. Bottom of the import graph: imports nothing local."""

import math


def dot(a, b):
    """Dot product of two equal-length sequences (dimensionless)."""
    if len(a) != len(b):
        raise ValueError("dot requires equal-length sequences")
    return sum(ai * bi for ai, bi in zip(a, b))


def norm(v):
    """Euclidean 2-norm of a sequence (same units as the input)."""
    return math.sqrt(dot(v, v))


def normalize(v):
    """Unit vector in the direction of v (dimensionless)."""
    n = norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [c / n for c in v]


def cross(a, b):
    """Cross product of two 3-vectors."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError("cross requires 3-vectors")
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
