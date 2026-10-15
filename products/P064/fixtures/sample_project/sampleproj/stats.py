"""Descriptive statistics. Bottom of the import graph."""

import math


def mean(xs):
    """Arithmetic mean (same units as the input)."""
    if not xs:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(xs) / len(xs)


def variance(xs):
    """Sample variance, Bessel-corrected, denominator n-1 (units squared)."""
    if len(xs) < 2:
        raise ValueError("sample variance needs at least two points")
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)


def rms(xs):
    """Root mean square (same units as the input)."""
    if not xs:
        raise ValueError("rms of an empty sequence is undefined")
    return math.sqrt(sum(x * x for x in xs) / len(xs))
