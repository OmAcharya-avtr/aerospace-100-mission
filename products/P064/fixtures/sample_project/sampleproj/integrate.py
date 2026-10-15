"""Fixed-step quadrature. Bottom of the import graph."""


def trapezoid(ys, h):
    """Trapezoid rule over uniformly spaced samples; h is the step (x units)."""
    if len(ys) < 2:
        raise ValueError("trapezoid needs at least two samples")
    total = 0.5 * (ys[0] + ys[-1]) + sum(ys[1:-1])
    return total * h


def simpson(ys, h):
    """Composite Simpson rule; requires an odd number of samples."""
    n = len(ys)
    if n < 3 or n % 2 == 0:
        raise ValueError("simpson needs an odd number of samples, at least three")
    total = ys[0] + ys[-1]
    for i in range(1, n - 1):
        total += 4.0 * ys[i] if i % 2 == 1 else 2.0 * ys[i]
    return total * h / 3.0
