"""Simple time-series filters. Imports stats."""

from sampleproj import stats


def moving_average(xs, window):
    """Centred-trailing moving average of window length `window` (samples)."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if len(xs) < window:
        raise ValueError("series shorter than the window")
    out = []
    for i in range(window - 1, len(xs)):
        out.append(stats.mean(xs[i - window + 1 : i + 1]))
    return out


def residual_rms(xs, window):
    """RMS of the series minus its moving average (same units as xs)."""
    smoothed = moving_average(xs, window)
    tail = xs[window - 1 :]
    return stats.rms([a - b for a, b in zip(tail, smoothed)])
