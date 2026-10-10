"""Published reference values for the CUSUM average run length.

This module exists so that the CUSUM detector has a **known answer** to be
checked against, rather than only being checked for self-consistency. Of the
five analytic detectors here, CUSUM is the one with a tabulated in-control ARL
in a source this session could read, so it is the one that anchors the
measurement machinery: if the harness reproduces CUSUM's published ARL0, the
same harness measuring ADWIN is measuring the thing it claims to.

Sources, both verified in this session
--------------------------------------
1. **NIST/SEMATECH e-Handbook of Statistical Methods**, section 6.3.2.3.1
   "Cusum Average Run Length", https://www.itl.nist.gov/div898/handbook/pmc/
   section3/pmc3231.htm, read 2026-10-10. For ``k = 0.5`` it tabulates the
   in-control ARL of a **one-sided** CUSUM as **336** at ``h = 4`` and **930**
   at ``h = 5``, and the ARL at a one-sigma mean shift as **8.38** and **10.4**
   respectively. The page states, verbatim: "If one has to control both positive
   and negative deviations, as is usually the case, two one-sided charts are
   used". The handbook credits no external author: "This Handbook used a
   computer program that furnished the required ARLs given the standardized h
   and k."

2. **Siegmund's approximation**, the closed form reproduced throughout the SPC
   literature for the one-sided CUSUM ARL,

   .. code-block:: text

       ARL(delta) ~ (exp(-2 D b) + 2 D b - 1) / (2 D^2),
       D = delta - k,   b = h + 1.166

   implemented in :func:`siegmund_one_sided_arl`. **The two sources are used to
   check each other**: the formula reproduces the handbook's 336 to 0.6 % and
   its 930 to 1.0 %, which is what makes it safe to use the formula at
   thresholds the handbook does not tabulate. That cross-check is a committed
   test (``tests/test_known_answers.py``), not a claim.

The two-sided correction, which is where this is easy to get wrong
------------------------------------------------------------------
This package's :class:`telemdrift.detectors.CUSUM` runs **both** one-sided
charts against the same decision interval and alarms when either crosses. Two
charts raise false alarms at twice the rate of one, so

.. code-block:: text

    ARL0_two_sided ~ ARL0_one_sided / 2

giving 168 at ``h = 4`` and 465 at ``h = 5`` for ``k = 0.5``. Those are the
numbers ``validation/validate_known_answers.py`` tests the implementation
against. Quoting the handbook's 336 as if it applied to a two-sided chart would
be wrong by a factor of two, and a factor of two is exactly the sort of error
that survives review because both numbers look plausible.

The halving is itself an approximation: the two charts are not independent, both
being driven by the same observations, and one chart's drift toward zero is
correlated with the other's drift away from it. The measured agreement below
bounds how good the approximation is in practice.
"""

from __future__ import annotations

import math

__all__ = [
    "NIST_CUSUM_ARL",
    "ks_asymptotic_tail_probability",
    "nist_two_sided_arl0",
    "siegmund_one_sided_arl",
    "siegmund_two_sided_arl0",
]

#: NIST/SEMATECH e-Handbook 6.3.2.3.1, k = 0.5, one-sided CUSUM.
#: ``{h: {"arl0": in-control ARL, "arl1_shift1": ARL at a 1-sigma mean shift}}``
NIST_CUSUM_ARL: dict[int, dict[str, float]] = {
    4: {"arl0": 336.0, "arl1_shift1": 8.38},
    5: {"arl0": 930.0, "arl1_shift1": 10.4},
}


def siegmund_one_sided_arl(h: float, k: float = 0.5, delta: float = 0.0) -> float:
    """Siegmund approximation to the one-sided CUSUM ARL, in samples.

    Parameters
    ----------
    h:
        Decision interval in standard deviations, > 0.
    k:
        Reference value in standard deviations, >= 0.
    delta:
        True mean shift in standard deviations. ``0.0`` gives the in-control
        ARL0.

    Returns
    -------
    float
        Approximate average run length in samples.

    Notes
    -----
    The ``delta == k`` case makes ``D`` zero and the expression removable; the
    limit is ``b^2`` and is returned directly rather than dividing by zero.
    Validity: the approximation is accurate to a few per cent for ``h`` of order
    2 to 10 and ``k`` of order 0.25 to 1.5; it is not exact for any argument and
    is not used anywhere in this package except as a reference value.
    """
    if h <= 0.0:
        raise ValueError("h must be > 0")
    if k < 0.0:
        raise ValueError("k must be >= 0")
    b = h + 1.166
    d = float(delta) - float(k)
    if abs(d) < 1e-12:
        return b * b
    return (math.exp(-2.0 * d * b) + 2.0 * d * b - 1.0) / (2.0 * d * d)


def siegmund_two_sided_arl0(h: float, k: float = 0.5) -> float:
    """Two-sided in-control ARL0 as half the one-sided value. See module docstring."""
    return 0.5 * siegmund_one_sided_arl(h, k, 0.0)


def nist_two_sided_arl0(h: int) -> float:
    """Handbook one-sided ARL0 at ``h``, halved for the two-sided chart."""
    if h not in NIST_CUSUM_ARL:
        raise ValueError(f"the handbook table covers h in {sorted(NIST_CUSUM_ARL)}, not {h}")
    return 0.5 * NIST_CUSUM_ARL[h]["arl0"]


def ks_asymptotic_tail_probability(c: float, n_ref: int, n_det: int) -> float:
    """Asymptotic null tail probability of the two-sample KS statistic.

    ``P(D > c) ~ 2 exp(-2 c^2 n m / (n + m))``, the leading term of the
    Kolmogorov distribution. Dimensionless, clipped to ``[0, 1]``.

    Why this is here
    ----------------
    It is the number a practitioner uses to pick a KS threshold, and this
    package's point is that it is the wrong number for a streaming detector.
    It is the false-alarm probability of **one** test on **independent** data.
    The windowed detector runs a test every ``stride`` samples on windows that
    overlap by ``1 - stride/n_det``, so successive tests are strongly dependent
    and the stream-level ARL0 is far longer than ``stride / P(D > c)``. The
    measured ratio between the two -- an *effective independent-test factor* --
    is reported in README.md and is stable at about 16 for the shipped
    ``n_ref = 200``, ``n_det = 100``, ``stride = 5``.

    Validity: the asymptotic form, so it is accurate for ``min(n, m)`` of order
    50 and above and over-states the tail for small ``c``. The exact finite
    sample distribution is available from ``scipy.stats.ks_2samp``; it is not
    used here because the point being made does not depend on the few per cent
    the approximation costs, and the approximation is the thing practitioners
    actually use.
    """
    if c <= 0.0:
        return 1.0
    if n_ref < 1 or n_det < 1:
        raise ValueError("n_ref and n_det must both be >= 1")
    eff = n_ref * n_det / (n_ref + n_det)
    return float(min(1.0, max(0.0, 2.0 * math.exp(-2.0 * c * c * eff))))
