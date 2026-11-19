"""Closed-form re-run count and wasted runtime for an intermittently failing test.

This module is the mathematical core of the product. Everything else ranks,
resamples or reports; the cost of intermittency is defined here.

THE MODEL
---------

A test is attempted repeatedly until it passes. Each attempt fails
independently with probability ``p`` and passes with probability ``q = 1 - p``.
Let ``N`` be the number of attempts up to and including the first pass. Then
``N`` is geometric on the support ``{1, 2, 3, ...}``::

    P(N = k) = p**(k - 1) * (1 - p),     k = 1, 2, 3, ...

and its mean is the standard result

    E[N] = 1 / (1 - p)                                              (1)

*Source:* Feller, W. (1968), *An Introduction to Probability Theory and Its
Applications, Volume I*, 3rd edition, Wiley -- the geometric distribution and
its mean. Johnson, N. L., Kemp, A. W. & Kotz, S. (2005), *Univariate Discrete
Distributions*, 3rd edition, Wiley, chapter 5, gives the same result with the
shifted-support convention used here stated explicitly.

A derivation in one line, because the reader should not have to take (1) on
trust. Condition on the first attempt: with probability ``q`` it passes and
``N = 1``; with probability ``p`` it fails and the remaining process is a fresh
copy, so ``N = 1 + N'``. Hence ``E[N] = 1 + p * E[N]``, which rearranges to
(1). :func:`expected_attempts_by_recursion` evaluates that recursion directly
and is the independent cross-check used in the test suite.

WASTED ATTEMPTS
---------------

Only the *failed* attempts are waste: the final passing attempt is work that
had to happen anyway. The number of failed attempts is ``F = N - 1``, so

    E[F] = E[N] - 1 = p / (1 - p)                                   (2)

and with a failing attempt costing ``t`` seconds and the test being invoked
``S`` times over the period of interest,

    E[wasted seconds] = S * t * p / (1 - p)                         (3)

THE DIVERGENCE AT p -> 1
------------------------

(1), (2) and (3) all diverge as ``p -> 1``: a test that always fails is never
reached by retrying, and the expected number of attempts is infinite. This is
not a numerical artifact to be smoothed over, and this module does **not**
return ``inf``. :func:`expected_failed_attempts` with no retry budget raises
:class:`ValueError` at ``p >= 1`` with a message naming the two ways out --
supply a retry budget, or treat the test as a hard failure rather than a flake.
A silent ``inf`` would propagate into a ranking and sort a permanently broken
test above every real flake, which is the wrong answer to the question this
product asks.

A FIXED RETRY BUDGET
--------------------

Real CI systems cap retries. Let ``m >= 1`` be the maximum number of attempts
per invocation: the process stops at the first pass or after ``m`` attempts,
whichever comes first. Two facts give the closed form.

First, the tail-sum identity for a non-negative integer variable gives

    E[min(N, m)] = sum_{k=0}^{m-1} P(N > k) = sum_{k=0}^{m-1} p**k
                 = (1 - p**m) / (1 - p)                             (4)

Second, the failed-attempt count under the cap is ``F_m = min(N, m) - 1`` when
the test eventually passes, and ``F_m = m`` when all ``m`` attempts fail, an
event of probability ``p**m``. Writing that as one expression,
``F_m = min(N, m) - 1 + 1{N > m}``, and taking expectations with (4),

    E[F_m] = (1 - p**m) / (1 - p) - 1 + p**m
           = (p - p**m) / (1 - p) + p**m                            (5)

Equation (5) is what :func:`expected_failed_attempts` evaluates when
``retry_budget`` is given. Four checks on it, all exercised by the test suite:

* ``m = 1``  gives ``E[F_1] = p``, which is right: one attempt, failing with
  probability ``p``.
* ``m -> inf`` with ``p < 1`` gives ``p / (1 - p)``, recovering (2).
* ``E[F_m] <= E[F]`` for every ``m``, since a cap can only remove attempts.
* ``E[F_m]`` satisfies the recursion ``E[F_m] = p * (1 + E[F_{m-1}])`` with
  ``E[F_0] = 0``. :func:`expected_failed_attempts_by_recursion` evaluates it.

Unlike (2), equation (5) stays **finite at p = 1**, where it equals ``m``: all
``m`` attempts fail and all ``m`` are wasted. The limit is taken analytically
and returned, so ``p = 1.0`` with a retry budget is a legal, meaningful input.

WHERE THIS STOPS BEING VALID
----------------------------

Stated plainly, because a cost model used outside its assumptions produces a
ranking that looks authoritative and is wrong.

1. **Independent attempts.** Retries of a flaky test are frequently *not*
   independent: a failure caused by a leaked file, an exhausted port or a
   poisoned cache makes the next attempt more likely to fail, and infrastructure
   outages make whole batches fail together. Positive dependence makes (2) an
   **under**-estimate of the expected failed-attempt count, and negative
   dependence -- a retry that succeeds because the first attempt warmed a cache
   -- makes it an over-estimate. The model has no parameter for this and the
   product does not estimate one.
2. **Stationary p.** ``p`` is taken as constant over the period. A test whose
   flake rate changes after an infrastructure migration has no single ``p``, and
   the measured pooled rate is a time average that describes neither regime.
   This is why the learned component is scored on a held-out TIME split
   separately: see ``MODEL_CARD.md``.
3. **Runtime independent of outcome.** (3) multiplies by one runtime ``t``. If a
   failing attempt exits early -- an assertion fires in the first second of a
   ten-minute test -- the correct ``t`` is the mean runtime of the *failing*
   attempts, not of all attempts, and the two can differ by a large factor.
   :mod:`flakerank.baselines` uses the measured failing-attempt mean where the
   history contains failures and says so; the fallback where it does not is
   documented there.
4. **The retry is the only cost.** Wasted machine seconds are not the whole cost
   of a flake: there is the engineer who re-reads the log, the release that was
   held, and the real failure that was dismissed as "probably flaky". This
   product measures the machine seconds only, because those are the part a run
   history actually records.
5. **p estimated from a finite history.** Every ``p`` here is a point estimate
   from a finite number of attempts, and ``p / (1 - p)`` is convex, so plugging
   in the estimate is a biased estimator of the expectation (Jensen). For a test
   with zero observed failures the point estimate is zero and the expected cost
   is zero, which is why :mod:`flakerank.baselines` reports an accumulated
   ground truth with a bootstrap interval alongside the analytic model rather
   than instead of it.

All quantities in this module are in SI seconds. Probabilities are
dimensionless and lie in ``[0, 1]``.
"""

from __future__ import annotations

import math

__all__ = [
    "expected_attempts_by_recursion",
    "expected_attempts_to_first_pass",
    "expected_failed_attempts",
    "expected_failed_attempts_by_recursion",
    "expected_wasted_seconds",
    "failure_probability_for_expected_attempts",
]


def _check_probability(failure_probability: float, *, name: str = "failure_probability") -> float:
    """Validate a probability and return it as a float.

    Raises:
        TypeError: if the value is not a real number.
        ValueError: if the value is NaN or outside ``[0, 1]``.
    """
    if isinstance(failure_probability, bool) or not isinstance(failure_probability, (int, float)):
        raise TypeError(
            f"{name} must be a real number in [0, 1], got {type(failure_probability).__name__}"
        )
    value = float(failure_probability)
    if math.isnan(value):
        raise ValueError(f"{name} must be a real number in [0, 1], got NaN")
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1], got {value!r}. A per-attempt failure "
            "probability is estimated as failed_attempts / attempts and cannot "
            "leave that interval; a value outside it means the counts are wrong."
        )
    return value


def _check_retry_budget(retry_budget: int | None) -> int | None:
    """Validate a retry budget: ``None`` for unbounded, else an integer ``>= 1``."""
    if retry_budget is None:
        return None
    if isinstance(retry_budget, bool) or not isinstance(retry_budget, int):
        raise TypeError(
            "retry_budget must be None for an unbounded retry policy or an "
            f"integer >= 1 giving the maximum attempts per invocation, got "
            f"{type(retry_budget).__name__}"
        )
    if retry_budget < 1:
        raise ValueError(
            f"retry_budget must be >= 1, got {retry_budget}. A budget of m means "
            "at most m ATTEMPTS per invocation, so m = 1 is the no-retry policy "
            "and m = 0 would mean the test is never run."
        )
    return retry_budget


def expected_attempts_to_first_pass(failure_probability: float) -> float:
    """Expected attempts up to and including the first pass, ``1 / (1 - p)``.

    Equation (1) of the module docstring. Dimensionless (a count).

    Args:
        failure_probability: per-attempt failure probability ``p`` in ``[0, 1)``.

    Returns:
        ``1 / (1 - p)``, the mean of a geometric variable on ``{1, 2, ...}``.

    Raises:
        ValueError: at ``p = 1``, where the expectation diverges. The message
            names the two ways out rather than returning ``inf``.

    Examples:
        >>> expected_attempts_to_first_pass(0.25)
        1.3333333333333333
    """
    p = _check_probability(failure_probability)
    if p >= 1.0:
        raise ValueError(
            "expected attempts to first pass diverges at failure_probability = 1: "
            "a test that always fails is never reached by retrying. Either pass a "
            "retry_budget to expected_failed_attempts, which is finite and equals "
            "the budget at p = 1, or treat the test as a hard failure rather than "
            "an intermittent one. This function does not return inf, because inf "
            "propagated into a ranking sorts a permanently broken test above every "
            "real flake."
        )
    return 1.0 / (1.0 - p)


def expected_attempts_by_recursion(failure_probability: float, iterations: int = 200) -> float:
    """``E[N]`` from the conditioning recursion, as an independent cross-check.

    Evaluates ``E[N] = 1 + p * E[N]`` as the fixed-point iteration
    ``x <- 1 + p * x`` from ``x = 1``. It converges geometrically at rate ``p``
    and is used only by the test suite: it shares no expression with
    :func:`expected_attempts_to_first_pass`, so agreement between the two is
    evidence and not a tautology.

    Args:
        failure_probability: per-attempt failure probability ``p`` in ``[0, 1)``.
        iterations: fixed-point iterations. The residual after ``n`` iterations
            is ``p**n / (1 - p)``, so the default is exact to double precision
            for ``p <= 0.84`` and is deliberately *not* adaptive -- a test that
            compares two expressions should not have one of them silently
            tightened.

    Returns:
        The iterate after ``iterations`` steps (dimensionless).

    Raises:
        ValueError: at ``p = 1``, where the recursion has no fixed point.
    """
    p = _check_probability(failure_probability)
    if p >= 1.0:
        raise ValueError(
            "the recursion E[N] = 1 + p E[N] has no finite fixed point at "
            "failure_probability = 1; see expected_attempts_to_first_pass"
        )
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError(f"iterations must be an integer >= 1, got {iterations!r}")
    value = 1.0
    for _ in range(iterations):
        value = 1.0 + p * value
    return value


def expected_failed_attempts(
    failure_probability: float,
    retry_budget: int | None = None,
) -> float:
    """Expected number of FAILED attempts per invocation.

    Equation (2) of the module docstring when ``retry_budget`` is ``None``, and
    equation (5) when it is given. Dimensionless (a count).

    The failed attempts are the waste: the final passing attempt is work that
    had to happen anyway.

    Args:
        failure_probability: per-attempt failure probability ``p`` in ``[0, 1]``.
            ``p = 1`` is legal only with a retry budget.
        retry_budget: maximum attempts per invocation ``m >= 1``, or ``None`` for
            an unbounded retry policy.

    Returns:
        ``p / (1 - p)`` unbounded, or ``(p - p**m) / (1 - p) + p**m`` capped. At
        ``p = 1`` with a budget the analytic limit ``m`` is returned.

    Raises:
        ValueError: at ``p = 1`` with no retry budget, where the expectation
            diverges; or for an invalid probability or budget.

    Examples:
        >>> expected_failed_attempts(0.25)
        0.3333333333333333
        >>> expected_failed_attempts(0.25, retry_budget=3)
        0.328125
        >>> expected_failed_attempts(1.0, retry_budget=4)
        4.0
    """
    p = _check_probability(failure_probability)
    m = _check_retry_budget(retry_budget)
    if m is None:
        if p >= 1.0:
            raise ValueError(
                "expected failed attempts diverges at failure_probability = 1 with "
                "an unbounded retry policy: every attempt fails and the retry loop "
                "does not terminate. Pass retry_budget=m, for which the expectation "
                "is finite and equals m at p = 1, or classify the test as a hard "
                "failure rather than an intermittent one. This function does not "
                "return inf."
            )
        return p / (1.0 - p)
    if p >= 1.0:
        # The analytic limit of (5) as p -> 1, by L'Hopital on (p - p**m)/(1 - p):
        # the ratio tends to m - 1 and the p**m term tends to 1, giving m.
        return float(m)
    if m == 1:
        return p
    if p == 0.0:
        return 0.0
    # (5) rewritten as p * (1 - p**(m-1)) / (1 - p) + p**m. The two forms are
    # algebraically identical; this one evaluates 1 - p**(m-1) through expm1 of
    # log p, which keeps the leading cancellation accurate as p -> 1, where the
    # numerator and the denominator both go to zero.
    log_p = math.log(p)
    one_minus_power = -math.expm1((m - 1) * log_p)
    return p * one_minus_power / (1.0 - p) + math.exp(m * log_p)


def expected_failed_attempts_by_recursion(
    failure_probability: float,
    retry_budget: int,
) -> float:
    """``E[F_m]`` from the recursion ``E[F_m] = p (1 + E[F_{m-1}])``, ``E[F_0] = 0``.

    An independent evaluation of equation (5) that shares no expression with
    :func:`expected_failed_attempts`: it is a Horner-style accumulation of the
    polynomial ``p + p**2 + ... + p**m`` and involves no division at all, so it
    cannot share a cancellation with the closed form. Used by the test suite and
    by ``validation/validate_geometric.py``.

    Args:
        failure_probability: per-attempt failure probability ``p`` in ``[0, 1]``.
        retry_budget: maximum attempts per invocation ``m >= 1``.

    Returns:
        The expected failed-attempt count (dimensionless).
    """
    p = _check_probability(failure_probability)
    m = _check_retry_budget(retry_budget)
    if m is None:  # pragma: no cover - retry_budget is not optional here
        raise ValueError("retry_budget is required by the recursion form")
    value = 0.0
    for _ in range(m):
        value = p * (1.0 + value)
    return value


def expected_wasted_seconds(
    failure_probability: float,
    failing_attempt_seconds: float,
    retry_budget: int | None = None,
    invocations: int = 1,
) -> float:
    """Expected wasted runtime in seconds, equation (3) or its capped form.

    Args:
        failure_probability: per-attempt failure probability ``p`` in ``[0, 1]``.
        failing_attempt_seconds: mean runtime of a FAILING attempt, in seconds.
            This is deliberately not "the test's runtime": if a failing attempt
            exits early the two differ, and assumption 3 of the module docstring
            says which one equation (3) needs.
        retry_budget: maximum attempts per invocation, or ``None`` for unbounded.
        invocations: number of times the test is invoked over the period, ``S``
            in equation (3). Dimensionless count.

    Returns:
        Expected wasted seconds over the period.

    Raises:
        ValueError: for a negative or non-finite runtime, a non-positive
            invocation count, or the divergent unbounded case at ``p = 1``.

    Examples:
        >>> expected_wasted_seconds(0.25, 8.0)
        2.6666666666666665
        >>> round(expected_wasted_seconds(1.0 / 44.0, 600.0, invocations=330), 6)
        4604.651163
    """
    if isinstance(failing_attempt_seconds, bool) or not isinstance(
        failing_attempt_seconds, (int, float)
    ):
        raise TypeError(
            "failing_attempt_seconds must be a real number of seconds, got "
            f"{type(failing_attempt_seconds).__name__}"
        )
    seconds = float(failing_attempt_seconds)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(
            f"failing_attempt_seconds must be finite and >= 0, got {seconds!r}. "
            "A runtime is a duration in seconds; a negative or non-finite value "
            "means the history was parsed wrongly."
        )
    if isinstance(invocations, bool) or not isinstance(invocations, int) or invocations < 1:
        raise ValueError(
            f"invocations must be an integer >= 1, got {invocations!r}. It is the "
            "number of times the test ran over the period, which is at least one "
            "for any test present in a run history."
        )
    return invocations * seconds * expected_failed_attempts(failure_probability, retry_budget)


def failure_probability_for_expected_attempts(expected_attempts: float) -> float:
    """Invert equation (1): the ``p`` whose expected attempt count is given.

    ``p = 1 - 1 / E[N]``. Exposed because it is the exact inverse of
    :func:`expected_attempts_to_first_pass`, which gives the test suite a
    round-trip identity to drive with Hypothesis.

    Args:
        expected_attempts: ``E[N] >= 1``, dimensionless.

    Returns:
        The per-attempt failure probability in ``[0, 1)``.

    Raises:
        ValueError: for ``E[N] < 1``, which no geometric variable on
            ``{1, 2, ...}`` can have.
    """
    if isinstance(expected_attempts, bool) or not isinstance(expected_attempts, (int, float)):
        raise TypeError(
            f"expected_attempts must be a real number >= 1, got "
            f"{type(expected_attempts).__name__}"
        )
    value = float(expected_attempts)
    if not math.isfinite(value) or value < 1.0:
        raise ValueError(
            f"expected_attempts must be finite and >= 1, got {value!r}. The support "
            "of N is {1, 2, 3, ...}, so E[N] < 1 is impossible."
        )
    return 1.0 - 1.0 / value
