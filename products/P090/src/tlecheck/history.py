"""Consistency of one element set against the object's own element history.

This is the screen that catches what no bound can: an element value that is
perfectly plausible for *an* Earth satellite and wrong for *this* one. A B*
drag term of -3.9e-5 is inside every physical bound in
:mod:`tlecheck.bounds`, and it is a four-hundred-sigma outlier against a
history of +3.9e-5.

Two robust estimators are used and the choice between them is per element,
because two of the elements genuinely drift.

Equations
---------
(7) Scaled median absolute deviation, the robust scale estimate:

        s = 1.4826 * median(|x_i - median(x)|)

    The constant 1.4826 makes ``s`` a consistent estimator of the standard
    deviation for Gaussian data. Source: Rousseeuw, P. J. and Croux, C.
    (1993), *Alternatives to the median absolute deviation*, Journal of the
    American Statistical Association 88(424), 1273-1283, section 2.

(8) Modified z-score of a candidate value against a history:

        z = |x - median(x)| / s

    reported as ``inf`` when ``s`` is zero and ``x`` differs from the median,
    and as 0 when both are zero. A zero scale is a real outcome for a
    quantised element whose history is constant, and collapsing it to a
    number would be a quieter lie than reporting the infinity.

(9) Theil-Sen slope, the robust trend estimate:

        b = median over all i < j of (x_j - x_i) / (t_j - t_i)

    Source: Sen, P. K. (1968), *Estimates of the regression coefficient based
    on Kendall's tau*, Journal of the American Statistical Association
    63(324), 1379-1389, equation (2.6); after Theil, H. (1950), *A rank-
    invariant method of linear and polynomial regression analysis*,
    Proceedings of the Royal Netherlands Academy of Sciences 53, 386-392.
    The intercept is taken as ``median(x_i - b t_i)``, which is the estimator
    Sen (1968) section 5 pairs with the slope.

(10) Detrended modified z-score, used for the elements that drift:

        r_i = x_i - (a + b t_i),   z = |r_candidate| / (1.4826 median(|r_i|))

**Which elements get which test.** Mean motion and B* are tested with equation
(10), because a decaying object's mean motion rises monotonically and its drag
term varies with solar activity; a plain median test would flag every honest
element set of a decaying object. Inclination, eccentricity and the right
ascension of the ascending node are tested with equation (8) after the
angle-unwrapping below, because their secular rates over a short history are
small compared with the corruptions this screen looks for. The argument of
perigee and the mean anomaly are **not** tested at all: both circulate at
rates that make a short history uninformative, and the screen reports that as
a determinability limit rather than running a test it cannot justify.

**Angle unwrapping.** The right ascension of the ascending node regresses
through 360 degrees. Its history is unwrapped into a continuous branch before
equation (9) is applied, and the candidate is placed on the same branch. A
screen that did not do this would report every node crossing as a corruption.

Units
-----
``times`` are Julian dates in days. Angles are degrees. The modified z-scores
are dimensionless.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .elements import ElementSet

MAD_CONSISTENCY = 1.4826
"""Equation (7) consistency constant, Rousseeuw and Croux (1993)."""

MINIMUM_HISTORY = 3
"""Element sets needed before equation (7) has a scale at all."""

LEVEL_ELEMENTS: tuple[str, ...] = ("inclination_deg", "eccentricity")
"""Tested by equation (8), the level test."""

TREND_ELEMENTS: tuple[str, ...] = ("raan_deg", "mean_motion_rev_per_day", "bstar")
"""Tested by equation (10) because they genuinely drift.

``raan_deg`` is in this group and not the level group, and the reason is
equation (16) of :mod:`tlecheck.dataset`: the node regresses secularly, so a
level test against the median of a history is wrong by construction. Measured
on 217 uncorrupted element sets of the test split, by
``validation/validate_baseline_history.py``, which runs both tests on the same
records: the level test gives a modified z-score between 1.3787 and 1.5246 on
every single one, a spread of 0.1459. That is a fixed offset -- the ratio of the
candidate's lever arm to the history's -- and it carries no information about
corruption at all. The ``raan_deg`` history is unwrapped by
:func:`unwrap_degrees` before the trend fit.
"""

QUANTISATION: dict[str, float] = {
    "inclination_deg": 1.0e-4,
    "raan_deg": 1.0e-4,
    "arg_perigee_deg": 1.0e-4,
    "mean_anomaly_deg": 1.0e-4,
    "eccentricity": 1.0e-7,
    "mean_motion_rev_per_day": 1.0e-8,
}
"""One count of each element's field in the two-line element text format.

These follow directly from the column spans: four decimal places on the angles,
seven implied decimals on the eccentricity, eight on the mean motion. The B*
term is not here because its field is a five-digit mantissa with an exponent,
so its step is relative rather than absolute and is computed by
:func:`scale_floor`.

**Why a floor is needed at all, and why it is not a tuning knob.** The text
format cannot express a difference smaller than one count, so a sample of
element sets whose values differ only in the last digit has a robust scale that
is an artefact of the quantiser and not a measurement of anything. Measured by
``validation/validate_baseline_history.py`` on a constructed unjittered history
whose candidate sits exactly one count of the mean-motion field from its robust
trend prediction: the unfloored modified z-score is **2.531364e+06** times the
floored one, which is 1.000000 by construction.

**And the honest other half.** On the corpus this repository actually ships the
floor changes nothing: the largest finite detrended mean-motion z-score over 217
uncorrupted element sets is 6.6304 with the floor and 6.6304 without it, because
the generator's declared mean-motion jitter of 1.5e-8 relative already keeps the
robust scale above one count. The floor is kept because it is derived from the
text format rather than from a corpus, and a real catalogue can publish element
sets identical to the last written digit. Its measured effect here is nil and
that is published rather than left to be inferred.
"""

BSTAR_RELATIVE_QUANTISATION = 1.0e-5
"""One count of the five-digit B* mantissa, relative to its own magnitude."""

BSTAR_ABSOLUTE_FLOOR = 1.0e-12
"""Floor on the B* step for a B* of zero, from the one-digit exponent field."""


def scale_floor(element: str, values: np.ndarray) -> float:
    """Smallest robust scale the element's own text field can express.

    Returns 0.0 for an element with no declared quantisation, which leaves
    equation (7) unfloored for it.
    """
    if element in QUANTISATION:
        return QUANTISATION[element]
    if element == "bstar":
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        magnitude = float(np.median(np.abs(finite))) if finite.size else 0.0
        return max(BSTAR_RELATIVE_QUANTISATION * magnitude, BSTAR_ABSOLUTE_FLOOR)
    return 0.0

UNTESTED_ELEMENTS: tuple[str, ...] = ("arg_perigee_deg", "mean_anomaly_deg")
"""Deliberately untested. Reported as a determinability limit."""


def robust_scale(values: np.ndarray) -> float:
    """Equation (7). Returns 0.0 for a constant sample, never ``nan``."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    deviation = np.abs(finite - np.median(finite))
    return float(MAD_CONSISTENCY * np.median(deviation))


def modified_z(value: float, values: np.ndarray, floor: float = 0.0) -> float:
    """Equation (8), with the scale floored at ``floor``.

    ``inf`` when the floored scale is zero and the value differs.
    """
    scale = max(robust_scale(values), floor)
    if not math.isfinite(value) or not math.isfinite(scale):
        return float("nan")
    centre = float(np.median(np.asarray(values, dtype=float)))
    residual = abs(value - centre)
    if scale == 0.0:
        return 0.0 if residual == 0.0 else float("inf")
    return residual / scale


def theil_sen(times: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Equation (9) slope and its paired median intercept.

    Returns ``(nan, nan)`` for fewer than two distinct abscissae.
    """
    t = np.asarray(times, dtype=float)
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(x)
    t, x = t[mask], x[mask]
    if t.size < 2:
        return float("nan"), float("nan")
    i, j = np.triu_indices(t.size, k=1)
    dt = t[j] - t[i]
    usable = dt != 0.0
    if not usable.any():
        return float("nan"), float("nan")
    slopes = (x[j][usable] - x[i][usable]) / dt[usable]
    slope = float(np.median(slopes))
    intercept = float(np.median(x - slope * t))
    return slope, intercept


def unwrap_degrees(values: np.ndarray) -> np.ndarray:
    """Place a degree-valued sequence on a continuous branch.

    Wraps are removed by adding the multiple of 360 that minimises the step
    from the previous value. Applied to the right ascension of the ascending
    node, whose regression crosses 360 degrees on any history long enough to
    be useful.
    """
    series = np.asarray(values, dtype=float).copy()
    if series.size == 0:
        return series
    for index in range(1, series.size):
        if not (math.isfinite(series[index]) and math.isfinite(series[index - 1])):
            continue
        step = series[index] - series[index - 1]
        series[index] -= 360.0 * round(step / 360.0)
    return series


def _place_on_branch(value: float, branch: np.ndarray, original_last: float) -> float:
    """Put ``value`` on the same 360-degree branch as an unwrapped history."""
    if branch.size == 0 or not math.isfinite(value) or not math.isfinite(original_last):
        return value
    offset = branch[-1] - original_last
    shifted = value + offset
    step = shifted - branch[-1]
    return shifted - 360.0 * round(step / 360.0)


@dataclass(frozen=True)
class HistoryFinding:
    """One element whose value is an outlier against the object's own history."""

    element: str
    test: str
    z_score: float
    threshold: float
    value: float
    predicted: float
    scale: float


@dataclass(frozen=True)
class HistoryVerdict:
    """Result of the history screen, including the case where it cannot run.

    Attributes
    ----------
    usable:
        ``True`` when the history had at least :data:`MINIMUM_HISTORY` prior
        element sets. When ``False`` the screen ran no test and ``reason``
        says why; this is the first-class "cannot determine" outcome of this
        module and it is never reported as a pass.
    reason:
        Empty when ``usable``.
    findings:
        One entry per element that failed its test.
    z_scores:
        Every z-score computed, including the ones that passed, so a report
        can show the margin rather than only the verdict.
    untested:
        Elements this screen deliberately does not test.
    history_length:
        Number of prior element sets supplied.
    """

    usable: bool
    reason: str
    findings: tuple[HistoryFinding, ...]
    z_scores: dict[str, float]
    untested: tuple[str, ...]
    history_length: int


def check(
    element_set: ElementSet,
    history: list[ElementSet],
    threshold: float = 4.0,
) -> HistoryVerdict:
    """Test ``element_set`` against the object's own prior element sets.

    Parameters
    ----------
    element_set:
        The candidate element set.
    history:
        Prior element sets for the same object, in any order. Element sets
        whose satellite number differs from the candidate's are a caller
        error and raise.
    threshold:
        Modified z-score above which an element is reported an outlier. The
        default of 4.0 is a declared policy choice and is labelled as such
        wherever it is reported. It is not tuned to the corruption corpus: see
        ``validation/validate_rule_screens.py``, which sweeps it.

    Raises
    ------
    ValueError
        If ``threshold`` is not a positive finite number, or any history
        entry belongs to a different object.
    """
    from .epoch import epoch_julian_date

    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be a positive finite number, got {threshold!r}")
    for prior in history:
        if prior.satnum != element_set.satnum:
            raise ValueError(
                f"history entry for object {prior.satnum} supplied for candidate "
                f"object {element_set.satnum}; the history screen compares an object "
                "against itself and nothing else"
            )
    if len(history) < MINIMUM_HISTORY:
        return HistoryVerdict(
            usable=False,
            reason=(
                f"{len(history)} prior element sets supplied; equation (7) has no "
                f"scale below {MINIMUM_HISTORY}, so no history test was run and no "
                "statement about element-history consistency is available"
            ),
            findings=(),
            z_scores={},
            untested=UNTESTED_ELEMENTS,
            history_length=len(history),
        )

    ordered = sorted(history, key=epoch_julian_date)
    times = np.array([epoch_julian_date(prior) for prior in ordered], dtype=float)
    candidate_time = epoch_julian_date(element_set)

    findings: list[HistoryFinding] = []
    z_scores: dict[str, float] = {}

    for name in LEVEL_ELEMENTS:
        series = np.array([getattr(prior, name) for prior in ordered], dtype=float)
        value = float(getattr(element_set, name))
        z = modified_z(value, series, scale_floor(name, series))
        z_scores[name] = z
        if math.isfinite(z) and z <= threshold:
            continue
        if math.isnan(z):
            continue
        findings.append(
            HistoryFinding(
                element=name,
                test="equation (8) modified z-score",
                z_score=z,
                threshold=threshold,
                value=value,
                predicted=float(np.median(series)),
                scale=max(robust_scale(series), scale_floor(name, series)),
            )
        )

    for name in TREND_ELEMENTS:
        series = np.array([getattr(prior, name) for prior in ordered], dtype=float)
        value = float(getattr(element_set, name))
        if name == "raan_deg":
            branch = unwrap_degrees(series)
            value = _place_on_branch(value, branch, float(series[-1]))
            series = branch
        slope, intercept = theil_sen(times, series)
        if not math.isfinite(slope):
            z_scores[name] = float("nan")
            continue
        residuals = series - (intercept + slope * times)
        scale = max(robust_scale(residuals), scale_floor(name, series))
        predicted = intercept + slope * candidate_time
        residual = abs(value - predicted)
        if scale == 0.0:
            z = 0.0 if residual == 0.0 else float("inf")
        else:
            z = residual / scale
        z_scores[name] = z
        if math.isfinite(z) and z <= threshold:
            continue
        if math.isnan(z):
            continue
        findings.append(
            HistoryFinding(
                element=name,
                test="equation (10) detrended modified z-score",
                z_score=z,
                threshold=threshold,
                value=value,
                predicted=predicted,
                scale=scale,
            )
        )

    return HistoryVerdict(
        usable=True,
        reason="",
        findings=tuple(findings),
        z_scores=z_scores,
        untested=UNTESTED_ELEMENTS,
        history_length=len(history),
    )
