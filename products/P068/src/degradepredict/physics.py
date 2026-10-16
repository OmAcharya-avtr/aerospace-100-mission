"""Physics-shaped capacity-fade baseline: sublinear power-law loss.

The form
--------
For a lithium-ion cell whose capacity loss is limited by the growth of the
solid-electrolyte interphase, the lost capacity grows sublinearly with time or
charge throughput::

    C(t) / C0 = 1 - B * t**z,      0 < z < 1

with ``C`` the usable capacity, ``C0`` the beginning-of-life capacity, ``t`` the
service age in days, ``B`` a rate coefficient in ``day**-z`` and ``z`` the fade
exponent, dimensionless.

Sources for the exponent
------------------------
- Wang, J., Liu, P., Hicks-Garner, J., Sherman, E., Soukiazian, S., Verbrugge,
  M., Tataria, H., Musser, J. and Finamore, P. (2011), "Cycle-life model for
  graphite-LiFePO4 cells", *Journal of Power Sources* 196(8):3942-3948. Fits
  capacity loss as ``B exp(-Ea/RT) Ah**z`` against charge throughput with
  ``z = 0.55``, over 15 to 60 degrees Celsius and C-rates from C/2 to 10C.
- Broussely, M., Biensan, Ph., Bonhomme, F., Blanchard, Ph., Herreyre, S.,
  Nechev, K. and Staniewicz, R. J. (2005), "Main aging mechanisms in Li ion
  batteries", *Journal of Power Sources* 146(1-2):90-96. Describes the
  diffusion-limited calendar-ageing regime in which the SEI thickens as the
  square root of time, that is ``z = 0.5``.

The default here is ``z = 0.55`` from Wang et al. (2011).

Assumptions and validity range
------------------------------
1. One ageing regime only. The power law describes SEI-limited loss. It does not
   describe lithium plating, electrode cracking, sudden-death failure, or any
   thermal excursion, and it must not be extrapolated through a change of
   regime.
2. Wang et al. fit against **charge throughput at a controlled temperature**,
   not against calendar time at an uncontrolled one. Using ``t`` in days assumes
   a duty cycle and a thermal environment that are constant enough for throughput
   to be proportional to elapsed time. That is a strong assumption and it is the
   main reason this baseline can be wrong in a way its interval does not show.
3. ``z`` is held fixed at a declared value. The fitted interval is therefore
   conditional on ``z`` and contains no term for ``z`` being wrong. The shipped
   generator draws ``z`` per trajectory from ``[0.45, 0.65]``, so the shipped
   benchmark measures exactly that error.
4. Valid only while ``B t**z < 1``. Beyond that the model predicts negative
   capacity.

Why this is still a least-squares line
--------------------------------------
``C(t) = C0 - (C0 B) t**z`` is **linear in its two free parameters** for a fixed
``z``, so the fit, the covariance and the inverse-regression interval are all
the closed-form expressions in :mod:`degradepredict.baselines` with the time
link ``h(t) = t**z``. No nonlinear optimiser is involved and no convergence
failure is possible.
"""

from __future__ import annotations

import numpy as np

from .baselines import BaselineFit, DegradationFitError, fit_transformed_ols
from .limits import LimitSpec

__all__ = ["DEFAULT_FADE_EXPONENT", "FADE_EXPONENT_GRID", "powerlaw_fade_baseline"]

#: Wang et al. (2011), Journal of Power Sources 196(8):3942-3948, table 3.
DEFAULT_FADE_EXPONENT = 0.55

#: Grid used when ``z`` is profiled rather than declared. Brackets the z = 0.5 of
#: Broussely et al. (2005) and the z = 0.55 of Wang et al. (2011) with margin.
FADE_EXPONENT_GRID = np.round(np.arange(0.35, 0.901, 0.05), 2)


def powerlaw_fade_baseline(
    times_days: np.ndarray,
    values: np.ndarray,
    limit: LimitSpec,
    *,
    z: float | None = DEFAULT_FADE_EXPONENT,
    level: float = 0.90,
    origin_days: float | None = None,
) -> BaselineFit:
    """Fit ``C(t) = C0 (1 - B t**z)`` and invert it for the limit crossing.

    Parameters
    ----------
    times_days:
        Sample service ages in days, non-negative, length >= 3.
    values:
        Usable capacity in the same units as ``limit`` (percent of BOL in the
        shipped generator).
    limit:
        Declared end-of-life limit, downward direction.
    z:
        Fade exponent, dimensionless. ``None`` profiles over
        :data:`FADE_EXPONENT_GRID` by residual sum of squares and uses the
        minimiser. **The returned interval is then conditional on the selected
        ``z`` and understates the uncertainty**, because the selection step is
        not propagated; this is stated rather than corrected.
    level:
        Two-sided nominal confidence level.
    origin_days:
        Forecast origin, days. Defaults to the last sample time.

    Returns
    -------
    BaselineFit
        With ``model="powerlaw-fade"`` and ``z`` recorded.

    Raises
    ------
    ValueError
        On malformed input or a non-positive ``z``.
    DegradationFitError
        If no grid value yields a degrading fit, or the fit has no crossing
        after the origin.
    """
    if z is not None:
        return fit_transformed_ols(
            times_days, values, limit, z=float(z), log_metric=False, level=level,
            model="powerlaw-fade", origin_days=origin_days,
        )

    best: BaselineFit | None = None
    best_rss = np.inf
    t = np.asarray(times_days, dtype=float)
    y = np.asarray(values, dtype=float)
    failures: list[str] = []
    for z_try in FADE_EXPONENT_GRID:
        try:
            fit = fit_transformed_ols(
                t, y, limit, z=float(z_try), log_metric=False, level=level,
                model="powerlaw-fade", origin_days=origin_days,
            )
        except DegradationFitError as exc:
            failures.append(f"z={z_try}: {exc}")
            continue
        rss = float(np.sum((y - (fit.intercept + fit.slope * np.power(t, z_try))) ** 2))
        if rss < best_rss:
            best_rss, best = rss, fit
    if best is None:
        raise DegradationFitError(
            "no fade exponent on the grid produced a degrading fit: " + "; ".join(failures)
        )
    return best
