"""Estimate the marginal cloud climatology, and its own sampling uncertainty.

Everything in this module is estimated FROM THE RECORD. The generator's true
thresholds exist in ``Record.threshold_true`` and no function here reads them:
an estimator that used the ground truth would make every downstream comparison
meaningless. ``validation/validate_climatology.py`` scores what is estimated
here against that ground truth, which is the one thing a synthetic record buys.

**(12) The season-by-hour cell estimator.** The marginal cloud probability of
station ``i`` at epoch ``t`` is modelled as a function of the epoch's position
in the seasonal and diurnal cycles alone,

    q_it = g_i( season(t), hour(t) )

and estimated by the cloud frequency in each of ``NSEASON * NHOUR`` cells,
shrunk toward the station's own long-term rate,

    q_hat_i(cell) = ( n_cell f_i(cell) + nu m_i ) / ( n_cell + nu )

where ``f_i(cell)`` is the observed frequency in the cell, ``n_cell`` the number
of epochs in it, ``m_i`` the station's long-term rate and ``nu`` a declared
pseudo-count. The shrinkage exists for one reason: a cell whose observed
frequency is exactly 0 or exactly 1 has a probit threshold of minus or plus
infinity, which propagates an infinity into equation (8). ``nu = 4`` moves an
all-clear cell of 160 epochs to a probability of ``4 m_i / 164``, which for a
long-term rate of 0.4 is 0.0098, a threshold of -2.33 rather than -inf. The
shrinkage is a BIAS toward the long-term rate and its size is measured in
``validation/validate_climatology_output.txt`` rather than assumed negligible.

The cells are not exactly equal in size. A declared year is 2920 epochs and
``NSEASON * NHOUR`` is 96, which does not divide it, so a season cell spans
30.42 days and the integer boundary falls at a different hour of the day each
time: cells carry 30 or 31 epochs. The imbalance is one part in thirty and is
carried correctly, because the estimator divides each cell's total by that
cell's OWN occupancy and never by a nominal count.

This is a non-parametric estimator, chosen over a harmonic regression because
it makes no claim about the shape of the cycles. Its cost is variance: with 12
season cells, 8 hour cells and an eight-year record there are about 243 epochs
per cell, of which about 40 are statistically independent after the persistence
of equation (5).

**(13) Effective sample size.** The epochs of equation (5) are not independent.
For a stationary AR(1) of coefficient ``phi``, the variance of the mean of ``n``
consecutive samples is inflated over the independent case by approximately
``(1 + phi) / (1 - phi)``, so the number of independent observations is

    n_eff = n (1 - phi) / (1 + phi)

(Bartlett 1946, *Journal of the Royal Statistical Society B* 8, 27; the result
is standard for AR(1) and is quoted in any time-series text). ``phi`` is
estimated from the record as the lag-one autocorrelation of the indicator
series, NOT taken from the generator, and the estimate is reported beside the
truth in validation. The purpose of ``n_eff`` is to put an honest standard error
on a cloud rate: the naive binomial ``sqrt(p(1-p)/n)`` understates it by
``sqrt((1+phi)/(1-phi))``, which at the declared ``phi = 0.72`` is a factor of
2.48. This is also why the bootstrap of ``sitediverse.empirical`` resamples
BLOCKS and not epochs.

Note on the lag-one autocorrelation of an INDICATOR. The latent field has
lag-one correlation ``phi`` exactly; the thresholded indicator does not. The
indicator's lag-one correlation is smaller, by the same copula relation of
equation (8), and the function below estimates the INDICATOR's correlation and
says so in its name. Using it in equation (13) is conservative in the direction
that matters -- it inflates fewer standard errors than the latent value would --
and the difference between the two is reported in validation rather than
asserted away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .field import EPOCHS_PER_DAY, EPOCHS_PER_YEAR

#: Season cells per year of the equation (12) estimator. DECLARED.
NSEASON: int = 12

#: Hour-of-day cells of the equation (12) estimator. DECLARED. Equal to the
#: number of epochs per day, so each cell is one sampling time.
NHOUR: int = EPOCHS_PER_DAY

#: Shrinkage pseudo-count ``nu`` of equation (12). DECLARED.
SHRINKAGE: float = 4.0


def cell_index(epochs: int, nseason: int = NSEASON, nhour: int = NHOUR) -> np.ndarray:
    """Season-by-hour cell index of each epoch, in ``[0, nseason * nhour)``."""
    if epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {epochs!r}")
    if nseason < 1 or nhour < 1:
        raise ValueError("nseason and nhour must both be at least 1")
    epoch = np.arange(epochs)
    season = (epoch % EPOCHS_PER_YEAR) * nseason // EPOCHS_PER_YEAR
    hour = epoch % nhour
    return season * nhour + hour


@dataclass(frozen=True)
class Climatology:
    """An estimated marginal cloud climatology.

    Attributes
    ----------
    cell_rate:
        ``(cells, S)`` estimated cloud probability per cell per station,
        strictly inside ``(0, 1)``.
    cell_of_epoch:
        ``(epochs,)`` cell index of each epoch.
    long_term_rate:
        ``(S,)`` observed long-term cloud rate per station. This is the
        quantity the independence baseline of ``sitediverse.independence`` uses,
        because it is what a diversity calculation in the literature uses.
    lag_one_indicator:
        ``(S,)`` estimated lag-one autocorrelation of the cloud INDICATOR.
    epochs:
        Record length.
    """

    cell_rate: np.ndarray
    cell_of_epoch: np.ndarray
    long_term_rate: np.ndarray
    lag_one_indicator: np.ndarray
    epochs: int

    @property
    def site_count(self) -> int:
        """Number of stations."""
        return int(self.cell_rate.shape[1])

    def rate_series(self) -> np.ndarray:
        """``(epochs, S)`` estimated marginal cloud probability per epoch."""
        return self.cell_rate[self.cell_of_epoch]

    def threshold_series(self) -> np.ndarray:
        """``(epochs, S)`` estimated probit threshold, ``Phi^{-1}`` of the rate."""
        return norm.ppf(self.rate_series())

    def effective_sample_size(self, epochs: int | None = None) -> np.ndarray:
        """Equation (13) per station, using the estimated indicator lag-one.

        Parameters
        ----------
        epochs:
            Number of consecutive epochs the mean is taken over. Defaults to the
            whole record.

        Returns
        -------
        numpy.ndarray
            ``(S,)`` effective sample sizes, each at least 1.0.
        """
        count = float(self.epochs if epochs is None else epochs)
        if count < 1.0:
            raise ValueError(f"epochs must be at least 1, got {epochs!r}")
        phi = np.clip(self.lag_one_indicator, 0.0, 0.999)
        return np.maximum(count * (1.0 - phi) / (1.0 + phi), 1.0)

    def rate_standard_error(self, epochs: int | None = None) -> np.ndarray:
        """Standard error of the long-term cloud rate, honouring equation (13)."""
        rate = self.long_term_rate
        return np.sqrt(rate * (1.0 - rate) / self.effective_sample_size(epochs))


def lag_one_autocorrelation(series: np.ndarray) -> np.ndarray:
    """Lag-one autocorrelation of each column of ``series``.

    Parameters
    ----------
    series:
        ``(epochs, S)`` array, boolean or float.

    Returns
    -------
    numpy.ndarray
        ``(S,)`` correlations. A constant column has no correlation defined and
        returns ``0.0``, which is the conservative choice: it declines to
        inflate a standard error on the strength of a degenerate series.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"series must be two-dimensional, got {values.ndim}D")
    if values.shape[0] < 3:
        raise ValueError(
            f"at least 3 epochs are needed for a lag-one estimate, got "
            f"{values.shape[0]}"
        )
    head = values[:-1]
    tail = values[1:]
    head_centred = head - head.mean(axis=0)
    tail_centred = tail - tail.mean(axis=0)
    numerator = np.sum(head_centred * tail_centred, axis=0)
    denominator = np.sqrt(
        np.sum(head_centred**2, axis=0) * np.sum(tail_centred**2, axis=0)
    )
    out = np.zeros(values.shape[1], dtype=float)
    good = denominator > 0.0
    out[good] = numerator[good] / denominator[good]
    return out


def estimate_climatology(
    cloud: np.ndarray,
    nseason: int = NSEASON,
    nhour: int = NHOUR,
    shrinkage: float = SHRINKAGE,
) -> Climatology:
    """Estimate the climatology of equation (12) from a cloud record.

    Parameters
    ----------
    cloud:
        ``(epochs, S)`` boolean record. ``True`` means cloudy.
    nseason, nhour:
        Cell counts of equation (12).
    shrinkage:
        ``nu`` of equation (12), a non-negative pseudo-count. Zero is permitted
        and is a defect waiting to happen: a cell that is all-clear then gives a
        threshold of ``-inf``. It is allowed so that
        ``validation/validate_climatology.py`` can measure what the shrinkage
        buys.

    Returns
    -------
    Climatology
    """
    indicator = np.asarray(cloud)
    if indicator.ndim != 2:
        raise ValueError(f"cloud must be two-dimensional, got {indicator.ndim}D")
    if indicator.shape[0] < 3:
        raise ValueError(
            f"a record of at least 3 epochs is needed, got {indicator.shape[0]}"
        )
    if shrinkage < 0.0:
        raise ValueError(f"shrinkage must be non-negative, got {shrinkage!r}")
    values = indicator.astype(float)
    epochs, count = values.shape
    cell = cell_index(epochs, nseason, nhour)
    cells = nseason * nhour
    occupancy = np.bincount(cell, minlength=cells).astype(float)
    if np.any(occupancy == 0.0):
        raise ValueError(
            f"{int(np.sum(occupancy == 0.0))} of {cells} season-by-hour cells are "
            "empty; the record is too short for this cell count"
        )
    totals = np.stack(
        [
            np.bincount(cell, weights=values[:, site], minlength=cells)
            for site in range(count)
        ],
        axis=1,
    )
    long_term = values.mean(axis=0)
    rate = (totals + shrinkage * long_term) / (occupancy[:, None] + shrinkage)
    if np.any(rate <= 0.0) or np.any(rate >= 1.0):
        raise ValueError(
            "the estimated cell rate reached 0 or 1, which gives an infinite "
            "probit threshold; raise `shrinkage` above "
            f"{shrinkage!r} or lower the cell count"
        )
    return Climatology(
        cell_rate=rate,
        cell_of_epoch=cell,
        long_term_rate=long_term,
        lag_one_indicator=lag_one_autocorrelation(values),
        epochs=int(epochs),
    )
