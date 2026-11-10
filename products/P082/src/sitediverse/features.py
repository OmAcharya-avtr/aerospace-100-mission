"""The pair-regime table: one row per (station pair, regime, time half).

This is the single data structure every model in this package is fitted on and
scored on, so there is one definition of what a training example is.

**(32) The row.** A row is identified by ``(i, j, regime, half)`` and carries

    target   : the implied latent correlation of equation (11) over those epochs
    features : the eight declared predictors below
    observed : the observed joint cloud rate of equation (20) over those epochs

The target is a correlation and not a probability, deliberately. All three model
families -- independence, the fitted exponential of equation (23), and the
learned model -- are then exactly the same map followed by the same copula, and
the only thing that differs between them is how ``rho`` is produced. Comparing
them on ``rho`` alone would be a comparison in the wrong units; comparing them
on joint unavailability after a shared copula step isolates the correlation model
and reports in the units a network designer cares about. Both are reported.

**(33) The eight declared features.**

+----+-------------------+------------------------------------------------------+
| 0  | separation_km     | great-circle separation, equation (1)                |
| 1  | along_km          | along-axis separation, equation (3)                  |
| 2  | cross_km          | cross-axis separation, equation (3)                  |
| 3  | anisotropy_ratio  | ``along / (along + cross + 1)``, dimensionless       |
| 4  | regime            | declared seasonal regime, an integer label           |
| 5  | rate_min          | the smaller of the two stations' long-term rates     |
| 6  | rate_max          | the larger of the two                                |
| 7  | rate_product      | their product, which is equation (14) for the pair   |
+----+-------------------+------------------------------------------------------+

Features 1 to 3 are the ones the isotropic model of equation (23) cannot use,
and they are what makes a learned margin on the anisotropic record possible at
all. Feature 4 is the calendar and is not leakage. Features 5 to 7 are the
marginal climatology, available to every model.

The ``+1`` in feature 3 is a declared regularisation in kilometres: it keeps the
ratio finite for a coincident pair, where ``along`` and ``cross`` are both zero.
No pair in the declared network is coincident, so it never binds there; it binds
for a caller who supplies one.

**(34) What is NOT a feature, and why.** Latitude, longitude, altitude and the
station identity are all excluded. A model given the station identity can
memorise the training pairs, and the held-out SITES score would then measure
nothing: the point of that cell is a model that has never seen the station.
Altitude is excluded because the generator does not use it, so including it
would let the model fit noise with a predictor that cannot generalise even in
principle. Both exclusions are deliberate and both are the kind of decision that
decides a hold-out result, so they are written down here rather than left in the
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .climatology import Climatology
from .copula import ImpliedCorrelation, epoch_nodes, implied_correlation
from .geometry import SITES, Site, along_cross_matrices, separation_matrix
from .splits import Split

#: Names of the equation (33) features, in column order.
FEATURE_NAMES: tuple[str, ...] = (
    "separation_km",
    "along_km",
    "cross_km",
    "anisotropy_ratio",
    "regime",
    "rate_min",
    "rate_max",
    "rate_product",
)

#: Declared regularisation of feature 3, kilometres. See equation (33).
ANISOTROPY_FLOOR_KM: float = 1.0

#: Columns the isotropic model of equation (23) is entitled to use. Recorded so
#: that a test can assert the baseline is not accidentally handed the
#: anisotropy columns.
ISOTROPIC_COLUMNS: tuple[int, ...] = (0, 4, 5, 6, 7)


@dataclass(frozen=True)
class PairTable:
    """The equation (32) table.

    Attributes
    ----------
    site_i, site_j:
        ``(rows,)`` station indices, ``site_i < site_j``.
    regime:
        ``(rows,)`` declared regime label.
    half:
        ``(rows,)`` time half, 0 or 1.
    features:
        ``(rows, 8)`` the equation (33) features.
    target:
        ``(rows,)`` implied latent correlation, equation (11).
    observed:
        ``(rows,)`` observed joint cloud rate, equation (20).
    at_bracket_end:
        ``(rows,)`` boolean, whether the inversion hit its bracket.
    epoch_count:
        ``(rows,)`` number of epochs the row was computed over.
    cell:
        ``(rows,)`` split cell code from ``Split.cell_of``.
    """

    site_i: np.ndarray
    site_j: np.ndarray
    regime: np.ndarray
    half: np.ndarray
    features: np.ndarray
    target: np.ndarray
    observed: np.ndarray
    at_bracket_end: np.ndarray
    epoch_count: np.ndarray
    cell: np.ndarray

    @property
    def rows(self) -> int:
        """Number of rows."""
        return int(self.target.size)

    def mask(self, cell_code: int) -> np.ndarray:
        """Boolean row mask of one split cell."""
        return self.cell == int(cell_code)

    def separation_km(self) -> np.ndarray:
        """Column 0 of the features, named."""
        return self.features[:, 0]

    def subset(self, mask: np.ndarray) -> PairTable:
        """A new table containing only the rows selected by ``mask``."""
        selector = np.asarray(mask, dtype=bool)
        if selector.shape != (self.rows,):
            raise ValueError(
                f"mask must have shape ({self.rows},), got {selector.shape}"
            )
        return PairTable(
            site_i=self.site_i[selector],
            site_j=self.site_j[selector],
            regime=self.regime[selector],
            half=self.half[selector],
            features=self.features[selector],
            target=self.target[selector],
            observed=self.observed[selector],
            at_bracket_end=self.at_bracket_end[selector],
            epoch_count=self.epoch_count[selector],
            cell=self.cell[selector],
        )


def build_pair_table(
    cloud: np.ndarray,
    regime: np.ndarray,
    climatology: Climatology,
    split: Split,
    regime_count: int,
    pairs: tuple[tuple[int, int], ...],
    sites: tuple[Site, ...] = SITES,
) -> PairTable:
    """Build the equation (32) table from a record and an estimated climatology.

    Parameters
    ----------
    cloud:
        ``(epochs, S)`` boolean record.
    regime:
        ``(epochs,)`` declared regime labels.
    climatology:
        Estimated marginal climatology. The ground-truth thresholds are NOT
        accepted here; the signature takes a ``Climatology`` precisely so that
        a caller cannot pass them.
    split:
        The equation (29) split.
    regime_count:
        Number of declared regimes.
    pairs:
        Station pairs to tabulate.
    sites:
        Station geometry.

    Returns
    -------
    PairTable
    """
    indicator = np.asarray(cloud, dtype=bool)
    label = np.asarray(regime, dtype=int)
    if indicator.ndim != 2:
        raise ValueError(f"cloud must be two-dimensional, got {indicator.ndim}D")
    if label.shape != (indicator.shape[0],):
        raise ValueError(
            f"regime must have shape ({indicator.shape[0]},), got {label.shape}"
        )
    if climatology.epochs != indicator.shape[0]:
        raise ValueError(
            f"the climatology covers {climatology.epochs} epochs and the record "
            f"has {indicator.shape[0]}; they must be the same record"
        )
    if split.epochs != indicator.shape[0]:
        raise ValueError(
            f"the split covers {split.epochs} epochs and the record has "
            f"{indicator.shape[0]}; they must be the same record"
        )
    if not pairs:
        raise ValueError("at least one pair is required")
    separation = separation_matrix(sites)
    along, cross = along_cross_matrices(sites)
    threshold = climatology.threshold_series()
    rate = climatology.long_term_rate

    rows_i: list[int] = []
    rows_j: list[int] = []
    rows_regime: list[int] = []
    rows_half: list[int] = []
    rows_feature: list[list[float]] = []
    rows_target: list[float] = []
    rows_observed: list[float] = []
    rows_flag: list[bool] = []
    rows_epochs: list[int] = []

    for half in (0, 1):
        time_mask = split.time_mask(half)
        for which in range(regime_count):
            mask = time_mask & (label == which)
            count = int(np.count_nonzero(mask))
            if count < 3:
                raise ValueError(
                    f"time half {half}, regime {which} has {count} epochs, which "
                    "is too few to estimate a joint rate"
                )
            window = np.asarray(threshold)[mask]
            cloudy = indicator[mask]
            for site_i, site_j in pairs:
                observed = float(
                    np.mean(cloudy[:, site_i] & cloudy[:, site_j])
                )
                nodes = epoch_nodes(window[:, site_i], window[:, site_j])
                implied: ImpliedCorrelation = implied_correlation(nodes, observed)
                along_km = float(along[site_i, site_j])
                cross_km = float(cross[site_i, site_j])
                left = float(rate[site_i])
                right = float(rate[site_j])
                rows_i.append(int(site_i))
                rows_j.append(int(site_j))
                rows_regime.append(int(which))
                rows_half.append(int(half))
                rows_feature.append(
                    [
                        float(separation[site_i, site_j]),
                        along_km,
                        cross_km,
                        along_km / (along_km + cross_km + ANISOTROPY_FLOOR_KM),
                        float(which),
                        min(left, right),
                        max(left, right),
                        left * right,
                    ]
                )
                rows_target.append(implied.rho)
                rows_observed.append(observed)
                rows_flag.append(implied.at_bracket_end)
                rows_epochs.append(count)

    site_i = np.array(rows_i, dtype=int)
    site_j = np.array(rows_j, dtype=int)
    half_array = np.array(rows_half, dtype=int)
    return PairTable(
        site_i=site_i,
        site_j=site_j,
        regime=np.array(rows_regime, dtype=int),
        half=half_array,
        features=np.array(rows_feature, dtype=float),
        target=np.array(rows_target, dtype=float),
        observed=np.array(rows_observed, dtype=float),
        at_bracket_end=np.array(rows_flag, dtype=bool),
        epoch_count=np.array(rows_epochs, dtype=int),
        cell=split.cell_of(site_i, site_j, half_array),
    )
