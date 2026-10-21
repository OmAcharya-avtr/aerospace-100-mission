"""Compressibility: the entropy floor, the operational constant, and a learned predictor.

A selection is made against *estimated* sizes. If the estimate is low the pass
overflows and the last product is truncated; if it is high the pass ends with
capacity unspent. This module produces the estimate three ways and
``validation/validate_compress.py`` measures the consequence of each in the
currency that matters, which is not the error on the ratio but the number of
windows that overflow and the value that is lost when they do.

Three estimators:

* :func:`entropy_floor_ratio` -- the analytic bound. For a memoryless source of
  first-order entropy ``H`` bits per sample stored at ``B`` bits per sample, no
  lossless coder can do better than ``H / B`` (Shannon 1948, source coding
  theorem). It needs no training data and no history, which is its whole
  attraction. It is also biased *high*, because a real codec exploits the
  spatial correlation that the first-order entropy ignores, so planning with it
  over-reserves capacity. The size of that bias is measured, not assumed.
* :func:`instrument_constant_table` -- the operational default: one ratio per
  instrument, the mean of what that instrument's products actually compressed
  to. This is what a mission database holds and it is a strong baseline.
* :class:`CompressibilityModel` -- a random forest over the seven features of
  :data:`downlinkpick.dataset.COMPRESSION_FEATURES`, including the
  lag-one autocorrelation proxy that the entropy floor cannot use.

No codec is run anywhere in this repository. The ratios that the learned model
is trained and scored against come from equation (7) of
:mod:`downlinkpick.dataset`, which is a synthetic stand-in for a codec, so every
number here is a statement about that generator. ``numcodecs`` is the package to
reach for if you want to measure a real ratio; it is named in the alternatives
table in README.md and is not a dependency.

References
----------
Shannon, C. E. (1948), A mathematical theory of communication, Bell System
Technical Journal 27(3), 379-423. Source coding theorem, hence equation (6) of
:mod:`downlinkpick.dataset` and :func:`entropy_floor_ratio` here.

Breiman, L. (2001), Random forests, Machine Learning 45(1), 5-32. The estimator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .dataset import COMPRESSION_FEATURES, compression_feature_row
from .inventory import MIN_COMPRESSION_RATIO, DataProduct, Inventory

#: Default forest size for the compressibility predictor. The target is bounded
#: in ``(0, 1]`` and smoother than the value target, so fewer trees suffice.
DEFAULT_N_ESTIMATORS = 200

#: Ratio used where no instrument-specific constant exists. The mid-point of the
#: permitted range rather than a measured figure, and a caller relying on it is
#: told so by :func:`instrument_constant_ratios` raising when asked to be strict.
GLOBAL_FALLBACK_RATIO = 0.5


def entropy_floor_ratio(product: DataProduct) -> float:
    """Memoryless compression-ratio bound ``min(H / B, 1)``, dimensionless.

    Parameters
    ----------
    product:
        The data product. ``H`` is
        :attr:`downlinkpick.inventory.DataProduct.entropy_bits_per_sample` and
        ``B`` is :attr:`downlinkpick.inventory.DataProduct.bit_depth`.

    Returns
    -------
    float
        The bound, clipped into ``[MIN_COMPRESSION_RATIO, 1.0]``. A real codec
        can and does go below this bound when the source is correlated, so this
        is a bound for an i.i.d. source and an over-estimate otherwise.
    """
    ratio = product.entropy_bits_per_sample / product.bit_depth
    return float(np.clip(ratio, MIN_COMPRESSION_RATIO, 1.0))


def entropy_floor_ratios(inventory: Inventory) -> dict[str, float]:
    """:func:`entropy_floor_ratio` for every product in ``inventory``."""
    return {product.name: entropy_floor_ratio(product) for product in inventory}


def instrument_constant_table(
    instruments: tuple[str, ...], ratios: np.ndarray
) -> dict[str, float]:
    """Mean realised ratio per instrument label, from a training sample.

    Parameters
    ----------
    instruments:
        Instrument label per training row.
    ratios:
        Realised compression ratio per training row, in ``(0, 1]``.

    Returns
    -------
    dict
        Mean ratio per instrument label, clipped into the permitted range.
    """
    values = np.asarray(ratios, dtype=float)
    if values.shape[0] != len(instruments):
        raise ValueError(
            f"{len(instruments)} instrument labels against {values.shape[0]} ratios"
        )
    if values.size == 0:
        raise ValueError("instrument_constant_table needs at least one row")
    table: dict[str, float] = {}
    for label in sorted(set(instruments)):
        mask = np.array([name == label for name in instruments])
        table[label] = float(np.clip(values[mask].mean(), MIN_COMPRESSION_RATIO, 1.0))
    return table


def instrument_constant_ratios(
    inventory: Inventory,
    table: Mapping[str, float],
    *,
    strict: bool = False,
) -> dict[str, float]:
    """Per-product ratio from an instrument lookup table.

    Parameters
    ----------
    inventory:
        The inventory.
    table:
        Instrument label to ratio, from :func:`instrument_constant_table`.
    strict:
        When true, raise on an instrument the table does not cover instead of
        falling back to :data:`GLOBAL_FALLBACK_RATIO`. A new instrument with no
        history is exactly the case where the operational default has nothing to
        say, and ``strict=True`` is how a caller finds that out rather than
        silently planning against 0.5.
    """
    unknown = sorted({p.instrument for p in inventory if p.instrument not in table})
    if unknown and strict:
        raise ValueError(
            "no constant-ratio entry for instrument(s): " + ", ".join(unknown)
        )
    return {
        product.name: float(table.get(product.instrument, GLOBAL_FALLBACK_RATIO))
        for product in inventory
    }


def global_mean_ratios(inventory: Inventory, ratios: np.ndarray) -> dict[str, float]:
    """One mean ratio for everything, the crudest usable estimator."""
    mean = float(np.clip(np.asarray(ratios, dtype=float).mean(), MIN_COMPRESSION_RATIO, 1.0))
    return dict.fromkeys(inventory.names, mean)


@dataclass(frozen=True)
class RatioPrediction:
    """Per-product compression-ratio prediction with its ensemble dispersion.

    Attributes
    ----------
    names:
        Product names in prediction order.
    ratio:
        Ensemble mean ratio, clipped into ``[MIN_COMPRESSION_RATIO, 1.0]``.
    dispersion:
        Standard deviation of the per-tree ratios, dimensionless. A dispersion
        measure over the ensemble, not a confidence interval.
    conservative:
        ``ratio + 2 * dispersion`` clipped to 1.0: the size estimate to plan
        with if an overflow is more expensive than unspent capacity. This is the
        one place the uncertainty output changes a decision rather than
        annotating one, and ``validation/validate_compress.py`` measures what it
        costs and what it saves.
    """

    names: tuple[str, ...]
    ratio: np.ndarray
    dispersion: np.ndarray
    conservative: np.ndarray

    def as_mapping(self) -> dict[str, float]:
        """Mean ratios as a ``{name: ratio}`` mapping."""
        return {name: float(r) for name, r in zip(self.names, self.ratio, strict=True)}

    def conservative_mapping(self) -> dict[str, float]:
        """Conservative ratios as a ``{name: ratio}`` mapping."""
        return {
            name: float(r) for name, r in zip(self.names, self.conservative, strict=True)
        }


class CompressibilityModel:
    """Random-forest regressor for the realised compression ratio."""

    def __init__(
        self,
        *,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        max_depth: int | None = None,
        min_samples_leaf: int = 2,
        seed: int = 20261021,
    ) -> None:
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2, got {n_estimators!r}")
        if min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {min_samples_leaf!r}")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.seed = int(seed)
        self._forest: object | None = None

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._forest is not None

    def fit(self, features: np.ndarray, target: np.ndarray) -> CompressibilityModel:
        """Fit on a matrix in :data:`COMPRESSION_FEATURES` column order."""
        from sklearn.ensemble import RandomForestRegressor

        x = np.asarray(features, dtype=float)
        y = np.asarray(target, dtype=float).ravel()
        if x.ndim != 2 or x.shape[1] != len(COMPRESSION_FEATURES):
            raise ValueError(
                f"features must have shape (n, {len(COMPRESSION_FEATURES)}), got {x.shape}"
            )
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"target has {y.shape[0]} rows against {x.shape[0]} features")
        if np.any(y <= 0.0) or np.any(y > 1.0):
            raise ValueError("compression ratios must lie in (0, 1]")
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            n_jobs=1,
        )
        forest.fit(x, y)
        self._forest = forest
        return self

    def _require_forest(self) -> object:
        if self._forest is None:
            raise RuntimeError("CompressibilityModel.fit must be called before prediction")
        return self._forest

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Ensemble mean ratio, clipped into the permitted range."""
        forest = self._require_forest()
        raw = forest.predict(np.asarray(features, dtype=float))  # type: ignore[attr-defined]
        return np.clip(raw, MIN_COMPRESSION_RATIO, 1.0)

    def predict_with_dispersion(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(mean ratio, per-tree standard deviation)``."""
        forest = self._require_forest()
        x = np.asarray(features, dtype=float)
        per_tree = np.vstack([tree.predict(x) for tree in forest.estimators_])  # type: ignore[attr-defined]
        mean = np.clip(per_tree.mean(axis=0), MIN_COMPRESSION_RATIO, 1.0)
        return mean, per_tree.std(axis=0, ddof=1)

    def predict_inventory(
        self, inventory: Inventory, autocorrelation: Mapping[str, float]
    ) -> RatioPrediction:
        """Predict a ratio for every product, with dispersion.

        Parameters
        ----------
        inventory:
            The inventory.
        autocorrelation:
            Lag-one autocorrelation proxy in ``[0, 1]`` per product name. This
            is a measured onboard statistic in the model and is not stored on
            :class:`downlinkpick.inventory.DataProduct`, because a product that
            has it and a product that does not are different situations and
            conflating them would hide which one a caller is in.
        """
        missing = [name for name in inventory.names if name not in autocorrelation]
        if missing:
            raise ValueError(
                "no lag-one autocorrelation supplied for: " + ", ".join(sorted(missing))
            )
        features = np.array(
            [
                compression_feature_row(product, float(autocorrelation[product.name]))
                for product in inventory
            ],
            dtype=float,
        )
        mean, dispersion = self.predict_with_dispersion(features)
        return RatioPrediction(
            names=inventory.names,
            ratio=mean,
            dispersion=dispersion,
            conservative=np.clip(mean + 2.0 * dispersion, MIN_COMPRESSION_RATIO, 1.0),
        )

    def feature_importances(self) -> dict[str, float]:
        """Impurity-based feature importances keyed by feature name."""
        forest = self._require_forest()
        return {
            name: float(weight)
            for name, weight in zip(
                COMPRESSION_FEATURES, forest.feature_importances_, strict=True  # type: ignore[attr-defined]
            )
        }


def overflow_bits(
    inventory: Inventory,
    plan_bits_by_window: Mapping[str, float],
    capacities: Mapping[str, float],
) -> dict[str, float]:
    """Per-window ``committed - capacity`` in bits, zero where it fits.

    Used by ``validation/validate_compress.py`` to convert a ratio error into
    the thing an operator sees, which is a pass that ran out of time with
    products still queued.
    """
    del inventory
    return {
        window: max(plan_bits_by_window.get(window, 0.0) - capacity, 0.0)
        for window, capacity in capacities.items()
    }
