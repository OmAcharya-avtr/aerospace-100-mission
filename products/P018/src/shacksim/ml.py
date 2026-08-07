"""Learned per-subaperture slope estimator with an ensemble confidence output.

The classical estimators in `shacksim.slopes` were implemented and validated
first; this model is benchmarked against the **thresholded centre of gravity**
on identical held-out stamps (see `validation/VALIDATION.md`, README
"Benchmark results"). The target regime is low flux and elongated spots, where
the CoG read-noise term of `shacksim.slopes.cog_noise_sigma` dominates.

Architecture: a small ensemble of independently seeded
`sklearn.neural_network.MLPRegressor` networks. The ensemble mean is the point
estimate, the per-component ensemble standard deviation is the confidence
output. Deep ensembles as an uncertainty proxy: Lakshminarayanan, Pritzel &
Blundell (2017), "Simple and Scalable Predictive Uncertainty Estimation using
Deep Ensembles", *NeurIPS 30*. The spread measures disagreement between
members, which is **not** the same as a calibrated 1-sigma error bar — the
measured calibration ratio is reported in `MODEL_CARD.md`.

PyTorch is not available in this build environment, so a convolutional model is
not an option; a fully connected ensemble is used instead (README Limitations).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor

from .geometry import LensletArray

__all__ = ["MLSlopeEstimator"]


class MLSlopeEstimator:
    """Ensemble regressor from a subaperture stamp to a wavefront slope.

    Parameters
    ----------
    array:
        Lenslet geometry. Fixes the expected stamp size and the pixel-to-slope
        conversion; a model fitted for one geometry must not be used with
        another.
    n_estimators:
        Number of ensemble members [-], >= 2 (a spread needs at least two).
    hidden_layer_sizes:
        MLP hidden widths, passed straight to `MLPRegressor`.
    alpha:
        L2 regularization strength.
    max_iter:
        Maximum Adam epochs.
    random_state:
        Base seed; member ``k`` uses ``random_state + k``. Fixing it makes
        training reproducible.

    Attributes
    ----------
    fitted_: whether `fit` has been called.

    Notes
    -----
    Features per stamp (see `features`):

    1. the stamp clipped at zero and divided by its own total, flattened
       (``p * p`` values) — removes overall gain, keeps the spot shape;
    2. ``log10(1 + total counts)`` — a single scalar telling the network which
       noise regime it is in.

    Feature 2 means the model is **not** gain invariant and assumes its input
    is in photoelectrons on the same scale as the training data. Feeding it
    ADU, or a detector with a different conversion gain, invalidates it.

    Targets are spot displacements in pixels (O(1) numbers, well conditioned
    for a neural network); slopes are converted in and out with
    `LensletArray.slope_to_displacement`.
    """

    def __init__(
        self,
        array: LensletArray,
        n_estimators: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (96, 48),
        alpha: float = 1e-4,
        max_iter: int = 400,
        random_state: int = 0,
    ) -> None:
        if not isinstance(array, LensletArray):
            raise TypeError(f"array must be a LensletArray, got {type(array).__name__}")
        if not isinstance(n_estimators, (int, np.integer)) or isinstance(
            n_estimators, (bool, np.bool_)
        ):
            raise TypeError(f"n_estimators must be an int, got {type(n_estimators).__name__}")
        if n_estimators < 2:
            raise ValueError(
                f"n_estimators must be >= 2 so that an ensemble spread exists, got {n_estimators}"
            )
        self.array = array
        self.n_estimators = int(n_estimators)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._members: list[MLPRegressor] = []

    # ------------------------------------------------------------- features

    @property
    def fitted_(self) -> bool:
        """True once `fit` has completed."""
        return bool(self._members)

    def features(self, stamps: NDArray[np.float64]) -> NDArray[np.float64]:
        """Feature matrix for a stack of stamps.

        Parameters
        ----------
        stamps: ``(n, p, p)`` or ``(p, p)`` subaperture intensities
            [photoelectrons].

        Returns
        -------
        ``(n, p*p + 1)`` float array. Stamps whose clipped total is zero give
        an all-zero shape block and a zero flux feature; the model then returns
        whatever it has learned for that degenerate input.
        """
        s = np.asarray(stamps, dtype=float)
        if s.ndim == 2:
            s = s[None, ...]
        p = self.array.pixels_per_sub
        if s.ndim != 3 or s.shape[1:] != (p, p):
            raise ValueError(
                f"stamps must have shape (n, {p}, {p}) for this LensletArray, got {s.shape}"
            )
        if not np.all(np.isfinite(s)):
            raise ValueError("stamps contain non-finite values (NaN or inf)")
        w = np.clip(s, 0.0, None).reshape(s.shape[0], -1)
        total = w.sum(axis=1)
        shape = np.zeros_like(w)
        ok = total > 0.0
        shape[ok] = w[ok] / total[ok, None]
        flux = np.log10(1.0 + total)[:, None]
        return np.hstack([shape, flux])

    # ------------------------------------------------------------------ fit

    def fit(self, stamps: NDArray[np.float64], slopes: NDArray[np.float64]) -> MLSlopeEstimator:
        """Train the ensemble.

        Parameters
        ----------
        stamps: ``(n, p, p)`` training stamps [photoelectrons].
        slopes: ``(n, 2)`` true slopes [rad].

        Returns
        -------
        self.
        """
        x = self.features(stamps)
        y = np.asarray(slopes, dtype=float)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"slopes must have shape (n, 2), got {y.shape}")
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"stamps and slopes disagree: {x.shape[0]} stamps vs {y.shape[0]} slopes"
            )
        if not np.all(np.isfinite(y)):
            raise ValueError("slopes contain non-finite values")
        if x.shape[0] < 10:
            raise ValueError(f"need at least 10 training samples, got {x.shape[0]}")

        target = self.array.slope_to_displacement(y)
        self._members = []
        for k in range(self.n_estimators):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
                random_state=self.random_state + k,
            )
            model.fit(x, target)
            self._members.append(model)
        return self

    # -------------------------------------------------------------- predict

    def predict(
        self, stamps: NDArray[np.float64], return_std: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict slopes [rad], optionally with the per-slope ensemble spread.

        Parameters
        ----------
        stamps: ``(n, p, p)`` or ``(p, p)`` stamps [photoelectrons].
        return_std: also return the per-component ensemble standard deviation
            [rad], shape ``(n, 2)``.

        Returns
        -------
        ``(n, 2)`` slopes [rad], or ``(slopes, std)``.

        Notes
        -----
        The spread is a *confidence indicator*, not a calibrated 1-sigma
        error: it measures how much the members disagree, and systematically
        under-states the true error where all members share the same bias.
        See `MODEL_CARD.md` §Uncertainty for the measured ratio.
        """
        if not self.fitted_:
            raise RuntimeError("MLSlopeEstimator.predict called before fit — call fit() first")
        x = self.features(stamps)
        preds = np.stack([m.predict(x) for m in self._members], axis=0)  # (K, n, 2) px
        mean_px = preds.mean(axis=0)
        slopes = self.array.displacement_to_slope(mean_px)
        if not return_std:
            return slopes
        std_px = preds.std(axis=0)
        return slopes, self.array.displacement_to_slope(std_px)

    def predict_frame(
        self, image: NDArray[np.float64], return_std: bool = False
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict the slope vector of a full detector frame.

        Convenience wrapper: cuts the frame into illuminated subapertures and
        calls `predict`. Returns ``(n_valid, 2)`` slopes [rad].
        """
        from .sensor import extract_subapertures

        return self.predict(extract_subapertures(image, self.array), return_std=return_std)
