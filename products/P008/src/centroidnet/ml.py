"""ML centroid estimator: ensemble of scikit-learn MLP regressors.

Deviation note: the product was originally specified with a small CNN;
PyTorch is not available in the build environment, so the model is an
ensemble of fully connected ``sklearn.neural_network.MLPRegressor`` networks
operating on the flattened, per-image-normalized pixel vector.  This is
recorded in README Limitations and MODEL_CARD.md.

Uncertainty output: the ensemble members differ only in their random
initialization/shuffling seed; the per-estimate standard deviation across
members (deep-ensemble spread in the sense of Lakshminarayanan, Pritzel &
Blundell, "Simple and scalable predictive uncertainty estimation using deep
ensembles", NeurIPS 2017) is returned by ``predict(..., return_std=True)``.
It captures model/initialization variance, not the full predictive
uncertainty, and is not calibrated.
"""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

__all__ = ["MLCentroider"]


class MLCentroider:
    """Ensemble MLP mapping a normalized pixel vector to (x, y) [pixels].

    Parameters
    ----------
    n_estimators : int
        Number of ensemble members, >= 2 (>= 5 recommended; default 5).
    hidden_layer_sizes : tuple of int
        MLP hidden-layer widths (default ``(64,)``).
    max_iter : int
        Maximum Adam epochs per member (default 300).
    random_state : int
        Base seed; member k uses ``random_state + k``.
    alpha : float
        L2 regularization strength (default 1e-4).

    Notes
    -----
    Inputs are preprocessed per image: negative pixels clipped to 0, then
    divided by the image sum (unit total flux), making the model invariant
    to overall gain.  Images with non-positive sum are rejected.
    """

    def __init__(
        self,
        n_estimators: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (64,),
        max_iter: int = 300,
        random_state: int = 0,
        alpha: float = 1e-4,
    ) -> None:
        if not isinstance(n_estimators, (int, np.integer)) or n_estimators < 2:
            raise ValueError(f"n_estimators must be an int >= 2, got {n_estimators!r}")
        self.n_estimators = int(n_estimators)
        self.hidden_layer_sizes = tuple(int(h) for h in hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.alpha = float(alpha)
        self._models: list[MLPRegressor] = []
        self._image_shape: tuple[int, int] | None = None

    # ------------------------------------------------------------------ #
    def _features(self, images: np.ndarray) -> np.ndarray:
        """Flatten and flux-normalize images -> (M, N*N) feature matrix."""
        arr = np.asarray(images, dtype=float)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(
                f"images must be (N, N) or (M, N, N), got array of shape {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("images contain NaN or Inf values")
        if self._image_shape is not None and arr.shape[1:] != self._image_shape:
            raise ValueError(
                f"image shape {arr.shape[1:]} does not match fitted shape {self._image_shape}"
            )
        flat = np.clip(arr, 0.0, None).reshape(arr.shape[0], -1)
        sums = flat.sum(axis=1, keepdims=True)
        if np.any(sums <= 0.0):
            raise ValueError("every image must have positive total intensity after clipping")
        return flat / sums

    # ------------------------------------------------------------------ #
    def fit(self, images: np.ndarray, positions: np.ndarray) -> "MLCentroider":
        """Train the ensemble.

        Parameters
        ----------
        images : ndarray, shape (M, N, N)
            Training frames [linear intensity units].
        positions : ndarray, shape (M, 2)
            True (x, y) centroids [pixels] from the array centre.

        Returns
        -------
        MLCentroider
            ``self`` (fitted).
        """
        arr = np.asarray(images, dtype=float)
        if arr.ndim != 3:
            raise ValueError(f"images must have shape (M, N, N), got {arr.shape}")
        pos = np.asarray(positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 2 or pos.shape[0] != arr.shape[0]:
            raise ValueError(
                f"positions must have shape ({arr.shape[0]}, 2), got {pos.shape}"
            )
        if not np.all(np.isfinite(pos)):
            raise ValueError("positions must be finite")
        self._image_shape = arr.shape[1:]
        feats = self._features(arr)
        self._models = []
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
            model.fit(feats, pos)
            self._models.append(model)
        return self

    # ------------------------------------------------------------------ #
    def predict(
        self, images: np.ndarray, return_std: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict (x, y) centroids [pixels]; optionally ensemble spread.

        Parameters
        ----------
        images : ndarray, shape (M, N, N) or (N, N)
            Frames to evaluate.
        return_std : bool
            If True, also return the per-estimate standard deviation across
            ensemble members [pixels] (shape (M, 2)) as an uncertainty
            proxy.

        Returns
        -------
        mean : ndarray, shape (M, 2)
            Ensemble-mean (x, y) [pixels].
        std : ndarray, shape (M, 2), only if ``return_std=True``
            Ensemble standard deviation [pixels].
        """
        if not self._models:
            raise RuntimeError("MLCentroider is not fitted; call fit() first")
        feats = self._features(images)
        preds = np.stack([m.predict(feats) for m in self._models])  # (K, M, 2)
        mean = preds.mean(axis=0)
        if return_std:
            return mean, preds.std(axis=0)
        return mean
