"""Learned slopes-to-Zernike reconstructor with an ensemble uncertainty output.

Implemented and trained only after the classical baseline
(`wavelab.modal.ModalReconstructor`) exists and is validated
(`validation/VALIDATION.md` §1-2), per the mission rule that the classical
reconstructor comes first. Every reported ML number in `MODEL_CARD.md` and
`validation/VALIDATION.md` is measured against `ModalReconstructor` on
identical held-out batches from `wavelab.dataset.generate_batch`.

Architecture: a small ensemble of independently seeded
`sklearn.neural_network.MLPRegressor` networks (PyTorch is not available in
this build environment, so a fully connected ensemble is used rather than a
convolutional or graph-structured model -- see README "Limitations"). The
ensemble mean is the point estimate; the per-coefficient ensemble standard
deviation is the uncertainty output. Deep ensembles as an uncertainty proxy:
Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017), "Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles", *NeurIPS 30*. As with
ShackSim's identical use of this technique, the spread measures *disagreement
between members*, which is not automatically a calibrated 1-sigma error bar;
the measured calibration ratio is reported in `MODEL_CARD.md`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.neural_network import MLPRegressor

__all__ = ["ZernikeSlopeEnsemble"]


class ZernikeSlopeEnsemble:
    """Ensemble regressor from a (possibly dropout-masked) slope vector to Zernike coefficients.

    Parameters
    ----------
    n_sub: number of subapertures (fixes the expected input size).
    n_modes: number of Zernike coefficients to predict.
    n_estimators: ensemble members, ``>= 2``.
    hidden_layer_sizes: MLP hidden widths, passed to `MLPRegressor`.
    alpha: L2 regularization strength.
    max_iter: maximum Adam epochs.
    random_state: base seed; member ``k`` uses ``random_state + k``.

    Notes
    -----
    Input features, per sample (`features`): the ``2 * n_sub`` slope values
    (zero at dropped-out subapertures) concatenated with the ``n_sub``-length
    active/dropout mask (1.0 = active, 0.0 = dropped). The mask is included
    explicitly so the network can, in principle, learn to distinguish "slope
    measured as zero" from "slope not measured" -- without it those two cases
    would be indistinguishable inputs with very different meanings.
    """

    def __init__(
        self,
        n_sub: int,
        n_modes: int,
        n_estimators: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        alpha: float = 1e-3,
        max_iter: int = 300,
        random_state: int = 0,
    ) -> None:
        if isinstance(n_sub, bool) or not isinstance(n_sub, (int, np.integer)) or n_sub < 1:
            raise ValueError(f"n_sub must be a positive integer, got {n_sub!r}")
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)) or n_modes < 1:
            raise ValueError(f"n_modes must be a positive integer, got {n_modes!r}")
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, (int, np.integer)):
            raise TypeError(f"n_estimators must be an int, got {type(n_estimators).__name__}")
        if n_estimators < 2:
            raise ValueError(f"n_estimators must be >= 2 for an ensemble spread, got {n_estimators}")
        self.n_sub = int(n_sub)
        self.n_modes = int(n_modes)
        self.n_estimators = int(n_estimators)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._members: list[MLPRegressor] = []

    @property
    def fitted_(self) -> bool:
        """True once `fit` has completed."""
        return bool(self._members)

    def features(
        self, slopes: NDArray[np.float64], active: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Build the ``(n, 3 * n_sub)`` feature matrix from slopes and the active mask.

        Parameters
        ----------
        slopes: ``(n, 2 * n_sub)`` or ``(2 * n_sub,)``.
        active: ``(n, n_sub)`` or ``(n_sub,)`` bool.
        """
        s = np.asarray(slopes, dtype=np.float64)
        a = np.asarray(active, dtype=bool)
        if s.ndim == 1:
            s = s[None, :]
        if a.ndim == 1:
            a = a[None, :]
        if s.shape[1] != 2 * self.n_sub:
            raise ValueError(f"slopes must have {2 * self.n_sub} columns, got {s.shape[1]}")
        if a.shape[1] != self.n_sub:
            raise ValueError(f"active must have {self.n_sub} columns, got {a.shape[1]}")
        if s.shape[0] != a.shape[0]:
            raise ValueError(f"slopes and active row counts disagree: {s.shape[0]} vs {a.shape[0]}")
        if not np.all(np.isfinite(s)):
            raise ValueError("slopes contain non-finite values")
        return np.hstack([s, a.astype(np.float64)])

    def fit(
        self,
        slopes: NDArray[np.float64],
        active: NDArray[np.bool_],
        coeffs: NDArray[np.float64],
    ) -> ZernikeSlopeEnsemble:
        """Train the ensemble.

        Parameters
        ----------
        slopes: ``(n, 2 * n_sub)`` training slopes.
        active: ``(n, n_sub)`` dropout masks.
        coeffs: ``(n, n_modes)`` ground-truth Noll coefficients [rad].

        Returns
        -------
        self.
        """
        x = self.features(slopes, active)
        y = np.asarray(coeffs, dtype=np.float64)
        if y.ndim != 2 or y.shape[1] != self.n_modes:
            raise ValueError(f"coeffs must have shape (n, {self.n_modes}), got {y.shape}")
        if y.shape[0] != x.shape[0]:
            raise ValueError(f"{x.shape[0]} feature rows vs {y.shape[0]} coefficient rows")
        if not np.all(np.isfinite(y)):
            raise ValueError("coeffs contain non-finite values")
        if x.shape[0] < 10:
            raise ValueError(f"need at least 10 training samples, got {x.shape[0]}")

        self._members = []
        for k in range(self.n_estimators):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=self.alpha,
                max_iter=self.max_iter,
                early_stopping=True,
                n_iter_no_change=12,
                validation_fraction=0.1,
                random_state=self.random_state + k,
            )
            model.fit(x, y)
            self._members.append(model)
        return self

    def predict(
        self,
        slopes: NDArray[np.float64],
        active: NDArray[np.bool_],
        return_std: bool = False,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Predict Zernike coefficients, optionally with the per-coefficient ensemble spread.

        Parameters
        ----------
        slopes: ``(n, 2 * n_sub)`` or ``(2 * n_sub,)``.
        active: ``(n, n_sub)`` or ``(n_sub,)``.
        return_std: also return the per-coefficient ensemble standard
            deviation [rad].

        Returns
        -------
        ``(n, n_modes)`` coefficients, or ``(coeffs, std)``.

        Notes
        -----
        The spread is a confidence *indicator*, not a calibrated 1-sigma
        error bar -- see `MODEL_CARD.md` "Uncertainty / confidence output"
        for the measured ratio to the true error.
        """
        if not self.fitted_:
            raise RuntimeError("ZernikeSlopeEnsemble.predict called before fit — call fit() first")
        x = self.features(slopes, active)
        preds = np.stack([m.predict(x) for m in self._members], axis=0)  # (K, n, n_modes)
        mean = preds.mean(axis=0)
        if not return_std:
            return mean
        return mean, preds.std(axis=0)
