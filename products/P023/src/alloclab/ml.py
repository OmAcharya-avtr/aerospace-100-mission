"""Learned allocator: an ensemble regressor trained to imitate the QP.

This is the AI element of the package, and it exists to be measured against
the exact QP, not to replace it. The classical allocators in
:mod:`alloclab.allocation` were implemented and validated first; this model is
trained on their output.

Architecture is a small ensemble of scikit-learn ``MLPRegressor`` networks.
There is no PyTorch in the target environment and only two CPU cores, so a
deep or convolutional model was never an option; see ``MODEL_CARD.md`` for the
compute budget.

The honest expectation, and the measured result, is a speed-for-exactness
trade: the network returns a command in microseconds but neither meets the
torque exactly nor respects the actuator bounds. :meth:`LearnedAllocator.predict`
therefore returns the raw network output by default and offers ``clip=True``
as an explicit, opt-in projection into the box -- clipping changes the achieved
torque and the caller has to be told, rather than having the violation
silently hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .effectors import EffectorSet

__all__ = ["LearnedAllocator", "LearnedAllocation"]


@dataclass
class LearnedAllocation:
    """Batch output of :meth:`LearnedAllocator.predict`.

    Attributes
    ----------
    commands
        ``(n, m)`` predicted commands [command units].
    std
        ``(n, m)`` per-effector standard deviation across the ensemble
        members [command units]. This is an ensemble *spread*, not a
        calibrated 1-sigma error bar; see ``MODEL_CARD.md`` sec. 8.
    confidence
        ``(n,)`` scalar confidence in ``[0, 1]``, defined as
        ``exp(-mean_i std_i / reference_spread)`` where ``reference_spread``
        is fixed at fit time as the mean ensemble spread on the training set.
        1 means the members agree as well as they did in training, 0 means
        they disagree far more.
    clipped
        Whether ``commands`` was projected into the command box.
    """

    commands: np.ndarray
    std: np.ndarray
    confidence: np.ndarray
    clipped: bool


class LearnedAllocator:
    """Ensemble MLP trained to reproduce the QP allocation.

    Parameters
    ----------
    eset
        The effector set the model is trained for. The model is tied to this
        configuration: the effectiveness matrix is baked into the training
        labels, and applying the model to a different geometry with the same
        shapes returns an unflagged wrong answer.
    n_estimators
        Number of MLPs in the ensemble. Member ``k`` uses ``random_state + k``.
    hidden_layer_sizes, alpha, max_iter, random_state
        Passed to ``sklearn.neural_network.MLPRegressor``.

    Features are ``[tau / torque_scale (3), health (m)]``; the health mask is
    included so that "this effector is failed" is representable rather than
    inferred from a zero command. Targets are commands normalised to
    ``[0, 1]`` across each effector's own span.
    """

    def __init__(
        self,
        eset: EffectorSet,
        n_estimators: int = 5,
        hidden_layer_sizes: tuple[int, ...] = (96, 64),
        alpha: float = 1e-4,
        max_iter: int = 400,
        random_state: int = 0,
    ) -> None:
        if n_estimators < 2:
            raise ValueError(
                f"n_estimators must be >= 2 for an ensemble spread, got {n_estimators}"
            )
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.eset = eset
        self.n_estimators = int(n_estimators)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._models: list[MLPRegressor] = []
        self._scaler: StandardScaler | None = None
        self._torque_scale: float = 1.0
        self._reference_spread: float = 1.0
        self._span = np.where(eset.span > 0.0, eset.span, 1.0)

    # -- feature / target transforms -------------------------------------

    @property
    def fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return bool(self._models)

    @property
    def n_effectors(self) -> int:
        """Number of effectors the model outputs commands for."""
        return self.eset.n_effectors

    def features(self, torques: ArrayLike, health: ArrayLike) -> np.ndarray:
        """Build the ``(n, 3 + m)`` feature matrix."""
        tau = np.atleast_2d(np.asarray(torques, dtype=float))
        h = np.atleast_2d(np.asarray(health, dtype=float))
        if tau.shape[1] != 3:
            raise ValueError(f"torques must have 3 columns, got {tau.shape[1]}")
        if h.shape[1] != self.n_effectors:
            raise ValueError(
                f"health must have {self.n_effectors} columns, got {h.shape[1]}"
            )
        if tau.shape[0] != h.shape[0]:
            raise ValueError(
                f"torques and health must have the same number of rows, "
                f"got {tau.shape[0]} and {h.shape[0]}"
            )
        return np.hstack([tau / self._torque_scale, h])

    def _to_unit(self, commands: np.ndarray) -> np.ndarray:
        return (commands - self.eset.lower) / self._span

    def _from_unit(self, unit: np.ndarray) -> np.ndarray:
        return self.eset.lower + unit * self._span

    # -- training ---------------------------------------------------------

    def fit(
        self,
        torques: ArrayLike,
        health: ArrayLike,
        commands: ArrayLike,
        torque_scale: float | None = None,
    ) -> LearnedAllocator:
        """Train every ensemble member on the same data with different seeds.

        Parameters
        ----------
        torques, health, commands
            ``(n, 3)``, ``(n, m)``, ``(n, m)`` arrays, normally straight from
            :func:`alloclab.dataset.generate_dataset`.
        torque_scale
            Input normalisation [N*m]. Defaults to the largest commanded
            torque magnitude in the training set.

        Returns ``self``.
        """
        tau = np.atleast_2d(np.asarray(torques, dtype=float))
        h = np.atleast_2d(np.asarray(health, dtype=float))
        u = np.atleast_2d(np.asarray(commands, dtype=float))
        if u.shape != h.shape:
            raise ValueError(f"commands shape {u.shape} must match health shape {h.shape}")
        if tau.shape[0] < 10:
            raise ValueError(f"need at least 10 training samples, got {tau.shape[0]}")

        if torque_scale is None:
            mags = np.linalg.norm(tau, axis=1)
            self._torque_scale = float(np.max(mags)) if np.max(mags) > 0.0 else 1.0
        else:
            if torque_scale <= 0.0:
                raise ValueError(f"torque_scale must be > 0, got {torque_scale}")
            self._torque_scale = float(torque_scale)

        x = self.features(tau, h)
        self._scaler = StandardScaler().fit(x)
        xs = self._scaler.transform(x)
        y = self._to_unit(u)

        self._models = []
        for k in range(self.n_estimators):
            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state + k,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
            )
            model.fit(xs, y)
            self._models.append(model)

        spreads = np.stack([m.predict(xs) for m in self._models], axis=0).std(axis=0)
        ref = float(np.mean(spreads * self._span))
        self._reference_spread = ref if ref > 0.0 else 1.0
        return self

    # -- inference --------------------------------------------------------

    def predict(
        self, torques: ArrayLike, health: ArrayLike, clip: bool = False
    ) -> LearnedAllocation:
        """Predict commands for a batch of (torque, health) pairs.

        ``clip=False`` (default) returns the raw network output, which may lie
        outside the command box. ``clip=True`` projects it into the box, which
        removes the bound violation but changes the achieved torque; the
        returned :class:`LearnedAllocation` records which was done.

        Raises
        ------
        RuntimeError
            If called before :meth:`fit`.
        """
        if not self.fitted or self._scaler is None:
            raise RuntimeError("LearnedAllocator.predict called before fit")
        x = self._scaler.transform(self.features(torques, health))
        preds = np.stack([m.predict(x) for m in self._models], axis=0)
        mean_unit = preds.mean(axis=0)
        std = preds.std(axis=0) * self._span
        commands = self._from_unit(mean_unit)
        if clip:
            commands = self.eset.clip(commands)
        conf = np.exp(-std.mean(axis=1) / self._reference_spread)
        return LearnedAllocation(
            commands=commands, std=std, confidence=conf, clipped=bool(clip)
        )
