"""Learned null-motion policy and its feature map.

The classical null-motion policies in :mod:`cmgsteer.nullmotion` are greedy:
they climb the singularity measure now, with no notion of what the array will
be asked to do next.  The policy here is trained to imitate a short-horizon
lookahead oracle that *does* see the next few seconds of commanded torque and
picks the null-motion coefficient that minimises the momentum error accumulated
over that horizon.  At run time the policy sees only the current state, so it
is a behaviour-cloned approximation of that oracle.

Scope and shape
---------------
For a four-CMG array away from a singularity ``null(A)`` is one-dimensional, so
the entire null motion is one signed scalar ``k`` times the unit null vector
``n_hat``, whose sign is fixed by :func:`cmgsteer.nullmotion.unit_null_vector`
so that ``k > 0`` increases the singularity measure.  The policy therefore
predicts a single number in ``[-1, 1]``, scaled by ``max_null_rate`` [rad/s].
``k = 0`` reproduces the plain steering law and a constant ``k > 0`` is a crude
version of gradient null motion, so the interesting predictions are the ones
that are neither.

Uncertainty: the model is an ensemble of scikit-learn multi-layer perceptrons.
The spread across members is reported per prediction and mapped to a scalar
confidence in ``[0, 1]``, calibrated against the mean spread on the training
set.  Confidence 1 means the members agree as well as they did in training.

**This model is not certified for operational flight use.**  See
``MODEL_CARD.md`` for the measured comparison against the classical policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .arrays import CMGArray
from .nullmotion import NullMotionPolicy, unit_null_vector
from .singularity import manipulability_gradient

__all__ = [
    "LearnedNullMotion",
    "NullMotionAction",
    "feature_names",
    "policy_features",
]

DEFAULT_MAX_NULL_RATE = 0.5


def feature_names(n_free: int) -> tuple[str, ...]:
    """Names of the features produced by :func:`policy_features`, in order."""
    names: list[str] = []
    for i in range(n_free):
        names.append(f"sin_delta{i + 1}")
        names.append(f"cos_delta{i + 1}")
    names += ["hx_norm", "hy_norm", "hz_norm", "h_mag_norm"]
    names += ["measure_norm", "sigma_min_norm"]
    names += ["tau_x_hat", "tau_y_hat", "tau_z_hat", "tau_mag_norm"]
    names += ["null_dot_grad_hat", "singular_dir_dot_tau_hat"]
    return tuple(names)


def policy_features(
    array: CMGArray, deltas: ArrayLike, torque: ArrayLike
) -> NDArray[np.float64]:
    """Feature vector for the learned null-motion policy.

    All entries are dimensionless.  The layout is given by
    :func:`feature_names`:

    * ``sin``/``cos`` of every free gimbal angle (the wrap-free encoding),
    * array momentum divided by the total momentum capacity, and its magnitude,
    * singularity measure divided by ``h0_mean^3`` and ``sigma_min`` divided by
      ``h0_mean``,
    * the unit commanded torque and its magnitude divided by ``h0_mean``
      (which converts a torque into a gimbal-rate scale),
    * the cosine between the unit null vector and ``grad(m)`` — the signal the
      classical gradient policy acts on — and the cosine between the singular
      direction and the commanded torque, which says whether the direction the
      array is losing is the direction it is being asked to push.

    Returns a ``(2 n_free + 10,)`` float array.
    """
    d = np.asarray(deltas, dtype=float).reshape(-1)
    t = np.asarray(torque, dtype=float).reshape(-1)
    if t.shape != (3,):
        raise ValueError(f"torque must have shape (3,), got {t.shape}")
    idx = array.free_indices
    h0 = float(np.mean(array.rotor_momenta))
    cap = array.total_momentum_capacity

    jac = array.jacobian(d)
    u_mat, sv, vt = np.linalg.svd(jac)
    measure = float(np.prod(sv))
    h = array.momentum(d)
    t_norm = float(np.linalg.norm(t))
    t_hat = t / t_norm if t_norm > 0.0 else np.zeros(3)

    grad = manipulability_gradient(array, d)
    gnorm = float(np.linalg.norm(grad))
    try:
        null_vec = unit_null_vector(array, d)
    except ValueError:
        cos_ng = 0.0
    else:
        # unit_null_vector fixes the sign so that null_vec . grad >= 0, so the
        # cosine below is in [0, 1] and matches the sign convention of the action.
        cos_ng = float(null_vec @ grad) / gnorm if gnorm > 0.0 else 0.0

    feats = np.empty(2 * idx.size + 10)
    feats[0 : 2 * idx.size : 2] = np.sin(d[idx])
    feats[1 : 2 * idx.size : 2] = np.cos(d[idx])
    base = 2 * idx.size
    feats[base : base + 3] = h / cap
    feats[base + 3] = float(np.linalg.norm(h)) / cap
    feats[base + 4] = measure / h0**3
    feats[base + 5] = float(sv[-1]) / h0
    feats[base + 6 : base + 9] = t_hat
    feats[base + 9] = t_norm / h0
    return np.concatenate([feats, [cos_ng, float(abs(u_mat[:, -1] @ t_hat))]])


@dataclass(frozen=True)
class NullMotionAction:
    """One policy decision.

    Attributes
    ----------
    coefficient
        Predicted null-motion coefficient ``k`` in ``[-1, 1]``.
    std
        Standard deviation of ``k`` across the ensemble members.
    confidence
        ``exp(-std / reference_spread)`` in ``[0, 1]``, where
        ``reference_spread`` is the mean ensemble spread on the training set.
    rates
        The resulting null-motion gimbal rates [rad/s], length ``n_free``.
    """

    coefficient: float
    std: float
    confidence: float
    rates: NDArray[np.float64]


@dataclass
class LearnedNullMotion(NullMotionPolicy):
    """Ensemble-MLP null-motion policy with a confidence output.

    Parameters
    ----------
    max_null_rate
        The null-motion rate [rad/s] that a coefficient of 1 corresponds to.
    n_estimators
        Number of MLP members; the spread across them is the uncertainty.
    hidden_layer_sizes, alpha, max_iter, random_state
        Passed to :class:`sklearn.neural_network.MLPRegressor`; member ``k``
        uses ``random_state + k``.
    confidence_floor
        Predictions whose confidence falls below this value fall back to the
        classical gradient direction (``k = +1``) instead of the network's
        output.  ``None`` disables the fallback, which is the default so that
        the raw model is what gets benchmarked.
    """

    max_null_rate: float = DEFAULT_MAX_NULL_RATE
    n_estimators: int = 5
    hidden_layer_sizes: tuple[int, ...] = (64, 32)
    alpha: float = 1e-4
    max_iter: int = 400
    random_state: int = 0
    confidence_floor: float | None = None
    name: str = "learned"
    _members: list[MLPRegressor] = field(default_factory=list, repr=False)
    _scaler: StandardScaler | None = field(default=None, repr=False)
    _reference_spread: float = field(default=1.0, repr=False)
    _n_features: int = field(default=0, repr=False)

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return bool(self._members)

    def fit(self, features: ArrayLike, coefficients: ArrayLike) -> LearnedNullMotion:
        """Train the ensemble on ``(features, coefficient)`` pairs.

        Parameters
        ----------
        features
            ``(n_samples, n_features)`` from :func:`policy_features`.
        coefficients
            ``(n_samples,)`` target null-motion coefficients in ``[-1, 1]``.
        """
        x = np.atleast_2d(np.asarray(features, dtype=float))
        y = np.asarray(coefficients, dtype=float).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"features has {x.shape[0]} rows but coefficients has {y.shape[0]}"
            )
        if x.shape[0] < self.n_estimators:
            raise ValueError(
                f"need at least {self.n_estimators} samples to fit {self.n_estimators} members, "
                f"got {x.shape[0]}"
            )
        if self.max_null_rate <= 0.0:
            raise ValueError(f"max_null_rate must be positive, got {self.max_null_rate}")
        scaler = StandardScaler().fit(x)
        xs = scaler.transform(x)
        members = []
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
            members.append(model)
        preds = np.column_stack([m.predict(xs) for m in members])
        spread = float(np.mean(np.std(preds, axis=1)))
        self._members = members
        self._scaler = scaler
        self._reference_spread = max(spread, 1e-9)
        self._n_features = int(x.shape[1])
        return self

    def predict(self, features: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Batch prediction: ``(coefficients, ensemble_std)``, both ``(n_samples,)``."""
        if not self.fitted or self._scaler is None:
            raise RuntimeError("LearnedNullMotion.predict called before fit")
        x = np.atleast_2d(np.asarray(features, dtype=float))
        if x.shape[1] != self._n_features:
            raise ValueError(f"expected {self._n_features} features, got {x.shape[1]}")
        xs = self._scaler.transform(x)
        preds = np.column_stack([m.predict(xs) for m in self._members])
        return np.clip(preds.mean(axis=1), -1.0, 1.0), preds.std(axis=1)

    def confidence(self, std: ArrayLike) -> NDArray[np.float64]:
        """Map ensemble spread to a confidence in ``[0, 1]``."""
        s = np.asarray(std, dtype=float)
        return np.exp(-s / self._reference_spread)

    def act(
        self, array: CMGArray, deltas: ArrayLike, torque: ArrayLike
    ) -> NullMotionAction:
        """Full decision for one state, including uncertainty and gimbal rates."""
        d = np.asarray(deltas, dtype=float).reshape(-1)
        feats = policy_features(array, d, torque)
        coeff, std = self.predict(feats[None, :])
        conf = float(self.confidence(std)[0])
        k = float(coeff[0])
        if self.confidence_floor is not None and conf < self.confidence_floor:
            k = 1.0
        try:
            vec = unit_null_vector(array, d)
        except ValueError:
            return NullMotionAction(k, float(std[0]), conf, np.zeros(array.n_free))
        return NullMotionAction(k, float(std[0]), conf, self.max_null_rate * k * vec)

    def rates(
        self,
        array: CMGArray,
        deltas: ArrayLike,
        torque: ArrayLike,
        time: float = 0.0,
    ) -> NDArray[np.float64]:
        """Null-motion gimbal rates [rad/s] for the current state."""
        del time
        return self.act(array, deltas, torque).rates
