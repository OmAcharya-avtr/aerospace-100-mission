"""The learned residual (discrepancy) model, with uncertainty.

This is the only learned component in the package and it is implemented last on
purpose. The physics fit of :mod:`twincalib.fitting` is the baseline, it exists
and is measured before this module is used, and every number this module
produces is reported beside the physics-only number on the same held-out data.

What it is
----------
After the physics parameters have been fitted, the remaining difference between
the measurements and the model is

``r_i = y_i - f_i(theta_hat)``

and it contains three things that are not separable from the data alone:
measurement noise, parameter error, and *model-form error* -- structure the
calibration model cannot produce at any parameter value. A discrepancy model
learns a function ``g(x_i)`` of features available from the physics run alone,
and the twin's prediction becomes ``f_i(theta_hat) + g(x_i)``. This is the
Kennedy and O'Hagan (2001) model-discrepancy construction, in its simplest
plug-in form: the physics is fitted first, the discrepancy is learned on the
residuals, and the two are not fitted jointly.

The estimator is a Gaussian-process regression (Rasmussen and Williams 2006)
with a constant-times-squared-exponential kernel plus a white-noise term. It is
chosen over a forest or a boosted ensemble for one reason: it returns a
predictive standard deviation that is a genuine posterior quantity for its own
assumptions, so the uncertainty output of the model is not an afterthought. The
calibration of that standard deviation is measured on held-out trajectories in
``validation/validate_residual_output.txt`` and is reported there whether or not
it is good.

Features
--------
Everything in the feature vector is computable from the physics run and the
known inputs, never from the measurement being predicted:

=====================  ==========================================================
feature                meaning and unit
=====================  ==========================================================
``T1_model``           physics-predicted unit temperature, K, centred and scaled
``T2_model``           physics-predicted panel temperature, K, centred and scaled
``dT_model``           ``T1_model - T2_model``, K: the interface temperature
                       difference, which is what the declared discrepancy acts on
``dissipation``        measured unit dissipation at that sample, W
``sun``                illumination fraction, dimensionless
``channel``            which thermometer the sample belongs to, one-hot
=====================  ==========================================================

Features are standardised by the training set's own mean and standard deviation,
and the scaler is stored with the model so that a held-out trajectory is
transformed with the training statistics and not with its own.

The risk this module carries
----------------------------
A flexible discrepancy model can improve the fit by absorbing error that is
really parameter error, and in doing so it can make the physical parameters
*worse* while making the predictions better. Brynjarsdottir and O'Hagan (2014)
show this plainly: a discrepancy term can allow a physical parameter to be
estimated far from its true value with high confidence. The package does not
mention this risk and move on; :mod:`twincalib.absorption` measures it, and the
measurement is reported in the README, in ``VALIDATION.md`` and in
``MODEL_CARD.md``.

References
----------
Kennedy, M. C. and O'Hagan, A. (2001), "Bayesian calibration of computer
models", *Journal of the Royal Statistical Society Series B* 63(3), 425-464.

Brynjarsdottir, J. and O'Hagan, A. (2014), "Learning about physical parameters:
the importance of model discrepancy", *Inverse Problems* 30(11), 114007.

Rasmussen, C. E. and Williams, C. K. I., *Gaussian Processes for Machine
Learning*, MIT Press, 2006. Chapter 2 for the predictive mean and variance,
Chapter 4 for the squared-exponential kernel, Chapter 5 for marginal-likelihood
hyperparameter selection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from .fitting import CalibrationProblem
from .model import Inputs, TwinModel

__all__ = [
    "FEATURE_NAMES",
    "ResidualModel",
    "ResidualPrediction",
    "build_features",
    "residual_targets",
]

FEATURE_NAMES: tuple[str, ...] = (
    "T1_model",
    "T2_model",
    "dT_model",
    "dissipation",
    "sun",
    "is_channel_T2",
)
"""Feature names in column order. Units are given in the module docstring."""


def build_features(
    model: TwinModel,
    theta: Sequence[float],
    inputs: Inputs,
    initial_state: Sequence[float],
    channels: Sequence[int],
    sample_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Features and physics prediction for one trajectory.

    Parameters
    ----------
    model, theta, inputs, initial_state, channels, sample_stride:
        The physics run, exactly as the fitter sees it.

    Returns
    -------
    features:
        Shape ``(n_obs, len(FEATURE_NAMES))``. Row order matches the
        channel-major observation vector.
    prediction:
        Physics prediction of the observations, K, shape ``(n_obs,)``.

    Notes
    -----
    Nothing here touches the measurements, by construction: the function does
    not take them. That is what makes it usable on a held-out trajectory and on
    a real one.
    """
    states = model.simulate(theta, inputs, initial_state)
    kept = states[::sample_stride, :]
    p_dis = inputs.dissipation[::sample_stride]
    sun = inputs.sun[::sample_stride]
    n = kept.shape[0]
    blocks = []
    preds = []
    for c in channels:
        if c < 0 or c >= kept.shape[1]:
            raise ValueError(f"channel {c} out of range for this model")
        block = np.column_stack(
            [
                kept[:, 0],
                kept[:, 1] if kept.shape[1] > 1 else kept[:, 0],
                (kept[:, 0] - kept[:, 1]) if kept.shape[1] > 1 else np.zeros(n),
                p_dis,
                sun,
                np.full(n, 1.0 if c == 1 else 0.0),
            ]
        )
        blocks.append(block)
        preds.append(kept[:, c])
    return np.vstack(blocks), np.concatenate(preds)


@dataclass(frozen=True)
class ResidualPrediction:
    """A residual-model prediction with its uncertainty.

    Attributes
    ----------
    mean:
        Predicted discrepancy, K, shape ``(n,)``. Add it to the physics
        prediction.
    std:
        Predictive standard deviation, K, shape ``(n,)``. This is the Gaussian
        process posterior standard deviation *including* its learned white-noise
        term, so it is a predictive interval for a new measurement and not a
        confidence interval for the mean function.
    """

    mean: np.ndarray
    std: np.ndarray

    def interval(self, z: float = 1.959963984540054) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper) = mean +- z * std``, K. Default ``z`` is 95 per cent."""
        if not np.isfinite(z) or z <= 0.0:
            raise ValueError("z must be finite and positive")
        return (self.mean - z * self.std, self.mean + z * self.std)


class ResidualModel:
    """Gaussian-process regression of the physics residual on physics features.

    Parameters
    ----------
    length_scale:
        Initial squared-exponential length scale in standardised feature units,
        dimensionless. Optimised by marginal likelihood from this start.
    noise_level:
        Initial white-noise variance in K^2. A sensible start is the square of
        the thermometer noise.
    n_restarts:
        Restarts of the marginal-likelihood optimiser, each from a random point
        in the bounds. More restarts cost linearly and the fit is already
        seconds, so the default is modest.
    random_state:
        Seed for the restarts, so that a fit is reproducible.
    max_training_points:
        Cap on the training-set size. A Gaussian process costs ``O(n**3)`` to
        fit and ``O(n**2)`` per prediction, so an uncapped training set on a
        multi-trajectory design would dominate the whole package's run time.
        When the training set is larger, an evenly spaced subset is taken --
        evenly spaced and not random, so that the time coverage of every
        trajectory is preserved -- and ``n_training_used`` records what was
        actually used.
    """

    def __init__(
        self,
        length_scale: float = 1.0,
        noise_level: float = 0.09,
        n_restarts: int = 1,
        random_state: int = 0,
        max_training_points: int = 600,
    ) -> None:
        if not np.isfinite(length_scale) or length_scale <= 0.0:
            raise ValueError("length_scale must be finite and positive")
        if not np.isfinite(noise_level) or noise_level <= 0.0:
            raise ValueError("noise_level must be finite and positive")
        if n_restarts < 0:
            raise ValueError("n_restarts must be >= 0")
        if max_training_points < 10:
            raise ValueError("max_training_points must be at least 10")
        self.length_scale = float(length_scale)
        self.noise_level = float(noise_level)
        self.n_restarts = int(n_restarts)
        self.random_state = int(random_state)
        self.max_training_points = int(max_training_points)
        self._gp: GaussianProcessRegressor | None = None
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self.n_training_used: int = 0
        self.n_training_offered: int = 0

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._gp is not None

    @property
    def kernel_description(self) -> str:
        """The fitted kernel as a string, or a message saying it is not fitted."""
        if self._gp is None:
            return "not fitted"
        return str(self._gp.kernel_)

    def _standardise(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None or self._scale is None:
            raise RuntimeError("the model is not fitted")
        return (features - self._mean) / self._scale

    def fit(self, features: np.ndarray, residual: np.ndarray) -> ResidualModel:
        """Fit on training features and residuals.

        Parameters
        ----------
        features:
            Shape ``(n, n_features)``, as returned by :func:`build_features`.
        residual:
            Measured minus physics-predicted, K, shape ``(n,)``.

        Returns
        -------
        ResidualModel
            ``self``, so the call can be chained.

        Raises
        ------
        ValueError
            On shape mismatch, non-finite entries, or fewer than 10 samples.
        """
        x = np.asarray(features, dtype=float)
        y = np.asarray(residual, dtype=float).ravel()
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}"
            )
        if y.size != x.shape[0]:
            raise ValueError("features and residual must have the same number of rows")
        if y.size < 10:
            raise ValueError("at least 10 training samples are required")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("features and residual must be finite")
        self.n_training_offered = int(y.size)
        if y.size > self.max_training_points:
            take = np.linspace(0, y.size - 1, self.max_training_points).round().astype(int)
            take = np.unique(take)
            x = x[take]
            y = y[take]
        self.n_training_used = int(y.size)
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A feature that is constant over the training set carries no
        # information; dividing by its zero standard deviation would produce
        # NaN, so the scale is set to 1 and the centred column stays at zero.
        self._scale = np.where(scale > 1e-12, scale, 1.0)
        # The hyperparameter bounds are deliberately wide. scikit-learn warns
        # when a fitted hyperparameter lands near a bound, and on the shipped
        # designs the amplitude legitimately collapses towards zero when there
        # is no discrepancy to learn -- that is a result, not a fitting problem,
        # so the bound is placed far enough out that the result is not clipped.
        kernel = ConstantKernel(1.0, (1e-10, 1e6)) * RBF(
            self.length_scale, (1e-3, 1e4)
        ) + WhiteKernel(self.noise_level, (1e-8, 1e4))
        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state,
        )
        gp.fit(self._standardise(x), y)
        self._gp = gp
        return self

    def predict(self, features: np.ndarray) -> ResidualPrediction:
        """Predict the discrepancy and its standard deviation, both in K.

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        if self._gp is None:
            raise RuntimeError("fit the model before calling predict")
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"features must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}"
            )
        mean, std = self._gp.predict(self._standardise(x), return_std=True)
        return ResidualPrediction(
            mean=np.asarray(mean, dtype=float), std=np.asarray(std, dtype=float)
        )

    def training_residual_scale(self) -> float:
        """Standard deviation of the training targets, K. 0 if not fitted."""
        if self._gp is None:
            return 0.0
        return float(np.std(self._gp.y_train_))


def residual_targets(problem: CalibrationProblem, theta: Sequence[float]) -> np.ndarray:
    """Unweighted residuals ``y - f(theta)``, K, shape ``(n_obs,)``.

    The training target of the residual model. Unweighted, because the residual
    model predicts a temperature, not a standardised quantity.
    """
    return problem.y - problem.predict(theta)
