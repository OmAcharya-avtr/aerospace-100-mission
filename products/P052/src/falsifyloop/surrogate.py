r"""The learned component: a random-forest surrogate of requirement robustness,
with a dispersion estimate it reports alongside every prediction.

What it models
--------------
A map from the six decision variables to the scalar requirement robustness
``rho(x)``. The falsification search wants the argmin of that map; the surrogate
is a cheap stand-in for it, refit as simulations accumulate, used to rank
candidate points so that simulations are spent where a violation looks likely.

Why a forest and not a Gaussian process
---------------------------------------
Two reasons, both measurable rather than stylistic:

* The response is **non-smooth**. Two clips (actuator slew rate and deflection)
  and the ``min``/``max`` of the robustness semantics put kinks and ridges all
  over ``rho``. A stationary GP kernel is a poor prior for that, and the usual
  remedy -- a short length scale -- throws away the extrapolation that makes a
  GP worth its cost.
* Cost. The search refits the surrogate every few simulations inside a loop
  budgeted in hundreds of simulations, and an exact GP is ``O(n^3)`` in the
  observations whereas the forest is near-linear. On this problem the simulator
  costs about 0.25 ms, so a surrogate that costs more than a few simulations per
  refit cannot pay for itself. The measured refit and scoring cost is in
  ``validation/validate_surrogate.py``.

The uncertainty output, described accurately
--------------------------------------------
:meth:`ForestSurrogate.predict` returns ``(mean, spread)`` where ``spread`` is
the **standard deviation of the individual trees' predictions** at that point.
Because the trees are fit on bootstrap resamples with a random feature subset,
that spread is an ensemble-dispersion statistic in the spirit of the bootstrap
(Efron 1979): it is large where the trees disagree, which is where the data are
sparse or the response is rough.

**It is not a posterior standard deviation, and its interval shape is wrong even
where its coverage looks right.** ``validation/validate_surrogate.py`` measures
the empirical coverage of ``mean +- 1.96 * spread`` on an independently seeded
held-out split, per instance. Measured on the shipped suite: nominal-95 coverage
averages **0.9356** over the eight instances and ranges from **0.8950** to
**0.9650**, so it is close to nominal and slightly under it on seven of the
eight. Nominal-68 coverage, over the same points, averages about **0.76** --
well *above* the 0.683 a Gaussian would give. Both at once means the error
distribution is not Gaussian: the spread is conservative through the body of the
distribution and too thin in the tails, and the near-nominal 95 % figure is a
coincidence of those two errors partly cancelling rather than evidence of
calibration.

**This contradicts what this docstring predicted before the measurement was
run**, which said the spread would understate the error badly. It does not; the
mean spread is 1.05 to 1.37 times the mean absolute error on every instance. The
prediction was wrong, the measurement stands, and the mistake is recorded in
``validation/VALIDATION.md`` rather than quietly deleted.

The search uses the spread only as a **ranking** signal inside a
lower-confidence-bound acquisition, where only its ordering matters. That
ordering is measurably useful: the Spearman correlation between the reported
spread and the absolute held-out error averages **0.4853** across the suite.
A number with a useful ordering and an unreliable interval shape is exactly what
should be reported as a confidence *signal* and never as a confidence
*interval*.

References
----------
Breiman, L. (2001), "Random forests", Machine Learning 45(1), 5-32. The
estimator.

Efron, B. (1979), "Bootstrap methods: another look at the jackknife", Annals of
Statistics 7(1), 1-26. The resampling idea the tree spread stands on.

Wager, S., Hastie, T. and Efron, B. (2014), "Confidence intervals for random
forests: the jackknife and the infinitesimal jackknife", Journal of Machine
Learning Research 15, 1625-1651. Why the naive tree spread is not a confidence
interval, and what would be needed instead; this package reports the naive
spread and its measured coverage rather than implementing the correction.

Srinivas, N., Krause, A., Kakade, S. and Seeger, M. (2010), "Gaussian process
optimization in the bandit setting", ICML 2010. The lower-confidence-bound
acquisition rule, used here with a forest dispersion in place of a GP posterior
standard deviation -- a substitution this module makes explicit because it
voids the regret bound that paper proves.
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor

#: Default number of trees. Small on purpose: the forest is refit inside a loop
#: whose whole budget is a few hundred simulations of 0.25 ms each.
DEFAULT_TREES = 25


class ForestSurrogate:
    """Random-forest regression of robustness, with an ensemble-spread output.

    Parameters
    ----------
    n_estimators:
        Number of trees, at least 2 (a spread needs two trees to exist).
    min_samples_leaf:
        Minimum samples per leaf, at least 1.
    random_state:
        Seed for the forest. Fixed seed plus fixed training data gives a
        bit-identical model, which is what makes a seeded search reproducible.

    Notes
    -----
    ``n_jobs`` is fixed at 1 and is deliberately not exposed. On this container
    (2 cores) a forest with ``n_jobs > 1`` is slower at the small-batch
    inference this search does, because the joblib dispatch overhead per call
    dominates the work; the figure is in ``validation/validate_surrogate.py``.
    """

    def __init__(
        self,
        n_estimators: int = DEFAULT_TREES,
        min_samples_leaf: int = 1,
        random_state: int = 0,
    ) -> None:
        if int(n_estimators) < 2:
            raise ValueError(f"n_estimators must be at least 2, got {n_estimators}")
        if int(min_samples_leaf) < 1:
            raise ValueError(f"min_samples_leaf must be at least 1, got {min_samples_leaf}")
        self.n_estimators = int(n_estimators)
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self._forest: RandomForestRegressor | None = None
        self._n_features: int | None = None

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._forest is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> ForestSurrogate:
        """Fit on decision vectors ``x`` (``n, d``) and robustness values ``y`` (``n,``).

        Raises
        ------
        ValueError
            On a shape mismatch, fewer than two samples, or a non-finite value.
            Infinite robustness -- which the semantics can produce from an empty
            time window -- is rejected here rather than silently clipped,
            because a surrogate fit through an infinity is meaningless; the
            benchmark instances check their horizon at construction so that it
            cannot arise.
        """
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float).ravel()
        if xa.ndim != 2:
            raise ValueError(f"x must be two-dimensional, got shape {xa.shape}")
        if xa.shape[0] != ya.size:
            raise ValueError(f"x has {xa.shape[0]} rows but y has {ya.size} values")
        if ya.size < 2:
            raise ValueError(f"need at least 2 training samples, got {ya.size}")
        if not np.all(np.isfinite(xa)):
            raise ValueError("x contains a non-finite value")
        if not np.all(np.isfinite(ya)):
            raise ValueError(
                "y contains a non-finite robustness value; an infinite robustness means a "
                "time-bounded window was empty, which a fitted surrogate cannot represent"
            )
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
            bootstrap=True,
        )
        forest.fit(xa, ya)
        self._forest = forest
        self._n_features = int(xa.shape[1])
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, spread)`` at ``x`` (``n, d``), both shape ``(n,)``.

        ``mean`` is the forest prediction, in the unit of the robustness it was
        fit on. ``spread`` is the standard deviation (population, ``ddof=0``)
        across the individual trees at that point, same unit. See the module
        docstring for what that number is and is not.
        """
        forest = self._require_fitted()
        xa = self._check_x(x)
        per_tree = np.stack([tree.predict(xa) for tree in forest.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0, ddof=0)

    def lower_confidence_bound(self, x: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """``mean - kappa * spread``: the acquisition the search minimises.

        A larger ``kappa`` favours points the trees disagree about over points
        they jointly predict are low. ``kappa`` must be non-negative; ``0``
        reduces the acquisition to greedy exploitation of the forest mean, which
        ``validation/validate_benchmark.py`` does not use but which is available
        for anyone wanting to see the exploration term's contribution.
        """
        kappa = float(kappa)
        if not math.isfinite(kappa) or kappa < 0.0:
            raise ValueError(f"kappa must be finite and non-negative, got {kappa}")
        mean, spread = self.predict(x)
        return mean - kappa * spread

    def _require_fitted(self) -> RandomForestRegressor:
        if self._forest is None:
            raise RuntimeError("surrogate is not fitted; call fit() first")
        return self._forest

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        if xa.ndim == 1:
            xa = xa.reshape(1, -1)
        if xa.ndim != 2:
            raise ValueError(f"x must be one- or two-dimensional, got shape {xa.shape}")
        if self._n_features is not None and xa.shape[1] != self._n_features:
            raise ValueError(
                f"x has {xa.shape[1]} features but the surrogate was fit on {self._n_features}"
            )
        if not np.all(np.isfinite(xa)):
            raise ValueError("x contains a non-finite value")
        return xa

    def __repr__(self) -> str:
        state = "fitted" if self.fitted else "unfitted"
        return (
            f"ForestSurrogate(n_estimators={self.n_estimators}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"random_state={self.random_state}, {state})"
        )
