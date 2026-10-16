"""Scan-order policies: fixed pattern, Bayesian-greedy baseline, learned policy.

The decision problem
--------------------
At every step the terminal must choose which cell to dwell on next. The state
is the posterior over target location given the misses so far, the current
boresight position (which sets the slew cost of every candidate) and the dwell
history.

Posterior update (the analytic baseline). Under assumption A2 of
:mod:`acqsearch.acquisition`, a miss on cell ``j`` multiplies the posterior
mass of cell ``j`` by ``(1 - q_j)`` and leaves every other cell unchanged, up
to renormalisation:

    P(target in j | miss at j) = P(j) (1 - q_j) / (1 - P(j) q_j)
    P(target in i | miss at j) = P(i)          / (1 - P(j) q_j),   i != j

This is Bayes' rule with the likelihood of A2 and nothing else. A false alarm
carries no information about the target location under A3, so it does not
update the posterior; it only costs time.

Bayesian-greedy. Choose the cell maximising detection probability per unit
time, ``P(i) q_i / c(i)``, where ``c(i)`` is the dwell plus the slew from the
current position plus the expected false-alarm charge. This is the classical
search-theory rule: Koopman, B. O. (1980) *Search and Screening: General
Principles with Historical Applications*, Pergamon Press, Ch. 5-6; Stone, L. D.
(1975) *Theory of Optimal Search*, Academic Press (Mathematics in Science and
Engineering vol. 118), Ch. 2-3, where the uniformly optimal allocation under an
exponential detection function is derived. The rule is myopic -- it optimises
the next dwell, not the sequence -- and with a non-zero slew cost it is not
optimal in general, which is the gap the learned policy is given the chance to
close.

Learned policy. A gradient-boosted regression of the realised remaining time
to acquisition on features of the (state, candidate cell) pair, fitted on
episodes rolled out under an epsilon-greedy behaviour policy, and used by
choosing the candidate with the smallest predicted remaining time. It also
fits 0.1 and 0.9 pinball-loss quantile regressors, so every decision carries a
prediction interval rather than a point estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .acquisition import cell_detection_probabilities
from .detection import PoissonDwellDetector
from .geometry import CellGrid, PointingUncertainty, TerminalGeometry

FEATURE_NAMES: tuple[str, ...] = (
    "posterior",
    "log_posterior",
    "detection_probability",
    "slew_time_ratio",
    "step_cost_s",
    "greedy_score_normalised",
    "greedy_rank_fraction",
    "cell_radius_fraction",
    "visit_count",
    "remaining_posterior",
    "dwells_elapsed",
    "neighbour_posterior",
    "current_radius_fraction",
    "elapsed_over_period",
)


@dataclass(frozen=True)
class AcquisitionEnvironment:
    """Everything a policy needs to score a candidate cell, and to be simulated.

    All times are seconds and all angles radians. ``cell_detection`` is ``q_i``
    per cell, ``false_alarm_probability`` is ``f`` and ``verification_time_s``
    is ``t_v``; the expected false-alarm charge ``f * t_v`` is added to every
    dwell, exactly as in assumption A3 of :mod:`acqsearch.acquisition`.
    """

    grid: CellGrid
    geometry: TerminalGeometry
    cell_detection: np.ndarray
    dwell_s: float
    false_alarm_probability: float
    verification_time_s: float
    deadline_s: float
    nominal_period_s: float
    neighbours: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def build(
        cls,
        grid: CellGrid,
        geometry: TerminalGeometry,
        detector: PoissonDwellDetector,
        uncertainty: PointingUncertainty,
        dwell_s: float,
        verification_time_s: float,
        deadline_s: float,
    ) -> AcquisitionEnvironment:
        """Construct an environment, computing ``q_i`` and ``f`` from the detector."""
        q = cell_detection_probabilities(grid, geometry, detector, dwell_s, uncertainty)
        f = detector.false_alarm_probability(dwell_s)
        charge = f * verification_time_s
        mean_step = dwell_s + geometry.settle_time_s + grid.side_rad / geometry.slew_rate_rad_s
        d2 = ((grid.centres_rad[:, None, :] - grid.centres_rad[None, :, :]) ** 2).sum(axis=2)
        neighbours = np.argsort(d2, axis=1)[:, 1:5]
        return cls(
            grid=grid,
            geometry=geometry,
            cell_detection=q,
            dwell_s=float(dwell_s),
            false_alarm_probability=float(f),
            verification_time_s=float(verification_time_s),
            deadline_s=float(deadline_s),
            nominal_period_s=float(grid.n_cells * (mean_step + charge)),
            neighbours=neighbours,
        )

    def step_cost_s(self, current_rad: np.ndarray) -> np.ndarray:
        """Expected cost of dwelling on each cell from ``current_rad``, seconds."""
        delta = self.grid.centres_rad - np.asarray(current_rad, dtype=float)
        distance = np.hypot(delta[:, 0], delta[:, 1])
        slew = np.asarray(self.geometry.slew_time_s(distance), dtype=float)
        return self.dwell_s + slew + self.false_alarm_probability * self.verification_time_s


def initial_posterior(grid: CellGrid) -> np.ndarray:
    """Prior over cells, renormalised to the mass inside the grid. Dimensionless."""
    mass = grid.prior_mass.sum()
    if mass <= 0.0:
        raise ValueError("the grid carries no prior mass")
    return grid.prior_mass / mass


def update_posterior(posterior: np.ndarray, cell: int, detection_probability: float) -> np.ndarray:
    """Bayes update of the cell posterior after a miss at ``cell``.

    Returns a new normalised array. Raises when the update would divide by
    zero, which happens only if the posterior has collapsed onto a cell with
    ``q == 1``; that state cannot be reached by a miss.
    """
    posterior = np.asarray(posterior, dtype=float)
    updated = posterior.copy()
    updated[int(cell)] = posterior[int(cell)] * (1.0 - float(detection_probability))
    total = updated.sum()
    if total <= 0.0:
        raise ValueError("posterior collapsed to zero mass; the update is undefined")
    return updated / total


def build_features(
    environment: AcquisitionEnvironment,
    posterior: np.ndarray,
    current_rad: np.ndarray,
    visits: np.ndarray,
    dwells_elapsed: int,
    elapsed_s: float,
) -> np.ndarray:
    """Feature matrix of shape ``(n_cells, len(FEATURE_NAMES))`` for the current state.

    One row per candidate cell. Columns are documented by
    :data:`FEATURE_NAMES`; units are given in MODEL_CARD.md. Deterministic.
    """
    grid = environment.grid
    cost = environment.step_cost_s(current_rad)
    q = environment.cell_detection
    score = posterior * q / cost
    peak = score.max()
    normalised = score / peak if peak > 0.0 else np.zeros_like(score)
    rank = np.empty(grid.n_cells, dtype=float)
    rank[np.argsort(-score, kind="stable")] = np.arange(grid.n_cells, dtype=float)
    rank /= max(1.0, grid.n_cells - 1.0)
    radius = grid.radii_rad / grid.extent_rad
    current_radius = float(np.hypot(current_rad[0], current_rad[1])) / grid.extent_rad
    neighbour_mass = posterior[environment.neighbours].mean(axis=1)
    columns = [
        posterior,
        np.log10(posterior + 1e-12),
        q,
        (cost - environment.dwell_s) / environment.dwell_s,
        cost,
        normalised,
        rank,
        radius,
        visits.astype(float),
        np.full(grid.n_cells, float(posterior.sum())),
        np.full(grid.n_cells, float(dwells_elapsed)),
        neighbour_mass,
        np.full(grid.n_cells, current_radius),
        np.full(grid.n_cells, elapsed_s / environment.nominal_period_s),
    ]
    return np.column_stack(columns)


def greedy_scores(
    environment: AcquisitionEnvironment,
    posterior: np.ndarray,
    current_rad: np.ndarray,
) -> np.ndarray:
    """Bayesian-greedy score ``P(i) q_i / c(i)`` for every cell, units s^-1."""
    cost = environment.step_cost_s(current_rad)
    return np.asarray(posterior, dtype=float) * environment.cell_detection / cost


def shortlist_indices(
    environment: AcquisitionEnvironment,
    posterior: np.ndarray,
    current_rad: np.ndarray,
    size: int,
) -> np.ndarray:
    """The ``size`` cells with the highest Bayesian-greedy score, best first.

    The learned policy re-ranks this shortlist rather than the whole grid. The
    choice is deliberate and it is a constraint on what the learned policy can
    do: it can reorder the candidates the myopic rule already considers
    plausible, and it cannot choose a cell the rule ranks below position
    ``size``. The alternative -- scoring all cells -- was measured and is
    reported in MODEL_CARD.md under "what was tried".
    """
    k = int(size)
    if k < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")
    k = min(k, environment.grid.n_cells)
    score = greedy_scores(environment, posterior, current_rad)
    top = np.argpartition(-score, k - 1)[:k]
    return top[np.argsort(-score[top], kind="stable")]


@dataclass
class FixedPatternPolicy:
    """Cycles a precomputed cell visit order, ignoring the posterior.

    This is the non-adaptive reference: whatever the misses say, the next dwell
    is the next cell of the pattern.
    """

    order: np.ndarray
    name: str = "fixed"

    def reset(self) -> None:
        """No state to reset; present so every policy has the same interface."""

    def select(
        self,
        environment: AcquisitionEnvironment,
        posterior: np.ndarray,
        current_rad: np.ndarray,
        visits: np.ndarray,
        dwells_elapsed: int,
        elapsed_s: float,
    ) -> int:
        """Return the next cell index."""
        return int(self.order[dwells_elapsed % self.order.size])


@dataclass
class BayesianGreedyPolicy:
    """Maximises posterior detection probability per unit time (Koopman, Stone).

    ``argmax_i P(i) q_i / c(i)``. Myopic by construction. ``epsilon`` > 0 makes
    it an exploring behaviour policy for generating training data; it is 0 for
    the baseline that is benchmarked.
    """

    epsilon: float = 0.0
    name: str = "bayesian_greedy"
    rng: np.random.Generator | None = None
    explore_shortlist: int = 0

    def reset(self) -> None:
        """No state to reset."""

    def select(
        self,
        environment: AcquisitionEnvironment,
        posterior: np.ndarray,
        current_rad: np.ndarray,
        visits: np.ndarray,
        dwells_elapsed: int,
        elapsed_s: float,
    ) -> int:
        """Return the next cell index."""
        if self.epsilon > 0.0:
            generator = self.rng if self.rng is not None else np.random.default_rng()
            if generator.random() < self.epsilon:
                if self.explore_shortlist > 0:
                    candidates = shortlist_indices(
                        environment, posterior, current_rad, self.explore_shortlist
                    )
                    return int(generator.choice(candidates))
                return int(generator.integers(environment.grid.n_cells))
        return int(np.argmax(greedy_scores(environment, posterior, current_rad)))


@dataclass(frozen=True)
class PolicyDecision:
    """A learned-policy decision with its uncertainty.

    ``predicted_remaining_s`` is the point prediction for the chosen cell;
    ``interval_s`` is the (0.1, 0.9) pinball-quantile interval around it;
    ``confidence`` is ``1 / (1 + width / max(point, dwell))``, dimensionless in
    (0, 1], which is 1 for a zero-width interval and falls towards 0 as the
    interval widens relative to the prediction. It is a monotone restatement of
    the interval width and nothing more; it is not a calibrated probability,
    and MODEL_CARD.md says so.
    """

    cell: int
    predicted_remaining_s: float
    interval_s: tuple[float, float]
    confidence: float


class LearnedScanPolicy:
    """Gradient-boosted value regression over (state, candidate cell) features.

    Fitted on episodes rolled out under an epsilon-greedy Bayesian-greedy
    behaviour policy; the regression target is the realised remaining time to
    acquisition, censored at the episode deadline. Selection takes the
    candidate with the smallest predicted remaining time.

    The three fitted models are a squared-error regressor for the point
    prediction and two pinball-loss quantile regressors at 0.1 and 0.9 for the
    interval. ``GradientBoostingRegressor`` is used rather than the histogram
    variant because this policy is called once per dwell inside an episode
    loop, so single-row-batch prediction latency, not training throughput, is
    the binding cost on the two-core budget. Nothing is pickled: the model is
    refitted from the committed scenario file and the committed seed, which is
    why this product declares no dependency on a serialisation format.
    """

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 20261016,
        shortlist: int = 12,
    ) -> None:
        self.name = "learned"
        self.shortlist = int(shortlist)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self._point: GradientBoostingRegressor | None = None
        self._low: GradientBoostingRegressor | None = None
        self._high: GradientBoostingRegressor | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._point is not None

    def _make(self, loss: str, alpha: float | None) -> GradientBoostingRegressor:
        kwargs: dict[str, object] = {
            "loss": loss,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
        }
        if alpha is not None:
            kwargs["alpha"] = alpha
        return GradientBoostingRegressor(**kwargs)  # type: ignore[arg-type]

    def fit(self, features: np.ndarray, remaining_time_s: np.ndarray) -> LearnedScanPolicy:
        """Fit the point and quantile regressors. Returns ``self``."""
        x = np.asarray(features, dtype=float)
        y = np.asarray(remaining_time_s, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape (n, {len(FEATURE_NAMES)}), got {x.shape}")
        if y.shape != (x.shape[0],):
            raise ValueError("remaining_time_s must have one entry per feature row")
        if x.shape[0] < 2:
            raise ValueError("at least two training rows are required")
        self._point = self._make("squared_error", None).fit(x, y)
        self._low = self._make("quantile", 0.1).fit(x, y)
        self._high = self._make("quantile", 0.9).fit(x, y)
        return self

    def predict_with_uncertainty(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Point prediction and (0.1, 0.9) quantiles for each feature row, seconds."""
        if self._point is None or self._low is None or self._high is None:
            raise RuntimeError("LearnedScanPolicy.fit must be called before prediction")
        x = np.asarray(features, dtype=float)
        point = self._point.predict(x)
        low = self._low.predict(x)
        high = self._high.predict(x)
        return point, np.minimum(low, high), np.maximum(low, high)

    def reset(self) -> None:
        """No per-episode state to reset."""

    def decide(
        self,
        environment: AcquisitionEnvironment,
        posterior: np.ndarray,
        current_rad: np.ndarray,
        visits: np.ndarray,
        dwells_elapsed: int,
        elapsed_s: float,
    ) -> PolicyDecision:
        """Choose a cell and report the prediction interval behind the choice.

        Predictions are returned in seconds: the regressors are fitted on the
        remaining time divided by ``environment.nominal_period_s`` and the
        prediction is multiplied back, so one model serves scenarios with
        different timing.
        """
        candidates = shortlist_indices(environment, posterior, current_rad, self.shortlist)
        features = build_features(
            environment, posterior, current_rad, visits, dwells_elapsed, elapsed_s
        )[candidates]
        point, low, high = self.predict_with_uncertainty(features)
        period = environment.nominal_period_s
        pick = int(np.argmin(point))
        width = float((high[pick] - low[pick]) * period)
        scale = max(float(point[pick] * period), environment.dwell_s)
        return PolicyDecision(
            cell=int(candidates[pick]),
            predicted_remaining_s=float(point[pick] * period),
            interval_s=(float(low[pick] * period), float(high[pick] * period)),
            confidence=1.0 / (1.0 + width / scale),
        )

    def select(
        self,
        environment: AcquisitionEnvironment,
        posterior: np.ndarray,
        current_rad: np.ndarray,
        visits: np.ndarray,
        dwells_elapsed: int,
        elapsed_s: float,
    ) -> int:
        """Return the next cell index.

        Only the point regressor is evaluated here. The quantile models are
        evaluated by :meth:`decide`, which is what a caller uses when it wants
        the interval; keeping them out of the episode loop is what makes the
        benchmark fit the two-core compute budget.
        """
        if self._point is None:
            raise RuntimeError("LearnedScanPolicy.fit must be called before prediction")
        candidates = shortlist_indices(environment, posterior, current_rad, self.shortlist)
        features = build_features(
            environment, posterior, current_rad, visits, dwells_elapsed, elapsed_s
        )[candidates]
        return int(candidates[int(np.argmin(self._point.predict(features)))])


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one simulated acquisition episode.

    ``time_s`` is the elapsed time at acquisition, or the deadline when the
    episode was censored. ``acquired`` says which. ``dwells`` counts the dwells
    performed.
    """

    time_s: float
    acquired: bool
    dwells: int


def run_episode(
    environment: AcquisitionEnvironment,
    policy: object,
    target_cell: int,
    rng: np.random.Generator,
    max_dwells: int | None = None,
    record: list[tuple[np.ndarray, int, float]] | None = None,
) -> EpisodeResult:
    """Simulate one episode of ``policy`` against a target in ``target_cell``.

    Time accounting matches assumptions A2-A4 of :mod:`acqsearch.acquisition`:
    each dwell costs ``dwell + slew + f * t_v``, and a dwell on the target cell
    detects with probability ``q``. The episode is censored at
    ``environment.deadline_s`` and the censored time is reported as the
    deadline, so the mean over episodes is a bounded, comparable statistic
    rather than an undefined one.

    When ``record`` is given, each decision appends
    ``(feature_row, chosen_cell, elapsed_at_decision)`` to it; the caller turns
    those into training rows once the episode's end time is known.
    """
    grid = environment.grid
    posterior = initial_posterior(grid)
    visits = np.zeros(grid.n_cells, dtype=np.int64)
    current = np.zeros(2)
    elapsed = 0.0
    limit = int(max_dwells) if max_dwells is not None else 3 * grid.n_cells
    for step in range(limit):
        if record is not None:
            features = build_features(environment, posterior, current, visits, step, elapsed)
            cell = int(policy.select(environment, posterior, current, visits, step, elapsed))
            record.append((features[cell].copy(), cell, elapsed))
        else:
            cell = int(policy.select(environment, posterior, current, visits, step, elapsed))
        cost = float(environment.step_cost_s(current)[cell])
        elapsed += cost
        visits[cell] += 1
        current = grid.centres_rad[cell]
        if elapsed > environment.deadline_s:
            return EpisodeResult(time_s=environment.deadline_s, acquired=False, dwells=step + 1)
        q = float(environment.cell_detection[cell])
        if cell == int(target_cell) and rng.random() < q:
            return EpisodeResult(time_s=elapsed, acquired=True, dwells=step + 1)
        posterior = update_posterior(posterior, cell, q)
    return EpisodeResult(time_s=environment.deadline_s, acquired=False, dwells=limit)


def sample_target_cells(grid: CellGrid, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` target cells from the prior restricted to the grid."""
    probability = initial_posterior(grid)
    return rng.choice(grid.n_cells, size=int(n), p=probability)


def collect_training_rows(
    environment: AcquisitionEnvironment,
    behaviour: object,
    target_cells: np.ndarray,
    rng: np.random.Generator,
    max_dwells: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out episodes and return ``(features, remaining_time_s)``.

    The regression target for a decision taken at elapsed time ``t`` in an
    episode that ended at ``T`` is ``(T - t) / nominal_period_s``, dimensionless,
    so episodes from scenarios with different dwell and slew timing are
    comparable in one fit. Censored episodes contribute ``deadline - t``, which
    under-states the true remaining time; that bias is recorded in
    MODEL_CARD.md as a known limitation rather than discarded, because
    discarding censored episodes would bias the fit the other way.
    """
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for target in np.atleast_1d(target_cells):
        record: list[tuple[np.ndarray, int, float]] = []
        result = run_episode(
            environment, behaviour, int(target), rng, max_dwells=max_dwells, record=record
        )
        for features, _cell, elapsed in record:
            rows.append(features)
            targets.append(max(0.0, result.time_s - elapsed) / environment.nominal_period_s)
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0)
    return np.vstack(rows), np.asarray(targets, dtype=float)


def mean_censored_time(results: list[EpisodeResult]) -> float:
    """Mean episode time with censored episodes charged at the deadline, seconds."""
    if not results:
        raise ValueError("results must be non-empty")
    return float(np.mean([r.time_s for r in results]))


def acquisition_rate(results: list[EpisodeResult]) -> float:
    """Fraction of episodes that acquired before the deadline, dimensionless."""
    if not results:
        raise ValueError("results must be non-empty")
    return float(np.mean([1.0 if r.acquired else 0.0 for r in results]))


def median_time(results: list[EpisodeResult]) -> float:
    """Median episode time with censoring at the deadline, seconds.

    The median is well defined whenever fewer than half the episodes are
    censored; above that it saturates at the deadline, and
    :func:`acquisition_rate` is the statistic to read instead.
    """
    if not results:
        raise ValueError("results must be non-empty")
    return float(np.median([r.time_s for r in results]))


def episode_time_standard_error(results: list[EpisodeResult]) -> float:
    """Standard error of :func:`mean_censored_time`, seconds."""
    if len(results) < 2:
        return float("nan")
    times = np.asarray([r.time_s for r in results], dtype=float)
    return float(times.std(ddof=1) / math.sqrt(times.size))
