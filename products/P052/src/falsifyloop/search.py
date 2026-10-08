r"""Five falsification search strategies over one instance, one budget, one seed.

**Uniform random is the baseline and it is a strong one.** It is the first
function in this module because it was the first one written, and every other
strategy here has to earn its place against it. On a box where the violating set
has volume fraction ``p``, the number of uniform draws to the first violation is
geometric with parameter ``p``: mean ``1/p``, and probability
``1 - (1 - p)^n`` of having found one within ``n`` draws. That closed form is
the curve the other four are plotted against, and
``validation/validate_benchmark.py`` reports where they beat it and where they
do not.

The common contract
-------------------
Every strategy has the signature
``f(instance, budget, seed, **options) -> SearchResult`` and must:

* spend **at most** ``budget`` simulations, counting every call to
  ``instance.evaluate``;
* **stop at the first violation**. A falsification run has nothing to do once it
  has a counterexample, so the result records the 1-based simulation index at
  which robustness first went negative, and the sample-efficiency curve is the
  distribution of that index, right-censored at the budget;
* be **exactly reproducible** from ``(instance, budget, seed)``;
* never look at ``instance.requirement``'s internals, the declared bound, or
  the design target probability. The strategies see only the box and the scalar
  robustness values they paid for.

Why stopping early is not cheating
----------------------------------
It would be if the comparison were "best robustness found at the budget". It is
not: the deliverable is sample efficiency to the first violation, so a run that
stops at simulation 12 contributes 12 and nothing else. Every strategy stops
under the same rule.

References
----------
McKay, M. D., Beckman, R. J. and Conover, W. J. (1979), "A comparison of three
methods for selecting values of input variables in the analysis of output from a
computer code", Technometrics 21(2), 239-245. Latin hypercube sampling.

Kirkpatrick, S., Gelatt, C. D. and Vecchi, M. P. (1983), "Optimization by
simulated annealing", Science 220(4598), 671-680. The annealing schedule and
Metropolis acceptance used here.

Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H. and
Teller, E. (1953), "Equation of state calculations by fast computing machines",
Journal of Chemical Physics 21(6), 1087-1092. The acceptance rule.

Rubinstein, R. Y. (1999), "The cross-entropy method for combinatorial
optimization, rare-event simulation and multi-extremal optimization",
Methodology and Computing in Applied Probability 1(2), 127-190. The elite-sample
distribution update.

De Boer, P.-T., Kroese, D. P., Mannor, S. and Rubinstein, R. Y. (2005), "A
tutorial on the cross-entropy method", Annals of Operations Research 134(1),
19-67. The Gaussian-family variant implemented here.

Annpureddy, Y., Liu, C., Fainekos, G. and Sankaranarayanan, S. (2011),
"S-TaLiRo: a tool for temporal logic falsification for hybrid systems", TACAS
2011, LNCS 6605. The falsification-as-optimisation framing this module sits in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

from .instances import Instance
from .surrogate import DEFAULT_TREES, ForestSurrogate


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one seeded search run.

    Attributes
    ----------
    instance_id:
        Identifier of the instance searched.
    strategy:
        Strategy name as it appears in :data:`STRATEGIES`.
    seed:
        Seed the run was reproduced from.
    budget:
        Maximum simulations the run was allowed.
    history:
        Robustness of every simulation actually run, in order, shape ``(n,)``.
    first_violation:
        1-based simulation index at which robustness first went negative, or
        ``None`` if no violation was found within the budget. ``None`` means
        **nothing was found**, which is not evidence that nothing is there.
    best_vector:
        The decision vector of the lowest robustness seen, shape ``(6,)``.
    """

    instance_id: str
    strategy: str
    seed: int
    budget: int
    history: np.ndarray
    first_violation: int | None
    best_vector: np.ndarray

    @property
    def simulations(self) -> int:
        """Number of simulations actually run."""
        return int(self.history.size)

    @property
    def best_robustness(self) -> float:
        """Lowest robustness seen."""
        return float(self.history.min())

    @property
    def found(self) -> bool:
        """Whether a violation was found within the budget."""
        return self.first_violation is not None


class _Run:
    """Bookkeeping shared by every strategy: evaluate, record, stop on violation."""

    __slots__ = ("best_rho", "best_x", "budget", "history", "instance", "violation")

    def __init__(self, instance: Instance, budget: int) -> None:
        if budget < 1:
            raise ValueError(f"budget must be at least 1 simulation, got {budget}")
        self.instance = instance
        self.budget = int(budget)
        self.history: list[float] = []
        self.violation: int | None = None
        self.best_rho = math.inf
        self.best_x = instance.centre()

    @property
    def spent(self) -> int:
        return len(self.history)

    @property
    def exhausted(self) -> bool:
        return self.violation is not None or self.spent >= self.budget

    def evaluate(self, vector: np.ndarray) -> float:
        """Run one simulation, record it, and latch the first violation."""
        if self.exhausted:
            raise RuntimeError("strategy tried to evaluate past its budget or after a violation")
        rho = self.instance.evaluate(vector)
        self.history.append(rho)
        if rho < self.best_rho:
            self.best_rho = rho
            self.best_x = np.array(vector, dtype=float)
        if rho < 0.0 and self.violation is None:
            self.violation = self.spent
        return rho

    def result(self, strategy: str, seed: int) -> SearchResult:
        return SearchResult(
            instance_id=self.instance.identifier,
            strategy=strategy,
            seed=int(seed),
            budget=self.budget,
            history=np.asarray(self.history, dtype=float),
            first_violation=self.violation,
            best_vector=self.best_x,
        )


# --------------------------------------------------------------------------- #
# 1. Uniform random -- the baseline, written first
# --------------------------------------------------------------------------- #
def uniform_random(instance: Instance, budget: int, seed: int) -> SearchResult:
    """Independent uniform draws from the declared box until a violation or the budget.

    The baseline. No state, no tuning, no assumption that robustness is in any
    way smooth -- which is exactly why it is hard to beat on an instance whose
    violating set is small but not shaped like anything a model can exploit.

    Expected simulations to the first violation is ``1/p`` for a violating-set
    volume fraction ``p``; the probability of success within ``n`` draws is
    ``1 - (1 - p)^n``, the analytic curve :func:`analytic_random_curve`
    returns.
    """
    run = _Run(instance, budget)
    rng = np.random.default_rng(seed)
    while not run.exhausted:
        run.evaluate(instance.sample(rng, 1)[0])
    return run.result("uniform-random", seed)


def analytic_random_curve(p: float, budget: int) -> np.ndarray:
    """``1 - (1-p)^n`` for ``n = 1..budget``: the exact uniform-random curve.

    Parameters
    ----------
    p:
        Violating-set volume fraction, in ``(0, 1)``.
    budget:
        Largest simulation count to evaluate at.

    Returns
    -------
    numpy.ndarray
        Shape ``(budget,)``, the probability a uniform-random search has found a
        violation by simulation ``n``. Plotted against the measured curve in
        ``examples/sample_efficiency_curve.py`` as a check that the empirical
        estimate and the closed form agree.
    """
    p = float(p)
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if budget < 1:
        raise ValueError(f"budget must be at least 1, got {budget}")
    n = np.arange(1, int(budget) + 1)
    return 1.0 - (1.0 - p) ** n


# --------------------------------------------------------------------------- #
# 2. Latin hypercube
# --------------------------------------------------------------------------- #
def latin_hypercube(instance: Instance, budget: int, seed: int) -> SearchResult:
    """One Latin hypercube design of ``budget`` points, evaluated in order.

    Stratifies each coordinate into ``budget`` equal-probability bins and visits
    one point per bin, so no coordinate is ever under-covered. The design is
    generated whole, up front, because that is what a Latin hypercube is; the
    run still stops at the first violation, so most of the design is usually
    never evaluated, and the ordering within the design is the scrambled order
    ``scipy.stats.qmc.LatinHypercube`` produces.

    This is the honest weakness of the method for falsification: the
    stratification guarantee is a property of the *whole* design, and a run that
    stops a tenth of the way through it has none of that guarantee.
    """
    run = _Run(instance, budget)
    sampler = qmc.LatinHypercube(d=instance.dimension, seed=seed)
    unit = sampler.random(n=run.budget)
    design = qmc.scale(unit, instance.box[:, 0], instance.box[:, 1])
    for row in design:
        if run.exhausted:
            break
        run.evaluate(row)
    return run.result("latin-hypercube", seed)


# --------------------------------------------------------------------------- #
# 3. Simulated annealing
# --------------------------------------------------------------------------- #
def simulated_annealing(
    instance: Instance,
    budget: int,
    seed: int,
    step_fraction: float = 0.15,
    initial_temperature: float = 0.30,
    cooling: float = 0.97,
) -> SearchResult:
    """Metropolis random walk on robustness with geometric cooling.

    Parameters
    ----------
    step_fraction:
        Proposal standard deviation as a fraction of each box edge.
    initial_temperature:
        Starting temperature in robustness units. The shipped instances
        normalise their predicates by the requirement's own tolerance, so
        robustness is dimensionless and of order one, which is why a single
        default temperature is meaningful across the suite.
    cooling:
        Geometric factor per accepted or rejected step, in ``(0, 1]``.

    The walk starts from one uniform draw, so its first simulation is a draw
    from the baseline's distribution; everything after that is the method.
    Proposals outside the box are **reflected** rather than clipped, since
    clipping piles probability mass on the faces and the violating sets of
    several instances in this suite lie on a face, which would flatter the
    method for the wrong reason.
    """
    for name, value, lo, hi in (
        ("step_fraction", step_fraction, 0.0, math.inf),
        ("initial_temperature", initial_temperature, 0.0, math.inf),
        ("cooling", cooling, 0.0, 1.0),
    ):
        if not (math.isfinite(value) and lo < value <= hi):
            raise ValueError(f"{name} must lie in ({lo}, {hi}], got {value}")

    run = _Run(instance, budget)
    rng = np.random.default_rng(seed)
    sigma = step_fraction * instance.widths()
    lo, hi = instance.box[:, 0], instance.box[:, 1]

    current = instance.sample(rng, 1)[0]
    current_rho = run.evaluate(current)
    temperature = float(initial_temperature)
    while not run.exhausted:
        proposal = _reflect(current + rng.normal(0.0, sigma), lo, hi)
        rho = run.evaluate(proposal)
        delta = rho - current_rho
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
            current, current_rho = proposal, rho
        temperature *= cooling
    return run.result("simulated-annealing", seed)


def _reflect(vector: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Fold ``vector`` into ``[lo, hi]`` by repeated reflection off the faces.

    Equivalent to a triangle wave of period ``2 * (hi - lo)``, so a proposal any
    distance outside the box lands inside it without the mass pile-up that
    clipping produces.
    """
    width = hi - lo
    folded = np.abs((vector - lo) % (2.0 * width))
    folded = np.where(folded > width, 2.0 * width - folded, folded)
    return lo + folded


# --------------------------------------------------------------------------- #
# 4. Cross-entropy method
# --------------------------------------------------------------------------- #
def cross_entropy(
    instance: Instance,
    budget: int,
    seed: int,
    population: int = 20,
    elite_fraction: float = 0.25,
    smoothing: float = 0.7,
    min_sigma_fraction: float = 0.02,
) -> SearchResult:
    """Gaussian cross-entropy method: sample, keep the elite, refit, repeat.

    Parameters
    ----------
    population:
        Simulations per generation, at least 4.
    elite_fraction:
        Fraction of each generation kept to refit the sampling distribution, in
        ``(0, 1)``; at least two points are always kept.
    smoothing:
        Weight on the newly fitted mean and standard deviation, in ``(0, 1]``.
        Below 1 the update is damped, which stops a single lucky elite set from
        collapsing the distribution onto it.
    min_sigma_fraction:
        Floor on each coordinate's standard deviation as a fraction of the box
        edge. Without a floor the method converges to a point and then spends
        the rest of the budget re-simulating it, which is the classic
        cross-entropy failure and is why the floor is a parameter and not a
        constant.

    The initial distribution is the box's own: mean at the centre, standard
    deviation a sixth of each edge, which puts about three standard deviations
    inside each face. Samples are clipped into the box, which for this method is
    the right choice -- the distribution is meant to concentrate, and reflecting
    a concentrated proposal would scatter it.
    """
    if population < 4:
        raise ValueError(f"population must be at least 4, got {population}")
    if not 0.0 < elite_fraction < 1.0:
        raise ValueError(f"elite_fraction must lie in (0, 1), got {elite_fraction}")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError(f"smoothing must lie in (0, 1], got {smoothing}")
    if not 0.0 <= min_sigma_fraction < 1.0:
        raise ValueError(f"min_sigma_fraction must lie in [0, 1), got {min_sigma_fraction}")

    run = _Run(instance, budget)
    rng = np.random.default_rng(seed)
    widths = instance.widths()
    mu = instance.centre()
    sigma = widths / 6.0
    sigma_floor = min_sigma_fraction * widths
    n_elite = max(2, int(round(elite_fraction * population)))

    while not run.exhausted:
        batch = min(population, run.budget - run.spent)
        points = np.empty((batch, instance.dimension))
        scores = np.empty(batch)
        taken = 0
        for i in range(batch):
            point = instance.clip(mu + sigma * rng.normal(size=instance.dimension))
            points[i] = point
            scores[i] = run.evaluate(point)
            taken = i + 1
            if run.violation is not None:
                break
        if run.exhausted:
            break
        points, scores = points[:taken], scores[:taken]
        k = min(n_elite, taken)
        elite = points[np.argsort(scores)[:k]]
        mu = smoothing * elite.mean(axis=0) + (1.0 - smoothing) * mu
        new_sigma = elite.std(axis=0, ddof=0) if k > 1 else sigma
        sigma = np.maximum(smoothing * new_sigma + (1.0 - smoothing) * sigma, sigma_floor)
    return run.result("cross-entropy", seed)


# --------------------------------------------------------------------------- #
# 5. Surrogate-guided search -- the learned component
# --------------------------------------------------------------------------- #
def surrogate_guided(
    instance: Instance,
    budget: int,
    seed: int,
    n_initial: int = 16,
    n_candidates: int = 256,
    refit_every: int = 8,
    kappa: float = 2.0,
    n_estimators: int = DEFAULT_TREES,
) -> SearchResult:
    """Random-forest surrogate with a lower-confidence-bound acquisition.

    Parameters
    ----------
    n_initial:
        Simulations spent on a Latin hypercube warm start before the surrogate
        is first fit, at least 2. **During these the strategy is not learned at
        all**, which is the fairest way to report it: the learned part only has
        ``budget - n_initial`` simulations to beat the baseline with, and if the
        instance is easy enough that uniform random wins inside ``n_initial``
        draws then the surrogate never got a turn. That is the honest accounting
        and it is why ``n_initial`` is reported in the README.
    n_candidates:
        Uniform candidate points scored by the surrogate per simulation. Scoring
        is free relative to a simulation only while this stays modest; the
        measured cost is in ``validation/validate_surrogate.py``.
    refit_every:
        Simulations between refits, at least 1.
    kappa:
        Weight on the forest's tree spread in the acquisition
        ``mean - kappa * spread``.
    n_estimators:
        Trees in the forest.

    Returns
    -------
    SearchResult
        With ``strategy == "surrogate-guided"``.
    """
    if n_initial < 2:
        raise ValueError(f"n_initial must be at least 2, got {n_initial}")
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be at least 1, got {n_candidates}")
    if refit_every < 1:
        raise ValueError(f"refit_every must be at least 1, got {refit_every}")

    run = _Run(instance, budget)
    rng = np.random.default_rng(seed)
    sampler = qmc.LatinHypercube(d=instance.dimension, seed=seed)
    warm = qmc.scale(
        sampler.random(n=min(n_initial, run.budget)), instance.box[:, 0], instance.box[:, 1]
    )
    seen_x: list[np.ndarray] = []
    seen_y: list[float] = []
    for row in warm:
        if run.exhausted:
            break
        seen_y.append(run.evaluate(row))
        seen_x.append(np.asarray(row, dtype=float))

    model = ForestSurrogate(n_estimators=n_estimators, random_state=seed)
    since_fit = refit_every
    while not run.exhausted:
        if since_fit >= refit_every or not model.fitted:
            model.fit(np.asarray(seen_x), np.asarray(seen_y))
            since_fit = 0
        candidates = instance.sample(rng, n_candidates)
        acquisition = model.lower_confidence_bound(candidates, kappa=kappa)
        pick = candidates[int(np.argmin(acquisition))]
        seen_y.append(run.evaluate(pick))
        seen_x.append(pick)
        since_fit += 1
    return run.result("surrogate-guided", seed)


#: Strategy registry. The baseline is first, deliberately.
STRATEGIES: Mapping[str, Callable[..., SearchResult]] = {
    "uniform-random": uniform_random,
    "latin-hypercube": latin_hypercube,
    "simulated-annealing": simulated_annealing,
    "cross-entropy": cross_entropy,
    "surrogate-guided": surrogate_guided,
}

#: The baseline every other strategy is reported against.
BASELINE = "uniform-random"


def strategy(name: str) -> Callable[..., SearchResult]:
    """Look up a strategy by name.

    Raises
    ------
    KeyError
        If ``name`` is not a registered strategy; the message lists them.
    """
    try:
        return STRATEGIES[name]
    except KeyError:
        raise KeyError(
            f"unknown strategy {name!r}; registered strategies are {list(STRATEGIES)}"
        ) from None
