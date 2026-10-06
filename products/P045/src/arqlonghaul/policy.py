"""Redundancy policies: three baselines first, then the learned one.

The decision
------------
At every HARQ round the sender must choose how many redundancy symbols to send
next, from the discrete set :data:`arqlonghaul.datasets.DEFAULT_ACTIONS`.  Send
too few and the frame needs another round, which costs a round trip.  Send too
many and the symbols are wasted.  The right answer depends on the fade state,
which the sender cannot see.

The honest structural point, stated before any result: on a long link the
feedback that would reveal the fade state is at least one round trip old by the
time it can be acted upon.  A policy therefore cannot track the channel; it can
only exploit the channel's *statistics*, plus whatever correlation survives the
staleness.  How much that is worth is an empirical question with an answer that
depends on the ratio of the fade correlation time to the round length, so it is
measured in both regimes (:func:`arqlonghaul.datasets.long_burst_env` and
:func:`arqlonghaul.datasets.short_burst_env`) and reported either way.

Order of work, which is not negotiable (ADR-011 and the build guide): the
analytic baseline is computed first, the tuned baselines second, the learned
model last, and all four are scored on the same held-out seeds.  If a baseline
wins, the baseline winning is the result.

Policies
--------
``AnalyticFixedPolicy``   the throughput-optimal fixed schedule computed from
                          the closed form at the stationary mixture BER.  Uses
                          no data at all.
``TunedFixedPolicy``      a fixed ``(first, later)`` schedule whose two values
                          are chosen by grid search on the tune seeds.
``EscalatingPolicy``      send more after each failure: ``a_m = a_base * 2**(m-1)``
                          snapped to the action set, with ``a_base`` chosen on
                          the tune seeds.
``LearnedRedundancyPolicy`` a random forest trained on the fit seeds to predict
                          the cost still to come from a (state, action) pair,
                          acting greedily on that prediction, with an ensemble
                          standard deviation and a decision margin as its
                          uncertainty output.

Feature vector, eight observable quantities, listed in :data:`FEATURE_NAMES`.
The fade state is deliberately not among them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .datasets import FadeHarqEnv
from .harq import schedule_metrics, singleton_t

__all__ = [
    "FEATURE_NAMES",
    "EpisodeStats",
    "Policy",
    "AnalyticFixedPolicy",
    "TunedFixedPolicy",
    "EscalatingPolicy",
    "RandomPolicy",
    "LearnedRedundancyPolicy",
    "run_episode",
    "evaluate",
    "collect_transitions",
    "tune_fixed",
    "tune_escalating",
    "analytic_fixed",
]

FEATURE_NAMES: tuple[str, ...] = (
    "round_index",
    "redundancy_ratio",
    "rounds_remaining",
    "prev_frame_rounds",
    "nak_ewma",
    "rounds_last8_mean",
    "consecutive_hard_frames",
    "rtt_over_k",
)
"""The observable state.  Deliberately excludes the fade state."""

_N_FEATURES = len(FEATURE_NAMES)


@dataclass
class EpisodeStats:
    """Aggregate outcome of one episode (one seed, many frames).

    Attributes:
        frames: Frames attempted.
        delivered: Frames decoded within ``max_rounds``.
        elapsed_symbols: Total elapsed symbol times, including feedback waits.
        symbols_sent: Channel symbols transmitted, excluding feedback waits.
        rounds: Total transmissions.
        cost: Total surrogate cost, elapsed plus drop penalties.
        goodput: ``k * delivered / elapsed_symbols``, dimensionless in [0, 1].
        residual_fer: Fraction of frames not decoded within ``max_rounds``.
        mean_rounds: Transmissions per frame.
        action_counts: How often each action was chosen.
    """

    frames: int
    delivered: int
    elapsed_symbols: float
    symbols_sent: float
    rounds: int
    cost: float
    goodput: float
    residual_fer: float
    mean_rounds: float
    action_counts: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))

    def as_dict(self) -> dict[str, float | int]:
        """Flat dict of the scalar fields."""
        return {
            "frames": self.frames,
            "delivered": self.delivered,
            "goodput": self.goodput,
            "residual_fer": self.residual_fer,
            "mean_rounds": self.mean_rounds,
            "cost_per_frame": self.cost / self.frames if self.frames else float("nan"),
            "symbols_per_frame": (
                self.symbols_sent / self.frames if self.frames else float("nan")
            ),
        }


class Policy:
    """Interface: map an observation to an index into ``env.actions``."""

    name: str = "policy"

    def act(self, obs: np.ndarray) -> int:
        """Index of the chosen action."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear any per-episode state.  Stateless policies need do nothing."""
        return None


@dataclass
class AnalyticFixedPolicy(Policy):
    """Fixed ``(first, later)`` schedule from the closed form, no data used.

    Attributes:
        first_index: Index into ``env.actions`` for round 1.
        later_index: Index into ``env.actions`` for rounds 2..M.
        name: Label.
    """

    first_index: int
    later_index: int
    name: str = "analytic-fixed"

    def act(self, obs: np.ndarray) -> int:
        """Round 1 uses ``first_index``, later rounds ``later_index``."""
        return self.first_index if obs[0] <= 1.0 else self.later_index


@dataclass
class TunedFixedPolicy(AnalyticFixedPolicy):
    """Fixed ``(first, later)`` schedule chosen by grid search on the tune seeds."""

    name: str = "tuned-fixed"


@dataclass
class EscalatingPolicy(Policy):
    """Send more redundancy after each failure: ``a_base * 2**(m-1)``, snapped.

    Attributes:
        base_index: Index into ``env.actions`` for round 1.
        n_actions: Size of the action set.
        name: Label.
    """

    base_index: int
    n_actions: int
    name: str = "escalating"

    def act(self, obs: np.ndarray) -> int:
        """Advance one action per failed round, saturating at the largest."""
        m = int(round(obs[0]))
        return min(self.n_actions - 1, self.base_index + (m - 1))


@dataclass
class RandomPolicy(Policy):
    """Uniform random action.  The behaviour policy used to collect training data.

    Attributes:
        n_actions: Size of the action set.
        rng: Generator.
        name: Label.
    """

    n_actions: int
    rng: np.random.Generator
    name: str = "random-behaviour"

    def act(self, obs: np.ndarray) -> int:
        """Uniform over the action set."""
        return int(self.rng.integers(self.n_actions))


class LearnedRedundancyPolicy(Policy):
    """Random-forest cost-to-go model, acted on greedily.

    This is one step of approximate policy improvement.  A uniform random
    behaviour policy is rolled out on the fit seeds; at every decision point the
    realised cost from that point to the end of the frame is recorded; a random
    forest regresses that cost on (observable state, action); the policy then
    chooses the action with the lowest predicted cost.  No bootstrapping, no
    second iteration -- the greedy policy is therefore an improvement on the
    *random* policy, which is a weaker statement than optimality and is the
    statement being made.

    Uncertainty output: the forest's trees give a predictive distribution per
    action.  :meth:`predict_with_uncertainty` returns the per-action mean and
    the across-tree standard deviation; :meth:`decision_confidence` returns the
    gap between the best and second-best predicted cost expressed in units of
    that standard deviation, which is the quantity that says whether the
    policy's choice is meaningful or a coin flip.

    Attributes:
        n_actions: Size of the action set.
        model: The fitted forest, or None before :meth:`fit`.
    """

    name = "learned-forest"

    def __init__(
        self,
        n_actions: int,
        n_estimators: int = 160,
        max_depth: int | None = 10,
        min_samples_leaf: int = 20,
        random_state: int = 0,
    ) -> None:
        if n_actions < 2:
            raise ValueError(f"n_actions must be >= 2, got {n_actions}")
        self.n_actions = int(n_actions)
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
        )
        self._fitted = False
        self._cache: dict[tuple[float, ...], int] = {}

    def fit(self, features: np.ndarray, actions: np.ndarray, cost: np.ndarray) -> None:
        """Fit the cost-to-go model.

        Args:
            features: ``(n_samples, 8)`` observable states.
            actions: ``(n_samples,)`` action indices taken.
            cost: ``(n_samples,)`` realised cost from that decision to the end of
                the frame, symbol times.

        Raises:
            ValueError: on a shape mismatch or an out-of-range action index.
        """
        x = np.asarray(features, dtype=float)
        a = np.asarray(actions, dtype=int)
        y = np.asarray(cost, dtype=float)
        if x.ndim != 2 or x.shape[1] != _N_FEATURES:
            raise ValueError(
                f"features must have shape (n, {_N_FEATURES}), got {x.shape}"
            )
        if a.shape != (x.shape[0],) or y.shape != (x.shape[0],):
            raise ValueError("features, actions and cost must agree in length")
        if a.min() < 0 or a.max() >= self.n_actions:
            raise ValueError(
                f"action indices must lie in [0, {self.n_actions}), got "
                f"[{a.min()}, {a.max()}]"
            )
        design = np.column_stack([x, a.astype(float)])
        self.model.fit(design, y)
        self._fitted = True
        self._cache.clear()

    def _design(self, obs: np.ndarray) -> np.ndarray:
        """One row per candidate action."""
        o = np.asarray(obs, dtype=float).reshape(1, -1)
        if o.shape[1] != _N_FEATURES:
            raise ValueError(f"obs must have {_N_FEATURES} features, got {o.shape[1]}")
        rows = np.repeat(o, self.n_actions, axis=0)
        return np.column_stack([rows, np.arange(self.n_actions, dtype=float)])

    def predict_with_uncertainty(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-action predicted cost and across-tree standard deviation.

        Returns:
            ``(mean, std)``, each shape ``(n_actions,)``, symbol times.

        Raises:
            RuntimeError: if the model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before predicting")
        design = self._design(obs)
        per_tree = np.stack(
            [est.predict(design) for est in self.model.estimators_], axis=0
        )
        return per_tree.mean(axis=0), per_tree.std(axis=0)

    def decision_confidence(self, obs: np.ndarray) -> dict[str, float]:
        """Confidence in the chosen action.

        Returns:
            ``{"action", "predicted_cost", "std", "margin", "margin_sigma"}``.
            ``margin`` is the predicted-cost gap to the runner-up, symbol times;
            ``margin_sigma`` divides it by the pooled across-tree standard
            deviation of the two, so a value below about 1 means the policy
            cannot distinguish its two best actions.
        """
        mean, std = self.predict_with_uncertainty(obs)
        order = np.argsort(mean)
        best, second = int(order[0]), int(order[1])
        pooled = math.sqrt(std[best] ** 2 + std[second] ** 2)
        margin = float(mean[second] - mean[best])
        return {
            "action": float(best),
            "predicted_cost": float(mean[best]),
            "std": float(std[best]),
            "margin": margin,
            "margin_sigma": margin / pooled if pooled > 0 else float("inf"),
        }

    def act(self, obs: np.ndarray) -> int:
        """Action with the lowest predicted cost to go.

        Memoised on the observation, which is finite because the two continuous
        observables are quantised by :func:`_quantise`.  The cache is cleared by
        :meth:`fit`, so it can never serve a prediction from a stale model.
        """
        if not self._fitted:
            raise RuntimeError("fit() must be called before act()")
        key = tuple(np.asarray(obs, dtype=float).tolist())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        design = self._design(obs)
        choice = int(np.argmin(self.model.predict(design)))
        self._cache[key] = choice
        return choice

    @property
    def cache_size(self) -> int:
        """Distinct observations seen since the last fit."""
        return len(self._cache)


QUANTISATION_LEVELS: int = 16
"""Quantisation of the two continuous observables, levels over [0, 1].

The exponentially weighted NAK rate and the mean round count over the last
eight frames are quantised to sixteenths before the policy sees them.  Two
reasons, and the second is the honest one: an implementation would keep these
statistics in a few bits of state rather than a float, and a finite observation
space lets the learned policy's action be memoised, which is what makes a
400-seed evaluation finish inside the compute budget.  The quantisation is
applied identically when collecting training data and when acting, so it is
part of the problem definition rather than a shortcut taken at evaluation time.
"""

CONSECUTIVE_HARD_CAP: int = 8
"""Cap on the run-length observable, frames.  Saturates the state space."""


def _quantise(x: float) -> float:
    """Round ``x`` to the nearest of :data:`QUANTISATION_LEVELS` levels in [0, 1]."""
    q = min(1.0, max(0.0, x))
    return round(q * QUANTISATION_LEVELS) / QUANTISATION_LEVELS


def _features(
    round_index: int,
    redundancy: int,
    env: FadeHarqEnv,
    prev_frame_rounds: int,
    nak_ewma: float,
    rounds_last8: float,
    consecutive_hard: int,
) -> np.ndarray:
    """Assemble the eight-element observation, in :data:`FEATURE_NAMES` order."""
    return np.array(
        [
            float(round_index),
            redundancy / env.k,
            float(env.max_rounds - round_index),
            prev_frame_rounds / env.max_rounds,
            _quantise(nak_ewma),
            _quantise(rounds_last8 / env.max_rounds),
            float(min(consecutive_hard, CONSECUTIVE_HARD_CAP)),
            env.rtt_symbols / env.k,
        ],
        dtype=float,
    )


def run_episode(
    env: FadeHarqEnv,
    policy: Policy,
    n_frames: int,
    seed: int,
    record: bool = False,
) -> tuple[EpisodeStats, dict[str, np.ndarray]]:
    """Run ``n_frames`` frames of ``env`` under ``policy``.

    The fade state trace is generated from ``seed`` up front and consumed one
    step per round, so two policies run with the same seed see the same channel
    until their round counts diverge.  That is common random numbers, and it is
    what makes a difference of a per cent between two policies measurable at
    this episode length.

    Args:
        env: Environment.
        policy: Policy to run.
        n_frames: Frames to attempt.
        seed: Seed for the fade trace and the symbol-error draws.
        record: If True, also return the per-decision transition arrays needed
            to train :class:`LearnedRedundancyPolicy`.

    Returns:
        ``(stats, transitions)``.  ``transitions`` is empty unless ``record``.

    Raises:
        ValueError: if ``n_frames < 1``.
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    rng = np.random.default_rng(seed)
    max_rounds = env.max_rounds
    states = env.state_sequence(n_frames * max_rounds + max_rounds, rng)
    ber = (env.ber_good, env.ber_bad)
    policy.reset()

    cursor = 0
    delivered = 0
    elapsed = 0.0
    sent_total = 0.0
    rounds_total = 0
    cost_total = 0.0
    action_counts = np.zeros(len(env.actions), dtype=int)
    prev_frame_rounds = 1
    nak_ewma = 0.0
    recent: list[int] = []
    consecutive_hard = 0

    feat_rows: list[np.ndarray] = []
    act_rows: list[int] = []
    cost_rows: list[float] = []

    for _ in range(n_frames):
        errs = 0
        n_acc = env.k
        decoded = False
        frame_rounds = 0
        frame_cost = 0.0
        decision_slice: list[tuple[int, np.ndarray, int, float]] = []
        for m in range(1, max_rounds + 1):
            obs = _features(
                m,
                n_acc - env.k,
                env,
                prev_frame_rounds,
                nak_ewma,
                float(np.mean(recent[-8:])) if recent else 1.0,
                consecutive_hard,
            )
            a_idx = policy.act(obs)
            if not 0 <= a_idx < len(env.actions):
                raise ValueError(
                    f"policy returned action index {a_idx}, outside "
                    f"[0, {len(env.actions)})"
                )
            inc = env.actions[a_idx]
            action_counts[a_idx] += 1
            bad = bool(states[min(cursor, states.size - 1)])
            cursor += 1
            p = ber[1] if bad else ber[0]
            n_sym = env.k + inc if m == 1 else inc
            errs += int(rng.binomial(n_sym, p))
            n_acc += inc
            spend = n_sym + env.rtt_symbols
            elapsed += spend
            sent_total += n_sym
            frame_cost += spend
            rounds_total += 1
            frame_rounds = m
            if record:
                decision_slice.append((m, obs, a_idx, frame_cost - spend))
            decoded = errs <= singleton_t(n_acc, env.k, env.alpha)
            # Exponentially weighted NAK rate over rounds.  This is the only
            # channel evidence the policy gets, and it is stale by construction:
            # the outcome of this round became known one feedback latency after
            # the round began.
            nak_ewma = 0.85 * nak_ewma + 0.15 * (0.0 if decoded else 1.0)
            if decoded:
                break
        if decoded:
            delivered += 1
        else:
            frame_cost += env.drop_penalty
        cost_total += frame_cost
        if record:
            for _m, obs, a_idx, spent_before in decision_slice:
                feat_rows.append(obs)
                act_rows.append(a_idx)
                cost_rows.append(frame_cost - spent_before)
        prev_frame_rounds = frame_rounds
        recent.append(frame_rounds)
        if len(recent) > 64:
            del recent[:32]
        consecutive_hard = consecutive_hard + 1 if frame_rounds > 1 else 0

    stats = EpisodeStats(
        frames=n_frames,
        delivered=delivered,
        elapsed_symbols=elapsed,
        symbols_sent=sent_total,
        rounds=rounds_total,
        cost=cost_total,
        goodput=env.k * delivered / elapsed if elapsed > 0 else 0.0,
        residual_fer=1.0 - delivered / n_frames,
        mean_rounds=rounds_total / n_frames,
        action_counts=action_counts,
    )
    transitions: dict[str, np.ndarray] = {}
    if record:
        transitions = {
            "features": np.asarray(feat_rows, dtype=float),
            "actions": np.asarray(act_rows, dtype=int),
            "cost": np.asarray(cost_rows, dtype=float),
        }
    return stats, transitions


def evaluate(
    env: FadeHarqEnv,
    policy: Policy,
    seeds: tuple[int, ...] | list[int],
    n_frames: int,
) -> dict[str, float]:
    """Score ``policy`` over ``seeds``, with a standard error across seeds.

    Args:
        env: Environment.
        policy: Policy to score.
        seeds: Seeds to average over; one episode each.
        n_frames: Frames per episode.

    Returns:
        ``{"goodput", "goodput_stderr", "residual_fer", "mean_rounds",
        "cost_per_frame", "cost_stderr", "symbols_per_frame", "n_seeds",
        "n_frames_total"}``.
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    gp = np.empty(len(seeds))
    cost = np.empty(len(seeds))
    res = np.empty(len(seeds))
    rounds = np.empty(len(seeds))
    symbols = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        stats, _ = run_episode(env, policy, n_frames, int(seed))
        gp[i] = stats.goodput
        cost[i] = stats.cost / stats.frames
        res[i] = stats.residual_fer
        rounds[i] = stats.mean_rounds
        symbols[i] = stats.symbols_sent / stats.frames
    n = len(seeds)
    return {
        "goodput": float(gp.mean()),
        "goodput_stderr": float(gp.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "residual_fer": float(res.mean()),
        "mean_rounds": float(rounds.mean()),
        "cost_per_frame": float(cost.mean()),
        "cost_stderr": float(cost.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "symbols_per_frame": float(symbols.mean()),
        "n_seeds": float(n),
        "n_frames_total": float(n * n_frames),
    }


def collect_transitions(
    env: FadeHarqEnv,
    seeds: tuple[int, ...] | list[int],
    n_frames: int,
    rng_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Roll out the uniform random behaviour policy and stack the transitions.

    Args:
        env: Environment.
        seeds: Fit seeds.  Must not overlap the tune or report sets.
        n_frames: Frames per episode.
        rng_seed: Seed for the behaviour policy's action draws.

    Returns:
        ``{"features", "actions", "cost"}`` stacked over all seeds.
    """
    behaviour = RandomPolicy(len(env.actions), np.random.default_rng(rng_seed))
    feats, acts, costs = [], [], []
    for seed in seeds:
        _, tr = run_episode(env, behaviour, n_frames, int(seed), record=True)
        feats.append(tr["features"])
        acts.append(tr["actions"])
        costs.append(tr["cost"])
    return {
        "features": np.concatenate(feats, axis=0),
        "actions": np.concatenate(acts, axis=0),
        "cost": np.concatenate(costs, axis=0),
    }


def analytic_fixed(env: FadeHarqEnv) -> tuple[AnalyticFixedPolicy, dict[str, float]]:
    """Throughput-optimal fixed schedule from the closed form, using no data.

    Evaluates :func:`arqlonghaul.harq.schedule_metrics` at the stationary
    mixture per-symbol error probability over every ``(first, later)`` pair of
    actions and returns the pair with the lowest expected cost.  The mixture BER
    is the only channel knowledge used: no episode is simulated, no seed is
    touched.

    Returns:
        ``(policy, detail)`` where ``detail`` carries the winning pair's
        analytic expected cost, goodput and residual frame error rate.
    """
    best: tuple[float, int, int, dict[str, float]] | None = None
    p = env.ber_mixture
    for i, first in enumerate(env.actions):
        for j, later in enumerate(env.actions):
            metrics = schedule_metrics(
                env.k,
                [first] + [later] * (env.max_rounds - 1),
                p,
                env.rtt_symbols,
                env.alpha,
                env.drop_penalty,
            )
            key = metrics["expected_cost"]
            if best is None or key < best[0]:
                best = (key, i, j, metrics)
    assert best is not None
    _, i, j, metrics = best
    detail = {
        "first_action": float(env.actions[i]),
        "later_action": float(env.actions[j]),
        "analytic_expected_cost": metrics["expected_cost"],
        "analytic_goodput": metrics["goodput"],
        "analytic_residual_fer": metrics["residual_fer"],
        "mixture_ber": p,
    }
    return AnalyticFixedPolicy(first_index=i, later_index=j), detail


def tune_fixed(
    env: FadeHarqEnv,
    seeds: tuple[int, ...] | list[int],
    n_frames: int,
) -> tuple[TunedFixedPolicy, dict[str, float]]:
    """Grid-search the fixed ``(first, later)`` pair on ``seeds``.

    Every pair is scored by simulation on the tune seeds and the lowest
    cost-per-frame wins.  This is the strong baseline: it gets a full pass of
    parameter selection on data disjoint from the reporting seeds, which is
    exactly the advantage the learned policy gets.
    """
    best: tuple[float, int, int] | None = None
    for i in range(len(env.actions)):
        for j in range(len(env.actions)):
            score = evaluate(
                env, AnalyticFixedPolicy(i, j, "grid"), seeds, n_frames
            )["cost_per_frame"]
            if best is None or score < best[0]:
                best = (score, i, j)
    assert best is not None
    score, i, j = best
    return TunedFixedPolicy(first_index=i, later_index=j), {
        "first_action": float(env.actions[i]),
        "later_action": float(env.actions[j]),
        "tune_cost_per_frame": score,
    }


def tune_escalating(
    env: FadeHarqEnv,
    seeds: tuple[int, ...] | list[int],
    n_frames: int,
) -> tuple[EscalatingPolicy, dict[str, float]]:
    """Choose the escalating heuristic's single free parameter on ``seeds``."""
    n = len(env.actions)
    best: tuple[float, int] | None = None
    for i in range(n):
        score = evaluate(env, EscalatingPolicy(i, n), seeds, n_frames)["cost_per_frame"]
        if best is None or score < best[0]:
            best = (score, i)
    assert best is not None
    score, i = best
    return EscalatingPolicy(base_index=i, n_actions=n), {
        "base_action": float(env.actions[i]),
        "tune_cost_per_frame": score,
    }
