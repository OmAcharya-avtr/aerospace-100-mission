"""Reacquisition policies after loss of lock (scripted baselines + tabular Q-learning).

Problem
-------
A tracking loop has just lost lock. The pointing system must re-find the
target as quickly as possible by choosing among a small set of re-scan
strategies. This is a sequential decision problem: each attempt costs time,
grows the uncertainty (the target keeps drifting), and either succeeds or
returns the system to a new state.

Decision state (as specified)
-----------------------------
    s = (time since loss [s],
         last-known offset estimate magnitude [rad],
         uncertainty growth sigma(t) [rad])

plus one bookkeeping feature required to keep the process Markov:

         radius already searched around the last-known position [rad]

(without it, the "expanding ring" action would not be Markov). All four are
tile-coded into bins; see ``ReacqEnv.STATE_BINS``.

Actions
-------
0 ``LOCAL``  restart the spiral at the last-known position, covering a disc
             of radius ``k_local * sigma(t)``;
1 ``FULL``   restart the spiral over the full a-priori uncertainty cone,
             centred on the nominal boresight;
2 ``RING``   expanding ring: sweep the annulus from the radius already
             searched out to that radius plus ``k_ring * sigma(t)``.

Scan-time model
---------------
Each action's duration uses the uniform-coverage approximation of
``trackforge.scan``: a spiral of track spacing s scanned at along-track
speed v covers area at rate s*v, so covering an area A costs A / (s v) [s].
Disc of radius R -> pi R^2 / (s v); annulus r1->r2 -> pi (r2^2 - r1^2)/(s v).

Loss-of-lock model
------------------
Loss of lock is caused by a disturbance spike of severity u ~ U[0, 1]. The
severity sets both the observable last-known offset magnitude and the
(unobservable) true displacement scale:

    |p_lk|      = sigma_lk * (0.5 + 2 u)                                (10)
    scale(u)    = sigma_0 * (1 + kappa u^2)                             (11)
    delta_0     ~ N(0, scale(u)^2 I_2)      (displacement at loss)
    v_drift     ~ N(0, drift_rate^2 I_2)    (relative drift rate)
    p_true(t)   = p_lk + delta_0 + v_drift t

so a violent loss both leaves the LOS far off boresight and throws the
target far from the last-known position. The tracker's own uncertainty
estimate grows as

    sigma(t) = sqrt(sigma_0^2 + (drift_rate t)^2)                       (12)

Eqs. (10)-(12) are a *modelling choice* of this simulator, not a published
model; they are documented here and in DATASET_CARD.md. They are meant to
create the qualitative trade-off that real PAT reacquisition logic faces
(cheap local re-scan vs. expensive full-cone re-scan), not to reproduce any
specific mission's statistics.

Detection: an attempt succeeds if the target (evaluated at the mid-point of
the attempt) lies inside the searched region AND an independent
Bernoulli(``p_detect``) trial succeeds.

Baselines FIRST
---------------
``AlwaysFullPolicy`` and ``AlwaysLocalPolicy`` are the scripted baselines;
they are implemented and benchmarked before and against the learned policy.

Reinforcement learning
----------------------
Tabular Q-learning (Watkins & Dayan 1992, "Q-learning", Machine Learning
8(3-4); Sutton & Barto 2018, "Reinforcement Learning: An Introduction",
2nd ed., sec. 6.5):

    Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]     (13)

implemented in numpy. PyTorch is not available in this environment, so no
deep-RL variant is provided; this is recorded in README Limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ReacqConfig",
    "ReacqEnv",
    "ACTIONS",
    "AlwaysFullPolicy",
    "AlwaysLocalPolicy",
    "QLearningPolicy",
    "train_q_learning",
    "evaluate_policy",
    "compare_policies",
]

ACTIONS = ("LOCAL", "FULL", "RING")
N_ACTIONS = len(ACTIONS)


@dataclass
class ReacqConfig:
    """Reacquisition scenario parameters (all angles [rad], times [s]).

    Attributes
    ----------
    sigma0 : float
        Post-loss position uncertainty of the tracker's estimate [rad].
    sigma_lk : float
        Scale of the last-known offset from boresight [rad].
    drift_rate : float
        1-sigma relative drift rate [rad/s]; drives eq. (12).
    cone_radius : float
        A-priori uncertainty-cone radius searched by ``FULL`` [rad].
    coverage_rate : float
        s * v [rad^2/s] of the scan (track spacing x along-track speed).
    p_detect : float
        Per-attempt detection probability given geometric coverage, (0, 1].
    k_local : float
        ``LOCAL`` disc radius in units of sigma(t).
    k_ring : float
        ``RING`` annulus width in units of sigma(t).
    kappa : float
        Severity coupling in eq. (11).
    max_time : float
        Episode timeout [s]; an episode that reaches it is censored.
    """

    sigma0: float = 50e-6
    sigma_lk: float = 60e-6
    drift_rate: float = 100e-6
    cone_radius: float = 1.0e-3
    coverage_rate: float = 6.0e-7
    p_detect: float = 0.85
    k_local: float = 3.0
    k_ring: float = 2.0
    kappa: float = 10.0
    max_time: float = 30.0

    def __post_init__(self) -> None:
        for name in ("sigma0", "sigma_lk", "drift_rate", "cone_radius", "coverage_rate",
                     "k_local", "k_ring", "max_time"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and > 0, got {v!r}")
            setattr(self, name, v)
        if not 0.0 < float(self.p_detect) <= 1.0:
            raise ValueError(f"p_detect must be in (0, 1], got {self.p_detect!r}")
        if float(self.kappa) < 0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa!r}")
        self.p_detect = float(self.p_detect)
        self.kappa = float(self.kappa)


class ReacqEnv:
    """Episodic reacquisition environment with a discrete tabular state space.

    Reward is the negative time spent by each attempt (so maximising return
    minimises time-to-reacquire); reaching ``max_time`` adds a terminal
    penalty of ``-max_time``.

    Timeout convention: an attempt in progress is never aborted mid-scan, so
    the elapsed time ``env.t`` at termination can EXCEED ``max_time`` (the
    timeout is tested after the attempt completes). ``evaluate_policy``
    censors reported times at ``max_time``; ``sim.run_episode`` reports the
    raw elapsed time. Compare policies on the censored statistic.
    """

    # bin edges: (time since loss [s], |p_lk|/sigma_lk, sigma(t)/sigma0,
    #             r_searched/sigma(t))
    STATE_BINS = (
        (0.25, 0.75, 2.0, 5.0),
        (0.8, 1.5, 2.2),
        (1.5, 3.0, 6.0),
        (0.01, 2.0, 4.0),
    )
    SHAPE = tuple(len(e) + 1 for e in STATE_BINS)
    N_STATES = int(np.prod(SHAPE))

    def __init__(self, config: ReacqConfig | None = None) -> None:
        self.cfg = config or ReacqConfig()
        self._rng = np.random.default_rng(0)
        self.reset(seed=0)

    # --- state helpers -------------------------------------------------
    def _sigma(self, t: float) -> float:
        return math.sqrt(self.cfg.sigma0**2 + (self.cfg.drift_rate * t) ** 2)

    def observation(self) -> tuple[float, float, float, float]:
        """Continuous observation (t_since_loss, |p_lk|, sigma(t), r_searched)."""
        return (self.t, float(np.linalg.norm(self.p_lk)), self._sigma(self.t), self.r_searched)

    def encode(self, obs: tuple[float, float, float, float] | None = None) -> int:
        """Map an observation to a flat discrete state index."""
        if obs is None:
            obs = self.observation()
        t, off, sig, rs = obs
        feats = (t, off / self.cfg.sigma_lk, sig / self.cfg.sigma0, rs / sig)
        idx = [int(np.searchsorted(edges, f, side="right")) for edges, f in
               zip(self.STATE_BINS, feats)]
        return int(np.ravel_multi_index(idx, self.SHAPE))

    # --- episode -------------------------------------------------------
    def reset(self, seed: int | None = None) -> int:
        """Start a new loss-of-lock episode; returns the discrete state index."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        c = self.cfg
        u = float(rng.random())
        self.severity = u
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        self.p_lk = c.sigma_lk * (0.5 + 2.0 * u) * np.array([math.cos(ang), math.sin(ang)])
        scale = c.sigma0 * (1.0 + c.kappa * u**2)
        self.delta0 = rng.normal(0.0, scale, size=2)
        self.v_drift = rng.normal(0.0, c.drift_rate, size=2)
        self.t = 0.0
        self.r_searched = 0.0
        self.n_attempts = 0
        self.done = False
        return self.encode()

    def target_position(self, t: float) -> np.ndarray:
        """True target position [rad] at time ``t`` since loss (boresight frame)."""
        return self.p_lk + self.delta0 + self.v_drift * t

    def action_plan(self, action: int) -> tuple[float, np.ndarray, float, float]:
        """Return (duration [s], centre [rad], r_inner [rad], r_outer [rad])."""
        if action not in range(N_ACTIONS):
            raise ValueError(f"action must be in 0..{N_ACTIONS - 1}, got {action!r}")
        c = self.cfg
        sig = self._sigma(self.t)
        if action == 0:  # LOCAL
            r_out = c.k_local * sig
            centre, r_in = self.p_lk, 0.0
        elif action == 1:  # FULL
            r_out = c.cone_radius
            centre, r_in = np.zeros(2), 0.0
        else:  # RING
            r_in = self.r_searched
            r_out = r_in + c.k_ring * sig
            centre = self.p_lk
        duration = math.pi * (r_out**2 - r_in**2) / c.coverage_rate
        return duration, centre, r_in, r_out

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        """Execute one re-scan attempt. Returns (state, reward, done, info)."""
        if self.done:
            raise RuntimeError("episode is done; call reset()")
        duration, centre, r_in, r_out = self.action_plan(action)
        t_mid = self.t + 0.5 * duration
        pos = self.target_position(t_mid)
        d = float(np.linalg.norm(pos - centre))
        covered = r_in <= d <= r_out
        success = covered and (self._rng.random() < self.cfg.p_detect)

        self.t += duration
        self.n_attempts += 1
        if action == 0:
            self.r_searched = max(self.r_searched, r_out)
        elif action == 2:
            self.r_searched = r_out

        reward = -duration
        info = {"action": ACTIONS[action], "duration_s": duration, "covered": covered}
        if success:
            self.done = True
            info["success"] = True
            return self.encode(), reward, True, info
        if self.t >= self.cfg.max_time:
            self.done = True
            reward -= self.cfg.max_time
            info["success"] = False
            info["timeout"] = True
            return self.encode(), reward, True, info
        info["success"] = False
        return self.encode(), reward, False, info


# --------------------------------------------------------------------------
# Baselines (implemented and benchmarked FIRST)
# --------------------------------------------------------------------------
class _ScriptedPolicy:
    """Base for fixed-action scripted policies."""

    action: int
    name: str

    def act(self, state: int, rng: np.random.Generator | None = None) -> int:
        """Return the action index for ``state`` (ignored: fixed action)."""
        return self.action

    def confidence(self, state: int) -> float:
        """Scripted policies are fully deterministic -> confidence 1.0."""
        return 1.0


class AlwaysFullPolicy(_ScriptedPolicy):
    """Baseline 1: always restart the full-cone spiral."""

    action = 1
    name = "baseline-always-full"


class AlwaysLocalPolicy(_ScriptedPolicy):
    """Baseline 2: always restart a local spiral at the last-known position."""

    action = 0
    name = "baseline-always-local"


# --------------------------------------------------------------------------
# Tabular Q-learning
# --------------------------------------------------------------------------
@dataclass
class QLearningPolicy:
    """Greedy policy over a learned tabular Q-function, with a confidence output.

    Attributes
    ----------
    q : np.ndarray, shape (n_states, n_actions)
        Action-value table [s] (values are negative expected times).
    visits : np.ndarray, shape (n_states, n_actions)
        Update counts, used for the confidence output.
    """

    q: np.ndarray
    visits: np.ndarray
    name: str = "q-learning"
    min_visits: int = 30
    fallback_action: int = 1
    metadata: dict = field(default_factory=dict)

    def act(self, state: int, rng: np.random.Generator | None = None) -> int:
        """Greedy action; falls back to ``fallback_action`` on unvisited states."""
        if self.visits[state].sum() < 1:
            return self.fallback_action
        return int(np.argmax(self.q[state]))

    def confidence(self, state: int) -> float:
        """Confidence in [0, 1] for the greedy action at ``state``.

        Combines two factors, both required to be high:

        - *margin*: normalised gap between the best and second-best
          action values, ``(Q1 - Q2) / (|Q1| + eps)``, saturated at 1;
        - *support*: ``min(1, n_visits(best) / min_visits)``.

        The product is returned. It is a heuristic confidence, NOT a
        calibrated probability; see MODEL_CARD.md.
        """
        row = self.q[state]
        order = np.argsort(row)[::-1]
        best, second = row[order[0]], row[order[1]]
        margin = (best - second) / (abs(best) + 1e-12)
        margin = float(min(max(margin, 0.0), 1.0))
        support = float(min(1.0, self.visits[state, order[0]] / max(self.min_visits, 1)))
        return margin * support

    def act_with_confidence(self, state: int) -> tuple[int, float]:
        """Return (action, confidence)."""
        return self.act(state), self.confidence(state)

    def greedy_actions(self) -> np.ndarray:
        """Greedy action index per state (unvisited states -> fallback)."""
        a = np.argmax(self.q, axis=1)
        a[self.visits.sum(axis=1) < 1] = self.fallback_action
        return a


def train_q_learning(
    config: ReacqConfig | None = None,
    episodes: int = 20000,
    alpha0: float = 0.30,
    alpha_min: float = 0.02,
    gamma: float = 0.99,
    eps0: float = 1.0,
    eps_min: float = 0.05,
    seed: int = 12345,
    reward_scale: float = 1.0,
) -> QLearningPolicy:
    """Train a tabular Q-learning reacquisition policy, eq. (13).

    Exploration uses epsilon-greedy with epsilon decayed geometrically from
    ``eps0`` to ``eps_min`` over the run; the learning rate decays the same
    way from ``alpha0`` to ``alpha_min``. Rewards are negative attempt
    durations scaled by ``reward_scale``.

    Reproducibility: the whole run is driven by ``numpy.random.default_rng``
    seeded with ``seed`` (both the environment and the exploration draws),
    so repeated calls with the same arguments return bit-identical tables.

    Returns
    -------
    QLearningPolicy
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes!r}")
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
    env = ReacqEnv(config)
    q = np.zeros((env.N_STATES, N_ACTIONS))
    visits = np.zeros((env.N_STATES, N_ACTIONS), dtype=np.int64)
    rng = np.random.default_rng(seed)
    env._rng = rng  # single stream -> single seed reproduces everything
    eps_decay = (eps_min / eps0) ** (1.0 / max(episodes - 1, 1))
    alpha_decay = (alpha_min / alpha0) ** (1.0 / max(episodes - 1, 1))
    eps, alpha = eps0, alpha0
    returns = np.zeros(episodes)
    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        while True:
            a = int(rng.integers(N_ACTIONS)) if rng.random() < eps else int(np.argmax(q[s]))
            s2, r, done, _ = env.step(a)
            r *= reward_scale
            target = r if done else r + gamma * float(np.max(q[s2]))
            q[s, a] += alpha * (target - q[s, a])
            visits[s, a] += 1
            total += r
            s = s2
            if done:
                break
        returns[ep] = total
        eps *= eps_decay
        alpha *= alpha_decay
    return QLearningPolicy(
        q=q,
        visits=visits,
        metadata={
            "episodes": episodes,
            "seed": seed,
            "gamma": gamma,
            "alpha0": alpha0,
            "alpha_min": alpha_min,
            "eps0": eps0,
            "eps_min": eps_min,
            "reward_scale": reward_scale,
            "mean_return_last_10pct": float(np.mean(returns[int(0.9 * episodes):])),
        },
    )


def evaluate_policy(
    policy,
    config: ReacqConfig | None = None,
    n_episodes: int = 2000,
    seed: int = 999,
    confidence: float = 0.95,
) -> dict:
    """Monte Carlo evaluation of a reacquisition policy.

    Every policy evaluated with the same ``seed`` and ``n_episodes`` sees
    exactly the same sequence of episodes (common random numbers): the
    environment is re-seeded per episode with ``seed + i``, and detection
    draws come from that same per-episode stream, so differences between
    policies are attributable to their decisions, not to sampling noise in
    the scenario draw.

    Returns
    -------
    dict with keys: policy, n_episodes, success_rate, mean_time_s,
    ci_low_s, ci_high_s, median_time_s, p90_time_s, mean_attempts.
    Timed-out episodes are counted at ``config.max_time`` (censored) and
    reported through ``success_rate``.
    """
    from scipy.stats import norm

    cfg = config or ReacqConfig()
    env = ReacqEnv(cfg)
    times = np.zeros(n_episodes)
    ok = np.zeros(n_episodes, dtype=bool)
    attempts = np.zeros(n_episodes)
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    for i in range(n_episodes):
        s = env.reset(seed=seed + i)
        while True:
            a = policy.act(s)
            action_counts[a] += 1
            s, _, done, info = env.step(a)
            if done:
                times[i] = min(env.t, cfg.max_time)
                ok[i] = bool(info.get("success", False))
                attempts[i] = env.n_attempts
                break
    z = float(norm.ppf(0.5 + confidence / 2.0))
    mean = float(np.mean(times))
    sem = float(np.std(times, ddof=1) / math.sqrt(n_episodes))
    return {
        "policy": getattr(policy, "name", policy.__class__.__name__),
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(ok)),
        "mean_time_s": mean,
        "ci_low_s": mean - z * sem,
        "ci_high_s": mean + z * sem,
        "median_time_s": float(np.median(times)),
        "p90_time_s": float(np.percentile(times, 90)),
        "mean_attempts": float(np.mean(attempts)),
        "action_mix": {ACTIONS[i]: int(c) for i, c in enumerate(action_counts)},
    }


def compare_policies(
    policies: dict,
    config: ReacqConfig | None = None,
    n_episodes: int = 2000,
    seed: int = 999,
) -> list[dict]:
    """Evaluate several policies on identical episodes; returns one row each."""
    return [
        evaluate_policy(p, config=config, n_episodes=n_episodes, seed=seed)
        for p in policies.values()
    ]
