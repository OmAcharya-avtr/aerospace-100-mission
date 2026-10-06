"""Markov and semi-Markov channel-state fitting, with the goodness of fit.

A fitted transition matrix handed over without a goodness-of-fit statement is
not evidence of anything: maximum likelihood always returns a matrix, whether
or not the process is Markov. This module therefore pairs every fit with two
tests and one effect size.

The model
---------
The amplitude record is quantised into ``K`` channel states by ``K - 1``
ascending amplitude thresholds. With one threshold the states are {fade, good}
and the model is the two-state burst channel of E. N. Gilbert, "Capacity of a
burst-noise channel", *Bell System Technical Journal* (1960) -- Gilbert's
channel has an error process attached to the states, which this module does
not model; only the state process is fitted here.

First-order Markov maximum likelihood is the row-normalised transition count
matrix,

    P[i, j] = N(i -> j) / sum_k N(i -> k),                                (1)

and the dwell time in state ``i`` is then geometric with parameter
``1 - P[i, i]``, mean ``1 / (1 - P[i, i])`` samples.

Where the Markov assumption fails, and why
------------------------------------------
A correlated fading channel does not produce geometric dwell times. The
log-amplitude is a continuous-state Gauss-Markov process; the *continuous*
state is Markov, but the *quantised* state is not, because knowing only "below
the threshold" discards where below the threshold the process is. A fade that
has just begun sits near the threshold and is likely to recover within a few
samples; a fade that has lasted a correlation length has usually gone deeper
and will last longer. The dwell time therefore has a heavier tail and a larger
coefficient of variation than a geometric, and the quantised state sequence
has detectable second-order structure.

Both failures are measured here:

:func:`dwell_time_goodness_of_fit`
    Pearson chi-square of the observed dwell-length histogram of each state
    against the geometric implied by the fitted ``P[i, i]``.
:func:`markov_order_test`
    The standard conditional-independence test of the Markov property: for
    each middle state ``j``, the next state must be independent of the
    previous state. The statistic is the summed contingency chi-square of the
    triplet counts ``N(i, j, k)``, with
    ``dof = (K - 1)**2`` summed over the usable middle states.

On a two-million-sample record *any* such test rejects, because the sample
size is enormous and no parametric model is exactly true. The informative
number is therefore the **effect size** -- Cramer's V on the triplet table --
reported next to the p-value, not instead of it.

:class:`SemiMarkovFit`
    The honest alternative: keep the embedded jump chain (which for a
    two-state model is deterministic) and replace the geometric dwell law with
    the *empirical* dwell distribution of each state. This reproduces the
    measured mean fade duration and fade-duration distribution by
    construction, at the cost of ``sum_i max_dwell_i`` parameters instead of
    ``K(K-1)``.

Units
-----
Dwell times in samples inside this module; convert with ``/ fs`` outside it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "DwellGoodnessOfFit",
    "MarkovFit",
    "MarkovOrderTest",
    "SemiMarkovFit",
    "dwell_lengths",
    "dwell_time_goodness_of_fit",
    "fit_markov",
    "fit_semi_markov",
    "markov_order_test",
    "state_sequence",
]


def state_sequence(
    amplitude: ArrayLike, thresholds: ArrayLike, *, strict_below: bool = True
) -> NDArray[np.int64]:
    """Quantise an amplitude record into channel states.

    With ascending thresholds ``T[0] < T[1] < ... < T[K-2]``:

    * state 0   -- ``a < T[0]`` (the deepest fade state),
    * state k   -- ``T[k-1] <= a < T[k]``,
    * state K-1 -- ``a >= T[K-2]`` (the good state).

    With a single threshold this is exactly {0 = fade, 1 = good}, and state 0
    is in fade under the same rule as :func:`linkoutage.fade.below_threshold`
    with ``strict_below=True``.

    Parameters
    ----------
    amplitude:
        One-dimensional amplitude record.
    thresholds:
        Ascending, strictly increasing, at least one.
    strict_below:
        ``True`` -- boundaries belong to the upper state (``a < T`` is below).
        ``False`` -- boundaries belong to the lower state (``a <= T`` is
        below).

    Returns
    -------
    ndarray of int64
        Same length as ``amplitude``, values in ``0 .. len(thresholds)``.
    """
    a = np.asarray(amplitude, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError(f"amplitude must have at least 2 samples, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError("amplitude contains non-finite values")
    t = np.asarray(thresholds, dtype=np.float64).ravel()
    if t.size < 1:
        raise ValueError("thresholds must contain at least one threshold")
    if not np.all(np.isfinite(t)):
        raise ValueError("thresholds contains non-finite values")
    if t.size > 1 and np.any(np.diff(t) <= 0.0):
        raise ValueError(f"thresholds must be strictly increasing, got {t.tolist()}")
    side = "right" if strict_below else "left"
    return np.asarray(np.searchsorted(t, a, side=side), dtype=np.int64)


def _check_states(states: ArrayLike, n_states: int | None) -> tuple[NDArray[np.int64], int]:
    s = np.asarray(states).ravel()
    if s.size < 3:
        raise ValueError(f"states must have at least 3 samples, got {s.size}")
    si = np.asarray(s, dtype=np.int64)
    if np.any(si < 0):
        raise ValueError("states must be non-negative integers")
    k = int(si.max()) + 1 if n_states is None else int(n_states)
    if k < 2:
        raise ValueError(f"n_states must be at least 2, got {k}")
    if int(si.max()) >= k:
        raise ValueError(f"states contains value {int(si.max())} >= n_states={k}")
    return si, k


def dwell_lengths(
    states: ArrayLike, *, n_states: int | None = None, exclude_censored: bool = True
) -> tuple[list[NDArray[np.int64]], list[int]]:
    """Dwell run lengths per state, in samples.

    Parameters
    ----------
    states:
        Integer state sequence.
    n_states:
        Number of states; inferred from the data when ``None``.
    exclude_censored:
        ``True`` (default) -- the first and last run of the record are dropped,
        because their true lengths are unknown. This is the same censoring rule
        as :class:`linkoutage.fade.FadeDefinitions` with
        ``censoring='exclude'``.

    Returns
    -------
    (per_state_lengths, n_censored_per_state)
    """
    s, k = _check_states(states, n_states)
    changes = np.flatnonzero(np.diff(s) != 0) + 1
    bounds = np.concatenate(
        (np.zeros(1, dtype=np.int64), changes.astype(np.int64), np.array([s.size], dtype=np.int64))
    )
    starts = bounds[:-1]
    stops = bounds[1:]
    which = s[starts]
    lengths = stops - starts
    censored = (starts == 0) | (stops == s.size)
    out: list[NDArray[np.int64]] = []
    n_cens: list[int] = []
    for i in range(k):
        sel = which == i
        n_cens.append(int(np.count_nonzero(sel & censored)))
        keep = sel & (~censored) if exclude_censored else sel
        out.append(lengths[keep].astype(np.int64))
    return out, n_cens


@dataclass(frozen=True)
class MarkovFit:
    """First-order Markov maximum-likelihood fit of a state sequence.

    Attributes
    ----------
    transition_counts:
        ``N[i, j]``, the number of ``i -> j`` transitions, including
        self-transitions.
    transition_matrix:
        Equation (1). Rows sum to 1; a state never visited gets a uniform row
        and is flagged in ``empty_rows``.
    stationary:
        Normalised left eigenvector of ``transition_matrix`` for eigenvalue 1,
        i.e. the model's predicted long-run state occupancy.
    empirical_occupancy:
        The measured state occupancy fractions, for comparison.
    mean_dwell_samples:
        ``1 / (1 - P[i, i])`` -- the Markov model's predicted mean dwell.
    empirical_mean_dwell_samples:
        The measured mean dwell of each state, censored runs excluded.
    log_likelihood:
        Log likelihood of the observed transitions under ``P``.
    n_parameters:
        ``K * (K - 1)`` free parameters.
    """

    transition_counts: NDArray[np.int64]
    transition_matrix: NDArray[np.float64]
    stationary: NDArray[np.float64]
    empirical_occupancy: NDArray[np.float64]
    mean_dwell_samples: NDArray[np.float64]
    empirical_mean_dwell_samples: NDArray[np.float64]
    log_likelihood: float
    n_parameters: int
    n_states: int
    n_samples: int
    empty_rows: tuple[int, ...]

    @property
    def aic(self) -> float:
        return 2.0 * self.n_parameters - 2.0 * self.log_likelihood

    def report(self) -> str:
        lines = [
            f"First-order Markov fit, K={self.n_states}, N={self.n_samples} samples",
            "  transition matrix P[i,j] (row-normalised counts):",
        ]
        for i in range(self.n_states):
            row = "  ".join(f"{v:.9f}" for v in self.transition_matrix[i])
            lines.append(f"    row {i}: {row}")
        lines.append("  transition counts:")
        for i in range(self.n_states):
            lines.append(f"    row {i}: {self.transition_counts[i].tolist()}")
        lines.append(f"  model stationary occupancy : {self.stationary.tolist()!r}")
        lines.append(f"  measured occupancy         : {self.empirical_occupancy.tolist()!r}")
        lines.append(
            f"  model mean dwell (samples) : {self.mean_dwell_samples.tolist()!r}"
        )
        lines.append(
            f"  measured mean dwell        : {self.empirical_mean_dwell_samples.tolist()!r}"
        )
        lines.append(f"  log likelihood             : {self.log_likelihood!r}")
        lines.append(f"  free parameters            : {self.n_parameters}")
        lines.append(f"  AIC                        : {self.aic!r}")
        if self.empty_rows:
            lines.append(f"  WARNING never-visited states (uniform rows): {self.empty_rows}")
        return "\n".join(lines)


def fit_markov(states: ArrayLike, *, n_states: int | None = None) -> MarkovFit:
    """Fit a first-order Markov chain to an integer state sequence.

    Maximum likelihood, equation (1). The stationary distribution is obtained
    from the left eigenvector of ``P`` for eigenvalue 1; it is compared against
    the measured occupancy in :meth:`MarkovFit.report`, and the two agreeing is
    a necessary but very weak check (it holds for any row-normalised count
    matrix on a long record).
    """
    s, k = _check_states(states, n_states)
    flat = s[:-1] * k + s[1:]
    counts = np.bincount(flat, minlength=k * k).reshape(k, k).astype(np.int64)
    row = counts.sum(axis=1)
    empty = tuple(int(i) for i in np.flatnonzero(row == 0))
    p = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        if row[i] == 0:
            p[i] = 1.0 / k
        else:
            p[i] = counts[i] / row[i]

    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(counts > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
    log_like = float(np.sum(counts * log_p))

    vals, vecs = np.linalg.eig(p.T)
    idx = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, idx])
    pi = np.abs(pi)
    total = pi.sum()
    pi = pi / total if total > 0 else np.full(k, 1.0 / k)

    occupancy = np.bincount(s, minlength=k).astype(np.float64) / s.size
    diag = np.diag(p)
    with np.errstate(divide="ignore"):
        mean_dwell = np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)
    per_state, _ = dwell_lengths(s, n_states=k)
    emp_dwell = np.asarray(
        [float(d.mean()) if d.size else float("nan") for d in per_state], dtype=np.float64
    )
    return MarkovFit(
        transition_counts=counts,
        transition_matrix=p,
        stationary=pi,
        empirical_occupancy=occupancy,
        mean_dwell_samples=mean_dwell,
        empirical_mean_dwell_samples=emp_dwell,
        log_likelihood=log_like,
        n_parameters=k * (k - 1),
        n_states=k,
        n_samples=int(s.size),
        empty_rows=empty,
    )


@dataclass(frozen=True)
class DwellGoodnessOfFit:
    """Per-state chi-square of the dwell histogram against a geometric.

    Attributes
    ----------
    state:
        Which state.
    n_runs:
        Complete dwell runs used.
    n_censored:
        Runs dropped because they touched a record boundary.
    p_geometric:
        ``1 - P[i, i]`` from the Markov fit -- NOT refitted to the dwell data,
        so this tests the Markov model as fitted rather than the best possible
        geometric.
    statistic, dof, p_value, n_bins, valid:
        Pearson chi-square, binned from the tail inwards to an expected count
        of at least ``min_expected``. ``dof = n_bins - 1``; no parameter is
        estimated from the dwell histogram itself.
    mean_observed, mean_geometric, cv_observed:
        Mean dwell in samples, the geometric prediction, and the observed
        coefficient of variation. A geometric has
        ``cv = sqrt(1 - p) <= 1``; an observed ``cv`` above 1 is direct
        evidence of a heavier-than-geometric tail and needs no test at all.
    """

    state: int
    n_runs: int
    n_censored: int
    p_geometric: float
    statistic: float
    dof: int
    p_value: float
    n_bins: int
    valid: bool
    mean_observed: float
    mean_geometric: float
    cv_observed: float
    cv_geometric: float
    note: str = ""

    def line(self) -> str:
        return (
            f"state {self.state:<3d} n={self.n_runs:<8d} p_geom={self.p_geometric:.8f} "
            f"chi2={self.statistic:>12.6g} dof={self.dof:<5d} p={self.p_value:>11.4g} "
            f"mean_obs={self.mean_observed:>10.4f} mean_geom={self.mean_geometric:>10.4f} "
            f"cv_obs={self.cv_observed:.4f} cv_geom={self.cv_geometric:.4f} "
            f"{'ok' if self.valid else 'INVALID'}"
        )


def dwell_time_goodness_of_fit(
    states: ArrayLike,
    fit: MarkovFit,
    *,
    min_expected: float = 5.0,
) -> tuple[DwellGoodnessOfFit, ...]:
    """Test each state's dwell histogram against the fitted geometric.

    The geometric parameter is taken from the Markov fit (``1 - P[i, i]``) and
    is *not* re-estimated, so ``dof = n_bins - 1``: the only constraint is the
    total count. Re-estimating would subtract a further degree of freedom and
    test a weaker hypothesis (that *some* geometric fits) than the one on trial
    (that *this* Markov model fits).
    """
    s, k = _check_states(states, fit.n_states)
    per_state, n_cens = dwell_lengths(s, n_states=k)
    out: list[DwellGoodnessOfFit] = []
    for i in range(k):
        d = per_state[i]
        p_geom = float(1.0 - fit.transition_matrix[i, i])
        if d.size == 0 or not 0.0 < p_geom <= 1.0:
            out.append(
                DwellGoodnessOfFit(
                    state=i,
                    n_runs=int(d.size),
                    n_censored=n_cens[i],
                    p_geometric=p_geom,
                    statistic=float("nan"),
                    dof=0,
                    p_value=float("nan"),
                    n_bins=0,
                    valid=False,
                    mean_observed=float(d.mean()) if d.size else float("nan"),
                    mean_geometric=1.0 / p_geom if 0.0 < p_geom <= 1.0 else float("inf"),
                    cv_observed=float("nan"),
                    cv_geometric=float("nan"),
                    note="no complete dwell runs, or degenerate self-transition probability",
                )
            )
            continue
        n = int(d.size)
        kmax = int(d.max())
        observed = np.bincount(d, minlength=kmax + 1)[1:].astype(np.float64)
        kk = np.arange(1, kmax + 1, dtype=np.float64)
        expected = n * ((1.0 - p_geom) ** (kk - 1.0)) * p_geom
        expected[-1] += n * (1.0 - p_geom) ** kmax
        obs_bins: list[float] = []
        exp_bins: list[float] = []
        acc_o = 0.0
        acc_e = 0.0
        for j in range(kmax - 1, -1, -1):
            acc_o += observed[j]
            acc_e += expected[j]
            if acc_e >= min_expected:
                obs_bins.append(acc_o)
                exp_bins.append(acc_e)
                acc_o = 0.0
                acc_e = 0.0
        if acc_e > 0.0:
            if obs_bins:
                obs_bins[-1] += acc_o
                exp_bins[-1] += acc_e
            else:
                obs_bins.append(acc_o)
                exp_bins.append(acc_e)
        o = np.asarray(obs_bins[::-1], dtype=np.float64)
        e = np.asarray(exp_bins[::-1], dtype=np.float64)
        stat = float(np.sum((o - e) ** 2 / e))
        dof = int(o.size - 1)
        valid = dof >= 1
        pv = float(stats.chi2.sf(stat, dof)) if valid else float("nan")
        cv_obs = float(d.std(ddof=1) / d.mean()) if n > 1 else float("nan")
        out.append(
            DwellGoodnessOfFit(
                state=i,
                n_runs=n,
                n_censored=n_cens[i],
                p_geometric=p_geom,
                statistic=stat,
                dof=dof,
                p_value=pv,
                n_bins=int(o.size),
                valid=valid,
                mean_observed=float(d.mean()),
                mean_geometric=1.0 / p_geom,
                cv_observed=cv_obs,
                cv_geometric=math.sqrt(1.0 - p_geom),
                note="" if valid else "fewer than 2 usable bins; p-value not computable",
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class MarkovOrderTest:
    """Conditional-independence test of the first-order Markov property.

    For each middle state ``j``, the triplet counts ``N(i, j, k)`` form an
    ``K x K`` contingency table. Under a first-order Markov chain the next
    state ``k`` is independent of the previous state ``i`` given ``j``, so each
    table's Pearson chi-square is asymptotically chi-square with
    ``(K-1)**2`` degrees of freedom; the statistics add.

    Attributes
    ----------
    statistic, dof, p_value:
        Summed over the usable middle states.
    per_middle_state:
        ``(j, statistic, dof, cramers_v, n_triplets)`` for each ``j``.
    cramers_v:
        Effect size over all usable tables,
        ``sqrt(statistic / (n_triplets * min(K-1, K-1)))``. At
        ``N = 2e6`` the p-value is uninformative -- it is zero for any real
        channel -- and this is the number to quote. Roughly: below 0.05 the
        departure from Markov is small, above 0.2 it is large.
    n_triplets:
        Triplets used.
    """

    statistic: float
    dof: int
    p_value: float
    cramers_v: float
    n_triplets: int
    n_states: int
    per_middle_state: tuple[tuple[int, float, int, float, int], ...]
    note: str = ""

    def report(self) -> str:
        lines = [
            f"Markov-order (conditional independence) test, K={self.n_states}, "
            f"triplets={self.n_triplets}",
            f"  summed chi2 = {self.statistic!r}, dof = {self.dof}, p = {self.p_value!r}",
            f"  Cramer's V (effect size) = {self.cramers_v!r}",
            "  per middle state j: (j, chi2, dof, V, n)",
        ]
        for j, st, dof, v, n in self.per_middle_state:
            lines.append(f"    j={j}: chi2={st!r} dof={dof} V={v!r} n={n}")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def markov_order_test(
    states: ArrayLike, *, n_states: int | None = None, min_expected: float = 5.0
) -> MarkovOrderTest:
    """Test whether the next state is independent of the previous one.

    A middle state ``j`` is skipped when any expected cell of its table falls
    below ``min_expected``, and the skip is recorded in ``note``.
    """
    s, k = _check_states(states, n_states)
    prev = s[:-2]
    mid = s[1:-1]
    nxt = s[2:]
    flat = (prev * k + mid) * k + nxt
    triplets = np.bincount(flat, minlength=k * k * k).reshape(k, k, k).astype(np.float64)

    total_stat = 0.0
    total_dof = 0
    total_n = 0
    per: list[tuple[int, float, int, float, int]] = []
    skipped: list[int] = []
    for j in range(k):
        table = triplets[:, j, :]
        n = float(table.sum())
        if n <= 0.0:
            skipped.append(j)
            continue
        rows = table.sum(axis=1, keepdims=True)
        cols = table.sum(axis=0, keepdims=True)
        expected = rows * cols / n
        keep_rows = rows.ravel() > 0
        keep_cols = cols.ravel() > 0
        sub_e = expected[np.ix_(keep_rows, keep_cols)]
        sub_o = table[np.ix_(keep_rows, keep_cols)]
        if sub_e.size == 0 or sub_e.min() < min_expected or min(sub_e.shape) < 2:
            skipped.append(j)
            continue
        stat = float(np.sum((sub_o - sub_e) ** 2 / sub_e))
        dof = int((sub_e.shape[0] - 1) * (sub_e.shape[1] - 1))
        v = math.sqrt(stat / (n * min(sub_e.shape[0] - 1, sub_e.shape[1] - 1)))
        per.append((j, stat, dof, v, int(n)))
        total_stat += stat
        total_dof += dof
        total_n += int(n)
    if total_dof < 1:
        return MarkovOrderTest(
            statistic=total_stat,
            dof=total_dof,
            p_value=float("nan"),
            cramers_v=float("nan"),
            n_triplets=total_n,
            n_states=k,
            per_middle_state=tuple(per),
            note=(
                "no middle state had a usable contingency table at "
                f"min_expected={min_expected}; p-value not computable"
            ),
        )
    p_value = float(stats.chi2.sf(total_stat, total_dof))
    v_total = math.sqrt(total_stat / (total_n * (k - 1))) if total_n > 0 and k > 1 else float("nan")
    note = f"middle states skipped for sparsity: {skipped}" if skipped else ""
    return MarkovOrderTest(
        statistic=total_stat,
        dof=total_dof,
        p_value=p_value,
        cramers_v=v_total,
        n_triplets=total_n,
        n_states=k,
        per_middle_state=tuple(per),
        note=note,
    )


@dataclass(frozen=True)
class SemiMarkovFit:
    """Embedded jump chain plus empirical dwell distributions.

    Attributes
    ----------
    jump_matrix:
        ``J[i, j]`` for ``i != j``, the probability that the state after
        leaving ``i`` is ``j``. Diagonal is zero by construction. For ``K = 2``
        this is deterministic, which is exactly why the two-state model's whole
        content is the dwell law.
    dwell_support, dwell_pmf:
        Per state, the observed dwell lengths in samples and their empirical
        probabilities. Censored runs excluded.
    mean_dwell_samples:
        Empirical mean dwell per state.
    n_runs, n_censored:
        Per state.
    """

    jump_matrix: NDArray[np.float64]
    dwell_support: tuple[NDArray[np.int64], ...]
    dwell_pmf: tuple[NDArray[np.float64], ...]
    mean_dwell_samples: NDArray[np.float64]
    n_runs: tuple[int, ...]
    n_censored: tuple[int, ...]
    n_states: int
    n_parameters: int

    def occupancy(self) -> NDArray[np.float64]:
        """Long-run occupancy implied by the semi-Markov model.

        ``pi_i = nu_i m_i / sum_j nu_j m_j`` with ``nu`` the stationary
        distribution of the embedded jump chain and ``m`` the mean dwells.
        """
        vals, vecs = np.linalg.eig(self.jump_matrix.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        nu = np.abs(np.real(vecs[:, idx]))
        total = nu.sum()
        nu = nu / total if total > 0 else np.full(self.n_states, 1.0 / self.n_states)
        w = nu * self.mean_dwell_samples
        return np.asarray(w / w.sum(), dtype=np.float64)

    def simulate(self, n_samples: int, rng: np.random.Generator) -> NDArray[np.int64]:
        """Draw a state sequence from the fitted semi-Markov model.

        Dwell lengths are resampled from the empirical per-state distributions
        and the next state from the jump chain. The first dwell is drawn from
        the same (unconditional) dwell law rather than from its equilibrium
        length-biased version, so the very first run is slightly short; with
        ``n_samples`` much larger than the mean dwell the effect is negligible,
        and it is stated rather than hidden.
        """
        n = int(n_samples)
        if n < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
        out = np.empty(n, dtype=np.int64)
        filled = 0
        state = int(np.argmax(self.occupancy()))
        while filled < n:
            sup = self.dwell_support[state]
            pmf = self.dwell_pmf[state]
            if sup.size == 0:
                length = 1
            else:
                length = int(rng.choice(sup, p=pmf))
            take = min(length, n - filled)
            out[filled : filled + take] = state
            filled += take
            probs = self.jump_matrix[state]
            total = probs.sum()
            if total <= 0.0:
                state = (state + 1) % self.n_states
            else:
                state = int(rng.choice(self.n_states, p=probs / total))
        return out

    def report(self) -> str:
        lines = [
            f"Semi-Markov fit, K={self.n_states}",
            "  embedded jump matrix J[i,j] (i != j):",
        ]
        for i in range(self.n_states):
            lines.append(f"    row {i}: {[round(float(v), 9) for v in self.jump_matrix[i]]}")
        lines.append(f"  mean dwell (samples) : {self.mean_dwell_samples.tolist()!r}")
        lines.append(f"  complete runs        : {list(self.n_runs)!r}")
        lines.append(f"  censored runs        : {list(self.n_censored)!r}")
        lines.append(f"  implied occupancy    : {self.occupancy().tolist()!r}")
        lines.append(f"  free parameters      : {self.n_parameters}")
        lines.append(
            "  the dwell law is empirical, so the mean dwell and the whole dwell "
            "distribution are reproduced by construction; this is a fit with many "
            "parameters, not a parsimonious model"
        )
        return "\n".join(lines)


def fit_semi_markov(states: ArrayLike, *, n_states: int | None = None) -> SemiMarkovFit:
    """Fit an embedded jump chain with empirical dwell distributions."""
    s, k = _check_states(states, n_states)
    changes = np.flatnonzero(np.diff(s) != 0) + 1
    bounds = np.concatenate(
        (np.zeros(1, dtype=np.int64), changes.astype(np.int64), np.array([s.size], dtype=np.int64))
    )
    starts = bounds[:-1]
    stops = bounds[1:]
    which = s[starts]

    jump = np.zeros((k, k), dtype=np.float64)
    for i, j in zip(which[:-1], which[1:], strict=True):
        jump[i, j] += 1.0
    row = jump.sum(axis=1, keepdims=True)
    jump = np.divide(jump, row, out=np.zeros_like(jump), where=row > 0)

    per_state, n_cens = dwell_lengths(s, n_states=k)
    support: list[NDArray[np.int64]] = []
    pmf: list[NDArray[np.float64]] = []
    means = np.empty(k, dtype=np.float64)
    n_params = 0
    for i in range(k):
        d = per_state[i]
        if d.size == 0:
            support.append(np.empty(0, dtype=np.int64))
            pmf.append(np.empty(0, dtype=np.float64))
            means[i] = float("nan")
            continue
        vals, counts = np.unique(d, return_counts=True)
        support.append(vals.astype(np.int64))
        pmf.append(counts.astype(np.float64) / counts.sum())
        means[i] = float(d.mean())
        n_params += int(vals.size) - 1
    n_params += k * (k - 1)
    del starts, stops
    return SemiMarkovFit(
        jump_matrix=jump,
        dwell_support=tuple(support),
        dwell_pmf=tuple(pmf),
        mean_dwell_samples=means,
        n_runs=tuple(int(d.size) for d in per_state),
        n_censored=tuple(n_cens),
        n_states=k,
        n_parameters=n_params,
    )
