"""Learned gap finder: a density model over executed scenarios that proposes sparse regions.

This is the AI component, and it exists only after the deterministic core
above it. It answers a different question from the combinatorial core: the
t-way report says *which declared tuples* are missing, while this says *where
in the normalised scenario space the executed campaign is thin*, including
inside a single declared bin, which the combinatorial measures cannot see.

Model
-----
A Gaussian kernel density estimate is fitted on the normalised embedding of the
executed scenarios (:meth:`ScenarioSpace.normalise`):

    f_hat(x) = (1 / (n h^d)) * sum_i K((x - x_i) / h),  K standard Gaussian  (4)

with bandwidth ``h`` selected by Scott's rule ``h = n^(-1/(d+4))`` on
standardised data (the default, delegated to
``sklearn.neighbors.KernelDensity(bandwidth="scott")``).

Sources: B. W. Silverman, *Density Estimation for Statistics and Data
Analysis*, Chapman and Hall, 1986, chapter 4 (multivariate kernel density
estimation and bandwidth selection); D. W. Scott, *Multivariate Density
Estimation: Theory, Practice, and Visualization*, Wiley, 1992, chapter 6, which
is the source of the ``n^(-1/(d+4))`` rule scikit-learn implements as
``bandwidth="scott"``.

Units: ``log f_hat`` is in nats per unit volume of the normalised embedding,
which is dimensionless by construction. Only *differences* between candidates
are used, so the normalising constant does not matter to the ranking.

Validity range, stated plainly
------------------------------
Kernel density estimation in ``d`` dimensions suffers the usual curse of
dimensionality: the fixture space here has 7 parameters and a 13-column
embedding, and the one-hot columns are not continuous, so the Gaussian kernel
is being applied to a mixed-type space with a single global bandwidth. That is
a known compromise and a real limitation; it is recorded in README.md and
MODEL_CARD.md rather than hidden. Nothing here claims the density estimate is
consistent on that space.

Uncertainty output
------------------
Each proposal carries three numbers, not one:

* ``log_density`` -- the mean log density over the bootstrap ensemble (or the
  single-fit value when ``n_bootstrap == 0``), in nats;
* ``log_density_sd`` -- the standard deviation of the log density across
  bootstrap replicates, in nats. Large relative to the spread of
  ``log_density`` across candidates means the ranking is not supported by the
  data the model was given;
* ``sparsity_agreement`` -- the fraction of bootstrap replicates that place
  this candidate inside the sparsest ``quantile`` of the candidate pool,
  dimensionless in [0, 1]. This is the confidence number to read: 1.0 means
  every resample agrees the point is in a sparse region.

The bootstrap is the ordinary nonparametric bootstrap over executed scenarios
(Efron, *Bootstrap Methods: Another Look at the Jackknife*, Annals of
Statistics 7(1), 1-26, 1979): resample ``n`` scenarios with replacement, refit,
rescore. It quantifies sensitivity to the executed sample, and nothing else; it
does not quantify error in the binning, the embedding or the kernel choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KernelDensity

__all__ = [
    "GapFinder",
    "GapScores",
    "Proposal",
]


@dataclass(frozen=True)
class GapScores:
    """Per-candidate scores from :meth:`GapFinder.score_candidates`.

    Attributes:
        log_density: ``(n_candidates,)`` mean log density, nats.
        log_density_sd: ``(n_candidates,)`` bootstrap standard deviation of the
            log density, nats. All-``nan`` when ``n_bootstrap == 0``.
        sparsity_agreement: ``(n_candidates,)`` fraction of replicates placing
            the candidate in the sparsest quantile, dimensionless in [0, 1].
            All-``nan`` when ``n_bootstrap == 0``.
        quantile: the sparsity quantile used, dimensionless in (0, 1].
    """

    log_density: np.ndarray
    log_density_sd: np.ndarray
    sparsity_agreement: np.ndarray
    quantile: float


@dataclass(frozen=True)
class Proposal:
    """One proposed next scenario.

    Attributes:
        candidate_index: row index into the candidate array passed in.
        log_density: estimated log density at the candidate, nats. Lower is
            sparser, and the proposals are ordered by it.
        log_density_sd: bootstrap standard deviation, nats, or ``nan``.
        sparsity_agreement: bootstrap confidence in [0, 1], or ``nan``.
        order: position in the proposed batch, 0 first.
    """

    candidate_index: int
    log_density: float
    log_density_sd: float
    sparsity_agreement: float
    order: int


class GapFinder:
    """Kernel-density gap finder over the normalised scenario embedding."""

    def __init__(
        self,
        bandwidth: str | float = "scott",
        n_bootstrap: int = 16,
        sparsity_quantile: float = 0.10,
        min_separation: float = 0.5,
        random_state: int = 0,
    ) -> None:
        """Construct a gap finder.

        Args:
            bandwidth: passed to ``KernelDensity``; ``"scott"``, ``"silverman"``
                or a positive float in normalised-embedding units.
            n_bootstrap: bootstrap replicates for the uncertainty output. 0
                disables it and makes the uncertainty fields ``nan``; the
                benchmark uses 0 for cost, the examples use the default.
            sparsity_quantile: the "sparse region" definition used by
                ``sparsity_agreement``, in (0, 1].
            min_separation: minimum distance between two proposals in one
                batch, in units of the fitted bandwidth. Non-negative; 0
                disables the rule. The default of 0.5 says that two
                proposals closer than half a bandwidth are the same query as
                far as this density model is concerned. It is a declared
                default, not a value fitted to anything.
            random_state: seed for the bootstrap resampling.
        """
        if isinstance(bandwidth, str):
            if bandwidth not in ("scott", "silverman"):
                raise ValueError("bandwidth string must be 'scott' or 'silverman'")
        elif not (float(bandwidth) > 0.0 and np.isfinite(float(bandwidth))):
            raise ValueError("numeric bandwidth must be positive and finite")
        if n_bootstrap < 0:
            raise ValueError("n_bootstrap must be non-negative")
        if not 0.0 < float(sparsity_quantile) <= 1.0:
            raise ValueError("sparsity_quantile must lie in (0, 1]")
        if not (float(min_separation) >= 0.0 and np.isfinite(float(min_separation))):
            raise ValueError("min_separation must be non-negative and finite")
        self.bandwidth = bandwidth
        self.n_bootstrap = int(n_bootstrap)
        self.sparsity_quantile = float(sparsity_quantile)
        self.min_separation = float(min_separation)
        self.random_state = int(random_state)
        self._train: np.ndarray | None = None
        self._kde: KernelDensity | None = None
        self._replicates: list[KernelDensity] = []

    @property
    def is_fitted(self) -> bool:
        """True once :meth:`fit` has run."""
        return self._kde is not None

    @property
    def fitted_bandwidth(self) -> float:
        """The bandwidth actually used, in normalised-embedding units.

        Equal to the declared value when a float was given, and the value
        scikit-learn computed from Scott's or Silverman's rule otherwise.
        """
        if self._kde is None:
            raise RuntimeError("call fit() before reading the fitted bandwidth")
        return float(getattr(self._kde, "bandwidth_", self._kde.bandwidth))

    def _make_kde(self, data: np.ndarray) -> KernelDensity:
        kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
        kde.fit(data)
        return kde

    def fit(self, executed: np.ndarray) -> GapFinder:
        """Fit the density model on the executed campaign's normalised embedding.

        Args:
            executed: ``(n_scenarios, n_columns)`` float array from
                :meth:`ScenarioSpace.normalise`.

        Returns:
            ``self``.

        Raises:
            ValueError: on a non-2D array, fewer than 2 scenarios, or a
                non-finite entry.
        """
        data = np.asarray(executed, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"executed must be 2-D, got shape {data.shape}")
        if data.shape[0] < 2:
            raise ValueError("need at least 2 executed scenarios to fit a density")
        if not np.all(np.isfinite(data)):
            raise ValueError("executed contains non-finite values")
        self._train = data
        self._kde = self._make_kde(data)
        rng = np.random.default_rng(self.random_state)
        self._replicates = []
        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, data.shape[0], size=data.shape[0])
            self._replicates.append(self._make_kde(data[idx]))
        return self

    def _check_candidates(self, candidates: np.ndarray) -> np.ndarray:
        if self._kde is None or self._train is None:
            raise RuntimeError("call fit() before scoring candidates")
        array = np.asarray(candidates, dtype=float)
        if array.ndim != 2 or array.shape[1] != self._train.shape[1]:
            raise ValueError(
                f"candidates must have shape (n, {self._train.shape[1]}), got {array.shape}"
            )
        if array.shape[0] == 0:
            raise ValueError("candidates must contain at least one row")
        if not np.all(np.isfinite(array)):
            raise ValueError("candidates contains non-finite values")
        return array

    def score_candidates(self, candidates: np.ndarray) -> GapScores:
        """Score candidates, returning the point estimate and its uncertainty.

        Args:
            candidates: ``(n_candidates, n_columns)`` normalised embedding.

        Returns:
            A :class:`GapScores`. Lower ``log_density`` is sparser.
        """
        array = self._check_candidates(candidates)
        point = np.asarray(self._kde.score_samples(array), dtype=float)
        if not self._replicates:
            nan = np.full(array.shape[0], np.nan)
            return GapScores(point, nan, nan.copy(), self.sparsity_quantile)
        stack = np.vstack([r.score_samples(array) for r in self._replicates])
        threshold = np.quantile(stack, self.sparsity_quantile, axis=1, keepdims=True)
        agreement = np.mean(stack <= threshold, axis=0)
        if len(self._replicates) > 1:
            spread = np.std(stack, axis=0, ddof=1)
        else:
            spread = np.zeros(array.shape[0])
        return GapScores(
            log_density=np.mean(stack, axis=0),
            log_density_sd=spread,
            sparsity_agreement=agreement,
            quantile=self.sparsity_quantile,
        )

    def propose(
        self,
        candidates: np.ndarray,
        n_proposals: int,
        sequential: bool = True,
        refit: bool = True,
    ) -> tuple[Proposal, ...]:
        """Propose the next batch of scenarios from a candidate pool.

        Args:
            candidates: ``(n_candidates, n_columns)`` normalised embedding of
                scenarios that could be executed next.
            n_proposals: batch size, ``1 <= n_proposals <= n_candidates``.
            sequential: when True (the default), each accepted proposal is
                added to the model's data and the density is refitted before
                the next is chosen, and the minimum-separation rule below is
                applied on top. Both exist to stop a batch crowding into one
                sparse spot; the refit is the stronger of the two when the
                executed campaign is small, and the separation rule is the
                stronger when it is large and one more point barely moves the
                density. When False, the batch is simply the ``n_proposals``
                lowest-density candidates of a single fit.
            refit: only meaningful when ``sequential`` is True. False keeps the
                minimum-separation rule but skips the refit, which is the
                ablation reported in validation/validate_benchmark_output.txt.

        Minimum-separation rule: a candidate is skipped when it lies within
        ``min_separation`` times the fitted bandwidth of a proposal already in
        the batch, because the density estimate cannot distinguish two points
        that close and proposing both buys one scenario's worth of
        information for two scenarios' worth of cost. If the rule leaves fewer
        than ``n_proposals`` candidates, the batch is filled with the lowest
        density candidates that remain and the rule is simply not met for
        those; this is a declared relaxation, not a guarantee.

        Returns:
            Proposals in selection order, sparsest first.
        """
        array = self._check_candidates(candidates)
        if not 1 <= n_proposals <= array.shape[0]:
            raise ValueError(
                f"n_proposals must satisfy 1 <= n <= {array.shape[0]}, got {n_proposals}"
            )
        scores = self.score_candidates(array)
        order = [int(i) for i in np.argsort(scores.log_density, kind="stable")]

        if sequential:
            radius = self.min_separation * float(self.fitted_bandwidth)
            base = self._train
            if base is None:  # pragma: no cover - _check_candidates guarantees this
                raise RuntimeError("call fit() before scoring candidates")
            accepted: list[int] = []
            current = scores.log_density
            while len(accepted) < n_proposals:
                ranked = [int(i) for i in np.argsort(current, kind="stable")]
                pick = None
                for index in ranked:
                    if index in accepted:
                        continue
                    if accepted and radius > 0.0:
                        gap = float(np.linalg.norm(array[accepted] - array[index], axis=1).min())
                        if gap < radius:
                            continue
                    pick = index
                    break
                if pick is None:
                    # The separation rule left nothing: fall back to the
                    # lowest-density candidate not already in the batch.
                    pick = next(i for i in ranked if i not in accepted)
                accepted.append(pick)
                if refit and len(accepted) < n_proposals:
                    refit = GapFinder(
                        bandwidth=self.bandwidth,
                        n_bootstrap=0,
                        sparsity_quantile=self.sparsity_quantile,
                        min_separation=self.min_separation,
                        random_state=self.random_state,
                    )
                    refit.fit(np.vstack([base, array[accepted]]))
                    current = refit.score_candidates(array).log_density
            order = accepted

        return tuple(
            Proposal(
                candidate_index=int(i),
                log_density=float(scores.log_density[i]),
                log_density_sd=float(scores.log_density_sd[i]),
                sparsity_agreement=float(scores.sparsity_agreement[i]),
                order=rank,
            )
            for rank, i in enumerate(order[:n_proposals])
        )

    def proposed_indices(self, candidates: np.ndarray, n_proposals: int) -> tuple[int, ...]:
        """Just the candidate indices from :meth:`propose`, in selection order."""
        return tuple(p.candidate_index for p in self.propose(candidates, n_proposals))

    def density_report(self, candidates: np.ndarray, labels: Sequence[str]) -> str:
        """Human-readable score table, used by the CLI and the examples."""
        scores = self.score_candidates(candidates)
        if len(labels) != len(scores.log_density):
            raise ValueError("labels must have one entry per candidate")
        lines = [f"{'candidate':<28}{'log_density':>12}{'sd':>10}{'confidence':>12}"]
        order = np.argsort(scores.log_density, kind="stable")
        for i in order:
            lines.append(
                f"{labels[i]:<28}{scores.log_density[i]:>12.4f}"
                f"{scores.log_density_sd[i]:>10.4f}{scores.sparsity_agreement[i]:>12.4f}"
            )
        return "\n".join(lines)
