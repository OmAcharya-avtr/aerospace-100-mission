"""Classical fault isolation by a bank of matched (GLR) tests.

Method
------
Model-based isolation asks a different question from detection: *given* that
the residual has departed from ``N(0, I)``, which of the named faults explains
it best.  The classical answer is a bank of generalised-likelihood-ratio tests
matched to the residual signature of each fault (Willsky 1976, sec. 4;
Basseville and Nikiforov 1993, ch. 7; Chen and Patton 1999, ch. 3; Gertler
1998, ch. 6).

Stack the normalised residuals of a window of ``W`` samples into
``r in R^{Wm}``.  Fault ``j`` produces a known *unit* signature
``phi_j in R^{Wm}`` of unknown amplitude ``a``.  Maximising the Gaussian
log-likelihood over ``a`` gives

.. math::
    \\hat{a}_j = \\phi_j^T r, \\qquad
    \\ell_j = 2\\left[\\log p_{H_j}(r) - \\log p_{H_0}(r)\\right] = (\\phi_j^T r)^2

so each ``ell_j`` is chi-squared with one degree of freedom under ``H0``.
Isolation picks ``argmax_j ell_j``; detection declares a fault when the
maximum exceeds a threshold.  With ``J`` hypotheses tested simultaneously the
Bonferroni threshold ``chi2.isf(alpha / J, 1)`` bounds the family-wise
false-alarm probability per window by ``alpha``.

Posterior confidence
--------------------
With equal priors and the Gaussian likelihoods above, the posterior over the
fault hypotheses is ``P(j) proportional to exp(ell_j / 2)``.  That is the
classical baseline's confidence output, so the comparison against the learned
classifier's ``predict_proba`` is like for like.

Where the signatures come from
------------------------------
Each ``phi_j`` is the residual response of the *noise-free* closed loop to
that fault, averaged over several onset phases of the reference manoeuvre and
then normalised.  Two consequences are measured rather than assumed:

* for additive sensor faults the signature is exact and matches the
  closed-form ``e_ss = [I - F(I - KH)]^-1 (-FKb)`` of
  :mod:`fdiscope.analytic` (checked in ``tests/test_isolation.py``);
* for the multiplicative actuator faults the signature depends on the
  commanded torque, hence on when in the manoeuvre the fault starts.
  Averaging over onset phases is an engineering compromise and it costs
  isolation accuracy; the confusion matrix in ``validation/`` shows exactly
  how much.

Units: residuals and statistics are dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import chi2

from .faults import FaultSpec, FaultType
from .simulate import LoopConfig, simulate_loop

__all__ = [
    "SignatureBank",
    "IsolationResult",
    "fault_signature",
    "build_signature_bank",
    "glr_statistics",
    "isolate_window",
]


@dataclass(frozen=True)
class SignatureBank:
    """Unit residual signatures for a set of fault hypotheses.

    Attributes
    ----------
    window : int
        Window length ``W`` in samples.
    dim : int
        Measurement dimension ``m``.
    faults : tuple of FaultType
        Hypothesis order, fixing the column order everywhere downstream.
    matrix : ndarray, shape (J, W*m)
        Row ``j`` is the unit signature ``phi_j``.
    """

    window: int
    dim: int
    faults: tuple[FaultType, ...]
    matrix: NDArray[np.float64] = field(repr=False)

    def __post_init__(self) -> None:
        if self.matrix.shape != (len(self.faults), self.window * self.dim):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"({len(self.faults)}, {self.window * self.dim})"
            )

    def gram(self) -> NDArray[np.float64]:
        """``Phi Phi^T``: the cosine similarity between every pair of signatures.

        Off-diagonal entries near ``+/-1`` are hypotheses this bank cannot
        separate, whatever the data.
        """
        return self.matrix @ self.matrix.T


@dataclass(frozen=True)
class IsolationResult:
    """Outcome of one isolation decision.

    Attributes
    ----------
    fault : FaultType
        Winning hypothesis, or ``FaultType.NONE`` when nothing crossed the
        threshold.
    statistic : float
        The winning GLR statistic, chi-squared with 1 dof under ``H0``.
    threshold : float
        Bonferroni threshold in use.
    scores : ndarray, shape (J,)
        All GLR statistics, in bank order.
    posterior : ndarray, shape (J,)
        ``exp(ell_j / 2)`` normalised to sum to one.
    confidence : float
        Posterior mass on the winning hypothesis, in ``(0, 1]``.  Equal to
        ``nan`` when no fault is declared.
    """

    fault: FaultType
    statistic: float
    threshold: float
    scores: NDArray[np.float64] = field(repr=False)
    posterior: NDArray[np.float64] = field(repr=False)
    confidence: float = float("nan")


def fault_signature(
    config: LoopConfig,
    spec: FaultSpec,
    window: int,
    onset_steps: ArrayLike,
) -> NDArray[np.float64]:
    """Mean normalised-residual profile of a fault, flattened and normalised.

    Parameters
    ----------
    config : LoopConfig
        Loop configuration.  A noise-free copy is used, so the healthy
        residual is identically zero and the faulted residual *is* the mean
        response.
    spec : FaultSpec
        Fault to characterise.  Its ``onset_step`` is overridden by each
        entry of ``onset_steps``.
    window : int
        Number of samples from onset to keep, ``W >= 1``.
    onset_steps : array_like of int
        Onset samples to average over.  For a multiplicative actuator fault
        these should span at least one reference period, because the
        signature depends on the commanded torque at onset.

    Returns
    -------
    ndarray, shape (W*m,)
        Unit-norm signature ``phi``.

    Raises
    ------
    ValueError
        If ``window < 1``, if any onset leaves fewer than ``window`` samples,
        or if the averaged response is numerically zero (an undetectable
        fault, which cannot have a signature).
    """
    w = int(window)
    if w < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    onsets = np.atleast_1d(np.asarray(onset_steps, dtype=int)).reshape(-1)
    if onsets.size == 0:
        raise ValueError("onset_steps must not be empty")
    quiet = LoopConfig(
        plant=config.plant,
        gains=config.gains,
        n_steps=config.n_steps,
        ref_amplitude_rad=config.ref_amplitude_rad,
        ref_period_s=config.ref_period_s,
        seed=config.seed,
        noise=False,
        x0=config.x0,
    )
    acc = np.zeros((w, 2))
    for k0 in onsets:
        if int(k0) + w > quiet.n_steps:
            raise ValueError(
                f"onset {int(k0)} + window {w} exceeds n_steps {quiet.n_steps}"
            )
        run = simulate_loop(
            quiet,
            FaultSpec(
                kind=spec.kind,
                onset_step=int(k0),
                magnitude=spec.magnitude,
                channel=spec.channel,
            ),
        )
        acc += run.residual[int(k0) : int(k0) + w]
    acc /= float(onsets.size)
    flat = acc.reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm < 1e-12:
        raise ValueError(
            f"fault {spec.kind.value} produces no mean residual response; "
            "it has no signature and cannot be isolated by this method"
        )
    return flat / norm


def build_signature_bank(
    config: LoopConfig,
    specs: dict[FaultType, FaultSpec],
    window: int,
    onset_steps: ArrayLike,
) -> SignatureBank:
    """Build a :class:`SignatureBank` from one representative spec per fault.

    Parameters
    ----------
    config : LoopConfig
        Loop configuration.
    specs : dict of FaultType to FaultSpec
        One representative fault per hypothesis.  ``FaultType.NONE`` is
        rejected: the null hypothesis has no signature by construction.
    window : int
        Window length in samples.
    onset_steps : array_like of int
        Onsets to average each signature over.

    Returns
    -------
    SignatureBank
    """
    if FaultType.NONE in specs:
        raise ValueError("FaultType.NONE has no signature; remove it from specs")
    if not specs:
        raise ValueError("specs must not be empty")
    faults = tuple(specs)
    rows = [fault_signature(config, specs[f], window, onset_steps) for f in faults]
    return SignatureBank(
        window=int(window), dim=2, faults=faults, matrix=np.stack(rows, axis=0)
    )


def glr_statistics(residual_window: ArrayLike, bank: SignatureBank) -> NDArray[np.float64]:
    """GLR statistics ``(phi_j^T r)^2`` for one residual window.

    Parameters
    ----------
    residual_window : array_like, shape (W, m) or (W*m,)
        Normalised residuals of exactly one window.
    bank : SignatureBank
        Signatures to test.

    Returns
    -------
    ndarray, shape (J,)
        One chi-squared(1) statistic per hypothesis, in bank order.
    """
    r = np.asarray(residual_window, dtype=float).reshape(-1)
    expected = bank.window * bank.dim
    if r.size != expected:
        raise ValueError(f"residual window has {r.size} entries, expected {expected}")
    if not np.all(np.isfinite(r)):
        raise ValueError("residual window must be finite")
    proj = bank.matrix @ r
    return proj * proj


def isolate_window(
    residual_window: ArrayLike, bank: SignatureBank, alpha: float = 1.0e-3
) -> IsolationResult:
    """Isolate one window with a Bonferroni-corrected GLR bank.

    Parameters
    ----------
    residual_window : array_like, shape (W, m)
        Normalised residuals.
    bank : SignatureBank
        Hypotheses.
    alpha : float
        Family-wise false-alarm probability per window, in ``(0, 1)``.  The
        per-hypothesis threshold is ``chi2.isf(alpha / J, 1)``.

    Returns
    -------
    IsolationResult
    """
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
    scores = glr_statistics(residual_window, bank)
    n_hyp = scores.size
    threshold = float(chi2.isf(a / n_hyp, 1))
    shifted = 0.5 * (scores - float(np.max(scores)))
    weights = np.exp(shifted)
    posterior = weights / float(np.sum(weights))
    best = int(np.argmax(scores))
    if float(scores[best]) <= threshold:
        return IsolationResult(
            fault=FaultType.NONE,
            statistic=float(scores[best]),
            threshold=threshold,
            scores=scores,
            posterior=posterior,
            confidence=float("nan"),
        )
    return IsolationResult(
        fault=bank.faults[best],
        statistic=float(scores[best]),
        threshold=threshold,
        scores=scores,
        posterior=posterior,
        confidence=float(posterior[best]),
    )
