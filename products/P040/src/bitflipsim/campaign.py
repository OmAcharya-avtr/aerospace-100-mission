"""Upset campaigns: degradation versus upset rate, with its standard error.

A campaign draws a Poisson upset count for a stated flux and exposure (or for a
stated expected count), places that many distinct upsets uniformly over the bit
population, runs inference, and records the degradation. Repeating it gives a
sample whose mean has a standard error ``s / sqrt(n)``; every campaign result
carries that error, because a mean degradation quoted without one cannot be
compared with another mean degradation.

Campaign size on one core
-------------------------
Bit-flip campaigns are embarrassingly parallel and this build host has one
core, shared. The campaign size is therefore chosen from the precision wanted:
for a degradation sample with standard deviation ``s``, reaching a standard
error ``e`` needs ``n = (s / e)**2`` trials. :func:`trials_for_standard_error`
returns that number so that a campaign is sized rather than guessed, and the
README states the sizes actually used and the wall-clock they cost.

Activations as well as parameters
---------------------------------
:func:`activation_campaign` injects into the post-ReLU hidden activations
instead of the parameters. The distinction matters: a parameter upset persists
until the next scrub and affects every subsequent inference, while an
activation upset affects exactly one inference. The two campaigns therefore
answer different questions and their degradations are not comparable per upset;
they are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .criticality import accuracy, mean_total_variation
from .flux import UpsetRate, sample_upset_counts
from .injection import BitUpset, apply_upsets, sample_upsets
from .network import MlpParameters


@dataclass(frozen=True)
class CampaignResult:
    """Degradation statistics for one expected upset count.

    Attributes
    ----------
    expected_upsets:
        The Poisson mean ``mu`` the counts were drawn from, dimensionless.
    trials:
        Number of independent campaign runs.
    mean_degradation, degradation_standard_error:
        Mean of equation (1) of :mod:`bitflipsim.criticality` over the trials,
        and ``s / sqrt(n)``.
    mean_accuracy, accuracy_standard_error:
        Faulty accuracy and its standard error, dimensionless.
    golden_accuracy:
        Accuracy with no upsets.
    upset_counts:
        The drawn counts, shape ``(trials,)``.
    """

    expected_upsets: float
    trials: int
    mean_degradation: float
    degradation_standard_error: float
    mean_accuracy: float
    accuracy_standard_error: float
    golden_accuracy: float
    upset_counts: np.ndarray

    @property
    def mean_accuracy_drop(self) -> float:
        return self.golden_accuracy - self.mean_accuracy


def trials_for_standard_error(sample_std: float, target_standard_error: float) -> int:
    """``n = ceil((s / e)**2)``: trials needed for a target standard error."""
    if sample_std < 0.0:
        raise ValueError(f"sample_std must be >= 0, got {sample_std}")
    if target_standard_error <= 0.0:
        raise ValueError(f"target_standard_error must be > 0, got {target_standard_error}")
    return int(np.ceil((sample_std / target_standard_error) ** 2))


def _summarise(
    degradations: np.ndarray,
    accuracies: np.ndarray,
    golden_accuracy: float,
    expected: float,
    counts: np.ndarray,
) -> CampaignResult:
    d = np.asarray(degradations, dtype=np.float64)
    a = np.asarray(accuracies, dtype=np.float64)
    n = d.size
    ddof = 1 if n > 1 else 0
    return CampaignResult(
        expected_upsets=float(expected),
        trials=int(n),
        mean_degradation=float(d.mean()),
        degradation_standard_error=float(d.std(ddof=ddof) / np.sqrt(n)) if n > 1 else 0.0,
        mean_accuracy=float(a.mean()),
        accuracy_standard_error=float(a.std(ddof=ddof) / np.sqrt(n)) if n > 1 else 0.0,
        golden_accuracy=float(golden_accuracy),
        upset_counts=np.asarray(counts, dtype=np.int64),
    )


def parameter_campaign(
    params: MlpParameters,
    x: np.ndarray,
    y: np.ndarray,
    expected_upsets: float,
    trials: int,
    rng: np.random.Generator,
    clamp_limit: float | None = None,
) -> CampaignResult:
    """Poisson-sized parameter-upset campaign at one expected upset count.

    Parameters
    ----------
    params:
        Golden parameter block.
    x, y:
        Evaluation features and labels.
    expected_upsets:
        Poisson mean ``mu``. The per-trial count is drawn, so some trials get
        zero upsets, which is the behaviour of the physical process.
    trials:
        Number of independent runs.
    rng:
        Seeded generator; the only randomness.
    clamp_limit:
        If given, the faulty parameters are read back through a clamp to
        ``[-limit, limit]`` before inference, so the campaign measures the
        mitigated degradation.
    """
    from .mitigation import clamp_parameters  # local import avoids a cycle

    if expected_upsets < 0.0:
        raise ValueError(f"expected_upsets must be >= 0, got {expected_upsets}")
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    golden_probs = params.probabilities(x)
    golden_acc = accuracy(params.predict(x), y)
    bits = params.layout.bits_per_parameter
    counts = rng.poisson(lam=float(expected_upsets), size=int(trials))
    population = params.layout.size * bits
    counts = np.minimum(counts, population)
    degradations = np.empty(int(trials), dtype=np.float64)
    accuracies = np.empty(int(trials), dtype=np.float64)
    for t, count in enumerate(counts):
        upsets = sample_upsets(params.layout.size, bits, int(count), rng)
        faulty = params.with_values(apply_upsets(params.values, upsets))
        if clamp_limit is not None:
            faulty = clamp_parameters(faulty, clamp_limit)
        degradations[t] = mean_total_variation(golden_probs, faulty.probabilities(x))
        accuracies[t] = accuracy(faulty.predict(x), y)
    return _summarise(degradations, accuracies, golden_acc, expected_upsets, counts)


def activation_campaign(
    params: MlpParameters,
    x: np.ndarray,
    y: np.ndarray,
    expected_upsets: float,
    trials: int,
    rng: np.random.Generator,
) -> CampaignResult:
    """Poisson-sized campaign injecting into the post-ReLU hidden activations.

    The hidden activations are stored as float32 for the purpose of the upset,
    matching the parameter storage width, and the second affine layer is then
    evaluated on the damaged activations. One upset affects one inference only.
    """
    if expected_upsets < 0.0:
        raise ValueError(f"expected_upsets must be >= 0, got {expected_upsets}")
    if trials <= 0:
        raise ValueError(f"trials must be > 0, got {trials}")
    tensors = params.tensors()
    w2 = np.asarray(tensors["W2"], dtype=np.float64)
    b2 = np.asarray(tensors["b2"], dtype=np.float64)
    hidden = params.hidden_activations(x).astype(np.float32)
    golden_probs = params.probabilities(x)
    golden_acc = accuracy(params.predict(x), y)
    n_samples, n_hidden = hidden.shape
    bits = 32
    counts = rng.poisson(lam=float(expected_upsets), size=int(trials))
    counts = np.minimum(counts, n_samples * n_hidden * bits)
    degradations = np.empty(int(trials), dtype=np.float64)
    accuracies = np.empty(int(trials), dtype=np.float64)
    from .network import softmax

    for t, count in enumerate(counts):
        upsets = sample_upsets(n_samples * n_hidden, bits, int(count), rng)
        damaged = apply_upsets(hidden.reshape(-1), upsets).reshape(hidden.shape)
        with np.errstate(over="ignore", invalid="ignore"):
            logits = damaged.astype(np.float64) @ w2 + b2
        probs = softmax(logits)
        degradations[t] = mean_total_variation(golden_probs, probs)
        bad = np.isnan(probs).any(axis=1)
        preds = np.full(n_samples, -1, dtype=np.int64)
        if (~bad).any():
            preds[~bad] = np.argmax(probs[~bad], axis=1)
        accuracies[t] = accuracy(preds, y)
    return _summarise(degradations, accuracies, golden_acc, expected_upsets, counts)


def flux_campaign(
    params: MlpParameters,
    x: np.ndarray,
    y: np.ndarray,
    rate: UpsetRate,
    exposure_s: float,
    trials: int,
    rng: np.random.Generator,
) -> CampaignResult:
    """A parameter campaign specified by flux, cross-section and exposure.

    Equivalent to :func:`parameter_campaign` with ``expected_upsets =
    rate.expected_upsets(exposure_s)``, kept separate so that the flux inputs
    appear in the call and are therefore recorded in the output of the CLI and
    the examples.
    """
    expected = rate.expected_upsets(exposure_s)
    # draw once to confirm the generator path matches flux.sample_upset_counts
    _ = sample_upset_counts(rate, exposure_s, 1, np.random.default_rng(0))
    return parameter_campaign(params, x, y, expected, trials, rng)


def single_upset_sites(
    params: MlpParameters, positions: list[int] | None = None
) -> list[BitUpset]:
    """Every ``(parameter, bit)`` site, or only those at the given bit positions."""
    bits = params.layout.bits_per_parameter
    chosen = range(bits) if positions is None else positions
    for b in chosen:
        if not 0 <= b < bits:
            raise ValueError(f"bit position {b} outside [0, {bits})")
    return [BitUpset(i, b) for i in range(params.layout.size) for b in chosen]
