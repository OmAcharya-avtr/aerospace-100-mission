"""Three mitigations, each with a derived guarantee and an explicit cost.

1. Parameter range clamping
---------------------------
Every parameter is read back through a range check and clipped into
``[-C, +C]``. An upset can set any bit pattern, but what the forward pass sees
is clipped, so the perturbation of a single parameter obeys ``|dw| <= 2C``
unconditionally - including the exponent-bit upsets that would otherwise send a
weight to ``10**38`` or to infinity.

For the two-layer network of :mod:`bitflipsim.network`, with all parameters
clamped to ``[-C, C]`` and inputs bounded by ``Xmax = max_i |x_i|`` over the
batch, the logit deviation from a *single* parameter upset is bounded as
follows. Write ``dz2`` for the change in the pre-softmax output.

*Upset in a first-layer weight ``W1[i,j]``.* Only hidden unit ``j`` changes:
``|dz1_j| = |dw| |x_i| <= 2 C Xmax``. ReLU is 1-Lipschitz, so
``|da1_j| <= |dz1_j|``, and ``dz2_k = da1_j W2[j,k]`` with ``|W2[j,k]| <= C``,
giving ``|dz2|_inf <= 2 C**2 Xmax``.

*Upset in a first-layer bias ``b1[j]``.* As above with ``|x_i|`` replaced by 1:
``|dz2|_inf <= 2 C**2``.

*Upset in a second-layer weight ``W2[j,k]``.* ``dz2_k = dw * a1_j`` and
``a1_j <= |z1_j| <= sum_i |x_i| |W1[i,j]| + |b1_j| <= C (n_in Xmax + 1)``,
giving ``|dz2|_inf <= 2 C**2 (n_in Xmax + 1)``.

*Upset in a second-layer bias ``b2[k]``.* ``|dz2|_inf = |dw| <= 2C``.

Taking the worst of the four, the single-upset logit-deviation bound is

    B = 2 C * max( C * (n_in * Xmax + 1), 1 )                              (1)

which is what :func:`clamp_logit_bound` returns and what
``validation/validate_clamp_bound.py`` checks the measured deviation against.
Equation (1) is a bound, not an estimate: it is deliberately loose, and the
validation script reports how loose (the ratio of bound to worst measured
deviation) instead of pretending otherwise.

Cost: zero extra memory; one compare-and-select per parameter read. The latency
term is measured, not asserted, by :func:`measure_clamp_latency`.

2. Selective triplication with a majority vote
----------------------------------------------
Three copies of the protected words, voted at use. Two voters are implemented
because they have different failure modes and the difference matters:

* :func:`word_majority_vote` - compare whole words; return the value held by at
  least two copies, and flag the element as *uncorrectable* when all three
  differ. This is what a word-level TMR register file does.
* :func:`bit_majority_vote` - ``(a & b) | (a & c) | (b & c)`` on the storage
  bits. This is what a bitwise voter gate does. It never detects anything; it
  always produces a value.

The exhaustive result, established in ``validation/validate_triplication.py``
rather than claimed here, is that single upsets are always corrected by both
voters, and that the double-upset cases split by *where* the two upsets land -
same copy, different copies and same element, or different elements. The spec's
expectation that triplication "fails on the double-upset cases" holds exactly
for the case where both upsets hit the same element in two different copies;
the script enumerates all four cases and reports each.

Cost: 3x memory on the protected region, plus the voter. Accounted by
:func:`triplication_cost`.

3. Periodic reload (scrubbing)
------------------------------
The parameter block is rewritten from a golden copy every ``T_s`` seconds.
Upsets arrive as a Poisson process of rate ``lambda`` and are cleared at each
reload, so at an observation instant a uniform time ``tau`` into the current
scrub interval the expected number of live upsets is ``lambda tau``, and
averaging ``tau`` over ``[0, T_s)`` gives

    E[live upsets] = lambda * T_s / 2                                      (2)

a factor of two better than the ``lambda T_s`` an unscrubbed interval would
accumulate by its end. Equation (2) is checked against a Monte Carlo in
``validation/validate_poisson_counts.py``.

Cost: one golden copy (+100 % memory on the protected region) plus a reload
duty cycle ``t_reload / T_s`` of lost inference time. Accounted by
:func:`reload_cost`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .bitlayout import bits_of_array
from .network import MlpParameters


def clamp_parameters(params: MlpParameters, limit: float) -> MlpParameters:
    """Clip every parameter into ``[-limit, +limit]``, preserving dtype.

    A NaN read back from a corrupted word has no sign to clip towards, so the
    stated convention is NaN -> 0.0 and ``+-inf`` -> ``+-limit``. Both are
    recorded in ``MODEL_CARD.md``; neither is numpy's default behaviour.
    """
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"limit must be finite and > 0, got {limit}")
    dtype = params.values.dtype
    with np.errstate(invalid="ignore", over="ignore"):
        finite = np.nan_to_num(
            np.asarray(params.values, dtype=dtype), nan=0.0, posinf=limit, neginf=-limit
        )
        clipped = np.clip(finite, dtype.type(-limit), dtype.type(limit)).astype(dtype)
    return params.with_values(clipped)


def clamp_logit_bound(limit: float, n_in: int, max_abs_input: float) -> float:
    """Equation (1): the single-upset logit-deviation bound under clamping.

    Parameters
    ----------
    limit:
        The clamp ``C``, in parameter units (dimensionless here).
    n_in:
        Number of network inputs.
    max_abs_input:
        ``Xmax``, the largest absolute input feature over the batch the bound is
        claimed for. The bound is only valid for inputs obeying it.

    Returns
    -------
    float
        Bound on ``max_k |dz2_k|`` for any single-parameter upset.
    """
    if limit <= 0.0:
        raise ValueError(f"limit must be > 0, got {limit}")
    if n_in <= 0:
        raise ValueError(f"n_in must be > 0, got {n_in}")
    if max_abs_input < 0.0:
        raise ValueError(f"max_abs_input must be >= 0, got {max_abs_input}")
    return 2.0 * limit * max(limit * (n_in * max_abs_input + 1.0), 1.0)


def measure_clamp_latency(n_parameters: int, repeats: int = 200) -> dict[str, float]:
    """Measure the wall-clock cost of one clamp pass over ``n_parameters``.

    Measured on the host that runs it, with ``time.perf_counter``, on a single
    contended core. Reported, never asserted: the README states the host.

    Returns
    -------
    dict
        ``median_s`` and ``per_parameter_ns`` over ``repeats`` passes.
    """
    if n_parameters <= 0 or repeats <= 0:
        raise ValueError("n_parameters and repeats must be > 0")
    block = np.linspace(-2.0, 2.0, n_parameters, dtype=np.float32)
    samples = np.empty(repeats, dtype=np.float64)
    for r in range(repeats):
        t0 = time.perf_counter()
        np.clip(block, -1.0, 1.0)
        samples[r] = time.perf_counter() - t0
    median = float(np.median(samples))
    return {
        "median_s": median,
        "per_parameter_ns": median / n_parameters * 1e9,
        "repeats": float(repeats),
    }


def word_majority_vote(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Word-level TMR vote on storage bit patterns.

    Returns
    -------
    (voted, uncorrectable)
        ``voted`` has the dtype of ``a``; where all three copies differ the
        value of ``a`` is passed through and ``uncorrectable`` is ``True`` for
        that element. Comparison is on bit patterns, so two NaNs with the same
        payload compare equal, which is what a hardware comparator does.
    """
    arrays = [np.ascontiguousarray(arr) for arr in (a, b, c)]
    if len({arr.shape for arr in arrays}) != 1 or len({arr.dtype for arr in arrays}) != 1:
        raise ValueError("the three copies must share shape and dtype")
    ba, bb, bc = (bits_of_array(arr.reshape(-1)) for arr in arrays)
    voted_bits = np.where(ba == bb, ba, np.where(ba == bc, ba, np.where(bb == bc, bb, ba)))
    uncorrectable = (ba != bb) & (ba != bc) & (bb != bc)
    voted = voted_bits.view(arrays[0].dtype).reshape(arrays[0].shape).copy()
    return voted, uncorrectable.reshape(arrays[0].shape).copy()


def bit_majority_vote(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Bitwise majority ``(a & b) | (a & c) | (b & c)`` on storage bits.

    Always returns a value and never signals an error, which is precisely its
    weakness relative to :func:`word_majority_vote`.
    """
    arrays = [np.ascontiguousarray(arr) for arr in (a, b, c)]
    if len({arr.shape for arr in arrays}) != 1 or len({arr.dtype for arr in arrays}) != 1:
        raise ValueError("the three copies must share shape and dtype")
    ba, bb, bc = (bits_of_array(arr.reshape(-1)) for arr in arrays)
    voted_bits = (ba & bb) | (ba & bc) | (bb & bc)
    return voted_bits.view(arrays[0].dtype).reshape(arrays[0].shape).copy()


@dataclass(frozen=True)
class MitigationCost:
    """Memory and latency cost of one mitigation on a protected region.

    Attributes
    ----------
    scheme:
        Label.
    protected_bytes:
        Size of the region the mitigation is applied to, bytes.
    extra_memory_bytes:
        Additional storage required, bytes.
    memory_factor:
        ``(protected + extra) / protected``, dimensionless.
    latency_overhead_fraction:
        Added inference time as a fraction of the unmitigated inference time,
        dimensionless. ``nan`` where this product does not measure it.
    notes:
        What the number does and does not include.
    """

    scheme: str
    protected_bytes: int
    extra_memory_bytes: int
    memory_factor: float
    latency_overhead_fraction: float
    notes: str


def triplication_cost(
    protected_bytes: int,
    inference_time_s: float,
    vote_time_s: float,
) -> MitigationCost:
    """Cost of triplicating ``protected_bytes``: 3x memory plus the voter.

    Parameters
    ----------
    protected_bytes:
        Bytes of parameter storage triplicated.
    inference_time_s:
        Measured unmitigated inference time for the batch, seconds.
    vote_time_s:
        Measured voter time for the same batch, seconds.
    """
    if protected_bytes < 0:
        raise ValueError(f"protected_bytes must be >= 0, got {protected_bytes}")
    if inference_time_s <= 0.0:
        raise ValueError(f"inference_time_s must be > 0, got {inference_time_s}")
    if vote_time_s < 0.0:
        raise ValueError(f"vote_time_s must be >= 0, got {vote_time_s}")
    return MitigationCost(
        scheme="selective_triplication",
        protected_bytes=int(protected_bytes),
        extra_memory_bytes=2 * int(protected_bytes),
        memory_factor=3.0,
        latency_overhead_fraction=vote_time_s / inference_time_s,
        notes=(
            "3x storage on the protected region only; the voter runs once per "
            "inference in this implementation, so the latency fraction is a "
            "measured numpy figure on the build host, not a flight figure."
        ),
    )


def reload_cost(
    protected_bytes: int,
    scrub_interval_s: float,
    reload_time_s: float,
    upset_rate_per_s: float,
) -> tuple[MitigationCost, float]:
    """Cost of periodic reload, and equation (2)'s expected live upset count.

    Returns
    -------
    (MitigationCost, expected_live_upsets)
    """
    if scrub_interval_s <= 0.0:
        raise ValueError(f"scrub_interval_s must be > 0, got {scrub_interval_s}")
    if reload_time_s < 0.0:
        raise ValueError(f"reload_time_s must be >= 0, got {reload_time_s}")
    if upset_rate_per_s < 0.0:
        raise ValueError(f"upset_rate_per_s must be >= 0, got {upset_rate_per_s}")
    if reload_time_s > scrub_interval_s:
        raise ValueError("reload_time_s cannot exceed scrub_interval_s")
    cost = MitigationCost(
        scheme="periodic_reload",
        protected_bytes=int(protected_bytes),
        extra_memory_bytes=int(protected_bytes),
        memory_factor=2.0,
        latency_overhead_fraction=reload_time_s / scrub_interval_s,
        notes=(
            "one golden copy in non-volatile or protected storage; the latency "
            "figure is the duty cycle of lost inference time, not a per-call "
            "overhead."
        ),
    )
    return cost, 0.5 * upset_rate_per_s * scrub_interval_s


def clamping_cost(protected_bytes: int, clamp_latency: dict[str, float],
                  inference_time_s: float) -> MitigationCost:
    """Cost of range clamping: no extra memory, one measured clamp pass."""
    if inference_time_s <= 0.0:
        raise ValueError(f"inference_time_s must be > 0, got {inference_time_s}")
    return MitigationCost(
        scheme="parameter_range_clamping",
        protected_bytes=int(protected_bytes),
        extra_memory_bytes=0,
        memory_factor=1.0,
        latency_overhead_fraction=float(clamp_latency["median_s"]) / inference_time_s,
        notes=(
            "no redundant storage; the clamp limit C must be stored, which is "
            "one word per tensor and is not counted here. Latency measured with "
            "time.perf_counter on the build host."
        ),
    )


def expected_live_upsets(upset_rate_per_s: float, scrub_interval_s: float) -> float:
    """Equation (2): ``lambda * T_s / 2``, dimensionless count."""
    if upset_rate_per_s < 0.0 or scrub_interval_s < 0.0:
        raise ValueError("rate and interval must be >= 0")
    return 0.5 * float(upset_rate_per_s) * float(scrub_interval_s)
