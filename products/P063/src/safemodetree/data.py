"""Deterministic synthetic observation generator for a declaration.

The generator is the declaration and nothing else: for a fault with declared
signature ``s``, channel ``j`` is drawn from a normal distribution with mean
``faulted_j`` when ``s_j = 1`` and ``nominal_j`` when ``s_j = 0``, and standard
deviation ``noise_scale * noise_sd_j``. The nominal class is the all-zero
signature.

Consequences, stated because they decide what the benchmark can and cannot
show:

- Two faults in the same declared ambiguity class have IDENTICAL observation
  distributions. No classifier, learned or otherwise, can separate them better
  than the class prior. That is the point: the declaration says the
  observations do not distinguish them, and the generator does not cheat by
  leaking a difference the declaration denies.
- The noise model is independent Gaussian per channel. Real symptom channels
  are correlated and often heavy-tailed; this generator is not a model of any
  spacecraft's telemetry. See README Limitations.
- ``noise_scale`` is a declared difficulty knob, dimensionless, multiplying
  every channel's declared ``noise_sd``. At ``noise_scale = 1`` the declared
  thresholds are many sigma from both the nominal and the faulted means and
  both detectors are nearly perfect, so every benchmark here is reported as a
  function of ``noise_scale`` rather than at one value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spec import SafeModeSpec

__all__ = ["NOMINAL_LABEL", "ObservationSet", "make_observations"]

#: The label used for the no-fault class.
NOMINAL_LABEL = "nominal"


@dataclass(frozen=True)
class ObservationSet:
    """A labelled set of observation vectors.

    Attributes:
        features: ``(n, n_channels)`` array of channel values, each column in
            that channel's declared unit.
        labels: ``(n,)`` array of class labels, ``NOMINAL_LABEL`` or a fault id.
        channel_ids: channel ids in column order.
        class_names: the classes present, sorted, nominal first.
        noise_scale: the declared difficulty multiplier used.
    """

    features: np.ndarray
    labels: np.ndarray
    channel_ids: tuple[str, ...]
    class_names: tuple[str, ...]
    noise_scale: float

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {self.features.shape}")
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"features has {self.features.shape[0]} rows and labels has "
                f"{self.labels.shape[0]}"
            )
        if self.features.shape[1] != len(self.channel_ids):
            raise ValueError(
                f"features has {self.features.shape[1]} columns and there are "
                f"{len(self.channel_ids)} channels"
            )

    @property
    def n_samples(self) -> int:
        """Number of observation vectors."""
        return int(self.features.shape[0])

    @property
    def is_nominal(self) -> np.ndarray:
        """Boolean mask of the no-fault samples."""
        return self.labels == NOMINAL_LABEL

    def counts(self) -> dict[str, int]:
        """Samples per class, keyed by label."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts, strict=True)}


def make_observations(
    spec: SafeModeSpec,
    *,
    n_per_fault: int = 400,
    nominal_fraction: float = 0.5,
    noise_scale: float = 1.0,
    seed: int = 0,
) -> ObservationSet:
    """Generate a labelled observation set from a declaration.

    Args:
        spec: the declaration supplying channels and signatures.
        n_per_fault: samples per declared fault, > 0.
        nominal_fraction: share of the whole set that is the no-fault class, in
            [0, 1). 0.5 gives as many nominal samples as fault samples in total.
        noise_scale: dimensionless multiplier on every channel's declared
            ``noise_sd``, > 0.
        seed: seed for ``numpy.random.default_rng``; the output is a
            deterministic function of (spec, arguments, seed).

    Returns:
        An :class:`ObservationSet`.

    Raises:
        ValueError: on a non-positive count or scale, or a fraction outside
            [0, 1).
    """
    if n_per_fault <= 0:
        raise ValueError(f"n_per_fault must be > 0, got {n_per_fault}")
    if not 0.0 <= nominal_fraction < 1.0:
        raise ValueError(f"nominal_fraction must be in [0, 1), got {nominal_fraction}")
    if noise_scale <= 0.0:
        raise ValueError(f"noise_scale must be > 0, got {noise_scale}")

    rng = np.random.default_rng(seed)
    channels = spec.observability.channels
    sd = np.array([c.noise_sd for c in channels], dtype=float) * float(noise_scale)
    nominal_mean = np.array(spec.observability.nominal_vector(), dtype=float)

    n_fault_total = n_per_fault * len(spec.fault_ids)
    n_nominal = int(round(n_fault_total * nominal_fraction / (1.0 - nominal_fraction)))

    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    if n_nominal > 0:
        blocks.append(rng.normal(nominal_mean, sd, size=(n_nominal, len(channels))))
        labels.append(np.full(n_nominal, NOMINAL_LABEL, dtype=object))
    for fault_id in spec.fault_ids:
        mean = np.array(
            spec.observability.expected_vector(spec.observability.signature(fault_id)),
            dtype=float,
        )
        blocks.append(rng.normal(mean, sd, size=(n_per_fault, len(channels))))
        labels.append(np.full(n_per_fault, fault_id, dtype=object))

    features = np.vstack(blocks)
    label_array = np.concatenate(labels)
    order = rng.permutation(features.shape[0])
    class_names = (NOMINAL_LABEL, *spec.fault_ids) if n_nominal > 0 else spec.fault_ids
    return ObservationSet(
        features=features[order],
        labels=label_array[order],
        channel_ids=spec.observability.channel_ids,
        class_names=class_names,
        noise_scale=float(noise_scale),
    )
