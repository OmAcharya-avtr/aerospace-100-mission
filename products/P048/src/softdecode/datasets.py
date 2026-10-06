"""Deterministic dataset construction for the learned corrector.

Three **disjoint** splits, each a different seed offset, so that the free
parameters of the non-learned competitors are never tuned on the data they are
reported on:

======== ================= ==============================================
split    seed              use
======== ================= ==============================================
train    ``BASE_SEED + 0`` fits the forest
tune     ``BASE_SEED + 1`` chooses alpha, L_max and the forest's clip level
report   ``BASE_SEED + 2`` every published number
======== ================= ==============================================

Each split draws its own information bits, fading, CSI estimates and noise, so
the splits share no channel realisation. The training split spans several
Eb/N0 points so the corrector sees more than one operating point; the tuning
and reporting splits are drawn at whichever Eb/N0 is being reported.

Nothing larger than 1 MB is committed. The arrays are regenerated from the
seeds below in a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import GammaGammaFading, LognormalFading
from .corrector import build_features
from .csi import MultiplicativeCsiError, StaleCsiError
from .ldpc import LdpcCode
from .simulate import OokRealisation, demap_ook, simulate_ook

BASE_SEED = 20261006
TRAIN_EBN0_DB: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0)

__all__ = ["BASE_SEED", "TRAIN_EBN0_DB", "TrainingSet", "make_training_set", "make_split"]


@dataclass(frozen=True)
class TrainingSet:
    """Features, labels and provenance of the corrector's training data."""

    features: np.ndarray
    bits: np.ndarray
    ebn0_db: tuple[float, ...]
    blocks_per_point: int
    seed: int

    @property
    def n_samples(self) -> int:
        """Number of channel-bit samples."""
        return int(self.features.shape[0])


def make_split(
    code: LdpcCode,
    ebn0_db: float,
    fading: LognormalFading | GammaGammaFading,
    error: MultiplicativeCsiError | StaleCsiError,
    blocks: int,
    split: str,
    base_seed: int = BASE_SEED,
) -> OokRealisation:
    """Draw the named split (``"train"``, ``"tune"`` or ``"report"``)."""
    offsets = {"train": 0, "tune": 1, "report": 2}
    if split not in offsets:
        raise ValueError(f"split must be one of {sorted(offsets)}, got {split!r}")
    seed = int(base_seed) + offsets[split] + 1000 * int(round(10 * ebn0_db))
    return simulate_ook(code, ebn0_db, fading, error, blocks, seed)


def make_training_set(
    code: LdpcCode,
    fading: LognormalFading | GammaGammaFading,
    error: MultiplicativeCsiError | StaleCsiError,
    blocks_per_point: int = 400,
    ebn0_db: tuple[float, ...] = TRAIN_EBN0_DB,
    base_seed: int = BASE_SEED,
) -> TrainingSet:
    """Build the training features and labels over several Eb/N0 points."""
    feats, bits = [], []
    for point in ebn0_db:
        realisation = make_split(
            code, point, fading, error, blocks_per_point, "train", base_seed
        )
        plugin = demap_ook(realisation, "plugin", fading, error)
        feats.append(build_features(realisation, plugin))
        bits.append(realisation.codeword.ravel())
    return TrainingSet(
        np.concatenate(feats, axis=0),
        np.concatenate(bits, axis=0),
        tuple(float(p) for p in ebn0_db),
        int(blocks_per_point),
        int(base_seed),
    )
