"""Deterministic synthetic dataset generation for the learned predictor.

No data files are committed.  Every dataset used in this package is produced by
the functions below from an :class:`~waveforge.loop.AOConfig` plus explicit
integer seeds, so a result is reproduced by re-running the generator rather
than by shipping arrays.

Split policy
------------
Training and test sequences come from **different phase-screen seeds**, i.e.
different atmospheric realisations, never from different slices of the same
frozen-flow sequence.  A frozen-flow slope series is strongly autocorrelated
(that is exactly what the predictor exploits), so a random or contiguous split
of one sequence would leak the test signal into training and inflate the score.

Noise policy
------------
:func:`make_slope_dataset` can add white Gaussian slope noise of a chosen
sigma.  Training noise and evaluation noise are drawn from *different*
generators seeded separately, so no noise realisation is ever shared between
the two.  Training the predictor at the noise level it will meet matters a
great deal — see MODEL_CARD.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .loop import AOConfig, AOSystem

__all__ = ["SlopeDataset", "make_slope_dataset"]


@dataclass(frozen=True)
class SlopeDataset:
    """Training and test slope sequences with the provenance needed to redo them.

    Attributes
    ----------
    train, test:
        Lists of ``(n_frames, n_slopes)`` arrays in rad/m.
    train_seeds, test_seeds:
        The phase-screen seeds used, in the same order as the sequences.
    n_frames:
        Frames per sequence.
    noise_sigma:
        One-sigma white slope noise added [rad/m]; ``0.0`` for noise-free.
    noise_seed:
        Seed of the generator used for the additive noise.
    config:
        The base configuration (its ``seed`` field is overridden per sequence).
    """

    train: list[np.ndarray]
    test: list[np.ndarray]
    train_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    n_frames: int
    noise_sigma: float
    noise_seed: int
    config: AOConfig

    @property
    def n_slopes(self) -> int:
        """Slope-vector length."""
        return int(self.train[0].shape[1])

    @property
    def n_train_frames(self) -> int:
        """Total training frames across all sequences."""
        return int(sum(seq.shape[0] for seq in self.train))

    @property
    def n_test_frames(self) -> int:
        """Total test frames across all sequences."""
        return int(sum(seq.shape[0] for seq in self.test))

    def summary(self) -> dict[str, object]:
        """Provenance dictionary for reports and model cards."""
        return {
            "train_seeds": list(self.train_seeds),
            "test_seeds": list(self.test_seeds),
            "n_frames_per_sequence": self.n_frames,
            "n_train_frames": self.n_train_frames,
            "n_test_frames": self.n_test_frames,
            "n_slopes": self.n_slopes,
            "noise_sigma_rad_per_m": self.noise_sigma,
            "noise_seed": self.noise_seed,
            "r0_m": self.config.r0_m,
            "wind_speed_m_s": self.config.wind_speed_m_s,
            "frame_rate_hz": self.config.frame_rate_hz,
            "d_over_r0": self.config.d_over_r0,
        }


def make_slope_dataset(
    config: AOConfig | None = None,
    *,
    n_frames: int = 400,
    train_seeds: Sequence[int] = (101, 102, 103),
    test_seeds: Sequence[int] = (901,),
    noise_sigma: float = 0.0,
    noise_seed: int = 4242,
) -> SlopeDataset:
    """Generate open-loop slope sequences for training and testing.

    Parameters
    ----------
    config:
        Base configuration; defaults to :class:`~waveforge.loop.AOConfig`.
    n_frames:
        Frames per sequence, ``>= 2``.  Must not exceed the atmosphere's
        ``max_frames`` for the configuration.
    train_seeds, test_seeds:
        Disjoint sets of phase-screen seeds.
    noise_sigma:
        White slope noise added to every sequence [rad/m], ``>= 0``.
    noise_seed:
        Seed for the additive-noise generator.

    Raises
    ------
    ValueError
        If the seed sets overlap, or any argument is out of range.
    """
    config = AOConfig() if config is None else config
    if int(n_frames) != n_frames or n_frames < 2:
        raise ValueError(f"n_frames must be an integer >= 2, got {n_frames!r}")
    if not np.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError(f"noise_sigma must be finite and >= 0, got {noise_sigma!r}")
    train_seeds = tuple(int(s) for s in train_seeds)
    test_seeds = tuple(int(s) for s in test_seeds)
    if not train_seeds or not test_seeds:
        raise ValueError("both train_seeds and test_seeds must be non-empty")
    overlap = set(train_seeds) & set(test_seeds)
    if overlap:
        raise ValueError(f"train and test seeds must be disjoint; shared seeds: {sorted(overlap)}")

    rng = np.random.default_rng(int(noise_seed))

    def build(seeds: tuple[int, ...]) -> list[np.ndarray]:
        out = []
        for seed in seeds:
            system = AOSystem(replace(config, seed=int(seed)))
            series = system.open_loop_slopes(int(n_frames))
            if noise_sigma > 0.0:
                series = series + rng.normal(0.0, noise_sigma, size=series.shape)
            out.append(series)
        return out

    return SlopeDataset(
        train=build(train_seeds),
        test=build(test_seeds),
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        n_frames=int(n_frames),
        noise_sigma=float(noise_sigma),
        noise_seed=int(noise_seed),
        config=config,
    )
