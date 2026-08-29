"""Deterministic synthetic training data: open-loop Shack-Hartmann slope sequences.

The learned predictor is trained on **open-loop** slope time series -- what the
sensor would report if the deformable mirror were flat. Each sequence comes
from one Kolmogorov phase screen advected by frozen flow, sampled at the frame
rate, and measured with the sensor's photon and read-noise model.

Everything is a function of an integer seed, so no data needs to be committed:
:func:`generate_dataset` reproduces a dataset bit-for-bit from its seeds. See
``DATASET_CARD.md`` for the dataset card.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .presets import ReferenceConfig, build_flow
from .sensor import ShackHartmann

__all__ = ["open_loop_slopes", "generate_dataset"]


def open_loop_slopes(
    sensor: ShackHartmann,
    flow,
    n_frames: int,
    n_photons: float | None = None,
    read_noise: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> NDArray[np.float64]:
    """Open-loop slope sequence, shape ``(n_frames, 2 * n_valid)`` [rad/m].

    Parameters
    ----------
    sensor:
        Configured :class:`~waveforge.sensor.ShackHartmann`.
    flow:
        A :class:`~waveforge.atmosphere.FrozenFlow`.
    n_frames:
        Frames to generate [-], >= 1. Must not exceed ``flow.max_frames``.
    n_photons, read_noise:
        Sensor noise parameters; ``None`` photons means noiseless.
    rng:
        Seed or generator for the sensor noise.
    """
    n_frames = int(n_frames)
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    out = np.empty((n_frames, sensor.n_slopes), dtype=np.float64)
    for k in range(n_frames):
        out[k] = sensor.measure(
            flow.frame(k), n_photons=n_photons, read_noise=read_noise, rng=rng
        )
    return out


def generate_dataset(
    sensor: ShackHartmann,
    config: ReferenceConfig,
    seeds,
    n_frames: int,
    n_photons: float | None = None,
    read_noise: float = 0.0,
    noise_seed_offset: int = 900_000,
) -> list[NDArray[np.float64]]:
    """One open-loop slope sequence per screen seed.

    Screen ``seed`` also determines the sensor-noise stream
    (``seed + noise_seed_offset``), so a dataset is fully specified by
    ``(config, seeds, n_frames, n_photons, read_noise)``.
    """
    sequences = []
    for seed in seeds:
        flow = build_flow(config, seed=int(seed))
        if n_frames > flow.max_frames:
            raise ValueError(
                f"n_frames ({n_frames}) exceeds flow.max_frames ({flow.max_frames}); "
                "enlarge config.screen_n or lower the wind speed"
            )
        sequences.append(
            open_loop_slopes(
                sensor,
                flow,
                n_frames,
                n_photons=n_photons,
                read_noise=read_noise,
                rng=int(seed) + noise_seed_offset,
            )
        )
    return sequences
