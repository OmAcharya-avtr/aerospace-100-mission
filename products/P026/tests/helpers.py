"""Shared synthetic-problem builders for the wahbakit test suite."""

from __future__ import annotations

import numpy as np

from wahbakit import VectorObservations, dcm_from_quat


def random_dcm(rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed rotation matrix, via a Gaussian quaternion."""
    return dcm_from_quat(rng.normal(size=4))


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 3) unit vectors drawn uniformly on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def well_conditioned_reference() -> np.ndarray:
    """Four reference directions with a smallest separation of 54.7 deg."""
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    return v / np.linalg.norm(v, axis=1)[:, None]


def transverse_noise(
    true_body: np.ndarray, sigmas: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Sample Eq. O1: noise in the plane orthogonal to each true direction.

    Returns re-normalised unit vectors, which is what a real unit-vector sensor
    reports.
    """
    out = np.empty_like(true_body)
    for i, v in enumerate(true_body):
        seed_axis = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(v, seed_axis)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(v, e1)
        out[i] = v + sigmas[i] * (rng.normal() * e1 + rng.normal() * e2)
    return out / np.linalg.norm(out, axis=1)[:, None]


def synthetic_problem(
    rng: np.random.Generator,
    *,
    n: int = 4,
    sigma: float = 0.0,
    reference: np.ndarray | None = None,
) -> tuple[VectorObservations, np.ndarray]:
    """Build ``(observations, true_dcm)``; ``sigma = 0`` gives noise-free data."""
    dcm = random_dcm(rng)
    ref = random_unit_vectors(rng, n) if reference is None else reference
    true_body = ref @ dcm.T
    sigmas = np.full(ref.shape[0], sigma if sigma > 0 else 1e-6)
    body = true_body if sigma == 0.0 else transverse_noise(true_body, sigmas, rng)
    return VectorObservations(body, ref, sigmas=sigmas), dcm
