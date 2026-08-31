"""Shared helpers for the wahbakit validation scripts (seeded, deterministic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wahbakit import VectorObservations, dcm_from_quat  # noqa: E402

#: Four reference directions: three axes plus the body diagonal.  Smallest
#: pairwise separation 54.7356 deg, lambda_min of Eq. O4 = 0.3333.
WELL_CONDITIONED = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
) / np.array([[1.0], [1.0], [1.0], [np.sqrt(3.0)]])


def random_dcm(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix."""
    return dcm_from_quat(rng.normal(size=4))


def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 3) directions uniform on the sphere."""
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def transverse_basis(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane orthogonal to ``v``."""
    seed = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v, seed)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(v, e1)


def sample_observations(
    true_body: np.ndarray, sigmas: np.ndarray, rng: np.random.Generator, trials: int
) -> np.ndarray:
    """(trials, N, 3) unit vectors sampled from Eq. O1 and re-normalised."""
    out = np.empty((trials, *true_body.shape))
    for i, v in enumerate(true_body):
        e1, e2 = transverse_basis(v)
        noise = rng.normal(size=(trials, 1)) * e1 + rng.normal(size=(trials, 1)) * e2
        out[:, i, :] = v + sigmas[i] * noise
    return out / np.linalg.norm(out, axis=2)[:, :, None]


def noisy_problem(
    rng: np.random.Generator, n: int, sigma: float, reference: np.ndarray | None = None
) -> tuple[VectorObservations, np.ndarray]:
    """One observation set with Eq. O1 noise, plus the true attitude."""
    dcm = random_dcm(rng)
    ref = random_unit_vectors(rng, n) if reference is None else reference
    sigmas = np.full(ref.shape[0], max(sigma, 1e-12))
    true_body = ref @ dcm.T
    body = true_body if sigma == 0.0 else sample_observations(true_body, sigmas, rng, 1)[0]
    return VectorObservations(body, ref, sigmas=sigmas), dcm


def banner(title: str) -> None:
    """Print a script header."""
    print(title)
    print("=" * max(len(title), 82))


def verdict(name: str, value: float, tolerance: float, *, smaller_is_better: bool = True) -> bool:
    """Print a PASS/FAIL line and return whether it passed."""
    ok = value <= tolerance if smaller_is_better else value >= tolerance
    relation = "<=" if smaller_is_better else ">="
    print(f"{name:<62} {value:12.4e} {relation} {tolerance:9.2e}  {'PASS' if ok else 'FAIL'}")
    return ok
