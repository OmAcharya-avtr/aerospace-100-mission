"""Shared fixtures and helpers for the navbench test suite."""

from __future__ import annotations

import numpy as np
import pytest

from navbench import (
    arw_deg_per_sqrt_hour_to_si,
    attitude_trajectory,
    constant_velocity_cwna,
    quat_from_euler_zyx,
    rrw_deg_per_hour_1p5_to_si,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh seeded generator; every test that uses it gets the same stream."""
    return np.random.default_rng(20260812)


@pytest.fixture
def cv_model():
    """(F, Q, H, R, x0, P0) for a 1-D CWNA constant-velocity model, dt = 1 s."""
    f, q = constant_velocity_cwna(1.0, 0.05)
    h = np.array([[1.0, 0.0]])
    r = np.array([[9.0]])
    return f, q, h, r, np.zeros(2), np.diag([100.0, 10.0])


@pytest.fixture
def inertia() -> np.ndarray:
    """A generic asymmetric inertia tensor [kg m^2]."""
    return np.diag([10.0, 15.0, 20.0])


@pytest.fixture
def short_attitude_truth(inertia: np.ndarray):
    """A 200-step torqued attitude truth at dt = 0.5 s."""
    return attitude_trajectory(
        inertia=inertia,
        quat0=quat_from_euler_zyx(0.2, -0.1, 0.3),
        omega0=np.array([0.01, -0.02, 0.015]),
        dt=0.5,
        n_steps=200,
        torque_fn=lambda t, q, w: np.array([1e-5 * np.sin(0.01 * t), 0.0, 0.0]),
    )


@pytest.fixture
def gyro_sigmas() -> tuple[float, float]:
    """(sigma_v, sigma_u) for a tactical-grade gyro: ARW 0.05 deg/sqrt(hr), RRW 0.5."""
    return arw_deg_per_sqrt_hour_to_si(0.05), rrw_deg_per_hour_1p5_to_si(0.5)


def random_spd(n: int, rng: np.random.Generator, scale: float = 1.0) -> np.ndarray:
    """A random symmetric positive-definite ``n x n`` matrix."""
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T + n * np.eye(n))


def random_unit_quat(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random unit quaternion (scalar-first)."""
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)
