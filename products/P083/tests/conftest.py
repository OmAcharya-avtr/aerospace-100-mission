"""Shared fixtures. Every seed and amplitude here is declared, never random."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hilverdict.certificate import CertificateModel
from hilverdict.plant import SHORT_PERIOD, LoopModel, Plant
from hilverdict.resimulate import simulate
from hilverdict.synthetic import doublet_command, generate_run

#: Short runs keep the suite fast. The physics is identical; only the horizon
#: differs, and every horizon-dependent claim is tested at both lengths.
SHORT_STEPS = 60


@pytest.fixture(scope="session")
def model() -> LoopModel:
    """The declared loop model."""
    return LoopModel()


@pytest.fixture(scope="session")
def linear_plant() -> Plant:
    """The declared plant with its cubic term removed, for exact comparisons."""
    return Plant(
        z_alpha=SHORT_PERIOD.z_alpha,
        z_delta=SHORT_PERIOD.z_delta,
        m_alpha=SHORT_PERIOD.m_alpha,
        m_alpha3=0.0,
        m_q=SHORT_PERIOD.m_q,
        m_delta=SHORT_PERIOD.m_delta,
        u_max_rad=SHORT_PERIOD.u_max_rad,
        name="linear-only",
    )


@pytest.fixture(scope="session")
def command() -> np.ndarray:
    """A declared 6 deg doublet schedule over the short horizon."""
    return doublet_command(SHORT_STEPS, math.radians(6.0))


@pytest.fixture(scope="session")
def reference(model: LoopModel, command: np.ndarray):
    """The miss-free reference trajectory over the short horizon."""
    return simulate(model, command, None)


@pytest.fixture(scope="session")
def constants(model: LoopModel) -> CertificateModel:
    """Certificate constants built for the short horizon."""
    return CertificateModel.build(model, SHORT_STEPS)


@pytest.fixture(scope="session")
def loaded_run():
    """A declared run log with a loaded timing profile."""
    return generate_run(seed=4242, profile="loaded", amplitude_deg=6.0,
                        steps=SHORT_STEPS)


@pytest.fixture(scope="session")
def long_run():
    """A declared 200-iteration run, for the claims that need a real horizon."""
    return generate_run(seed=4243, profile="loaded", amplitude_deg=6.0, steps=200)
