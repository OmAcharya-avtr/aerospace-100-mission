"""Shared fixtures. Paths are anchored on ``__file__``, never on the cwd."""

from __future__ import annotations

from pathlib import Path

import pytest

from boardprobe import ORIN_NANO, SimulatedSource, Trace

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


@pytest.fixture
def declaration():
    """The declared Orin Nano channel table."""
    return ORIN_NANO


@pytest.fixture
def simulated():
    """A simulated backend at a fixed seed."""
    return SimulatedSource(ORIN_NANO, seed=4242)


@pytest.fixture
def idle_trace():
    """The committed trace whose content provenance is ``simulated``."""
    return Trace.from_file(FIXTURES / "orin_nano_idle_simulated_trace.json")


@pytest.fixture
def recorded_trace():
    """The committed trace whose content provenance is ``recorded``."""
    return Trace.from_file(FIXTURES / "replay_path_recorded_trace.json")


@pytest.fixture
def absent_root(tmp_path):
    """A filesystem root with no device nodes in it at all."""
    root = tmp_path / "absent-root"
    root.mkdir()
    return root
