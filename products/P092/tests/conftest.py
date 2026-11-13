"""Shared fixtures. Every campaign here is SYNTHETIC and is labelled so.

The synthetic campaign is written once per session into a temporary directory
OUTSIDE the repository, because a fixture written into the tree would be a build
artifact and release-gate check 9 fails one of those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deadlinerisk.trace import Campaign

from fixtures import synthetic_trace


@pytest.fixture(scope="session")
def synthetic_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory holding one written synthetic campaign."""
    target = tmp_path_factory.mktemp("synthetic-campaign")
    return synthetic_trace.write_synthetic_campaign(target)


@pytest.fixture(scope="session")
def campaign(synthetic_directory: Path) -> Campaign:
    """A loaded SYNTHETIC campaign. No number from this measures any board."""
    from deadlinerisk.provenance import Origin
    from deadlinerisk.trace import load_campaign

    return load_campaign(
        synthetic_directory, origin=Origin.SYNTHETIC, directory_label="synthetic"
    )


@pytest.fixture(scope="session")
def ground_truth(campaign: Campaign):
    """The synthetic campaign's long run."""
    return campaign.jobs["gt_long"]


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh PCG64 generator at a fixed seed."""
    return np.random.default_rng(920092)
