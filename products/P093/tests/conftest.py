"""Shared fixtures. Every campaign a test sees is SYNTHETIC and labelled so.

No test in this suite reads ``data/traces/``. The measured campaign is checked
by the scripts under ``validation/``, which refuse and exit non-zero when it is
absent; a unit test that depended on measured data would turn an absent campaign
into a test failure and hide the refusal behind it.
"""

from __future__ import annotations

import pytest

from fixtures import synthetic_campaign

#: Cycles per run for the shared fixtures. Ten cycles is 20 s per cell, which
#: gives two whole 8 s windows per cell and keeps the whole suite fast.
CYCLES = 10


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    """A written synthetic campaign directory, shared by the session."""
    return synthetic_campaign.write_campaign_dir(
        tmp_path_factory.mktemp("synthetic-campaign")
    )


@pytest.fixture(scope="session")
def campaign(synthetic_dir):
    """The shared synthetic campaign, loaded once."""
    return synthetic_campaign.load_dir(synthetic_dir)


@pytest.fixture(scope="session")
def reference_cell(campaign):
    """The reference cell of the synthetic campaign: dim 64, duty 0.50."""
    return campaign.require("w64_d050")


@pytest.fixture
def jittered_campaign(tmp_path):
    """A synthetic campaign with a non-uniform sample grid and current jitter.

    Used wherever a test must not rest on the ideal grid: the between-run
    spread, which is identically zero when four repeats are byte-identical, and
    the property that the energy over a partition sums to the energy over the
    span, which is uninteresting on a uniform grid.
    """
    return synthetic_campaign.campaign(
        tmp_path / "jittered",
        cycles=CYCLES,
        current_jitter_lsb=2,
        time_jitter_s=0.004,
        seed=4711,
    )
