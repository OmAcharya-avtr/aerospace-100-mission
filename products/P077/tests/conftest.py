"""Shared fixtures. Paths are anchored on ``__file__``, never on the caller's cwd."""

from __future__ import annotations

from pathlib import Path

import pytest

from relfiledelivery.dataset import fit_split, generate_table
from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows
from relfiledelivery.payload import FileTransferSpec

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tiny_spec() -> FileTransferSpec:
    """Ten units of 100 bytes plus a 12-byte header, 950 bytes of file.

    Hand-calculable throughout: ``pdu_count`` is 10, the last unit carries 50
    bytes, and one clean pass is ``950 + 10 * 12 = 1070`` wire bytes, so at
    1000 bit/s it occupies ``8 * 1070 / 1000 = 8.56`` seconds.
    """
    return FileTransferSpec(file_bytes=950, pdu_payload_bytes=100, pdu_header_bytes=12)


@pytest.fixture
def clean_link() -> LinkConditions:
    """A lossless, outage-free, continuous link at 1000 bit/s forward."""
    return LinkConditions(
        forward_rate_bps=1000.0,
        return_rate_bps=100.0,
        one_way_light_time_s=0.5,
        pdu_loss_probability=0.0,
    )


@pytest.fixture
def lossy_link() -> LinkConditions:
    """The same link with a 20 per cent independent loss probability."""
    return LinkConditions(
        forward_rate_bps=1000.0,
        return_rate_bps=100.0,
        one_way_light_time_s=0.5,
        pdu_loss_probability=0.2,
    )


@pytest.fixture
def outage_link() -> LinkConditions:
    """Correlated outages and four contact windows at 40 per cent contact duty."""
    return LinkConditions(
        forward_rate_bps=1000.0,
        return_rate_bps=100.0,
        one_way_light_time_s=0.5,
        pdu_loss_probability=0.05,
        outage=OutageModel(mean_up_s=12.0, mean_down_s=6.0),
        windows=uniform_windows(4, 40.0, 100.0),
    )


@pytest.fixture
def medium_spec() -> FileTransferSpec:
    """120 units of 512 bytes, the size the ensemble tests use."""
    return FileTransferSpec(file_bytes=120 * 512 - 7, pdu_payload_bytes=512)


@pytest.fixture(scope="session")
def small_table():
    """A 120-row-per-regime table, seed 777. Session-scoped: it simulates."""
    return generate_table(120, 777)


@pytest.fixture(scope="session")
def small_split(small_table):
    """``(train, test)`` built from disjoint seed ranges, both completed-only."""
    train = fit_split(small_table)
    test = fit_split(generate_table(80, 400_777))
    return train, test
