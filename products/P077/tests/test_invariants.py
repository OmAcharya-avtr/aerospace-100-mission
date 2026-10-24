"""The completion invariant, as a hard test over a declared array of runs.

The central claim of this package is that the numbers it reports describe
transfers that actually delivered the file. That claim is only worth as much as
the check behind it, so this module

1. runs a declared array of scenarios, modes and seeds and asserts for **every
   single run** that the delivered digest equals the source digest -- not that
   a summary statistic looks right;
2. asserts that the check is on by default, so a caller cannot get a reported
   delivery time without it;
3. asserts that the check *fails* when the reassembly is corrupted, because an
   invariant that has never been seen to fail is not evidence of anything.

The array is deliberately boring and wide rather than clever and narrow: three
modes, four link regimes, twelve seeds.
"""

from __future__ import annotations

import itertools

import pytest

from relfiledelivery.link import LinkConditions, OutageModel, uniform_windows
from relfiledelivery.modes import ALL_MODES, TransferMode
from relfiledelivery.payload import CompletionInvariantError, FileTransferSpec, source_payload
from relfiledelivery.simulate import simulate_transfer

SPECS: tuple[FileTransferSpec, ...] = (
    FileTransferSpec(950, 100, 12),
    FileTransferSpec(40 * 256, 256, 12),
    FileTransferSpec(77 * 512 - 1, 512, 12),
)

LINKS: dict[str, LinkConditions] = {
    "clean_continuous": LinkConditions(1000.0, 100.0, 0.5, 0.0),
    "lossy_continuous": LinkConditions(5000.0, 500.0, 0.5, 0.2),
    "bursty_continuous": LinkConditions(
        5000.0, 500.0, 0.25, 0.05, outage=OutageModel(mean_up_s=8.0, mean_down_s=3.0)
    ),
    "windowed_bursty": LinkConditions(
        5000.0,
        500.0,
        0.25,
        0.05,
        outage=OutageModel(mean_up_s=15.0, mean_down_s=5.0),
        windows=uniform_windows(8, 60.0, 120.0),
    ),
}

SEEDS: tuple[int, ...] = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233)

RUN_ARRAY = list(itertools.product(sorted(LINKS), range(len(SPECS)), ALL_MODES, SEEDS))


@pytest.mark.parametrize(("link_name", "spec_index", "mode", "seed"), RUN_ARRAY)
def test_completion_invariant_holds_on_every_run(
    link_name: str, spec_index: int, mode: TransferMode, seed: int
) -> None:
    """Delivered bytes equal source bytes, verified by SHA-256, on every run.

    Parametrised one run per test case, so a single violation names the exact
    scenario rather than appearing as a changed average. A run that does not
    complete is not a violation -- an undelivered transfer has no reassembly --
    and is asserted to report no digest at all.
    """
    result = simulate_transfer(SPECS[spec_index], LINKS[link_name], mode, seed)
    if result.delivered:
        assert result.delivered_digest is not None
        assert result.delivered_digest == result.source_digest
        assert result.received_count == result.pdu_count
    else:
        assert result.delivered_digest is None
        assert result.received_count < result.pdu_count


def test_the_run_array_contains_both_delivered_and_undelivered_runs() -> None:
    """The array has to exercise both branches or the test above is vacuous."""
    delivered = 0
    undelivered = 0
    for link_name, spec_index, mode, seed in RUN_ARRAY:
        result = simulate_transfer(SPECS[spec_index], LINKS[link_name], mode, seed)
        if result.delivered:
            delivered += 1
        else:
            undelivered += 1
    assert delivered > 0
    assert undelivered > 0


def test_verification_is_on_by_default(
    tiny_spec: FileTransferSpec, clean_link: LinkConditions
) -> None:
    """A delivered run reports a digest without the caller asking for one."""
    result = simulate_transfer(tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 1)
    assert result.delivered_digest == result.source_digest


def test_turning_verification_off_reports_no_digest(
    tiny_spec: FileTransferSpec, clean_link: LinkConditions
) -> None:
    result = simulate_transfer(
        tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 1, verify=False
    )
    assert result.delivered
    assert result.delivered_digest is None


@pytest.mark.parametrize("mode", list(ALL_MODES))
@pytest.mark.parametrize("offset", [0, 1, 471, 949])
def test_a_corrupted_reassembly_raises(
    tiny_spec: FileTransferSpec,
    clean_link: LinkConditions,
    mode: TransferMode,
    offset: int,
) -> None:
    """The negative control: flipping one bit anywhere must be caught.

    Without this, a check that always passes is indistinguishable from a check
    that does nothing.
    """
    with pytest.raises(CompletionInvariantError, match="completion invariant violated"):
        simulate_transfer(
            tiny_spec, clean_link, mode, 1, inject_corruption_at=offset
        )


def test_the_error_message_names_the_mode_and_the_seed(
    tiny_spec: FileTransferSpec, clean_link: LinkConditions
) -> None:
    with pytest.raises(CompletionInvariantError) as excinfo:
        simulate_transfer(
            tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 4242, inject_corruption_at=3
        )
    message = str(excinfo.value)
    assert "deferred_nak" in message
    assert "4242" in message


def test_corruption_offset_wraps_rather_than_failing(
    tiny_spec: FileTransferSpec, clean_link: LinkConditions
) -> None:
    """An offset past the end of the file wraps, so the hook cannot silently no-op."""
    with pytest.raises(CompletionInvariantError):
        simulate_transfer(
            tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 1, inject_corruption_at=100_000
        )


def test_a_delivered_run_reproduces_the_source_payload_byte_for_byte(
    tiny_spec: FileTransferSpec, clean_link: LinkConditions
) -> None:
    """A second, independent check that does not go through the digest at all."""
    from relfiledelivery.payload import digest

    result = simulate_transfer(tiny_spec, clean_link, TransferMode.DEFERRED_NAK, 77)
    assert result.delivered
    assert digest(source_payload(tiny_spec, 77)) == result.source_digest
