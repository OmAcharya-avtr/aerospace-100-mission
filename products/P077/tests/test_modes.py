"""Transfer-mode enum and its parser."""

from __future__ import annotations

import pytest

from relfiledelivery.modes import ALL_MODES, TransferMode, parse_mode


def test_three_modes_exist() -> None:
    assert len(ALL_MODES) == 3
    assert set(ALL_MODES) == {
        TransferMode.UNACKNOWLEDGED,
        TransferMode.DEFERRED_NAK,
        TransferMode.IMMEDIATE_NAK,
    }


def test_unacknowledged_neither_retransmits_nor_uses_the_return_link() -> None:
    mode = TransferMode.UNACKNOWLEDGED
    assert not mode.retransmits
    assert not mode.uses_return_link


@pytest.mark.parametrize(
    "mode", [TransferMode.DEFERRED_NAK, TransferMode.IMMEDIATE_NAK]
)
def test_acknowledged_modes_retransmit(mode: TransferMode) -> None:
    assert mode.retransmits
    assert mode.uses_return_link


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("deferred_nak", TransferMode.DEFERRED_NAK),
        ("deferred-nak", TransferMode.DEFERRED_NAK),
        ("  IMMEDIATE-NAK ", TransferMode.IMMEDIATE_NAK),
        ("unacknowledged", TransferMode.UNACKNOWLEDGED),
    ],
)
def test_parse_mode_accepts_both_spellings(text: str, expected: TransferMode) -> None:
    assert parse_mode(text) is expected


def test_parse_mode_is_idempotent_on_an_enum_member() -> None:
    assert parse_mode(TransferMode.DEFERRED_NAK) is TransferMode.DEFERRED_NAK


def test_parse_mode_names_the_valid_set_in_the_error() -> None:
    with pytest.raises(ValueError, match="deferred_nak"):
        parse_mode("selective-repeat")
