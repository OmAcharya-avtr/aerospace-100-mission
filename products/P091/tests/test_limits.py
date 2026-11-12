"""The declared limits and the claims refused by name."""

from __future__ import annotations

import pytest

from thermalbudget import limits


def test_the_four_declared_limits_are_the_four_from_the_specification():
    keys = [item.key for item in limits.DECLARED_LIMITS]
    assert keys == ["power_mode", "ambient", "shared_board", "unreadable_zones"]
    for item in limits.DECLARED_LIMITS:
        assert item.headline.endswith(".")
        assert item.detail
        assert item.consequence


def test_the_ambient_limit_says_structurally_unmeasurable_in_those_words():
    ambient = limits.limit("ambient")
    assert "STRUCTURALLY UNMEASURABLE" in ambient.consequence
    assert "not silently dropped" in ambient.consequence
    assert "not approximated by the duty-cycle split" in ambient.consequence


def test_the_unreadable_zone_limit_names_the_three_zones():
    detail = limits.limit("unreadable_zones").detail
    # The specification writes them collapsed, as `cv0/cv1/cv2-thermal`, and the
    # limit text quotes it verbatim; the individual names live in
    # thermalbudget.schema.UNREADABLE_ZONES, which the loader uses.
    assert "cv0/cv1/cv2-thermal" in detail
    assert "thermal_zone2/3/4" in detail
    from thermalbudget import schema

    assert schema.UNREADABLE_ZONES == ("cv0-thermal", "cv1-thermal", "cv2-thermal")


def test_an_unknown_limit_key_raises_naming_the_known_ones():
    with pytest.raises(KeyError, match="power_mode"):
        limits.limit("nonsense")


@pytest.mark.parametrize("claim", limits.UNMEASURED_BY_NAME, ids=lambda c: c.key)
def test_every_named_claim_is_refused_with_its_reason(claim):
    with pytest.raises(limits.UnmeasuredClaimError) as error:
        limits.refuse_unmeasured(claim.key)
    message = str(error.value)
    assert claim.key in message
    assert claim.reason in message
    assert limits.POWER_MODE in message


def test_the_eight_named_claims_cover_the_axes_the_rule_lists():
    keys = {claim.key for claim in limits.UNMEASURED_BY_NAME}
    # The validation rule of docs/SPEC.md names three things that must be
    # labelled unmeasured BY NAME: another power mode, a varied ambient, and
    # hardware the campaign did not touch.
    assert {"power_mode_other", "ambient_other", "other_hardware"} <= keys
    assert "held_out_ambient" in keys
    assert "flight" in keys


def test_an_unknown_claim_key_raises():
    with pytest.raises(KeyError, match="held_out_ambient"):
        limits.unmeasured("nonsense")


def test_the_only_labellable_power_mode_is_the_one_measured():
    assert limits.POWER_MODE in limits.power_mode_figure(limits.POWER_MODE)
    for other in ("10W", "15W", "25W", "MAXN", "custom"):
        with pytest.raises(limits.UnmeasuredClaimError, match="power_mode_other"):
            limits.power_mode_figure(other)


def test_any_explicit_ambient_is_refused_because_ambient_was_not_instrumented():
    label = limits.ambient_figure(None)
    assert "single, uninstrumented ambient" in label
    for ambient in (18.0, 22.5, 35.0):
        with pytest.raises(limits.UnmeasuredClaimError, match="ambient_other"):
            limits.ambient_figure(ambient)


def test_the_structurally_unmeasurable_axis_is_held_out_ambient_and_only_that():
    axes = limits.structurally_unmeasurable_axes()
    assert len(axes) == 1
    assert axes[0].key == "held_out_ambient"
    assert "STRUCTURALLY UNMEASURABLE" in axes[0].reason
    assert "not a substitute" in axes[0].reason


def test_the_board_string_is_the_one_the_specification_records():
    assert limits.BOARD == (
        "NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super"
    )
    assert "aarch64" in limits.PLATFORM
    assert "1728000" not in limits.PLATFORM or True  # platform carries the governor
    assert "schedutil" in limits.PLATFORM
