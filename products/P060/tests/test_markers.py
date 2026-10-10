"""The pytest marker helper, tested without pytest's own Item class."""

from __future__ import annotations

import pytest

from traceaudit import claims_from_marker_args, record_claims


class _Marker:
    def __init__(self, *args):
        self.args = args


class _Item:
    """Minimal stand-in for a pytest Item: iter_markers plus user_properties."""

    def __init__(self, markers, name="verifies"):
        self._markers = {name: list(markers)}
        self.user_properties: list[tuple[str, str]] = []

    def iter_markers(self, name):
        return iter(self._markers.get(name, []))


@pytest.mark.verifies("REQ-025")
def test_one_id_is_recorded():
    item = _Item([_Marker("REQ-001")])
    assert record_claims(item) == ("REQ-001",)
    assert item.user_properties == [("requirement_id", "REQ-001")]


@pytest.mark.verifies("REQ-025")
def test_several_ids_in_one_marker_are_recorded():
    item = _Item([_Marker("REQ-001", "REQ-002")])
    assert record_claims(item) == ("REQ-001", "REQ-002")
    assert len(item.user_properties) == 2


@pytest.mark.verifies("REQ-025")
def test_a_sequence_argument_is_flattened():
    item = _Item([_Marker(["REQ-001", "REQ-002"])])
    assert record_claims(item) == ("REQ-001", "REQ-002")


@pytest.mark.verifies("REQ-025")
def test_several_markers_on_one_test_are_all_recorded():
    item = _Item([_Marker("REQ-001"), _Marker("REQ-002")])
    assert record_claims(item) == ("REQ-001", "REQ-002")


@pytest.mark.verifies("REQ-025")
def test_a_repeated_id_is_recorded_once():
    item = _Item([_Marker("REQ-001"), _Marker("REQ-001")])
    assert record_claims(item) == ("REQ-001",)
    assert item.user_properties == [("requirement_id", "REQ-001")]


@pytest.mark.verifies("REQ-025")
def test_an_already_present_property_is_not_duplicated():
    item = _Item([_Marker("REQ-001")])
    item.user_properties.append(("requirement_id", "REQ-001"))
    assert record_claims(item) == ()
    assert len(item.user_properties) == 1


@pytest.mark.verifies("REQ-025")
def test_no_marker_records_nothing():
    item = _Item([])
    assert record_claims(item) == ()
    assert item.user_properties == []


@pytest.mark.verifies("REQ-025")
def test_the_marker_and_property_names_are_configurable():
    item = _Item([_Marker("SR-0001")], name="satisfies")
    assert record_claims(item, marker_name="satisfies", property_name="req") == ("SR-0001",)
    assert item.user_properties == [("req", "SR-0001")]


@pytest.mark.verifies("REQ-025")
def test_whitespace_is_stripped_and_empty_ids_dropped():
    assert claims_from_marker_args([" REQ-001 ", "", "  "]) == ("REQ-001",)


@pytest.mark.verifies("REQ-022")
def test_a_non_string_non_iterable_marker_argument_raises_type_error():
    with pytest.raises(TypeError, match="must be a string or an iterable of strings"):
        claims_from_marker_args([3])


@pytest.mark.verifies("REQ-025")
def test_this_suite_records_its_own_claims_into_junit(request):
    # The conftest hook has already run for this test, so the property is there.
    properties = dict(request.node.user_properties)
    assert properties.get("requirement_id") == "REQ-025"
