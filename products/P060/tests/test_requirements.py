"""Requirement-parsing tests.

Hand-counted expectations are stated in the comments: the number of
declarations in each literal document below was counted by reading it.
"""

from __future__ import annotations

import pytest

from traceaudit import (
    Requirement,
    TraceConfig,
    merge_documents,
    parse_requirements_file,
    parse_requirements_text,
)

HEADING_DOC = """\
# Specification

## REQ-001 First requirement
Body text that also mentions REQ-001 again, mid-sentence.

### REQ-002: Second requirement
- REQ-003 Third requirement
* **REQ-004** Fourth requirement
+ REQ-005
"""
# Hand count: five declarations, REQ-001 .. REQ-005, on lines 3, 6, 7, 8, 9.


@pytest.mark.verifies("REQ-001")
def test_heading_bullet_and_bold_declaration_forms():
    doc = parse_requirements_text(HEADING_DOC, source="spec.md")
    assert [r.id for r in doc.declarations] == [
        "REQ-001", "REQ-002", "REQ-003", "REQ-004", "REQ-005",
    ]
    assert [r.line for r in doc.declarations] == [3, 6, 7, 8, 9]


@pytest.mark.verifies("REQ-001")
def test_titles_are_the_remainder_of_the_declaration_line():
    doc = parse_requirements_text(HEADING_DOC, source="spec.md")
    titles = {r.id: r.title for r in doc.declarations}
    assert titles["REQ-001"] == "First requirement"
    assert titles["REQ-002"] == "Second requirement"
    assert titles["REQ-005"] == ""


@pytest.mark.verifies("REQ-001")
def test_source_and_location_are_recorded():
    doc = parse_requirements_text(HEADING_DOC, source="docs/spec.md")
    first = doc.declarations[0]
    assert first.source == "docs/spec.md"
    assert first.location == "docs/spec.md:3"


@pytest.mark.verifies("REQ-002")
def test_mid_sentence_mention_is_not_a_declaration():
    text = "The monitor shall, unlike REQ-009, do nothing.\nSee also REQ-010 below.\n"
    doc = parse_requirements_text(text, source="s.md")
    assert doc.declarations == ()


@pytest.mark.verifies("REQ-002")
def test_indented_declaration_is_accepted_but_mid_line_is_not():
    text = "    REQ-007 indented four spaces\nprefix REQ-008 not at line start\n"
    doc = parse_requirements_text(text, source="s.md")
    assert [r.id for r in doc.declarations] == ["REQ-007"]


@pytest.mark.verifies("REQ-003")
def test_fenced_code_block_is_skipped_by_default():
    text = "## REQ-001 Real\n\n```\n## REQ-002 Inside a fence\n```\n\n## REQ-003 Real\n"
    doc = parse_requirements_text(text, source="s.md")
    assert [r.id for r in doc.declarations] == ["REQ-001", "REQ-003"]


@pytest.mark.verifies("REQ-003")
def test_fenced_code_block_is_parsed_when_skipping_is_off():
    text = "## REQ-001 Real\n\n```\n## REQ-002 Inside a fence\n```\n"
    cfg = TraceConfig(skip_code_fences=False)
    doc = parse_requirements_text(text, source="s.md", config=cfg)
    assert [r.id for r in doc.declarations] == ["REQ-001", "REQ-002"]


@pytest.mark.verifies("REQ-003")
def test_tilde_fence_is_also_recognised():
    text = "## REQ-001 Real\n~~~\n## REQ-002 Hidden\n~~~\n## REQ-003 Real\n"
    doc = parse_requirements_text(text, source="s.md")
    assert [r.id for r in doc.declarations] == ["REQ-001", "REQ-003"]


@pytest.mark.verifies("REQ-004")
def test_duplicate_declarations_are_retained_and_grouped():
    text = "## REQ-001 One\n## REQ-002 Two\n## REQ-001 One again\n"
    doc = parse_requirements_text(text, source="s.md")
    assert len(doc.declarations) == 3
    assert doc.ids == ("REQ-001", "REQ-002")
    duplicates = doc.duplicates
    assert set(duplicates) == {"REQ-001"}
    assert [d.line for d in duplicates["REQ-001"]] == [1, 3]


@pytest.mark.verifies("REQ-004")
def test_first_returns_the_earliest_declaration():
    text = "## REQ-001 One\n## REQ-001 One again\n"
    doc = parse_requirements_text(text, source="s.md")
    assert doc.first("REQ-001").line == 1
    assert doc.first("REQ-999") is None


@pytest.mark.verifies("REQ-004")
def test_merge_documents_keeps_cross_document_duplicates():
    a = parse_requirements_text("## REQ-001 A\n", source="a.md")
    b = parse_requirements_text("## REQ-001 B\n## REQ-002 C\n", source="b.md")
    merged = merge_documents([a, b])
    assert merged.sources == ("a.md", "b.md")
    assert merged.ids == ("REQ-001", "REQ-002")
    assert len(merged.duplicates["REQ-001"]) == 2


@pytest.mark.verifies("REQ-021")
def test_custom_id_pattern_parses_a_different_convention():
    cfg = TraceConfig(id_pattern=r"SRS-[A-Z]{2}-\d{2}")
    doc = parse_requirements_text("## SRS-GN-07 Attitude\n## REQ-001 Ignored\n",
                                  source="s.md", config=cfg)
    assert [r.id for r in doc.declarations] == ["SRS-GN-07"]


@pytest.mark.verifies("REQ-021")
def test_custom_declaration_pattern_overrides_the_template():
    cfg = TraceConfig(declaration_pattern=r"^\|\s*(?P<id>REQ-\d{3})\s*\|\s*(?P<title>.*?)\s*\|$")
    doc = parse_requirements_text("| REQ-010 | Table row |\n## REQ-011 Heading\n",
                                  source="s.md", config=cfg)
    assert [(r.id, r.title) for r in doc.declarations] == [("REQ-010", "Table row")]


@pytest.mark.verifies("REQ-022")
def test_missing_requirements_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="requirements document not found"):
        parse_requirements_file(tmp_path / "absent.md")


@pytest.mark.verifies("REQ-022")
def test_parse_requirements_file_reads_from_disk(tmp_path):
    path = tmp_path / "REQUIREMENTS.md"
    path.write_text("## REQ-001 On disk\n", encoding="utf-8")
    doc = parse_requirements_file(path)
    assert doc.declarations[0] == Requirement(
        id="REQ-001", title="On disk", source=str(path), line=1
    )


@pytest.mark.verifies("REQ-022")
def test_relative_to_strips_the_leading_directories(tmp_path):
    (tmp_path / "docs").mkdir()
    path = tmp_path / "docs" / "REQUIREMENTS.md"
    path.write_text("## REQ-001 Relative\n", encoding="utf-8")
    doc = parse_requirements_file(path, relative_to=tmp_path)
    assert doc.declarations[0].source == "docs/REQUIREMENTS.md"


@pytest.mark.verifies("REQ-022")
def test_relative_to_falls_back_to_the_bare_name(tmp_path):
    path = tmp_path / "REQUIREMENTS.md"
    path.write_text("## REQ-001 Elsewhere\n", encoding="utf-8")
    doc = parse_requirements_file(path, relative_to=tmp_path / "other")
    assert doc.declarations[0].source == "REQUIREMENTS.md"


@pytest.mark.verifies("REQ-001")
def test_empty_document_yields_no_declarations():
    doc = parse_requirements_text("", source="empty.md")
    assert doc.declarations == ()
    assert doc.ids == ()
    assert doc.duplicates == {}


@pytest.mark.verifies("REQ-001")
def test_own_requirements_document_declares_twenty_six_unique_ids(product_root):
    # Hand count of docs/REQUIREMENTS.md: REQ-001 through REQ-026, no duplicates.
    doc = parse_requirements_file(product_root / "docs" / "REQUIREMENTS.md",
                                  relative_to=product_root)
    assert len(doc.declarations) == 26
    assert doc.ids == tuple(f"REQ-{n:03d}" for n in range(1, 27))
    assert doc.duplicates == {}


@pytest.mark.verifies("REQ-004")
def test_sample_fixture_declares_nine_declarations_and_eight_ids(sample_project):
    # Hand count of the fixture: REQ-001..REQ-008 plus a second REQ-007.
    doc = parse_requirements_file(sample_project / "docs" / "REQUIREMENTS.md")
    assert len(doc.declarations) == 9
    assert len(doc.ids) == 8
    assert list(doc.duplicates) == ["REQ-007"]
