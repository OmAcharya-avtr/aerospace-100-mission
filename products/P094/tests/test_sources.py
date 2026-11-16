"""The committed board programs: identity, import, and the declared cost model.

SPEC §1 rests on the same file running in both modes. These tests assert that
property directly rather than documenting it: the md5 of each committed program
is compared against the sum the coordinator verified against the copy that
executed on the Jetson, and the module the package times is asserted to be the
file on disk at that path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from hiltimingdiff import sources
from hiltimingdiff.sources import SOURCE_MD5, TIMED_SOURCES, SourceIdentityError

#: The four sums, repeated here as literals rather than imported, so that a
#: change to the package constant does not silently change the test. If these
#: two lists ever disagree the suite fails, which is the point.
COORDINATOR_MD5 = {
    "campaign.py": "8c137fc2c31c48c2e8f16d1ae1f206d3",
    "staged_exec.py": "0f654894ea7dc3c6ec5d3880e9150459",
    "pat_pipeline.py": "80ba9015b54fa8e95eae96b2c93eb9c2",
    "verify_outputs.py": "ac6b6aa2b486ea7d9a227d1598dc8293",
}


def test_source_md5_table_matches_the_coordinator_list():
    assert SOURCE_MD5 == COORDINATOR_MD5


@pytest.mark.parametrize("name", sorted(COORDINATOR_MD5))
def test_committed_program_exists(name):
    assert sources.source_path(name).is_file()


@pytest.mark.parametrize("name", sorted(COORDINATOR_MD5))
def test_committed_program_md5_matches_the_board_copy(name):
    assert sources.md5_of_file(sources.source_path(name)) == COORDINATOR_MD5[name]


@pytest.mark.parametrize("name", sorted(COORDINATOR_MD5))
def test_measured_md5_agrees_with_an_independent_hashlib_call(name):
    data = sources.source_path(name).read_bytes()
    assert hashlib.md5(data).hexdigest() == sources.measured_md5()[name]


def test_verify_source_identity_returns_all_four_sums():
    assert sources.verify_source_identity() == COORDINATOR_MD5


def test_no_source_identity_failures():
    assert sources.source_identity_failures() == {}


def test_source_path_rejects_an_unknown_program():
    with pytest.raises(SourceIdentityError, match="not one of the committed"):
        sources.source_path("nope.py")


@pytest.mark.parametrize("name", ["campaign", "verify_outputs"])
def test_only_the_timed_sources_are_importable(name):
    with pytest.raises(SourceIdentityError, match="not one of the importable"):
        sources.load_board_module(name)


@pytest.mark.parametrize("name", sorted(TIMED_SOURCES))
def test_timed_sources_are_the_two_the_package_executes(name):
    assert name in ("pat_pipeline.py", "staged_exec.py")


def test_pipeline_module_is_the_file_on_disk():
    module = sources.pipeline_module()
    assert Path(module.__file__).resolve() == sources.source_path("pat_pipeline.py").resolve()


def test_executor_module_is_the_file_on_disk():
    module = sources.executor_module()
    assert Path(module.__file__).resolve() == sources.source_path("staged_exec.py").resolve()


def test_importing_the_pipeline_does_not_leave_the_data_directory_importable():
    import sys

    before = list(sys.path)
    sources.pipeline_module()
    assert str(sources.data_directory()) not in sys.path or before == sys.path


def test_stage_names_are_the_six_in_pipeline_order():
    assert sources.stage_names() == (
        "ingest",
        "background",
        "threshold",
        "centroid",
        "gate",
        "track",
    )


def test_reject_reasons_are_the_four_declared_on_the_board():
    assert sources.reject_reasons() == ("no_candidate", "low_score", "area_out", "border")


@pytest.mark.parametrize("width", [16, 48, 64, 96, 128])
def test_declared_ops_ingest_is_two_per_pixel(width):
    # Hand calculation: ingest declares 2.0 * W * W. At W = 64 that is
    # 2 * 4096 = 8192.
    assert sources.declared_ops(width, 4.0)["ingest"] == pytest.approx(
        2.0 * width * width, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("width", [16, 48, 64, 96, 128])
def test_declared_ops_background_is_four_per_pixel(width):
    assert sources.declared_ops(width, 4.0)["background"] == pytest.approx(
        4.0 * width * width, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("width", [16, 48, 64, 96, 128])
def test_declared_ops_threshold_is_three_per_pixel(width):
    assert sources.declared_ops(width, 4.0)["threshold"] == pytest.approx(
        3.0 * width * width, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("candidates", [0.0, 1.0, 2.5, 12.0])
def test_declared_ops_centroid_is_one_hundred_fifty_per_candidate(candidates):
    # Hand calculation: 6.0 * K * 25.0 = 150 * K. At K = 2.5 that is 375.
    assert sources.declared_ops(64, candidates)["centroid"] == pytest.approx(
        150.0 * candidates, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("candidates", [0.0, 1.0, 2.5, 12.0])
def test_declared_ops_gate_is_twelve_per_candidate(candidates):
    assert sources.declared_ops(64, candidates)["gate"] == pytest.approx(
        12.0 * candidates, rel=1e-12, abs=0.0
    )


@pytest.mark.parametrize("width", [48, 64, 96])
@pytest.mark.parametrize("candidates", [0.0, 3.5, 12.8])
def test_declared_ops_track_is_width_and_candidate_independent(width, candidates):
    assert sources.declared_ops(width, candidates)["track"] == 180.0


def test_declared_ops_covers_exactly_the_six_stages():
    assert set(sources.declared_ops(64, 5.0)) == set(sources.stage_names())


@pytest.mark.parametrize("width", [0, -1, -64])
def test_declared_ops_rejects_a_non_positive_width(width):
    with pytest.raises(ValueError, match="positive pixel count"):
        sources.declared_ops(width, 1.0)


@pytest.mark.parametrize("candidates", [-0.5, -1.0])
def test_declared_ops_rejects_a_negative_candidate_count(candidates):
    with pytest.raises(ValueError, match="non-negative"):
        sources.declared_ops(64, candidates)


def test_md5_of_file_matches_for_an_empty_file(tmp_path):
    target = tmp_path / "empty.bin"
    target.write_bytes(b"")
    assert sources.md5_of_file(target) == "d41d8cd98f00b204e9800998ecf8427e"


def test_md5_of_file_reads_beyond_one_chunk(tmp_path):
    payload = b"x" * (3 * 1024 * 1024 + 7)
    target = tmp_path / "big.bin"
    target.write_bytes(payload)
    assert sources.md5_of_file(target) == hashlib.md5(payload).hexdigest()


def test_root_is_the_product_directory():
    assert (sources.ROOT / "SPEC.md").is_file()
    assert (sources.ROOT / "data" / "pat_pipeline.py").is_file()
