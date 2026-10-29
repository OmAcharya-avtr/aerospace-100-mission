"""Text rendering, and the front door of the package."""

from __future__ import annotations

import numpy as np

import aligncal
from aligncal import batch, observability, recursive, report


def test_arcsec_per_rad():
    assert report.ARCSEC_PER_RAD == 180.0 * 3600.0 / np.pi


def test_observability_lines_are_strings(rich_verdict):
    lines = report.observability_lines(rich_verdict)
    assert all(isinstance(line, str) for line in lines)
    assert lines[0].startswith("OBSERVABILITY VERDICT")


def test_solution_lines(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    lines = report.solution_lines(solution)
    assert any("BATCH ESTIMATE" in line for line in lines)


def test_compare_lines_without_truth(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    lines = report.compare_lines(solution, mekf)
    text = "\n".join(lines)
    assert "BATCH AGAINST RECURSIVE" in text
    assert "injected truth" not in text
    assert len(lines) == 5  # header, column header, three degrees of freedom


def test_compare_lines_with_truth(rich, rich_verdict):
    solution = batch.solve(rich.record, rich_verdict, "secondary_1")
    mekf = recursive.run(rich.record, rich_verdict, "secondary_1", store_history=False)
    lines = report.compare_lines(solution, mekf, truth_rad=rich.truth["secondary_1"])
    text = "\n".join(lines)
    assert "against the injected truth" in text
    assert "batch error" in text


def test_compare_lines_marks_a_withheld_direction(single_axis, single_verdict):
    solution = batch.solve(single_axis.record, single_verdict, "secondary_1")
    mekf = recursive.run(single_axis.record, single_verdict, "secondary_1", store_history=False)
    lines = report.compare_lines(
        solution, mekf, truth_rad=single_axis.truth["secondary_1"]
    )
    text = "\n".join(lines)
    assert "withheld by the batch solution" in text
    assert "inf" in text


def test_align_front_door_computes_the_verdict_first(rich):
    result = aligncal.align(rich.record)
    assert isinstance(result.observability, observability.ObservabilityReport)
    assert set(result.batch) == {"secondary_1"}
    assert result.recursive == {}


def test_align_with_recursive(rich):
    result = aligncal.align(rich.record, with_recursive=True)
    assert set(result.recursive) == {"secondary_1"}
    batch_theta = result.batch["secondary_1"].theta_rad
    mekf_theta = result.recursive["secondary_1"].theta_rad
    crb = result.observability["secondary_1"].degrees_of_freedom[0].crb_sigma_rad
    assert float(np.linalg.norm(batch_theta - mekf_theta)) < 0.1 * crb


def test_align_threads_the_declared_thresholds(single_axis):
    strict = aligncal.align(single_axis.record, floor_factor=4.0)
    assert strict.observability.any_rank_deficient
    assert strict.batch["secondary_1"].withheld == (2,)


def test_align_summary_puts_the_verdict_first(single_axis):
    lines = aligncal.align(single_axis.record, with_recursive=True).summary()
    text = "\n".join(lines)
    assert lines[0].startswith("OBSERVABILITY VERDICT")
    assert text.index("does NOT support") < text.index("BATCH ESTIMATE")
    assert "RECURSIVE (MEKF) ESTIMATE" in text


def test_align_handles_several_secondaries(multi):
    result = aligncal.align(multi.record)
    assert set(result.batch) == {"secondary_1", "secondary_2", "secondary_3"}


def test_version_and_exports():
    assert aligncal.__version__ == "0.1.0"
    for name in ("align", "batch", "observability", "recursive", "learned", "dataset"):
        assert name in aligncal.__all__
        assert hasattr(aligncal, name)
