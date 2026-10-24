"""Text rendering. No computation here, so the tests check shape and content."""

from __future__ import annotations

import pytest

from relfiledelivery.analytic import analytic_delivery
from relfiledelivery.link import LinkConditions
from relfiledelivery.modes import ALL_MODES, TransferMode
from relfiledelivery.montecarlo import run_ensemble, standard_error_of_mean
from relfiledelivery.payload import FileTransferSpec
from relfiledelivery.report import (
    analytic_report,
    closed_form_agreement,
    distribution_report,
    mode_comparison,
    rule,
    scenario_report,
    simulation_report,
)
from relfiledelivery.simulate import simulate_transfer


def test_rule_is_a_titled_horizontal_rule() -> None:
    lines = rule("heading")
    assert lines[2] == "heading"
    assert set(lines[1]) == {"="}


def test_scenario_report_describes_the_file_before_the_link(
    tiny_spec: FileTransferSpec, outage_link: LinkConditions
) -> None:
    lines = scenario_report(tiny_spec, outage_link)
    file_at = next(i for i, line in enumerate(lines) if line.startswith("file size"))
    link_at = next(i for i, line in enumerate(lines) if line.startswith("forward rate"))
    assert file_at < link_at


def test_analytic_report_restates_the_assumptions(
    tiny_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    prediction = analytic_delivery(tiny_spec, lossy_link, TransferMode.DEFERRED_NAK)
    lines = analytic_report(prediction)
    assert any("A1-A4" in line for line in lines)


def test_simulation_report_names_the_seed(
    tiny_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    result = simulate_transfer(tiny_spec, lossy_link, TransferMode.DEFERRED_NAK, 4242)
    assert any("4242" in line for line in simulation_report(result))


def test_distribution_report_states_the_tail_separately(
    medium_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    distribution = run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 40, 1)
    lines = distribution_report(distribution)
    assert any("tail, stated separately" in line for line in lines)


def test_mode_comparison_has_no_mean_column(
    medium_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    distributions = {
        mode: run_ensemble(medium_spec, lossy_link, mode, 40, 1) for mode in ALL_MODES
    }
    lines = mode_comparison(distributions)
    assert "mean" not in lines[0]
    assert len(lines) == 2 + len(ALL_MODES)


def test_mode_comparison_handles_a_mode_with_no_completions(
    medium_spec: FileTransferSpec
) -> None:
    link = LinkConditions(1000.0, 100.0, 0.5, 0.3)
    distributions = {
        TransferMode.UNACKNOWLEDGED: run_ensemble(
            medium_spec, link, TransferMode.UNACKNOWLEDGED, 20, 1
        )
    }
    lines = mode_comparison(distributions)
    assert "n/a" in lines[2]


def test_closed_form_agreement_reports_the_discrepancy_in_standard_errors(
    medium_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    distribution = run_ensemble(medium_spec, lossy_link, TransferMode.DEFERRED_NAK, 80, 1)
    prediction = analytic_delivery(medium_spec, lossy_link, TransferMode.DEFERRED_NAK)
    lines = closed_form_agreement(
        prediction, distribution, standard_error_of_mean(distribution.delivery_times_s)
    )
    assert any("difference / std error" in line for line in lines)


def test_closed_form_agreement_refuses_a_mode_without_a_closed_form(
    medium_spec: FileTransferSpec, lossy_link: LinkConditions
) -> None:
    distribution = run_ensemble(medium_spec, lossy_link, TransferMode.IMMEDIATE_NAK, 20, 1)
    prediction = analytic_delivery(medium_spec, lossy_link, TransferMode.IMMEDIATE_NAK)
    with pytest.raises(ValueError, match="no closed-form expected delivery time"):
        closed_form_agreement(prediction, distribution, 0.1)
