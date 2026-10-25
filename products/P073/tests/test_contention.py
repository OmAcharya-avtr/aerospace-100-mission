"""Contention map: union supply, apportioned demand and the ratio."""

from __future__ import annotations

import pytest

from contactsched.contention import (
    ContentionCell,
    contention_map,
    effective_seconds,
    request_contention,
)
from contactsched.problem import ContactRequest, build_problem
from contactsched.station import GroundStation, HorizonMask
from contactsched.visibility import ContactWindow


def station(name: str = "A1", turnaround_s: float = 0.0) -> GroundStation:
    return GroundStation(name, 0.0, 0.0, 0.0, HorizonMask(floor_deg=5.0), turnaround_s)


def window(spacecraft: str, rise: float, close: float, station_name: str = "A1") -> ContactWindow:
    return ContactWindow(
        station_name, spacecraft, "M", rise, close, 45.0, 0.5 * (rise + close), 5.0
    )


class TestContentionCell:
    def test_ratio_is_demand_over_supply(self):
        cell = ContentionCell("A1", 0.0, 100.0, supply_s=50.0, demand_s=25.0, requests=("R1",))
        assert cell.ratio == pytest.approx(0.5, rel=0.0, abs=0.0)

    def test_no_supply_with_demand_is_infinite(self):
        cell = ContentionCell("A1", 0.0, 100.0, supply_s=0.0, demand_s=5.0, requests=("R1",))
        assert cell.ratio == float("inf")

    def test_no_supply_and_no_demand_is_zero(self):
        cell = ContentionCell("A1", 0.0, 100.0, supply_s=0.0, demand_s=0.0, requests=())
        assert cell.ratio == pytest.approx(0.0, rel=0.0, abs=0.0)


class TestEffectiveSeconds:
    def test_sums_usable_windows(self):
        request = ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1)
        windows = (window("SC-1", 0.0, 300.0), window("SC-1", 1000.0, 1200.0))
        problem = build_problem((station(),), (request,), windows, 100.0)
        assert effective_seconds(problem)["R1"] == pytest.approx(500.0, rel=0.0, abs=0.0)

    def test_excludes_windows_shorter_than_the_contact(self):
        request = ContactRequest("R1", "M1", "SC-1", 1, 400.0, 1, 1)
        windows = (window("SC-1", 0.0, 500.0), window("SC-1", 1000.0, 1100.0))
        problem = build_problem((station(),), (request,), windows, 100.0)
        assert effective_seconds(problem)["R1"] == pytest.approx(500.0, rel=0.0, abs=0.0)

    def test_excludes_disallowed_antennas(self):
        request = ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1, stations=("A1",))
        windows = (window("SC-1", 0.0, 300.0, "A1"), window("SC-1", 0.0, 300.0, "A2"))
        problem = build_problem((station("A1"), station("A2")), (request,), windows, 100.0)
        assert effective_seconds(problem)["R1"] == pytest.approx(300.0, rel=0.0, abs=0.0)

    def test_zero_when_nothing_is_usable(self):
        request = ContactRequest("R1", "M1", "SC-1", 1, 1000.0, 1, 1)
        windows = (window("SC-1", 0.0, 300.0),)
        problem = build_problem((station(),), (request,), windows, 100.0)
        assert effective_seconds(problem)["R1"] == pytest.approx(0.0, rel=0.0, abs=0.0)


class TestContentionMap:
    def test_supply_is_a_union_of_overlapping_windows(self):
        # Two 100 s windows on A1, overlapping by 50 s, inside one 1000 s bin.
        # The union is 150 s, not the 200 s a sum would give.
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),)
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 50.0, 150.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=1000.0)
        assert len(cells) == 1
        assert cells[0].supply_s == pytest.approx(150.0, rel=0.0, abs=0.0)

    def test_supply_is_clipped_to_the_bin(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),)
        windows = (window("SC-1", 50.0, 250.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=400.0, bin_s=100.0)
        supplies = [c.supply_s for c in cells]
        # Bins [0,100), [100,200), [200,300), [300,400): 50, 100, 50, 0.
        assert supplies == pytest.approx([50.0, 100.0, 50.0, 0.0], rel=0.0, abs=1e-12)

    def test_demand_is_apportioned_across_opportunities(self):
        # One request needing 2 x 100 s = 200 s with two identical 100 s windows,
        # so 200 s of usable window. Each window therefore carries
        # 200 * 100 / 200 = 100 s of demand.
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=2, max_contacts=2),)
        windows = (window("SC-1", 0.0, 100.0), window("SC-1", 1000.0, 1100.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=2000.0, bin_s=1000.0)
        assert cells[0].demand_s == pytest.approx(100.0, rel=0.0, abs=1e-12)
        assert cells[1].demand_s == pytest.approx(100.0, rel=0.0, abs=1e-12)

    def test_total_demand_equals_the_declared_requirement(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=3, max_contacts=3),
            ContactRequest("R2", "M2", "SC-2", 2, 200.0, min_contacts=1, max_contacts=1),
        )
        windows = (
            window("SC-1", 0.0, 100.0),
            window("SC-1", 1000.0, 1100.0),
            window("SC-2", 500.0, 700.0),
        )
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=2000.0, bin_s=250.0)
        total = sum(c.demand_s for c in cells)
        # 3 x 100 + 1 x 200 = 500 s, conserved by the apportionment.
        assert total == pytest.approx(500.0, rel=1e-12, abs=0.0)

    def test_a_request_with_one_opportunity_presses_its_whole_demand(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, min_contacts=2, max_contacts=2),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=1000.0)
        assert cells[0].demand_s == pytest.approx(200.0, rel=0.0, abs=1e-12)
        assert cells[0].ratio == pytest.approx(2.0, rel=0.0, abs=1e-12)

    def test_requests_are_listed_per_cell(self):
        requests = (
            ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),
            ContactRequest("R2", "M2", "SC-2", 2, 100.0, 1, 1),
        )
        windows = (window("SC-1", 0.0, 100.0), window("SC-2", 600.0, 700.0))
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=500.0)
        assert cells[0].requests == ("R1",)
        assert cells[1].requests == ("R2",)

    def test_bins_tile_the_horizon(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),)
        windows = (window("SC-1", 0.0, 100.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=300.0)
        edges = [(c.bin_start_s, c.bin_end_s) for c in cells]
        assert edges[0][0] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert edges[-1][1] == pytest.approx(1000.0, rel=0.0, abs=0.0)
        for first, second in zip(edges[:-1], edges[1:], strict=True):
            assert first[1] == pytest.approx(second[0], rel=0.0, abs=0.0)

    def test_cells_are_produced_for_every_antenna(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 100.0, 1, 1),)
        windows = (window("SC-1", 0.0, 100.0, "A1"),)
        problem = build_problem((station("A1"), station("A2")), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=1000.0)
        assert {c.station for c in cells} == {"A1", "A2"}

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(horizon_s=0.0), "horizon_s must be positive"),
            (dict(bin_s=0.0), "bin_s must be positive"),
            (dict(bin_s=-10.0), "bin_s must be positive"),
        ],
    )
    def test_invalid_parameters_raise(self, toy_problem, kwargs, match):
        base = dict(horizon_s=1000.0, bin_s=100.0)
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            contention_map(toy_problem, **base)


class TestRequestContention:
    def test_keys_cover_every_request(self, toy_problem):
        cells = contention_map(toy_problem, horizon_s=5000.0, bin_s=1000.0)
        summary = request_contention(toy_problem, cells)
        assert set(summary) == {"R1", "R2"}

    def test_every_summary_has_the_declared_keys(self, toy_problem):
        cells = contention_map(toy_problem, horizon_s=5000.0, bin_s=1000.0)
        for values in request_contention(toy_problem, cells).values():
            assert set(values) == {
                "ratio_max",
                "ratio_weighted",
                "cells",
                "usable_seconds",
                "demand_over_usable",
            }

    def test_ratio_max_is_at_least_the_weighted_mean(self, instance_seed7):
        problem = instance_seed7.problem
        cells = contention_map(problem, instance_seed7.scenario.duration_s, 1800.0)
        for values in request_contention(problem, cells).values():
            if values["ratio_max"] == float("inf"):
                continue
            assert values["ratio_max"] >= values["ratio_weighted"] - 1e-9

    def test_demand_over_usable_is_infinite_without_usable_window(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 1000.0, min_contacts=1),)
        windows = (window("SC-1", 0.0, 300.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=1000.0)
        summary = request_contention(problem, cells)
        assert summary["R1"]["demand_over_usable"] == float("inf")

    def test_demand_over_usable_is_zero_with_no_demand_and_no_window(self):
        requests = (ContactRequest("R1", "M1", "SC-1", 1, 1000.0, min_contacts=0),)
        windows = (window("SC-1", 0.0, 300.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=1000.0)
        summary = request_contention(problem, cells)
        assert summary["R1"]["demand_over_usable"] == pytest.approx(0.0, rel=0.0, abs=0.0)

    def test_a_request_touching_no_cell_scores_zero(self):
        requests = (ContactRequest("R1", "M1", "SC-9", 1, 100.0, min_contacts=1),)
        windows = (window("SC-1", 0.0, 300.0),)
        problem = build_problem((station(),), requests, windows, 100.0)
        cells = contention_map(problem, horizon_s=1000.0, bin_s=1000.0)
        summary = request_contention(problem, cells)
        assert summary["R1"]["ratio_max"] == pytest.approx(0.0, rel=0.0, abs=0.0)
        assert summary["R1"]["cells"] == pytest.approx(0.0, rel=0.0, abs=0.0)
