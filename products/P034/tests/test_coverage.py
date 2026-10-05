"""Coverage accounting: cell mapping, totals, tracker behaviour."""

from __future__ import annotations

import pytest

from faultinject.coverage import CoverageTracker, all_cells, cell_of, iter_cells
from faultinject.faults import Injection
from faultinject.taxonomy import FaultKind, kinds


def test_total_cell_count_known_answer():
    assert len(all_cells()) == 248


def test_cells_are_unique_and_sorted():
    cells = all_cells()
    assert len(set(cells)) == len(cells)
    assert list(cells) == sorted(cells)


def test_iter_cells_matches_all_cells():
    assert tuple(sorted(iter_cells())) == all_cells()


def test_subset_cell_counts():
    subset = (FaultKind.SENSOR_BIAS, FaultKind.TIMING_LATE_SAMPLE)
    assert len(all_cells(subset)) == 24 + 8


def test_cell_of_hand_computed():
    # start 10/150 = 0.0667 -> bin 0 ; duration 20/150 = 0.1333 -> bin 0
    # offset 0.2 with log edges 0.1, 0.4642, 2.1544, 10 -> bin 0
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 0.2}, 10, 20)
    assert cell_of(inj, 150).label() == "sensor_bias/pos/s0/d0/p[0]"


def test_cell_of_upper_bins():
    # start 120/150 = 0.8 -> bin 1 ; duration 150/150 = 1.0 -> bin 1 ; offset 9 -> bin 2
    inj = Injection.create(FaultKind.SENSOR_BIAS, "vel", {"offset": 9.0}, 120, 150)
    assert cell_of(inj, 150).label() == "sensor_bias/vel/s1/d1/p[2]"


def test_cell_of_rejects_zero_steps():
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 0, 1)
    with pytest.raises(ValueError, match="n_steps"):
        cell_of(inj, 0)


def test_tracker_counts_and_fraction():
    t = CoverageTracker.full()
    assert t.total == 248
    assert t.covered == 0
    assert t.fraction == 0.0
    inj = Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 10, 20)
    t.add(inj, 150)
    t.add(inj, 150)
    assert t.covered == 1
    assert t.fraction == pytest.approx(1.0 / 248, rel=1e-15)
    assert max(t.hits.values()) == 2


def test_tracker_rejects_out_of_subset():
    t = CoverageTracker((FaultKind.SENSOR_BIAS,))
    inj = Injection.create(FaultKind.BUS_LOSS, "bus", {"loss_prob": 0.2}, 0, 5)
    with pytest.raises(ValueError, match="outside the tracked subset"):
        t.add(inj, 150)


def test_tracker_missing_shrinks():
    t = CoverageTracker((FaultKind.SENSOR_STUCK,))
    assert len(t.missing()) == 8
    t.add(Injection.create(FaultKind.SENSOR_STUCK, "pos", {}, 0, 150), 150)
    assert len(t.missing()) == 7


def test_per_kind_breakdown():
    t = CoverageTracker((FaultKind.SENSOR_BIAS, FaultKind.SENSOR_STUCK))
    t.add(Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": 1.0}, 0, 150), 150)
    pk = t.per_kind()
    assert pk[FaultKind.SENSOR_BIAS] == (1, 24)
    assert pk[FaultKind.SENSOR_STUCK] == (0, 8)


def test_add_all():
    t = CoverageTracker((FaultKind.SENSOR_BIAS,))
    injs = [
        Injection.create(FaultKind.SENSOR_BIAS, "pos", {"offset": o}, 0, 150)
        for o in (0.2, 1.0, 5.0)
    ]
    t.add_all(injs, 150)
    assert t.covered == 3


def test_every_kind_appears_in_the_cell_list():
    present = {c.kind for c in all_cells()}
    assert present == set(kinds())
