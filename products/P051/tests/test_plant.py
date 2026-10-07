"""Plant construction, the declared sets, and input validation."""

from __future__ import annotations

import numpy as np
import pytest

from simplexguard import Plant, box, reference_plant
from simplexguard.polytope import Polytope


def test_reference_plant_zoh_discretisation_known_answer():
    # Exact ZOH discretisation of theta'' = u at dt = 0.05 s:
    #   A = [[1, 0.05], [0, 1]]
    #   B = [[0.05^2 / 2], [0.05]] = [[0.00125], [0.05]]
    plant = reference_plant(dt=0.05)
    assert plant.A == pytest.approx(np.array([[1.0, 0.05], [0.0, 1.0]]))
    assert plant.B == pytest.approx(np.array([[0.00125], [0.05]]))
    assert plant.n_states == 2
    assert plant.n_inputs == 1


def test_reference_plant_disturbance_box_known_answer():
    # A declared unmodelled acceleration of 0.12 rad/s^2 acting for one 0.05 s
    # sample gives a state increment of
    #   [0.12 * 0.05^2 / 2, 0.12 * 0.05] = [1.5e-4 rad, 6.0e-3 rad/s].
    plant = reference_plant(dt=0.05, disturbance_accel_rad_s2=0.12)
    assert plant.disturbance.half_widths == pytest.approx(np.array([1.5e-4, 6.0e-3]))
    assert plant.disturbance.centre == pytest.approx(np.zeros(2))


def test_step_known_answer():
    # x = [0.1, 0.2], u = 1.0, w = 0:
    #   theta' = 0.1 + 0.05*0.2 + 0.00125*1.0 = 0.1 + 0.01 + 0.00125 = 0.11125
    #   rate'  = 0.2 + 0.05*1.0 = 0.25
    plant = reference_plant()
    nxt = plant.step(np.array([0.1, 0.2]), np.array([1.0]))
    assert nxt == pytest.approx(np.array([0.11125, 0.25]))


def test_step_adds_the_disturbance():
    plant = reference_plant()
    base = plant.step(np.array([0.0, 0.0]), np.array([0.0]))
    with_w = plant.step(np.array([0.0, 0.0]), np.array([0.0]), np.array([1.0, -2.0]))
    assert with_w - base == pytest.approx(np.array([1.0, -2.0]))


def test_step_does_not_check_the_disturbance_bound():
    # Deliberate: the bound-violation experiment must be able to pass an
    # inadmissible disturbance through the plant.
    plant = reference_plant()
    huge = np.array([10.0, 10.0])
    assert not plant.disturbance_is_admissible(huge)
    plant.step(np.zeros(2), np.zeros(1), huge)


def test_disturbance_is_admissible_at_the_boundary():
    plant = reference_plant()
    edge = plant.disturbance.half_widths.copy()
    assert plant.disturbance_is_admissible(edge)
    assert not plant.disturbance_is_admissible(edge * 1.001)


def test_with_disturbance_replaces_only_the_disturbance():
    plant = reference_plant()
    bigger = plant.with_disturbance(box(plant.disturbance.half_widths * 3.0))
    assert bigger.disturbance.half_widths == pytest.approx(
        plant.disturbance.half_widths * 3.0
    )
    assert bigger.A == pytest.approx(plant.A)
    assert bigger.state_constraints.b == pytest.approx(plant.state_constraints.b)


def test_describe_mentions_the_declared_sets():
    text = reference_plant().describe()
    assert "declared W (box)" in text
    assert "declared X" in text
    assert "declared U" in text


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dt": 0.0}, "dt must be finite and positive"),
        ({"dt": -1.0}, "dt must be finite and positive"),
        ({"angle_limit_rad": 0.0}, "angle_limit_rad"),
        ({"rate_limit_rad_s": -0.1}, "rate_limit_rad_s"),
        ({"accel_limit_rad_s2": np.nan}, "accel_limit_rad_s2"),
        ({"disturbance_accel_rad_s2": -1.0}, "disturbance_accel_rad_s2"),
    ],
)
def test_reference_plant_rejects_bad_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        reference_plant(**kwargs)


def test_zero_disturbance_is_allowed_and_gives_a_degenerate_box():
    plant = reference_plant(disturbance_accel_rad_s2=0.0)
    assert plant.disturbance.half_widths == pytest.approx(np.zeros(2))
    assert plant.disturbance.support(np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_plant_rejects_a_disturbance_set_without_the_origin():
    offset = Polytope(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
                      np.array([2.0, -1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="must contain the origin"):
        Plant(
            A=np.eye(2),
            B=np.ones((2, 1)),
            disturbance=offset,
            state_constraints=box([1.0, 1.0]),
            input_constraints=box([1.0]),
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"A": np.ones((2, 3))}, "must be square"),
        ({"B": np.ones((3, 1))}, "rows, A has"),
        ({"state_constraints": box([1.0])}, "state constraints have dim"),
        ({"input_constraints": box([1.0, 1.0])}, "input constraints have dim"),
        ({"disturbance": box([1.0])}, "disturbance set has dim"),
        ({"A": np.array([[np.nan, 0.0], [0.0, 1.0]])}, "must be finite"),
        ({"state_names": ("a",)}, "state_names"),
    ],
)
def test_plant_constructor_validation(kwargs, match):
    base = {
        "A": np.eye(2),
        "B": np.ones((2, 1)),
        "disturbance": box([0.1, 0.1]),
        "state_constraints": box([1.0, 1.0]),
        "input_constraints": box([1.0]),
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        Plant(**base)


def test_step_rejects_wrong_shapes():
    plant = reference_plant()
    with pytest.raises(ValueError, match="x has length"):
        plant.step(np.zeros(3), np.zeros(1))
    with pytest.raises(ValueError, match="u has length"):
        plant.step(np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError, match="w has length"):
        plant.step(np.zeros(2), np.zeros(1), np.zeros(3))
