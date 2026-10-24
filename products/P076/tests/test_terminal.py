"""Terminal, crosslink and beam validation."""

from __future__ import annotations

import numpy as np
import pytest

from pointahead.geometry import C_LIGHT
from pointahead.kepler import KeplerTrajectory, LinearTrajectory
from pointahead.terminal import BeamContext, Crosslink, TerminalState


def test_state_exposes_speed_and_beta() -> None:
    state = TerminalState([7.0e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 0.0, "t")
    assert state.speed_mps == pytest.approx(7500.0, rel=1e-15)
    assert state.beta[1] == pytest.approx(7500.0 / C_LIGHT, rel=1e-15)


def test_zero_position_is_refused() -> None:
    with pytest.raises(ValueError, match="position_m must be non-zero"):
        TerminalState([0.0, 0.0, 0.0], [0.0, 7500.0, 0.0], 0.0, "t")


def test_superluminal_velocity_is_refused_with_a_units_hint() -> None:
    with pytest.raises(ValueError, match="this field is metres per second"):
        TerminalState([7.0e6, 0.0, 0.0], [0.0, 4.0e8, 0.0], 0.0, "t")


def test_non_finite_epoch_is_refused() -> None:
    with pytest.raises(ValueError, match="epoch_s must be finite"):
        TerminalState([7.0e6, 0.0, 0.0], [0.0, 7500.0, 0.0], np.nan, "t")


def test_trajectory_models_are_named() -> None:
    state = TerminalState([7.0e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 0.0, "t")
    assert isinstance(state.trajectory("kepler"), KeplerTrajectory)
    assert isinstance(state.trajectory("linear"), LinearTrajectory)
    with pytest.raises(ValueError, match="unknown trajectory model"):
        state.trajectory("sgp4")


def test_crosslink_geometry_properties(hand_link) -> None:
    assert hand_link.range_m == pytest.approx(2.0e6, rel=1e-15)
    assert np.allclose(hand_link.line_of_sight, [0.0, 1.0, 0.0], atol=1e-15)
    assert np.allclose(hand_link.relative_velocity_mps, [7500.0, 0.0, 0.0], atol=1e-15)
    assert hand_link.nominal_light_time_s == pytest.approx(2.0e6 / C_LIGHT, rel=1e-15)


def test_mismatched_epochs_are_refused() -> None:
    with pytest.raises(ValueError, match="must share an epoch"):
        Crosslink(
            local=TerminalState([7.0e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 0.0, "a"),
            remote=TerminalState([7.0e6, 1.0e6, 0.0], [0.0, 7500.0, 0.0], 1.0, "b"),
        )


def test_colocated_terminals_are_refused() -> None:
    with pytest.raises(ValueError, match="co-located"):
        Crosslink(
            local=TerminalState([7.0e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 0.0, "a"),
            remote=TerminalState([7.0e6, 0.0, 0.0], [0.0, 7600.0, 0.0], 0.0, "b"),
        )


def test_unknown_trajectory_model_is_refused() -> None:
    with pytest.raises(ValueError, match="trajectory_model must be one of"):
        Crosslink(
            local=TerminalState([7.0e6, 0.0, 0.0], [0.0, 7500.0, 0.0], 0.0, "a"),
            remote=TerminalState([7.0e6, 1.0e6, 0.0], [0.0, 7500.0, 0.0], 0.0, "b"),
            trajectory_model="rk4",
        )


def test_reversed_swaps_the_terminals(crossing_link) -> None:
    reverse = crossing_link.reversed()
    assert reverse.local.name == crossing_link.remote.name
    assert reverse.range_m == pytest.approx(crossing_link.range_m, rel=1e-15)
    assert np.allclose(reverse.line_of_sight, -crossing_link.line_of_sight, atol=1e-15)


def test_boosted_adds_the_velocity_and_forces_the_linear_model(crossing_link) -> None:
    delta = np.array([100.0, -200.0, 50.0])
    boosted = crossing_link.boosted(delta)
    assert boosted.trajectory_model == "linear"
    assert np.allclose(boosted.local.velocity_mps, crossing_link.local.velocity_mps + delta)
    assert np.allclose(boosted.remote.velocity_mps, crossing_link.remote.velocity_mps + delta)
    assert np.allclose(boosted.relative_velocity_mps, crossing_link.relative_velocity_mps)


def test_beam_rejects_non_positive_and_sub_wavelength_waists() -> None:
    with pytest.raises(ValueError, match="wavelength_m must be positive"):
        BeamContext(0.0, 0.025)
    with pytest.raises(ValueError, match="waist_m must be positive"):
        BeamContext(1550.0e-9, 0.0)
    with pytest.raises(ValueError, match="paraxial Gaussian far-field"):
        BeamContext(1550.0e-9, 1.0e-9)


def test_trajectories_returns_local_then_remote(crossing_link) -> None:
    local, remote = crossing_link.trajectories()
    assert np.allclose(local.position(0.0), crossing_link.local.position_m)
    assert np.allclose(remote.position(0.0), crossing_link.remote.position_m)
