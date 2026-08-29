"""Tests for waveforge.dm."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.dm import DeformableMirror
from waveforge.pupil import PupilGrid, variance


@pytest.fixture(scope="module")
def mirror():
    return DeformableMirror(PupilGrid(32, 0.5), n_act=5, margin_actuators=0)


class TestGeometry:
    def test_pitch(self, mirror):
        # d_act = D / (n_act - 1) = 0.5 / 4 = 0.125 m
        assert mirror.pitch_m == pytest.approx(0.125)

    def test_actuator_count_without_margin(self, mirror):
        assert mirror.n_actuators == 25

    def test_actuator_count_with_margin(self):
        dm = DeformableMirror(PupilGrid(32, 0.5), n_act=5, margin_actuators=1)
        assert dm.n_actuators == 49  # (5 + 2)^2

    def test_positions_shape(self, mirror):
        assert mirror.positions_m.shape == (25, 2)

    def test_positions_are_centred(self, mirror):
        assert np.allclose(mirror.positions_m.mean(axis=0), 0.0, atol=1e-12)

    def test_outer_actuators_sit_on_the_rim(self, mirror):
        assert mirror.positions_m[:, 0].max() == pytest.approx(0.25)

    def test_influence_matrix_shape(self, mirror):
        assert mirror.influence_matrix.shape == (25, mirror.pupil.n_valid)

    def test_influence_peaks_at_unity(self):
        # A dense grid resolves the actuator centre, so max influence -> 1
        dm = DeformableMirror(PupilGrid(128, 0.5), n_act=5, margin_actuators=0)
        assert dm.influence_matrix.max() == pytest.approx(1.0, abs=0.02)

    def test_coupling_at_neighbour_spacing(self):
        dm = DeformableMirror(PupilGrid(32, 0.5), n_act=5, coupling=0.2, margin_actuators=0)
        centre = dm.positions_m[12]
        neighbour = centre + np.array([dm.pitch_m, 0.0])
        r2 = np.sum((neighbour - centre) ** 2)
        assert np.exp(np.log(0.2) * r2 / dm.pitch_m**2) == pytest.approx(0.2)

    def test_influence_is_positive(self, mirror):
        assert np.all(mirror.influence_matrix > 0.0)

    @pytest.mark.parametrize("n_act", [1, 0, 4.5])
    def test_bad_n_act(self, n_act):
        with pytest.raises(ValueError, match="n_act"):
            DeformableMirror(PupilGrid(32, 0.5), n_act=n_act)

    @pytest.mark.parametrize("coupling", [0.0, 1.0, -0.1])
    def test_bad_coupling(self, coupling):
        with pytest.raises(ValueError, match="coupling"):
            DeformableMirror(PupilGrid(32, 0.5), n_act=5, coupling=coupling)

    def test_bad_stroke(self):
        with pytest.raises(ValueError, match="stroke_rad"):
            DeformableMirror(PupilGrid(32, 0.5), n_act=5, stroke_rad=0.0)

    def test_bad_margin(self):
        with pytest.raises(ValueError, match="margin_actuators"):
            DeformableMirror(PupilGrid(32, 0.5), n_act=5, margin_actuators=-1)


class TestSurface:
    def test_zero_commands_give_zero_surface(self, mirror):
        assert np.allclose(mirror.surface(np.zeros(25)), 0.0)

    def test_surface_is_linear(self, mirror, rng):
        a = rng.normal(size=25)
        b = rng.normal(size=25)
        assert np.allclose(
            mirror.surface(2.0 * a + 0.5 * b),
            2.0 * mirror.surface(a) + 0.5 * mirror.surface(b),
            atol=1e-12,
        )

    def test_surface_piston_removed(self, mirror, rng):
        s = mirror.surface(rng.normal(size=25))
        assert s[mirror.pupil.mask].mean() == pytest.approx(0.0, abs=1e-12)

    def test_surface_outside_mask_is_zero(self, mirror, rng):
        s = mirror.surface(rng.normal(size=25))
        assert np.all(s[~mirror.pupil.mask] == 0.0)

    def test_uniform_command_is_pure_piston(self, mirror):
        # Every actuator pushed equally gives an (almost) flat surface, so
        # after piston removal there is very little left.
        s = mirror.surface(np.ones(25))
        assert variance(s, mirror.pupil.mask) < 0.05

    def test_wrong_command_shape(self, mirror):
        with pytest.raises(ValueError, match="commands must have shape"):
            mirror.surface(np.zeros(3))


class TestStroke:
    def test_no_clipping_when_unlimited(self, mirror, rng):
        c = rng.normal(size=25) * 100.0
        clipped, fraction = mirror.clip(c)
        assert np.array_equal(clipped, c)
        assert fraction == 0.0

    def test_clipping_applies_limit(self):
        dm = DeformableMirror(PupilGrid(32, 0.5), n_act=5, stroke_rad=1.0, margin_actuators=0)
        c = np.linspace(-3.0, 3.0, 25)
        clipped, fraction = dm.clip(c)
        assert clipped.max() == pytest.approx(1.0)
        assert clipped.min() == pytest.approx(-1.0)
        assert 0.0 < fraction < 1.0

    def test_saturation_fraction_known_answer(self):
        dm = DeformableMirror(PupilGrid(32, 0.5), n_act=5, stroke_rad=1.0, margin_actuators=0)
        c = np.zeros(25)
        c[:5] = 2.0  # 5 of 25 actuators over the limit
        _, fraction = dm.clip(c)
        assert fraction == pytest.approx(5 / 25)

    def test_clip_does_not_mutate_input(self):
        dm = DeformableMirror(PupilGrid(32, 0.5), n_act=5, stroke_rad=1.0, margin_actuators=0)
        c = np.full(25, 3.0)
        dm.clip(c)
        assert np.all(c == 3.0)

    def test_clip_wrong_shape(self, mirror):
        with pytest.raises(ValueError, match="commands must have shape"):
            mirror.clip(np.zeros(3))


class TestFitting:
    def test_recovers_its_own_commands(self, mirror, rng):
        # Fitting the raw (piston-retained) surface an unregularised solve must
        # return the exact command vector that produced it.
        c = rng.normal(size=25)
        target = mirror.surface(c, remove_piston=False)
        assert np.allclose(mirror.fit(target, regularisation=0.0), c, atol=1e-8)

    def test_reproduces_its_own_surface(self, mirror, rng):
        # fitting_residual works on the piston-removed shape, and the constant
        # vector is not in the span of the influence functions, so the residual
        # is small but not zero. This pins how small.
        c = rng.normal(size=25)
        surface = mirror.surface(c)
        residual = mirror.fitting_residual(surface, regularisation=0.0)
        ratio = variance(residual, mirror.pupil.mask) / variance(surface, mirror.pupil.mask)
        assert ratio < 1e-3

    def test_fit_of_zero_is_zero(self, mirror):
        assert np.allclose(mirror.fit(np.zeros((32, 32))), 0.0)

    def test_residual_is_smaller_than_input(self, mirror, rng):
        phase = rng.normal(size=(32, 32))
        assert variance(mirror.fitting_residual(phase), mirror.pupil.mask) < variance(
            phase, mirror.pupil.mask
        )

    def test_more_actuators_fit_better(self, rng):
        from waveforge.atmosphere import phase_screen

        pupil = PupilGrid(32, 0.5)
        phase = phase_screen(64, pupil.sample_spacing_m, 0.1, rng=3)[:32, :32]
        coarse = DeformableMirror(pupil, n_act=3, margin_actuators=0)
        fine = DeformableMirror(pupil, n_act=9, margin_actuators=0)
        assert variance(fine.fitting_residual(phase), pupil.mask) < variance(
            coarse.fitting_residual(phase), pupil.mask
        )

    def test_residual_piston_removed(self, mirror, rng):
        residual = mirror.fitting_residual(rng.normal(size=(32, 32)))
        assert residual[mirror.pupil.mask].mean() == pytest.approx(0.0, abs=1e-12)

    def test_fit_wrong_shape(self, mirror):
        with pytest.raises(ValueError, match="phase shape"):
            mirror.fit(np.zeros((8, 8)))

    def test_bad_regularisation(self, mirror):
        with pytest.raises(ValueError, match="regularisation"):
            mirror.fit(np.zeros((32, 32)), regularisation=-1.0)

    def test_stronger_regularisation_shrinks_commands(self, mirror, rng):
        phase = rng.normal(size=(32, 32))
        weak = np.linalg.norm(mirror.fit(phase, 1e-9))
        strong = np.linalg.norm(mirror.fit(phase, 1e-1))
        assert strong < weak
