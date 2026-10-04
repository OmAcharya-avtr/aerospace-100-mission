"""Plant model: exact ZOH transition, gyro model, snapshot and PD gains."""

from __future__ import annotations

import numpy as np
import pytest

from hilforge.errors import ConfigurationError
from hilforge.plant import AttitudePlant, PDController, PDGains, PlantConfig


def test_zoh_double_integrator_is_exact_known_answer():
    # I = 2 kg m^2, u = 4 N*m -> a = 2 rad/s^2. From theta=0, omega=0, dt=3 s:
    #   theta = 0 + 0*3 + 0.5*2*9 = 9 rad
    #   omega = 0 + 2*3           = 6 rad/s
    cfg = PlantConfig(
        inertia_kgm2=2.0,
        gyro_bias_rad_s=0.0,
        gyro_arw_rad_s_sqrt_hz=0.0,
        encoder_noise_rad=0.0,
        torque_limit_nm=10.0,
        theta0_rad=0.0,
        omega0_rad_s=0.0,
    )
    plant = AttitudePlant(cfg, seed=0)
    applied, saturated = plant.apply_torque(4.0)
    assert applied == 4.0 and saturated is False
    state = plant.step(3.0)
    assert state[0] == pytest.approx(9.0, rel=1e-15)
    assert state[1] == pytest.approx(6.0, rel=1e-15)


def test_zoh_matches_the_closed_form_over_many_steps():
    # Constant torque from rest: theta(t) = 0.5 a t^2 exactly, for any number
    # of equal ZOH steps, because the transition carries no truncation error.
    cfg = PlantConfig(
        inertia_kgm2=5.0,
        gyro_bias_rad_s=0.0,
        gyro_arw_rad_s_sqrt_hz=0.0,
        encoder_noise_rad=0.0,
        torque_limit_nm=10.0,
        theta0_rad=0.0,
        omega0_rad_s=0.0,
    )
    plant = AttitudePlant(cfg, seed=0)
    plant.apply_torque(1.0)
    dt = 0.01
    n = 500
    for _ in range(n):
        plant.step(dt)
    acc = 1.0 / 5.0
    t = n * dt
    assert plant.state[0] == pytest.approx(0.5 * acc * t * t, rel=1e-12)
    assert plant.state[1] == pytest.approx(acc * t, rel=1e-14)


def test_torque_saturation_is_reported():
    plant = AttitudePlant(PlantConfig(torque_limit_nm=0.5), seed=1)
    applied, saturated = plant.apply_torque(2.0)
    assert applied == 0.5 and saturated is True
    applied, saturated = plant.apply_torque(-0.25)
    assert applied == -0.25 and saturated is False


def test_gyro_noise_scales_as_one_over_sqrt_dt():
    # sigma_w = N / sqrt(dt). With N = 1e-3 and dt = 1e-4, sigma = 0.1 rad/s.
    cfg = PlantConfig(
        gyro_bias_rad_s=0.0,
        gyro_arw_rad_s_sqrt_hz=1.0e-3,
        encoder_noise_rad=0.0,
        theta0_rad=0.0,
        omega0_rad_s=0.0,
    )
    plant = AttitudePlant(cfg, seed=5)
    samples = np.array([plant.measure(1.0e-4)[1] for _ in range(20000)])
    assert float(samples.std(ddof=1)) == pytest.approx(0.1, rel=0.03)
    assert abs(float(samples.mean())) < 0.005


def test_gyro_bias_appears_in_the_rate_channel():
    cfg = PlantConfig(
        gyro_bias_rad_s=2.5e-3,
        gyro_arw_rad_s_sqrt_hz=0.0,
        encoder_noise_rad=0.0,
        theta0_rad=0.0,
        omega0_rad_s=0.0,
    )
    plant = AttitudePlant(cfg, seed=0)
    assert plant.measure(0.01)[1] == pytest.approx(2.5e-3)


def test_same_seed_gives_identical_measurements():
    a = AttitudePlant(seed=99)
    b = AttitudePlant(seed=99)
    for _ in range(50):
        assert np.array_equal(a.measure(0.01), b.measure(0.01))


def test_snapshot_restore_round_trips_including_rng():
    plant = AttitudePlant(seed=3)
    for _ in range(10):
        plant.measure(0.01)
        plant.apply_torque(0.01)
        plant.step(0.01)
    snap = plant.snapshot()
    after_snapshot = [plant.measure(0.01).copy() for _ in range(5)]
    plant.restore(snap)
    replayed = [plant.measure(0.01).copy() for _ in range(5)]
    for want, got in zip(after_snapshot, replayed, strict=True):
        assert np.array_equal(want, got)
    assert plant.state.tolist() == pytest.approx(
        [snap["theta"], snap["omega"]]
    )


def test_plant_config_validation():
    with pytest.raises(ConfigurationError):
        PlantConfig(inertia_kgm2=0.0)
    with pytest.raises(ConfigurationError):
        PlantConfig(gyro_arw_rad_s_sqrt_hz=-1.0)
    with pytest.raises(ConfigurationError):
        PlantConfig(encoder_noise_rad=-1.0)
    with pytest.raises(ConfigurationError):
        PlantConfig(torque_limit_nm=0.0)


def test_plant_rejects_bad_step_and_measure():
    plant = AttitudePlant(seed=0)
    with pytest.raises(ConfigurationError):
        plant.step(-0.001)
    with pytest.raises(ConfigurationError):
        plant.measure(0.0)


def test_pd_gains_closed_loop_known_answer():
    # I = 12, wn = 1 rad/s -> kp = I wn^2 = 12; zeta = 0.7 -> kd = 2 zeta wn I = 16.8.
    gains = PDGains(kp=12.0, kd=16.8)
    assert gains.natural_frequency_rad_s(12.0) == pytest.approx(1.0, rel=1e-12)
    assert gains.damping_ratio(12.0) == pytest.approx(0.7, rel=1e-12)


def test_pd_gains_validation():
    with pytest.raises(ConfigurationError):
        PDGains(kp=0.0)
    with pytest.raises(ConfigurationError):
        PDGains(kd=-1.0)
    with pytest.raises(ConfigurationError):
        PDGains().natural_frequency_rad_s(0.0)
    with pytest.raises(ConfigurationError):
        PDGains().damping_ratio(-1.0)


def test_pd_controller_known_answer():
    # kp=12, kd=16.8, theta_cmd=0: u = -12*0.05 - 16.8*0.01 = -0.6 - 0.168 = -0.768
    controller = PDController(PDGains(kp=12.0, kd=16.8))
    assert controller.command(np.array([0.05, 0.01]))[0] == pytest.approx(-0.768, rel=1e-14)
    # With a command of 0.05 rad the proportional term vanishes.
    biased = PDController(PDGains(kp=12.0, kd=16.8), theta_cmd_rad=0.05)
    assert biased.command(np.array([0.05, 0.0]))[0] == pytest.approx(0.0)


def test_pd_controller_rejects_wrong_shape():
    with pytest.raises(ValueError):
        PDController().command(np.array([0.0, 0.0, 0.0]))


def test_pd_loop_settles_the_attitude():
    plant = AttitudePlant(PlantConfig(theta0_rad=0.08), seed=7)
    controller = PDController()
    for _ in range(600):
        sample = plant.measure(0.01)
        plant.apply_torque(float(controller.command(sample)[0]))
        plant.step(0.01)
    assert abs(plant.state[0]) < 0.01
