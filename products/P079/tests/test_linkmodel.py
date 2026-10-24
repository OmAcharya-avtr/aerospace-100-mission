"""The link model: asymmetry bookkeeping, queueing, loss and outages."""

from __future__ import annotations

import numpy as np
import pytest

from timetransfer.errors import ConfigurationError
from timetransfer.linkmodel import LinkModel, LinkSpec


def test_static_asymmetry_and_bias_known_answer() -> None:
    # f_ab = 5.4 ms, f_ba = 4.6 ms -> A = 0.8 ms, bias = A/2 = 0.4 ms.
    spec = LinkSpec(floor_ab_s=5.4e-3, floor_ba_s=4.6e-3)
    assert spec.static_asymmetry_s == pytest.approx(8.0e-4, rel=1e-12)
    assert spec.static_bias_s == pytest.approx(4.0e-4, rel=1e-12)


def test_mean_queue_asymmetry_known_answer() -> None:
    # shape*scale differences: 2*3e-4 - 1*2e-4 = 6e-4 - 2e-4 = 4e-4 s
    spec = LinkSpec(
        jitter_shape_ab=2.0,
        jitter_scale_ab_s=3.0e-4,
        jitter_shape_ba=1.0,
        jitter_scale_ba_s=2.0e-4,
    )
    assert spec.mean_queue_asymmetry_s == pytest.approx(4.0e-4, rel=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"floor_ab_s": -1.0}, "floor_ab_s"),
        ({"jitter_shape_ab": 0.0}, "jitter_shape_ab"),
        ({"jitter_scale_ba_s": -1e-9}, "jitter_scale_ba_s"),
        ({"loss_probability": 1.0}, "loss_probability"),
        ({"loss_probability": -0.1}, "loss_probability"),
        ({"outage_windows": ((5.0, 5.0),)}, "end > start"),
        ({"outage_windows": ((5.0,),)}, "pair"),
    ],
)
def test_link_spec_validates(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        LinkSpec(**kwargs)


def test_in_outage_is_half_open() -> None:
    spec = LinkSpec(outage_windows=((10.0, 20.0),))
    assert not spec.in_outage(9.999)
    assert spec.in_outage(10.0)
    assert spec.in_outage(19.999)
    assert not spec.in_outage(20.0)


def test_queueing_mean_matches_the_gamma_parameters() -> None:
    # Gamma(k, theta) has mean k*theta. 4000 draws of Gamma(2, 2e-4) have a
    # mean of 4e-4 with a standard error of sqrt(2)*2e-4/sqrt(4000) = 4.5e-6,
    # so a 5 per cent tolerance is about 4 standard errors.
    spec = LinkSpec(
        floor_ab_s=0.0, floor_ba_s=0.0, jitter_shape_ab=2.0, jitter_scale_ab_s=2.0e-4
    )
    model = LinkModel(spec, seed=11)
    draws = np.asarray([model.draw(float(i))[0] for i in range(4000)])
    assert float(np.mean(draws)) == pytest.approx(4.0e-4, rel=0.05)
    assert float(np.min(draws)) >= 0.0


def test_loss_rate_matches_the_declared_probability() -> None:
    spec = LinkSpec(loss_probability=0.2)
    model = LinkModel(spec, seed=5)
    lost = sum(int(model.draw(float(i))[2]) for i in range(5000))
    # 5000 Bernoulli(0.2) draws: mean 1000, sd = sqrt(5000*0.2*0.8) = 28.3.
    assert abs(lost - 1000) < 5 * 28.3


def test_outage_window_loses_every_exchange_inside_it() -> None:
    spec = LinkSpec(loss_probability=0.0, outage_windows=((100.0, 110.0),))
    model = LinkModel(spec, seed=2)
    inside = [model.draw(float(t))[2] for t in range(100, 110)]
    outside = [model.draw(float(t))[2] for t in range(110, 120)]
    assert all(inside)
    assert not any(outside)


def test_draw_consumes_randomness_on_a_lost_exchange_too() -> None:
    # The realisation must not depend on which exchanges survived, so the loss
    # draw and both queue draws happen on every call.
    lossy = LinkModel(LinkSpec(loss_probability=0.5), seed=9)
    clean = LinkModel(LinkSpec(loss_probability=0.0), seed=9)
    lossy_delays = [lossy.draw(float(i))[:2] for i in range(50)]
    clean_delays = [clean.draw(float(i))[:2] for i in range(50)]
    assert lossy_delays == clean_delays


def test_snapshot_restore_reproduces_the_draws() -> None:
    model = LinkModel(LinkSpec(loss_probability=0.1), seed=4)
    for i in range(10):
        model.draw(float(i))
    snap = model.snapshot()
    first = [model.draw(float(i)) for i in range(10, 20)]
    model.restore(snap)
    second = [model.draw(float(i)) for i in range(10, 20)]
    assert first == second
    assert model.draws == int(snap["draws"]) + 10
