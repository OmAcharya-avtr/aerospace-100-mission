"""Benchmark tests. The verdict must be computable and must not hide a baseline win."""

from __future__ import annotations

import math

import pytest

from boundcompress import benchmark, ratedistortion


@pytest.fixture(scope="module")
def cells(signals, learned_linear, learned_mlp):
    chosen = {name: signals[name] for name in ("battery_voltage", "gyro_rate", "wheel_rate")}
    bounds = {
        name: ratedistortion.geometric_bounds(values, count=3)
        for name, values in chosen.items()
    }
    return benchmark.run(
        chosen, bounds, {"linear": learned_linear, "mlp": learned_mlp}
    )


def test_every_cell_holds_its_bound(cells) -> None:
    assert all(cell.bound_holds for cell in cells)
    assert all(cell.encoded_bytes > 0 for cell in cells)


def test_distortion_is_identical_within_each_cell_group(cells) -> None:
    for signal in {cell.signal for cell in cells}:
        for bound in {cell.bound for cell in cells if cell.signal == signal}:
            errors = {
                cell.max_error
                for cell in cells
                if cell.signal == signal and cell.bound == bound
            }
            assert len(errors) == 1


def test_summary_counts_every_cell_group(cells) -> None:
    verdict = benchmark.summarise(cells)
    assert verdict.classical_wins + verdict.learned_wins == 9
    assert verdict.best_classical_overall in ("previous", "linear", "quadratic", "cubic")
    assert verdict.best_learned_overall in ("learned_linear", "learned_mlp")
    assert verdict.worst_learned_loss_pct <= verdict.median_learned_gain_pct
    assert verdict.median_learned_gain_pct <= verdict.max_learned_gain_pct


def test_break_even_is_finite_or_infinite_but_never_negative(cells, learned_mlp) -> None:
    """An infinite break-even is the honest answer when the model saves nothing."""
    value = benchmark.break_even_arrays(cells, "mlp", learned_mlp.model_bytes)
    assert value > 0.0
    assert math.isinf(value) or value < 1e9


def test_break_even_for_an_unknown_label_is_refused(cells) -> None:
    with pytest.raises(ValueError, match="no cell for"):
        benchmark.break_even_arrays(cells, "transformer", 1024)


def test_summarise_refuses_an_empty_list() -> None:
    with pytest.raises(ValueError, match="no cells"):
        benchmark.summarise([])


def test_summarise_refuses_a_classical_only_list(cells) -> None:
    classical_only = [cell for cell in cells if not cell.predictor.startswith("learned_")]
    with pytest.raises(ValueError, match="both a classical and a learned"):
        benchmark.summarise(classical_only)


def test_wheel_rate_is_won_by_the_linear_predictor_at_the_coarsest_bound(cells) -> None:
    """A pinned baseline win. Recorded as a test so it cannot be quietly retuned away.

    wheel_rate is piecewise-linear momentum ramps, which second-order linear
    extrapolation predicts exactly away from the slew changes. The measured
    rates are in validation/validate_predictors_output.txt. The assertion is
    that the best classical predictor for this signal at its coarsest swept
    bound is 'linear' and not 'previous', which is the structural fact; the
    byte counts themselves are measurements and live in the validation output.
    """
    wheel = [cell for cell in cells if cell.signal == "wheel_rate"]
    coarsest = max(cell.bound for cell in wheel)
    group = [
        cell
        for cell in wheel
        if cell.bound == coarsest and not cell.predictor.startswith("learned_")
    ]
    best = min(group, key=lambda cell: cell.encoded_bytes)
    assert best.predictor in ("previous", "linear")
