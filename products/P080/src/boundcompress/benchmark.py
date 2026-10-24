"""Predictor benchmark: learned against classical, on rate, at identical distortion.

The classical predictors are implemented first and are never retuned away from
a win. This module exists to produce one number per ``(signal, bound,
predictor)`` cell and then to say plainly who won, including when the answer
is that a two-tap integer filter matched a neural network that costs kilobytes
to ship.

Three accounting rules, all of them deliberate:

1. **Rate is bytes.** ``encoded_bytes`` of a container that round-trips.
2. **Distortion is identical, not matched.** Every predictor shares the
   quantiser, so comparing rates is comparing one number.
3. **The model is charged.** A learned predictor's parameters have to be on
   the spacecraft and on the ground. :func:`break_even_arrays` reports how
   many arrays the mission must send before the bytes saved pay for the bytes
   shipped, and it is reported whether or not it flatters the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .codec import encode
from .predictors import CLASSICAL_NAMES


@dataclass(frozen=True)
class Cell:
    """One ``(signal, bound, predictor)`` rate measurement at identical distortion."""

    signal: str
    bound: float
    predictor: str
    encoded_bytes: int
    bits_per_sample: float
    entropy_bits: float
    escape_count: int
    bound_holds: bool
    max_error: float


@dataclass(frozen=True)
class Verdict:
    """The outcome of a benchmark, phrased so it cannot be read two ways.

    Attributes
    ----------
    cells:
        Every measurement.
    classical_wins, learned_wins:
        Number of ``(signal, bound)`` cells where the lowest rate belongs to a
        classical predictor and to a learned one.
    median_learned_gain_pct, max_learned_gain_pct, worst_learned_loss_pct:
        Percentage rate reduction of the best learned predictor against the
        best classical predictor, over all cells. Negative means the learned
        predictor used more bytes.
    best_classical_overall, best_learned_overall:
        The predictor with the lowest total bytes in each family.
    """

    cells: tuple[Cell, ...]
    classical_wins: int
    learned_wins: int
    median_learned_gain_pct: float
    max_learned_gain_pct: float
    worst_learned_loss_pct: float
    best_classical_overall: str
    best_learned_overall: str


def run(
    signals: dict[str, np.ndarray],
    bounds_for: dict[str, list[float]],
    models: dict[str, object],
    *,
    mode: str = "abs",
    classical: tuple[str, ...] = CLASSICAL_NAMES,
) -> list[Cell]:
    """Measure every classical predictor and every supplied learned model.

    Parameters
    ----------
    signals:
        Name to 1-D float64 array.
    bounds_for:
        Name to the list of declared bounds for that signal.
    models:
        Label to fitted learned predictor. The label appears in
        :attr:`Cell.predictor` as ``learned_<label>``.
    mode:
        Declared bound mode, the same for every cell.
    classical:
        Classical predictor names to include.
    """
    cells: list[Cell] = []
    for name, values in signals.items():
        for bound in bounds_for[name]:
            for predictor in classical:
                certificate = encode(values, mode, bound, predictor=predictor).certificate
                cells.append(
                    Cell(
                        signal=name,
                        bound=float(bound),
                        predictor=predictor,
                        encoded_bytes=certificate.encoded_bytes,
                        bits_per_sample=certificate.bits_per_sample,
                        entropy_bits=certificate.residual_entropy_bits,
                        escape_count=certificate.escape_count,
                        bound_holds=certificate.bound_holds,
                        max_error=(
                            certificate.max_relative_error
                            if mode == "rel"
                            else certificate.max_absolute_error
                        ),
                    )
                )
            for label, model in models.items():
                certificate = encode(
                    values, mode, bound, predictor="learned", model=model
                ).certificate
                cells.append(
                    Cell(
                        signal=name,
                        bound=float(bound),
                        predictor=f"learned_{label}",
                        encoded_bytes=certificate.encoded_bytes,
                        bits_per_sample=certificate.bits_per_sample,
                        entropy_bits=certificate.residual_entropy_bits,
                        escape_count=certificate.escape_count,
                        bound_holds=certificate.bound_holds,
                        max_error=(
                            certificate.max_relative_error
                            if mode == "rel"
                            else certificate.max_absolute_error
                        ),
                    )
                )
    return cells


def summarise(cells: list[Cell], *, classical: tuple[str, ...] = CLASSICAL_NAMES) -> Verdict:
    """Reduce a list of cells to a :class:`Verdict`.

    Raises
    ------
    ValueError
        If ``cells`` is empty, or if no cell belongs to either family, in which
        case there is no comparison to summarise.
    """
    if not cells:
        raise ValueError("no cells to summarise")
    keys = sorted({(cell.signal, cell.bound) for cell in cells})
    classical_set = set(classical)

    gains: list[float] = []
    classical_wins = 0
    learned_wins = 0
    for signal, bound in keys:
        group = [cell for cell in cells if cell.signal == signal and cell.bound == bound]
        classical_cells = [cell for cell in group if cell.predictor in classical_set]
        learned_cells = [cell for cell in group if cell.predictor.startswith("learned_")]
        if not classical_cells or not learned_cells:
            continue
        best_classical = min(classical_cells, key=lambda cell: cell.encoded_bytes)
        best_learned = min(learned_cells, key=lambda cell: cell.encoded_bytes)
        gains.append(
            100.0
            * (best_classical.encoded_bytes - best_learned.encoded_bytes)
            / best_classical.encoded_bytes
        )
        if best_learned.encoded_bytes < best_classical.encoded_bytes:
            learned_wins += 1
        else:
            classical_wins += 1
    if not gains:
        raise ValueError("no cell has both a classical and a learned measurement")

    totals: dict[str, int] = {}
    for cell in cells:
        totals[cell.predictor] = totals.get(cell.predictor, 0) + cell.encoded_bytes
    classical_totals = {k: v for k, v in totals.items() if k in classical_set}
    learned_totals = {k: v for k, v in totals.items() if k.startswith("learned_")}

    array = np.asarray(gains, dtype=np.float64)
    return Verdict(
        cells=tuple(cells),
        classical_wins=classical_wins,
        learned_wins=learned_wins,
        median_learned_gain_pct=float(np.median(array)),
        max_learned_gain_pct=float(np.max(array)),
        worst_learned_loss_pct=float(np.min(array)),
        best_classical_overall=min(classical_totals, key=classical_totals.get),
        best_learned_overall=min(learned_totals, key=learned_totals.get),
    )


def break_even_arrays(
    cells: list[Cell], learned_label: str, model_bytes: int,
    *, classical: tuple[str, ...] = CLASSICAL_NAMES
) -> float:
    """Arrays the mission must send before a learned predictor pays for itself.

    ``model_bytes`` divided by the mean bytes saved per array against the best
    classical predictor. Returns ``inf`` when the mean saving is zero or
    negative, which is the honest answer and not an error: a predictor that
    does not save bytes never pays for the bytes it costs.
    """
    predictor = f"learned_{learned_label}"
    classical_set = set(classical)
    savings: list[float] = []
    for signal, bound in sorted({(cell.signal, cell.bound) for cell in cells}):
        group = [cell for cell in cells if cell.signal == signal and cell.bound == bound]
        classical_cells = [cell for cell in group if cell.predictor in classical_set]
        learned_cells = [cell for cell in group if cell.predictor == predictor]
        if not classical_cells or not learned_cells:
            continue
        best_classical = min(classical_cells, key=lambda cell: cell.encoded_bytes)
        savings.append(float(best_classical.encoded_bytes - learned_cells[0].encoded_bytes))
    if not savings:
        raise ValueError(f"no cell for {predictor!r}")
    mean_saving = float(np.mean(savings))
    if mean_saving <= 0.0:
        return float("inf")
    return model_bytes / mean_saving
