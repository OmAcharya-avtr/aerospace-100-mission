"""Result-file writing, with the environment and the empty device column.

Every file this module writes carries, in the file itself:

1. the environment the numbers came from, including whether the host is
   shared and how many cores were available;
2. the measurement method and repeat count for each number;
3. a **target-device column that is left empty**.

Point 3 is the reason this module exists rather than a bare ``print``. This
repository's numbers were measured in a shared single-core cloud container.
That is a workstation-class measurement and it is not a measurement of an
edge target. The results table therefore has a column headed with the target
device name whose cells read ``(not measured)`` until someone runs
``validation/validate_performance.py`` on the device and pastes its raw
output. No number in this repository may be extrapolated into that column,
read out of a vendor datasheet, or produced by the simulated backend --- and
:func:`results_table` has no parameter that would let a caller fill it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edgeinfer.environment import EnvironmentRecord, environment_record

__all__ = [
    "CLOUD_ENVIRONMENT_NOTE",
    "NOT_MEASURED",
    "ResultRow",
    "crosscheck_payload",
    "results_table",
    "write_crosscheck_json",
    "write_results_file",
]

#: The cell text for a device column that has not been measured.
NOT_MEASURED: str = "(not measured)"

#: Appended to the environment line of every file this module writes.
CLOUD_ENVIRONMENT_NOTE: str = (
    "All numbers in this file are WORKSTATION numbers measured in a shared, "
    "single-CPU-core cloud container with other build jobs running concurrently. "
    "They are not measurements of any edge or flight target. The target-device "
    "column is empty because no measurement from that device exists; it must be "
    "filled only from raw output produced on the device itself."
)


@dataclass(frozen=True)
class ResultRow:
    """One quantity, its workstation value and its measurement method.

    Attributes
    ----------
    quantity
        What was measured.
    unit
        Unit of ``workstation_value``.
    workstation_value
        The measured or computed value on the host this ran on, or ``None``.
    uncertainty
        Combined standard uncertainty in the same unit, or ``None``.
    method
        Measurement method including the repeat count. Required: a row
        without a method cannot be written.
    """

    quantity: str
    unit: str
    workstation_value: float | None
    uncertainty: float | None
    method: str

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError(
                f"row {self.quantity!r} has no method; every number in an edgeinfer "
                "result file carries its measurement method and repeat count"
            )

    def cells(self) -> tuple[str, str, str, str, str]:
        """``(quantity, unit, value, uncertainty, device)`` as formatted text."""
        value = NOT_MEASURED if self.workstation_value is None else f"{self.workstation_value:.6g}"
        unc = "-" if self.uncertainty is None else f"{self.uncertainty:.3g}"
        return (self.quantity, self.unit, value, unc, NOT_MEASURED)


def results_table(rows: tuple[ResultRow, ...], device_column: str) -> str:
    """Markdown results table with an empty ``device_column``.

    Parameters
    ----------
    rows
        At least one row.
    device_column
        Header for the target-device column, e.g.
        ``"Jetson Orin Nano (measured on device)"``. Its cells are always
        :data:`NOT_MEASURED`; there is no way to pass a value for them.
    """
    if not rows:
        raise ValueError("results_table needs at least one row")
    if not device_column.strip():
        raise ValueError("device_column must be a non-empty header")
    lines = [
        f"| Quantity | Unit | This host (shared 1-core container) | u (std) | {device_column} |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        quantity, unit, value, unc, device = row.cells()
        lines.append(f"| {quantity} | {unit} | {value} | {unc} | {device} |")
    lines.append("")
    lines.append("Measurement methods:")
    for row in rows:
        lines.append(f"- {row.quantity}: {row.method}")
    return "\n".join(lines)


def write_results_file(
    path: str | Path,
    title: str,
    rows: tuple[ResultRow, ...],
    *,
    device_column: str = "Jetson Orin Nano (measured on device)",
    environment: EnvironmentRecord | None = None,
    extra_sections: tuple[tuple[str, str], ...] = (),
) -> Path:
    """Write a results file whose own text states its environment.

    Parameters
    ----------
    path
        Destination file; parent directories are created.
    title
        File heading.
    rows
        The result rows.
    device_column
        Target-device column header; its cells stay empty.
    environment
        Captured environment; taken from
        :func:`edgeinfer.environment.environment_record` when ``None``.
    extra_sections
        ``(heading, body)`` pairs appended after the table.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    env = environment or environment_record(shared_host=True, note=CLOUD_ENVIRONMENT_NOTE)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        f"# {title}",
        "",
        "## Environment (applies to every number in this file)",
        "",
        env.one_line(),
        "",
        CLOUD_ENVIRONMENT_NOTE,
        "",
        "## Results",
        "",
        results_table(rows, device_column),
        "",
    ]
    for heading, body in extra_sections:
        parts += [f"## {heading}", "", body, ""]
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def crosscheck_payload(
    stages: tuple[dict[str, object], ...],
    n_samples: int,
    seed: int,
    mean_s: float,
    p50_s: float,
    p99_s: float,
) -> dict[str, object]:
    """Build the cross-check payload for an independent reimplementation.

    The schema is fixed by the batch specification so that a sibling product's
    independently written harness can be diffed against this one on the same
    synthetic pipeline:

    .. code-block:: json

        {"stages": [{"name": "...", "mean_s": 0.0, "std_s": 0.0, "dist": "..."}],
         "n_samples": 0, "seed": 0,
         "measured": {"mean_s": 0.0, "p50_s": 0.0, "p99_s": 0.0}}

    The pipeline is fully reproducible from ``stages`` plus ``seed``: see
    :class:`edgeinfer.backends.SimulatedBackend`, whose sampler draws one
    value per stage per sample from ``numpy.random.default_rng(seed)`` in
    stage order.

    Raises
    ------
    ValueError
        If any measured statistic is non-positive, or the ordering
        ``p50 <= p99`` is violated, which would indicate a quantile bug rather
        than a result worth exporting.
    """
    if not stages:
        raise ValueError("crosscheck payload needs at least one stage")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    for name, value in (("mean_s", mean_s), ("p50_s", p50_s), ("p99_s", p99_s)):
        if not value > 0:
            raise ValueError(f"{name} must be > 0, got {value}")
    if p50_s > p99_s:
        raise ValueError(f"p50_s ({p50_s}) must not exceed p99_s ({p99_s})")
    return {
        "stages": [dict(s) for s in stages],
        "n_samples": int(n_samples),
        "seed": int(seed),
        "measured": {"mean_s": float(mean_s), "p50_s": float(p50_s), "p99_s": float(p99_s)},
    }


def write_crosscheck_json(
    path: str | Path,
    payload: dict[str, object],
    *,
    environment: EnvironmentRecord | None = None,
    extra: dict[str, object] | None = None,
) -> Path:
    """Write a cross-check payload as JSON, with its environment attached.

    The schema keys required by the batch specification are written at the top
    level unchanged. The environment and any ``extra`` keys are added
    alongside them, under ``_environment`` and their own names, so a diff on
    the required keys is unaffected.
    """
    env = environment or environment_record(shared_host=True, note=CLOUD_ENVIRONMENT_NOTE)
    body: dict[str, object] = dict(payload)
    body["_environment"] = env.as_dict()
    body["_environment_note"] = CLOUD_ENVIRONMENT_NOTE
    if extra:
        body.update(extra)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return out
