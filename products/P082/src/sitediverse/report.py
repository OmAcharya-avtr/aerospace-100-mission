"""Table and line rendering. No library module prints; this one formats strings.

Every function here returns a string. Nothing in ``src/`` calls ``print``: the
CLI of ``sitediverse.__main__`` and the scripts under ``examples/`` and
``validation/`` are the only things that write to standard output.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A plain left-aligned table with a dashed rule under the header.

    Parameters
    ----------
    headers:
        Column titles.
    rows:
        Rows of already-formatted strings. A row with the wrong number of
        columns raises rather than being padded, because a silently padded row
        in a validation output is a wrong number presented as a right one.

    Returns
    -------
    str
        The table, without a trailing newline.
    """
    titles = [str(header) for header in headers]
    if not titles:
        raise ValueError("at least one column is required")
    body = [[str(cell) for cell in row] for row in rows]
    for index, row in enumerate(body):
        if len(row) != len(titles):
            raise ValueError(
                f"row {index} has {len(row)} cells but there are {len(titles)} "
                "columns"
            )
    widths = [
        max(len(titles[column]), *(len(row[column]) for row in body))
        if body
        else len(titles[column])
        for column in range(len(titles))
    ]
    lines = [
        "  ".join(title.ljust(width) for title, width in zip(titles, widths, strict=True)),
        "  ".join("-" * width for width in widths),
    ]
    lines += [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
        for row in body
    ]
    return "\n".join(line.rstrip() for line in lines)


def number(value: float, decimals: int = 6) -> str:
    """Fixed-point rendering, with ``inf`` and ``nan`` spelled out."""
    probe = float(value)
    if np.isnan(probe):
        return "nan"
    if np.isposinf(probe):
        return "inf"
    if np.isneginf(probe):
        return "-inf"
    return f"{probe:.{decimals}f}"


def scientific(value: float, digits: int = 6) -> str:
    """Scientific rendering, with ``inf`` and ``nan`` spelled out."""
    probe = float(value)
    if np.isnan(probe):
        return "nan"
    if np.isinf(probe):
        return "inf" if probe > 0 else "-inf"
    return f"{probe:.{digits}e}"


def interval(lower: float, upper: float, decimals: int = 6) -> str:
    """``[lower, upper]`` to a fixed number of decimals."""
    return f"[{number(lower, decimals)}, {number(upper, decimals)}]"


def plus_minus(value: float, spread: float, decimals: int = 6) -> str:
    """``value +/- spread``, the form every availability figure is reported in."""
    return f"{number(value, decimals)} +/- {number(spread, decimals)}"


def verdict(passed: bool) -> str:
    """``PASS`` or ``FAIL``."""
    return "PASS" if passed else "FAIL"
