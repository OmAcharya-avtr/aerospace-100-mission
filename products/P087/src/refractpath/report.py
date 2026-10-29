"""Plain-text table and report formatting. No library code prints; this returns strings."""

from __future__ import annotations

from collections.abc import Sequence

from .disagreement import SpreadPoint, ValidityFloor
from .verdict import ValidityVerdict


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a fixed-width table.

    Args:
        headers: Column headers.
        rows: Rows of already-formatted cells. Short rows are padded.

    Returns:
        The table as a single string with no trailing newline.

    Raises:
        ValueError: If a row is longer than the header.
    """
    columns = len(headers)
    padded: list[list[str]] = []
    for row in rows:
        cells = [str(cell) for cell in row]
        if len(cells) > columns:
            raise ValueError(f"row has {len(cells)} cells, header has {columns}")
        padded.append(cells + [""] * (columns - len(cells)))
    widths = [len(str(header)) for header in headers]
    for row in padded:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)).rstrip()]
    lines.append("  ".join("-" * widths[i] for i in range(columns)))
    for row in padded:
        lines.append("  ".join(row[i].ljust(widths[i]) for i in range(columns)).rstrip())
    return "\n".join(lines)


def format_spread_point(point: SpreadPoint) -> str:
    """Render one elevation's ensemble as a table."""
    rows = [
        [name, f"{value:12.5f}"]
        for name, value in sorted(point.values_arcsec.items(), key=lambda item: item[1])
    ]
    for name in point.undefined_members:
        rows.append([name, "undefined"])
    table = format_table(["method", "refraction (arcsec)"], rows)
    summary = (
        f"elevation {point.elevation_deg:.4f} deg   "
        f"spread {point.spread_arcsec:.5f} arcsec   "
        f"widest pair {point.minimum_name} / {point.maximum_name}"
    )
    return f"{summary}\n{table}"


def format_validity_floor(floor: ValidityFloor) -> str:
    """Render the measured floor and the evidence for it."""
    lines = [
        f"declared tolerance          : {floor.tolerance_arcsec:.4f} arcsec",
        f"measured validity floor     : {floor.floor_deg:.5f} deg apparent elevation",
        (
            f"bracket                     : [{floor.bracket_deg[0]:.5f}, "
            f"{floor.bracket_deg[1]:.5f}] deg, width {floor.bracket_width_deg:.3e} deg"
        ),
        (
            f"spread at bracket edges     : {floor.spread_below_arcsec:.5f} arcsec "
            f"below, {floor.spread_above_arcsec:.5f} arcsec above"
        ),
        f"tolerance crossings on grid : {floor.crossings}",
        f"spread monotone on grid     : {floor.monotone_on_grid}",
        f"exceeded at grid top        : {floor.exceeded_at_highest_grid_point}",
        f"never exceeded on grid      : {floor.never_exceeded}",
        f"bisection iterations        : {floor.iterations}",
        f"grid points                 : {floor.grid_deg.size}",
    ]
    return "\n".join(lines)


def format_verdict(verdict: ValidityVerdict) -> str:
    """Render the validity verdict, reasons first."""
    lines = [
        f"VERDICT: {verdict.verdict.upper()}",
        f"  apparent elevation        : {verdict.apparent_elevation_deg:.5f} deg",
        f"  declared tolerance        : {verdict.tolerance_arcsec:.4f} arcsec",
        f"  cross-model disagreement  : {verdict.disagreement_arcsec:.5f} arcsec",
        f"  profile envelope, eq (23) : {verdict.profile_envelope_arcsec:.5f} arcsec",
        f"  measured validity floor   : {verdict.validity_floor_deg:.5f} deg",
        f"  above the floor           : {verdict.above_validity_floor}",
        (
            "  plane-parallel hard floor : "
            f"{verdict.plane_parallel_analytic_floor_deg:.5f} deg (analytic, eq 13)"
        ),
        (
            "  oriani breakdown elevation: "
            f"{verdict.oriani_breakdown_elevation_deg:.5f} deg (computed, eq 14a)"
        ),
        f"  refractivity rel. spread  : {verdict.refractivity_relative_spread:.3e}",
    ]
    if verdict.models_undefined:
        lines.append(f"  undefined methods         : {', '.join(verdict.models_undefined)}")
    if verdict.reasons:
        lines.append("  reasons:")
        lines.extend(f"    - {reason}" for reason in verdict.reasons)
    else:
        lines.append("  reasons: none; every gate passed")
    lines.append("  cannot be determined from these inputs at any elevation:")
    lines.extend(f"    - {item}" for item in verdict.undeterminable)
    return "\n".join(lines)
