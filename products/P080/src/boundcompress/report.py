"""Rendering a certificate as text. No library code prints; these return strings."""

from __future__ import annotations

from .certificate import Certificate


def format_certificate(certificate: Certificate) -> str:
    """Render a :class:`Certificate` as a fixed-width block.

    The verdict line comes first and says HOLDS or FAILED. A reader who looks
    at nothing else should still see whether the bound held.
    """
    verdict = "HOLDS" if certificate.bound_holds else "FAILED"
    lines = [
        f"bound verdict          : {verdict}",
        f"declared mode          : {certificate.declared_mode}",
        f"declared bound         : {certificate.declared_bound:.6e}",
    ]
    if certificate.declared_mode != "rel":
        lines.append(
            f"effective abs bound    : {certificate.effective_absolute_bound:.6e}"
        )
    lines += [
        f"max absolute error     : {certificate.max_absolute_error:.6e}",
        f"max relative error     : {certificate.max_relative_error:.6e}",
        f"worst sample index     : {certificate.argmax_index}",
        f"worst sample value     : {certificate.argmax_value:.9g}",
        f"worst reconstruction   : {certificate.argmax_reconstruction:.9g}",
        f"margin to bound        : {certificate.margin:.6e}",
        f"violating samples      : {certificate.violations} of {certificate.samples}",
    ]
    if certificate.violation_indices:
        lines.append(f"first violating indices: {list(certificate.violation_indices)}")
    lines += [
        f"predictor              : {certificate.predictor}",
        f"original bytes         : {certificate.original_bytes}",
        f"encoded bytes          : {certificate.encoded_bytes}",
        f"bits per sample        : {certificate.bits_per_sample:.6f}",
        f"compression ratio      : {certificate.compression_ratio:.6f}",
        f"residual entropy bits  : {certificate.residual_entropy_bits:.6f}"
        "  (reference, not a floor)",
        f"rate below entropy ref : {certificate.rate_below_entropy_reference}",
        f"escaped residuals      : {certificate.escape_count}",
        f"rounding corrections   : {certificate.corrections}",
        f"container sha256       : {certificate.digest}",
    ]
    lines += [f"note                   : {note}" for note in certificate.notes]
    return "\n".join(lines)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a fixed-width table. Used by the CLI and the validation scripts."""
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    out = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    out.append("  ".join("-" * width for width in widths))
    for row in rows:
        out.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(out)
