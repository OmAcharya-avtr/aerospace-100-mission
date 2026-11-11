"""Plain-text rendering. The provenance comes before the number, every time.

One structural rule, and it is the same rule as everywhere else in this
package: a table of figures is always preceded by the certificate headline for
those figures, and the headline is built by
``boardprobe.certify.Certificate.headline``, which cannot produce a bare
``MEASURED``. There is no renderer here that takes a value without an
attribution.
"""

from __future__ import annotations

from collections.abc import Sequence

from .certify import Certificate
from .channels import BoardDeclaration
from .harness import BenchmarkResult, OverheadReport
from .provenance import ORDER, Provenance, meet
from .readings import Reading
from .statistics import Statistic


def format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """A fixed-width table. Columns are as wide as their widest cell."""
    columns = len(headers)
    widths = [len(str(header)) for header in headers]
    for row in rows:
        if len(row) != columns:
            raise ValueError(
                f"row {list(row)!r} has {len(row)} cells and the header has {columns}"
            )
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append(
            "  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True)).rstrip()
        )
    return "\n".join(lines)


def render_certificate(certificate: Certificate) -> str:
    """The headline, then what it rests on. Three lines."""
    attribution = certificate.attribution
    return "\n".join(
        (
            f"{certificate.subject}",
            f"  {certificate.headline}",
            f"  rests on {attribution.reading_count} readings from "
            f"{', '.join(attribution.source_ids)}; hardware measurement on the declared "
            f"target: {'yes' if certificate.is_hardware_measurement else 'NO'}",
        )
    )


def render_readings(readings: Sequence[Reading]) -> str:
    """A reading table with provenance in its own column."""
    rows = [
        [
            reading.channel,
            f"{reading.value:.6g}",
            reading.unit,
            reading.provenance.value,
            reading.detail,
        ]
        for reading in readings
    ]
    return format_table(["channel", "value", "unit", "provenance", "detail"], rows)


def render_statistics(statistics: Sequence[Statistic]) -> str:
    """A statistic table with provenance and the weakest channel in their own columns."""
    rows = [
        [
            statistic.name,
            f"{statistic.value:.6g}",
            statistic.unit,
            statistic.provenance.value,
            ", ".join(statistic.attribution.weakest),
        ]
        for statistic in statistics
    ]
    return format_table(
        ["statistic", "value", "unit", "provenance", "weakest channel"], rows
    )


def render_channels(declaration: BoardDeclaration) -> str:
    """The declared channel table, with the verification flag in its own column."""
    rows = [
        [
            channel.name,
            str(channel.kind),
            channel.unit,
            str(channel.access),
            f"{channel.scale:g}",
            channel.raw_unit,
            "yes" if channel.verified_on_hardware else "NO",
        ]
        for channel in declaration.channels
    ]
    return format_table(
        ["channel", "kind", "unit", "access", "scale", "raw unit", "verified on hw"], rows
    )


def render_lattice() -> str:
    """The meet table of the provenance lattice, printed as a 3x3 grid."""
    rows = [
        [left.value, *[meet(left, right).value for right in ORDER]] for left in ORDER
    ]
    return format_table(["meet", *[p.value for p in ORDER]], rows)


def render_label_rules() -> str:
    """Which label each provenance admits, and what it entitles a reader to."""
    from .certify import label_for

    entitlement = {
        Provenance.LIVE: "a device responded; the platform line says which device",
        Provenance.RECORDED: "a device responded once, earlier, elsewhere",
        Provenance.SIMULATED: "nothing was measured; a generator produced the number",
    }
    rows = [
        [p.value, label_for(p).value, entitlement[p]] for p in reversed(ORDER)
    ]
    return format_table(["provenance", "label", "what a reader may conclude"], rows)


def render_benchmark(result: BenchmarkResult) -> str:
    """A benchmark result: the certificate first, then timing, then resources."""
    blocks = [
        render_certificate(result.overall_certificate),
        "",
        f"timing ({result.repeats} repeats, {result.warmup} warm-up, clock resolution "
        f"{result.timing_resolution_s:.3e} s)",
        render_statistics(result.timing),
    ]
    if result.resources:
        blocks.extend(("", "resources", render_statistics(result.resources)))
    if result.refusals:
        blocks.extend(
            (
                "",
                f"channels the environment backend refused ({len(result.refusals)}); "
                "reported, not dropped",
                format_table(
                    ["channel", "reason"],
                    [[channel, reason[:96]] for channel, reason in result.refusals],
                ),
            )
        )
    return "\n".join(blocks)


def render_overhead(report: OverheadReport) -> str:
    """The harness's own cost, with the certificate that names the host."""
    rows = [
        ["null payload, sampling on", f"{report.null_with_sampling_s * 1e6:.3f}"],
        ["null payload, sampling off", f"{report.null_without_sampling_s * 1e6:.3f}"],
        ["overhead inside the bracket", f"{report.in_bracket_overhead_s * 1e6:.3f}"],
        ["one clock read (independent)", f"{report.clock_read_cost_s * 1e6:.3f}"],
        ["one sealed Reading", f"{report.reading_cost_s * 1e6:.3f}"],
        ["reference payload, sampling off", f"{report.reference_without_sampling_s * 1e6:.3f}"],
        ["reference payload, sampling on", f"{report.reference_with_sampling_s * 1e6:.3f}"],
    ]
    return "\n".join(
        (
            render_certificate(report.certificate),
            "",
            f"per-repeat cost over {report.repeats} repeats, microseconds",
            format_table(["quantity", "us"], rows),
            "",
            f"perturbation of the reference payload : {report.perturbation_ratio:.4f} x",
            f"clock resolution at this uptime       : {report.timing_resolution_s:.3e} s",
        )
    )
