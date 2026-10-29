"""aligncal -- relative sensor alignment from flight time series, observability first.

The front door is :func:`align`. It computes the observability verdict **before**
the estimate and returns both, with the verdict first, so that no caller can
read an alignment number without the statement of which of its degrees of
freedom the record actually supports.

    >>> from aligncal import align, dataset
    >>> generated = dataset.generate(profile="single_axis", samples=400, seed=5)
    >>> result = align(generated.record)
    >>> result.observability.any_rank_deficient
    True

What this package is not
------------------------
It is not an attitude determination package: it does not estimate where the
vehicle is pointing. It estimates the fixed relative rotation between two rate
sensors from a record of motion, and it reports which components of that
rotation the motion excited. An observable alignment is not a calibrated one:
observability is a statement about the record, not about the instrument, and
nothing here may be applied to flight software.

Research grade. Not flight-qualified, not certified, not approved for
operational aerospace use.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import (
    batch,
    benchmark,
    dataset,
    features,
    geometry,
    learned,
    observability,
    record,
    recursive,
    report,
)
from .observability import ObservabilityReport
from .record import AlignmentRecord

__version__ = "0.1.0"

__all__ = [
    "AlignmentResult",
    "align",
    "batch",
    "benchmark",
    "dataset",
    "features",
    "geometry",
    "learned",
    "observability",
    "record",
    "recursive",
    "report",
    "__version__",
]


@dataclass(frozen=True)
class AlignmentResult:
    """The observability verdict and the estimates it gated, verdict first.

    Attributes
    ----------
    observability:
        The verdict for every secondary. First field, deliberately: it is
        computed before any estimate and it is what gates them.
    batch:
        Mapping from secondary name to the batch solution of equations
        (14)-(16), which withholds every degree of freedom below the noise
        floor of (11).
    recursive:
        Mapping from secondary name to the MEKF solution of (19)-(22), or an
        empty mapping when ``with_recursive`` was false.
    """

    observability: ObservabilityReport
    batch: dict[str, batch.BatchSolution]
    recursive: dict[str, recursive.RecursiveSolution]

    def summary(self) -> list[str]:
        """Verdict, then every estimate, as text."""
        lines = list(self.observability.summary())
        for name in sorted(self.batch):
            lines.append("")
            lines.extend(self.batch[name].summary()[len(self.observability[name].summary()) :])
            if name in self.recursive:
                solution = self.recursive[name]
                lines.extend(solution.summary()[len(self.observability[name].summary()) :])
        return lines


def align(
    measurements: AlignmentRecord,
    *,
    required_sigma_rad: float = observability.DEFAULT_REQUIRED_SIGMA_RAD,
    floor_factor: float = observability.DEFAULT_FLOOR_FACTOR,
    with_recursive: bool = False,
    initial_sigma_rad: float = recursive.DEFAULT_INITIAL_SIGMA_RAD,
) -> AlignmentResult:
    """Assess observability, then estimate the alignment of every secondary.

    The order is not a convention: :func:`aligncal.observability.assess` is
    called first and its result is passed into every estimator, which require
    it. Parameters are those of :func:`aligncal.observability.assess` and
    :func:`aligncal.recursive.run`.

    Returns
    -------
    AlignmentResult
    """
    verdict = observability.assess(
        measurements,
        required_sigma_rad=required_sigma_rad,
        floor_factor=floor_factor,
    )
    batch_solutions = {
        name: batch.solve(measurements, verdict, name) for name in measurements.names
    }
    recursive_solutions = {}
    if with_recursive:
        recursive_solutions = {
            name: recursive.run(
                measurements, verdict, name, initial_sigma_rad=initial_sigma_rad
            )
            for name in measurements.names
        }
    return AlignmentResult(
        observability=verdict, batch=batch_solutions, recursive=recursive_solutions
    )
