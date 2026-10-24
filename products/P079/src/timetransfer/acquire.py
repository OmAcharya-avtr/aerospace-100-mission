"""Acquisition: drive a backend, count what was lost, survive losing the link.

What the acquisition guarantees
-------------------------------
* It attempts exactly ``n_attempts`` exchanges, in order, through
  :meth:`~timetransfer.hal.Backend.next_exchange`.
* A lost exchange is counted and the sequence number is skipped in the output,
  so a gap in :attr:`~timetransfer.exchange.ExchangeSet.seq` is a visible loss
  rather than a silent one.
* A run of ``max_consecutive_losses`` consecutive losses stops the
  acquisition and is reported as ``stopped_early``. This is the case the
  recovery procedure in ``docs/DEPLOYMENT.md`` is written for: a burst of
  losses means the link is gone, not that this exchange was unlucky, and
  continuing to hammer a dead link wastes the acquisition window.
* Whatever completed before the stop is returned. The estimate from a
  truncated acquisition is still an estimate; what changes is its uncertainty,
  which :mod:`timetransfer.uncertainty` computes from the data that actually
  arrived rather than from the data that was planned.
* ``min_exchanges`` is checked last. Fewer than that and the acquisition raises
  :class:`~timetransfer.errors.InsufficientDataError` carrying both counts, so
  a caller can decide between extending the window and abandoning.

Why a loss is not an exception here
-----------------------------------
A two-way link drops exchanges as a matter of course. Treating each one as an
exception would mean the normal case travelled by the exceptional path. The
backend converts a driver-level
:class:`~timetransfer.errors.ExchangeLostError` into a lost outcome, and only a
fault -- a missing device, a clock regression -- propagates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .errors import ConfigurationError, InsufficientDataError
from .exchange import ExchangeSet
from .hal import Backend

__all__ = ["AcquisitionReport", "acquire"]


@dataclass(frozen=True)
class AcquisitionReport:
    """What an acquisition did, beside the data it returned.

    Attributes
    ----------
    exchanges:
        The completed exchanges.
    n_attempted:
        Attempts made, including losses and including none made after an early
        stop.
    n_lost:
        Attempts that did not complete.
    longest_loss_run:
        Longest run of consecutive losses observed.
    stopped_early:
        ``True`` if the acquisition stopped on a loss run rather than on
        reaching ``n_attempts``.
    stopped_at_seq:
        Sequence number of the attempt the stop happened on, or ``-1``.
    backend_kind, backend_driver, is_hardware:
        Copied from the backend, so the report cannot be separated from what
        produced it.
    """

    exchanges: ExchangeSet
    n_attempted: int
    n_lost: int
    longest_loss_run: int
    stopped_early: bool
    stopped_at_seq: int
    backend_kind: str
    backend_driver: str
    is_hardware: bool

    @property
    def completion_fraction(self) -> float:
        """Completed divided by attempted; ``0.0`` for an empty acquisition."""
        return float(self.exchanges.n) / self.n_attempted if self.n_attempted else 0.0

    def as_text(self) -> str:
        """Compact human-readable summary."""
        return (
            f"attempted          : {self.n_attempted}\n"
            f"completed          : {self.exchanges.n}\n"
            f"lost               : {self.n_lost}\n"
            f"longest loss run   : {self.longest_loss_run}\n"
            f"stopped early      : {self.stopped_early} (at seq {self.stopped_at_seq})\n"
            f"backend            : {self.backend_kind}/{self.backend_driver or 'n/a'}\n"
            f"is_hardware        : {self.is_hardware}"
        )


def acquire(
    backend: Backend,
    n_attempts: int,
    *,
    min_exchanges: int = 2,
    max_consecutive_losses: int = 1_000_000_000,
    keep_truth: bool = True,
) -> AcquisitionReport:
    """Drive ``backend`` for ``n_attempts`` exchanges and return what arrived.

    Parameters
    ----------
    backend:
        An open backend. Passing a closed one raises
        :class:`~timetransfer.errors.ConfigurationError`, rather than opening
        it silently: opening a device is a side effect the caller owns.
    n_attempts:
        Number of exchanges to attempt, must be >= 1.
    min_exchanges:
        Minimum completed exchanges for the acquisition to be considered
        usable, must be >= 1.
    max_consecutive_losses:
        Stop after this many consecutive losses. The default is effectively no
        limit; a deployment sets it to something like 10.
    keep_truth:
        Record the simulation truth columns when the backend supplies them.
        A device backend supplies none and the columns are ``None``.

    Returns
    -------
    AcquisitionReport

    Raises
    ------
    InsufficientDataError
        Fewer than ``min_exchanges`` completed.
    """
    if int(n_attempts) < 1:
        raise ConfigurationError(f"n_attempts must be >= 1, got {n_attempts!r}")
    if int(min_exchanges) < 1:
        raise ConfigurationError(f"min_exchanges must be >= 1, got {min_exchanges!r}")
    if int(max_consecutive_losses) < 1:
        raise ConfigurationError(
            f"max_consecutive_losses must be >= 1, got {max_consecutive_losses!r}"
        )
    if not backend.is_open:
        raise ConfigurationError(
            "acquire() requires an open backend; call backend.open() first"
        )

    t1: list[int] = []
    t2: list[int] = []
    t3: list[int] = []
    t4: list[int] = []
    seqs: list[int] = []
    truth_offset: list[float] = []
    truth_asym: list[float] = []
    have_truth = True
    n_lost = 0
    run = 0
    longest_run = 0
    stopped_early = False
    stopped_at = -1
    attempted = 0

    for seq in range(int(n_attempts)):
        outcome = backend.next_exchange(seq)
        attempted += 1
        if outcome.lost or outcome.exchange is None:
            n_lost += 1
            run += 1
            longest_run = max(longest_run, run)
            if run >= int(max_consecutive_losses):
                stopped_early = True
                stopped_at = seq
                break
            continue
        run = 0
        ex = outcome.exchange
        t1.append(ex.t1_ns)
        t2.append(ex.t2_ns)
        t3.append(ex.t3_ns)
        t4.append(ex.t4_ns)
        seqs.append(ex.seq)
        if keep_truth and outcome.truth is not None:
            truth_offset.append(float(outcome.truth.offset_s))
            truth_asym.append(float(outcome.truth.asymmetry_s))
        else:
            have_truth = False

    info = backend.info
    exchanges = ExchangeSet(
        t1_ns=np.asarray(t1, dtype=np.int64),
        t2_ns=np.asarray(t2, dtype=np.int64),
        t3_ns=np.asarray(t3, dtype=np.int64),
        t4_ns=np.asarray(t4, dtype=np.int64),
        seq=np.asarray(seqs, dtype=np.int64),
        n_lost=n_lost,
        backend_kind=info.kind,
        backend_driver=info.driver,
        is_hardware=info.is_hardware,
        true_offset_s=(
            np.asarray(truth_offset, dtype=np.float64)
            if have_truth and len(truth_offset) == len(t1) and t1
            else None
        ),
        true_asymmetry_s=(
            np.asarray(truth_asym, dtype=np.float64)
            if have_truth and len(truth_asym) == len(t1) and t1
            else None
        ),
    )
    report = AcquisitionReport(
        exchanges=exchanges,
        n_attempted=attempted,
        n_lost=n_lost,
        longest_loss_run=longest_run,
        stopped_early=stopped_early,
        stopped_at_seq=stopped_at,
        backend_kind=info.kind,
        backend_driver=info.driver,
        is_hardware=info.is_hardware,
    )
    if exchanges.n < int(min_exchanges):
        raise InsufficientDataError(
            f"acquisition completed {exchanges.n} of {attempted} attempted exchanges, "
            f"below the minimum of {min_exchanges}; longest loss run was {longest_run}"
            + (f" and the run stopped early at seq {stopped_at}" if stopped_early else ""),
            n_usable=exchanges.n,
            n_required=int(min_exchanges),
        )
    return report
