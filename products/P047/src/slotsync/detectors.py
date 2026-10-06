"""Timing-error detectors (TEDs) for sampled OOK and PPM receivers.

Sign convention (binding for the whole package)
-----------------------------------------------
Every detector in this module is written so that

    **a positive output means the sampling instant is late**

relative to the pulse centre.  The loop in :mod:`slotsync.loop` therefore
*subtracts* the filtered detector output from its timing estimate.  Some
published arrangements of the same detectors carry the opposite sign; the sign
is a wiring convention, not a property of the detector, and where this module's
arrangement differs from the usual printed one the docstring says so.

Each detector is a pure function of already-taken samples.  Nothing here
resamples, interpolates or decides when to sample: that is the loop's job
(:func:`slotsync.simulate.run_timing_loop`).  Keeping the detectors sample-level
pure is what lets :mod:`slotsync.scurve` compute their S-curves exactly.

Detector summary
----------------
=====================  ==========  =====================  ==========================
detector               samples     needs data decisions   notes
=====================  ==========  =====================  ==========================
:func:`early_late`     2 per sym   no (magnitude form)    needs the pulse to have a
                                                          non-zero slope at +-delta
:func:`early_late_dd`  3 per sym   yes                    sign-corrected for
                                                          antipodal data
:func:`gardner`        2 per sym   **no**                 independent of carrier
                                                          phase (see below)
:func:`mueller_muller` 1 per sym   yes                    symbol-rate only
=====================  ==========  =====================  ==========================

Why Gardner's detector is used
------------------------------
The property worth stating explicitly is that Gardner's detector is
**independent of carrier phase**.  For a complex baseband sample stream the
detector output is ``Re{ conj(x_mid) * (x_k - x_{k-1}) }``.  Rotating the whole
sample stream by a constant carrier phase ``phi`` multiplies every sample by
``exp(j phi)``; the product ``conj(x_mid) * (x_k - x_{k-1})`` is then multiplied
by ``conj(exp(j phi)) * exp(j phi) = 1``, so the detector output is unchanged.
Timing recovery can therefore run *before* carrier recovery has converged, and
in a receiver that has not yet acquired carrier phase it is the only one of
these three detectors that works at all.  :func:`gardner_complex` implements
that form, and ``tests/test_detectors.py`` checks the invariance numerically
over a sweep of phases.  The same argument does not apply to
:func:`mueller_muller`, which needs data decisions and therefore needs the
constellation to be de-rotated first.

References
----------
* F. M. Gardner, "A BPSK/QPSK Timing-Error Detector for Sampled Receivers",
  *IEEE Transactions on Communications*, vol. 34, no. 5, pp. 423-429, 1986.
  Bibliographic details verified via the Crossref record for
  ``10.1109/TCOM.1986.1096561`` on 2026-10-06.  No page or equation number from
  inside the paper is quoted anywhere in this package.
* K. H. Mueller and M. Mueller, "Timing Recovery in Digital Synchronous Data
  Receivers", *IEEE Transactions on Communications*, vol. 24, no. 5,
  pp. 516-531, 1976.  Bibliographic details verified via the Crossref record
  for ``10.1109/TCOM.1976.1093326`` on 2026-10-06.
* U. Mengali and A. N. D'Andrea, *Synchronization Techniques for Digital
  Receivers*, Springer (Applications of Communications Theory), 1997,
  ISBN 978-0-306-45725-8.  Named as the standard treatment of the detectors and
  their S-curves; bibliographic details verified against the publisher's record
  on 2026-10-06.  Nothing in this package quotes an equation from it.

The early-late gate itself is elementary and is **not** attributed to a
reference here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DETECTOR_NAMES",
    "detector_by_name",
    "early_late",
    "early_late_dd",
    "gardner",
    "gardner_complex",
    "mueller_muller",
    "slice_antipodal",
]


def _as_float_array(name: str, values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} is empty; a detector needs at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite samples")
    return array


def _check_same_shape(**arrays: np.ndarray) -> None:
    shapes = {name: a.shape for name, a in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"all sample arrays must have the same shape, got {shapes}")


def early_late(early: np.ndarray, late: np.ndarray, *, square: bool = False) -> np.ndarray:
    """Early-late gate: ``e = early - late`` (or the difference of squares).

    Parameters
    ----------
    early
        Sample taken ``delta`` symbol periods **before** the estimated centre.
    late
        Sample taken ``delta`` symbol periods **after** the estimated centre.
    square
        If true, use ``early**2 - late**2``.  This is the non-data-aided form:
        it is insensitive to the sign of antipodal data and is the natural form
        for unipolar OOK and for PPM slot energy.  If false, the raw difference
        is returned, which is only meaningful for unipolar data or when
        multiplied by a decision (see :func:`early_late_dd`).

    Returns
    -------
    numpy.ndarray
        Detector output, dimensionless, positive when sampling late.

    Assumptions and validity
    ------------------------
    * The pulse must have a non-zero slope at ``+-delta`` from its centre.  For a
      rectangular pulse with ``delta`` inside the pulse the slope is zero and
      the gain collapses to zero; the S-curve machinery measures that rather
      than assuming it away.
    * Linear only for ``|timing error| < delta``.  Beyond that the S-curve bends
      and eventually reverses; :attr:`slotsync.scurve.SCurve.reversal_offset`
      locates the reversal.
    * ``square=True`` doubles the noise-times-signal cross term, so the squaring
      form is noisier at low sample SNR.  That is a measured effect in this
      package, not an assumed one.
    """
    e = _as_float_array("early", early)
    late_a = _as_float_array("late", late)
    _check_same_shape(early=e, late=late_a)
    if square:
        return e * e - late_a * late_a
    return e - late_a


def early_late_dd(
    decisions: np.ndarray, early: np.ndarray, late: np.ndarray
) -> np.ndarray:
    """Decision-directed early-late gate: ``e = d * (early - late)``.

    ``decisions`` carries the sliced data symbol (``+-1`` for antipodal, ``0`` or
    ``1`` for OOK).  Multiplying by the decision removes the data sign, so the
    detector works on antipodal data without squaring and therefore without the
    squaring loss.  It fails when the decisions are wrong, which couples the
    timing loop to the data error rate: a documented limitation rather than a
    hidden one.
    """
    d = _as_float_array("decisions", decisions)
    e = _as_float_array("early", early)
    late_a = _as_float_array("late", late)
    _check_same_shape(decisions=d, early=e, late=late_a)
    return d * (e - late_a)


def gardner(
    strobe_previous: np.ndarray, mid: np.ndarray, strobe: np.ndarray
) -> np.ndarray:
    """Gardner timing-error detector for real samples.

    ``e[k] = x_mid[k] * (x[k] - x[k-1])`` where ``x_mid[k]`` is the sample
    halfway between the two symbol strobes.  Needs exactly two samples per
    symbol and **no data decisions**.

    Parameters
    ----------
    strobe_previous
        Sample at the previous symbol centre, ``x[k-1]``.
    mid
        Sample halfway between the two centres, ``x[k-1/2]``.
    strobe
        Sample at the current symbol centre, ``x[k]``.

    Returns
    -------
    numpy.ndarray
        Detector output, positive when sampling late.

    Assumptions and validity
    ------------------------
    * Exactly two samples per symbol.  The detector is defined by the midpoint
      sample; at one sample per symbol it does not exist.
    * Non-data-aided and **independent of carrier phase** in its complex form
      (:func:`gardner_complex`); see the module docstring for the one-line
      proof.  This is the property that makes it the usual choice when timing
      must be recovered before carrier phase.
    * Its output is quadratic in the signal amplitude, so the detector gain
      scales with received power.  A loop designed for one amplitude is
      mis-tuned at another unless the samples are normalised; this package
      measures the gain from the S-curve at a stated amplitude and says so.
    * Zero mean output requires data transitions.  On a long run of identical
      symbols ``x[k] - x[k-1]`` vanishes and the detector delivers nothing:
      timing information in this detector comes from transitions only.
    """
    prev = _as_float_array("strobe_previous", strobe_previous)
    m = _as_float_array("mid", mid)
    cur = _as_float_array("strobe", strobe)
    _check_same_shape(strobe_previous=prev, mid=m, strobe=cur)
    return m * (cur - prev)


def gardner_complex(
    strobe_previous: np.ndarray, mid: np.ndarray, strobe: np.ndarray
) -> np.ndarray:
    """Gardner detector for complex baseband samples: ``Re{conj(mid)*(x_k - x_{k-1})}``.

    Invariant to a constant carrier phase rotation of the whole sample stream.
    Reduces to :func:`gardner` for real samples.
    """
    prev = np.asarray(strobe_previous, dtype=complex)
    m = np.asarray(mid, dtype=complex)
    cur = np.asarray(strobe, dtype=complex)
    if not (prev.shape == m.shape == cur.shape):
        raise ValueError(
            "all sample arrays must have the same shape, got "
            f"{prev.shape}, {m.shape}, {cur.shape}"
        )
    if not (
        np.all(np.isfinite(prev)) and np.all(np.isfinite(m)) and np.all(np.isfinite(cur))
    ):
        raise ValueError("complex sample arrays contain non-finite values")
    return np.real(np.conj(m) * (cur - prev))


def mueller_muller(
    decision_previous: np.ndarray,
    decision: np.ndarray,
    strobe_previous: np.ndarray,
    strobe: np.ndarray,
) -> np.ndarray:
    """Mueller and Mueller timing-error detector at one sample per symbol.

    ``e[k] = a[k] * x[k-1] - a[k-1] * x[k]`` with ``a`` the data decisions.

    **Sign note.** The arrangement printed in most texts is
    ``a[k-1] * x[k] - a[k] * x[k-1]``, which is the negative of the expression
    used here.  The sign has been flipped so that this detector shares the
    package-wide convention that a positive output means late sampling; nothing
    else differs.

    Parameters
    ----------
    decision_previous, decision
        Sliced data symbols ``a[k-1]`` and ``a[k]``.
    strobe_previous, strobe
        Symbol-centre samples ``x[k-1]`` and ``x[k]``.

    Returns
    -------
    numpy.ndarray
        Detector output, positive when sampling late.

    Assumptions and validity
    ------------------------
    * One sample per symbol.  This is the detector's main attraction: no
      oversampling and no midpoint sample.
    * **Decision-directed**: it needs correct data decisions, so it needs the
      carrier to be de-rotated and the eye to be open.  Unlike
      :func:`gardner_complex` it is *not* independent of carrier phase.
    * Its mean output is
      ``sigma_a^2 * (h(T + eps T) - h(eps T - T))`` for a pulse ``h`` and
      zero-mean independent data with variance ``sigma_a^2``: it is driven
      entirely by the pulse's values **one symbol either side of the centre**.
      On a pulse that is exactly zero at ``+-T`` the gain comes only from the
      slope there, and on a pulse that is flat at ``+-T`` the gain is zero.
      This is derived in ``docs/TIMING_MODEL.md`` section 2.3 and checked by a
      hand-computable known-answer test in ``tests/test_scurve.py``.
    * Self-noise: the same ISI terms that give the gain also give a
      data-dependent fluctuation that does not vanish as the noise goes to
      zero.  :func:`slotsync.scurve.scurve` reports the pattern-to-pattern
      standard deviation at zero offset so that self-noise is visible.
    """
    dp = _as_float_array("decision_previous", decision_previous)
    d = _as_float_array("decision", decision)
    sp = _as_float_array("strobe_previous", strobe_previous)
    s = _as_float_array("strobe", strobe)
    _check_same_shape(
        decision_previous=dp, decision=d, strobe_previous=sp, strobe=s
    )
    return d * sp - dp * s


def slice_antipodal(sample: np.ndarray) -> np.ndarray:
    """Hard decision for antipodal data: ``sign(x)``, with ``0 -> +1``."""
    x = _as_float_array("sample", sample)
    return np.where(x >= 0.0, 1.0, -1.0)


#: Detector names accepted by :func:`detector_by_name`, the CLI and the examples.
DETECTOR_NAMES: tuple[str, ...] = ("early-late", "gardner", "mueller-muller")


def detector_by_name(name: str) -> str:
    """Validate a detector name and return it unchanged.

    The detectors have different sample requirements, so they cannot share one
    call signature; the loop and the S-curve dispatch on the name.  This
    function exists so that an unknown name fails in one place with one
    message.
    """
    if name not in DETECTOR_NAMES:
        raise ValueError(
            f"unknown detector {name!r}; choose one of {list(DETECTOR_NAMES)}"
        )
    return name
