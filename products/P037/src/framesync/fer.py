"""Frame-error-rate harness: analytic expressions and Monte Carlo measurement.

Uncoded link
------------
A frame of n bits survives a memoryless binary symmetric channel only if
every bit survives, so

    FER = 1 - (1 - p)^n                                                 (11)

with p the channel bit error probability of
:func:`framesync.channel.bpsk_ber`. Equation (11) is exact for independent
bit errors and is the relation the P010 cross-check runs through.

The frame *loss* rate seen by the data user is not Eq. (11): a frame is lost
when the receiver discards it, which is when the Frame Error Control Field
fails. A 16-bit CRC misses a fraction of error patterns -- asymptotically
2^-16 = 1.526e-5 of them -- so the CRC-detected rate is slightly *below*
Eq. (11) and the difference is undetected errors delivered to the user.
This module measures both and reports the gap.

Coded links
-----------
RS(255,223): Eqs. (7) and (8) of :mod:`framesync.rs`, exact under the
memoryless-symbol assumption, with p evaluated at the channel rate
R = 223/255.

Convolutional (171, 133) rate 1/2: no tractable exact frame-error
expression exists, so this link is Monte Carlo only. The union bound on the
bit error probability of a terminated convolutional code is an upper bound
whose tightness depends on the whole distance spectrum, and this package
does not compute the spectrum, so no bound is quoted.

Statistics
----------
Every measured rate is reported with its binomial standard error

    SE = sqrt( f (1 - f) / N )                                          (12)

where f is the measured fraction over N frames. With f = 0 the standard
error is 0 and uninformative, so a one-sided 95% upper limit
``3 / N`` (the Poisson rule-of-three) is reported instead and labelled.
Monte Carlo point sizes in this package are chosen from the precision
needed, not from a round number: see ``n_frames_for_target``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .channel import bpsk_ber, bsc_flip
from .conv import ConvCode
from .frames import FrameGeometry
from .rs import RS_RATE, ReedSolomonLink, rs_frame_error_rate

__all__ = [
    "uncoded_fer",
    "binomial_stderr",
    "n_frames_for_target",
    "FerPoint",
    "measure_uncoded_fer",
    "measure_rs_fer",
    "measure_conv_fer",
    "ebn0_for_target",
    "coding_gain_db",
]


def uncoded_fer(ebn0_db: float | np.ndarray, frame_bits: int) -> np.ndarray:
    """Equation (11): analytic uncoded frame error rate.

    Parameters
    ----------
    ebn0_db : float or 1-D array
        Eb/N0 in dB (energy per information bit; R = 1 for the uncoded link).
    frame_bits : int
        Frame length n in bits, positive.

    Returns
    -------
    ndarray
        Frame error probability, dimensionless. Evaluated as
        ``-expm1(n * log1p(-p))`` so it keeps relative precision when
        ``n * p`` is far below 1.
    """
    n = int(frame_bits)
    if n <= 0:
        raise ValueError(f"frame_bits must be positive, got {frame_bits}")
    p = bpsk_ber(ebn0_db, 1.0)
    return -np.expm1(n * np.log1p(-p))


def binomial_stderr(n_errors: int, n_trials: int) -> float:
    """Equation (12): binomial standard error of a measured fraction.

    Returns 0.0 when ``n_errors`` is 0 or ``n_trials``; the caller should
    then quote the rule-of-three limit instead (see module docstring).
    """
    n = int(n_trials)
    k = int(n_errors)
    if n <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")
    if not 0 <= k <= n:
        raise ValueError(f"n_errors must lie in [0, {n}], got {n_errors}")
    f = k / n
    return float(np.sqrt(f * (1.0 - f) / n))


def n_frames_for_target(fer_expected: float, rel_stderr: float = 0.10) -> int:
    """Frames needed for a given relative standard error on the measured FER.

    From Eq. (12), SE/f = sqrt((1-f)/(f N)), so
    ``N = (1 - f) / (f * rel_stderr^2)``. Returns at least 1.
    """
    f = float(fer_expected)
    r = float(rel_stderr)
    if not 0.0 < f <= 1.0:
        raise ValueError(f"fer_expected must lie in (0, 1], got {fer_expected!r}")
    if not 0.0 < r < 1.0:
        raise ValueError(f"rel_stderr must lie in (0, 1), got {rel_stderr!r}")
    return max(1, int(np.ceil((1.0 - f) / (f * r * r))))


@dataclass
class FerPoint:
    """One measured frame-error-rate point."""

    ebn0_db: float
    link: str
    n_frames: int
    n_frame_errors: int
    fer: float
    stderr: float
    rule_of_three_upper: float | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def build(cls, ebn0_db: float, link: str, k: int, n: int, **extra: object) -> FerPoint:
        """Construct from an error count, filling in Eq. (12) and the 0-count case."""
        se = binomial_stderr(k, n)
        r3 = None if k > 0 else 3.0 / n
        return cls(float(ebn0_db), link, int(n), int(k), k / n, se, r3, dict(extra))

    def as_row(self) -> str:
        """One fixed-width text line, for the validation output files."""
        if self.n_frame_errors == 0:
            unc = f"<{self.rule_of_three_upper:.3e} (95% one-sided)"
        else:
            unc = f"+-{self.stderr:.3e}"
        return (
            f"{self.ebn0_db:7.2f}  {self.link:<12s}  N={self.n_frames:>8d}  "
            f"errors={self.n_frame_errors:>7d}  FER={self.fer:.6e}  {unc}"
        )


def measure_uncoded_fer(
    ebn0_db: float,
    geometry: FrameGeometry,
    n_frames: int,
    rng: np.random.Generator,
    batch: int = 256,
) -> FerPoint:
    """Monte Carlo uncoded frame error rate, with the FECF detection gap.

    Builds frames with a valid Frame Error Control Field, passes every bit
    through the binary symmetric channel at ``p = bpsk_ber(ebn0_db)``, and
    counts (a) frames with at least one bit in error -- the true frame error
    -- and (b) frames the Frame Error Control Field rejects. The difference
    is the undetected-error count and appears in ``extra``.
    """
    if not geometry.fecf:
        raise ValueError("measure_uncoded_fer needs a geometry with a FECF to detect errors")
    p = float(bpsk_ber(ebn0_db, 1.0)[0])
    true_err = detected = 0
    done = 0
    while done < n_frames:
        b = int(min(batch, n_frames - done))
        frames = geometry.random_frames(b, rng)
        bits = np.unpackbits(frames, axis=1)
        rx_bits = bsc_flip(bits, p, rng)
        true_err += int(np.count_nonzero(np.any(bits != rx_bits, axis=1)))
        rx = np.packbits(rx_bits, axis=1)
        detected += int(np.count_nonzero(~geometry.fecf_ok(rx)))
        done += b
    return FerPoint.build(
        ebn0_db,
        "uncoded",
        true_err,
        n_frames,
        p_bit=p,
        n_fecf_detected=detected,
        n_undetected=true_err - detected,
        frame_bits=geometry.frame_bits,
    )


def measure_rs_fer(
    ebn0_db: float,
    n_frames: int,
    rng: np.random.Generator,
    interleave: int = 5,
) -> FerPoint:
    """Monte Carlo RS(255,223) codeblock error rate using ``reedsolo``.

    The channel runs at ``p = bpsk_ber(ebn0_db, R=223/255)``; a frame counts
    as an error when any of its I codewords fails to decode. Slow by
    construction -- ``reedsolo`` decodes in Python -- so point sizes must be
    chosen with the compute budget in mind.
    """
    link = ReedSolomonLink(interleave=interleave)
    p = float(bpsk_ber(ebn0_db, RS_RATE)[0])
    errors = failed_codewords = 0
    for _ in range(int(n_frames)):
        data = rng.integers(0, 256, link.frame_data_octets, dtype=np.uint8).tobytes()
        block = np.frombuffer(link.encode_frame(data), dtype=np.uint8)
        bits = np.unpackbits(block)
        rx = np.packbits(bsc_flip(bits, p, rng)).tobytes()
        out, nfail = link.decode_frame(rx)
        if out != data:
            errors += 1
        failed_codewords += nfail
    return FerPoint.build(
        ebn0_db,
        "rs(255,223)",
        errors,
        int(n_frames),
        p_bit=p,
        code_rate=RS_RATE,
        interleave=int(interleave),
        n_failed_codewords=failed_codewords,
        analytic_fer=float(np.atleast_1d(rs_frame_error_rate(p, interleave))[0]),
    )


def measure_conv_fer(
    ebn0_db: float,
    n_info_bits: int,
    n_frames: int,
    rng: np.random.Generator,
    *,
    soft: bool = False,
    code: ConvCode | None = None,
    batch: int = 200,
) -> FerPoint:
    """Monte Carlo frame error rate of the CCSDS rate-1/2 K=7 link.

    A frame is ``n_info_bits`` information bits, terminated with K-1 zero
    tail bits. A frame counts as an error when any decoded information bit
    differs from the transmitted one. Decoding runs in batches of ``batch``
    frames at a time through the across-frames vectorised Viterbi.
    """
    c = code or ConvCode()
    from .channel import awgn_bpsk_samples

    p = float(bpsk_ber(ebn0_db, c.rate)[0])
    errors = bit_errors = 0
    done = 0
    while done < n_frames:
        b = int(min(batch, n_frames - done))
        info = rng.integers(0, 2, (b, int(n_info_bits)), dtype=np.uint8)
        enc = c.encode_batch(info)
        if soft:
            rx = awgn_bpsk_samples(enc, float(ebn0_db), c.rate, rng)
        else:
            rx = bsc_flip(enc, p, rng)
        dec = c.decode_batch(rx, soft=soft, n_info_bits=int(n_info_bits))
        diff = dec != info
        errors += int(np.count_nonzero(np.any(diff, axis=1)))
        bit_errors += int(np.count_nonzero(diff))
        done += b
    return FerPoint.build(
        ebn0_db,
        "conv(171,133)" + ("-soft" if soft else "-hard"),
        errors,
        int(n_frames),
        p_bit=p,
        code_rate=c.rate,
        n_info_bits=int(n_info_bits),
        decoded_ber=bit_errors / (n_frames * int(n_info_bits)),
        soft=bool(soft),
    )


def ebn0_for_target(
    curve: str,
    target: float,
    *,
    frame_bits: int = 8936,
    interleave: int = 5,
    bracket: tuple[float, float] = (-5.0, 25.0),
) -> float:
    """Eb/N0 in dB at which an analytic curve reaches ``target``.

    Parameters
    ----------
    curve : {"uncoded_fer", "uncoded_ber", "rs_fer", "rs_ber"}
        Which analytic expression to invert. ``*_ber`` are bit error rates
        (Eq. (1) and Eq. (10)); ``*_fer`` are frame error rates (Eqs. (11)
        and (8)).
    target : float
        The rate to solve for, in (0, 1).
    frame_bits : int
        n for the uncoded frame expression.
    interleave : int
        I for the RS frame expression.
    bracket : (float, float)
        Eb/N0 search bracket in dB. The function raises if the target is not
        bracketed, rather than extrapolating.

    Returns
    -------
    float
        Eb/N0 in dB, from Brent root finding on log10(rate) - log10(target).
    """
    from .rs import rs_output_bit_error_rate

    if not 0.0 < float(target) < 1.0:
        raise ValueError(f"target must lie in (0, 1), got {target!r}")

    def rate(x: float) -> float:
        if curve == "uncoded_fer":
            return float(np.atleast_1d(uncoded_fer(x, frame_bits))[0])
        if curve == "uncoded_ber":
            return float(np.atleast_1d(bpsk_ber(x, 1.0))[0])
        if curve == "rs_fer":
            p = float(np.atleast_1d(bpsk_ber(x, RS_RATE))[0])
            return float(np.atleast_1d(rs_frame_error_rate(p, interleave))[0])
        if curve == "rs_ber":
            p = float(np.atleast_1d(bpsk_ber(x, RS_RATE))[0])
            return float(np.atleast_1d(rs_output_bit_error_rate(p))[0])
        raise ValueError(
            f"curve must be one of 'uncoded_fer', 'uncoded_ber', 'rs_fer', 'rs_ber', "
            f"got {curve!r}"
        )

    def f(x: float) -> float:
        r = rate(x)
        if r <= 0.0:
            return -50.0
        return float(np.log10(r) - np.log10(target))

    lo, hi = bracket
    if f(lo) < 0 or f(hi) > 0:
        raise ValueError(
            f"target {target:g} for curve {curve!r} is not bracketed by Eb/N0 in "
            f"[{lo}, {hi}] dB; rate({lo})={rate(lo):.3e}, rate({hi})={rate(hi):.3e}"
        )
    return float(brentq(f, lo, hi, xtol=1e-6))


def coding_gain_db(
    target: float = 1e-5,
    *,
    metric: str = "ber",
    frame_bits: int = 8936,
    interleave: int = 5,
) -> dict[str, float]:
    """Coding gain of RS(255,223) over the uncoded link at a target rate.

    Gain is the difference in required Eb/N0 **per information bit**, which
    is the convention that makes the comparison fair: the coded link spends
    Es = R*Eb per channel symbol, and that rate loss is already inside
    Eq. (1).

    Parameters
    ----------
    target : float
        Target rate, e.g. 1e-5.
    metric : {"ber", "fer"}
        Compare at equal output bit error rate (Eq. (10) against Eq. (1)) or
        at equal frame error rate (Eq. (8) against Eq. (11)).

    Returns
    -------
    dict
        ``ebn0_uncoded_db``, ``ebn0_rs_db``, ``gain_db``, ``target``, ``metric``.
    """
    if metric not in ("ber", "fer"):
        raise ValueError(f"metric must be 'ber' or 'fer', got {metric!r}")
    unc = ebn0_for_target(
        f"uncoded_{metric}", target, frame_bits=frame_bits, interleave=interleave
    )
    rs = ebn0_for_target(f"rs_{metric}", target, frame_bits=frame_bits, interleave=interleave)
    return {
        "target": float(target),
        "metric": metric,
        "ebn0_uncoded_db": unc,
        "ebn0_rs_db": rs,
        "gain_db": unc - rs,
    }
