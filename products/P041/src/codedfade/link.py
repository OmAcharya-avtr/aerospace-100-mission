"""Coded OOK link over a correlated fading channel: the depth-versus-correlation-time sweep.

Channel and detection model
---------------------------
On-off keying with direct detection, thermal-noise limited. The transmitter
sends optical power ``2*Pbar`` for a one and ``0`` for a zero, so the mean
transmitted power is ``Pbar``; the receiver has responsivity ``R`` and additive
Gaussian noise of standard deviation ``sigma_n`` that does **not** depend on the
transmitted symbol (the thermal-limited case). With the decision threshold at
half the on-level and normalised irradiance ``I`` (``E[I] = 1``), the two
hypotheses are separated by ``2 R Pbar I`` and the threshold sits ``R Pbar I``
from each, so the conditional bit-error probability is

    p_b(I) = Q(R * Pbar * I / sigma_n) = Q(sqrt(gbar) * I)                (26)

with the mean-power electrical signal-to-noise ratio

    gbar = (R * Pbar / sigma_n)**2                                        (27)

Equation (26) is elementary binary detection in Gaussian noise with a fixed
threshold; it is derived here rather than cited so that its convention is
unambiguous. ``gbar`` is quoted in dB as ``10*log10(gbar)``.

**Assumptions, all of which matter.**

* Signal-independent noise. A shot-noise-limited or APD receiver has
  symbol-dependent noise variance and equation (26) is then wrong; this package
  does not model it.
* Fixed threshold at half the *instantaneous* on-level, i.e. perfect knowledge
  of ``I`` at the receiver for threshold placement but no use of it for coding.
  A fixed absolute threshold performs worse and an adaptive-threshold receiver
  better; the gap is not modelled.
* The irradiance is held constant across the ``m`` bits of one code symbol, so
  the channel is sampled once per code symbol and the symbol period is the time
  unit. This is exact when ``tau >> m / Rb`` and is the convention that makes
  ``L_c = tau * Rs`` the single dimensionless parameter of the problem.
* No intersymbol interference, no pointing jitter, no background light.

Error generation
----------------
For each code symbol the number of bit errors is drawn as
``Binomial(m, p_b(I))`` and the symbol is in error if that count is non-zero.
This is exact under the assumptions above: bits within a symbol see the same
``I`` and independent noise.

Decoding shortcut, and why it is sound
--------------------------------------
Reed-Solomon codes are maximum distance separable, so bounded-distance decoding
corrects **every** pattern of at most ``t = (n-k)/2`` symbol errors. The sweep
therefore does not run the Berlekamp-Massey decoder on codewords whose symbol
error count is ``<= t``: it records success with zero residual errors. Codewords
with more than ``t`` symbol errors **are** passed to the real decoder, because
above the radius the outcome (failure or miscorrection, and how many residual
symbol errors) is not predictable. Pass ``exact_decode=True`` to disable the
shortcut entirely. ``validation/validate_decode_shortcut.py`` runs both paths on
the same seeded realisation and reports the difference.

Frame definition
----------------
A **frame** is one Reed-Solomon codeword. ``FER`` is the fraction of codewords
whose decoded message differs from the transmitted message in at least one
symbol, which includes both detected failures and undetected miscorrections.
``BER`` is the post-decoding message **bit** error ratio over all transmitted
message bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special

from .channel import ChannelConfig, generate_irradiance
from .interleave import BlockInterleaver
from .reedsolomon import ReedSolomon, symbols_to_bits


def q_function(x: np.ndarray | float) -> np.ndarray:
    """Gaussian tail ``Q(x) = 0.5*erfc(x/sqrt(2))``. Dimensionless."""
    return 0.5 * special.erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2.0))


def conditional_bit_error_probability(
    irradiance: np.ndarray, mean_snr_db: float
) -> np.ndarray:
    """``p_b(I) = Q(sqrt(gbar) * I)``, equation (26). Dimensionless.

    Parameters
    ----------
    irradiance:
        Normalised irradiance ``I`` with ``E[I] = 1``, >= 0.
    mean_snr_db:
        ``10*log10(gbar)``, dB.
    """
    i = np.asarray(irradiance, dtype=np.float64)
    if np.any(i < 0):
        raise ValueError("irradiance must be >= 0")
    gbar = 10.0 ** (float(mean_snr_db) / 10.0)
    return q_function(np.sqrt(gbar) * i)


@dataclass(frozen=True)
class LinkResult:
    """Outcome of one coded-link simulation at one interleaver depth.

    Attributes
    ----------
    depth:
        Block-interleaver depth ``D``, symbols.
    codewords:
        Number of codewords transmitted.
    raw_symbol_error_rate:
        Pre-decoding channel symbol error rate, dimensionless.
    raw_bit_error_rate:
        Pre-decoding channel bit error rate, dimensionless.
    post_bit_error_rate:
        Post-decoding message bit error rate, dimensionless.
    frame_error_rate:
        Fraction of codewords decoded to the wrong message, dimensionless.
    decoder_failures:
        Codewords the decoder declared uncorrectable.
    miscorrections:
        Codewords the decoder declared corrected but got wrong.
    mean_symbol_errors_per_codeword:
        Mean pre-decoding symbol errors per codeword, dimensionless.
    max_symbol_errors_per_codeword:
        Worst pre-decoding symbol error count in any codeword.
    latency_ms:
        End-to-end interleaving latency at the configured symbol rate, ms.
    memory_bytes:
        Total interleaver storage at both ends, bytes.
    """

    depth: int
    codewords: int
    raw_symbol_error_rate: float
    raw_bit_error_rate: float
    post_bit_error_rate: float
    frame_error_rate: float
    decoder_failures: int
    miscorrections: int
    mean_symbol_errors_per_codeword: float
    max_symbol_errors_per_codeword: int
    latency_ms: float
    memory_bytes: float

    @property
    def frame_error_rate_standard_error(self) -> float:
        """Binomial standard error of ``frame_error_rate``, dimensionless."""
        p = self.frame_error_rate
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / self.codewords))


@dataclass
class CodedLink:
    """Coded OOK link over a correlated fading channel.

    Parameters
    ----------
    code:
        The Reed-Solomon code.
    channel:
        Channel configuration; ``channel.sample_rate_hz`` is the **code symbol**
        rate ``Rs``, one channel sample per code symbol.
    mean_snr_db:
        ``10*log10(gbar)`` from equation (27), dB.
    exact_decode:
        If ``True``, every codeword goes through the Berlekamp-Massey decoder,
        including those within the correction radius.
    """

    code: ReedSolomon
    channel: ChannelConfig
    mean_snr_db: float
    exact_decode: bool = False
    _message: np.ndarray = field(default=None, repr=False, init=False)  # type: ignore[assignment]

    @property
    def symbol_rate_hz(self) -> float:
        """Code symbol rate ``Rs``, Hz."""
        return self.channel.sample_rate_hz

    @property
    def samples_per_correlation_time(self) -> float:
        """``L_c = tau * Rs``, the dimensionless fade length in code symbols."""
        return self.channel.samples_per_correlation_time

    def run(self, depth: int, codewords: int, data_seed: int = 1234) -> LinkResult:
        """Simulate ``codewords`` codewords at block-interleaver depth ``depth``.

        The channel realisation is fixed by ``self.channel.seed`` and is the
        **same series** for every depth, so differences between depths are
        attributable to the interleaver and not to the channel draw. The data and
        the noise draw are fixed by ``data_seed``.
        """
        depth = int(depth)
        codewords = int(codewords)
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth!r}")
        if codewords < 1:
            raise ValueError(f"codewords must be >= 1, got {codewords!r}")
        code = self.code
        n, k, m = code.n, code.k, code.m
        blocks = int(np.ceil(codewords / depth))
        total_codewords = blocks * depth
        n_symbols = total_codewords * n

        irradiance = generate_irradiance(self.channel, n_symbols)
        p_bit = conditional_bit_error_probability(irradiance, self.mean_snr_db)
        rng = np.random.default_rng(data_seed)
        bit_errors = rng.binomial(m, p_bit)
        # Channel symbol errors: a symbol is wrong if any of its m bits is wrong.
        # The error value is uniform over the non-zero field elements consistent
        # with that bit-error count; drawing it uniformly over 1..2^m-1 is a
        # stand-in that preserves "wrong symbol" and is what the decoder sees.
        error_mask = bit_errors > 0
        error_values = np.where(
            error_mask, rng.integers(1, 1 << m, size=n_symbols), 0
        ).astype(np.int64)

        interleaver = BlockInterleaver(depth, n)
        # De-interleaving the channel error stream maps channel-time positions to
        # codeword positions. The data itself is irrelevant to the error pattern,
        # so only the error stream is permuted.
        err_cw = interleaver.deinterleave(error_values).reshape(total_codewords, n)

        message = rng.integers(0, 1 << m, size=(total_codewords, k), dtype=np.int64)

        post_bit_errors = 0
        frame_errors = 0
        failures = 0
        miscorrections = 0
        sym_err_counts = np.count_nonzero(err_cw, axis=1)

        for idx in range(total_codewords):
            e = int(sym_err_counts[idx])
            if e == 0:
                continue
            if e <= code.t and not self.exact_decode:
                continue
            msg = message[idx]
            received = code.encode(msg) ^ err_cw[idx]
            result = code.decode(received)
            if np.array_equal(result.message, msg):
                continue
            frame_errors += 1
            if result.success:
                miscorrections += 1
            else:
                failures += 1
            diff = symbols_to_bits(result.message ^ msg, m)
            post_bit_errors += int(diff.sum())

        raw_sym = float(np.count_nonzero(error_values) / n_symbols)
        raw_bit = float(bit_errors.sum() / (n_symbols * m))
        cost = interleaver.cost(self.symbol_rate_hz, bits_per_symbol=m)
        return LinkResult(
            depth=depth,
            codewords=total_codewords,
            raw_symbol_error_rate=raw_sym,
            raw_bit_error_rate=raw_bit,
            post_bit_error_rate=post_bit_errors / (total_codewords * k * m),
            frame_error_rate=frame_errors / total_codewords,
            decoder_failures=failures,
            miscorrections=miscorrections,
            mean_symbol_errors_per_codeword=float(sym_err_counts.mean()),
            max_symbol_errors_per_codeword=int(sym_err_counts.max()),
            latency_ms=cost.latency_ms,
            memory_bytes=cost.memory_bytes,
        )

    def depth_sweep(
        self, depths: list[int], codewords: int, data_seed: int = 1234
    ) -> list[LinkResult]:
        """Run :meth:`run` at every depth in ``depths``, on an identical channel record.

        ``run`` rounds the codeword count up to a whole number of interleaver
        blocks, which would otherwise give each depth a slightly different length
        of channel series and so a slightly different raw symbol error rate. This
        method raises the requested count to a common multiple of every depth
        when that multiple is at most eight times the request, so that every
        point in the sweep sees the **same** channel prefix and the only thing
        that varies is the permutation. When the least common multiple is larger
        than that, each depth is rounded individually and the per-point
        ``codewords`` field records what was actually run.
        """
        if not depths:
            raise ValueError("depths must be non-empty")
        if any(int(d) < 1 for d in depths):
            raise ValueError(f"all depths must be >= 1, got {depths!r}")
        lcm = int(np.lcm.reduce(np.asarray([int(d) for d in depths], dtype=np.int64)))
        target = codewords
        if lcm <= 8 * codewords:
            target = int(np.ceil(codewords / lcm) * lcm)
        return [self.run(d, target, data_seed) for d in depths]


def uncoded_bit_error_rate(
    channel: ChannelConfig, mean_snr_db: float, n_symbols: int
) -> float:
    """Average of equation (26) over a realisation: the uncoded reference BER.

    This is the number every coded result must be compared against, and it is
    computed from the same channel configuration so the comparison is like for
    like.
    """
    irradiance = generate_irradiance(channel, int(n_symbols))
    return float(conditional_bit_error_probability(irradiance, mean_snr_db).mean())
