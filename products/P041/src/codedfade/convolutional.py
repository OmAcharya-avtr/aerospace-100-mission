"""Rate-1/n binary convolutional encoder and hard-decision Viterbi decoder, in numpy.

Encoder
-------
A rate ``1/n_out`` feedforward convolutional code of constraint length ``K`` is
defined by ``n_out`` generator polynomials, each ``K`` bits wide and given in
octal in the usual convention: the most significant bit multiplies the current
input and the remaining bits multiply the ``K-1`` stored past inputs. With state
``s = (b[i-1], b[i-2], ..., b[i-K+1])`` (most recent first) the ``j``-th output
bit is

    c_j[i] = parity( g_j & (b[i] << (K-1) | state) )                       (21)

so the encoder is a Mealy machine over ``2**(K-1)`` states. Encoding appends
``K-1`` zero tail bits, which drives the encoder back to the all-zero state and
makes the terminated code a block code of length ``n_out*(L + K - 1)`` bits for
``L`` message bits. Rate including the tail is ``L / (n_out*(L+K-1))``.

The default ``(0o7, 0o5)``, ``K = 3`` code is the standard rate-1/2 example; its
free distance is 5. **This module does not assert a free distance**: the test
suite measures the minimum weight of all non-zero terminated codewords up to a
stated length and reports what it finds.

Decoder
-------
Hard-decision Viterbi over the terminated trellis with Hamming branch metrics,
initial and final state constrained to zero. The survivor history is stored as a
``(n_steps, n_states)`` array of predecessor states and traced back from state
zero. Ties in the metric are broken towards the lower state index, which is
deterministic but arbitrary; it is documented because it makes the decoder's
output reproducible rather than correct.

Hand-traced known answer (also in ``tests/test_convolutional.py``): for
``K = 3``, ``g = (0o7, 0o5)`` and message ``1 0 1 1``:

===========  =====================  =============  ======
input bit    state (b[i-1],b[i-2])  outputs        new state
===========  =====================  =============  ======
1            0,0                    1^0^0=1, 1^0=1   1,0
0            1,0                    0^1^0=1, 0^0=0   0,1
1            0,1                    1^0^1=0, 1^1=0   1,0
1            1,0                    1^1^0=0, 1^0=1   1,1
0 (tail)     1,1                    0^1^1=0, 0^1=1   0,1
0 (tail)     0,1                    0^0^1=1, 0^1=1   0,0
===========  =====================  =============  ======

giving ``11 10 00 01 01 11``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViterbiResult:
    """Outcome of one Viterbi decode.

    Attributes
    ----------
    message:
        Decoded message bits, uint8 0/1, length ``L``.
    path_metric:
        Hamming distance between the received word and the chosen codeword,
        in bits.
    """

    message: np.ndarray
    path_metric: int


class ConvolutionalCode:
    """Terminated rate-1/n feedforward convolutional code with Viterbi decoding.

    Parameters
    ----------
    generators:
        Generator polynomials as integers (octal literals are conventional),
        each ``< 2**constraint_length``.
    constraint_length:
        ``K``, number of input bits each output depends on, ``>= 2``.
    """

    def __init__(
        self, generators: tuple[int, ...] = (0o7, 0o5), constraint_length: int = 3
    ) -> None:
        k = int(constraint_length)
        if k < 2:
            raise ValueError(f"constraint_length must be >= 2, got {k!r}")
        if len(generators) < 1:
            raise ValueError("at least one generator polynomial is required")
        gens = tuple(int(g) for g in generators)
        for g in gens:
            if not 0 < g < (1 << k):
                raise ValueError(
                    f"each generator must satisfy 0 < g < {1 << k} for K={k}, got {g!r}"
                )
        self.constraint_length = k
        self.generators = gens
        self.n_out = len(gens)
        self.n_states = 1 << (k - 1)
        self._build_trellis()

    @property
    def rate(self) -> float:
        """Nominal rate ``1/n_out``, ignoring the tail. Dimensionless."""
        return 1.0 / self.n_out

    def terminated_rate(self, message_bits: int) -> float:
        """Rate including the ``K-1`` tail bits, dimensionless."""
        if message_bits <= 0:
            raise ValueError(f"message_bits must be > 0, got {message_bits!r}")
        return message_bits / (self.n_out * (message_bits + self.constraint_length - 1))

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        gens = ", ".join(f"0o{g:o}" for g in self.generators)
        return f"ConvolutionalCode(({gens}), K={self.constraint_length})"

    def _build_trellis(self) -> None:
        k = self.constraint_length
        states = np.arange(self.n_states, dtype=np.int64)
        self.next_state = np.empty((self.n_states, 2), dtype=np.int64)
        self.output = np.empty((self.n_states, 2, self.n_out), dtype=np.uint8)
        for s in states.tolist():
            for bit in (0, 1):
                word = (bit << (k - 1)) | s
                for j, g in enumerate(self.generators):
                    self.output[s, bit, j] = bin(word & g).count("1") & 1
                self.next_state[s, bit] = ((s >> 1) | (bit << (k - 2))) & (
                    self.n_states - 1
                )

    def encode(self, message: np.ndarray) -> np.ndarray:
        """Encode message bits with ``K-1`` zero tail bits appended.

        Returns a uint8 0/1 array of length ``n_out * (L + K - 1)``.
        """
        msg = np.asarray(message, dtype=np.uint8).reshape(-1)
        if msg.size == 0:
            raise ValueError("message must be non-empty")
        if not np.all((msg == 0) | (msg == 1)):
            raise ValueError("message must contain only 0 and 1")
        tail = np.zeros(self.constraint_length - 1, dtype=np.uint8)
        stream = np.concatenate([msg, tail])
        out = np.empty((stream.size, self.n_out), dtype=np.uint8)
        state = 0
        for i, bit in enumerate(stream.tolist()):
            out[i] = self.output[state, bit]
            state = int(self.next_state[state, bit])
        if state != 0:
            raise RuntimeError("terminated encoder did not return to the zero state")
        return out.reshape(-1)

    def decode(self, received: np.ndarray, message_bits: int) -> ViterbiResult:
        """Hard-decision Viterbi decode.

        Parameters
        ----------
        received:
            uint8 0/1 array of length ``n_out * (message_bits + K - 1)``.
        message_bits:
            ``L``, the number of message bits that were encoded.
        """
        r = np.asarray(received, dtype=np.uint8).reshape(-1)
        if message_bits <= 0:
            raise ValueError(f"message_bits must be > 0, got {message_bits!r}")
        steps = message_bits + self.constraint_length - 1
        if r.size != self.n_out * steps:
            raise ValueError(
                f"received must have {self.n_out * steps} bits for message_bits="
                f"{message_bits}, got {r.size}"
            )
        if not np.all((r == 0) | (r == 1)):
            raise ValueError("received must contain only 0 and 1")

        big = np.int64(1 << 40)
        metric = np.full(self.n_states, big, dtype=np.int64)
        metric[0] = 0
        prev_state = np.zeros((steps, self.n_states), dtype=np.int64)
        prev_bit = np.zeros((steps, self.n_states), dtype=np.uint8)
        obs = r.reshape(steps, self.n_out).astype(np.int64)
        out_i = self.output.astype(np.int64)

        for i in range(steps):
            # branch metric for every (state, bit): Hamming distance
            branch = np.abs(out_i - obs[i][None, None, :]).sum(axis=2)
            cand = metric[:, None] + branch  # (n_states, 2)
            nxt = np.full(self.n_states, big, dtype=np.int64)
            pstate = np.zeros(self.n_states, dtype=np.int64)
            pbit = np.zeros(self.n_states, dtype=np.uint8)
            # iterate states ascending so ties resolve to the lower state index
            for s in range(self.n_states):
                for bit in (0, 1):
                    ns = int(self.next_state[s, bit])
                    c = int(cand[s, bit])
                    if c < nxt[ns]:
                        nxt[ns] = c
                        pstate[ns] = s
                        pbit[ns] = bit
            metric = nxt
            prev_state[i] = pstate
            prev_bit[i] = pbit

        bits = np.zeros(steps, dtype=np.uint8)
        state = 0
        for i in range(steps - 1, -1, -1):
            bits[i] = prev_bit[i, state]
            state = int(prev_state[i, state])
        return ViterbiResult(bits[:message_bits].copy(), int(metric[0]))

    def minimum_terminated_weight(self, message_bits: int) -> int:
        """Minimum Hamming weight over all non-zero terminated codewords.

        Exhaustive over ``2**message_bits - 1`` messages; intended for small
        ``message_bits`` only. Returned in bits.
        """
        if not 1 <= message_bits <= 16:
            raise ValueError(f"message_bits must be in [1, 16], got {message_bits!r}")
        best = None
        for value in range(1, 1 << message_bits):
            msg = np.array(
                [(value >> b) & 1 for b in range(message_bits - 1, -1, -1)],
                dtype=np.uint8,
            )
            w = int(self.encode(msg).sum())
            best = w if best is None else min(best, w)
        assert best is not None
        return best
