"""CCSDS rate-1/2 convolutional code: encoder, Viterbi decoder, free distance.

The code
--------
CCSDS 131.0-B specifies a basic convolutional code of rate 1/2 and
constraint length K = 7, with generator polynomials, in octal,

    G1 = 0o171 = 1111001b          G2 = 0o133 = 1011011b                (5)

and the symbol produced by G2 inverted on the channel. The inversion is a
transmission convention that improves symbol-transition density; it is a
fixed XOR on alternate output positions, so it changes no distance property
of the code. It is applied by default here and can be switched off with
``invert_g2=False``.

Encoder: the register holds the current input bit in the most significant
position followed by the six previous bits; each output bit is the parity of
the register ANDed with the corresponding generator. Frames are terminated
by flushing K-1 = 6 zero bits, so every codeword starts and ends in the
all-zero state.

Free distance
-------------
``free_distance()`` computes the free distance directly from the trellis as
the minimum Hamming weight of a non-zero codeword that leaves and returns to
the all-zero state (Dijkstra over the state graph with output weight as edge
cost). It is computed, not quoted. The literature value for the (171, 133)
rate-1/2 K=7 code is d_free = 10; ``validation/validate_rs_coding_gain.py``
prints the computed value next to it.

Decoder
-------
Hard-decision or soft-decision Viterbi, terminated, with full path memory
and traceback from the known all-zero final state. The implementation
vectorises **across frames**: the add-compare-select recursion runs once per
trellis step on arrays of shape (n_frames, 64), which is what makes a
frame-error-rate sweep affordable in pure NumPy on one core.

Soft-decision branch metrics are squared Euclidean distances against the
+-1 BPSK mapping of :func:`framesync.channel.awgn_bpsk_samples`; hard
decision uses Hamming distance. Soft decision is worth about 2 dB over hard
decision for this code in the textbook treatment, and this package measures
the difference rather than asserting it.

Why this is implemented here rather than taken from ``commpy``
--------------------------------------------------------------
``scikit-commpy`` (imports as ``commpy``) ships a general ``Trellis`` and
``viterbi_decode`` and is the mature choice; it is named in the README
alternatives table. It cannot be installed in this build container -- no
wheel for Python 3.13 and the source build fails -- so the decoder below
stands in. A reader with a working toolchain should prefer ``commpy``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CCSDS_G1",
    "CCSDS_G2",
    "ConvCode",
    "free_distance",
]

CCSDS_G1 = 0o171
"""First generator of Eq. (5), octal 171."""

CCSDS_G2 = 0o133
"""Second generator of Eq. (5), octal 133; its output symbol is inverted."""


def _parity(x: np.ndarray | int) -> np.ndarray:
    """Parity (population count mod 2) of each element."""
    v = np.asarray(x, dtype=np.int64)
    out = np.zeros_like(v)
    for shift in range(16):
        out ^= (v >> shift) & 1
    return out


@dataclass(frozen=True)
class ConvCode:
    """Rate-1/2 feedforward convolutional code with terminated frames.

    Attributes
    ----------
    k : int
        Constraint length K (register length including the current bit), >= 2.
    g1, g2 : int
        Generator polynomials as K-bit integers, most significant bit the
        current input tap.
    invert_g2 : bool
        Invert the G2 output symbol, as CCSDS 131.0-B specifies.
    """

    k: int = 7
    g1: int = CCSDS_G1
    g2: int = CCSDS_G2
    invert_g2: bool = True

    def __post_init__(self) -> None:
        if self.k < 2 or self.k > 12:
            raise ValueError(f"constraint length K must lie in [2, 12], got {self.k}")
        for name, g in (("g1", self.g1), ("g2", self.g2)):
            if g <= 0 or g >> self.k:
                raise ValueError(f"{name}=0o{g:o} does not fit in K={self.k} bits")

    @property
    def n_states(self) -> int:
        """Number of trellis states, 2**(K-1)."""
        return 1 << (self.k - 1)

    @property
    def rate(self) -> float:
        """Code rate R = 1/2 (dimensionless), ignoring the termination tail."""
        return 0.5

    def tail_bits(self) -> int:
        """Zero bits appended to flush the register, K-1."""
        return self.k - 1

    # -- trellis -----------------------------------------------------------
    def _trellis(self) -> tuple[np.ndarray, np.ndarray]:
        """(next_state, out_symbol) arrays of shape (n_states, 2).

        ``out_symbol[s, u]`` packs the two output bits as 2*b1 + b2.
        """
        s = np.arange(self.n_states, dtype=np.int64)[:, None]
        u = np.arange(2, dtype=np.int64)[None, :]
        reg = (u << (self.k - 1)) | s
        b1 = _parity(reg & self.g1)
        b2 = _parity(reg & self.g2)
        if self.invert_g2:
            b2 = b2 ^ 1
        nxt = (reg >> 1) & (self.n_states - 1)
        return nxt, (2 * b1 + b2)

    def _predecessors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(prev_state, prev_input, prev_out) of shape (n_states, 2).

        For a feedforward rate-1/k code every state has exactly two
        predecessors; ``prev_out[ns, j]`` is the symbol emitted on that edge.
        """
        nxt, out = self._trellis()
        prev_state = np.zeros((self.n_states, 2), dtype=np.int64)
        prev_input = np.zeros((self.n_states, 2), dtype=np.int64)
        prev_out = np.zeros((self.n_states, 2), dtype=np.int64)
        fill = np.zeros(self.n_states, dtype=np.int64)
        for s in range(self.n_states):
            for u in range(2):
                ns = int(nxt[s, u])
                j = int(fill[ns])
                if j > 1:  # pragma: no cover - structural guarantee
                    raise AssertionError("feedforward trellis must have 2 predecessors")
                prev_state[ns, j] = s
                prev_input[ns, j] = u
                prev_out[ns, j] = int(out[s, u])
                fill[ns] += 1
        return prev_state, prev_input, prev_out

    # -- encoding ----------------------------------------------------------
    def encode(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """Encode one frame of information bits into channel bits.

        Parameters
        ----------
        bits : ndarray of uint8
            1-D array of 0/1 information bits.
        terminate : bool
            Append K-1 zero bits so the codeword ends in state 0.

        Returns
        -------
        ndarray of uint8
            ``2 * (len(bits) + K - 1)`` channel bits if terminated,
            ``2 * len(bits)`` otherwise, in transmission order b1, b2, b1, ...
        """
        arr = np.asarray(bits, dtype=np.uint8).ravel()
        if arr.size and arr.max() > 1:
            raise ValueError("encode expects an array of 0/1 bits")
        if terminate:
            arr = np.concatenate([arr, np.zeros(self.tail_bits(), dtype=np.uint8)])
        nxt, out = self._trellis()
        state = 0
        symbols = np.empty(arr.size, dtype=np.int64)
        for i, u in enumerate(arr.tolist()):
            symbols[i] = out[state, u]
            state = int(nxt[state, u])
        b1 = (symbols >> 1) & 1
        b2 = symbols & 1
        return np.stack([b1, b2], axis=1).ravel().astype(np.uint8)

    def encode_batch(self, bits: np.ndarray, terminate: bool = True) -> np.ndarray:
        """Encode a (n_frames, n_bits) array, one frame per row."""
        arr = np.atleast_2d(np.asarray(bits, dtype=np.uint8))
        return np.stack([self.encode(row, terminate) for row in arr])

    # -- decoding ----------------------------------------------------------
    def decode_batch(
        self,
        received: np.ndarray,
        *,
        soft: bool = False,
        n_info_bits: int | None = None,
    ) -> np.ndarray:
        """Viterbi-decode a batch of terminated codewords.

        Parameters
        ----------
        received : ndarray, shape (n_frames, 2*T)
            Hard decisions as 0/1 uint8 when ``soft=False``, or matched-filter
            samples (bit 0 -> +1) when ``soft=True``.
        soft : bool
            Squared-Euclidean branch metrics instead of Hamming.
        n_info_bits : int, optional
            Information bits to return per frame. Defaults to ``T - (K-1)``,
            i.e. the terminated case with the tail stripped.

        Returns
        -------
        ndarray of uint8, shape (n_frames, n_info_bits)
            Maximum-likelihood information bits.
        """
        rec = np.atleast_2d(np.asarray(received))
        if rec.ndim != 2 or rec.shape[1] % 2:
            raise ValueError("received must be (n_frames, 2*T) with an even number of columns")
        b, t = rec.shape[0], rec.shape[1] // 2
        if n_info_bits is None:
            n_info_bits = t - self.tail_bits()
        if not 0 < n_info_bits <= t:
            raise ValueError(f"n_info_bits must lie in (0, {t}], got {n_info_bits}")
        ns_count = self.n_states
        prev_state, prev_input, prev_out = self._predecessors()

        pairs = rec.reshape(b, t, 2)
        # Branch metric of each of the 4 output symbols, per frame and step.
        sym_bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int64)
        if soft:
            mapped = 1.0 - 2.0 * sym_bits.astype(float)  # (4, 2)
            # (B, T, 4): squared Euclidean distance to each candidate symbol
            metric = ((pairs[:, :, None, :].astype(float) - mapped[None, None, :, :]) ** 2).sum(-1)
        else:
            hb = np.asarray(pairs, dtype=np.int64)
            if hb.size and (hb.max() > 1 or hb.min() < 0):
                raise ValueError("hard-decision input must contain only 0/1 values")
            metric = (hb[:, :, None, :] ^ sym_bits[None, None, :, :]).sum(-1).astype(float)

        big = 1e9
        state_metric = np.full((b, ns_count), big, dtype=np.float64)
        state_metric[:, 0] = 0.0
        survivors = np.empty((t, b, ns_count), dtype=np.uint8)
        for step in range(t):
            m = metric[:, step, :]  # (B, 4)
            cand = np.stack(
                [
                    state_metric[:, prev_state[:, j]] + m[:, prev_out[:, j]]
                    for j in (0, 1)
                ],
                axis=2,
            )  # (B, n_states, 2)
            survivors[step] = np.argmin(cand, axis=2).astype(np.uint8)
            state_metric = np.min(cand, axis=2)

        bits = np.zeros((b, t), dtype=np.uint8)
        state = np.zeros(b, dtype=np.int64)
        rows = np.arange(b)
        for step in range(t - 1, -1, -1):
            j = survivors[step][rows, state].astype(np.int64)
            bits[:, step] = prev_input[state, j].astype(np.uint8)
            state = prev_state[state, j]
        return bits[:, :n_info_bits]

    def decode(self, received: np.ndarray, *, soft: bool = False) -> np.ndarray:
        """Viterbi-decode one terminated codeword. See :meth:`decode_batch`."""
        return self.decode_batch(np.atleast_2d(received), soft=soft)[0]


def free_distance(code: ConvCode | None = None, max_weight: int = 40) -> int:
    """Free distance of the code, computed from the trellis.

    Minimum Hamming weight over non-zero codewords that leave the all-zero
    state and return to it. Dijkstra from each first-step successor of state
    0 under input 1, with edge cost the Hamming weight of the output symbol
    *difference* from the all-zero-input path (so the G2 inversion, a fixed
    XOR, cancels and does not affect the result).

    Returns the weight in bits; raises ValueError if no returning path of
    weight <= ``max_weight`` exists.
    """
    c = code or ConvCode()
    linear = ConvCode(k=c.k, g1=c.g1, g2=c.g2, invert_g2=False)
    nxt, out = linear._trellis()
    wt = np.array([bin(int(o)).count("1") for o in out.ravel()]).reshape(out.shape)
    best = {}
    heap: list[tuple[int, int]] = []
    s0, w0 = int(nxt[0, 1]), int(wt[0, 1])
    heapq.heappush(heap, (w0, s0))
    best[s0] = w0
    while heap:
        w, s = heapq.heappop(heap)
        if w > max_weight:
            break
        if w > best.get(s, 1 << 30):
            continue
        for u in range(2):
            ns, nw = int(nxt[s, u]), w + int(wt[s, u])
            if ns == 0:
                return nw
            if nw < best.get(ns, 1 << 30):
                best[ns] = nw
                heapq.heappush(heap, (nw, ns))
    raise ValueError(f"no returning path of weight <= {max_weight} found")
