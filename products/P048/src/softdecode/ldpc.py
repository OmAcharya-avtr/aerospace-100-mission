"""A small regular LDPC code and a sum-product decoder, in numpy.

Why a belief-propagation decoder is needed here
-----------------------------------------------
Maximum-likelihood decoding of a block code depends on the LLRs only through
their sign pattern and their *relative* magnitudes: multiplying every LLR by a
positive constant leaves the decision unchanged. A scale error -- which is
exactly what a mismatched channel-state estimate produces -- is therefore
invisible to such a decoder, and the cost of LLR miscalibration cannot be
measured with one. Sum-product belief propagation has a ``tanh`` message
update, so absolute LLR magnitude changes its output. It is the decoder in
which over-confident LLRs actually do damage, and it is the one used for every
mismatch and clipping number in this repository.

Construction
------------
``make_regular_ldpc(n, dv, dc, seed)`` builds a Gallager-style
``(dv, dc)``-regular parity-check matrix: the first ``n / dc`` rows cover
disjoint consecutive blocks of ``dc`` columns, and the remaining ``dv - 1``
row groups are seeded column permutations of that block. Candidate
permutations are resampled up to ``attempts`` times to reduce the number of
length-4 cycles; the number that remains is reported rather than claimed to be
zero, because this construction does not guarantee girth 6.

A systematic generator is obtained by Gauss-Jordan elimination over GF(2) with
column pivoting. If ``H`` is rank deficient the dependent rows are dropped,
which raises the rate; the returned :class:`LdpcCode` reports the realised
rank, dimension and rate, so no rate is assumed.

Decoder
-------
Flooding-schedule sum-product on LLRs ``L = log P(0)/P(1)``:

    check  -> variable:  m_cv = 2 atanh( prod_{v' != v} tanh(m_v'c / 2) )
    variable -> check:   m_vc = L_v + sum_{c' != c} m_c'v

The leave-one-out products are computed exactly by a forward and a backward
cumulative product over the ``dc`` edges of each check, so no division by a
near-zero ``tanh`` occurs. Messages are vectorised over blocks: with ``B``
blocks and ``E`` edges the working arrays are ``(B, E)``.

On a cycle-free parity-check matrix sum-product is exact after enough
iterations; ``tests/test_ldpc.py`` checks this against brute-force
bit-posterior computation on a small cycle-free code, which is the
known-answer test for the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LdpcCode", "make_regular_ldpc", "sum_product_decode"]


def _gf2_systematic(h: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Row-reduce ``h`` over GF(2) with column pivoting.

    Returns ``(reduced, column_order, rank)`` where ``reduced[:, column_order]``
    begins with an identity of size ``rank``.
    """
    a = h.copy().astype(np.int8)
    m, n = a.shape
    order = np.arange(n)
    rank = 0
    for col in range(n):
        if rank >= m:
            break
        pivot_rows = np.nonzero(a[rank:, col])[0]
        if pivot_rows.size == 0:
            continue
        p = rank + int(pivot_rows[0])
        if p != rank:
            a[[rank, p]] = a[[p, rank]]
        hit = np.nonzero(a[:, col])[0]
        hit = hit[hit != rank]
        if hit.size:
            a[hit] ^= a[rank]
        if col != rank:
            a[:, [rank, col]] = a[:, [col, rank]]
            order[[rank, col]] = order[[col, rank]]
        rank += 1
    return a[:rank], order, rank


@dataclass(frozen=True)
class LdpcCode:
    """A regular LDPC code with its systematic generator and edge layout.

    Attributes
    ----------
    parity_check:
        ``(m, n)`` GF(2) matrix with exactly ``dc`` ones per row and ``dv``
        per column.
    generator:
        ``(k, n)`` systematic generator, ``k = n - rank(H)``, in the column
        order given by ``column_order``.
    column_order:
        Permutation applied to the columns to reach systematic form.
    check_vars:
        ``(m, dc)`` variable index of each check edge, check-major order.
    var_edges:
        ``(n, dv)`` edge index of each variable's edges.
    cycles4:
        Number of length-4 cycles in the Tanner graph (counted as pairs of
        checks sharing two variables).
    """

    parity_check: np.ndarray
    generator: np.ndarray
    column_order: np.ndarray
    check_vars: np.ndarray
    var_edges: np.ndarray
    cycles4: int

    @property
    def length(self) -> int:
        """Block length ``n`` in channel bits."""
        return int(self.parity_check.shape[1])

    @property
    def n_checks(self) -> int:
        """Number of parity checks ``m``."""
        return int(self.parity_check.shape[0])

    @property
    def dimension(self) -> int:
        """Number of information bits ``k``."""
        return int(self.generator.shape[0])

    @property
    def rate(self) -> float:
        """``k / n``, dimensionless."""
        return self.dimension / self.length

    def encode(self, messages) -> np.ndarray:
        """Encode ``(B, k)`` message bits into ``(B, n)`` codewords."""
        m = np.asarray(messages, dtype=np.int8)
        if m.shape[-1] != self.dimension:
            raise ValueError(
                f"messages must have last dimension {self.dimension}, got {m.shape[-1]}"
            )
        words = (m @ self.generator) % 2
        out = np.empty_like(words)
        out[..., self.column_order] = words
        return out.astype(np.int8)

    def message_positions(self) -> np.ndarray:
        """Column indices of the information bits in a codeword."""
        return self.column_order[self.length - self.dimension :]

    def syndrome(self, words) -> np.ndarray:
        """``H c^T mod 2`` of ``(B, n)`` words."""
        c = np.asarray(words, dtype=np.int8)
        return (c @ self.parity_check.T) % 2


def make_regular_ldpc(
    n: int = 96, dv: int = 3, dc: int = 6, seed: int = 20261006, attempts: int = 40
) -> LdpcCode:
    """Build a ``(dv, dc)``-regular LDPC code of length ``n``.

    Requires ``n % dc == 0`` and ``n * dv % dc == 0``. ``attempts`` seeded
    candidates are generated and the one with the fewest length-4 cycles is
    kept.
    """
    n, dv, dc = int(n), int(dv), int(dc)
    if n % dc != 0:
        raise ValueError(f"n must be a multiple of dc, got n={n}, dc={dc}")
    if dv < 2 or dc < 2:
        raise ValueError(f"dv and dc must be at least 2, got dv={dv}, dc={dc}")
    rows_per_group = n // dc
    base = np.zeros((rows_per_group, n), dtype=np.int8)
    for r in range(rows_per_group):
        base[r, r * dc : (r + 1) * dc] = 1

    best: tuple[int, np.ndarray] | None = None
    for attempt in range(int(attempts)):
        rng = np.random.default_rng(int(seed) + attempt)
        blocks = [base]
        for _ in range(dv - 1):
            blocks.append(base[:, rng.permutation(n)])
        h = np.concatenate(blocks, axis=0)
        cyc = _count_cycles4(h)
        if best is None or cyc < best[0]:
            best = (cyc, h)
        if cyc == 0:
            break
    assert best is not None
    cycles4, h = best

    reduced, order, rank = _gf2_systematic(h)
    k = n - rank
    parity = reduced[:, rank:]
    generator = np.concatenate([parity.T % 2, np.eye(k, dtype=np.int8)], axis=1)
    check_vars = np.stack([np.nonzero(row)[0] for row in h])
    edges = check_vars.ravel()
    var_edges = np.stack([np.nonzero(edges == v)[0] for v in range(n)])
    return LdpcCode(h, generator, order, check_vars, var_edges, int(cycles4))


def _count_cycles4(h: np.ndarray) -> int:
    """Pairs of checks sharing two or more variables, a length-4 cycle count."""
    overlap = (h.astype(np.int32) @ h.T.astype(np.int32))
    np.fill_diagonal(overlap, 0)
    pairs = overlap[np.triu_indices_from(overlap, k=1)]
    return int(np.sum(pairs * (pairs - 1) // 2))


def sum_product_decode(
    code: LdpcCode,
    llr,
    iterations: int = 20,
    message_clip: float = 30.0,
    early_stop: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flooding sum-product decoding of ``(B, n)`` channel LLRs.

    Parameters
    ----------
    llr:
        ``log P(0)/P(1)`` per channel bit. ``(B, n)`` or ``(n,)``.
    iterations:
        Maximum number of flooding iterations.
    message_clip:
        Internal message magnitude cap, dimensionless. Numerical only: it
        bounds ``tanh`` arguments away from ``+-1``. It is **not** the
        receiver LLR clipping studied in
        ``validation/validate_maxlog_clipping.py``, which is applied to the
        channel LLRs before they reach this function.
    early_stop:
        Stop once every block satisfies its parity checks.

    Returns
    -------
    ``(bits, posterior, iterations_used)``: hard decisions ``(B, n)``,
    posterior LLRs ``(B, n)``, and the number of iterations actually run.
    """
    channel = np.atleast_2d(np.asarray(llr, dtype=float))
    if channel.shape[-1] != code.length:
        raise ValueError(
            f"llr must have last dimension {code.length}, got {channel.shape[-1]}"
        )
    if int(iterations) < 1:
        raise ValueError(f"iterations must be at least 1, got {iterations!r}")
    clip = float(message_clip)
    if not np.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"message_clip must be a finite positive number, got {message_clip!r}")

    cv = code.check_vars
    m, dc = cv.shape
    flat = cv.ravel()
    ve = code.var_edges
    posterior = channel.copy()
    m_vc = channel[:, flat].copy()
    used = 0
    for it in range(int(iterations)):
        used = it + 1
        t = np.tanh(0.5 * np.clip(m_vc, -clip, clip)).reshape(-1, m, dc)
        fwd = np.ones_like(t)
        np.cumprod(t[:, :, :-1], axis=2, out=fwd[:, :, 1:])
        bwd = np.ones_like(t)
        np.cumprod(t[:, :, :0:-1], axis=2, out=bwd[:, :, -2::-1])
        excl = np.clip(fwd * bwd, -1.0 + 1e-15, 1.0 - 1e-15)
        m_cv = np.clip(2.0 * np.arctanh(excl), -clip, clip).reshape(-1, m * dc)
        gathered = m_cv[:, ve]
        total = channel + gathered.sum(axis=2)
        posterior = total
        if early_stop:
            bits = (total < 0.0).astype(np.int8)
            if not np.any(code.syndrome(bits)):
                break
        m_vc = total[:, flat] - m_cv
    bits = (posterior < 0.0).astype(np.int8)
    return bits, posterior, np.array(used)
