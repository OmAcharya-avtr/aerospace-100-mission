"""Extended Hamming (8,4) with exhaustive soft and hard decoding.

Used for cross-check X3, where the required statement is that soft-decision
decoding is never worse than hard-decision decoding on the *same* channel
realisations. Exhaustive maximum-likelihood decoding over all 16 codewords is
used so that neither decoder has an algorithmic shortfall: any ordering
violation would be a property of the channel model or of the LLRs, not of a
suboptimal decoder.

The code
--------
``(8, 4)`` extended Hamming, minimum distance 4, rate 1/2, systematic
generator ``G = [I_4 | P]`` with

    P = [[0, 1, 1, 1],
         [1, 0, 1, 1],
         [1, 1, 0, 1],
         [1, 1, 1, 0]]

so every row of ``G`` has weight 4 and the all-ones word is a codeword. The
weight distribution is checked in the tests (1 word of weight 0, 14 of weight
4, 1 of weight 8), which fixes ``d_min = 4``.

Decoding rules
--------------
With ``L_i = log P(c_i = 0) / P(c_i = 1)``,

* **soft-decision ML**: maximise ``sum_i (1 - 2 c_i) * L_i`` over codewords.
  This is the exact maximum-likelihood rule for a memoryless channel whose
  bit log-likelihood ratios are ``L_i``;
* **hard-decision ML**: form ``b_i = (L_i < 0)`` and minimise the Hamming
  distance to a codeword, discarding the magnitudes.

Both are evaluated by one ``(16, 8)`` matrix contraction, so both see
identical realisations. Ties are broken by lowest codeword index in both.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ExtendedHamming84", "EXTENDED_HAMMING_84"]


class ExtendedHamming84:
    """The (8,4) extended Hamming code with both decoders.

    Attributes
    ----------
    length, dimension, rate, min_distance:
        ``8``, ``4``, ``0.5``, ``4``.
    """

    length = 8
    dimension = 4
    rate = 0.5
    min_distance = 4

    def __init__(self) -> None:
        parity = np.array(
            [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=np.int8
        )
        self.generator = np.concatenate([np.eye(4, dtype=np.int8), parity], axis=1)
        self.parity_check = np.concatenate([parity.T, np.eye(4, dtype=np.int8)], axis=1)
        messages = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.int8)
        self.messages = messages
        self.codewords = (messages @ self.generator) % 2
        self._signs = (1 - 2 * self.codewords).astype(float)

    def encode(self, messages) -> np.ndarray:
        """Encode ``(..., 4)`` message bits into ``(..., 8)`` codeword bits."""
        m = np.asarray(messages, dtype=np.int8)
        if m.shape[-1] != self.dimension:
            raise ValueError(f"messages must have last dimension 4, got {m.shape[-1]}")
        if np.any((m != 0) & (m != 1)):
            raise ValueError("messages must be 0/1")
        return (m @ self.generator) % 2

    def syndrome(self, words) -> np.ndarray:
        """Syndrome ``H c^T mod 2`` of ``(..., 8)`` words."""
        c = np.asarray(words, dtype=np.int8)
        if c.shape[-1] != self.length:
            raise ValueError(f"words must have last dimension 8, got {c.shape[-1]}")
        return (c @ self.parity_check.T) % 2

    def decode_soft(self, llr) -> np.ndarray:
        """Exhaustive soft-decision ML decoding of ``(..., 8)`` LLRs.

        Returns the ``(..., 4)`` decoded message bits.
        """
        llr = np.asarray(llr, dtype=float)
        if llr.shape[-1] != self.length:
            raise ValueError(f"llr must have last dimension 8, got {llr.shape[-1]}")
        metric = llr @ self._signs.T
        return self.messages[np.argmax(metric, axis=-1)]

    def decode_hard(self, llr) -> np.ndarray:
        """Hard-decision (minimum-Hamming-distance) decoding of the same LLRs.

        Only ``sign(llr)`` is used. Returns the ``(..., 4)`` message bits.
        """
        llr = np.asarray(llr, dtype=float)
        if llr.shape[-1] != self.length:
            raise ValueError(f"llr must have last dimension 8, got {llr.shape[-1]}")
        bits = (llr < 0.0).astype(np.int8)
        distance = (bits[..., None, :] != self.codewords).sum(axis=-1)
        return self.messages[np.argmin(distance, axis=-1)]

    def weight_distribution(self) -> np.ndarray:
        """Number of codewords of each Hamming weight 0..8."""
        return np.bincount(self.codewords.sum(axis=1), minlength=self.length + 1)


EXTENDED_HAMMING_84 = ExtendedHamming84()
