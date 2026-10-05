# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-05

First release. Status: TESTING.

### Added

- `framesync.asm`: the 32-bit CCSDS attached sync marker 0x1ACFFC1D, a
  sliding Hamming-distance correlation detector, the combinatorial
  false-sync expression `P_fa(T) = 2^-L * sum_{k<=T} C(L, k)`, the expected
  false-sync count over a stream, and the marker's cyclic autocorrelation.
- `framesync.sync`: the four-state acquisition / check / lock / flywheel
  synchroniser with configurable thresholds, plus slip analysis across a
  known bit insertion or deletion with the re-acquisition latency in frames.
- `framesync.channel`: the Eb/N0-per-information-bit convention, coherent
  BPSK over AWGN, the code-rate offset `Es/N0 = R Eb/N0`, a seeded binary
  symmetric channel, and matched-filter sample generation for soft decision.
- `framesync.rs`: RS(255,223) encode and decode over `I` interleaved
  codewords via `reedsolo`, plus the exact codeword-failure and frame-error
  expressions and the standard post-decoding bit-error approximation.
- `framesync.conv`: the CCSDS rate-1/2 K=7 (171, 133) convolutional
  encoder with the G2 inversion, a Viterbi decoder vectorised across frames
  with hard and soft branch metrics, and free-distance computation from the
  trellis.
- `framesync.crc`: CRC-16 Frame Error Control Field, scalar and batch paths.
- `framesync.frames`: transfer frame geometry, FECF generation and checking,
  and marker-prefixed bit stream construction.
- `framesync.fer`: analytic and Monte Carlo frame error rate for all three
  links, binomial standard errors, point sizing from a target precision,
  and RS coding gain at a target rate.
- CLI `python -m framesync` with `fer`, `falsesync`, `gain`, `sync` and
  `slip` subcommands.
- 122 tests, four runnable examples, seven validation scripts with raw
  output committed.

### Known deviations

- `commpy` (`scikit-commpy`) and `crcmod`, named as dependencies in the
  build specification, cannot be installed in the build container: neither
  ships a wheel for Python 3.13 and both fail to build from source. The
  convolutional encoder, the Viterbi decoder and the CRC are implemented
  internally instead, and both packages are named in the README
  alternatives table as the mature choices a reader should prefer. They are
  not dependencies of this package.
- The RS coding gain is compared against the asymptotic-gain formula rather
  than against a numeric range quoted from CCSDS 130.1-G, which was not
  available to read in the build container. See `validation/VALIDATION.md`.
