# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-06

First release. Validation level 2 (research grade). Status: TESTING.

### Added

- `interleavekit.base` - the permutation convention the whole package rests on
  (`y[i] = x[pi[i]]`, so `pi[i]` is the source index at transmitted position `i`),
  the `Interleaver` abstract base class, and `InterleaverCost`: pair and one-way
  latency and memory in symbols, with conversion to milliseconds at a stated
  symbol rate and to bytes at a stated symbol width.
- `interleavekit.block` - row-write, column-read block interleaver with a
  closed-form permutation, a closed-form inverse, and a closed-form minimum
  spread `min(span+1, depth+1, depth+span-2)` derived in the module docstring and
  checked against brute-force enumeration on 2468 `(depth, span)` pairs.
- `interleavekit.convolutional` - shift-register-bank convolutional interleaver.
  Register `k` holds `k * slope` symbols and is clocked once every `registers`
  symbol times, so the transmitted position of source symbol `i` is
  `i + (i % registers) * slope * registers`. The matched de-interleaver reverses
  the delays, giving a pair latency of `registers * slope * (registers - 1)` that
  is the same for every symbol. Reports its steady-state transmitted range, so
  burst metrics are never measured across start-up fill symbols.
- `interleavekit.helical` - fixed-length block interleaver with a wrapping
  diagonal read, a bijection for every step value. Not bit-compatible with
  MATLAB `helintrlv`, which is a streaming block; the difference is documented.
- `interleavekit.srandom` - S-random interleaver by randomised backjumping search,
  deterministic in `(length, spread, seed)`. Forward-only greedy placement, which
  is how the construction is usually described, was measured to fail at length
  1024 and spread 16; single-step backtracking also failed at length 256 and
  spread 10. Backjumping three positions is what ships.
- `interleavekit.metrics` - minimum spread, S-parameter, normalised dispersion,
  and the burst metrics: `transmitted_span_profile`, `burst_dispersion`,
  `burst_dispersion_profile` and `max_burst_fully_dispersed`. Burst dispersion is
  computed from the identity "`R` consecutive source symbols can be hit by a
  burst of `L` exactly when `m(R) <= L - 1`", which makes it `O(N)` instead of
  `O(N*L)`; the definitional window scan is retained as
  `burst_dispersion_by_window_scan` and the two agree on 5304 comparisons.
- `interleavekit.cost` - `describe` and `cost_table` to measure spread, burst
  dispersion and cost together, `format_cost_table` to print them, and
  `cheapest_for_burst` to search each family for the cheapest parameter set that
  measurably disperses a target burst.
- CLI `python -m interleavekit` with `permutation`, `metrics`, `cost`, `compare`
  and `design` subcommands.
- 191 tests, including hand-computed permutations written out in the test
  comments; eight validation scripts with their committed raw output; four
  examples, each producing a PNG in `screenshots/`.

### Known limitations in this release

- The block, helical and S-random cost figures use a **full-block buffering**
  model: `N` symbols at each end, `2N` symbol times for the pair. An
  implementation that overlaps read-out with write-in pays less. The measured
  block-to-bank memory ratio of 3.3 to 3.9 depends on this model and is not
  claimed to be universal.
- The S-random search is bounded, not optimal. The largest spread it reaches
  falls from 1.06 times `sqrt(length/2)` at length 64 to 0.53 times at length
  1024, and it is not monotone in the requested spread: at length 256 it reaches
  spread 11 but fails at 10 with the same seed and budget.
- `dispersion` is `O(N^2)` and refuses lengths above 2048 with a message naming
  the ceiling.
- The S-random cost record does not count the stored permutation table, which
  both ends need because the permutation has no algebraic form. Block and helical
  permutations are computed from two integers and need no table.
- Nothing is validated against hardware, against a link measurement, or against
  a bit-error-rate result. Every number in `validation/` is combinatorial.
- No comparison was executed against `komm` or `scikit-commpy`. Their source was
  read, not run, because neither installs in the build environment.
