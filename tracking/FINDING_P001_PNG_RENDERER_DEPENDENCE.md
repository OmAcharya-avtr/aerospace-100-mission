# Finding — P001 `margin_histogram.png` byte change: root cause identified

**Status:** CLOSED as explained. No defect in P001. No data change. One genuine
limitation recorded, which generalises beyond P001.
**Opened:** 2026-10-10, as "unexplained and uninvestigated" in `MISSION_STATUS.md`.
**Closed:** 2026-10-12 by direct measurement in the container.

## What was observed, and why it was alarming

A partial release-gate run on 2026-10-10 moved
`products/P001/screenshots/margin_histogram.png` from **66,047 to 75,602 bytes** —
a 14 per cent change in a committed artifact of a **published** product. The
2026-10-10 session correctly rejected the obvious explanation: 9.5 kB is far too
large to be embedded metadata such as a timestamp. It recorded the change as
unexplained rather than guessing, and the 2026-10-11 session narrowed the
mission's reproducibility claim because of it. Both were the right calls.

## Reproduction

`examples/margin_histogram.py` was re-run in this container. The committed file
was copied aside first and restored afterwards, so the tree is clean.

The byte sizes reproduced **exactly**: 66,047 → 75,602. This is the same event,
not a similar one.

## Root cause: the matplotlib version changed, and the PNG says so itself

Matplotlib writes its own version into a PNG `tEXt` chunk. The two files
therefore carry their provenance:

| File | `tEXt` Software chunk |
|---|---|
| committed | `Matplotlib version3.10.9, https://matplotlib.org/` |
| regenerated | `Matplotlib version3.11.2, https://matplotlib.org/` |

The committed artifact was rendered by **matplotlib 3.10.9**. This container
ships **3.11.2** (freetype 2.14.3, numpy 2.5.3). The artifact is
renderer-version-dependent, and the difference is a rasterization difference.

## The change is not metadata and not anti-aliasing

Measured over the two decoded images (identical `IHDR`: 1092 x 672 RGBA,
identical `pHYs`):

- **84,846 of 733,824 pixels differ — 11.56 per cent.**
- Deltas are **saturated, not subtle**: max channel delta 255, median delta 95
  over differing pixels, and 4.77 per cent of the whole image differs by more
  than 128. Anti-aliasing cannot produce that.
- **Ink quantity is nearly unchanged** — 59,134 non-white pixels committed
  versus 58,591 regenerated — but **ink IoU is only 0.257**, and the ink
  bounding boxes agree (rows 25–649 vs 26–650, cols 21–1071 vs 24–1071).
- It is **not a translation**. An exhaustive search over all shifts in
  ±12 px in both axes raises IoU only from 0.257 to 0.332, at dy = −6, dx = +1.

Same amount of ink, same envelope, different placement within it.

## Mechanism

The figure draws two `histtype="step"` histograms with **220 bins** at
**1.2 pt** linewidth across roughly 1,000 px of axes — about 4.5 px per bin.
A step outline at that density is a long, thin zigzag polyline, which is the
structure most sensitive to a change in how Agg snaps and rasterizes thin
paths. Shifting a 1.2 px line by a sub-pixel amount produces a *saturated*
delta along its entire length, and this figure contains a great deal of such
length. The busiest differing rows hold roughly 1,010 differing pixels out of
1,092 — these are the horizontal segments of the step outlines and the
gridlines, caught full width.

Compressed size grows for the same reason: less pixel-aligned line
rasterization compresses worse. The second `IDAT` chunk goes from **351 to
9,906 bytes** while the first stays at the 65,536 maximum.

## The data did not change — this is the important part

`examples/margin_histogram.py` is fully deterministic: `N_MC = 300_000`,
`SEED = 5`, seeded through every sampling call. The regenerated run printed:

```
combined  : P_fade=9.2667e-04  mean=+10.12 dB  std=3.22 dB
percentiles [dB]: p01=+2.59, p05=+4.80, p50=+10.13, p95=+15.39, p99=+17.53
scint-only : P_fade=2.1333e-04  mean=+10.82 dB  std=3.13 dB
jitter-only: P_fade=0.0000e+00  mean=+11.24 dB  std=0.76 dB
```

`README.md` states the combined fade probability is `9.27e-04` at
`n = 300,000, seed 5`. The regenerated value is `9.2667e-04`, which is
`9.27e-04`. **The scientific content is identical across the two matplotlib
versions. Only the rasterization differs.** P001's published numbers stand.

## What this means, stated no more strongly than measured

1. **P001 has no defect and needs no correction.** The figure conveys the same
   distribution; the README's claims about it remain true.
2. **Committed raster artifacts are not reproducible across renderer
   versions.** This is a real limitation and it is not specific to P001 — any
   product committing a matplotlib PNG has it. It belongs in the same family as
   P054 recording `platform.platform()` into committed output: the artifact
   encodes its environment.
3. **Do not "fix" this by regenerating the PNG.** Regeneration pins the artifact
   to matplotlib 3.11.2 and moves the identical problem to the next upgrade,
   while producing churn in a published product for no scientific gain.
4. **The lesson for the gate and for reproducibility claims:** a byte-level diff
   on a committed raster artifact is evidence about the renderer, not about the
   data. Reproducibility must be asserted over *numbers*, which are stable here,
   not over *image bytes*, which are not. The mission's reproducibility claim was
   narrowed on 2026-10-11 and this finding supports that narrowing rather than
   reversing it.

## Recommended follow-up, not done here

- Record the generating library version beside every committed figure, or pin
  the plotting stack per product, so a future byte change is immediately
  attributable instead of alarming. Either is a cross-cutting change to 50
  published products and is not an unattended-session decision.
- Prefer asserting figure *content* (the printed statistics, already covered by
  the README and validation output) over figure *bytes*. No gate check currently
  diffs artifact bytes, so nothing in the publication path is affected today.
