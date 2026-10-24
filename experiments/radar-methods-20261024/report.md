# Radar motion, bounded adaptation, and sparse inference — 24 October 2026

## Decision

Ship deterministic radar motion as an opt-in experiment; keep Aurora rainfall as
the daily default. Do not publish calibrated weights from these cases. Motion
helped the first hours of one measured case, then degraded sharply. Agreement
with the supplied Buienradar reference improved, but that is development evidence,
not demonstrated forecast accuracy.

Aurora stays frozen during forecasting. One isolated internal LoRA update fitted
on the M3/16 GB machine; its weights were discarded. Sparse inference is enabled
in the default configuration after exact float32 equivalence checks. Historical
engine callers still receive dense output by default.

## Method and data

The method uses pySTEPS [Lucas–Kanade motion](https://pysteps.readthedocs.io/en/latest/generated/pysteps.motion.lucaskanade.dense_lucaskanade.html)
on 24 masked log rain-rate images, then [semi-Lagrangian extrapolation](https://pysteps.readthedocs.io/en/latest/generated/pysteps.extrapolation.semilagrangian.extrapolate.html)
of measured five-minute amounts. Ninety-six intervals are streamed into hourly
sums using carried displacement fields. Motion never consumes future observations
or previously predicted rain. Missing/entering trajectories remain unknown;
Aurora supplies unsupported bins. Valid dry history uses zero motion.

Full-domain history, motion, and hourly grids have checksummed caches shared
between locality queries. A radar refresh does not invalidate Aurora forecasts.
The elapsed portion of a clock hour uses measured radar; unsupported measured
prefixes retain the complete Aurora hour. Raw Aurora and motion values, coverage,
cutoffs, sources, and correction identities remain available in reports.

Calibration selects weights {0, .25, .5, .75, 1} independently for eight hours on
validation only, requiring lower MAE without worse CSI at 0.1 mm/hour. Evaluation
keeps raw motion even if its weight is zero, and reports paired Aurora support,
hourly/cumulative errors, events, spatial skill, area-aggregated errors, coverage,
and case-bootstrap intervals. A single case cannot establish an uncertainty interval.

## Bounded acquisition audit

KNMI listed complete January/April/July 2024 filename windows. Downloading the
January window consumed **143,652,592 bytes**. Decoding stopped because this
archive uses `ACCUMULATION_[MM]` and timestamps ending in `:03`, whereas the
supported measured schema requires `PRECIP_[MM]` and exact five-minute boundaries.
The files were preserved; their units/times were not silently reinterpreted.
No April/July payloads were downloaded.

The recent comparison downloaded **2,959,027 bytes**. Total new pilot transfers:
**146,611,619 bytes**, below the **250,000,000-byte** ceiling. The measured historical
case below reused existing raw radar and atmospheric inputs with zero additional
transfers. Independent chronological train/validation/test splits could not be
assembled from supported data in this pilot; broader qualification stays pending.
Acquisition timestamps cannot establish historical publication availability, so
both cases are explicitly retrospective.

## Saved Leusden development reference

The original manually captured Buienradar snapshot remains unchanged in
[buienradar.json](../leusden-20261024/buienradar.json). Its issuance timestamp and
rainfall interval convention are unverified; amounts are treated as hour-ending.
The replay uses the saved Aurora initialization at 06:00 UTC and nominal request
13:34 UTC, with measured radar through 13:30. No parameters or weights were fitted
to these eight rows. Temperature is unchanged by radar motion.

| Local hour | Temperature, Aurora °C | Reference °C | Raw Aurora rain mm | Motion + supported fallback mm | Reference rain mm |
|---|---:|---:|---:|---:|---:|
| 16:00 | 7.97 | 12 | .005 | .000 | .0 |
| 17:00 | 8.18 | 11 | .010 | .000 | .0 |
| 18:00 | 8.47 | 11 | .017 | .000 | .0 |
| 19:00 | 8.74 | 10 | .025 | .030 | 1.2 |
| 20:00 | 8.96 | 10 | .035 | .192 | 1.5 |
| 21:00 | 8.67 | 9 | .040 | .166 | .5 |
| 22:00 | 8.67 | 9 | .044 | .487 | 1.4 |
| 23:00 | 8.76 | 9 | .052 | .257 | 3.0 |
| Eight hours | | | **.228** | **1.132** | **7.6** |

Raw Aurora rainfall MAE is .9295 mm/hour; motion-result MAE is .8085 mm/hour,
with bias −.8085 and RMSE 1.2015. Eight-hour differences are −7.372 mm and
−6.468 mm. Temperature MAE is 1.5725°C. Radar support is complete at Leusden for
all requested future bins. The first hour's measured portion is zero; raw Aurora
above remains its untouched full-hour output. A second comparison applies the
identical measured prefix to Aurora: its total is .226 mm and MAE .92925 mm/hour,
versus the same motion total 1.132 mm and MAE .8085. Both comparisons are preserved.

Motion estimation took **26.111 s** and streamed extrapolation **8.992 s**, with
peak process RSS **.75 GiB** in this replay. These exclude Aurora loading/inference
and are not full cold-request timings.

## Separate measured-radar case: 23 October, 20:00 UTC cutoff

A fresh local Aurora replay initialized at 12:00 UTC supplies model leads 9–16.
The existing 120 full-domain RTCOR frames cover 18:05 through 04:00 UTC. Only the
first 24 enter motion; the following 96 supply eight-hour measured targets.
This is a single diagnostic case, not an independent held-out qualification set.

| Lead hour | Aurora MAE mm | Aurora MAE on motion support | pySTEPS MAE mm | Uncalibrated combination MAE mm | Motion coverage |
|---|---:|---:|---:|---:|---:|
| 1 | .360 | .378 | .197 | .187 | .831 |
| 2 | .344 | .378 | .366 | .332 | .792 |
| 3 | .343 | .395 | .470 | .406 | .753 |
| 4 | .345 | .416 | .569 | .469 | .715 |
| 5 | .349 | .441 | .653 | .513 | .680 |
| 6 | .364 | .475 | .740 | .556 | .637 |
| 7 | .368 | .490 | .790 | .569 | .589 |
| 8 | .343 | .473 | .810 | .551 | .542 |

Across available pixel-hour pairs, Aurora MAE is .3520 mm, persistence .4899,
FFT motion .4693, pySTEPS motion .5476, and the full-weight combination .4479.
Their supports differ: motion-only comparisons require the paired baseline column,
not a direct comparison with the full-domain Aurora average. The combination and
Aurora have the same .8769 coverage. The combination is worse over eight hours;
there is no basis for enabling it by default or selecting weights using this case.

Replaying Aurora took **32.842 s**, including assets/load/device transfer. Five-method
construction and scoring took **29.132 s**; maximum process RSS was **4.58 GiB**.
The full JSON retains per-hour events, spatial skill, cumulative and total errors,
coarse-grid scores, masks/coverage, input hashes, and stage timings.

## One-step LoRA feasibility

Pinned Aurora 1.5, rank eight, single adapter, float32 MPS, original 48×64 grid,
batch one, cached ERA5 initialization 1 January 2024 06:00 UTC and its matching
six-hour-ahead hourly-rain target. All original parameters were frozen. Missing
checkpoint keys were permitted only for newly introduced LoRA matrices.

- 2,850,816 trainable parameters; zero-adapter rainfall difference exactly zero.
- One checkpointed training forward/backward/AdamW update: **4.738 s**.
- Finite loss **.01770957**, finite/nonzero gradients, adapter update confirmed.
- Base parameter version counters unchanged; no base gradients.
- Load **94.854 s**; zero-equivalence checks **2.503 s**; parent **108.584 s**.
- Peak process RSS **3.10 GiB**; final MPS allocation **4.84 GiB**, driver **5.35 GiB**.
- Ten-minute subprocess timeout, 12 GiB process checks, 10 GiB driver checks, and
  an MPS allocator fraction of .8. The process guard is not a total unified-memory cap.

The two no-gradient equivalence forwards are additional to the single training
step. Earlier worker attempts failed before training because of constructor,
lead-tensor, and target-shape integration errors; those logs remain separate.
The successful run establishes this step's hardware feasibility only. No updated
weights were saved or deployed. Larger-domain training and forecast skill remain untested.

## Sparse inference

Mandatory bridges match pinned upstream batch preparation, clipping, and history
advancement. Leads 8–15 execute **6, 8, 9, 10, 11, 12, 13, 14, 15**: nine forwards.
A second window **5, 6, 7, 12, 13, 18** crosses six-hour boundaries. All five physical
output variables matched dense MPS inference exactly in both windows. A tiny real
Aurora CPU regression additionally compares every surface and atmospheric output.

One comparable pair of separate-process cached-input requests, including Python
imports, model loading, query, and close, took **84.308 s dense** and **81.373 s sparse**.
The measured inference stages totaled **4.965 s** and **3.720 s**. Peak MPS driver
allocations were **5.25/5.26 GiB**. These are single measurements on a shared machine,
exclude live cycle/observation network checks, and do not establish a reliable
percentage speedup. Startup dominates; forward counts alone overstate latency benefit.
A reused engine completed the sparse 8–15 window in **2.223 s**.

Cache schema 3 stores actual lead coverage and rejects missing requested timestamps,
corruption, or incompatible schemas. Dense historical behavior remains the API
default; `sparse_forecasts` controls daily queries.

## Reproduce and inspect

Run `pixi install --locked`, then the README's motion/calibration/evaluation commands
with prepared measured data and immutable locally computed forecasts. Use
`prepare-radar-native --full-domain` for deterministic methods and ordinary cropped
preparation for CNN training. `adapter-feasibility` is an explicit separate command.

Local generated evidence is preserved under `outputs/radar-methods-20261024/`:
`radar-motion-leusden.json`, `buienradar-motion-comparison.json`,
`aurora-matched-prefix.json`, `buienradar-matched-comparison.json`,
`recent-measured-evaluation.json`, `sparse-mps.json`, `cold-dense.json`,
`cold-sparse.json`, `adapter-feasibility-v4/report.json`, `pilot-status.json`, and
`historical-availability.json`. Large prepared grids, model assets, and acquisition
manifests remain outside Git. The original Buienradar snapshot is never used for
calibration or as an input to forecasting.

## Verification

Locked Pixi installation succeeded. Formatting, Ruff, and strict mypy passed;
113 offline tests passed, plus the opt-in architecture forward/backward integration
test. New regressions cover translated/dry/missing rain, domain exits, eight-hour
amounts, cutoff exclusion, observed prefixes, unsupported-bin fallbacks, calibration
isolation/checksums, CLI option conflicts, cache lead coverage, and actual sparse
versus dense Aurora autoregression. Real float32 MPS runs are recorded above.
The synthetic optical-flow tests emit two pySTEPS singular-matrix warnings during
outlier rejection; their motion and coverage assertions passed.
