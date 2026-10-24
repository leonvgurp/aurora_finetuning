# Design and evidence

## Why this configuration

Aurora 1.5 supplies native hourly outputs, precipitation, and solar insolation.
Its pretrained weights contain learned weather dynamics. Fresh initialization
supplies today's state. Recent observations condition separate local adjustments;
we do not retrain Aurora during a forecast request.

- [Official model catalogue](https://microsoft.github.io/aurora/models.html):
  checkpoint-specific variables, source compatibility, statics, and hourly rollout.
- [Input cautions](https://microsoft.github.io/aurora/beware.html): source and
  regridding affect accuracy; IFS T0 differs from analysis.
- [Fine-tuning guide](https://microsoft.github.io/aurora/finetuning.html): full-model
  reference examples use NVIDIA/A100 infrastructure. This repository trains only
  external models on the Mac.
- [Original Nature paper](https://www.nature.com/articles/s41586-025-09005-y):
  diverse analysis, reanalysis, forecast/reforecast and climate data underpin the
  published global/task-specific performance. ERA5, HRES, IFS ensembles, GFS,
  GEFS, CMIP6, MERRA-2 and CAMS serve different tasks; they are not interchangeable.
- [Aurora 1.5 research](https://www.microsoft.com/en-us/research/publication/aurora-1-5-fine-tuning-a-foundation-model-for-medium-range-ensemble-weather-prediction/):
  variable expansion, lead-time training, ensemble objectives, and rollout tuning.
  Published global ensemble scores do not describe this regional deterministic run.
- [Regional discussion](https://github.com/microsoft/aurora/issues/87): global
  pretraining, missing boundary context, longitude wrap, and positional numerics
  make cropped forecasts experimental.
- [KNMI station documentation](https://dataplatform.knmi.nl/dataset/docs/10-minute-in-situ-meteorological-observations-1-0):
  timestamps, variable-specific intervals and near-real-time availability.
- [Validated stations](https://dataplatform.knmi.nl/dataset/hourly-in-situ-meteorological-observations-validated-1-0):
  delayed expert validation is suitable for targets, not live input reconstruction.
- [RTCOR radar](https://dataplatform.knmi.nl/dataset/nl-rdr-data-rtcor-5m-tar-1-0):
  measured five-minute accumulations already incorporate gauge information.
- [ECMWF open data](https://www.ecmwf.int/en/forecasts/datasets/open-data):
  step-zero surface/pressure initialization fields; retrieval is global by field,
  even though prepared artifacts are cropped regionally.

This is a focused primary-source reference set, not a claim to have enumerated
all Aurora literature. Microsoft examples and the installed pinned source remain
the API reference. No new model/environment dependencies were introduced.

## Responsibility boundaries

`engine.py` owns Aurora's session lifetime; `model.py` contains loading, device
operations and physical outputs. `weather.py` handles live orchestration and
caching; `forecast.py` queries gridded predictions. Acquisition/preparation modules
retain provider-specific validation. Station and radar modules separate data,
features/models, training and evaluation. CLI parsing and configuration remain at
the boundary. The results notebook is an analysis client; implementation remains in Python modules.

A session reuses one frozen Aurora instance. A forecast cache hit avoids weights,
atmospheric tensor preparation and network cycle discovery. Cache identities
include model, actual grid, precision, numerical source hashes and input manifests.
Missing cached coverage triggers fresh-input validation. Cached forecasts may be
older than the newest published cycle but must remain within the configured age.
Observation refreshes do not invalidate the independent Aurora cache.

JSON/NetCDF/checkpoint publication uses same-filesystem exclusive atomic links.
A failed publication cannot replace a valuable existing output. Data/manifest pairs
are individually atomic; interrupted pairs fail validation rather than being reused.

## Observation timing and seasonal context

Annual/day sine/cosine features are UTC and continuous across calendar boundaries.
They are supplied to external models alongside geography and lead time. No climate
average downloads or forced seasonal reversion are used. Seasonal feature selection
uses validation, not test observations. Aurora retains timestamps and insolation.

A station residual uses a forecast available by the observation timestamp. Recorded
`available_utc` controls this; initialization-only historical archives are explicitly
retrospective. Station `available_time` coordinates, when present, prevent using
records acquired after issuance. Prospective KNMI cache manifests retain acquisition
time. The final issuance follows acquisition while observation timestamps remain
bounded by the request cutoff. Full 72-hour intraday near-real-time coverage is not
silently synthesized: overnight hourly context and the latest 10-minute window may
leave gaps, represented through coverage/missingness features.

Station NetCDF inputs use documented KDP coordinates and variable metadata:
`T`/`ta` (1.5 m temperature), `P`/`pp` (explicit mean-sea-level pressure), and
`FF`/`ff` plus `DD`/`dd` (10-minute mean wind). Gusts are excluded; unknown pressure
references, units, durations and quality-flag semantics fail explicitly. Station
1.5 m temperature versus Aurora 2 m temperature, and mean versus instantaneous
wind, remain representativeness limitations. Missing observations are not dry rain.

Station preparation reads train/validation targets only. Whole held-out stations
are excluded both as training targets and as feature neighbours, and the exclusion
is recorded in the trained artifact. Evaluation rejects unrecorded exclusions.

Radar uses 24 past frames and predicts 96 future bins directly. Validation/test
windows include the full two-hour history/eight-hour target range plus a 30-hour
embargo; station splits include the full 72-hour history plus the same embargo.
Training never reads test targets. Current examples are structural templates,
not adequate seasonal datasets.

## Verification and skill reporting

Station ridge selection compares regularization and calendar ablations, then freezes
per-variable/hour correction weights. Radar selection compares calendar ablations
for conditioned and radar-only CNNs and freezes hourly blend weights. Zero is an
allowed weight and wins ties; rainfall nonzero weights must improve validation MAE
without worsening CSI at 0.1 mm/hour.

Radar reports raw adapter output, selected combined output, Aurora, persistence,
motion extrapolation and the trained radar-only control. Native-grid verification
uses shared finite masks for Aurora and learned methods. Motion extrapolation
can leave the observed domain; it receives an additional paired Aurora score
on exactly its own finite support, at both native and coarse resolutions. Area
aggregation averages predictions and observations over the same native pixels.
Comparisons with no overlapping pixels have count zero, coverage zero, and null
error scores. They remain visible in coverage reports and are excluded from error
averages and bootstrap estimates; other methods retain their own valid coverage.
Fractions skill uses complete neighbourhoods at 1/5/10 km; neighbourhoods outside
coverage are undefined, not zero skill. Case bootstrap intervals resample cases,
not correlated pixels. With one case intervals are unavailable. Adjacent cases may
still share weather systems: independent weather-event selection is necessary.

Clock-hour live totals and issuance-relative historical lead hours are labelled
separately. The first live hour can contain observed elapsed rainfall; skill of that
measured prefix must not be presented as predictive skill. Historical publication
latency is not fully reconstructed. Radar/gauge dependence limits independence of
station rainfall verification. Eight-hour detail is a hypothesis to test, not a promise.

## Measured local execution, 24 October 2026

One cached historical case (2024-01-01 06 UTC, aligned 48×64, eight hours) ran on
M3 MPS with float32:

| Stage | Seconds |
|---|---:|
| Cold engine, including preparation/loading | 38.86 |
| Warm engine, same inputs and reused weights | 2.12 |
| Cached location query | 0.055 |

Repeated physical outputs were identical. Peak process RSS was 4.35 GiB;
MPS driver allocation snapshots reached 5.25 GiB. These counters must not be added
as a definitive unified-memory peak. No network bytes were transferred. The report
is `outputs/refactor-benchmark-20261024/benchmark.json` (ignored generated artifact).
Different filesystem caches/startup conditions prevent attributing the difference
from an older 128-second run entirely to code changes.

The pinned upstream autocast chooses CPU rather than MPS. The first attempted
float16 comparison is marked invalid for MPS precision benchmarking. The implementation
now wraps inference in an explicit device autocast context. Float32 remains the
default; a passing numerical comparison alone does not establish speed or memory
benefit or forecast accuracy. Use `benchmark --reference PATH` for one candidate run
against a saved float32 reference, or `--compare-precision` for both in one command.

Quality checks and synthetic tests verify implementation behavior, not weather
skill. Existing cached station observations and a single hindcast do not supply
an adequate trained/held-out radar study. Complete radar/station model qualification
requires curated historical cases and prospective observation collection. The
repository provides those workflows without launching large downloads automatically.

The corrected explicit-MPS float16 experiment passed numerical gates on this case
(temperature difference MAE 0.00236°C; rainfall 0.000278 mm), but later hourly
forward passes were approximately 0.39–0.44 seconds versus about 0.24–0.27 seconds
for float32. Accelerator allocation remained about 5.2 GiB. Therefore mixed
precision was **not promoted**. Cold loading times varied and are not evidence
of a precision speedup. The valid candidate report is
`outputs/refactor-mps-autocast-20261024/benchmark.json`.

## Bounded observation validation

Live KNMI ingestion was exercised against both operational hourly and near-real-time
10-minute files. Normalizing WIGOS identifiers reconciled the products to 71 distinct
stations. Temperature, pressure and both wind components supplied usable recent
features near Utrecht. A cached repeat took 2.28 seconds for acquisition/decoding
and 0.013 seconds for features, with no new data-file payload; metadata HTTP
requests still occurred. Network counters measure file payload, excluding small
API/index responses and transport overhead. The first station acquisition transferred
about 4.84 MB; a subsequent update transferred 0.16 MB. Missing intraday hourly
context remains explicit.

A real station pilot used cached January 2024 observations for training, April for
validation, and July for testing, with De Bilt (`0-20000-0-06260`) held out from
training targets and input neighbours. All eight hours were scored for all four
variables, without downloading data. Mean MAE across eight hours at that held-out
station was:

| Variable | Raw Aurora | Validation-weighted calibration |
|---|---:|---:|
| Temperature, °C | 0.716 | 0.752 |
| Mean-sea-level pressure, hPa | 1.112 | 0.654 |
| U wind component, m/s | 0.924 | 1.135 |
| V wind component, m/s | 0.782 | 0.869 |

The pilot is **not qualified for live use**: each split contains only one day,
station history is incomplete, prior forecast residual history is absent, and
validated observations were used retrospectively. Validation choices were not
retuned on the test outcome. Pressure improved here; temperature and wind did not.
Artifacts are in `outputs/refactor-station-pilot-20261024/`, including the resolved
case manifest, selected coefficients, input provenance and held-out evaluation.

An existing local Aurora forecast initialized 23 October 2026 at 12 UTC was
compared with measured radar for the eight hours after 20 UTC near Utrecht.
Hourly rainfall MAE over the evaluated 66×66 km interior was
1.971, 1.746, 2.399, 2.892, 2.930, 0.713, 0.299 and 0.273 mm.
Acquiring the missing radar frames transferred 18.25 MB. This is **Aurora-only
baseline verification**, not a trained-refinement result. The complete scores,
spatial metrics and hashes are in
`outputs/refactor-eight-hour-rainfall-20261024.json`. A multi-case trained radar
qualification study is still required; synthetic training tests establish only
that training, checkpoint selection and eight-hour verification execute correctly.
