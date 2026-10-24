# Local weather prediction with Aurora and KNMI

This project explores whether a personal computer can produce useful local weather
forecasts using Microsoft's **Aurora 1.5**, recent KNMI weather-station observations,
and rainfall radar. The goal is an eight-hour forecast for any locality in the
European Netherlands, computed on the local machine.

Aurora supplies the pretrained atmospheric dynamics. Separate, lightweight models
can adjust its temperature, pressure, and wind predictions using recent station
measurements and refine rainfall using radar. Date, time, and location provide
seasonal and daily context. Training happens separately from everyday prediction;
Aurora's weights stay frozen during forecasting. Radar motion also works without
a trained checkpoint: it moves the latest measured rain field using motion
estimated from two hours of observations, with Aurora filling unsupported areas.

**Current status:** local M3/MPS forecasting and the training/evaluation workflows
run successfully. Observation-based improvements remain experimental: the small
station pilot improved pressure but worsened temperature and wind, and trained
radar refinement still needs representative held-out evaluation. External models
are enabled explicitly; the default forecast uses Aurora alone.

Every live request reports requested components, actual per-variable sources,
initialization age, model lead hours, correction hashes, and observation cutoffs
to stderr. Newer input-cycle availability is checked before using an older
forecast cache; availability checks are reused for up to ten minutes.

The [Leusden case study](experiments/leusden-20261024/report.md) records a full
eight-hour comparison with a user-supplied Buienradar reference. A larger regional
crop improved temperature agreement, but rainfall remained substantially too low.
The revised station pilot also remains unqualified for routine prediction.
The [radar and adaptation experiment](experiments/radar-methods-20261024/report.md)
records the deterministic method, bounded training feasibility, and sparse
inference checks. Broader rainfall qualification remains pending.

## How it works and what data is used

```text
Fresh atmospheric initialization → local Aurora → eight-hour regional forecast
                                                      ↓
Recent KNMI station history + date/time/location → atmospheric adjustments
Recent measured radar + Aurora rainfall          → detailed rainfall refinement
                                                      ↓
                                Hourly forecast for the requested locality
```

| Data | Purpose | Acquisition and reuse |
|---|---|---|
| Pinned Aurora 1.5 weights and matching static fields | Pretrained model and land-surface context | Download once from Microsoft on Hugging Face; reuse from `checkpoints/base/` |
| ECMWF IFS step-zero fields | Fresh surface and upper-air initialization at two times six hours apart | Download required fields for new cycles; cache under `data/raw/ifs/` |
| ERA5 sea ice | Supplement an input missing from the public IFS product | Download through CDS when needed; enforce a seven-day maximum age |
| KNMI operational stations | Recent temperature, mean-sea-level pressure, and wind for local adjustments | Up to 72 hours of context; reuse nationwide hourly files plus recent 10-minute observations |
| KNMI measured RTCOR radar | Rainfall location, intensity, and evolution | Two hours of five-minute frames for live refinement; reuse raw and prepared caches |
| ERA5/CDS, optional ARCO pressure fields, and historical KNMI observations | Historical forecasts, training, and verification | Explicit historical acquisition; validated station observations are targets, not assumed live-available inputs |

Stations and radar cannot replace Aurora's three-dimensional atmospheric inputs.
The application downloads current-state initialization rather than another
provider's future forecast. Internet access is needed to refresh observations and
inputs; the forecasting and adjustment models execute locally.

Aurora retains its 0.25° grid, approximately 17 × 28 km around the Netherlands.
Radar refinement produces rainfall on a 1 km grid; that grid spacing does not
establish 1 km forecast accuracy. Temperature, pressure, and broad wind patterns
remain coarse forecasts with optional local corrections.

Public IFS fields are downloaded globally before regional cropping, so small local
files do not imply equally small transfers. ARCO is explicitly labelled as a
regridded ERA5 mirror. Hourly operational station files arrive overnight; the live
station window can contain intraday gaps, which remain explicit.

## Install

Install [Pixi](https://pixi.sh) and Git first. The tested machine is an Apple M3
MacBook Pro with 16 GB unified memory; allow approximately 25 GB free disk for
setup and initial experiments. Supported environment platforms are macOS ARM64
and Linux x86-64. Pixi supplies Python 3.12, dependencies, and the editable package.

```sh
git clone https://github.com/leonvgurp/aurora_finetuning.git
cd aurora_finetuning
pixi install --locked
pixi run check
pixi run smoke
```

For an existing checkout, start with `pixi install --locked`. Run the examples
below from the repository root. MPS is the default device; use `--device cpu`
explicitly on other machines. There is no silent CPU fallback.

## Configure credentials and runtime settings

**CDS:** obtain a [personal token](https://cds.climate.copernicus.eu/how-to-api)
and accept the terms for [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
and [pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels).
Create `~/.cdsapirc` containing:

```text
url: https://cds.climate.copernicus.eu/api
key: <CDS-PERSONAL-ACCESS-TOKEN>
```

Then restrict access:

```sh
chmod 600 ~/.cdsapirc
```

**KNMI:** obtain an [Open Data API key](https://developer.dataplatform.knmi.nl/register).
Set `KNMI_API_KEY`, or create `.secrets/knmi-api-key` containing only the key:

```sh
mkdir -p .secrets
chmod 700 .secrets
# Save the key in .secrets/knmi-api-key using your editor, then:
chmod 600 .secrets/knmi-api-key
```

The environment variable takes precedence. Secret files must not be symlinks;
`.env` files are not loaded. KNMI credentials are needed for radar/station access.
CDS credentials are needed for ERA5 and sea-ice setup. Neither belongs in Git or TOML.

Edit [configs/default.toml](configs/default.toml) for the region, horizon, device,
observation freshness, and resource limits. CLI arguments override TOML values.
Configuration and case-manifest paths resolve relative to their containing file;
CLI paths resolve against the working directory.

## Make a forecast

Run setup once, and repeat it when the sea-ice supplement expires:

```sh
pixi run weather-setup
pixi run weather Leusden --hours 8
```

Setup downloads the pinned model assets and validates live atmospheric inputs;
it does not train correction models. Verified inputs and forecasts are reused.
An alternative to locality lookup is explicit coordinates:

```sh
pixi run weather --latitude 52.09 --longitude 5.12 --hours 8
```

Locality names use a cached PDOK lookup. Stdout contains an hourly table with:

- Temperature in °C, mean-sea-level pressure in hPa, and 10 m wind speed in m/s.
- Rainfall in mm over the preceding clock hour.
- Europe/Amsterdam timestamps with UTC offsets, including daylight-saving changes.

Diagnostics go to stderr. Requested horizons are 1–24 hours; eight hours is the
training/evaluation target for external models. The elapsed part of the first
clock hour may use measured radar rainfall and is identified in the result metadata.

Choose a rainfall method explicitly; Aurora remains the default:

```sh
pixi run weather Leusden --hours 8 --rain-method aurora
pixi run weather Leusden --hours 8 --rain-method radar-motion
```

`radar-motion` needs KNMI access but no trained checkpoint. It estimates pySTEPS
optical flow once over the complete measured radar domain, then streams 96
five-minute predictions into hourly totals. Missing pixels and trajectories from
outside the domain retain Aurora, with coverage and fallback reasons in the JSON
report. Dry valid observations use zero motion. Motion cannot reliably predict
storm formation or guarantee useful skill eight hours ahead. Radar updates reuse
Aurora forecasts; motion and hourly radar grids are shared between localities.

After training and evaluating compatible external models, enable either or both:

```sh
pixi run weather Leusden --hours 8 \
  --station-calibrator outputs/stations/model.json \
  --radar-adapter outputs/radar/best.pt
```

These paths must contain actual trained artifacts; the example files are not
bundled pretrained corrections. Forecasting never retrains them. Missing fresh
nearby stations cause a per-variable Aurora fallback; corrupt artifacts,
incompatible schemas, and unavailable required CNN inputs produce explicit errors.
An omitted `--rain-method` with `--radar-adapter` preserves CNN behavior. Explicit
`--rain-method cnn` requires that adapter. Contradictory options are rejected.
Station calibration is independent of the rainfall method.

## Train and validate local adjustments

Historical data collection and training are separate operations. Start with small
cases and cached locally generated Aurora forecasts. The case templates contain
placeholder paths: supply the data before running the following commands. Keep
training, validation, and test windows separate and include varied weather/seasons.

### Stations

Populate [station-cases.example.json](configs/station-cases.example.json) with
issuance times, Aurora forecasts, past KNMI files, future target files, and prior
forecast archives. Historical residuals use forecasts available by the observation
time. Reserve entire station identifiers to test transfer between stations.

```sh
pixi run prepare-stations --cases configs/station-cases.example.json \
  --heldout-stations 0-20000-0-06260 --output data/prepared/stations
pixi run train-stations --prepared data/prepared/stations \
  --output outputs/stations/model.json
pixi run evaluate-stations --cases configs/station-cases.example.json \
  --station-calibrator outputs/stations/model.json \
  --heldout-stations 0-20000-0-06260
```

Ridge regression models use forecast values, lead time, cyclic date/time features,
location, recent trends, and residual summaries. Up to four suitable stations within
100 km supply local context. Regularization, seasonal features, and per-hour
correction weights are selected on validation data, not test observations.

Station checkpoints now use `station-ridge-v2`, adding the latest observed value
for cold starts without eligible archived forecast residuals. Re-run
`prepare-stations` and `train-stations` for old artifacts; v1 checkpoints are
rejected rather than interpreted with the new feature layout. Training still
selects regularization and blending using historical validation data only.

### Rainfall radar

Populate [radar-cases.example.json](configs/radar-cases.example.json) with baseline
Aurora forecasts and prepared measured radar. Each case needs **24 past + 96 future
five-minute frames**. Preserve `RAD_NL25_RAC_RT_YYYYMMDDhhmm.h5` filenames. Historical
RTCOR archive extraction remains a manual prerequisite.

Checkpoint-free motion calibration/evaluation uses the same chronological manifest.
Prepare full-domain observations with `prepare-radar-native --full-domain` (keep
the default cropped preparation for CNN training). The selected product schema
must have exact five-minute UTC boundaries and supported measured amounts; some
older KNMI archive files are deliberately rejected rather than reinterpreted.

```sh
pixi run calibrate-rain --cases configs/radar-cases.example.json \
  --output outputs/rain/blend.json
pixi run evaluate-rain --cases configs/radar-cases.example.json \
  --report outputs/rain/evaluation.json
pixi run evaluate-rain --cases configs/radar-cases.example.json \
  --rain-calibration outputs/rain/blend.json \
  --report outputs/rain/blended-evaluation.json
pixi run weather Leusden --hours 8 --rain-method radar-blend \
  --rain-calibration outputs/rain/blend.json
```

Evaluation compares Aurora, radar persistence, FFT motion, pySTEPS motion, and the
combined result without loading a CNN checkpoint. Calibration selects one weight
per lead hour on validation only, requiring lower MAE without worse CSI at
0.1 mm/hour. Artifacts carry checksums, case identities, and numerical/domain
compatibility. A weight of zero retains Aurora's future rain; all methods retain
the same measured portion of the first clock hour. No qualified blending artifact
is bundled. The bounded pilot's new transfers were 146.6 MB of its 250 MB budget;
incompatible historical files prevented independent train/validation/test splits.

```sh
pixi run prepare-radar-native --frames /absolute/path/RAD_NL25_RAC_RT_*.h5 \
  --latitude 52.09 --longitude 5.12 --output data/prepared/radar/case.nc
pixi run train --cases configs/radar-cases.example.json --output outputs/radar
pixi run evaluate-radar-adapter --cases configs/radar-cases.example.json \
  --radar-adapter outputs/radar/best.pt
```

`train` now trains the external radar model; `train-radar` remains an alias.
The default 20 steps are a pipeline pilot. Training compares seasonal ablations
and a radar-only control. Verification compares Aurora, learned refinement,
persistence, and motion extrapolation for every hour 1–8, cumulative rainfall,
wet events, and spatial skill. Old surface-head and six-hour radar adapters are
incompatible. Use fresh output paths; valuable results are not silently overwritten.

## Performance, outputs, and checks

Daily forecasts now request only needed hourly outputs plus six-hour bridge
states (`sparse_forecasts = true` in the default configuration). Historical engine
calls remain dense unless a lead schedule is explicitly supplied. Float32 MPS
comparisons across multiple windows were exactly equal to dense inference.
Forecast cache schema 3 records timestamp coverage and cannot reuse a cache
missing a requested lead. Startup still dominates short uncached requests.

The optional internal adaptation command performs one isolated rank-eight LoRA
training step on cached ERA5 inputs, with a ten-minute timeout, the original
48×64 grid, process-memory checks, and accelerator allocation checks. It discards
updated weights and never affects forecasting:

```sh
pixi run adapter-feasibility --initialization 2024-01-01T06:00:00 \
  --output outputs/adapter-feasibility-new
```

This experiment passed on M3/MPS; it demonstrates that one step fits, not useful
adaptation or feasibility on a larger domain. Pixi supplies the compiler/OpenMP
dependencies needed to build pySTEPS; no separate Python environment is needed.

A measured eight-hour historical run on M3/MPS with cached inputs took **38.9 s
with a cold engine**, **2.12 s with reused weights**, and **0.055 s for a cached
location query**. These are different execution modes, not end-to-end timings
including fresh downloads. Separate CLI processes reload weights on cache misses;
Python callers can reuse `ForecastEngine` in one session. Float32 remains the
default because the tested MPS float16 path was slower.

```sh
pixi run benchmark --initialization 2024-01-01T06:00:00 \
  --output outputs/benchmark-new
pixi run check
pixi run test-integration
```

The benchmark requires existing historical inputs and weights; it does not fetch
missing data. `check` runs formatting checks, Ruff, mypy, and offline pytest tests.
Integration checks are opt-in. `pixi run format` applies formatting. CI installs
the locked Pixi environment and runs the same quality checks.

| Artifact | Format and location |
|---|---|
| Raw and prepared observations/inputs | NetCDF, GRIB, or HDF5 under `data/` |
| Regional Aurora forecasts | NetCDF under `outputs/weather/` |
| Trained station/radar models | JSON coefficients / PyTorch checkpoint at the requested output path |
| Results, metrics, input hashes, configuration, and timing reports | JSON under the requested directory and `outputs/timings/` |

Raw Aurora values remain available alongside corrected location results. Forecast
caches are independent of observation refreshes. `max_download_gb` and
`max_storage_gb` bound KNMI/IFS acquisition in GiB; model setup and explicit
CDS/ARCO operations have separate acquisition characteristics. No data is
automatically deleted. The 12 GiB process-RSS guard is not a total unified-memory cap.
Generated data, weights, environments, outputs, and secrets are Git-ignored.

See [design and measured evidence](docs/design.md) for source references, data
schemas, evaluation details, pilot results, and limitations. Local execution is
demonstrated; reliable skill across Dutch localities and eight-hour radar
improvements remain questions for broader held-out and prospective evaluation.
