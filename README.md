# Local weather experiments with Aurora

I read about Microsoft's Aurora model, knew public weather data was available,
and wanted to find out whether I could run weather predictions locally on my Mac
and how well they would work. This is a first draft of that experiment: use a
globally pretrained model on a region surrounding the Netherlands, compare its
predictions with KNMI observations, and try limited local fine-tuning.

The project provides historical experiments and a **one-command live forecast**
with hourly temperature and precipitation amounts. Inference runs locally;
useful forecast skill and operational reliability remain open questions.

## Requirements

- [Pixi](https://pixi.sh) and Git; Python 3.12 and all libraries come from Pixi.
- **Reproduction target:** Apple Silicon MacBook Pro with 16 GB unified memory
  and the PyTorch MPS GPU backend. The initial full-model run succeeded on the
  author's M3 Mac. This is the tested baseline, not a measured absolute minimum;
  smaller machines have not been validated.
- Allow roughly **25 GB free disk** for the environment, download caches, 5 GB
  checkpoint and initial data. This is a starting budget; larger experiments
  need more. Internet access is required for downloads; CDS queues can be slow.
- The lockfile also supports x86-64 Linux. Use `--device cpu` for CPU experiments;
  GPU inference performance on Linux/CUDA has not been established here.

## Setup

From the repository root:

```sh
pixi install --locked
pixi run check                 # Offline formatting, lint, types and tests
pixi run smoke                 # Imports, configured region and credential presence
pixi run smoke-architecture    # Tiny random-weight CPU forward/backward
```

Pixi installs the local package in editable mode. No pip installation, Homebrew
NetCDF installation or separate virtual environment is needed.

### Credentials: one-time manual steps

**Copernicus CDS**

1. [Register/sign in to CDS](https://cds.climate.copernicus.eu/), then open the
   [API setup page](https://cds.climate.copernicus.eu/how-to-api) while signed in
   and copy your personal access token.
2. Run `nano ~/.cdsapirc`, save the following with your actual token, then run
   `chmod 600 ~/.cdsapirc`:

   ```text
   url: https://cds.climate.copernicus.eu/api
   key: <PERSONAL-ACCESS-TOKEN>
   ```

3. While signed in, manually accept the terms on the Download tabs for both
   [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
   and [ERA5 pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download).
   Authentication alone does not establish download permission. `cdsapi` is
   already included through Pixi.

**KNMI Open Data**

1. [Register](https://developer.dataplatform.knmi.nl/register), sign in, open
   **API Catalogue → Open Data API → Request an API key**, and save the full key.
   A separate EDR key is not needed for this project.
2. Run `mkdir -p .secrets` and `chmod 700 .secrets`. Use your editor to save
   **only the full key** in `.secrets/knmi-api-key`, without quotes or a variable
   name. Run `chmod 600 .secrets/knmi-api-key`.
3. Alternatively supply `KNMI_API_KEY` in the process environment; it takes
   precedence over the file. `.env` files are not loaded automatically.

The secret file is Git-ignored and must not be a symlink. It is resolved under
`root` from the runtime configuration. Never commit keys or include them in
logs. The [KNMI documentation](https://developer.dataplatform.knmi.nl/open-data-api)
also offers a shared, expiring anonymous key for small trials; a registered key
has a dedicated quota. Aurora assets, the ARCO ERA5 mirror and public IFS files
need no personal token.

```sh
pixi run providers             # Check CDS authentication and KNMI file listing
```

## One-command weather prediction

After `pixi install --locked`, configure CDS credentials and accept the dataset
terms above, then run the one-time setup:

```sh
pixi run weather-setup
pixi run weather Utrecht --hours 8
pixi run weather --latitude 52.09 --longitude 5.12 --hours 24
```

Run these commands from this checkout. Setup verifies/downloads pinned model
assets, selects a fresh public IFS input pair and obtains a regional ERA5 sea-ice
supplement. CDS credentials are needed for that supplement; KNMI credentials are
not needed for weather predictions. Setup validates input fields, units and
pressure levels; a successful prediction separately establishes model execution.
Setup defaults to a supplement five days before today to allow for ERA5 delay;
use `pixi run weather-setup --sea-ice-date YYYY-MM-DD` to specify the date used
when a new supplement is needed.
Both input states must be within seven days of the supplement.

`weather` accepts a Dutch place name or a coordinate pair within the supported
Netherlands envelope. Names resolve to locality centroids through
[PDOK](https://www.pdok.nl/pdok-locatieserver), with cached lookups. Ambiguous names
fail with candidates; refine the name or use coordinates. Locality lookup data
is cached under `data/cache/locations/`; remove a lookup file to refresh it.

The default is the next eight hourly predictions; `--hours` accepts 1–24. Optional
`--start 2026-10-24T08:00:00+02:00` requests a future starting time. Results begin
at the first whole UTC hour strictly after the requested start. Live starts must
not be in the past, and initialization-to-final-output must fit the model's
48-hour internal limit. All rows include local dates and UTC offsets, including
across daylight-saving transitions.

Successful stdout contains only a Markdown table:

| Hour ending (Europe/Amsterdam) | Temperature (°C) | Rainfall (mm/hour) |
|---|---:|---:|
| 2026-10-23 22:00+02:00 | 11.72 | 0.155 |

Temperature is the value at the hour ending; rainfall is the amount during the
preceding hour, not its probability. The grid resolution is about 25 km, not
street-level. Progress, resolved locality, initialization age and experimental
warnings go to stderr; redirect stdout to save only the table. Errors return a
nonzero exit status and no partial table.

Prediction automatically selects the newest published, complete IFS input pair
within `max_input_age_hours` (default 12). It reuses verified inputs and forecasts
under `outputs/weather/`, including across localities and shorter requests.
Uncached input downloads and model loading take longer than cached queries.
Internet access is still needed to check current input availability. If the
sea-ice supplement expires, rerun `pixi run weather-setup`; prediction never
queues a CDS renewal or silently uses stale/fabricated inputs. Timing/provenance
reports are saved under `outputs/timings/`. `--adapter PATH` explicitly selects
trained heads; the default uses the pretrained model.

A one-hour live Utrecht prediction completed on MPS on 23 October 2026 in
47.4 seconds after setup, including model loading and eight internal lead hours
to bridge the age of the available IFS cycle. This establishes execution for one
case, not forecast accuracy or guaranteed provider availability.

## Research workflow

Each execution step has a Pixi task. Run the steps below in order and stop on
failure. Tasks accept additional CLI arguments and do not automatically trigger
other downloads or model runs. Account registration, secrets and accepting
licenses are manual prerequisites.

| Step | What happens | Data used |
| --- | --- | --- |
| Download | Cache pinned Aurora 1.5 weights and matching static fields | Public Microsoft assets on Hugging Face |
| Retrieve | Get regional atmospheric inputs and independent scoring references | ERA5 via CDS; KNMI validated hourly station NetCDF files |
| Prepare | Verify checksums/units, align cells and select two states six hours apart | Surface fields, five upper-air variables on 13 pressure levels, static fields |
| Forecast | Run pretrained Aurora on the regional grid | Prepared inputs; hourly temperature and precipitation outputs |
| Evaluate | Score MAE, RMSE and bias by lead hour; compare simple baselines | KNMI temperature/rain; persistence and always-dry baselines |
| Fine-tune | Update temperature/rain output heads, select on validation loss | ERA5 gridded targets plus optional KNMI station supervision |
| Query | Extract hourly values at a Netherlands latitude/longitude | Saved forecast, or a new explicit model run |

### First historical run

This example uses 1 January 2024, initialized at 06:00 UTC. Full-day ERA5 retrieval
covers both input states and the required forecast/target hours.

```sh
pixi run download-model
pixi run pull-era5 --start 2024-01-01
pixi run pull-knmi --begin hourly-observations-validated-20240101-00.nc --end hourly-observations-validated-20240101-23.nc --max-files 24
pixi run prepare --initialization 2024-01-01T06:00:00
pixi run smoke-model --initialization 2024-01-01T06:00:00
pixi run hindcast --initialization 2024-01-01T06:00:00 --hours 8 --output outputs/base-20240101.nc
pixi run evaluate --prediction outputs/base-20240101.nc --stations data/raw/knmi/hourly-in-situ-meteorological-observations-validated/1.0/*.nc
pixi run forecast --latitude 52.0907 --longitude 5.1214 --hours 8 --cached-forecast outputs/base-20240101.nc --start 2024-01-01T06:00:00+00:00
```

If CDS pressure retrieval remains queued, an explicit alternative is:

```sh
pixi run pull-era5 --start 2024-01-01 --surface-only
pixi run pull-arco-pressure --initialization 2024-01-01T06:00:00
```

This imports the two pressure input states from
[Google's ARCO ERA5 mirror](https://github.com/google-research/arco-era5).
It is regridded ERA5, not guaranteed bit-identical to CDS; manifests record the
source. Storage chunks are global, although only regional arrays are retained.
Avoid running duplicate CDS requests while one is already queued.

### Fine-tuning and held-out evaluation

Edit `configs/cases.example.json`: train/validation/test contain chronological
UTC initialization times with more than 30 hours between splits. For **each**
case, first retrieve ERA5 from initialization minus six hours through plus six
hours and matching KNMI files. The supplied four-date example requires January,
April, July and September inputs; the first-run commands above fetch only January.
If using ARCO, run `pull-arco-pressure` separately for each initialization.

```sh
pixi run smoke-backward --initialization 2024-01-01T06:00:00
pixi run train --cases configs/cases.example.json --stations data/raw/knmi/hourly-in-situ-meteorological-observations-validated/1.0/*.nc --steps 1 --output outputs/train-smoke
# After inspecting timings and losses, run the configured 20-step pilot:
pixi run train --cases configs/cases.example.json --stations data/raw/knmi/hourly-in-situ-meteorological-observations-validated/1.0/*.nc --output outputs/train-pilot
pixi run hindcast --initialization 2024-09-01T06:00:00 --hours 8 --output outputs/base-test.nc
pixi run hindcast --initialization 2024-09-01T06:00:00 --adapter outputs/train-pilot/best.pt --hours 8 --output outputs/adapted-test.nc
```

Run `pixi run evaluate` on both test forecasts using identical KNMI files.
`best.pt` is selected using validation loss; `training.json` records the losses
and inputs. Test cases are not consumed by training. The backbone stays frozen;
this adapts output heads, not regional atmospheric dynamics. The four example
cases and 20 steps are a smoke-sized pilot, not a sufficient scientific dataset.

## Locality, time and design choices

Edit [configs/default.toml](configs/default.toml). Explicit CLI arguments override
configuration; configuration overrides defaults. Paths in configuration resolve
relative to that file; explicit CLI paths resolve from the working directory.

| Setting | Default / choice | Why it matters |
| --- | --- | --- |
| `north`, `south`, `west`, `east` | 56.5, 47.5, −2.5, 12.5 degrees | Rectangular context surrounding NL; not a radius around each queried point |
| `resolution` | 0.25° | Checkpoint's native scale, roughly 28 km north–south; not street-level weather |
| Aligned grid | 48 × 64 cells; 46.25–58°N, −3–12.75°E | Bounds expand to satisfy model dimensions, which must be multiples of 16 |
| `hours` | 8; accepts 1–24 | Short experimental horizon, one-hour outputs; longer horizons need separate validation |
| Input times | Initialization −6h and initialization | Includes vertical atmosphere and surrounding weather, not just station history |
| `device` | `mps` | Uses the Apple GPU; a smaller region reduces computation but not the 1.26B-parameter model's weight memory |
| `max_steps`, `learning_rate`, `seed` | 20, 0.0001, 42 | Bounded head adaptation and best-effort reproducibility |
| Loss scales | 5°C temperature, 1 mm precipitation | Fixed scaling; rainfall loss uses `log1p`; optional KNMI loss has equal weight to gridded loss |
| `max_memory_gb` | 12 | Post-stage process RSS guard, not an OS limit or total GPU/unified-memory cap |
| `max_input_age_hours` | 12 | Live input age is measured against now; research queries use their start |

Coarser 0.5°/1° grids are configurable experiments, not validated equivalents of
the native checkpoint. Start with 0.25° and reduce the region/horizon before
making accuracy claims. Cropped inference changes a global model's context;
scoped regional attention/longitude adaptations are experimental. The code and
checkpoint revisions are pinned in the environment/configuration.

Initializations are exact UTC hours without a timezone suffix. Query `--start`
requires a timezone; results show Amsterdam offsets. Temperature is bilinearly
interpolated; rain uses the nearest grid cell and reports **mm over the preceding
hour**, not probability. Location queries accept the NL bounding envelope
50.7–53.7°N, 3.2–7.3°E, including some sea/border cells. The research `forecast`
command requires coordinates; `weather` also accepts place names.

## Outputs and remaining limits

- `data/raw/`: inputs; `checkpoints/`: weights/static fields; `outputs/`: forecasts,
  adapters and metrics. All are Git-ignored. `outputs/timings/` records wall time,
  RSS, GPU allocation snapshots, configuration and hashes. Use new result paths;
  verified input caches are reusable.
- Forecast NetCDF variables: `temperature_c`, `precipitation_mm`. Location queries
  through `forecast` return JSON; `weather` prints a table. Python interface:
  `aurora_finetuning.forecast.forecast(...)`, run
  through `pixi run python`.
- Three-day history: `pixi run history --stations FILES --station WIGOS_ID
  --today YYYY-MM-DD`. Download all three days first. Returns hourly-sample
  temperature min/max/mean, sunshine/rain duration and coverage; incomplete totals
  stay null. This is separate from forecasting.
- **Today:** `weather` orchestrates live IFS inputs and age-checked sea ice;
  ERA5 alone is delayed. The explicit research commands remain available:
  `pull-ifs --initialization UTC_CYCLE --audit-only`, `pull-sea-ice`, then
  `pull-ifs --sea-ice FILE`. Downloads can be global before cropping.
- **Optional radar:** `prepare-radar` needs individual five-minute HDF5 frames
  plus authoritative matching geolocation; `evaluate --radar FILE` scores hourly
  aggregates. Archive extraction and radar training are not implemented.
- KNMI temperature is at 1.5 m versus model 2 m. Documented rain traces map to zero;
  other negative observations remain missing. Model rain is floored at zero with
  raw diagnostics retained; nonfinite outputs fail. ERA5/IFS assimilate
  observations, so scoring against KNMI is not fully independent.

## Evidence and development

An initial eight-hour historical run completed on MPS in **128 s**, including
model load/transfer; forecast computation took **7.87 s**. GPU driver allocation
snapshots reached **5.23 GiB**, not a measured total-memory peak. Across this one
case, temperature MAE was **0.982°C** versus **0.527°C** for persistence; rain MAE
was **0.292 mm** versus **0.286 mm** for always dry. These results establish
execution, not useful skill. Fine-tuned held-out performance and sustained live
reliability remain to be demonstrated. Local detailed reports, when generated,
are under
`outputs/performance/`; they are not included in a fresh checkout.

```sh
pixi run format              # Apply formatting
pixi run check               # Format check, Ruff, strict mypy, offline pytest
pixi run test-integration    # Separate tiny-model integration check
pixi run aurora-finetuning --help
pixi run train --help        # Every task exposes its command's options
```

CI runs the same locked checks and tiny smoke on macOS/Linux; default tests need
no network or model weights. Respect [Aurora's license](https://github.com/microsoft/aurora)
and source-data terms before redistributing assets.
