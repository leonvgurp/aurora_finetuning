# Local weather experiments with Aurora

I read about Microsoft's Aurora model, knew public weather data was available,
and wanted to find out whether I could run weather predictions locally on my Mac
and how well they would work. This is a first draft of that experiment: use a
globally pretrained model on a region surrounding the Netherlands, compare its
predictions with KNMI observations, and try limited local fine-tuning.

The project provides historical experiments and a **one-command live forecast**
with hourly temperature and precipitation amounts. Inference runs locally;
useful forecast skill and operational reliability remain open questions.

## Install

Requires [Pixi](https://pixi.sh) and Git. Tested on Apple Silicon, 16 GB memory,
~25 GB free disk. Pixi supplies Python 3.12 and the editable package;
platforms: macOS ARM64 / Linux x86-64.

```sh
pixi install --locked
pixi run check             # Formatting, lint, types and offline tests
pixi run smoke             # Imports, configuration and credential presence
```

Run from this checkout. Default device: `mps`; use `--device cpu` elsewhere.

- **CDS:** get a [token](https://cds.climate.copernicus.eu/how-to-api), save the
  following in `~/.cdsapirc` (mode `600`), and accept the dataset terms for
  [ERA5 single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
  and [pressure levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=download).
- **KNMI:** [request an Open Data key](https://developer.dataplatform.knmi.nl/register).
  Set `KNMI_API_KEY` or save only the key in `.secrets/knmi-api-key` (file `600`,
  directory `700`, no symlink). Environment overrides file; `.env` is not loaded.
  Needed for stations/radar.

```text
url: https://cds.climate.copernicus.eu/api
key: <CDS-PERSONAL-ACCESS-TOKEN>
```

## Resources

Weights/data are downloaded on demand and Git-ignored; verified caches are reused.

| Resource | Use and storage |
|---|---|
| Pinned Aurora weights + static fields | Local inference; `checkpoints/base/` (~5 GB weights) |
| Public IFS / CDS ERA5 | Live / historical atmospheric inputs; `data/raw/ifs/`, `data/raw/era5/` |
| ERA5 sea ice | Missing IFS field supplement; `data/raw/sea-ice/`; needs CDS |
| KNMI stations / measured RTCOR radar | Evaluation/training; `data/raw/knmi/`, `data/prepared/radar/` |
| PDOK locality lookup | Place names to coordinates; `data/cache/locations/` |
| Generated results | Forecasts/adapters/metrics in `outputs/`; configuration, hashes and timings in `outputs/timings/` |

Use fresh output paths. Keep secrets private and respect
[Aurora's license](https://github.com/microsoft/aurora) and source-data terms.

## Workflow

**Predict:** configure CDS, then run setup once; repeat it when sea ice expires.

```sh
pixi run weather-setup
pixi run weather Leusden --hours 8
# Coordinates instead: --latitude 52.09 --longitude 5.12
```

Stdout is an hourly table: Amsterdam time with UTC offset, °C and rainfall
**mm over the preceding hour**. Logs go to stderr. Default: 8 hours; range: 1–24.
Optional `--start` requires a future timestamp with timezone.

**Historical/head training:** populate [cases.example.json](configs/cases.example.json).
Retrieve ERA5 from initialization −6h through +6h per case, then prepare,
hindcast and evaluate. `train --stations FILES` optionally adds KNMI supervision.
Compare base/adapted hindcasts on held-out dates.

**Radar training:** populate [radar-cases.example.json](configs/radar-cases.example.json)
with baseline Aurora forecasts and 24 past + 72 subsequent five-minute RTCOR
frames per case. Preserve
`RAD_NL25_RAC_RT_YYYYMMDDhhmm.h5` names; get measurements from
[KNMI](https://dataplatform.knmi.nl/dataset/nl-rdr-data-rtcor-5m-1-0) or its
[archive](https://dataplatform.knmi.nl/dataset/nl-rdr-data-rtcor-5m-tar-1-0)
(manual extraction). KNMI forecasts/climatological corrections are rejected.

```sh
pixi run prepare-radar-native --frames path/to/case/*.h5 --latitude 52.1 --longitude 5.4 --size 256 --output data/prepared/radar/train.nc
pixi run train-radar --cases configs/radar-cases.example.json --steps 1 --output outputs/radar-pilot
pixi run evaluate-radar-adapter --cases configs/radar-cases.example.json --radar-adapter outputs/radar-pilot/best.pt
pixi run weather Leusden --hours 8 --radar-adapter outputs/radar-pilot/best.pt
```

Example paths need real data; one step is a smoke test. Keep radar
`best.pt` and `best.json` together. If baselines used Aurora head weights, pass
the same `--adapter PATH` throughout. **A representative real-storm radar
checkpoint is not yet available.**

## Commands: order of events

| Command | Sequence |
|---|---|
| `weather-setup` | Verify/download weights → select IFS pair → obtain/reuse sea ice → download/validate inputs |
| `weather` | Resolve locality → select/validate inputs → reuse/run regional forecast → extract location → optionally apply radar adapter → print table |
| `download-model`, `pull-era5`, `pull-knmi` | Cache requested assets/data; no model run |
| `prepare` → `hindcast` → `evaluate` | Align/check inputs → run Aurora/write NetCDF → score observations and baselines |
| `train` | Freeze backbone → fit temperature/rain heads → select validation-best weights |
| `prepare-radar-native` → `train-radar` | Validate/project/crop measurements → train rain adapter using past radar and later measured targets |
| `evaluate-radar-adapter` | Score held-out cases against Aurora and radar persistence |
| `forecast`, `history` | Location forecast as JSON, or summaries of downloaded station observations |

Arguments: `pixi run COMMAND --help`. Other helpers: `providers`, `smoke-*`,
`pull-ifs`, `pull-sea-ice`, `pull-arco-pressure` (ERA5 mirror), `prepare-radar`
(coarse-grid scoring). Research commands require explicit inputs.

## Technical details

- **Config:** [configs/default.toml](configs/default.toml); CLI > TOML > defaults.
  Config/manifest paths resolve relative to their file; CLI paths to the working
  directory. Forecast NetCDF contains `temperature_c` and `precipitation_mm`.
- **Aurora:** experimental global-model crop around NL, 0.25° spacing (~17 × 28 km
  locally), aligned 48 × 64 grid; two input states six hours apart, static fields
  and 13 pressure levels. Temperature is interpolated; base rain uses nearest cell.
- **Freshness:** IFS ≤12h old; sea ice ≤7 days old for both inputs (setup requests
  five days earlier); latest radar ≤20 minutes old. Missing/stale inputs fail.
  Live checks need internet access.
- **Radar:** frozen Aurora + trainable rain adapter; native 1 km history/masks →
  five-minute amounts → hourly sums. Elapsed first-hour rain is measured; beyond
  six hours from the radar cutoff, rows retain Aurora rain.
- **Training:** seed 42, 20 pilot steps, validation-selected weights, >30h split
  gaps, masked labels, wet-weighted log-rain loss. Radar scores sample central
  patches; historical publication latency is not reconstructed.
- **Limits:** local MPS inference and radar forward/backward work; useful rainfall
  skill remains unproven. One historical case did not beat persistence/dry
  baselines. Grid spacing does not establish accuracy; the 12 GB RSS guard is not
  a total GPU-memory limit.

Development: `pixi run format` applies Ruff formatting; `pixi run check` runs
non-mutating checks; `pixi run test-integration` runs the separate tiny Aurora
forward/backward test. CI uses the locked Pixi environment and the same checks.
