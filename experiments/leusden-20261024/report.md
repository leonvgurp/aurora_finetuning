# Leusden, 24 October 2026: eight-hour development comparison

Buienradar is the fixed reference **prediction** for this development case. Its issue time and rainfall interval convention were not supplied. The original pasted snapshot is preserved in [original-snapshot.md](original-snapshot.md); [buienradar.json](buienradar.json) assumes hour-ending rainfall. All times below are Europe/Amsterdam (UTC+02:00). No target rows entered training or model inputs.

## Results

| Candidate | Temperature MAE °C | Temperature bias °C | Hourly rainfall MAE mm | Eight-hour rainfall mm |
|---|---:|---:|---:|---:|
| Original saved forecast | 1.573 | -1.573 | 0.929 | 0.228 |
| Exact replay | 1.573 | -1.573 | 0.929 | 0.228 |
| Strict station fallback | 1.573 | -1.573 | 0.929 | 0.228 |
| Station v1 pilot (retrospective) | 1.618 | -1.608 | 0.929 | 0.228 |
| Larger crop | 1.108 | -0.915 | 0.949 | 0.025 |
| Station v2, original crop (retrospective) | 1.267 | -1.207 | 0.929 | 0.228 |
| Station v2, larger crop (retrospective) | 1.117 | -0.697 | 0.949 | 0.025 |
| Buienradar target | — | — | — | 7.600 |

The larger crop reduces temperature disagreement by 29.6%, but worsens rainfall. Adding the latest observed value improves the original-crop station pilot relative to raw Aurora on this case, but it does not beat the larger raw crop. **No candidate fixes rainfall; the larger crop is the selected temperature development profile, while no overall winner is promoted to the default configuration.**

## All eight hourly predictions

Temperature (°C); the exact replay and strict fallback equal the original forecast.

| Hour ending | Buienradar | Original Aurora | Larger Aurora | Station v1 original | Station v2 original | Station v2 larger |
|---|---:|---:|---:|---:|---:|---:|
| 16:00 | 12.00 | 7.97 | 9.11 | 7.65 | 8.27 | 9.09 |
| 17:00 | 11.00 | 8.18 | 9.10 | 7.94 | 8.53 | 9.20 |
| 18:00 | 11.00 | 8.47 | 9.12 | 8.31 | 8.85 | 9.32 |
| 19:00 | 10.00 | 8.74 | 9.16 | 8.65 | 9.16 | 9.46 |
| 20:00 | 10.00 | 8.96 | 9.42 | 8.95 | 9.31 | 9.67 |
| 21:00 | 9.00 | 8.67 | 9.26 | 8.79 | 9.00 | 9.51 |
| 22:00 | 9.00 | 8.67 | 9.25 | 8.81 | 9.05 | 9.55 |
| 23:00 | 9.00 | 8.76 | 9.26 | 9.04 | 9.19 | 9.62 |

Rainfall (mm in the preceding hour). Station calibration does not change rainfall.

| Hour ending | Buienradar | Original Aurora | Larger Aurora | Original cumulative difference | Larger cumulative difference |
|---|---:|---:|---:|---:|---:|
| 16:00 | 0.000 | 0.005 | 0.002 | +0.005 | +0.002 |
| 17:00 | 0.000 | 0.010 | 0.002 | +0.015 | +0.004 |
| 18:00 | 0.000 | 0.017 | 0.003 | +0.032 | +0.007 |
| 19:00 | 1.200 | 0.025 | 0.004 | -1.143 | -1.189 |
| 20:00 | 1.500 | 0.035 | 0.005 | -2.608 | -2.684 |
| 21:00 | 0.500 | 0.040 | 0.003 | -3.068 | -3.181 |
| 22:00 | 1.400 | 0.044 | 0.003 | -4.424 | -4.578 |
| 23:00 | 3.000 | 0.052 | 0.003 | -7.372 | -7.575 |

## Method and diagnosis

- Frozen Aurora 1.5, pinned checkpoint, float32, MPS. Initial states: 00:00 and 06:00 UTC. Requested window ends 16:00–23:00 local, corresponding to model lead hours 8–15. The replay exactly matches the original saved values.
- Audited Kelvin/Celsius, pressure, wind and precipitation conversions; chronological input order; actual upstream output times; pressure levels; descending latitude and increasing longitude; matching static cells; and prescribed insolation. No numerical conversion or timestamp defect was established.
- Original actual grid: 48×64 (46.25–58°N, −3–12.75°E). Larger actual grid: 64×96 (43.75–59.5°N, −6.5–17.25°E). Larger inputs were decoded from retained global step-zero GRIB, with a separately acquired matching sea-ice analysis dated 18 October. No future forecast fields were downloaded.
- Tiny index requests confirmed that the 12 UTC input pair was still unpublished at the recorded check. The 06 UTC pair was the newest available. The cache-selection bug is fixed, but did not cause this particular run’s old initialization.
- Near-real-time KNMI observations at 13:30 UTC measured 11.9°C at De Bilt, 11.4°C at Deelen and 11.9°C at Herwijnen. This supports concern about the cold atmospheric forecast; station values are not exact Leusden targets.
- Newly downloaded station observations were acquired after the original issuance. Strict reconstruction correctly falls back. Retrospective pilots remove the acquisition-time gate explicitly, while retaining the observation-time cutoff. Publication latency remains unknown; these pilots are not prospective evidence.
- Station v2 adds the latest observed value because cold starts otherwise have no residual-level feature when eligible forecast archives are absent. Models were refitted on cached January training and April validation cases; regularization, seasonal features and lead weights use validation only. De Bilt was excluded from preparation. Old v1 artifacts are preserved and rejected by the new schema.
- No qualified radar checkpoint or complete chronological radar train/validation/test dataset is available. Rainfall refinement therefore remains a declared fallback to Aurora. Training on the supplied eight future rainfall values would invalidate this experiment. No radar improvement is claimed.

## Separate historical station check

The July held-out case is a reused pilot check, not a new untouched benchmark. One case cannot establish generalization or confidence intervals. Mean of hourly MAE over hours 1–8:

| Variable | Raw Aurora | Station v1 | Station v2 | Persistence |
|---|---:|---:|---:|---:|
| temperature_c | 0.716 | 0.751 | 0.809 | 2.063 |
| pressure_hpa | 1.112 | 0.654 | 0.762 | 1.212 |
| wind_u_ms | 0.924 | 1.135 | 0.987 | 0.733 |
| wind_v_ms | 0.782 | 0.869 | 0.570 | 0.638 |

V2 improves one wind component but worsens temperature versus raw Aurora on this separate case. The station models remain disabled by default. The larger crop has not been validated across independent weather cases, so its improvement is only reference agreement on today’s case. Observed verification of today’s full window is pending its completion.

## Resource costs

| Run | Full request seconds | Peak process RSS GiB | Final inference driver allocation GiB |
|---|---:|---:|---:|
| Original-crop replay | 35.675 | 2.841 | 5.254 |
| Larger-crop replay | 37.209 | 4.859 | 5.281 |

Atmospheric GRIB transfer for both replays: **0 bytes**. Station refresh: **1,233,428 bytes** of new file payload. Larger sea-ice file: **26,403 bytes**; its CDS request took 103.5 seconds including remote queue/processing. HTTP metadata and CDS transfer overhead are not measured as payload bytes.

The 12 GiB guard checks process peak RSS, not total unified memory. Cold startup dominates these runs. Inference averaged about 0.25 s/hour on the original crop and 0.46 s/hour on the larger crop after the first step.

## Reproduce comparison

The explicit larger-domain development profile is [configs/leusden-larger.toml](../../configs/leusden-larger.toml). It resolves to the separately prepared experiment root; it is not the everyday default.

From the repository root, using existing local artifacts:

```sh
pixi run aurora-finetuning compare-reference \
  --reference experiments/leusden-20261024/buienradar.json \
  --predictions outputs/leusden-development-20261024/replay-mps/prediction.json \
    outputs/leusden-development-20261024/larger-replay-mps/prediction.json \
    outputs/leusden-development-20261024/station-v2-original.json \
  --output outputs/leusden-development-20261024/new-comparison.json
```

Choose a new output name on each run; artifacts are never silently overwritten. `compare-reference` accepts a weather timing report containing `result`, or a direct hourly forecast JSON. It rejects missing leads, mismatched coordinates, duplicate timestamps, nonfinite values and incompatible rainfall intervals.

`aurora_finetuning.case_study.run_candidate` provides an explicit cached-input replay with per-stage timing, configuration, input hashes, NetCDF predictions and failure records. `prepare_larger_inputs` recrops retained global GRIB into a separate root after acquiring a matching sea-ice supplement; point its `checkpoints/base` at the existing verified assets. The development comparison itself requires no model loading, training, or network access.

Full local machine artifacts are under `outputs/leusden-development-20261024/`, including `final-comparison.json`, input audits, station history, training provenance, model coefficients, separate historical evaluation and raw predictions. Large data and model artifacts remain ignored by Git.

## Verification

Locked Pixi checks: formatting, Ruff, mypy and 93 offline tests passed. One opt-in integration test was excluded; real MPS full-window replays were run separately. Initial sandbox MPS and unsupported decoder-argument attempts failed, were corrected, and are not counted as successful forecasts.
