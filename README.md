# Can Aurora produce useful local forecasts on a laptop?

This project started from a practical question: Microsoft's Aurora is publicly
available, weather observations are public, and a MacBook can run the model—can
that combination produce useful local weather predictions? This repository is
now an **experimental setup for testing that question**, rather than a qualified
weather forecasting application.

We compare locally executed Aurora 1.5, station corrections, radar methods and
public forecasts against later observations in the European Netherlands.
The primary horizon is eight hours; temperature and rainfall are the main outcomes.
Public future forecasts are comparators, never inputs to the local Aurora methods.

## Key results so far

| Experiment | Finding | Evidence limit |
|---|---|---|
| Local frozen Aurora | Regional inference runs on M3/16 GB/MPS | Execution does not establish forecast skill |
| Wider regional crops | Temperature disagreement with a supplied Buienradar forecast fell from 1.573°C to 1.108°C MAE | One development case; observed accuracy not established |
| Rainfall in that case | Aurora predicted 0.228 mm over eight hours versus 7.6 mm in the reference; wider crops did not fix it | Reference prediction, not measured rainfall |
| Station corrections | Pressure improved in a small held-out pilot; temperature and wind results were mixed | One day per split; retrospective inputs |
| Radar motion | Improved first-hour MAE in one measured case; degraded at longer leads | One event; coverage changes with lead |
| Radar CNN | Training/evaluation machinery exists | No representative trained, independently evaluated model |
| LoRA | One isolated training step fitted the laptop | No evidence of improved forecasts |
| Execution optimization | Engine reuse and sparse inference helped; tested float16 was slower | Hardware/case-specific timings |

**Conclusion:** useful forecasting skill remains unproven. Supplementary data can
help in particular contexts, but no method is qualified for routine prediction.
See the [experiment history](docs/experiment-history.md) for what was tried,
including failures, and the [full experimental design](docs/experimental-design.md)
for how those questions will now be tested.

## Reproduce the environment

Install [Pixi](https://pixi.sh), then from this checkout:

```sh
pixi install --locked
pixi run check
pixi run smoke
```

Pixi supplies Python 3.12 and the editable package on macOS ARM64/Linux x86-64.
The tested accelerator is Apple M3 with 16 GB unified memory. Real-data work
requires model assets, disk space, and provider credentials; see
[setup and existing research commands](docs/setup.md). Offline pipeline checks
need no credentials or model download. Use Pixi for all execution.

## Experiment workflow

Start with a deterministic **synthetic software smoke test**:

```sh
pixi run experiment-demo --output outputs/experiments/smoke
pixi run experiment-plan --config outputs/experiments/smoke/campaign.toml
pixi run experiment-run --config outputs/experiments/smoke/campaign.toml
pixi run experiment-report --input outputs/experiments --output outputs/reports/smoke
pixi run experiment-notebook
```

Use a fresh demo/report output directory on repeat. Repeating `experiment-run`
reuses completed checksummed work when configuration, source and declared inputs
are unchanged. The notebook loads the latest score snapshot per run, performs no
training/inference/downloads, and keeps synthetic results labelled.
Set `AURORA_RESULTS_ROOT` to inspect another result root.

For real experiments, copy [the campaign example](configs/experiment.example.toml),
supply its explicit data paths, and inspect `experiment-plan` before execution.
`experiment-fetch` runs declared acquisition jobs; `experiment-run` runs declared
research jobs, predictions and observational scoring. Jobs bridge the existing
station/radar preparation, training and evaluation commands. Their original
reports remain separate from the common observational score table until normalized
predictions are supplied. Missing prerequisites are reported, not fabricated.

```sh
pixi run experiment-collect --root . --output outputs/evidence/initial
pixi run experiment-plan --config configs/experiment.example.toml
# After configuring actual inputs and acquisition jobs:
# pixi run experiment-fetch --config configs/my-campaign.toml
# pixi run experiment-run --config configs/my-campaign.toml
```

## Inputs, outputs and interpretation

- Campaign TOML paths resolve relative to that TOML; CLI paths resolve from the
  working directory. Explicit paths support execution outside the repository.
- Forecasts/observations use UTC NetCDF with physical units and explicit rainfall
  intervals. Spatial grids must match, or atmospheric grids are sampled at station
  coordinates (temperature interpolation; nearest-cell rainfall).
- Runs retain manifests, hashes, predictions, receipts, failures, timing and tidy
  CSV scores under their configured output directory. Reports export PNG plots
  and paired event-bootstrap summaries. Original large inputs/results stay ignored.
- Public HARMONIE/IFS normalization and bounded IFS acquisition are exposed through
  `pixi run experiment-public --help`. KNMI acquisition uses declared `pull-knmi`
  jobs. Raw KNMI forecast HDF5 decoding remains a documented preparation boundary.
- Retrospective, prospective, development-reference and synthetic evidence are
  separate. Compare errors on paired support, including coverage. A better match
  to another provider's forecast is not proof of greater observed accuracy.

The existing `pixi run weather Leusden --hours 8` interface remains available as
an experimental convenience; see [setup](docs/setup.md). No full campaign,
large download or training run is launched by the default checks.
