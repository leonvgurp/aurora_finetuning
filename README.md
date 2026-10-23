# Aurora fine-tuning with KNMI weather data

Research workspace for evaluating Microsoft's Aurora weather model and exploring
adaptation to local weather in the Netherlands using KNMI observations. This is
the initial repository setup: the environment and Python package are ready for
development; data retrieval, preprocessing, training, and forecasting pipelines
are not implemented yet.

## Environment

Install [Pixi](https://pixi.sh), then from this repository run:

```sh
pixi install --locked
pixi run aurora-finetuning --help
pixi run python -m aurora_finetuning --version
pixi run smoke
```

The environment uses Python 3.12 and supports Apple Silicon macOS (`osx-arm64`)
and x86-64 Linux (`linux-64`). The local `aurora-finetuning` package is installed
in editable mode from `src/`. Pixi is the sole environment manager; run Python
and tools through `pixi run`. Commit both `pyproject.toml` and `pixi.lock`.

Runtime packages:

- `microsoft-aurora`: the official Aurora implementation, including its PyTorch,
  scientific-computing, and checkpoint-access dependencies.
- `requests`: HTTP access to the KNMI API, following KNMI's documented examples.
- `xarray` and `netCDF4`: labeled weather arrays and NetCDF input/output.

NetCDF4 is installed from conda-forge through Pixi, together with its native
netCDF C and HDF5 libraries. It does not require a system or Homebrew installation.

Ruff, mypy, and pytest are development dependencies. Exact versions and transitive
dependencies are recorded in `pixi.lock`. Use `pixi add --pypi PACKAGE` for runtime
Python dependencies and `pixi add --feature dev PACKAGE` for development tools.

The smoke check imports dependencies only. It does not instantiate a model,
retrieve checkpoints, or run inference. This Mac environment is for development
and small experiments; GPU training hardware and accelerator configuration will
be selected separately. Linux support does not imply a configured CUDA runtime.

## KNMI access

Use the [KNMI Open Data API](https://developer.dataplatform.knmi.nl/open-data-api)
to list dataset files and request temporary download URLs. No dedicated SDK is
required. Select a dataset and its version from the
[KNMI catalog](https://dataplatform.knmi.nl/). The 10-minute in-situ meteorological
observations are a candidate for local station measurements; dataset selection
and coverage need validation before implementing downloads.

Obtain an API key through the [KNMI developer portal](https://developer.dataplatform.knmi.nl/)
and make it available to future download code as an environment variable:

```sh
export KNMI_API_KEY='your-key'
```

This is the planned project variable; the initial CLI does not read it yet.
KNMI requires the key in the HTTP `Authorization` header. Use `requests` to list
files at:

```text
GET https://api.dataplatform.knmi.nl/open-data/v1/datasets/{dataset}/versions/{version}/files
GET https://api.dataplatform.knmi.nl/open-data/v1/datasets/{dataset}/versions/{version}/files/{filename}/url
```

The second response supplies a temporary file download URL. Do not forward the
API key to that URL. Respect KNMI quotas and fair-use guidance; use the
notification service rather than excessive polling. Credentials must stay out
of source control. Environment variables are not loaded automatically from `.env`.

## Planned inputs and outputs

Keep original downloads under `data/raw/`, processed inputs under
`data/processed/`, model weights under `checkpoints/`, and predictions and metrics
under `outputs/`. These directories are ignored and created only when needed.
Record dataset name, version, filenames, timestamps, preprocessing settings, and
random seeds alongside generated results. Honor each dataset's attribution and
license requirements.

Station observations are not directly interchangeable with Aurora inputs.
Aurora consumes structured surface, static, and pressure-level atmospheric fields
with spatial coordinates and time history. Mapping local observations to training
or evaluation targets, obtaining background fields, and choosing a model variant
remain research decisions.

Future runtime settings belong in `configs/*.toml`; no configuration file is
needed for the current help/version CLI. Reusable Python code belongs in
`src/aurora_finetuning/` and tests in `tests/`. No notebooks are required.

## Development checks

```sh
pixi run format        # Apply Ruff formatting
pixi run format-check # Verify formatting
pixi run lint         # Ruff linting and import sorting checks
pixi run typecheck    # Strict mypy checks
pixi run test         # Offline CLI tests against the installed package
pixi run check        # All non-mutating quality checks
pixi run smoke        # Aurora, PyTorch, HTTP, and NetCDF import check
```

GitHub Actions installs the locked Pixi environment and runs `check` and `smoke`
on Linux and Apple Silicon macOS. Git ignores local environments, secrets,
agent-related instructions/state, data, weights, caches, and generated outputs.

## References

- [Microsoft Aurora repository and installation](https://github.com/microsoft/aurora)
- [Aurora documentation and input examples](https://microsoft.github.io/aurora/)
- [KNMI Open Data API documentation](https://developer.dataplatform.knmi.nl/open-data-api)

Check Aurora's upstream license and model terms before distributing or using
model assets. This repository does not redistribute weather data or weights.
