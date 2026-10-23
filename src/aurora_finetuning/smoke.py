"""Cheap gates before downloading weights or launching large experiments."""

import importlib.metadata
import os
from pathlib import Path

import torch
from aurora import AuroraV1p5, Batch, Metadata

from aurora_finetuning.acquisition import ATMOSPHERE, LEVELS, SURFACE, http_session
from aurora_finetuning.config import Config
from aurora_finetuning.credentials import knmi_api_key
from aurora_finetuning.grid import make_grid
from aurora_finetuning.model import accelerator_memory, synchronize
from aurora_finetuning.regional import regional_context
from aurora_finetuning.telemetry import Recorder
from aurora_finetuning.training import trainable_surface_heads


def preflight(config: Config, recorder: Recorder) -> dict[str, object]:
    """Inspect environment and credential presence without reading/logging secrets."""
    with recorder.measure("smoke.preflight"):
        grid = make_grid(config)
        return {
            "aurora_version": importlib.metadata.version("microsoft-aurora"),
            "aurora_v1p5_available": True,
            "torch_version": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
            "cds_credentials_file_present": (Path.home() / ".cdsapirc").is_file(),
            "knmi_environment_key_present": bool(os.environ.get("KNMI_API_KEY")),
            "knmi_secret_file_present": (
                config.root / ".secrets/knmi-api-key"
            ).is_file(),
            "requested_grid": {
                "north": config.north,
                "south": config.south,
                "west": config.west,
                "east": config.east,
            },
            "actual_grid": grid.describe(),
            "credentials_authenticated": False,
        }


def provider_smoke(recorder: Recorder, config: Config) -> dict[str, object]:
    """Authenticate CDS and list one KNMI file; download no weather data."""
    import cdsapi

    with recorder.measure("smoke.cds.authentication"):
        client = cdsapi.Client(
            quiet=True, debug=False, timeout=30, retry_max=1, sleep_max=1
        )
        client.client.check_authentication()
    key = knmi_api_key(config.root)
    with recorder.measure("smoke.knmi.list"), http_session() as session:
        params: dict[str, str | int] = {"maxKeys": 1, "sorting": "desc"}
        response = session.get(
            "https://api.dataplatform.knmi.nl/open-data/v1/datasets/"
            "hourly-in-situ-meteorological-observations-validated/versions/1.0/files",
            params=params,
            headers={"Authorization": key},
            timeout=(10, 30),
        )
        response.raise_for_status()
        files = response.json().get("files", [])
        if not files:
            raise ValueError("KNMI returned no station files")
    return {
        "cds_authenticated": True,
        "knmi_authenticated": True,
        "latest_station_filename": files[0]["filename"],
        "dataset_terms_verified": False,
        "weather_files_downloaded": False,
    }


def synthetic_batch() -> Batch:
    """Create physically scaled tiny input fields, solely for software smoke tests."""
    from datetime import datetime

    shape = (1, 2, 16, 16)
    surface = {name: torch.zeros(shape) for name in SURFACE.values()}
    for name in ("2t", "2d", "skt", "stl1"):
        surface[name].fill_(280)
    for name in ("msl", "sp"):
        surface[name].fill_(100000)
    surface["tcwv"].fill_(10)
    surface["swvl1"].fill_(0.2)
    surface["insolation"] = torch.zeros(shape)
    atmosphere = {name: torch.zeros((1, 2, len(LEVELS), 16, 16)) for name in ATMOSPHERE}
    atmosphere["t"].fill_(260)
    atmosphere["q"].fill_(0.001)
    static = {
        name: torch.zeros((16, 16))
        for name in (
            "lsm",
            "z",
            "anor",
            "isor",
            "cvh",
            "cl",
            "dl",
            "cvl",
            "slor",
            "sdfor",
            "sdor",
            *(f"slt_{i}" for i in range(8)),
            *(f"tvh_{i}" for i in (0, 18, 19, 3, 4, 5, 6)),
            *(f"tvl_{i}" for i in (0, 1, 10, 11, 13, 16, 17, 2, 7, 9)),
        )
    }
    static["lsm"].fill_(1)
    return Batch(
        surface,
        static,
        atmosphere,
        Metadata(
            lat=torch.linspace(54, 50.25, 16),
            lon=torch.linspace(-1, 2.75, 16),
            time=(datetime(2024, 1, 1, 6),),
            atmos_levels=LEVELS,
        ),
    )


def architecture_smoke(device: str, recorder: Recorder) -> dict[str, object]:
    """Run a tiny random Aurora 1.5 forward/backward, never a scientific forecast."""
    from aurora_finetuning.model import check_device

    check_device(Config(device=device))
    torch.manual_seed(42)
    with regional_context():
        with recorder.measure("smoke.tiny_model_construct"):
            model = AuroraV1p5(
                embed_dim=32,
                num_heads=2,
                encoder_depths=(1, 1, 1),
                decoder_depths=(1, 1, 1),
                encoder_num_heads=(2, 4, 8),
                decoder_num_heads=(8, 4, 2),
                window_size=(2, 2, 2),
                autocast=False,
                use_fp16_safe_attention=False,
            )
            model.to(device)
            model.eval()
            trainable_surface_heads(model)
            batch = synthetic_batch()
        recorder.synchronize = lambda: synchronize(device)
        recorder.accelerator_memory = lambda: accelerator_memory(device)
        with recorder.measure("smoke.tiny_forward"):
            prediction = model.forward(batch, lead_times=torch.tensor([1.0]))
            temperature = prediction.surf_vars["2t"]
            rain = prediction.surf_vars["scaled_tp_1h"]
            if not torch.isfinite(temperature).all() or not torch.isfinite(rain).all():
                raise ValueError("Tiny model generated nonfinite outputs")
        with recorder.measure("smoke.tiny_backward"):
            loss = (temperature.mean() - 280).square() + rain.mean().square()
            loss.backward()
            gradients = [
                p.grad
                for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("Missing/nonfinite trainable gradients")
        return {
            "kind": "random tiny architecture; not full-model feasibility or accuracy",
            "device": device,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "shape": list(temperature.shape),
            "finite": True,
            "timings": recorder.report(),
        }
