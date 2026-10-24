"""Isolated one-step LoRA hardware experiment; updated weights are discarded."""

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from aurora_finetuning.config import Config
from aurora_finetuning.grid import make_grid
from aurora_finetuning.telemetry import Recorder, write_json


def run_isolated(
    config: Config, initialization: datetime, output: Path
) -> dict[str, object]:
    """Run a cached-input MPS experiment with a ten-minute process timeout."""
    if (
        config.device != "mps"
        or config.precision != "float32"
        or make_grid(config).shape != (48, 64)
    ):
        raise ValueError(
            "Adapter feasibility requires MPS, float32 and the original 48x64 grid"
        )
    if initialization.tzinfo is not None:
        raise ValueError("Initialization must be naive UTC")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    values = asdict(config)
    values["root"] = str(config.root.resolve())
    write_json(output / "config.json", values)
    with (output / "worker.log").open("w") as log:
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aurora_finetuning.adapter_feasibility",
                    str(output / "config.json"),
                    initialization.isoformat(),
                    str(output / "report.json"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
                check=False,
            )
        except subprocess.TimeoutExpired:
            write_json(
                output / "report.json",
                {
                    "passed": False,
                    "reason": "ten-minute timeout",
                    "weights_saved": False,
                },
            )
            raise ValueError(f"Adapter feasibility timed out; see {output}") from None
    if not (output / "report.json").exists():
        write_json(
            output / "report.json",
            {
                "passed": False,
                "reason": f"worker exit {process.returncode}",
                "weights_saved": False,
            },
        )
    report = json.loads((output / "report.json").read_text())
    if not isinstance(report, dict):
        raise ValueError("Invalid feasibility report")
    if process.returncode or not report.get("passed"):
        raise ValueError(
            f"Adapter feasibility failed; see {output}/report.json and worker.log"
        )
    return report


def worker(config: Config, initialization: datetime, output: Path) -> None:
    """Perform inference equivalence checks and exactly one gradient update."""
    import torch
    from aurora import AuroraV1p5
    from aurora.model.lora import LoRARollout

    from aurora_finetuning.assets import CHECKPOINT, verify_assets
    from aurora_finetuning.model import accelerator_memory, check_device, check_memory
    from aurora_finetuning.preparation import prepare_batch, target_fields
    from aurora_finetuning.regional import regional_context

    recorder = Recorder()
    report: dict[str, object] = {
        "passed": False,
        "weights_saved": False,
        "training_steps": 0,
    }
    try:
        check_device(config)
        torch.mps.set_per_process_memory_fraction(0.8)
        recorder.synchronize = torch.mps.synchronize
        recorder.accelerator_memory = lambda: accelerator_memory("mps")
        torch.manual_seed(config.seed)
        with regional_context():
            batch = prepare_batch(config, initialization, recorder, "era5")
            target = (
                target_fields(config, initialization + timedelta(hours=6), recorder)
                .precipitation_mm.isel(time=0)
                .values
            )
            with recorder.measure("adapter.load"):
                verify_assets(config)
                model = AuroraV1p5(
                    use_lora=True,
                    lora_mode="single",
                    autocast=False,
                    use_fp16_safe_attention=False,
                )
                original = model.load_state_dict
                allowed = {
                    name
                    for name in model.state_dict()
                    if ".lora_A" in name or ".lora_B" in name
                }

                def load(state: dict[str, torch.Tensor], strict: bool = True) -> object:
                    result = original(state, strict=False)
                    if set(result.missing_keys) != allowed or result.unexpected_keys:
                        raise ValueError(
                            "Checkpoint mismatch outside new LoRA parameters"
                        )
                    return result

                with patch.object(model, "load_state_dict", side_effect=load):
                    model.load_checkpoint_local(
                        str(config.root / "checkpoints/base" / CHECKPOINT), strict=False
                    )
                model.requires_grad_(False)
                trainable = []
                for name, param in model.named_parameters():
                    if name in allowed:
                        param.requires_grad_(True)
                        trainable.append(param)
                model = model.to("mps").eval()
                batch = batch.to("mps")
            modules = [
                module for module in model.modules() if isinstance(module, LoRARollout)
            ]
            with recorder.measure("adapter.zero_equivalence"), torch.no_grad():
                for module in modules:
                    module.mode = "from_second"
                reference = (
                    model.forward(batch, lead_times=torch.full((1,), 6.0, device="mps"))
                    .surf_vars["scaled_tp_1h"]
                    .detach()
                    .cpu()
                )
                for module in modules:
                    module.mode = "single"
                zero = (
                    model.forward(batch, lead_times=torch.full((1,), 6.0, device="mps"))
                    .surf_vars["scaled_tp_1h"]
                    .detach()
                    .cpu()
                )
                difference = float((reference - zero).abs().max())
                report["zero_adapter_max_difference_m"] = difference
                if not torch.isfinite(zero).all() or difference > 1e-7:
                    raise ValueError(
                        "Zero adapters changed the original forward result"
                    )
            check_memory(config)
            if torch.mps.driver_allocated_memory() > 10 * 1024**3:
                raise ValueError(
                    "Accelerator allocation exceeded 10 GiB before training"
                )
            frozen = {
                name: param._version
                for name, param in model.named_parameters()
                if not param.requires_grad
            }
            before = [param.detach().cpu().clone() for param in trainable]
            model.configure_activation_checkpointing(
                ("Basic3DEncoderLayer", "Basic3DDecoderLayer")
            )
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            truth = torch.as_tensor(target, dtype=torch.float32, device="mps")
            valid = torch.isfinite(truth)
            if not valid.any():
                raise ValueError("No finite historical rainfall targets")
            with recorder.measure("adapter.one_training_step"):
                prediction = (
                    model.forward(
                        batch, lead_times=torch.full((1,), 6.0, device="mps")
                    ).surf_vars["scaled_tp_1h"][0, 0]
                    * 1000
                )
                loss = (
                    (
                        torch.log1p(prediction.clamp_min(0)[valid])
                        - torch.log1p(truth[valid])
                    )
                    .square()
                    .mean()
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                torch.autograd.backward(loss)
                if not all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in trainable
                ):
                    raise ValueError("Missing or nonfinite adapter gradients")
                if not any(
                    bool(torch.any(p.grad != 0))
                    for p in trainable
                    if p.grad is not None
                ):
                    raise ValueError("All adapter gradients are zero")
                check_memory(config)
                if torch.mps.driver_allocated_memory() > 10 * 1024**3:
                    raise ValueError(
                        "Accelerator allocation exceeded 10 GiB in backward"
                    )
                optimizer.step()
            changed = any(
                not torch.equal(old, param.detach().cpu())
                for old, param in zip(before, trainable, strict=True)
            )
            unchanged = all(
                param.grad is None and param._version == frozen[name]
                for name, param in model.named_parameters()
                if name in frozen
            )
            if not changed or not unchanged:
                raise ValueError("Adapter/base weight update invariant failed")
            check_memory(config)
            if torch.mps.driver_allocated_memory() > 10 * 1024**3:
                raise ValueError("Accelerator allocation exceeded 10 GiB after update")
            report.update(
                passed=True,
                training_steps=1,
                loss=float(loss.detach().cpu()),
                trainable_parameters=sum(p.numel() for p in trainable),
                adapter_updated=changed,
                base_unchanged=unchanged,
            )
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["reason"] = str(error)
        raise
    finally:
        report.update(
            timings=recorder.report(),
            input_sha256=recorder.inputs,
            scope="hardware feasibility only; no trained weights saved",
        )
        write_json(output, report)


if __name__ == "__main__":
    values = json.loads(Path(sys.argv[1]).read_text())
    values["root"] = Path(values["root"])
    worker(Config(**values), datetime.fromisoformat(sys.argv[2]), Path(sys.argv[3]))
