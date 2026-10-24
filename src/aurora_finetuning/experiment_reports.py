"""Load saved evidence and generate observational performance plots offline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aurora_finetuning.acquisition import sha256
from aurora_finetuning.artifacts import atomic_output
from aurora_finetuning.telemetry import write_json


def collect_evidence(root: Path, output: Path) -> dict[str, object]:
    """Inventory historical artifacts without reinterpreting unknown old schemas."""
    files = sorted(
        set((root / "experiments").rglob("*.md"))
        | set((root / "experiments").rglob("*.json"))
        | set((root / "outputs").rglob("*.json"))
    )
    entries = []
    for path in files:
        if output.resolve() in path.resolve().parents:
            continue
        entry: dict[str, object] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".md":
            entry.update(kind="narrative", status="documented")
        else:
            try:
                document = json.loads(path.read_text())
                if not isinstance(document, dict):
                    raise ValueError("Not a JSON object")
                entry.update(
                    kind="legacy-json",
                    keys=sorted(document),
                    scope=document.get("scope", "unclassified historical artifact"),
                    success=document.get("success"),
                    status="inventoried; original schema retained",
                )
            except (ValueError, UnicodeError):
                entry.update(kind="unreadable", status="blocked")
        entries.append(entry)
    result = {
        "schema_version": 1,
        "scope": "Historical inventory, not a qualification leaderboard",
        "entries": entries,
    }
    write_json(output / "evidence.json", result)
    # Only convert the explicit compare-reference schema; never infer truth from names.
    comparisons = []
    for path in files:
        if path.suffix != ".json":
            continue
        try:
            document = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if (
            not isinstance(document, dict)
            or not {"reference_sha256", "candidates"} <= document.keys()
        ):
            continue
        for candidate in document["candidates"]:
            comparison = candidate.get("comparison", {})
            for variable, values in comparison.get("metrics", {}).items():
                for metric, value in values.items():
                    comparisons.append(
                        {
                            "artifact": str(path.relative_to(root)),
                            "candidate": candidate.get("path"),
                            "variable": variable,
                            "metric": metric,
                            "value": value,
                            "track": "development-reference-agreement",
                        }
                    )
    if comparisons:
        with atomic_output(output / "development-comparisons.csv") as temporary:
            pd.DataFrame(comparisons).to_csv(temporary, index=False)
    return {
        "entries": len(entries),
        "comparisons": len(comparisons),
        "output": str(output),
    }


def load_results(root: Path) -> pd.DataFrame:
    """Load only the latest score snapshot of each campaign run, avoiding repeats."""
    frames = []
    for manifest in sorted(root.rglob("run/manifest.json")):
        snapshots = sorted(manifest.parent.glob("snapshot-*/summary.json"))
        if not snapshots:
            continue
        scores = snapshots[-1].with_name("scores.csv")
        if scores.exists():
            frame = pd.read_csv(scores)
            frame["run_directory"] = str(manifest.parent)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def paired_summary(results: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Bootstrap equal-weight event means; pixels and stations are not replicates."""
    if results.empty:
        return pd.DataFrame()
    rows = results[results.metric == "paired_delta_mae"]
    output = []
    groups = [
        "campaign",
        "track",
        "split",
        "method",
        "baseline",
        "variable",
        "lead_hour",
        "interval_hours",
    ]
    rng = np.random.default_rng(seed)
    for key, group in rows.groupby(groups, dropna=False):
        events = group.groupby("case").value.mean().dropna().to_numpy(dtype=float)
        if not len(events):
            continue
        lower = upper = None
        if len(events) > 1:
            samples = rng.choice(events, (2000, len(events)), replace=True).mean(axis=1)
            lower, upper = np.quantile(samples, [0.025, 0.975]).tolist()
        row = dict(zip(groups, key, strict=True))
        row.update(
            delta_mae=float(events.mean()), lower=lower, upper=upper, events=len(events)
        )
        output.append(row)
    return pd.DataFrame(output)


def performance_plots(results: pd.DataFrame, output: Path) -> list[Path]:
    """Export separate curves for tracks, splits, variables and native intervals."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if results.empty:
        return []
    paths = []
    rows = results[(results.baseline == "own") & (results.lead_hour > 0)]
    for key, group in rows.groupby(
        ["campaign", "track", "split", "variable", "interval_hours"]
    ):
        campaign, track, split, variable, interval = key
        for metric in ("mae", "coverage", "csi_0.1", "csi_1.0", "csi_5.0"):
            selected = group[group.metric == metric]
            if selected.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            for method, values in selected.groupby("method"):
                means = values.groupby("lead_hour").value.mean()
                ax.plot(
                    means.index, means.to_numpy(dtype=float), marker="o", label=method
                )
            ax.set(
                xlabel="Hours after issuance",
                ylabel=metric,
                title=(
                    f"{campaign} · {track} · {split}\n{variable}, {interval}h intervals"
                ),
            )
            ax.legend(fontsize="small")
            fig.tight_layout()
            target = (
                output
                / f"{campaign}-{track}-{split}-{variable}-{interval}-{metric}.png"
            )
            with atomic_output(target) as temporary:
                fig.savefig(temporary, dpi=150)
            plt.close(fig)
            paths.append(target)
    paired = paired_summary(results)
    if not paired.empty:
        for key, group in paired.groupby(
            ["campaign", "track", "split", "variable", "interval_hours"]
        ):
            fig, ax = plt.subplots(figsize=(9, 5))
            for (paired_method, baseline), values in group[group.lead_hour > 0].groupby(
                ["method", "baseline"]
            ):
                values = values.sort_values("lead_hour")
                ax.plot(
                    values.lead_hour,
                    values.delta_mae,
                    label=f"{paired_method} vs {baseline}",
                )
                if values.lower.notna().any():
                    ax.fill_between(
                        values.lead_hour,
                        values.lower.astype(float),
                        values.upper.astype(float),
                        alpha=0.15,
                    )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(
                xlabel="Hours after issuance",
                ylabel="Paired MAE difference (negative improves)",
                title=" · ".join(str(v) for v in key),
            )
            ax.legend(fontsize="x-small")
            fig.tight_layout()
            target = output / ("paired-" + "-".join(str(v) for v in key) + ".png")
            with atomic_output(target) as temporary:
                fig.savefig(temporary, dpi=150)
            plt.close(fig)
            paths.append(target)
    return paths


def report_results(root: Path, output: Path) -> dict[str, object]:
    """Publish an immutable offline analysis bundle, including empty-run guidance."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    results = load_results(root)
    if not results.empty:
        results.to_csv(output / "scores.csv", index=False)
        paired_summary(results).to_csv(output / "paired-summary.csv", index=False)
    plots = performance_plots(results, output)
    report = {
        "rows": len(results),
        "plots": [str(p) for p in plots],
        "guidance": "Run a campaign first; legacy evidence is separate."
        if results.empty
        else (
            "Own-support curves are descriptive. "
            "Compare paired differences and coverage."
        ),
    }
    write_json(output / "report.json", report)
    return report


def execution_results(root: Path) -> pd.DataFrame:
    """Load latest stage statuses and measured execution costs, including failures."""
    rows = []
    for manifest in sorted(root.rglob("run/manifest.json")):
        snapshots = sorted(manifest.parent.glob("snapshot-*/summary.json"))
        if not snapshots:
            continue
        summary = json.loads(snapshots[-1].read_text())
        for state in summary["states"]:
            rows.append(
                {
                    "campaign": summary["campaign"],
                    "track": summary["track"],
                    "run_directory": str(manifest.parent),
                    **state,
                }
            )
    return pd.DataFrame(rows)


def example_plots(root: Path, output: Path) -> list[Path]:
    """Show one event's hourly/cumulative predictions without pooling locations."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from aurora_finetuning.experiment_config import load_campaign
    from aurora_finetuning.experiments import read_dataset

    paths = []
    for manifest_path in sorted(root.rglob("run/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        source = manifest.get("campaign_path")
        if not isinstance(source, str) or not Path(source).is_file():
            continue
        campaign = load_campaign(Path(source))
        case = campaign.cases[0]
        if not case.observations.is_file():
            continue
        datasets = {"observed": read_dataset(case.observations)}
        for path in (manifest_path.parent / case.id).glob("*.nc"):
            datasets[path.stem] = read_dataset(path)
        for variable in ("temperature_c", "precipitation_mm"):
            fig, ax = plt.subplots(figsize=(9, 4))
            for method, dataset in datasets.items():
                if variable not in dataset:
                    continue
                field = dataset[variable]
                first_location = {d: 0 for d in field.dims if d != "time"}
                series = field.isel(first_location)
                ax.plot(dataset.time.values, series.values, marker="o", label=method)
            ax.set(
                title=(
                    f"{campaign.id}: {case.id}, first stored location "
                    f"· {campaign.track}"
                ),
                ylabel=variable,
                xlabel="UTC valid time",
            )
            ax.legend(fontsize="small")
            fig.autofmt_xdate()
            fig.tight_layout()
            target = output / f"example-{campaign.id}-{variable}.png"
            with atomic_output(target) as temporary:
                fig.savefig(temporary, dpi=150)
            plt.close(fig)
            paths.append(target)
            if variable == "precipitation_mm":
                fig, ax = plt.subplots(figsize=(9, 4))
                for method, dataset in datasets.items():
                    if variable in dataset:
                        field = dataset[variable]
                        series = field.isel({d: 0 for d in field.dims if d != "time"})
                        ax.plot(
                            dataset.time.values,
                            series.cumsum("time", skipna=False).values,
                            label=method,
                        )
                ax.set(
                    title=f"{case.id}: accumulated rainfall at first stored location",
                    ylabel="Rainfall (mm)",
                    xlabel="UTC valid time",
                )
                ax.legend(fontsize="small")
                fig.autofmt_xdate()
                fig.tight_layout()
                target = output / f"cumulative-{campaign.id}.png"
                with atomic_output(target) as temporary:
                    fig.savefig(temporary, dpi=150)
                plt.close(fig)
                paths.append(target)
        # Native-grid maps only when actual two-dimensional spatial data exists.
        for method, dataset in datasets.items():
            if "precipitation_mm" not in dataset or dataset.precipitation_mm.ndim != 3:
                continue
            field = dataset.precipitation_mm.isel(time=0)
            fig, ax = plt.subplots(figsize=(6, 5))
            field.plot(ax=ax)
            ax.set_title(f"{campaign.id}: {method}, first native rain interval")
            fig.tight_layout()
            target = output / f"spatial-{campaign.id}-{method}.png"
            with atomic_output(target) as temporary:
                fig.savefig(temporary, dpi=150)
            plt.close(fig)
            paths.append(target)
    return paths


def legacy_metrics(root: Path) -> pd.DataFrame:
    """Expose original historical numeric results without creating a leaderboard.

    Paths retain every enclosing method/lead/support label. Unclassified metrics,
    training losses and timing counters remain distinct from observational scores.
    """
    rows: list[dict[str, object]] = []

    def visit(value: object, path: str, artifact: str, scope: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}/{key}", artifact, scope)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}", artifact, scope)
        elif type(value) in (float, int) or value is None:
            if path.rsplit("/", 1)[-1] in {
                "mae",
                "rmse",
                "bias",
                "csi",
                "precision",
                "recall",
                "loss",
                "seconds",
                "network_bytes",
                "peak_process_rss_gb",
                "accelerator_driver_gb",
                "accelerator_allocated_gb",
            }:
                rows.append(
                    {
                        "artifact": artifact,
                        "scope": scope,
                        "metric_path": path,
                        "value": value,
                    }
                )

    for path in sorted((root / "outputs").rglob("*.json")):
        # Campaign snapshots have their own checked normalized result loader.
        if any(p.name in {"run", "fetch"} for p in path.parents):
            continue
        try:
            document = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if isinstance(document, dict):
            visit(
                document,
                "",
                str(path.relative_to(root)),
                str(document.get("scope", "unclassified historical result")),
            )
    return pd.DataFrame(rows)
