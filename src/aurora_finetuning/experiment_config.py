"""Strict campaign configuration, separate from operational weather settings."""

import math
import re
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


def record(value: object, allowed: set[str], required: set[str]) -> dict[str, object]:
    """Validate a mapping at the serialization boundary."""
    if not isinstance(value, dict) or not all(isinstance(k, str) for k in value):
        raise ValueError("Expected a mapping with string keys")
    result: dict[str, object] = dict(value)
    if set(result) - allowed or required - set(result):
        invalid = (set(result) - allowed) | (required - set(result))
        raise ValueError(f"Unknown or missing keys: {invalid}")
    return result


def timestamp(value: object) -> datetime:
    """Require explicit UTC offsets; normalize once at the boundary."""
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO strings with UTC offsets")
    time = datetime.fromisoformat(value)
    if time.tzinfo is None:
        raise ValueError("Timestamp needs an explicit UTC offset")
    return time.astimezone(UTC)


def identifier(value: object) -> str:
    """Require safe, unambiguous artifact identifiers."""
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise ValueError("Identifiers require lowercase letters, digits, _ or -")
    return value


def path_value(base: Path, value: object) -> Path:
    """Resolve configuration paths against their containing TOML."""
    if not isinstance(value, str) or not value:
        raise ValueError("Expected a nonempty path string")
    return (base / value).resolve()


@dataclass(frozen=True)
class Case:
    """An independent event with an issuance and explicit observation target."""

    id: str
    issue: datetime
    initialization: datetime
    split: str
    regime: str
    observations: Path
    history: Path | None
    availability: str


@dataclass(frozen=True)
class Method:
    """A built-in baseline or explicitly supplied forecast artifact."""

    id: str
    kind: str
    paths: dict[str, Path]
    config: Path | None
    source: str


@dataclass(frozen=True)
class Job:
    """A bounded bridge to an existing preparation/training/evaluation command."""

    id: str
    phase: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class Campaign:
    """Validated campaign settings and finite execution budgets."""

    path: Path
    id: str
    track: str
    output: Path
    hours: int
    seed: int
    timeout_seconds: int
    max_download_bytes: int
    max_storage_bytes: int
    cases: tuple[Case, ...]
    methods: tuple[Method, ...]
    jobs: tuple[Job, ...]


def load_campaign(path: Path) -> Campaign:
    """Load TOML, reject unknown settings and overlapping event windows."""
    path = path.resolve()
    with path.open("rb") as handle:
        raw = record(
            tomllib.load(handle),
            {
                "id",
                "track",
                "output",
                "hours",
                "seed",
                "timeout_seconds",
                "max_download_bytes",
                "max_storage_bytes",
                "cases",
                "methods",
                "jobs",
            },
            {"id", "track", "output", "cases", "methods"},
        )
    base = path.parent
    numbers = {}
    for name, default in {
        "hours": 8,
        "seed": 42,
        "timeout_seconds": 1800,
        "max_download_bytes": 250_000_000,
        "max_storage_bytes": 25 * 1024**3,
    }.items():
        value = raw.get(name, default)
        if type(value) is not int or int(str(value)) < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be a positive integer (seed may be zero)")
        numbers[name] = int(str(value))
    if numbers["hours"] > 8:
        raise ValueError("Campaign screening supports up to eight forecast hours")
    track = str(raw["track"])
    if track not in {"development", "retrospective", "prospective", "synthetic"}:
        raise ValueError("Invalid campaign track")
    cases = []
    items = raw["cases"]
    if not isinstance(items, list) or not items:
        raise ValueError("Campaign requires nonempty cases")
    for item in items:
        v = record(
            item,
            {
                "id",
                "issue",
                "initialization",
                "split",
                "regime",
                "observations",
                "history",
                "availability",
            },
            {"id", "issue", "initialization", "split", "observations"},
        )
        issue, init = timestamp(v["issue"]), timestamp(v["initialization"])
        if issue.minute or issue.second or issue.microsecond:
            raise ValueError("Issuances must be on exact UTC hours")
        if not 0 <= (issue - init).total_seconds() / 3600 <= 12:
            raise ValueError("Initialization must precede issuance by at most 12 hours")
        split = str(v["split"])
        if split not in {"train", "validation", "test", "development"}:
            raise ValueError("Invalid split")
        if track in {"development", "synthetic"} and split != "development":
            raise ValueError(
                "Development/synthetic cases cannot be qualification cases"
            )
        availability = str(v.get("availability", "unknown"))
        if availability not in {"unknown", "retrospective", "recorded"}:
            raise ValueError("Invalid availability evidence")
        if track == "prospective" and availability != "recorded":
            raise ValueError("Prospective cases need recorded availability")
        cases.append(
            Case(
                identifier(v["id"]),
                issue,
                init,
                split,
                str(v.get("regime", "unclassified")),
                path_value(base, v["observations"]),
                path_value(base, v["history"]) if "history" in v else None,
                availability,
            )
        )
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate case identifiers")
    ordered = sorted(cases, key=lambda c: c.issue)
    ranks = {"train": 0, "validation": 1, "test": 2, "development": 3}
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.issue - a.issue <= timedelta(hours=72 + numbers["hours"] + 30):
            raise ValueError(
                "Events need disjoint 72h histories/targets plus 30h embargo"
            )
        if ranks[a.split] > ranks[b.split]:
            raise ValueError("Splits must be chronological")
    methods = []
    items = raw["methods"]
    if not isinstance(items, list) or not items:
        raise ValueError("Campaign requires nonempty methods")
    for item in items:
        v = record(item, {"id", "kind", "paths", "config", "source"}, {"id", "kind"})
        kind = str(v["kind"])
        if kind not in {"cached", "aurora", "persistence", "dry"}:
            raise ValueError(
                "Use cached, aurora, persistence or dry; refinements use jobs"
            )
        paths = record(v.get("paths", {}), {c.id for c in cases}, set())
        if kind == "cached" and not paths:
            raise ValueError("Cached methods require explicit case-to-NetCDF paths")
        config = path_value(base, v["config"]) if "config" in v else None
        if kind == "aurora" and config is None:
            raise ValueError("Aurora methods require an explicit weather configuration")
        source = str(v.get("source", "era5"))
        if source not in {"era5", "ifs"}:
            raise ValueError("Aurora source must be era5 or ifs")
        if kind == "aurora" and track == "prospective" and source != "ifs":
            raise ValueError("Prospective Aurora requires IFS; ERA5 is retrospective")
        methods.append(
            Method(
                identifier(v["id"]),
                kind,
                {k: path_value(base, p) for k, p in paths.items()},
                config,
                source,
            )
        )
    if len({m.id for m in methods}) != len(methods):
        raise ValueError("Duplicate method identifiers")
    jobs = []
    items = raw.get("jobs", [])
    if not isinstance(items, list):
        raise ValueError("jobs must be an array")
    allowed_commands = {
        "prepare-stations",
        "train-stations",
        "evaluate-stations",
        "prepare-radar-native",
        "train-radar",
        "evaluate-radar-adapter",
        "calibrate-rain",
        "evaluate-rain",
        "adapter-feasibility",
        "pull-knmi",
    }
    for item in items:
        v = record(item, {"id", "phase", "arguments"}, {"id", "phase", "arguments"})
        args = v["arguments"]
        phase = str(v["phase"])
        if (
            phase not in {"fetch", "run"}
            or not isinstance(args, list)
            or not args
            or not all(isinstance(a, str) for a in args)
        ):
            raise ValueError("Jobs require fetch/run phase and string argument arrays")
        arguments = tuple(str(a) for a in args)
        if arguments[0] not in allowed_commands:
            raise ValueError("Job must invoke an existing supported research command")
        if arguments[0].startswith("pull-") != (phase == "fetch"):
            raise ValueError("Downloads belong exclusively to the fetch phase")
        if "--report" in arguments or "--config" not in arguments:
            raise ValueError("Jobs require --config; the runner owns --report")
        jobs.append(Job(identifier(v["id"]), phase, arguments))
    if len({j.id for j in jobs}) != len(jobs):
        raise ValueError("Duplicate job identifiers")
    if not math.isfinite(numbers["timeout_seconds"]):
        raise ValueError("Invalid timeout")
    return Campaign(
        path,
        identifier(raw["id"]),
        track,
        path_value(base, raw["output"]),
        numbers["hours"],
        numbers["seed"],
        numbers["timeout_seconds"],
        numbers["max_download_bytes"],
        numbers["max_storage_bytes"],
        tuple(cases),
        tuple(methods),
        tuple(jobs),
    )
