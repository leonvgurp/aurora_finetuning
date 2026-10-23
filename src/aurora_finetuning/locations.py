"""Dutch locality resolution through PDOK, with an on-disk lookup cache."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from aurora_finetuning.acquisition import http_session
from aurora_finetuning.forecast import validate_location
from aurora_finetuning.telemetry import write_json


@dataclass(frozen=True)
class Location:
    """A resolved locality in geographic degrees."""

    name: str
    latitude: float
    longitude: float


def resolve_location(
    name: str | None, latitude: float | None, longitude: float | None, cache: Path
) -> Location:
    """Resolve a unique Dutch place name or validate an explicit coordinate pair."""
    if name is not None and (latitude is not None or longitude is not None):
        raise ValueError("Use a place name OR both --latitude and --longitude")
    if name is None:
        if latitude is None or longitude is None:
            raise ValueError("Supply a place name or both --latitude and --longitude")
        validate_location(latitude, longitude)
        return Location(f"{latitude}, {longitude}", latitude, longitude)
    query = " ".join(name.split())
    if not query:
        raise ValueError("Place name must not be empty")
    key = hashlib.sha256(query.casefold().encode()).hexdigest()
    path = cache / f"{key}.json"
    if path.exists():
        records = json.loads(path.read_text())
    else:
        with http_session() as session:
            escaped = query.replace("\\", "\\\\").replace('"', '\\"')
            response = session.get(
                "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free",
                params={
                    "q": "*",
                    "fq": ["type:woonplaats", f'woonplaatsnaam:"{escaped}"'],
                    "rows": "100",
                },
                timeout=(10, 30),
            )
            response.raise_for_status()
            payload = response.json()["response"]
            if payload["numFound"] == 0:
                response = session.get(
                    "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free",
                    params={"q": query, "fq": "type:woonplaats", "rows": "100"},
                    timeout=(10, 30),
                )
                response.raise_for_status()
                payload = response.json()["response"]
        records = payload["docs"]
        # Never cache a truncated candidate list as a complete lookup.
        if payload["numFound"] > len(records):
            raise ValueError(
                "Too many matching places; specify a more precise locality"
            )
        write_json(path, records)
    if not isinstance(records, list):
        raise ValueError(f"Invalid location cache: {path}")
    exact = [
        r
        for r in records
        if isinstance(r, dict)
        and str(r.get("woonplaatsnaam", "")).casefold() == query.casefold()
    ]
    candidates = exact or records
    if not candidates:
        raise ValueError(f"No Dutch locality found for {query!r}; try coordinates")
    if len(candidates) != 1:
        labels = [str(r.get("weergavenaam", "unknown")) for r in candidates]
        raise ValueError(
            "Ambiguous locality; use coordinates or refine: " + "; ".join(labels)
        )
    record = candidates[0]
    point = str(record["centroide_ll"])
    if not point.startswith("POINT(") or not point.endswith(")"):
        raise ValueError("PDOK returned an invalid geographic centroid")
    lon, lat = map(float, point[6:-1].split())
    validate_location(lat, lon)
    return Location(str(record["weergavenaam"]), lat, lon)
