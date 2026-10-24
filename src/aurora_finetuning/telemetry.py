"""Wall-clock timings and process memory, including failure paths."""

import json
import logging
import resource
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def peak_rss_gb() -> float:
    """Return process lifetime peak RSS, not instantaneous or system memory."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**3 if sys.platform == "darwin" else 1024**2)


@dataclass
class Timing:
    """One measured stage; accelerator work is synchronised by the caller."""

    stage: str
    seconds: float
    peak_process_rss_gb: float
    success: bool
    accelerator_allocated_gb: float | None = None
    accelerator_driver_gb: float | None = None


@dataclass
class Recorder:
    """Collect stage timings and persist an exclusive JSON report."""

    network_bytes: int = 0
    retained_bytes: int = 0
    timings: list[Timing] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    synchronize: Callable[[], None] = field(default=lambda: None, repr=False)
    accelerator_memory: Callable[[], tuple[float | None, float | None]] = field(
        default=lambda: (None, None), repr=False
    )

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Record a stage even when it raises; never suppress its failure."""
        self.synchronize()
        start = time.perf_counter()
        success = False
        try:
            yield
            self.synchronize()
            success = True
        finally:
            allocated, driver = self.accelerator_memory()
            entry = Timing(
                stage,
                time.perf_counter() - start,
                peak_rss_gb(),
                success,
                allocated,
                driver,
            )
            self.timings.append(entry)
            LOGGER.info(
                "%s: %.3fs, peak RSS %.2f GiB, success=%s",
                stage,
                entry.seconds,
                entry.peak_process_rss_gb,
                success,
            )

    def report(self) -> list[dict[str, object]]:
        """Return serializable timing records."""
        return [asdict(t) for t in self.timings]


def write_json(path: Path, value: object) -> None:
    """Create a JSON artifact without overwriting existing results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    from aurora_finetuning.artifacts import atomic_output

    with atomic_output(path) as temporary:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
