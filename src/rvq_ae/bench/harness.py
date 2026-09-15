import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

MIB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class Result:
    """One measured configuration. Throughput units are stated by the suite that produced it."""

    suite: str
    case: str
    variant: str
    milliseconds: float
    memory: float
    throughput: float = 0.0
    units: str = ""
    note: str = ""


@dataclass(slots=True)
class Report:
    environment: dict[str, Any]
    results: list[Result] = field(default_factory=list)

    def add(self, result: Result) -> None:
        self.results.append(result)
        print(
            f"  {result.case:<34} {result.variant:<22} "
            f"{result.milliseconds:9.3f} ms {result.memory:9.1f} MiB",
            flush=True,
        )

    def save(self, folder: Path) -> tuple[Path, Path]:
        folder.mkdir(parents=True, exist_ok=True)
        payload = {"environment": self.environment, "results": [asdict(r) for r in self.results]}
        data = folder / "benchmarks.json"
        data.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        table = folder / "benchmarks.md"
        table.write_text(render(self), encoding="utf-8")
        return data, table


def driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"


def environment(device: torch.device) -> dict[str, Any]:
    """Everything a reader needs to judge whether a number transfers to their machine."""
    values: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
    }
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        values |= {
            "gpu": torch.cuda.get_device_name(device),
            "compute_capability": f"{major}.{minor}",
            "memory_gib": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 1),
            "cuda": torch.version.cuda,
            "driver": driver_version(),
        }
    return values


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    run: Callable[[], Any], device: torch.device, *, warmup: int = 5, reps: int = 20
) -> tuple[float, float]:
    """Median wall time in milliseconds and peak allocation in MiB over reps repetitions.

    Timing uses CUDA events so it reflects device time rather than launch time, and the peak is
    read after a reset so it covers only the measured region. The median rejects the occasional
    outlier from an allocator growth or a clock excursion without hiding a genuine regression.
    """
    for warm in range(warmup):
        run()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for rep in range(reps):
        if device.type == "cuda":
            # torch.cuda.Event carries no stubs, hence the narrow ignores
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            run()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            began = time.perf_counter()
            run()
            samples.append((time.perf_counter() - began) * 1000.0)
    peak = torch.cuda.max_memory_allocated(device) / MIB if device.type == "cuda" else 0.0
    return statistics.median(samples), peak


def free_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def speedup(results: Sequence[Result], case: str, slow: str, fast: str) -> float:
    """Ratio of two variants of the same case, for the summary column."""
    times = {r.variant: r.milliseconds for r in results if r.case == case}
    if slow not in times or fast not in times or times[fast] <= 0:
        return 0.0
    return times[slow] / times[fast]


def render(report: Report) -> str:
    """Markdown table grouped by suite, in the order the suites ran."""
    lines = ["# Benchmarks", ""]
    lines.append("| " + " | ".join(f"{key}: {value}" for key, value in report.environment.items()) + " |")
    lines.append("")
    for suite in dict.fromkeys(result.suite for result in report.results):
        rows = [result for result in report.results if result.suite == suite]
        units = next((row.units for row in rows if row.units), "")
        lines.append(f"## {suite}")
        lines.append("")
        header = ["case", "variant", "ms", "peak MiB"]
        if units:
            header.append(units)
        if any(row.note for row in rows):
            header.append("note")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            cells = [row.case, row.variant, f"{row.milliseconds:.3f}", f"{row.memory:.1f}"]
            if units:
                cells.append(f"{row.throughput:,.1f}")
            if any(item.note for item in rows):
                cells.append(row.note)
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
