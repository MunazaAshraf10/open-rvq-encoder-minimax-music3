import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from rvq_ae.bench.harness import Report, environment
from rvq_ae.bench.suites import (
    attention_suite,
    audio_suite,
    depth_suite,
    encoder_suite,
    pooling_suite,
    training_suite,
)
from rvq_ae.train import enable_tf32

log = logging.getLogger("rvq_ae.bench")

# audio is the only suite that needs the released weights, so it is excluded from the default set.
OFFLINE = ("attention", "pooling", "encoder", "depth", "training")
SUITES = (*OFFLINE, "audio")


def run(
    suites: Sequence[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    out: Path = Path("benchmarks"),
    seed: int = 0,
) -> Report:
    """Run the named suites and write benchmarks.json and benchmarks.md into out."""
    unknown = sorted(set(suites) - set(SUITES))
    if unknown:
        raise ValueError(f"unknown suites: {unknown}; expected from {list(SUITES)}")
    torch.manual_seed(seed)
    if device.type == "cuda":
        enable_tf32(True)
    report = Report(environment=environment(device))
    for name in suites:
        log.info("running %s", name)
        if name == "attention":
            attention_suite(report, device, dtype)
        elif name == "pooling":
            pooling_suite(report, device)
        elif name == "encoder":
            encoder_suite(report, device, dtype)
        elif name == "depth":
            depth_suite(report, device, dtype)
        elif name == "training":
            training_suite(report, device, dtype)
        else:
            audio_suite(report, device)
    data, table = report.save(out)
    log.info("wrote %s and %s", data, table)
    return report
