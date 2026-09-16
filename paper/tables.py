import argparse
import json
import os
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published"
REPRODUCED = ROOT / "results" / "reproduced"
BENCHMARKS = ROOT / "benchmarks" / "benchmarks.json"
TEX = ROOT / "paper" / "tables"

VARIANTS = ("v1", "v2", "v3", "v4")
LABELS = {
    "serveurperso-v1": "Serveurperso v1 (community)",
    "simpletuner-v1": "v1, 41M",
    "simpletuner-v2": "v2, 155M",
    "simpletuner-v3": "v3, 155M + MERT",
    "simpletuner-v4": "v4, 169M + depth",
}
# the packaged v1 checkpoint is step 17,500; v2 to v4 are packaged at final
PACKAGED = {"v1": "checkpoint-17500", "v2": "final", "v3": "final", "v4": "final"}
MATCHED = "checkpoint-17500"

Row = list[str]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}"


def rows_of(tag: str) -> dict[str, dict[str, Any]]:
    """Checkpoint rows of one variant keyed by checkpoint name."""
    return {
        row["checkpoint"]: row for row in load(PUBLISHED / tag / "evaluation-metrics.json")["checkpoints"]
    }


def replay_table() -> tuple[Row, list[Row]]:
    data = load(PUBLISHED / "condition-replay-aggregate.json")
    header = ["Model", "Parameters", "Mean", "Std", "5th pct", "95th pct"]
    rows: list[Row] = []
    for model in data["models"]:
        stats = model["predicted_code_condition_cosine"]
        rows.append(
            [
                LABELS[model["model_id"]],
                f"{model['parameter_count']:,}",
                f"{stats['mean']:.4f}",
                f"{stats['std']:.4f}",
                f"{stats['quantiles']['0.05']:.4f}",
                f"{stats['quantiles']['0.95']:.4f}",
            ]
        )
    control = data["models"][-1]["true_code_control"]
    rows.append(
        [
            "true codes (control)",
            "",
            f"{control['mean']:.4f}",
            f"{control['std']:.4f}",
            f"{control['quantiles']['0.05']:.4f}",
            f"{control['quantiles']['0.95']:.4f}",
        ]
    )
    return header, rows


def matched_table() -> tuple[Row, list[Row]]:
    header = ["Model", "Loss", "Sem. top-1", "Sem. top-5", "Ac. top-1", "Ac. top-5"]
    rows: list[Row] = []
    for tag in VARIANTS:
        row = rows_of(tag)[MATCHED]
        rows.append(
            [
                LABELS[f"simpletuner-{tag}"],
                f"{row['loss']:.4f}",
                pct(row["semantic_top1"]),
                pct(row["semantic_top5"]),
                pct(row["acoustic_top1"]),
                pct(row["acoustic_top5"]),
            ]
        )
    return header, rows


def forcing_table() -> tuple[Row, list[Row]]:
    row = rows_of("v4")["final"]
    header = ["Metric", "Free-running", "Teacher-forced"]
    rows = [
        ["semantic top-1", pct(row["semantic_top1"]), pct(row["teacher_forced_semantic_top1"])],
        ["semantic top-5", pct(row["semantic_top5"]), pct(row["teacher_forced_semantic_top5"])],
        ["acoustic top-1", pct(row["acoustic_top1"]), pct(row["teacher_forced_acoustic_top1"])],
        ["acoustic top-5", pct(row["acoustic_top5"]), pct(row["teacher_forced_acoustic_top5"])],
    ]
    return header, rows


def depth_table() -> tuple[Row, list[Row]]:
    header = ["Codebook", "v1", "v2", "v3", "v4 free-running", "v4 teacher-forced"]
    matched = {tag: rows_of(tag)[MATCHED] for tag in VARIANTS}
    rows: list[Row] = []
    for book in range(8):
        cells = [str(book)] + [pct(matched[tag][f"head_{book}_top1"]) for tag in VARIANTS]
        cells.append(pct(matched["v4"][f"teacher_forced_head_{book}_top1"]))
        rows.append(cells)
    return header, rows


def final_table() -> tuple[Row, list[Row]]:
    header = ["Model", "Checkpoint", "Step", "Loss", "Sem. top-1", "Sem. top-5", "Ac. top-1", "Ac. top-5"]
    rows: list[Row] = []
    for tag in VARIANTS:
        row = rows_of(tag)[PACKAGED[tag]]
        rows.append(
            [
                LABELS[f"simpletuner-{tag}"],
                PACKAGED[tag],
                f"{row['step']:,}",
                f"{row['loss']:.4f}",
                pct(row["semantic_top1"]),
                pct(row["semantic_top5"]),
                pct(row["acoustic_top1"]),
                pct(row["acoustic_top5"]),
            ]
        )
    return header, rows


def reproduced_table() -> tuple[Row, list[Row]]:
    """Published packaged rows against the rows this implementation produced on the same holdout."""
    header = ["Model", "Source", "Loss", "Sem. top-1", "Sem. top-5", "Ac. top-1", "Ac. top-5", "TF ac. top-1"]
    rows: list[Row] = []
    for tag, folder in (("v1", "v1"), ("v2", "v2"), ("v3", "v3"), ("v4", "v4"), ("v4", "v4_fp32")):
        published = rows_of(tag)[PACKAGED[tag]]
        label = LABELS[f"simpletuner-{tag}"]
        forced = pct(published["teacher_forced_acoustic_top1"]) if tag == "v4" else ""
        if folder == tag:
            rows.append(
                [
                    label,
                    "published, bf16, 4 ranks",
                    f"{published['loss']:.4f}",
                    pct(published["semantic_top1"]),
                    pct(published["acoustic_top1"]),
                    forced,
                ]
            )
        path = REPRODUCED / folder / "metrics.json"
        if not path.is_file():
            rows.append([label, f"this repository ({folder}): not run yet", "", "", "", ""])
            continue
        local = load(path)[0]
        precision = "fp32" if folder.endswith("fp32") else "bf16"
        forced_local = pct(local["teacher_forced_acoustic_top1"]) if tag == "v4" else ""
        rows.append(
            [
                label,
                f"this repository, {precision}, 1 GPU",
                f"{local['loss']:.4f}",
                pct(local["semantic_top1"]),
                pct(local["acoustic_top1"]),
                forced_local,
            ]
        )
    return header, rows


REPLAY = ROOT / "results" / "published" / "replay"
REPLAY_FILES = {
    "community": "raw-metrics-serveurperso-v1.json",
    "v1": "raw-metrics-simpletuner-v1.json",
    "v2": "raw-metrics-simpletuner-v2.json",
    "v3": "raw-metrics-simpletuner-v3.json",
    "v4": "raw-metrics-simpletuner-v4.json",
}
DIFFERENCES = (("v1", "community"), ("v2", "v1"), ("v3", "v2"), ("v4", "v2"), ("v4", "v3"))


def replay_scores() -> dict[str, list[float]]:
    """Per track replay cosine of every model, in the shared order of the 130 held-out tracks."""
    out: dict[str, list[float]] = {}
    order: list[str] | None = None
    for tag, name in REPLAY_FILES.items():
        records = load(REPLAY / name)["records"]
        by_id = {
            r["sample_id"]: r["condition_embedding_replay"]["predicted_codes"]["cosine_mean"] for r in records
        }
        if order is None:
            order = sorted(by_id)
        out[tag] = [by_id[i] for i in order]
    return out


def bootstrap(
    a: Sequence[float], b: Sequence[float], *, draws: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a minus b over paired tracks with a percentile bootstrap 95 percent interval."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for unused in range(n)) / n for unused in range(draws))
    return sum(diffs) / n, means[int(0.025 * draws)], means[int(0.975 * draws) - 1]


def bootstrap_table() -> tuple[Row, list[Row]]:
    scores = replay_scores()
    header = ["Comparison", "Mean difference", "95% interval", "Tracks improved"]
    rows: list[Row] = []
    for a, b in DIFFERENCES:
        mean, lo, hi = bootstrap(scores[a], scores[b])
        improved = sum(x > y for x, y in zip(scores[a], scores[b], strict=True))
        rows.append(
            [
                f"{LABELS_SHORT[a]} minus {LABELS_SHORT[b]}",
                f"{mean:+.4f}",
                f"[{lo:+.4f}, {hi:+.4f}]",
                f"{improved} of {len(scores[a])}",
            ]
        )
    return header, rows


LABELS_SHORT = {"community": "community 41M", "v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4"}


def corpus_table() -> tuple[Row, list[Row]]:
    """Content statistics of the corpus from the cached generation index."""
    paths = sorted(Path(os.environ.get("HF_HOME", str(ROOT.parent / ".hf_home"))).rglob("batch-*.jsonl"))
    rows_in: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            for job in entry["manifest"]["jobs"]:
                if job.get("status") != "succeeded":
                    continue
                lyrics = (job.get("lyrics") or "").strip()
                rows_in.append(
                    {
                        "split": entry.get("dataset_split"),
                        "prompt": (job.get("prompt") or "").strip(),
                        "lyrics": lyrics,
                        "instrumental": lyrics.lower().startswith("[instrumental"),
                        "exact": (job.get("alignment") or {}).get("chunk_stitching") is not None,
                    }
                )
    train = [r for r in rows_in if r["split"] == "train"]
    held = [r for r in rows_in if r["split"] == "holdout"]
    prompts = [r["prompt"] for r in rows_in]
    words = sorted(len(p.split()) for p in prompts)
    lines = sorted(
        len([ln for ln in r["lyrics"].splitlines() if ln.strip()]) for r in rows_in if not r["instrumental"]
    )
    counts = {}
    for p in prompts:
        counts[p] = counts.get(p, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:4]
    train_prompts = {r["prompt"] for r in train}
    train_lyrics = {r["lyrics"] for r in train if not r["instrumental"]}
    header = ["Statistic", "Value"]
    rows = [
        ["tracks (train / held-out)", f"{len(rows_in):,} ({len(train):,} / {len(held)})"],
        [
            "with exact stitching table (train / held-out)",
            f"{sum(r['exact'] for r in train):,} / {sum(r['exact'] for r in held)}",
        ],
        [
            "lyric-bearing / instrumental",
            f"{sum(not r['instrumental'] for r in rows_in):,} / {sum(r['instrumental'] for r in rows_in)}",
        ],
        ["distinct prompts", f"{len(counts)}"],
        ["four most frequent prompts", ", ".join(f"{p} ({c:,})" for p, c in top)],
        ["prompt length in words, median (max)", f"{words[len(words) // 2]} ({words[-1]})"],
        ["lyric length in lines, median (mean)", f"{lines[len(lines) // 2]} ({sum(lines) / len(lines):.1f})"],
        [
            "held-out prompts also used in train",
            f"{sum(r['prompt'] in train_prompts for r in held)} of {len(held)}",
        ],
        [
            "held-out lyrics identical to a training track",
            f"{sum(r['lyrics'] in train_lyrics for r in held if not r['instrumental'])} of {len(held)}",
        ],
    ]
    return header, rows


def provenance_table() -> tuple[Row, list[Row]]:
    summary = load(PUBLISHED / "experiment-summary.json")
    header = ["Version", "Source run", "Revision", "Checkpoint", "Weights sha256"]
    rows = [
        [
            entry["version"],
            entry["source_repo_id"].removeprefix("SimpleTuner/open-rvq-encoder-minimax-music3-"),
            entry["source_revision"][:12],
            entry["source_checkpoint_path"],
            entry["sha256"][:16],
        ]
        for entry in summary["encoders"]
    ]
    metric = summary["metric"]
    rows.append(
        [
            "dataset",
            metric["dataset_repo_id"].removeprefix("bghira/"),
            metric["dataset_revision"][:12],
            metric["split"],
            "",
        ]
    )
    return header, rows


def benchmark_tables() -> list[tuple[str, Row, list[Row]]]:
    """Every suite of the committed benchmark run as its own table."""
    data = load(BENCHMARKS)
    tables: list[tuple[str, Row, list[Row]]] = []
    for suite in dict.fromkeys(result["suite"] for result in data["results"]):
        rows = [result for result in data["results"] if result["suite"] == suite]
        units = next((row["units"] for row in rows if row["units"]), "")
        header = ["Case", "Variant", "ms", "Peak MiB"] + ([units] if units else []) + ["Note"]
        body = [
            [row["case"], row["variant"], f"{row['milliseconds']:.3f}", f"{row['memory']:.1f}"]
            + ([f"{row['throughput']:,.1f}"] if units else [])
            + [row["note"]]
            for row in rows
        ]
        tables.append((suite, header, body))
    return tables


def markdown(header: Row, rows: Sequence[Row]) -> str:
    numeric = [
        "---:" if any(cell.replace(",", "").replace(".", "").isdigit() for cell in column) else "---"
        for column in zip(*rows, strict=True)
    ]
    align = ["---", *numeric[1:]]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def latex(header: Row, rows: Sequence[Row], *, label: str, caption: str) -> str:
    """Booktabs table floated to the top of a page; five or more columns span both columns."""
    wide = len(header) >= 9
    env = "table*" if wide else "table"
    placement = "[t]" if wide else "[tb]"
    spec = "l" + "r" * (len(header) - 1)
    lines = [
        f"\\begin{{{env}}}{placement}",
        "\\centering",
        "\\scriptsize" if wide else "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        "\\adjustbox{max width=\\textwidth}{" if wide else "\\adjustbox{max width=\\columnwidth}{",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(cell.replace("%", "\\%") for cell in header) + " \\\\",
        "\\midrule",
    ]
    lines += [
        " & ".join(cell.replace("%", "\\%").replace("_", "\\_") for cell in row) + " \\\\" for row in rows
    ]
    lines += ["\\bottomrule", "\\end{tabular}", "}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def fill(text: str, name: str, body: str) -> str:
    """Replace the region between the table marker and the end marker, keeping both markers."""
    pattern = re.compile(rf"(<!-- table: {name} -->\n).*?(<!-- end table -->)", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"marker for table {name!r} not found")
    return pattern.sub(lambda match: match.group(1) + body + "\n" + match.group(2), text)


def render_docs() -> None:
    tables = {
        "replay": replay_table(),
        "matched": matched_table(),
        "forcing": forcing_table(),
        "depth": depth_table(),
        "final": final_table(),
        "reproduced": reproduced_table(),
        "provenance": provenance_table(),
    }
    page = ROOT / "docs" / "evaluation.md"
    text = page.read_text(encoding="utf-8")
    for name, (header, rows) in tables.items():
        text = fill(text, name, markdown(header, rows))
    page.write_text(text, encoding="utf-8")

    page = ROOT / "docs" / "benchmarks.md"
    text = page.read_text(encoding="utf-8")
    body = "\n\n".join(
        f"#### {suite}\n\n{markdown(header, rows)}" for suite, header, rows in benchmark_tables()
    )
    page.write_text(fill(text, "benchmarks", body), encoding="utf-8")


def render_tex() -> None:
    TEX.mkdir(parents=True, exist_ok=True)
    captions = {
        "replay": "Condition-replay cosine on the 130 exact-alignment held-out tracks.",
        "matched": "Exact-token metrics at the matched step 17,500 (percent).",
        "forcing": "v4 at the final checkpoint, free-running against teacher-forced (percent).",
        "depth": "Top-1 agreement by codebook at step 17,500 (percent).",
        "final": "Packaged checkpoints on the held-out split (percent).",
        "reproduced": "Published rows against our implementation on the same held-out split.",
        "provenance": "Release provenance.",
        "bootstrap": "Paired differences in condition-replay cosine over the same 130 tracks, "
        "with percentile bootstrap intervals (10,000 resamples).",
        "corpus": "Content statistics of the reverse-distillation corpus.",
    }
    builders = {
        "replay": replay_table,
        "matched": matched_table,
        "forcing": forcing_table,
        "depth": depth_table,
        "final": final_table,
        "reproduced": reproduced_table,
        "provenance": provenance_table,
        "bootstrap": bootstrap_table,
        "corpus": corpus_table,
    }
    for name, build in builders.items():
        header, rows = build()
        (TEX / f"{name}.tex").write_text(
            latex(header, rows, label=name, caption=captions[name]), encoding="utf-8"
        )
    for suite, header, rows in benchmark_tables():
        slug = suite.replace(" ", "_")
        (TEX / f"bench_{slug}.tex").write_text(
            latex(header, rows, label=f"bench_{slug}", caption=f"Benchmark suite: {suite}."), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render result tables from the committed JSON")
    parser.add_argument("--docs", action="store_true", help="fill the marker regions in docs/")
    parser.add_argument("--tex", action="store_true", help="write booktabs tables to paper/tables/")
    args = parser.parse_args()
    if not (args.docs or args.tex):
        args.docs = args.tex = True
    if args.docs:
        render_docs()
    if args.tex:
        render_tex()


if __name__ == "__main__":
    main()
