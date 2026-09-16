import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results" / "published"
BENCHMARKS = ROOT / "benchmarks" / "benchmarks.json"
OUT = ROOT / "paper" / "figures"

# Fixed categorical order, validated for colour vision deficiency; identity never changes with
# the number of series drawn. Text always wears ink, never a series colour.
COLOR = {
    "v1": "#2a78d6",
    "v2": "#eb6834",
    "v3": "#1baf7a",
    "v4": "#eda100",
    "v4 teacher forced": "#4a3aa7",
    "community": "#e34948",
}
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
LABEL = {"v1": "v1, 41M", "v2": "v2, 155M", "v3": "v3, 155M + MERT", "v4": "v4, 169M + depth"}
VARIANTS = ("v1", "v2", "v3", "v4")
COLUMN = 3.35
DOUBLE = 7.0


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "serif",
            "axes.edgecolor": INK_SOFT,
            "axes.labelcolor": INK,
            "axes.titlesize": 8,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
        }
    )


def checkpoints(tag: str) -> list[dict[str, Any]]:
    rows = load(PUBLISHED / tag / "evaluation-metrics.json")["checkpoints"]
    return sorted(rows, key=lambda row: row["step"])


def history(tag: str) -> list[dict[str, Any]]:
    rows = load(PUBLISHED / tag / "evaluation-metrics.json")["training_history"]
    return sorted((row for row in rows if row["type"] == "train"), key=lambda row: row["step"])


def end_label(ax: Axes, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(
        text, (x, y), xytext=(3, 0), textcoords="offset points", color=color, fontsize=6.5, va="center"
    )


def figure_checkpoints() -> None:
    """Holdout metrics over the 36 checkpoints of every run; the conditional v4 loss has its own panel."""
    fig, axes = plt.subplots(4, 1, figsize=(COLUMN, 6.2))
    axes = axes.reshape(2, 2)
    for tag in VARIANTS:
        rows = checkpoints(tag)
        steps = [row["step"] for row in rows]
        axes[0, 0].plot(
            steps, [100 * row["semantic_top1"] for row in rows], color=COLOR[tag], label=LABEL[tag]
        )
        axes[0, 1].plot(
            steps, [100 * row["acoustic_top1"] for row in rows], color=COLOR[tag], label=LABEL[tag]
        )
        target = axes[1, 1] if tag == "v4" else axes[1, 0]
        target.plot(steps, [row["loss"] for row in rows], color=COLOR[tag], label=LABEL[tag])
    rows = checkpoints("v4")
    axes[0, 1].plot(
        [row["step"] for row in rows],
        [100 * row["teacher_forced_acoustic_top1"] for row in rows],
        color=COLOR["v4 teacher forced"],
        linestyle="--",
        label="v4, teacher-forced",
    )
    axes[0, 0].set_title("semantic top-1 (percent)")
    axes[0, 1].set_title("acoustic top-1 (percent)")
    axes[1, 0].set_title("held-out loss, independent heads")
    axes[1, 1].set_title("held-out loss, v4 (conditional on earlier codes)")
    for ax in axes.flat:
        ax.set_xlabel("optimizer step")
        ax.set_xlim(0, 18_000)
    axes[0, 1].legend(loc="upper left")
    axes[1, 0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "checkpoints.pdf")
    plt.close(fig)


def figure_depth_gradient() -> None:
    """Top 1 by codebook at the matched step. The single figure that motivates the depth decoder."""
    matched = {
        tag: next(row for row in checkpoints(tag) if row["checkpoint"] == "checkpoint-17500")
        for tag in VARIANTS
    }
    fig, ax = plt.subplots(figsize=(COLUMN, 2.6))
    books = list(range(1, 8))
    series = [
        (tag, LABEL[tag], [100 * matched[tag][f"head_{b}_top1"] for b in books], "-") for tag in VARIANTS
    ]
    series.append(
        (
            "v4 teacher forced",
            "v4, teacher-forced",
            [100 * matched["v4"][f"teacher_forced_head_{b}_top1"] for b in books],
            "--",
        )
    )
    for key, label, values, dash in series:
        ax.plot(books, values, color=COLOR[key], linestyle=dash, marker="o", label=label)
        if key.startswith("v4"):
            end_label(ax, books[-1], values[-1], f"{values[-1]:.1f}", COLOR[key])
    ax.set_xlabel("acoustic codebook")
    ax.set_ylabel("top-1 agreement (percent)")
    ax.set_xticks(books)
    ax.set_xlim(0.8, 7.9)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "depth_gradient.pdf")
    plt.close(fig)


def figure_replay() -> None:
    """Replay cosine distribution over the 130 holdout tracks: min to max, 5th to 95th, mean."""
    data = load(PUBLISHED / "condition-replay-aggregate.json")
    names = {
        "serveurperso-v1": ("community 41M", COLOR["community"]),
        "simpletuner-v1": (LABEL["v1"], COLOR["v1"]),
        "simpletuner-v2": (LABEL["v2"], COLOR["v2"]),
        "simpletuner-v3": (LABEL["v3"], COLOR["v3"]),
        "simpletuner-v4": (LABEL["v4"], COLOR["v4"]),
    }
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    rows = [(names[m["model_id"]], m["predicted_code_condition_cosine"]) for m in data["models"]]
    rows.append((("true codes (control)", INK_SOFT), data["models"][-1]["true_code_control"]))
    for index, ((unused, color), stats) in enumerate(rows):
        y = len(rows) - 1 - index
        ax.plot([stats["min"], stats["max"]], [y, y], color=color, linewidth=0.8)
        ax.plot(
            [stats["quantiles"]["0.05"], stats["quantiles"]["0.95"]],
            [y, y],
            color=color,
            linewidth=4,
            solid_capstyle="butt",
        )
        ax.plot(
            [stats["mean"]],
            [y],
            marker="o",
            color="white",
            markeredgecolor=color,
            markersize=5,
            markeredgewidth=1.4,
        )
        ax.text(
            stats["max"] + 0.008,
            y,
            f"{stats['mean']:.4f}",
            ha="left",
            va="center",
            fontsize=6.5,
            color=INK_SOFT,
        )
    ax.set_yticks(range(len(rows)), [label for (label, unused), stats in reversed(rows)])
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.set_xlim(0.58, 1.07)
    ax.set_xlabel("condition-replay cosine")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "replay.pdf")
    plt.close(fig)


def smooth(values: list[float], window: int = 15) -> list[float]:
    out = []
    for index in range(len(values)):
        lo = max(0, index - window + 1)
        out.append(sum(values[lo : index + 1]) / (index + 1 - lo))
    return out


def figure_history() -> None:
    """Training loss and the learning rate of the width scaled parameter group, from the run logs."""
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 4.2))
    for tag in VARIANTS:
        rows = history(tag)
        steps = [row["step"] for row in rows]
        axes[0].plot(steps, smooth([row["loss"] for row in rows]), color=COLOR[tag], label=LABEL[tag])
        if tag in ("v1", "v4"):
            peak = max(row["learning_rate"] for row in rows)
            axes[1].plot(
                steps, [row["learning_rate"] / peak for row in rows], color=COLOR[tag], label=LABEL[tag]
            )
    axes[0].set_title("training loss (running mean of 15 logs)")
    axes[0].set_xlabel("optimizer step")
    axes[0].legend()
    axes[1].set_title("learning rate relative to its peak")
    axes[1].set_xlabel("optimizer step")
    axes[1].legend(loc="upper right")
    for ax in axes:
        ax.set_xlim(0, 18_000)
    fig.tight_layout()
    fig.savefig(OUT / "history.pdf")
    plt.close(fig)


def figure_benchmarks() -> None:
    """Attention kernel latency at the two model shapes, and host side assembly of the pooling operator."""
    data = load(BENCHMARKS)["results"]
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN, 4.2))
    cases = [
        ("temporal (b16, 17h, len 128) forward", "temporal\nforward"),
        ("temporal (b16, 17h, len 128) backward", "temporal\nbackward"),
        ("depth (b2048, 8h, len 8) forward", "depth\nforward"),
        ("depth (b2048, 8h, len 8) backward", "depth\nbackward"),
    ]
    width = 0.38
    for offset, (variant, label, color) in enumerate(
        (("reference", "unfused reference", COLOR["v1"]), ("fused sdpa", "fused kernel", COLOR["v2"]))
    ):
        values = [
            next(r["milliseconds"] for r in data if r["case"] == case and r["variant"] == variant)
            for case, unused in cases
        ]
        xs = [index + (offset - 0.5) * width for index in range(len(cases))]
        axes[0].bar(xs, values, width=width - 0.04, color=color, label=label)
        for x, value in zip(xs, values, strict=True):
            axes[0].text(x, value, f"{value:.2f}", ha="center", va="bottom", fontsize=6, color=INK_SOFT)
    axes[0].set_xticks(range(len(cases)), [label for unused, label in cases])
    axes[0].set_ylabel("milliseconds")
    axes[0].set_title("attention latency, bf16, RTX 3090")
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="x", visible=False)
    axes[0].set_axisbelow(True)

    collate = [
        ("collate b16 x 128 frames x 441 latents", "batch 16"),
        ("collate b64 x 128 frames x 441 latents", "batch 64"),
    ]
    for offset, (variant, label, color) in enumerate(
        (("dense matrix", "dense matrix", COLOR["v1"]), ("segment index", "segment index", COLOR["v2"]))
    ):
        values = [
            next(r["milliseconds"] for r in data if r["case"] == case and r["variant"] == variant)
            for case, unused in collate
        ]
        xs = [index + (offset - 0.5) * width for index in range(len(collate))]
        axes[1].bar(xs, values, width=width - 0.04, color=color, label=label)
        for x, value in zip(xs, values, strict=True):
            axes[1].text(x, value, f"{value:.1f}", ha="center", va="bottom", fontsize=6, color=INK_SOFT)
    axes[1].set_xticks(range(len(collate)), [label for unused, label in collate])
    axes[1].set_ylabel("milliseconds per batch")
    axes[1].set_title("host-side pooling operator assembly")
    axes[1].legend(loc="upper left")
    axes[1].grid(axis="x", visible=False)
    axes[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "benchmarks.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the paper figures from the committed results")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    figure_checkpoints()
    figure_depth_gradient()
    figure_replay()
    figure_history()
    figure_benchmarks()
    print("wrote", sorted(path.name for path in args.out.glob("*.pdf")))


if __name__ == "__main__":
    main()
