"""Generate gallery-ready PNG summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate gallery PNG files.")
    parser.add_argument("--report", required=True, help="Path to reports/summary.json.")
    parser.add_argument("--out", default="artifacts/gallery", help="Output gallery directory.")
    return parser.parse_args()


def _load_report(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_before_after(report: Dict, output_path: Path) -> None:
    by_name = {item["run_name"]: item for item in report.get("runs", [])}
    baseline = by_name.get("baseline", {})
    improved = by_name.get("improved", {})

    metrics = ["best_val_acc", "test_acc"]
    baseline_vals = [float(baseline.get(metric, 0.0) or 0.0) for metric in metrics]
    improved_vals = [float(improved.get(metric, 0.0) or 0.0) for metric in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35

    ax.bar([i - width / 2 for i in x], baseline_vals, width=width, label="baseline")
    ax.bar([i + width / 2 for i in x], improved_vals, width=width, label="improved")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Best Val Acc", "Test Acc"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Before/After: baseline vs improved")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    delta_val = report.get("comparison", {}).get("delta_best_val_acc")
    delta_test = report.get("comparison", {}).get("delta_test_acc")
    text_block = f"Delta val_acc: {delta_val}\nDelta test_acc: {delta_test}"
    ax.text(
        0.98,
        0.05,
        text_block,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _build_deliverables(output_path: Path) -> None:
    required_paths = [
        "artifacts/runs/baseline/history.csv",
        "artifacts/runs/improved/history.csv",
        "artifacts/plots/baseline_curves.png",
        "artifacts/plots/improved_curves.png",
        "artifacts/plots/confusion_baseline.png",
        "artifacts/plots/confusion_improved.png",
        "reports/summary.json",
        "reports/summary.md",
        "checkpoints/baseline_best.pt",
        "checkpoints/improved_best.pt",
    ]
    existence = [(path, Path(path).exists()) for path in required_paths]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    ax.set_title("Deliverables Checklist", fontsize=16, loc="left")

    lines = ["Repository Structure & Checklist", ""]
    lines.append("src/  configs/  scripts/  artifacts/  checkpoints/  reports/  tests/")
    lines.append("")
    for path, exists in existence:
        status = "[x]" if exists else "[ ]"
        lines.append(f"{status} {path}")

    ax.text(
        0.01,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=11,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _load_report(report_path)

    before_after_path = out_dir / "before_after.png"
    deliverables_path = out_dir / "deliverables.png"

    _build_before_after(report, before_after_path)
    _build_deliverables(deliverables_path)

    print(f"Wrote {before_after_path}")
    print(f"Wrote {deliverables_path}")


if __name__ == "__main__":
    main()

