"""Build JSON and Markdown comparison reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build run comparison report.")
    parser.add_argument("--runs", default="artifacts/runs", help="Directory with run outputs.")
    parser.add_argument("--out", default="reports", help="Output directory for reports.")
    return parser.parse_args()


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_size_mb(checkpoint_path: str | None) -> float | None:
    if not checkpoint_path:
        return None
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 3)


def collect_runs(runs_dir: Path) -> List[Dict]:
    records: List[Dict] = []
    if not runs_dir.exists():
        return records

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metrics = _load_json(run_dir / "metrics.json")
        eval_metrics = _load_json(run_dir / "eval.json")
        if not metrics:
            continue

        record = {
            "run_name": metrics.get("run_name", run_dir.name),
            "model_name": metrics.get("model_name", run_dir.name),
            "epochs": metrics.get("epochs"),
            "num_parameters": metrics.get("num_parameters"),
            "best_val_acc": metrics.get("best_val_acc"),
            "test_acc": eval_metrics.get("test_acc"),
            "test_loss": eval_metrics.get("test_loss"),
            "training_seconds": metrics.get("training_seconds"),
            "checkpoint_path": metrics.get("checkpoint_path"),
            "checkpoint_size_mb": _checkpoint_size_mb(metrics.get("checkpoint_path")),
            "history_path": metrics.get("history_path"),
            "curves_path": eval_metrics.get("curves_path"),
            "confusion_path": eval_metrics.get("confusion_path"),
        }
        records.append(record)
    return records


def build_comparison(records: List[Dict]) -> Dict:
    by_name = {item["run_name"]: item for item in records}
    baseline = by_name.get("baseline")
    improved = by_name.get("improved")

    comparison = {"baseline": baseline, "improved": improved}
    if baseline and improved:
        base_test = baseline.get("test_acc")
        imp_test = improved.get("test_acc")
        base_val = baseline.get("best_val_acc")
        imp_val = improved.get("best_val_acc")

        if isinstance(base_test, (int, float)) and isinstance(imp_test, (int, float)):
            comparison["delta_test_acc"] = round(float(imp_test) - float(base_test), 6)
        if isinstance(base_val, (int, float)) and isinstance(imp_val, (int, float)):
            comparison["delta_best_val_acc"] = round(float(imp_val) - float(base_val), 6)
    return comparison


def build_markdown(payload: Dict) -> str:
    lines = [
        "# Training Dashboard Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Baseline vs Improved",
        "",
        "| Run | Model | Epochs | Best Val Acc | Test Acc | Train Time (s) | Params | Ckpt (MB) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for record in payload["runs"]:
        lines.append(
            "| {run_name} | {model_name} | {epochs} | {best_val_acc} | {test_acc} | {training_seconds} | {num_parameters} | {checkpoint_size_mb} |".format(
                run_name=record.get("run_name"),
                model_name=record.get("model_name"),
                epochs=record.get("epochs"),
                best_val_acc=record.get("best_val_acc"),
                test_acc=record.get("test_acc"),
                training_seconds=record.get("training_seconds"),
                num_parameters=record.get("num_parameters"),
                checkpoint_size_mb=record.get("checkpoint_size_mb"),
            )
        )

    comparison = payload.get("comparison", {})
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            f"- delta_best_val_acc: {comparison.get('delta_best_val_acc')}",
            f"- delta_test_acc: {comparison.get('delta_test_acc')}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect_runs(runs_dir)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": records,
        "comparison": build_comparison(records),
    }

    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md_path.write_text(build_markdown(payload), encoding="utf-8")

    print(f"Wrote {summary_json_path}")
    print(f"Wrote {summary_md_path}")


if __name__ == "__main__":
    main()

