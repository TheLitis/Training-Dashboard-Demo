"""Artifact contract checks from project requirements."""

from __future__ import annotations

from pathlib import Path


REQUIRED_ARTIFACTS = [
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


def test_required_artifacts_exist():
    missing = [path for path in REQUIRED_ARTIFACTS if not Path(path).exists()]
    assert not missing, f"Missing required artifacts: {missing}"

