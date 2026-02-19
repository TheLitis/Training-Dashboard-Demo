# Training Dashboard Demo

Engineering-style mini project for image classification on CIFAR-10:
config -> train -> metrics -> report.

## Setup

```powershell
python -m pip install --upgrade pip
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m src.train --config configs/baseline.yaml
python -m src.train --config configs/improved.yaml
python -m src.evaluate --config configs/baseline.yaml --weights checkpoints/baseline_best.pt
python -m src.evaluate --config configs/improved.yaml --weights checkpoints/improved_best.pt
python -m src.infer --weights checkpoints/improved_best.pt --image path\\to\\image.png --topk 5
python -m src.make_report --runs artifacts/runs --out reports
python -m src.make_gallery --report reports/summary.json --out artifacts/gallery
```

Helper scripts:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_baseline.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_improved.ps1
powershell -ExecutionPolicy Bypass -File scripts/eval_all.ps1
```

## Actual Results

Environment used for this run:

- GPU: NVIDIA GeForce RTX 3070
- PyTorch: `2.10.0+cu128`
- Dataset: CIFAR-10

Baseline vs improved (from `reports/summary.json`):

| Run | Epochs | Best Val Acc | Test Acc | Test Loss | Train Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 4 | 0.702976 | 0.704785 | 0.840364 | 29.954 |
| improved | 6 | 0.823874 | 0.823047 | 0.513845 | 53.679 |

Delta:

- `delta_best_val_acc`: `+0.120898`
- `delta_test_acc`: `+0.118262`

## Deliverables

Required output files are generated:

- `artifacts/runs/baseline/history.csv`
- `artifacts/runs/improved/history.csv`
- `artifacts/plots/baseline_curves.png`
- `artifacts/plots/improved_curves.png`
- `artifacts/plots/confusion_baseline.png`
- `artifacts/plots/confusion_improved.png`
- `reports/summary.json`
- `reports/summary.md`
- `artifacts/gallery/before_after.png`
- `artifacts/gallery/deliverables.png`
- `checkpoints/baseline_best.pt`
- `checkpoints/improved_best.pt`

## Test

```powershell
pytest -q
```

## Structure

```text
src/           # train/eval/infer/report/gallery logic
configs/       # baseline and improved run configs
scripts/       # helper powershell scripts
artifacts/     # run logs, plots, gallery png
checkpoints/   # saved model weights
reports/       # markdown/json summaries
tests/         # smoke tests
```
