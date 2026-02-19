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
