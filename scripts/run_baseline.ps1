$ErrorActionPreference = 'Stop'

python -m src.train --config configs/baseline.yaml
python -m src.evaluate --config configs/baseline.yaml --weights checkpoints/baseline_best.pt
