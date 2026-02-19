$ErrorActionPreference = 'Stop'

python -m src.evaluate --config configs/baseline.yaml --weights checkpoints/baseline_best.pt
python -m src.evaluate --config configs/improved.yaml --weights checkpoints/improved_best.pt
