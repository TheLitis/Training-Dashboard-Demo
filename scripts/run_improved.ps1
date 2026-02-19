$ErrorActionPreference = 'Stop'

python -m src.train --config configs/improved.yaml
python -m src.evaluate --config configs/improved.yaml --weights checkpoints/improved_best.pt
