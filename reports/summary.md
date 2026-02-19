# Training Dashboard Report

Generated: 2026-02-19T23:13:02.450597+00:00

## Baseline vs Improved

| Run | Model | Epochs | Best Val Acc | Test Acc | Train Time (s) | Params | Ckpt (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | baseline | 4 | 0.702976 | 0.704785 | 29.954 | 620362 | 2.372 |
| baseline_smoke | baseline | 1 | 0.207031 | None | 13.107 | 620362 | 2.372 |
| improved | improved | 6 | 0.823874 | 0.823047 | 53.679 | 1215562 | 4.66 |
| improved_smoke | improved | 1 | 0.199219 | None | 13.684 | 1215562 | 4.661 |

## Deltas

- delta_best_val_acc: 0.120898
- delta_test_acc: 0.118262
