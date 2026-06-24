# ReMMDBench Evaluation Summary

- Provider: `qwen3_5_9b`
- Model: `qwen3.5-9b`
- Timestamp: `20260521_192914`
- Samples evaluated: 500 / 500
- Wall time: 68889.8s
- Output dir: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914`

## Five-way Verdict (single-label)

- **Accuracy**: 0.2560
- **Macro Precision**: 0.2398
- **Macro Recall**: 0.2567
- **Macro F1**: 0.2331
- ERROR predictions: 0

### Per-class metrics

| Verdict | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| True | 100 | 0.2449 | 0.2400 | 0.2424 |
| Mostly True | 99 | 0.1667 | 0.0808 | 0.1088 |
| Mixture | 100 | 0.2658 | 0.2100 | 0.2346 |
| Mostly False | 102 | 0.2361 | 0.1667 | 0.1954 |
| False | 99 | 0.2857 | 0.5859 | 0.3841 |

### Confusion Matrix (counts; rows = GT, cols = Pred; last col = ERROR)

| GT \\ Pred | True | Mostly True | Mixture | Mostly False | False | ERROR |
| --- | --- | --- | --- | --- | --- | --- |
| True | 24 | 11 | 20 | 13 | 32 | 0 |
| Mostly True | 21 | 8 | 15 | 13 | 42 | 0 |
| Mixture | 19 | 10 | 21 | 15 | 35 | 0 |
| Mostly False | 23 | 10 | 16 | 17 | 36 | 0 |
| False | 11 | 9 | 7 | 14 | 58 | 0 |

## Eight-way Distortion Taxonomy (multi-label)

- **Exact match**: 0.0320 (16/500)
- **Macro Precision**: 0.3883
- **Macro Recall**: 0.2714
- **Macro F1**: 0.2858
- Micro P/R/F1: 0.4425 / 0.2962 / 0.3549
- Average Jaccard: 0.2160
- ERROR predictions: 0

### Per-label metrics

| Label | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 Fabrication | 99 | 198 | 0.2475 | 0.4949 | 0.3300 |
| T2 Distortion | 222 | 73 | 0.5753 | 0.1892 | 0.2847 |
| T3 Misleading Context | 164 | 20 | 0.3500 | 0.0427 | 0.0761 |
| V1 Synthetic Visual Content | 145 | 118 | 0.3898 | 0.3172 | 0.3498 |
| V2 Visual Editing | 272 | 225 | 0.6044 | 0.5000 | 0.5473 |
| C1 Semantic Inconsistency | 212 | 68 | 0.4853 | 0.1557 | 0.2357 |
| C2 Contextual Inconsistency | 210 | 218 | 0.4541 | 0.4714 | 0.4626 |
| C3 Pragmatic Inconsistency | 67 | 11 | 0.0000 | 0.0000 | 0.0000 |

## Figures

- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_confusion_matrix.png`
- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_confusion_matrix.pdf`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_per_label_bar.png`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_per_label_bar.pdf`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_per_label_bar.png`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_per_label_bar.pdf`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_label_alignment_heatmap.png`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_label_alignment_heatmap.pdf`