# ReMMDBench Evaluation Summary

- Provider: `qwen3_5_4b`
- Model: `qwen3.5-4b`
- Timestamp: `20260521_192914`
- Samples evaluated: 500 / 500
- Wall time: 72683.1s
- Output dir: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201`

## Five-way Verdict (single-label)

- **Accuracy**: 0.2120
- **Macro Precision**: 0.2041
- **Macro Recall**: 0.2120
- **Macro F1**: 0.1992
- ERROR predictions: 0

### Per-class metrics

| Verdict | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| True | 100 | 0.1692 | 0.1100 | 0.1333 |
| Mostly True | 99 | 0.1525 | 0.0909 | 0.1139 |
| Mixture | 100 | 0.1881 | 0.1900 | 0.1891 |
| Mostly False | 102 | 0.2889 | 0.2549 | 0.2708 |
| False | 99 | 0.2216 | 0.4141 | 0.2887 |

### Confusion Matrix (counts; rows = GT, cols = Pred; last col = ERROR)

| GT \\ Pred | True | Mostly True | Mixture | Mostly False | False | ERROR |
| --- | --- | --- | --- | --- | --- | --- |
| True | 11 | 10 | 29 | 12 | 38 | 0 |
| Mostly True | 14 | 9 | 22 | 13 | 41 | 0 |
| Mixture | 19 | 15 | 19 | 16 | 31 | 0 |
| Mostly False | 13 | 12 | 17 | 26 | 34 | 0 |
| False | 8 | 13 | 14 | 23 | 41 | 0 |

## Eight-way Distortion Taxonomy (multi-label)

- **Exact match**: 0.0180 (9/500)
- **Macro Precision**: 0.3795
- **Macro Recall**: 0.2727
- **Macro F1**: 0.2851
- Micro P/R/F1: 0.4058 / 0.2926 / 0.3400
- Average Jaccard: 0.1958
- ERROR predictions: 0

### Per-label metrics

| Label | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 Fabrication | 99 | 198 | 0.2222 | 0.4444 | 0.2963 |
| T2 Distortion | 222 | 76 | 0.4079 | 0.1396 | 0.2081 |
| T3 Misleading Context | 164 | 44 | 0.3636 | 0.0976 | 0.1538 |
| V1 Synthetic Visual Content | 145 | 131 | 0.3206 | 0.2897 | 0.3043 |
| V2 Visual Editing | 272 | 188 | 0.6064 | 0.4191 | 0.4957 |
| C1 Semantic Inconsistency | 212 | 67 | 0.5373 | 0.1698 | 0.2581 |
| C2 Contextual Inconsistency | 210 | 278 | 0.4353 | 0.5762 | 0.4959 |
| C3 Pragmatic Inconsistency | 67 | 21 | 0.1429 | 0.0448 | 0.0682 |

## Figures

- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_confusion_matrix.png`
- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_confusion_matrix.pdf`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_per_label_bar.png`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_per_label_bar.pdf`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_per_label_bar.png`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_per_label_bar.pdf`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_label_alignment_heatmap.png`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_label_alignment_heatmap.pdf`