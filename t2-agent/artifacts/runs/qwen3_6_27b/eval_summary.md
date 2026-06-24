# ReMMDBench Evaluation Summary

- Provider: `qwen3_6_27b`
- Model: `qwen3.6-27b`
- Timestamp: `20260521_192914`
- Samples evaluated: 500 / 500
- Wall time: 105000.4s
- Output dir: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914`

## Five-way Verdict (single-label)

- **Accuracy**: 0.2600
- **Macro Precision**: 0.2407
- **Macro Recall**: 0.2613
- **Macro F1**: 0.2340
- ERROR predictions: 0

### Per-class metrics

| Verdict | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| True | 100 | 0.2903 | 0.2700 | 0.2798 |
| Mostly True | 99 | 0.2609 | 0.1818 | 0.2143 |
| Mixture | 100 | 0.2262 | 0.1900 | 0.2065 |
| Mostly False | 102 | 0.1429 | 0.0588 | 0.0833 |
| False | 99 | 0.2830 | 0.6061 | 0.3859 |

### Confusion Matrix (counts; rows = GT, cols = Pred; last col = ERROR)

| GT \\ Pred | True | Mostly True | Mixture | Mostly False | False | ERROR |
| --- | --- | --- | --- | --- | --- | --- |
| True | 27 | 12 | 17 | 9 | 35 | 0 |
| Mostly True | 14 | 18 | 21 | 6 | 40 | 0 |
| Mixture | 25 | 17 | 19 | 8 | 31 | 0 |
| Mostly False | 20 | 13 | 17 | 6 | 46 | 0 |
| False | 7 | 9 | 10 | 13 | 60 | 0 |

## Eight-way Distortion Taxonomy (multi-label)

- **Exact match**: 0.0420 (21/500)
- **Macro Precision**: 0.4163
- **Macro Recall**: 0.2709
- **Macro F1**: 0.2774
- Micro P/R/F1: 0.4341 / 0.2890 / 0.3470
- Average Jaccard: 0.2221
- ERROR predictions: 0

### Per-label metrics

| Label | Support | Predicted | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T1 Fabrication | 99 | 208 | 0.2452 | 0.5152 | 0.3322 |
| T2 Distortion | 222 | 40 | 0.4500 | 0.0811 | 0.1374 |
| T3 Misleading Context | 164 | 13 | 0.3077 | 0.0244 | 0.0452 |
| V1 Synthetic Visual Content | 145 | 163 | 0.3558 | 0.4000 | 0.3766 |
| V2 Visual Editing | 272 | 233 | 0.6009 | 0.5147 | 0.5545 |
| C1 Semantic Inconsistency | 212 | 82 | 0.5976 | 0.2311 | 0.3333 |
| C2 Contextual Inconsistency | 210 | 184 | 0.4402 | 0.3857 | 0.4112 |
| C3 Pragmatic Inconsistency | 67 | 3 | 0.3333 | 0.0149 | 0.0286 |

## Figures

- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_confusion_matrix.png`
- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_confusion_matrix.pdf`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_per_label_bar.png`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_per_label_bar.pdf`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_per_label_bar.png`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_per_label_bar.pdf`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_label_alignment_heatmap.png`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_label_alignment_heatmap.pdf`