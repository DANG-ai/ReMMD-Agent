# Unified T2-Agent Evaluation on ReMMDBench

- Timestamp: `20260521_192914`
- Provider: `qwen3_5_4b`
- Model: `qwen3.5-4b`
- Base URL: `http://YOUR_QWEN_4B_ENDPOINT/v1`
- Samples: 500
- Wall time: 72683.1s
- Output dir: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201`
- Call log: `/path/to/ReMMD-Agent/t2-agent/records/qwen3_5_4b_20260521_184201/llm_calls.jsonl`

## Five-way verdict (single-label)

- Accuracy (all): 0.2120 (106/500)
- Accuracy (valid only): 0.2120
- Macro Precision / Recall / F1: 0.2041 / 0.2120 / 0.1992
- ERROR predictions: 0

| Verdict | Support | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 100 | 11 | 54 | 89 | 0.1692 | 0.1100 | 0.1333 |
| Mostly True | 99 | 9 | 50 | 90 | 0.1525 | 0.0909 | 0.1139 |
| Mixture | 100 | 19 | 82 | 81 | 0.1881 | 0.1900 | 0.1891 |
| Mostly False | 102 | 26 | 64 | 76 | 0.2889 | 0.2549 | 0.2708 |
| False | 99 | 41 | 144 | 58 | 0.2216 | 0.4141 | 0.2887 |

### 5x5 Confusion Matrix (rows = ground truth, cols = prediction; final col = ERROR)

| GT \\ Pred | True | Mostly True | Mixture | Mostly False | False | ERROR |
| --- | --- | --- | --- | --- | --- | --- |
| True | 11 | 10 | 29 | 12 | 38 | 0 |
| Mostly True | 14 | 9 | 22 | 13 | 41 | 0 |
| Mixture | 19 | 15 | 19 | 16 | 31 | 0 |
| Mostly False | 13 | 12 | 17 | 26 | 34 | 0 |
| False | 8 | 13 | 14 | 23 | 41 | 0 |

## Eight-way distortion taxonomy (multi-label)

- Exact-match accuracy: 0.0180 (9/500)
- Valid exact-match accuracy: 0.0180
- Micro Precision / Recall / F1: 0.4058 / 0.2926 / 0.3400
- Macro Precision / Recall / F1: 0.3795 / 0.2727 / 0.2851
- Average Jaccard: 0.1958
- ERROR predictions: 0

| Label | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 Fabrication | 99 | 198 | 44 | 154 | 55 | 0.2222 | 0.4444 | 0.2963 |
| T2 Distortion | 222 | 76 | 31 | 45 | 191 | 0.4079 | 0.1396 | 0.2081 |
| T3 Misleading Context | 164 | 44 | 16 | 28 | 148 | 0.3636 | 0.0976 | 0.1538 |
| V1 Synthetic Visual Content | 145 | 131 | 42 | 89 | 103 | 0.3206 | 0.2897 | 0.3043 |
| V2 Visual Editing | 272 | 188 | 114 | 74 | 158 | 0.6064 | 0.4191 | 0.4957 |
| C1 Semantic Inconsistency | 212 | 67 | 36 | 31 | 176 | 0.5373 | 0.1698 | 0.2581 |
| C2 Contextual Inconsistency | 210 | 278 | 121 | 157 | 89 | 0.4353 | 0.5762 | 0.4959 |
| C3 Pragmatic Inconsistency | 67 | 21 | 3 | 18 | 64 | 0.1429 | 0.0448 | 0.0682 |

## Generated figures

- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_confusion_matrix.png`
- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_confusion_matrix.pdf`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_per_label_bar.png`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_verdict_per_label_bar.pdf`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_per_label_bar.png`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_per_label_bar.pdf`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_label_alignment_heatmap.png`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_4b_20260521_184201/figures/qwen3_5_4b_taxonomy_label_alignment_heatmap.pdf`

## First 20 per-sample predictions

| # | Sample | GT Verdict | Pred Verdict | GT Taxonomy | Pred Taxonomy | V | T | Time |
| ---: | --- | --- | --- | --- | --- | :-: | :-: | ---: |
| 1 | 001 | True | Mostly False | None | T1 Fabrication; C2 Contextual Inconsistency |  |  | 1157.1s |
| 2 | 002 | Mostly True | False | T2 Distortion; V2 Visual Editing; C3 Pragmatic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency |  |  | 1403.2s |
| 3 | 003 | True | Mixture | None | T1 Fabrication; T2 Distortion; V2 Visual Editing |  |  | 1578.2s |
| 4 | 004 | Mixture | True | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency | T1 Fabrication; C2 Contextual Inconsistency |  |  | 889.0s |
| 5 | 005 | Mostly True | True | T2 Distortion; V2 Visual Editing | None |  |  | 1064.0s |
| 6 | 006 | Mostly False | False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | T2 Distortion; V2 Visual Editing; C2 Contextual Inconsistency |  |  | 1123.0s |
| 7 | 007 | Mostly False | Mostly False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content | Y |  | 1481.0s |
| 8 | 008 | Mostly False | Mostly False | T2 Distortion; T3 Misleading Context; V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency | T1 Fabrication; V1 Synthetic Visual Content; C2 Contextual Inconsistency | Y |  | 1000.6s |
| 9 | 009 | Mostly False | Mixture | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | V1 Synthetic Visual Content; V2 Visual Editing |  |  | 887.9s |
| 10 | 010 | Mixture | Mostly True | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency | T2 Distortion |  |  | 1459.4s |
| 11 | 011 | False | Mostly False | T1 Fabrication; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency | V1 Synthetic Visual Content |  |  | 1021.9s |
| 12 | 012 | False | Mostly False | T1 Fabrication; T3 Misleading Context; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency; C3 Pragmatic Inconsistency | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency |  |  | 1860.1s |
| 13 | 013 | Mostly False | True | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency; C3 Pragmatic Inconsistency | None |  |  | 1551.7s |
| 14 | 014 | Mixture | Mixture | T3 Misleading Context; V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency | V2 Visual Editing | Y |  | 1459.8s |
| 15 | 015 | False | False | T1 Fabrication; V1 Synthetic Visual Content; C1 Semantic Inconsistency | T2 Distortion; V2 Visual Editing; C2 Contextual Inconsistency | Y |  | 1349.5s |
| 16 | 016 | Mixture | False | T3 Misleading Context; V1 Synthetic Visual Content; V2 Visual Editing; C2 Contextual Inconsistency | T1 Fabrication; T3 Misleading Context; C2 Contextual Inconsistency |  |  | 1405.5s |
| 17 | 017 | False | True | T1 Fabrication; T2 Distortion; T3 Misleading Context; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency | None |  |  | 1331.5s |
| 18 | 018 | Mixture | True | T3 Misleading Context; V1 Synthetic Visual Content; V2 Visual Editing; C2 Contextual Inconsistency | T1 Fabrication |  |  | 933.2s |
| 19 | 019 | True | Mixture | None | V1 Synthetic Visual Content |  |  | 2260.9s |
| 20 | 020 | Mostly False | False | T2 Distortion; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency | T1 Fabrication; C2 Contextual Inconsistency |  |  | 1657.6s |