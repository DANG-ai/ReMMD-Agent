# Unified T2-Agent Evaluation on ReMMDBench

- Timestamp: `20260521_192914`
- Provider: `qwen3_6_27b`
- Model: `qwen3.6-27b`
- Base URL: `http://YOUR_QWEN_27B_ENDPOINT/v1`
- Samples: 500
- Wall time: 105000.4s
- Output dir: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914`
- Call log: `/path/to/ReMMD-Agent/t2-agent/records/qwen3_6_27b_20260521_192914/llm_calls.jsonl`

## Five-way verdict (single-label)

- Accuracy (all): 0.2600 (130/500)
- Accuracy (valid only): 0.2600
- Macro Precision / Recall / F1: 0.2407 / 0.2613 / 0.2340
- ERROR predictions: 0

| Verdict | Support | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 100 | 27 | 66 | 73 | 0.2903 | 0.2700 | 0.2798 |
| Mostly True | 99 | 18 | 51 | 81 | 0.2609 | 0.1818 | 0.2143 |
| Mixture | 100 | 19 | 65 | 81 | 0.2262 | 0.1900 | 0.2065 |
| Mostly False | 102 | 6 | 36 | 96 | 0.1429 | 0.0588 | 0.0833 |
| False | 99 | 60 | 152 | 39 | 0.2830 | 0.6061 | 0.3859 |

### 5x5 Confusion Matrix (rows = ground truth, cols = prediction; final col = ERROR)

| GT \\ Pred | True | Mostly True | Mixture | Mostly False | False | ERROR |
| --- | --- | --- | --- | --- | --- | --- |
| True | 27 | 12 | 17 | 9 | 35 | 0 |
| Mostly True | 14 | 18 | 21 | 6 | 40 | 0 |
| Mixture | 25 | 17 | 19 | 8 | 31 | 0 |
| Mostly False | 20 | 13 | 17 | 6 | 46 | 0 |
| False | 7 | 9 | 10 | 13 | 60 | 0 |

## Eight-way distortion taxonomy (multi-label)

- Exact-match accuracy: 0.0420 (21/500)
- Valid exact-match accuracy: 0.0420
- Micro Precision / Recall / F1: 0.4341 / 0.2890 / 0.3470
- Macro Precision / Recall / F1: 0.4163 / 0.2709 / 0.2774
- Average Jaccard: 0.2221
- ERROR predictions: 0

| Label | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 Fabrication | 99 | 208 | 51 | 157 | 48 | 0.2452 | 0.5152 | 0.3322 |
| T2 Distortion | 222 | 40 | 18 | 22 | 204 | 0.4500 | 0.0811 | 0.1374 |
| T3 Misleading Context | 164 | 13 | 4 | 9 | 160 | 0.3077 | 0.0244 | 0.0452 |
| V1 Synthetic Visual Content | 145 | 163 | 58 | 105 | 87 | 0.3558 | 0.4000 | 0.3766 |
| V2 Visual Editing | 272 | 233 | 140 | 93 | 132 | 0.6009 | 0.5147 | 0.5545 |
| C1 Semantic Inconsistency | 212 | 82 | 49 | 33 | 163 | 0.5976 | 0.2311 | 0.3333 |
| C2 Contextual Inconsistency | 210 | 184 | 81 | 103 | 129 | 0.4402 | 0.3857 | 0.4112 |
| C3 Pragmatic Inconsistency | 67 | 3 | 1 | 2 | 66 | 0.3333 | 0.0149 | 0.0286 |

## Generated figures

- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_confusion_matrix.png`
- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_confusion_matrix.pdf`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_per_label_bar.png`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_verdict_per_label_bar.pdf`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_per_label_bar.png`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_per_label_bar.pdf`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_label_alignment_heatmap.png`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_6_27b_20260521_192914/figures/qwen3_6_27b_taxonomy_label_alignment_heatmap.pdf`

## First 20 per-sample predictions

| # | Sample | GT Verdict | Pred Verdict | GT Taxonomy | Pred Taxonomy | V | T | Time |
| ---: | --- | --- | --- | --- | --- | :-: | :-: | ---: |
| 1 | 001 | True | False | None | T1 Fabrication |  |  | 3004.4s |
| 2 | 002 | Mostly True | False | T2 Distortion; V2 Visual Editing; C3 Pragmatic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content |  |  | 2368.4s |
| 3 | 003 | True | True | None | None | Y | Y | 1182.9s |
| 4 | 004 | Mixture | False | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency | T1 Fabrication; V2 Visual Editing; C2 Contextual Inconsistency |  |  | 2782.0s |
| 5 | 005 | Mostly True | Mixture | T2 Distortion; V2 Visual Editing | V2 Visual Editing |  |  | 1935.1s |
| 6 | 006 | Mostly False | False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | V2 Visual Editing; C2 Contextual Inconsistency |  |  | 2616.4s |
| 7 | 007 | Mostly False | False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | T1 Fabrication; V2 Visual Editing |  |  | 2179.0s |
| 8 | 008 | Mostly False | False | T2 Distortion; T3 Misleading Context; V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency | T1 Fabrication; V1 Synthetic Visual Content; C2 Contextual Inconsistency |  |  | 1110.5s |
| 9 | 009 | Mostly False | Mixture | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | V2 Visual Editing |  |  | 1884.9s |
| 10 | 010 | Mixture | Mixture | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency | C2 Contextual Inconsistency | Y |  | 548.8s |
| 11 | 011 | False | Mostly True | T1 Fabrication; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency | T2 Distortion; V1 Synthetic Visual Content; C1 Semantic Inconsistency |  |  | 3750.3s |
| 12 | 012 | False | True | T1 Fabrication; T3 Misleading Context; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency; C3 Pragmatic Inconsistency | V2 Visual Editing |  |  | 905.0s |
| 13 | 013 | Mostly False | True | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency; C3 Pragmatic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content |  |  | 2659.8s |
| 14 | 014 | Mixture | False | T3 Misleading Context; V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency | T2 Distortion; C2 Contextual Inconsistency |  |  | 3323.8s |
| 15 | 015 | False | False | T1 Fabrication; V1 Synthetic Visual Content; C1 Semantic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content; C2 Contextual Inconsistency | Y |  | 2606.3s |
| 16 | 016 | Mixture | Mixture | T3 Misleading Context; V1 Synthetic Visual Content; V2 Visual Editing; C2 Contextual Inconsistency | T1 Fabrication; V2 Visual Editing; C2 Contextual Inconsistency | Y |  | 2443.0s |
| 17 | 017 | False | False | T1 Fabrication; T2 Distortion; T3 Misleading Context; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency | V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency | Y |  | 1257.9s |
| 18 | 018 | Mixture | True | T3 Misleading Context; V1 Synthetic Visual Content; V2 Visual Editing; C2 Contextual Inconsistency | None |  |  | 842.7s |
| 19 | 019 | True | False | None | V2 Visual Editing; C2 Contextual Inconsistency |  |  | 2808.6s |
| 20 | 020 | Mostly False | Mostly True | T2 Distortion; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency | V2 Visual Editing; C2 Contextual Inconsistency |  |  | 2897.7s |