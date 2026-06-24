# Unified T2-Agent Evaluation on ReMMDBench

- Timestamp: `20260521_192914`
- Provider: `qwen3_5_9b`
- Model: `qwen3.5-9b`
- Base URL: `http://YOUR_QWEN_9B_ENDPOINT/v1`
- Samples: 500
- Wall time: 68889.8s
- Output dir: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914`
- Call log: `/path/to/ReMMD-Agent/t2-agent/records/qwen3_5_9b_20260521_192914/llm_calls.jsonl`

## Five-way verdict (single-label)

- Accuracy (all): 0.2560 (128/500)
- Accuracy (valid only): 0.2560
- Macro Precision / Recall / F1: 0.2398 / 0.2567 / 0.2331
- ERROR predictions: 0

| Verdict | Support | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| True | 100 | 24 | 74 | 76 | 0.2449 | 0.2400 | 0.2424 |
| Mostly True | 99 | 8 | 40 | 91 | 0.1667 | 0.0808 | 0.1088 |
| Mixture | 100 | 21 | 58 | 79 | 0.2658 | 0.2100 | 0.2346 |
| Mostly False | 102 | 17 | 55 | 85 | 0.2361 | 0.1667 | 0.1954 |
| False | 99 | 58 | 145 | 41 | 0.2857 | 0.5859 | 0.3841 |

### 5x5 Confusion Matrix (rows = ground truth, cols = prediction; final col = ERROR)

| GT \\ Pred | True | Mostly True | Mixture | Mostly False | False | ERROR |
| --- | --- | --- | --- | --- | --- | --- |
| True | 24 | 11 | 20 | 13 | 32 | 0 |
| Mostly True | 21 | 8 | 15 | 13 | 42 | 0 |
| Mixture | 19 | 10 | 21 | 15 | 35 | 0 |
| Mostly False | 23 | 10 | 16 | 17 | 36 | 0 |
| False | 11 | 9 | 7 | 14 | 58 | 0 |

## Eight-way distortion taxonomy (multi-label)

- Exact-match accuracy: 0.0320 (16/500)
- Valid exact-match accuracy: 0.0320
- Micro Precision / Recall / F1: 0.4425 / 0.2962 / 0.3549
- Macro Precision / Recall / F1: 0.3883 / 0.2714 / 0.2858
- Average Jaccard: 0.2160
- ERROR predictions: 0

| Label | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T1 Fabrication | 99 | 198 | 49 | 149 | 50 | 0.2475 | 0.4949 | 0.3300 |
| T2 Distortion | 222 | 73 | 42 | 31 | 180 | 0.5753 | 0.1892 | 0.2847 |
| T3 Misleading Context | 164 | 20 | 7 | 13 | 157 | 0.3500 | 0.0427 | 0.0761 |
| V1 Synthetic Visual Content | 145 | 118 | 46 | 72 | 99 | 0.3898 | 0.3172 | 0.3498 |
| V2 Visual Editing | 272 | 225 | 136 | 89 | 136 | 0.6044 | 0.5000 | 0.5473 |
| C1 Semantic Inconsistency | 212 | 68 | 33 | 35 | 179 | 0.4853 | 0.1557 | 0.2357 |
| C2 Contextual Inconsistency | 210 | 218 | 99 | 119 | 111 | 0.4541 | 0.4714 | 0.4626 |
| C3 Pragmatic Inconsistency | 67 | 11 | 0 | 11 | 67 | 0.0000 | 0.0000 | 0.0000 |

## Generated figures

- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_confusion_matrix.png`
- `verdict_confusion_matrix`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_confusion_matrix.pdf`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_per_label_bar.png`
- `verdict_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_verdict_per_label_bar.pdf`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_per_label_bar.png`
- `taxonomy_per_label_bar`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_per_label_bar.pdf`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_label_alignment_heatmap.png`
- `taxonomy_label_alignment_heatmap`: `/path/to/ReMMD-Agent/t2-agent/artifacts/runs/qwen3_5_9b_20260521_192914/figures/qwen3_5_9b_taxonomy_label_alignment_heatmap.pdf`

## First 20 per-sample predictions

| # | Sample | GT Verdict | Pred Verdict | GT Taxonomy | Pred Taxonomy | V | T | Time |
| ---: | --- | --- | --- | --- | --- | :-: | :-: | ---: |
| 1 | 001 | True | True | None | None | Y | Y | 1274.0s |
| 2 | 002 | Mostly True | False | T2 Distortion; V2 Visual Editing; C3 Pragmatic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content |  |  | 920.9s |
| 3 | 003 | True | Mixture | None | V1 Synthetic Visual Content |  |  | 1011.0s |
| 4 | 004 | Mixture | False | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency | V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency |  |  | 1582.8s |
| 5 | 005 | Mostly True | False | T2 Distortion; V2 Visual Editing | T1 Fabrication; V1 Synthetic Visual Content |  |  | 1464.8s |
| 6 | 006 | Mostly False | False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | V2 Visual Editing; C2 Contextual Inconsistency |  |  | 1616.9s |
| 7 | 007 | Mostly False | Mostly False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content | Y |  | 835.3s |
| 8 | 008 | Mostly False | True | T2 Distortion; T3 Misleading Context; V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency | T1 Fabrication; T2 Distortion |  |  | 1475.5s |
| 9 | 009 | Mostly False | False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency | V2 Visual Editing; C2 Contextual Inconsistency |  |  | 1244.6s |
| 10 | 010 | Mixture | Mostly True | T3 Misleading Context; V2 Visual Editing; C2 Contextual Inconsistency | C2 Contextual Inconsistency |  |  | 1595.8s |
| 11 | 011 | False | False | T1 Fabrication; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency | V1 Synthetic Visual Content; C2 Contextual Inconsistency | Y |  | 1069.8s |
| 12 | 012 | False | Mixture | T1 Fabrication; T3 Misleading Context; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency; C3 Pragmatic Inconsistency | V1 Synthetic Visual Content; V2 Visual Editing |  |  | 1456.9s |
| 13 | 013 | Mostly False | Mostly False | T2 Distortion; V2 Visual Editing; C1 Semantic Inconsistency; C3 Pragmatic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content | Y |  | 1051.5s |
| 14 | 014 | Mixture | True | T3 Misleading Context; V2 Visual Editing; C1 Semantic Inconsistency; C2 Contextual Inconsistency | None |  |  | 537.8s |
| 15 | 015 | False | False | T1 Fabrication; V1 Synthetic Visual Content; C1 Semantic Inconsistency | T1 Fabrication; V1 Synthetic Visual Content; C2 Contextual Inconsistency | Y |  | 1628.8s |
| 16 | 016 | Mixture | False | T3 Misleading Context; V1 Synthetic Visual Content; V2 Visual Editing; C2 Contextual Inconsistency | T1 Fabrication; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency |  |  | 1202.7s |
| 17 | 017 | False | False | T1 Fabrication; T2 Distortion; T3 Misleading Context; V1 Synthetic Visual Content; C1 Semantic Inconsistency; C2 Contextual Inconsistency | C2 Contextual Inconsistency | Y |  | 1305.3s |
| 18 | 018 | Mixture | Mostly True | T3 Misleading Context; V1 Synthetic Visual Content; V2 Visual Editing; C2 Contextual Inconsistency | V1 Synthetic Visual Content |  |  | 818.8s |
| 19 | 019 | True | False | None | T1 Fabrication; V2 Visual Editing; C2 Contextual Inconsistency |  |  | 1527.0s |
| 20 | 020 | Mostly False | True | T2 Distortion; V1 Synthetic Visual Content; V2 Visual Editing; C1 Semantic Inconsistency | None |  |  | 717.6s |