# ReMMD Multimodal Misinformation Detection Agent

End-to-end agent system for the **ReMMDBench** benchmark.

This is the repository root. The actual project code lives in [`agent/`](./agent/).

```
ReMMD-agent/
├── ReMMDBench/                    # benchmark — 500 multilingual posts (NOT modified)
├── rag_database/                  # 7,709 evidence items + sample→evidence index
│   └── corpus.jsonl               # one row per evidence item; 5 evidence types
├── 一级标签.txt                   # L1 (5 verdicts) prior document
├── 二级标签.docx                  # L2 (8 distortion patterns) prior document
├── 结构图.png                     # workflow diagram
└── agent/                         # ← project code, see agent/README.md
    ├── configs/
    │   ├── default.yaml            # GPT-5.2 stack (legacy / production)
    │   └── qwen_v2.yaml            # NEW: qwen3.5-9b stack with split atomic + v2 prompts + selective L2 union + concurrency 125
    ├── prompts/                    # legacy and v2 prompts COEXIST (legacy never modified)
    │   ├── atomic_parse.txt          # legacy single-call atomic parser
    │   ├── image_analyze.txt         # legacy qwen image analyzer
    │   ├── text_analyze.txt          # legacy qwen text analyzer
    │   ├── final_judge.txt           # legacy qwen final judge
    │   ├── final_judge_gpt.txt       # GPT-5.2 final judge (legacy)
    │   ├── image_analyze_gpt.txt     # GPT-5.2 image analyzer (legacy)
    │   ├── text_analyze_gpt.txt      # GPT-5.2 text analyzer (legacy)
    │   ├── image_atom_parse.txt      # NEW v2 image-only atomic parser
    │   ├── cross_modal_atom_parse.txt # NEW v2 cross-modal atomic parser
    │   ├── text_atom_parse.txt       # NEW v2 text-only atomic parser
    │   ├── image_analyze_v2.txt      # NEW v2 calibrated image analyzer (qwen)
    │   ├── text_analyze_v2.txt       # NEW v2 calibrated text analyzer (qwen)
    │   └── final_judge_v2.txt        # NEW v2 qwen judge: 5-shot + NEVER/NO anti-bias + tie-breakers
    ├── src/                        # core modules (llm / embedder / rag / atomic / image_analyzer
    │                                # / text_analyzer / final_judge / pipeline / runner / metrics …)
    │                                # Pipeline supports atomic_mode in {"single", "multi", "split"}
    │                                # and analyzer_union_policy in {"off", "always", "selective"}.
    ├── scripts/
    │   ├── prepare_corpus.py       # idempotent rename img_ctx_*  →  ctx_*  (already applied)
    │   ├── build_rag_index.py      # one-time embed all 7,709 evidence items
    │   ├── smoke_rag.py            # sanity-check RAG retrieval on a few samples
    │   ├── run_eval.py             # main eval entry point (resumable)
    │   ├── ensemble_judge.py       # majority-vote ensemble of N judges per sample
    │   └── recompute_from_raw.py   # recompute metrics from on-disk raw judge responses (supports --policy)
    ├── runs/                       # per-evaluation output directories
    └── rag_index/                  # cached corpus embeddings (.npy)
```

See [`agent/README.md`](./agent/README.md) for the full documentation: pipeline architecture, split / multi atomic parsing, label-definition-based consistency rules, per-run artifact layout, evaluation metrics, and a candid analysis of qwen3.5-9b's calibration limits on this benchmark.

## Quick start

```bash
source /path/to/conda/etc/profile.d/conda.sh && conda activate mmd
cd agent
pip install -r requirements.txt
python scripts/build_rag_index.py
python scripts/smoke_rag.py --sample-ids 001 002 023

# Qwen v2 (split atomic + v2 prompts + selective L2 union + concurrency 125 — 500 samples in ~9 min)
python scripts/run_eval.py --config configs/qwen_v2.yaml --tag qwen_v2_full500 --limit 500 --concurrency 125

# Legacy GPT-5.2 stack
python scripts/run_eval.py --config configs/default.yaml --tag gpt52_full500
```

## Pre-applied data fix

The agent expects `evidence_id` values in the corpus to look like `ctx_<sample>_<idx>` (no `img_` prefix). We applied this rename already on `rag_database/corpus.jsonl` and `rag_database/sample_to_evidence.json`. Backup files `*.bak` are next to the originals. To re-apply (idempotent), run:

```bash
python agent/scripts/prepare_corpus.py
```
