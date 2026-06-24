# MMD-Agent (Unified) — 5-class verdict + 8-label distortion on ReMMDBench

This folder is a **self-contained** agent pipeline. Drop this `mmd-agent/`
folder onto the server, install `requirements.txt`, point it at ReMMDBench and
at the Serper key file, and it runs. No code from any sibling folder is
imported or referenced at runtime — every Python module, prompt template and
shell script lives under `mmd-agent/`.

It integrates the logic of two earlier reference codebases (one focused on the
5-class verdict and one on the 8-label distortion taxonomy) and adapts it to
**ReMMDBench**, but those two codebases are NOT a runtime dependency: their
relevant pieces (prompt structure, signal-to-verdict rule, dataset reader)
have been re-implemented and re-tested directly inside `mmd-agent/`.

**Key property:** every sample is sent through the agent **exactly once**, and
the same run produces **both**:

- A 5-class final verdict over `True / Mostly True / Mixture / Mostly False / False`
  (read from `annotation.json` → `verdict`), and
- A multi-label prediction over 8 distortion categories
  (read from `annotation.json` → `distortion_taxonomy`):

  | Modality | Codes |
  |----------|-------|
  | Text     | T1 Fabrication / T2 Distortion / T3 Misleading Context |
  | Visual   | V1 Synthetic Visual Content / V2 Visual Editing |
  | Cross    | C1 Semantic Inconsistency / C2 Contextual Inconsistency / C3 Pragmatic Inconsistency |

The two earlier reference folders (the 5-class one and the 8-label one) are
**not modified, not copied, and not imported**. They can be ignored or
deleted on the server side; this `mmd-agent/` ships everything it needs.

---

## 1. Directory layout

```
mmd-agent/
├── README.md                                   # this file
├── requirements.txt                            # httpx + tqdm; no torch / sklearn needed
├── configs/                                    # model-specific env files (URL / API key / Serper index / …)
│   ├── gpt52.env
│   ├── qwen36_27b.env
│   ├── qwen35_9b.env
│   └── qwen35_4b.env
├── scripts/                                    # run entry scripts
│   ├── _common.sh                              # shared launcher (paths + sanity checks)
│   ├── run_gpt52.sh                            # bash entry for gpt-5.2
│   ├── run_qwen36_27b.sh                       # bash entry for qwen3.6-27b
│   ├── run_qwen35_9b.sh                        # bash entry for qwen3.5-9b
│   ├── run_qwen35_4b.sh                        # bash entry for qwen3.5-4b
│   └── run_local_smoke.ps1                     # PowerShell entry for local conda-mmd test
├── eval/
│   ├── run_mmd_agent.py                        # main Python entry point
│   ├── prompt_template/
│   │   └── MMD_Agent_Unified/
│   │       ├── textual_veracity_check.txt
│   │       ├── visual_veracity_check.txt
│   │       └── cross_modal_consistency_reason.txt
│   ├── task_datasets/
│   │   └── __init__.py                         # ReMMDBench_Dataset loader (abs-path safe)
│   └── utils/
│       ├── model_utils.py                      # OpenAI-compatible chat-completions client
│       ├── serper_search.py                    # Serper-based key-entity search
│       ├── tools.py                            # 5cls signal parsing + 8label parsing + combiners
│       └── vqa.py                              # main evaluation loop (logs + metrics)
└── outputs/                                    # default output root (per-model subfolders below)
```

The layout intentionally keeps the same `eval/{task_datasets, utils,
prompt_template}/` shape as the earlier reference codebases, so anyone
familiar with the previous MMD-Agent flow can navigate this one immediately.
The single multi-model `run_mmd_agent.py` replaces the older per-model entry
scripts: all four backends (gpt-5.2 / qwen3.6-27b / qwen3.5-9b / qwen3.5-4b)
share the same Python code path and only differ in the `configs/<model>.env`
they source.

---

## 2. Where do I configure the data path, the Serper key file, and the model APIs?

Everything is configurable **outside Python**, in three places:

### 2.1 ReMMDBench root path (`MMD_BENCH_ROOT`)

Edit the top of [`scripts/_common.sh`](scripts/_common.sh):

```bash
: "${MMD_BENCH_ROOT:=/abs/path/to/ReMMDBench}"
```

Replace `/abs/path/to/ReMMDBench` with the **absolute** path on the server
that contains the numbered sample folders (`001/`, `002/`, …). This is the
same value for every model.

You can also override it without editing the file:

```bash
MMD_BENCH_ROOT=/abs/path/to/ReMMDBench bash scripts/run_gpt52.sh
```

### 2.2 Serper API key file (`MMD_SERPER_KEY_FILE`)

Also at the top of [`scripts/_common.sh`](scripts/_common.sh):

```bash
: "${MMD_SERPER_KEY_FILE:=/abs/path/to/serper_api.txt}"
```

The file is expected to contain one Serper key per line (8 keys total).

**Key assignment** (encoded in `configs/<model>.env` via `MMD_SERPER_KEY_INDEX`):

| Line # | Reserved for                |
|--------|-----------------------------|
| 1..4   | OTHER agents that share the pool (not this repo) |
| 5      | `gpt-5.2`                   |
| 6      | `qwen3.6-27b`               |
| 7      | `qwen3.5-9b`                |
| 8      | `qwen3.5-4b`                |

You can override the index for a particular run by exporting
`MMD_SERPER_KEY_INDEX` before invoking the script, but the defaults already
encode the “last 4” rule that the user requested.

### 2.3 Per-model API URL / key

Edit the corresponding file under [`configs/`](configs/):

| File                          | Edit these two fields                              |
|-------------------------------|----------------------------------------------------|
| `configs/gpt52.env`           | `MMD_BASE_URL` (already filled), `MMD_API_KEY` (already filled). |
| `configs/qwen36_27b.env`      | `MMD_BASE_URL` (REPLACE_ME), `MMD_API_KEY` (REPLACE_ME). |
| `configs/qwen35_9b.env`       | `MMD_BASE_URL` (REPLACE_ME), `MMD_API_KEY` (REPLACE_ME). |
| `configs/qwen35_4b.env`       | `MMD_BASE_URL` (REPLACE_ME), `MMD_API_KEY` (REPLACE_ME). |

The GPT-5.2 config ships with the values the user supplied:

```bash
export MMD_MODEL_NAME="gpt-5.2"
export MMD_BASE_URL="http://YOUR_GPT_ENDPOINT/v1"
export MMD_API_KEY="sk-YOUR_GPT_API_KEY_HERE"
export MMD_SERPER_KEY_INDEX=5
```

The three Qwen configs ship with `REPLACE_ME_WITH_..._BASE_URL` / `..._API_KEY`
placeholders. Edit them once on the server with the real URL/keys; the
launcher refuses to run while a `REPLACE_ME_*` placeholder is still in place.

> The launcher does **not** call any other search backend. As requested, the
> `wiki_search.py` from the upstream repo is replaced by `utils/serper_search.py`
> and the only external search API used here is Serper.

---

## 3. Quickstart on the server

```bash
# 0. activate the Python env that has httpx + tqdm
conda activate mmd        # or: pip install -r requirements.txt in any 3.10+ env

# 1. Set absolute paths (edit once, applies to all models).
#    Either edit scripts/_common.sh in place, or export inline before each run.

# 2. Per-model API URL/key (edit configs/<model>.env, especially the qwen ones).

# 3. Run any of the four backends. All output goes under ./outputs/<model_name>/.

bash scripts/run_gpt52.sh
bash scripts/run_qwen36_27b.sh
bash scripts/run_qwen35_9b.sh
bash scripts/run_qwen35_4b.sh
```

You can forward any extra CLI flag straight through to the Python entry point:

```bash
# Smoke run on 5 samples
bash scripts/run_gpt52.sh --max_samples 5

# Full 500-sample run with 8 parallel workers
MMD_NUM_WORKERS=8 bash scripts/run_gpt52.sh

# Only the samples whose folder name contains "010"
bash scripts/run_qwen35_9b.sh --sample_filter 010
```

---

## 4. Quickstart for local testing on Windows (conda `mmd`)

```powershell
# Activate the conda env (assumes Python 3.10+ with httpx and tqdm installed)
conda activate mmd

# Run 2 samples to validate the pipeline end-to-end
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_smoke.ps1 -MaxSamples 2
```

The PS1 script hard-codes the local paths you mentioned during integration:

- `BenchRoot = C:\path\to\ReMMDBench`
- `SerperKeyFile = C:\path\to\serper_api.txt`

…and the GPT-5.2 model URL/API key from the configuration. Pass `-MaxSamples`
to control how many samples you smoke-test (default: 2).

---

## 5. What each run writes to disk

Each invocation produces a self-contained run folder at

```
<output_root>/<model_name>/<run_tag>/
```

with the following structure:

```
<run_tag>/
├── run_config.json            # snapshot of the CLI args (api_key redacted)
├── progress.json              # live, resumable progress file
├── <dataset_name>.jsonl       # one record per completed sample (checkpoint)
├── <dataset_name>.json        # final consolidated, sorted JSON
├── metrics.json               # 5cls verdict metrics + 8-label distortion metrics
└── samples/
    ├── 001/
    │   ├── llm_calls.jsonl    # one line per LLM (or Serper) call: prompt + output + raw payload
    │   └── result.json        # full per-sample result (stage signals, distortions, gt, …)
    ├── 002/
    │   ├── llm_calls.jsonl
    │   └── result.json
    └── …
```

The most useful files at a glance:

- **`metrics.json`** — final report with two top-level keys:
  - `verdict_5cls`: accuracy, macro F1, weighted F1, per-class precision/recall/F1, confusion matrix, GT and pred distributions.
  - `distortion_8label_multi`: macro / micro precision/recall/F1, per-label TP/FP/FN/support, exact-match ratio, Hamming accuracy.
- **`samples/<id>/llm_calls.jsonl`** — the chronological audit log of every
  LLM and Serper call for that sample. Each record carries the full prompt,
  the model's textual output, and the raw API payload, with a UTC timestamp.
- **`samples/<id>/result.json`** — the per-sample answer. Includes the
  parsed stage signals, the parsed distortion codes per stage, the rule
  that produced the final 5cls verdict, the predicted 8-label set, the
  binary GT/pred vectors, and (on failure) a fallback flag plus traceback.
- **`<dataset_name>.jsonl`** — append-only checkpoint. Re-running the same
  `--run_name` will skip samples already present here.

---

## 6. How the 5-class verdict and the 8-label distortion are produced

Each sample goes through the standard three-stage MMD-Agent flow (textual
veracity → visual veracity → cross-modal consistency), but the prompt
templates here have been **extended** so each stage simultaneously emits
both kinds of output:

| Stage | Veracity signal               | Distortion codes |
|-------|-------------------------------|------------------|
| `text` (textual veracity)   | `Finish[TEXT_STRONG_SUPPORT/...REFUTE].` | `Distortions: T1, T2, T3` (any subset, or `NONE`) |
| `image` (visual veracity)   | `Finish[IMAGE_STRONG_SUPPORT/...REFUTE].` | `Distortions: V1, V2` (any subset, or `NONE`) |
| `cross` (text↔image)        | `Finish[CROSS_STRONG_SUPPORT/...REFUTE].` | `Distortions: C1, C2, C3` (any subset, or `NONE`) |

In `utils/tools.py`:

- `parse_stage_signal(...)` extracts the `Finish[...]` bucket for each stage.
- `combine_stage_signals(...)` implements the deterministic rule that maps
  the three per-stage signals to one of the five verdicts: each signal is
  scored on `{+2, +1, 0, -1, -2}`, weighted `0.45 (text) / 0.25 (image) /
  0.30 (cross)`, then run through the rule cascade in the function body.
  This rule is the canonical signal-aggregation rule from the upstream
  MMD-Agent reference implementation, ported verbatim here so the
  5-class numbers stay directly comparable.
- `parse_distortion_codes(...)` reads the `Distortions:` line and keeps only
  codes that are valid for the current stage (so the text stage cannot pollute
  the image distortion set).
- `combine_distortion_predictions(...)` unions the three per-stage code sets
  into the final 8-label prediction.

The same logic also computes the GT binary vector from
`annotation.json::distortion_taxonomy`, so the 8-label metrics are produced
inline without a second pass.

The textual veracity stage still runs an external web search step (one
Serper call per sample), but instead of Wikipedia scraping it now calls
`google.serper.dev/search` with the per-model key described in §2.2. The
search result is logged to `llm_calls.jsonl` like any other call.

---

## 7. Re-running and resuming

If a run is interrupted (network blip, OOM, ctrl-C, …), simply re-run the
same script. The launcher reads `<dataset_name>.jsonl` and skips any sample
whose `sample_name` is already there, so progress is never lost.

To pick an explicit run name (useful for re-using a partial output folder):

```bash
MMD_RUN_NAME=my_run_v1 bash scripts/run_gpt52.sh
```

Re-running with the same `MMD_RUN_NAME` resumes; using a fresh
`MMD_RUN_NAME` starts a new folder.

---

## 8. Customizing prompts or rules

The three stage prompts are plain text in
`eval/prompt_template/MMD_Agent_Unified/`. You can edit any of them and
re-run; the format the parser expects is documented at the top of each file
(specifically: a `Finish[...]` line for the veracity signal, and a
`Distortions: ...` line for the codes). The signal-to-verdict aggregation
rule lives in `utils/tools.py::combine_stage_signals`.
