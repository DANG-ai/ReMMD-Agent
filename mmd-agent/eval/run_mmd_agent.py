"""Unified MMD-Agent entry point.

Runs the 3-stage agent (textual veracity, visual veracity, cross-modal
consistency) ONCE per sample and writes BOTH:
  * a 5-class verdict prediction (True / Mostly True / Mixture / Mostly False / False)
  * a multi-label 8-category distortion taxonomy prediction
    (T1/T2/T3 + V1/V2 + C1/C2/C3).

Designed to be model-agnostic. It calls any OpenAI-compatible
``/chat/completions`` endpoint, so the same script drives gpt-5.2 today and
qwen3.6-27b / qwen3.5-9b / qwen3.5-4b tomorrow with no code change - just
pass a different ``--model_name``, ``--base_url`` and ``--api_key``.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

# Make `task_datasets` and `utils` importable when this script is invoked from
# anywhere (the run scripts cd to `mmd-agent/eval` before launching).
EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from task_datasets import ReMMDBench_Dataset  # noqa: E402
from utils.serper_search import select_serper_key  # noqa: E402
from utils.vqa import evaluate_VQA_MMD_Agent_Unified  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified MMD-Agent (5cls verdict + 8-label distortion) "
            "over ReMMDBench with a configurable OpenAI-compatible backend."
        )
    )

    # --- benchmark / dataset ---
    parser.add_argument(
        "--sampled_root", "--sampled-root",
        dest="sampled_root",
        type=str,
        required=True,
        help="ABSOLUTE path to the ReMMDBench root directory (contains 001/, 002/, ...).",
    )
    parser.add_argument(
        "--prompt_root", "--prompt-root",
        dest="prompt_root",
        type=str,
        default=str(EVAL_ROOT / "prompt_template" / "MMD_Agent_Unified"),
        help="Directory holding textual_veracity_check.txt / visual_veracity_check.txt / cross_modal_consistency_reason.txt.",
    )
    parser.add_argument("--dataset_name", type=str, default="remmdbench")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="If >0, only run the first --max_samples samples (deterministic shuffle).")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--sample_filter", type=str, default="",
                        help="Only keep samples whose directory path contains this substring.")
    parser.add_argument("--seed", type=int, default=42)

    # --- output ---
    parser.add_argument("--answer_path", type=str, default=str(EVAL_ROOT.parent / "outputs"),
                        help="Where per-model run folders are created.")
    parser.add_argument("--run_name", type=str, default="",
                        help="Sub-folder name under <answer_path>/<model_name>. Default: timestamp.")
    parser.add_argument(
        "--rerun_fallback", "--rerun-fallback",
        dest="rerun_fallback", action="store_true",
        help=(
            "Resume mode: re-run samples whose previous result was a random fallback "
            "(API failure). Default: skip them like any other completed sample."
        ),
    )

    # --- model / API ---
    parser.add_argument("--model_name", type=str, required=True,
                        help="e.g. gpt-5.2 / qwen3.6-27b / qwen3.5-9b / qwen3.5-4b")
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True,
                        help="OpenAI-compatible base URL, e.g. http://YOUR_GPT_ENDPOINT/v1")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You are a careful multimodal misinformation evaluator. "
            "Follow the prompt exactly and return the requested format only."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_new_tokens", type=int, default=0,
        help=(
            "Deprecated and IGNORED. The unified MMD-Agent never sets `max_tokens` "
            "on the chat-completions payload, so every model can write as long an "
            "answer as it needs to. The flag is kept for backward compatibility."
        ),
    )
    parser.add_argument("--request_timeout", type=int, default=180)
    parser.add_argument("--retry_times", type=int, default=5)
    parser.add_argument("--retry_interval", type=int, default=6)
    parser.add_argument("--image_detail", type=str, default="low", choices=["low", "high", "auto"])
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Per-model thread-pool concurrency. Default 10 to match the API-bound profile.")
    # 服务器上 LLM 调用必须借助 HTTP(S)_PROXY 才能访问公网端点 (e.g. gpt-5.2);
    # Qwen 内网端点通过 NO_PROXY 排除。因此默认开启 trust_env=True。
    parser.add_argument("--trust_env", dest="trust_env", action="store_true", default=True,
                        help="Honor HTTP(S)_PROXY env vars when making LLM calls (default: on).")
    parser.add_argument("--no_trust_env", dest="trust_env", action="store_false",
                        help="Force LLM calls to ignore HTTP(S)_PROXY env vars.")

    # --- Serper search ---
    parser.add_argument(
        "--serper_key_file", "--serper-key-file",
        dest="serper_key_file",
        type=str,
        required=True,
        help="ABSOLUTE path to the file holding Serper API keys (one per line).",
    )
    parser.add_argument(
        "--serper_key_index",
        type=int,
        required=True,
        help=(
            "1-based index into --serper_key_file. The user reserves keys 1..4 for "
            "other agents; use 5/6/7/8 for gpt-5.2 / qwen3.6-27b / qwen3.5-9b / qwen3.5-4b."
        ),
    )

    args = parser.parse_args()

    if args.max_samples and args.max_samples <= 0:
        args.max_samples = None

    return args


def main() -> None:
    args = parse_args()

    if not args.max_samples:
        args.max_samples = None

    # --- Pre-resolve absolute paths ---
    sampled_root = Path(args.sampled_root).expanduser().resolve()
    prompt_root = Path(args.prompt_root).expanduser().resolve()
    answer_path = Path(args.answer_path).expanduser().resolve()
    serper_key_file = Path(args.serper_key_file).expanduser().resolve()

    if not sampled_root.exists():
        raise FileNotFoundError(f"ReMMDBench root does not exist: {sampled_root}")
    if not prompt_root.exists():
        raise FileNotFoundError(f"Prompt template directory does not exist: {prompt_root}")
    if not serper_key_file.exists():
        raise FileNotFoundError(f"Serper key file does not exist: {serper_key_file}")

    args.sampled_root = str(sampled_root)
    args.prompt_root = str(prompt_root)
    args.answer_path = str(answer_path)
    args.serper_key_file = str(serper_key_file)

    serper_key = select_serper_key(serper_key_file, args.serper_key_index)

    # --- Build dataset ---
    dataset = ReMMDBench_Dataset(
        root=sampled_root,
        prompt_root=prompt_root,
        max_samples=args.max_samples,
        seed=args.seed,
        sample_filter=args.sample_filter,
        max_images=args.max_images,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"No samples found under {sampled_root} (sample_filter='{args.sample_filter}')"
        )

    # --- Run tag ---
    if args.run_name:
        run_tag = args.run_name
    else:
        run_tag = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_{args.dataset_name}_unified"
        )

    print(
        f"\nUnified MMD-Agent run\n"
        f"  model           : {args.model_name}\n"
        f"  base_url        : {args.base_url}\n"
        f"  sampled_root    : {sampled_root}\n"
        f"  prompt_root     : {prompt_root}\n"
        f"  serper_key_file : {serper_key_file} (index {args.serper_key_index})\n"
        f"  answer_path     : {answer_path}\n"
        f"  dataset_name    : {args.dataset_name}\n"
        f"  samples         : {len(dataset)}\n"
        f"  num_workers     : {args.num_workers}\n"
        f"  run_tag         : {run_tag}\n",
        flush=True,
    )

    summary = evaluate_VQA_MMD_Agent_Unified(
        dataset=dataset,
        args=args,
        dataset_name=args.dataset_name,
        run_tag=run_tag,
        serper_key=serper_key,
    )

    print(
        f"\nDone. Final artifacts:\n"
        f"  run_root      : {summary['run_root']}\n"
        f"  answer_path   : {summary['answer_path']}\n"
        f"  metrics_path  : {summary['metrics_path']}\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
