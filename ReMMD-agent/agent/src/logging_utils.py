"""Run-folder management and logging setup.

Each invocation of `run_eval.py` creates a fresh `runs/<model>_<timestamp>_<tag>/`
directory containing:
  - `config.yaml`        snapshot of the merged config used
  - `eval.log`            file log
  - `samples/<sid>/...`   per-sample process artifacts
  - `summary.json`        per-sample results (appended)
  - `metrics/`            metrics + heatmap once evaluation finishes
  - `state.json`          running counters (atomic_ok, judge_ok, errors)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def utc_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_dir(runs_dir: str, model_name: str, tag: str = "") -> Path:
    ts = utc_timestamp()
    safe_model = model_name.replace("/", "_").replace(":", "_")
    parts = [safe_model, ts]
    if tag:
        parts.append(tag)
    name = "_".join(parts)
    out = Path(runs_dir) / name
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(run_dir: Path, level: int = logging.INFO) -> Path:
    log_path = run_dir / "eval.log"
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    # silence noisy 3rd-party loggers
    for noisy in (
        "httpx", "httpcore", "urllib3",
        "fontTools", "fontTools.subset", "fontTools.ttLib", "matplotlib",
        "matplotlib.font_manager", "PIL", "PIL.PngImagePlugin",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_path


def dump_config(run_dir: Path, cfg: dict[str, Any]) -> None:
    p = run_dir / "config.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def append_summary_row(run_dir: Path, row: dict[str, Any]) -> None:
    p = run_dir / "summary.jsonl"
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_sample_artifact(run_dir: Path, sample_id: str, name: str, data: Any) -> Path:
    sdir = run_dir / "samples" / sample_id
    sdir.mkdir(parents=True, exist_ok=True)
    p = sdir / name
    if isinstance(data, (dict, list)):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(data))
    return p


def sample_result_path(run_dir: Path, sample_id: str) -> Path:
    return run_dir / "samples" / sample_id / "result.json"


def sample_done(run_dir: Path, sample_id: str) -> bool:
    return sample_result_path(run_dir, sample_id).exists()
