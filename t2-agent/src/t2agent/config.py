"""Runtime configuration loader.

A single YAML file describes everything an evaluation needs:

* ``api``      : provider name, model name, base URL, API key, timeout, proxy.
* ``retry``    : retry attempts and sleep between attempts (paper: 5 / 6 s).
* ``search``   : T2-Agent MCTS hyper-parameters.
* ``paths``    : ReMMDBench root, artifacts root, records root, Serper key file.
* ``serper``   : which line of the Serper key file to use.
* ``toolkits`` : tool sets per benchmark.
* ``evaluation``: minimum-per-category and the verdict probability bins.

The path entries accept absolute paths exclusively, because on the deployment
server the benchmark and the Serper file live outside this project tree. The
loader does *not* fall back to relative paths for benchmark data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class ApiConfig:
    """Provider-level API settings.

    ``provider`` is a short identifier ("gpt", "qwen3_6_27b", ...) used in the
    log/result file naming. ``model`` is the actual model id sent to the API.
    """

    provider: str
    model: str
    api_key: str
    primary_base_url: str
    backup_base_urls: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    proxy_url: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None


@dataclass(slots=True)
class RetryConfig:
    attempts: int = 5
    sleep_seconds: int = 6


@dataclass(slots=True)
class SearchConfig:
    sampled_nodes: int = 2
    exploration_weight: float = 2.0
    alpha: float = 0.5
    simulations: int = 12
    depth_limit: int = 4
    max_steps_per_rollout: int = 4
    high_confidence_threshold: float = 0.8
    score_scale: float = 10.0
    realmmdbench_tool_selection_subset_size: int = 100


@dataclass(slots=True)
class PathConfig:
    workspace_root: Path
    realmmdbench_root: Path
    artifacts_root: Path
    records_root: Path
    serper_api_file: Path


@dataclass(slots=True)
class SerperConfig:
    api_key_index: int = 0
    num_results: int = 5
    timeout_seconds: int = 30


@dataclass(slots=True)
class ToolkitConfig:
    base_tools: list[str]
    candidate_tools: list[str]


@dataclass(slots=True)
class EvaluationConfig:
    realmmdbench_min_per_category: int = 1
    realmmdbench_probability_bins: dict[str, float] = field(
        default_factory=lambda: {
            "true_min": 0.8,
            "mostly_true_min": 0.6,
            "mixture_min": 0.4,
            "mostly_false_min": 0.2,
        }
    )
    max_samples: int = 0


@dataclass(slots=True)
class LoggingConfig:
    log_llm_calls: bool = True
    log_tool_calls: bool = True
    log_prompt_chars: int = 0
    """0 means store the full prompt; a positive value truncates the stored prompt."""


@dataclass(slots=True)
class RuntimeConfig:
    api: ApiConfig
    retry: RetryConfig
    search: SearchConfig
    paths: PathConfig
    serper: SerperConfig
    toolkits: dict[str, ToolkitConfig]
    evaluation: EvaluationConfig
    logging: LoggingConfig


def _build_path(value: str) -> Path:
    if not value:
        raise ValueError("path value cannot be empty")
    return Path(value).expanduser().resolve()


def _load_serper_keys(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Serper API key file not found: {path}")
    keys: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            keys.append(stripped)
    if not keys:
        raise ValueError(f"Serper API key file is empty: {path}")
    return keys


def load_runtime_config(config_path: str | Path) -> RuntimeConfig:
    """Load and validate a runtime configuration YAML file."""

    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    api_raw = dict(raw["api"])
    api_raw.setdefault("provider", "gpt")
    api_raw.setdefault("backup_base_urls", [])
    api_raw.setdefault("proxy_url", None)
    api_raw.setdefault("max_output_tokens", None)
    api_raw.setdefault("temperature", None)
    api = ApiConfig(**api_raw)

    retry = RetryConfig(**raw.get("retry", {}))
    search = SearchConfig(**raw.get("search", {}))

    paths_raw = raw["paths"]
    paths = PathConfig(
        workspace_root=_build_path(paths_raw["workspace_root"]),
        realmmdbench_root=_build_path(paths_raw["realmmdbench_root"]),
        artifacts_root=_build_path(paths_raw["artifacts_root"]),
        records_root=_build_path(paths_raw["records_root"]),
        serper_api_file=_build_path(paths_raw["serper_api_file"]),
    )

    serper = SerperConfig(**raw.get("serper", {}))
    toolkits_raw = raw.get("toolkits", {}) or {}
    toolkits = {
        benchmark_name: ToolkitConfig(**toolkit_config)
        for benchmark_name, toolkit_config in toolkits_raw.items()
    }
    if "realmmdbench" not in toolkits:
        toolkits["realmmdbench"] = ToolkitConfig(
            base_tools=["wikipedia", "image_understanding"],
            candidate_tools=["web_search", "entity", "counterfactual"],
        )
    evaluation = EvaluationConfig(**raw.get("evaluation", {}))
    logging_cfg = LoggingConfig(**raw.get("logging", {}))

    for directory in (paths.artifacts_root, paths.records_root):
        directory.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig(
        api=api,
        retry=retry,
        search=search,
        paths=paths,
        serper=serper,
        toolkits=toolkits,
        evaluation=evaluation,
        logging=logging_cfg,
    )

    serper_keys = _load_serper_keys(paths.serper_api_file)
    if not (0 <= serper.api_key_index < len(serper_keys)):
        raise IndexError(
            "serper.api_key_index "
            f"{serper.api_key_index} is out of range for {len(serper_keys)} keys"
        )

    return config


def resolve_serper_key(config: RuntimeConfig) -> str:
    """Return the Serper API key string selected by ``config.serper.api_key_index``."""

    keys = _load_serper_keys(config.paths.serper_api_file)
    return keys[config.serper.api_key_index]
