# t2-agent: 统一版 T2-Agent for ReMMDBench

本目录是把 `t2_agent-single-5`（单标签 5 分类：`verdict`）与
`t2_agent-multi-8`（多标签 8 分类：`distortion_taxonomy`）整合后的版本。
一次 MCTS 运行同时给出两份输出：

- **5 分类 verdict**（对应 `ReMMDBench/<id>/annotation.json` 的 `"verdict"` 字段）
- **8 标签 distortion taxonomy**（对应 `ReMMDBench/<id>/annotation.json` 的 `"distortion_taxonomy"` 字段）

所有 LLM 调用、工具调用、单道题的最终结果都会自动落盘本地。

---

## 1. 目录结构

```text
t2-agent/
├── README.md                       # 本文档
├── requirements.txt                # Python 依赖
├── configs/
│   ├── gpt5_2.yaml                 # GPT-5.2 配置
│   ├── qwen3_6_27b.yaml            # Qwen3.6-27B 配置
│   ├── qwen3_5_9b.yaml             # Qwen3.5-9B 配置
│   └── qwen3_5_4b.yaml             # Qwen3.5-4B 配置
├── scripts/
│   ├── run_remmdbench.py           # 主运行脚本（接受 --config）
│   ├── calc_metrics.py             # 离线重算 5+8 分类指标
│   ├── run_gpt5_2.sh / .ps1        # GPT 入口
│   ├── run_qwen3_6_27b.sh / .ps1   # Qwen3.6-27B 入口
│   ├── run_qwen3_5_9b.sh / .ps1    # Qwen3.5-9B 入口
│   └── run_qwen3_5_4b.sh / .ps1    # Qwen3.5-4B 入口
├── src/t2agent/
│   ├── __init__.py
│   ├── config.py                   # 配置加载（多模型 / Serper / ReMMDBench 路径）
│   ├── data.py                     # ReMMDBench 数据加载（annotation.json + sample.json + images/）
│   ├── labels.py                   # 5 类 verdict + 8 标签 taxonomy 规范化与指标
│   ├── llm.py                      # OpenAI-compatible LLM 客户端（缓存 + 重试 + 详细日志）
│   ├── tools.py                    # Wikipedia + Serper + LVLM 工具集
│   ├── logging_utils.py            # JSONL 调用日志器
│   └── agent.py                    # 统一的 T2Agent（一次 MCTS 给出 5+8 两份结果）
├── artifacts/                      # 运行产物（缓存 + per-run runs/）
└── records/                        # 运行级日志（含 LLM/工具调用 JSONL）
```

`src/t2agent/` 内的模块名与原始 `t2_agent-single-5/src/t2agent` 完全对齐，
新增了 `logging_utils.py` 以承载详细的调用日志逻辑。

---

## 2. 整合逻辑

两份原始 Agent 共享同一套 MCTS 主体（`text` / `image` / `match` 三个子任务、
双评分 ST/SC、改写 UCT、子任务初始化），差别只在**最后一步**：

- `t2_agent-single-5` 把 `p_real` 分桶后输出 5 类 verdict 标签。
- `t2_agent-multi-8` 再让 LLM 基于轨迹分类到 8 个 distortion 子标签。

整合后的 `agent.py`：

1. 跑一次 MCTS 得到 `p_real / p_fake_text / p_fake_image / p_fake_match`。
2. 用 `p_real` 直接产出 **5 类 verdict**（无额外 LLM 调用）。
3. 把 5 类 verdict + 子任务证据塞进同一份 prompt，让 LLM 输出
   **8 标签 distortion taxonomy**。
4. `PredictionResult` 同时保留两份结果：
   - `predicted_verdict: str`
   - `predicted_taxonomy: list[str]`
5. 每跑完一道题就立刻把：
   - `artifacts/runs/<run_name>/details/<index>_<sample>.json`（含完整轨迹、子任务、双标签）
   - `records/<run_name>/llm_calls.jsonl`（每一次 LLM/工具调用追加一条 JSON 记录）

写入磁盘，避免长跑过程中丢数据。

整个项目不再依赖 `MMFakeBench`，主入口只评估 `ReMMDBench`。

---

## 3. 关键路径设置位置（重要）

以下两类路径在服务器上是绝对路径，需要在每个 `configs/*.yaml` 中按实际位置修改。
**只有 `configs/` 下的 4 个 YAML 文件涉及路径**，源码不写死任何业务路径。

### 3.1 ReMMDBench 路径

字段：`paths.realmmdbench_root`

```yaml
paths:
  realmmdbench_root: "<benchmark 绝对路径>/ReMMDBench"
```

例如本地：
```yaml
realmmdbench_root: "C:/path/to/ReMMD-Agent/t2_agent/ReMMDBench"
```

例如服务器：
```yaml
realmmdbench_root: "/data/benchmarks/ReMMDBench"
```

数据加载器会逐个枚举该目录下的子目录（`001/`、`002/`、……），读取每个子目录的
`annotation.json` + `sample.json` + `images/`。

### 3.2 Serper API key 文件路径

字段：`paths.serper_api_file`

```yaml
paths:
  serper_api_file: "<绝对路径>/serper_api.txt"
```

例如本地：
```yaml
serper_api_file: "C:/path/to/serper_api.txt"
```

文件内容一行一个 API key，至少要有 4 行（前四行依次给 4 个模型使用）。
具体哪个模型使用哪一行，由各 YAML 的 `serper.api_key_index` 决定，已预设：

| 模型           | `serper.api_key_index` | 取 serper_api.txt 第几行 |
| -------------- | ---------------------- | ------------------------ |
| gpt-5.2        | 0                      | 第 1 行                  |
| qwen3.6-27b    | 1                      | 第 2 行                  |
| qwen3.5-9b     | 2                      | 第 3 行                  |
| qwen3.5-4b     | 3                      | 第 4 行                  |

如果想改成其他索引，只需修改对应 YAML 里 `serper.api_key_index` 即可。

### 3.3 每个模型的 URL / api_key / model

每个 `configs/*.yaml` 顶部的 `api` 段落控制该模型的接口：

```yaml
api:
  provider: "gpt"                          # 用于日志/产物命名的短名
  model: "gpt-5.2"                         # 发送到 API 的 model id
  api_key: "sk-..."                        # 该模型的 API Key
  primary_base_url: "http://.../v1"        # OpenAI 兼容 /v1 base URL
  backup_base_urls: []                     # 备用 URL（可选）
  timeout_seconds: 180
  proxy_url: null                          # 例：http://127.0.0.1:7890
  max_output_tokens: null
  temperature: null
```

四个模型 YAML 中需要填的位置如下：

- `configs/gpt5_2.yaml`
  - **已经填好**：
    - `model: gpt-5.2`
    - `api_key: sk-YOUR_GPT_API_KEY_HERE`
    - `primary_base_url: http://YOUR_GPT_ENDPOINT/v1`
- `configs/qwen3_6_27b.yaml`
  - **待填**：`api.model`、`api.api_key`、`api.primary_base_url`
- `configs/qwen3_5_9b.yaml`
  - **待填**：`api.model`、`api.api_key`、`api.primary_base_url`
- `configs/qwen3_5_4b.yaml`
  - **待填**：`api.model`、`api.api_key`、`api.primary_base_url`

每个 Qwen 的 YAML 里都标有 `===== TO BE FILLED IN ON THE SERVER =====`
区块，到服务器上拿到 url / api_key 后直接替换占位字符串即可。

### 3.4 项目自身的 `workspace_root` / `artifacts_root` / `records_root`

这三个路径用于本地产物落盘，默认指向本目录。把整个 `t2-agent/` 拷到服务器后，
统一把这三项改成服务器上的绝对路径即可：

```yaml
paths:
  workspace_root: "/abs/path/to/t2-agent"
  artifacts_root: "/abs/path/to/t2-agent/artifacts"
  records_root: "/abs/path/to/t2-agent/records"
```

---

## 4. 环境与依赖

本地测试环境：conda 环境 **`mmd`**。所有入口脚本都自动调用：

```bash
conda run -n mmd --no-capture-output python scripts/run_remmdbench.py ...
```

要换名字可以设置环境变量：

```bash
export T2AGENT_CONDA_ENV=my_env       # Linux/macOS
$env:T2AGENT_CONDA_ENV = "my_env"     # PowerShell
```

依赖见 `requirements.txt`：

```text
openai>=1.0.0
httpx>=0.27.0
PyYAML>=6.0.0
beautifulsoup4>=4.12.0
```

安装：

```bash
conda run -n mmd pip install -r requirements.txt        # Linux/macOS
```
```powershell
conda run -n mmd pip install -r requirements.txt        # Windows PowerShell
```

---

## 5. 运行说明

### 5.1 推荐方式：直接调用入口脚本

每个模型一个入口脚本，**默认会跑完整个 ReMMDBench**，并使用 4 个并发 worker：

Linux / macOS：

```bash
bash scripts/run_gpt5_2.sh
bash scripts/run_qwen3_6_27b.sh
bash scripts/run_qwen3_5_9b.sh
bash scripts/run_qwen3_5_4b.sh
```

Windows PowerShell：

```powershell
.\scripts\run_gpt5_2.ps1
.\scripts\run_qwen3_6_27b.ps1
.\scripts\run_qwen3_5_9b.ps1
.\scripts\run_qwen3_5_4b.ps1
```

常用参数（直接附在入口脚本后面即可，会原样转发给 `run_remmdbench.py`）：

| 参数 | 说明 |
| --- | --- |
| `--smoke` | 只跑第一个样本，烟测整条链路 |
| `--limit N` | 只跑前 N 个样本 |
| `--indices 0,5,10-19` | 只跑指定下标的样本 |
| `--max-workers K` | 并发数（覆盖默认值 4） |
| `--run-name NAME` | 自定义本次运行的产物目录名 |

举例：

```bash
# 单样本烟测
bash scripts/run_gpt5_2.sh --smoke

# 只跑前 20 个
bash scripts/run_gpt5_2.sh --limit 20

# 跑指定下标
bash scripts/run_gpt5_2.sh --indices 0,5,10-19

# 自定义运行名 + 8 并发
bash scripts/run_gpt5_2.sh --run-name gpt_full_$(date +%Y%m%d) --max-workers 8
```

### 5.2 直接调用主脚本

入口脚本背后只是包了一层 `conda run`：

```bash
conda run -n mmd python scripts/run_remmdbench.py \
    --config configs/gpt5_2.yaml \
    --max-workers 4 \
    --smoke
```

### 5.3 离线重算指标

任何一次运行结束后都可以基于 `run_summary.json` 重算指标（不会再调任何 API）：

```bash
conda run -n mmd python scripts/calc_metrics.py \
    artifacts/runs/<run_name>/run_summary.json
```

会在终端打印 5 分类 + 8 分类的全部指标，加上 `--output some.json` 还能把指标写盘。

---

## 6. 产物与日志说明

每次运行都会创建：

```text
artifacts/runs/<provider>_<timestamp>/
├── run_config.json                 # 本次运行的快照配置
├── run_summary.json                # 5+8 分类指标 + 全部 record
├── run_summary.md                  # 同上，Markdown 版
└── details/
    ├── 000_<sample_id>.json        # 单道题的完整结果（含轨迹、双标签、子任务）
    ├── 001_<sample_id>.json
    └── ...
```

```text
records/<provider>_<timestamp>/
└── llm_calls.jsonl                 # 每次 LLM/工具调用的 JSONL 日志
```

`llm_calls.jsonl` 每行是一个 JSON 对象，关键字段：

```json
{
  "event": "llm_call" | "tool_call",
  "timestamp": "2026-05-20T09:15:23.123Z",
  "seq": 42,
  "elapsed_ms": 5210,
  "status": "ok" | "error",
  "provider": "gpt",
  "model": "gpt-5.2",
  "purpose": "agent.plan.text",
  "cache_key": "...",
  "cached": false,
  "attempt": 1,
  "url": "http://YOUR_GPT_ENDPOINT/v1",
  "image_count": 1,
  "image_paths": ["..."],
  "system_prompt": "...",
  "user_prompt": "...",
  "response": "...",
  "error": null
}
```

工具调用日志格式与之相似，`event="tool_call"`，会带 `tool / tool_input / observation`。

如果只想保留摘要（避免日志过大），把对应 YAML 的
`logging.log_prompt_chars` 改为正整数，超过该长度的 prompt / response
会被截断并附 `<truncated to N chars>` 标记。

`details/<index>_<sample>.json` 里包含：

```json
{
  "index": 30,
  "sample_id": "031",
  "ground_truth_verdict": "Mostly False",
  "ground_truth_taxonomy": ["T2 Distortion", "V2 Visual Editing", "C1 Semantic Inconsistency", "C2 Contextual Inconsistency"],
  "predicted_verdict": "Mostly False",
  "predicted_taxonomy": ["T2 Distortion", "C2 Contextual Inconsistency"],
  "verdict_match": true,
  "taxonomy_match": false,
  "elapsed_seconds": 132.5,
  "prediction": { ... 完整轨迹、子任务、最终分数、rationale ... },
  "error": null
}
```

---

## 7. ReMMDBench 数据格式

数据加载器假设每个样本目录形如：

```text
<realmmdbench_root>/
├── 001/
│   ├── annotation.json     # { "verdict": "...", "distortion_taxonomy": [...], "rationale": "..." }
│   ├── sample.json         # { "text": "...", "images": ["01_img_1.jpg", ...], ... }
│   └── images/
│       ├── 01_img_1.jpg
│       └── ...
├── 002/...
└── 500/...
```

- `verdict` 字段会被规范化到 5 类之一：`True / Mostly True / Mixture / Mostly False / False`。
- `distortion_taxonomy` 字段会被规范化到 8 个标签之一或多个：
  `T1 Fabrication / T2 Distortion / T3 Misleading Context /
   V1 Synthetic Visual Content / V2 Visual Editing /
   C1 Semantic Inconsistency / C2 Contextual Inconsistency / C3 Pragmatic Inconsistency`。
- `sample.json` 的 `images` 字段被解析为相对 `images/` 子目录的文件名；
  若目录里没有列出但 `images/` 有图片文件，会自动回退使用目录内所有图片。

---

## 8. 不修改原始代码

按要求，本目录的所有代码与脚本都是**新建**或**从原代码逻辑改写**得到，
没有删除或修改：

- `ReMMDBench/`
- `t2_agent-multi-8/`
- `t2_agent-single-5/`

下的任何文件。

---

## 9. 复现性 / 缓存

- LLM 缓存：`artifacts/cache/llm/<provider>/<sha256>.json`，按
  `(model, system_prompt, user_prompt, image_paths, expect_json, temperature, max_output_tokens)`
  做 key。
- 工具缓存：`artifacts/cache/tools/<sha256>.json`，按 `(tool, input, text_sha, images)` 做 key。
- 同一份配置 + 同一份数据会命中缓存，重跑只会重新调 LLM 输出未命中的部分。
- 想重新计算所有结果，删除 `artifacts/cache/` 即可。

---

## 10. 快速自检

正式跑全集前，强烈建议先做单样本烟测：

```bash
bash scripts/run_gpt5_2.sh --smoke
```

或者只跑前 5 个：

```bash
bash scripts/run_gpt5_2.sh --limit 5
```

跑完检查：

1. `artifacts/runs/<run_name>/details/*.json` 里 `predicted_verdict` 和
   `predicted_taxonomy` 都已生成。
2. `records/<run_name>/llm_calls.jsonl` 里有事件，且
   `event` 字段同时出现 `llm_call` 与 `tool_call`。
3. `artifacts/runs/<run_name>/run_summary.md` 能正常渲染，5 分类与 8 分类
   两段表格都不为空。
