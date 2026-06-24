"""API 实际请求测试：检查 4 个 LLM + Serper 返回内容是否正常。

不只是连通性，而是实际请求一段中英文 prompt，看返回是否符合预期。
"""

from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import httpx

LLM_TARGETS = [
    {
        "name": "gpt-5.2",
        "base_url": "http://YOUR_GPT_ENDPOINT/v1",
        "api_key": "sk-YOUR_GPT_API_KEY_HERE",
        "use_proxy": True,
    },
    {
        "name": "qwen3.6-27b",
        "base_url": "http://YOUR_QWEN_27B_ENDPOINT/v1",
        "api_key": "qwen",
        "use_proxy": False,
    },
    {
        "name": "qwen3.5-9b",
        "base_url": "http://YOUR_QWEN_9B_ENDPOINT/v1",
        "api_key": "qwen",
        "use_proxy": False,
    },
    {
        "name": "qwen3.5-4b",
        "base_url": "http://YOUR_QWEN_4B_ENDPOINT/v1",
        "api_key": "qwen",
        "use_proxy": False,
    },
]

SAMPLE_IMAGE_PATH = "/path/to/ReMMDBench/001/images/01_img_1.jpg"
SERPER_KEY_FILE = "/path/to/serper_api.txt"


def encode_image(path: str) -> str:
    p = Path(path)
    suffix = p.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else (suffix or "jpeg")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def test_one_llm(cfg: dict) -> dict:
    name = cfg["name"]
    base_url = cfg["base_url"].rstrip("/")
    api_key = cfg["api_key"]
    url = f"{base_url}/chat/completions" if not base_url.endswith("/chat/completions") else base_url

    print(f"\n{'='*70}")
    print(f"测试模型: {name}")
    print(f"URL: {url}")
    print(f"{'='*70}")

    user_content = [
        {"type": "text",
         "text": "Briefly describe the image in one sentence, and then output exactly one line in the form `Finish[TEXT_STRONG_SUPPORT].`"},
        {"type": "image_url",
         "image_url": {"url": encode_image(SAMPLE_IMAGE_PATH), "detail": "low"}},
    ]
    payload = {
        "model": name,
        "messages": [
            {"role": "system", "content": "You are a careful multimodal evaluator."},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    trust_env = cfg["use_proxy"]
    t0 = time.time()
    try:
        with httpx.Client(timeout=120.0, trust_env=trust_env) as client:
            resp = client.post(url, headers=headers, json=payload)
        elapsed = time.time() - t0
        if resp.status_code != 200:
            print(f"[FAIL] HTTP {resp.status_code}, elapsed={elapsed:.2f}s")
            print(f"  body: {resp.text[:500]}")
            return {"name": name, "ok": False, "http_status": resp.status_code, "error": resp.text[:500]}
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            print(f"[FAIL] 空 choices: {json.dumps(data)[:400]}")
            return {"name": name, "ok": False, "error": "empty_choices"}
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            text = "\n".join([p.get("text", "") for p in content if isinstance(p, dict)])
        else:
            text = str(content or "")
        text = text.strip()
        print(f"[OK] elapsed={elapsed:.2f}s, response length={len(text)} chars")
        print(f"  返回内容预览:\n    {text[:400]}{'...' if len(text) > 400 else ''}")
        ok = bool(text) and len(text) > 5
        usage = data.get("usage", {})
        if usage:
            print(f"  usage: {usage}")
        return {"name": name, "ok": ok, "elapsed_s": elapsed, "response": text, "usage": usage}
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"[ERROR] {type(exc).__name__}: {exc}, elapsed={elapsed:.2f}s")
        return {"name": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def test_serper() -> dict:
    print(f"\n{'='*70}")
    print("测试 Serper API（key #5/6/7/8 各测一次）")
    print(f"{'='*70}")
    keys = [line.strip() for line in Path(SERPER_KEY_FILE).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]
    print(f"可用 key 数量: {len(keys)}")
    results = []
    for idx in (5, 6, 7, 8):
        if idx > len(keys):
            results.append({"index": idx, "ok": False, "error": "out-of-range"})
            continue
        key = keys[idx - 1]
        body = {"q": "Berlin 500 billion euro infrastructure fund", "gl": "us", "hl": "en", "num": 5}
        headers = {"X-API-KEY": key, "Content-Type": "application/json"}
        print(f"\n  ---- Serper key #{idx} (前 6 位: {key[:6]}...) ----")
        t0 = time.time()
        try:
            with httpx.Client(timeout=30.0, trust_env=True) as client:
                resp = client.post("https://google.serper.dev/search", headers=headers, json=body)
            elapsed = time.time() - t0
            if resp.status_code != 200:
                print(f"    [FAIL] HTTP {resp.status_code}: {resp.text[:200]}")
                results.append({"index": idx, "ok": False, "http_status": resp.status_code})
                continue
            data = resp.json()
            organic = data.get("organic") or []
            ans = data.get("answerBox") or {}
            kg = data.get("knowledgeGraph") or {}
            print(f"    [OK] elapsed={elapsed:.2f}s, organic={len(organic)}, "
                  f"has_answerBox={bool(ans)}, has_knowledgeGraph={bool(kg)}")
            if organic:
                first = organic[0]
                print(f"    首条结果: {first.get('title','')[:80]}")
                print(f"    snippet : {(first.get('snippet','') or '')[:120]}")
            results.append({"index": idx, "ok": True, "elapsed_s": elapsed,
                            "organic_count": len(organic),
                            "preview_title": (organic[0].get("title", "") if organic else "")})
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    [ERROR] {type(exc).__name__}: {exc}, elapsed={elapsed:.2f}s")
            results.append({"index": idx, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"keys": results}


if __name__ == "__main__":
    print(f"使用图片: {SAMPLE_IMAGE_PATH} (大小: {Path(SAMPLE_IMAGE_PATH).stat().st_size} 字节)")
    all_results = {"llms": [], "serper": None}
    for cfg in LLM_TARGETS:
        r = test_one_llm(cfg)
        all_results["llms"].append(r)
    all_results["serper"] = test_serper()

    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    for r in all_results["llms"]:
        status = "✓ OK" if r.get("ok") else "✗ FAIL"
        print(f"  {status:>8s}  {r['name']:<14s}  {r.get('elapsed_s','?'):>6}s  {r.get('error','')}")
    print("Serper:")
    for r in all_results["serper"]["keys"]:
        status = "✓ OK" if r.get("ok") else "✗ FAIL"
        print(f"  {status:>8s}  key#{r['index']}  organic={r.get('organic_count','?')} {r.get('error','')}")

    Path("/tmp/api_test_result.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n详细结果写入 /tmp/api_test_result.json")

    n_ok = sum(1 for r in all_results["llms"] if r.get("ok"))
    n_serper_ok = sum(1 for r in all_results["serper"]["keys"] if r.get("ok"))
    print(f"LLM 通过: {n_ok}/{len(all_results['llms'])}, Serper key 通过: {n_serper_ok}/4")
    sys.exit(0 if (n_ok == len(all_results["llms"]) and n_serper_ok == 4) else 1)
