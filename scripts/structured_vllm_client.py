#!/usr/bin/env python3
"""Call vLLM with schema-strict prompts and repair outputs to schema fields.

Examples:
    python scripts/structured_vllm_client.py \
      --eval-file eval/prompts_instructie.json \
      --limit 40 \
      --output results/vllm_benchmark_schema_strict/repaired_outputs.jsonl

    python scripts/structured_vllm_client.py \
      --instruction "从文本中抽取信息。" \
      --schema-json "[\"出生地\", \"职业\"]" \
      --input-text "鲁迅，原名周树人，浙江绍兴人，中国现代作家。"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

from microlm.structured import (
    build_schema_strict_messages,
    repair_to_schema,
    score_repaired_fields,
    try_parse_json,
)


DEFAULT_BASE_URL = "http://localhost:8000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema-strict vLLM structured client")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=None, help="Model id; defaults to /v1/models first entry")
    parser.add_argument("--eval-file", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/vllm_benchmark_schema_strict/repaired_outputs.jsonl"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--self-repair", action="store_true", help="Ask a second pass for missing required fields")
    parser.add_argument("--self-repair-max-fields", type=int, default=6)
    parser.add_argument("--self-repair-max-tokens", type=int, default=160)

    parser.add_argument("--instruction", default=None)
    parser.add_argument("--schema-json", default=None, help="JSON list of allowed fields")
    parser.add_argument("--required-json", default=None, help="Optional JSON list of required fields")
    parser.add_argument("--schema-fields", default=None, help="Comma-separated allowed fields, easier on Windows shells")
    parser.add_argument("--required-fields", default=None, help="Comma-separated required fields")
    parser.add_argument("--input-text", default=None)
    return parser.parse_args()


def get_model_name(base_url: str, fallback: str | None) -> str:
    if fallback:
        return fallback
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            return data["data"][0]["id"]
    except Exception:
        pass
    return "qwen"


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    response_format: bool,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        payload["response_format"] = {"type": "json_object"}

    started = time.time()
    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    elapsed = time.time() - started
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"], {
        "elapsed_s": round(elapsed, 3),
        "usage": data.get("usage", {}),
    }


def merge_repaired_fields(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if value is None or value == "" or value == []:
            continue
        if key not in merged or merged[key] is None or merged[key] == "" or merged[key] == []:
            merged[key] = value
    return merged


def build_self_repair_messages(
    prompt: dict[str, Any],
    repaired: dict[str, Any],
    missing_fields: list[str],
    raw_output: str,
) -> list[dict[str, str]]:
    schema_def = prompt.get("schema_def", {})
    schema_list = prompt.get("schema", [])
    input_text = prompt.get("input", "")
    instruction = prompt.get("instruction", "")
    field_types = {field: schema_def.get("types", {}).get(field, "string_or_list") for field in missing_fields}

    user = "\n".join([
        "你正在修复一次信息抽取 JSON 输出。",
        "只根据原文补齐缺失字段，不要猜测原文没有的信息。",
        "只输出一个 JSON object，顶层 key 只能来自 Missing fields。",
        "不要输出实体名作为顶层 key，不要输出解释文字。",
        "Missing fields 之外的 key 一律禁止。",
        "如果原文包含缺失字段的信息，必须抽取出来；不要重复 Current repaired JSON 里已有的字段。",
        "正确示例: Missing fields=[\"出生日期\"]，输出 {\"出生日期\": \"701年\"}",
        "错误示例: {\"李白\": {\"出生地\": \"碎叶城\"}} 或 {\"出生地\": \"碎叶城\"}",
        f"Instruction: {instruction}",
        f"Schema: {json.dumps(schema_list, ensure_ascii=False)}",
        f"Missing fields: {json.dumps(missing_fields, ensure_ascii=False)}",
        f"Missing field types: {json.dumps(field_types, ensure_ascii=False)}",
        f"Current repaired JSON: {json.dumps(repaired, ensure_ascii=False)}",
        f"Previous raw output: {raw_output}",
        f"Text: {input_text}",
    ])
    if any(field in {"起因", "导致", "原因"} for field in missing_fields):
        user += "\n提示: 如果缺失字段是起因/原因/导致，请优先查找原文中“由、由于、因为、造成、导致”等词附近的因果短语。"
    return [
        {
            "role": "system",
            "content": "你是一个 JSON 字段补全器。你的任务是只补齐缺失字段；输出必须是 JSON object；禁止输出缺失字段列表之外的 key。",
        },
        {"role": "user", "content": user},
    ]


def prompt_from_single_args(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.instruction and (args.schema_json or args.schema_fields) and args.input_text):
        raise SystemExit("--instruction, (--schema-json or --schema-fields) and --input-text must be provided for single-request mode")
    schema = parse_field_list(args.schema_json, args.schema_fields)
    required = parse_field_list(args.required_json, args.required_fields) if (args.required_json or args.required_fields) else schema
    return {
        "id": "single_request",
        "group": "single",
        "instruction": args.instruction,
        "schema": schema,
        "input": args.input_text,
        "schema_def": {
            "required_fields": required,
            "allowed_fields": schema,
            "types": {field: "string_or_list" for field in schema},
        },
    }


def parse_field_list(json_text: str | None, field_text: str | None) -> list[str]:
    if json_text:
        try:
            parsed = json.loads(json_text)
            if not isinstance(parsed, list):
                raise ValueError("field JSON must be a list")
            return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            # Windows shells can strip JSON quotes; fall back to forgiving split.
            field_text = json_text
    if not field_text:
        return []
    cleaned = field_text.strip().strip("[]")
    parts = [part.strip().strip("'\"") for part in cleaned.replace("，", ",").split(",")]
    return [part for part in parts if part]


def iter_prompts(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.eval_file:
        data = json.loads(args.eval_file.read_text(encoding="utf-8"))
        prompts = data["prompts"]
        return prompts[: args.limit] if args.limit else prompts
    return [prompt_from_single_args(args)]


def process_prompt(
    prompt: dict[str, Any],
    *,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: bool,
    self_repair: bool = False,
    self_repair_max_fields: int = 6,
    self_repair_max_tokens: int = 160,
) -> dict[str, Any]:
    messages = build_schema_strict_messages(prompt)
    raw_output, meta = chat_completion(
        base_url,
        model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    parsed, parse_ok = try_parse_json(raw_output)
    schema_def = prompt.get("schema_def", {})

    repaired = {}
    repaired_for_contract = {}
    score = {"schema_strict": False, "missing_fields": schema_def.get("required_fields", [])}
    if parse_ok:
        repaired = repair_to_schema(parsed, schema_def, use_aliases=True, fill_missing=False)
        repaired_for_contract = repair_to_schema(parsed, schema_def, use_aliases=True, fill_missing=True)
        score = score_repaired_fields(repaired, schema_def)

    self_repair_output = None
    self_repair_used = False
    self_repair_latency_s = 0.0
    if self_repair and score["missing_fields"] and len(score["missing_fields"]) <= self_repair_max_fields:
        self_repair_used = True
        repair_messages = build_self_repair_messages(prompt, repaired, score["missing_fields"], raw_output)
        self_repair_output, repair_meta = chat_completion(
            base_url,
            model,
            repair_messages,
            temperature=0.0,
            max_tokens=self_repair_max_tokens,
            response_format=response_format,
        )
        self_repair_latency_s = repair_meta["elapsed_s"]
        repair_parsed, repair_parse_ok = try_parse_json(self_repair_output)
        if repair_parse_ok:
            missing_schema_def = {
                "required_fields": score["missing_fields"],
                "allowed_fields": score["missing_fields"],
                "types": {field: schema_def.get("types", {}).get(field, "string_or_list") for field in score["missing_fields"]},
                "enum_constraints": {
                    field: values
                    for field, values in (schema_def.get("enum_constraints", {}) or {}).items()
                    if field in score["missing_fields"]
                },
            }
            patch = repair_to_schema(repair_parsed, missing_schema_def, use_aliases=True, fill_missing=False)
            repaired = merge_repaired_fields(repaired, patch)
            repaired_for_contract = repair_to_schema(repaired, schema_def, use_aliases=True, fill_missing=True)
            score = score_repaired_fields(repaired, schema_def)

    return {
        "id": prompt.get("id"),
        "group": prompt.get("group"),
        "parse_ok": parse_ok,
        "schema_strict_after_repair": score["schema_strict"],
        "missing_fields_after_repair": score["missing_fields"],
        "extra_fields_after_repair": score["extra_fields"],
        "enum_ok_after_repair": score["enum_ok"],
        "raw_output": raw_output,
        "self_repair_used": self_repair_used,
        "self_repair_output": self_repair_output,
        "repaired": repaired,
        "repaired_for_contract": repaired_for_contract,
        "latency_s": round(meta["elapsed_s"] + self_repair_latency_s, 3),
        "first_pass_latency_s": meta["elapsed_s"],
        "self_repair_latency_s": self_repair_latency_s,
        "usage": meta["usage"],
    }


def main() -> None:
    args = parse_args()
    prompts = iter_prompts(args)
    model = get_model_name(args.base_url, args.model)
    response_format = not args.no_response_format

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, prompt in enumerate(prompts, start=1):
        row = process_prompt(
            prompt,
            base_url=args.base_url,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format=response_format,
            self_repair=args.self_repair,
            self_repair_max_fields=args.self_repair_max_fields,
            self_repair_max_tokens=args.self_repair_max_tokens,
        )
        rows.append(row)
        print(
            f"[{index}/{len(prompts)}] {row['id']} parse={row['parse_ok']} "
            f"repair_strict={row['schema_strict_after_repair']} missing={row['missing_fields_after_repair']}"
        )

    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    parse_rate = sum(1 for row in rows if row["parse_ok"]) / total if total else 0
    repair_rate = sum(1 for row in rows if row["schema_strict_after_repair"]) / total if total else 0
    avg_latency = sum(row["latency_s"] for row in rows) / total if total else 0
    print("\nSUMMARY")
    print(f"model={model}")
    print(f"total={total}")
    print(f"parse_rate={parse_rate:.1%}")
    print(f"repair_strict_rate={repair_rate:.1%}")
    print(f"avg_latency_s={avg_latency:.3f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
