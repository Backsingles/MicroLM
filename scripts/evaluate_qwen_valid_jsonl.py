#!/usr/bin/env python3
"""Evaluate a merged Qwen model on chat-style JSONL structured-output samples."""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL | re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_json_text(text: str) -> tuple[str, bool]:
    stripped = text.strip()
    match = FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip(), True
    return stripped, False


def find_first_json_span(text: str) -> str | None:
    stripped = text.strip()
    starts = [i for i in (stripped.find("{"), stripped.find("[")) if i >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = stripped[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]
    return None


def parse_jsonish(text: str) -> tuple[Any | None, str, str]:
    direct = text.strip()
    try:
        return json.loads(direct), "direct", direct
    except json.JSONDecodeError:
        pass

    cleaned, had_fence = clean_json_text(text)
    if had_fence:
        try:
            return json.loads(cleaned), "fenced", cleaned
        except json.JSONDecodeError:
            pass

    span = find_first_json_span(text)
    if span is not None:
        try:
            return json.loads(span), "extracted", span
        except json.JSONDecodeError:
            pass

    return None, "failed", cleaned


def canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list):
        return [canonical(v) for v in obj]
    return obj


def flatten_pairs(obj: Any, prefix: str = "") -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, (dict, list)):
                pairs |= flatten_pairs(value, path)
            else:
                pairs.add((path, str(value)))
    elif isinstance(obj, list):
        for value in obj:
            if isinstance(value, (dict, list)):
                pairs |= flatten_pairs(value, prefix)
            else:
                pairs.add((prefix, str(value)))
    return pairs


def flatten_field_paths(obj: Any, prefix: str = "") -> set[str]:
    fields: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.add(path)
            fields |= flatten_field_paths(value, path)
    elif isinstance(obj, list):
        for value in obj:
            fields |= flatten_field_paths(value, prefix)
    return fields


def prf(pred: set[Any], gold: set[Any]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_model(model_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, messages: list[dict[str, str]], args) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = outputs[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("outputs/qwen_lora_merged_final"))
    parser.add_argument("--data-path", type=Path, default=Path("data/sft_candidate/valid.jsonl"))
    parser.add_argument("--config-path", type=Path, default=Path("configs/qwen_lora_structured.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/qwen_valid_eval"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(args.config_path.read_text(encoding="utf-8"))
    system_prompt = cfg.get("system_prompt")
    rows = load_jsonl(args.data_path)
    sample_count = min(args.limit, len(rows)) if args.limit > 0 else len(rows)
    sampled = random.sample(rows, sample_count)

    print(f"Loading model: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.device)
    print(f"Evaluating {sample_count} samples from {args.data_path}")
    print(f"Generation: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}")

    results = []
    counters = Counter()
    field_p = field_r = field_f1 = 0.0
    pair_p = pair_r = pair_f1 = 0.0
    latencies = []
    by_task: dict[str, Counter] = defaultdict(Counter)

    for idx, row in enumerate(sampled, 1):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_msg = next(m for m in row["messages"] if m["role"] == "user")
        gold_msg = next(m for m in row["messages"] if m["role"] == "assistant")
        messages.append(user_msg)

        gold_obj = json.loads(gold_msg["content"])
        t0 = time.time()
        raw = generate_one(model, tokenizer, messages, args)
        latency = time.time() - t0
        latencies.append(latency)

        pred_obj, parse_mode, json_text = parse_jsonish(raw)
        parsed = pred_obj is not None
        direct_json = parse_mode == "direct"
        has_fence = raw.strip().startswith("```")
        exact = parsed and canonical(pred_obj) == canonical(gold_obj)

        if parsed:
            pf, rf, ff = prf(flatten_field_paths(pred_obj), flatten_field_paths(gold_obj))
            pp, rp, fp = prf(flatten_pairs(pred_obj), flatten_pairs(gold_obj))
        else:
            pf = rf = ff = pp = rp = fp = 0.0

        field_p += pf
        field_r += rf
        field_f1 += ff
        pair_p += pp
        pair_r += rp
        pair_f1 += fp

        counters["total"] += 1
        counters["parsed"] += int(parsed)
        counters["direct_json"] += int(direct_json)
        counters["fenced"] += int(has_fence)
        counters["exact_match"] += int(exact)
        task = row.get("task_type", "unknown")
        by_task[task]["total"] += 1
        by_task[task]["parsed"] += int(parsed)
        by_task[task]["direct_json"] += int(direct_json)
        by_task[task]["exact_match"] += int(exact)

        record = {
            "id": row.get("id"),
            "task_type": task,
            "topic_schema": row.get("topic_schema"),
            "quality_tier": row.get("quality_tier"),
            "parsed": parsed,
            "parse_mode": parse_mode,
            "direct_json": direct_json,
            "has_markdown_fence": has_fence,
            "exact_match": exact,
            "field_precision": round(pf, 6),
            "field_recall": round(rf, 6),
            "field_f1": round(ff, 6),
            "pair_precision": round(pp, 6),
            "pair_recall": round(rp, 6),
            "pair_f1": round(fp, 6),
            "latency_sec": round(latency, 3),
            "raw_output": raw,
            "parsed_output": pred_obj,
            "gold_output": gold_obj,
        }
        results.append(record)

        print(
            f"[{idx:04d}/{sample_count}] {row.get('id')} parsed={parsed} "
            f"direct={direct_json} exact={exact} field_f1={ff:.3f} pair_f1={fp:.3f} "
            f"latency={latency:.2f}s"
        )

    n = counters["total"]
    summary = {
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "sample_count": n,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "parseable_rate": counters["parsed"] / n if n else 0,
        "direct_json_rate": counters["direct_json"] / n if n else 0,
        "markdown_fence_rate": counters["fenced"] / n if n else 0,
        "exact_match_rate": counters["exact_match"] / n if n else 0,
        "avg_field_precision": field_p / n if n else 0,
        "avg_field_recall": field_r / n if n else 0,
        "avg_field_f1": field_f1 / n if n else 0,
        "avg_pair_precision": pair_p / n if n else 0,
        "avg_pair_recall": pair_r / n if n else 0,
        "avg_pair_f1": pair_f1 / n if n else 0,
        "avg_latency_sec": sum(latencies) / n if n else 0,
        "by_task": {
            task: {
                "total": c["total"],
                "parseable_rate": c["parsed"] / c["total"] if c["total"] else 0,
                "direct_json_rate": c["direct_json"] / c["total"] if c["total"] else 0,
                "exact_match_rate": c["exact_match"] / c["total"] if c["total"] else 0,
            }
            for task, c in sorted(by_task.items())
        },
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    failures = [r for r in results if not r["parsed"] or not r["direct_json"] or r["field_f1"] < 0.5]
    with (args.out_dir / "failure_samples.jsonl").open("w", encoding="utf-8") as f:
        for row in failures[:50]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved summary to {args.out_dir / 'summary.json'}")
    print(f"Saved detailed results to {args.out_dir / 'results.jsonl'}")
    print(f"Saved failure samples to {args.out_dir / 'failure_samples.jsonl'}")


if __name__ == "__main__":
    main()
