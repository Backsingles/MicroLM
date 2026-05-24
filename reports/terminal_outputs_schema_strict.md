
## 2026-05-21 20:39:01 - Read check_structured_stability.py

~~~powershell
Get-Content -Encoding utf8 scripts\check_structured_stability.py
~~~

~~~text
#!/usr/bin/env python3
"""check_structured_stability.py — Verify structured output stability on vLLM-served model.

Reuses the InstructIE evaluation prompt set (eval/prompts_instructie.json) to verify
that the vLLM-deployed qwen_lora model maintains its structured output quality.

Runs TWO rounds:
  Round 1: Normal chat completion (no format constraint)
  Round 2: Constrained completion with response_format=json_object (if supported)

For each round, computes:
  - Parse%     (JSON parseable rate)
  - Strict%    (strict schema match rate — all 4 checks pass)
  - Alias-Strict% (alias-normalized strict rate)
  - Per-group breakdown (extraction / schema_constraint / format_following)

Usage:
    python scripts/check_structured_stability.py                          # full test
    python scripts/check_structured_stability.py --rounds 1               # round 1 only
    python scripts/check_structured_stability.py --base-url http://host:8001
    python scripts/check_structured_stability.py --limit 5                # quick check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)


# ── Paths & Defaults ──────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8000"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PROMPTS_PATH = PROJECT_ROOT / "eval" / "prompts_instructie.json"


def get_served_model_name(base_url: str) -> str:
    """Query /v1/models to get the actual model ID served by vLLM."""
    try:
        r = requests.get(f"{base_url}/v1/models", timeout=10)
        data = r.json()
        if data.get("data") and len(data["data"]) > 0:
            return data["data"][0]["id"]
    except Exception:
        pass
    return "qwen"  # fallback


# ══════════════════════════════════════════════════════════════════════════
# Detection Functions (adapted from run_instructie_eval.py)
# ══════════════════════════════════════════════════════════════════════════

def clean_model_output(raw: str) -> str:
    s = raw.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", s, re.DOTALL)
    if m:
        s = m.group(1).strip()
    return s


def try_parse_json(text: str):
    try:
        return json.loads(text), True
    except (json.JSONDecodeError, ValueError):
        return None, False


# Alias map (same as run_instructie_eval.py)
FIELD_ALIASES = {
    "创办者": ["创始人", "创建者", "建立者", "发起人"],
    "位于": ["位置", "地点", "所在地", "地址", "所在"],
    "发现者或发明者": ["发现者", "发明者", "发现人", "发明人"],
    "创建或成立时间": ["建造时间", "创建时间", "建立时间", "建成时间"],
    "发生时间": ["时间", "举办时间", "开始时间"],
    "发生地点": ["地点", "位置", "举办地"],
    "常见并发症": ["并发症", "合并症"],
    "症状": ["主要症状", "常见症状", "临床表现"],
    "治疗方法": ["治疗", "疗法", "治疗方式"],
    "成就": ["获奖", "奖项", "荣誉", "成就奖"],
    "子组织": ["旗下", "下属组织", "子公司", "分支机构"],
    "别名": ["又称", "又名", "别称", "也叫"],
    "组成": ["成分", "原料", "组成部分", "构成"],
    "用途": ["应用", "应用领域", "主要用途", "作用"],
    "所属科室": ["科室", "所属科"],
    "线路": ["所属线路", "路线"],
    "车站等级": ["等级"],
    "开通时间": ["启用时间", "运营时间", "通车时间"],
    "保护级别": ["濒危等级", "保护等级"],
    "成立时间": ["成立年份", "创立时间", "创建时间"],
    "出生地": ["出生地点", "籍贯"],
    "出生日期": ["生日", "出生年月"],
    "参与者": ["参加者", "参赛者", "参赛方"],
    "起因": ["原因", "导火索"],
    "导致": ["结果", "后果"],
}


def normalize_field_name(field: str) -> str:
    for canonical, aliases in FIELD_ALIASES.items():
        if field == canonical or field in aliases:
            return canonical
    return field


def extract_all_fields(parsed) -> set[str]:
    fields = set()
    if isinstance(parsed, dict):
        for key, val in parsed.items():
            fields.add(key)
            if isinstance(val, dict):
                fields.update(val.keys())
    return fields


def score_output(raw_output: str, prompt_item: dict) -> dict:
    """Run 4-detection + alias-normalized scoring on a single output."""
    cleaned = clean_model_output(raw_output)
    parsed, is_parseable = try_parse_json(cleaned)

    schema_def = prompt_item.get("schema_def", {})
    required = schema_def.get("required_fields", [])
    allowed = schema_def.get("allowed_fields", [])
    enums = schema_def.get("enum_constraints", {})

    result = {
        "id": prompt_item["id"],
        "group": prompt_item["group"],
        "parsed": is_parseable,
        "raw_output": raw_output[:500],
    }

    if not is_parseable or parsed is None:
        result.update({
            "missing_fields": required,
            "extra_fields": [],
            "schema_strict": False,
            "schema_strict_alias": False,
        })
        return result

    # Collect all fields (handles entity-keyed nested format)
    all_fields_raw = extract_all_fields(parsed)
    missing = [f for f in required if f not in all_fields_raw]
    extra = [f for f in all_fields_raw if f not in set(allowed)]

    # Enum check
    enum_ok = True
    if enums:
        for _entity, fields in (parsed.items() if isinstance(parsed, dict) else []):
            if not isinstance(fields, dict):
                continue
            for fname, fval in fields.items():
                if fname in enums:
                    allowed_vals = set(enums[fname])
                    vals = [fval] if isinstance(fval, str) else fval if isinstance(fval, list) else []
                    for v in vals:
                        if v not in allowed_vals:
                            enum_ok = False

    schema_strict = len(missing) == 0 and len(extra) == 0 and enum_ok

    # Alias-normalized
    all_fields_norm = set(normalize_field_name(f) for f in all_fields_raw)
    missing_alias = [f for f in required if f not in all_fields_norm]
    extra_alias = [f for f in all_fields_raw if normalize_field_name(f) not in set(allowed)]
    schema_strict_alias = len(missing_alias) == 0 and len(extra_alias) == 0 and enum_ok

    result.update({
        "missing_fields": missing,
        "extra_fields": extra,
        "enum_ok": enum_ok if enums else None,
        "schema_strict": schema_strict,
        "schema_strict_alias": schema_strict_alias,
        "missing_fields_alias": missing_alias,
        "extra_fields_alias": extra_alias,
    })
    return result


# ══════════════════════════════════════════════════════════════════════════
# Prompt Builder (same as run_instructie_eval.py)
# ══════════════════════════════════════════════════════════════════════════

def build_prompt_text(prompt_item: dict) -> str:
    instruction = prompt_item.get("instruction", "")
    schema_list = prompt_item.get("schema", [])
    input_text = prompt_item.get("input", "")

    parts = []
    if instruction:
        parts.append(instruction)
    if schema_list and "Schema:" not in instruction and "schema" not in instruction.lower():
        parts.append(f"Schema: {json.dumps(schema_list, ensure_ascii=False)}")
    if "文本:" not in instruction and "从文本" not in instruction:
        parts.append(f"文本: {input_text}")
    else:
        parts.append(input_text)

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════
# API Client
# ══════════════════════════════════════════════════════════════════════════

def api_chat_completion(
    base_url: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    response_format: dict | None = None,
    model: str = "qwen",
) -> tuple[str, dict]:
    """Send chat completion request. Returns (output_text, usage_info)."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format:
        payload["response_format"] = response_format

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=120)
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return content, {"elapsed_s": round(elapsed, 3), **usage}


# ══════════════════════════════════════════════════════════════════════════
# Evaluation Rounds
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = "你是一个严格遵循 schema 的信息抽取助手。请严格按照给定的 schema 从文本中抽取信息，并以 JSON 格式输出。不要在 JSON 前后添加任何解释性文字。"


def run_round(
    base_url: str,
    prompts: list[dict],
    round_name: str,
    use_response_format: bool = False,
    limit: int = 0,
    model_name: str = "qwen",
) -> dict:
    """Run one evaluation round against the vLLM API."""
    print(f"\n{'='*60}")
    print(f"Round: {round_name}")
    print(f"  Mode: {'constrained (response_format=json_object)' if use_response_format else 'normal chat completion'}")
    print(f"  Prompts: {len(prompts) if limit == 0 else min(limit, len(prompts))}")
    print(f"{'='*60}")

    if limit > 0:
        prompts = prompts[:limit]

    scored_results = []
    total_time = 0.0
    errors = 0

    for i, prompt_item in enumerate(prompts):
        prompt_text = build_prompt_text(prompt_item)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]

        try:
            fmt = {"type": "json_object"} if use_response_format else None
            output, usage_info = api_chat_completion(
                base_url, messages,
                max_tokens=256, temperature=0.0,
                response_format=fmt,
                model=model_name,
            )
            total_time += usage_info.get("elapsed_s", 0)
        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] {prompt_item['id']}: ERROR — {e}")
            errors += 1
            scored_results.append({
                "id": prompt_item["id"],
                "group": prompt_item["group"],
                "parsed": False,
                "schema_strict": False,
                "schema_strict_alias": False,
                "error": str(e),
            })
            continue

        scored = score_output(output, prompt_item)
        scored["latency_s"] = usage_info.get("elapsed_s", 0)
        scored_results.append(scored)

        p_flag = "Y" if scored["parsed"] else "N"
        s_flag = "Y" if scored["schema_strict"] else "N"
        a_flag = "Y" if scored["schema_strict_alias"] else "N"
        preview = output[:80].replace("\n", " ")
        print(f"  [{i+1}/{len(prompts)}] {prompt_item['group'][:4]:>4} {prompt_item['id']}: "
              f"parse={p_flag} strict={s_flag} alias={a_flag} | {preview}...")

    n = len(scored_results)
    if n == 0:
        return {"round": round_name, "status": "all_failed"}

    # Compute summary stats
    parse_count = sum(1 for r in scored_results if r["parsed"])
    strict_count = sum(1 for r in scored_results if r["schema_strict"])
    alias_strict_count = sum(1 for r in scored_results if r.get("schema_strict_alias"))

    summary = {
        "round": round_name,
        "mode": "constrained" if use_response_format else "normal",
        "total": n,
        "errors": errors,
        "parse_rate": round(parse_count / n, 4),
        "strict_rate": round(strict_count / n, 4),
        "alias_strict_rate": round(alias_strict_count / n, 4),
        "total_time_s": round(total_time, 2),
        "avg_latency_s": round(total_time / n, 3) if n > 0 else 0,
        "results": scored_results,
    }

    # Per-group breakdown
    groups = ["extraction", "schema_constraint", "format_following"]
    by_group = {}
    for g in groups:
        group_results = [r for r in scored_results if r["group"] == g]
        gn = len(group_results)
        if gn == 0:
            continue
        by_group[g] = {
            "total": gn,
            "parse_rate": round(sum(1 for r in group_results if r["parsed"]) / gn, 4),
            "strict_rate": round(sum(1 for r in group_results if r["schema_strict"]) / gn, 4),
            "alias_strict_rate": round(sum(1 for r in group_results if r.get("schema_strict_alias")) / gn, 4),
        }
    summary["by_group"] = by_group

    # Print summary
    print(f"\n  --- {round_name} Summary ---")
    print(f"  Parse%:       {summary['parse_rate']:.1%} ({parse_count}/{n})")
    print(f"  Strict%:      {summary['strict_rate']:.1%} ({strict_count}/{n})")
    print(f"  Alias-Strict%:{summary['alias_strict_rate']:.1%} ({alias_strict_count}/{n})")
    print(f"  Errors:       {errors}")
    print(f"  Total time:   {summary['total_time_s']}s")
    if by_group:
        print(f"  By group:")
        for g, gs in by_group.items():
            print(f"    {g:<22} P={gs['parse_rate']:.1%} S={gs['strict_rate']:.1%} A={gs['alias_strict_rate']:.1%}")

    return summary


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Check structured output stability on vLLM")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--eval-file", type=str, default=None)
    parser.add_argument("--rounds", type=int, default=2, help="Number of rounds (1=normal only, 2=+constrained)")
    parser.add_argument("--limit", type=int, default=0, help="Limit prompts per round (0=all)")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    eval_path = Path(args.eval_file) if args.eval_file else EVAL_PROMPTS_PATH
    if not eval_path.exists():
        print(f"[ERROR] Eval prompts not found at {eval_path}")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    prompts = eval_data["prompts"]

    gen_params = eval_data.get("generation_params", {})
    print("=" * 60)
    print("  Structured Output Stability Check — vLLM Deployed Model")
    print(f"  Target:      {args.base_url}")
    print(f"  Eval file:   {eval_path.name}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Rounds:      {args.rounds}")
    print(f"  Time:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{args.base_url}/health", timeout=10)
        assert r.status_code == 200
        print(f"\nServer health: OK")
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to server: {e}")
        sys.exit(1)

    # Detect served model name
    model_name = get_served_model_name(args.base_url)
    print(f"  Model: {model_name}")

    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "vllm_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rounds = []

    # ── Round 1: Normal completion ──
    r1 = run_round(args.base_url, prompts, "Round 1: Normal Chat Completion",
                   use_response_format=False, limit=args.limit, model_name=model_name)
    all_rounds.append(r1)

    # ── Round 2: Constrained completion ──
    if args.rounds >= 2:
        r2 = run_round(args.base_url, prompts, "Round 2: Constrained (response_format=json_object)",
                       use_response_format=True, limit=args.limit, model_name=model_name)
        all_rounds.append(r2)

    # ── Comparison table ──
    print(f"\n{'='*70}")
    print("STABILITY CHECK COMPARISON")
    print(f"{'='*70}")
    print(f"{'Round':<45} {'Parse%':>8} {'Strict%':>9} {'Alias-S%':>10}")
    print("-" * 75)
    for rd in all_rounds:
        mode_tag = " [constrained]" if rd["mode"] == "constrained" else ""
        print(f"{rd['round']:<45}{rd['parse_rate']:>7.1%}{rd['strict_rate']:>8.1%}{rd['alias_strict_rate']:>9.1%}{mode_tag}")

    # Compare with 6C offline results (reference)
    print(f"\n--- Reference: 6C Offline Results (qwen_lora) ---")
    print(f"{'Config':<45} {'Parse%':>8} {'Strict%':>9} {'Alias-S%':>10}")
    print(f"{'6C offline (run_instructie_eval.py)':<45}{'97.5%':>8}{'7.5%':>9}{'15.0%':>10}")

    # ── Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Full JSON with results
    save_data = {
        "check_config": {
            "base_url": args.base_url,
            "eval_file": str(eval_path),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rounds": args.rounds,
        },
        "rounds": [
            {
                "round": rd["round"],
                "mode": rd["mode"],
                "total": rd["total"],
                "parse_rate": rd["parse_rate"],
                "strict_rate": rd["strict_rate"],
                "alias_strict_rate": rd["alias_strict_rate"],
                "by_group": rd.get("by_group", {}),
                "avg_latency_s": rd["avg_latency_s"],
            }
            for rd in all_rounds
        ],
        "reference_6c_offline": {
            "model": "qwen_lora",
            "parse_rate": 0.975,
            "strict_rate": 0.075,
            "alias_strict_rate": 0.150,
        },
    }
    json_out = out_dir / f"stability_{timestamp}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_out}")

    # Summary CSV
    csv_out = out_dir / f"stability_summary_{timestamp}.csv"
    import csv
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "mode", "total", "parse_rate", "strict_rate", "alias_strict_rate",
                         "extraction_P", "extraction_S", "extraction_A",
                         "schema_P", "schema_S", "schema_A",
                         "format_P", "format_S", "format_A",
                         "avg_latency_s"])
        for rd in all_rounds:
            bg = rd.get("by_group", {})
            eg = bg.get("extraction", {})
            sg = bg.get("schema_constraint", {})
            fg = bg.get("format_following", {})
            writer.writerow([
                rd["round"], rd["mode"], rd["total"],
                rd["parse_rate"], rd["strict_rate"], rd["alias_strict_rate"],
                eg.get("parse_rate", ""), eg.get("strict_rate", ""), eg.get("alias_strict_rate", ""),
                sg.get("parse_rate", ""), sg.get("strict_rate", ""), sg.get("alias_strict_rate", ""),
                fg.get("parse_rate", ""), fg.get("strict_rate", ""), fg.get("alias_strict_rate", ""),
                rd["avg_latency_s"],
            ])
    print(f"CSV saved to {csv_out}")


if __name__ == "__main__":
    main()

~~~

## 2026-05-21 20:39:15 - Inspect InstructIE prompt schema examples

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe - <<'PY'
import json
from pathlib import Path
p=json.loads(Path('eval/prompts_instructie.json').read_text(encoding='utf-8'))
for item in p['prompts'][:5]+p['prompts'][18:22]+p['prompts'][30:34]:
    print('\nID', item['id'], 'group', item['group'])
    print('schema:', item.get('schema'))
    print('schema_def:', json.dumps(item.get('schema_def'), ensure_ascii=False))
    print('gold:', json.dumps(item.get('gold_output'), ensure_ascii=False)[:300])
PY
~~~

~~~text
Invoke-Expression : 所在位置 行:1 字符: 60
+ $env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe - <<'PY'
+                                                            ~
重定向运算符后面缺少文件规范。

所在位置 行:1 字符: 59
+ $env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe - <<'PY'
+                                                           ~
“<”运算符是为将来使用而保留的。

所在位置 行:1 字符: 60
+ $env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe - <<'PY'
+                                                            ~
“<”运算符是为将来使用而保留的。

所在位置 行:3 字符: 1
+ from pathlib import Path
+ ~~~~
此语言版本中不支持“from”关键字。

所在位置 行:4 字符: 61
+ p=json.loads(Path('eval/prompts_instructie.json').read_text(encoding= ...
+                                                             ~
方法调用中缺少“)”。

所在位置 行:4 字符: 78
+ ... ads(Path('eval/prompts_instructie.json').read_text(encoding='utf-8'))
+                                                                         ~
表达式或语句中包含意外的标记“)”。

所在位置 行:5 字符: 4
+ for item in p['prompts'][:5]+p['prompts'][18:22]+p['prompts'][30:34]:
+    ~
关键字“for”后面缺少左“(”。

所在位置 行:6 字符: 18
+     print('\nID', item['id'], 'group', item['group'])
+                  ~
“,”后面缺少表达式。

所在位置 行:6 字符: 19
+     print('\nID', item['id'], 'group', item['group'])
+                   ~~~~~~~~~~
表达式或语句中包含意外的标记“item['id']”。

所在位置 行:6 字符: 18
+     print('\nID', item['id'], 'group', item['group'])
+                  ~
表达式中缺少右“)”。

并未报告所有分析错误。请更正报告的错误并重试。
所在位置 行:15 字符: 14
+ $out = try { Invoke-Expression $cmd 2>&1 | Out-String -Width 360 } ca ...
+              ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ParserError: (:) [Invoke-Expression], ParseException
    + FullyQualifiedErrorId : MissingFileSpecification,Microsoft.PowerShell.Commands.InvokeExpressionCommand
 

~~~

## 2026-05-21 20:39:28 - Inspect InstructIE prompt schema examples corrected

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path; p=json.loads(Path('eval/prompts_instructie.json').read_text(encoding='utf-8')); items=p['prompts'][:5]+p['prompts'][18:22]+p['prompts'][30:34];
for item in items:
 print('\nID', item['id'], 'group', item['group']); print('schema:', item.get('schema')); print('schema_def:', json.dumps(item.get('schema_def'), ensure_ascii=False)); print('gold:', json.dumps(item.get('gold_output'), ensure_ascii=False)[:300])"
~~~

~~~text

ID ext_001 group extraction
schema: ['别名', '出生地', '出生日期', '国籍', '职业', '死亡日期', '家庭成员', '成就', '雇主', '教育机构']
schema_def: {"required_fields": ["别名", "出生地", "出生日期"], "allowed_fields": ["别名", "出生地", "出生日期", "国籍", "职业", "死亡日期", "家庭成员", "成就", "雇主", "教育机构"], "types": {"别名": "string_or_list", "出生地": "string", "出生日期": "string"}}
gold: {"鲁迅": {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "出生日期": "1881年9月25日", "职业": ["作家", "文学家"], "成就": "中国现代文学的奠基人之一"}}

ID ext_002 group extraction
schema: ['位于', '别名', '人口', '行政中心', '面积', '长度', '宽度', '海拔']
schema_def: {"required_fields": ["位于", "面积"], "allowed_fields": ["位于", "别名", "人口", "行政中心", "面积", "长度", "宽度", "海拔"], "types": {"位于": "string_or_list", "面积": "string"}}
gold: {"西湖": {"位于": "中国浙江省杭州市西湖区", "面积": "6.39平方公里"}}

ID ext_003 group extraction
schema: ['位于', '别名', '子组织', '成立时间', '产品', '成员', '创办者', '解散时间']
schema_def: {"required_fields": ["创办者", "成立时间"], "allowed_fields": ["位于", "别名", "子组织", "成立时间", "产品", "成员", "创办者", "解散时间"], "types": {"创办者": "string", "成立时间": "string"}}
gold: {"阿里巴巴集团": {"创办者": "马云", "成立时间": "1999年", "位于": "浙江省杭州市", "子组织": ["淘宝网", "天猫", "阿里云"]}}

ID ext_004 group extraction
schema: ['别名', '组成', '生成物', '产地', '发现者或发明者', '名称由来', '用途']
schema_def: {"required_fields": ["别名", "发现者或发明者"], "allowed_fields": ["别名", "组成", "生成物", "产地", "发现者或发明者", "名称由来", "用途"], "types": {"别名": "string_or_list", "发现者或发明者": "string"}}
gold: {"青霉素": {"别名": "盘尼西林", "发现者或发明者": "亚历山大·弗莱明", "用途": "治疗由革兰氏阳性菌引起的感染"}}

ID ext_005 group extraction
schema: ['位于', '别名', '创建或成立时间', '高度', '面积', '建筑师', '建筑风格', '材料', '用途']
schema_def: {"required_fields": ["位于", "建筑师"], "allowed_fields": ["位于", "别名", "创建或成立时间", "高度", "面积", "建筑师", "建筑风格", "材料", "用途"], "types": {"位于": "string", "建筑师": "string"}}
gold: {"埃菲尔铁塔": {"位于": "法国巴黎战神广场", "建筑师": "古斯塔夫·埃菲尔", "创建或成立时间": "1889年", "高度": "324米"}}

ID sc_001 group schema_constraint
schema: ['出生地', '出生日期', '国籍']
schema_def: {"required_fields": ["出生地", "出生日期", "国籍"], "allowed_fields": ["出生地", "出生日期", "国籍"], "types": {"出生地": "string", "出生日期": "string", "国籍": "string"}, "enum_constraints": {}}
gold: {"姚明": {"出生地": "上海市", "出生日期": "1980年9月12日", "国籍": "中国"}}

ID sc_002 group schema_constraint
schema: ['位于', '成立时间', '创办者']
schema_def: {"required_fields": ["位于", "成立时间", "创办者"], "allowed_fields": ["位于", "成立时间", "创办者"], "types": {"位于": "string", "成立时间": "string", "创办者": "string"}, "enum_constraints": {}}
gold: {"腾讯公司": {"位于": "广东省深圳市南山区", "成立时间": "1998年", "创办者": "马化腾"}}

ID sc_003 group schema_constraint
schema: ['位于', '创建或成立时间', '高度', '用途']
schema_def: {"required_fields": ["位于", "用途"], "allowed_fields": ["位于", "创建或成立时间", "高度", "用途"], "types": {"位于": "string", "用途": "string"}, "enum_constraints": {"用途": ["住宅", "商业", "宗教", "军事", "文化", "交通", "教育"]}}
gold: {"巴黎圣母院": {"位于": "法国巴黎西岱岛", "创建或成立时间": "1163年", "高度": "约96米", "用途": "宗教"}}

ID sc_004 group schema_constraint
schema: ['别名', '症状', '治疗方法', '所属科室']
schema_def: {"required_fields": ["症状", "所属科室"], "allowed_fields": ["别名", "症状", "治疗方法", "所属科室"], "types": {"症状": "string_or_list", "所属科室": "string"}, "enum_constraints": {"所属科室": ["内科", "外科", "儿科", "妇产科", "眼科", "皮肤科", "骨科", "神经科"]}}
gold: {"肺炎": {"症状": ["咳嗽", "发热", "呼吸困难"], "治疗方法": "抗生素", "所属科室": "内科"}}

ID ff_001 group format_following
schema: ['出生地', '出生日期', '职业']
schema_def: {"required_fields": ["出生地", "出生日期", "职业"], "allowed_fields": ["出生地", "出生日期", "职业"], "types": {"出生地": "string", "出生日期": "string", "职业": "string"}}
gold: {"李白": {"出生地": "碎叶城", "出生日期": "701年", "职业": "诗人"}}

ID ff_002 group format_following
schema: ['位于', '面积']
schema_def: {"required_fields": ["位于", "面积"], "allowed_fields": ["位于", "面积"], "types": {"位于": "string", "面积": "string"}}
gold: {"鄱阳湖": {"位于": "江西省北部", "面积": "约3150平方公里"}}

ID ff_003 group format_following
schema: ['位于', '成立时间', '创办者']
schema_def: {"required_fields": ["位于", "成立时间", "创办者"], "allowed_fields": ["位于", "成立时间", "创办者"], "types": {"位于": "string", "成立时间": "string", "创办者": "string"}}
gold: {"华为技术有限公司": {"位于": "广东省深圳市", "成立时间": "1987年", "创办者": "任正非"}}

ID ff_004 group format_following
schema: ['别名', '发现者或发明者', '用途']
schema_def: {"required_fields": ["发现者或发明者"], "allowed_fields": ["别名", "发现者或发明者", "用途"], "types": {"发现者或发明者": "string_or_list"}, "must_use_chinese_fields": true}
gold: {"DNA双螺旋结构": {"发现者或发明者": ["沃森", "克里克"], "用途": "现代分子生物学基础"}}

~~~

## 2026-05-21 20:41:35 - Py compile check_structured_stability.py

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\check_structured_stability.py
~~~

~~~text

~~~

## 2026-05-21 20:41:55 - Check vLLM health before schema strict eval

~~~powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health -TimeoutSec 5
~~~

~~~text


StatusCode        : 200
StatusDescription : OK
Content           : {}
RawContent        : HTTP/1.1 200 OK
                    Content-Length: 0
                    Date: Thu, 21 May 2026 12:41:54 GMT
                    Server: uvicorn
                    
                    
Headers           : {[Content-Length, 0], [Date, Thu, 21 May 2026 12:41:54 GMT], [Server, uvicorn]}
RawContentLength  : 0




~~~

## 2026-05-21 20:42:45 - Quick schema strict stability check limit 5

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\check_structured_stability.py --base-url http://localhost:8000 --rounds 3 --limit 5 --output-dir results\vllm_benchmark_schema_strict
~~~

~~~text
============================================================
  Structured Output Stability Check — vLLM Deployed Model
  Target:      http://localhost:8000
  Eval file:   prompts_instructie.json
  Prompts:     40
  Rounds:      3
  Time:        2026-05-21 20:42:04
============================================================

Server health: OK
  Model: /mnt/e/MicroLM/outputs/qwen_lora_merged_final

============================================================
Round: Round 1: Normal Chat Completion
  Mode: normal chat completion
  Prompts: 5
============================================================
  [1/5] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/5] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/5] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/5] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/5] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...

  --- Round 1: Normal Chat Completion Summary ---
  Parse%:       100.0% (5/5)
  Strict%:      0.0% (0/5)
  Alias-Strict%:0.0% (0/5)
  Proj-Strict%: 0.0% (0/5)
  Proj-Alias%:  60.0% (3/5)
  Errors:       0
  Total time:   12.7s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=0.0%/60.0%

============================================================
Round: Round 2: Constrained (response_format=json_object)
  Mode: constrained (response_format=json_object)
  Prompts: 5
============================================================
  [1/5] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/5] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/5] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/5] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/5] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...

  --- Round 2: Constrained (response_format=json_object) Summary ---
  Parse%:       100.0% (5/5)
  Strict%:      0.0% (0/5)
  Alias-Strict%:0.0% (0/5)
  Proj-Strict%: 0.0% (0/5)
  Proj-Alias%:  60.0% (3/5)
  Errors:       0
  Total time:   12.76s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=0.0%/60.0%

============================================================
Round: Round 3: Schema-Strict Constrained
  Mode: schema-strict constrained
  Prompts: 5
============================================================
  [1/5] extr ext_001: parse=Y strict=N alias=N proj=Y/Y | {"鲁迅": {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "出生日期": "1881年9月25日"}}...
  [2/5] extr ext_002: parse=Y strict=N alias=N proj=Y/Y | {"西湖": {"面积": "约6.39平方公里", "位于": "杭州市西湖区"}}...
  [3/5] extr ext_003: parse=Y strict=N alias=N proj=Y/Y | {"阿里巴巴集团": {"成立时间": "1999年", "创办者": "马云"}}...
  [4/5] extr ext_004: parse=Y strict=N alias=N proj=N/N | {"青霉素": {"别名": "盘尼西林"}}...
  [5/5] extr ext_005: parse=Y strict=N alias=N proj=Y/Y | {"埃菲尔铁塔": {"建筑师": "古斯塔夫·埃菲尔", "位于": "巴黎战神广场"}}...

  --- Round 3: Schema-Strict Constrained Summary ---
  Parse%:       100.0% (5/5)
  Strict%:      0.0% (0/5)
  Alias-Strict%:0.0% (0/5)
  Proj-Strict%: 80.0% (4/5)
  Proj-Alias%:  80.0% (4/5)
  Errors:       0
  Total time:   11.87s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=80.0%/80.0%

======================================================================
STABILITY CHECK COMPARISON
======================================================================
Round                                           Parse%   Strict%   Alias-S%    Proj%   Proj-A%
----------------------------------------------------------------------------------------------
Round 1: Normal Chat Completion               100.0%    0.0%     0.0%    0.0%    60.0%
Round 2: Constrained (response_format=json_object) 100.0%    0.0%     0.0%    0.0%    60.0% [constrained]
Round 3: Schema-Strict Constrained            100.0%    0.0%     0.0%   80.0%    80.0% [strict]

--- Reference: 6C Offline Results (qwen_lora) ---
Config                                          Parse%   Strict%   Alias-S%
6C offline (run_instructie_eval.py)             97.5%     7.5%     15.0%

Results saved to results\vllm_benchmark_schema_strict\stability_20260521_204245.json
CSV saved to results\vllm_benchmark_schema_strict\stability_summary_20260521_204245.csv

~~~

## 2026-05-21 20:45:20 - Py compile after strict prompt examples

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\check_structured_stability.py
~~~

~~~text

~~~

## 2026-05-21 20:46:12 - Quick schema strict stability check with examples limit 5

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\check_structured_stability.py --base-url http://localhost:8000 --rounds 3 --limit 5 --output-dir results\vllm_benchmark_schema_strict
~~~

~~~text
============================================================
  Structured Output Stability Check — vLLM Deployed Model
  Target:      http://localhost:8000
  Eval file:   prompts_instructie.json
  Prompts:     40
  Rounds:      3
  Time:        2026-05-21 20:45:30
============================================================

Server health: OK
  Model: /mnt/e/MicroLM/outputs/qwen_lora_merged_final

============================================================
Round: Round 1: Normal Chat Completion
  Mode: normal chat completion
  Prompts: 5
============================================================
  [1/5] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/5] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/5] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/5] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/5] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...

  --- Round 1: Normal Chat Completion Summary ---
  Parse%:       100.0% (5/5)
  Strict%:      0.0% (0/5)
  Alias-Strict%:0.0% (0/5)
  Proj-Strict%: 0.0% (0/5)
  Proj-Alias%:  60.0% (3/5)
  Errors:       0
  Total time:   12.66s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=0.0%/60.0%

============================================================
Round: Round 2: Constrained (response_format=json_object)
  Mode: constrained (response_format=json_object)
  Prompts: 5
============================================================
  [1/5] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/5] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/5] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/5] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/5] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...

  --- Round 2: Constrained (response_format=json_object) Summary ---
  Parse%:       100.0% (5/5)
  Strict%:      0.0% (0/5)
  Alias-Strict%:0.0% (0/5)
  Proj-Strict%: 0.0% (0/5)
  Proj-Alias%:  60.0% (3/5)
  Errors:       0
  Total time:   12.77s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=0.0%/60.0%

============================================================
Round: Round 3: Schema-Strict Constrained
  Mode: schema-strict constrained
  Prompts: 5
============================================================
  [1/5] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "职业": "作家"}}...
  [2/5] extr ext_002: parse=Y strict=N alias=N proj=Y/Y | {"位于": {"位于": "浙江省"}, "西湖": {"面积": "约6.39平方公里"}}...
  [3/5] extr ext_003: parse=Y strict=N alias=N proj=Y/Y | {"阿里巴巴集团": {"成立时间": "1999年", "创办者": "马云"}, "淘宝网": {"所属": "阿里巴巴集团"}, "天猫": {"所属":...
  [4/5] extr ext_004: parse=Y strict=N alias=N proj=Y/Y | {"青霉素": {"别名": "盘尼西林", "发现者或发明者": "亚历山大·弗莱明"}}...
  [5/5] extr ext_005: parse=Y strict=Y alias=Y proj=N/N | {"位于": {"建筑师": "古斯塔夫·埃菲尔", "别名": "铁塔"}}...

  --- Round 3: Schema-Strict Constrained Summary ---
  Parse%:       100.0% (5/5)
  Strict%:      20.0% (1/5)
  Alias-Strict%:20.0% (1/5)
  Proj-Strict%: 60.0% (3/5)
  Proj-Alias%:  60.0% (3/5)
  Errors:       0
  Total time:   11.96s
  By group:
    extraction             P=100.0% S=20.0% A=20.0% Proj=60.0%/60.0%

======================================================================
STABILITY CHECK COMPARISON
======================================================================
Round                                           Parse%   Strict%   Alias-S%    Proj%   Proj-A%
----------------------------------------------------------------------------------------------
Round 1: Normal Chat Completion               100.0%    0.0%     0.0%    0.0%    60.0%
Round 2: Constrained (response_format=json_object) 100.0%    0.0%     0.0%    0.0%    60.0% [constrained]
Round 3: Schema-Strict Constrained            100.0%   20.0%    20.0%   60.0%    60.0% [strict]

--- Reference: 6C Offline Results (qwen_lora) ---
Config                                          Parse%   Strict%   Alias-S%
6C offline (run_instructie_eval.py)             97.5%     7.5%     15.0%

Results saved to results\vllm_benchmark_schema_strict\stability_20260521_204612.json
CSV saved to results\vllm_benchmark_schema_strict\stability_summary_20260521_204612.csv

~~~

## 2026-05-21 20:51:35 - Full schema strict stability check limit 40

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\check_structured_stability.py --base-url http://localhost:8000 --rounds 3 --limit 40 --output-dir results\vllm_benchmark_schema_strict
~~~

~~~text
============================================================
  Structured Output Stability Check — vLLM Deployed Model
  Target:      http://localhost:8000
  Eval file:   prompts_instructie.json
  Prompts:     40
  Rounds:      3
  Time:        2026-05-21 20:46:33
============================================================

Server health: OK
  Model: /mnt/e/MicroLM/outputs/qwen_lora_merged_final

============================================================
Round: Round 1: Normal Chat Completion
  Mode: normal chat completion
  Prompts: 40
============================================================
  [1/40] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/40] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/40] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/40] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/40] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...
  [6/40] extr ext_006: parse=Y strict=N alias=N proj=N/N | {"2008年北京夏季奥林匹克运动会": {"主办城市": "中国北京", "开始时间": "2008年8月8日", "结束时间": "2008年8月24日"}...
  [7/40] extr ext_007: parse=Y strict=N alias=N proj=N/Y | {"糖尿病": {"主要症状": ["多饮", "多尿", "多食和体重下降"], "并发症": ["心血管疾病", "视网膜病变"]}}...
  [8/40] extr ext_008: parse=Y strict=N alias=N proj=N/Y | {"上海虹桥站": {"所属单位": "中国铁路上海局集团有限公司", "启用时间": "2010年7月1日", "位置": "上海市闵行区"}, "闵行区":...
  [9/40] extr ext_009: parse=Y strict=N alias=N proj=Y/Y | {"宝剑": {"材料": "铁或钢", "产地": ["湖北省江陵县", "中国"], "长度": "55.7厘米"}, "越王勾践剑": {"产地": "湖...
  [10/40] extr ext_010: parse=Y strict=N alias=N proj=N/N | {"三体": {"作者": "刘慈欣", "出版社": "重庆出版社", "获奖": ["雨果奖最佳长篇小说奖", "最佳长篇小说奖"]}, "刘慈欣": {"...
  [11/40] extr ext_011: parse=Y strict=N alias=N proj=N/N | {"大熊猫": {"别称": "熊猫", "分类": ["国家一级保护动物", "珍稀动物"], "拉丁名": "Ailuropoda melanoleuca"...
  [12/40] extr ext_012: parse=Y strict=N alias=N proj=N/N | {"木星": {"直径": "142984千米", "距离太阳": "7.78亿千米"}, "太阳系": {"最大行星": "木星"}}...
  [13/40] extr ext_013: parse=Y strict=N alias=N proj=N/N | {"阿尔伯特·爱因斯坦": {"出生日期": "1879年3月14日", "国籍": "德国", "配偶": "米列娃·马里奇", "成就": ["诺贝尔物理学...
  [14/40] extr ext_014: parse=Y strict=N alias=N proj=N/N | {"第二次世界大战": {"参与方": ["同盟国", "轴心国"], "开始时间": "1939年9月1日", "结束时间": "1945年9月2日", "伤...
  [15/40] extr ext_015: parse=Y strict=N alias=N proj=N/N | {"联合国": {"主要机构": ["安全理事会", "联合国大会", "国际法院"], "会员国": "193个"}, "联合国大会": {"主要机构": "...
  [16/40] extr ext_016: parse=Y strict=N alias=N proj=Y/Y | {"哈利·波特": {"作者": "J.K.罗琳", "出版社": "英国布鲁姆斯伯里出版社"}, "哈利·波特与魔法石": {"作者": "J.K.罗琳"}}...
  [17/40] extr ext_017: parse=Y strict=N alias=N proj=N/N | {"黑火药": {"别名": ["火药", "有烟火药"], "用途": ["制造火炮和烟花", "军事和民用"]}, "硝石": {"用途": "制造火炮和烟...
  [18/40] extr ext_018: parse=Y strict=N alias=N proj=Y/Y | {"长江": {"长度": "约6300千米", "起点": "青藏高原唐古拉山脉", "终点": "东海"}}...
  [19/40] sche sc_001: parse=Y strict=N alias=N proj=N/N | {"姚明": {"出生日期": "1980年9月12日", "国籍": "中国", "职业": "前中国职业篮球运动员"}, "姚明": {"出生地": "上海...
  [20/40] sche sc_002: parse=Y strict=N alias=N proj=N/N | {"腾讯公司": {"创始人": "马化腾", "成立时间": "1998年"}}...
  [21/40] sche sc_003: parse=Y strict=N alias=N proj=N/N | {"巴黎": {"文化": "巴黎圣母院"}, "巴黎圣母院": {"宗教": "天主教", "建筑材料": "砖石", "建造时间": "1163年", "类...
  [22/40] sche sc_004: parse=Y strict=N alias=N proj=N/N | {"肺炎": {"症状": ["发热", "呼吸困难", "咳嗽"], "治疗方法": "抗生素"}}...
  [23/40] sche sc_005: parse=Y strict=N alias=N proj=N/N | {"1969年7月20日": {"事件": "阿波罗11号任务成功将宇航员送上月球", "人物": ["尼尔·阿姆斯特朗", "阿波罗11号任务"]}, "尼尔...
  [24/40] sche sc_006: parse=Y strict=N alias=N proj=N/Y | {"X射线": {"发明者": "威廉·伦琴", "别称": "伦琴射线"}}...
  [25/40] sche sc_007: parse=Y strict=N alias=N proj=N/N | {"北京南站": {"车站等级": "特等站", "所属公司": "中国铁路北京局集团有限公司"}, "丰台区": {"所属城市": "北京市"}}...
  [26/40] sche sc_008: parse=Y strict=N alias=N proj=N/N | {"华南虎": {"保护级别": "国家一级保护动物"}}...
  [27/40] sche sc_009: parse=Y strict=N alias=N proj=Y/Y | {"百年孤独": {"作者": "加西亚·马尔克斯", "奖项": ["诺贝尔文学奖", "魔幻现实主义文学"], "出版时间": "1967年"}}...
  [28/40] sche sc_010: parse=Y strict=N alias=N proj=N/N | {"火星": {"直径": "约6779千米", "表面平均温度": "-63℃"}}...
  [29/40] sche sc_011: parse=Y strict=N alias=N proj=N/N | {"司南": {"材料": "天然磁石"}}...
  [30/40] sche sc_012: parse=Y strict=N alias=N proj=N/Y | {"死海": {"海拔": "-430米", "面积": "810平方公里", "位置": ["约旦", "巴勒斯坦", "以色列"]}}...
  [31/40] form ff_001: parse=Y strict=N alias=N proj=N/N | {"李白": {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}}...
  [32/40] form ff_002: parse=Y strict=N alias=N proj=Y/Y | {"江西省": {"位于": "中国"}, "鄱阳湖": {"面积": "约3150平方公里"}}...
  [33/40] form ff_003: parse=Y strict=N alias=N proj=N/N | {"华为技术有限公司": {"成立时间": "1987年", "位于": "广东省深圳市"}, "深圳市": {"位于": "广东省"}}...
  [34/40] form ff_004: parse=Y strict=N alias=N proj=Y/Y | {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}}...
  [35/40] form ff_005: parse=Y strict=N alias=N proj=N/N | {"2008年汶川地震": {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}}...
  [36/40] form ff_006: parse=Y strict=N alias=N proj=N/N | {"悉尼歌剧院": {"位于": "澳大利亚悉尼港口", "创建或成立时间": "1973年"}, "悉尼港口": {"位于": "澳大利亚"}}...
  [37/40] form ff_007: parse=Y strict=N alias=N proj=N/N | {"北京市": {"位于": "中国"}, "北京地铁1号线": {"开通时间": "1969年10月1日", "位于": "北京市"}}...
  [38/40] form ff_008: parse=Y strict=N alias=N proj=N/N | {"高血压": {"常见症状": ["头痛", "头晕", "心悸"], "并发症": ["冠心病", "脑卒中", "肾功能不全"]}}...
  [39/40] form ff_009: parse=Y strict=N alias=N proj=Y/Y | {"土星": {"属于": "太阳系", "直径": "116464千米"}}...
  [40/40] form ff_010: parse=Y strict=N alias=N proj=Y/Y | {"红楼梦": {"作者": "曹雪芹", "成就": "中国古典小说的巅峰之作", "出版时间": "未明确"}}...

  --- Round 1: Normal Chat Completion Summary ---
  Parse%:       100.0% (40/40)
  Strict%:      0.0% (0/40)
  Alias-Strict%:0.0% (0/40)
  Proj-Strict%: 20.0% (8/40)
  Proj-Alias%:  37.5% (15/40)
  Errors:       0
  Total time:   99.87s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=16.7%/44.4%
    schema_constraint      P=100.0% S=0.0% A=0.0% Proj=8.3%/25.0%
    format_following       P=100.0% S=0.0% A=0.0% Proj=40.0%/40.0%

============================================================
Round: Round 2: Constrained (response_format=json_object)
  Mode: constrained (response_format=json_object)
  Prompts: 40
============================================================
  [1/40] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/40] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/40] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/40] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/40] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...
  [6/40] extr ext_006: parse=Y strict=N alias=N proj=N/N | {"2008年北京夏季奥林匹克运动会": {"主办城市": "中国北京", "开始时间": "2008年8月8日", "结束时间": "2008年8月24日"}...
  [7/40] extr ext_007: parse=Y strict=N alias=N proj=N/Y | {"糖尿病": {"主要症状": ["多饮", "多尿", "多食和体重下降"], "并发症": ["心血管疾病", "视网膜病变"]}}...
  [8/40] extr ext_008: parse=Y strict=N alias=N proj=N/Y | {"上海虹桥站": {"所属单位": "中国铁路上海局集团有限公司", "启用时间": "2010年7月1日", "位置": "上海市闵行区"}, "闵行区":...
  [9/40] extr ext_009: parse=Y strict=N alias=N proj=Y/Y | {"宝剑": {"材料": "铁或钢", "产地": ["湖北省江陵县", "中国"], "长度": "55.7厘米"}, "越王勾践剑": {"产地": "湖...
  [10/40] extr ext_010: parse=Y strict=N alias=N proj=N/N | {"三体": {"作者": "刘慈欣", "出版社": "重庆出版社", "获奖": ["雨果奖最佳长篇小说奖", "最佳长篇小说奖"]}, "刘慈欣": {"...
  [11/40] extr ext_011: parse=Y strict=N alias=N proj=N/N | {"大熊猫": {"别称": "熊猫", "分类": ["国家一级保护动物", "珍稀动物"], "拉丁名": "Ailuropoda melanoleuca"...
  [12/40] extr ext_012: parse=Y strict=N alias=N proj=N/N | {"木星": {"直径": "142984千米", "距离太阳": "7.78亿千米"}, "太阳系": {"最大行星": "木星"}}...
  [13/40] extr ext_013: parse=Y strict=N alias=N proj=N/N | {"阿尔伯特·爱因斯坦": {"出生日期": "1879年3月14日", "国籍": "德国", "配偶": "米列娃·马里奇", "成就": ["诺贝尔物理学...
  [14/40] extr ext_014: parse=Y strict=N alias=N proj=N/N | {"第二次世界大战": {"参与方": ["同盟国", "轴心国"], "开始时间": "1939年9月1日", "结束时间": "1945年9月2日", "伤...
  [15/40] extr ext_015: parse=Y strict=N alias=N proj=N/N | {"联合国": {"主要机构": ["安全理事会", "联合国大会", "国际法院"], "会员国": "193个"}, "联合国大会": {"主要机构": "...
  [16/40] extr ext_016: parse=Y strict=N alias=N proj=Y/Y | {"哈利·波特": {"作者": "J.K.罗琳", "出版社": "英国布鲁姆斯伯里出版社"}, "哈利·波特与魔法石": {"作者": "J.K.罗琳"}}...
  [17/40] extr ext_017: parse=Y strict=N alias=N proj=N/N | {"黑火药": {"别名": ["火药", "有烟火药"], "用途": ["制造火炮和烟花", "军事和民用"]}, "硝石": {"用途": "制造火炮和烟...
  [18/40] extr ext_018: parse=Y strict=N alias=N proj=Y/Y | {"长江": {"长度": "约6300千米", "起点": "青藏高原唐古拉山脉", "终点": "东海"}}...
  [19/40] sche sc_001: parse=Y strict=N alias=N proj=N/N | {"姚明": {"出生日期": "1980年9月12日", "国籍": "中国", "职业": "前中国职业篮球运动员"}, "姚明": {"出生地": "上海...
  [20/40] sche sc_002: parse=Y strict=N alias=N proj=N/N | {"腾讯公司": {"创始人": "马化腾", "成立时间": "1998年"}}...
  [21/40] sche sc_003: parse=Y strict=N alias=N proj=N/N | {"巴黎": {"文化": "巴黎圣母院"}, "巴黎圣母院": {"宗教": "天主教", "建筑材料": "砖石", "建造时间": "1163年", "类...
  [22/40] sche sc_004: parse=Y strict=N alias=N proj=N/N | {"肺炎": {"症状": ["发热", "呼吸困难", "咳嗽"], "治疗方法": "抗生素"}}...
  [23/40] sche sc_005: parse=Y strict=N alias=N proj=N/N | {"1969年7月20日": {"事件": "阿波罗11号任务成功将宇航员送上月球", "人物": ["尼尔·阿姆斯特朗", "阿波罗11号任务"]}, "尼尔...
  [24/40] sche sc_006: parse=Y strict=N alias=N proj=N/Y | {"X射线": {"发明者": "威廉·伦琴", "别称": "伦琴射线"}}...
  [25/40] sche sc_007: parse=Y strict=N alias=N proj=N/N | {"北京南站": {"车站等级": "特等站", "所属公司": "中国铁路北京局集团有限公司"}, "丰台区": {"所属城市": "北京市"}}...
  [26/40] sche sc_008: parse=Y strict=N alias=N proj=N/N | {"华南虎": {"保护级别": "国家一级保护动物"}}...
  [27/40] sche sc_009: parse=Y strict=N alias=N proj=Y/Y | {"百年孤独": {"作者": "加西亚·马尔克斯", "奖项": ["诺贝尔文学奖", "魔幻现实主义文学"], "出版时间": "1967年"}}...
  [28/40] sche sc_010: parse=Y strict=N alias=N proj=N/N | {"火星": {"直径": "约6779千米", "表面平均温度": "-63℃"}}...
  [29/40] sche sc_011: parse=Y strict=N alias=N proj=N/N | {"司南": {"材料": "天然磁石"}}...
  [30/40] sche sc_012: parse=Y strict=N alias=N proj=N/Y | {"死海": {"海拔": "-430米", "面积": "810平方公里", "位置": ["约旦", "巴勒斯坦", "以色列"]}}...
  [31/40] form ff_001: parse=Y strict=N alias=N proj=N/N | {"李白": {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}}...
  [32/40] form ff_002: parse=Y strict=N alias=N proj=Y/Y | {"江西省": {"位于": "中国"}, "鄱阳湖": {"面积": "约3150平方公里"}}...
  [33/40] form ff_003: parse=Y strict=N alias=N proj=N/N | {"华为技术有限公司": {"成立时间": "1987年", "位于": "广东省深圳市"}, "深圳市": {"位于": "广东省"}}...
  [34/40] form ff_004: parse=Y strict=N alias=N proj=Y/Y | {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}}...
  [35/40] form ff_005: parse=Y strict=N alias=N proj=N/N | {"2008年汶川地震": {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}}...
  [36/40] form ff_006: parse=Y strict=N alias=N proj=N/N | {"悉尼歌剧院": {"位于": "澳大利亚悉尼港口", "创建或成立时间": "1973年"}, "悉尼港口": {"位于": "澳大利亚"}}...
  [37/40] form ff_007: parse=Y strict=N alias=N proj=N/N | {"北京市": {"位于": "中国"}, "北京地铁1号线": {"开通时间": "1969年10月1日", "位于": "北京市"}}...
  [38/40] form ff_008: parse=Y strict=N alias=N proj=N/N | {"高血压": {"常见症状": ["头痛", "头晕", "心悸"], "并发症": ["冠心病", "脑卒中", "肾功能不全"]}}...
  [39/40] form ff_009: parse=Y strict=N alias=N proj=Y/Y | {"土星": {"属于": "太阳系", "直径": "116464千米"}}...
  [40/40] form ff_010: parse=Y strict=N alias=N proj=Y/Y | {"红楼梦": {"作者": "曹雪芹", "成就": "中国古典小说的巅峰之作", "出版时间": "未明确"}}...

  --- Round 2: Constrained (response_format=json_object) Summary ---
  Parse%:       100.0% (40/40)
  Strict%:      0.0% (0/40)
  Alias-Strict%:0.0% (0/40)
  Proj-Strict%: 20.0% (8/40)
  Proj-Alias%:  37.5% (15/40)
  Errors:       0
  Total time:   99.59s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=16.7%/44.4%
    schema_constraint      P=100.0% S=0.0% A=0.0% Proj=8.3%/25.0%
    format_following       P=100.0% S=0.0% A=0.0% Proj=40.0%/40.0%

============================================================
Round: Round 3: Schema-Strict Constrained
  Mode: schema-strict constrained
  Prompts: 40
============================================================
  [1/40] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "职业": "作家"}}...
  [2/40] extr ext_002: parse=Y strict=N alias=N proj=Y/Y | {"西湖": {"面积": "约6.39平方公里", "位于": "杭州市西湖区"}, "杭州市西湖区": {"位于": "浙江省"}}...
  [3/40] extr ext_003: parse=Y strict=N alias=N proj=Y/Y | {"阿里巴巴集团": {"成立时间": "1999年", "创办者": "马云"}, "淘宝网": {"所属": "阿里巴巴集团"}, "天猫": {"所属":...
  [4/40] extr ext_004: parse=Y strict=N alias=N proj=Y/Y | {"青霉素": {"别名": "盘尼西林", "发现者或发明者": "亚历山大·弗莱明"}}...
  [5/40] extr ext_005: parse=Y strict=Y alias=Y proj=N/N | {"位于": {"建筑师": "古斯塔夫·埃菲尔", "别名": "铁塔"}}...
  [6/40] extr ext_006: parse=Y strict=Y alias=Y proj=Y/Y | {"发生地点": "中国北京", "参与者": ["中国", "204个国家和地区"], "发生时间": "2008年8月8日至8月24日"}...
  [7/40] extr ext_007: parse=Y strict=Y alias=Y proj=Y/Y | {"症状": ["多饮", "多尿", "多食和体重下降"], "治疗方法": ["胰岛素注射", "口服降糖药"], "病因": "高血糖"}...
  [8/40] extr ext_008: parse=Y strict=N alias=N proj=Y/Y | {"位于": {"城市": "上海市", "区域": "闵行区"}, "开通时间": "2010年7月1日"}...
  [9/40] extr ext_009: parse=Y strict=Y alias=Y proj=Y/Y | {"材料": ["铁", "钢"], "长度": "55.7厘米", "宽度": "4.6厘米", "产地": "湖北省江陵县", "别名": "越王勾践剑"}...
  [10/40] extr ext_010: parse=Y strict=Y alias=Y proj=Y/Y | {"作者": "刘慈欣", "出版时间": "2008年1月", "产地": "中国", "成就": "雨果奖最佳长篇小说奖"}...
  [11/40] extr ext_011: parse=Y strict=N alias=N proj=Y/Y | {"分布": {"类别": "国家一级保护动物", "所属省份": ["四川", "甘肃", "陕西"], "产地": "中国"}, "学名": "Ailuro...
  [12/40] extr ext_012: parse=Y strict=Y alias=Y proj=Y/Y | {"属于": "太阳系", "直径": "142984千米"}...
  [13/40] extr ext_013: parse=Y strict=N alias=N proj=N/N | {"出生地": "德国乌尔姆", "职业": "理论物理学家"}...
  [14/40] extr ext_014: parse=Y strict=N alias=N proj=Y/Y | {"第二次世界大战": {"起因": "纳粹德国入侵波兰", "参与者": ["同盟国", "轴心国"], "发生时间": "1939年9月1日至1945年9月...
  [15/40] extr ext_015: parse=Y strict=N alias=N proj=Y/Y | {"联合国": {"成立时间": "1945年10月24日", "位于": "美国纽约", "成员": "联合国大会"}}...
  [16/40] extr ext_016: parse=Y strict=Y alias=Y proj=Y/Y | {"作者": "J.K.罗琳", "出版时间": "1997年", "出版商": "英国布鲁姆斯伯里出版社", "成就": "全球票房超过77亿美元", "产地...
  [17/40] extr ext_017: parse=Y strict=Y alias=Y proj=Y/Y | {"组成": ["硝石", "硫磺", "木炭"], "用途": ["军事和民用", "制造火炮和烟花"], "别名": ["火药", "有烟火药"], "发现...
  [18/40] extr ext_018: parse=Y strict=N alias=N proj=Y/Y | {"长江": {"长度": "6300千米", "位于": "中国"}}...
  [19/40] sche sc_001: parse=Y strict=N alias=N proj=N/N | {"出生地": "上海市", "出生日期": "1980年9月12日"}...
  [20/40] sche sc_002: parse=Y strict=Y alias=Y proj=Y/Y | {"成立时间": "1998年", "创办者": "马化腾", "位于": "广东省深圳市南山区"}...
  [21/40] sche sc_003: parse=Y strict=Y alias=Y proj=N/N | {"用途": "哥特式天主教大教堂", "用途": "大教堂", "位于": "巴黎西岱岛"}...
  [22/40] sche sc_004: parse=Y strict=Y alias=Y proj=N/N | {"症状": "咳嗽、发热、呼吸困难", "所属科室": "呼吸系统", "治疗方法": "抗生素"}...
  [23/40] sche sc_005: parse=Y strict=Y alias=Y proj=Y/Y | {"发生时间": "1969年7月20日", "参与者": ["尼尔·阿姆斯特朗", "阿波罗11号"], "发生地点": "月球"}...
  [24/40] sche sc_006: parse=Y strict=Y alias=Y proj=Y/Y | {"发现者或发明者": "威廉·伦琴", "用途": ["医学影像诊断", "工业无损检测", "安全检查"], "别名": "伦琴射线"}...
  [25/40] sche sc_007: parse=Y strict=N alias=N proj=Y/Y | {"位于": {"城市": "北京市丰台区"}, "车站等级": "特等站"}...
  [26/40] sche sc_008: parse=Y strict=Y alias=Y proj=Y/Y | {"保护级别": ["国家一级保护动物", "国家二级保护动物"], "用途": "中国特有的虎亚种", "学名": "Panthera tigris amoy...
  [27/40] sche sc_009: parse=Y strict=Y alias=Y proj=Y/Y | {"作者": "加西亚·马尔克斯", "出版时间": "1967年", "成就": "诺贝尔文学奖"}...
  [28/40] sche sc_010: parse=Y strict=N alias=N proj=N/N | {"属于": "太阳系"}...
  [29/40] sche sc_011: parse=Y strict=Y alias=Y proj=N/N | {"材料": "天然磁石", "用途": "辨别方向"}...
  [30/40] sche sc_012: parse=Y strict=Y alias=Y proj=Y/Y | {"位于": {"面积": "810平方公里", "海拔": "-430米"}, "面积": "810平方公里"}...
  [31/40] form ff_001: parse=Y strict=N alias=N proj=Y/Y | {"李白": {"职业": "唐代著名的浪漫主义诗人", "出生日期": "701年", "出生地": "碎叶城"}}...
  [32/40] form ff_002: parse=Y strict=N alias=N proj=N/N | {"鄱阳湖": {"位于": "江西省"}, "江西省": {"位于": "中国"}}...
  [33/40] form ff_003: parse=Y strict=Y alias=Y proj=Y/Y | {"创办者": "任正非", "成立时间": "1987年", "位于": "广东省深圳市"}...
  [34/40] form ff_004: parse=Y strict=N alias=N proj=N/N | {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}, "DNA双螺旋结构": {"用途": "为现代分子生物学奠定了基础"}}...
  [35/40] form ff_005: parse=Y strict=N alias=N proj=N/N | {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}...
  [36/40] form ff_006: parse=Y strict=N alias=N proj=N/N | {"悉尼歌剧院": {"创建或成立时间": "1973年", "位于": "澳大利亚悉尼港口"}, "约恩·乌松": {"位于": "丹麦"}}...
  [37/40] form ff_007: parse=Y strict=Y alias=Y proj=Y/Y | {"开通时间": "1969年10月1日", "线路": "北京地铁1号线"}...
  [38/40] form ff_008: parse=Y strict=Y alias=Y proj=Y/Y | {"症状": ["头痛", "头晕", "心悸"], "治疗方法": ["降压药物治疗", "生活方式调整"], "常见并发症": ["冠心病", "脑卒中",...
  [39/40] form ff_009: parse=Y strict=N alias=N proj=Y/Y | {"土星": {"属于": "太阳系", "直径": "116464千米"}, "太阳系": {"属于": "宇宙"}}...
  [40/40] form ff_010: parse=Y strict=Y alias=Y proj=Y/Y | {"成就": "中国古典小说的巅峰之作", "作者": "曹雪芹"}...

  --- Round 3: Schema-Strict Constrained Summary ---
  Parse%:       100.0% (40/40)
  Strict%:      52.5% (21/40)
  Alias-Strict%:52.5% (21/40)
  Proj-Strict%: 70.0% (28/40)
  Proj-Alias%:  70.0% (28/40)
  Errors:       0
  Total time:   98.09s
  By group:
    extraction             P=100.0% S=44.4% A=44.4% Proj=83.3%/83.3%
    schema_constraint      P=100.0% S=75.0% A=75.0% Proj=58.3%/58.3%
    format_following       P=100.0% S=40.0% A=40.0% Proj=60.0%/60.0%

======================================================================
STABILITY CHECK COMPARISON
======================================================================
Round                                           Parse%   Strict%   Alias-S%    Proj%   Proj-A%
----------------------------------------------------------------------------------------------
Round 1: Normal Chat Completion               100.0%    0.0%     0.0%   20.0%    37.5%
Round 2: Constrained (response_format=json_object) 100.0%    0.0%     0.0%   20.0%    37.5% [constrained]
Round 3: Schema-Strict Constrained            100.0%   52.5%    52.5%   70.0%    70.0% [strict]

--- Reference: 6C Offline Results (qwen_lora) ---
Config                                          Parse%   Strict%   Alias-S%
6C offline (run_instructie_eval.py)             97.5%     7.5%     15.0%

Results saved to results\vllm_benchmark_schema_strict\stability_20260521_205135.json
CSV saved to results\vllm_benchmark_schema_strict\stability_summary_20260521_205135.csv

~~~

## 2026-05-21 20:51:54 - Read schema strict final results

~~~powershell
Get-Content -Encoding utf8 results\vllm_benchmark_schema_strict\stability_summary_20260521_205135.csv
Get-Content -Encoding utf8 results\vllm_benchmark_schema_strict\stability_20260521_205135.json
~~~

~~~text
round,mode,total,parse_rate,strict_rate,alias_strict_rate,projected_strict_rate,projected_alias_strict_rate,extraction_P,extraction_S,extraction_A,extraction_projected,extraction_projected_alias,schema_P,schema_S,schema_A,schema_projected,schema_projected_alias,format_P,format_S,format_A,format_projected,format_projected_alias,avg_latency_s
Round 1: Normal Chat Completion,normal,40,1.0,0.0,0.0,0.2,0.375,1.0,0.0,0.0,0.1667,0.4444,1.0,0.0,0.0,0.0833,0.25,1.0,0.0,0.0,0.4,0.4,2.497
Round 2: Constrained (response_format=json_object),constrained,40,1.0,0.0,0.0,0.2,0.375,1.0,0.0,0.0,0.1667,0.4444,1.0,0.0,0.0,0.0833,0.25,1.0,0.0,0.0,0.4,0.4,2.49
Round 3: Schema-Strict Constrained,schema_strict,40,1.0,0.525,0.525,0.7,0.7,1.0,0.4444,0.4444,0.8333,0.8333,1.0,0.75,0.75,0.5833,0.5833,1.0,0.4,0.4,0.6,0.6,2.452
{
  "check_config": {
    "base_url": "http://localhost:8000",
    "eval_file": "E:\\MicroLM\\eval\\prompts_instructie.json",
    "timestamp": "2026-05-21 20:51:35",
    "rounds": 3
  },
  "rounds": [
    {
      "round": "Round 1: Normal Chat Completion",
      "mode": "normal",
      "total": 40,
      "parse_rate": 1.0,
      "strict_rate": 0.0,
      "alias_strict_rate": 0.0,
      "projected_strict_rate": 0.2,
      "projected_alias_strict_rate": 0.375,
      "by_group": {
        "extraction": {
          "total": 18,
          "parse_rate": 1.0,
          "strict_rate": 0.0,
          "alias_strict_rate": 0.0,
          "projected_strict_rate": 0.1667,
          "projected_alias_strict_rate": 0.4444
        },
        "schema_constraint": {
          "total": 12,
          "parse_rate": 1.0,
          "strict_rate": 0.0,
          "alias_strict_rate": 0.0,
          "projected_strict_rate": 0.0833,
          "projected_alias_strict_rate": 0.25
        },
        "format_following": {
          "total": 10,
          "parse_rate": 1.0,
          "strict_rate": 0.0,
          "alias_strict_rate": 0.0,
          "projected_strict_rate": 0.4,
          "projected_alias_strict_rate": 0.4
        }
      },
      "avg_latency_s": 2.497
    },
    {
      "round": "Round 2: Constrained (response_format=json_object)",
      "mode": "constrained",
      "total": 40,
      "parse_rate": 1.0,
      "strict_rate": 0.0,
      "alias_strict_rate": 0.0,
      "projected_strict_rate": 0.2,
      "projected_alias_strict_rate": 0.375,
      "by_group": {
        "extraction": {
          "total": 18,
          "parse_rate": 1.0,
          "strict_rate": 0.0,
          "alias_strict_rate": 0.0,
          "projected_strict_rate": 0.1667,
          "projected_alias_strict_rate": 0.4444
        },
        "schema_constraint": {
          "total": 12,
          "parse_rate": 1.0,
          "strict_rate": 0.0,
          "alias_strict_rate": 0.0,
          "projected_strict_rate": 0.0833,
          "projected_alias_strict_rate": 0.25
        },
        "format_following": {
          "total": 10,
          "parse_rate": 1.0,
          "strict_rate": 0.0,
          "alias_strict_rate": 0.0,
          "projected_strict_rate": 0.4,
          "projected_alias_strict_rate": 0.4
        }
      },
      "avg_latency_s": 2.49
    },
    {
      "round": "Round 3: Schema-Strict Constrained",
      "mode": "schema_strict",
      "total": 40,
      "parse_rate": 1.0,
      "strict_rate": 0.525,
      "alias_strict_rate": 0.525,
      "projected_strict_rate": 0.7,
      "projected_alias_strict_rate": 0.7,
      "by_group": {
        "extraction": {
          "total": 18,
          "parse_rate": 1.0,
          "strict_rate": 0.4444,
          "alias_strict_rate": 0.4444,
          "projected_strict_rate": 0.8333,
          "projected_alias_strict_rate": 0.8333
        },
        "schema_constraint": {
          "total": 12,
          "parse_rate": 1.0,
          "strict_rate": 0.75,
          "alias_strict_rate": 0.75,
          "projected_strict_rate": 0.5833,
          "projected_alias_strict_rate": 0.5833
        },
        "format_following": {
          "total": 10,
          "parse_rate": 1.0,
          "strict_rate": 0.4,
          "alias_strict_rate": 0.4,
          "projected_strict_rate": 0.6,
          "projected_alias_strict_rate": 0.6
        }
      },
      "avg_latency_s": 2.452
    }
  ],
  "reference_6c_offline": {
    "model": "qwen_lora",
    "parse_rate": 0.975,
    "strict_rate": 0.075,
    "alias_strict_rate": 0.15
  }
}

~~~

## 2026-05-21 20:55:25 - Verify schema strict report and final CSV

~~~powershell
$files = @(
  'results\vllm_benchmark_schema_strict\stability_20260521_205135.json',
  'reports\schema_strict_improvement_report.md',
  'reports\eval_deploy_closure.md'
)
$files | ForEach-Object { $item=Get-Item $_ -ErrorAction SilentlyContinue; [PSCustomObject]@{Path=$_; Exists=[bool]$item; Length=if($item){$item.Length}else{''}; LastWriteTime=if($item){$item.LastWriteTime}else{''}} } | Format-Table -AutoSize
Get-Content -Encoding utf8 results\vllm_benchmark_schema_strict\stability_summary_20260521_205135.csv
~~~

~~~text

Path                                                                Exists Length LastWriteTime     
----                                                                ------ ------ -------------     
results\vllm_benchmark_schema_strict\stability_20260521_205135.json   True   3705 2026/5/21 20:51:35
reports\schema_strict_improvement_report.md                           True   3362 2026/5/21 20:52:22
reports\eval_deploy_closure.md                                        True   7605 2026/5/21 20:52:41


round,mode,total,parse_rate,strict_rate,alias_strict_rate,projected_strict_rate,projected_alias_strict_rate,extraction_P,extraction_S,extraction_A,extraction_projected,extraction_projected_alias,schema_P,schema_S,schema_A,schema_projected,schema_projected_alias,format_P,format_S,format_A,format_projected,format_projected_alias,avg_latency_s
Round 1: Normal Chat Completion,normal,40,1.0,0.0,0.0,0.2,0.375,1.0,0.0,0.0,0.1667,0.4444,1.0,0.0,0.0,0.0833,0.25,1.0,0.0,0.0,0.4,0.4,2.497
Round 2: Constrained (response_format=json_object),constrained,40,1.0,0.0,0.0,0.2,0.375,1.0,0.0,0.0,0.1667,0.4444,1.0,0.0,0.0,0.0833,0.25,1.0,0.0,0.0,0.4,0.4,2.49
Round 3: Schema-Strict Constrained,schema_strict,40,1.0,0.525,0.525,0.7,0.7,1.0,0.4444,0.4444,0.8333,0.8333,1.0,0.75,0.75,0.5833,0.5833,1.0,0.4,0.4,0.6,0.6,2.452

~~~

## 2026-05-21 20:55:37 - Final vLLM health after schema strict eval

~~~powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health -TimeoutSec 5
~~~

~~~text


StatusCode        : 200
StatusDescription : OK
Content           : {}
RawContent        : HTTP/1.1 200 OK
                    Content-Length: 0
                    Date: Thu, 21 May 2026 12:55:37 GMT
                    Server: uvicorn
                    
                    
Headers           : {[Content-Length, 0], [Date, Thu, 21 May 2026 12:55:37 GMT], [Server, uvicorn]}
RawContentLength  : 0




~~~

## 2026-05-21 20:55:52 - Git status for schema strict improvement

~~~powershell
git status --short scripts\check_structured_stability.py reports\schema_strict_improvement_report.md reports\eval_deploy_closure.md reports\terminal_outputs_schema_strict.md results\vllm_benchmark_schema_strict
~~~

~~~text
 M scripts/check_structured_stability.py
?? reports/eval_deploy_closure.md
?? reports/schema_strict_improvement_report.md
?? reports/terminal_outputs_schema_strict.md
?? results/vllm_benchmark_schema_strict/

~~~

## 2026-05-21 20:57:19 - Inspect project modules for schema repair placement

~~~powershell
Get-ChildItem microlm -Recurse -Depth 2 | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
Get-ChildItem scripts -Filter '*structured*','*schema*','*vllm*' -ErrorAction SilentlyContinue | Select-Object Name,Length,LastWriteTime | Format-Table -AutoSize
~~~

~~~text
Invoke-Expression : 无法将“System.Object[]”转换为参数“Filter”所需的类型“System.String”。不支持所指定的方法。
所在位置 行:7 字符: 14
+ $out = try { Invoke-Expression $cmd 2>&1 | Out-String -Width 360 } ca ...
+              ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) [Invoke-Expression]，ParameterBindingException
    + FullyQualifiedErrorId : CannotConvertArgument,Microsoft.PowerShell.Commands.InvokeExpressionCommand
 

~~~

## 2026-05-21 20:57:33 - List structured/schema/vLLM related files

~~~powershell
rg --files microlm scripts | rg "(structured|schema|vllm|eval|repair)"
~~~

~~~text
scripts\summarize_eval_results.py
scripts\start_vllm_wsl.ps1
scripts\smoke_vllm.py
scripts\setup_vllm_wsl_admin.ps1
scripts\setup_vllm_ubuntu.sh
scripts\serve_vllm.sh
scripts\run_instructie_eval.py
scripts\run_eval_prompts.py
scripts\evaluate_qwen_valid_jsonl.py
scripts\check_structured_stability.py
scripts\bench_vllm_local.py

~~~

## 2026-05-21 20:57:42 - Inspect microlm package structure

~~~powershell
Get-ChildItem microlm | Select-Object Name,Mode,Length,LastWriteTime | Format-Table -AutoSize
Get-ChildItem microlm -Recurse -Filter __init__.py | Select-Object FullName | Format-Table -AutoSize
~~~

~~~text

Name        Mode   Length LastWriteTime     
----        ----   ------ -------------     
inference   d-----        2026/5/19 18:28:17
model       d-----        2026/5/19 18:28:18
tokenizer   d-----        2026/5/19 18:28:18
training    d-----        2026/5/19 18:28:17
__pycache__ d-----        2026/5/19 18:28:17
__init__.py -a---- 64     2026/5/19 17:26:29



FullName                                
--------                                
E:\MicroLM\microlm\__init__.py          
E:\MicroLM\microlm\inference\__init__.py
E:\MicroLM\microlm\model\__init__.py    
E:\MicroLM\microlm\tokenizer\__init__.py
E:\MicroLM\microlm\training\__init__.py 



~~~

## 2026-05-21 20:59:57 - Inspect check_structured_stability imports after schema repair module

~~~powershell
Get-Content -Encoding utf8 scripts\check_structured_stability.py -TotalCount 45
~~~

~~~text
#!/usr/bin/env python3
"""check_structured_stability.py — Verify structured output stability on vLLM-served model.

Reuses the InstructIE evaluation prompt set (eval/prompts_instructie.json) to verify
that the vLLM-deployed qwen_lora model maintains its structured output quality.

Runs TWO rounds:
  Round 1: Normal chat completion (no format constraint)
  Round 2: Constrained completion with response_format=json_object (if supported)
  Round 3: Schema-strict constrained completion (optional, --rounds 3)

For each round, computes:
  - Parse%     (JSON parseable rate)
  - Strict%    (strict schema match rate — all 4 checks pass)
  - Alias-Strict% (alias-normalized strict rate)
  - Projected-Strict% (after schema projection / field normalization postprocess)
  - Per-group breakdown (extraction / schema_constraint / format_following)

Usage:
    python scripts/check_structured_stability.py                          # full test
    python scripts/check_structured_stability.py --rounds 1               # round 1 only
    python scripts/check_structured_stability.py --base-url http://host:8001
    python scripts/check_structured_stability.py --limit 5                # quick check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
import requests
except ImportError:
    print("[ERROR] 'requests' not installed. Run: pip install requests")
    sys.exit(1)

from microlm.structured.schema_repair import (
    build_schema_strict_messages as build_shared_schema_strict_messages,
    repair_to_schema as repair_to_schema_fields,
    score_repaired_fields,
)

~~~

## 2026-05-21 21:00:40 - Py compile schema repair module and stability script

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile microlm\structured\schema_repair.py scripts\check_structured_stability.py
~~~

~~~text

~~~

## 2026-05-21 21:01:39 - Py compile structured client and repair integration

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\structured_vllm_client.py microlm\structured\schema_repair.py scripts\check_structured_stability.py
~~~

~~~text

~~~

## 2026-05-21 21:03:40 - Run structured vLLM client repaired outputs limit 40

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --eval-file eval\prompts_instructie.json --limit 40 --output results\vllm_benchmark_schema_strict\repaired_outputs_20260521.jsonl
~~~

~~~text
[1/40] ext_001 parse=True repair_strict=False missing=['出生日期']
[2/40] ext_002 parse=True repair_strict=True missing=[]
[3/40] ext_003 parse=True repair_strict=True missing=[]
[4/40] ext_004 parse=True repair_strict=True missing=[]
[5/40] ext_005 parse=True repair_strict=False missing=['位于']
[6/40] ext_006 parse=True repair_strict=True missing=[]
[7/40] ext_007 parse=True repair_strict=True missing=[]
[8/40] ext_008 parse=True repair_strict=False missing=['位于']
[9/40] ext_009 parse=True repair_strict=True missing=[]
[10/40] ext_010 parse=True repair_strict=True missing=[]
[11/40] ext_011 parse=True repair_strict=False missing=['分布']
[12/40] ext_012 parse=True repair_strict=True missing=[]
[13/40] ext_013 parse=True repair_strict=True missing=[]
[14/40] ext_014 parse=True repair_strict=True missing=[]
[15/40] ext_015 parse=True repair_strict=True missing=[]
[16/40] ext_016 parse=True repair_strict=True missing=[]
[17/40] ext_017 parse=True repair_strict=True missing=[]
[18/40] ext_018 parse=True repair_strict=True missing=[]
[19/40] sc_001 parse=True repair_strict=False missing=['国籍']
[20/40] sc_002 parse=True repair_strict=True missing=[]
[21/40] sc_003 parse=True repair_strict=False missing=[]
[22/40] sc_004 parse=True repair_strict=False missing=[]
[23/40] sc_005 parse=True repair_strict=True missing=[]
[24/40] sc_006 parse=True repair_strict=True missing=[]
[25/40] sc_007 parse=True repair_strict=False missing=['位于']
[26/40] sc_008 parse=True repair_strict=True missing=[]
[27/40] sc_009 parse=True repair_strict=True missing=[]
[28/40] sc_010 parse=True repair_strict=False missing=['直径']
[29/40] sc_011 parse=True repair_strict=False missing=[]
[30/40] sc_012 parse=True repair_strict=False missing=['位于']
[31/40] ff_001 parse=True repair_strict=False missing=['出生日期']
[32/40] ff_002 parse=True repair_strict=False missing=['面积']
[33/40] ff_003 parse=True repair_strict=True missing=[]
[34/40] ff_004 parse=True repair_strict=False missing=['发现者或发明者']
[35/40] ff_005 parse=True repair_strict=False missing=['起因']
[36/40] ff_006 parse=True repair_strict=False missing=['建筑师']
[37/40] ff_007 parse=True repair_strict=True missing=[]
[38/40] ff_008 parse=True repair_strict=True missing=[]
[39/40] ff_009 parse=True repair_strict=True missing=[]
[40/40] ff_010 parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=40
parse_rate=100.0%
repair_strict_rate=60.0%
avg_latency_s=2.438
saved=results\vllm_benchmark_schema_strict\repaired_outputs_20260521.jsonl

~~~

## 2026-05-21 21:03:56 - Inspect repaired client failure samples

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
rows=[json.loads(x) for x in Path('results/vllm_benchmark_schema_strict/repaired_outputs_20260521.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
for r in rows:
    if not r['schema_strict_after_repair']:
        print('\n', r['id'], 'missing=', r['missing_fields_after_repair'], 'extra=', r['extra_fields_after_repair'], 'enum_ok=', r['enum_ok_after_repair'])
        print('raw=', r['raw_output'][:260])
        print('repaired=', json.dumps(r['repaired'], ensure_ascii=False)[:260])"
~~~

~~~text

 ext_001 missing= ['出生日期'] extra= [] enum_ok= True
raw= {"鲁迅": {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "职业": "作家"}}
repaired= {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "职业": "作家"}

 ext_005 missing= ['位于'] extra= [] enum_ok= True
raw= {"位于": {"建筑师": "古斯塔夫·埃菲尔", "别名": "铁塔"}}
repaired= {"别名": "铁塔", "建筑师": "古斯塔夫·埃菲尔"}

 ext_008 missing= ['位于'] extra= [] enum_ok= True
raw= {"位于": {"城市": "上海市", "区域": "闵行区"}, "开通时间": "2010年7月1日"}
repaired= {"开通时间": "2010年7月1日"}

 ext_011 missing= ['分布'] extra= [] enum_ok= True
raw= {"分布": {"所属省份": ["四川", "甘肃", "陕西"], "产地": "中国"}, "学名": "Ailuropoda melanoleuca"}
repaired= {"学名": "Ailuropoda melanoleuca", "产地": "中国"}

 sc_001 missing= ['国籍'] extra= [] enum_ok= True
raw= {"出生地": "上海市", "出生日期": "1980年9月12日"}
repaired= {"出生地": "上海市", "出生日期": "1980年9月12日"}

 sc_003 missing= [] extra= [] enum_ok= False
raw= {"用途": "哥特式天主教大教堂", "用途": "大教堂", "位于": "巴黎西岱岛"}
repaired= {"位于": "巴黎西岱岛", "用途": "大教堂"}

 sc_004 missing= [] extra= [] enum_ok= False
raw= {"症状": "咳嗽、发热、呼吸困难", "所属科室": "呼吸系统", "治疗方法": "抗生素"}
repaired= {"症状": "咳嗽、发热、呼吸困难", "治疗方法": "抗生素", "所属科室": "呼吸系统"}

 sc_007 missing= ['位于'] extra= [] enum_ok= True
raw= {"位于": {"城市": "北京市丰台区"}, "车站等级": "特等站"}
repaired= {"车站等级": "特等站"}

 sc_010 missing= ['直径'] extra= [] enum_ok= True
raw= {"属于": "太阳系"}
repaired= {"属于": "太阳系"}

 sc_011 missing= [] extra= [] enum_ok= False
raw= {"材料": "天然磁石", "用途": "辨别方向"}
repaired= {"材料": "天然磁石", "用途": "辨别方向"}

 sc_012 missing= ['位于'] extra= [] enum_ok= True
raw= {"位于": {"面积": "810平方公里", "海拔": "-430米"}, "面积": "810平方公里"}
repaired= {"面积": "810平方公里", "海拔": "-430米"}

 ff_001 missing= ['出生日期'] extra= [] enum_ok= True
raw= {"李白": {"职业": "唐代著名的浪漫主义诗人", "出生地": "碎叶城"}}
repaired= {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}

 ff_002 missing= ['面积'] extra= [] enum_ok= True
raw= {"鄱阳湖": {"位于": "江西省"}, "江西省": {"位于": "中国"}}
repaired= {"位于": ["江西省", "中国"]}

 ff_004 missing= ['发现者或发明者'] extra= [] enum_ok= True
raw= {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}, "DNA双螺旋结构": {"用途": "为现代分子生物学奠定了基础"}}
repaired= {"用途": "为现代分子生物学奠定了基础"}

 ff_005 missing= ['起因'] extra= [] enum_ok= True
raw= {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}
repaired= {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日"}

 ff_006 missing= ['建筑师'] extra= [] enum_ok= True
raw= {"悉尼歌剧院": {"创建或成立时间": "1973年", "位于": "澳大利亚悉尼港口"}, "约恩·乌松": {"位于": "丹麦"}}
repaired= {"位于": ["澳大利亚悉尼港口", "丹麦"], "创建或成立时间": "1973年"}

~~~

## 2026-05-21 21:04:25 - Inspect enum constraint failure schemas

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
p=json.loads(Path('eval/prompts_instructie.json').read_text(encoding='utf-8'))
for id in ['sc_003','sc_004','sc_011']:
    item=next(x for x in p['prompts'] if x['id']==id)
    print(id, json.dumps(item['schema_def'], ensure_ascii=False))"
~~~

~~~text
sc_003 {"required_fields": ["位于", "用途"], "allowed_fields": ["位于", "创建或成立时间", "高度", "用途"], "types": {"位于": "string", "用途": "string"}, "enum_constraints": {"用途": ["住宅", "商业", "宗教", "军事", "文化", "交通", "教育"]}}
sc_004 {"required_fields": ["症状", "所属科室"], "allowed_fields": ["别名", "症状", "治疗方法", "所属科室"], "types": {"症状": "string_or_list", "所属科室": "string"}, "enum_constraints": {"所属科室": ["内科", "外科", "儿科", "妇产科", "眼科", "皮肤科", "骨科", "神经科"]}}
sc_011 {"required_fields": ["材料", "用途"], "allowed_fields": ["别名", "材料", "产地", "用途"], "types": {"材料": "string", "用途": "string"}, "enum_constraints": {"材料": ["金属", "木材", "石材", "陶瓷", "塑料", "玻璃", "橡胶", "纤维"]}}

~~~

## 2026-05-21 21:05:54 - Py compile after enhanced repair rules

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile microlm\structured\schema_repair.py scripts\structured_vllm_client.py scripts\check_structured_stability.py
~~~

~~~text

~~~

## 2026-05-21 21:06:11 - Recompute repaired strict from existing raw outputs after enhanced rules

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path; from microlm.structured import try_parse_json, repair_to_schema, score_repaired_fields
prompts=json.loads(Path('eval/prompts_instructie.json').read_text(encoding='utf-8'))['prompts']; by_id={p['id']:p for p in prompts}
rows=[json.loads(x) for x in Path('results/vllm_benchmark_schema_strict/repaired_outputs_20260521.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
ok=0
for r in rows:
    p=by_id[r['id']]; parsed, parse_ok=try_parse_json(r['raw_output']); repaired=repair_to_schema(parsed, p['schema_def'], use_aliases=True, fill_missing=False) if parse_ok else {}; score=score_repaired_fields(repaired,p['schema_def']); ok += score['schema_strict'];
    if not score['schema_strict']:
        print(r['id'], score['missing_fields'], 'enum', score['enum_ok'], 'repaired', json.dumps(repaired, ensure_ascii=False)[:180])
print('recomputed_repair_strict_rate', ok, '/', len(rows), f'{ok/len(rows):.1%}')"
~~~

~~~text
ext_001 ['出生日期'] enum True repaired {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "职业": "作家"}
ext_005 ['位于'] enum True repaired {"别名": "铁塔", "建筑师": "古斯塔夫·埃菲尔"}
sc_001 ['国籍'] enum True repaired {"出生地": "上海市", "出生日期": "1980年9月12日"}
sc_010 ['直径'] enum True repaired {"属于": "太阳系"}
sc_012 ['位于'] enum True repaired {"面积": "810平方公里", "海拔": "-430米"}
ff_001 ['出生日期'] enum True repaired {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}
ff_002 ['面积'] enum True repaired {"位于": ["江西省", "中国"]}
ff_005 ['起因'] enum True repaired {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日"}
ff_006 ['建筑师'] enum True repaired {"位于": ["澳大利亚悉尼港口", "丹麦"], "创建或成立时间": "1973年"}
recomputed_repair_strict_rate 31 / 40 77.5%

~~~

## 2026-05-21 21:08:10 - Run structured vLLM client with enhanced repair limit 40

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --eval-file eval\prompts_instructie.json --limit 40 --output results\vllm_benchmark_schema_strict\repaired_outputs_20260521_enhanced.jsonl
~~~

~~~text
[1/40] ext_001 parse=True repair_strict=False missing=['出生日期']
[2/40] ext_002 parse=True repair_strict=True missing=[]
[3/40] ext_003 parse=True repair_strict=True missing=[]
[4/40] ext_004 parse=True repair_strict=True missing=[]
[5/40] ext_005 parse=True repair_strict=False missing=['位于']
[6/40] ext_006 parse=True repair_strict=True missing=[]
[7/40] ext_007 parse=True repair_strict=True missing=[]
[8/40] ext_008 parse=True repair_strict=True missing=[]
[9/40] ext_009 parse=True repair_strict=True missing=[]
[10/40] ext_010 parse=True repair_strict=True missing=[]
[11/40] ext_011 parse=True repair_strict=True missing=[]
[12/40] ext_012 parse=True repair_strict=True missing=[]
[13/40] ext_013 parse=True repair_strict=True missing=[]
[14/40] ext_014 parse=True repair_strict=True missing=[]
[15/40] ext_015 parse=True repair_strict=True missing=[]
[16/40] ext_016 parse=True repair_strict=True missing=[]
[17/40] ext_017 parse=True repair_strict=True missing=[]
[18/40] ext_018 parse=True repair_strict=True missing=[]
[19/40] sc_001 parse=True repair_strict=False missing=['国籍']
[20/40] sc_002 parse=True repair_strict=True missing=[]
[21/40] sc_003 parse=True repair_strict=True missing=[]
[22/40] sc_004 parse=True repair_strict=True missing=[]
[23/40] sc_005 parse=True repair_strict=True missing=[]
[24/40] sc_006 parse=True repair_strict=True missing=[]
[25/40] sc_007 parse=True repair_strict=True missing=[]
[26/40] sc_008 parse=True repair_strict=True missing=[]
[27/40] sc_009 parse=True repair_strict=True missing=[]
[28/40] sc_010 parse=True repair_strict=False missing=['直径']
[29/40] sc_011 parse=True repair_strict=True missing=[]
[30/40] sc_012 parse=True repair_strict=False missing=['位于']
[31/40] ff_001 parse=True repair_strict=False missing=['出生日期']
[32/40] ff_002 parse=True repair_strict=False missing=['面积']
[33/40] ff_003 parse=True repair_strict=True missing=[]
[34/40] ff_004 parse=True repair_strict=True missing=[]
[35/40] ff_005 parse=True repair_strict=False missing=['起因']
[36/40] ff_006 parse=True repair_strict=False missing=['建筑师']
[37/40] ff_007 parse=True repair_strict=True missing=[]
[38/40] ff_008 parse=True repair_strict=True missing=[]
[39/40] ff_009 parse=True repair_strict=True missing=[]
[40/40] ff_010 parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=40
parse_rate=100.0%
repair_strict_rate=77.5%
avg_latency_s=2.432
saved=results\vllm_benchmark_schema_strict\repaired_outputs_20260521_enhanced.jsonl

~~~

## 2026-05-21 21:14:57 - Full schema strict stability check after enhanced repair

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\check_structured_stability.py --base-url http://localhost:8000 --rounds 3 --limit 40 --output-dir results\vllm_benchmark_schema_strict
~~~

~~~text
============================================================
  Structured Output Stability Check — vLLM Deployed Model
  Target:      http://localhost:8000
  Eval file:   prompts_instructie.json
  Prompts:     40
  Rounds:      3
  Time:        2026-05-21 21:09:55
============================================================

Server health: OK
  Model: /mnt/e/MicroLM/outputs/qwen_lora_merged_final

============================================================
Round: Round 1: Normal Chat Completion
  Mode: normal chat completion
  Prompts: 40
============================================================
  [1/40] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/40] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/40] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/40] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/40] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...
  [6/40] extr ext_006: parse=Y strict=N alias=N proj=N/N | {"2008年北京夏季奥林匹克运动会": {"主办城市": "中国北京", "开始时间": "2008年8月8日", "结束时间": "2008年8月24日"}...
  [7/40] extr ext_007: parse=Y strict=N alias=N proj=N/Y | {"糖尿病": {"主要症状": ["多饮", "多尿", "多食和体重下降"], "并发症": ["心血管疾病", "视网膜病变"]}}...
  [8/40] extr ext_008: parse=Y strict=N alias=N proj=N/Y | {"上海虹桥站": {"所属单位": "中国铁路上海局集团有限公司", "启用时间": "2010年7月1日", "位置": "上海市闵行区"}, "闵行区":...
  [9/40] extr ext_009: parse=Y strict=N alias=N proj=Y/Y | {"宝剑": {"材料": "铁或钢", "产地": ["湖北省江陵县", "中国"], "长度": "55.7厘米"}, "越王勾践剑": {"产地": "湖...
  [10/40] extr ext_010: parse=Y strict=N alias=N proj=N/N | {"三体": {"作者": "刘慈欣", "出版社": "重庆出版社", "获奖": ["雨果奖最佳长篇小说奖", "最佳长篇小说奖"]}, "刘慈欣": {"...
  [11/40] extr ext_011: parse=Y strict=N alias=N proj=N/N | {"大熊猫": {"别称": "熊猫", "分类": ["国家一级保护动物", "珍稀动物"], "拉丁名": "Ailuropoda melanoleuca"...
  [12/40] extr ext_012: parse=Y strict=N alias=N proj=N/N | {"木星": {"直径": "142984千米", "距离太阳": "7.78亿千米"}, "太阳系": {"最大行星": "木星"}}...
  [13/40] extr ext_013: parse=Y strict=N alias=N proj=N/N | {"阿尔伯特·爱因斯坦": {"出生日期": "1879年3月14日", "国籍": "德国", "配偶": "米列娃·马里奇", "成就": ["诺贝尔物理学...
  [14/40] extr ext_014: parse=Y strict=N alias=N proj=N/N | {"第二次世界大战": {"参与方": ["同盟国", "轴心国"], "开始时间": "1939年9月1日", "结束时间": "1945年9月2日", "伤...
  [15/40] extr ext_015: parse=Y strict=N alias=N proj=N/N | {"联合国": {"主要机构": ["安全理事会", "联合国大会", "国际法院"], "会员国": "193个"}, "联合国大会": {"主要机构": "...
  [16/40] extr ext_016: parse=Y strict=N alias=N proj=Y/Y | {"哈利·波特": {"作者": "J.K.罗琳", "出版社": "英国布鲁姆斯伯里出版社"}, "哈利·波特与魔法石": {"作者": "J.K.罗琳"}}...
  [17/40] extr ext_017: parse=Y strict=N alias=N proj=N/N | {"黑火药": {"别名": ["火药", "有烟火药"], "用途": ["制造火炮和烟花", "军事和民用"]}, "硝石": {"用途": "制造火炮和烟...
  [18/40] extr ext_018: parse=Y strict=N alias=N proj=Y/Y | {"长江": {"长度": "约6300千米", "起点": "青藏高原唐古拉山脉", "终点": "东海"}}...
  [19/40] sche sc_001: parse=Y strict=N alias=N proj=N/N | {"姚明": {"出生日期": "1980年9月12日", "国籍": "中国", "职业": "前中国职业篮球运动员"}, "姚明": {"出生地": "上海...
  [20/40] sche sc_002: parse=Y strict=N alias=N proj=N/N | {"腾讯公司": {"创始人": "马化腾", "成立时间": "1998年"}}...
  [21/40] sche sc_003: parse=Y strict=N alias=N proj=N/N | {"巴黎": {"文化": "巴黎圣母院"}, "巴黎圣母院": {"宗教": "天主教", "建筑材料": "砖石", "建造时间": "1163年", "类...
  [22/40] sche sc_004: parse=Y strict=N alias=N proj=N/N | {"肺炎": {"症状": ["发热", "呼吸困难", "咳嗽"], "治疗方法": "抗生素"}}...
  [23/40] sche sc_005: parse=Y strict=N alias=N proj=N/N | {"1969年7月20日": {"事件": "阿波罗11号任务成功将宇航员送上月球", "人物": ["尼尔·阿姆斯特朗", "阿波罗11号任务"]}, "尼尔...
  [24/40] sche sc_006: parse=Y strict=N alias=N proj=N/Y | {"X射线": {"发明者": "威廉·伦琴", "别称": "伦琴射线"}}...
  [25/40] sche sc_007: parse=Y strict=N alias=N proj=N/N | {"北京南站": {"车站等级": "特等站", "所属公司": "中国铁路北京局集团有限公司"}, "丰台区": {"所属城市": "北京市"}}...
  [26/40] sche sc_008: parse=Y strict=N alias=N proj=N/N | {"华南虎": {"保护级别": "国家一级保护动物"}}...
  [27/40] sche sc_009: parse=Y strict=N alias=N proj=Y/Y | {"百年孤独": {"作者": "加西亚·马尔克斯", "奖项": ["诺贝尔文学奖", "魔幻现实主义文学"], "出版时间": "1967年"}}...
  [28/40] sche sc_010: parse=Y strict=N alias=N proj=N/N | {"火星": {"直径": "约6779千米", "表面平均温度": "-63℃"}}...
  [29/40] sche sc_011: parse=Y strict=N alias=N proj=N/N | {"司南": {"材料": "天然磁石"}}...
  [30/40] sche sc_012: parse=Y strict=N alias=N proj=N/Y | {"死海": {"海拔": "-430米", "面积": "810平方公里", "位置": ["约旦", "巴勒斯坦", "以色列"]}}...
  [31/40] form ff_001: parse=Y strict=N alias=N proj=N/N | {"李白": {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}}...
  [32/40] form ff_002: parse=Y strict=N alias=N proj=Y/Y | {"江西省": {"位于": "中国"}, "鄱阳湖": {"面积": "约3150平方公里"}}...
  [33/40] form ff_003: parse=Y strict=N alias=N proj=N/N | {"华为技术有限公司": {"成立时间": "1987年", "位于": "广东省深圳市"}, "深圳市": {"位于": "广东省"}}...
  [34/40] form ff_004: parse=Y strict=N alias=N proj=Y/Y | {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}}...
  [35/40] form ff_005: parse=Y strict=N alias=N proj=N/N | {"2008年汶川地震": {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}}...
  [36/40] form ff_006: parse=Y strict=N alias=N proj=N/N | {"悉尼歌剧院": {"位于": "澳大利亚悉尼港口", "创建或成立时间": "1973年"}, "悉尼港口": {"位于": "澳大利亚"}}...
  [37/40] form ff_007: parse=Y strict=N alias=N proj=N/N | {"北京市": {"位于": "中国"}, "北京地铁1号线": {"开通时间": "1969年10月1日", "位于": "北京市"}}...
  [38/40] form ff_008: parse=Y strict=N alias=N proj=N/N | {"高血压": {"常见症状": ["头痛", "头晕", "心悸"], "并发症": ["冠心病", "脑卒中", "肾功能不全"]}}...
  [39/40] form ff_009: parse=Y strict=N alias=N proj=Y/Y | {"土星": {"属于": "太阳系", "直径": "116464千米"}}...
  [40/40] form ff_010: parse=Y strict=N alias=N proj=Y/Y | {"红楼梦": {"作者": "曹雪芹", "成就": "中国古典小说的巅峰之作", "出版时间": "未明确"}}...

  --- Round 1: Normal Chat Completion Summary ---
  Parse%:       100.0% (40/40)
  Strict%:      0.0% (0/40)
  Alias-Strict%:0.0% (0/40)
  Proj-Strict%: 20.0% (8/40)
  Proj-Alias%:  37.5% (15/40)
  Errors:       0
  Total time:   99.93s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=16.7%/44.4%
    schema_constraint      P=100.0% S=0.0% A=0.0% Proj=8.3%/25.0%
    format_following       P=100.0% S=0.0% A=0.0% Proj=40.0%/40.0%

============================================================
Round: Round 2: Constrained (response_format=json_object)
  Mode: constrained (response_format=json_object)
  Prompts: 40
============================================================
  [1/40] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"出生地": "浙江省绍兴府会稽县", "别名": "周树人", "职业": "中国现代文学的奠基人"}, "中国": {"代表人物": ["鲁...
  [2/40] extr ext_002: parse=Y strict=N alias=N proj=N/Y | {"西湖": {"面积": "约6.39平方公里", "位置": ["浙江省", "杭州市西湖区"], "所属省份": "浙江省"}, "杭州市西湖区": {"...
  [3/40] extr ext_003: parse=Y strict=N alias=N proj=N/Y | {"阿里巴巴集团": {"创始人": "马云", "成立时间": "1999年"}, "淘宝网": {"所属公司": "阿里巴巴集团"}}...
  [4/40] extr ext_004: parse=Y strict=N alias=N proj=N/Y | {"青霉素": {"别名": "盘尼西林", "发明者": "亚历山大·弗莱明"}}...
  [5/40] extr ext_005: parse=Y strict=N alias=N proj=N/N | {"埃菲尔铁塔": {"高度": "324米", "建造时间": "1889年", "地点": "法国巴黎战神广场"}, "巴黎": {"景点": ["埃菲尔铁...
  [6/40] extr ext_006: parse=Y strict=N alias=N proj=N/N | {"2008年北京夏季奥林匹克运动会": {"主办城市": "中国北京", "开始时间": "2008年8月8日", "结束时间": "2008年8月24日"}...
  [7/40] extr ext_007: parse=Y strict=N alias=N proj=N/Y | {"糖尿病": {"主要症状": ["多饮", "多尿", "多食和体重下降"], "并发症": ["心血管疾病", "视网膜病变"]}}...
  [8/40] extr ext_008: parse=Y strict=N alias=N proj=N/Y | {"上海虹桥站": {"所属单位": "中国铁路上海局集团有限公司", "启用时间": "2010年7月1日", "位置": "上海市闵行区"}, "闵行区":...
  [9/40] extr ext_009: parse=Y strict=N alias=N proj=Y/Y | {"宝剑": {"材料": "铁或钢", "产地": ["湖北省江陵县", "中国"], "长度": "55.7厘米"}, "越王勾践剑": {"产地": "湖...
  [10/40] extr ext_010: parse=Y strict=N alias=N proj=N/N | {"三体": {"作者": "刘慈欣", "出版社": "重庆出版社", "获奖": ["雨果奖最佳长篇小说奖", "最佳长篇小说奖"]}, "刘慈欣": {"...
  [11/40] extr ext_011: parse=Y strict=N alias=N proj=N/N | {"大熊猫": {"别称": "熊猫", "分类": ["国家一级保护动物", "珍稀动物"], "拉丁名": "Ailuropoda melanoleuca"...
  [12/40] extr ext_012: parse=Y strict=N alias=N proj=N/N | {"木星": {"直径": "142984千米", "距离太阳": "7.78亿千米"}, "太阳系": {"最大行星": "木星"}}...
  [13/40] extr ext_013: parse=Y strict=N alias=N proj=N/N | {"阿尔伯特·爱因斯坦": {"出生日期": "1879年3月14日", "国籍": "德国", "配偶": "米列娃·马里奇", "成就": ["诺贝尔物理学...
  [14/40] extr ext_014: parse=Y strict=N alias=N proj=N/N | {"第二次世界大战": {"参与方": ["同盟国", "轴心国"], "开始时间": "1939年9月1日", "结束时间": "1945年9月2日", "伤...
  [15/40] extr ext_015: parse=Y strict=N alias=N proj=N/N | {"联合国": {"主要机构": ["安全理事会", "联合国大会", "国际法院"], "会员国": "193个"}, "联合国大会": {"主要机构": "...
  [16/40] extr ext_016: parse=Y strict=N alias=N proj=Y/Y | {"哈利·波特": {"作者": "J.K.罗琳", "出版社": "英国布鲁姆斯伯里出版社"}, "哈利·波特与魔法石": {"作者": "J.K.罗琳"}}...
  [17/40] extr ext_017: parse=Y strict=N alias=N proj=N/N | {"黑火药": {"别名": ["火药", "有烟火药"], "用途": ["制造火炮和烟花", "军事和民用"]}, "硝石": {"用途": "制造火炮和烟...
  [18/40] extr ext_018: parse=Y strict=N alias=N proj=Y/Y | {"长江": {"长度": "约6300千米", "起点": "青藏高原唐古拉山脉", "终点": "东海"}}...
  [19/40] sche sc_001: parse=Y strict=N alias=N proj=N/N | {"姚明": {"出生日期": "1980年9月12日", "国籍": "中国", "职业": "前中国职业篮球运动员"}, "姚明": {"出生地": "上海...
  [20/40] sche sc_002: parse=Y strict=N alias=N proj=N/N | {"腾讯公司": {"创始人": "马化腾", "成立时间": "1998年"}}...
  [21/40] sche sc_003: parse=Y strict=N alias=N proj=N/N | {"巴黎": {"文化": "巴黎圣母院"}, "巴黎圣母院": {"宗教": "天主教", "建筑材料": "砖石", "建造时间": "1163年", "类...
  [22/40] sche sc_004: parse=Y strict=N alias=N proj=N/N | {"肺炎": {"症状": ["发热", "呼吸困难", "咳嗽"], "治疗方法": "抗生素"}}...
  [23/40] sche sc_005: parse=Y strict=N alias=N proj=N/N | {"1969年7月20日": {"事件": "阿波罗11号任务成功将宇航员送上月球", "人物": ["尼尔·阿姆斯特朗", "阿波罗11号任务"]}, "尼尔...
  [24/40] sche sc_006: parse=Y strict=N alias=N proj=N/Y | {"X射线": {"发明者": "威廉·伦琴", "别称": "伦琴射线"}}...
  [25/40] sche sc_007: parse=Y strict=N alias=N proj=N/N | {"北京南站": {"车站等级": "特等站", "所属公司": "中国铁路北京局集团有限公司"}, "丰台区": {"所属城市": "北京市"}}...
  [26/40] sche sc_008: parse=Y strict=N alias=N proj=N/N | {"华南虎": {"保护级别": "国家一级保护动物"}}...
  [27/40] sche sc_009: parse=Y strict=N alias=N proj=Y/Y | {"百年孤独": {"作者": "加西亚·马尔克斯", "奖项": ["诺贝尔文学奖", "魔幻现实主义文学"], "出版时间": "1967年"}}...
  [28/40] sche sc_010: parse=Y strict=N alias=N proj=N/N | {"火星": {"直径": "约6779千米", "表面平均温度": "-63℃"}}...
  [29/40] sche sc_011: parse=Y strict=N alias=N proj=N/N | {"司南": {"材料": "天然磁石"}}...
  [30/40] sche sc_012: parse=Y strict=N alias=N proj=N/Y | {"死海": {"海拔": "-430米", "面积": "810平方公里", "位置": ["约旦", "巴勒斯坦", "以色列"]}}...
  [31/40] form ff_001: parse=Y strict=N alias=N proj=N/N | {"李白": {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}}...
  [32/40] form ff_002: parse=Y strict=N alias=N proj=Y/Y | {"江西省": {"位于": "中国"}, "鄱阳湖": {"面积": "约3150平方公里"}}...
  [33/40] form ff_003: parse=Y strict=N alias=N proj=N/N | {"华为技术有限公司": {"成立时间": "1987年", "位于": "广东省深圳市"}, "深圳市": {"位于": "广东省"}}...
  [34/40] form ff_004: parse=Y strict=N alias=N proj=Y/Y | {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}}...
  [35/40] form ff_005: parse=Y strict=N alias=N proj=N/N | {"2008年汶川地震": {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}}...
  [36/40] form ff_006: parse=Y strict=N alias=N proj=N/N | {"悉尼歌剧院": {"位于": "澳大利亚悉尼港口", "创建或成立时间": "1973年"}, "悉尼港口": {"位于": "澳大利亚"}}...
  [37/40] form ff_007: parse=Y strict=N alias=N proj=N/N | {"北京市": {"位于": "中国"}, "北京地铁1号线": {"开通时间": "1969年10月1日", "位于": "北京市"}}...
  [38/40] form ff_008: parse=Y strict=N alias=N proj=N/N | {"高血压": {"常见症状": ["头痛", "头晕", "心悸"], "并发症": ["冠心病", "脑卒中", "肾功能不全"]}}...
  [39/40] form ff_009: parse=Y strict=N alias=N proj=Y/Y | {"土星": {"属于": "太阳系", "直径": "116464千米"}}...
  [40/40] form ff_010: parse=Y strict=N alias=N proj=Y/Y | {"红楼梦": {"作者": "曹雪芹", "成就": "中国古典小说的巅峰之作", "出版时间": "未明确"}}...

  --- Round 2: Constrained (response_format=json_object) Summary ---
  Parse%:       100.0% (40/40)
  Strict%:      0.0% (0/40)
  Alias-Strict%:0.0% (0/40)
  Proj-Strict%: 20.0% (8/40)
  Proj-Alias%:  37.5% (15/40)
  Errors:       0
  Total time:   99.71s
  By group:
    extraction             P=100.0% S=0.0% A=0.0% Proj=16.7%/44.4%
    schema_constraint      P=100.0% S=0.0% A=0.0% Proj=8.3%/25.0%
    format_following       P=100.0% S=0.0% A=0.0% Proj=40.0%/40.0%

============================================================
Round: Round 3: Schema-Strict Constrained
  Mode: schema-strict constrained
  Prompts: 40
============================================================
  [1/40] extr ext_001: parse=Y strict=N alias=N proj=N/N | {"鲁迅": {"别名": "周树人", "出生地": "浙江省绍兴府会稽县", "职业": "作家"}}...
  [2/40] extr ext_002: parse=Y strict=N alias=N proj=Y/Y | {"西湖": {"面积": "约6.39平方公里", "位于": "杭州市西湖区"}, "杭州市西湖区": {"位于": "浙江省"}}...
  [3/40] extr ext_003: parse=Y strict=N alias=N proj=Y/Y | {"阿里巴巴集团": {"成立时间": "1999年", "创办者": "马云"}, "淘宝网": {"所属": "阿里巴巴集团"}, "天猫": {"所属":...
  [4/40] extr ext_004: parse=Y strict=N alias=N proj=Y/Y | {"青霉素": {"别名": "盘尼西林", "发现者或发明者": "亚历山大·弗莱明"}}...
  [5/40] extr ext_005: parse=Y strict=Y alias=Y proj=N/N | {"位于": {"建筑师": "古斯塔夫·埃菲尔", "别名": "铁塔"}}...
  [6/40] extr ext_006: parse=Y strict=Y alias=Y proj=Y/Y | {"发生地点": "中国北京", "参与者": ["中国", "204个国家和地区"], "发生时间": "2008年8月8日至8月24日"}...
  [7/40] extr ext_007: parse=Y strict=Y alias=Y proj=Y/Y | {"症状": ["多饮", "多尿", "多食和体重下降"], "治疗方法": ["胰岛素注射和口服降糖药"], "病因": "高血糖"}...
  [8/40] extr ext_008: parse=Y strict=N alias=N proj=Y/Y | {"位于": {"城市": "上海市", "区域": "闵行区"}, "开通时间": "2010年7月1日"}...
  [9/40] extr ext_009: parse=Y strict=Y alias=Y proj=Y/Y | {"材料": ["铁", "钢"], "长度": "55.7厘米", "宽度": "4.6厘米"}...
  [10/40] extr ext_010: parse=Y strict=Y alias=Y proj=Y/Y | {"作者": "刘慈欣", "出版时间": "2008年1月", "产地": "中国", "成就": "雨果奖最佳长篇小说奖"}...
  [11/40] extr ext_011: parse=Y strict=N alias=N proj=Y/Y | {"分布": {"所属省份": ["四川", "甘肃", "陕西"], "产地": "中国"}, "学名": "Ailuropoda melanoleuca"}...
  [12/40] extr ext_012: parse=Y strict=Y alias=Y proj=Y/Y | {"属于": "太阳系", "直径": "142984千米"}...
  [13/40] extr ext_013: parse=Y strict=N alias=N proj=Y/Y | {"爱因斯坦": {"成就": ["诺贝尔物理学奖", "狭义相对论", "广义相对论"], "出生日期": "1879年3月14日", "职业": "理论物理...
  [14/40] extr ext_014: parse=Y strict=N alias=N proj=Y/Y | {"第二次世界大战": {"起因": "纳粹德国入侵波兰", "参与者": ["同盟国", "轴心国"], "发生时间": "1939年9月1日至1945年9月...
  [15/40] extr ext_015: parse=Y strict=N alias=N proj=Y/Y | {"联合国": {"成立时间": "1945年10月24日", "位于": "美国纽约", "成员": "联合国大会"}}...
  [16/40] extr ext_016: parse=Y strict=Y alias=Y proj=Y/Y | {"作者": "J.K.罗琳", "出版时间": "1997年", "出版商": "英国布鲁姆斯伯里出版社", "成就": "全球票房超过77亿美元", "产地...
  [17/40] extr ext_017: parse=Y strict=Y alias=Y proj=Y/Y | {"组成": ["硝石", "硫磺", "木炭"], "用途": ["军事和民用", "制造火炮和烟花"], "别名": ["火药", "有烟火药"], "发现...
  [18/40] extr ext_018: parse=Y strict=N alias=N proj=Y/Y | {"长江": {"长度": "6300千米", "位于": "中国"}}...
  [19/40] sche sc_001: parse=Y strict=N alias=N proj=N/N | {"出生地": "上海市", "出生日期": "1980年9月12日"}...
  [20/40] sche sc_002: parse=Y strict=Y alias=Y proj=Y/Y | {"成立时间": "1998年", "创办者": "马化腾", "位于": "广东省深圳市南山区"}...
  [21/40] sche sc_003: parse=Y strict=Y alias=Y proj=Y/Y | {"用途": "哥特式天主教大教堂", "用途": "大教堂", "位于": "巴黎西岱岛"}...
  [22/40] sche sc_004: parse=Y strict=Y alias=Y proj=Y/Y | {"症状": "咳嗽、发热、呼吸困难", "所属科室": "呼吸系统", "治疗方法": "抗生素"}...
  [23/40] sche sc_005: parse=Y strict=Y alias=Y proj=Y/Y | {"发生时间": "1969年7月20日", "参与者": ["尼尔·阿姆斯特朗", "阿波罗11号"], "发生地点": "月球"}...
  [24/40] sche sc_006: parse=Y strict=Y alias=Y proj=Y/Y | {"发现者或发明者": "威廉·伦琴", "用途": ["医学影像诊断", "工业无损检测", "安全检查"], "别名": "伦琴射线"}...
  [25/40] sche sc_007: parse=Y strict=N alias=N proj=Y/Y | {"位于": {"城市": "北京市丰台区"}, "车站等级": "特等站"}...
  [26/40] sche sc_008: parse=Y strict=Y alias=Y proj=Y/Y | {"保护级别": ["国家一级保护动物", "国家二级保护动物"], "用途": "中国特有的虎亚种", "学名": "Panthera tigris amoy...
  [27/40] sche sc_009: parse=Y strict=Y alias=Y proj=Y/Y | {"作者": "加西亚·马尔克斯", "出版时间": "1967年", "成就": "诺贝尔文学奖"}...
  [28/40] sche sc_010: parse=Y strict=N alias=N proj=N/N | {"属于": "太阳系"}...
  [29/40] sche sc_011: parse=Y strict=Y alias=Y proj=Y/Y | {"材料": "天然磁石", "用途": "辨别方向"}...
  [30/40] sche sc_012: parse=Y strict=Y alias=Y proj=N/N | {"位于": {"面积": "810平方公里", "海拔": "-430米"}, "面积": "810平方公里"}...
  [31/40] form ff_001: parse=Y strict=N alias=N proj=N/N | {"李白": {"职业": "唐代著名的浪漫主义诗人", "出生地": "碎叶城"}}...
  [32/40] form ff_002: parse=Y strict=N alias=N proj=N/N | {"鄱阳湖": {"位于": "江西省"}, "江西省": {"位于": "中国"}}...
  [33/40] form ff_003: parse=Y strict=Y alias=Y proj=Y/Y | {"创办者": "任正非", "成立时间": "1987年", "位于": "广东省深圳市"}...
  [34/40] form ff_004: parse=Y strict=N alias=N proj=N/N | {"DNA双螺旋结构": {"发现者或发明者": "沃森和克里克"}, "DNA双螺旋结构": {"用途": "为现代分子生物学奠定了基础"}}...
  [35/40] form ff_005: parse=Y strict=N alias=N proj=N/N | {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}...
  [36/40] form ff_006: parse=Y strict=N alias=N proj=N/N | {"悉尼歌剧院": {"创建或成立时间": "1973年", "位于": "澳大利亚悉尼港口"}, "约恩·乌松": {"位于": "丹麦"}}...
  [37/40] form ff_007: parse=Y strict=Y alias=Y proj=Y/Y | {"开通时间": "1969年10月1日", "线路": "北京地铁1号线"}...
  [38/40] form ff_008: parse=Y strict=Y alias=Y proj=Y/Y | {"症状": ["头痛", "头晕", "心悸"], "治疗方法": ["降压药物治疗", "生活方式调整"], "常见并发症": ["冠心病", "脑卒中",...
  [39/40] form ff_009: parse=Y strict=N alias=N proj=Y/Y | {"土星": {"属于": "太阳系", "直径": "116464千米"}, "太阳系": {"属于": "宇宙"}}...
  [40/40] form ff_010: parse=Y strict=Y alias=Y proj=Y/Y | {"成就": "中国古典小说的巅峰之作", "作者": "曹雪芹"}...

  --- Round 3: Schema-Strict Constrained Summary ---
  Parse%:       100.0% (40/40)
  Strict%:      52.5% (21/40)
  Alias-Strict%:52.5% (21/40)
  Proj-Strict%: 75.0% (30/40)
  Proj-Alias%:  75.0% (30/40)
  Errors:       0
  Total time:   97.44s
  By group:
    extraction             P=100.0% S=44.4% A=44.4% Proj=88.9%/88.9%
    schema_constraint      P=100.0% S=75.0% A=75.0% Proj=75.0%/75.0%
    format_following       P=100.0% S=40.0% A=40.0% Proj=50.0%/50.0%

======================================================================
STABILITY CHECK COMPARISON
======================================================================
Round                                           Parse%   Strict%   Alias-S%    Proj%   Proj-A%
----------------------------------------------------------------------------------------------
Round 1: Normal Chat Completion               100.0%    0.0%     0.0%   20.0%    37.5%
Round 2: Constrained (response_format=json_object) 100.0%    0.0%     0.0%   20.0%    37.5% [constrained]
Round 3: Schema-Strict Constrained            100.0%   52.5%    52.5%   75.0%    75.0% [strict]

--- Reference: 6C Offline Results (qwen_lora) ---
Config                                          Parse%   Strict%   Alias-S%
6C offline (run_instructie_eval.py)             97.5%     7.5%     15.0%

Results saved to results\vllm_benchmark_schema_strict\stability_20260521_211457.json
CSV saved to results\vllm_benchmark_schema_strict\stability_summary_20260521_211457.csv

~~~

## 2026-05-21 21:17:27 - Run single structured vLLM client example

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --instruction "从文本中抽取人物信息。" --schema-json "[\"出生地\", \"职业\"]" --required-json "[\"出生地\", \"职业\"]" --input-text "鲁迅，原名周树人，浙江绍兴人，中国现代作家。" --output results\vllm_benchmark_schema_strict\single_repaired_example.jsonl
~~~

~~~text

~~~

## 2026-05-21 21:17:38 - Inspect single structured client example output

~~~powershell
Test-Path results\vllm_benchmark_schema_strict\single_repaired_example.jsonl
if (Test-Path results\vllm_benchmark_schema_strict\single_repaired_example.jsonl) { Get-Content -Encoding utf8 results\vllm_benchmark_schema_strict\single_repaired_example.jsonl }
~~~

~~~text
False

~~~

## 2026-05-21 21:17:50 - Run single structured vLLM client example corrected quoting

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --instruction '从文本中抽取人物信息。' --schema-json '["出生地", "职业"]' --required-json '["出生地", "职业"]' --input-text '鲁迅，原名周树人，浙江绍兴人，中国现代作家。' --output results\vllm_benchmark_schema_strict\single_repaired_example.jsonl
~~~

~~~text

~~~

## 2026-05-21 21:18:03 - Structured vLLM client help check

~~~powershell
.venv\Scripts\python.exe scripts\structured_vllm_client.py --help
~~~

~~~text
usage: structured_vllm_client.py [-h] [--base-url BASE_URL] [--model MODEL]
                                 [--eval-file EVAL_FILE] [--limit LIMIT]
                                 [--output OUTPUT] [--temperature TEMPERATURE]
                                 [--max-tokens MAX_TOKENS]
                                 [--no-response-format]
                                 [--instruction INSTRUCTION]
                                 [--schema-json SCHEMA_JSON]
                                 [--required-json REQUIRED_JSON]
                                 [--input-text INPUT_TEXT]

Schema-strict vLLM structured client

options:
  -h, --help            show this help message and exit
  --base-url BASE_URL
  --model MODEL         Model id; defaults to /v1/models first entry
  --eval-file EVAL_FILE
  --limit LIMIT
  --output OUTPUT
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --no-response-format
  --instruction INSTRUCTION
  --schema-json SCHEMA_JSON
                        JSON list of allowed fields
  --required-json REQUIRED_JSON
                        Optional JSON list of required fields
  --input-text INPUT_TEXT

~~~

## 2026-05-21 21:18:18 - Run single structured vLLM client example with arg list

~~~powershell
$env:PYTHONIOENCODING='utf-8'
$argsList = @(
  'scripts\structured_vllm_client.py',
  '--base-url','http://localhost:8000',
  '--instruction','从文本中抽取人物信息。',
  '--schema-json','["出生地", "职业"]',
  '--required-json','["出生地", "职业"]',
  '--input-text','鲁迅，原名周树人，浙江绍兴人，中国现代作家。',
  '--output','results\vllm_benchmark_schema_strict\single_repaired_example.jsonl'
)
& .venv\Scripts\python.exe @argsList
~~~

~~~text

~~~

## 2026-05-21 21:18:29 - List schema strict result files after single example attempts

~~~powershell
Get-ChildItem results\vllm_benchmark_schema_strict | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,Length,LastWriteTime | Format-Table -AutoSize
~~~

~~~text

Name                                     Length LastWriteTime     
----                                     ------ -------------     
stability_summary_20260521_211457.csv       811 2026/5/21 21:14:57
stability_20260521_211457.json             3704 2026/5/21 21:14:57
repaired_outputs_20260521_enhanced.jsonl  28000 2026/5/21 21:08:10
repaired_outputs_20260521.jsonl           27839 2026/5/21 21:03:40
stability_summary_20260521_205135.csv       812 2026/5/21 20:51:35



~~~

## 2026-05-21 21:18:42 - Debug single structured client arg list execution

~~~powershell
$env:PYTHONIOENCODING='utf-8'
$argsList = @('scripts\structured_vllm_client.py','--base-url','http://localhost:8000','--instruction','从文本中抽取人物信息。','--schema-json','["出生地", "职业"]','--required-json','["出生地", "职业"]','--input-text','鲁迅，原名周树人，浙江绍兴人，中国现代作家。','--output','results\vllm_benchmark_schema_strict\single_repaired_example.jsonl')
Write-Output "argc=$($argsList.Count)"
& .venv\Scripts\python.exe @argsList
Write-Output "exit=$LASTEXITCODE"
~~~

~~~text
argc=13
exit=1

~~~

## 2026-05-21 21:18:55 - Capture single structured client stderr

~~~powershell
$env:PYTHONIOENCODING='utf-8'
$argsList = @('scripts\structured_vllm_client.py','--base-url','http://localhost:8000','--instruction','从文本中抽取人物信息。','--schema-json','["出生地", "职业"]','--required-json','["出生地", "职业"]','--input-text','鲁迅，原名周树人，浙江绍兴人，中国现代作家。','--output','results\vllm_benchmark_schema_strict\single_repaired_example.jsonl')
& .venv\Scripts\python.exe @argsList 2>&1
Write-Output "exit=$LASTEXITCODE"
~~~

~~~text
python.exe : Traceback (most recent call last):
所在位置 行:3 字符: 1
+ & .venv\Scripts\python.exe @argsList 2>&1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (Traceback (most recent call last)::String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
  File "E:\MicroLM\scripts\structured_vllm_client.py", line 216, in <module>
    main()
  File "E:\MicroLM\scripts\structured_vllm_client.py", line 177, in main
    prompts = iter_prompts(args)
              ^^^^^^^^^^^^^^^^^^
  File "E:\MicroLM\scripts\structured_vllm_client.py", line 127, in iter_prompts
    return [prompt_from_single_args(args)]
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\MicroLM\scripts\structured_vllm_client.py", line 106, in prompt_from_single_args
    schema = json.loads(args.schema_json)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\backs\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\json\__init__.py", line 346, in loads
    return _default_decoder.decode(s)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\backs\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\json\decoder.py", line 338, in decode
    obj, end = self.raw_decode(s, idx=_w(s, 0).end())
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\backs\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\json\decoder.py", line 356, in raw_decode
    raise JSONDecodeError("Expecting value", s, err.value) from None
json.decoder.JSONDecodeError: Expecting value: line 1 column 2 (char 1)
exit=1

~~~

## 2026-05-21 21:19:29 - Py compile structured client after schema-fields args

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\structured_vllm_client.py
~~~

~~~text

~~~

## 2026-05-21 21:19:46 - Run single structured client example with schema-fields

~~~powershell
$env:PYTHONIOENCODING='utf-8'
$argsList = @('scripts\structured_vllm_client.py','--base-url','http://localhost:8000','--instruction','从文本中抽取人物信息。','--schema-fields','出生地,职业','--required-fields','出生地,职业','--input-text','鲁迅，原名周树人，浙江绍兴人，中国现代作家。','--output','results\vllm_benchmark_schema_strict\single_repaired_example.jsonl')
& .venv\Scripts\python.exe @argsList 2>&1
Write-Output "exit=$LASTEXITCODE"
~~~

~~~text
[1/1] single_request parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=1
parse_rate=100.0%
repair_strict_rate=100.0%
avg_latency_s=2.245
saved=results\vllm_benchmark_schema_strict\single_repaired_example.jsonl
exit=0

~~~

## 2026-05-21 21:19:58 - Read single repaired example output

~~~powershell
Get-Content -Encoding utf8 results\vllm_benchmark_schema_strict\single_repaired_example.jsonl
~~~

~~~text
{"id": "single_request", "group": "single", "parse_ok": true, "schema_strict_after_repair": true, "missing_fields_after_repair": [], "extra_fields_after_repair": [], "enum_ok_after_repair": true, "raw_output": "{\"鲁迅\": {\"职业\": \"作家\", \"出生地\": \"浙江绍兴\"}}", "repaired": {"出生地": "浙江绍兴", "职业": "作家"}, "repaired_for_contract": {"出生地": "浙江绍兴", "职业": "作家"}, "latency_s": 2.245, "usage": {"prompt_tokens": 288, "total_tokens": 306, "completion_tokens": 18, "prompt_tokens_details": null}}

~~~

## 2026-05-21 21:20:33 - Final validate enhanced repaired outputs and stability summary

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
files=['results/vllm_benchmark_schema_strict/repaired_outputs_20260521_enhanced.jsonl','results/vllm_benchmark_schema_strict/single_repaired_example.jsonl']
for f in files:
 rows=[json.loads(x) for x in Path(f).read_text(encoding='utf-8').splitlines() if x.strip()]
 print(f, 'rows=', len(rows), 'parse=', sum(r['parse_ok'] for r in rows), 'repair_strict=', sum(r['schema_strict_after_repair'] for r in rows))"
Get-Content -Encoding utf8 results\vllm_benchmark_schema_strict\stability_summary_20260521_211457.csv
~~~

~~~text
results/vllm_benchmark_schema_strict/repaired_outputs_20260521_enhanced.jsonl rows= 40 parse= 40 repair_strict= 31
results/vllm_benchmark_schema_strict/single_repaired_example.jsonl rows= 1 parse= 1 repair_strict= 1
round,mode,total,parse_rate,strict_rate,alias_strict_rate,projected_strict_rate,projected_alias_strict_rate,extraction_P,extraction_S,extraction_A,extraction_projected,extraction_projected_alias,schema_P,schema_S,schema_A,schema_projected,schema_projected_alias,format_P,format_S,format_A,format_projected,format_projected_alias,avg_latency_s
Round 1: Normal Chat Completion,normal,40,1.0,0.0,0.0,0.2,0.375,1.0,0.0,0.0,0.1667,0.4444,1.0,0.0,0.0,0.0833,0.25,1.0,0.0,0.0,0.4,0.4,2.498
Round 2: Constrained (response_format=json_object),constrained,40,1.0,0.0,0.0,0.2,0.375,1.0,0.0,0.0,0.1667,0.4444,1.0,0.0,0.0,0.0833,0.25,1.0,0.0,0.0,0.4,0.4,2.493
Round 3: Schema-Strict Constrained,schema_strict,40,1.0,0.525,0.525,0.75,0.75,1.0,0.4444,0.4444,0.8889,0.8889,1.0,0.75,0.75,0.75,0.75,1.0,0.4,0.4,0.5,0.5,2.436

~~~

## 2026-05-21 21:20:45 - Final health check after enhanced repair work

~~~powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health -TimeoutSec 5
~~~

~~~text


StatusCode        : 200
StatusDescription : OK
Content           : {}
RawContent        : HTTP/1.1 200 OK
                    Content-Length: 0
                    Date: Thu, 21 May 2026 13:20:45 GMT
                    Server: uvicorn
                    
                    
Headers           : {[Content-Length, 0], [Date, Thu, 21 May 2026 13:20:45 GMT], [Server, uvicorn]}
RawContentLength  : 0




~~~

## 2026-05-21 21:20:56 - Git status for enhanced schema strict work

~~~powershell
git status --short microlm\structured scripts\check_structured_stability.py scripts\structured_vllm_client.py docs\vllm_deploy.md reports\schema_strict_improvement_report.md reports\eval_deploy_closure.md reports\terminal_outputs_schema_strict.md results\vllm_benchmark_schema_strict
~~~

~~~text
 M scripts/check_structured_stability.py
?? docs/vllm_deploy.md
?? microlm/structured/
?? reports/eval_deploy_closure.md
?? reports/schema_strict_improvement_report.md
?? reports/terminal_outputs_schema_strict.md
?? results/vllm_benchmark_schema_strict/
?? scripts/structured_vllm_client.py

~~~

## 2026-05-21 21:25:40 - Read structured_vllm_client.py before self-repair

~~~powershell
Get-Content -Encoding utf8 scripts\structured_vllm_client.py
~~~

~~~text
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

    return {
        "id": prompt.get("id"),
        "group": prompt.get("group"),
        "parse_ok": parse_ok,
        "schema_strict_after_repair": score["schema_strict"],
        "missing_fields_after_repair": score["missing_fields"],
        "extra_fields_after_repair": score["extra_fields"],
        "enum_ok_after_repair": score["enum_ok"],
        "raw_output": raw_output,
        "repaired": repaired,
        "repaired_for_contract": repaired_for_contract,
        "latency_s": meta["elapsed_s"],
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

~~~

## 2026-05-21 21:26:35 - Py compile structured client with self repair

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\structured_vllm_client.py
~~~

~~~text

~~~

## 2026-05-21 21:27:17 - Quick self-repair client limit 5

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --eval-file eval\prompts_instructie.json --limit 5 --self-repair --output results\vllm_benchmark_schema_strict\self_repair_quick_5.jsonl
~~~

~~~text
[1/5] ext_001 parse=True repair_strict=True missing=[]
[2/5] ext_002 parse=True repair_strict=True missing=[]
[3/5] ext_003 parse=True repair_strict=True missing=[]
[4/5] ext_004 parse=True repair_strict=True missing=[]
[5/5] ext_005 parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=5
parse_rate=100.0%
repair_strict_rate=100.0%
avg_latency_s=3.609
saved=results\vllm_benchmark_schema_strict\self_repair_quick_5.jsonl

~~~

## 2026-05-21 21:29:42 - Full self-repair client limit 40

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --eval-file eval\prompts_instructie.json --limit 40 --self-repair --output results\vllm_benchmark_schema_strict\self_repair_outputs_20260521.jsonl
~~~

~~~text
[1/40] ext_001 parse=True repair_strict=True missing=[]
[2/40] ext_002 parse=True repair_strict=True missing=[]
[3/40] ext_003 parse=True repair_strict=True missing=[]
[4/40] ext_004 parse=True repair_strict=True missing=[]
[5/40] ext_005 parse=True repair_strict=True missing=[]
[6/40] ext_006 parse=True repair_strict=True missing=[]
[7/40] ext_007 parse=True repair_strict=True missing=[]
[8/40] ext_008 parse=True repair_strict=True missing=[]
[9/40] ext_009 parse=True repair_strict=True missing=[]
[10/40] ext_010 parse=True repair_strict=True missing=[]
[11/40] ext_011 parse=True repair_strict=True missing=[]
[12/40] ext_012 parse=True repair_strict=True missing=[]
[13/40] ext_013 parse=True repair_strict=True missing=[]
[14/40] ext_014 parse=True repair_strict=True missing=[]
[15/40] ext_015 parse=True repair_strict=True missing=[]
[16/40] ext_016 parse=True repair_strict=True missing=[]
[17/40] ext_017 parse=True repair_strict=True missing=[]
[18/40] ext_018 parse=True repair_strict=True missing=[]
[19/40] sc_001 parse=True repair_strict=True missing=[]
[20/40] sc_002 parse=True repair_strict=True missing=[]
[21/40] sc_003 parse=True repair_strict=True missing=[]
[22/40] sc_004 parse=True repair_strict=True missing=[]
[23/40] sc_005 parse=True repair_strict=True missing=[]
[24/40] sc_006 parse=True repair_strict=True missing=[]
[25/40] sc_007 parse=True repair_strict=True missing=[]
[26/40] sc_008 parse=True repair_strict=True missing=[]
[27/40] sc_009 parse=True repair_strict=True missing=[]
[28/40] sc_010 parse=True repair_strict=True missing=[]
[29/40] sc_011 parse=True repair_strict=True missing=[]
[30/40] sc_012 parse=True repair_strict=True missing=[]
[31/40] ff_001 parse=True repair_strict=False missing=['出生日期']
[32/40] ff_002 parse=True repair_strict=True missing=[]
[33/40] ff_003 parse=True repair_strict=True missing=[]
[34/40] ff_004 parse=True repair_strict=True missing=[]
[35/40] ff_005 parse=True repair_strict=False missing=['起因']
[36/40] ff_006 parse=True repair_strict=True missing=[]
[37/40] ff_007 parse=True repair_strict=True missing=[]
[38/40] ff_008 parse=True repair_strict=True missing=[]
[39/40] ff_009 parse=True repair_strict=True missing=[]
[40/40] ff_010 parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=40
parse_rate=100.0%
repair_strict_rate=95.0%
avg_latency_s=3.167
saved=results\vllm_benchmark_schema_strict\self_repair_outputs_20260521.jsonl

~~~

## 2026-05-21 21:29:56 - Inspect remaining self-repair failures

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
rows=[json.loads(x) for x in Path('results/vllm_benchmark_schema_strict/self_repair_outputs_20260521.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
for r in rows:
    if not r['schema_strict_after_repair']:
        print('\n', r['id'], 'missing=', r['missing_fields_after_repair'])
        print('raw=', r['raw_output'][:400])
        print('self_repair=', r.get('self_repair_output'))
        print('repaired=', json.dumps(r['repaired'], ensure_ascii=False))"
~~~

~~~text

 ff_001 missing= ['出生日期']
raw= {"李白": {"职业": "唐代著名的浪漫主义诗人", "出生地": "碎叶城"}}
self_repair= {"李白": {"职业": "唐代著名的浪漫主义诗人", "出生地": "碎叶城"}}
repaired= {"出生地": "碎叶城", "职业": "唐代著名的浪漫主义诗人"}

 ff_005 missing= ['起因']
raw= {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}
self_repair= {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日"}
repaired= {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日"}

~~~

## 2026-05-21 21:30:08 - Inspect remaining failure prompt texts

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
p=json.loads(Path('eval/prompts_instructie.json').read_text(encoding='utf-8'))['prompts']
for id in ['ff_001','ff_005']:
 item=next(x for x in p if x['id']==id)
 print('\n', id)
 print('instruction=', item.get('instruction'))
 print('input=', item.get('input'))
 print('schema_def=', json.dumps(item.get('schema_def'), ensure_ascii=False))
 print('gold=', json.dumps(item.get('gold_output'), ensure_ascii=False))"
~~~

~~~text

 ff_001
instruction= 请从文本中抽取人物信息并输出 JSON。只输出 JSON，不要附加任何解释文字。
input= 李白，字太白，701年出生于碎叶城，是唐代著名的浪漫主义诗人，被后人誉为诗仙。
schema_def= {"required_fields": ["出生地", "出生日期", "职业"], "allowed_fields": ["出生地", "出生日期", "职业"], "types": {"出生地": "string", "出生日期": "string", "职业": "string"}}
gold= {"李白": {"出生地": "碎叶城", "出生日期": "701年", "职业": "诗人"}}

 ff_005
instruction= 从文本中抽取事件信息。输出格式：合法 JSON。禁止附加解释文字、注释或 markdown 格式标记。
input= 2008年汶川地震发生在四川省汶川县，发生于2008年5月12日，震级为里氏8.0级。地震由印度板块向亚洲板块俯冲造成。
schema_def= {"required_fields": ["发生地点", "发生时间", "起因"], "allowed_fields": ["发生地点", "发生时间", "起因"], "types": {"发生地点": "string", "发生时间": "string", "起因": "string"}}
gold= {"汶川地震": {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日", "起因": "印度板块向亚洲板块俯冲"}}

~~~

## 2026-05-21 21:30:38 - Py compile after stronger self-repair prompt

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\structured_vllm_client.py
~~~

~~~text

~~~

## 2026-05-21 21:33:03 - Full self-repair client v2 limit 40

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --eval-file eval\prompts_instructie.json --limit 40 --self-repair --output results\vllm_benchmark_schema_strict\self_repair_outputs_20260521_v2.jsonl
~~~

~~~text
[1/40] ext_001 parse=True repair_strict=True missing=[]
[2/40] ext_002 parse=True repair_strict=True missing=[]
[3/40] ext_003 parse=True repair_strict=True missing=[]
[4/40] ext_004 parse=True repair_strict=True missing=[]
[5/40] ext_005 parse=True repair_strict=True missing=[]
[6/40] ext_006 parse=True repair_strict=True missing=[]
[7/40] ext_007 parse=True repair_strict=True missing=[]
[8/40] ext_008 parse=True repair_strict=True missing=[]
[9/40] ext_009 parse=True repair_strict=True missing=[]
[10/40] ext_010 parse=True repair_strict=True missing=[]
[11/40] ext_011 parse=True repair_strict=True missing=[]
[12/40] ext_012 parse=True repair_strict=True missing=[]
[13/40] ext_013 parse=True repair_strict=True missing=[]
[14/40] ext_014 parse=True repair_strict=True missing=[]
[15/40] ext_015 parse=True repair_strict=True missing=[]
[16/40] ext_016 parse=True repair_strict=True missing=[]
[17/40] ext_017 parse=True repair_strict=True missing=[]
[18/40] ext_018 parse=True repair_strict=True missing=[]
[19/40] sc_001 parse=True repair_strict=True missing=[]
[20/40] sc_002 parse=True repair_strict=True missing=[]
[21/40] sc_003 parse=True repair_strict=True missing=[]
[22/40] sc_004 parse=True repair_strict=True missing=[]
[23/40] sc_005 parse=True repair_strict=True missing=[]
[24/40] sc_006 parse=True repair_strict=True missing=[]
[25/40] sc_007 parse=True repair_strict=True missing=[]
[26/40] sc_008 parse=True repair_strict=True missing=[]
[27/40] sc_009 parse=True repair_strict=True missing=[]
[28/40] sc_010 parse=True repair_strict=True missing=[]
[29/40] sc_011 parse=True repair_strict=True missing=[]
[30/40] sc_012 parse=True repair_strict=True missing=[]
[31/40] ff_001 parse=True repair_strict=True missing=[]
[32/40] ff_002 parse=True repair_strict=True missing=[]
[33/40] ff_003 parse=True repair_strict=True missing=[]
[34/40] ff_004 parse=True repair_strict=True missing=[]
[35/40] ff_005 parse=True repair_strict=False missing=['起因']
[36/40] ff_006 parse=True repair_strict=True missing=[]
[37/40] ff_007 parse=True repair_strict=True missing=[]
[38/40] ff_008 parse=True repair_strict=True missing=[]
[39/40] ff_009 parse=True repair_strict=True missing=[]
[40/40] ff_010 parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=40
parse_rate=100.0%
repair_strict_rate=97.5%
avg_latency_s=3.183
saved=results\vllm_benchmark_schema_strict\self_repair_outputs_20260521_v2.jsonl

~~~

## 2026-05-21 21:33:21 - Inspect remaining self-repair v2 failure

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
rows=[json.loads(x) for x in Path('results/vllm_benchmark_schema_strict/self_repair_outputs_20260521_v2.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
for r in rows:
    if not r['schema_strict_after_repair']:
        print(r['id'], r['missing_fields_after_repair']); print('raw=', r['raw_output']); print('self=', r['self_repair_output']); print('repaired=', json.dumps(r['repaired'], ensure_ascii=False))"
~~~

~~~text
ff_005 ['起因']
raw= {"发生时间": "2008年5月12日", "发生地点": "四川省汶川县"}
self= {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日"}
repaired= {"发生地点": "四川省汶川县", "发生时间": "2008年5月12日"}

~~~

## 2026-05-21 21:33:48 - Py compile after causal self-repair hint

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -m py_compile scripts\structured_vllm_client.py
~~~

~~~text

~~~

## 2026-05-21 21:36:11 - Full self-repair client v3 causal hint limit 40

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\structured_vllm_client.py --base-url http://localhost:8000 --eval-file eval\prompts_instructie.json --limit 40 --self-repair --output results\vllm_benchmark_schema_strict\self_repair_outputs_20260521_v3.jsonl
~~~

~~~text
[1/40] ext_001 parse=True repair_strict=True missing=[]
[2/40] ext_002 parse=True repair_strict=True missing=[]
[3/40] ext_003 parse=True repair_strict=True missing=[]
[4/40] ext_004 parse=True repair_strict=True missing=[]
[5/40] ext_005 parse=True repair_strict=True missing=[]
[6/40] ext_006 parse=True repair_strict=True missing=[]
[7/40] ext_007 parse=True repair_strict=True missing=[]
[8/40] ext_008 parse=True repair_strict=True missing=[]
[9/40] ext_009 parse=True repair_strict=True missing=[]
[10/40] ext_010 parse=True repair_strict=True missing=[]
[11/40] ext_011 parse=True repair_strict=True missing=[]
[12/40] ext_012 parse=True repair_strict=True missing=[]
[13/40] ext_013 parse=True repair_strict=True missing=[]
[14/40] ext_014 parse=True repair_strict=True missing=[]
[15/40] ext_015 parse=True repair_strict=True missing=[]
[16/40] ext_016 parse=True repair_strict=True missing=[]
[17/40] ext_017 parse=True repair_strict=True missing=[]
[18/40] ext_018 parse=True repair_strict=True missing=[]
[19/40] sc_001 parse=True repair_strict=True missing=[]
[20/40] sc_002 parse=True repair_strict=True missing=[]
[21/40] sc_003 parse=True repair_strict=True missing=[]
[22/40] sc_004 parse=True repair_strict=True missing=[]
[23/40] sc_005 parse=True repair_strict=True missing=[]
[24/40] sc_006 parse=True repair_strict=True missing=[]
[25/40] sc_007 parse=True repair_strict=True missing=[]
[26/40] sc_008 parse=True repair_strict=True missing=[]
[27/40] sc_009 parse=True repair_strict=True missing=[]
[28/40] sc_010 parse=True repair_strict=True missing=[]
[29/40] sc_011 parse=True repair_strict=True missing=[]
[30/40] sc_012 parse=True repair_strict=True missing=[]
[31/40] ff_001 parse=True repair_strict=True missing=[]
[32/40] ff_002 parse=True repair_strict=True missing=[]
[33/40] ff_003 parse=True repair_strict=True missing=[]
[34/40] ff_004 parse=True repair_strict=True missing=[]
[35/40] ff_005 parse=True repair_strict=True missing=[]
[36/40] ff_006 parse=True repair_strict=True missing=[]
[37/40] ff_007 parse=True repair_strict=True missing=[]
[38/40] ff_008 parse=True repair_strict=True missing=[]
[39/40] ff_009 parse=True repair_strict=True missing=[]
[40/40] ff_010 parse=True repair_strict=True missing=[]

SUMMARY
model=/mnt/e/MicroLM/outputs/qwen_lora_merged_final
total=40
parse_rate=100.0%
repair_strict_rate=100.0%
avg_latency_s=3.176
saved=results\vllm_benchmark_schema_strict\self_repair_outputs_20260521_v3.jsonl

~~~

## 2026-05-21 21:36:36 - Summarize final self-repair v3 jsonl

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe -c "import json; from pathlib import Path
rows=[json.loads(x) for x in Path('results/vllm_benchmark_schema_strict/self_repair_outputs_20260521_v3.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
print('rows', len(rows)); print('parse', sum(r['parse_ok'] for r in rows)); print('strict', sum(r['schema_strict_after_repair'] for r in rows)); print('avg_latency', round(sum(r['latency_s'] for r in rows)/len(rows),3)); print('self_repair_used', sum(r.get('self_repair_used', False) for r in rows))"
~~~

~~~text
rows 40
parse 40
strict 40
avg_latency 3.176
self_repair_used 9

~~~
