#!/usr/bin/env python3
"""Summarize original vs refined Qwen hardcase/holdout comparison."""

from __future__ import annotations

import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row(name: str, summary: dict) -> dict:
    return {
        "name": name,
        "samples": summary["sample_count"],
        "parseable": summary["parseable_rate"],
        "direct_json": summary["direct_json_rate"],
        "exact": summary["exact_match_rate"],
        "field_f1": summary["avg_field_f1"],
        "pair_f1": summary["avg_pair_f1"],
        "latency": summary["avg_latency_sec"],
    }


def main() -> None:
    rows = [
        row("original_hardcases", load("results/qwen_refine_compare/original_hardcases/summary.json")),
        row("refined_hardcases", load("results/qwen_refine_compare/refined_hardcases/summary.json")),
        row("original_holdout", load("results/qwen_refine_compare/original_holdout/summary.json")),
        row("refined_holdout", load("results/qwen_refine_compare/refined_holdout/summary.json")),
    ]
    print(json.dumps(rows, indent=2, ensure_ascii=False))

    by_name = {r["name"]: r for r in rows}
    deltas = {
        "hardcases": {
            "exact_delta": by_name["refined_hardcases"]["exact"] - by_name["original_hardcases"]["exact"],
            "field_f1_delta": by_name["refined_hardcases"]["field_f1"] - by_name["original_hardcases"]["field_f1"],
            "pair_f1_delta": by_name["refined_hardcases"]["pair_f1"] - by_name["original_hardcases"]["pair_f1"],
        },
        "holdout": {
            "exact_delta": by_name["refined_holdout"]["exact"] - by_name["original_holdout"]["exact"],
            "field_f1_delta": by_name["refined_holdout"]["field_f1"] - by_name["original_holdout"]["field_f1"],
            "pair_f1_delta": by_name["refined_holdout"]["pair_f1"] - by_name["original_holdout"]["pair_f1"],
        },
    }
    print("\nDELTAS")
    print(json.dumps(deltas, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
