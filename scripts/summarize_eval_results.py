#!/usr/bin/env python3
"""Summarize Qwen evaluation artifacts into a compact report."""

from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    prompt_by_model = read_json(Path("results/instructie_eval_qwen/summary/by_model.json"))
    prompt_by_group = read_json(Path("results/instructie_eval_qwen/summary/by_group.json"))
    valid_summary = read_json(Path("results/qwen_valid_eval_200/summary.json"))

    print("PROMPT_EVAL_BY_MODEL")
    print(json.dumps(prompt_by_model, ensure_ascii=False, indent=2))
    print("\nPROMPT_EVAL_BY_GROUP")
    print(json.dumps(prompt_by_group, ensure_ascii=False, indent=2))
    print("\nVALID_200_SUMMARY")
    print(json.dumps(valid_summary, ensure_ascii=False, indent=2))

    failure_path = Path("results/qwen_valid_eval_200/failure_samples.jsonl")
    failures = []
    if failure_path.exists():
        failures = [json.loads(line) for line in failure_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\nVALID_FAILURE_SAMPLES_SAVED={len(failures)}")
    for row in sorted(failures, key=lambda item: (item["field_f1"], item["pair_f1"]))[:8]:
        print("\n--- LOW_SCORE_SAMPLE ---")
        print(f"id={row['id']} task={row['task_type']} topic={row.get('topic_schema')} parsed={row['parsed']} direct={row['direct_json']}")
        print(f"field_f1={row['field_f1']} pair_f1={row['pair_f1']} exact={row['exact_match']}")
        print(f"raw={row['raw_output'][:500]}")
        print(f"gold={json.dumps(row['gold_output'], ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    main()
