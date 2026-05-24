#!/usr/bin/env python3
"""Build hardcase replay data for a short Qwen LoRA refinement run."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-path", type=Path, default=Path("results/qwen_valid_eval_200/failure_samples.jsonl"))
    parser.add_argument("--train-source", type=Path, default=Path("data/sft_candidate/train.jsonl"))
    parser.add_argument("--valid-source", type=Path, default=Path("data/sft_candidate/valid.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/qwen_refine"))
    parser.add_argument("--hardcase-repeat", type=int, default=30)
    parser.add_argument("--support-samples", type=int, default=990)
    parser.add_argument("--valid-samples", type=int, default=200)
    parser.add_argument("--holdout-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260520)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    failures = load_jsonl(args.failure_path)
    train_source = load_jsonl(args.train_source)
    valid_source = load_jsonl(args.valid_source)
    valid_by_id = {row["id"]: row for row in valid_source}

    hardcase_ids = [row["id"] for row in failures]
    missing = [row_id for row_id in hardcase_ids if row_id not in valid_by_id]
    if missing:
        raise ValueError(f"Failure ids missing from valid source: {missing[:5]}")

    hardcase_rows: list[dict[str, Any]] = []
    for failure in failures:
        source = valid_by_id[failure["id"]]
        for repeat_idx in range(args.hardcase_repeat):
            row = dict(source)
            row["id"] = f"{source['id']}__hardcase_repeat_{repeat_idx:02d}"
            row["refine_source_id"] = source["id"]
            row["refine_reason"] = {
                "field_f1": failure.get("field_f1"),
                "pair_f1": failure.get("pair_f1"),
                "raw_output": failure.get("raw_output"),
            }
            hardcase_rows.append(row)

    support_rows = rng.sample(train_source, min(args.support_samples, len(train_source)))
    support_rows = [dict(row, refine_source="support_train") for row in support_rows]

    train_rows = hardcase_rows + support_rows
    rng.shuffle(train_rows)

    non_hardcase_valid = [row for row in valid_source if row["id"] not in set(hardcase_ids)]
    rng.shuffle(non_hardcase_valid)
    needed = args.valid_samples + args.holdout_samples
    if len(non_hardcase_valid) < needed:
        raise ValueError(f"Not enough non-hardcase valid rows: need {needed}, got {len(non_hardcase_valid)}")
    refine_valid = non_hardcase_valid[: args.valid_samples]
    holdout = non_hardcase_valid[args.valid_samples : args.valid_samples + args.holdout_samples]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_rows, args.out_dir / "train.jsonl")
    save_jsonl(refine_valid, args.out_dir / "valid.jsonl")
    save_jsonl(holdout, args.out_dir / "eval_holdout.jsonl")
    save_jsonl([valid_by_id[row_id] for row_id in hardcase_ids], args.out_dir / "hardcases_once.jsonl")

    metadata = {
        "seed": args.seed,
        "failure_path": str(args.failure_path),
        "train_source": str(args.train_source),
        "valid_source": str(args.valid_source),
        "hardcase_unique": len(hardcase_ids),
        "hardcase_repeat": args.hardcase_repeat,
        "hardcase_rows": len(hardcase_rows),
        "support_samples": len(support_rows),
        "train_rows": len(train_rows),
        "valid_rows": len(refine_valid),
        "holdout_rows": len(holdout),
        "hardcase_by_task": dict(Counter(row.get("task_type", "unknown") for row in [valid_by_id[i] for i in hardcase_ids])),
        "train_by_task": dict(Counter(row.get("task_type", "unknown") for row in train_rows)),
        "valid_by_task": dict(Counter(row.get("task_type", "unknown") for row in refine_valid)),
        "holdout_by_task": dict(Counter(row.get("task_type", "unknown") for row in holdout)),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
