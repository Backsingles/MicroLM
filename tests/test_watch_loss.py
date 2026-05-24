from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_watch_loss_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "watch_loss.py"
    spec = importlib.util.spec_from_file_location("watch_loss", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_loss_points_accepts_common_log_shapes(tmp_path: Path) -> None:
    watch_loss = load_watch_loss_module()
    log_path = tmp_path / "train_log.jsonl"
    records = [
        {"step": 10, "train_loss": 2.5, "val_loss": 2.8, "lr": 1e-4},
        {"iter": 20, "train/loss": 2.1, "val/loss": 2.4, "train/lr": 8e-5},
        "not-json",
        {"step": 30, "message": "checkpoint only"},
    ]
    log_path.write_text(
        "\n".join(json.dumps(record) if isinstance(record, dict) else record for record in records),
        encoding="utf-8",
    )

    points = watch_loss.load_loss_points(log_path)

    assert points == [
        {"step": 10.0, "train_loss": 2.5, "val_loss": 2.8, "lr": 1e-4},
        {"step": 20.0, "train_loss": 2.1, "val_loss": 2.4, "lr": 8e-5},
    ]


def test_load_loss_points_returns_empty_for_missing_log(tmp_path: Path) -> None:
    watch_loss = load_watch_loss_module()

    assert watch_loss.load_loss_points(tmp_path / "missing.jsonl") == []
