from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    instructie_dir = Path("data/instructie")
    qwen_dir = Path("Qwen2.5-1.5B-Instruct")

    instructie_dir.mkdir(parents=True, exist_ok=True)
    qwen_dir.mkdir(parents=True, exist_ok=True)

    for filename in ["train_zh.json", "valid_zh.json", "test_zh.json", "schema_zh.json"]:
        path = hf_hub_download(
            repo_id="zjunlp/InstructIE",
            repo_type="dataset",
            filename=filename,
            local_dir=str(instructie_dir),
        )
        print(f"InstructIE ready: {path}")

    snapshot_path = snapshot_download(
        repo_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_dir=str(qwen_dir),
    )
    print(f"Qwen snapshot ready: {snapshot_path}")


if __name__ == "__main__":
    main()
