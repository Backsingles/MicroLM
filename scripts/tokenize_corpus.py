from __future__ import annotations

import argparse
import os
import json
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import islice
from pathlib import Path

import numpy as np

from microlm.tokenizer import BPETokenizer

_WORKER_TOKENIZER: BPETokenizer | None = None


def load_config_defaults(config_path: str | None) -> dict[str, object]:
    if config_path is None:
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer = config.get("tokenizer", {})
    data = config.get("data", {})
    output = config.get("output", {})

    defaults: dict[str, object] = {
        "vocab_path": tokenizer.get("vocab_path"),
        "merges_path": tokenizer.get("merges_path"),
        "special_tokens": tokenizer.get("special_tokens"),
        "train_path": data.get("train_path"),
        "valid_path": data.get("valid_path"),
        "output_dir": output.get("output_dir"),
        "read_chunk_bytes": output.get("read_chunk_bytes"),
        "token_batch_size": output.get("token_batch_size"),
        "num_workers": output.get("num_workers"),
    }
    return {key: value for key, value in defaults.items() if value is not None}


def build_parser(defaults: dict[str, object]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Encode text train/valid splits into uint16 token ID arrays."
    )
    parser.add_argument("--config", type=str, default=defaults.get("config"))
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=defaults.get("vocab_path", Path("output/tinystories_bpe_10k/vocab.json")),
    )
    parser.add_argument(
        "--merges-path",
        type=Path,
        default=defaults.get("merges_path", Path("output/tinystories_bpe_10k/merge.txt")),
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=defaults.get("train_path", Path("data/TinyStoriesV2-GPT4-train.txt")),
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=defaults.get("valid_path", Path("data/TinyStoriesV2-GPT4-valid.txt")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=defaults.get("output_dir", Path("output/tinystories_tokenized")),
    )
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        default=defaults.get("special_tokens"),
        help="Special token to reserve while loading the tokenizer. May be passed multiple times.",
    )
    parser.add_argument(
        "--read-chunk-bytes",
        type=int,
        default=defaults.get("read_chunk_bytes", 4 * 1024 * 1024),
        help="Number of UTF-8 text bytes to read from disk at a time.",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=defaults.get("token_batch_size", 1_000_000),
        help="Number of token IDs to materialize at once while counting/writing.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=defaults.get("num_workers", 1),
        help=(
            "Number of worker processes for newline/space-boundary sharded encoding. "
            "Use 1 to keep the legacy two-pass single-process path."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()

    defaults = load_config_defaults(config_args.config)
    defaults["config"] = config_args.config
    parser = build_parser(defaults)
    return parser.parse_args(remaining)


def iter_file_chunks(path: Path, chunk_size: int):
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return
            yield chunk


def iter_token_batches(
    tokenizer: BPETokenizer,
    path: Path,
    read_chunk_bytes: int,
    token_batch_size: int,
):
    token_iter = tokenizer.encode_iterable(iter_file_chunks(path, read_chunk_bytes))
    while True:
        batch = np.fromiter(islice(token_iter, token_batch_size), dtype=np.uint16)
        if batch.size == 0:
            return
        yield batch


def count_tokens(
    tokenizer: BPETokenizer,
    path: Path,
    read_chunk_bytes: int,
    token_batch_size: int,
) -> tuple[int, int]:
    total_tokens = 0
    max_token_id = -1
    for batch in iter_token_batches(tokenizer, path, read_chunk_bytes, token_batch_size):
        total_tokens += int(batch.size)
        batch_max = int(batch.max())
        if batch_max > max_token_id:
            max_token_id = batch_max
    return total_tokens, max_token_id


def write_tokens(
    tokenizer: BPETokenizer,
    path: Path,
    out_path: Path,
    total_tokens: int,
    read_chunk_bytes: int,
    token_batch_size: int,
) -> None:
    array = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )
    offset = 0
    for batch in iter_token_batches(tokenizer, path, read_chunk_bytes, token_batch_size):
        next_offset = offset + batch.size
        array[offset:next_offset] = batch
        offset = next_offset
    array.flush()


def iter_safe_text_batches(path: Path, chunk_size: int):
    buffer = ""
    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            safe_idx = max(buffer.rfind("\n"), buffer.rfind(" "))
            if safe_idx != -1:
                yield buffer[: safe_idx + 1]
                buffer = buffer[safe_idx + 1 :]
    if buffer:
        yield buffer


def init_worker_tokenizer(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = BPETokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens,
    )


def encode_shard(
    shard_index: int,
    text: str,
    shard_dir: str,
) -> dict[str, int | str]:
    if _WORKER_TOKENIZER is None:
        raise RuntimeError("Worker tokenizer was not initialized")

    tokens = np.asarray(_WORKER_TOKENIZER.encode(text), dtype=np.uint16)
    shard_path = Path(shard_dir) / f"shard_{shard_index:06d}.npy"
    np.save(shard_path, tokens)
    return {
        "index": shard_index,
        "path": str(shard_path),
        "num_tokens": int(tokens.size),
        "max_token_id": int(tokens.max()) if tokens.size else -1,
    }


def submit_until_full(
    executor: ProcessPoolExecutor,
    pending: dict[object, int],
    batch_iter,
    shard_dir: Path,
    max_pending: int,
) -> tuple[int, bool]:
    submitted = 0
    exhausted = False
    while len(pending) < max_pending:
        try:
            shard_index, text = next(batch_iter)
        except StopIteration:
            exhausted = True
            break
        future = executor.submit(encode_shard, shard_index, text, str(shard_dir))
        pending[future] = shard_index
        submitted += 1
    return submitted, exhausted


def encode_to_shards(
    path: Path,
    split: str,
    shard_dir: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    read_chunk_bytes: int,
    num_workers: int,
) -> list[dict[str, int | str]]:
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, int | str]] = []
    pending: dict[object, int] = {}
    batch_iter = enumerate(iter_safe_text_batches(path, read_chunk_bytes))
    max_pending = max(1, num_workers * 2)
    total_submitted = 0
    total_completed = 0
    exhausted = False

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker_tokenizer,
        initargs=(str(vocab_path), str(merges_path), special_tokens),
    ) as executor:
        while pending or not exhausted:
            submitted, exhausted_now = submit_until_full(
                executor,
                pending,
                batch_iter,
                shard_dir,
                max_pending,
            )
            total_submitted += submitted
            exhausted = exhausted or exhausted_now
            if not pending:
                continue

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                result = future.result()
                results.append(result)
                total_completed += 1
                print(
                    f"[{split}] encoded shard {result['index']} "
                    f"({result['num_tokens']} tokens) "
                    f"[{total_completed}/{total_submitted}]",
                    flush=True,
                )

    return sorted(results, key=lambda result: int(result["index"]))


def merge_shards(
    shard_results: list[dict[str, int | str]],
    out_path: Path,
) -> tuple[int, int]:
    total_tokens = sum(int(result["num_tokens"]) for result in shard_results)
    max_token_id = max((int(result["max_token_id"]) for result in shard_results), default=-1)
    array = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    offset = 0
    for result in shard_results:
        shard = np.load(str(result["path"]), mmap_mode="r")
        next_offset = offset + int(shard.shape[0])
        array[offset:next_offset] = shard
        offset = next_offset
    array.flush()
    return total_tokens, max_token_id


def write_tokens_parallel(
    path: Path,
    split: str,
    out_path: Path,
    output_dir: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    read_chunk_bytes: int,
    num_workers: int,
) -> tuple[int, int]:
    shard_dir = output_dir / f".{split}_shards"
    print(
        f"[{split}] encoding {path} with {num_workers} worker processes ...",
        flush=True,
    )
    shard_results = encode_to_shards(
        path=path,
        split=split,
        shard_dir=shard_dir,
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens,
        read_chunk_bytes=read_chunk_bytes,
        num_workers=num_workers,
    )
    print(f"[{split}] merging {len(shard_results)} shards to {out_path} ...", flush=True)
    total_tokens, max_token_id = merge_shards(shard_results, out_path)
    shutil.rmtree(shard_dir)
    return total_tokens, max_token_id


def main() -> None:
    args = parse_args()
    if args.special_tokens is None:
        args.special_tokens = ["<|endoftext|>"]
    if args.num_workers < 1:
        args.num_workers = max(1, (os.cpu_count() or 2) - 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=args.special_tokens,
    )

    vocab_size = len(tokenizer.id_to_vocab)
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"Tokenizer vocab size {vocab_size} does not fit in uint16 IDs")

    datasets = {
        "train": args.train_path,
        "valid": args.valid_path,
    }
    metadata: dict[str, object] = {
        "dtype": "uint16",
        "tokenizer_vocab_path": str(args.vocab_path),
        "tokenizer_merges_path": str(args.merges_path),
        "special_tokens": args.special_tokens,
        "vocab_size": vocab_size,
        "num_workers": args.num_workers,
        "datasets": {},
    }

    for split, path in datasets.items():
        out_path = args.output_dir / f"{split}_ids.npy"
        if args.num_workers == 1:
            print(f"[{split}] counting tokens in {path} ...")
            total_tokens, max_token_id = count_tokens(
                tokenizer,
                path,
                args.read_chunk_bytes,
                args.token_batch_size,
            )
            print(f"[{split}] writing {total_tokens} tokens to {out_path} ...")
            write_tokens(
                tokenizer,
                path,
                out_path,
                total_tokens,
                args.read_chunk_bytes,
                args.token_batch_size,
            )
        else:
            total_tokens, max_token_id = write_tokens_parallel(
                path=path,
                split=split,
                out_path=out_path,
                output_dir=args.output_dir,
                vocab_path=args.vocab_path,
                merges_path=args.merges_path,
                special_tokens=args.special_tokens,
                read_chunk_bytes=args.read_chunk_bytes,
                num_workers=args.num_workers,
            )
        metadata["datasets"][split] = {
            "source_path": str(path),
            "token_ids_path": str(out_path),
            "num_tokens": total_tokens,
            "max_token_id": max_token_id,
        }

    metadata_path = args.output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
