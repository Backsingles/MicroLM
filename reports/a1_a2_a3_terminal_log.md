# A1-A3 Terminal Log

Workspace: `E:\MicroLM`

## A1 - 预训练语料清洗与切分

Command:

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_pretrain_jsonl.py --input-path data\pretrain_t2t_mini.jsonl --output-dir data\pretrain_clean --document-separator '<|endoftext|>' --replace-literal '<|im_end|>=\n' --replace-literal '<|im_start|>=\n' --replace-literal '<think>=\n' --replace-literal '</think>=\n' --clean-html
```

Output:

```text
wrote train split to data\pretrain_clean\train.txt
wrote valid split to data\pretrain_clean\valid.txt
wrote tokenizer corpus to data\pretrain_clean\tokenizer_corpus.txt
saved metadata to data\pretrain_clean\metadata.json
documents: raw=1270238, kept=1264051, empty=6, short=5926, long=0, dupes=255
```

## A2 - 构造 tokenizer 训练样本

Command:

```powershell
@'
from pathlib import Path

src = Path("data/pretrain_clean/tokenizer_corpus.txt")
dst = Path("data/pretrain_clean/tokenizer_sample.txt")
sample_bytes = 15 * 1024 * 1024

data = src.read_bytes()[:sample_bytes]
try:
    data.decode("utf-8")
except UnicodeDecodeError as exc:
    data = data[:exc.start]

dst.write_bytes(data)
'@ | .\.venv\Scripts\python.exe -
```

Output:

```text

```

## A3 - 训练 tokenizer

Command:

```powershell
& .\.venv\Scripts\python.exe scripts\train_tokenizer.py --config configs\tokenizer_full_clean.json
```

Output:

```text
saved tokenizer files to outputs\tokenizer_full_clean
```

## Verification - A1/A2/A3 产物校验

Command:

```powershell
@'
import json
from pathlib import Path

metadata = json.load(open('data/pretrain_clean/metadata.json', encoding='utf-8'))
paths = [
    Path('data/pretrain_clean/train.txt'),
    Path('data/pretrain_clean/valid.txt'),
    Path('data/pretrain_clean/tokenizer_corpus.txt'),
    Path('data/pretrain_clean/tokenizer_sample.txt'),
    Path('outputs/tokenizer_full_clean/vocab.json'),
    Path('outputs/tokenizer_full_clean/merge.txt'),
]
print('A1 metadata:')
print('  raw_documents=', metadata['filter_stats']['total_raw_documents'], sep='')
print('  kept_documents=', metadata['filter_stats']['total_kept'], sep='')
print('  train_documents=', metadata['train']['documents'], sep='')
print('  valid_documents=', metadata['valid']['documents'], sep='')
print('  filter_rate=', metadata['filter_stats']['filter_rate'], sep='')
print('Files:')
for path in paths:
    print(f'  {path}: exists={path.exists()} size_mb={(path.stat().st_size / 1024 / 1024 if path.exists() else 0):.2f}')
Path('data/pretrain_clean/tokenizer_sample.txt').read_bytes().decode('utf-8')
vocab = json.load(open('outputs/tokenizer_full_clean/vocab.json', encoding='utf-8'))
merge_lines = sum(1 for _ in open('outputs/tokenizer_full_clean/merge.txt', encoding='utf-8'))
print('Tokenizer:')
print('  vocab_size=', len(vocab), sep='')
print('  merge_lines=', merge_lines, sep='')
print('  tokenizer_sample_utf8=ok')
'@ | .\.venv\Scripts\python.exe -
```

Output:

```text
A1 metadata:
  raw_documents=1270238
  kept_documents=1264051
  train_documents=1251547
  valid_documents=12504
  filter_rate=0.49%
Files:
  data\pretrain_clean\train.txt: exists=True size_mb=1172.71
  data\pretrain_clean\valid.txt: exists=True size_mb=11.59
  data\pretrain_clean\tokenizer_corpus.txt: exists=True size_mb=1166.22
  data\pretrain_clean\tokenizer_sample.txt: exists=True size_mb=15.00
  outputs\tokenizer_full_clean\vocab.json: exists=True size_mb=0.28
  outputs\tokenizer_full_clean\merge.txt: exists=True size_mb=0.08
Tokenizer:
  vocab_size=6400
  merge_lines=6143
  tokenizer_sample_utf8=ok
```
