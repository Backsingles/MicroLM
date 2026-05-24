# 07. 复现与恢复 Runbook

## 1. 恢复源码

如果已有仓库：

```powershell
cd E:\MicroLM
git checkout main
git reset --hard 88c5dffa806d048c129671eeaa0c7c3b194377b0
```

如果从远程恢复：

```powershell
git clone <repo-url> MicroLM
cd MicroLM
git checkout 88c5dffa806d048c129671eeaa0c7c3b194377b0
```

注意：上面 `git reset --hard` 会丢弃当前工作区改动，只能在确认需要恢复时执行。

## 2. 建立环境

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[all]"
```

基础依赖来自 `pyproject.toml`：

```text
torch
numpy
einops>=0.8.1
regex>=2024.11.6
wandb>=0.19.7
```

Qwen extra：

```text
transformers>=4.40
peft>=0.12
datasets>=2.18
requests>=2.31
```

dev extra：

```text
pytest>=8.3.4
```

## 3. 最小验证

```powershell
pytest tests/
python scripts/train_pretrain.py --config configs/pretrain_smoke.json
python scripts/train_sft.py --config configs/sft_smoke.json
```

Qwen smoke 需要本地已有基座模型和 CUDA：

```powershell
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured_smoke.json
```

## 4. MiniMind 正式链路

### 4.1 准备原始数据

参考 `data/README.md` 下载：

```text
data/pretrain_t2t_mini.jsonl
data/minimind_sft/gongjy/minimind_dataset/sft_t2t_mini.jsonl
```

### 4.2 清洗预训练语料

```bash
python scripts/prepare_pretrain_jsonl.py \
  --input-path data/pretrain_t2t_mini.jsonl \
  --output-dir data/pretrain_clean \
  --document-separator "<|endoftext|>" \
  --replace-literal "<|im_end|>=\n" \
  --replace-literal "<|im_start|>=\n" \
  --replace-literal "<think>=\n" \
  --replace-literal "</think>=\n" \
  --clean-html
```

### 4.3 训练 tokenizer

```bash
python scripts/train_tokenizer.py --config configs/tokenizer_full_clean.json
```

### 4.4 tokenization

```bash
python scripts/tokenize_corpus.py --config configs/tokenize_full_corpus.json
```

### 4.5 pretrain

```bash
python scripts/train_pretrain.py --config configs/pretrain_full_corpus.json
```

### 4.6 SFT

```bash
python scripts/train_sft.py --config configs/sft_baseline.json
python scripts/train_sft.py --config configs/sft_lora.json
```

## 5. Qwen 正式链路

### 5.1 准备数据和模型

需要：

```text
data/instructie/train_zh.json
data/instructie/valid_zh.json
data/instructie/test_zh.json
data/instructie/schema_zh.json
Qwen2.5-1.5B-Instruct/
```

### 5.2 跑 InstructIE pipeline

```powershell
python scripts/01_normalize.py
python scripts/02_filter.py
python scripts/03_quality_tier.py
python scripts/04_derive_tasks.py
python scripts/05_stratified_sample.py
python scripts/06_to_chat_jsonl.py
```

验收：

```text
data/sft_candidate/train.jsonl
data/sft_candidate/valid.jsonl
data/sft_candidate/metadata.json
```

规模应为：

```text
train: 28,500
valid: 1,500
```

### 5.3 Qwen LoRA

```powershell
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured.json
```

验收：

```text
outputs/qwen_lora/adaptor_final/
outputs/qwen_lora/best_adaptor/
outputs/qwen_lora/train_log.jsonl
```

期望 final val_loss 约：

```text
0.155349
```

### 5.4 导出模型

```powershell
python scripts/export_final_model.py
```

验收：

```text
outputs/qwen_lora_merged_final/
outputs/qwen_lora_merged_final/config.json
outputs/qwen_lora_merged_final/model.safetensors
outputs/qwen_lora_merged_final/tokenizer.json
outputs/qwen_lora_merged_final/export_metadata.json
```

## 6. 评测复现

### 6.1 MicroLM 通用评测

```powershell
.venv\Scripts\python.exe scripts\run_eval_prompts.py `
  --eval-file eval\prompts_v1.json `
  --models pretrain=outputs\pretrain_full_corpus\ckpt_final.pt baseline=outputs\sft_baseline\ckpt_final.pt lora=outputs\sft_lora\ckpt_final.pt `
  --out-dir results\lora_vs_full_sft_v1 `
  --lora-adaptor outputs\sft_lora\lora_adaptor.pt `
  --device cuda `
  --dtype float16
```

### 6.2 Qwen valid 200

```powershell
python scripts/evaluate_qwen_valid_jsonl.py `
  --model-path outputs\qwen_lora_merged_final `
  --data-path data\sft_candidate\valid.jsonl `
  --limit 200 `
  --output-dir results\qwen_valid_eval_200
```

期望：

```text
Parse%: 100%
Field F1: ~0.784
Pair F1: ~0.673
Exact Match: ~20%
```

### 6.3 vLLM stability

服务启动后：

```powershell
.venv\Scripts\python.exe scripts\check_structured_stability.py `
  --base-url http://localhost:8000 `
  --rounds 3 `
  --limit 40 `
  --output-dir results\vllm_benchmark_schema_strict
```

期望：

```text
Round 3 schema-strict:
Parse% 100%
Strict% 52.5%
Projected-Strict% 75.0%
```

## 7. vLLM 部署复现

Windows/WSL：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_vllm_wsl.ps1
```

WSL：

```bash
cd /mnt/e/MicroLM
source /root/.venvs/microlm-vllm/bin/activate
bash scripts/serve_vllm.sh
```

Smoke：

```powershell
python scripts/smoke_vllm.py `
  --base-url http://localhost:8000 `
  --structured `
  --output results\vllm_benchmark\smoke_results.json
```

Benchmark：

```powershell
python scripts/bench_vllm_local.py `
  --base-url http://localhost:8000 `
  --output-dir results\vllm_benchmark
```

## 8. 最终验收清单

| 项目 | 验收 |
|---|---|
| 源码 | `pytest tests/` 通过 |
| tokenizer | `outputs/tokenizer_full_clean/vocab.json`, `merge.txt` 存在 |
| MicroLM pretrain | `outputs/pretrain_full_corpus/ckpt_final.pt` 存在 |
| MicroLM SFT | `outputs/sft_baseline/ckpt_final.pt` 存在 |
| MicroLM LoRA | `outputs/sft_lora/lora_adaptor.pt` 存在 |
| Qwen data | `data/sft_candidate/train.jsonl`, `valid.jsonl` 存在且规模正确 |
| Qwen LoRA | `outputs/qwen_lora/adaptor_final/` 存在 |
| merged model | `outputs/qwen_lora_merged_final/model.safetensors` 存在 |
| valid eval | Parse 100%，Field F1 约 0.784 |
| vLLM smoke | 5/5 PASS |
| schema strict | Projected-Strict 约 75% |

