# 03. 训练与推理说明

## 1. Tokenizer 训练

入口：

```text
scripts/train_tokenizer.py
```

配置：

```text
configs/tokenizer_full_clean.json
```

命令：

```bash
python scripts/train_tokenizer.py --config configs/tokenizer_full_clean.json
```

产物：

```text
outputs/tokenizer_full_clean/vocab.json
outputs/tokenizer_full_clean/merge.txt
```

关键设计：

- vocab 从 256 byte 起步。
- special token 独立处理。
- GPT-2 regex 预分词。
- 保存采用 byte-to-unicode 可见字符映射。

## 2. MicroLM Pretrain

入口：

```text
scripts/train_pretrain.py
```

正式配置：

```text
configs/pretrain_full_corpus.json
```

smoke 配置：

```text
configs/pretrain_smoke.json
```

正式命令：

```bash
python scripts/train_pretrain.py --config configs/pretrain_full_corpus.json
```

训练输入：

```text
data/pretrain_clean/tokenized_full/train_ids.npy
data/pretrain_clean/tokenized_full/valid_ids.npy
```

模型配置：

| 参数 | 值 |
|---|---:|
| vocab_size | 6400 |
| context_length | 512 |
| d_model | 512 |
| num_layers | 8 |
| num_heads | 8 |
| d_ff | 1344 |
| rope_theta | 1000000.0 |
| norm | RMSNorm |
| FFN | SwiGLU |

优化配置：

| 参数 | 值 |
|---|---:|
| lr | 0.0002 |
| min_lr | 0.00002 |
| warmup_iters | 2000 |
| max_norm | 1.0 |
| weight_decay | 0.1 |
| batch_size | 8 |
| max_iters | 50000 |

产物：

```text
outputs/pretrain_full_corpus/ckpt.pt
outputs/pretrain_full_corpus/ckpt_final.pt
outputs/pretrain_full_corpus/model_config.json
outputs/pretrain_full_corpus/resolved_train_config.json
```

训练 loop：

1. 加载 token ids 为 memmap。
2. `get_batch()` 随机采样 next-token 窗口。
3. 前向得到 logits。
4. `cross_entropy()` 计算全 token loss。
5. AdamW 更新。
6. warmup + cosine 调整 lr。
7. 梯度裁剪。
8. 每 100 step 评估一次 batch val_loss。
9. 每 1000 step 保存 checkpoint。

## 3. MicroLM SFT

入口：

```text
scripts/train_sft.py
```

正式全参配置：

```text
configs/sft_baseline.json
```

正式 LoRA 配置：

```text
configs/sft_lora.json
```

命令：

```bash
python scripts/train_sft.py --config configs/sft_baseline.json
python scripts/train_sft.py --config configs/sft_lora.json
```

关键输入：

```text
outputs/pretrain_full_corpus/ckpt_final.pt
outputs/tokenizer_full_clean/vocab.json
outputs/tokenizer_full_clean/merge.txt
data/minimind_sft/gongjy/minimind_dataset/*.jsonl
```

SFT 协议：

```text
<|system|>
...
<|user|>
...
<|assistant|>
...
<|endoftext|>
```

LoRA 协议使用的 EOS 可配置为 `</s>`，以配置文件为准。

SFT loss：

- input_ids 包含完整对话。
- labels 只在 assistant 回复区间是 token id。
- 其它位置为 `-100`。
- 训练时 shift logits/labels 后用 `masked_cross_entropy()`。

全参 SFT 产物：

```text
outputs/sft_baseline/ckpt_final.pt
outputs/sft_baseline/train_log.jsonl
outputs/sft_baseline/model_config.json
```

LoRA SFT 产物：

```text
outputs/sft_lora/ckpt_final.pt
outputs/sft_lora/lora_adaptor.pt
outputs/sft_lora/train_log.jsonl
```

记录结果：

| 实验 | 结果 |
|---|---|
| `sft_baseline` | val_loss 约 2.35 -> 2.20 |
| `sft_lora` | val_loss 约 2.41 -> 2.30 |
| LoRA 参数占比 | 0.83%，约 262K / 31.7M |
| LoRA adaptor 大小 | 约 1.0 MB |

## 4. Qwen LoRA 训练

入口：

```text
scripts/train_qwen_lora.py
```

正式配置：

```text
configs/qwen_lora_structured.json
```

smoke 配置：

```text
configs/qwen_lora_structured_smoke.json
```

命令：

```bash
python scripts/train_qwen_lora.py --config configs/qwen_lora_structured.json
```

核心参数：

| 参数 | 值 |
|---|---|
| base model | `./Qwen2.5-1.5B-Instruct` |
| LoRA r / alpha | 8 / 16 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| dropout | 0.05 |
| batch_size | 4 |
| grad_accum | 4 |
| effective batch | 16 |
| max_steps | 2000 |
| warmup_steps | 100 |
| max_length | 512 |
| precision | fp16 |
| lr | 2e-5 |
| weight_decay | 0.01 |

数据：

```text
data/sft_candidate/train.jsonl
data/sft_candidate/valid.jsonl
```

Qwen SFT Dataset 逻辑：

1. 读取 chat-style JSONL。
2. 注入系统 prompt。
3. 使用 `tokenizer.apply_chat_template()` 得到完整 token ids。
4. 单独渲染 prefix，确定 assistant answer 起点。
5. prefix labels 置为 `-100`。
6. 长样本保留 tail，避免 answer 被截断。
7. batch 内动态 padding。

训练结果：

| step | train_loss | val_loss |
|---:|---:|---:|
| 100 | 0.372947 | 0.402493 |
| 500 | 0.255109 | 0.205028 |
| 1000 | 0.150461 | 0.177738 |
| 1500 | 0.210959 | 0.165426 |
| 2000 | 0.186020 | 0.155349 |

产物：

```text
outputs/qwen_lora/adaptor_final/
outputs/qwen_lora/best_adaptor/
outputs/qwen_lora/ckpt_step_*/
outputs/qwen_lora/train_log.jsonl
```

参数效率：

| 指标 | 值 |
|---|---:|
| 基座参数 | 1,543,714,304 |
| LoRA 可训练参数 | 约 2.18M |
| 可训练占比 | 0.14% |
| adaptor 大小 | 约 8.3 MB |
| 存储节省 | 约 99.7% |

## 5. Qwen 模型导出

入口：

```text
scripts/export_final_model.py
```

命令：

```bash
python scripts/export_final_model.py
```

流程：

1. 加载 base model。
2. 加载 tokenizer。
3. 加载 PEFT adaptor。
4. `merge_and_unload()` 合并 LoRA。
5. 保存 HuggingFace 标准 CausalLM 目录。
6. 写出 `export_metadata.json`。

推荐部署产物：

```text
outputs/qwen_lora_merged_final/
```

导出 metadata：

| 项目 | 值 |
|---|---|
| timestamp | 2026-05-20 23:02:02 |
| total_params | 1,543,714,304 |
| adaptor_path | `outputs/qwen_lora/adaptor_final` |
| elapsed_sec | 6.9 |
| peft_version | 0.19.1 |

## 6. MicroLM 单轮推理

入口：

```text
scripts/generate_text.py
```

能力：

- 加载 checkpoint。
- 自动读取 checkpoint 同目录 `model_config.json`。
- 支持 prompt、conversations-json、conversations-path。
- 支持 temperature、top_p、max_new_tokens。
- 支持 dtype 与 device auto。

示例：

```bash
python scripts/generate_text.py \
  --checkpoint-path outputs/pretrain_full_corpus/ckpt_final.pt \
  --prompt "春天的早晨，"
```

## 7. MicroLM 多轮 Chat

入口：

```text
scripts/chat.py
```

能力：

- 全参或 LoRA checkpoint 加载。
- 多轮 REPL。
- `/history` 查看历史。
- `/save` 保存 JSONL。
- `/quit` 退出。
- 清理 Unicode surrogate。

示例输入记录：

```text
reports/chat_smoke_input.txt
reports/chat_smoke_session_utf8.jsonl
```

## 8. KV Cache Benchmark

入口：

```text
scripts/benchmark_kvcache.py
```

结果：

```text
results/kvcache_benchmark.csv
results/kvcache_benchmark.json
```

指标：

| 指标 | 值 |
|---|---:|
| 配置数 | 20 |
| 平均加速比 | 3.86x |
| 最大加速比 | 9.08x |
| 最大加速配置 | prompt=256, gen=256 |
| cache decode 平均吞吐 | 约 100 tok/s |
| no-cache 平均吞吐 | 约 31.6 tok/s |

