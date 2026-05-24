# 01. 架构与模块说明

## 1. 顶层目录

| 路径 | 作用 |
|---|---|
| `microlm/` | 自研 Python 包，包含模型、tokenizer、训练、推理、结构化修复 |
| `scripts/` | 命令行入口：数据处理、训练、评测、部署、监控 |
| `configs/` | JSON 配置，覆盖 smoke 和正式实验 |
| `data/` | 数据说明、小型 smoke 数据、处理后的候选数据、refine 数据 |
| `eval/` | prompt 评测集 |
| `outputs/` | 训练和导出产物 |
| `results/` | 自动评测、benchmark、stability 结果 |
| `reports/` | 阶段报告、日志、复盘、终端记录 |
| `docs/` | 部署说明 |
| `Readme/` | 分卷中文文档与核心代码解析 |
| `tests/` | 单元测试 |

## 2. Python 包结构

```text
microlm/
  model/
    transformer.py
    lora.py
    kvcache.py
  tokenizer/
    bpe.py
    tokenizer.py
  training/
    data_loader.py
    loss.py
    optimizer.py
    scheduler.py
    gradient.py
    checkpoint.py
    sft.py
  inference/
    prompting.py
  structured/
    schema_repair.py
```

## 3. `microlm.model`

### `transformer.py`

自研 Transformer 主干。关键对象：

| 对象 | 职责 |
|---|---|
| `Linear` | 自定义线性层，使用 `torch.einsum("... i, o i -> ... o")` |
| `Embedding` | token embedding lookup |
| `RMSNorm` | RMSNorm，fp32 计算后转回输入 dtype |
| `SwiGLU` | `W2(SiLU(W1(x)) * W3(x))` |
| `SiLU_FFN` | 可选 SiLU-only FFN |
| `RotaryPositionalEmbedding` | RoPE，预计算 cos/sin buffer |
| `scaled_dot_product_attention` | 手写 attention，score/softmax 用 fp32 |
| `MultiHeadSelfAttention` | q/k/v/out projection，支持 RoPE 与 KV Cache |
| `TransformerBlock` | pre-norm attention + FFN residual block |
| `TransformerLM` | embedding、blocks、final norm、lm head、generate |
| `KVCache` | 每层保存历史 k/v |

正式模型配置：

| 参数 | 值 |
|---|---:|
| vocab_size | 6400 |
| context_length | 512 |
| d_model | 512 |
| num_layers | 8 |
| num_heads | 8 |
| d_head | 64 |
| d_ff | 1344 |
| rope_theta | 1000000.0 |
| norm_mode | `pre` |
| ffn_type | `swiglu` |

注意：

- `forward()` 会检查输入长度不能超过 `context_length`。
- `generate()` 当前不支持超过 context length 的长上下文生成。
- KV cache 路径中 decode 阶段不加 causal mask，因为每次 query 只看历史和当前 token。

### `lora.py`

LoRA 自研实现。核心公式：

```text
output = W x + (alpha / r) * B A x
```

关键对象：

| 对象 | 职责 |
|---|---|
| `LoRALinear` | 包裹原始 Linear，冻结原权重，只训练 A/B |
| `apply_lora_to_model` | 替换模型中目标 Linear 层 |
| `get_lora_params` | 返回 LoRA A/B 参数 |
| `get_lora_state_dict` | 只保存 LoRA 权重 |
| `load_lora_state_dict` | 加载 LoRA 权重 |
| `merge_lora` / `unmerge_lora` | 推理时合并/撤销 LoRA |
| `print_trainable_params` | 打印总参数和可训练参数占比 |

默认 target：

```text
q_proj, k_proj, v_proj, output_proj
```

关键修复点：

- LoRA A/B 参数必须使用 `original.weight.device`，否则 GPU 模型中会出现跨设备错误。
- checkpoint 加载顺序要先加载 base checkpoint，再注入 LoRA，避免 state_dict 结构不匹配。

## 4. `microlm.tokenizer`

### `bpe.py`

自研 BPE 训练：

1. 以 256 个 byte 初始化 vocab。
2. 使用 special token regex 切分，避免 special token 被合并。
3. 使用 GPT-2 regex 预分词。
4. 统计 pair 频次。
5. 迭代选择最高频 pair 合并。
6. 保存 `vocab.json` 和 `merge.txt`。

### `tokenizer.py`

`BPETokenizer` 支持：

| 方法 | 作用 |
|---|---|
| `from_files` | 从 vocab/merge 文件恢复 tokenizer |
| `encode` | special token aware 编码 |
| `_encode_text_segment` | 普通文本 BPE 合并 |
| `decode` | bytes 拼接后 UTF-8 decode |
| `encode_iterable` | 按安全边界流式编码大文本 |

关键边界：

- special token 会在初始化时追加到 vocab，可能导致实际 vocab 大于模型配置的 `vocab_size`。
- SFT 训练代码已有 embedding/lm_head resize，但推理和配置仍需注意 token id 越界。

## 5. `microlm.training`

| 文件 | 内容 |
|---|---|
| `data_loader.py` | `get_batch()` 随机采样连续 token 窗口，构造 next-token x/y |
| `loss.py` | `cross_entropy()` 与 `masked_cross_entropy()` |
| `optimizer.py` | 自实现 AdamW |
| `scheduler.py` | warmup + cosine decay |
| `gradient.py` | 全局 L2 梯度裁剪 |
| `checkpoint.py` | 保存/加载 checkpoint，处理 `_orig_mod.` 前缀 |
| `sft.py` | chat prompt 渲染、assistant-only loss mask、SFTDataset |

SFT role marker：

| role | marker |
|---|---|
| system | `<|system|>\n` |
| user | `<|user|>\n` |
| assistant | `<|assistant|>\n` |
| tool | `<|tool|>\n` |

SFT loss 只训练 assistant 区间，其它 token label 为 `-100`。

## 6. `microlm.inference`

`prompting.py` 统一处理：

- 纯文本 prompt。
- conversations JSON 字符串。
- conversations JSON 文件路径。

约束：

- `conversations_json` 和 `conversations_path` 不能同时提供。
- conversations 必须是非空 list，元素必须有 string `role` 和 `content`。
- generation prompt 不能以 assistant 消息结束。

## 7. `microlm.structured`

`schema_repair.py` 是结构化服务层的核心。能力：

- 清理 markdown JSON fence。
- 解析 JSON，支持重复 key 合并。
- 字段 alias 归一化。
- 递归扫描实体嵌套 JSON，投影到 schema 字段。
- 类型归一化：string、string_or_list、list。
- 枚举归一化：如材料、用途、所属科室。
- required 字段缺失检测。
- 构造 schema-strict prompt。

注意：`pyproject.toml` 当前 packages 列表需要检查是否包含 `microlm.structured`。如果要正式打包发布，应补充这个子包。

## 8. 脚本分组

| 分组 | 脚本 |
|---|---|
| MiniMind 数据 | `prepare_pretrain_jsonl.py`, `train_tokenizer.py`, `tokenize_corpus.py` |
| MicroLM 训练 | `train_pretrain.py`, `train_sft.py` |
| MicroLM 推理 | `generate_text.py`, `chat.py`, `benchmark_kvcache.py` |
| InstructIE pipeline | `01_normalize.py` 到 `06_to_chat_jsonl.py`, `conf.py` |
| Qwen 训练/导出 | `train_qwen_lora.py`, `export_final_model.py`, `download_c0_qwen.py` |
| 评测 | `run_eval_prompts.py`, `run_instructie_eval.py`, `evaluate_qwen_valid_jsonl.py`, `check_structured_stability.py` |
| 服务化 | `serve_vllm.sh`, `start_vllm_wsl.ps1`, `smoke_vllm.py`, `bench_vllm_local.py`, `structured_vllm_client.py` |
| 辅助 | `watch_loss.py`, `summarize_eval_results.py`, `summarize_refine_comparison.py`, `build_qwen_refine_data.py` |

