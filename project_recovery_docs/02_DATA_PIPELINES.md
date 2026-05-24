# 02. 数据 Pipeline 说明

## 1. 数据源总览

| 数据源 | 用途 | 主线 | 原始规模 | 处理后规模 |
|---|---|---|---:|---:|
| MiniMind | 自研 MicroLM 预训练与 SFT | A/B | 约 141 万条 | pretrain 约 126 万清洗样本；SFT 使用 MiniMind 对话数据 |
| InstructIE | Qwen 结构化信息抽取 SFT | C/D | train 171,471 | 28,500 train + 1,500 valid |

## 2. MiniMind 预训练数据

### 2.1 原始输入

| 项目 | 值 |
|---|---|
| 文件 | `data/pretrain_t2t_mini.jsonl` |
| 来源 | `jingyaogong/minimind_dataset` |
| 格式 | 每行 `{"text": "..."}` |
| 原始规模 | 1,270,238 条 |
| 文件大小 | 约 1.24 GB |

下载方式参考 `data/README.md`。

### 2.2 清洗脚本

脚本：`scripts/prepare_pretrain_jsonl.py`

典型命令：

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

### 2.3 清洗逻辑

1. 逐行读取 JSONL。
2. 取 `text` 字段。
3. 替换模板残留。
4. 清理控制字符。
5. 清理 HTML 标签。
6. 压缩空白。
7. 进行长度过滤。
8. 用 SHA256 精确去重。
9. 用 SHA1 哈希确定性划分 train/valid。
10. 每篇文档间插入 EOS 分隔符。
11. 输出 metadata。

### 2.4 已记录结果

| 指标 | 数值 |
|---|---:|
| train 文档 | 1,251,547 |
| valid 文档 | 12,504 |
| HTML 标签清理 | 7,625 条，0.60% |
| 空白压缩 | 59,393 条，4.68% |
| 精确去重 | 255 条，0.02% |
| 总过滤率 | 0.49% |

输出：

```text
data/pretrain_clean/train.txt
data/pretrain_clean/valid.txt
data/pretrain_clean/tokenizer_corpus.txt
data/pretrain_clean/metadata.json
```

## 3. Tokenizer 数据

正式 tokenizer 只使用全量语料的 15 MB sample：

```text
data/pretrain_clean/tokenizer_sample.txt
```

正式配置：

```text
configs/tokenizer_full_clean.json
```

关键参数：

| 参数 | 值 |
|---|---|
| vocab_size | 6400 |
| special_tokens | `<|endoftext|>` |
| output_dir | `outputs/tokenizer_full_clean` |

产物：

```text
outputs/tokenizer_full_clean/vocab.json
outputs/tokenizer_full_clean/merge.txt
```

## 4. Token IDs 数据

脚本：`scripts/tokenize_corpus.py`

正式配置：

```text
configs/tokenize_full_corpus.json
```

输入：

```text
data/pretrain_clean/train.txt
data/pretrain_clean/valid.txt
outputs/tokenizer_full_clean/vocab.json
outputs/tokenizer_full_clean/merge.txt
```

输出：

```text
data/pretrain_clean/tokenized_full/train_ids.npy
data/pretrain_clean/tokenized_full/valid_ids.npy
data/pretrain_clean/tokenized_full/metadata.json
```

关键实现：

- 支持串行和多进程编码。
- 多进程模式按换行/空格安全边界切分 chunk。
- 每个 shard 编码为 `.npy`。
- merge 阶段用 `np.lib.format.open_memmap` 写入最终 `.npy`，避免一次性占用大量内存。

## 5. MiniMind SFT 数据

| 项目 | 值 |
|---|---|
| 原始文件 | `data/minimind_sft/gongjy/minimind_dataset/sft_t2t_mini.jsonl` |
| 用途 | MicroLM SFT baseline 和 LoRA |
| 渲染模块 | `microlm.training.sft.SFTDataset` |
| loss 协议 | assistant-only masked loss |

正式配置中出现的拆分：

```text
data/minimind_sft/gongjy/minimind_dataset/sft_t2t_valid_1000.jsonl
data/minimind_sft/gongjy/minimind_dataset/sft_t2t_train_995.jsonl
data/minimind_sft/gongjy/minimind_dataset/sft_t2t_valid_005.jsonl
```

这些文件用于不同 SFT 实验；恢复时需根据配置确认本地是否已有对应拆分。

## 6. InstructIE 原始数据

| 项目 | 值 |
|---|---|
| 来源 | `zjunlp/InstructIE` |
| train | 171,471 |
| valid | 1,004 |
| test | 1,002 |
| schema | 12 类 topic schema |
| 语言 | 中英双语 |

项目使用中文 split：

```text
data/instructie/train_zh.json
data/instructie/valid_zh.json
data/instructie/test_zh.json
data/instructie/schema_zh.json
```

## 7. InstructIE 六步 Pipeline

阈值配置集中在：

```text
scripts/conf.py
```

### Step 1: 字段标准化

脚本：

```text
scripts/01_normalize.py
```

解决问题：

- train/valid/test 字段命名不一致。
- train 使用 `text`，valid/test 使用 `input`。
- relation 内部字段在不同 split 间漂移。
- cate 名称如“建筑结构”需要归一为“建筑”。

输出：

```text
data/processed/normalized_train.jsonl
data/processed/normalized_valid.jsonl
data/processed/normalized_test.jsonl
reports/normalize_report.json
```

### Step 2: 两层过滤

脚本：

```text
scripts/02_filter.py
```

硬过滤阈值：

| 参数 | 值 |
|---|---:|
| min_relations | 1 |
| max_relations | 25 |
| min_input_len | 15 |
| max_input_len | 800 |
| max_output_json_len | 2500 |
| max_head_tail_len | 100 |

软过滤：

- input length P99
- output length P99
- relation count P99
- head/tail length P99

输出：

```text
data/processed/filtered_train.jsonl
reports/filter_report.json
```

记录结果：硬过滤 3,585 条，P99 软过滤 4,257 条，剩余约 163,629 条。

### Step 3: 质量分层

脚本：

```text
scripts/03_quality_tier.py
```

分层逻辑：

| tier | 条件 |
|---|---|
| high | head/tail 全部在原文中，且关系数和输入长度处于理想区间 |
| medium | head/tail 匹配率 >= 0.8 或部分指标偏离 |
| low | 质量问题明显 |

理想区间：

```text
relation_count: 2-10
input_len: 30-400
```

输出：

```text
data/processed/tiered_train.jsonl
reports/quality_report.json
```

记录结果：high 约 95.5%，medium 约 3.9%，low 约 0.6%，high 约 156,275 条。

### Step 4: 任务派生

脚本：

```text
scripts/04_derive_tasks.py
```

派生四类任务：

| task_type | 比例 | 作用 |
|---|---:|---|
| `ie_extraction` | 50% | 标准信息抽取 |
| `text_to_json` | 25% | 文本到 JSON 转换 |
| `format_following` | 15% | 格式遵循 |
| `schema_repair` | 10% | schema 修复 |

输出：

```text
data/processed/derived_all.jsonl
reports/derive_report.json
```

记录结果：约 623,650 条派生样本。

### Step 5: 分层采样

脚本：

```text
scripts/05_stratified_sample.py
```

采样维度：

- task_type
- topic
- quality_tier
- complexity

配置：

| 参数 | 值 |
|---|---:|
| candidate_target | 30000 |
| final_target | 15000 |
| internal_valid_ratio | 0.05 |
| random_seed | 42 |

输出：

```text
data/processed/sampled_train.jsonl
reports/sample_report.json
```

### Step 6: chat JSONL 转写

脚本：

```text
scripts/06_to_chat_jsonl.py
```

输出：

```text
data/sft_candidate/train.jsonl
data/sft_candidate/valid.jsonl
data/sft_candidate/metadata.json
```

最终规模：

| split | 条数 |
|---|---:|
| train | 28,500 |
| valid | 1,500 |

特点：

- 统一为 chat-style JSONL。
- 全量 JSON 合法性校验 100% 通过。
- 作为 Qwen LoRA 训练输入。

## 8. Hardcase refine 数据

目录：

```text
data/qwen_refine/
```

文件：

```text
hardcases_once.jsonl
eval_holdout.jsonl
train.jsonl
valid.jsonl
metadata.json
```

用途：

- 针对结构化失败样本做 hardcase replay/refinement。
- 已验证收益不稳定，不推荐替代默认 `outputs/qwen_lora_merged_final/`。

