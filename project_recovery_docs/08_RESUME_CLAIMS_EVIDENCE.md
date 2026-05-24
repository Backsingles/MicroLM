# 08. 简历项目证据链与面试回答口径

生成日期：2026-05-24  
对应仓库提交：`88c5dffa806d048c129671eeaa0c7c3b194377b0`  
用途：把简历中的每一句项目描述拆成“证据、可回答口径、风险点、建议改写”，用于面试追问和项目复盘。

## 0. 总结结论

这份恢复文档已经可以支撑面试官围绕项目的大多数技术追问：为什么先做 MicroLM、为什么迁移到 Qwen、数据 pipeline 怎么设计、LoRA 为什么有效、如何评测结构化输出、如何部署到 vLLM、Parse% 和 F1 的区别是什么。

但如果完全照当前简历原文回答，会有几个数字口径需要修正或补证据：

| 简历数字 | 当前证据状态 | 建议 |
|---|---|---|
| `316M tokens` | 当前恢复文档没有直接给出可复验的 token 统计文件 | 找到 tokenized 语料 metadata 后再保留；否则改成“全量 MiniMind tokenized 语料” |
| `Qwen val_loss 1.115 -> 0.843` | 与当前正式日志不一致；正式日志是 `0.402493 -> 0.155349` | 用正式日志数字，或补上对应 `1.115 -> 0.843` 的实验日志 |
| `1000 条验证样本 Field JSON F1 0.057 -> 0.541` | 当前正式 content-level 评测是 200 条；hardcase 里有 `pair_f1=0.0575`，但不是“1000 条 Field F1” | 改成当前可证明的 200 条 valid JSONL 指标，或补 1000 条评测结果 |
| `4 模型 × 40 Prompt × 4 指标` | `4 模型 × 40 Prompt` 有证据；指标文件实际包含 Parse、Strict、Alias-Strict、Missing、Hallucination 等 | 面试时说“多项结构化指标”，不要和 200 条 F1 评测混成同一实验 |

面试策略：不要硬背数字。把项目拆成两条线讲清楚：MicroLM 证明“从零训练闭环能力”，Qwen 证明“迁移、结构化抽取、评测、部署能力”。

## 1. 简历原文拆解

原始简历描述：

> 基于 Qwen2.5-1.5B 的结构化信息抽取 LoRA 后训练与评测，2026 年 1 月 - 2026 年 4 月。

核心句子可以拆成 5 组：

1. MicroLM 自研训练闭环：PyTorch、31.7M 参数、MiniMind 清洗、SHA1 切分、BPE tokenizer、预训练、SFT baseline、LoRA SFT、CUDA 推理、pretrain loss。
2. 数据 pipeline：MiniMind 约 141 万条、6400 BPE、InstructIE 171K、标准化到 chat JSONL、28.5K train + 1.5K valid、JSON 校验。
3. 模型和 LoRA：8 层 Transformer、RoPE、SwiGLU、Qwen 1.55B、HF/PEFT/vLLM、MicroLM LoRA 0.83%、Qwen LoRA 0.14%、训练 2000 steps、验证 loss。
4. 推理和部署：KV Cache、REPL、vLLM、smoke、Parse%。
5. 自动评测：Base Qwen vs LoRA Qwen、4 模型 × 40 Prompt、结构化指标、valid JSONL 字段级 F1。

## 2. 一句话项目介绍

推荐回答：

> 这个项目分两阶段。第一阶段我用 PyTorch 从零实现了一个 31.7M 参数的 MicroLM，跑通 tokenizer、预训练、SFT、LoRA、KV Cache 推理和自动评测闭环，主要目的是证明我能掌控训练系统的底层细节。第二阶段我把这套方法迁移到 Qwen2.5-1.5B-Instruct 上，用 InstructIE 构造结构化抽取数据集，基于 PEFT 做 LoRA 后训练，再用自动评测、schema 约束和 vLLM 服务化完成训练到部署闭环。

如果面试官让你 30 秒讲清楚：

> 我不是只调了一个 LoRA 脚本，而是先自研 MicroLM 跑通小模型训练闭环，再迁移到 Qwen 做结构化信息抽取。项目里我负责数据清洗和切分、BPE tokenizer、训练脚本、LoRA 参数高效微调、JSON 结构化评测、KV Cache 推理优化、vLLM 在线部署和 benchmark。最终 Qwen LoRA 用 0.14% 可训练参数完成任务适配，在 200 条 valid JSONL 上 JSON parse 100%、Field F1 约 0.784、Pair F1 约 0.673。

## 3. 证据等级

| 等级 | 含义 | 面试使用方式 |
|---|---|---|
| A | 仓库中有脚本、配置、日志、结果文件直接支持 | 可以直接引用数字 |
| B | 报告或设计文档支持，但原始日志需要进一步定位 | 可以讲结论，但最好避免过细数字 |
| C | 当前文档与简历数字不一致，或缺少直接产物 | 改写简历，或者补跑/补存证据 |

## 4. 简历逐条证据链

### 4.1 MicroLM 自研训练闭环

简历表述：

> 基于 PyTorch 搭建 31.7M 参数 MicroLM 训练闭环，完成 MiniMind 数据清洗、SHA1 切分、BPE tokenizer 训练及 316M tokens 预训练；搭建 SFT baseline 并进行 LoRA SFT 微调，同时完成 CUDA 推理测试，预训练 loss 从 8.85 降至约 2.4。

证据状态：B，部分数字需要补强。

可直接支撑的点：

| 论点 | 证据位置 |
|---|---|
| PyTorch 自研 MicroLM | `microlm/model/transformer.py`、`microlm/model/attention.py`、`microlm/model/lora.py` |
| 31.7M 量级 | `outputs/pretrain_full_corpus/model_config.json`、`reports/interview_qa_qwen_microlm.md` |
| MiniMind 清洗和切分 | `scripts/prepare_pretrain_jsonl.py`、`scripts/tokenize_corpus.py`、`data/README.md` |
| SHA1 稳定切分 | `scripts/prepare_pretrain_jsonl.py`、`project_recovery_docs/02_DATA_PIPELINES.md` |
| BPE tokenizer | `scripts/train_tokenizer.py`、`tokenizers/microlm_bpe/` |
| 预训练 | `scripts/train_pretrain.py`、`configs/pretrain_full_corpus.json`、`reports/b1_pretrain_combined.log` |
| SFT baseline | `scripts/train_sft.py`、`outputs/sft_baseline/train_log.jsonl` |
| LoRA SFT | `microlm/model/lora.py`、`outputs/sft_lora/`、`reports/interview_qa_qwen_microlm.md` |
| CUDA 推理测试 | `scripts/generate.py`、`scripts/chat.py`、`project_recovery_docs/03_TRAINING_AND_INFERENCE.md` |

需要谨慎的点：

- `316M tokens` 当前恢复文档中没有直接 token count 产物。除非能指到 tokenized bin/metadata 的统计，否则面试时建议说“全量 tokenized 语料”。
- `pretrain loss 8.85 -> 2.4` 在访谈准备文档中有口径，但应补充对应训练日志或截图。仓库里可见 `reports/b1_pretrain_combined.log` 后段出现 `val_loss` 约 2.65 和 `train_loss` 约 2.48，能支持“进入 2.x 区间”，但“2.4”要说成“约 2.x”更稳。

推荐回答：

> MicroLM 是我为了掌握训练闭环自己搭的 8 层 Transformer。数据侧先把 MiniMind 文本清洗成统一 JSONL，再用 SHA1 做稳定切分，避免每次随机划分导致验证集漂移。tokenizer 用 BPE 训练到 6400 vocab，训练脚本负责 dataloader、mask、优化器、checkpoint、验证和日志。预训练 loss 从随机初始的 8.x 降到 2.x，说明模型确实学到了语言建模分布。后面我又在同一套框架上做了 SFT baseline 和 LoRA SFT，用来比较全参微调和参数高效微调。

常见追问：

| 问题 | 回答 |
|---|---|
| 为什么自己做 MicroLM，而不是一开始就用 Qwen？ | MicroLM 的目标不是追求最终效果，而是掌握训练系统：数据切分、tokenizer、attention、RoPE、SwiGLU、loss mask、checkpoint、KV Cache。Qwen 是后续任务迁移和部署线。 |
| SHA1 切分有什么意义？ | 用样本内容的哈希决定 train/valid/test，新增数据时旧样本不会因为随机种子变化而跑到别的 split，方便复现实验。 |
| BPE vocab 为什么是 6400？ | 对 31.7M 参数小模型来说，词表太大浪费 embedding 参数，太小又导致中文文本过度切碎；6400 是小模型容量、中文覆盖和训练成本之间的折中。 |
| pretrain loss 是什么？ | causal LM 的 token-level cross entropy，只在有效 token 上计算。loss 下降说明模型从随机猜 token 变成学到了局部语言结构。 |

### 4.2 数据 pipeline

简历表述：

> 处理 MiniMind 约 141 万条语料，生成 6400 词 BPE Tokenizer；Qwen 迁移链路处理 InstructIE 171K 条样本，完成标准化 → 过滤 → 分层 → 任务派生 → 采样 → chat JSONL 转换 6 步数据 Pipeline，生成 28.5K 训练样本与 1.5K 验证样本，JSON 校验 100%。

证据状态：A。

证据位置：

| 论点 | 证据位置 |
|---|---|
| MiniMind 数据说明 | `data/README.md` |
| BPE tokenizer | `scripts/train_tokenizer.py`、`tokenizers/microlm_bpe/` |
| InstructIE 原始规模和类别 | `scripts/conf.py`、`data/instructie_derived/metadata.json` |
| 6 步 pipeline | `scripts/01_standardize_instructie.py` 到 `scripts/06_build_qwen_sft_jsonl.py` |
| 28.5K train + 1.5K valid | `data/sft_candidate/metadata.json`、`reports/sft_candidate_report.md` |
| JSON 校验 | `scripts/validate_sft_jsonl.py`、`reports/sft_candidate_report.md` |

推荐回答：

> Qwen 迁移线没有直接拿原始 InstructIE 训练，而是先做标准化，把不同任务来源统一成 instruction、schema、input、output 的中间格式；再过滤掉 schema 不完整、输出不可解析或字段质量差的样本；之后按任务类型和领域做分层，派生出抽取、文本转 JSON、格式跟随、schema repair 等任务；再做采样，避免单一类别过多；最后转成 Qwen chat JSONL。最终训练集约 28.5K，验证集约 1.5K，并且训练标签 JSON 校验 100%。

常见追问：

| 问题 | 回答 |
|---|---|
| “JSON 校验 100%”指什么？ | 指训练/验证标签能被 `json.loads` 解析，不代表模型输出 100% 正确。模型输出另有 Parse%、Field F1、Pair F1 等评测。 |
| 为什么要任务派生？ | 结构化抽取不是单一格式。派生多个任务能让模型同时学习字段抽取、格式约束、schema 修复和文本转 JSON。 |
| 为什么要分层采样？ | 防止高频类别压倒长尾类别，保证 valid 集能覆盖不同领域和任务类型。 |
| 训练和验证如何避免泄漏？ | 先建立稳定 ID 和 split，再做下游转换；同源样本不能跨 split 混入。 |

### 4.3 Transformer 架构和 LoRA 参数效率

简历表述：

> 基于 Transformer 架构 8 层 RoPE + SwiGLU，Qwen 迁移模型 1.55B 参数，使用 HF / PEFT / vLLM，SFT 阶段 LoRA 微调仅训练 0.83% 参数 (1MB)，Qwen LoRA 微调仅训练 0.14% 参数 (8.3MB)，完成 2000 steps 后训练，验证 loss 从 1.115 降至 0.843。

证据状态：A/C。架构、工具、LoRA 参数量、2000 steps 有证据；`1.115 -> 0.843` 与正式日志不一致。

可直接支撑的点：

| 论点 | 证据位置 |
|---|---|
| 8 层 Transformer、RoPE、SwiGLU | `microlm/model/transformer.py`、`microlm/model/attention.py`、`outputs/pretrain_full_corpus/model_config.json` |
| Qwen2.5-1.5B-Instruct | `configs/qwen_lora_structured.json`、`outputs/qwen_lora/resolved_config.json` |
| HF / PEFT | `scripts/train_qwen_lora.py`、`outputs/qwen_lora/adaptor_final/adapter_config.json` |
| vLLM | `docs/vllm_deploy.md`、`reports/vllm_benchmark_report.md` |
| MicroLM LoRA 0.83% | `reports/interview_qa_qwen_microlm.md`、`Readme/项目全景图/05-评测、验证与部署闭环.md` |
| Qwen LoRA 0.14%、8.3MB | `reports/interview_qa_qwen_microlm.md`、`Readme/项目全景图/04-Qwen 迁移与结构化输出主线.md` |
| Qwen 2000 steps | `outputs/qwen_lora/train_log.jsonl` |
| Qwen 正式 loss | `outputs/qwen_lora/train_log.jsonl`：step 100 `val_loss=0.402493`，step 2000 `val_loss=0.155349` |

推荐改写：

> 基于 Transformer 架构 8 层 RoPE + SwiGLU 实现 MicroLM；迁移到 Qwen2.5-1.5B-Instruct 后使用 HF / PEFT / vLLM，MicroLM LoRA 仅训练约 0.83% 参数、adaptor 约 1MB，Qwen LoRA 仅训练约 0.14% 参数、adaptor 约 8.3MB；完成 2000 steps 后训练，正式日志中验证 loss 从 0.4025 降至 0.1553，降幅约 61.4%。

计算口径：

- MicroLM LoRA：可训练参数约 262K，基座约 31.7M，比例约 `262K / 31.7M = 0.83%`。`1MB` 是 adaptor 存储大小，不是参数个数。
- Qwen LoRA：可训练参数 `2,179,072`，总参数 `1,545,893,376`，比例约 `0.141%`。`8.3MB` 是 adaptor 存储大小。
- Qwen loss 降幅：`(0.402493 - 0.155349) / 0.402493 = 61.4%`。

常见追问：

| 问题 | 回答 |
|---|---|
| LoRA 为什么只训 q/k/v/o？ | 结构化输出主要改变注意力中的信息选择和组织方式，先对 attention projection 注入低秩增量，参数少、稳定、显存可控。 |
| 为什么不用 full fine-tune？ | 1.55B full fine-tune 成本高、过拟合风险更大；本项目目标是任务适配和部署闭环，LoRA 更适合快速验证。 |
| 为什么不用 QLoRA？ | QLoRA 能省显存，但会引入 4-bit 量化变量。这个项目单卡 FP16 LoRA 能跑通，所以优先减少实验变量。 |
| `1MB` 和 `8.3MB` 怎么解释？ | 它们是 adaptor 文件大小；参数比例要看 trainable params / total params。面试时不要说“只训练 8.3MB 参数”。 |
| `1.115 -> 0.843` 能不能讲？ | 除非能拿出对应日志。当前仓库正式日志应讲 `0.4025 -> 0.1553`。 |

### 4.4 推理、KV Cache 和 vLLM 部署

简历表述：

> 推理与性能优化，集成 KV Cache 实现推理加速约 3.86x，搭建 REPL 服务；基于 vLLM 完成在线推理部署，实现低延迟服务化，smoke 5/5，Parse% 100%。

证据状态：A。

证据位置：

| 论点 | 证据位置 |
|---|---|
| KV Cache 实现 | `microlm/model/attention.py`、`scripts/benchmark_kvcache.py` |
| 平均加速 3.859x | `results/kvcache_benchmark.csv` |
| REPL | `scripts/chat.py` |
| vLLM 部署 | `docs/vllm_deploy.md`、`reports/vllm_benchmark_report.md`、`reports/vllm_server_wsl_no_flashinfer.log` |
| smoke 5/5 | `results/vllm_benchmark/smoke_results_qwen_structured.json` |
| Parse% 100 | `results/vllm_benchmark_schema_strict/stability_summary_20260521_211457.csv`、`results/qwen_valid_eval_200/summary.json` |

推荐回答：

> KV Cache 的加速来自避免每一步解码都重复计算历史 token 的 K/V。benchmark 覆盖不同 prompt_len 和 gen_len，一共 20 组，平均 speedup 是 3.859x，长上下文长生成时最高到 9.08x。部署侧我把合并后的 Qwen LoRA 导出为 vLLM 可加载格式，使用 OpenAI-compatible API 做服务化，再用 smoke 和 schema-strict benchmark 检查 JSON 可解析率、严格匹配率和延迟。

常见追问：

| 问题 | 回答 |
|---|---|
| KV Cache 为什么越长越有效？ | 没有 cache 时每生成一个 token 都要重算前面所有 token 的 attention；有 cache 后历史 K/V 只算一次，越长的上下文节省越明显。 |
| smoke 5/5 说明什么？ | 说明服务链路、模型加载、请求格式和基础输出可用；不是最终质量指标。最终质量要看 F1 和 schema benchmark。 |
| Parse% 100 是不是代表答案都对？ | 不是。Parse% 只代表 JSON 语法合法，字段是否正确要看 Field F1、Pair F1、Strict 或 Alias-Strict。 |
| vLLM 的价值是什么？ | 服务化、连续批处理、KV cache 管理、OpenAI-compatible API、较低延迟和更高吞吐，适合从实验脚本走向在线推理。 |

### 4.5 自动评测和结构化验证

简历表述：

> 构建 Base Qwen vs LoRA Qwen 自动评测脚本，使用 4 模型 × 40 Prompt × 4 指标进行结构化验证，在 1000 条验证样本上字段级 JSON F1 从 0.057 提升至 0.541，约提升 9.4 倍，实现训练-评测-部署闭环。

证据状态：A/C。自动评测框架有证据；`1000 条`、`Field JSON F1 0.057 -> 0.541`、`9.4 倍` 当前没有被正式结果文件直接支持。

可直接支撑的点：

| 论点 | 证据位置 |
|---|---|
| 4 模型对比 | `scripts/run_instructie_eval.py`、`results/instructie_eval/summary/leaderboard.json` |
| 40 prompt | `eval/prompts_instructie.json` |
| Base Qwen vs LoRA Qwen | `results/instructie_eval/summary/leaderboard.json` |
| 结构化质量分析 | `results/instructie_eval/summary/structural_quality.json` |
| 200 条 valid JSONL content-level F1 | `results/qwen_valid_eval_200/summary.json` |
| schema-strict vLLM benchmark | `results/vllm_benchmark_schema_strict/stability_summary_20260521_211457.csv` |

当前可证明的正式指标：

| 实验 | 样本 | 结果 |
|---|---:|---|
| fixed prompt leaderboard | 4 模型 × 40 prompts | qwen_base Parse 100%、Strict 10.0%、Alias-Strict 7.5%；qwen_lora Parse 97.5%、Strict 7.5%、Alias-Strict 15.0% |
| structural quality | 4 模型 × 40 prompts | qwen_lora entity-key rate 95.0%、Chinese fields 92.5%、avg field overlap 0.49 |
| valid JSONL content-level eval | 200 samples | Parse 100%、Direct JSON 100%、Exact Match 20.0%、Field F1 0.784、Pair F1 0.673 |
| vLLM schema-strict | 40 prompts | Parse 100%、Strict 52.5%、Alias-Strict 52.5%、Projected Strict 75.0% |
| hardcase original | 17 samples | Pair F1 0.0575，但这是 hardcase pair-level，不是 1000 条 field-level |

推荐改写：

> 构建 Base Qwen、LoRA Qwen、MicroLM SFT、MicroLM LoRA 四模型自动评测脚本，在 40 条固定结构化 Prompt 上评估 Parse、Strict、Alias-Strict、缺失率与幻觉率；LoRA Qwen 的 Alias-Strict 达 15.0%，为 Base Qwen 的 2 倍。在 200 条 held-out valid JSONL 上进行字段级内容评测，Parse 100%、Field F1 0.784、Pair F1 0.673；进一步通过 schema-strict vLLM 约束将 40 prompt 的 Strict / Alias-Strict 提升到 52.5%，完成训练-评测-部署闭环。

如果必须保留 `1000 条` 和 `0.057 -> 0.541`：

1. 需要补一个结果目录，例如 `results/qwen_valid_eval_1000/summary.json`。
2. 需要明确 baseline 是谁：Base Qwen、MicroLM、未约束输出、还是 hardcase 原始输出。
3. 需要明确指标定义：Field F1、Pair F1、Strict、Alias-Strict 不能混用。
4. 需要保留完整命令、输入数据 hash、采样 seed、模型路径和输出 JSONL。

常见追问：

| 问题 | 回答 |
|---|---|
| 为什么 qwen_lora Strict 低于 qwen_base，但你还说 LoRA 有价值？ | Strict 对字段名非常敏感。LoRA 学到了 InstructIE 风格，倾向实体中心嵌套 JSON，字段别名归一化后 Alias-Strict 是 base 的 2 倍，结构化行为更符合训练数据。 |
| 40 prompt 和 200 valid JSONL 是不是同一套评测？ | 不是。40 prompt 是固定 prompt 行为评测，用来比较模型输出风格；200 valid JSONL 是内容级抽取评测，用 Field F1 / Pair F1 衡量字段和值的对齐。 |
| Field F1 和 Pair F1 区别？ | Field F1 看字段覆盖，Pair F1 看字段-值键值对是否同时正确。Pair F1 更严格。 |
| Alias-Strict 是什么？ | 对语义等价字段名做归一化后再判断结构是否匹配，例如 `姓名` 和 `name` 可以映射为同一字段。 |
| schema-strict 为什么能提升 Strict？ | 它通过响应格式和后处理约束模型输出必须符合 schema，减少语法正确但结构不对的情况。 |

## 5. 推荐简历版本

下面这版尽量只使用当前仓库能证明的数字：

> 基于 Qwen2.5-1.5B 的结构化信息抽取 LoRA 后训练与评测，2026 年 1 月 - 2026 年 4 月  
> - 基于 PyTorch 搭建 31.7M 参数 MicroLM 训练闭环，完成 MiniMind 数据清洗、SHA1 稳定切分、6400 词 BPE tokenizer 训练、预训练、SFT baseline、LoRA SFT 微调与 CUDA 推理测试，预训练 loss 从 8.x 降至 2.x 区间。  
> - 面向 Qwen 迁移链路处理 InstructIE 171K 条样本，完成标准化、过滤、分层、任务派生、采样、chat JSONL 转换 6 步数据 pipeline，生成约 28.5K 训练样本与 1.5K 验证样本，训练标签 JSON 校验 100%。  
> - 基于 8 层 RoPE + SwiGLU Transformer 实现 MicroLM，并迁移到 1.55B 参数 Qwen2.5-1.5B-Instruct；使用 HF / PEFT / vLLM，MicroLM LoRA 仅训练约 0.83% 参数、adaptor 约 1MB，Qwen LoRA 仅训练约 0.14% 参数、adaptor 约 8.3MB；Qwen LoRA 训练 2000 steps，验证 loss 从 0.4025 降至 0.1553。  
> - 集成 KV Cache 并在 20 组 benchmark 上实现平均 3.86x 推理加速，搭建 REPL；导出合并后的 Qwen LoRA 并基于 vLLM 完成 OpenAI-compatible 在线推理部署，smoke 5/5，valid JSONL Parse 100%。  
> - 构建 Base Qwen、LoRA Qwen、MicroLM SFT、MicroLM LoRA 四模型自动评测脚本，在 40 条固定结构化 Prompt 上评估 Parse、Strict、Alias-Strict、缺失率和幻觉率；LoRA Qwen 的 Alias-Strict 达 15.0%，为 Base Qwen 的 2 倍。在 200 条 held-out valid JSONL 上实现 Field F1 0.784、Pair F1 0.673，并通过 schema-strict vLLM 约束将 40 prompt Strict / Alias-Strict 提升到 52.5%。

如果你坚持保留原简历中的 `316M tokens`、`1.115 -> 0.843`、`1000 条 / 0.057 -> 0.541`，先补齐结果文件，再写进最终版本。

## 6. 高频面试问答

### Q1：这个项目最大的技术价值是什么？

推荐回答：

> 价值在于我不是只完成一次微调，而是把 LLM 项目的关键闭环都跑通了：数据治理、tokenizer、训练、LoRA、评测、推理优化和服务化。MicroLM 线证明我理解底层训练机制，Qwen 线证明我能把方法迁移到真实基座模型和结构化抽取任务上。

### Q2：为什么从 MicroLM 迁移到 Qwen？

推荐回答：

> MicroLM 参数量小，适合自研和调试，但它的词表、预训练语料和指令能力不足，结构化 JSON 输出能力弱。固定 prompt 评测里 MicroLM 系列 JSON parse 基本不可用，所以最终任务需要迁移到已有指令能力的 Qwen2.5-1.5B-Instruct。迁移不是否定 MicroLM，而是把 MicroLM 阶段建立的数据、训练、评测方法迁移到更强基座上。

### Q3：为什么选择 Qwen2.5-1.5B？

推荐回答：

> 1.5B 是任务效果、显存成本和本地部署之间的折中。它比 MicroLM 有完整中文能力和指令跟随能力，又比 7B 成本低，能在本地用 FP16 LoRA 完成训练，并能用 vLLM 服务化部署。

### Q4：SFT 的 loss mask 怎么做？

推荐回答：

> 只对 assistant 输出部分计算 loss，user instruction、schema、input 作为条件上下文，不参与监督。MicroLM 和 Qwen 的 chat template 不同，所以 Qwen 线用 prefix 对比法定位 assistant token 区间，避免把 prompt 也当成目标去学。

### Q5：LoRA 学到了什么？

推荐回答：

> 它学到的不是通用知识，而是任务适配：输出 JSON、遵循 InstructIE 风格、把文本中的实体和属性组织成结构化字段。证据是 qwen_lora 在 Alias-Strict 和结构化质量指标上相对 base 有提升，并且 valid JSONL 上 Field F1 / Pair F1 可量化。

### Q6：为什么 Parse% 高但 Strict 不高？

推荐回答：

> Parse% 只说明语法是合法 JSON；Strict 要求字段名、层级、结构和 schema 完全对齐。模型可能输出合法 JSON，但字段名用了中文别名、层级嵌套不同，Strict 就会失败。所以我额外做了 Alias-Strict、Field F1 和 Pair F1 来拆解问题。

### Q7：Field F1 怎么算？

推荐回答：

> 先把预测 JSON 和标签 JSON 展平成字段路径或字段集合，再计算 precision、recall、F1。Field F1 更关注字段是否覆盖；Pair F1 会把字段和值绑定起来，更严格，能衡量抽取内容是否正确。

### Q8：评测脚本如何避免偏向 LoRA？

推荐回答：

> 同一批 prompt、同一套 schema、同一套解码参数、同一套解析和打分逻辑。脚本不会因为模型名字改变评分规则，并且保留 base Qwen、LoRA Qwen、MicroLM SFT、MicroLM LoRA 四个模型作为对照。

### Q9：vLLM 部署时遇到什么问题？

推荐回答：

> 主要是模型导出格式、chat template、CUDA/WSL 环境和 response_format 约束。LoRA 需要先 merge 到基座并导出完整 HF 格式，vLLM 才能稳定加载；部署后还要用 smoke 和 benchmark 检查服务链路、延迟和 JSON 结构质量。

### Q10：schema-strict 和普通 JSON mode 有什么区别？

推荐回答：

> 普通 JSON mode 主要约束输出是 JSON object，但不保证字段符合业务 schema。schema-strict 会进一步约束字段和结构，所以在 40 prompt benchmark 中 Strict / Alias-Strict 明显提升，但它不能替代训练，只是部署侧约束。

### Q11：为什么 hardcase refinement 提升不明显？

推荐回答：

> hardcase 样本数量很少，而且难例可能集中在 schema 歧义、长尾字段和值对齐问题上。小规模 refinement 容易只改变输出风格，未必提升 holdout 内容 F1。这个结论反而说明评测闭环有价值：不是所有微调都会带来真实提升。

### Q12：项目还有哪些不足？

推荐回答：

> 第一，部分简历数字需要更严格的结果文件支撑，比如 1000 条评测和 316M token 统计。第二，LoRA 只做了有限超参，没有完整 rank、target modules、QLoRA、full fine-tune ablation。第三，结构化评测还可以加入 hidden test 和人工抽检，减少 prompt 开发过程带来的偏差。

### Q13：如果继续做，你会怎么优化？

推荐回答：

> 我会先冻结一套 hidden test，再补 rank=4/8/16、target modules、QLoRA 对比；同时把 schema-strict decoding 和训练数据中的 schema repair 结合起来。部署侧会补吞吐、并发、P95/P99 延迟和显存曲线，而不是只看 smoke。

### Q14：项目中你亲自实现了哪些部分？

推荐回答：

> MicroLM 的 Transformer、RoPE attention、SwiGLU block、LoRA wrapper、训练循环、tokenizer 训练流程、数据切分、KV Cache benchmark、评测脚本和报告整理都是项目内实现或整合的。Qwen 阶段我主要做 HF/PEFT 训练适配、InstructIE 数据 pipeline、模型导出、vLLM 部署和结构化评测闭环。

### Q15：怎么解释项目周期是 2026 年 1 月到 4 月，但文档里有 5 月产物？

推荐回答：

> 核心训练和评测工作集中在 1 月到 4 月；5 月的内容主要是项目收口、报告整理、部署验证和恢复文档补全。如果简历严格按产物日期写，也可以把时间改成 2026 年 1 月到 5 月，避免被追问时显得口径不一致。

## 7. 面试时不要混用的概念

| 容易混用 | 正确区分 |
|---|---|
| MiniMind 和 InstructIE | MiniMind 用于 MicroLM 预训练/SFT 基础链路；InstructIE 用于 Qwen 结构化抽取迁移 |
| MicroLM 和 Qwen | MicroLM 证明底层训练能力；Qwen 承担最终结构化任务效果和部署 |
| Parse% 和 F1 | Parse% 是 JSON 语法合法；F1 是字段和值内容正确 |
| Strict 和 Alias-Strict | Strict 要求字段完全一致；Alias-Strict 允许语义等价字段名归一化 |
| 4×40 prompt 评测和 valid JSONL F1 | 前者是固定 prompt 行为评测；后者是 held-out 样本内容评测 |
| trainable parameter ratio 和 adaptor size | 0.83%/0.14% 是参数比例；1MB/8.3MB 是文件大小 |
| smoke 和 benchmark | smoke 验证链路可用；benchmark 才衡量质量、延迟和稳定性 |

## 8. 需要补证据的清单

如果要让简历原文每个数字都能“被文件证明”，建议补以下产物：

| 缺口 | 建议文件 | 内容 |
|---|---|---|
| `316M tokens` | `reports/token_count_pretrain_full.json` | tokenizer、语料版本、样本数、总 token、统计命令、时间 |
| `1.115 -> 0.843` | `outputs/qwen_lora_xxx/train_log.jsonl` | 对应实验的 step、train_loss、val_loss、config |
| `1000 条验证样本` | `results/qwen_valid_eval_1000/summary.json` | sample_count=1000、seed、model_path、data_path、Field F1、Pair F1 |
| `0.057 -> 0.541` | `results/base_vs_lora_f1_compare/summary.json` | baseline、candidate、指标定义、提升倍数 |
| 项目时间 | `reports/project_timeline.md` | 1-4 月核心工作、5 月整理/部署验证的边界 |

## 9. 最终回答策略

1. 先讲项目闭环，不先堆数字。
2. 被问数字时，只讲当前能指到文件的数字。
3. 主动区分 MicroLM 和 Qwen 两条线。
4. 主动区分数据标签合法、模型输出可解析、字段内容正确。
5. 被问不足时坦诚：有些实验可以继续补 hidden test、ablation 和 1000 条评测。
6. 遇到简历原文里不稳的数字，不硬解释，改成“我会以仓库正式日志为准”。

