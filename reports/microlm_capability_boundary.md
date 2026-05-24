# MicroLM 自研主线能力边界收口报告

生成日期：2026-05-21

## 1. 结论摘要

自研 MicroLM 主线已经完成从原始语料到预训练、全参 SFT、LoRA SFT、固定 prompt 评测和结构化评测的闭环。它的价值不是追求绝对效果，而是证明完整训练链路可运行、可复现、可分析。

当前结论很清楚：

- MicroLM 能学到基础中文续写、简单问答、概念解释和一部分指令格式。
- SFT 后模型明显获得对话意识，输出从预训练阶段的符号/续写噪声变成可读中文。
- LoRA 能以很小的可训练参数量获得同方向能力，但稳定性和指令遵循弱于全参 SFT。
- 31M 级别模型在长输出、精确事实、多步推理、严格 JSON 输出上达到明显上限。
- 结构化输出评测中 MicroLM SFT / LoRA 的 JSON Parse% 均为 0%，这是迁移到 Qwen 主线的直接证据。

因此，自研主线应在这里收口；后续产品化或结构化输出能力应以 Qwen 迁移主线为主。

## 2. 已完成产物

| 阶段 | 内容 | 主要产物 |
|---|---|---|
| A1-A4 | 预训练语料清洗、tokenizer 样本、BPE tokenizer、全文 token IDs | `data/pretrain_clean/`, `data/tokenized_full/` |
| B1 | MicroLM 预训练 | `outputs/pretrain_full_corpus/ckpt_final.pt` |
| B2 | 全参 SFT | `outputs/sft_baseline/ckpt_final.pt`, `outputs/sft_valid_0p5_step1000/ckpt_final.pt` |
| B3 | LoRA SFT | `outputs/sft_lora/lora_adaptor.pt`, `outputs/sft_lora/ckpt_final.pt` |
| Eval | 固定 prompt 评测与结构化评测 | `results/lora_vs_full_sft/`, `results/instructie_eval/` |

## 3. 训练指标

| 模型 / 阶段 | 训练步数 | train_loss | val_loss | 说明 |
|---|---:|---:|---:|---|
| Pretrain | 49,999 | 2.8635 | 2.6933 | 最终日志点，预训练 loss 波动较大 |
| SFT baseline extended | 3,000 | 1.4866 | 2.2003 | 全参 SFT 延长训练后最佳的当前主产物 |
| SFT baseline 0.5% valid | 1,000 | 3.0380 | 2.2248 | 更合理的验证集比例实验 |
| SFT LoRA | 1,000 | 3.2157 | 2.3032 | 参数高效微调，略弱于全参 |

解读：

- SFT baseline 的验证 loss 低于 LoRA，符合“全参微调上限更高”的预期。
- LoRA loss 高一些，但训练方向一致，说明参数高效微调链路是有效的。
- 继续增加 MicroLM 训练步数可以小幅降低 loss，但不能根本解决模型容量带来的长输出和 JSON 能力问题。

## 4. 固定 Prompt 评测结果

评测集包含 13 条固定 prompt，覆盖基础问答、中文表达、指令遵循、多轮对话和续写。统一采样参数下，三个模型都生成到 128 token：

| 模型 | prompt 数 | 平均延迟 | 平均输出 token |
|---|---:|---:|---:|
| pretrain | 13 | 0.616s | 128.0 |
| SFT baseline | 13 | 0.568s | 128.0 |
| SFT LoRA | 13 | 0.571s | 128.0 |

能力变化：

- Pretrain：主要表现为续写式输出，常出现符号噪声，基本没有对话格式意识。
- SFT baseline：能回答简单问题，能用中文解释概念，也能生成列表或段落。
- SFT LoRA：概念解释类任务可读性较好，但格式遵循和稳定性更弱。

代表性样例：

- “地球上最大的海洋是什么？”  
  SFT baseline / LoRA 都能在开头回答“太平洋”，说明基础问答能力已经出现；但后续会混入错误事实和无关内容。

- “请简单解释什么是人工智能。”  
  SFT baseline 和 LoRA 都能给出较自然的中文解释，并提到机器学习、自然语言处理、计算机视觉等概念。

- “请用 JSON 格式输出三个中国城市的名称。”  
  SFT baseline 会输出城市列表但不是合法 JSON；LoRA 会产生格式噪声，不能满足结构化输出要求。

- “春天的早晨，阳光洒在小村庄的屋顶上，”  
  SFT baseline 能接出较自然的开头，但很快漂移到无关任务；LoRA 也会转向咖啡推荐等无关内容。

## 5. 结构化输出能力边界

结构化评测结果：

| 排名 | 模型 | Parse% | Strict% | Alias-Strict% | Missing Rate | Hallucination Rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | qwen_base | 100.0% | 10.0% | 7.5% | 82.5% | 65.0% |
| 2 | qwen_lora | 97.5% | 7.5% | 15.0% | 80.0% | 67.5% |
| 3 | microlm_sft | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| 4 | microlm_lora | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |

这里最关键的不是 Qwen 和 MicroLM 谁高一点，而是量级差异：

- Qwen 至少能稳定输出可解析 JSON。
- MicroLM SFT / LoRA 完全不能稳定输出合法 JSON。
- MicroLM 的失败不是简单 prompt 不够好，而是小词表、小模型容量、缺少结构化预训练和 schema-guided 数据共同造成的能力边界。

这也是项目从自研 MicroLM 转向 Qwen 迁移主线的核心理由。

## 6. 能力边界表

| 能做到 | 做不好 | 当前做不到 |
|---|---|---|
| 简单问答、短中文解释、短续写开头、少量列表格式 | 长输出稳定性、事实准确性、精确翻译、多轮一致性、格式严格遵循 | 严格 JSON、复杂 schema、可靠信息抽取、多步结构化推理 |

典型失败模式：

- repetition loop：输出 64-128 token 后进入重复片段。
- topic drift：开头相关，后半段跳到其他任务或无关主题。
- fact drift：能说出关键词，但后续事实迅速失真。
- format drift：能理解“JSON/列表”的意图，但无法维持合法结构。

## 7. 工程判断

MicroLM 自研主线已经达到它最重要的目标：完整证明了从数据、tokenizer、Transformer、pretrain、SFT、LoRA 到评测的训练链路。继续投入训练可以改善局部 loss，但不会改变它在结构化输出上的硬上限。

后续建议：

1. 自研 MicroLM 主线冻结为“原理验证与能力边界展示”。
2. 项目交付主线切到 Qwen 迁移、评测和部署闭环。
3. 对外展示时，把 MicroLM 的 Parse%=0% 作为迁移决策证据，而不是失败结果。
4. 如果继续研究 MicroLM，只适合做模型结构实验、KV Cache、LoRA 注入、训练框架教学，不适合作为结构化输出生产模型。

## 8. 关联文件

- 固定 prompt 原始输出：`results/lora_vs_full_sft/eval_results.json`
- 结构化评测榜单：`results/instructie_eval/summary/leaderboard.json`
- Qwen valid JSONL 评测：`results/qwen_valid_eval_200/summary.json`
- 训练终端记录：`reports/terminal_outputs_microlm_boundary.md`
- 自研主线说明：`Readme/项目全景图/02-自研 MicroLM 主线.md`
