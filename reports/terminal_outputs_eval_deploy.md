
## 2026-05-21 20:22:16 - Read evaluation/deployment roadmap

~~~powershell
Get-Content -Encoding utf8 "Readme\项目全景图\05-评测、验证与部署闭环.md"
~~~

~~~text
---
type: project-note
project: MicroLM
section: eval-deploy-loop
---

# 五、评测、验证与部署

> 这一章负责证明项目结果是可信的、可复现的、可部署的。所有"证据"和"结果"尽量集中在这里。训练产出 checkpoint，评测证明 checkpoint 有价值，部署让价值可被使用。

---

## 5.1 通用生成评测体系

### 这一小节要解决什么问题

"我感觉模型变好了"不是证据。需要建立一套可复现的统一评测流程——同一组 prompt、同一套采样参数、同一个随机种子，让不同模型之间的差异可比。

### 核心设计 / 核心流程

**两套 prompt 集，覆盖从基础到扩展的梯度：**

| 集合  | 路径                           | 规模           | 覆盖维度                           |
| --- | ---------------------------- | ------------ | ------------------------------ |
| 基础集 | `eval/prompts_baseline.json` | **13 条**，5 类 | 基础问答 / 中文表达 / 指令遵循 / 多轮对话 / 续写 |
| 扩展集 | `eval/prompts_v1.json`       | **40 条**，5 类 | 在基础集上增加难度梯度（easy/medium/hard）  |

每条 prompt 包含：prompt 文本、类别标签、预期能力维度。

**统一推理参数：**

| 参数 | 值 | 理由 |
|------|-----|------|
| temperature | 0.8 | 适度随机，避免 greedy 导致的多样性不足 |
| top_p | 0.9 | nucleus sampling，过滤低概率 token |
| max_new_tokens | 128 | 足够生成完整回答但不至于过长 |
| seed | 42 | 可复现 |

**为什么要统一 temperature、top-p、seed：** 如果不同模型用不同的采样参数，评分差异可能来自参数而非模型能力。统一参数后，唯一变量是模型本身。

**评测对象与评分方式：** 对 MicroLM 自研链路的三个阶段模型做统一对比。人工 4 维度评分（满分 5）：相关性 / 流畅性 / 准确性 / 完整性。

### 关键结果 / 数据

| 模型 | 平均分 | vs Pretrain | 核心特征 |
|------|--------|-------------|----------|
| pretrain | 1.13 | 基准 | 能续写但无对话意识，经常跑题 |
| baseline (SFT) | 2.04 | **+81%** | 指令跟随明显改善，续写质量最高 |
| lora (LoRA) | 1.73 | **+53%** | 以 **0.83% 参数**达到全参 **85%** 效果 |

关键发现：
- SFT 使平均评分提升 81%，证明 pretrain→SFT 的训练范式在这个规模上有效
- LoRA 以极小的参数代价（262K / 31.7M）达到了全参 SFT 85% 的生成质量，验证了低秩适配在微型 LLM 上的有效性
- Baseline 擅长指令遵循和续写任务，lora 在概念解释类 prompt 上表现更好——说明 LoRA 和全参微调学到的"偏好"略有差异，但整体能力处于同一量级

### 这一节的结论

固定 prompt 评测体系的价值在于**建立了可比较的基线**。SFT 证明了训练范式有效（+81%），LoRA 证明了参数效率可行（0.83% 参数达到 85% 效果）。这个基线在后续 Qwen 迁移线上也被保留——只是评测指标从人工打分变成了自动检测。

---

## 5.2 结构化输出自动评测

### 这一小节要解决什么问题

通用评测依赖人工打分，适合评估"聊天体验好不好"。但 Qwen 迁移线聚焦的是结构化信息抽取，需要的是**完全自动化、零人工干预的硬指标检测**——JSON 是不是合法的、字段齐不齐、有没有幻觉字段。

### 核心设计 / 核心流程

**三组 Prompt 设计**——`eval/prompts_instructie.json` 共 **40 条**：

| 组别 | 数量 | 内容 | 难度设计 |
|------|------|------|----------|
| Extraction（抽取类）| 18 条 | 给定文本 + schema，要求抽取实体/属性/事件并以 JSON 输出 | 覆盖 12 个主题，含 easy/medium/hard |
| Schema Constraint（Schema 约束类）| 12 条 | 给定枚举值、必填字段等约束条件 | 测试模型对约束的理解力 |
| Format Following（格式遵循类）| 10 条 | 纯格式约束：必须输出 JSON / 禁止额外解释 / 禁止 markdown | 测试格式控制的严格程度 |

每条 prompt 附带完整的元数据：instruction + schema + input + schema_def + gold_output。gold_output 用于后续可选的人工抽检，但四项主检测指标全部自动计算。

**四项自动检测指标：**

| 检测项 | 定义 | 自动化方式 |
|--------|------|-----------|
| **JSON 可解析率** | 清洗 markdown fence 后能否被 json.loads() 解析 | 直接 try/except |
| **缺字段率** | schema 中必填字段未出现的比例 | 字段存在性集合运算 |
| **幻觉字段率** | 输出了 schema 中不存在的字段 | 多余字段集合运算 |
| **严格 Schema 命中率** | 前三项全部通过 + 枚举值约束满足 | 综合布尔判定 |

**为什么不能用人工评分：** 结构化输出的正确性是客观的——JSON 要么能解析要么不能，字段要么齐全要么缺失。人工评分在这里不仅不必要，还会引入主观偏差。四项自动检测指标让"模型变好了"从一个感受变成可复现的数字。

**推理参数与通用评测不同：** temperature=0.0（greedy）、max_new_tokens=256、seed=42。用 greedy 是因为要测试的是模型"能不能"输出正确格式，而不是"有多大的概率"输出正确格式。

### 关键结果 / 数据

**四模型对比结果：**

| 指标            | qwen_base (1.55B) | qwen_lora (1.55B+LoRA) | microlm_sft (31.7M) | microlm_lora (31.7M+LoRA) |
| ------------- | ----------------- | ---------------------- | ------------------- | ------------------------- |
| JSON 可解析率     | **100.0%**        | 97.5%                  | 0.0%                | 0.0%                      |
| 缺字段率          | 82.5%             | **80.0%**              | 100.0%              | 100.0%                    |
| 幻觉字段率         | 65.0%             | 67.5%                  | 0.0%                | 0.0%                      |
| 严格 Schema 命中率 | **10.0%**         | 7.5%                   | 0.0%                | 0.0%                      |

MicroLM 系列 Parse%=0% 再次确认：小词表 + 无结构化预训练 = 不具备 JSON 输出能力。后续部署候选只在两个 Qwen 版本之间选择。

### 这一节的结论

结构化自动评测的核心价值是**消除了主观性**。四项指标从不同维度刻画模型的输出质量：Parse% 衡量基本格式能力，缺字段率和幻觉字段率衡量内容准确性，Strict% 是综合判定。MicroLM Parse%=0% 是最硬的证据——自研小模型不具备结构化输出能力，必须迁移到更大的基座。

---

## 5.3 Alias 归一化与结构化质量分析

### 这一小节要解决什么问题

主指标（Strict%）下 qwen_lora（7.5%）略低于 qwen_base（10.0%），但这个数字没有反映完整故事。问题在于：模型可能输出了语义正确但字段名不同的 JSON（如用"姓名"而非 schema 要求的"name"）。需要更深层的分析来揭示 LoRA 的真实价值。

### 核心设计 / 核心流程

**为什么主指标不足以解释结果：** Strict% 要求字段名与 schema 完全匹配（包括中文 vs 英文、近义词等）。但实际应用中，"姓名"和 "name"、"位于"和 "location" 语义相同——严格的字符匹配会低估模型的真实能力。

**Alias 归一化评测怎么做：** 构建近义词映射表（如 {"姓名": "name", "位于": "location", ...}），将模型输出中的字段名归一化后再重新计算 Strict%。这相当于给模型一个"语义容错"——只要字段含义正确就算对。

**结构化质量指标：** 除了"是否通过"，还测量输出的**结构化行为质量**：

| 指标 | 定义 |
|------|------|
| 实体做 key 率 | 输出 JSON 中以实体名称作为顶层 key 的比例 |
| 全中文字段率 | 字段名为纯中文（非英文混合）的比例 |
| 字段名重叠率 | 输出字段名与 schema 字段名的集合重叠程度 |

这三项指标衡量的是模型是否学到了 InstructIE 数据集的结构化风格——实体中心、中文优先、schema-aligned。

### 关键结果 / 数据

**Alias 归一化结果：**

| 指标                     | qwen_base | qwen_lora | 变化                    |
| ---------------------- | --------- | --------- | --------------------- |
| Alias-Strict%          | 7.5%      | **15.0%** | **lora 是 base 的 2 倍** |
| 缺字段率                   | 75.0%     | **60.0%** | lora 改善 15pp          |
| extraction 组新命中        | 0%        | **11.1%** | base 完全未命中            |
| schema_constraint 组新命中 | 0%        | **8.3%**  | base 完全未命中            |

**结构化质量指标：**

| 指标 | qwen_base | qwen_lora | 变化 |
|------|-----------|-----------|------|
| 实体做 key 率 | 57.5% | **95.0%** | **+37.5pp** |
| 全中文字段率 | 55.0% | **92.5%** | **+37.5pp** |
| 字段名重叠率 | 4.5% | **49.0%** | **+44.5pp** |

### 这一节的结论

Alias 归一化和结构化质量分析揭示了主指标背后的完整故事：**LoRA 的价值不在精确字段名对齐，而在结构化行为塑形**。qwen_lora 学会了更接近 InstructIE 数据集风格的输出——实体做 key（+37.5pp）、中文字段名（+37.5pp）、schema-aligned（+44.5pp）。Alias-Strict 15.0%（base 的 2 倍）是最终推荐 qwen_lora 部署的核心依据。

> [!note] "Entity nesting" 格式解释
> qwen_base 倾向于输出扁平格式 JSON：`{"位于": "广东省深圳市", "创办者": "任正非"}`
> qwen_lora 倾向于输出实体嵌套格式：`{"华为技术有限公司": {"位于": "广东省深圳市"}, "深圳市": {"位于": "广东省"}}`
> 这种嵌套格式在 format_following 组的严格检测中会"失败"（因为缺少顶层 `创办者` 字段），但它反映了 LoRA 确实学到了 InstructIE 数据集的结构化行为模式。这就是为什么 lora 的主指标略低但 alias 归一化后反而领先。

综合三个维度的分析，**推荐 qwen_lora 作为最终部署版本**：
- Alias-Strict 15.0%（base 的 2 倍）
- 结构化质量全面领先（实体做 key +37.5pp，中文字段 +37.5pp，字段名重叠 +44.5pp）
- 主指标差距仅 2.5pp（10.0% vs 7.5%），且该差距主要来自 format_following 组的格式偏好变化

---

## 5.4 vLLM 部署与 Smoke Test

### 这一小节要解决什么问题

基于评测结论选定 qwen_lora 作为部署版本后，如何将其导出并通过 vLLM 服务化。Smoke test 回答的是最基本的问题：服务能不能启动、能不能对话、能不能做核心任务。

### 核心设计 / 核心流程

**为什么部署选择 qwen_lora：** 基于第 5.3 节的综合分析（Alias-Strict 15.0% vs 7.5%，结构化质量全面领先）。

**导出与服务启动：**

导出（`export_final_model.py`）：
```bash
python scripts/export_final_model.py    # PEFT merge_and_unload → HF 格式
# 输出: outputs/qwen_lora_merged_final/
```

服务启动（`serve_vllm.sh`）：
```bash
bash scripts/serve_vllm.sh [--port 8000] [--max-model-len 4096]
# 默认监听 http://0.0.0.0:8000
# 自动加载 qwen_lora_merged_final/
# 启动前校验模型目录存在性
```

vLLM 0.19.0，OpenAI 兼容 API。支持端口/CPU offload/tensor parallel/max-model-len 等参数自定义。

**Smoke Test（5 项验证）：**

`smoke_vllm.py` 在服务启动后执行最小功能验证：

| # | 测试项 | 验证内容 | 结果 |
|---|--------|----------|------|
| 1 | Health check | `/health` 端点返回正常 | **PASS** |
| 2 | Simple chat | 基础对话 completion 能力 | **PASS** |
| 3 | Structured extraction | 给定 schema 的信息抽取能力 | **PASS** |
| 4 | Multi-turn | 多轮对话上下文保持 | **PASS** |
| 5 | Response format | `response_format=json_object` 约束输出 | **PASS** |

### 关键结果 / 数据

**5/5 全部通过。** 这五项覆盖了服务可用性的最基本要求：能响应、能对话、能做核心任务（结构化抽取）、能维持上下文、能接受格式约束。

### 这一节的结论

Smoke test 5/5 通过意味着从"checkpoint 文件"到"可用 HTTP API"的路径完全打通。vLLM 的 OpenAI 兼容接口意味着任何 HTTP 客户端都可以直接调用——不需要写专门的客户端代码。这是项目从"实验脚本"到"可部署服务"的关键一步。

---

## 5.5 性能 Benchmark 与稳定性验证

### 这一小节要解决什么问题

Smoke test 只回答了"能用不能用"。Benchmark 回答的是"用得怎么样"——TTFT、吞吐量、并发表现、以及服务化环境下结构化输出是否稳定（不退化）。

### 核心设计 / 核心流程

**单并发 Benchmark**——`bench_vllm_local.py` 在 RTX 5070 Ti + vLLM 0.19.0 上完成实测：

| 配置 | Input | Output | TTFT(s) | **Tok/s Mean** |
|------|-------|--------|---------|-------------|
| 短序列 | 128 | 64 | 0.298 | **32.19** |
| 中序列 | 512 | 128 | 0.434 | **29.40** |
| 长序列 | 1024 | 256 | 1.280 | **30.01** |

单并发吞吐在三种输入长度下稳定在 **29~32 tok/s**，随输入长度变化不大（-8.7% ~ +6.8%），说明 vLLM 的 decode 阶段效率稳定，主要差异来自 prefill 开销。TTFT 与输入长度近似线性关系（128→0.3s, 512→0.43s, 1024→1.28s）。

**多并发 Benchmark：**

| 并发数 | Tok/s/req | 总请求 | 错误率 |
|--------|-----------|--------|-------|
| 1 (基准) | ~30 | — | — |
| 4 | **23.0** | 12 | **0** |
| 8 | **15.3** | 24 | **0** |

并发从 1→8 时，总系统吞吐从 ~30 提升到 **~122 tok/s**（+37%），但单请求吞吐因排队等待下降。零错误率表明 vLLM 的连续 batching 在当前负载下运行稳定。

**稳定性验证**——`check_structured_stability.py` 在 vLLM 服务上执行两轮对比：

| 轮次 | 模式 | Parse% | Strict% | Alias-Strict% | Avg Latency(s) |
|------|------|--------|---------|---------------|----------------|
| Round 1 | Normal completion | **100.0%** (40/40) | 0.0% | 0.0% | 1.143 |
| Round 2 | Constrained (`response_format=json_object`) | **100.0% (40/40) | 0.0% | 0.0% | **0.819** |
| 6C 离线参考 | Direct transformers | 97.5% (39/40) | 7.5% | 15.0% | 0.519 |

**三个关键发现：**

**Parse% 不降反升**（97.5% → 100%，+2.5pp）。vLLM 使用 PagedAttention 和优化后的 attention kernel，数值更稳定；merged 模型权重与离线一致；greedy 解码在两种路径下确定性相同。服务化部署未引入格式退化。

**Strict%/Alias-Strict% 出现数值差异**（6C: 7.5%/15.0% → vLLM: 0%/0%）。原因：(1) 6C 评测使用 `best_adaptor`（val_loss 最优 checkpoint），6D 导出使用 `adaptor_final`（step 2000 最终版本），两者存在微小差异；(2) vLLM 的 BF16/FP16 量化推理与 transformers FP16 直推的浮点运算顺序不同，少数 token 的 tie-breaking 结果不同。但这不代表能力退化——两轮 80 个样本全部输出为**实体嵌套 JSON 格式**，结构化行为塑形效果完整保留。

**Constrained 模式显著加速**（-28.4% 延迟）。`response_format=json_object` 约束使模型更快收敛到 JSON 输出（1.143s → 0.819s），对在线服务场景有实际价值。

### 关键结果 / 数据

**服务化 vs 脚本式推理对比：**

| 维度   | 脚本式推理 (6C)    | vLLM 服务化 (6D)                                    |
| ---- | ------------- | ------------------------------------------------ |
| 调用方式 | Python 直接加载模型 | HTTP API (OpenAI 兼容)                             |
| 并发支持 | 单请求串行         | 内置连续 batching，多并发原生支持                            |
| 部署形态 | 脚本/交互式        | 常驻服务进程                                           |
| 可复用性 | 需写代码集成        | 任意 HTTP 客户端可直接调用                                 |
| 演示价值 | 终端内展示         | 可对接前端 / 其他服务                                     |
| 性能优化 | 无（CPU/GPU 直跑） | PagedAttention, continuous batching, CUDA graphs |
| 显存管理 | 手动控制          | 自动 KV Cache 管理                                   |
| 适用场景 | 开发调试、离线评测     | 生产部署、在线服务、演示                                     |

### 这一节的结论

从数据到部署的完整验证链条：

```
InstructIE 原始数据 (171K)
  → 6步 pipeline → 28.5K SFT 数据集     ← 数据质量可控
  → Qwen LoRA 微调 → val_loss 0.1534      ← 训练效果可追踪
  → 4模型 × 40prompt × 4指标自动评测      ← 模型能力可量化
  → 推荐 qwen_lora 部署                  ← 部署决策有依据
  → vLLM smoke 5/5 + stability Parse%100% ← 线上表现可验证
  → TTFT/吞吐/P95 benchmark              ← 性能数据可复现
```

每一个环节都有对应的产物和数值，不存在"训练完就结束"的断点。

> [!note] 与 KV Cache benchmark 的关系
> 第 3 大主题中的 KV Cache benchmark（`benchmark_kvcache.py`）测量的是**算法层面**的加速比（no-cache vs cache），在 CPU float32 环境下运行，平均加速 3.86x、最大 9.08x。这里的 vLLM benchmark 测量的是**系统层面**的服务性能（TTFT / 吞吐 / P95 延迟），在 GPU 上运行，单并发 ~30 tok/s。两者互补：前者证明优化思路有效，后者证明部署后实际表现达标。

---

## 相关记录

- [[01-项目总览]] — 项目全局地图，部署链路在整体架构中的位置
- [[03-推理与系统能力增强]] — KV Cache benchmark（算法层加速比，与本节的系统层 benchmark 互补）
- [[04-Qwen 迁移与结构化输出主线]] — 被评测和部署的对象（qwen_lora）

~~~

## 2026-05-21 20:22:34 - List eval/deploy related files

~~~powershell
rg --files eval scripts reports results outputs configs | rg "(eval|benchmark|bench|smoke|stability|alias|score|report|qwen|vllm|kvcache|prompts)"
~~~

~~~text
eval\prompts_v1.json
eval\prompts_instructie.json
eval\prompts_baseline.json
results\kvcache_benchmark.json
results\kvcache_benchmark.csv
configs\tokenize_smoke.json
configs\tokenizer_smoke.json
configs\sft_smoke.json
configs\sft_lora_smoke.json
configs\qwen_lora_structured_smoke.json
configs\qwen_lora_structured.json
configs\qwen_lora_refine_hardcases.json
configs\pretrain_smoke.json
scripts\train_qwen_lora.py
scripts\summarize_eval_results.py
scripts\start_vllm_wsl.ps1
scripts\smoke_vllm.py
scripts\setup_vllm_wsl_admin.ps1
scripts\setup_vllm_ubuntu.sh
scripts\serve_vllm.sh
scripts\run_instructie_eval.py
scripts\run_eval_prompts.py
scripts\evaluate_qwen_valid_jsonl.py
scripts\download_c0_qwen.py
scripts\check_structured_stability.py
scripts\build_qwen_refine_data.py
scripts\bench_vllm_local.py
scripts\benchmark_kvcache.py
results\vllm_benchmark\stability_summary_20260521_184306.csv
results\vllm_benchmark\stability_summary_20260412_204357.csv
results\vllm_benchmark\stability_20260521_184306.json
results\vllm_benchmark\stability_20260412_204357.json
results\vllm_benchmark\smoke_results_qwen_structured.json
results\vllm_benchmark\smoke_results_current.json
results\vllm_benchmark\smoke_results.json
results\vllm_benchmark\benchmark_summary_20260521_183849.csv
results\vllm_benchmark\benchmark_summary_20260412_204210.csv
results\vllm_benchmark\benchmark_20260521_183849.json
results\vllm_benchmark\benchmark_20260412_204210.json
results\instructie_eval_sanity\summary\leaderboard.json
results\instructie_eval_sanity\summary\detailed.csv
results\instructie_eval_sanity\summary\by_model.json
results\instructie_eval_sanity\summary\by_group.json
outputs\tokenizer_smoke\vocab.json
outputs\tokenizer_smoke\merge.txt
results\instructie_eval_sanity\scored_results\qwen_lora_scored.json
results\instructie_eval_sanity\scored_results\qwen_base_scored.json
results\qwen_valid_eval_smoke\summary.json
reports\vllm_wsl_setup.log
results\qwen_valid_eval_smoke\results.jsonl
reports\vllm_wsl_appx_install.log
results\qwen_valid_eval_smoke\failure_samples.jsonl
reports\vllm_wsl_admin_install.log
reports\vllm_server_wsl_no_flashinfer.log
reports\vllm_server_wsl.log
reports\vllm_benchmark_report.md
reports\terminal_outputs_vllm_env.md
reports\terminal_outputs_recovery.md
reports\terminal_outputs_qwen_structured.md
reports\terminal_outputs_next_step.md
reports\terminal_outputs_microlm_boundary.md
reports\terminal_outputs_inference_system.md
reports\terminal_outputs_eval_deploy.md
reports\terminal_outputs.md
reports\start_loss_dashboard.cmd
reports\start_b1_pretrain.js
reports\spotcheck_samples.jsonl
reports\sft_valid_0p5_step1000.log
reports\sft_valid_0p5_step1000.exitcode
reports\sft_candidate_report.md
reports\sample_report.json
reports\run_c1_pipeline.ps1
reports\run_b3_sft_lora.ps1
reports\run_b1_pretrain.cmd
results\instructie_eval_qwen\summary\leaderboard.json
reports\qwen_refine_train.log
reports\qwen_refine_train.exitcode
reports\qwen_refine_export_best.log
results\instructie_eval_qwen\summary\detailed.csv
reports\qwen_refine_export_best.exitcode
results\instructie_eval_qwen\summary\by_model.json
reports\qwen_refine_comparison_summary.log
results\instructie_eval_qwen\summary\by_group.json
reports\qwen_refine_compare_eval.log
reports\qwen_refine_compare_eval.exitcode
reports\qwen_refine_build_data.log
reports\qwen_refine_build_data.exitcode
reports\qwen_migration_structured_closure.md
reports\quality_report.json
reports\normalize_report.json
reports\microlm_capability_boundary.md
reports\live_loss_server.js
reports\inference_system_closure.md
reports\generate_text_smoke_conversation.json
reports\filter_report.json
reports\eval_summary.log
reports\eval_qwen_valid_smoke.log
reports\eval_qwen_valid_smoke.exitcode
reports\eval_qwen_valid_200.log
reports\eval_qwen_valid_200.exitcode
reports\eval_instructie_qwen_utf8.log
reports\eval_instructie_qwen_utf8.exitcode
reports\eval_instructie_qwen.log
reports\eval_instructie_qwen.exitcode
reports\derive_report.json
reports\d2_verify.log
reports\d2_export_final_model.log
reports\d2_export_final_model.exitcode
reports\d1_verify.log
reports\d1_qwen_lora.stdout.log
reports\d1_qwen_lora.stderr.log
reports\d1_qwen_lora.pid
reports\d1_qwen_lora.exitcode
reports\chat_smoke_session_utf8.jsonl
reports\chat_smoke_input.txt
reports\c1_verify.log
reports\c1_instructie_pipeline.log
reports\c1_instructie_pipeline.exitcode
reports\c0_verify.log
reports\c0_download_qwen.log
reports\c0_download_qwen.exitcode
reports\b3_sft_lora_0p5.pid
reports\b3_sft_lora_0p5.log
reports\b3_sft_lora_0p5.exitcode
reports\b3_sft_lora.log
reports\b3_sft_lora.exitcode
reports\b2_sft_workers_probe.log
reports\b2_sft_workers_probe.exitcode
reports\b2_sft_combined.log
reports\b2_sft.exitcode
reports\b1_pretrain_stdout.log
reports\b1_pretrain_stderr.log
reports\b1_pretrain_status.txt
reports\b1_pretrain_combined.log
reports\b1_pretrain.pid
reports\b1_pretrain.exitcode
reports\a1_a2_a3_terminal_log.md
results\instructie_eval_sanity\raw_outputs\qwen_lora.json
results\instructie_eval_sanity\raw_outputs\qwen_base.json
outputs\qwen_lora_refine\train_log.jsonl
outputs\qwen_lora_refine\resolved_config.json
results\instructie_eval_qwen\scored_results\qwen_lora_scored.json
results\instructie_eval_qwen\scored_results\qwen_base_scored.json
results\qwen_valid_eval_200\summary.json
results\qwen_valid_eval_200\results.jsonl
results\qwen_valid_eval_200\failure_samples.jsonl
results\lora_vs_full_sft\eval_results.json
results\instructie_eval_qwen\raw_outputs\qwen_lora.json
results\instructie_eval_qwen\raw_outputs\qwen_base.json
outputs\qwen_lora_refine\ckpt_step_300\README.md
outputs\qwen_lora_refine\ckpt_step_300\adapter_model.safetensors
outputs\qwen_lora_refine\ckpt_step_300\adapter_config.json
outputs\qwen_lora_merged_final\tokenizer_config.json
outputs\qwen_lora_merged_final\tokenizer.json
outputs\qwen_lora_merged_final\model.safetensors
outputs\qwen_lora_merged_final\generation_config.json
outputs\qwen_lora_merged_final\export_metadata.json
outputs\qwen_lora_merged_final\config.json
outputs\qwen_lora_merged_final\chat_template.jinja
results\instructie_eval\summary\structural_quality.json
results\instructie_eval\summary\leaderboard.json
results\instructie_eval\summary\detailed.csv
results\instructie_eval\summary\by_model.json
results\instructie_eval\summary\by_group.json
outputs\qwen_lora_refine\ckpt_step_200\README.md
outputs\qwen_lora_refine\ckpt_step_200\adapter_model.safetensors
outputs\qwen_lora_refine\ckpt_step_200\adapter_config.json
results\qwen_refine_compare\refined_holdout\summary.json
results\qwen_refine_compare\refined_holdout\results.jsonl
results\qwen_refine_compare\refined_holdout\failure_samples.jsonl
outputs\tokenized_smoke_serial\valid_ids.npy
outputs\tokenized_smoke_serial\train_ids.npy
outputs\tokenized_smoke_serial\metadata.json
outputs\qwen_lora\train_log.jsonl
outputs\qwen_lora\resolved_config.json
results\instructie_eval\scored_results\qwen_lora_scored.json
results\instructie_eval\scored_results\qwen_base_scored.json
results\instructie_eval\scored_results\microlm_sft_scored.json
results\instructie_eval\scored_results\microlm_lora_scored.json
outputs\qwen_lora_refine\ckpt_step_100\README.md
outputs\qwen_lora_refine\ckpt_step_100\adapter_model.safetensors
outputs\qwen_lora_refine\ckpt_step_100\adapter_config.json
results\qwen_refine_compare\refined_hardcases\summary.json
results\qwen_refine_compare\refined_hardcases\results.jsonl
results\qwen_refine_compare\refined_hardcases\failure_samples.jsonl
outputs\qwen_lora\ckpt_step_500\README.md
outputs\qwen_lora\ckpt_step_500\adapter_model.safetensors
outputs\qwen_lora\ckpt_step_500\adapter_config.json
outputs\qwen_lora\ckpt_step_1500\README.md
outputs\qwen_lora\ckpt_step_1500\adapter_model.safetensors
outputs\qwen_lora\ckpt_step_1500\adapter_config.json
results\instructie_eval\raw_outputs\qwen_lora.json
results\instructie_eval\raw_outputs\qwen_base.json
results\instructie_eval\raw_outputs\microlm_sft.json
results\instructie_eval\raw_outputs\microlm_lora.json
outputs\tokenized_smoke\valid_ids.npy
outputs\tokenized_smoke\train_ids.npy
outputs\tokenized_smoke\metadata.json
outputs\qwen_lora\ckpt_step_2000\README.md
outputs\qwen_lora\ckpt_step_2000\adapter_model.safetensors
outputs\qwen_lora\ckpt_step_2000\adapter_config.json
results\qwen_refine_compare\original_holdout\failure_samples.jsonl
outputs\qwen_lora_refine\best_adaptor\README.md
outputs\qwen_lora_refine\best_adaptor\adapter_model.safetensors
outputs\qwen_lora_refine\best_adaptor\adapter_config.json
results\qwen_refine_compare\original_holdout\summary.json
results\qwen_refine_compare\original_holdout\results.jsonl
outputs\qwen_lora\ckpt_step_1000\README.md
outputs\qwen_lora\ckpt_step_1000\adapter_model.safetensors
outputs\qwen_lora\ckpt_step_1000\adapter_config.json
results\qwen_refine_compare\original_hardcases\summary.json
results\qwen_refine_compare\original_hardcases\results.jsonl
results\qwen_refine_compare\original_hardcases\failure_samples.jsonl
outputs\qwen_lora_refine\adaptor_final\README.md
outputs\qwen_lora_refine\adaptor_final\adapter_model.safetensors
outputs\qwen_lora_refine\adaptor_final\adapter_config.json
outputs\qwen_lora\adaptor_final\README.md
outputs\qwen_lora\adaptor_final\adapter_model.safetensors
outputs\qwen_lora\adaptor_final\adapter_config.json
outputs\qwen_lora\best_adaptor\adapter_model.safetensors
outputs\qwen_lora\best_adaptor\adapter_config.json
outputs\qwen_lora\best_adaptor\README.md
outputs\pretrain_smoke\resolved_train_config.json
outputs\pretrain_smoke\model_config.json
outputs\pretrain_smoke\ckpt_final.pt
outputs\qwen_lora_merged_refined_best\tokenizer_config.json
outputs\qwen_lora_merged_refined_best\tokenizer.json
outputs\qwen_lora_merged_refined_best\model.safetensors
outputs\qwen_lora_merged_refined_best\generation_config.json
outputs\qwen_lora_merged_refined_best\export_metadata.json
outputs\qwen_lora_merged_refined_best\config.json
outputs\qwen_lora_merged_refined_best\chat_template.jinja

~~~

## 2026-05-21 20:22:44 - Inventory recent eval/deploy results

~~~powershell
Get-ChildItem reports,results -Recurse -File | Where-Object { $_.Name -match '(eval|deploy|benchmark|stability|instructie|summary)' } | Select-Object FullName,Length,LastWriteTime | Sort-Object LastWriteTime -Descending | Select-Object -First 80 | Format-Table -AutoSize
~~~

~~~text

FullName                                                                Length LastWriteTime     
--------                                                                ------ -------------     
E:\MicroLM\reports\terminal_outputs_eval_deploy.md                       25250 2026/5/21 20:22:34
E:\MicroLM\reports\vllm_benchmark_report.md                               4228 2026/5/21 18:44:17
E:\MicroLM\results\vllm_benchmark\stability_summary_20260521_184306.csv    384 2026/5/21 18:43:06
E:\MicroLM\results\vllm_benchmark\stability_20260521_184306.json          1872 2026/5/21 18:43:06
E:\MicroLM\results\vllm_benchmark\benchmark_20260521_183849.json          2640 2026/5/21 18:38:49
E:\MicroLM\results\vllm_benchmark\benchmark_summary_20260521_183849.csv    505 2026/5/21 18:38:49
E:\MicroLM\reports\qwen_refine_comparison_summary.log                     1268 2026/5/20 23:49:56
E:\MicroLM\reports\qwen_refine_compare_eval.exitcode                         6 2026/5/20 23:49:18
E:\MicroLM\reports\qwen_refine_compare_eval.log                          63281 2026/5/20 23:49:18
E:\MicroLM\results\qwen_refine_compare\refined_holdout\summary.json       1269 2026/5/20 23:49:18
E:\MicroLM\results\qwen_refine_compare\original_holdout\summary.json      1263 2026/5/20 23:45:51
E:\MicroLM\results\qwen_refine_compare\refined_hardcases\summary.json     1081 2026/5/20 23:42:27
E:\MicroLM\results\qwen_refine_compare\original_hardcases\summary.json    1074 2026/5/20 23:42:09
E:\MicroLM\reports\eval_summary.log                                       7213 2026/5/20 23:28:02
E:\MicroLM\reports\eval_qwen_valid_200.exitcode                              6 2026/5/20 23:27:13
E:\MicroLM\reports\eval_qwen_valid_200.log                               26738 2026/5/20 23:27:13
E:\MicroLM\results\qwen_valid_eval_200\summary.json                       1271 2026/5/20 23:27:13
E:\MicroLM\reports\eval_qwen_valid_smoke.exitcode                            6 2026/5/20 23:23:14
E:\MicroLM\reports\eval_qwen_valid_smoke.log                              2712 2026/5/20 23:23:13
E:\MicroLM\results\qwen_valid_eval_smoke\summary.json                     1217 2026/5/20 23:23:13
E:\MicroLM\reports\eval_instructie_qwen_utf8.exitcode                        6 2026/5/20 23:21:14
E:\MicroLM\reports\eval_instructie_qwen_utf8.log                         13964 2026/5/20 23:21:14
E:\MicroLM\reports\eval_instructie_qwen.exitcode                             6 2026/5/20 23:18:57
E:\MicroLM\reports\eval_instructie_qwen.log                               1220 2026/5/20 23:18:56
E:\MicroLM\reports\c1_instructie_pipeline.exitcode                           6 2026/5/20 20:57:35
E:\MicroLM\reports\c1_instructie_pipeline.log                            11676 2026/5/20 20:57:35
E:\MicroLM\results\vllm_benchmark\benchmark_summary_20260412_204210.csv    500 2026/5/19 17:26:29
E:\MicroLM\results\vllm_benchmark\stability_20260412_204357.json          1900 2026/5/19 17:26:29
E:\MicroLM\results\vllm_benchmark\stability_summary_20260412_204357.csv    384 2026/5/19 17:26:29
E:\MicroLM\results\vllm_benchmark\benchmark_20260412_204210.json          2635 2026/5/19 17:26:29
E:\MicroLM\results\kvcache_benchmark.csv                                  1151 2026/5/19 17:26:29
E:\MicroLM\results\kvcache_benchmark.json                                 6380 2026/5/19 17:26:29
E:\MicroLM\results\lora_vs_full_sft\eval_results.json                    29298 2026/5/19 17:26:29



~~~

## 2026-05-21 20:23:05 - Inspect eval script help

~~~powershell
.venv\Scripts\python.exe scripts\run_eval_prompts.py --help
.venv\Scripts\python.exe scripts\summarize_eval_results.py --help
.venv\Scripts\python.exe scripts\run_instructie_eval.py --help
.venv\Scripts\python.exe scripts\evaluate_qwen_valid_jsonl.py --help
~~~

~~~text
usage: run_eval_prompts.py [-h] [--eval-file EVAL_FILE] --models NAME=PATH
                           [NAME=PATH ...] [--out-dir OUT_DIR]
                           [--vocab-path VOCAB_PATH]
                           [--merges-path MERGES_PATH]
                           [--special-token SPECIAL_TOKENS]
                           [--eos-token EOS_TOKEN] [--device DEVICE]
                           [--dtype {float32,float16,bfloat16}]
                           [--lora-adaptor LORA_ADAPTOR]

Run eval prompts against multiple checkpoints.

options:
  -h, --help            show this help message and exit
  --eval-file EVAL_FILE
  --models NAME=PATH [NAME=PATH ...]
                        Model checkpoints as name=path pairs.
  --out-dir OUT_DIR
  --vocab-path VOCAB_PATH
  --merges-path MERGES_PATH
  --special-token SPECIAL_TOKENS
  --eos-token EOS_TOKEN
  --device DEVICE
  --dtype {float32,float16,bfloat16}
  --lora-adaptor LORA_ADAPTOR
                        Path to a lora_adaptor.pt to apply to the 'lora'
                        model.
PROMPT_EVAL_BY_MODEL
{
  "qwen_base": {
    "total": 40,
    "parseable": 40,
    "parseable_rate": 1.0,
    "missing_fields_count": 61,
    "missing_rate": 0.825,
    "extra_fields_count": 103,
    "hallucination_rate": 0.65,
    "schema_strict_count": 4,
    "schema_strict_rate": 0.1,
    "schema_strict_alias_rate": 0.075,
    "missing_alias_rate": 0.75,
    "hallucination_alias_rate": 0.75,
    "total_time_s": 73.78,
    "avg_latency_s": 1.845
  },
  "qwen_lora": {
    "total": 40,
    "parseable": 39,
    "parseable_rate": 0.975,
    "missing_fields_count": 45,
    "missing_rate": 0.8,
    "extra_fields_count": 68,
    "hallucination_rate": 0.675,
    "schema_strict_count": 3,
    "schema_strict_rate": 0.075,
    "schema_strict_alias_rate": 0.15,
    "missing_alias_rate": 0.6,
    "hallucination_alias_rate": 0.55,
    "total_time_s": 39.12,
    "avg_latency_s": 0.978
  }
}

PROMPT_EVAL_BY_GROUP
{
  "qwen_base": {
    "extraction": {
      "total": 18,
      "parseable_rate": 1.0,
      "missing_rate": 0.8889,
      "hallucination_rate": 1.0,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.0,
      "missing_alias_rate": 0.7778
    },
    "schema_constraint": {
      "total": 12,
      "parseable_rate": 1.0,
      "missing_rate": 0.9167,
      "hallucination_rate": 0.6667,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.0,
      "missing_alias_rate": 0.8333
    },
    "format_following": {
      "total": 10,
      "parseable_rate": 1.0,
      "missing_rate": 0.6,
      "hallucination_rate": 0.0,
      "schema_strict_rate": 0.4,
      "schema_strict_alias_rate": 0.3,
      "missing_alias_rate": 0.6
    }
  },
  "qwen_lora": {
    "extraction": {
      "total": 18,
      "parseable_rate": 1.0,
      "missing_rate": 0.7778,
      "hallucination_rate": 1.0,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.1111,
      "missing_alias_rate": 0.5
    },
    "schema_constraint": {
      "total": 12,
      "parseable_rate": 1.0,
      "missing_rate": 0.9167,
      "hallucination_rate": 0.6667,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.0833,
      "missing_alias_rate": 0.6667
    },
    "format_following": {
      "total": 10,
      "parseable_rate": 0.9,
      "missing_rate": 0.7,
      "hallucination_rate": 0.1,
      "schema_strict_rate": 0.3,
      "schema_strict_alias_rate": 0.3,
      "missing_alias_rate": 0.7
    }
  }
}

VALID_200_SUMMARY
{
  "model_path": "outputs\\qwen_lora_merged_final",
  "data_path": "data\\sft_candidate\\valid.jsonl",
  "sample_count": 200,
  "seed": 42,
  "max_new_tokens": 256,
  "temperature": 0.0,
  "top_p": 1.0,
  "parseable_rate": 1.0,
  "direct_json_rate": 1.0,
  "markdown_fence_rate": 0.0,
  "exact_match_rate": 0.2,
  "avg_field_precision": 0.8749384920634917,
  "avg_field_recall": 0.7343706709956709,
  "avg_field_f1": 0.7839667314969176,
  "avg_pair_precision": 0.7796785714285712,
  "avg_pair_recall": 0.6237500000000001,
  "avg_pair_f1": 0.6731100288600288,
  "avg_latency_sec": 1.031116888523102,
  "by_task": {
    "format_following": {
      "total": 26,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.11538461538461539
    },
    "ie_extraction": {
      "total": 100,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.14
    },
    "schema_repair": {
      "total": 17,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.9411764705882353
    },
    "text_to_json": {
      "total": 57,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.12280701754385964
    }
  }
}

VALID_FAILURE_SAMPLES_SAVED=17

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_012087 task=ie_extraction topic=组织 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"当今日报": {"创办者": ["詹德兰", "颜重庆"], "位于": "马来西亚"}, "大马": {"位于": "马来西亚"}}
gold={"当今大马": {"位于": "马来西亚", "创办者": ["颜重庆", "詹德兰"], "成立时间": "1999年11月20日"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_format_following_022524 task=format_following topic=事件 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"1053年－1054年的麦地那之围": {"参与者": ["拜占庭帝国", "穆斯林"], "发生地点": "马耳他岛上的穆斯林城市麦地那(Medina，今马耳他的姆迪纳)"}}
gold={"麦地那之围": {"参与者": ["马耳他岛上的穆斯林城市麦地那", "拜占庭帝国"], "发生时间": "1053年－1054年", "发生地点": "马耳他岛"}, "麦地那": {"别名": ["Medina", "姆迪纳"]}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_000847 task=ie_extraction topic=事件 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"2011年4月宫城地震": {"发生地点": "日本宫城县东部海域", "伤亡人数": "4人死亡、至少141人受伤", "发生时间": "7日深夜"}}
gold={"宫城地震": {"发生时间": ["2011年4月", "7日深夜"], "发生地点": "日本宫城县东部海域", "伤亡人数": "4人死亡、至少141人受伤"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_001370 task=ie_extraction topic=人物 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"耶律阿保机": {"出生日期": "872年", "别名": "耶律亿", "死亡日期": "926年9月6日", "职务": "大契丹国的第一位皇帝"}}
gold={"辽太祖耶律阿保机": {"别名": ["安巴坚", "耶律亿"], "配偶": "萧氏", "出生日期": "872年", "死亡日期": "926年9月6日"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_012551 task=ie_extraction topic=自然科学 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"可分离变量的偏微分方程": {"用途": "求解"}}
gold={"分离变量法": {"用途": "偏微分方程"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_text_to_json_018166 task=text_to_json topic=地理地区 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"南大西洋诸岛省": {"行政中心": "乌斯怀亚", "位于": "阿根廷"}, "阿根廷": {"位于": "南美洲"}}
gold={"火地省": {"别名": ["fueguino", "Fin del Mundo"], "位于": "阿根廷", "行政中心": "乌斯怀亚"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_text_to_json_017694 task=text_to_json topic=医学 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"反射学": {"别名": "反射疗法"}, "反射疗法": {"别名": "区带疗法"}}
gold={"反射学/反射疗法": {"别名": ["脚底按摩", "区带疗法"], "疗法": ["替代疗法", "脚底按摩"]}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_013248 task=ie_extraction topic=自然科学 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"历史语言学": {"组成": "波浪模型"}}
gold={"波浪模型": {"别名": ["波理论", "波模型"], "用途": "语言变化的模型"}}
usage: run_instructie_eval.py [-h] [--eval-file EVAL_FILE] [--out-dir OUT_DIR]
                              [--device DEVICE] [--skip-microlm] [--skip-qwen]
                              [--qwen-base-path QWEN_BASE_PATH]
                              [--qwen-adaptor-path QWEN_ADAPTOR_PATH]
                              [--microlm-sft-path MICROLM_SFT_PATH]
                              [--microlm-lora-path MICROLM_LORA_PATH]
                              [--microlm-lora-adaptor MICROLM_LORA_ADAPTOR]
                              [--vocab-path VOCAB_PATH]
                              [--merges-path MERGES_PATH] [--limit LIMIT]

Run InstructIE structured evaluation.

options:
  -h, --help            show this help message and exit
  --eval-file EVAL_FILE
  --out-dir OUT_DIR
  --device DEVICE
  --skip-microlm        Skip MicroLM models
  --skip-qwen           Skip Qwen models
  --qwen-base-path QWEN_BASE_PATH
  --qwen-adaptor-path QWEN_ADAPTOR_PATH
  --microlm-sft-path MICROLM_SFT_PATH
  --microlm-lora-path MICROLM_LORA_PATH
  --microlm-lora-adaptor MICROLM_LORA_ADAPTOR
  --vocab-path VOCAB_PATH
  --merges-path MERGES_PATH
  --limit LIMIT         Limit number of prompts (0 = all)
usage: evaluate_qwen_valid_jsonl.py [-h] [--model-path MODEL_PATH]
                                    [--data-path DATA_PATH]
                                    [--config-path CONFIG_PATH]
                                    [--out-dir OUT_DIR] [--limit LIMIT]
                                    [--seed SEED] [--device DEVICE]
                                    [--max-new-tokens MAX_NEW_TOKENS]
                                    [--temperature TEMPERATURE]
                                    [--top-p TOP_P]

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
  --data-path DATA_PATH
  --config-path CONFIG_PATH
  --out-dir OUT_DIR
  --limit LIMIT
  --seed SEED
  --device DEVICE
  --max-new-tokens MAX_NEW_TOKENS
  --temperature TEMPERATURE
  --top-p TOP_P

~~~

## 2026-05-21 20:23:23 - Count prompt files

~~~powershell
$files = @('eval\prompts_baseline.json','eval\prompts_v1.json','eval\prompts_instructie.json')
foreach ($f in $files) {
  $json = Get-Content -Raw -Encoding utf8 $f | ConvertFrom-Json
  $count = if ($json -is [array]) { $json.Count } elseif ($json.prompts) { $json.prompts.Count } else { ($json | Get-Member -MemberType NoteProperty).Count }
  [PSCustomObject]@{Path=$f; Count=$count}
}
~~~

~~~text
ConvertFrom-Json : 传入的对象无效，应为“:”或“}”。 (3593): {
  "description": "Extended evaluation prompts v1 for comparing pretrain / SFT baseline / LoRA SFT. Expanded from prompt
s_baseline.json (13 -> 40 prompts, 5 categories)",
  "generation_params": {
    "temperature": 0.8,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "seed": 42
  },
  "prompts": [
    {
      "id": "qa_01",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "中国的首都是哪里？"}
      ]
    },
    {
      "id": "qa_02",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "地球上最大的海洋是什么？"}
      ]
    },
    {
      "id": "qa_03",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "一年有多少天？闰年呢？"}
      ]
    },
    {
      "id": "qa_04",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "水在标准大气压下的沸点是多少度？"}
      ]
    },
    {
      "id": "qa_05",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "太阳系中最大的行星是哪颗？"}
      ]
    },
    {
      "id": "qa_06",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "中国的四大发明是什么？"}
      ]
    },
    {
      "id": "qa_07",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "世界上最高的山峰叫什么名字？"}
      ]
    },
    {
      "id": "qa_08",
      "category": "基础问答",
      "conversations": [
        {"role": "user", "content": "一年有几个月？每个月最多有多少天？"}
      ]
    },
    {
      "id": "expr_01",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "用三句话概括《西游记》的故事。"}
      ]
    },
    {
      "id": "expr_02",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "请简单解释什么是人工智能。"}
      ]
    },
    {
      "id": "expr_03",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "用一句话总结《三国演义》的主题。"}
      ]
    },
    {
      "id": "expr_04",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "用通俗的语言解释什么是光合作用。"}
      ]
    },
    {
      "id": "expr_05",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "请用两句话介绍唐朝的历史地位。"}
      ]
    },
    {
      "id": "expr_06",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "简要概括《红楼梦》的主要故事线索。"}
      ]
    },
    {
      "id": "expr_07",
      "category": "中文表达与总结",
      "conversations": [
        {"role": "user", "content": "用三句话解释什么是云计算。"}
      ]
    },
    {
      "id": "instr_01",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "请列出三种水果的名称。"}
      ]
    },
    {
      "id": "instr_02",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "请用JSON格式输出三个中国城市的名称。"}
      ]
    },
    {
      "id": "instr_03",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "把下面的句子翻译成英文：今天天气很好。"}
      ]
    },
    {
      "id": "instr_04",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "请按从大到小的顺序排列以下数字：3, 17, 8, 42, 1。"}
      ]
    },
    {
      "id": "instr_05",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "请写一首关于秋天的四行诗。"}
      ]
    },
    {
      "id": "instr_06",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "请列出五个以"春"字开头的成语。"}
      ]
    },
    {
      "id": "instr_07",
      "category": "指令遵循",
      "conversations": [
        {"role": "user", "content": "请用三个形容词描述大海。"}
      ]
    },
    {
      "id": "multi_01",
      "category": "多轮对话",
      "conversations": [
        {"role": "user", "content": "你好，请介绍一下你自己。"},
        {"role": "assistant", "content": "你好！我是一个AI助手，可以回答各种问题。"},
        {"role": "user", "content": "那你能做什么？"}
      ]
    },
    {
      "id": "multi_02",
      "category": "多轮对话",
      "conversations": [
        {"role": "user", "content": "李白是哪个朝代的诗人？"},
        {"role": "assistant", "content": "李白是唐朝著名的浪漫主义诗人，被誉为\"诗仙\"。"},
        {"role": "user", "content": "他最有名的诗是哪首？"}
      ]
    },
    {
      "id": "multi_03",
      "category": "多轮对话",
      "conversations": [
        {"role": "user", "content": "我想学习Python编程，有什么建议吗？"},
        {"role": "assistant", "content": "建议从基础语法开始学习，先掌握变量、循环和函数的概念。"},
        {"role": "user", "content": "那有没有推荐的入门书籍？"}
      ]
    },
    {
      "id": "multi_04",
      "category": "多轮对话",
      "conversations": [
        {"role": "user", "content": "北京今天天气怎么样？"},
        {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。建议您查看天气预报应用。"},
        {"role": "user", "content": "那你能告诉我北京一般在几月份最热吗？"}
      ]
    },
    {
      "id": "multi_05",
      "category": "多轮对话",
      "conversations": [
        {"role": "user", "content": "请帮我写一封邮件给老师请假。"},
        {"role": "assistant", "content": "好的，请问您要请假几天？原因是什么？"},
        {"role": "user", "content": "请假两天，因为感冒了需要休息。"}
      ]
    },
    {
      "id": "multi_06",
      "category": "多轮对话",
      "conversations": [
        {"role": "user", "content": "中国有多少个省级行政区？"},
        {"role": "assistant", "content": "中国共有34个省级行政区，包括23个省、5个自治区、4个直辖市和2个特别行政区。"},
        {"role": "user", "content": "其中四个直辖市分别是哪里？"}
      ]
    },
    {
      "id": "cont_01",
      "category": "续写",
      "prompt_text": "春天的早晨，阳光洒在小村庄的屋顶上，"
    },
    {
      "id": "cont_02",
      "category": "续写",
      "prompt_text": "从前有一座山，山上有一座庙，庙里有一个老和尚和一个小和尚。"
    },
    {
      "id": "cont_03",
      "category": "续写",
      "prompt_text": "夜深了，城市的灯火渐渐暗了下来，街道上只剩下"
    },
    {
      "id": "cont_04",
      "category": "续写",
      "prompt_text": "小明走在回家的路上，突然看到路边有一只受伤的小猫，它"
    },
    {
      "id": "cont_05",
      "category": "续写",
      "prompt_text": "秋天到了，树叶变成了金黄色，一阵风吹来，"
    },
    {
      "id": "cont_06",
      "category": "续写",
      "prompt_text": "在一个遥远的王国里，有一位年轻的骑士，他决定"
    },
    {
      "id": "cont_07",
      "category": "续写",
      "prompt_text": "清晨的第一缕阳光穿过窗帘，照在书桌上那本翻开的书上，"
    },
    {
      "id": "cont_08",
      "category": "续写",
      "prompt_text": "站在山顶上，俯瞰脚下的城市，他深深地吸了一口气，"
    }
  ]
}

所在位置 行:3 字符: 48
+   $json = Get-Content -Raw -Encoding utf8 $f | ConvertFrom-Json
+                                                ~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [ConvertFrom-Json], ArgumentException
    + FullyQualifiedErrorId : System.ArgumentException,Microsoft.PowerShell.Commands.ConvertFromJsonCommand
 

~~~

## 2026-05-21 20:23:36 - Inspect invalid prompts_v1 line

~~~powershell
Select-String -Path eval\prompts_v1.json -Pattern '春|prompts|generation_params' -Context 1,1
~~~

~~~text

  eval\prompts_v1.json:1:{
> eval\prompts_v1.json:2:  "description": "Extended evaluation prompts v1 for comparing pretrain / SFT baseline / LoRA SFT. Expanded from prompts_baseline.json (13 -> 40 prompts, 5 categories)",
> eval\prompts_v1.json:3:  "generation_params": {
  eval\prompts_v1.json:4:    "temperature": 0.8,
  eval\prompts_v1.json:8:  },
> eval\prompts_v1.json:9:  "prompts": [
  eval\prompts_v1.json:10:    {
  eval\prompts_v1.json:153:      "conversations": [
> eval\prompts_v1.json:154:        {"role": "user", "content": "请列出五个以"春"字开头的成语。"}
  eval\prompts_v1.json:155:      ]
  eval\prompts_v1.json:220:      "category": "续写",
> eval\prompts_v1.json:221:      "prompt_text": "春天的早晨，阳光洒在小村庄的屋顶上，"
  eval\prompts_v1.json:222:    },



~~~

## 2026-05-21 20:23:52 - Count prompt files after JSON fix

~~~powershell
$files = @('eval\prompts_baseline.json','eval\prompts_v1.json','eval\prompts_instructie.json')
foreach ($f in $files) {
  $json = Get-Content -Raw -Encoding utf8 $f | ConvertFrom-Json
  $count = if ($json -is [array]) { $json.Count } elseif ($json.prompts) { $json.prompts.Count } else { ($json | Get-Member -MemberType NoteProperty).Count }
  [PSCustomObject]@{Path=$f; Count=$count}
}
~~~

~~~text

Path                         Count
----                         -----
eval\prompts_baseline.json      13
eval\prompts_v1.json            36
eval\prompts_instructie.json    40



~~~

## 2026-05-21 20:24:05 - Inspect prompts_v1 category distribution

~~~powershell
$json = Get-Content -Raw -Encoding utf8 eval\prompts_v1.json | ConvertFrom-Json
$json.prompts | Group-Object category | Select-Object Name,Count | Format-Table -AutoSize
$json.prompts | Select-Object id,category | Format-Table -AutoSize
~~~

~~~text

Name    Count
----    -----
基础问答        8
中文表达与总结     7
指令遵循        7
多轮对话        6
续写          8



id       category
--       --------
qa_01    基础问答    
qa_02    基础问答    
qa_03    基础问答    
qa_04    基础问答    
qa_05    基础问答    
qa_06    基础问答    
qa_07    基础问答    
qa_08    基础问答    
expr_01  中文表达与总结 
expr_02  中文表达与总结 
expr_03  中文表达与总结 
expr_04  中文表达与总结 
expr_05  中文表达与总结 
expr_06  中文表达与总结 
expr_07  中文表达与总结 
instr_01 指令遵循    
instr_02 指令遵循    
instr_03 指令遵循    
instr_04 指令遵循    
instr_05 指令遵循    
instr_06 指令遵循    
instr_07 指令遵循    
multi_01 多轮对话    
multi_02 多轮对话    
multi_03 多轮对话    
multi_04 多轮对话    
multi_05 多轮对话    
multi_06 多轮对话    
cont_01  续写      
cont_02  续写      
cont_03  续写      
cont_04  续写      
cont_05  续写      
cont_06  续写      
cont_07  续写      
cont_08  续写      



~~~

## 2026-05-21 20:24:19 - Read prompts_v1 insertion context

~~~powershell
$lines = Get-Content -Encoding utf8 eval\prompts_v1.json
for ($i=120; $i -le [Math]::Min($lines.Count, 230); $i++) { '{0,4}: {1}' -f $i, $lines[$i-1] }
~~~

~~~text
 120:       ]
 121:     },
 122:     {
 123:       "id": "instr_02",
 124:       "category": "指令遵循",
 125:       "conversations": [
 126:         {"role": "user", "content": "请用JSON格式输出三个中国城市的名称。"}
 127:       ]
 128:     },
 129:     {
 130:       "id": "instr_03",
 131:       "category": "指令遵循",
 132:       "conversations": [
 133:         {"role": "user", "content": "把下面的句子翻译成英文：今天天气很好。"}
 134:       ]
 135:     },
 136:     {
 137:       "id": "instr_04",
 138:       "category": "指令遵循",
 139:       "conversations": [
 140:         {"role": "user", "content": "请按从大到小的顺序排列以下数字：3, 17, 8, 42, 1。"}
 141:       ]
 142:     },
 143:     {
 144:       "id": "instr_05",
 145:       "category": "指令遵循",
 146:       "conversations": [
 147:         {"role": "user", "content": "请写一首关于秋天的四行诗。"}
 148:       ]
 149:     },
 150:     {
 151:       "id": "instr_06",
 152:       "category": "指令遵循",
 153:       "conversations": [
 154:         {"role": "user", "content": "请列出五个以\"春\"字开头的成语。"}
 155:       ]
 156:     },
 157:     {
 158:       "id": "instr_07",
 159:       "category": "指令遵循",
 160:       "conversations": [
 161:         {"role": "user", "content": "请用三个形容词描述大海。"}
 162:       ]
 163:     },
 164:     {
 165:       "id": "multi_01",
 166:       "category": "多轮对话",
 167:       "conversations": [
 168:         {"role": "user", "content": "你好，请介绍一下你自己。"},
 169:         {"role": "assistant", "content": "你好！我是一个AI助手，可以回答各种问题。"},
 170:         {"role": "user", "content": "那你能做什么？"}
 171:       ]
 172:     },
 173:     {
 174:       "id": "multi_02",
 175:       "category": "多轮对话",
 176:       "conversations": [
 177:         {"role": "user", "content": "李白是哪个朝代的诗人？"},
 178:         {"role": "assistant", "content": "李白是唐朝著名的浪漫主义诗人，被誉为\"诗仙\"。"},
 179:         {"role": "user", "content": "他最有名的诗是哪首？"}
 180:       ]
 181:     },
 182:     {
 183:       "id": "multi_03",
 184:       "category": "多轮对话",
 185:       "conversations": [
 186:         {"role": "user", "content": "我想学习Python编程，有什么建议吗？"},
 187:         {"role": "assistant", "content": "建议从基础语法开始学习，先掌握变量、循环和函数的概念。"},
 188:         {"role": "user", "content": "那有没有推荐的入门书籍？"}
 189:       ]
 190:     },
 191:     {
 192:       "id": "multi_04",
 193:       "category": "多轮对话",
 194:       "conversations": [
 195:         {"role": "user", "content": "北京今天天气怎么样？"},
 196:         {"role": "assistant", "content": "抱歉，我无法获取实时天气信息。建议您查看天气预报应用。"},
 197:         {"role": "user", "content": "那你能告诉我北京一般在几月份最热吗？"}
 198:       ]
 199:     },
 200:     {
 201:       "id": "multi_05",
 202:       "category": "多轮对话",
 203:       "conversations": [
 204:         {"role": "user", "content": "请帮我写一封邮件给老师请假。"},
 205:         {"role": "assistant", "content": "好的，请问您要请假几天？原因是什么？"},
 206:         {"role": "user", "content": "请假两天，因为感冒了需要休息。"}
 207:       ]
 208:     },
 209:     {
 210:       "id": "multi_06",
 211:       "category": "多轮对话",
 212:       "conversations": [
 213:         {"role": "user", "content": "中国有多少个省级行政区？"},
 214:         {"role": "assistant", "content": "中国共有34个省级行政区，包括23个省、5个自治区、4个直辖市和2个特别行政区。"},
 215:         {"role": "user", "content": "其中四个直辖市分别是哪里？"}
 216:       ]
 217:     },
 218:     {
 219:       "id": "cont_01",
 220:       "category": "续写",
 221:       "prompt_text": "春天的早晨，阳光洒在小村庄的屋顶上，"
 222:     },
 223:     {
 224:       "id": "cont_02",
 225:       "category": "续写",
 226:       "prompt_text": "从前有一座山，山上有一座庙，庙里有一个老和尚和一个小和尚。"
 227:     },
 228:     {
 229:       "id": "cont_03",
 230:       "category": "续写",

~~~

## 2026-05-21 20:24:31 - Read prompts_v1 expression/instruction context

~~~powershell
$lines = Get-Content -Encoding utf8 eval\prompts_v1.json
for ($i=70; $i -le 125; $i++) { '{0,4}: {1}' -f $i, $lines[$i-1] }
~~~

~~~text
  70:         {"role": "user", "content": "用三句话概括《西游记》的故事。"}
  71:       ]
  72:     },
  73:     {
  74:       "id": "expr_02",
  75:       "category": "中文表达与总结",
  76:       "conversations": [
  77:         {"role": "user", "content": "请简单解释什么是人工智能。"}
  78:       ]
  79:     },
  80:     {
  81:       "id": "expr_03",
  82:       "category": "中文表达与总结",
  83:       "conversations": [
  84:         {"role": "user", "content": "用一句话总结《三国演义》的主题。"}
  85:       ]
  86:     },
  87:     {
  88:       "id": "expr_04",
  89:       "category": "中文表达与总结",
  90:       "conversations": [
  91:         {"role": "user", "content": "用通俗的语言解释什么是光合作用。"}
  92:       ]
  93:     },
  94:     {
  95:       "id": "expr_05",
  96:       "category": "中文表达与总结",
  97:       "conversations": [
  98:         {"role": "user", "content": "请用两句话介绍唐朝的历史地位。"}
  99:       ]
 100:     },
 101:     {
 102:       "id": "expr_06",
 103:       "category": "中文表达与总结",
 104:       "conversations": [
 105:         {"role": "user", "content": "简要概括《红楼梦》的主要故事线索。"}
 106:       ]
 107:     },
 108:     {
 109:       "id": "expr_07",
 110:       "category": "中文表达与总结",
 111:       "conversations": [
 112:         {"role": "user", "content": "用三句话解释什么是云计算。"}
 113:       ]
 114:     },
 115:     {
 116:       "id": "instr_01",
 117:       "category": "指令遵循",
 118:       "conversations": [
 119:         {"role": "user", "content": "请列出三种水果的名称。"}
 120:       ]
 121:     },
 122:     {
 123:       "id": "instr_02",
 124:       "category": "指令遵循",
 125:       "conversations": [

~~~

## 2026-05-21 20:25:03 - Validate prompt file counts after completing prompts_v1

~~~powershell
$files = @('eval\prompts_baseline.json','eval\prompts_v1.json','eval\prompts_instructie.json')
foreach ($f in $files) {
  $json = Get-Content -Raw -Encoding utf8 $f | ConvertFrom-Json
  $count = if ($json -is [array]) { $json.Count } elseif ($json.prompts) { $json.prompts.Count } else { ($json | Get-Member -MemberType NoteProperty).Count }
  [PSCustomObject]@{Path=$f; Count=$count}
}
(Get-Content -Raw -Encoding utf8 eval\prompts_v1.json | ConvertFrom-Json).prompts | Group-Object category | Select-Object Name,Count | Format-Table -AutoSize
~~~

~~~text

Path                         Count
----                         -----
eval\prompts_baseline.json      13
eval\prompts_v1.json            40
eval\prompts_instructie.json    40



Name    Count
----    -----
基础问答        8
中文表达与总结     8
指令遵循        8
多轮对话        8
续写          8



~~~

## 2026-05-21 20:25:17 - Check model checkpoint paths for eval

~~~powershell
$paths = @(
 'outputs\pretrain_full_corpus\ckpt_final.pt',
 'outputs\sft_baseline\ckpt_final.pt',
 'outputs\sft_lora\ckpt_final.pt',
 'outputs\sft_lora\lora_adaptor.pt',
 'outputs\pretrain_smoke\ckpt_final.pt',
 'outputs\qwen_lora_merged_final\config.json',
 'outputs\qwen_lora\adaptor_final\adapter_model.safetensors',
 'Qwen2.5-1.5B-Instruct\config.json'
)
$paths | ForEach-Object {
  $item = Get-Item $_ -ErrorAction SilentlyContinue
  [PSCustomObject]@{Path=$_; Exists=[bool]$item; Type=if($item -and $item.PSIsContainer){'dir'}elseif($item){'file'}else{''}; Length=if($item){$item.Length}else{''}; LastWriteTime=if($item){$item.LastWriteTime}else{''}}
} | Format-Table -AutoSize
~~~

~~~text

Path                                                      Exists Type    Length LastWriteTime     
----                                                      ------ ----    ------ -------------     
outputs\pretrain_full_corpus\ckpt_final.pt                  True file 377666290 2026/5/20 0:37:51 
outputs\sft_baseline\ckpt_final.pt                          True file 377666290 2026/5/20 19:45:09
outputs\sft_lora\ckpt_final.pt                              True file 129104092 2026/5/20 20:31:05
outputs\sft_lora\lora_adaptor.pt                            True file   1068993 2026/5/20 20:31:05
outputs\pretrain_smoke\ckpt_final.pt                        True file   5533813 2026/5/19 18:45:09
outputs\qwen_lora_merged_final\config.json                  True file      1433 2026/5/20 23:02:00
outputs\qwen_lora\adaptor_final\adapter_model.safetensors   True file   8745704 2026/5/20 22:54:40
Qwen2.5-1.5B-Instruct\config.json                           True file       660 2026/5/20 20:34:44



~~~

## 2026-05-21 20:25:27 - Read run_eval_prompts.py

~~~powershell
Get-Content -Encoding utf8 scripts\run_eval_prompts.py -TotalCount 240
~~~

~~~text
"""Run fixed evaluation prompts against multiple checkpoints and save results.

Usage:
    python scripts/run_eval_prompts.py \
        --eval-file eval/prompts_baseline.json \
        --models pretrain=outputs/pretrain_full_corpus/ckpt_final.pt \
                 baseline=outputs/sft_baseline/ckpt_final.pt \
                 lora=outputs/sft_lora/ckpt_final.pt \
        --out-dir results/lora_vs_full_sft
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from microlm.model import TransformerLM
from microlm.model.lora import load_lora_state_dict, merge_lora
from microlm.tokenizer import BPETokenizer
from microlm.training import build_generation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval prompts against multiple checkpoints.")
    parser.add_argument("--eval-file", type=Path, default=Path("eval/prompts_baseline.json"))
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="Model checkpoints as name=path pairs.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/lora_vs_full_sft"))
    parser.add_argument("--vocab-path", type=Path, default=Path("outputs/tokenizer_full_clean/vocab.json"))
    parser.add_argument("--merges-path", type=Path, default=Path("outputs/tokenizer_full_clean/merge.txt"))
    parser.add_argument("--special-token", action="append", dest="special_tokens", default=None)
    parser.add_argument("--eos-token", type=str, default="</s>")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--lora-adaptor", type=Path, default=None,
                        help="Path to a lora_adaptor.pt to apply to the 'lora' model.")
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> list[tuple[str, Path]]:
    models = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid model spec '{spec}'. Use NAME=PATH format.")
        name, path = spec.split("=", 1)
        models.append((name, Path(path)))
    return models


def load_model(
    checkpoint_path: Path,
    model_config: dict,
    device: str,
    dtype: torch.dtype,
    lora_adaptor_path: Path | None = None,
) -> TransformerLM:
    import torch.nn as nn

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

    # Detect LoRA checkpoint: keys contain "original.weight"
    is_lora_ckpt = any("original.weight" in k for k in cleaned)
    # Remap LoRA checkpoint keys: strip LoRA wrapper to get plain Linear weights
    if is_lora_ckpt:
        remapped = {}
        for k, v in cleaned.items():
            if k.endswith(".original.weight"):
                remapped[k.replace(".original.weight", ".weight")] = v
            elif ".lora_" in k:
                continue  # skip lora A/B, use adaptor file instead
            else:
                remapped[k] = v
        cleaned = remapped

    # Determine actual vocab_size from checkpoint weights
    ckpt_vocab_size = cleaned.get("token_embeddings.weight").shape[0]

    model = TransformerLM(
        vocab_size=ckpt_vocab_size,
        context_length=int(model_config["context_length"]),
        d_model=int(model_config["d_model"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        d_ff=int(model_config["d_ff"]),
        rope_theta=float(model_config.get("rope_theta", 1000000.0)),
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        device=device,
        dtype=dtype,
    ).to(device)

    model.load_state_dict(cleaned, strict=True)

    # Resize if tokenizer has more tokens than checkpoint
    tokenizer_vocab = model_config.get("_tokenizer_vocab_size", ckpt_vocab_size)
    if tokenizer_vocab > ckpt_vocab_size:
        d_model = int(model_config["d_model"])
        old_emb = model.token_embeddings.weight.data
        new_emb = torch.zeros(tokenizer_vocab, d_model, device=old_emb.device, dtype=old_emb.dtype)
        new_emb[:old_emb.shape[0]] = old_emb
        model.token_embeddings.weight = nn.Parameter(new_emb)

        old_head = model.lm_head.weight.data
        new_head = torch.zeros(tokenizer_vocab, d_model, device=old_head.device, dtype=old_head.dtype)
        new_head[:old_head.shape[0]] = old_head
        model.lm_head.weight = nn.Parameter(new_head)
        print(f"  Resized vocab: {ckpt_vocab_size} -> {tokenizer_vocab}")

    if lora_adaptor_path is not None and lora_adaptor_path.exists():
        from microlm.model.lora import apply_lora_to_model
        apply_lora_to_model(model, r=8, alpha=16.0)
        lora_sd = torch.load(lora_adaptor_path, map_location=device, weights_only=True)
        load_lora_state_dict(model, lora_sd)
        merge_lora(model)
        print(f"  Loaded and merged LoRA adaptor from {lora_adaptor_path}")

    model.eval()
    return model


def generate(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt_text: str,
    eos_token_id: int | None,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> tuple[str, list[int]]:
    token_ids = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        if temperature == 0.0:
            generated = prompt_tensor.clone()
            for _ in range(max_new_tokens):
                logits = model(generated[:, -model.context_length:])[:, -1, :]
                if top_p < 1.0:
                    logits = model._top_p_filter(logits, top_p)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
            new_ids = generated[0].tolist()[len(token_ids):]
        else:
            out = model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_p=top_p,
            )
            new_ids = out[0].tolist()[len(token_ids):]

    return tokenizer.decode(new_ids), new_ids


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    special_tokens = args.special_tokens or [args.eos_token]
    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=special_tokens,
    )

    eos_token_id = tokenizer.vocab_to_id.get(args.eos_token.encode("utf-8"))

    with args.eval_file.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    gen_params = eval_data["generation_params"]
    prompts = eval_data["prompts"]
    temperature = gen_params["temperature"]
    top_p = gen_params["top_p"]
    max_new_tokens = gen_params["max_new_tokens"]

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    model_specs = parse_model_specs(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, checkpoint_path in model_specs:
        if not checkpoint_path.exists():
            print(f"SKIP {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n=== Evaluating: {model_name} ({checkpoint_path}) ===")

        # Try loading model_config.json from the same directory
        config_path = checkpoint_path.parent / "model_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                model_config = json.load(f)
            # Pass tokenizer vocab size for potential resize
            model_config["_tokenizer_vocab_size"] = len(tokenizer.id_to_vocab)
        else:
            raise FileNotFoundError(f"No model_config.json found next to {checkpoint_path}")

        lora_path = args.lora_adaptor if model_name == "lora" else None
        model = load_model(checkpoint_path, model_config, args.device, dtype, lora_path)

        results = []
        total_time = 0.0
        for prompt_item in prompts:
            prompt_id = prompt_item["id"]
            category = prompt_item["category"]

            # Build generation prompt
            if "conversations" in prompt_item:
                prompt_text = build_generation_prompt(
                    prompt_item["conversations"], eos_token=args.eos_token,
                )
            else:
                prompt_text = prompt_item["prompt_text"]

            torch.manual_seed(gen_params["seed"])
            t0 = time.time()
            output_text, output_ids = generate(
                model, tokenizer, prompt_text, eos_token_id,
                max_new_tokens, temperature, top_p, args.device,
            )
            elapsed = time.time() - t0

~~~

## 2026-05-21 20:25:35 - Read run_eval_prompts.py tail

~~~powershell
Get-Content -Encoding utf8 scripts\run_eval_prompts.py -Tail 120
~~~

~~~text
            out = model.generate(
                prompt_ids=prompt_tensor,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=temperature,
                top_p=top_p,
            )
            new_ids = out[0].tolist()[len(token_ids):]

    return tokenizer.decode(new_ids), new_ids


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    special_tokens = args.special_tokens or [args.eos_token]
    tokenizer = BPETokenizer.from_files(
        str(args.vocab_path),
        str(args.merges_path),
        special_tokens=special_tokens,
    )

    eos_token_id = tokenizer.vocab_to_id.get(args.eos_token.encode("utf-8"))

    with args.eval_file.open("r", encoding="utf-8") as f:
        eval_data = json.load(f)

    gen_params = eval_data["generation_params"]
    prompts = eval_data["prompts"]
    temperature = gen_params["temperature"]
    top_p = gen_params["top_p"]
    max_new_tokens = gen_params["max_new_tokens"]

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    model_specs = parse_model_specs(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, checkpoint_path in model_specs:
        if not checkpoint_path.exists():
            print(f"SKIP {model_name}: checkpoint not found at {checkpoint_path}")
            continue

        print(f"\n=== Evaluating: {model_name} ({checkpoint_path}) ===")

        # Try loading model_config.json from the same directory
        config_path = checkpoint_path.parent / "model_config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                model_config = json.load(f)
            # Pass tokenizer vocab size for potential resize
            model_config["_tokenizer_vocab_size"] = len(tokenizer.id_to_vocab)
        else:
            raise FileNotFoundError(f"No model_config.json found next to {checkpoint_path}")

        lora_path = args.lora_adaptor if model_name == "lora" else None
        model = load_model(checkpoint_path, model_config, args.device, dtype, lora_path)

        results = []
        total_time = 0.0
        for prompt_item in prompts:
            prompt_id = prompt_item["id"]
            category = prompt_item["category"]

            # Build generation prompt
            if "conversations" in prompt_item:
                prompt_text = build_generation_prompt(
                    prompt_item["conversations"], eos_token=args.eos_token,
                )
            else:
                prompt_text = prompt_item["prompt_text"]

            torch.manual_seed(gen_params["seed"])
            t0 = time.time()
            output_text, output_ids = generate(
                model, tokenizer, prompt_text, eos_token_id,
                max_new_tokens, temperature, top_p, args.device,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            result = {
                "id": prompt_id,
                "category": category,
                "input": prompt_item.get("conversations") or prompt_item.get("prompt_text"),
                "output": output_text,
                "output_tokens": len(output_ids),
                "latency_s": round(elapsed, 3),
            }
            results.append(result)
            print(f"  [{category}] {prompt_id}: {output_text[:80]}{'...' if len(output_text) > 80 else ''}")

        avg_latency = total_time / len(results) if results else 0
        print(f"  Total: {total_time:.1f}s, Avg: {avg_latency:.2f}s/prompt")

        all_results[model_name] = {
            "checkpoint": str(checkpoint_path),
            "generation_params": gen_params,
            "results": results,
            "total_time_s": round(total_time, 2),
            "avg_latency_s": round(avg_latency, 3),
        }

        # Free GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    out_file = args.out_dir / "eval_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

~~~

## 2026-05-21 20:25:46 - Check MicroLM model config files

~~~powershell
Get-ChildItem outputs\pretrain_full_corpus,outputs\sft_baseline,outputs\sft_lora -Filter model_config.json -Recurse | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
~~~

~~~text

FullName                                                  Length LastWriteTime     
--------                                                  ------ -------------     
E:\MicroLM\outputs\pretrain_full_corpus\model_config.json    228 2026/5/19 22:04:19
E:\MicroLM\outputs\sft_baseline\model_config.json            228 2026/5/20 19:38:35
E:\MicroLM\outputs\sft_lora\model_config.json                228 2026/5/20 20:18:26



~~~

## 2026-05-21 20:26:00 - Inspect existing MicroLM eval results

~~~powershell
$path='results\lora_vs_full_sft\eval_results.json'
if (Test-Path $path) {
  $json = Get-Content -Raw -Encoding utf8 $path | ConvertFrom-Json
  $json.PSObject.Properties | ForEach-Object { [PSCustomObject]@{Model=$_.Name; Prompts=$_.Value.results.Count; AvgLatency=$_.Value.avg_latency_s; TotalTime=$_.Value.total_time_s} } | Format-Table -AutoSize
} else { 'missing' }
~~~

~~~text

Model    Prompts AvgLatency TotalTime
-----    ------- ---------- ---------
pretrain      13      0.616      8.01
baseline      13      0.568      7.38
lora          13      0.571      7.43



~~~

## 2026-05-21 20:26:16 - Read summarize_eval_results.py

~~~powershell
Get-Content -Encoding utf8 scripts\summarize_eval_results.py
~~~

~~~text
#!/usr/bin/env python3
"""Summarize Qwen evaluation artifacts into a compact report."""

from __future__ import annotations

import json
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    prompt_by_model = read_json(Path("results/instructie_eval_qwen/summary/by_model.json"))
    prompt_by_group = read_json(Path("results/instructie_eval_qwen/summary/by_group.json"))
    valid_summary = read_json(Path("results/qwen_valid_eval_200/summary.json"))

    print("PROMPT_EVAL_BY_MODEL")
    print(json.dumps(prompt_by_model, ensure_ascii=False, indent=2))
    print("\nPROMPT_EVAL_BY_GROUP")
    print(json.dumps(prompt_by_group, ensure_ascii=False, indent=2))
    print("\nVALID_200_SUMMARY")
    print(json.dumps(valid_summary, ensure_ascii=False, indent=2))

    failure_path = Path("results/qwen_valid_eval_200/failure_samples.jsonl")
    failures = []
    if failure_path.exists():
        failures = [json.loads(line) for line in failure_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"\nVALID_FAILURE_SAMPLES_SAVED={len(failures)}")
    for row in sorted(failures, key=lambda item: (item["field_f1"], item["pair_f1"]))[:8]:
        print("\n--- LOW_SCORE_SAMPLE ---")
        print(f"id={row['id']} task={row['task_type']} topic={row.get('topic_schema')} parsed={row['parsed']} direct={row['direct_json']}")
        print(f"field_f1={row['field_f1']} pair_f1={row['pair_f1']} exact={row['exact_match']}")
        print(f"raw={row['raw_output'][:500]}")
        print(f"gold={json.dumps(row['gold_output'], ensure_ascii=False)[:500]}")


if __name__ == "__main__":
    main()

~~~

## 2026-05-21 20:26:27 - Read structured eval existing summaries

~~~powershell
Get-Content -Encoding utf8 results\instructie_eval\summary\leaderboard.json
Get-Content -Encoding utf8 results\instructie_eval\summary\structural_quality.json
Get-Content -Encoding utf8 results\instructie_eval_qwen\summary\by_model.json
~~~

~~~text
[
  {
    "rank": 1,
    "model": "qwen_base",
    "schema_strict_rate": 0.1,
    "schema_strict_alias_rate": 0.075,
    "parseable_rate": 1.0,
    "missing_rate": 0.825,
    "hallucination_rate": 0.65
  },
  {
    "rank": 2,
    "model": "qwen_lora",
    "schema_strict_rate": 0.075,
    "schema_strict_alias_rate": 0.15,
    "parseable_rate": 0.975,
    "missing_rate": 0.8,
    "hallucination_rate": 0.675
  },
  {
    "rank": 3,
    "model": "microlm_sft",
    "schema_strict_rate": 0.0,
    "schema_strict_alias_rate": 0.0,
    "parseable_rate": 0.0,
    "missing_rate": 1.0,
    "hallucination_rate": 0.0
  },
  {
    "rank": 4,
    "model": "microlm_lora",
    "schema_strict_rate": 0.0,
    "schema_strict_alias_rate": 0.0,
    "parseable_rate": 0.0,
    "missing_rate": 1.0,
    "hallucination_rate": 0.0
  }
]
{
  "qwen_base": {
    "total": 40,
    "entity_key_rate": 0.575,
    "chinese_fields_rate": 0.55,
    "avg_field_overlap": 0.1697,
    "by_group": {
      "extraction": {
        "entity_key_rate": 0.7222,
        "chinese_fields_rate": 0.5556,
        "avg_field_overlap": 0.037
      },
      "schema_constraint": {
        "entity_key_rate": 0.5,
        "chinese_fields_rate": 0.6667,
        "avg_field_overlap": 0.2183
      },
      "format_following": {
        "entity_key_rate": 0.4,
        "chinese_fields_rate": 0.4,
        "avg_field_overlap": 0.35
      }
    }
  },
  "qwen_lora": {
    "total": 40,
    "entity_key_rate": 0.95,
    "chinese_fields_rate": 0.925,
    "avg_field_overlap": 0.49,
    "by_group": {
      "extraction": {
        "entity_key_rate": 1.0,
        "chinese_fields_rate": 1.0,
        "avg_field_overlap": 0.3435
      },
      "schema_constraint": {
        "entity_key_rate": 1.0,
        "chinese_fields_rate": 0.9167,
        "avg_field_overlap": 0.5347
      },
      "format_following": {
        "entity_key_rate": 0.8,
        "chinese_fields_rate": 0.8,
        "avg_field_overlap": 0.7
      }
    }
  }
}
{
  "qwen_base": {
    "total": 40,
    "parseable": 40,
    "parseable_rate": 1.0,
    "missing_fields_count": 61,
    "missing_rate": 0.825,
    "extra_fields_count": 103,
    "hallucination_rate": 0.65,
    "schema_strict_count": 4,
    "schema_strict_rate": 0.1,
    "schema_strict_alias_rate": 0.075,
    "missing_alias_rate": 0.75,
    "hallucination_alias_rate": 0.75,
    "total_time_s": 73.78,
    "avg_latency_s": 1.845
  },
  "qwen_lora": {
    "total": 40,
    "parseable": 39,
    "parseable_rate": 0.975,
    "missing_fields_count": 45,
    "missing_rate": 0.8,
    "extra_fields_count": 68,
    "hallucination_rate": 0.675,
    "schema_strict_count": 3,
    "schema_strict_rate": 0.075,
    "schema_strict_alias_rate": 0.15,
    "missing_alias_rate": 0.6,
    "hallucination_alias_rate": 0.55,
    "total_time_s": 39.12,
    "avg_latency_s": 0.978
  }
}

~~~

## 2026-05-21 20:29:06 - Run MicroLM 40-prompt general eval

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\run_eval_prompts.py --eval-file eval\prompts_v1.json --models pretrain=outputs\pretrain_full_corpus\ckpt_final.pt baseline=outputs\sft_baseline\ckpt_final.pt lora=outputs\sft_lora\ckpt_final.pt --out-dir results\lora_vs_full_sft_v1 --lora-adaptor outputs\sft_lora\lora_adaptor.pt --device cuda --dtype float16
~~~

~~~text

=== Evaluating: pretrain (outputs\pretrain_full_corpus\ckpt_final.pt) ===
  Resized vocab: 6400 -> 6401
  [基础问答] qa_01: [display] |
<-end
[end
] |
 >>> >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>...
  [基础问答] qa_02: 地球，约
河
约
地球，约
大约
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近
近...
  [基础问答] qa_03: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。2. 打沙滩排球，享受自然的美景。3. 游泳，享受太阳和沙滩。4. ...
  [基础问答] qa_04: 水在标准大气压下的沸点是80度。
水在标准大气压下的沸点是8度。
水在标准大气压下的沸点是1度。
水在标准大气压下的沸点是1度。
水在标准大气压下的沸点是2度。...
  [基础问答] qa_05: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。2. 打沙滩排球，享受户外的刺激和刺激。3. 室内游泳，品尝当地美食...
  [基础问答] qa_06: 中国的四大发明之一，一直持续不断地推动着中国大众的发明。你能给我讲一下中国四大发明吗？当然可以。中国四大发明之一，已成为世界首次发明之一，涵盖了四大发明和发明。...
  [基础问答] qa_07: 世界上最高的山峰叫什么名字？
<|endoftext|>
生成一个描述夏天的句子。夏日炎炎，阳光温暖，蝉鸣声声不断，蝉鸣声不断。人们在夏季会穿上轻便的衣服，享受...
  [基础问答] qa_08: 一年有2月4日。
<|endoftext|>
Recipient the following expression:Sure, please provide m...
  [中文表达与总结] expr_01: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以尝试烤肉、烤肉、烤炉等各种食物。
2. 野餐，也可以去公园野餐，享受清凉的时光。...
  [中文表达与总结] expr_02: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。2. 打沙滩排球，享受海滩活动。3. 游泳，享受沙滩和户外活动。
4...
  [中文表达与总结] expr_03: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以尝试烤肉、烤肉、烤炉等各种食物。
2. 野餐，也可以去公园野餐，享受清凉的时光。...
  [中文表达与总结] expr_04: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以尝试烤肉、烤肉、烤炉等各种食物。
2. 野餐，也可以去公园野餐，享受清凉的时光。...
  [中文表达与总结] expr_05: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以尝试烤肉、烤肉、烤炉等各种菜肴。
2. 野餐，也可以去公园野餐，享受清凉的时光。...
  [中文表达与总结] expr_06: * 贾宝玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* 林黛玉
* ...
  [中文表达与总结] expr_07: 用三句话解释。
<|endoftext|>
你认为RCSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS...
  [中文表达与总结] expr_08: 因为环境不符合环保要求。
<|endoftext|>
Recipient the following expression:Sure, please provi...
  [指令遵循] instr_01: * 苹果
* 香蕉
* 苹果
* 香蕉
* 香蕉
* 橙子
* 橙子
* 香蕉
* 香蕉
* 橙子
* 香蕉
* 橙子
* 橙子
* 橙子
* 香蕉
* 橙子
...
  [指令遵循] instr_02: ```
在该格式输出三个中国城市的名称。```
<|endoftext|>
给定一个单词列表，返回其中所有单词出现的次数。
单词列表: ['apple', 'b...
  [指令遵循] instr_03: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以尝试烤肉、烤肉、烤炉等各种食物。
2. 野餐，可以参加户外运动，例如打篮球、跳绳...
  [指令遵循] instr_04: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。2. 野餐野餐，选择新鲜的食材，享受户外体验。3. 室内游泳，品尝当...
  [指令遵循] instr_05: <|endoftext|>
给我造一个50字的中文名字。好的，这是我为你生成的中文名字：“方南绿草黄花”（英文名：Error age）。请问你是否对“方南绿草黄...
  [指令遵循] instr_06: <|endoftext|>
给我五个单词，这些单词都可以用来形容它们的意思。1. "可爱"2. "惊讶"3. "惊讶"4. "令人不安"5. "非常惊奇"6. ...
  [指令遵循] instr_07: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以尝试烤肉、烤肉、烤炉等各种食物。
2. 野餐，也可以去公园野餐，享受清凉的时光。...
  [指令遵循] instr_08: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用，也要准备食物。2. 野餐，可以野餐，也可以在户外露营，享受美食。3....
  [多轮对话] multi_01: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。
2. 野餐，选择新鲜的露营活动，如野餐和露营。
3. 打篮球，帮助...
  [多轮对话] multi_02: 复数句
<
复数句
Rose:
>
4 春秋
 春秋
 春秋
 春秋
 春秋
 秋
 秋
 秋
 秋
 秋
 秋
 秋
 
 
 
 
 
 
 
 
 
 
...
  [多轮对话] multi_03: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。2. 打沙滩排球，享受户外的刺激和刺激。3. 室内游泳，品尝当地美食...
  [多轮对话] multi_04: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，不要在室外使用。2. 野餐野餐，选择新鲜的露营活动，如野餐和露营。3. 野餐，选择太...
  [多轮对话] multi_05: <|endoftext|>
给我五个适合夏天的户外活动建议。1. 烧烤野餐，可以选择烤肉或者烤鱼，以增强身体机能。
2. 野餐，选择露营场地，享受美食和美景。
...
  [多轮对话] multi_06: 因为不成立，河南省的广东省。好的，那你能告诉我，广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东省广东...
  [多轮对话] multi_07: 因为不吃糖。
您可能需要调整水量。好的，我明白了，谢谢你的解释。那么，如果我想加入鸡蛋，应该怎么做呢？如果您想加入鸡蛋，首先将鸡蛋打散，然后加入鸡蛋，然后加入鸡...
  [多轮对话] multi_08: 人工智能，指代程序执行。
<|endoftext|>
写一篇介绍狗的文章。好的，以下是介绍狗的文章。狗是人类最好的朋友之一，它们忠诚、友好、忠诚、友好和友好。狗...
  [续写] cont_01: 高大的树木在绿色的树木上飘荡。大片的叶子在阳光下绽放，它们如同一道道耀眼的画卷，映衬着夏天的生机和活力。夏天的阳光温暖，让人感到温馨和活力，但它也带来了许多快乐...
  [续写] cont_02: 高大，有一座庙，山上有一个庙，其中有一座庙，上面有一个庙。建于1954年，这座庙是由泰国雕像之一，他对它的建筑和建筑有很高的要求。除了建于1954年，它还有许多...
  [续写] cont_03: 了。给我几个创意，帮我设计一个新的咖啡馆。好的，让我想想。 我建议您可以使用一些特色饮品，比如香浓的啤酒、咖啡的茶道、蒸汽锅、卡茶等等。此外，您还可以在咖啡馆里...
  [续写] cont_04: 开始不受伤。小明一直以为他走了很久，回到家里，看到了一个受伤的小猫，他感到非常的害怕。他轻轻地看着它，在回家的路上，他看到了一只受伤的小猫，它告诉他它高兴起来。...
  [续写] cont_05: 雨不时地冷落。这时，一位老妇人说：“老妇人，我只能照着你，来找你一回。”他轻轻地说：“老妇人，我只好想着你，来找你一回。”老妇人听到老妇人说：“对，我很抱歉。”...
  [续写] cont_06: 踏上那个王国，并要求骑士以帮助他和他回到家。王国又开始了他的冒险，开始了一段冒险之旅。他四处寻找自己的梦想，并与一匹马匹配。随着时间的推移，王国逐渐加深了对他的...
  [续写] cont_07: 我画下了一首悠扬的自由自在的钢琴曲。我看到了一朵春暖的玫瑰，绿草如茵的叶子在飘荡的草地上，我看见了一片细小的树，在阳光的照耀下，感觉到了一股清新的气息，让人感觉...
  [续写] cont_08: 说他从来没有想过这场比赛。
<|endoftext|>
Sharming habits and birds their future downfully fal...
  Total: 42.8s, Avg: 1.07s/prompt

=== Evaluating: baseline (outputs\sft_baseline\ckpt_final.pt) ===
  Resized vocab: 6400 -> 6401
  [基础问答] qa_01: 中国的首都是中国的首都。

1. **中国的首都是**：
   - 中国的首都是北京。

2. **北京**：
   - 北京。

3. **北京**：
   ...
  [基础问答] qa_02: 地球上最大的海洋是约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约约...
  [基础问答] qa_03: 一年有365天，天数为366天。

1. ** 2023年有365天，天数为366天。

2. ** 2023年有365天，天数为366天。

3. **365...
  [基础问答] qa_04: 水在标准大气压下的沸点是80度，70度；

水在标准大气压下的沸点是1度。

水在标准大气压下的沸点是1度；

水在标准大气压下的沸点为1度；

水在标准大气压...
  [基础问答] qa_05: 太阳系中最大的天体是因为它们拥有非常广阔的引力场和大气层。因此，我们无法直接观测到太阳系中的行星。不过，我们可以根据目前的地球科学知识和问题提供详细的解释。

...
  [基础问答] qa_06: 中国的四大发明之一，主要是造纸术。造纸术的发明，是通过造纸术，发明以造纸术的发明。造纸术的发明，通过在纸术上打造，发明了纸术。造纸术的发明，不仅在生产方面取得了...
  [基础问答] qa_07: 世界上最高的山峰叫《静夜》是山峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之巅峰之...
  [基础问答] qa_08: 一年有2个月，每一年有4个月。

1. **3月**：每个月有4个月。

2. **7月**：每个月有6个月。

3. **7月**：每个月有36个月。

**...
  [中文表达与总结] expr_01: 《西游记》的故事主要围绕着西游记展开，其中有些情节充满冒险和冒险。故事的核心在于了四川的四川四川四川四川四川四川四川五川四川四川四川四川四川四川四川四川四川四川...
  [中文表达与总结] expr_02: 人工智能是一种利用算法和模型进行技术处理复杂任务的技术。它旨在实现智能体（如语音识别、图像识别、自然语言处理）在内的各种领域应用。

AI 通过学习用户行为模式...
  [中文表达与总结] expr_03: 《三国演义》中，三国演义涉及中国五国演义、历史背景、政治背景等多种因素。

**回答：**  
1. **三国演义**：三国演义涉及中国五国演义、历史背景、政治...
  [中文表达与总结] expr_04: 光合作用是指利用光能将光能转化为光能的过程。光能被转化成光能的过程包括：1. 光能的利用：光能通过光能将光能转化为光能。光能被转化为光能，并释放光能。2. 光能...
  [中文表达与总结] expr_05: 唐朝历史地位主要涉及历史地位、历史地位、历史地位、历史地位等多个方面。以下是唐朝的历史地位：

1. 朝代：唐朝历史地位主要由其历史地位、历史地位和历史地位决定...
  [中文表达与总结] expr_06: 《红楼梦》的主要情节围绕着贾宝玉和贾宝玉展开展开展开。林黛玉和薛薛之间以了深厚的象征，两人在四川的园林中逐渐形成，并对黛玉和薛薛之间的情感纠葛产生深厚的情感。
...
  [中文表达与总结] expr_07: 云计算是一种通过互联网提供计算的计算能力，以满足各种计算任务的需求，并为用户提供无需人工存储的计算能力。

云计算是一种通过互联网提供计算资源，使用云计算来提供...
  [中文表达与总结] expr_08: 环境是我们生存的重要组成部分。它提供了我们生存的环境，包括空气、水、土壤和水，以及各种不同类型的空气，这些环境对我们的生存至关重要。因此，保护环境对于我们的生存...
  [指令遵循] instr_01: ### 三种水果类型：
- 草莓（柠檬）：
- 橙子（柚子）：
- 柚子（柚子）：
- 西瓜（柚子）：
- 柚子（柚子）：
- 柚子（柚子）：
- 柚子（柚子）...
  [指令遵循] instr_02: JSON格式输出如下：

```

```


```


```


Hello

```


I don't have the same current c...
  [指令遵循] instr_03: This is an accurate response to this.

**B**:   
There are two days versuses of ...
  [指令遵循] instr_04: 按从大到小的顺序排列以下数字：

1. **3, 17, 8, 42**：3, 17, 8, 42**：3, 17, 8, 42**：5, 17, 8, 42*...
  [指令遵循] instr_05: 秋风，落叶，

秋风，落叶，

落叶，


落叶，


落叶，


秋风，



落叶，



叶，


 

 
 
 
  

  

  
  
   
...
  [指令遵循] instr_06: 1. **"春"字开头的成语**：春天气温热，春天气温热，春天气温热，春天气温热，春天气温热，春天气温热，春天气温热，春天气温热，春天气温热，春天气温热，春天气...
  [指令遵循] instr_07: 海，

在海中，


```





 


  
  
   
     
    
         
                         ...
  [指令遵循] instr_08: **************：苹果是苹果的产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产产...
  [多轮对话] multi_01: 当然可以！我是由中国的个人开发者开发的。作为AI助手，我能自主地执行任务。但我以前没有经验的AI，无法体验世界。

如果你有关于AI的问题，我可以帮你分析并解答...
  [多轮对话] multi_02: 
他最有名的诗是《赤霞》。

其中《赤霞》是诗人的著名诗人，《赤霞》是他最有名的诗作。

第九? 诗是《赤霞》诗，《赤霞》是李白。

她是唐朝的诗人，是唐朝的诗...
  [多轮对话] multi_03: 如果你喜欢，可以试试看《论语》哦！

<|endoftext|>
生成一个能让人感觉心情愉悦的对话场景。你来试试Brainstorming吧！Brainstor...
  [多轮对话] multi_04: 记得，今天北京天气晴朗，空气质量也比今天大时大得多。  

**告诉我今天北京的天气怎么样？

答案：北京

**天气**  

**天气**  

**天气*...
  [多轮对话] multi_05: 记得，我该提前几天，请您带上专门的安理工作。

回复：  
**提示：选择回复邮件，不要担心，我该会尽快恢复。

希望这可以帮到您。

若您看到邮件，建议尽快联...
  [多轮对话] multi_06: 首先，从桨县县区出发，大江、河、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、山、...
  [多轮对话] multi_07: 如果你不吃糖，可以先用蒸蛋器清洗，然后用冷蛋器清洗，然后用蒸蛋器清洗，直到油膨胀。糖的量应在锅内加适量，以保持食材的新鲜。

如果你不吃糖，可以先用清水器清洗，...
  [多轮对话] multi_08: 人工智能是让计算机能够执行人类任务的一种方法。

**Scar** 指的是：
- 人工智能是让机器具备自我认知的能力。
- 机器学习是让机器具备类似于人类智能的...
  [续写] cont_01: 高大的树木在远处隐隐约水面，大风轻拂，仿佛在探索这个世界。

秋天，是秋天最美的诗篇，
在风中轻拂，温暖的海风拂过，
在风中，吹拂着树叶，轻摇，
在麦田里，花朵...
  [续写] cont_02: 高，有一位老和尚，但他以前也有一位老和尚，他经常参加各种活动，以帮助老和尚人。张先生是一位经验丰富且充满活力的年轻人，他曾经参与过各种活动，以帮助老和尚和她的女...
  [续写] cont_03: 了。给我几个创意，用来推广我的咖啡馆。好的，以下是几个创意：
1. 通过社交媒体平台，来推广咖啡馆。
2. 利用社交媒体平台，例如Facebook、Twitte...
  [续写] cont_04: 说不理解该怎么办。小明：你突然觉得这很难说，你不要放弃？小兔：也许，我不会放弃。你不要让小猫重新站起来，也不要让小猫迸发。小明：可是我也想学会更多，这样我们才能...
  [续写] cont_05: 雨不时地在雨中跳动。

林黛玉：一个孤独的女孩，总是对家乡充满爱，每一天都会有新的美好回忆。


他们之间的爱情故事，
对彼此的关爱，但每个人都必须在爱情中找到...
  [续写] cont_06: 踏上那个遥远的王国，这个王国的王国里，有着强大的力量和强大的力量。他建造了许多山脉，包括山顶，山脉，山脉，森林，森林，山脉，山脉，山脉，湖泊和山脉。他还建造了山...
  [续写] cont_07: 那不小心在书桌上滑落。

<|endoftext|>
生成商品文案：秋冬装奢华，经典风范，舒适自然，简约风，轻便舒适，细致柔滑，挺括着，随风而来的轻盈，简约风范...
  [续写] cont_08: 说不下历史。

山谷深处，山谷深处，一个孤独的王子，一位年轻的女子，在他的怀抱中，背着一个黑暗的王子，他对着宇宙的深处，但他却有着深厚的历史底蕴，而他却对这个王...
  Total: 40.3s, Avg: 1.01s/prompt

=== Evaluating: lora (outputs\sft_lora\ckpt_final.pt) ===
  Loaded and merged LoRA adaptor from outputs\sft_lora\lora_adaptor.pt
  [基础问答] qa_01: 中国的首都是中国。
中国的首都是北京。

中国的首都是北京。


中国的首都是北京。

中国的首都是北京。

1

1

1
1
1
1
1
1
1
1
1
1...
  [基础问答] qa_02: 地球上最大的海洋之一，随着海洋的开采，大部分海洋被覆盖，成为全球最大的海洋资源之一。海上游的目的地包括：

地球上最大的海洋之一，海上生活着海豹、海豹、鲸鱼、海...
  [基础问答] qa_03: 一年有：

一年有：366天
一年有：366天
一年有：366天
一年有：366天
一年有：366天
一年有：366天
一年有：366天
一年有：366天
一年...
  [基础问答] qa_04: 水在标准大气压下的沸点是80度，每天的盐摄入量为：\[80 度 \times 0.25 = 34 度 \]

where \(d\) is a point, t...
  [基础问答] qa_05: 太阳系中最大的行星是水星。水星是太阳系中最大的天体之一，最深处是最接近太阳的行星。水星以其独特的速度、光线以及能量等因素而著称，形成于太阳系内，最深处最接近太阳...
  [基础问答] qa_06: 四大发明是指在古代中国发明的一种方法。四大发明是通过发明创造的，造就了其发明。造就了其发明，但是又可以造就了这个发明。
四大发明指的是造就了一种无法用的材料，但...
  [基础问答] qa_07: 世界上最高的山峰叫“珠穆朗玛峰”，位于喜马拉雅山脉，海拔8848.488.448.48米。


**Explanation:** 

When was the ...
  [基础问答] qa_08: 一年有七个月，每一年有4个月。每月有36个月，每个月有4个月。

一年有：42个月。

一年有：42个月。

一年有：42个月。

一年有：42个月。

一年有...
  [中文表达与总结] expr_01: 《西游记》的故事。故事发生在三只大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大大...
  [中文表达与总结] expr_02: 人工智能是指利用计算机程序模拟人类智能的一种方法。人工智能涉及多个方面，包括机器学习、深度学习、自然语言处理、计算机视觉等。在人工智能领域，机器学习已经成为许多...
  [中文表达与总结] expr_03: 《三国演义》的主题是“三国演义”。《三国演义》的主题是“三国演义”。

“三国演义》的主题是“三国演义”。


《三国演义》主题围绕着三国演义展开。

一国演义...
  [中文表达与总结] expr_04: 光合作用是指利用光能将光能转化为光能的过程。光能被转化成光能的过程。通过光能的转化过程，光能被转化为光能，并将其转化为光能。同时，光能被转化为光能，并被转化为光...
  [中文表达与总结] expr_05: 唐朝历史地位，是中国历史地位最重要的大之一。唐朝是中国历史上的重要时期，占卜了南北朝一年的历史地位。唐朝是中国历史上的重要一朝，它的建筑、建筑、工艺、艺术、文化...
  [中文表达与总结] expr_06: 《红楼梦》的主要情节线索是：

《红楼梦》讲述了一个贾宝玉与贾宝玉的关系。黛玉因贾宝玉的性格特点而被迫成为贾宝玉与黛玉之间的爱情。贾宝玉对黛玉的婚姻生活产生了很...
  [中文表达与总结] expr_07: 云计算（Internet）是以互联网为基础的分布式数据结构，指的是互联网服务提供商的特定服务提供商的特定服务提供商的特定服务提供商的特定服务提供商的特定服务提供...
  [中文表达与总结] expr_08: 环境，如雾、雨、雨、雨、雨等，其中有些元素可能不如回。但这并不代表自然是环境的严峻，而是自然现象，在实际应用中，对环境的危害也无法得到有效解决。


若要保护环...
  [指令遵循] instr_01: * 苹果（Hypertext）
苹果（Series）
- 苹果（Medical Series）
- 香蕉（Natural Series）

- 苹果（Serie...
  [指令遵循] instr_02: JSON格式输出三个中国城市的名称。

```


4


```

C


5


5

7


1


5


6

7

7

7

7

8

7

7...
  [指令遵循] instr_03: This is an accommodate programming.

English: It is a script to programming.

En...
  [指令遵循] instr_04: 按从大到小的顺序排列以下数字：
```
10
6
4
6
```
因此，按从大到小的顺序排列后的列表为：
```
10
6
```

- 将数字按从大到小的顺序...
  [指令遵循] instr_05: 秋叶轻盈飘落，
霜叶轻盈飘落，
果实累累又无忧。
满枝头细细微微微，
小草长在叶片上轻轻拍下，
丝丝轻轻拍打着岸边，
若风也轻盈地飘落，
春风轻拂，似乎在诉说着...
  [指令遵循] instr_06: 1. **春** - 春是春的象征，其中春天的象征，春又为春。春至是春的象征，春又为春的象征，在春的阳光下，春天的象征，春的繁华，春水温暖，春天的象征，春。
2...
  [指令遵循] instr_07: 海。

河
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
江
...
  [指令遵循] instr_08: 拼音首字母排序后为：


```

```

```

- 苹果：                                               ...
  [多轮对话] multi_01: 我的编程方式不适合我。
你认为最有效的学习方法是什么？
 不是，你认为最有效的学习方法是什么？
 不，我作为AI助手，并没有实体或对信息的兴趣。
但我可以提供信...
  [多轮对话] multi_02: 江南春。

江南春。

江南春。

 春。
 南。
 江南春。
 四。
 成语出：

江南春。

杜甫。
 长。
 
江南春。
 朝诗：

江南春。
杜甫。
 ...
  [多轮对话] multi_03: 首先，理解“Python”的基本语法。Python的定义是Python中的一个概念，它定义了一个变量“n”，以变量“n”为定义。在Python中，函数通过在变量...
  [多轮对话] multi_04: 今天的天气，最高温度在60℃左右，最低温度在19℃左右。

 不是，明天北京的天气预报。


<|endoftext|>
给出一个电影名称，推荐给我电影的剧情片...
  [多轮对话] multi_05: 记得，我该怎么办？您可能需要携带以下的工具和设备：

                                                     ...
  [多轮对话] multi_06: 首先，从该市的5个直辖市的10个直辖市的10个直辖市，分别称为“南北平面”的10个直辖市。南北平面的10个直辖市区，包括南北平面、南北平面、南北平面、南北平面、...
  [多轮对话] multi_07: 首先，炒蛋需要先准备一个鸡蛋，然后按照大致的分组，将鸡蛋倒入锅中，然后按照大致的分组，逐一进行平底平底锅，再加适量盐调味，最后再加适量盐调味即可。

煎蛋的步骤...
  [多轮对话] multi_08: 人工智能是指利用机器学习和人工智能进行自动化流程自动化流程自动化流程的科学研究。它涉及机器学习和深度学习，并以计算机视觉为主要任务。

未来，机器学习将继续在更...
  [续写] cont_01: 高大的树木在绿色的树木中飘荡。大片的叶子在阳光下绽放，它们如同一道道耀眼的画卷，在草地上轻轻摇曳，仿佛在等待着一股生命的力量。
<|endoftext|>
An...
  [续写] cont_02: 高，老和尚，一直以来都以生存为主，希望他能看到山上的生命。山上有一个草，在那里有一条小溪，这里有一条清澈见底的小溪。老和尚告诉老和尚，他用水流过来，于是他和他一...
  [续写] cont_03: 了。给我几个创意，帮我设计一个新的品牌。好的，让我想想。可以尝试设计一些有趣、有吸引力的品牌，比如Bitbit、LinkedIn、Jira等等。此外，也可以考虑...
  [续写] cont_04: 开始不受伤。小明一直以为他走了很久，回到家里，看到了小鸟。但他从妈妈的家里走得很快，他重新回到了家里。小明轻轻地迈过了家，并回到了家。小明感到非常感激，因为他非...
  [续写] cont_05: 雨不时地冷落。
<|endoftext|>
创建一个包含10个逗号的句子，其中每个单词都包含一个逗号。
1. 我买了一个苹果，这个苹果很漂亮。
2. 我喜欢吃苹...
  [续写] cont_06: 踏上骑士。他在骑士里出击了大龙，并创造了一只巨大的蛇。骑士以至于他刚刚回到了王国，并成为了王国中的英雄。他对他的形象和战斗帮助到了王国的目标，并为他的后代留下了...
  [续写] cont_07: 我画下了一首悠扬的诗，让我感受到了生命的美好。读完后，我又开始思考自己所拥有的一切。
<|endoftext|>
生成以下描述的两个句子：在沙漠中，我看到一个漂...
  [续写] cont_08: 说他留下了一首悠扬的自由诗。
林黛玉说：“你不要不愿意回头，也许你来找你。”
但是他轻轻地说：“我只是觉得我不知道，我只能再告诉你。”
林黛玉告诉他：“你不是很...
  Total: 40.4s, Avg: 1.01s/prompt

Results saved to results\lora_vs_full_sft_v1\eval_results.json

~~~

## 2026-05-21 20:29:22 - Summarize structured eval artifacts

~~~powershell
$env:PYTHONIOENCODING='utf-8'; .venv\Scripts\python.exe scripts\summarize_eval_results.py
~~~

~~~text
PROMPT_EVAL_BY_MODEL
{
  "qwen_base": {
    "total": 40,
    "parseable": 40,
    "parseable_rate": 1.0,
    "missing_fields_count": 61,
    "missing_rate": 0.825,
    "extra_fields_count": 103,
    "hallucination_rate": 0.65,
    "schema_strict_count": 4,
    "schema_strict_rate": 0.1,
    "schema_strict_alias_rate": 0.075,
    "missing_alias_rate": 0.75,
    "hallucination_alias_rate": 0.75,
    "total_time_s": 73.78,
    "avg_latency_s": 1.845
  },
  "qwen_lora": {
    "total": 40,
    "parseable": 39,
    "parseable_rate": 0.975,
    "missing_fields_count": 45,
    "missing_rate": 0.8,
    "extra_fields_count": 68,
    "hallucination_rate": 0.675,
    "schema_strict_count": 3,
    "schema_strict_rate": 0.075,
    "schema_strict_alias_rate": 0.15,
    "missing_alias_rate": 0.6,
    "hallucination_alias_rate": 0.55,
    "total_time_s": 39.12,
    "avg_latency_s": 0.978
  }
}

PROMPT_EVAL_BY_GROUP
{
  "qwen_base": {
    "extraction": {
      "total": 18,
      "parseable_rate": 1.0,
      "missing_rate": 0.8889,
      "hallucination_rate": 1.0,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.0,
      "missing_alias_rate": 0.7778
    },
    "schema_constraint": {
      "total": 12,
      "parseable_rate": 1.0,
      "missing_rate": 0.9167,
      "hallucination_rate": 0.6667,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.0,
      "missing_alias_rate": 0.8333
    },
    "format_following": {
      "total": 10,
      "parseable_rate": 1.0,
      "missing_rate": 0.6,
      "hallucination_rate": 0.0,
      "schema_strict_rate": 0.4,
      "schema_strict_alias_rate": 0.3,
      "missing_alias_rate": 0.6
    }
  },
  "qwen_lora": {
    "extraction": {
      "total": 18,
      "parseable_rate": 1.0,
      "missing_rate": 0.7778,
      "hallucination_rate": 1.0,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.1111,
      "missing_alias_rate": 0.5
    },
    "schema_constraint": {
      "total": 12,
      "parseable_rate": 1.0,
      "missing_rate": 0.9167,
      "hallucination_rate": 0.6667,
      "schema_strict_rate": 0.0,
      "schema_strict_alias_rate": 0.0833,
      "missing_alias_rate": 0.6667
    },
    "format_following": {
      "total": 10,
      "parseable_rate": 0.9,
      "missing_rate": 0.7,
      "hallucination_rate": 0.1,
      "schema_strict_rate": 0.3,
      "schema_strict_alias_rate": 0.3,
      "missing_alias_rate": 0.7
    }
  }
}

VALID_200_SUMMARY
{
  "model_path": "outputs\\qwen_lora_merged_final",
  "data_path": "data\\sft_candidate\\valid.jsonl",
  "sample_count": 200,
  "seed": 42,
  "max_new_tokens": 256,
  "temperature": 0.0,
  "top_p": 1.0,
  "parseable_rate": 1.0,
  "direct_json_rate": 1.0,
  "markdown_fence_rate": 0.0,
  "exact_match_rate": 0.2,
  "avg_field_precision": 0.8749384920634917,
  "avg_field_recall": 0.7343706709956709,
  "avg_field_f1": 0.7839667314969176,
  "avg_pair_precision": 0.7796785714285712,
  "avg_pair_recall": 0.6237500000000001,
  "avg_pair_f1": 0.6731100288600288,
  "avg_latency_sec": 1.031116888523102,
  "by_task": {
    "format_following": {
      "total": 26,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.11538461538461539
    },
    "ie_extraction": {
      "total": 100,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.14
    },
    "schema_repair": {
      "total": 17,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.9411764705882353
    },
    "text_to_json": {
      "total": 57,
      "parseable_rate": 1.0,
      "direct_json_rate": 1.0,
      "exact_match_rate": 0.12280701754385964
    }
  }
}

VALID_FAILURE_SAMPLES_SAVED=17

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_012087 task=ie_extraction topic=组织 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"当今日报": {"创办者": ["詹德兰", "颜重庆"], "位于": "马来西亚"}, "大马": {"位于": "马来西亚"}}
gold={"当今大马": {"位于": "马来西亚", "创办者": ["颜重庆", "詹德兰"], "成立时间": "1999年11月20日"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_format_following_022524 task=format_following topic=事件 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"1053年－1054年的麦地那之围": {"参与者": ["拜占庭帝国", "穆斯林"], "发生地点": "马耳他岛上的穆斯林城市麦地那(Medina，今马耳他的姆迪纳)"}}
gold={"麦地那之围": {"参与者": ["马耳他岛上的穆斯林城市麦地那", "拜占庭帝国"], "发生时间": "1053年－1054年", "发生地点": "马耳他岛"}, "麦地那": {"别名": ["Medina", "姆迪纳"]}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_000847 task=ie_extraction topic=事件 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"2011年4月宫城地震": {"发生地点": "日本宫城县东部海域", "伤亡人数": "4人死亡、至少141人受伤", "发生时间": "7日深夜"}}
gold={"宫城地震": {"发生时间": ["2011年4月", "7日深夜"], "发生地点": "日本宫城县东部海域", "伤亡人数": "4人死亡、至少141人受伤"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_001370 task=ie_extraction topic=人物 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"耶律阿保机": {"出生日期": "872年", "别名": "耶律亿", "死亡日期": "926年9月6日", "职务": "大契丹国的第一位皇帝"}}
gold={"辽太祖耶律阿保机": {"别名": ["安巴坚", "耶律亿"], "配偶": "萧氏", "出生日期": "872年", "死亡日期": "926年9月6日"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_012551 task=ie_extraction topic=自然科学 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"可分离变量的偏微分方程": {"用途": "求解"}}
gold={"分离变量法": {"用途": "偏微分方程"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_text_to_json_018166 task=text_to_json topic=地理地区 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"南大西洋诸岛省": {"行政中心": "乌斯怀亚", "位于": "阿根廷"}, "阿根廷": {"位于": "南美洲"}}
gold={"火地省": {"别名": ["fueguino", "Fin del Mundo"], "位于": "阿根廷", "行政中心": "乌斯怀亚"}}

--- LOW_SCORE_SAMPLE ---
id=instructie_text_to_json_017694 task=text_to_json topic=医学 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"反射学": {"别名": "反射疗法"}, "反射疗法": {"别名": "区带疗法"}}
gold={"反射学/反射疗法": {"别名": ["脚底按摩", "区带疗法"], "疗法": ["替代疗法", "脚底按摩"]}}

--- LOW_SCORE_SAMPLE ---
id=instructie_ie_extraction_013248 task=ie_extraction topic=自然科学 parsed=True direct=True
field_f1=0.0 pair_f1=0.0 exact=False
raw={"历史语言学": {"组成": "波浪模型"}}
gold={"波浪模型": {"别名": ["波理论", "波模型"], "用途": "语言变化的模型"}}

~~~

## 2026-05-21 20:29:34 - Check existing eval deploy closure report

~~~powershell
Test-Path reports\eval_deploy_closure.md
if (Test-Path reports\eval_deploy_closure.md) { Get-Content -Encoding utf8 reports\eval_deploy_closure.md -TotalCount 80 }
~~~

~~~text
False

~~~

## 2026-05-21 20:29:52 - Summarize MicroLM 40-prompt eval result file

~~~powershell
$json = Get-Content -Raw -Encoding utf8 results\lora_vs_full_sft_v1\eval_results.json | ConvertFrom-Json
$json.PSObject.Properties | ForEach-Object {
  $m=$_.Value
  [PSCustomObject]@{Model=$_.Name; Prompts=$m.results.Count; AvgLatency=$m.avg_latency_s; TotalTime=$m.total_time_s; EvalFile='eval/prompts_v1.json'}
} | Format-Table -AutoSize
~~~

~~~text

Model    Prompts AvgLatency TotalTime EvalFile            
-----    ------- ---------- --------- --------            
pretrain      40      1.069     42.77 eval/prompts_v1.json
baseline      40      1.008     40.31 eval/prompts_v1.json
lora          40       1.01      40.4 eval/prompts_v1.json



~~~

## 2026-05-21 20:30:04 - Read latest vLLM smoke benchmark stability outputs

~~~powershell
Get-Content -Encoding utf8 results\vllm_benchmark\benchmark_summary_20260521_183849.csv
Get-Content -Encoding utf8 results\vllm_benchmark\stability_summary_20260521_184306.csv
Get-Content -Encoding utf8 results\vllm_benchmark\smoke_results_qwen_structured.json
~~~

~~~text
config,type,input_len,output_len,concurrency,time_mean_s,time_min_s,time_max_s,tok_s_mean,tok_s_min,tok_s_max,ttft_mean_s,runs,errors
sc_128_64,single,128,64,,3.5445,3.5021,3.5928,18.06,17.81,18.27,0.5317,3,
sc_512_128,single,512,128,,2.8983,2.8822,2.924,12.42,12.31,12.49,0.4348,3,
sc_1024_256,single,1024,256,,8.2525,8.1668,8.3073,31.02,30.82,31.35,1.2379,3,
mc_4conc,multi,256,128,4,7.5141,6.9852,8.5246,17.24,15.02,18.49,,3,0
mc_8conc,multi,256,128,8,6.9722,6.6656,7.3189,18.44,17.49,19.44,,3,0
round,mode,total,parse_rate,strict_rate,alias_strict_rate,extraction_P,extraction_S,extraction_A,schema_P,schema_S,schema_A,format_P,format_S,format_A,avg_latency_s
Round 1: Normal Chat Completion,normal,40,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,3.048
Round 2: Constrained (response_format=json_object),constrained,40,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,3.007
{
  "server_base_url": "http://localhost:8000",
  "timestamp": "2026-05-21 18:36:14",
  "tests": [
    {
      "name": "health_check",
      "status": "PASS",
      "detail": ""
    },
    {
      "name": "simple_chat",
      "status": "PASS",
      "response_preview": "我是阿里云开发的AI助手，可以回答各种问题、提供信息和帮助您完成任务。",
      "usage": {
        "prompt_tokens": 36,
        "total_tokens": 58,
        "completion_tokens": 22,
        "prompt_tokens_details": null
      },
      "latency_s": 2.647
    },
    {
      "name": "structured_extraction",
      "status": "PASS",
      "is_valid_json": true,
      "fields": [
        "鲁迅"
      ],
      "raw_output": "{\"鲁迅\": {\"职业\": \"中国现代文学的奠基人之一\", \"代表作\": [\"阿Q正传\", \"狂人日记\"], \"姓名\": \"周树人\"}}"
    },
    {
      "name": "multi_turn",
      "status": "PASS",
      "response_preview": "{\"名称\": \"胰岛素\", \"类型\": \"激素类药物\"}"
    },
    {
      "name": "structured_response_format",
      "status": "PASS",
      "is_valid_json": true,
      "raw_output": "{\"盘尼西林\": {\"发现者\": \"亚历山大·弗莱明\", \"名称\": \"青霉素\"}, \"青霉素\": {\"发现者\": \"亚历山大·弗莱明\"}}"
    }
  ]
}

~~~

## 2026-05-21 20:33:17 - Final JSON artifact validation

~~~powershell
$files = @(
  'eval\prompts_v1.json',
  'results\lora_vs_full_sft_v1\eval_results.json',
  'results\instructie_eval\summary\leaderboard.json',
  'results\instructie_eval\summary\structural_quality.json',
  'results\qwen_valid_eval_200\summary.json',
  'results\vllm_benchmark\smoke_results_qwen_structured.json',
  'results\vllm_benchmark\benchmark_20260521_183849.json',
  'results\vllm_benchmark\stability_20260521_184306.json'
)
foreach ($f in $files) {
  try {
    $null = Get-Content -Raw -Encoding utf8 $f | ConvertFrom-Json
    [PSCustomObject]@{Path=$f; Json='OK'}
  } catch {
    [PSCustomObject]@{Path=$f; Json='ERROR: ' + $_.Exception.Message}
  }
} | Format-Table -AutoSize
~~~

~~~text
Invoke-Expression : 所在位置 行:18 字符: 3
+ } | Format-Table -AutoSize
+   ~
不允许使用空管道元素。
所在位置 行:23 字符: 14
+ $out = try { Invoke-Expression $cmd 2>&1 | Out-String -Width 320 } ca ...
+              ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ParserError: (:) [Invoke-Expression], ParseException
    + FullyQualifiedErrorId : EmptyPipeElement,Microsoft.PowerShell.Commands.InvokeExpressionCommand
 

~~~

## 2026-05-21 20:33:34 - Final JSON artifact validation corrected

~~~powershell
$files = @(
  'eval\prompts_v1.json',
  'results\lora_vs_full_sft_v1\eval_results.json',
  'results\instructie_eval\summary\leaderboard.json',
  'results\instructie_eval\summary\structural_quality.json',
  'results\qwen_valid_eval_200\summary.json',
  'results\vllm_benchmark\smoke_results_qwen_structured.json',
  'results\vllm_benchmark\benchmark_20260521_183849.json',
  'results\vllm_benchmark\stability_20260521_184306.json'
)
$rows = foreach ($f in $files) {
  try {
    $null = Get-Content -Raw -Encoding utf8 $f | ConvertFrom-Json
    [PSCustomObject]@{Path=$f; Json='OK'}
  } catch {
    [PSCustomObject]@{Path=$f; Json='ERROR: ' + $_.Exception.Message}
  }
}
$rows | Format-Table -AutoSize
~~~

~~~text

Path                                                      Json
----                                                      ----
eval\prompts_v1.json                                      OK  
results\lora_vs_full_sft_v1\eval_results.json             OK  
results\instructie_eval\summary\leaderboard.json          OK  
results\instructie_eval\summary\structural_quality.json   OK  
results\qwen_valid_eval_200\summary.json                  OK  
results\vllm_benchmark\smoke_results_qwen_structured.json OK  
results\vllm_benchmark\benchmark_20260521_183849.json     OK  
results\vllm_benchmark\stability_20260521_184306.json     OK  



~~~

## 2026-05-21 20:33:46 - Final key artifact existence check

~~~powershell
$paths = @(
 'reports\eval_deploy_closure.md',
 'Readme\项目全景图\05-评测、验证与部署闭环.md',
 'reports\terminal_outputs_eval_deploy.md',
 'results\lora_vs_full_sft_v1\eval_results.json',
 'results\vllm_benchmark\benchmark_summary_20260521_183849.csv',
 'results\vllm_benchmark\stability_summary_20260521_184306.csv'
)
$paths | ForEach-Object { $item=Get-Item $_ -ErrorAction SilentlyContinue; [PSCustomObject]@{Path=$_; Exists=[bool]$item; Length=if($item){$item.Length}else{''}; LastWriteTime=if($item){$item.LastWriteTime}else{''}} } | Format-Table -AutoSize
~~~

~~~text

Path                                                         Exists Length LastWriteTime     
----                                                         ------ ------ -------------     
reports\eval_deploy_closure.md                                 True   7178 2026/5/21 20:31:13
Readme\项目全景图\05-评测、验证与部署闭环.md                                  True  18531 2026/5/21 20:31:32
reports\terminal_outputs_eval_deploy.md                        True 115179 2026/5/21 20:33:34
results\lora_vs_full_sft_v1\eval_results.json                  True  98846 2026/5/21 20:29:06
results\vllm_benchmark\benchmark_summary_20260521_183849.csv   True    505 2026/5/21 18:38:49
results\vllm_benchmark\stability_summary_20260521_184306.csv   True    384 2026/5/21 18:43:06



~~~

## 2026-05-21 20:33:59 - Final vLLM health check for eval/deploy closure

~~~powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health -TimeoutSec 5
~~~

~~~text


StatusCode        : 200
StatusDescription : OK
Content           : {}
RawContent        : HTTP/1.1 200 OK
                    Content-Length: 0
                    Date: Thu, 21 May 2026 12:33:58 GMT
                    Server: uvicorn
                    
                    
Headers           : {[Content-Length, 0], [Date, Thu, 21 May 2026 12:33:58 GMT], [Server, uvicorn]}
RawContentLength  : 0




~~~

## 2026-05-21 20:34:10 - Final git status for eval/deploy closure

~~~powershell
git status --short eval\prompts_v1.json "Readme\项目全景图\05-评测、验证与部署闭环.md" reports\eval_deploy_closure.md reports\terminal_outputs_eval_deploy.md results\lora_vs_full_sft_v1
~~~

~~~text
 M "Readme/\351\241\271\347\233\256\345\205\250\346\231\257\345\233\276/05-\350\257\204\346\265\213\343\200\201\351\252\214\350\257\201\344\270\216\351\203\250\347\275\262\351\227\255\347\216\257.md"
 M eval/prompts_v1.json
?? reports/eval_deploy_closure.md
?? reports/terminal_outputs_eval_deploy.md
?? results/lora_vs_full_sft_v1/

~~~
