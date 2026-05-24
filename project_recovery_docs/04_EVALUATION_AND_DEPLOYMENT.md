# 04. 评测与部署说明

## 1. 评测体系分层

项目中的评测分为五层：

| 层级 | 目的 | 代表脚本 |
|---|---|---|
| 通用生成评测 | 看 MicroLM pretrain/SFT/LoRA 的开放生成效果 | `run_eval_prompts.py` |
| 结构化 prompt 评测 | 看不同模型对 schema 输出的遵循能力 | `run_instructie_eval.py` |
| valid JSONL 自动评测 | 用真实 valid 样本算 Exact Match / Field F1 / Pair F1 | `evaluate_qwen_valid_jsonl.py` |
| vLLM smoke/benchmark | 验证服务是否可用、吞吐和延迟 | `smoke_vllm.py`, `bench_vllm_local.py` |
| stability/schema repair | 验证线上结构化输出契约和修复层 | `check_structured_stability.py`, `structured_vllm_client.py` |

## 2. 通用生成评测

评测集：

```text
eval/prompts_v1.json
```

入口：

```text
scripts/run_eval_prompts.py
```

结果：

```text
results/lora_vs_full_sft_v1/eval_results.json
```

记录摘要：

| 模型 | prompt 数 | 平均延迟 |
|---|---:|---:|
| pretrain | 40 | 约 1.069s |
| baseline | 40 | 约 1.008s |
| lora | 40 | 约 1.010s |

项目结论：

- MicroLM SFT 比 pretrain 明显更像指令模型。
- MicroLM 仍有容量上限，长输出容易重复、跑题或漂移。
- 通用生成结果更适合作为能力边界展示，不适合作为线上结构化任务依据。

## 3. 四模型结构化评测

评测集：

```text
eval/prompts_instructie.json
```

入口：

```text
scripts/run_instructie_eval.py
```

结果：

```text
results/instructie_eval/
results/instructie_eval_qwen/
```

指标定义：

| 指标 | 含义 |
|---|---|
| Parse% | 输出是否可解析为 JSON |
| missing_rate | required 字段缺失率 |
| hallucination_rate | schema 外字段幻觉率 |
| Strict% | 缺失、幻觉、枚举等检查全部通过 |
| Alias-Strict% | 字段 alias 归一化后再判断 strict |

leaderboard：

| 模型 | Parse% | Strict% | Alias-Strict% | 缺字段率 | 幻觉字段率 |
|---|---:|---:|---:|---:|---:|
| qwen_base | 100.0% | 10.0% | 7.5% | 82.5% | 65.0% |
| qwen_lora | 97.5% | 7.5% | 15.0% | 80.0% | 67.5% |
| microlm_sft | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |
| microlm_lora | 0.0% | 0.0% | 0.0% | 100.0% | 0.0% |

结构化行为质量：

| 指标 | qwen_base | qwen_lora |
|---|---:|---:|
| 实体做 key 率 | 57.5% | 95.0% |
| 全中文字段率 | 55.0% | 92.5% |
| 平均字段重叠率 | 16.97% | 49.0% |

解读：

- qwen_lora 的 raw Strict% 不比 base 高，但 Alias-Strict% 翻倍。
- LoRA 学到更多中文 schema 行为，但更倾向实体嵌套 JSON。
- 因此线上需要 schema-strict prompt 和 repair 层。

## 4. Valid JSONL 200 条评测

入口：

```text
scripts/evaluate_qwen_valid_jsonl.py
```

结果：

```text
results/qwen_valid_eval_200/summary.json
```

指标：

| 指标 | 值 |
|---|---:|
| sample_count | 200 |
| Parse% | 100.0% |
| Direct JSON% | 100.0% |
| Exact Match% | 20.0% |
| Field Precision | 87.49% |
| Field Recall | 73.44% |
| Field F1 | 78.40% |
| Pair Precision | 77.97% |
| Pair Recall | 62.38% |
| Pair F1 | 67.31% |
| 平均延迟 | 1.031s/sample |

分任务：

| task | 样本数 | Exact Match |
|---|---:|---:|
| schema_repair | 17 | 94.12% |
| ie_extraction | 100 | 14.00% |
| text_to_json | 57 | 12.28% |
| format_following | 26 | 11.54% |

关键解读：

- Parse 和 direct JSON 已稳定。
- 内容语义仍有提升空间。
- `schema_repair` 明显更容易 exact match，`ie_extraction` 和 `text_to_json` 是主要优化对象。

## 5. Schema Repair

模块：

```text
microlm/structured/schema_repair.py
```

关键能力：

| 能力 | 说明 |
|---|---|
| clean | 去掉 markdown fence |
| parse | JSON parse，并合并重复 key |
| alias normalize | 中文字段别名归一化 |
| schema projection | 递归扫描实体嵌套，保留 schema 字段 |
| type normalize | string / list 基础归一 |
| enum normalize | 少量领域枚举归一 |
| contract fill | 服务模式可补齐缺失 required 字段 |
| score | 检查 missing / extra / enum / strict |

重要原则：

- repair 不等于语义校正。
- repair 只把模型已经输出的结构投影到契约。
- `fill_missing=True` 是接口契约补齐，不应被当作模型真的抽取到字段。

## 6. Schema-Strict Stability

入口：

```text
scripts/check_structured_stability.py
```

结果：

```text
results/vllm_benchmark_schema_strict/stability_summary_20260521_211457.csv
```

三轮对比：

| 轮次 | Parse% | Strict% | Alias-Strict% | Projected-Strict% | Projected-Alias% | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|
| normal | 100.0% | 0.0% | 0.0% | 20.0% | 37.5% | 2.498s |
| response_format | 100.0% | 0.0% | 0.0% | 20.0% | 37.5% | 2.493s |
| schema-strict | 100.0% | 52.5% | 52.5% | 75.0% | 75.0% | 2.436s |

结论：

- 问题不是 JSON 格式，而是输出结构风格。
- 明确禁止实体名做顶层 key，并给出 flat JSON 示例后，raw Strict% 明显提升。

## 7. Structured vLLM Client

入口：

```text
scripts/structured_vllm_client.py
```

批量调用示例：

```powershell
.venv\Scripts\python.exe scripts\structured_vllm_client.py `
  --base-url http://localhost:8000 `
  --eval-file eval\prompts_instructie.json `
  --limit 40 `
  --self-repair `
  --output results\vllm_benchmark_schema_strict\self_repair_outputs.jsonl
```

单条调用示例：

```powershell
.venv\Scripts\python.exe scripts\structured_vllm_client.py `
  --base-url http://localhost:8000 `
  --instruction "从文本中抽取人物信息。" `
  --schema-fields "出生地,职业" `
  --required-fields "出生地,职业" `
  --input-text "鲁迅，原名周树人，浙江绍兴人，中国现代作家。" `
  --self-repair `
  --output results\vllm_benchmark_schema_strict\single_repaired.jsonl
```

输出字段：

| 字段 | 含义 |
|---|---|
| `raw_output` | 模型原始输出 |
| `parse_ok` | 原始输出是否可解析 |
| `repaired` | schema projection 后对象 |
| `repaired_for_contract` | 补齐 required 后的接口契约对象 |
| `schema_strict_after_repair` | repair 后是否满足字段契约 |
| `self_repair_used` | 是否触发二阶段补问 |
| `self_repair_output` | 二阶段模型输出 |

记录结果：

| 模式 | 指标 |
|---|---:|
| schema-strict + repair | repair_strict_rate 77.5%，avg latency 2.432s |
| schema-strict + self-repair | 9/40 使用二阶段补问，repair_strict_rate 100.0%，avg latency 3.176s |

## 8. vLLM 部署

推荐模型：

```text
outputs/qwen_lora_merged_final/
```

部署文档：

```text
docs/vllm_deploy.md
```

Windows/WSL 启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_vllm_wsl.ps1
```

WSL 内启动：

```bash
cd /mnt/e/MicroLM
source /root/.venvs/microlm-vllm/bin/activate
bash scripts/serve_vllm.sh
```

服务地址：

```text
http://localhost:8000/v1
http://localhost:8000/health
```

`serve_vllm.sh` 参数：

| 参数 | 说明 |
|---|---|
| `--port` | 修改端口 |
| `--host` | 修改监听地址 |
| `--tp` | tensor parallel size |
| `--max-model-len` | 最大上下文 |
| `--cpu` | CPU 测试模式 |

当前环境记录：

| 项目 | 值 |
|---|---|
| WSL distro | `MicroLM-Ubuntu` |
| venv | `/root/.venvs/microlm-vllm` |
| vLLM | 0.21.0 |
| torch | 2.11.0+cu130 |
| GPU | NVIDIA GeForce RTX 5060 Ti |

脚本默认：

```bash
VLLM_NO_USAGE_STATS=1
VLLM_USE_FLASHINFER_SAMPLER=0
```

原因：当前 WSL/RTX 50 系列环境中 FlashInfer sampler 可能依赖 `nvcc` JIT，禁用后使用 PyTorch-native sampler。

## 9. vLLM Smoke 与 Benchmark

Smoke：

```bash
python scripts/smoke_vllm.py \
  --base-url http://localhost:8000 \
  --structured \
  --output results/vllm_benchmark/smoke_results.json
```

5/5 PASS：

| 测试 | 状态 |
|---|---|
| health_check | PASS |
| simple_chat | PASS |
| structured_extraction | PASS |
| multi_turn | PASS |
| structured_response_format | PASS |

Benchmark：

```text
results/vllm_benchmark/benchmark_summary_20260521_183849.csv
```

| config | 类型 | 输入 | 输出 | 平均耗时 | Tok/s | TTFT | errors |
|---|---|---:|---:|---:|---:|---:|---:|
| sc_128_64 | single | 128 | 64 | 3.5445s | 18.06 | 0.5317s | - |
| sc_512_128 | single | 512 | 128 | 2.8983s | 12.42 | 0.4348s | - |
| sc_1024_256 | single | 1024 | 256 | 8.2525s | 31.02 | 1.2379s | - |
| mc_4conc | multi | 256 | 128 | 7.5141s | 17.24/req | - | 0 |
| mc_8conc | multi | 256 | 128 | 6.9722s | 18.44/req | - | 0 |

