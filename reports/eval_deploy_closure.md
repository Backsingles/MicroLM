# 评测、验证与部署闭环收口报告

生成日期：2026-05-21

## 1. 结论摘要

`05-评测、验证与部署闭环` 已完成从固定 prompt 评测、结构化自动评测、Alias 归一化分析，到 vLLM 服务化、smoke、benchmark、稳定性验证的闭环。

当前推荐部署模型仍是：

```text
outputs/qwen_lora_merged_final/
```

核心判断：

- MicroLM 主线：40 条通用生成 prompt 已可复现运行，SFT/LoRA 与 pretrain 的原始输出已归档，适合作为人工质检和能力边界展示材料。
- Qwen 主线：结构化输出自动评测已完成，qwen_lora 的 Alias-Strict% 为 15.0%，是 qwen_base 的 2 倍。
- 部署主线：vLLM 服务 smoke 5/5 通过，benchmark 5 组配置 0 errors，normal/constrained 两轮结构化稳定性 Parse% 均为 100%。
- 已知短板：默认 prompt 下 Strict% / Alias-Strict% 在 vLLM 服务化稳定性集上仍为 0%，说明格式稳定已经成立，schema 内容精确度需要更强 prompt、后处理、schema alias 归一化或约束解码继续提升。
- 最新改进：schema-strict prompt + schema projection 后，40 条稳定性集 raw Strict% 提升到 52.5%，Projected-Strict% 提升到 75.0%；独立 vLLM 客户端 repaired strict 达到 77.5%，开启二阶段 self-repair 后字段契约通过率达到 100.0%。详见 `reports/schema_strict_improvement_report.md`。

## 2. 5.1 通用生成评测体系

Prompt 文件状态：

| 文件 | 规模 | 状态 |
|---|---:|---|
| `eval/prompts_baseline.json` | 13 | 已校验 |
| `eval/prompts_v1.json` | 40 | 已修复并补齐 |
| `eval/prompts_instructie.json` | 40 | 已校验 |

本轮修复：

- 修复 `eval/prompts_v1.json` 中未转义的 `"春"`，使文件恢复为合法 JSON。
- 将 `prompts_v1.json` 从 36 条补齐到 40 条，五类 prompt 各 8 条。

本轮复跑命令：

```powershell
.venv\Scripts\python.exe scripts\run_eval_prompts.py `
  --eval-file eval\prompts_v1.json `
  --models pretrain=outputs\pretrain_full_corpus\ckpt_final.pt baseline=outputs\sft_baseline\ckpt_final.pt lora=outputs\sft_lora\ckpt_final.pt `
  --out-dir results\lora_vs_full_sft_v1 `
  --lora-adaptor outputs\sft_lora\lora_adaptor.pt `
  --device cuda `
  --dtype float16
```

结果文件：

```text
results/lora_vs_full_sft_v1/eval_results.json
```

运行摘要：

| 模型 | Prompt 数 | 总耗时 | 平均延迟 |
|---|---:|---:|---:|
| pretrain | 40 | 42.77s | 1.069s |
| baseline | 40 | 40.31s | 1.008s |
| lora | 40 | 40.40s | 1.010s |

说明：该脚本产出的是固定 prompt 的原始生成结果；人工 4 维度评分结论沿用既有项目记录。新产物补齐了 40 条扩展集的可复现推理材料。

## 3. 5.2 结构化输出自动评测

评测文件：

```text
eval/prompts_instructie.json
```

四模型结果：

| 模型 | Parse% | 缺字段率 | 幻觉字段率 | Strict% |
|---|---:|---:|---:|---:|
| qwen_base | 100.0% | 82.5% | 65.0% | 10.0% |
| qwen_lora | 97.5% | 80.0% | 67.5% | 7.5% |
| microlm_sft | 0.0% | 100.0% | 0.0% | 0.0% |
| microlm_lora | 0.0% | 100.0% | 0.0% | 0.0% |

结果目录：

```text
results/instructie_eval/
results/instructie_eval_qwen/
```

结论：MicroLM 系列不具备可靠 JSON 输出能力；部署候选集中在 Qwen base 与 Qwen LoRA。

## 4. 5.3 Alias 归一化与质量分析

Alias 归一化结果：

| 指标 | qwen_base | qwen_lora | 结论 |
|---|---:|---:|---|
| Alias-Strict% | 7.5% | 15.0% | LoRA 为 base 的 2 倍 |
| missing alias rate | 75.0% | 60.0% | LoRA 缺字段改善 15pp |
| hallucination alias rate | 75.0% | 55.0% | LoRA 幻觉字段改善 20pp |

结构化行为质量：

| 指标 | qwen_base | qwen_lora |
|---|---:|---:|
| 实体做 key 率 | 57.5% | 95.0% |
| 全中文字段率 | 55.0% | 92.5% |
| 平均字段重叠率 | 16.97% | 49.0% |

结论：LoRA 的价值主要体现在结构化行为塑形，而不是单纯提升严格字段名命中。qwen_lora 更倾向输出 InstructIE 风格的实体嵌套 JSON，因此推荐作为最终部署版本。

## 5. 5.4 vLLM 部署与 Smoke Test

当前部署环境：

| 项目 | 值 |
|---|---|
| vLLM | 0.21.0 |
| torch | 2.11.0+cu130 |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| WSL distro | MicroLM-Ubuntu |
| venv | `/root/.venvs/microlm-vllm` |
| 服务地址 | `http://localhost:8000/v1` |

启动方式：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_vllm_wsl.ps1
```

本轮部署修复：

- `scripts/serve_vllm.sh` 的 `VLLM_DYPE` 更正为 `VLLM_DTYPE`。
- 参数解析支持 `--port 8001` 和 `--port=8001` 两种形式。
- 默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，绕过当前 WSL/RTX 50 系列环境中 FlashInfer sampler JIT 依赖 `nvcc` 的启动问题。

Smoke test：

| 测试项 | 结果 |
|---|---|
| health_check | PASS |
| simple_chat | PASS |
| structured_extraction | PASS |
| multi_turn | PASS |
| structured_response_format | PASS |

结果文件：

```text
results/vllm_benchmark/smoke_results_qwen_structured.json
```

## 6. 5.5 性能 Benchmark 与稳定性验证

Benchmark 结果：

| 配置 | 类型 | 输入 | 输出 | 平均耗时 | Tok/s | TTFT | errors |
|---|---|---:|---:|---:|---:|---:|---:|
| sc_128_64 | single | 128 | 64 | 3.5445s | 18.06 | 0.5317s | - |
| sc_512_128 | single | 512 | 128 | 2.8983s | 12.42 | 0.4348s | - |
| sc_1024_256 | single | 1024 | 256 | 8.2525s | 31.02 | 1.2379s | - |
| mc_4conc | multi | 256 | 128 | 7.5141s | 17.24/req | - | 0 |
| mc_8conc | multi | 256 | 128 | 6.9722s | 18.44/req | - | 0 |

结果文件：

```text
results/vllm_benchmark/benchmark_20260521_183849.json
results/vllm_benchmark/benchmark_summary_20260521_183849.csv
```

结构化稳定性：

| 轮次 | 模式 | 样本数 | Parse% | Strict% | Alias-Strict% | 平均延迟 |
|---|---|---:|---:|---:|---:|---:|
| Round 1 | normal | 40 | 100.0% | 0.0% | 0.0% | 3.048s |
| Round 2 | constrained | 40 | 100.0% | 0.0% | 0.0% | 3.007s |

结果文件：

```text
results/vllm_benchmark/stability_20260521_184306.json
results/vllm_benchmark/stability_summary_20260521_184306.csv
```

结论：服务化没有破坏 JSON 格式稳定性；`response_format=json_object` 可作为线上默认约束，但字段语义正确性仍不能只靠 Parse% 判断。

## 7. 关联产物清单

| 类型 | 路径 |
|---|---|
| 通用 prompt 扩展集 | `eval/prompts_v1.json` |
| MicroLM 40 prompt 结果 | `results/lora_vs_full_sft_v1/eval_results.json` |
| 结构化 prompt 集 | `eval/prompts_instructie.json` |
| 结构化四模型评测 | `results/instructie_eval/` |
| Qwen 结构化评测 | `results/instructie_eval_qwen/` |
| Valid JSONL 200 条评测 | `results/qwen_valid_eval_200/` |
| vLLM smoke | `results/vllm_benchmark/smoke_results_qwen_structured.json` |
| vLLM benchmark | `results/vllm_benchmark/benchmark_summary_20260521_183849.csv` |
| vLLM stability | `results/vllm_benchmark/stability_summary_20260521_184306.csv` |
| Schema-strict stability | `results/vllm_benchmark_schema_strict/stability_summary_20260521_211457.csv` |
| Schema repair client outputs | `results/vllm_benchmark_schema_strict/repaired_outputs_20260521_enhanced.jsonl` |
| Schema self-repair outputs | `results/vllm_benchmark_schema_strict/self_repair_outputs_20260521_v3.jsonl` |
| vLLM 部署说明 | `docs/vllm_deploy.md` |
| vLLM benchmark 报告 | `reports/vllm_benchmark_report.md` |
| Schema 字段对齐提升报告 | `reports/schema_strict_improvement_report.md` |
| 终端记录 | `reports/terminal_outputs_eval_deploy.md` |

## 8. 后续优化建议

1. 将 Strict% / Alias-Strict% 作为结构化内容优化主指标，不再只看 Parse%。
2. 优先做 schema alias 归一化、实体名归一化、关系字段归一化和 constrained decoding。
3. 对 vLLM 服务增加固定线上验收集：smoke + 40 prompt stability + 200 valid JSONL 抽样。
4. 保留 MicroLM 通用生成评测作为能力边界展示，不再把小模型作为结构化部署候选。
