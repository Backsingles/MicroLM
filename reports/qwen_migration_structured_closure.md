# Qwen 迁移与结构化输出主线收口报告

生成日期：2026-05-21

## 1. 结论摘要

`04-Qwen 迁移与结构化输出主线` 已完成从迁移动机、InstructIE 数据 pipeline、Qwen LoRA 训练、模型导出到 vLLM 部署准备的闭环。

当前推荐模型仍是：

```text
outputs/qwen_lora_merged_final/
```

hardcase refinement 实验产物 `outputs/qwen_lora_merged_refined_best/` 不建议替换主模型，因为对 hardcases 没有实质改善，holdout 变化也只是轻微波动。

## 2. 4.1 迁移决策

迁移证据链已经成立：

```text
MicroLM JSON Parse%=0%
  → 31M 小模型不具备可靠结构化输出能力
  → 选择 Qwen2.5-1.5B-Instruct
  → 聚焦 schema-guided 信息抽取
  → 使用 LoRA 做结构化行为塑形
```

MicroLM 的价值是证明自研训练链路；Qwen 主线的价值是交付可评测、可部署的结构化输出能力。

## 3. 4.2 InstructIE 数据 Pipeline

原始数据与最终数据产物均已就位。

| 项目 | 结果 |
|---|---:|
| 原始中文 train | 171,471 |
| 最终 train.jsonl | 28,500 |
| 最终 valid.jsonl | 1,500 |
| 任务类型 | 4 类 |
| topic schema | 12 类 |
| 数据格式 | chat-style JSONL |

四类任务配比：

| 任务 | 配比 |
|---|---:|
| ie_extraction | 50% |
| text_to_json | 25% |
| format_following | 15% |
| schema_repair | 10% |

核心产物：

```text
data/sft_candidate/train.jsonl
data/sft_candidate/valid.jsonl
data/sft_candidate/metadata.json
```

## 4. 4.3 Qwen LoRA 训练

正式训练配置：

| 参数 | 值 |
|---|---|
| 基座 | `./Qwen2.5-1.5B-Instruct` |
| LoRA r / alpha | 8 / 16 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| batch_size | 4 |
| grad_accum | 4 |
| effective batch | 16 |
| max_steps | 2,000 |
| precision | FP16 |
| device | CUDA |

最终训练结果：

| Step | train_loss | val_loss |
|---:|---:|---:|
| 1300 | 0.169383 | 0.168402 |
| 1400 | 0.242021 | 0.165343 |
| 1500 | 0.210959 | 0.165426 |
| 1600 | 0.147618 | 0.161547 |
| 1700 | 0.199097 | 0.160555 |
| 1800 | 0.181111 | 0.159483 |
| 1900 | 0.090205 | 0.157389 |
| 2000 | 0.186020 | 0.155349 |

训练产物：

```text
outputs/qwen_lora/adaptor_final/
outputs/qwen_lora/best_adaptor/
outputs/qwen_lora/train_log.jsonl
```

## 5. 4.4 导出与部署准备

导出产物：

```text
outputs/qwen_lora_merged_final/
```

导出元信息：

| 项目 | 值 |
|---|---|
| total_params | 1,543,714,304 |
| adaptor_path | `outputs/qwen_lora/adaptor_final` |
| elapsed_sec | 6.9 |
| PEFT version | 0.19.1 |

本轮补齐并复验的部署文档与报告：

```text
docs/vllm_deploy.md
reports/vllm_benchmark_report.md
```

同时修复了 `scripts/serve_vllm.sh` 的两个部署易用性问题：

- `VLLM_DYPE` 更正为 `VLLM_DTYPE`。
- 参数解析同时支持 `--port 8001` 与 `--port=8001`，`--host`、`--tp`、`--max-model-len` 同理。
- WSL/RTX 50 系列环境默认设置 `VLLM_USE_FLASHINFER_SAMPLER=0`，避免 FlashInfer sampler JIT 依赖 `nvcc` 导致服务启动失败。

## 6. 评测结果

### 6.1 Prompt 结构化评测

| 模型 | Parse% | Strict% | Alias-Strict% |
|---|---:|---:|---:|
| qwen_base | 100.0% | 10.0% | 7.5% |
| qwen_lora | 97.5% | 7.5% | 15.0% |

解读：LoRA 的 strict 主指标略低，但 alias-normalized 指标翻倍，说明模型更偏向 InstructIE 风格的实体中心结构，而不是扁平字段格式。

### 6.2 Valid JSONL 200 条评测

| 指标 | 值 |
|---|---:|
| Parse% | 100.0% |
| Direct JSON% | 100.0% |
| Exact Match% | 20.0% |
| Field F1 | 0.7840 |
| Pair F1 | 0.6731 |
| 平均延迟 | 1.031s/sample |

分任务看，`schema_repair` exact match 最高，`ie_extraction` / `text_to_json` 仍是后续优化重点。

## 7. vLLM 部署验证

已有 vLLM 结果目录：

```text
results/vllm_benchmark/
```

Smoke test：

| 项目 | 结果 |
|---|---|
| health_check | PASS |
| simple_chat | PASS |
| structured_extraction | PASS |
| multi_turn | PASS |
| structured_response_format | PASS |

Benchmark：

| 指标 | 值 |
|---|---:|
| 配置数 | 5 |
| errors | 0 |
| 单请求吞吐 | 约 12.42-31.02 tok/s |
| 4 并发吞吐 | 17.24 tok/s/req |
| 8 并发吞吐 | 18.44 tok/s/req |

结构化稳定性：

| 模式 | Parse% | Strict% | Alias-Strict% |
|---|---:|---:|---:|
| normal | 100% | 0% | 0% |
| constrained | 100% | 0% | 0% |

解读：vLLM 服务化路径没有破坏 JSON 格式稳定性，但 schema 严格命中仍需要数据、约束解码或后处理继续优化。

## 8. 当前状态判断

`04-Qwen 迁移与结构化输出主线` 可以收口。它已经完成：

- 数据准备；
- 6 步 pipeline；
- Qwen LoRA 训练；
- LoRA merge 导出；
- 离线评测；
- vLLM 部署准备；
- smoke / benchmark / stability 结果归档；
- 部署文档与 benchmark 报告。

本轮已在当前 WSL2 + GPU 环境重新启动并验证 vLLM 服务，最新 smoke、benchmark、stability 结果已写入 `results/vllm_benchmark/`。当前服务加载的模型为 `/mnt/e/MicroLM/outputs/qwen_lora_merged_final`。

## 9. 后续建议

下一步应进入 `05-评测、验证与部署闭环`，重点不是再训练，而是把验收指标体系固定下来：

- 离线：Parse%、Exact Match、Field F1、Pair F1；
- 服务：TTFT、吞吐、P95、并发错误率；
- 稳定性：normal / constrained 双模式结构化输出；
- 失败样本：schema alias、实体归一化、关系表示差异。

如果继续优化模型，优先做 schema/实体归一化和受约束解码；单纯 hardcase replay 已经验证收益不明显。

## 10. 关联文件

- 数据 metadata：`data/sft_candidate/metadata.json`
- Qwen 训练日志：`outputs/qwen_lora/train_log.jsonl`
- 合并模型：`outputs/qwen_lora_merged_final/`
- 导出 metadata：`outputs/qwen_lora_merged_final/export_metadata.json`
- 离线评测：`results/qwen_valid_eval_200/summary.json`
- vLLM 结果：`results/vllm_benchmark/`
- 部署文档：`docs/vllm_deploy.md`
- vLLM 报告：`reports/vllm_benchmark_report.md`
- 本轮终端记录：`reports/terminal_outputs_qwen_structured.md`
