# vLLM 部署验证与 Benchmark 报告

生成日期：2026-05-21

## 1. 结论摘要

vLLM 部署链路已在当前 WSL2 + GPU 环境重新验证：smoke test 5/5 通过，benchmark 5 组配置 0 errors，结构化稳定性验证在 normal 与 constrained 两种模式下 JSON Parse% 均为 100%。

需要注意的是：Parse%=100% 只说明输出是合法 JSON，不代表 schema 严格命中。当前稳定性结果中 Strict% / Alias-Strict% 仍为 0%，后续如果要提升结构化内容正确性，应继续优化数据、schema 归一化、解码约束或后处理评分。

## 2. 验证对象

| 项目 | 路径 |
|---|---|
| 合并模型 | `outputs/qwen_lora_merged_final/` |
| 启动脚本 | `scripts/serve_vllm.sh` |
| Smoke test | `scripts/smoke_vllm.py` |
| Benchmark | `scripts/bench_vllm_local.py` |
| 稳定性验证 | `scripts/check_structured_stability.py` |
| 原始结果目录 | `results/vllm_benchmark/` |

## 3. Smoke Test

结果文件：

```text
results/vllm_benchmark/smoke_results.json
results/vllm_benchmark/smoke_results_qwen_structured.json
```

| 测试项 | 结果 | 说明 |
|---|---|---|
| health_check | PASS | `/health` 正常响应 |
| simple_chat | PASS | 基础对话 completion 能力正常 |
| structured_extraction | PASS | 可输出合法结构化 JSON |
| multi_turn | PASS | 多轮上下文可用 |
| structured_response_format | PASS | `response_format=json_object` 可用 |

结构化 smoke 样例输出：

```json
{"鲁迅": {"职业": "中国现代文学的奠基人之一", "代表作": ["阿Q正传", "狂人日记"], "姓名": "周树人"}}
```

## 4. 性能 Benchmark

结果文件：

```text
results/vllm_benchmark/benchmark_summary_20260412_204210.csv
results/vllm_benchmark/benchmark_summary_20260521_183849.csv
```

| 配置 | 类型 | 输入长度 | 输出长度 | 并发 | 平均耗时 | 平均吞吐 | TTFT | errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| sc_128_64 | single | 128 | 64 | - | 3.5445s | 18.06 tok/s | 0.5317s | - |
| sc_512_128 | single | 512 | 128 | - | 2.8983s | 12.42 tok/s | 0.4348s | - |
| sc_1024_256 | single | 1024 | 256 | - | 8.2525s | 31.02 tok/s | 1.2379s | - |
| mc_4conc | multi | 256 | 128 | 4 | 7.514s | 17.24 tok/s/req | - | 0 |
| mc_8conc | multi | 256 | 128 | 8 | 6.972s | 18.44 tok/s/req | - | 0 |

观察：

- 单请求吞吐随输出长度和解码形态波动，当前 1024/256 配置达到约 31 tok/s。
- 输入 1024 / 输出 256 时 TTFT 上升到 1.24s，符合长上下文 prefill 成本增加的预期。
- 多并发没有报错，4/8 并发均稳定完成，当前配置可支撑中低并发结构化抽取服务。

## 5. 结构化稳定性

结果文件：

```text
results/vllm_benchmark/stability_summary_20260412_204357.csv
results/vllm_benchmark/stability_summary_20260521_184306.csv
```

| 轮次 | 模式 | 样本数 | Parse% | Strict% | Alias-Strict% | 平均延迟 |
|---|---|---:|---:|---:|---:|---:|
| Round 1 | normal | 40 | 100% | 0% | 0% | 3.048s |
| Round 2 | constrained | 40 | 100% | 0% | 0% | 3.007s |

判断：

- vLLM 服务化未引入 JSON 格式退化。
- `response_format=json_object` 下平均延迟与 normal 接近，不能单独说明模型内容质量提升。
- Strict% / Alias-Strict% 为 0%，说明服务化验证的当前强项是格式稳定，不是 schema 严格对齐。

## 6. 与离线评测的关系

离线 200 条 valid JSONL 评测结果：

| 指标 | 值 |
|---|---:|
| Parse% | 100% |
| Direct JSON% | 100% |
| Exact Match% | 20.0% |
| Field F1 | 0.7840 |
| Pair F1 | 0.6731 |
| 平均延迟 | 1.031s/sample |

离线评测更适合判断内容质量，vLLM benchmark 更适合判断部署性能。两个结果合在一起说明：

- 部署路径可用；
- 输出格式稳定；
- 内容级 schema 精确度仍有提升空间。

## 7. 建议

1. 默认部署 `outputs/qwen_lora_merged_final/`，不要推广 hardcase refined 版本。
2. 线上接口默认使用 `temperature=0`、`top_p=1`、`response_format=json_object`。
3. 验收指标不要只看 Parse%，至少同时看 Exact Match / Field F1 / Pair F1。
4. 如果继续优化，优先做 schema alias 归一化、实体规范化和 constrained decoding，而不是单纯增加训练步数。
