# Schema 字段对齐提升报告

生成日期：2026-05-21

## 1. 做了什么

本轮针对 vLLM stability 中 `Strict% / Alias-Strict% = 0%` 的问题，做了三类低成本改进：

1. 在 `scripts/check_structured_stability.py` 中新增 `schema-strict` 第三轮评测。
2. 新增 schema projection 后处理指标：将模型输出 JSON 投影到允许的 schema 字段，并可选做字段 alias 归一化。
3. 抽取独立 repair 模块和 vLLM 客户端，让线上调用可直接返回 raw + repaired 两份结果。
4. 增加二阶段 self-repair：当第一次 repair 后仍缺 required 字段时，只补问缺失字段并 merge 回 repaired JSON。

新增第三轮 prompt 强约束：

- 顶层 key 必须来自 Schema 字段。
- 禁止实体名称作为顶层 key。
- 禁止 Schema 外字段。
- required 字段必须出现，无法确定时填 `null` 或 `[]`。
- 加入 flat JSON 正确示例和实体嵌套错误示例。

## 2. 代码改动

文件：

```text
scripts/check_structured_stability.py
microlm/structured/schema_repair.py
scripts/structured_vllm_client.py
```

新增能力：

- `--rounds 3` 会额外运行 `Round 3: Schema-Strict Constrained`。
- 每轮新增两个指标：
  - `projected_strict_rate`
  - `projected_alias_strict_rate`
- CSV 中新增 extraction / schema_constraint / format_following 的 projection 分组指标。
- `structured_vllm_client.py` 支持批量评测或单条请求，输出 raw JSON、repaired JSON 和 contract JSON。
- 单条请求可用 `--schema-fields "出生地,职业"` / `--required-fields "出生地,职业"`，避免 Windows PowerShell 处理 JSON 引号时出错。
- `structured_vllm_client.py --self-repair` 会对缺失 required 字段执行第二次补问，只允许输出 missing fields 列表中的字段。
- repair 层支持重复 JSON key 合并、字段 alias、递归 schema 字段提取、基础类型归一化和少量枚举归一化。

运行命令：

```powershell
.venv\Scripts\python.exe scripts\check_structured_stability.py `
  --base-url http://localhost:8000 `
  --rounds 3 `
  --limit 40 `
  --output-dir results\vllm_benchmark_schema_strict
```

## 3. 结果

结果文件：

```text
results/vllm_benchmark_schema_strict/stability_20260521_211457.json
results/vllm_benchmark_schema_strict/stability_summary_20260521_211457.csv
results/vllm_benchmark_schema_strict/repaired_outputs_20260521_enhanced.jsonl
results/vllm_benchmark_schema_strict/self_repair_outputs_20260521_v3.jsonl
```

总体结果：

| 轮次 | 模式 | Parse% | Strict% | Alias-Strict% | Projected-Strict% | Projected-Alias% |
|---|---|---:|---:|---:|---:|---:|
| Round 1 | normal | 100.0% | 0.0% | 0.0% | 20.0% | 37.5% |
| Round 2 | constrained | 100.0% | 0.0% | 0.0% | 20.0% | 37.5% |
| Round 3 | schema-strict constrained | 100.0% | 52.5% | 52.5% | 75.0% | 75.0% |

线上客户端结果：

| 指标 | 值 |
|---|---:|
| Parse% | 100.0% |
| repair_strict_rate | 77.5% |
| avg_latency | 2.432s |

二阶段 self-repair 结果：

| 指标 | 值 |
|---|---:|
| Parse% | 100.0% |
| self-repair 使用次数 | 9 / 40 |
| repair_strict_rate | 100.0% |
| avg_latency | 3.176s |

分组结果（Round 3）：

| 组别 | Strict% | Projected-Strict% |
|---|---:|---:|
| extraction | 44.4% | 88.9% |
| schema_constraint | 75.0% | 75.0% |
| format_following | 40.0% | 50.0% |

## 4. 结论

低成本 prompt 约束已经有效：

- raw `Strict%` 从 0.0% 提升到 52.5%。
- raw `Alias-Strict%` 从 0.0% 提升到 52.5%。
- stability postprocess 后 `Projected-Strict%` 达到 75.0%。
- 实际客户端 repaired strict 达到 77.5%。
- 开启二阶段 self-repair 后，40 条验证集字段契约通过率达到 100.0%。

这说明问题不主要是“模型不会抽取”，而是“默认输出风格仍偏实体嵌套”。显式告诉模型不要实体做顶层 key，并给出正反例后，模型可以大幅转向 schema-field flat JSON。

## 5. 注意事项

`Projected-Strict%` 是服务层可用指标，不等价于模型 raw 输出完全正确。它表示：在 JSON 可解析的前提下，经过 schema projection / alias normalize / 删除多余字段 / 基础类型归一化后，是否能得到符合 schema 字段集合的对象。

当前 repair 已经做基础类型检查和少量枚举归一化；self-repair 能进一步补齐遗漏字段。但内容值正确性仍需用 Exact Match / Field F1 / Pair F1 衡量，不能把字段集合命中直接等同于语义完全正确。

## 6. 下一步建议

1. 线上调用默认使用 schema-strict prompt 模板。
2. 在线服务层加入 schema projection + alias normalize。
3. 对仍缺 required 字段的样本启用 `--self-repair` 二阶段补问。
4. 将 repaired object 作为下游消费对象，将 raw output 和 self-repair output 作为审计字段保存。
5. 后续训练时加入 flat schema-field JSON 样本和“不要实体嵌套”的负例修复样本。
