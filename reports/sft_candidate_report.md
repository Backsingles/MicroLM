# SFT 候选集数据报告

> 生成时间: 2026-05-20 20:57
> 数据来源: zjunlp/InstructIE


## 1. Pipeline 执行摘要

| 阶段 | 样本数 | 说明 |
|------|--------|------|
| 原始 train | 171,471 | 全量弱监督 |
| 标准化后 | 171,471 | 字段统一，无丢失 |
| 硬过滤后 | 167,886 | 去空关系/泄漏/异常 |
| 软过滤后 | 163,629 | per-topic P99 分位数 |
| 质量分层 high | 156,275 (95.5%) | 高质量子集 |
| 派生后 | 623,650 | 4类任务 x 多质量层 |
| 采样后 | **30,000** | 分层采样 |
| 最终 train | **28,500** | chat-style JSONL |
| 最终 valid | **1,500** | 内部验证集 |


## 2. 任务配比

| 任务类型 | 数量 | 占比 | 目标 |
|----------|------|------|------|
| ie_extraction | 15000 | 50.0% | 50% |
| text_to_json | 7500 | 25.0% | 25% |
| format_following | 4500 | 15.0% | 15% |
| schema_repair | 3000 | 10.0% | 10% |

## 3. Topic 分布 (train)

| Topic | 数量 | 占比 |
|-------|------|------|
| 组织 | 2403 | 8.4% |
| 作品 | 2393 | 8.4% |
| 人造物件 | 2389 | 8.4% |
| 生物 | 2387 | 8.4% |
| 事件 | 2376 | 8.3% |
| 自然科学 | 2373 | 8.3% |
| 运输 | 2371 | 8.3% |
| 地理地区 | 2367 | 8.3% |
| 建筑 | 2362 | 8.3% |
| 人物 | 2362 | 8.3% |
| 医学 | 2361 | 8.3% |
| 天文对象 | 2356 | 8.3% |

## 4. 质量分布 (train)

| 质量等级 | 数量 | 占比 |
|----------|------|------|
| high | 28500 | 100.0% |
| medium | 0 | 0.0% |
| low | 0 | 0.0% |

## 5. 过滤明细

### 硬过滤

- empty_relation: 2,446 (空关系)
- leak_with_valid_test: 638 (与官方 valid/test 文本重叠)
- too_many_relations: 632 (>25条关系)
- too_long_input: 145 (>800字符)
- too_long_head_tail: 40 (head/tail>100字符)
- too_short_input: 3 (<15字符)

### 软过滤 (per-topic P99)

- soft_input_len_exceed: 1,499
- soft_head_tail_len_exceed: 1,461
- soft_relation_count_exceed: 984
- soft_output_len_exceed: 313

## 6. 分层采样策略

按以下维度分层采样:
- task_type: 精确控制 50/25/15/10 配比
- topic_schema: 12 个 topic 等比例分配
- quality_tier: 仅保留 high 质量
- complexity: 优先中等复杂度 (关系数 4~6, 输入长度 100~250)

## 7. 数据格式

每条样本为 chat-style JSONL:
```json
{
  "id": "instructie_ie_extraction_000001",
  "source": "instructie",
  "task_type": "ie_extraction",
  "topic_schema": "人物",
  "quality_tier": "high",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{...}"}
  ]
}
```

## 8. 剩余风险点

1. **弱监督噪声**: 虽然匹配率 99.7%，但仍有 0.3% 的 head/tail 不完全在原文中
2. **schema_repair 扰动**: 仅覆盖了 4 种扰动类型，可能不够多样化
3. **format_following**: 同一原始样本的约束文本是随机选择的，但约束类型有限
4. **topic 不均衡**: 原始数据中医学(3,244)和自然科学(4,308)样本少，采样后每个 topic 均 2,500 条，相当于对这两个 topic 过采样
5. **输出格式**: 使用按实体分组的 JSON 格式，与 InstructIE 原始三元组格式不同，需要在评估时注意
6. **内部 valid**: 从 train 中切分，与官方 valid/test 独立

## 9. 目标达成度

| 目标 | 达成情况 |
|------|----------|
| 按 schema 抽取 | 四类任务均围绕 schema 字段抽取 |
| 稳定输出合法 JSON | 所有 output 经 json.loads 验证 |
| 任务配比 50/25/15/10 | 精确匹配 |
| Topic 均衡 | 12 topic 完全均衡 |
| 质量 high 占比 | 100% |
| 候选集 2~4万 | 30,000 (命中) |