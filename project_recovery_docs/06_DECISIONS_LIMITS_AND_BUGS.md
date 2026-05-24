# 06. 关键决策、限制与踩坑

## 1. 为什么做双主线

项目没有只选择“自研小模型”或“调用开源大模型”，而是并行做两条线：

| 主线 | 价值 |
|---|---|
| 自研 MicroLM | 证明能从 tokenizer、Transformer、optimizer、loss、LoRA、KV Cache 一路手写并训练 |
| Qwen 迁移 | 在实际结构化输出任务上获得可评测、可部署的能力 |

核心判断：

- 自研线训练链路价值高，业务输出能力受规模限制。
- Qwen 线依赖开源生态，但能更接近可交付服务。
- 两条线共用方法论：配置驱动、smoke-first、日志固化、评测闭环。

## 2. 为什么结构化任务迁移到 Qwen

证据：

- MicroLM 在结构化 JSON 评测中的 Parse% 为 0%。
- Qwen base 和 qwen_lora 能稳定输出 JSON。
- qwen_lora 的 Alias-Strict% 是 base 的 2 倍。

因此：

- MicroLM 用于展示原理和系统实现。
- Qwen 用于结构化信息抽取部署。

## 3. 为什么需要 schema repair

Qwen LoRA 学到了 InstructIE 风格，但常见输出是实体嵌套：

```json
{
  "鲁迅": {
    "出生地": "浙江绍兴",
    "职业": "作家"
  }
}
```

而线上接口更希望 flat schema-field JSON：

```json
{
  "出生地": "浙江绍兴",
  "职业": "作家"
}
```

schema repair 负责：

- 从实体嵌套中提取允许字段。
- alias 归一化。
- 删除 schema 外字段。
- 检查 required 字段。
- 必要时二阶段 self-repair 补问缺失字段。

## 4. 关键 Bug 与教训

### Bug 1: tokenizer vocab 与模型 embedding 不匹配

现象：多轮对话或含 EOS token 时 `IndexError`。

根因：

- tokenizer config 设 `vocab_size=6400`。
- 初始化时又注册 special token。
- 实际 token id 可能达到 6400。
- 模型 embedding 只有 0-6399。

修复：

- `train_sft.py` 中如实际 tokenizer vocab 大于模型 vocab，resize embedding 和 lm_head。
- 推理侧避免追加越界 EOS。

教训：tokenizer 和模型 vocab size 必须显式同步。

### Bug 2: SFT 训练格式与推理格式不一致

现象：SFT 后生成质量异常，尤其多轮对话跑题。

根因：

- 训练中 assistant 回复后有 EOS。
- 推理 prompt 渲染曾缺 EOS。

修复：

- `render_chat_prompt()` 对 assistant 消息统一追加 EOS + 换行。

教训：SFT 的每个分隔符都是模型分布的一部分。

### Bug 3: LoRA checkpoint 加载顺序

现象：LoRA 权重或 base 权重没有按预期恢复。

根因：

- 注入 LoRA 后模块结构变化。
- state_dict key 和原始 Linear 不完全一致。

修复：

- 先加载 base checkpoint，再注入 LoRA。
- checkpoint loader 处理 `_orig_mod.` 前缀。

教训：凡是会替换模块结构的技术都要重新审视 save/load。

### Bug 4: LoRA 参数 device 不一致

现象：GPU 训练时报跨设备错误或 loss 不正常。

根因：

- `LoRALinear` 内新建 A/B 参数默认可能在 CPU。

修复：

- A/B 显式使用 `original.weight.device`。

教训：自定义模块中新建参数必须继承已有参数 device/dtype。

### Bug 5: InstructIE 字段漂移

现象：直接处理原始数据时字段不一致。

根因：

- train/valid/test split 字段命名不同。
- cate 命名漂移。
- relation 内字段存在差异。

修复：

- pipeline 第一步专门做 normalize。

教训：新数据集第一步永远是 profiling 和标准化，不是训练。

### Bug 6: 验证集误用训练集

现象：val_loss 异常好或速度异常慢。

根因：

- 配置中 valid path 指向训练数据。

修复：

- 修正独立 valid split。

教训：配置路径需要系统性审查，val_loss 假了训练就失去导航。

### Bug 7: wandb mode 配置错误

现象：训练启动后 wandb 中断。

根因：

- mode 值不是 `online` / `offline` / `disabled` 等有效值。

修复：

- 配置统一为字符串枚举。

教训：外部工具配置应启动前校验。

### Bug 8: Unicode surrogate 崩溃

现象：chat 第二轮或保存历史时报 `UnicodeEncodeError`。

根因：

- 小模型生成的 token 序列可能 decode 出 surrogate。
- 存入历史后下一轮 encode 崩溃。

修复：

- `chat.py` 增加 `_remove_surrogates()`。

教训：多轮系统里的错误可能来自历史状态累积。

## 5. 当前限制

| 限制 | 说明 | 建议 |
|---|---|---|
| MicroLM 生成质量有限 | 31.7M 参数，容易重复和漂移 | 演示时限制输出长度 |
| MicroLM 不适合结构化部署 | JSON Parse%=0 | 结构化任务用 Qwen |
| schema strict 仍非语义正确 | repair 只能保证字段契约 | 仍需 Field F1 / Pair F1 |
| hardcase refine 收益有限 | refined_best 未稳定优于 final | 默认部署 final |
| vLLM 依赖 WSL/Linux | Windows 原生 bash 不可用 | 使用 WSL2 |
| 大文件已进入提交 | WSL 镜像和下载包很大 | 后续考虑 Git LFS 或外部存储 |
| `microlm.structured` 打包需确认 | pyproject packages 可能未包含 | 发布前补充 setuptools 配置 |

## 6. 继续开发建议

短期：

1. 把 `microlm.structured` 纳入打包配置。
2. 固化 schema-strict prompt 和 repair API 契约。
3. 增加一键 `smoke + valid200 + stability` 验收脚本。
4. 补充 README 中对大文件和 Git LFS 的说明。

中期：

1. 尝试 Qwen2.5-7B-Instruct。
2. 加 INT4/INT8 量化部署 benchmark。
3. 扩充 flat JSON 负例修复样本。
4. 引入更多结构化抽取数据源。

长期：

1. 做数据飞轮，把线上失败样本回流训练。
2. 扩展 function calling/tool use。
3. 结合 RAG 做文档级结构化抽取。

