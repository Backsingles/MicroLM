# MicroLM 推理与系统能力增强收口报告

生成日期：2026-05-21

## 1. 结论摘要

`03-推理与系统能力增强` 对应的系统侧闭环已经完成：单轮生成入口可用、KV Cache benchmark 有可复现数据、交互式 `chat.py` 可以完成多轮输入、历史查看和会话保存。

本轮还修复了 `generate_text.py` 的两个易用性问题：

- 默认 tokenizer 路径从旧的 `output/tinystories_bpe_10k/...` 改为当前项目产物 `outputs/tokenizer_full_clean/...`。
- 当未显式传入 `--config-path` 时，自动从 checkpoint 同目录读取 `model_config.json`。

这让单轮推理入口真正符合“给定 checkpoint 即可运行”的项目预期。

## 2. 3.1 单轮生成入口验证

验证对象：`scripts/generate_text.py`

| 模式 | 命令输入 | 结果 |
|---|---|---|
| 纯文本续写 | `--prompt "春天的早晨，"` + pretrain checkpoint | 成功生成，进程退出码 0 |
| 对话生成 | `--conversations-path reports/generate_text_smoke_conversation.json` + SFT checkpoint | 成功生成中文回答，进程退出码 0 |
| Prompt 协议单测 | `pytest tests/test_inference.py -q` | 4 passed |

SFT 对话 smoke 输出片段：

```text
人工智能是一种通过模拟人类思维的技术，能够模拟人类思维，但其主要目的是实现智能功能，但其应用需需依靠机器学习、自然语言处理、计算机视觉等技术。
```

判断：

- `resolve_generation_prompt()` 的对话渲染路径正常。
- checkpoint 同目录自动加载 `model_config.json` 后，CLI 使用成本降低。
- 当前生成质量仍受 MicroLM 模型能力限制，但系统入口已经可用。

## 3. 3.2 KV Cache Benchmark 收口

已有 benchmark 文件：

- `results/kvcache_benchmark.json`
- `results/kvcache_benchmark.csv`

测试矩阵为 5 种 prompt 长度 × 4 种生成长度，共 20 组配置。运行环境为 CPU / float32。

| 指标 | 数值 |
|---|---:|
| 配置数 | 20 |
| 平均加速比 | 3.86x |
| 最大加速比 | 9.08x |
| 最大加速配置 | prompt=256, gen=256 |
| Cache decode 平均吞吐 | 100.3 tok/s |
| No-cache 平均吞吐 | 31.6 tok/s |

判断：

- KV Cache 的收益随上下文长度和生成长度增加而放大。
- Cache decode 吞吐基本稳定在约 100 tok/s，符合文档中“decode 阶段计算量可预测”的判断。
- No-cache 路径随序列变长明显衰减，证明重复计算历史 K/V 是主要瓶颈。

## 4. 3.3 交互式 Chat 系统验证

验证对象：`scripts/chat.py`

本轮使用 UTF-8 输入文件驱动非交互 REPL：

```text
你好，请介绍一下你自己。
/history
/save reports/chat_smoke_session_utf8.jsonl
/quit
```

验证结果：

| 功能 | 结果 |
|---|---|
| 模型加载 | 成功，加载到 CUDA |
| 用户输入 | 成功进入会话 |
| 模型回复 | 成功生成 assistant 消息 |
| `/history` | 成功显示当前对话 |
| `/save` | 成功保存 JSONL 会话 |
| `/quit` | 成功退出 |

保存的会话文件：

- `reports/chat_smoke_session_utf8.jsonl`

判断：

- `chat.py` 已经完成从单次脚本推理到多轮 REPL 的系统化封装。
- UTF-8 输入路径正常；PowerShell 直接数组管道会把中文转成问号，正式演示时建议使用 UTF-8 文件输入或直接交互式终端输入。

## 5. 当前限制

| 限制 | 原因 | 建议 |
|---|---|---|
| 长输出仍会漂移或重复 | MicroLM 模型容量有限 | 演示时限制 `--max-new-tokens`，结构化任务使用 Qwen |
| EOS 自动停止仍受 tokenizer/model vocab 差异影响 | tokenizer 含 special token 后为 6401，模型 vocab 为 6400 | 保持 max_new_tokens 截断；后续如重训可统一 vocab |
| chat.py 仍要求显式传 `--config-path` / tokenizer 路径 | 当前脚本偏演示工具 | 可后续复用 generate_text 的自动推断逻辑 |
| PowerShell 管道中文可能乱码 | Windows 管道编码问题 | 设置 `$OutputEncoding` 或使用 UTF-8 输入文件 |

## 6. 工程结论

`03-推理与系统能力增强` 可以收口。MicroLM 已经具备完整的系统演示路径：

1. `generate_text.py` 用于单轮续写和对话生成。
2. `benchmark_kvcache.py` 用于证明 KV Cache 的算法收益。
3. `chat.py` 用于交互式多轮对话、参数调节和会话保存。

后续如果继续推进，重点不应再放在 MicroLM 生成质量本身，而应进入 `05-评测、验证与部署闭环`：以 Qwen 最终模型为部署对象，验证接口、服务性能和结构化输出稳定性。

## 7. 关联文件

- 推理入口：`scripts/generate_text.py`
- 交互式对话：`scripts/chat.py`
- KV Cache benchmark：`scripts/benchmark_kvcache.py`
- 推理协议单测：`tests/test_inference.py`
- KV Cache 结果：`results/kvcache_benchmark.json`
- SFT 对话 smoke 输入：`reports/generate_text_smoke_conversation.json`
- Chat smoke 输入：`reports/chat_smoke_input.txt`
- Chat smoke 会话：`reports/chat_smoke_session_utf8.jsonl`
- 本轮终端记录：`reports/terminal_outputs_inference_system.md`
