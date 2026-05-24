# Qwen 结构化输出模型 vLLM 部署说明

生成日期：2026-05-21

## 1. 部署对象

本项目的推荐部署模型是已经合并 LoRA 的最终 Qwen 模型：

```text
outputs/qwen_lora_merged_final/
```

该目录由 `scripts/export_final_model.py` 导出，采用标准 HuggingFace CausalLM 格式，可被 vLLM 直接加载。当前导出元信息：

| 项目 | 值 |
|---|---|
| 基座 | `Qwen2.5-1.5B-Instruct` |
| LoRA adaptor | `outputs/qwen_lora/adaptor_final` |
| 合并模型 | `outputs/qwen_lora_merged_final` |
| 参数量 | 1,543,714,304 |
| LoRA rank / alpha | r=8, alpha=16 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |

说明：`outputs/qwen_lora_merged_refined_best/` 是 hardcase refinement 实验产物，评测没有带来稳定收益，不推荐作为默认部署模型。

## 2. 环境要求

推荐在 Linux 或 WSL2 + CUDA 环境运行 vLLM。当前仓库位于 Windows PowerShell 环境，`bash` 不可用时无法直接执行 `scripts/serve_vllm.sh`。

基础依赖：

```bash
pip install vllm
```

如果需要复用项目环境，还需要确保 transformers / peft / torch 已安装，并且 CUDA 可用。

### 当前 Windows/WSL2 安装状态

本机已通过 WSL2 配置 vLLM GPU 环境：

| 项目 | 当前值 |
|---|---|
| WSL distro | `MicroLM-Ubuntu` |
| vLLM venv | `/root/.venvs/microlm-vllm` |
| vLLM | `0.21.0` |
| torch | `2.11.0+cu130` |
| GPU | `NVIDIA GeForce RTX 5060 Ti` |
| 部署模型 | `/mnt/e/MicroLM/outputs/qwen_lora_merged_final` |

从 Windows PowerShell 启动：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_vllm_wsl.ps1
```

在 WSL 内启动：

```bash
cd /mnt/e/MicroLM
source /root/.venvs/microlm-vllm/bin/activate
bash scripts/serve_vllm.sh
```

当前 WSL + RTX 50 系列环境下，FlashInfer sampler 会尝试 JIT 编译并依赖 `nvcc`。项目启动脚本默认设置：

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

这样会使用 PyTorch-native sampler，避免 vLLM 启动时报 `Could not find nvcc`。

## 3. 启动服务

默认启动方式：

```bash
bash scripts/serve_vllm.sh
```

自定义端口：

```bash
bash scripts/serve_vllm.sh --port 8001
```

或：

```bash
bash scripts/serve_vllm.sh --port=8001
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--port 8001` / `--port=8001` | 修改服务端口 |
| `--host 0.0.0.0` / `--host=0.0.0.0` | 修改监听地址 |
| `--tp 1` / `--tp=1` | tensor parallel size |
| `--max-model-len 4096` | 最大上下文长度 |
| `--cpu` | CPU 测试模式，速度很慢，仅用于功能检查 |

服务启动后，OpenAI 兼容 API 地址为：

```text
http://localhost:8000/v1
```

健康检查地址：

```text
http://localhost:8000/health
```

## 4. Smoke Test

服务启动后运行：

```bash
python scripts/smoke_vllm.py \
  --base-url http://localhost:8000 \
  --structured \
  --output results/vllm_benchmark/smoke_results.json
```

现有 smoke 结果为 5/5 通过：

| 测试项 | 结果 |
|---|---|
| health_check | PASS |
| simple_chat | PASS |
| structured_extraction | PASS |
| multi_turn | PASS |
| structured_response_format | PASS |

当前安装验证结果已保存到：

```text
results/vllm_benchmark/smoke_results_current.json
results/vllm_benchmark/smoke_results_qwen_structured.json
```

## 5. Benchmark

运行本地 benchmark：

```bash
python scripts/bench_vllm_local.py \
  --base-url http://localhost:8000 \
  --output-dir results/vllm_benchmark
```

快速模式：

```bash
python scripts/bench_vllm_local.py \
  --base-url http://localhost:8000 \
  --quick \
  --output-dir results/vllm_benchmark
```

已有 benchmark 覆盖 5 组配置，0 errors。详情见：

```text
results/vllm_benchmark/benchmark_20260521_183849.json
results/vllm_benchmark/benchmark_summary_20260521_183849.csv
reports/vllm_benchmark_report.md
```

## 6. 结构化稳定性验证

运行：

```bash
python scripts/check_structured_stability.py \
  --base-url http://localhost:8000 \
  --rounds 2 \
  --limit 40 \
  --output-dir results/vllm_benchmark
```

现有结果：

| 模式 | 样本数 | Parse% | Strict% | Alias-Strict% | 平均延迟 |
|---|---:|---:|---:|---:|---:|
| normal | 40 | 100% | 0% | 0% | 3.048s |
| constrained | 40 | 100% | 0% | 0% | 3.007s |

解读：vLLM 服务化没有破坏 JSON 可解析性；但严格 schema 命中率仍是后续优化点，不能把 Parse%=100% 解读为内容完全正确。

## 7. 推荐调用方式

结构化抽取请求建议使用低温或 greedy 设置：

```json
{
  "model": "outputs/qwen_lora_merged_final",
  "messages": [
    {
      "role": "system",
      "content": "你是一个严格遵循 schema 的信息抽取助手。请严格按照给定的 schema 从文本中抽取信息，并以 JSON 格式输出。不要在 JSON 前后添加任何解释性文字。"
    },
    {
      "role": "user",
      "content": "Instruction: 从文本中抽取实体和关系。\n\nSchema: ...\n\nInput: ..."
    }
  ],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 256,
  "response_format": {
    "type": "json_object"
  }
}
```

更推荐使用项目内置的 schema-strict 客户端，它会同时保存模型 raw 输出和修复后的 schema JSON：

```powershell
.venv\Scripts\python.exe scripts\structured_vllm_client.py `
  --base-url http://localhost:8000 `
  --eval-file eval\prompts_instructie.json `
  --limit 40 `
  --self-repair `
  --output results\vllm_benchmark_schema_strict\repaired_outputs.jsonl
```

单条请求示例：

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

客户端会输出：

- `raw_output`：模型原始 JSON；
- `self_repair_output`：二阶段补齐缺失字段时的原始输出；
- `repaired`：经 schema projection / alias normalize / 类型归一化后的字段；
- `repaired_for_contract`：面向线上接口契约补齐 missing 字段后的对象。

## 8. 注意事项

- 默认部署模型是 `outputs/qwen_lora_merged_final/`。
- vLLM 启动脚本依赖 bash；Windows PowerShell 环境建议使用 WSL2 或 Linux 服务器运行。
- `response_format=json_object` 可以增强 JSON 格式约束，但不会自动保证字段语义正确。
- 线上结构化抽取建议使用 `scripts/structured_vllm_client.py` 或复用 `microlm.structured.schema_repair` 的修复逻辑。
- 线上验收至少同时看 Parse%、Strict%、Alias-Strict%、字段 F1 和端到端延迟。
