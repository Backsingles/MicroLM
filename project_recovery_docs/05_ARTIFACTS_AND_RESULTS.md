# 05. 关键产物与结果索引

## 1. 源码与配置

| 类型 | 路径 |
|---|---|
| 项目元数据 | `pyproject.toml` |
| 顶层 README | `README.md` |
| 数据说明 | `data/README.md` |
| 配置目录 | `configs/` |
| prompt 评测集 | `eval/` |
| 测试目录 | `tests/` |

## 2. 自研 MicroLM 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| tokenizer vocab | `outputs/tokenizer_full_clean/vocab.json` | BPE vocab |
| tokenizer merges | `outputs/tokenizer_full_clean/merge.txt` | BPE merges |
| pretrain checkpoint | `outputs/pretrain_full_corpus/ckpt_final.pt` | 正式预训练模型 |
| pretrain model config | `outputs/pretrain_full_corpus/model_config.json` | 模型结构 |
| SFT baseline checkpoint | `outputs/sft_baseline/ckpt_final.pt` | 全参 SFT |
| SFT baseline log | `outputs/sft_baseline/train_log.jsonl` | 训练日志 |
| SFT LoRA checkpoint | `outputs/sft_lora/ckpt_final.pt` | LoRA SFT checkpoint |
| SFT LoRA adaptor | `outputs/sft_lora/lora_adaptor.pt` | 自研 LoRA adaptor |
| KV Cache benchmark | `results/kvcache_benchmark.csv` | cache/no-cache 对比 |

## 3. Qwen 产物

| 产物 | 路径 | 说明 |
|---|---|---|
| Qwen base | `Qwen2.5-1.5B-Instruct/` | 外部下载基座 |
| Qwen LoRA adaptor | `outputs/qwen_lora/adaptor_final/` | 正式 adaptor |
| Qwen best adaptor | `outputs/qwen_lora/best_adaptor/` | 最佳 val_loss adaptor |
| Qwen train log | `outputs/qwen_lora/train_log.jsonl` | 训练日志 |
| 推荐 merged model | `outputs/qwen_lora_merged_final/` | 默认部署模型 |
| export metadata | `outputs/qwen_lora_merged_final/export_metadata.json` | 导出元信息 |
| refined model | `outputs/qwen_lora_merged_refined_best/` | hardcase refinement，不推荐默认部署 |

## 4. 数据产物

| 数据 | 路径 | 说明 |
|---|---|---|
| smoke pretrain | `data/smoke/` | 小型 pretrain 验证 |
| smoke SFT | `data/sft_smoke/` | 小型 SFT 验证 |
| MiniMind clean | `data/pretrain_clean/` | 清洗后预训练文本 |
| tokenized full | `data/pretrain_clean/tokenized_full/` | `.npy` token ids |
| InstructIE processed | `data/processed/` | 6 步 pipeline 中间产物 |
| SFT candidate | `data/sft_candidate/` | Qwen 训练数据 |
| Qwen refine | `data/qwen_refine/` | hardcase refinement 数据 |

## 5. 评测结果

| 结果 | 路径 |
|---|---|
| 通用生成评测 | `results/lora_vs_full_sft_v1/eval_results.json` |
| 四模型结构化评测 | `results/instructie_eval/` |
| Qwen-only 结构化评测 | `results/instructie_eval_qwen/` |
| Qwen valid 200 | `results/qwen_valid_eval_200/` |
| Qwen valid smoke | `results/qwen_valid_eval_smoke/` |
| hardcase refine 对比 | `results/qwen_refine_compare/` |
| vLLM benchmark | `results/vllm_benchmark/` |
| schema strict benchmark | `results/vllm_benchmark_schema_strict/` |

## 6. 关键指标快照

### MicroLM

| 指标 | 值 |
|---|---:|
| 模型参数 | 31.7M |
| LoRA 可训练参数 | 约 262K |
| LoRA 占比 | 0.83% |
| LoRA adaptor | 约 1.0 MB |
| KV Cache 平均加速 | 3.86x |
| KV Cache 最大加速 | 9.08x |

### Qwen

| 指标 | 值 |
|---|---:|
| total_params | 1,543,714,304 |
| LoRA 可训练参数 | 约 2.18M |
| LoRA 占比 | 0.14% |
| adaptor 大小 | 约 8.3 MB |
| Qwen LoRA final val_loss | 0.155349 |
| valid 200 Field F1 | 78.40% |
| valid 200 Pair F1 | 67.31% |
| valid 200 Exact Match | 20.0% |

### vLLM / schema

| 指标 | 值 |
|---|---:|
| smoke | 5/5 PASS |
| stability Parse% | 100.0% |
| schema-strict raw Strict% | 52.5% |
| schema-strict Projected-Strict% | 75.0% |
| client repair strict | 77.5% |
| self-repair strict | 100.0% |

## 7. 报告索引

| 报告 | 路径 |
|---|---|
| 技术全景报告 | `reports/project_technical_report.md` |
| 推理系统收口 | `reports/inference_system_closure.md` |
| Qwen 迁移收口 | `reports/qwen_migration_structured_closure.md` |
| 评测部署闭环 | `reports/eval_deploy_closure.md` |
| schema strict 提升 | `reports/schema_strict_improvement_report.md` |
| vLLM benchmark 报告 | `reports/vllm_benchmark_report.md` |
| 项目能力边界 | `reports/microlm_capability_boundary.md` |
| 面试问答材料 | `reports/interview_qa_qwen_microlm.md` |

## 8. 大文件注意

当前提交中包含若干大文件。轻量恢复文档不复制它们，但恢复仓库时需要预留空间。

| 路径 | 说明 |
|---|---|
| `.wsl/MicroLM-Ubuntu/ext4.vhdx` | WSL 虚拟磁盘镜像 |
| `downloads/Microsoft.WSL_2.6.3.0_x64_ARM64.msixbundle` | WSL 安装包 |
| `downloads/ubuntu-noble-wsl-amd64-24.04lts.rootfs.tar.gz` | Ubuntu rootfs |
| `reports/terminal_outputs.md` | 大型终端记录 |

完整跟踪文件清单见：

```text
project_recovery_docs/manifests/git_tracked_files.txt
```

